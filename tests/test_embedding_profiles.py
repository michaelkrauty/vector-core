"""Model formatting, context limits and vector-space cache separation."""

import asyncio
import hashlib
import json
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingInputTooLongError,
    EmbeddingServiceError,
    SyncEmbeddingClient,
)
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import VectorCoreSettings


def wire(client, requests, *, fail_batch=False):
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if fail_batch and len(payload["input"]) > 1:
            raise RuntimeError("batch failure")
        assert set(payload) == {"model", "input", "encoding_format"}
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(sum(text.encode())), float(len(text))]}
                    for i, text in enumerate(payload["input"])
                ]
            },
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return client


@pytest.mark.parametrize(
    "model,prefix",
    [
        ("generic", ""),
        ("Qwen/Qwen3-Embedding-8B", "Instruct: Find relevant code\nQuery:"),
        ("qwen3_embedding_8b.gguf", "Instruct: Find relevant code\nQuery:"),
        ("nvidia/Nemotron-3-Embed-1B-BF16", "query: "),
        ("Nemotron-3-Embed-1B", "query: "),
    ],
)
async def test_profile_inputs(model, prefix):
    requests = []
    async with wire(
        EmbeddingClient(model=model, dim=2, query_instruction="Find relevant code"), requests
    ) as client:
        await client.embed_batch(["  text  "], role="query")
        await client.embed_single("  text  ")
    assert requests[0]["input"] == [prefix + "  text  "]
    document_prefix = "passage: " if "nemotron" in model.lower() else ""
    assert requests[1]["input"] == [document_prefix + "  text  "]


async def test_explicit_raw_override_and_custom_prefixes():
    requests = []
    async with wire(
        EmbeddingClient(
            model="Nemotron-3-Embed-1B",
            profile="raw",
            dim=2,
            query_prefix="search: ",
            document_prefix="text: ",
        ),
        requests,
    ) as client:
        await client.embed_all(["same"], role="query")
        await client.embed_all(["same"])
    assert [p["input"] for p in requests] == [["search: same"], ["text: same"]]


async def test_memory_cache_uses_role_and_full_input():
    requests = []
    async with wire(EmbeddingClient(model="Nemotron-3-Embed-1B", dim=2), requests) as client:
        query = await client.embed_single_cached("abcdef-one", role="query")
        other = await client.embed_single_cached("abcdef-two", role="query")
        assert other != query
        assert await client.embed_single_cached("abcdef-one", role="query") == query
        document = await client.embed_single_cached("abcdef-one")
        assert query != document
        assert await client.embed_single_cached("abcdef-one") == document
    assert [p["input"] for p in requests] == [
        ["query: abcdef-one"],
        ["query: abcdef-two"],
        ["passage: abcdef-one"],
    ]


async def test_auto_dimension_cached_concurrent_requests(monkeypatch):
    monkeypatch.setattr("vector_core.embeddings.client.settings.embedding_dim", 4096)
    requests = []
    async with wire(EmbeddingClient(model="generic", dim=0), requests) as client:
        assert client.dim == 0
        results = await asyncio.gather(*(client.embed_single_cached("same") for _ in range(3)))
        assert results[0] == results[1] == results[2]
        assert client.dim == 2
    assert len(requests) == 1


async def test_persistent_cache_segregates_roles_profiles_and_reuses_exact_input(tmp_path):
    requests = []
    kwargs: dict[str, Any] = {
        "model": "alias",
        "dim": 2,
        "cache_namespace": "v1",
        "cache_path": tmp_path / "cache.db",
    }
    for profile in ("raw", "nemotron3", "raw"):
        async with wire(EmbeddingClient(**kwargs, profile=profile), requests) as client:
            await client.embed_all(["same", "same"], role="query")
            await client.embed_all(["same"])
    assert len(requests) == 4
    assert [p["input"] for p in requests] == [
        ["same"],
        ["same"],
        ["query: same"],
        ["passage: same"],
    ]


async def test_batch_fallback_never_reapplies_prefix():
    requests = []
    async with wire(
        EmbeddingClient(model="Nemotron-3-Embed-1B", dim=2), requests, fail_batch=True
    ) as client:
        await client.embed_all(["a", "b"], role="query")
    assert [p["input"] for p in requests] == [["query: a", "query: b"], ["query: a"], ["query: b"]]


@pytest.mark.parametrize("text", ["", None])
async def test_empty_or_non_string_inputs_fail_before_network(text):
    requests = []
    async with wire(EmbeddingClient(model="Nemotron-3-Embed-1B", dim=2), requests) as client:
        with pytest.raises(ValueError, match="non-empty"):
            await client.embed_single_cached(text, role="query")
    assert requests == []


@pytest.mark.parametrize("role,prefix", [("query", "query: "), ("document", "passage: ")])
async def test_nemotron_utf8_byte_budget_includes_prefix(role, prefix):
    requests = []
    text = "🙂界"
    async with wire(
        EmbeddingClient(
            model="Nemotron-3-Embed-1B", dim=2, max_input_bytes=len((prefix + text).encode())
        ),
        requests,
    ) as client:
        await client.embed_batch([text], role=role)
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_batch([text + "é"], role=role)
    assert [p["input"] for p in requests] == [[prefix + text]]


@pytest.fixture
def tokenizer_file(tmp_path):
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "query": 1, ":": 2, "passage": 3, "word": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # Loading must disable tokenizer-level implicit truncation/padding.
    tokenizer.enable_truncation(1)
    tokenizer.enable_padding(length=20)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


async def test_local_tokenizer_preserves_coverage_and_counts_prefix(tokenizer_file):
    requests = []
    async with wire(
        EmbeddingClient(
            model="Nemotron-3-Embed-1B", dim=2, tokenizer_path=tokenizer_file, max_input_tokens=1000
        ),
        requests,
    ) as client:
        text = "word " * 998
        await client.embed_all([text], role="query")
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_all([text + "word "], role="query")
        assert client.max_input_bytes in (None, 0)
        assert (
            client.tokenizer_fingerprint == hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
        )
        tokenizer = client._tokenizer
        assert tokenizer is not None
    sent = requests[0]["input"][0]
    assert len(sent.encode()) > 4096
    assert sent == "query: " + text
    assert len(tokenizer.encode(sent, add_special_tokens=True).ids) == 1000


async def test_tokenizer_prefix_too_large_fails_locally(tokenizer_file):
    requests = []
    async with wire(
        EmbeddingClient(
            model="Nemotron-3-Embed-1B", dim=2, tokenizer_path=tokenizer_file, max_input_tokens=1
        ),
        requests,
    ) as client:
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_batch(["word"])
    assert requests == []


async def test_raw_tokenizer_budget_includes_postprocessor_tokens(tokenizer_file):
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.save(str(tokenizer_file))
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            profile="raw",
            dim=2,
            tokenizer_path=tokenizer_file,
            max_input_tokens=4,
        ),
        requests,
    ) as client:
        await client.embed_single("word word")
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_single("word word word")
    sent = requests[0]["input"][0]
    assert len(tokenizer.encode(sent, add_special_tokens=True).ids) == 4
    assert len(tokenizer.encode(sent, add_special_tokens=False).ids) == 2


@pytest.mark.parametrize("kind", ["wordpiece", "unigram"])
async def test_token_limit_preserves_later_merged_prefix(tmp_path, kind):
    if kind == "wordpiece":
        vocabulary = ["[UNK]", "a", "##b", "##c", "##d", "##e", "##f", "abcdef", "tail"]
        model = models.WordPiece(
            {token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    else:
        vocabulary = ["[UNK]", "a", "b", "c", "d", "e", "f", "abcdef", "tail"]
        model = models.Unigram([(token, -1.0) for token in vocabulary], unk_id=0)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "nonmonotonic-tokenizer.json"
    tokenizer.save(str(path))
    assert len(tokenizer.encode("abcd").ids) > 1
    assert len(tokenizer.encode("abcdef").ids) == 1
    requests = []
    async with wire(
        EmbeddingClient(model="generic", dim=2, tokenizer_path=path, max_input_tokens=1),
        requests,
    ) as client:
        await client.embed_single("abcdef")
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_single("abcdef tail tail")
    assert requests[0]["input"] == ["abcdef"]


def test_formatting_changes_identity_and_roundtrip(tokenizer_file):
    base = EmbeddingClient(model="alias", dim=2).configured_identity()
    changes: list[dict[str, Any]] = [
        {"profile": "nemotron3"},
        {"query_prefix": "query: "},
        {"document_prefix": "passage: "},
        {"max_input_bytes": 200},
        {"max_text_chars": 100},
        {"tokenizer_path": tokenizer_file, "max_input_tokens": 100},
        {
            "tokenizer_path": tokenizer_file,
            "max_input_tokens": 100,
            "tokenizer_add_special_tokens": False,
        },
        {"tokenizer_path": tokenizer_file, "max_input_tokens": 100, "reserved_tokens": 2},
    ]
    for changed in changes:
        identity = EmbeddingClient(model="alias", dim=2, **changed).configured_identity()
        assert identity.fingerprint != base.fingerprint
        assert EmbeddingIdentity.from_dict(identity.to_dict()) == identity
        assert str(tokenizer_file) not in json.dumps(identity.to_dict())


async def test_bound_identity_rejects_profile_mutation_on_cache_hit():
    requests = []
    async with wire(EmbeddingClient(model="alias", dim=2), requests) as client:
        await client.resolve_identity()
        await client.embed_single_cached("same")
        client.query_prefix = "query: "
        with pytest.raises(EmbeddingServiceError, match="configuration changed"):
            await client.embed_single_cached("same")


async def test_dimension_mismatch_fails_before_binding():
    requests = []
    async with wire(EmbeddingClient(model="Nemotron-3-Embed-1B", dim=2048), requests) as client:
        with pytest.raises(EmbeddingServiceError, match="dimension 2048"):
            await client.resolve_identity()
        assert client._identity is None


def test_sync_roles_reach_network():
    requests = []
    with SyncEmbeddingClient(model="Nemotron-3-Embed-1B", dim=2) as client:
        wire(client._client, requests)
        client.embed_single("a", role="query")
        client.embed_single_cached("b", role="query")
        client.embed_batch(["c"], role="query")
        client.embed_all(["d"], role="query")
    assert [p["input"] for p in requests] == [[f"query: {v}"] for v in "abcd"]


@pytest.mark.parametrize(
    "setting,value",
    [
        ("embedding_profile", "bad"),
        ("embedding_max_input_bytes", -1),
        ("embedding_max_input_tokens", 0),
    ],
)
def test_invalid_settings(setting, value):
    with pytest.raises(ValueError):
        VectorCoreSettings(**{setting: value})
