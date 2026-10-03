"""Explicit token-ID transport preserves input boundaries and embedding identity."""

import dataclasses
import hashlib
import json
from typing import Any

import httpx
import pytest
import tokenizers
from tokenizers import Tokenizer, models, normalizers

from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingInputRejectedError,
    EmbeddingInputTooLongError,
    EmbeddingRequestTooLargeError,
    EmbeddingServiceError,
    SyncEmbeddingClient,
)
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import VectorCoreSettings, VectorCoreSettingsMixin, settings

from . import test_embedding_coverage as coverage
from .test_embedding_coverage import assert_partition, embed, request_size

tokenizer_path = coverage.tokenizer_path
unset_limits = coverage.unset_limits


@pytest.fixture
def normalized_tokenizer_path(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.model = models.WordLevel(
        {**tokenizer.get_vocab(), "é": 4, "qword": 5}, unk_token="[UNK]"
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def wire(client, requests, *, failure=None):
    def respond(request):
        requests.append(request)
        inputs = json.loads(request.content)["input"]
        if failure == "retry" and len(requests) == 1:
            return httpx.Response(503)
        if failure == "fallback" and len(inputs) > 1:
            raise RuntimeError("batch failed")
        if failure == "reject":
            return httpx.Response(400, text="unsupported ID input")
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": i,
                        "embedding": [
                            float(sum(value.encode() if isinstance(value, str) else value)),
                            float(len(value)),
                        ],
                    }
                    for i, value in reversed(list(enumerate(inputs)))
                ]
            },
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return client


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
@pytest.mark.parametrize("add_special", [True, False])
async def test_exact_normalized_and_literal_special_ids(
    method, add_special, normalized_tokenizer_path
):
    source = " e\u0301 [SEP] word\t\n"
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            profile="raw",
            input_encoding="token_ids",
            tokenizer_path=normalized_tokenizer_path,
            tokenizer_add_special_tokens=add_special,
        ),
        requests,
    ) as client:
        assert client._prepare_texts([source], role="document") == [source]
        await embed(client, method, source)
    expected = [2, 4, 3, 1, 3] if add_special else [4, 3, 1]
    assert [json.loads(request.content)["input"] for request in requests] == [[expected]]


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
@pytest.mark.parametrize("role", ["query", "document"])
@pytest.mark.parametrize("add_special", [True, False])
async def test_full_boundary_counts_prefix_specials_and_reserved_overhead(
    method, role, add_special, normalized_tokenizer_path
):
    # q + word must merge before counting; encoding the prefix separately is wrong.
    expected = [2, 5, 3] if add_special else [5]
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            profile="raw",
            input_encoding="token_ids",
            query_prefix="q",
            document_prefix="q",
            tokenizer_path=normalized_tokenizer_path,
            tokenizer_add_special_tokens=add_special,
            reserved_tokens=2,
            max_input_tokens=len(expected) + 2,
        ),
        requests,
    ) as client:
        await embed(client, method, "word", role=role)
        with pytest.raises(EmbeddingInputTooLongError) as error:
            await embed(client, method, "word word", role=role)
        assert error.value.measured == len(expected) + 3
        assert error.value.role == role
    assert [json.loads(request.content)["input"] for request in requests] == [[expected]]


async def test_id_splitter_keeps_original_normalization_and_character_offsets(
    normalized_tokenizer_path,
):
    source = " e\u0301 [SEP] word\t\n" * 20
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=normalized_tokenizer_path,
            max_input_tokens=5,
        ),
        requests,
    ) as client:
        spans = client.split_text(source)
        assert_partition(source, spans)
        assert len(spans) > 1
        await client.embed_all([span.text for span in spans])
        tokenizer = client._tokenizer
        assert tokenizer is not None
    sent = [ids for request in requests for ids in json.loads(request.content)["input"]]
    assert sent == [tokenizer.encode(span.text, add_special_tokens=True).ids for span in spans]
    assert all(len(ids) <= 5 for ids in sent)
    assert "e\u0301" in "".join(span.text for span in spans)


@pytest.mark.parametrize("method", ["embed_batch", "embed_all"])
async def test_actual_id_json_budget_packs_whole_inputs_and_preserves_order(method, tokenizer_path):
    texts = ["word", "word word word", "[SEP]", "word word", "word"]
    ids = [[2, 1, 3], [2, 1, 1, 1, 3], [2, 3, 3], [2, 1, 1, 3], [2, 1, 3]]
    cap = request_size(ids[:2])
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            max_request_bytes=cap,
            batch_size=10,
            concurrency=1,
        ),
        requests,
    ) as client:
        result = await getattr(client, method)(texts)
    assert [json.loads(request.content)["input"] for request in requests] == [
        ids[:2],
        ids[2:4],
        ids[4:],
    ]
    assert len(requests[0].content) == cap
    assert all(len(request.content) <= cap for request in requests)
    assert result == [[float(sum(value)), float(len(value))] for value in ids]


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
async def test_actual_id_singleton_budget_accepts_exact_boundary_and_refuses_overflow(
    method, tokenizer_path
):
    ids = [[2, 1, 3]]
    cap = request_size(ids)
    assert cap != request_size(["word"])
    for budget in (cap - 1, cap):
        requests = []
        async with wire(
            EmbeddingClient(
                model="generic",
                dim=2,
                input_encoding="token_ids",
                tokenizer_path=tokenizer_path,
                max_request_bytes=budget,
            ),
            requests,
        ) as client:
            if budget < cap:
                with pytest.raises(EmbeddingRequestTooLargeError):
                    await embed(client, method, "word")
                assert requests == []
            else:
                await embed(client, method, "word")
                assert len(requests[0].content) == cap
                assert json.loads(requests[0].content)["input"] == ids


@pytest.mark.parametrize("token_limit", [None, 5])
async def test_request_budget_tokenization_work_is_linear(token_limit, tokenizer_path, monkeypatch):
    client = EmbeddingClient(
        model="generic",
        dim=2,
        input_encoding="token_ids",
        tokenizer_path=tokenizer_path,
        max_request_bytes=100000,
        max_input_tokens=token_limit,
        batch_size=100,
    )
    tokenizer = client._tokenizer
    assert tokenizer is not None
    encoded = []

    class CountingTokenizer:
        def encode(self, text, *, add_special_tokens):
            encoded.append(text)
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(client, "_tokenizer", CountingTokenizer())
    requests = []
    async with wire(client, requests):
        await client.embed_batch(["word"] * 100)
    assert len(encoded) == 200
    assert len(json.loads(requests[0].content)["input"]) == 100


@pytest.mark.parametrize("failure", ["retry", "fallback", "reject"])
async def test_retry_and_fallback_retain_ids_without_text_fallback(
    failure, tokenizer_path, monkeypatch
):
    async def no_sleep(delay):
        pass

    monkeypatch.setattr("vector_core.utils.retry.asyncio.sleep", no_sleep)
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            document_prefix="word ",
        ),
        requests,
        failure=failure,
    ) as client:
        if failure == "reject":
            with pytest.raises(EmbeddingInputRejectedError):
                await client.embed_batch(["word", "[SEP]"])
        else:
            await client.embed_batch(["word", "[SEP]"])
    inputs = [[2, 1, 1, 3], [2, 1, 3, 3]]
    expected = {
        "retry": [inputs, inputs],
        "fallback": [inputs, inputs[:1], inputs[1:]],
        "reject": [inputs],
    }[failure]
    assert [json.loads(request.content)["input"] for request in requests] == expected
    if failure == "retry":
        assert requests[0].content == requests[1].content


async def test_id_cache_separates_transport_and_preserves_duplicates_roles_and_order(
    tokenizer_path, tmp_path
):
    requests = []
    kwargs: dict[str, Any] = {
        "model": "generic",
        "dim": 2,
        "tokenizer_path": tokenizer_path,
        "cache_namespace": "revision",
        "cache_path": tmp_path / "cache.db",
        "concurrency": 1,
    }
    for mode in ("text", "token_ids", "text", "token_ids"):
        async with wire(EmbeddingClient(**kwargs, input_encoding=mode), requests) as client:
            result = await client.embed_all(["word", "[SEP]", "word"])
            assert result[0] == result[2]
            assert result[0] != result[1]
            assert await client.embed_all(["[SEP]", "word", "[SEP]"]) == [
                result[1],
                result[0],
                result[1],
            ]
            await client.embed_all(["word"], role="query")
    assert [json.loads(request.content)["input"] for request in requests] == [
        ["word", "[SEP]"],
        ["word"],
        [[2, 1, 3], [2, 3, 3]],
        [[2, 1, 3]],
    ]


def test_default_text_identity_and_memory_fingerprint_match_previous_schema(tokenizer_path):
    client = EmbeddingClient(
        base_url="http://example.test",
        model="generic",
        dim=2,
        profile="raw",
        tokenizer_path=tokenizer_path,
    )
    legacy = {
        "model": "generic",
        "namespace": None,
        "endpoint": "http://example.test",
        "dimension": 2,
        "preprocessing": "complete-input-fragments-v5",
        "max_text_chars": None,
        "profile": "raw",
        "query_prefix": "",
        "document_prefix": "",
        "max_input_bytes": 0,
        "max_input_tokens": None,
        "tokenizer_fingerprint": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokenizer_add_special_tokens": True,
        "reserved_tokens": 0,
    }
    identity = client.configured_identity()
    assert client.input_encoding == "text"
    assert identity.to_dict() == legacy
    assert EmbeddingIdentity.from_dict(legacy) == identity
    assert (
        identity.fingerprint
        == hashlib.sha256(
            json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    memory = {
        key: value for key, value in legacy.items() if key not in {"dimension", "preprocessing"}
    }
    memory.update(dim=2, role="document", input="word")
    assert (
        client._memory_cache_key("word", role="document")
        == hashlib.sha256(json.dumps(memory, sort_keys=True).encode()).hexdigest()
    )


def test_token_id_identity_tracks_tokenizer_implementation_version_hash_and_not_transport_budget(
    tokenizer_path,
):
    client = EmbeddingClient(
        model="generic", dim=2, input_encoding="token_ids", tokenizer_path=tokenizer_path
    )
    identity = client.configured_identity()
    assert identity.tokenizer_implementation == "tokenizers"
    assert identity.tokenizer_version == tokenizers.__version__
    assert identity.tokenizer_fingerprint == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert EmbeddingIdentity.from_dict(identity.to_dict()) == identity
    for change in (
        {"tokenizer_implementation": "different"},
        {"tokenizer_version": "different"},
        {"tokenizer_fingerprint": "0" * 64},
    ):
        assert dataclasses.replace(identity, **change).fingerprint != identity.fingerprint
    client.max_request_bytes = 100
    assert client.configured_identity() == identity
    assert str(tokenizer_path) not in json.dumps(identity.to_dict())


@pytest.mark.parametrize(
    "change",
    [
        {"input_encoding": "invalid"},
        {"tokenizer_implementation": None},
        {"tokenizer_version": ""},
        {"tokenizer_fingerprint": "invalid"},
    ],
)
def test_invalid_persisted_token_id_identity_is_rejected(change, tokenizer_path):
    value = (
        EmbeddingClient(
            model="generic", dim=2, input_encoding="token_ids", tokenizer_path=tokenizer_path
        )
        .configured_identity()
        .to_dict()
    )
    with pytest.raises(ValueError):
        EmbeddingIdentity.from_dict({**value, **change})


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_encoding", "text"),
        ("tokenizer_implementation", "different"),
        ("tokenizer_version", "different"),
        ("tokenizer_fingerprint", "0" * 64),
        ("tokenizer_add_special_tokens", False),
    ],
)
async def test_bound_token_id_configuration_cannot_change_on_cache_hit(
    field, value, tokenizer_path
):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic", dim=2, input_encoding="token_ids", tokenizer_path=tokenizer_path
        ),
        requests,
    ) as client:
        await client.resolve_identity()
        await client.embed_single_cached("word")
        setattr(client, field, value)
        with pytest.raises(EmbeddingServiceError, match="configuration changed"):
            await client.embed_single_cached("word")
    assert len(requests) == 2


def test_token_ids_require_tokenizer_and_settings_mixin_forward_mode(monkeypatch, tokenizer_path):
    monkeypatch.setattr(settings, "embedding_tokenizer_path", None)
    with pytest.raises(ValueError, match="requires a local tokenizer"):
        EmbeddingClient(model="generic", input_encoding="token_ids")
    with pytest.raises(ValueError, match="Unknown embedding input encoding"):
        EmbeddingClient(model="generic", input_encoding="invalid")  # type: ignore[arg-type]
    monkeypatch.setenv("VECTOR_EMBEDDING_INPUT_ENCODING", "token_ids")
    assert VectorCoreSettings().embedding_input_encoding == "token_ids"
    monkeypatch.setattr(settings, "embedding_input_encoding", "token_ids")
    assert VectorCoreSettingsMixin().embedding_input_encoding == "token_ids"
    assert (
        EmbeddingClient(model="generic", tokenizer_path=tokenizer_path).input_encoding
        == "token_ids"
    )
    monkeypatch.setenv("VECTOR_EMBEDDING_INPUT_ENCODING", "invalid")
    with pytest.raises(ValueError):
        VectorCoreSettings()


def test_sync_token_id_mode_reaches_every_network_entrypoint(tokenizer_path):
    requests = []
    with SyncEmbeddingClient(
        model="generic",
        dim=2,
        input_encoding="token_ids",
        tokenizer_path=tokenizer_path,
        query_prefix="word ",
    ) as client:
        wire(client._client, requests)
        client.embed_single("word", role="query")
        client.embed_single_cached("word", role="query")
        client.embed_batch(["word"], role="query")
        client.embed_all(["word"], role="query")
        assert client.configured_identity().input_encoding == "token_ids"
    assert [json.loads(request.content)["input"] for request in requests] == [[[2, 1, 1, 3]]] * 4


@pytest.mark.parametrize("dropout", [None, 0.0, 0.5])
def test_token_id_mode_requires_deterministic_bpe_but_text_behavior_is_unchanged(dropout, tmp_path):
    tokenizer = Tokenizer(
        models.BPE(vocab={"a": 0, "b": 1, "ab": 2}, merges=[("a", "b")], dropout=dropout)
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert EmbeddingClient(model="generic", tokenizer_path=path).input_encoding == "text"
    if dropout:
        with pytest.raises(ValueError, match="deterministic tokenizer"):
            EmbeddingClient(model="generic", tokenizer_path=path, input_encoding="token_ids")
    else:
        assert (
            EmbeddingClient(
                model="generic", tokenizer_path=path, input_encoding="token_ids"
            ).input_encoding
            == "token_ids"
        )


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
async def test_nonempty_source_with_no_ids_is_rejected_before_http(method, tokenizer_path):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            tokenizer_add_special_tokens=False,
        ),
        requests,
    ) as client:
        with pytest.raises(ValueError, match="empty sequence"):
            await embed(client, method, " \t\n")
    assert requests == []


@pytest.mark.parametrize("method", ["embed_batch", "embed_all"])
async def test_late_oversized_id_sequence_is_refused_before_any_http(method, tokenizer_path):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            max_input_tokens=3,
            batch_size=1,
        ),
        requests,
    ) as client:
        with pytest.raises(EmbeddingInputTooLongError) as error:
            await getattr(client, method)(["word", "word word"], role="query")
        assert error.value.index == 1
        assert error.value.role == "query"
    assert requests == []


@pytest.mark.parametrize(
    "source",
    [
        "word" + " " * 20 + "word",
        " " * 20 + "word",
        "word" + " " * 20,
        " " * 20 + "word" + " " * 20 + "word" + "\t\n" * 10,
    ],
)
async def test_splitter_merges_tokenless_runs_without_losing_source(source, tokenizer_path):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            tokenizer_add_special_tokens=False,
            max_input_tokens=1,
        ),
        requests,
    ) as client:
        spans = client.split_text(source)
        assert_partition(source, spans)
        await client.embed_all([span.text for span in spans])
    sent = [ids for request in requests for ids in json.loads(request.content)["input"]]
    assert sent == [[1]] * len(spans)


@pytest.mark.parametrize("limit", ["chars", "bytes"])
async def test_tokenless_run_can_be_distributed_between_bounded_neighbors(limit, tokenizer_path):
    source = "word" + " " * 16 + "word"
    kwargs: dict[str, Any] = {"max_text_chars" if limit == "chars" else "max_input_bytes": 12}
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            tokenizer_add_special_tokens=False,
            max_input_tokens=1,
            **kwargs,
        ),
        requests,
    ) as client:
        spans = client.split_text(source)
        assert_partition(source, spans)
        assert [span.text for span in spans] == ["word" + " " * 8, " " * 8 + "word"]
        await client.embed_all([span.text for span in spans])
    assert all(len(span.text.encode()) <= 12 for span in spans)
    assert [ids for request in requests for ids in json.loads(request.content)["input"]] == [
        [1],
        [1],
    ]


@pytest.mark.parametrize("limit", ["chars", "bytes"])
@pytest.mark.parametrize("source", [" " * 20, "word" + " " * 20, " " * 20 + "word"])
def test_splitter_refuses_unembeddable_tokenless_source_instead_of_returning_spans(
    source, limit, tokenizer_path
):
    kwargs: dict[str, Any] = {"max_text_chars" if limit == "chars" else "max_input_bytes": 12}
    client = EmbeddingClient(
        model="generic",
        dim=2,
        input_encoding="token_ids",
        tokenizer_path=tokenizer_path,
        tokenizer_add_special_tokens=False,
        max_input_tokens=1,
        **kwargs,
    )
    with pytest.raises(ValueError):
        client.split_text(source)


async def test_tokenless_source_is_embeddable_when_repeated_context_supplies_ids(tokenizer_path):
    source = " \t\n" * 20
    context = "word "
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            input_encoding="token_ids",
            tokenizer_path=tokenizer_path,
            tokenizer_add_special_tokens=False,
            max_input_tokens=1,
            max_text_chars=8,
        ),
        requests,
    ) as client:
        spans = client.split_text(source, role="query", context_prefix=context)
        assert_partition(source, spans)
        await client.embed_all([context + span.text for span in spans], role="query")
    assert [ids for request in requests for ids in json.loads(request.content)["input"]] == [
        [1] for _ in spans
    ]


def test_coalescing_long_tokenless_runs_keeps_tokenizer_work_linear(tokenizer_path, monkeypatch):
    client = EmbeddingClient(
        model="generic",
        dim=2,
        input_encoding="token_ids",
        tokenizer_path=tokenizer_path,
        tokenizer_add_special_tokens=False,
        max_input_tokens=1,
    )
    tokenizer = client._tokenizer
    assert tokenizer is not None
    encoded_sizes = []

    class CountingTokenizer:
        def encode(self, text, *, add_special_tokens):
            encoded_sizes.append(len(text))
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(client, "_tokenizer", CountingTokenizer())
    work = []
    for length in (4096, 8192):
        encoded_sizes.clear()
        source = "word" + " " * length + "word"
        spans = client.split_text(source)
        assert_partition(source, spans)
        assert len(spans) == 2
        work.append(sum(encoded_sizes))
        assert work[-1] <= 4 * len(source)
    assert work[1] <= 2 * work[0] + 32
