"""Lossless embedding validation, source partitioning, and transport budgets."""

import asyncio
import dataclasses
import json

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingInputRejectedError,
    EmbeddingInputTooLongError,
    EmbeddingRequestTooLargeError,
    EmbeddingSpan,
)
from vector_core.settings import VectorCoreSettings


@pytest.fixture(autouse=True)
def unset_limits(monkeypatch):
    for name in ("max_text_chars", "max_input_bytes", "max_input_tokens", "max_request_bytes"):
        monkeypatch.setattr(f"vector_core.embeddings.client.settings.embedding_{name}", None)


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer.enable_truncation(1)
    tokenizer.enable_padding(length=30)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def wire(client, requests):
    def respond(request):
        requests.append(request)
        texts = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(sum(text.encode())), float(len(text))]}
                    for i, text in reversed(list(enumerate(texts)))
                ]
            },
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._client_loop = asyncio.get_running_loop()
    return client


async def embed(client, method, text, *, role="document"):
    argument = [text] if method in {"embed_batch", "embed_all"} else text
    return await getattr(client, method)(argument, role=role)


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
@pytest.mark.parametrize("model", ["generic", "Qwen3-Embedding-8B", "Nemotron-3-Embed-1B"])
async def test_every_entrypoint_preserves_full_source_with_unset_limits(method, model):
    requests = []
    text = " \t界🙂é\n" + "word " * 10000 + " unique-tail\t\n"
    async with wire(EmbeddingClient(model=model, dim=2), requests) as client:
        await embed(client, method, text, role="query")
        assert client.max_input_tokens is None
        assert client.max_input_bytes in (None, 0)
        prefix = client.query_prefix
    assert [json.loads(request.content)["input"] for request in requests] == [[prefix + text]]


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
@pytest.mark.parametrize("limit", ["chars", "bytes", "tokens"])
async def test_every_entrypoint_rejects_excess_without_sending_a_prefix(
    method, limit, tokenizer_path
):
    kwargs = {
        "chars": {"max_text_chars": 4},
        "bytes": {"max_input_bytes": 4},
        "tokens": {"max_input_tokens": 2, "tokenizer_path": tokenizer_path},
    }[limit]
    requests = []
    async with wire(EmbeddingClient(model="generic", dim=2, **kwargs), requests) as client:
        with pytest.raises(EmbeddingInputTooLongError) as error:
            await embed(client, method, "word word")
        assert isinstance(error.value, ValueError)
        assert client._circuit_failure_count == 0
    assert requests == []


@pytest.mark.parametrize("method", ["embed_batch", "embed_all"])
async def test_late_oversized_input_is_validated_before_any_request(method):
    requests = []
    async with wire(EmbeddingClient(model="generic", dim=2, max_text_chars=4), requests) as client:
        with pytest.raises(EmbeddingInputTooLongError):
            await getattr(client, method)(["good", "long-tail"])
    assert requests == []


@pytest.mark.parametrize("persistent", [False, True])
async def test_cached_prefix_never_accepts_an_oversized_tail(persistent, tmp_path):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            max_text_chars=4,
            cache_namespace="coverage" if persistent else None,
            cache_path=tmp_path / "cache.db",
        ),
        requests,
    ) as client:
        method = "embed_all" if persistent else "embed_single_cached"
        await embed(client, method, "word")
        with pytest.raises(EmbeddingInputTooLongError):
            await embed(client, method, "word distinct-tail")
    assert len(requests) == 1


@pytest.mark.parametrize("role,prefix", [("query", "q: "), ("document", "d: ")])
@pytest.mark.parametrize("limit", ["chars", "bytes"])
async def test_exact_character_and_utf8_boundaries_include_prefix(role, prefix, limit):
    text = " é🙂\t\n"
    length = len(prefix + text) if limit == "chars" else len((prefix + text).encode())
    kwargs = {"max_text_chars" if limit == "chars" else "max_input_bytes": length}
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic", dim=2, query_prefix="q: ", document_prefix="d: ", **kwargs
        ),
        requests,
    ) as client:
        await client.embed_batch([text], role=role)
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_batch([text + "é"], role=role)
    assert json.loads(requests[0].content)["input"] == [prefix + text]
    assert len(requests) == 1


@pytest.mark.parametrize("profile", ["raw", "qwen3", "nemotron3"])
def test_token_limits_require_local_tokenizer_for_every_profile(profile):
    with pytest.raises(ValueError, match="tokenizer"):
        EmbeddingClient(profile=profile, max_input_tokens=100)


@pytest.mark.parametrize("profile", ["raw", "qwen3", "nemotron3"])
@pytest.mark.parametrize("add_special", [True, False])
async def test_explicit_token_policy_and_reserved_overhead(profile, add_special, tokenizer_path):
    requests = []
    overhead = 2 if add_special else 0
    async with wire(
        EmbeddingClient(
            model="generic",
            profile=profile,
            dim=2,
            document_prefix="",
            query_prefix="",
            tokenizer_path=tokenizer_path,
            max_input_tokens=2 + overhead + 3,
            tokenizer_add_special_tokens=add_special,
            reserved_tokens=3,
        ),
        requests,
    ) as client:
        await client.embed_single("word word")
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_single("word word word")
    assert [json.loads(request.content)["input"] for request in requests] == [["word word"]]


async def test_special_tokens_enabled_and_reserved_tokens_zero_by_default(tokenizer_path):
    requests = []
    async with wire(
        EmbeddingClient(model="generic", dim=2, tokenizer_path=tokenizer_path, max_input_tokens=3),
        requests,
    ) as client:
        await client.embed_single("word")
        with pytest.raises(EmbeddingInputTooLongError):
            await client.embed_single("word word")
    assert len(requests) == 1


def assert_partition(text, spans):
    assert spans
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert "".join(span.text for span in spans) == text
    for i, span in enumerate(spans):
        assert isinstance(span, EmbeddingSpan)
        assert span.start < span.end
        assert span.text == text[span.start : span.end]
        if i:
            assert spans[i - 1].end == span.start
    with pytest.raises(dataclasses.FrozenInstanceError):
        spans[0].start = 1


@pytest.mark.parametrize("role,prefix", [("query", "q: "), ("document", "d: ")])
@pytest.mark.parametrize("limit", ["chars", "bytes", "tokens", "combined"])
async def test_splitter_partitions_source_with_context_and_all_budgets(
    role, prefix, limit, tokenizer_path
):
    text = " word é🙂\nword\tword 界 word\nword word unique-tail \n"
    context = "ctx: "
    kwargs = {
        "chars": {"max_text_chars": 22},
        "bytes": {"max_input_bytes": 26},
        "tokens": {"max_input_tokens": 12, "tokenizer_path": tokenizer_path},
        "combined": {
            "max_text_chars": 22,
            "max_input_bytes": 26,
            "max_input_tokens": 12,
            "tokenizer_path": tokenizer_path,
        },
    }[limit]
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic", dim=2, query_prefix="q: ", document_prefix="d: ", **kwargs
        ),
        requests,
    ) as client:
        spans = client.split_text(text, role=role, context_prefix=context)
        assert_partition(text, spans)
        assert len(spans) > 1
        await client.embed_all([context + span.text for span in spans], role=role)
    sent = [value for request in requests for value in json.loads(request.content)["input"]]
    assert sent == [prefix + context + span.text for span in spans]
    for value in sent:
        if "max_text_chars" in kwargs:
            assert len(value) <= kwargs["max_text_chars"]
        if "max_input_bytes" in kwargs:
            assert len(value.encode()) <= kwargs["max_input_bytes"]
        if "max_input_tokens" in kwargs:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            assert len(tokenizer.encode(value).ids) <= kwargs["max_input_tokens"]


def test_unbounded_split_is_one_exact_span():
    text = " \nword " * 10000 + "tail\t"
    spans = EmbeddingClient(model="generic").split_text(text, context_prefix="context: ")
    assert_partition(text, spans)
    assert spans == [EmbeddingSpan(start=0, end=len(text), text=text)]


async def test_identity_probe_respects_a_small_valid_explicit_limit():
    requests = []
    async with wire(EmbeddingClient(model="generic", dim=2, max_text_chars=1), requests) as client:
        identity = await client.resolve_identity()
        assert identity.dimension == 2
    assert json.loads(requests[0].content)["input"] == ["x"]


def test_large_source_tokenization_has_bounded_windows_and_linear_work(tokenizer_path, monkeypatch):
    token_limit = 256
    role_prefix = "query: "
    context = "ctx: "
    prefix = role_prefix + context
    client = EmbeddingClient(
        model="generic",
        tokenizer_path=tokenizer_path,
        max_input_tokens=token_limit,
        query_prefix=role_prefix,
        reserved_tokens=3,
    )
    tokenizer = client._tokenizer
    assert tokenizer is not None
    encoded_sizes = []

    class CountingTokenizer:
        def encode(self, text, *, add_special_tokens):
            encoded_sizes.append(len(text))
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(client, "_tokenizer", CountingTokenizer())
    text = "word " * 200000 + " unique-tail\t\n"
    spans = client.split_text(text, role="query", context_prefix=context)
    assert_partition(text, spans)
    assert len(spans) > 500
    assert max(encoded_sizes) <= token_limit * 8 + len(prefix)
    assert sum(encoded_sizes) <= 8 * len(text)
    assert all(
        len(tokenizer.encode(prefix + span.text, add_special_tokens=True).ids) + 3 <= token_limit
        for span in spans
    )


async def test_whitespace_only_spans_are_preserved_and_embeddable():
    text = " \t\r\n" * 100
    requests = []
    async with wire(
        EmbeddingClient(model="generic", dim=2, max_text_chars=13, max_input_bytes=13), requests
    ) as client:
        spans = client.split_text(text)
        assert_partition(text, spans)
        assert len(spans) > 1
        await client.embed_all([span.text for span in spans])
    assert (
        "".join(value for request in requests for value in json.loads(request.content)["input"])
        == text
    )


def test_dense_tokens_adapt_windows_without_reencoding_every_accepted_span(tmp_path, monkeypatch):
    tokenizer = Tokenizer(models.BPE(vocab={" ": 0, "7": 1, "!": 2}, merges=[]))
    path = tmp_path / "character-tokenizer.json"
    tokenizer.save(str(path))
    budget = 256
    client = EmbeddingClient(model="generic", tokenizer_path=path, max_input_tokens=budget)
    encoded_sizes = []

    class CountingTokenizer:
        def encode(self, text, *, add_special_tokens):
            encoded_sizes.append(len(text))
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(client, "_tokenizer", CountingTokenizer())
    text = " 7" * 50000 + "!"
    spans = client.split_text(text)
    assert_partition(text, spans)
    assert all(len(tokenizer.encode(span.text)) <= budget for span in spans)
    # One initial density-discovery window is allowed; repeated eight-context
    # windows and duplicate validation of every accepted span are not.
    assert sum(encoded_sizes) <= len(text) + 16 * budget


@pytest.mark.parametrize(
    "kwargs,text,context",
    [
        ({"max_input_bytes": 3}, "🙂", ""),
        ({"max_text_chars": 4}, "word", "1234"),
        ({"max_input_bytes": 4}, "word", "🙂"),
    ],
)
def test_splitter_impossible_capacity_raises_instead_of_losing_source(kwargs, text, context):
    with pytest.raises(EmbeddingInputTooLongError):
        EmbeddingClient(model="generic", **kwargs).split_text(text, context_prefix=context)


@pytest.mark.parametrize("kind", ["wordpiece", "unigram"])
async def test_nonmonotonic_token_counts_split_without_losing_tail(kind, tmp_path):
    vocabulary = ["[UNK]", "a", "b", "c", "d", "e", "f", "abcdef", "tail"]
    if kind == "wordpiece":
        vocabulary[2:7] = ["##b", "##c", "##d", "##e", "##f"]
        model = models.WordPiece(
            {token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    else:
        model = models.Unigram([(token, -1.0) for token in vocabulary], unk_id=0)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "nonmonotonic.json"
    tokenizer.save(str(path))
    assert len(tokenizer.encode("abcd").ids) > 1
    assert len(tokenizer.encode("abcdef").ids) == 1
    requests = []
    text = "abcdef tail tail\n"
    async with wire(
        EmbeddingClient(model="generic", dim=2, tokenizer_path=path, max_input_tokens=1), requests
    ) as client:
        spans = client.split_text(text)
        assert_partition(text, spans)
        assert spans[0].text.startswith("abcdef")
        await client.embed_all([span.text for span in spans])
    assert all(len(tokenizer.encode(span.text).ids) <= 1 for span in spans)


def request_size(texts, model="generic"):
    request = httpx.Request(
        "POST",
        "http://example.test/v1/embeddings",
        json={
            "input": texts,
            "model": model,
            "encoding_format": "float",
        },
    )
    return len(request.content)


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
@pytest.mark.parametrize("role", ["query", "document"])
@pytest.mark.parametrize("status", [400, 422, 413])
@pytest.mark.parametrize("message", ["context window exceeded", "opaque rejection"])
async def test_unknown_capacity_backend_rejection_is_strict_and_never_cached(
    method, role, status, message, tmp_path
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(status, json={"error": {"message": message}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    client = EmbeddingClient(
        model="Nemotron-3-Embed-1B",
        dim=2,
        cache_namespace="coverage",
        cache_path=tmp_path / "cache.db",
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._client_loop = asyncio.get_running_loop()
    text = "界🙂 word " * 3000 + " unique-tail\t\n"
    expected_error = EmbeddingRequestTooLargeError if status == 413 else EmbeddingInputRejectedError
    async with client:
        with pytest.raises(expected_error) as error:
            await embed(client, method, text, role=role)
        assert isinstance(error.value, ValueError)
        assert len(requests) == 1
        assert client._circuit_failure_count == 0
        assert client._circuit_open_until is None
        assert client._query_cache == {}
        assert client.max_input_tokens is None
        assert client.max_input_bytes in (None, 0)
        prefix = client.query_prefix if role == "query" else client.document_prefix
        successful = await embed(client, method, text, role=role)
        assert successful == (
            [[1.0, 2.0]] if method in {"embed_batch", "embed_all"} else [1.0, 2.0]
        )
        if method in {"embed_single_cached", "embed_all"}:
            assert await embed(client, method, text, role=role) == successful
    assert [request["input"] for request in requests] == [[prefix + text], [prefix + text]]


@pytest.mark.parametrize("status", [400, 422, 413])
@pytest.mark.parametrize("method", ["embed_batch", "embed_all"])
async def test_diagnostic_singleton_fallback_preserves_backend_rejection_type(status, method):
    requests = []

    def respond(request):
        texts = json.loads(request.content)["input"]
        requests.append(texts)
        if len(texts) > 1:
            raise RuntimeError("batch failed before a response")
        return httpx.Response(status, json={"error": {"message": "opaque rejection"}})

    client = EmbeddingClient(model="Nemotron-3-Embed-1B", dim=2)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._client_loop = asyncio.get_running_loop()
    expected_error = EmbeddingRequestTooLargeError if status == 413 else EmbeddingInputRejectedError
    async with client:
        with pytest.raises(expected_error) as error:
            await getattr(client, method)(["first unique-tail", "second unique-tail"], role="query")
        assert isinstance(error.value, ValueError)
        assert client._circuit_failure_count == 0
        assert client._circuit_open_until is None
    assert requests == [
        ["query: first unique-tail", "query: second unique-tail"],
        ["query: first unique-tail"],
    ]


@pytest.mark.parametrize("method", ["embed_batch", "embed_all"])
async def test_request_byte_budget_splits_whole_inputs_and_preserves_order(method):
    texts = ["a", "b", "🙂" * 5, "界" * 6, 'tail\\"\n']
    cap = max(request_size([text]) for text in texts)
    assert request_size(texts) > cap
    requests = []
    progress = []
    async with wire(
        EmbeddingClient(
            model="generic", dim=2, batch_size=10, concurrency=1, max_request_bytes=cap
        ),
        requests,
    ) as client:
        if method == "embed_all":
            result = await client.embed_all(
                texts, progress_cb=lambda done, total: progress.append((done, total))
            )
        else:
            result = await client.embed_batch(texts)
    assert [json.loads(request.content)["input"] for request in requests] != [texts]
    assert [text for request in requests for text in json.loads(request.content)["input"]] == texts
    assert all(len(request.content) <= cap for request in requests)
    assert result == [[float(sum(text.encode())), float(len(text))] for text in texts]
    if method == "embed_all":
        completed = 0
        expected = []
        for request in requests:
            completed += len(json.loads(request.content)["input"])
            expected.append((completed, len(texts)))
        assert progress == expected


@pytest.mark.parametrize(
    "method", ["embed_batch", "embed_all", "embed_single", "embed_single_cached"]
)
async def test_singleton_transport_overflow_fails_without_network(method):
    text = '界🙂\\"\n'
    requests = []
    async with wire(
        EmbeddingClient(model="generic", dim=2, max_request_bytes=request_size([text]) - 1),
        requests,
    ) as client:
        with pytest.raises(EmbeddingRequestTooLargeError) as error:
            await embed(client, method, text)
        assert isinstance(error.value, ValueError)
    assert requests == []


async def test_exact_transport_boundary_is_independent_of_input_bytes():
    text = '🙂\\"\n'
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            max_input_bytes=len(text.encode()),
            max_request_bytes=request_size([text]),
        ),
        requests,
    ) as client:
        await client.embed_single(text)
    assert len(requests[0].content) == request_size([text])
    assert json.loads(requests[0].content)["input"] == [text]


async def test_transport_subbatches_count_cached_duplicates_correctly(tmp_path):
    requests = []
    cap = request_size(["long-tail-a"])
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            batch_size=10,
            concurrency=1,
            cache_namespace="coverage",
            cache_path=tmp_path / "cache.db",
            max_request_bytes=cap,
        ),
        requests,
    ) as client:
        await client.embed_all(["cached"])
        requests.clear()
        progress = []
        texts = ["long-tail-a", "cached", "long-tail-a", "long-tail-b", "long-tail-b"]
        result = await client.embed_all(
            texts, progress_cb=lambda done, total: progress.append((done, total))
        )
    assert [json.loads(request.content)["input"] for request in requests] == [
        ["long-tail-a"],
        ["long-tail-b"],
    ]
    assert progress == [(1, 5), (3, 5), (5, 5)]
    assert result == [[float(sum(text.encode())), float(len(text))] for text in texts]


async def test_concurrent_transport_batches_scatter_results_and_weight_progress(tmp_path):
    later_started = asyncio.Event()
    requests = []

    async def respond(request):
        texts = json.loads(request.content)["input"]
        requests.append(texts)
        if texts == ["aaaa"]:
            await asyncio.wait_for(later_started.wait(), timeout=2)
        else:
            later_started.set()
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(ord(text[0])), 1.0]}
                    for i, text in enumerate(texts)
                ]
            },
        )

    client = EmbeddingClient(
        model="generic",
        dim=2,
        concurrency=2,
        batch_size=10,
        max_request_bytes=request_size(["aaaa"]),
        cache_namespace="coverage",
        cache_path=tmp_path / "cache.db",
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._client_loop = asyncio.get_running_loop()
    progress = []
    texts = ["aaaa", "bbbb", "aaaa", "cccc", "cccc", "cccc"]
    async with client:
        result = await client.embed_all(
            texts, progress_cb=lambda done, total: progress.append((done, total))
        )
    assert result == [[float(ord(text[0])), 1.0] for text in texts]
    assert requests == [["aaaa"], ["bbbb"], ["cccc"]]
    completed = [done for done, _ in progress]
    assert completed[0] == 1
    assert completed[-1] == 6
    assert all(a < b for a, b in zip(completed, completed[1:], strict=False))
    increments = [completed[0]] + [b - a for a, b in zip(completed, completed[1:], strict=False)]
    assert sorted(increments) == [1, 2, 3]
    assert all(total == 6 for _, total in progress)


@pytest.mark.parametrize("persistent", [False, True])
async def test_equal_effective_inputs_still_have_separate_role_cache_entries(persistent, tmp_path):
    requests = []
    async with wire(
        EmbeddingClient(
            model="generic",
            dim=2,
            query_prefix="",
            document_prefix="",
            cache_namespace="coverage" if persistent else None,
            cache_path=tmp_path / "cache.db",
        ),
        requests,
    ) as client:
        for role in ("query", "document", "query", "document"):
            if persistent:
                await client.embed_all(["same-tail"], role=role)
            else:
                await client.embed_single_cached("same-tail", role=role)
    assert len(requests) == 2


def test_settings_limits_are_optional_and_unset_by_default(monkeypatch):
    for name in ("MAX_TEXT_CHARS", "MAX_INPUT_BYTES", "MAX_INPUT_TOKENS", "MAX_REQUEST_BYTES"):
        monkeypatch.delenv(f"VECTOR_EMBEDDING_{name}", raising=False)
    settings = VectorCoreSettings()
    for name in ("max_text_chars", "max_input_bytes", "max_input_tokens", "max_request_bytes"):
        assert getattr(settings, f"embedding_{name}") is None
    assert VectorCoreSettings(embedding_max_text_chars=None).embedding_max_text_chars is None
    assert VectorCoreSettings(embedding_max_input_bytes=0).embedding_max_input_bytes == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_text_chars": -1},
        {"max_text_chars": 0},
        {"max_input_bytes": -1},
        {"max_input_tokens": 0},
        {"max_request_bytes": -1},
        {"reserved_tokens": -1},
    ],
)
def test_invalid_constructor_limits(kwargs):
    with pytest.raises(ValueError):
        EmbeddingClient(model="generic", **kwargs)
