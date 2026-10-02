"""Lossless text references and REST byte-budgeted migration copies."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.api.points_api import jsonable_encoder
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import PointsList, PointStruct, SparseVector, WriteOrdering

from vector_core.embeddings.client import EmbeddingClient
from vector_core.settings import settings
from vector_core.storage import embedding_migration as migration
from vector_core.storage.embedding_fragments import source_hash
from vector_core.storage.embedding_sources import resolve_shared_embedding_text
from vector_core.storage.qdrant import QdrantStorage


@pytest.fixture
async def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    result = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    result._client = AsyncQdrantClient(location=":memory:")
    yield result
    await result.close()


def embedder(model):
    result = EmbeddingClient(model=model, dim=3, cache_namespace="batch-tests")
    result.embed_single = AsyncMock(return_value=[1.0, 0.0, 0.0])
    result.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[1.0, 0.0, 0.0] for _ in texts]
    )
    return result


def point(point_id, payload):
    return PointStruct(
        id=point_id,
        vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[3, 17], values=[0.5, 2.0])},
        payload=payload,
    )


async def seed(storage, points):
    await storage.create_collection("corpus", dense_dim=2)
    client = await storage.get_client()
    await client.upsert("corpus", points, wait=True)


async def records(storage, collection):
    client = await storage.get_client()
    result, _ = await client.scroll(collection, with_payload=True, with_vectors=True, limit=1000)
    return {record.id: record for record in result}


def rest_body(points):
    # Use Qdrant's actual REST encoder, including its exclude-unset/none behavior.
    return jsonable_encoder(PointsList(points=points)).encode("utf-8")


async def test_two_generations_restore_content_reference_without_domain_resolver(storage):
    original = [
        point(7, {"type": "document", "content": '原文 🦊\n"\\' * 200, "extra": {"keep": True}}),
        point("e6f2b1df-0123-4567-89ab-0123456789ab", {"content": " second source "}),
    ]
    await seed(storage, original)
    before = await records(storage, "corpus")
    resolver = AsyncMock(side_effect=lambda payload: payload["content"])
    first_embedder = embedder("first")
    first = await migration.ensure_embedding_collection(storage, "corpus", first_embedder, resolver)
    first_records = await records(storage, first.physical_name)
    assert resolver.await_count == len(original)
    assert [call.args[0][0] for call in first_embedder.embed_all.await_args_list] == [
        before[key].payload["content"] for key in before
    ]
    for point_id, source in before.items():
        copied = first_records[point_id]
        assert copied.payload == {
            **source.payload,
            "embedding_text_field": "content",
            "embedding_text_source": "legacy-reconstruction",
            "embedding_fragment": {
                "schema": 1,
                "parent_id": point_id,
                "source_hash": source_hash(source.payload["content"]),
                "start": 0,
                "end": len(source.payload["content"]),
                "index": 0,
                "count": 1,
            },
        }
        assert "embedding_text" not in copied.payload
        assert copied.vector["sparse"] == source.vector["sparse"]

    unavailable = AsyncMock(side_effect=AssertionError("domain source is unavailable"))
    second_embedder = embedder("second")
    second = await migration.ensure_embedding_collection(
        storage, "corpus", second_embedder, unavailable
    )
    unavailable.assert_not_called()
    after = await records(storage, second.physical_name)
    assert set(after) == {0, *before}
    for point_id in before:
        assert after[point_id].payload == first_records[point_id].payload
        assert after[point_id].vector["sparse"] == before[point_id].vector["sparse"]
    assert [call.args[0][0] for call in second_embedder.embed_all.await_args_list] == [
        before[key].payload["content"] for key in before
    ]
    assert await records(storage, "corpus") == before
    assert await records(storage, first.physical_name) == first_records
    assert await migration.active_embedding_collection(storage, "corpus") == second.physical_name


@pytest.mark.parametrize("retained", ["same content", "authoritative embedding input"])
async def test_original_embedding_text_is_preserved_even_when_it_duplicates_content(
    storage, retained
):
    payload = {"content": "same content", "embedding_text": retained, "extra": [1, "kept"]}
    await seed(storage, [point(7, payload)])
    before = await records(storage, "corpus")
    resolver = AsyncMock(side_effect=AssertionError("retained text is authoritative"))
    client = embedder("retained")
    result = await migration.ensure_embedding_collection(storage, "corpus", client, resolver)
    copied = (await records(storage, result.physical_name))[7]
    for key, value in payload.items():
        assert copied.payload[key] == value
    assert copied.vector["sparse"] == before[7].vector["sparse"]
    assert await records(storage, "corpus") == before
    client.embed_all.assert_awaited_once_with([retained], role="document")
    resolver.assert_not_called()


async def test_distinct_reconstructed_embedding_input_is_retained_in_full(storage):
    payload = {"content": "display content", "extra": {"keep": True}}
    raw_input = "embedding prefix\n" + 'Ω🦊\\"' * 300
    await seed(storage, [point(7, payload)])
    result = await migration.ensure_embedding_collection(
        storage, "corpus", embedder("distinct"), AsyncMock(return_value=raw_input)
    )
    copied = (await records(storage, result.physical_name))[7]
    assert copied.payload["embedding_text"] == raw_input
    assert "embedding_text_field" not in copied.payload
    for key, value in payload.items():
        assert copied.payload[key] == value


async def test_explicit_reference_resolves_arbitrary_retained_payload_field(storage):
    payload = {
        "content": "display content",
        "raw_embedding_input": 'authoritative source 🦊\n\\"',
        "embedding_text_field": "raw_embedding_input",
    }
    await seed(storage, [point(7, payload)])
    resolver = AsyncMock(side_effect=AssertionError("reference is self-contained"))
    client = embedder("custom-reference")
    result = await migration.ensure_embedding_collection(storage, "corpus", client, resolver)
    copied = (await records(storage, result.physical_name))[7]
    for key, value in payload.items():
        assert copied.payload[key] == value
    resolver.assert_not_called()
    client.embed_all.assert_awaited_once_with([payload["raw_embedding_input"]], role="document")


async def test_existing_reference_is_preserved_when_another_field_has_identical_text(storage):
    payload = {
        "content": "same exact raw input 🦊",
        "other": "same exact raw input 🦊",
        "embedding_text_field": "other",
    }
    await seed(storage, [point(7, payload)])
    before = await records(storage, "corpus")
    resolver = AsyncMock(side_effect=AssertionError("reference is authoritative"))
    client = embedder("preserved-reference")
    result = await migration.ensure_embedding_collection(storage, "corpus", client, resolver)
    copied = (await records(storage, result.physical_name))[7]
    assert {
        key: value for key, value in copied.payload.items() if key != "embedding_fragment"
    } == payload
    assert copied.vector["sparse"] == before[7].vector["sparse"]
    assert await records(storage, "corpus") == before
    client.embed_all.assert_awaited_once_with([payload["other"]], role="document")
    resolver.assert_not_called()


@pytest.mark.parametrize("with_reference", [False, True])
async def test_explicit_null_embedding_text_fails_closed_even_with_usable_fallback(
    storage, with_reference
):
    payload = {"content": "usable fallback", "embedding_text": None}
    if with_reference:
        payload["embedding_text_field"] = "content"
    await seed(storage, [point(7, payload)])
    before = await records(storage, "corpus")
    resolver = AsyncMock(return_value="usable domain fallback")
    client = embedder("null-retained-text")
    with pytest.raises(migration.EmbeddingMigrationError):
        await migration.ensure_embedding_collection(storage, "corpus", client, resolver)
    resolver.assert_not_called()
    client.embed_all.assert_not_called()
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert await records(storage, "corpus") == before


async def test_shared_resolver_uses_reference_before_glossary_source_lookup():
    store = SimpleNamespace(read=Mock(side_effect=AssertionError("source lookup is forbidden")))
    text = ' full glossary input 🦊\n\\" '
    payload = {"type": "glossary", "content": text, "embedding_text_field": "content"}
    assert await resolve_shared_embedding_text(payload, glossary_store=store) == text
    store.read.assert_not_called()


@pytest.mark.parametrize("field", [None, 17, "absent", "empty"])
async def test_shared_resolver_rejects_invalid_reference_before_legacy_note_fallback(field):
    payload = {
        "type": "note",
        "title": "otherwise valid legacy note",
        "embedding_text_field": field,
        "empty": "",
    }
    with pytest.raises(ValueError, match="reference"):
        await resolve_shared_embedding_text(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"embedding_text_field": None, "content": "usable"},
        {"embedding_text_field": 17, "content": "usable"},
        {"embedding_text_field": ["content"], "content": "usable"},
        {"embedding_text_field": "absent", "content": "usable"},
        {"embedding_text_field": "content"},
        {"embedding_text_field": "content", "content": 17},
        {"embedding_text_field": "content", "content": None},
        {"embedding_text_field": "content", "content": ""},
    ],
)
async def test_invalid_explicit_reference_fails_closed_without_pointer_promotion(storage, payload):
    await seed(storage, [point(7, payload)])
    before = await records(storage, "corpus")
    resolver = AsyncMock(return_value="must never fabricate fallback text")
    client = embedder("invalid-reference")
    with pytest.raises(migration.EmbeddingMigrationError):
        await migration.ensure_embedding_collection(storage, "corpus", client, resolver)
    resolver.assert_not_called()
    client.embed_all.assert_not_called()
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert await records(storage, "corpus") == before
    candidates = [name for name in await storage.list_collections() if name.startswith("vcgen_")]
    assert len(candidates) == 1
    metadata = await storage.get_metadata(candidates[0])
    assert metadata["embedding_generation"]["state"] == "building"


async def test_copy_batches_obey_exact_rest_bytes_and_preserve_every_point(monkeypatch):
    points = [point(index, {"content": '🦊é漢\n\t"\\' * (20 + index)}) for index in range(1, 8)]
    original = deepcopy(points)
    budget = max(len(rest_body([item])) for item in points)
    assert len(rest_body(points[:2])) > budget
    assert len(rest_body([points[0]])) > len(rest_body([points[0]]).decode("utf-8"))
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", budget)
    upsert = AsyncMock()
    await migration._upsert_copy_points(SimpleNamespace(upsert=upsert), "candidate", points)
    assert upsert.await_count == len(points)
    sent = []
    for call in upsert.await_args_list:
        collection = call.args[0] if call.args else call.kwargs["collection_name"]
        batch = call.args[1] if len(call.args) > 1 else call.kwargs["points"]
        assert collection == "candidate"
        assert call.kwargs["wait"] is True
        assert call.kwargs["ordering"] == WriteOrdering.STRONG
        assert len(rest_body(batch)) <= budget
        sent.extend(batch)
    assert sent == original
    assert points == original


async def test_copy_batch_accepts_exact_budget_boundary(monkeypatch):
    points = [point(1, {"content": '🦊\n\\"' * 20}), point(2, {"content": "unchanged"})]
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", len(rest_body(points)))
    upsert = AsyncMock()
    await migration._upsert_copy_points(SimpleNamespace(upsert=upsert), "candidate", points)
    upsert.assert_awaited_once()
    call = upsert.await_args
    assert call is not None
    batch = call.args[1] if len(call.args) > 1 else call.kwargs["points"]
    assert batch == points


async def test_oversized_single_point_fails_before_sending_its_batch(monkeypatch):
    points = [point(917, {"content": '🦊"\\\n' * 100})]
    serialized_bytes = len(rest_body(points))
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", serialized_bytes - 1)
    upsert = AsyncMock()
    with pytest.raises(migration.EmbeddingMigrationError) as caught:
        await migration._upsert_copy_points(SimpleNamespace(upsert=upsert), "candidate", points)
    assert "917" in str(caught.value)
    assert str(serialized_bytes) in str(caught.value)
    upsert.assert_not_called()


async def test_empty_read_error_retains_meaningful_type_and_cause(storage, monkeypatch):
    await seed(storage, [point(7, {"content": "source"})])
    before = await records(storage, "corpus")
    read_error = httpx.ReadError("")
    failure = ResponseHandlingException(read_error)
    assert str(failure) == ""
    monkeypatch.setattr(migration, "_upsert_copy_points", AsyncMock(side_effect=failure))
    with pytest.raises(migration.EmbeddingMigrationError) as caught:
        await migration.ensure_embedding_collection(
            storage, "corpus", embedder("read-error"), AsyncMock(return_value="source")
        )
    assert "ResponseHandlingException" in str(caught.value)
    assert "ReadError" in str(caught.value)
    assert caught.value.__cause__ is failure
    assert failure.source is read_error
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert await records(storage, "corpus") == before


async def test_oversized_migration_keeps_source_and_does_not_send_data(storage, monkeypatch):
    await seed(storage, [point(917, {"content": '🦊"\\\n' * 1000})])
    before = await records(storage, "corpus")
    raw = await storage.get_client()
    original_upsert = raw.upsert
    upsert = AsyncMock(wraps=original_upsert)
    monkeypatch.setattr(raw, "upsert", upsert)
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", 4096)
    with pytest.raises(migration.EmbeddingMigrationError, match="917.*bytes"):
        await migration.ensure_embedding_collection(
            storage,
            "corpus",
            embedder("oversized"),
            AsyncMock(side_effect=lambda payload: payload["content"]),
        )
    # Only the small building manifest was sent; the oversized data batch was rejected locally.
    upsert.assert_awaited_once()
    call = upsert.await_args
    assert call is not None
    sent = call.args[1] if len(call.args) > 1 else call.kwargs["points"]
    assert [item.id for item in sent] == [0]
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert await records(storage, "corpus") == before
