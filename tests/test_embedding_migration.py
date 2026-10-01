"""Migration contracts exercised against an isolated in-memory Qdrant instance."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PayloadIndexInfo, PointStruct, SparseVector, TextIndexParams

from vector_core.embeddings.client import EmbeddingClient, EmbeddingServiceError
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import settings
from vector_core.storage.embedding_migration import (
    EmbeddingMigrationError,
    _copy_indexes,
    active_embedding_collection,
    embedding_collection_lock,
    ensure_embedding_collection,
)
from vector_core.storage.qdrant import QdrantStorage


@pytest.fixture
async def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    result = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    result._client = AsyncQdrantClient(location=":memory:")
    yield result
    await result.close()


def embedder(model="new", dim=3, namespace="revision-1"):
    client = EmbeddingClient(model=model, dim=dim, cache_namespace=namespace)
    client.embed_single = AsyncMock(return_value=[1.0] + [0.0] * (dim - 1))
    client.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[1.0] + [0.0] * (dim - 1) for _ in texts]
    )
    return client


async def text(payload):
    return payload["content"]


async def seed(storage, count=2):
    await storage.create_collection("corpus", dense_dim=2)
    client = await storage.get_client()
    await client.upsert(
        "corpus",
        [
            PointStruct(
                id=index,
                vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[index], values=[1.0])},
                payload={"type": "document", "content": f"source {index}", "extra": {"kept": True}},
            )
            for index in range(1, count + 1)
        ],
        wait=True,
    )


async def records(storage, collection):
    client = await storage.get_client()
    points, _ = await client.scroll(collection, with_payload=True, with_vectors=True, limit=1000)
    return {point.id: point for point in points}


async def test_unknown_legacy_rebuild_preserves_points_sparse_and_original(storage):
    await seed(storage)
    before = await records(storage, "corpus")
    client = embedder()
    generation = await ensure_embedding_collection(storage, "corpus", client, text)
    assert generation.migrated
    assert generation.physical_name != "corpus"
    assert await active_embedding_collection(storage, "corpus") == generation.physical_name
    assert await records(storage, "corpus") == before
    after = await records(storage, generation.physical_name)
    assert set(after) == {0, 1, 2}
    for point_id in (1, 2):
        assert after[point_id].vector["sparse"] == before[point_id].vector["sparse"]
        assert after[point_id].payload == {
            **before[point_id].payload,
            "embedding_text_field": "content",
            "embedding_text_source": "legacy-reconstruction",
        }
        assert len(after[point_id].vector["dense"]) == 3
    again = await ensure_embedding_collection(storage, "corpus", client, text)
    assert again == generation
    client.embed_all.assert_awaited_once()


async def test_model_namespace_and_dimension_changes_create_distinct_spaces(storage):
    await seed(storage)
    clients = [
        embedder(),
        embedder(model="other"),
        embedder(model="other", namespace="revision-2"),
        embedder(model="other", namespace="", dim=4),
    ]
    generations = [
        await ensure_embedding_collection(storage, "corpus", client, text) for client in clients
    ]
    assert len({generation.physical_name for generation in generations}) == 4
    for client in clients[:-1]:
        with pytest.raises(EmbeddingMigrationError, match="superseded"):
            await ensure_embedding_collection(storage, "corpus", client, text)


async def test_returning_to_old_model_copies_current_generation(storage):
    await seed(storage)
    first = await ensure_embedding_collection(storage, "corpus", embedder(model="A"), text)
    second = await ensure_embedding_collection(storage, "corpus", embedder(model="B"), text)
    client = await storage.get_client()
    await client.delete(second.physical_name, points_selector=[1], wait=True)
    third = await ensure_embedding_collection(storage, "corpus", embedder(model="A"), text)
    assert third.physical_name != first.physical_name
    assert set(await records(storage, third.physical_name)) == {0, 2}
    assert set(await records(storage, first.physical_name)) == {0, 1, 2}


async def test_failure_never_publishes_partial_generation_and_retry_is_clean(storage):
    await seed(storage, count=140)
    before = await records(storage, "corpus")
    client = embedder()
    client.embed_all.side_effect = [
        [[1.0, 0.0, 0.0]] * 128,
        EmbeddingServiceError("offline"),
    ]
    with pytest.raises(EmbeddingMigrationError, match="offline"):
        await ensure_embedding_collection(storage, "corpus", client, text)
    assert await active_embedding_collection(storage, "corpus") == "corpus"
    assert await records(storage, "corpus") == before
    candidates = [name for name in await storage.list_collections() if name.startswith("vcgen_")]
    assert len(candidates) == 1
    failed = await storage.get_metadata(candidates[0])
    assert failed["embedding_generation"]["state"] == "building"
    raw = await storage.get_client()
    await raw.delete("corpus", points_selector=[1], wait=True)
    recovered = await ensure_embedding_collection(storage, "corpus", embedder(), text)
    assert recovered.physical_name != candidates[0]
    assert 1 not in await records(storage, recovered.physical_name)


async def test_missing_text_fails_closed_instead_of_skipping(storage):
    await seed(storage)
    missing = AsyncMock(side_effect=ValueError("source unavailable"))
    with pytest.raises(EmbeddingMigrationError, match="source unavailable"):
        await ensure_embedding_collection(storage, "corpus", embedder(), missing)
    assert await active_embedding_collection(storage, "corpus") == "corpus"


async def test_none_resolver_requires_an_explicit_source_finalizer(storage):
    await seed(storage)
    with pytest.raises(EmbeddingMigrationError, match="cannot be omitted"):
        await ensure_embedding_collection(
            storage, "corpus", embedder(), AsyncMock(return_value=None)
        )
    assert await active_embedding_collection(storage, "corpus") == "corpus"


async def test_exact_retained_text_needs_no_external_source(storage):
    await seed(storage)
    raw = await storage.get_client()
    await raw.set_payload("corpus", {"embedding_text": "authoritative text"}, points=[1, 2])
    resolver = AsyncMock(side_effect=AssertionError("must not access source"))
    generation = await ensure_embedding_collection(storage, "corpus", embedder(), resolver)
    assert len(await records(storage, generation.physical_name)) == 3
    resolver.assert_not_called()


async def test_finalizer_runs_before_publish_and_failure_keeps_source(storage):
    await seed(storage)
    resolver = AsyncMock(return_value=None)

    async def finalize(target):
        assert await active_embedding_collection(storage, "corpus") == "corpus"
        assert set(await records(storage, target)) == {0}
        raise RuntimeError("source group incomplete")

    with pytest.raises(EmbeddingMigrationError, match="source group incomplete"):
        await ensure_embedding_collection(
            storage, "corpus", embedder(), resolver, finalize_candidate=finalize
        )
    assert await active_embedding_collection(storage, "corpus") == "corpus"


async def test_concurrent_clients_share_one_completed_build(storage):
    await seed(storage)
    first, second = embedder(), embedder()
    generations = await asyncio.gather(
        ensure_embedding_collection(storage, "corpus", first, text),
        ensure_embedding_collection(storage, "corpus", second, text),
    )
    assert generations[0].physical_name == generations[1].physical_name
    assert first.embed_all.await_count + second.embed_all.await_count == 1


async def test_finalizer_metadata_survives_ready_transition(storage):
    await seed(storage)
    await storage.store_metadata("corpus", {"count": 1})

    async def finalize(target):
        await storage.store_metadata(target, {"count": 2, "rebuilt": True})

    generation = await ensure_embedding_collection(
        storage,
        "corpus",
        embedder(),
        text,
        finalize_candidate=finalize,
    )
    metadata = await storage.get_metadata(generation.physical_name)
    assert metadata["count"] == 2
    assert metadata["rebuilt"] is True
    assert metadata["embedding_generation"]["state"] == "ready"
    with pytest.raises(ValueError, match="cannot be overwritten"):
        await storage.store_metadata(generation.physical_name, {"embedding_generation": {}})


async def test_same_identity_client_rebinds_to_current_generation(storage):
    await seed(storage)
    first_client = embedder(model="A")
    first = await ensure_embedding_collection(storage, "corpus", first_client, text)
    await ensure_embedding_collection(storage, "corpus", embedder(model="B"), text)
    current = await ensure_embedding_collection(storage, "corpus", embedder(model="A"), text)
    rebound = await ensure_embedding_collection(storage, "corpus", first_client, text)
    assert rebound.physical_name == current.physical_name != first.physical_name


async def test_copy_indexes_preserves_custom_payload_schema(storage, monkeypatch):
    raw = await storage.get_client()
    params = TextIndexParams(
        type="text",
        tokenizer="whitespace",
        lowercase=False,
        min_token_len=2,
        max_token_len=40,
        on_disk=True,
    )
    schema = PayloadIndexInfo(data_type="text", params=params, points=2)
    monkeypatch.setattr(
        raw,
        "get_collection",
        AsyncMock(
            return_value=SimpleNamespace(
                payload_schema={"content": schema},
            )
        ),
    )
    create_index = AsyncMock()
    monkeypatch.setattr(raw, "create_payload_index", create_index)
    await _copy_indexes(storage, "source", "target", [])
    create_index.assert_awaited_once_with(
        collection_name="target",
        field_name="content",
        field_schema=params,
        wait=True,
    )


async def test_lock_is_task_reentrant_but_not_inherited_by_child(storage):
    entered = asyncio.Event()

    async def child():
        async with embedding_collection_lock(storage, "corpus"):
            entered.set()

    async with embedding_collection_lock(storage, "corpus"):
        async with embedding_collection_lock(storage, "corpus"):
            task = asyncio.create_task(child())
            await asyncio.sleep(0.05)
            assert not entered.is_set()
    await asyncio.wait_for(task, timeout=2)
    assert entered.is_set()


async def test_loopback_endpoint_aliases_share_the_same_lock(storage):
    storage.url = "http://localhost:6333/"
    other = QdrantStorage(url="http://127.0.0.1:6333")
    entered = asyncio.Event()

    async def child():
        async with embedding_collection_lock(other, "corpus"):
            entered.set()

    async with embedding_collection_lock(storage, "corpus"):
        task = asyncio.create_task(child())
        await asyncio.sleep(0.05)
        assert not entered.is_set()
    await asyncio.wait_for(task, timeout=2)
    assert entered.is_set()


async def test_cancelled_build_is_not_published(storage):
    await seed(storage)
    client = embedder()
    started = asyncio.Event()

    async def blocked(texts, **kwargs):
        started.set()
        await asyncio.Event().wait()

    client.embed_all.side_effect = blocked
    task = asyncio.create_task(ensure_embedding_collection(storage, "corpus", client, text))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await active_embedding_collection(storage, "corpus") == "corpus"
    recovered = await ensure_embedding_collection(storage, "corpus", embedder(), text)
    assert len(await records(storage, recovered.physical_name)) == 3


async def test_publication_finishes_before_cancellation_releases_operation_lock(
    storage, monkeypatch
):
    await seed(storage)
    raw = await storage.get_client()
    original_publish = raw.update_collection_aliases
    started = asyncio.Event()
    finish = asyncio.Event()

    async def delayed_publish(**kwargs):
        started.set()
        await finish.wait()
        return await original_publish(**kwargs)

    monkeypatch.setattr(raw, "update_collection_aliases", delayed_publish)
    task = asyncio.create_task(ensure_embedding_collection(storage, "corpus", embedder(), text))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    assert await active_embedding_collection(storage, "corpus") == "corpus"
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    target = await active_embedding_collection(storage, "corpus")
    assert target != "corpus"
    assert len(await records(storage, target)) == 3


async def test_resolved_identity_probes_once_and_rejects_mutation():
    client = embedder()
    identity = await client.resolve_identity()
    assert identity.dimension == 3
    assert await client.resolve_identity() == identity
    client.embed_single.assert_awaited_once()
    client.model = "changed"
    with pytest.raises(EmbeddingServiceError, match="configuration changed"):
        await client.resolve_identity()


def test_identity_is_stable_complete_and_normalizes_optional_namespace():
    base = EmbeddingIdentity("model", None, "http://example.invalid/", 3)
    same = EmbeddingIdentity("model", "", "http://example.invalid", 3)
    assert base == same
    assert EmbeddingIdentity.from_dict(base.to_dict()) == base
    assert base.fingerprint == same.fingerprint
    assert base.fingerprint != EmbeddingIdentity("model", "revision", base.endpoint, 3).fingerprint
