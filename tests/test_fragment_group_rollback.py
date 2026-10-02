"""Real local Qdrant mutations must be compensated before a failed writer returns."""

import asyncio
import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.api.points_api import jsonable_encoder
from qdrant_client.models import PointIdsList, PointsList, PointStruct, SparseVector, WriteOrdering

from vector_core.embeddings.client import EmbeddingClient
from vector_core.facts.indexer import FactIndexer
from vector_core.settings import settings
from vector_core.storage import embedding_migration as migration
from vector_core.storage import fragment_recovery as recovery
from vector_core.storage.embedding_fragments import (
    FRAGMENT_KEY,
    fragment_point,
    fragment_text,
    upsert_fragment_group,
)
from vector_core.storage.fragment_recovery import FragmentGroupRecoveryError
from vector_core.storage.qdrant import QdrantStorage


@pytest.fixture
async def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    storage = QdrantStorage(url="http://rollback-tests.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    await storage.create_collection("corpus", dense_dim=2)
    try:
        yield storage
    finally:
        await storage.close()


async def group(raw, point_id=7):
    embedder = EmbeddingClient(model="rollback-test", dim=2, max_input_bytes=40)
    embedder.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [
            [0.0, 1.0] if "TAILNEEDLE" in text else [1.0, 0.0] for text in texts
        ]
    )
    return await fragment_point(
        embedder,
        point_id=point_id,
        payload={"content": raw, "extra": {"raw": raw, "preserve": [True, 3]}},
        sparse=SparseVector(indices=[3, 77], values=[1.25, 2.5]),
        text=raw,
        vectorize=lambda text: SparseVector(indices=[91], values=[float(len(text))]),
    )


async def snapshot(storage):
    client = await storage.get_client()
    records, offset = await client.scroll(
        "corpus", limit=1000, with_payload=True, with_vectors=True
    )
    assert offset is None
    return deepcopy({record.id: (record.payload, record.vector) for record in records})


def journals():
    return list((settings.cache_dir / "embedding-fragment-recovery").glob("*.json"))


async def replacement(storage, monkeypatch, *, existing=True):
    old = await group("old alpha " * 35)
    if existing:
        await upsert_fragment_group(storage, "corpus", old)
    # Compensation must leave unrelated source points intact too.
    await upsert_fragment_group(storage, "corpus", await group("unrelated", point_id=8))
    assert journals() == []
    before = await snapshot(storage)
    new = await group("replacement beta " * 60 + "TAILNEEDLE")
    budget = max(len(jsonable_encoder(PointsList(points=[point])).encode()) for point in old + new)
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", budget)
    return old, new, before, budget


def assert_write(args, kwargs, budget):
    assert kwargs["wait"] is True
    assert kwargs["ordering"] == WriteOrdering.STRONG
    assert len(jsonable_encoder(PointsList(points=args[1])).encode()) <= budget


async def assert_new_coverage(storage, new, before):
    after = await snapshot(storage)
    assert set(after) == {8, *(point.id for point in new)}
    assert after[8] == before[8]
    assert after[7][0]["content"] == new[0].payload["content"]
    ordered = sorted(
        (payload for point_id, (payload, _) in after.items() if point_id != 8),
        key=lambda payload: payload[FRAGMENT_KEY]["index"],
    )
    assert "".join(fragment_text(payload) for payload in ordered) == new[0].payload["content"]
    client = await storage.get_client()
    hits = await client.query_points(
        "corpus", query=[0.0, 1.0], using="dense", score_threshold=0.99, limit=10
    )
    assert any("TAILNEEDLE" in fragment_text(hit.payload) for hit in hits.points)


@pytest.mark.parametrize("existing", [True, False], ids=["replace", "brand-new"])
async def test_later_batch_failure_restores_exact_group_and_retry_covers_tail(
    storage, monkeypatch, existing
):
    old, new, before, budget = await replacement(storage, monkeypatch, existing=existing)
    client = await storage.get_client()
    real_upsert = client.upsert
    calls = 0
    mutations = 0
    failure = RuntimeError("later batch rejected")

    async def fail_later_batch(*args, **kwargs):
        nonlocal calls, mutations
        assert_write(args, kwargs, budget)
        calls += 1
        if calls == 3:
            assert mutations == 2
            partial = await snapshot(storage)
            assert partial != before
            assert {point.id for point in new[1:]} & set(partial)
            raise failure
        result = await real_upsert(*args, **kwargs)
        mutations += 1
        return result

    monkeypatch.setattr(client, "upsert", fail_later_batch)
    with pytest.raises(RuntimeError) as caught:
        await upsert_fragment_group(storage, "corpus", new)
    assert caught.value is failure
    assert calls >= 3
    assert await snapshot(storage) == before
    assert journals() == []
    hits = await client.query_points(
        "corpus", query=[0.0, 1.0], using="dense", score_threshold=0.99, limit=10
    )
    assert hits.points == []
    monkeypatch.setattr(client, "upsert", real_upsert)
    await upsert_fragment_group(storage, "corpus", new)
    assert journals() == []
    await assert_new_coverage(storage, new, before)
    assert not {point.id for point in old[1:]} & set(await snapshot(storage))


async def test_cleanup_error_after_actual_delete_restores_old_payloads_and_all_vectors(
    storage, monkeypatch
):
    old, new, before, budget = await replacement(storage, monkeypatch)
    client = await storage.get_client()
    real_delete = client.delete
    real_upsert = client.upsert
    deleted = []
    failure = RuntimeError("cleanup response lost after partial delete")

    async def checked_upsert(*args, **kwargs):
        assert_write(args, kwargs, budget)
        return await real_upsert(*args, **kwargs)

    async def fail_after_partial_delete(*args, **kwargs):
        assert kwargs["wait"] is True
        assert kwargs["ordering"] == WriteOrdering.STRONG
        if not deleted:
            selector = kwargs["points_selector"]
            stale = selector.points if isinstance(selector, PointIdsList) else selector
            assert len(stale) > 1
            deleted.append(stale[0])
            await real_delete(
                args[0],
                points_selector=PointIdsList(points=deleted),
                wait=True,
                ordering=WriteOrdering.STRONG,
            )
            assert deleted[0] not in await snapshot(storage)
            raise failure
        return await real_delete(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", checked_upsert)
    monkeypatch.setattr(client, "delete", fail_after_partial_delete)
    with pytest.raises(RuntimeError) as caught:
        await upsert_fragment_group(storage, "corpus", new)
    assert caught.value is failure
    assert deleted[0] in {point.id for point in old[1:]}
    assert await snapshot(storage) == before
    assert journals() == []
    monkeypatch.setattr(client, "delete", real_delete)
    await upsert_fragment_group(storage, "corpus", new)
    assert journals() == []
    await assert_new_coverage(storage, new, before)


async def test_cancellation_settles_inflight_write_and_rollback_inside_caller_lock(
    storage, monkeypatch
):
    _, new, before, budget = await replacement(storage, monkeypatch)
    client = await storage.get_client()
    real_upsert = client.upsert
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()
    competitor_acquired = asyncio.Event()
    lock = asyncio.Lock()
    calls = 0

    async def blocked_second_write(*args, **kwargs):
        nonlocal calls
        assert_write(args, kwargs, budget)
        calls += 1
        if calls == 2:
            entered.set()
            await release.wait()
            result = await real_upsert(*args, **kwargs)
            settled.set()
            return result
        return await real_upsert(*args, **kwargs)

    async def writer():
        async with lock:
            await upsert_fragment_group(storage, "corpus", new)

    async def competitor():
        async with lock:
            competitor_acquired.set()
            assert settled.is_set()
            assert await snapshot(storage) == before

    monkeypatch.setattr(client, "upsert", blocked_second_write)
    task = asyncio.create_task(writer())
    competing_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert await snapshot(storage) != before
        competing_task = asyncio.create_task(competitor())
        task.cancel()
        # A loop barrier gives cancellation and the lock waiter a turn without polling.
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert not task.done()
        assert lock.locked()
        assert not competitor_acquired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.wait_for(competing_task, timeout=5)
        assert await snapshot(storage) == before
        assert journals() == []
    finally:
        release.set()
        await asyncio.gather(
            task, *([competing_task] if competing_task else []), return_exceptions=True
        )


@pytest.mark.parametrize("phase", ["create", "remove"])
async def test_journal_io_keeps_loop_responsive_and_settles_cancelled_workers(
    storage, monkeypatch, phase
):
    _, new, before, _ = await replacement(storage, monkeypatch)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    lock = asyncio.Lock()
    # Block actual serialization or journal removal, rather than substituting
    # an async mock that could conceal filesystem work on the event loop.
    owner, name = (recovery.json, "dump") if phase == "create" else (recovery, "_remove_journal")
    real_io = getattr(owner, name)

    def blocked_io(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=5), "event loop could not release the journal worker"
        try:
            return real_io(*args, **kwargs)
        finally:
            finished.set()

    async def writer():
        async with lock:
            await upsert_fragment_group(storage, "corpus", new)

    monkeypatch.setattr(owner, name, blocked_io)
    task = asyncio.create_task(writer())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        # An unrelated reader stays responsive while journal I/O is blocked.
        during_io = await asyncio.wait_for(snapshot(storage), timeout=1)
        if phase == "create":
            assert during_io == before
        else:
            assert during_io != before
        for _ in range(2):
            task.cancel()
            barrier = asyncio.Event()
            loop.call_soon(barrier.set)
            await barrier.wait()
            assert not task.done()
            assert lock.locked()
            assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
        assert not lock.locked()
        assert journals() == []
        if phase == "create":
            assert await snapshot(storage) == before
        else:
            # Cleanup starts only after the replacement has committed.
            await assert_new_coverage(storage, new, before)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_journal_serialization_failure_prevents_mutation_and_removes_partial_file(
    storage, monkeypatch
):
    _, new, before, _ = await replacement(storage, monkeypatch)
    client = await storage.get_client()
    upsert = AsyncMock(wraps=client.upsert)
    monkeypatch.setattr(client, "upsert", upsert)
    failure = OSError("journal disk failure")
    monkeypatch.setattr(recovery.json, "dump", Mock(side_effect=failure))
    with pytest.raises(OSError) as caught:
        await upsert_fragment_group(storage, "corpus", new)
    assert caught.value is failure
    upsert.assert_not_awaited()
    assert await snapshot(storage) == before
    assert journals() == []


async def test_double_failure_retains_private_complete_journal_without_replaying_later_edits(
    storage, monkeypatch
):
    _, new, before, budget = await replacement(storage, monkeypatch)
    client = await storage.get_client()
    real_upsert = client.upsert
    original = RuntimeError("replacement rejected")
    rollback = RuntimeError("compensation rejected")
    calls = 0
    retained = None

    async def reject_replacement_and_compensation(*args, **kwargs):
        nonlocal calls, retained
        assert_write(args, kwargs, budget)
        calls += 1
        if calls == 1:
            # The recovery evidence exists before any remote mutation can begin.
            paths = journals()
            assert len(paths) == 1
            retained = paths[0]
            assert retained.stat().st_mode & 0o777 == 0o600
            journal = json.loads(retained.read_text())
            assert journal["schema"] == 1
            assert journal["storage_scope"] == migration._storage_scope(storage.url)
            assert journal["collection"] == "corpus"
            assert journal["parent_id"] == 7
            assert set(journal["introduced_ids"]) == {point.id for point in new[1:]}
            previous = [PointStruct.model_validate(point) for point in journal["previous_points"]]
            assert {point.id: (point.payload, point.vector) for point in previous} == {
                point_id: record for point_id, record in before.items() if point_id != 8
            }
            assert await snapshot(storage) == before
        if calls == 3:
            assert await snapshot(storage) != before
            raise original
        if calls == 4:
            raise rollback
        return await real_upsert(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", reject_replacement_and_compensation)
    with pytest.raises(FragmentGroupRecoveryError) as caught:
        await upsert_fragment_group(storage, "corpus", new)
    assert retained is not None
    error = caught.value
    assert (error.original_error, error.rollback_error) == (original, rollback)
    assert error.__cause__ is rollback
    assert error.recovery_path == retained
    assert journals() == [retained]
    assert str(retained) in str(error)
    journal_bytes = retained.read_bytes()

    monkeypatch.setattr(client, "upsert", real_upsert)
    later = await group("later successful edit " * 15 + "TAILNEEDLE")
    await upsert_fragment_group(storage, "corpus", later)
    await assert_new_coverage(storage, later, before)
    # An obsolete failed-operation journal is evidence, never automatic replay input.
    assert journals() == [retained]
    assert retained.read_bytes() == journal_bytes
    await upsert_fragment_group(storage, "corpus", later)
    await assert_new_coverage(storage, later, before)
    assert retained.read_bytes() == journal_bytes


@pytest.mark.parametrize("recovery_required", [True, False], ids=["fatal", "compensated"])
async def test_fact_batch_propagates_recovery_required_and_stops_before_next_fact(
    storage, tmp_path, monkeypatch, recovery_required
):
    indexer = FactIndexer(
        fact_store=Mock(), storage=storage, global_vocab=Mock(), collection_name="corpus"
    )
    first, second = SimpleNamespace(id="first"), SimpleNamespace(id="second")
    monkeypatch.setattr(
        indexer, "_iter_fact_tokens", lambda: iter([(first, {"first"}), (second, {"second"})])
    )
    monkeypatch.setattr(indexer, "_get_indexed_fact_ids", AsyncMock(return_value=set()))
    original = RuntimeError("replacement rejected")
    failure = (
        FragmentGroupRecoveryError(tmp_path / "recovery.json", original, RuntimeError("rollback"))
        if recovery_required
        else original
    )
    write = AsyncMock(side_effect=[failure, None])
    monkeypatch.setattr(indexer, "_index_fact", write)
    if recovery_required:
        with pytest.raises(FragmentGroupRecoveryError) as caught:
            await indexer._index_all("corpus")
        assert caught.value is failure
        write.assert_awaited_once_with(first, "corpus")
    else:
        result = await indexer._index_all("corpus")
        assert result["indexed"] == 1
        assert write.await_count == 2
        assert write.await_args is not None
        assert write.await_args.args == (second, "corpus")
