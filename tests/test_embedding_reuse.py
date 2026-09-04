"""Persistent embedding reuse and global request coordination tests."""

import asyncio
import os
import signal
import sqlite3
import struct
import sys
import threading
from contextlib import asynccontextmanager
from multiprocessing import Process
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vector_core.embeddings import limiter as limiter_module
from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.client import EmbeddingClient, EmbeddingServiceError
from vector_core.embeddings.limiter import GlobalRequestLimiter
from vector_core.settings import VectorCoreSettings


def _initialize_shared_cache(path: Path, index: int) -> None:
    cache = EmbeddingCache(path)
    cache.set_many({f"key-{index}": [float(index), 1.0]}, expected_dim=2)
    cache.close()


def test_global_concurrency_setting_allows_zero_but_not_negative() -> None:
    assert VectorCoreSettings(embedding_global_concurrency=0).embedding_global_concurrency == 0
    with pytest.raises(ValueError, match="embedding_global_concurrency"):
        VectorCoreSettings(embedding_global_concurrency=-1)


def test_cache_key_covers_every_embedding_identity_field() -> None:
    base = {
        "text": "effective text",
        "namespace": "deployment-a",
        "model": "model-a",
        "dim": 2,
    }
    key = EmbeddingCache.make_key(**base)

    assert EmbeddingCache.make_key(**base) == key
    for changed in (
        {**base, "text": "other text"},
        {**base, "namespace": "deployment-b"},
        {**base, "model": "model-b"},
        {**base, "dim": 3},
        {**base, "preprocessing_version": "truncate-v2"},
    ):
        assert EmbeddingCache.make_key(**changed) != key


def test_bulk_cache_uses_float32_and_rejects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.db"
    cache = EmbeddingCache(path)
    cache.set_many({"a": [0.25, -0.5], "b": [1.0, 2.0]}, expected_dim=2)

    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT cache_key, typeof(embedding), length(embedding) "
        "FROM embedding_cache_v2 ORDER BY cache_key"
    ).fetchall()
    assert rows == [("a", "blob", 8), ("b", "blob", 8)]
    conn.execute(
        "INSERT INTO embedding_cache_v2 VALUES (?, ?, ?, ?, ?)",
        ("bad", struct.pack("<2f", float("nan"), 1.0), 2, "now", "now"),
    )
    conn.commit()
    conn.close()

    assert cache.get_many(["b", "bad", "a"], expected_dim=2) == {
        "a": [0.25, -0.5],
        "b": [1.0, 2.0],
    }
    conn = sqlite3.connect(path)
    assert (
        conn.execute("SELECT COUNT(*) FROM embedding_cache_v2 WHERE cache_key = 'bad'").fetchone()[
            0
        ]
        == 0
    )
    conn.close()
    cache.close()


def test_bulk_cache_write_validates_every_vector_before_transaction(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    with pytest.raises(ValueError, match="expected 2 finite values"):
        cache.set_many(
            {"would-be-valid": [1.0, 2.0], "invalid": [float("nan"), 2.0]},
            expected_dim=2,
        )

    assert cache.get_many(["would-be-valid"], expected_dim=2) == {}
    cache.close()


def test_concurrent_processes_initialize_one_cache(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.db"
    processes = [Process(target=_initialize_shared_cache, args=(path, index)) for index in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    cache = EmbeddingCache(path)
    assert len(cache.get_many([f"key-{index}" for index in range(6)], expected_dim=2)) == 6
    cache.close()


def test_bulk_cache_ends_read_transaction_before_access_updates(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    cache.set_many({"key": [1.0, 2.0]}, expected_dim=2)
    statements: list[str] = []
    cache._get_conn().set_trace_callback(statements.append)

    assert cache.get_many(["key"], expected_dim=2) == {"key": [1.0, 2.0]}

    read_commit = statements.index("COMMIT")
    update = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("UPDATE embedding_cache_v2")
    )
    assert read_commit < update
    cache.close()


@pytest.mark.asyncio
async def test_embed_all_caches_effective_input_and_scatters_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vector_core.embeddings.client.settings.embedding_max_text_chars", 3)
    path = tmp_path / "embeddings.db"
    first = EmbeddingClient(model="model-a", dim=2, cache_namespace="deployment-a", cache_path=path)
    first.embed_batch = AsyncMock(
        side_effect=lambda texts: [[float(ord(text[0])), 1.0] for text in texts]
    )

    result = await first.embed_all(["abcdef", "z", "abc-other", "z"])

    first.embed_batch.assert_awaited_once_with(["abc", "z"])
    assert result == [[97.0, 1.0], [122.0, 1.0], [97.0, 1.0], [122.0, 1.0]]
    await first.close()

    second = EmbeddingClient(
        model="model-a", dim=2, cache_namespace="deployment-a", cache_path=path
    )
    second.embed_batch = AsyncMock(side_effect=AssertionError("cache miss"))
    progress: list[tuple[int, int]] = []
    cached = await second.embed_all(
        ["abc-new suffix", "z"], progress_cb=lambda done, total: progress.append((done, total))
    )
    assert cached == [[97.0, 1.0], [122.0, 1.0]]
    assert progress == [(2, 2)]
    second.embed_batch.assert_not_awaited()
    await second.close()


@pytest.mark.asyncio
async def test_auto_dimension_deduplicates_before_first_request(tmp_path: Path) -> None:
    client = EmbeddingClient(
        model="model-a",
        dim=0,
        cache_namespace="deployment-a",
        cache_path=tmp_path / "embeddings.db",
    )

    async def embed_batch(texts: list[str]) -> list[list[float]]:
        client.dim = 2
        return [[float(ord(text[0])), 1.0] for text in texts]

    client.embed_batch = AsyncMock(side_effect=embed_batch)
    progress: list[tuple[int, int]] = []

    result = await client.embed_all(
        ["alpha", "beta", "alpha"],
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    client.embed_batch.assert_awaited_once_with(["alpha", "beta"])
    assert result == [[97.0, 1.0], [98.0, 1.0], [97.0, 1.0]]
    assert progress == [(3, 3)]
    await client.close()


def test_cache_entry_limit_applies_across_legacy_and_v2_tables(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.db", max_entries=3)
    cache.set("legacy-a", [1.0, 2.0])
    cache.set("legacy-b", [1.0, 2.0])
    cache.set_many(
        {"v2-a": [1.0, 2.0], "v2-b": [1.0, 2.0]},
        expected_dim=2,
    )

    conn = cache._get_conn()
    legacy = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    v2 = conn.execute("SELECT COUNT(*) FROM embedding_cache_v2").fetchone()[0]
    assert legacy + v2 <= 3
    assert v2 == 2
    cache.close()


@pytest.mark.asyncio
async def test_unset_namespace_never_opens_persistent_cache(tmp_path: Path) -> None:
    client = EmbeddingClient(dim=2, cache_namespace=None, cache_path=tmp_path / "bad")
    client.embed_batch = AsyncMock(return_value=[[1.0, 2.0]])

    with patch("vector_core.embeddings.client.EmbeddingCache") as cache_type:
        assert await client.embed_all(["text"]) == [[1.0, 2.0]]
    cache_type.assert_not_called()


@pytest.mark.asyncio
async def test_cache_read_error_fails_open(tmp_path: Path) -> None:
    client = EmbeddingClient(dim=2, cache_namespace="deployment", cache_path=tmp_path / "cache.db")
    cache = await client._get_embedding_cache()
    assert cache is not None
    cache.get_many = MagicMock(side_effect=sqlite3.DatabaseError("broken"))
    client.embed_batch = AsyncMock(
        side_effect=lambda texts: [[float(index), 1.0] for index, _ in enumerate(texts)]
    )

    assert await client.embed_all(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert client._persistent_cache_failed is True
    await client.close()


@pytest.mark.asyncio
async def test_cache_initialization_does_not_block_the_event_loop(
    tmp_path: Path,
) -> None:
    client = EmbeddingClient(
        dim=2,
        cache_namespace="deployment",
        cache_path=tmp_path / "cache.db",
    )
    started = threading.Event()
    release = threading.Event()

    class SlowCache(EmbeddingCache):
        def __init__(self, *args, **kwargs):
            started.set()
            release.wait(timeout=2.0)
            super().__init__(*args, **kwargs)

    with patch("vector_core.embeddings.client.EmbeddingCache", SlowCache):
        task = asyncio.create_task(client._get_embedding_cache())
        assert await asyncio.to_thread(started.wait, 1.0)
        ticked = False

        async def ticker() -> None:
            nonlocal ticked
            await asyncio.sleep(0)
            ticked = True

        await ticker()
        assert ticked
        release.set()
        assert await task is not None
    await client.close()


@pytest.mark.asyncio
async def test_close_cancellation_still_closes_cache_and_resets_state(
    tmp_path: Path,
) -> None:
    client = EmbeddingClient(dim=2)
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    cache._get_conn()
    client._embedding_cache = cache
    client._circuit_failure_count = 3
    client._circuit_open_until = 123.0
    started = asyncio.Event()

    class SlowHttpClient:
        async def aclose(self) -> None:
            started.set()
            await asyncio.Event().wait()

    client._client = SlowHttpClient()  # type: ignore[assignment]
    client._client_loop = asyncio.get_running_loop()
    task = asyncio.create_task(client.close())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client._client is None
    assert client._embedding_cache is None
    assert cache._all_conns == {}
    assert client._circuit_failure_count == 0
    assert client._circuit_open_until is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([{"index": 0, "embedding": [1.0, 2.0]}], "count"),
        (
            [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 1, "embedding": [float("inf"), 2.0]},
            ],
            "non-finite",
        ),
        (
            [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 1, "embedding": [10**400, 2.0]},
            ],
            "invalid response",
        ),
    ],
)
async def test_embed_batch_rejects_invalid_responses(data: list[dict], message: str) -> None:
    client = EmbeddingClient(dim=2)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": data}
    response.raise_for_status.return_value = None
    http = AsyncMock()
    http.post.return_value = response
    client._get_client = AsyncMock(return_value=http)

    with pytest.raises(EmbeddingServiceError, match=message):
        await client.embed_batch(["a", "b"])


@pytest.mark.asyncio
async def test_invalid_auto_dimension_response_does_not_poison_client() -> None:
    client = EmbeddingClient(dim=0)
    invalid = MagicMock()
    invalid.status_code = 200
    invalid.json.return_value = {
        "data": [
            {"index": 0, "embedding": [1.0, 2.0]},
            {"index": 1, "embedding": [1.0, 2.0, 3.0]},
        ]
    }
    valid = MagicMock()
    valid.status_code = 200
    valid.json.return_value = {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
    http = AsyncMock()
    http.post.side_effect = [invalid, valid]
    client._get_client = AsyncMock(return_value=http)

    with pytest.raises(EmbeddingServiceError, match="dimension"):
        await client.embed_batch(["a", "b"])
    assert client.dim == 0
    assert await client.embed_batch(["valid"]) == [[1.0, 2.0, 3.0]]
    assert client.dim == 3


def test_cache_namespaces_do_not_split_backend_limiter_scope(tmp_path: Path) -> None:
    first = EmbeddingClient(
        base_url="http://backend", model="model", cache_namespace="cache-a", limiter_dir=tmp_path
    )
    second = EmbeddingClient(
        base_url="http://backend", model="model", cache_namespace="cache-b", limiter_dir=tmp_path
    )
    other_model = EmbeddingClient(
        base_url="http://backend", model="other", cache_namespace="cache-a", limiter_dir=tmp_path
    )

    assert first._request_limiter._scope_dir == second._request_limiter._scope_dir
    assert first._request_limiter._scope_dir != other_model._request_limiter._scope_dir

    other_endpoint = EmbeddingClient(
        base_url="http://other-backend",
        model="model",
        dim=2,
        cache_namespace="cache-a",
        limiter_dir=tmp_path,
    )
    first.dim = 2
    assert first._persistent_cache_key("text") != other_endpoint._persistent_cache_key("text")


def test_limiter_closes_descriptor_on_unexpected_flock_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = GlobalRequestLimiter(1, "scope", tmp_path)
    opened: list[int] = []
    real_open = limiter_module.os.open

    def recording_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(limiter_module.os, "open", recording_open)
    monkeypatch.setattr(
        limiter_module.fcntl,
        "flock",
        MagicMock(side_effect=OSError("unexpected")),
    )

    with pytest.raises(OSError, match="unexpected"):
        limiter._try_acquire()
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_limiter_rejects_mixed_capacities_for_one_scope(tmp_path: Path) -> None:
    GlobalRequestLimiter(2, "shared-scope", tmp_path)
    GlobalRequestLimiter(2, "shared-scope", tmp_path)

    with pytest.raises(ValueError, match="differs from the existing scope"):
        GlobalRequestLimiter(3, "shared-scope", tmp_path)
    # Zero explicitly disables coordination and remains a backwards-compatible
    # opt-out even when enabled clients have established a manifest.
    GlobalRequestLimiter(0, "shared-scope", tmp_path)


@pytest.mark.asyncio
async def test_limiter_wraps_each_retry_attempt_not_retry_sleep() -> None:
    client = EmbeddingClient(dim=2, global_concurrency=1)
    attempts: list[str] = []

    @asynccontextmanager
    async def acquire():
        attempts.append("enter")
        try:
            yield
        finally:
            attempts.append("exit")

    client._request_limiter.acquire = acquire
    response = httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        request=httpx.Request("POST", "http://example.test/v1/embeddings"),
    )
    http = AsyncMock()
    http.post.side_effect = [httpx.ConnectError("one"), response]
    client._get_client = AsyncMock(return_value=http)

    with patch("vector_core.utils.retry.asyncio.sleep", new=AsyncMock()) as sleep:
        assert await client.embed_batch(["a"]) == [[1.0, 2.0]]

    assert attempts == ["enter", "exit", "enter", "exit"]
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_limiter_is_cancellation_safe_and_retains_slot_file(tmp_path: Path) -> None:
    first = GlobalRequestLimiter(1, "scope", tmp_path)
    second = GlobalRequestLimiter(1, "scope", tmp_path)

    async with first.acquire():
        waiter = asyncio.create_task(second.acquire().__aenter__())
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    async with second.acquire():
        pass
    assert len(list(tmp_path.rglob("slot-*.lock"))) == 1


@pytest.mark.asyncio
async def test_limiter_coordinates_with_another_process(tmp_path: Path) -> None:
    limiter = GlobalRequestLimiter(1, "shared-scope", tmp_path)
    code = """
import asyncio
import sys
from pathlib import Path
from vector_core.embeddings.limiter import GlobalRequestLimiter

async def main():
    print('ready', flush=True)
    async with GlobalRequestLimiter(1, 'shared-scope', Path(sys.argv[1])).acquire():
        print('acquired', flush=True)

asyncio.run(main())
"""
    async with limiter.acquire():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        assert await process.stdout.readline() == b"ready\n"
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(process.stdout.readline(), timeout=0.2)

    assert process.stdout is not None
    assert await asyncio.wait_for(process.stdout.readline(), timeout=2.0) == b"acquired\n"
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_limiter_closes_inherited_slot_descriptor_after_fork(tmp_path: Path) -> None:
    limiter = GlobalRequestLimiter(1, "fork-scope", tmp_path)
    child_pid: int | None = None
    in_child = False
    try:
        async with limiter.acquire():
            child_pid = os.fork()
            if child_pid == 0:
                in_child = True

        if in_child:
            # Normal context unwinding in the child must not close a reused fd
            # number after the at-fork hook closed the inherited slot.
            probe = os.open(tmp_path / "child-probe", os.O_CREAT | os.O_RDWR, 0o600)
            os.fstat(probe)
            os.close(probe)
            os._exit(0)

        async with asyncio.timeout(0.5):
            async with GlobalRequestLimiter(1, "fork-scope", tmp_path).acquire():
                pass
    finally:
        if child_pid and not in_child:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            await asyncio.to_thread(os.waitpid, child_pid, 0)
