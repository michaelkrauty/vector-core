"""Regression tests for concurrent SQLite store initialization."""

import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.glossary.store import GlossaryStore
from vector_core.storage.hash_registry import HashRegistry
from vector_core.utils.sqlite import SQLiteConfig

STORE_TYPES = (GlossaryStore, FactStore, GlobalVocabulary, EmbeddingCache, HashRegistry)


def _open_and_write(store_type, db_path, barrier, worker):
    """Open a real store and persist a unique entry after a simultaneous start."""
    barrier.wait(timeout=15)
    kwargs = {"cache_path" if store_type is EmbeddingCache else "db_path": db_path}
    with store_type(**kwargs) as store:
        conn = store._get_conn()
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert conn.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        if store_type is GlossaryStore:
            store.create(term=f"term{worker}", expansion="Expansion", definition="Definition")
        elif store_type is FactStore:
            store.create(subject=f"subject{worker}", predicate="is", object_value="fact")
        elif store_type is GlobalVocabulary:
            conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)", (f"worker{worker}", "ok")
            )
            conn.commit()
        elif store_type is EmbeddingCache:
            store.set(f"content{worker}", [0.1, 0.2], "test")
        else:
            store.register(f"hash{worker}", uuid4(), f"path{worker}", "test")


def _try_open_and_write(store_type, db_path, barrier, worker):
    """Keep failure messages without retaining tracebacks and their open connections."""
    try:
        _open_and_write(store_type, db_path, barrier, worker)
    except Exception as error:
        return repr(error)
    return None


@pytest.mark.parametrize("store_type", STORE_TYPES)
def test_concurrent_thread_initialization(tmp_path, store_type):
    """Independent store instances can initialize the same fresh database."""
    workers = 12
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for round_id in range(3):
            db_path = tmp_path / f"threads{round_id}.db"
            barrier = threading.Barrier(workers)
            futures = [
                pool.submit(_try_open_and_write, store_type, db_path, barrier, worker)
                for worker in range(workers)
            ]
            # Retrieve every result, including constructor failures.
            errors = [future.result(timeout=20) for future in futures]
            errors = [error for error in errors if error is not None]
            assert not errors, errors


def _process_open_and_write(directory, barrier, worker, results):
    """Report child failures explicitly so none disappear with a child process."""
    for store_type in STORE_TYPES:
        db_path = directory / f"{store_type.__name__}.db"
        error = _try_open_and_write(store_type, db_path, barrier, worker)
        results.put((store_type.__name__, worker, error))


def test_concurrent_process_initialization(tmp_path):
    """Spawned processes can initialize and write to the same fresh database."""
    context = multiprocessing.get_context("spawn")
    workers = 4
    barrier = context.Barrier(workers)
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_open_and_write,
            args=(tmp_path, barrier, worker, results),
        )
        for worker in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=25) for _ in range(workers * len(STORE_TYPES))]
        for process in processes:
            process.join(timeout=5)
        assert all(process.exitcode == 0 for process in processes)
        assert sorted((name, worker) for name, worker, _ in outcomes) == sorted(
            (store_type.__name__, worker) for store_type in STORE_TYPES for worker in range(workers)
        )
        assert not [error for _, _, error in outcomes if error is not None], outcomes
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()
        results.join_thread()


def test_journal_mode_retries_busy_upgrade(tmp_path):
    """A reserved lock rejects a journal-mode upgrade without invoking the busy handler."""
    path = tmp_path / "reserved.db"
    with sqlite3.connect(path) as holder:
        holder.execute("CREATE TABLE probe (value)")
    holder = sqlite3.connect(path)
    holder.execute("BEGIN IMMEDIATE")
    started = threading.Event()

    def configure():
        with sqlite3.connect(path, timeout=0) as conn:
            started.set()
            SQLiteConfig(busy_timeout_ms=1000).apply(conn)
            assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
            assert conn.execute("PRAGMA busy_timeout").fetchone() == (1000,)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(configure)
            assert started.wait(timeout=5)
            time.sleep(0.05)
            holder.rollback()
            future.result(timeout=5)
    finally:
        holder.close()


@pytest.mark.parametrize("timeout_ms", [0, 100])
def test_journal_mode_busy_timeout_is_bounded(tmp_path, timeout_ms):
    """Persistent initialization contention respects the configured busy timeout."""
    path = tmp_path / "locked.db"
    holder = sqlite3.connect(path)
    conn = sqlite3.connect(path, timeout=10)
    try:
        holder.execute("CREATE TABLE probe (value)")
        holder.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError) as exc:
            SQLiteConfig(busy_timeout_ms=timeout_ms).apply(conn)
        elapsed = time.monotonic() - started
        assert exc.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert timeout_ms / 1000 <= elapsed < timeout_ms / 1000 + 0.5
        assert conn.execute("PRAGMA busy_timeout").fetchone() == (timeout_ms,)
    finally:
        conn.close()
        holder.close()
