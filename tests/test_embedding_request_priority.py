"""Query requests go ahead of document requests at the global embedding limit."""

import asyncio
import fcntl
import os
import random
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from vector_core.embeddings import limiter as limiter_module
from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.limiter import GlobalRequestLimiter

# Children load the limiter file directly: it only needs the standard library,
# and skipping the package import keeps each child start in the tens of ms.
_CHILD_PRELUDE = """
import asyncio
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("limiter", sys.argv[1])
limiter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limiter)
lock_dir = Path(sys.argv[2])
"""


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limiter_module, "_POLL_INTERVAL", 0.001)


def _query_waiting(scope_dir: Path) -> bool:
    """Probe the interactive gate the way a document request does."""
    path = scope_dir / "interactive.lock"
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _slot_free(scope_dir: Path) -> bool:
    fd = os.open(scope_dir / "slot-0000.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.001)


def _gate(limiter: GlobalRequestLimiter, is_open: Callable[[], bool]) -> None:
    """Hold back a limiter's slot attempts (not its claim) until ``is_open()``."""
    attempt = limiter._try_acquire

    def gated(*, interactive: bool = False) -> int | None:
        return attempt(interactive=interactive) if is_open() else None

    limiter._try_acquire = gated  # type: ignore[method-assign]


async def _spawn(code: str, lock_dir: Path) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CHILD_PRELUDE + code,
        str(limiter_module.__file__),
        str(lock_dir),
        stdout=asyncio.subprocess.PIPE,
    )


_DOCUMENT_CHILD = """
async def main():
    document = limiter.GlobalRequestLimiter(1, "shared-scope", lock_dir)
    attempt = document._try_acquire
    reported = []

    def first_attempt(*, interactive=False):
        fd = attempt(interactive=interactive)
        if not reported and fd is None:
            print("waiting", flush=True)
        reported.append(fd)
        return fd

    document._try_acquire = first_attempt
    async with document.acquire():
        print("acquired", flush=True)

asyncio.run(main())
"""

_QUERY_CHILD = """
async def main():
    query = limiter.GlobalRequestLimiter(1, "shared-scope", lock_dir)
    async with query.acquire(interactive=True):
        print("acquired", flush=True)

asyncio.run(main())
"""


async def test_waiting_query_takes_released_slot_before_document_waiters(
    tmp_path: Path,
) -> None:
    held_before = set(limiter_module._held_fds)
    holder = GlobalRequestLimiter(1, "scope", tmp_path)
    query = GlobalRequestLimiter(1, "scope", tmp_path)
    documents = [GlobalRequestLimiter(1, "scope", tmp_path) for _ in range(3)]
    order: list[str] = []
    released = False
    query_may_attempt = False
    attempts_after_release = [0] * len(documents)
    leave_query = asyncio.Event()

    _gate(query, lambda: query_may_attempt)
    for index, document in enumerate(documents):
        attempt = document._try_acquire

        def counted(
            *, interactive: bool = False, index: int = index, attempt=attempt
        ) -> int | None:
            if released:
                attempts_after_release[index] += 1
            return attempt(interactive=interactive)

        document._try_acquire = counted  # type: ignore[method-assign]

    async def request(name: str, limiter: GlobalRequestLimiter, *, interactive: bool) -> None:
        async with limiter.acquire(interactive=interactive):
            order.append(name)
            if interactive:
                await leave_query.wait()
                order.append("query released")

    hold = holder.acquire()
    await hold.__aenter__()
    waiters = [
        asyncio.create_task(request(f"document-{index}", document, interactive=False))
        for index, document in enumerate(documents)
    ]
    waiters.append(asyncio.create_task(request("query", query, interactive=True)))
    await _until(lambda: _query_waiting(holder._scope_dir))
    await hold.__aexit__(None, None, None)
    released = True

    # Every document waiter tries the free slot while the query has not polled yet.
    await _until(lambda: all(attempts_after_release))
    assert order == []
    query_may_attempt = True
    await _until(lambda: bool(order))
    assert order == ["query"]
    leave_query.set()
    await asyncio.gather(*waiters)

    assert order[:2] == ["query", "query released"]
    assert sorted(order[2:]) == ["document-0", "document-1", "document-2"]
    assert not _query_waiting(holder._scope_dir)
    assert limiter_module._held_fds == held_before


async def test_query_goes_ahead_of_document_waiting_in_another_process(tmp_path: Path) -> None:
    holder = GlobalRequestLimiter(1, "shared-scope", tmp_path)
    query = GlobalRequestLimiter(1, "shared-scope", tmp_path)
    query_may_attempt = False
    query_entered = asyncio.Event()
    leave_query = asyncio.Event()
    _gate(query, lambda: query_may_attempt)

    async def wait_as_query() -> None:
        async with query.acquire(interactive=True):
            query_entered.set()
            await leave_query.wait()

    async with holder.acquire():
        query_task = asyncio.create_task(wait_as_query())
        await _until(lambda: _query_waiting(holder._scope_dir))
    # The slot is free and the query has claimed priority but not polled yet,
    # so the child's first attempt finds a free slot.
    process = await _spawn(_DOCUMENT_CHILD, tmp_path)
    assert process.stdout is not None
    assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"waiting\n"
    query_may_attempt = True
    await asyncio.wait_for(query_entered.wait(), timeout=10)
    leave_query.set()
    await query_task
    assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"acquired\n"
    assert await process.wait() == 0


async def test_killed_query_waiter_does_not_block_document_requests(tmp_path: Path) -> None:
    holder = GlobalRequestLimiter(1, "shared-scope", tmp_path)
    async with holder.acquire():
        process = await _spawn(_QUERY_CHILD, tmp_path)
        try:
            await _until(lambda: _query_waiting(holder._scope_dir))
        finally:
            process.kill()
            await process.wait()
        assert not _query_waiting(holder._scope_dir)

    async with asyncio.timeout(10):
        async with GlobalRequestLimiter(1, "shared-scope", tmp_path).acquire():
            pass
    assert process.stdout is not None
    assert await process.stdout.read() == b""


async def test_cancelled_query_waiter_releases_its_claim(tmp_path: Path) -> None:
    held_before = set(limiter_module._held_fds)
    holder = GlobalRequestLimiter(1, "scope", tmp_path)
    query = GlobalRequestLimiter(1, "scope", tmp_path)

    async with holder.acquire():
        waiter = asyncio.create_task(query.acquire(interactive=True).__aenter__())
        await _until(lambda: _query_waiting(holder._scope_dir))
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not _query_waiting(holder._scope_dir)

    assert limiter_module._held_fds == held_before
    async with asyncio.timeout(10):
        async with GlobalRequestLimiter(1, "scope", tmp_path).acquire():
            pass


async def test_cancelled_document_waiter_leaves_nothing_held(tmp_path: Path) -> None:
    held_before = set(limiter_module._held_fds)
    holder = GlobalRequestLimiter(1, "scope", tmp_path)
    document = GlobalRequestLimiter(1, "scope", tmp_path)
    attempts: list[int | None] = []
    attempt = document._try_acquire

    def counted(*, interactive: bool = False) -> int | None:
        attempts.append(fd := attempt(interactive=interactive))
        return fd

    document._try_acquire = counted  # type: ignore[method-assign]

    async with holder.acquire():
        waiter = asyncio.create_task(document.acquire().__aenter__())
        await _until(lambda: len(attempts) >= 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert len(limiter_module._held_fds) == len(held_before) + 1
        assert not _query_waiting(holder._scope_dir)

    assert attempts and all(fd is None for fd in attempts)
    assert limiter_module._held_fds == held_before
    assert _slot_free(holder._scope_dir)
    assert not _query_waiting(holder._scope_dir)


async def test_capacity_is_never_exceeded_under_mixed_priorities(tmp_path: Path) -> None:
    capacity = 2
    rng = random.Random(20261101)
    active = 0
    peak = 0
    served: list[bool] = []

    async def request(interactive: bool, delay: int, hold: int) -> None:
        nonlocal active, peak
        for _ in range(delay):
            await asyncio.sleep(0)
        async with GlobalRequestLimiter(capacity, "scope", tmp_path).acquire(
            interactive=interactive
        ):
            active += 1
            peak = max(peak, active)
            assert active <= capacity
            for _ in range(hold):
                await asyncio.sleep(0)
            active -= 1
            served.append(interactive)

    plan = [(rng.random() < 0.35, rng.randrange(20), rng.randrange(1, 5)) for _ in range(30)]
    async with asyncio.timeout(10):
        await asyncio.gather(*(request(*args) for args in plan))

    assert 1 <= peak <= capacity
    assert sorted(served) == sorted(interactive for interactive, _, _ in plan)
    assert not _query_waiting(next(tmp_path.iterdir()))


def _embedding_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        request=httpx.Request("POST", "http://example.test/v1/embeddings"),
    )


def _recording_client(tmp_path: Path) -> tuple[EmbeddingClient, list[bool]]:
    client = EmbeddingClient(
        dim=2,
        batch_size=1,
        global_concurrency=0,
        limiter_dir=tmp_path / "locks",
        cache_path=tmp_path / "cache.db",
        cache_namespace="",
    )
    priorities: list[bool] = []

    @asynccontextmanager
    async def acquire(*, interactive: bool = False):
        priorities.append(interactive)
        yield

    client._request_limiter.acquire = acquire  # type: ignore[method-assign]
    http = AsyncMock()
    http.post.side_effect = lambda *args, **kwargs: _embedding_response()
    client._get_client = AsyncMock(return_value=http)  # type: ignore[method-assign]
    return client, priorities


@pytest.mark.parametrize("role", ["query", "document"])
@pytest.mark.parametrize(
    ("method", "requests"),
    [("embed_batch", 3), ("embed_all", 3), ("embed_single", 1), ("embed_single_cached", 1)],
)
async def test_client_request_priority_follows_role(
    tmp_path: Path, method: str, requests: int, role: str
) -> None:
    client, priorities = _recording_client(tmp_path)
    call = getattr(client, method)
    if requests > 1:
        await call(["a", "b", "c"], role=role)
    else:
        await call("a", role=role)

    assert priorities == [role == "query"] * requests
    await client.close()


async def test_identity_probe_takes_query_priority_without_leaking_it(tmp_path: Path) -> None:
    client, priorities = _recording_client(tmp_path)

    await client.resolve_identity()
    await client.embed_batch(["document"])

    assert priorities == [True, False]
    await client.close()


async def test_zero_capacity_does_no_file_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr(
        limiter_module.fcntl, "flock", MagicMock(side_effect=AssertionError("file work"))
    )
    limiter = GlobalRequestLimiter(0, "scope", lock_dir)

    for interactive in (False, True):
        async with limiter.acquire(interactive=interactive):
            pass

    assert not lock_dir.exists()


@pytest.mark.parametrize(
    ("attempt", "failing_call"),
    [("document", 0), ("document", 1), ("query", 0), ("claim", 0)],
)
def test_unexpected_flock_error_closes_every_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt: str, failing_call: int
) -> None:
    limiter = GlobalRequestLimiter(1, "scope", tmp_path)
    held_before = set(limiter_module._held_fds)
    opened: list[int] = []
    calls = 0
    real_open = os.open
    real_flock = fcntl.flock

    def recording_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def failing_flock(fd: int, operation: int) -> None:
        nonlocal calls
        calls += 1
        if calls - 1 == failing_call:
            raise OSError("unexpected")
        real_flock(fd, operation)

    monkeypatch.setattr(limiter_module.os, "open", recording_open)
    monkeypatch.setattr(limiter_module.fcntl, "flock", failing_flock)
    run = {
        "document": limiter._try_acquire,
        "query": lambda: limiter._try_acquire(interactive=True),
        "claim": limiter._try_claim,
    }[attempt]

    with pytest.raises(OSError, match="unexpected"):
        run()
    assert len(opened) == failing_call + 1
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert limiter_module._held_fds == held_before


class _ChildOs:
    """The ``os`` module as seen from a forked child: same calls, different pid."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

    def getpid(self) -> int:
        return self._pid

    def __getattr__(self, name: str):
        return getattr(os, name)


async def test_query_waiter_resumed_after_fork_never_closes_inherited_claim_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = GlobalRequestLimiter(1, "scope", tmp_path)
    query = GlobalRequestLimiter(1, "scope", tmp_path)
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def wait_as_query() -> None:
        async with query.acquire(interactive=True):
            entered.set()
            await leave.wait()

    hold = holder.acquire()
    await hold.__aenter__()
    waiter = asyncio.create_task(wait_as_query())
    await _until(lambda: _query_waiting(holder._scope_dir))
    inherited = set(limiter_module._held_fds)
    # Run the at-fork hooks as a child that keeps driving this event loop would.
    limiter_module._before_fork()
    limiter_module._after_fork_child()
    monkeypatch.setattr(limiter_module, "os", _ChildOs(os.getpid() + 1))
    probes: list[int] = []
    try:
        while not inherited <= set(probes):
            assert len(probes) < 1024
            probes.append(os.open(tmp_path / f"probe-{len(probes)}", os.O_CREAT | os.O_RDWR))

        await asyncio.wait_for(entered.wait(), timeout=10)
        leave.set()
        await waiter
        await hold.__aexit__(None, None, None)
        for fd in probes:
            os.fstat(fd)
        assert limiter_module._held_fds == set()
    finally:
        for fd in probes:
            try:
                os.close(fd)
            except OSError:
                pass
