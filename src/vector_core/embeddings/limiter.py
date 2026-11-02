"""Cross-process concurrency limiting for embedding HTTP requests."""

import asyncio
import fcntl
import hashlib
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

_POLL_INTERVAL = 0.05
_CLAIM_RETRY_INTERVAL = 0.001
_INTERACTIVE_LOCK = "interactive.lock"

_held_fds: set[int] = set()
_held_fds_lock = threading.Lock()


def _before_fork() -> None:
    _held_fds_lock.acquire()


def _after_fork_parent() -> None:
    _held_fds_lock.release()


def _after_fork_child() -> None:
    for fd in _held_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    _held_fds.clear()
    _held_fds_lock.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)


class GlobalRequestLimiter:
    """Bound concurrent requests across processes using stable flock slot files.

    Each endpoint/model scope directory holds ``capacity`` slot files. A request
    holds an exclusive ``flock`` on one slot for the duration of one HTTP
    attempt, and waiters poll without blocking the event loop.

    Interactive (query) requests go ahead of document requests. An interactive
    request that finds no free slot claims priority with a shared ``flock`` on
    the scope's ``interactive.lock`` before it polls for a slot, retrying within
    about a millisecond while a document scan holds that file. A document
    request tries for a slot only while it can take that file exclusively, so a
    slot freed while a query waits goes to the query. Requests already holding
    a slot are never interrupted and capacity is never exceeded. Document
    requests can starve while interactive requests keep arriving, and resume as
    soon as none is waiting.

    Every lock lives on an open file description, which the kernel releases
    when its last descriptor closes, so a process that exits or is killed while
    waiting or holding leaves nothing behind. Linux and macOS both give
    ``flock`` per-open-file-description semantics, so separate opens of one
    file conflict even inside a single process. The lock directory must be on
    a local filesystem: Linux emulates ``flock`` on NFS with per-process POSIX
    record locks, which would not separate waiters within one process.

    These guarantees hold only while the lock files stay in place. Deleting the
    scope directory or a lock file while any process uses it lets later
    requests lock a fresh file, so the limit can be exceeded and priority
    claims are lost; stop every process using the scope before removing it.
    """

    def __init__(self, capacity: int, scope: str, lock_dir: Path) -> None:
        if capacity < 0:
            raise ValueError("global request capacity must be non-negative")
        self.capacity = capacity
        scope_hash = hashlib.sha256(scope.encode()).hexdigest()
        self._scope_dir = lock_dir / scope_hash
        self._validate_capacity_manifest()

    def _validate_capacity_manifest(self) -> None:
        """Reject conflicting capacities for an existing endpoint/model scope."""
        if self.capacity == 0:
            return
        self._scope_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._scope_dir / "capacity.lock"
        capacity_path = self._scope_dir / "capacity"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            capacity_fd = os.open(capacity_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                raw = os.read(capacity_fd, 64).decode("ascii").strip()
                if raw:
                    try:
                        configured = int(raw)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Invalid embedding limiter capacity manifest: {capacity_path}"
                        ) from exc
                    if configured != self.capacity:
                        raise ValueError(
                            "embedding global concurrency differs from the existing "
                            f"scope ({self.capacity} != {configured}); stop every process "
                            f"using this backend and remove {self._scope_dir} before changing it"
                        )
                else:
                    os.write(capacity_fd, f"{self.capacity}\n".encode("ascii"))
                    os.fsync(capacity_fd)
            finally:
                os.close(capacity_fd)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _open_lock_file(self, name: str) -> int:
        return os.open(self._scope_dir / name, os.O_CREAT | os.O_RDWR, 0o600)

    def _try_acquire(self, *, interactive: bool = False) -> int | None:
        """Take a free slot without blocking; document requests yield to waiting queries."""
        self._scope_dir.mkdir(parents=True, exist_ok=True)
        with _held_fds_lock:
            gate_fd: int | None = None
            if not interactive:
                # Keep the gate exclusive for the whole slot scan so this attempt
                # is ordered entirely before or after any query's claim. The held
                # descriptor lock defers os.fork() until the gate is closed again.
                gate_fd = self._open_lock_file(_INTERACTIVE_LOCK)
                try:
                    fcntl.flock(gate_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(gate_fd)
                    return None
                except BaseException:
                    os.close(gate_fd)
                    raise
            try:
                for slot in range(self.capacity):
                    fd = self._open_lock_file(f"slot-{slot:04d}.lock")
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        _held_fds.add(fd)
                        return fd
                    except BlockingIOError:
                        os.close(fd)
                    except BaseException:
                        os.close(fd)
                        raise
            finally:
                if gate_fd is not None:
                    os.close(gate_fd)
        return None

    def _try_claim(self) -> int | None:
        """Mark an interactive request as waiting; None while a document attempt holds the gate."""
        with _held_fds_lock:
            fd = self._open_lock_file(_INTERACTIVE_LOCK)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return None
            except BaseException:
                os.close(fd)
                raise
            _held_fds.add(fd)
            return fd

    @staticmethod
    def _release(fd: int) -> None:
        with _held_fds_lock:
            _held_fds.discard(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @asynccontextmanager
    async def acquire(self, *, interactive: bool = False) -> AsyncIterator[None]:
        """Hold one slot for the block, waiting without blocking the event loop.

        ``interactive`` marks a latency-sensitive query request: while it waits,
        no document request for this scope takes a free slot. Every descriptor
        is released on cancellation and on every other exit path.
        """
        if self.capacity == 0:
            yield
            return

        fd: int | None = None
        claim_fd: int | None = None
        owner_pid = os.getpid()
        try:
            fd = self._try_acquire(interactive=interactive)
            while fd is None:
                if interactive and claim_fd is None:
                    claim_fd = self._try_claim()
                # An unregistered query retries almost at once: a document scan
                # holds the gate only briefly, and the query has no priority
                # until its claim succeeds.
                unclaimed = interactive and claim_fd is None
                await asyncio.sleep(_CLAIM_RETRY_INTERVAL if unclaimed else _POLL_INTERVAL)
                if os.getpid() != owner_pid:
                    # Resumed in a child forked from this event loop: the
                    # at-fork hook closed the inherited claim, so forget its
                    # number and keep waiting as the child's own request.
                    claim_fd = None
                    owner_pid = os.getpid()
                fd = self._try_acquire(interactive=interactive)
            if claim_fd is not None:
                claim, claim_fd = claim_fd, None
                self._release(claim)
            yield
        finally:
            # The at-fork child hook already closes inherited descriptors. An
            # inherited context can still unwind normally in the child; never
            # close its stale integers again because the numbers may have been
            # reused for unrelated resources by then.
            if os.getpid() == owner_pid:
                try:
                    if claim_fd is not None:
                        self._release(claim_fd)
                finally:
                    if fd is not None:
                        self._release(fd)
