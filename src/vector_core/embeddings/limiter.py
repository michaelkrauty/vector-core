"""Cross-process concurrency limiting for embedding HTTP requests."""

import asyncio
import fcntl
import hashlib
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

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
    """Bound concurrent requests across processes using stable flock slot files."""

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

    def _try_acquire(self) -> int | None:
        self._scope_dir.mkdir(parents=True, exist_ok=True)
        with _held_fds_lock:
            for slot in range(self.capacity):
                path = self._scope_dir / f"slot-{slot:04d}.lock"
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _held_fds.add(fd)
                    return fd
                except BlockingIOError:
                    os.close(fd)
                except BaseException:
                    os.close(fd)
                    raise
        return None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Wait without blocking the event loop, releasing a held slot on cancellation."""
        if self.capacity == 0:
            yield
            return

        fd: int | None = None
        owner_pid = os.getpid()
        try:
            while fd is None:
                fd = self._try_acquire()
                if fd is None:
                    await asyncio.sleep(0.05)
            yield
        finally:
            # The at-fork child hook already closes inherited descriptors. An
            # inherited context can still unwind normally in the child; never
            # close its stale integer again because the number may have been
            # reused for an unrelated resource by then.
            if fd is not None and os.getpid() == owner_pid:
                with _held_fds_lock:
                    _held_fds.discard(fd)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
