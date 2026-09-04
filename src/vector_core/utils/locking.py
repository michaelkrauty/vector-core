"""Cross-process file locking utilities.

This module provides file-based locking for coordinating access across
processes or async tasks. Useful for:
- Preventing concurrent indexing of the same codebase
- Protecting file read-modify-write operations
- Coordinating access to shared resources

Note: Uses POSIX fcntl.flock(). Linux and macOS are supported; Windows is not.

Usage:
    # Synchronous context manager
    from vector_core.utils.locking import file_lock

    with file_lock(path, timeout=10.0):
        # Exclusive access to path
        process_file(path)

    # Async context manager
    from vector_core.utils.locking import async_file_lock

    async with async_file_lock("indexing", timeout=60.0):
        await index_codebase()
"""

import asyncio
import logging
import sys
import threading

if sys.platform == "win32":
    raise ImportError(
        "vector-core's file locking requires POSIX fcntl, which is not available on Windows. "
        "Linux and macOS are supported."
    )

import fcntl
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TypeVar

from vector_core.settings import settings

logger = logging.getLogger(__name__)
R = TypeVar("R")

_held_lock_fds: set[int] = set()
_fork_condition = threading.Condition()


class _ForkState:
    active_opens = 0
    forking = False


_fork_state = _ForkState()


def _before_fork() -> None:
    _fork_condition.acquire()
    _fork_state.forking = True
    while _fork_state.active_opens:
        _fork_condition.wait()


def _after_fork_parent() -> None:
    _fork_state.forking = False
    _fork_condition.notify_all()
    _fork_condition.release()


def _after_fork_child() -> None:
    # Closing a duplicated flock descriptor preserves the parent's lock. Calling
    # LOCK_UN here would unlock the shared open-file description for both.
    for fd in _held_lock_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    _held_lock_fds.clear()
    _fork_state.active_opens = 0
    _fork_state.forking = False
    _fork_condition.notify_all()
    _fork_condition.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)

# Legacy age threshold retained for compatibility. Age never permits deletion.
STALE_LOCK_SECONDS = 3600.0


async def _run_worker_uninterrupted(
    fn: Callable[..., R],
    *args: object,
) -> tuple[R, asyncio.CancelledError | None]:
    """Finish one fd-using worker before propagating task cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(fn, *args))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
            return result, cancellation
        except asyncio.CancelledError as exc:
            if worker.done():
                if worker.cancelled():
                    raise
                try:
                    return worker.result(), exc
                except BaseException as worker_error:
                    raise exc from worker_error
            cancellation = exc
        except BaseException as worker_error:
            if cancellation is not None:
                raise cancellation from worker_error
            raise


def _open_lock_fd(path: Path) -> int:
    """Open and register a descriptor atomically with respect to fork."""
    with _fork_condition:
        while _fork_state.forking:
            _fork_condition.wait()
        _fork_state.active_opens += 1
    fd: int | None = None
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
        return fd
    finally:
        with _fork_condition:
            if fd is not None:
                _held_lock_fds.add(fd)
            _fork_state.active_opens -= 1
            _fork_condition.notify_all()


def _close_owned_lock(fd: int, owner_pid: int, *, unlock: bool) -> None:
    """Release only in the process that acquired this context."""
    if os.getpid() != owner_pid:
        return
    with _fork_condition:
        try:
            if unlock:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            _held_lock_fds.discard(fd)


def _is_lock_stale(lock_path: Path, max_age: float = STALE_LOCK_SECONDS) -> bool:
    """Report whether a lock file exceeds the legacy age threshold."""
    try:
        if not lock_path.exists():
            return False
        mtime = lock_path.stat().st_mtime
        age = time.time() - mtime
        return age > max_age
    except OSError:
        return False


def _break_stale_lock(lock_path: Path, max_age: float = STALE_LOCK_SECONDS) -> bool:
    """Retain a lock file as the stable inode for its lock namespace.

    An unlocked inode is not stale: the kernel releases ``flock`` when the last
    owning descriptor closes. Deleting the pathname is unsafe even after a
    successful non-blocking lock, because another process may already have
    opened that inode but not attempted ``flock`` yet. It could later lock the
    unlinked inode while a new arrival locks a replacement inode.

    Kept as a no-op for compatibility with callers of this private helper.
    """
    _ = lock_path, max_age
    return False


@contextmanager
def file_lock(
    path: Path,
    timeout: float | None = None,
    *,
    shared: bool = False,
) -> Iterator[None]:
    """
    Acquire a POSIX file lock (synchronous, blocking with timeout).

    Creates a .lock file adjacent to the target path. Uses POSIX flock()
    which is released automatically when the process exits or file
    descriptor is closed.

    Args:
        path: Path to file to lock (lock file created as path.lock)
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)
        shared: Acquire a shared/read lock instead of an exclusive/write lock

    Yields:
        None (lock is held while in context)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Example:
        with file_lock(Path("/path/to/file.txt")):
            # Exclusive access
            modify_file(path)
    """
    if timeout is None:
        timeout = settings.file_lock_timeout
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    owner_pid = os.getpid()
    fd = _open_lock_fd(lock_path)
    lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    acquired = False

    try:
        while True:
            try:
                fcntl.flock(fd, lock_mode | fcntl.LOCK_NB)
                acquired = True
                break  # Lock acquired
            except BlockingIOError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {path} within {timeout}s"
                    ) from None
                time.sleep(0.05)  # Brief sleep before retry

        logger.debug(f"Acquired lock on {lock_path}")
        try:
            yield
        finally:
            logger.debug(f"Released lock on {lock_path}")
    finally:
        _close_owned_lock(fd, owner_pid, unlock=acquired)


@asynccontextmanager
async def async_file_lock(
    name: str,
    timeout: float = 60.0,
    lock_dir: Path | None = None,
    *,
    shared: bool = False,
) -> AsyncIterator[None]:
    """
    Acquire a POSIX file lock (async, non-blocking with timeout).

    Uses thread pool executor to avoid blocking the event loop while
    waiting for lock acquisition.

    Args:
        name: Lock name (used as filename in lock_dir)
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)
        lock_dir: Directory for lock files (default: settings.cache_dir / "locks")
        shared: Acquire a shared/read lock instead of an exclusive/write lock

    Yields:
        None (lock is held while in context)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Example:
        async with async_file_lock("codebase_abc"):
            await index_codebase()
    """
    lock_dir = lock_dir or (settings.cache_dir / "locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"

    fd: int | None = None
    owner_pid = os.getpid()
    acquired = False

    try:
        fd, cancellation = await _run_worker_uninterrupted(
            _open_lock_fd,
            lock_path,
        )
        if cancellation is not None:
            raise cancellation
        assert fd is not None

        start_time = asyncio.get_running_loop().time()

        while True:
            acquired, cancellation = await _run_worker_uninterrupted(
                _try_lock,
                fd,
                shared,
            )
            if cancellation is not None:
                raise cancellation
            if acquired:
                acquired = True
                logger.debug(f"Acquired async lock: {name}")
                break
            if asyncio.get_running_loop().time() - start_time >= timeout:
                raise TimeoutError(f"Timeout waiting for lock: {name} ({timeout}s)")
            await asyncio.sleep(0.1)

        try:
            yield
        finally:
            logger.debug(f"Released async lock: {name}")

    finally:
        if fd is not None:
            try:
                _close_owned_lock(fd, owner_pid, unlock=acquired)
            except OSError:
                pass


def _try_lock(fd: int, shared: bool = False) -> bool:
    """Try to acquire a shared or exclusive lock without blocking."""
    try:
        lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(fd, lock_mode | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _release_lock(fd: int) -> None:
    """Release exclusive lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass  # Lock may already be released


class LockManager:
    """
    Manage multiple named locks with automatic cleanup.

    Useful for tracking active locks and ensuring cleanup on shutdown.

    Example:
        manager = LockManager()

        async with manager.acquire("project_a"):
            await process_project_a()

        # On shutdown
        await manager.release_all()
    """

    def __init__(self, lock_dir: Path | None = None):
        """
        Initialize lock manager.

        Args:
            lock_dir: Directory for lock files (default: settings.cache_dir / "locks")
        """
        self._lock_dir = lock_dir or (settings.cache_dir / "locks")
        self._active_locks: dict[str, int] = {}  # name -> fd
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, name: str, timeout: float = 60.0) -> AsyncIterator[None]:
        """
        Acquire named lock and track it.

        Args:
            name: Lock name
            timeout: Max seconds to wait

        Yields:
            None (lock is held while in context)
        """
        async with async_file_lock(name, timeout=timeout, lock_dir=self._lock_dir):
            async with self._lock:
                # Track that we hold this lock (for cleanup)
                self._active_locks[name] = 0  # fd already managed by context
            try:
                yield
            finally:
                async with self._lock:
                    self._active_locks.pop(name, None)

    async def is_locked(self, name: str) -> bool:
        """Check if a lock is currently held by this manager."""
        async with self._lock:
            return name in self._active_locks

    def cleanup(self, force: bool = False) -> int:
        """Retain every lock file and return zero.

        ``force`` is accepted for API compatibility but cannot make pathname
        deletion safe while another process may have the inode open.
        """
        _ = force
        return 0


def cleanup_stale_locks(lock_dir: Path | None = None) -> int:
    """Retain every lock file and return zero.

    The argument and return value remain for compatibility. Lock files are
    empty, bounded by the number of lock names used, and must keep a stable
    inode to preserve mutual exclusion.
    """
    _ = lock_dir
    return 0
