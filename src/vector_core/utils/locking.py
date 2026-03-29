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

if sys.platform == "win32":
    raise ImportError(
        "vector-core's file locking requires POSIX fcntl, which is not available on Windows. "
        "Linux and macOS are supported."
    )

import fcntl
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from vector_core.settings import settings

logger = logging.getLogger(__name__)

# Stale lock threshold (1 hour) - locks older than this are considered abandoned
STALE_LOCK_SECONDS = 3600.0


def _is_lock_stale(lock_path: Path, max_age: float = STALE_LOCK_SECONDS) -> bool:
    """Check if a lock file is stale (older than max_age seconds)."""
    try:
        if not lock_path.exists():
            return False
        mtime = lock_path.stat().st_mtime
        age = time.time() - mtime
        return age > max_age
    except OSError:
        return False


def _break_stale_lock(lock_path: Path, max_age: float = STALE_LOCK_SECONDS) -> bool:
    """
    Remove a stale lock file if it exceeds max_age.

    Uses atomic acquire-then-delete to prevent TOCTOU race conditions:
    1. Check if lock appears stale (file is old)
    2. Try to acquire the lock atomically (non-blocking)
    3. If acquired, the lock is truly abandoned - safe to delete
    4. If blocked, another process holds it - not actually stale

    Returns True if a stale lock was removed, False otherwise.
    """
    if not _is_lock_stale(lock_path, max_age):
        return False

    try:
        # Open the lock file for atomic flock acquisition
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            # Non-blocking acquire - if this succeeds, no one holds the lock
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We now hold the lock - it's truly abandoned, safe to delete
            try:
                lock_path.unlink()
                logger.warning(f"Removed stale lock file: {lock_path} (age > {max_age}s)")
                return True
            except OSError as e:
                logger.debug(f"Could not remove stale lock {lock_path}: {e}")
                return False
            finally:
                # Release the lock before closing fd
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        except BlockingIOError:
            # Lock is actively held by another process - not actually stale
            # This prevents the race condition where we delete an active lock
            logger.debug(f"Lock {lock_path} appears stale but is held by another process")
            return False
        finally:
            os.close(fd)
    except OSError as e:
        logger.debug(f"Could not check stale lock {lock_path}: {e}")
        return False


@contextmanager
def file_lock(
    path: Path,
    timeout: float | None = None,
) -> Iterator[None]:
    """
    Acquire exclusive file lock (synchronous, blocking with timeout).

    Creates a .lock file adjacent to the target path. Uses POSIX flock()
    which is released automatically when the process exits or file
    descriptor is closed.

    Args:
        path: Path to file to lock (lock file created as path.lock)
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)

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

    # Check for and break stale locks before trying to acquire
    _break_stale_lock(lock_path)

    start = time.monotonic()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)

    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except BlockingIOError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {path} within {timeout}s"
                    )
                time.sleep(0.05)  # Brief sleep before retry

        logger.debug(f"Acquired lock on {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            logger.debug(f"Released lock on {lock_path}")
    finally:
        os.close(fd)
        # Clean up lock file (best effort)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass  # Ignore cleanup failures


@asynccontextmanager
async def async_file_lock(
    name: str,
    timeout: float = 60.0,
    lock_dir: Path | None = None,
) -> AsyncIterator[None]:
    """
    Acquire exclusive file lock (async, non-blocking with timeout).

    Uses thread pool executor to avoid blocking the event loop while
    waiting for lock acquisition.

    Args:
        name: Lock name (used as filename in lock_dir)
        timeout: Max seconds to wait for lock (raises TimeoutError if exceeded)
        lock_dir: Directory for lock files (default: settings.cache_dir / "locks")

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

    # Check for and break stale locks before trying to acquire
    _break_stale_lock(lock_path)

    loop = asyncio.get_running_loop()
    fd: int | None = None

    try:
        # Open lock file
        fd = await loop.run_in_executor(
            None, lambda: os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        )

        start_time = loop.time()

        while True:
            acquired = await loop.run_in_executor(None, _try_lock, fd)
            if acquired:
                logger.debug(f"Acquired async lock: {name}")
                break
            if loop.time() - start_time >= timeout:
                raise TimeoutError(f"Timeout waiting for lock: {name} ({timeout}s)")
            await asyncio.sleep(0.1)

        try:
            yield
        finally:
            await loop.run_in_executor(None, _release_lock, fd)
            logger.debug(f"Released async lock: {name}")

    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            # Clean up lock file (best effort)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def _try_lock(fd: int) -> bool:
    """Try to acquire exclusive lock (non-blocking)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    async def acquire(
        self, name: str, timeout: float = 60.0
    ) -> AsyncIterator[None]:
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
        """
        Clean up stale lock files in lock directory.

        By default, only removes locks older than STALE_LOCK_SECONDS (1 hour).
        Use force=True to remove all locks (e.g., on known clean startup).

        Args:
            force: If True, remove all locks regardless of age

        Returns:
            Number of lock files removed
        """
        removed = 0
        try:
            if self._lock_dir.exists():
                for lock_file in self._lock_dir.glob("*.lock"):
                    try:
                        if force or _is_lock_stale(lock_file):
                            lock_file.unlink()
                            removed += 1
                            if not force:
                                logger.info(f"Cleaned up stale lock: {lock_file}")
                    except OSError:
                        pass
        except OSError:
            pass
        return removed


def cleanup_stale_locks(lock_dir: Path | None = None) -> int:
    """
    Convenience function to clean up stale lock files.

    Removes lock files older than STALE_LOCK_SECONDS (1 hour).

    Args:
        lock_dir: Directory containing lock files (default: settings.cache_dir / "locks")

    Returns:
        Number of stale lock files removed
    """
    lock_dir = lock_dir or (settings.cache_dir / "locks")
    removed = 0
    try:
        if lock_dir.exists():
            for lock_file in lock_dir.glob("*.lock"):
                try:
                    if _is_lock_stale(lock_file):
                        lock_file.unlink()
                        removed += 1
                        logger.info(f"Cleaned up stale lock: {lock_file}")
                except OSError:
                    pass
    except OSError:
        pass
    return removed
