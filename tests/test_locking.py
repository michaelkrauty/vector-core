"""Tests for cross-process file locking utilities."""

import asyncio
import fcntl
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from vector_core.utils import locking as locking_module
from vector_core.utils.locking import (
    LockManager,
    _is_lock_stale,
    async_file_lock,
    cleanup_stale_locks,
    file_lock,
)


class TestFileLock:
    """Tests for the synchronous file_lock context manager."""

    def test_lock_acquires_and_releases(self, tmp_path: Path):
        """Lock is acquired and released properly."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with file_lock(test_file, timeout=1.0):
            # Lock file should exist
            lock_file = test_file.with_suffix(".txt.lock")
            assert lock_file.exists()

    def test_lock_is_exclusive(self, tmp_path: Path):
        """Cannot acquire same lock twice."""
        test_file = tmp_path / "exclusive.txt"
        test_file.touch()
        lock_file = test_file.with_suffix(".txt.lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Manually acquire lock
        fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            # Try to acquire same lock - should timeout
            with pytest.raises(TimeoutError):
                with file_lock(test_file, timeout=0.2):
                    pass
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_lock_cleanup_on_exception(self, tmp_path: Path):
        """Lock is released even if exception occurs inside context."""
        test_file = tmp_path / "exception.txt"
        test_file.touch()

        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            with file_lock(test_file, timeout=1.0):
                raise TestException("Test error")

        # Lock should be released - we should be able to acquire it again
        with file_lock(test_file, timeout=0.2):
            pass  # Should not timeout


class TestAsyncFileLock:
    """Tests for the async_file_lock context manager."""

    @pytest.mark.asyncio
    async def test_lock_acquires_and_releases(self, tmp_path: Path):
        """Lock is acquired and released properly."""
        lock_dir = tmp_path / "locks"
        lock_file = lock_dir / "test.lock"

        async with async_file_lock("test", lock_dir=lock_dir):
            # Lock file should exist
            assert lock_file.exists()

    @pytest.mark.asyncio
    async def test_lock_is_exclusive(self, tmp_path: Path):
        """Two locks on same name cannot be held simultaneously."""
        lock_dir = tmp_path / "locks"

        lock_acquired = asyncio.Event()
        lock_released = asyncio.Event()

        async def hold_lock():
            async with async_file_lock("exclusive_test", lock_dir=lock_dir):
                lock_acquired.set()
                await lock_released.wait()

        # Start first lock holder
        task = asyncio.create_task(hold_lock())
        await lock_acquired.wait()

        # Try to acquire same lock with short timeout - should fail
        with pytest.raises(TimeoutError) as exc_info:
            async with async_file_lock("exclusive_test", timeout=0.2, lock_dir=lock_dir):
                pass

        assert "Timeout waiting for lock" in str(exc_info.value)
        assert "exclusive_test" in str(exc_info.value)

        # Release first lock
        lock_released.set()
        await task

    @pytest.mark.asyncio
    async def test_lock_cleanup_on_exception(self, tmp_path: Path):
        """Lock is released even if exception occurs inside context."""
        lock_dir = tmp_path / "locks"

        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            async with async_file_lock("exception_test", lock_dir=lock_dir):
                raise TestException("Test error")

        # Lock should be released - we should be able to acquire it again
        async with async_file_lock("exception_test", timeout=0.2, lock_dir=lock_dir):
            pass  # Should not timeout

    @pytest.mark.asyncio
    async def test_lock_retry_loop(self, tmp_path: Path):
        """Lock retries when BlockingIOError occurs."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        # Mock flock to fail first 2 times, then succeed
        original_flock = fcntl.flock
        attempts = []

        def flock_with_retry(fd, operation):
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                attempts.append(time.time())
                if len(attempts) < 3:
                    raise BlockingIOError("Lock held")
            return original_flock(fd, operation)

        with patch("vector_core.utils.locking.fcntl.flock", side_effect=flock_with_retry):
            async with async_file_lock("retry_test", timeout=1.0, lock_dir=lock_dir):
                pass

        # Should have retried
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_repeated_cancellation_cannot_strand_or_reuse_fd(self, tmp_path: Path):
        """A cancelled waiter closes only its own descriptor and leaves the lock usable."""
        lock_dir = tmp_path / "locks"
        release = asyncio.Event()

        async def holder() -> None:
            async with async_file_lock("cancel", lock_dir=lock_dir):
                await release.wait()

        held = asyncio.create_task(holder())
        await asyncio.sleep(0.05)
        waiter = asyncio.create_task(async_file_lock("cancel", lock_dir=lock_dir).__aenter__())
        await asyncio.sleep(0.05)
        waiter.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        release.set()
        await held
        async with async_file_lock("cancel", timeout=0.5, lock_dir=lock_dir):
            pass

    @pytest.mark.asyncio
    async def test_open_error_after_cancellation_preserves_cancelled_error(self, tmp_path: Path):
        started = threading.Event()
        release = threading.Event()

        def fail_open(*_args):
            started.set()
            release.wait(timeout=2.0)
            raise OSError("late open failure")

        with patch.object(locking_module.os, "open", side_effect=fail_open):
            context = async_file_lock("late-open", lock_dir=tmp_path)
            task = asyncio.create_task(context.__aenter__())
            assert await asyncio.to_thread(started.wait, 1.0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task

        assert isinstance(exc_info.value.__cause__, OSError)

    @pytest.mark.asyncio
    async def test_flock_error_after_cancellation_preserves_cancelled_error(self, tmp_path: Path):
        started = threading.Event()
        release = threading.Event()

        def fail_lock(*_args):
            started.set()
            release.wait(timeout=2.0)
            raise OSError("late flock failure")

        with patch.object(locking_module, "_try_lock", side_effect=fail_lock):
            context = async_file_lock("late-flock", lock_dir=tmp_path)
            task = asyncio.create_task(context.__aenter__())
            assert await asyncio.to_thread(started.wait, 1.0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task

        assert isinstance(exc_info.value.__cause__, OSError)
        async with async_file_lock("late-flock", timeout=0.5, lock_dir=tmp_path):
            pass


class TestCleanupStaleLocks:
    """Tests for cleanup_stale_locks function."""

    def test_cleanup_returns_zero_if_no_lock_dir(self, tmp_path: Path):
        """Returns 0 if lock directory doesn't exist."""
        lock_dir = tmp_path / "nonexistent_locks"
        assert cleanup_stale_locks(lock_dir) == 0

    def test_cleanup_retains_stale_locks(self, tmp_path: Path):
        """Retains old lock files as stable inode anchors."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True)

        # Create a "stale" lock file (fake old mtime)
        stale_lock = lock_dir / "stale.lock"
        stale_lock.touch()
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(stale_lock, (old_time, old_time))

        # Create a "fresh" lock file
        fresh_lock = lock_dir / "fresh.lock"
        fresh_lock.touch()

        # Run cleanup
        removed = cleanup_stale_locks(lock_dir)

        assert removed == 0
        assert stale_lock.exists()
        assert fresh_lock.exists()

    def test_cleanup_handles_permission_error(self, tmp_path: Path):
        """Does not attempt to unlink lock files."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True)

        # Create a stale lock
        stale_lock = lock_dir / "stale.lock"
        stale_lock.touch()
        old_time = time.time() - 7200
        os.utime(stale_lock, (old_time, old_time))

        # Any unlink attempt would fail the test.
        with patch.object(Path, "unlink", side_effect=PermissionError("Cannot delete")):
            removed = cleanup_stale_locks(lock_dir)

        # Should handle error gracefully and return 0
        assert removed == 0
        assert stale_lock.exists()

    def test_cleanup_leaves_all_files(self, tmp_path: Path):
        """Leaves lock and non-lock files untouched."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True)

        # Create various files
        lock_file = lock_dir / "test.lock"
        lock_file.touch()
        old_time = time.time() - 7200
        os.utime(lock_file, (old_time, old_time))

        other_file = lock_dir / "test.txt"
        other_file.touch()
        os.utime(other_file, (old_time, old_time))

        removed = cleanup_stale_locks(lock_dir)

        assert removed == 0
        assert lock_file.exists()
        assert other_file.exists()  # Should not be touched


class TestIsLockStale:
    """Tests for _is_lock_stale helper."""

    def test_nonexistent_file_not_stale(self, tmp_path: Path):
        """Non-existent file is not considered stale."""
        assert not _is_lock_stale(tmp_path / "nonexistent.lock")

    def test_fresh_file_not_stale(self, tmp_path: Path):
        """Recently created file is not stale."""
        lock_file = tmp_path / "fresh.lock"
        lock_file.touch()
        assert not _is_lock_stale(lock_file)

    def test_old_file_is_stale(self, tmp_path: Path):
        """File older than threshold is stale."""
        lock_file = tmp_path / "old.lock"
        lock_file.touch()
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(lock_file, (old_time, old_time))
        assert _is_lock_stale(lock_file)


class TestLockManager:
    """Tests for LockManager class."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, tmp_path: Path):
        """Can acquire and release locks through manager."""
        manager = LockManager(lock_dir=tmp_path / "locks")

        async with manager.acquire("test_lock"):
            assert await manager.is_locked("test_lock")

        assert not await manager.is_locked("test_lock")

    @pytest.mark.asyncio
    async def test_cleanup_stale_locks(self, tmp_path: Path):
        """Manager retains stale-looking lock files."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True)
        manager = LockManager(lock_dir=lock_dir)

        # Create a stale lock
        stale_lock = lock_dir / "stale.lock"
        stale_lock.touch()
        old_time = time.time() - 7200
        os.utime(stale_lock, (old_time, old_time))

        removed = manager.cleanup()
        assert removed == 0
        assert stale_lock.exists()

    def test_force_cleanup_all_locks(self, tmp_path: Path):
        """Force cleanup cannot remove stable lock inode anchors."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir(parents=True)
        manager = LockManager(lock_dir=lock_dir)

        # Create a fresh lock
        fresh_lock = lock_dir / "fresh.lock"
        fresh_lock.touch()

        removed = manager.cleanup(force=True)
        assert removed == 0
        assert fresh_lock.exists()


class TestSharedLocks:
    """Tests for optional shared/read lock acquisition."""

    def test_sync_shared_locks_coexist_and_block_exclusive(self, tmp_path: Path):
        """Readers may overlap, while a writer waits for every reader."""
        target = tmp_path / "resource"
        with file_lock(target, timeout=1.0, shared=True):
            with file_lock(target, timeout=1.0, shared=True):
                with pytest.raises(TimeoutError):
                    with file_lock(target, timeout=0.1):
                        pass

        with file_lock(target, timeout=1.0):
            pass

    @pytest.mark.asyncio
    async def test_async_shared_locks_coexist_and_block_exclusive(self, tmp_path: Path):
        """Async readers overlap and an exclusive waiter enters after release."""
        lock_dir = tmp_path / "locks"
        acquired = asyncio.Event()

        async def writer() -> None:
            async with async_file_lock("rw", lock_dir=lock_dir):
                acquired.set()

        async with async_file_lock("rw", lock_dir=lock_dir, shared=True):
            async with async_file_lock("rw", lock_dir=lock_dir, shared=True):
                task = asyncio.create_task(writer())
                await asyncio.sleep(0.15)
                assert not acquired.is_set()

        await asyncio.wait_for(task, timeout=1.0)
        assert acquired.is_set()


class TestForkSafety:
    @pytest.mark.parametrize("shared", [False, True])
    def test_child_unwind_does_not_unlock_parent_sync(self, tmp_path: Path, shared: bool) -> None:
        target = tmp_path / "fork-sync"
        context = file_lock(target, timeout=1.0, shared=shared)
        context.__enter__()
        child = os.fork()
        if child == 0:
            context.__exit__(None, None, None)
            probe = os.open(tmp_path / "child-probe-sync", os.O_CREAT | os.O_RDWR, 0o600)
            os.fstat(probe)
            os.close(probe)
            os._exit(0)

        try:
            _pid, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            with pytest.raises(TimeoutError):
                with file_lock(target, timeout=0.1):
                    pass
        finally:
            context.__exit__(None, None, None)

        with file_lock(target, timeout=0.5):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shared", [False, True])
    async def test_child_unwind_does_not_unlock_parent_async(
        self, tmp_path: Path, shared: bool
    ) -> None:
        lock_dir = tmp_path / "locks"
        context = async_file_lock(
            "fork-async",
            timeout=1.0,
            lock_dir=lock_dir,
            shared=shared,
        )
        await context.__aenter__()
        child = os.fork()
        if child == 0:
            await context.__aexit__(None, None, None)
            probe = os.open(tmp_path / "child-probe-async", os.O_CREAT | os.O_RDWR, 0o600)
            os.fstat(probe)
            os.close(probe)
            os._exit(0)

        try:
            _pid, status = await asyncio.to_thread(os.waitpid, child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            with pytest.raises(TimeoutError):
                async with async_file_lock(
                    "fork-async",
                    timeout=0.1,
                    lock_dir=lock_dir,
                ):
                    pass
        finally:
            await context.__aexit__(None, None, None)

        async with async_file_lock(
            "fork-async",
            timeout=0.5,
            lock_dir=lock_dir,
        ):
            pass
