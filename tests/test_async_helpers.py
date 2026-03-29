"""Tests for async helper utilities."""

import asyncio
import threading
import time

import pytest

from vector_core.utils.async_helpers import (
    AsyncSingleton,
    SingletonInitError,
    clear_init_locks,
    get_async_init_lock,
    sync_cleanup_wrapper,
)


class TestAsyncSingleton:
    """Tests for AsyncSingleton class."""

    @pytest.fixture(autouse=True)
    def reset_locks(self):
        """Reset init locks before each test."""
        clear_init_locks()
        yield
        clear_init_locks()

    @pytest.mark.asyncio
    async def test_basic_initialization(self):
        """Singleton initializes correctly on first call."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        result = await singleton.get(lambda: "initialized")

        assert result == "initialized"
        assert singleton.is_initialized
        assert not singleton.has_error

    @pytest.mark.asyncio
    async def test_returns_same_instance(self):
        """Subsequent calls return the same instance."""
        singleton: AsyncSingleton[list] = AsyncSingleton("test")
        instance = []

        result1 = await singleton.get(lambda: instance)
        result2 = await singleton.get(lambda: [])

        assert result1 is result2
        assert result1 is instance

    @pytest.mark.asyncio
    async def test_async_factory(self):
        """Async factory functions work correctly."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        async def async_factory():
            await asyncio.sleep(0.01)
            return "async_initialized"

        result = await singleton.get(async_factory)

        assert result == "async_initialized"

    @pytest.mark.asyncio
    async def test_error_stored_and_reraised(self):
        """Initialization error is stored and re-raised on subsequent calls."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        def failing_factory():
            raise ValueError("init failed")

        # First call raises the original error
        with pytest.raises(ValueError, match="init failed"):
            await singleton.get(failing_factory)

        assert singleton.has_error
        assert not singleton.is_initialized

        # Subsequent calls raise SingletonInitError
        with pytest.raises(SingletonInitError) as exc_info:
            await singleton.get(lambda: "should not be called")

        assert "test" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, ValueError)

    @pytest.mark.asyncio
    async def test_recovery_after_cooldown(self):
        """Singleton recovers after cooldown period expires."""
        # Use a very short cooldown for testing
        singleton: AsyncSingleton[str] = AsyncSingleton(
            "test_recovery",
            recovery_delay=0.1,  # 100ms cooldown
        )
        attempt_count = [0]

        def sometimes_failing_factory():
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise ValueError("first attempt fails")
            return "recovered"

        # First call fails
        with pytest.raises(ValueError, match="first attempt fails"):
            await singleton.get(sometimes_failing_factory)

        assert singleton.has_error
        assert attempt_count[0] == 1

        # Immediate retry still fails with SingletonInitError
        with pytest.raises(SingletonInitError):
            await singleton.get(sometimes_failing_factory)
        assert attempt_count[0] == 1  # Factory not called again

        # Wait for recovery cooldown
        await asyncio.sleep(0.15)  # Wait slightly longer than cooldown

        # Now recovery should work
        result = await singleton.get(sometimes_failing_factory)
        assert result == "recovered"
        assert attempt_count[0] == 2  # Factory called again
        assert singleton.is_initialized
        assert not singleton.has_error

    @pytest.mark.asyncio
    async def test_recovery_disabled_when_delay_zero(self):
        """Recovery is disabled when recovery_delay is 0."""
        singleton: AsyncSingleton[str] = AsyncSingleton(
            "test_no_recovery",
            recovery_delay=0,  # Disable recovery
        )

        def failing_factory():
            raise ValueError("permanent failure")

        # First call fails
        with pytest.raises(ValueError):
            await singleton.get(failing_factory)

        # Even after time passes, still fails with SingletonInitError
        await asyncio.sleep(0.1)
        with pytest.raises(SingletonInitError):
            await singleton.get(lambda: "should not be called")

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Retries work for configured exception types."""
        singleton: AsyncSingleton[str] = AsyncSingleton(
            "test",
            max_retries=2,
            retry_exceptions=(ConnectionError,),
        )
        attempt_count = [0]

        def flaky_factory():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError("transient")
            return "success"

        result = await singleton.get(flaky_factory)

        assert result == "success"
        assert attempt_count[0] == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """All retries exhausted raises the last error."""
        singleton: AsyncSingleton[str] = AsyncSingleton(
            "test",
            max_retries=2,
            retry_exceptions=(ConnectionError,),
        )

        def always_failing():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError, match="always fails"):
            await singleton.get(always_failing)

        assert singleton.has_error

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        """Close clears singleton state."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        await singleton.get(lambda: "initialized")
        assert singleton.is_initialized

        await singleton.close()

        assert not singleton.is_initialized
        assert not singleton.has_error

    @pytest.mark.asyncio
    async def test_close_calls_cleanup(self):
        """Close calls cleanup function with instance."""
        singleton: AsyncSingleton[list] = AsyncSingleton("test")
        cleanup_called = []

        instance = await singleton.get(lambda: ["item"])

        async def cleanup(inst):
            cleanup_called.append(inst)
            inst.clear()

        await singleton.close(cleanup)

        assert cleanup_called == [instance]
        assert instance == []  # Cleanup was called

    @pytest.mark.asyncio
    async def test_close_sync_cleanup(self):
        """Close works with sync cleanup function."""
        singleton: AsyncSingleton[list] = AsyncSingleton("test")
        cleanup_called = []

        instance = await singleton.get(lambda: ["item"])

        def sync_cleanup(inst):
            cleanup_called.append(inst)

        await singleton.close(sync_cleanup)

        assert cleanup_called == [instance]

    @pytest.mark.asyncio
    async def test_close_handles_cleanup_error(self):
        """Close handles errors in cleanup gracefully."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        await singleton.get(lambda: "initialized")

        def failing_cleanup(inst):
            raise RuntimeError("cleanup failed")

        # Should not raise, just log
        await singleton.close(failing_cleanup)

        # State should still be cleared
        assert not singleton.is_initialized

    @pytest.mark.asyncio
    async def test_close_already_closed(self):
        """Closing already-closed singleton is a no-op."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")
        cleanup_count = [0]

        def cleanup(inst):
            cleanup_count[0] += 1

        await singleton.get(lambda: "initialized")
        await singleton.close(cleanup)
        await singleton.close(cleanup)  # Second close

        assert cleanup_count[0] == 1  # Only called once

    def test_reset_clears_state(self):
        """Reset synchronously clears state."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")
        singleton._instance = "test"
        singleton._initialized = True

        singleton.reset()

        assert not singleton.is_initialized
        assert singleton._instance is None

    @pytest.mark.asyncio
    async def test_reinitialize_after_close(self):
        """Can reinitialize after close."""
        singleton: AsyncSingleton[int] = AsyncSingleton("test")
        call_count = [0]

        def factory():
            call_count[0] += 1
            return call_count[0]

        result1 = await singleton.get(factory)
        await singleton.close()
        result2 = await singleton.get(factory)

        assert result1 == 1
        assert result2 == 2

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Concurrent calls all get the same instance."""
        singleton: AsyncSingleton[int] = AsyncSingleton("test")
        init_count = [0]

        async def slow_factory():
            init_count[0] += 1
            await asyncio.sleep(0.05)
            return init_count[0]

        # Start multiple concurrent gets
        results = await asyncio.gather(
            singleton.get(slow_factory),
            singleton.get(slow_factory),
            singleton.get(slow_factory),
        )

        # All should get the same instance
        assert results == [1, 1, 1]
        assert init_count[0] == 1  # Factory called only once


class TestAsyncSingletonCloseRace:
    """Tests for close() race condition fix."""

    @pytest.fixture(autouse=True)
    def reset_locks(self):
        """Reset init locks before each test."""
        clear_init_locks()
        yield
        clear_init_locks()

    @pytest.mark.asyncio
    async def test_close_during_slow_cleanup_no_deadlock(self):
        """Close doesn't deadlock with slow async cleanup."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")

        await singleton.get(lambda: "initialized")

        async def slow_cleanup(inst):
            await asyncio.sleep(0.1)

        # This should complete without deadlock
        await asyncio.wait_for(
            singleton.close(slow_cleanup),
            timeout=1.0,
        )

        assert not singleton.is_initialized

    @pytest.mark.asyncio
    async def test_close_allows_concurrent_get_after_clear(self):
        """After close clears state, concurrent get can proceed."""
        singleton: AsyncSingleton[int] = AsyncSingleton("test")
        call_count = [0]
        cleanup_started = asyncio.Event()
        cleanup_proceed = asyncio.Event()

        def factory():
            call_count[0] += 1
            return call_count[0]

        await singleton.get(factory)

        async def slow_cleanup(inst):
            cleanup_started.set()
            await cleanup_proceed.wait()

        # Start close (will be slow)
        close_task = asyncio.create_task(singleton.close(slow_cleanup))

        # Wait for cleanup to start
        await cleanup_started.wait()

        # Instance should be cleared immediately (before cleanup finishes)
        assert not singleton.is_initialized

        # Allow cleanup to proceed
        cleanup_proceed.set()
        await close_task

    @pytest.mark.asyncio
    async def test_threading_lock_not_held_during_await(self):
        """Threading lock is NOT held during async cleanup (the fix)."""
        singleton: AsyncSingleton[str] = AsyncSingleton("test")
        lock_held_during_cleanup = [False]

        await singleton.get(lambda: "initialized")

        async def check_lock_cleanup(inst):
            # Try to acquire the threading lock
            # If it's held, acquire() will block (return False with timeout=0)
            acquired = singleton._lock.acquire(blocking=False)
            if acquired:
                singleton._lock.release()
                lock_held_during_cleanup[0] = False
            else:
                lock_held_during_cleanup[0] = True
            await asyncio.sleep(0.01)

        await singleton.close(check_lock_cleanup)

        # Lock should NOT be held during cleanup (the fix ensures this)
        assert not lock_held_during_cleanup[0]


class TestGetAsyncInitLock:
    """Tests for get_async_init_lock function."""

    @pytest.fixture(autouse=True)
    def reset_locks(self):
        """Reset init locks before each test."""
        clear_init_locks()
        yield
        clear_init_locks()

    @pytest.mark.asyncio
    async def test_returns_asyncio_lock(self):
        """Returns an asyncio.Lock instance."""
        lock = get_async_init_lock()
        assert isinstance(lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_same_lock_same_loop(self):
        """Same lock returned for same event loop."""
        lock1 = get_async_init_lock()
        lock2 = get_async_init_lock()
        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_lock_works_for_mutual_exclusion(self):
        """Lock provides mutual exclusion."""
        lock = get_async_init_lock()
        order = []

        async def task(name, delay):
            async with lock:
                order.append(f"{name}_start")
                await asyncio.sleep(delay)
                order.append(f"{name}_end")

        await asyncio.gather(
            task("a", 0.02),
            task("b", 0.01),
        )

        # Tasks should be serialized
        assert order == ["a_start", "a_end", "b_start", "b_end"]


class TestSyncCleanupWrapper:
    """Tests for sync_cleanup_wrapper function."""

    def test_running_loop_schedules_task(self):
        """When loop is running, cleanup is scheduled as task."""
        task_scheduled = [False]

        async def cleanup():
            task_scheduled[0] = True

        singletons: list[AsyncSingleton] = []

        async def main():
            # Schedule cleanup while loop is running
            sync_cleanup_wrapper(cleanup, singletons)
            # Give task a chance to run
            await asyncio.sleep(0.05)

        asyncio.run(main())

        assert task_scheduled[0]

    def test_no_loop_uses_asyncio_run(self):
        """When no loop running, uses asyncio.run()."""
        cleanup_called = [False]

        async def cleanup():
            cleanup_called[0] = True

        singletons: list[AsyncSingleton] = []

        # Call outside any event loop
        sync_cleanup_wrapper(cleanup, singletons)

        assert cleanup_called[0]

    def test_fallback_resets_singletons(self):
        """On exception, falls back to resetting singletons."""
        singleton = AsyncSingleton("test")
        singleton._instance = "test"
        singleton._initialized = True

        async def failing_cleanup():
            raise RuntimeError("cleanup failed")

        # This should not raise
        sync_cleanup_wrapper(failing_cleanup, [singleton])

        # Singleton should be reset
        assert not singleton.is_initialized


class TestSingletonInitError:
    """Tests for SingletonInitError."""

    def test_error_message_includes_name(self):
        """Error message includes singleton name."""
        original = ValueError("original error")
        error = SingletonInitError("my_singleton", original)

        assert "my_singleton" in str(error)
        assert "original error" in str(error)

    def test_original_error_accessible(self):
        """Original error is accessible."""
        original = ValueError("test")
        error = SingletonInitError("name", original)

        assert error.original_error is original
        assert error.name == "name"

    def test_error_is_exception(self):
        """SingletonInitError is an Exception."""
        error = SingletonInitError("name", ValueError("test"))
        assert isinstance(error, Exception)
