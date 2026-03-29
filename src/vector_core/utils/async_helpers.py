"""Async utility functions for thread-safe singleton initialization."""

import asyncio
import logging
import threading
import time
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Thread lock for creating async init locks (prevents race condition)
_init_lock_creation_lock = threading.Lock()

# Store locks per event loop to avoid cross-loop issues
_init_locks: dict[int, asyncio.Lock] = {}


class SingletonInitError(Exception):
    """Raised when singleton initialization fails permanently."""

    def __init__(self, name: str, original_error: Exception):
        self.name = name
        self.original_error = original_error
        super().__init__(
            f"Failed to initialize singleton '{name}': {original_error}"
        )


class AsyncSingleton(Generic[T]):
    """
    Robust async singleton manager with error tracking.

    Handles:
    - Thread-safe double-checked locking
    - Initialization error tracking (fails fast after first error)
    - Proper cleanup on shutdown
    - Optional retry on transient errors

    Usage:
        _my_singleton: AsyncSingleton[MyClass] = AsyncSingleton("my_class")

        async def get_my_instance() -> MyClass:
            return await _my_singleton.get(lambda: MyClass())

        async def cleanup():
            await _my_singleton.close(lambda x: x.close())
    """

    def __init__(
        self,
        name: str,
        *,
        max_retries: int = 0,
        retry_exceptions: tuple[type[Exception], ...] | None = None,
        recovery_delay: float = 30.0,
    ):
        """
        Initialize singleton manager.

        Args:
            name: Human-readable name for logging/errors
            max_retries: Number of retries on transient errors (0 = fail immediately)
            retry_exceptions: Exception types to retry on (None = no retries)
            recovery_delay: Seconds to wait before allowing retry after a permanent error.
                          After this cooldown, the next get() call will attempt re-initialization.
                          Set to 0 to disable recovery (fail permanently after first error).
        """
        self._name = name
        self._instance: T | None = None
        self._init_error: Exception | None = None
        self._error_timestamp: float | None = None
        self._initialized = False
        self._max_retries = max_retries
        self._retry_exceptions = retry_exceptions or ()
        self._recovery_delay = recovery_delay
        self._lock = threading.Lock()
        # Per-instance async locks keyed by event loop id (avoids cross-singleton deadlocks)
        self._async_locks: dict[int, asyncio.Lock] = {}

    @property
    def is_initialized(self) -> bool:
        """Check if singleton is initialized (without triggering init)."""
        return self._initialized and self._instance is not None

    @property
    def has_error(self) -> bool:
        """Check if initialization failed."""
        return self._init_error is not None

    def _get_async_lock(self) -> asyncio.Lock:
        """
        Get or create an async lock for this singleton for the current event loop.

        Each AsyncSingleton has its own set of locks (one per event loop) to avoid
        deadlocks when one singleton's factory calls another singleton's get().

        SAFETY: When no running loop exists, returns a fresh ephemeral lock to avoid
        the bug where multiple different event loops share loop_id=0.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop - return ephemeral lock to avoid shared loop_id=0 bug
            # Caller must be in async context anyway, so this is safe
            return asyncio.Lock()

        with self._lock:
            existing = self._async_locks.get(loop_id)
            if existing is not None:
                # Validate lock's loop is still valid (handles loop GC/memory recycle)
                # After a loop is closed and GC'd, its memory id can be reused
                try:
                    # asyncio.Lock stores _loop reference; verify it matches current loop
                    lock_loop = getattr(existing, '_loop', None)
                    if lock_loop is loop or lock_loop is None:
                        return existing
                except AttributeError:
                    pass  # Python version compatibility
                # Lock is from a stale/different loop - remove and recreate
                del self._async_locks[loop_id]

            # Create new lock for this loop
            new_lock = asyncio.Lock()
            self._async_locks[loop_id] = new_lock
            return new_lock

    def _should_retry_after_error(self) -> bool:
        """
        Check if we should retry initialization after a previous error.

        Returns True if:
        - recovery_delay > 0 (recovery enabled)
        - error_timestamp is set (we had a failure)
        - enough time has passed since the error

        Must be called inside self._lock.
        """
        if self._recovery_delay <= 0:
            return False
        if self._error_timestamp is None:
            return False
        elapsed = time.monotonic() - self._error_timestamp
        return elapsed >= self._recovery_delay

    async def get(self, factory: Callable[[], T | Awaitable[T]]) -> T:
        """
        Get or create the singleton instance.

        Args:
            factory: Callable that creates the instance (can be sync or async)

        Returns:
            The singleton instance

        Raises:
            SingletonInitError: If initialization failed previously
            Exception: Whatever the factory raises on first failure
        """
        # Fast path: thread-safe read of state
        # Using threading.Lock ensures visibility across threads
        with self._lock:
            if self._instance is not None:
                return self._instance
            if self._init_error is not None:
                if self._should_retry_after_error():
                    # Recovery cooldown elapsed - clear error and try again
                    logger.info(
                        f"Singleton '{self._name}' recovery: attempting re-initialization "
                        f"after {self._recovery_delay}s cooldown"
                    )
                    self._init_error = None
                    self._error_timestamp = None
                else:
                    raise SingletonInitError(self._name, self._init_error)

        # Acquire per-instance async lock for initialization (coordinates async tasks)
        # Using per-instance lock avoids deadlocks when factories call other singletons
        async with self._get_async_lock():
            # Double-check after acquiring async lock (thread-safe)
            with self._lock:
                if self._instance is not None:
                    return self._instance
                if self._init_error is not None:
                    if self._should_retry_after_error():
                        # Recovery cooldown elapsed - clear error and try again
                        self._init_error = None
                        self._error_timestamp = None
                    else:
                        raise SingletonInitError(self._name, self._init_error)

            # Try to initialize with retries
            last_error: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    result = factory()
                    if asyncio.iscoroutine(result):
                        instance = await result
                    else:
                        instance = result
                    # Thread-safe assignment of instance
                    with self._lock:
                        self._instance = instance
                        self._initialized = True
                    logger.debug(f"Initialized singleton '{self._name}'")
                    return instance
                except self._retry_exceptions as e:
                    last_error = e
                    if attempt < self._max_retries:
                        logger.warning(
                            f"Singleton '{self._name}' init failed (attempt {attempt + 1}/"
                            f"{self._max_retries + 1}): {e}"
                        )
                        await asyncio.sleep(0.1 * (attempt + 1))  # Backoff
                    continue
                except Exception as e:
                    # Non-retryable error - store and raise (thread-safe)
                    with self._lock:
                        self._init_error = e
                        self._error_timestamp = time.monotonic()
                    logger.error(
                        f"Singleton '{self._name}' init failed: {e} "
                        f"(will retry after {self._recovery_delay}s)"
                    )
                    raise

            # Exhausted retries
            if last_error is not None:
                with self._lock:
                    self._init_error = last_error
                    self._error_timestamp = time.monotonic()
                logger.error(
                    f"Singleton '{self._name}' init failed after {self._max_retries + 1} attempts "
                    f"(will retry after {self._recovery_delay}s)"
                )
                raise last_error

            # Should never reach here
            raise RuntimeError(f"Unexpected state in singleton '{self._name}'")

    async def close(
        self,
        cleanup: Callable[[T], Awaitable[None] | None] | None = None,
    ) -> None:
        """
        Close and reset the singleton.

        The close operation happens in two phases:
        1. Atomically get instance and clear state (inside lock)
        2. Run cleanup OUTSIDE the lock (allows proper await)

        This prevents holding a threading.Lock while awaiting, which could
        block the event loop and cause deadlocks.

        Args:
            cleanup: Optional async cleanup function to call on the instance
        """
        # Phase 1: Atomically get instance and clear state
        with self._lock:
            instance = self._instance
            if instance is None:
                # Already closed or never initialized
                return
            # Clear instance immediately to prevent new callers from getting it
            self._instance = None
            self._initialized = False
            self._init_error = None
            self._error_timestamp = None

        # Phase 2: Run cleanup OUTSIDE lock (can await safely)
        if cleanup is not None:
            try:
                result = cleanup(instance)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error cleaning up singleton '{self._name}': {e}")

        logger.debug(f"Closed singleton '{self._name}'")

    def reset(self) -> None:
        """
        Reset singleton state without cleanup (for testing).

        WARNING: May leak resources if instance has open connections.
        """
        with self._lock:
            self._instance = None
            self._initialized = False
            self._init_error = None
            self._error_timestamp = None
            self._async_locks.clear()  # Clear locks for new event loops


def get_async_init_lock() -> asyncio.Lock:
    """
    Get or create an async initialization lock for the current event loop.

    This function is thread-safe and creates a separate lock per event loop
    to avoid issues with asyncio.Lock() being used across different loops.

    Use this for async-safe singleton initialization with double-checked locking:

        async def get_singleton() -> MySingleton:
            global _singleton
            if _singleton is None:
                async with get_async_init_lock():
                    if _singleton is None:  # Double-check after lock
                        _singleton = await MySingleton.create()
            return _singleton

    Returns:
        asyncio.Lock for the current event loop

    SAFETY: When no running loop exists, returns a fresh ephemeral lock to avoid
    the bug where multiple different event loops share loop_id=0.
    """
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No running loop - return ephemeral lock to avoid shared loop_id=0 bug
        return asyncio.Lock()

    with _init_lock_creation_lock:
        existing = _init_locks.get(loop_id)
        if existing is not None:
            # Validate lock's loop is still valid (handles loop GC/memory recycle)
            try:
                lock_loop = getattr(existing, '_loop', None)
                if lock_loop is loop or lock_loop is None:
                    return existing
            except AttributeError:
                pass
            del _init_locks[loop_id]

        new_lock = asyncio.Lock()
        _init_locks[loop_id] = new_lock
        return new_lock


def clear_init_locks() -> None:
    """
    Clear all cached init locks.

    Useful for testing to reset state between test runs.
    """
    global _init_locks
    with _init_lock_creation_lock:
        _init_locks = {}


def sync_cleanup_wrapper(
    cleanup_coro: Callable[[], Awaitable[None]],
    singletons: list["AsyncSingleton"],
) -> None:
    """
    Synchronous wrapper for async cleanup, suitable for atexit handlers.

    Handles the complexity of running async cleanup from sync contexts:
    - If an event loop is running, schedules cleanup as a task
    - Otherwise creates a new loop to run cleanup
    - On any exception, falls back to synchronously resetting all singletons

    This is the standard pattern used by MCP servers for shutdown cleanup.

    Usage:
        # At module level, define your singletons and cleanup
        _storage: AsyncSingleton[Storage] = AsyncSingleton("storage")
        _embedder: AsyncSingleton[Embedder] = AsyncSingleton("embedder")

        async def cleanup_resources() -> None:
            await _storage.close(lambda s: s.close())
            await _embedder.close(lambda e: e.close())

        def _sync_cleanup() -> None:
            singletons = [_storage, _embedder]
            if not any(s.is_initialized for s in singletons):
                return
            sync_cleanup_wrapper(cleanup_resources, singletons)

        atexit.register(_sync_cleanup)

    Args:
        cleanup_coro: Async function that performs cleanup (should be idempotent)
        singletons: List of AsyncSingleton instances to reset on exception
    """

    def _safe_reset_all() -> None:
        """Reset all singletons synchronously (fallback on error)."""
        for singleton in singletons:
            try:
                singleton.reset()
            except Exception:
                # During interpreter shutdown, even reset might fail
                pass

    try:
        # Check if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            # Loop is running - schedule cleanup as a task
            # This happens during testing or when called from async context
            # Note: Task may not complete before loop exits, but that's acceptable
            # for shutdown cleanup
            loop.create_task(cleanup_coro())
            logger.debug("Scheduled async cleanup task in running loop")
            return
        except RuntimeError:
            # No running loop - need to run synchronously
            pass

        # Create a new event loop and run cleanup
        # This is the most reliable approach when no loop is running
        try:
            asyncio.run(cleanup_coro())
            logger.debug("Completed async cleanup via asyncio.run()")
        except RuntimeError as e:
            # asyncio.run() failed (e.g., interpreter shutting down)
            logger.debug(f"asyncio.run() failed: {e}, falling back to sync reset")
            _safe_reset_all()

    except Exception as e:
        # Cleanup failures during shutdown should not raise
        # Just reset the singletons synchronously (may leak resources)
        logger.warning(f"Async cleanup failed, resetting singletons: {e}")
        _safe_reset_all()
