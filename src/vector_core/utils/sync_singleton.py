"""Thread-safe synchronous singleton manager.

This module provides a sync equivalent to AsyncSingleton for managing
non-async resources that need thread-safe lazy initialization.

Usage:
    from vector_core.utils.sync_singleton import SyncSingleton

    _store: SyncSingleton[NoteStore] = SyncSingleton("note_store")

    def get_store() -> NoteStore:
        def _create() -> NoteStore:
            store = NoteStore()
            store.ensure_directories()
            return store
        return _store.get(_create)
"""

import logging
import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class SyncSingleton(Generic[T]):
    """
    Thread-safe synchronous singleton manager using RLock.

    Mirrors the AsyncSingleton API for consistency:
    - .get(factory) - Get or create instance
    - .is_initialized - Check if initialized without triggering init
    - .reset() - Reset for testing

    Uses RLock to allow reentrant calls (useful if factory calls get()).

    Example:
        _db: SyncSingleton[Database] = SyncSingleton("database")

        def get_db() -> Database:
            return _db.get(lambda: Database(connection_string))

        # Later...
        def cleanup():
            _db.reset()  # For testing only - may leak resources
    """

    def __init__(self, name: str):
        """
        Initialize singleton manager.

        Args:
            name: Human-readable name for logging
        """
        self._name = name
        self._instance: T | None = None
        self._lock = threading.RLock()  # Reentrant for nested calls

    @property
    def name(self) -> str:
        """Get singleton name."""
        return self._name

    @property
    def is_initialized(self) -> bool:
        """Check if singleton is initialized (without triggering init)."""
        return self._instance is not None

    def get(self, factory: Callable[[], T]) -> T:
        """
        Get or create the singleton instance (thread-safe).

        Uses double-checked locking pattern:
        1. Fast path: return existing instance without lock
        2. Slow path: acquire lock, check again, create if needed

        Args:
            factory: Callable that creates the instance

        Returns:
            The singleton instance

        Raises:
            Exception: Whatever the factory raises on failure
        """
        # Fast path: instance already exists
        # Store in local variable to avoid TOCTOU race with reset()
        instance = self._instance
        if instance is not None:
            return instance

        # Slow path: acquire lock and check again
        with self._lock:
            # Double-check after acquiring lock
            if self._instance is None:
                logger.debug(f"Initializing singleton '{self._name}'")
                self._instance = factory()
                logger.debug(f"Initialized singleton '{self._name}'")
            return self._instance

    def reset(self) -> None:
        """
        Reset singleton state (for testing).

        WARNING: May leak resources if instance has open connections.
        For production cleanup, implement proper shutdown logic in factory.
        """
        with self._lock:
            if self._instance is not None:
                logger.debug(f"Resetting singleton '{self._name}'")
                self._instance = None

    def get_if_initialized(self) -> T | None:
        """
        Get instance only if already initialized.

        Useful for cleanup routines that should skip uninitialized singletons.

        Returns:
            The instance if initialized, None otherwise
        """
        return self._instance

    def set_instance(self, instance: T | None) -> None:
        """
        Set singleton instance directly (for testing).

        WARNING: This bypasses the factory pattern and should only be used
        in test fixtures to inject mock/temp instances.

        Args:
            instance: The instance to set, or None to clear
        """
        with self._lock:
            self._instance = instance
