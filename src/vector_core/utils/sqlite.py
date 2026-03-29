"""Thread-safe SQLite utilities for consistent database access patterns.

Provides a base class that eliminates the boilerplate of:
- Thread-local connection management
- Double-checked locking for connection initialization
- Standard PRAGMA configuration (WAL mode, timeouts)
- Proper resource cleanup across all threads

Usage:
    class MyStore(ThreadSafeSQLiteStore):
        def __init__(self, db_path: Path):
            super().__init__(db_path, config=SQLiteConfig(busy_timeout_ms=10000))
            self._init_db()

        def _init_db(self) -> None:
            conn = self._get_conn()
            conn.execute("CREATE TABLE IF NOT EXISTS ...")
            conn.commit()
"""

import atexit
import logging
import sqlite3
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

# Track all ThreadSafeSQLiteStore instances for cleanup at exit
# Using WeakSet so instances can still be garbage collected normally
_all_stores: weakref.WeakSet["ThreadSafeSQLiteStore"] = weakref.WeakSet()
_stores_lock = threading.Lock()
_atexit_registered = False


def _cleanup_all_stores() -> None:
    """Close all tracked SQLite stores at interpreter exit.

    This is registered via atexit to ensure connections are properly
    closed even if explicit close() calls are missed.
    """
    with _stores_lock:
        for store in list(_all_stores):
            try:
                store.close()
            except Exception:
                # Silently ignore - we're shutting down
                pass


@dataclass(frozen=True)
class SQLiteConfig:
    """Configuration for SQLite connections.

    Sensible defaults for concurrent access patterns:
    - WAL mode: Allows concurrent readers with single writer
    - 5-second busy timeout: Prevents immediate lock failures
    - NORMAL synchronous: Good balance of safety and performance
    """

    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000
    synchronous: str = "NORMAL"
    foreign_keys: bool = False
    connect_timeout: float = 30.0
    check_same_thread: bool = False

    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply configuration PRAGMAs to connection."""
        conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute(f"PRAGMA synchronous={self.synchronous}")
        if self.foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")


# Default configuration used when none specified
DEFAULT_SQLITE_CONFIG = SQLiteConfig()


class ThreadSafeSQLiteStore:
    """Base class for thread-safe SQLite stores.

    Handles:
    - Per-thread connection management (thread-local storage)
    - Double-checked locking for thread-safe initialization
    - Standard PRAGMA configuration via SQLiteConfig
    - Proper cleanup of all connections on close()

    Subclasses should:
    1. Call super().__init__(db_path, config) in __init__
    2. Call _init_db() to create schema after super().__init__
    3. Use _get_conn() to get thread-local connection
    4. Override _init_db() to create tables/indexes
    """

    def __init__(
        self,
        db_path: Path,
        config: SQLiteConfig | None = None,
    ):
        """Initialize the store.

        Args:
            db_path: Path to SQLite database file
            config: SQLite configuration (uses defaults if None)
        """
        global _atexit_registered

        self.db_path = db_path
        self._config = config or DEFAULT_SQLITE_CONFIG
        self._local = threading.local()
        self._conn_lock = threading.Lock()
        # Track all connections by thread ID for proper cleanup
        self._all_conns: dict[int, sqlite3.Connection] = {}

        # Register this instance for cleanup at exit
        with _stores_lock:
            _all_stores.add(self)
            # Lazily register atexit handler on first instance
            if not _atexit_registered:
                atexit.register(_cleanup_all_stores)
                _atexit_registered = True

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection with thread-safe initialization and health check.

        Uses double-checked locking pattern:
        1. Fast check without lock (most common case - connection exists)
        2. Validate existing connection with health check
        3. Acquire lock and reconnect if needed
        4. Double-check after lock (another thread may have initialized)

        Returns:
            Thread-local SQLite connection
        """
        # Fast path: check if we have a connection
        if hasattr(self._local, "conn") and self._local.conn is not None:
            # Validate connection is healthy
            if self._is_connection_healthy(self._local.conn):
                return cast(sqlite3.Connection, self._local.conn)
            else:
                # Connection unhealthy, clear it for reconnection
                logger.debug(
                    f"SQLite connection unhealthy for thread {threading.get_ident()}, reconnecting"
                )
                self._close_thread_conn()

        # Slow path: need to create connection
        with self._conn_lock:
            # Double-check after acquiring lock
            if not hasattr(self._local, "conn") or self._local.conn is None:
                conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=self._config.check_same_thread,
                    timeout=self._config.connect_timeout,
                )
                self._config.apply(conn)
                self._local.conn = conn
                self._all_conns[threading.get_ident()] = conn
                logger.debug(
                    f"Created SQLite connection for thread {threading.get_ident()}"
                )
        return cast(sqlite3.Connection, self._local.conn)

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is healthy with a simple query.

        Args:
            conn: Connection to validate

        Returns:
            True if connection responds to queries, False if dead/corrupted
        """
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def _close_thread_conn(self) -> None:
        """Close current thread's connection (called when unhealthy)."""
        thread_id = threading.get_ident()
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except sqlite3.Error:
                pass  # Ignore errors on dead connection
            self._local.conn = None
        with self._conn_lock:
            self._all_conns.pop(thread_id, None)

    def _ensure_parent_dir(self) -> None:
        """Ensure the parent directory of db_path exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Close all database connections across all threads.

        Should be called during graceful shutdown. Safe to call multiple times.
        """
        with self._conn_lock:
            for thread_id, conn in self._all_conns.items():
                try:
                    conn.close()
                    logger.debug(f"Closed SQLite connection for thread {thread_id}")
                except sqlite3.Error as e:
                    logger.debug(f"Error closing connection for thread {thread_id}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error closing connection: {e}")
            self._all_conns.clear()
        if hasattr(self._local, "conn"):
            self._local.conn = None

    def __del__(self) -> None:
        """Ensure connections are closed on garbage collection.

        This is a safety net - prefer explicit close() calls.
        Silent on errors since this runs during interpreter shutdown.
        """
        try:
            self.close()
        except Exception:
            # Silently ignore: during interpreter shutdown, logging/resources
            # may be partially destroyed, making any error handling unreliable
            pass

    def __enter__(self) -> "ThreadSafeSQLiteStore":
        """Support context manager protocol."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close on context exit."""
        self.close()
