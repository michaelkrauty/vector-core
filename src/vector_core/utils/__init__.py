"""Utility functions for vector-core."""

from vector_core.utils.async_helpers import (
    AsyncSingleton,
    SingletonInitError,
    clear_init_locks,
    get_async_init_lock,
)
from vector_core.utils.datetime import (
    DEFAULT_DATETIME,
    DEFAULT_TIMESTAMP,
    now_utc,
    parse_iso_datetime,
    parse_payload_timestamps,
)
from vector_core.utils.hashing import compute_file_hash, hash_content
from vector_core.utils.sentinel import UNSET, UnsetType, is_set
from vector_core.utils.sqlite import (
    DEFAULT_SQLITE_CONFIG,
    SQLiteConfig,
    ThreadSafeSQLiteStore,
)
from vector_core.utils.validation import (
    DEFAULT_MAX_LIMIT,
    DEFAULT_MIN_LIMIT,
    parse_uuid_or_none,
    validate_limit,
    validate_uuid_string,
)

__all__ = [
    # Hashing
    "hash_content",
    "compute_file_hash",
    # Async helpers
    "get_async_init_lock",
    "clear_init_locks",
    "AsyncSingleton",
    "SingletonInitError",
    # Datetime
    "parse_iso_datetime",
    "parse_payload_timestamps",
    "now_utc",
    "DEFAULT_DATETIME",
    "DEFAULT_TIMESTAMP",
    # Sentinel
    "UNSET",
    "UnsetType",
    "is_set",
    # SQLite
    "SQLiteConfig",
    "ThreadSafeSQLiteStore",
    "DEFAULT_SQLITE_CONFIG",
    # Validation
    "validate_limit",
    "validate_uuid_string",
    "parse_uuid_or_none",
    "DEFAULT_MIN_LIMIT",
    "DEFAULT_MAX_LIMIT",
]
