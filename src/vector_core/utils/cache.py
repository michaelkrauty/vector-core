"""Thread-safe LRU cache with TTL expiration.

This module provides a simple in-memory cache with:
- Time-based expiration (TTL)
- LRU eviction when full
- Thread-safe operations
- Prefix-based invalidation

Usage:
    from vector_core.utils.cache import TTLCache, CacheConfig

    cache: TTLCache[str] = TTLCache(CacheConfig(max_size=100, ttl_seconds=300))

    # Store and retrieve
    cache.set("query|hash", result)
    result = cache.get("query|hash")

    # Invalidate by prefix (e.g., after indexing)
    cache.invalidate_prefix("/path/to/project|")
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

V = TypeVar("V")


@dataclass
class CacheConfig:
    """Configuration for TTLCache.

    Attributes:
        max_size: Maximum number of entries before eviction
        ttl_seconds: Time-to-live in seconds (entries expire after this time)
        eviction_ratio: Fraction of entries to remove on eviction (0.0-1.0)
    """

    max_size: int = 100
    ttl_seconds: float = 300.0  # 5 minutes
    eviction_ratio: float = 0.2  # Remove 20% on eviction


class CacheStats:
    """Cache statistics for monitoring."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.expirations = 0
        self.evictions = 0

    def hit_rate(self) -> float:
        """Calculate hit rate (0.0-1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Return stats as dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expirations": self.expirations,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate(),
        }


class TTLCache(Generic[V]):
    """
    Thread-safe LRU cache with TTL expiration.

    Features:
    - True LRU behavior (access updates timestamp)
    - TTL expiration (entries become stale after ttl_seconds)
    - Configurable batch eviction (removes oldest entries when full)
    - Prefix-based invalidation (clear related entries)
    - Thread-safe via threading.Lock

    Example:
        config = CacheConfig(max_size=100, ttl_seconds=300)
        cache: TTLCache[str] = TTLCache(config)

        # Basic usage
        cache.set("key", "value")
        value = cache.get("key")  # Returns "value" or None if expired

        # Invalidation after updates
        cache.invalidate_prefix("project_a|")  # Clear all project_a entries

        # Statistics
        print(cache.stats.hit_rate())
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self._config = config or CacheConfig()
        # OrderedDict maintains insertion order for O(1) LRU eviction
        # Most recently used items are at the end
        self._data: OrderedDict[str, tuple[V, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get(self, key: str) -> V | None:
        """
        Get cached value if valid (thread-safe, updates LRU timestamp).

        Args:
            key: Cache key

        Returns:
            Cached value if present and not expired, None otherwise
        """
        with self._lock:
            if key not in self._data:
                self._stats.misses += 1
                return None

            value, timestamp = self._data[key]
            now = time.time()

            # Check expiration
            if now - timestamp > self._config.ttl_seconds:
                del self._data[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Update timestamp and move to end (LRU behavior) - O(1)
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            self._stats.hits += 1
            return value

    def set(self, key: str, value: V) -> None:
        """
        Store value in cache, evicting oldest entries if full (thread-safe).

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if len(self._data) >= self._config.max_size:
                self._evict_oldest()
            self._data[key] = (value, time.time())

    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if entry was present and removed, False otherwise
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Remove all entries with keys starting with prefix.

        Useful for clearing all cached results related to a specific project
        or path after indexing updates.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [k for k in self._data if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._data[key]
            if keys_to_remove:
                logger.debug(
                    f"Cache invalidated {len(keys_to_remove)} entries with prefix '{prefix}'"
                )
            return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            logger.debug(f"Cache cleared {count} entries")
            return count

    def size(self) -> int:
        """
        Get current number of entries in cache.

        Note: Some entries may be expired but not yet removed.

        Returns:
            Number of entries
        """
        return len(self._data)

    def _evict_oldest(self) -> None:
        """
        Remove oldest entries (called inside lock).

        Evicts a configurable fraction of entries based on eviction_ratio.
        Uses OrderedDict for O(1) per-item eviction (oldest items at front).
        """
        to_remove = max(1, int(self._config.max_size * self._config.eviction_ratio))
        # Pop from front (oldest items) - O(1) per item
        for _ in range(min(to_remove, len(self._data))):
            self._data.popitem(last=False)
            self._stats.evictions += 1

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Useful for periodic cleanup to free memory.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                k
                for k, (_, timestamp) in self._data.items()
                if now - timestamp > self._config.ttl_seconds
            ]
            for key in expired_keys:
                del self._data[key]
                self._stats.expirations += 1
            return len(expired_keys)

    def contains(self, key: str) -> bool:
        """
        Check if key exists and is not expired (without updating LRU).

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is valid
        """
        with self._lock:
            if key not in self._data:
                return False
            _, timestamp = self._data[key]
            return time.time() - timestamp <= self._config.ttl_seconds
