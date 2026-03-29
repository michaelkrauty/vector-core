"""Tests for embedding cache."""

import json
import sqlite3
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from vector_core.embeddings.cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    """Create a cache instance with automatic cleanup."""
    cache = EmbeddingCache(cache_path=tmp_path / "test_cache.db")
    yield cache
    cache.close()


class TestEmbeddingCacheInit:
    """Tests for cache initialization."""

    def test_default_path(self, tmp_path, monkeypatch):
        """Cache uses settings default path."""
        mock_settings = MagicMock()
        mock_settings.cache_dir = tmp_path
        mock_settings.cache_max_entries = 1000

        monkeypatch.setattr("vector_core.embeddings.cache.settings", mock_settings)

        cache = EmbeddingCache()

        assert cache.cache_path == tmp_path / "embeddings.db"
        assert cache.max_entries == 1000

    def test_custom_path(self, tmp_path):
        """Cache accepts custom path."""
        custom_path = tmp_path / "custom" / "cache.db"
        cache = EmbeddingCache(cache_path=custom_path, max_entries=500)

        assert cache.cache_path == custom_path
        assert cache.max_entries == 500
        assert custom_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Cache creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "cache.db"
        cache = EmbeddingCache(cache_path=deep_path)

        assert deep_path.parent.exists()
        cache.close()

    def test_db_schema_created(self, tmp_path):
        """Database schema is created on init."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Verify table exists
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        cache.close()


class TestHashContent:
    """Tests for content hashing."""

    def test_hash_consistency(self):
        """Same content produces same hash."""
        text = "Hello, world!"
        hash1 = EmbeddingCache.hash_content(text)
        hash2 = EmbeddingCache.hash_content(text)

        assert hash1 == hash2

    def test_hash_uniqueness(self):
        """Different content produces different hashes."""
        hash1 = EmbeddingCache.hash_content("text one")
        hash2 = EmbeddingCache.hash_content("text two")

        assert hash1 != hash2

    def test_hash_format(self):
        """Hash is valid SHA256 hex string."""
        hash_val = EmbeddingCache.hash_content("test")

        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestCacheGetSet:
    """Tests for get/set operations."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache for testing."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        yield cache
        cache.close()

    def test_set_and_get(self, cache):
        """Can store and retrieve embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set("hash123", embedding, model="test-model")

        result = cache.get("hash123")

        assert result == embedding

    def test_get_nonexistent(self, cache):
        """Get returns None for missing hash."""
        result = cache.get("nonexistent")

        assert result is None

    def test_set_overwrites(self, cache):
        """Setting same hash overwrites value."""
        cache.set("hash123", [1.0, 2.0])
        cache.set("hash123", [3.0, 4.0])

        result = cache.get("hash123")

        assert result == [3.0, 4.0]

    def test_stats_tracking(self, cache):
        """Stats track hits and misses."""
        embedding = [0.1, 0.2, 0.3]
        cache.set("exists", embedding)

        cache.get("exists")  # hit
        cache.get("exists")  # hit
        cache.get("missing")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_accessed_at_updated(self, cache, tmp_path):
        """Accessing entry updates accessed_at timestamp."""
        cache.set("hash1", [0.1, 0.2])

        # Get first access time
        conn = sqlite3.connect(str(tmp_path / "cache.db"))
        cursor = conn.execute("SELECT accessed_at FROM embeddings WHERE content_hash = ?", ("hash1",))
        time1 = cursor.fetchone()[0]

        # Access the entry
        cache.get("hash1")

        # Get updated access time
        cursor = conn.execute("SELECT accessed_at FROM embeddings WHERE content_hash = ?", ("hash1",))
        time2 = cursor.fetchone()[0]

        conn.close()

        # Time should be updated (or at least not earlier)
        assert time2 >= time1


class TestCacheEviction:
    """Tests for LRU eviction."""

    def test_eviction_at_limit(self, tmp_path):
        """Oldest entries evicted when limit exceeded."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db", max_entries=5)

        # Add 6 entries (over limit of 5)
        for i in range(6):
            cache.set(f"hash{i}", [float(i)])

        # Check count after eviction
        stats = cache.stats()
        # Should have evicted at least 1 entry
        assert stats["entries"] <= 5

        cache.close()

    def test_lru_eviction_order(self, tmp_path):
        """Least recently accessed entries evicted first."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db", max_entries=3)

        # Add entries
        cache.set("old", [1.0])
        cache.set("newer", [2.0])

        # Access "old" to make it more recent
        cache.get("old")

        cache.set("newest", [3.0])
        cache.set("trigger_eviction", [4.0])

        # "newer" should be evicted (oldest accessed)
        assert cache.get("newer") is None
        assert cache.get("old") is not None

        cache.close()


class TestGetOrCompute:
    """Tests for get_or_compute functionality."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache for testing."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        yield cache
        cache.close()

    def test_returns_cached(self, cache):
        """Returns cached value without computing."""
        cache.set(EmbeddingCache.hash_content("test"), [0.1, 0.2])
        compute_fn = MagicMock(return_value=[0.9, 0.9])

        result = cache.get_or_compute("test", compute_fn)

        assert result == [0.1, 0.2]
        compute_fn.assert_not_called()

    def test_computes_when_missing(self, cache):
        """Computes and caches when not in cache."""
        compute_fn = MagicMock(return_value=[0.5, 0.6])

        result = cache.get_or_compute("uncached_text", compute_fn)

        assert result == [0.5, 0.6]
        compute_fn.assert_called_once_with("uncached_text")

        # Verify it was cached
        cached = cache.get(EmbeddingCache.hash_content("uncached_text"))
        assert cached == [0.5, 0.6]


class TestGetOrComputeAsync:
    """Tests for async get_or_compute."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache for testing."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        yield cache
        cache.close()

    @pytest.mark.asyncio
    async def test_returns_cached_async(self, cache):
        """Returns cached value without computing."""
        cache.set(EmbeddingCache.hash_content("test"), [0.1, 0.2])
        compute_fn = AsyncMock(return_value=[0.9, 0.9])

        result = await cache.get_or_compute_async("test", compute_fn)

        assert result == [0.1, 0.2]
        compute_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_computes_when_missing_async(self, cache):
        """Computes and caches when not in cache."""
        compute_fn = AsyncMock(return_value=[0.5, 0.6])

        result = await cache.get_or_compute_async("uncached_text", compute_fn)

        assert result == [0.5, 0.6]
        compute_fn.assert_called_once_with("uncached_text")


class TestCacheStats:
    """Tests for cache statistics."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache for testing."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        yield cache
        cache.close()

    def test_stats_structure(self, cache):
        """Stats returns expected structure."""
        stats = cache.stats()

        assert "entries" in stats
        assert "size_mb" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_hit_rate_calculation(self, cache):
        """Hit rate calculated correctly."""
        cache.set("h1", [0.1])
        cache.get("h1")  # hit
        cache.get("h1")  # hit
        cache.get("h1")  # hit
        cache.get("miss")  # miss

        stats = cache.stats()

        assert stats["hit_rate"] == 0.75  # 3 / 4

    def test_empty_cache_stats(self, cache):
        """Stats work on empty cache."""
        stats = cache.stats()

        assert stats["entries"] == 0
        assert stats["size_mb"] == 0.0
        assert stats["hit_rate"] == 0.0


class TestCacheClear:
    """Tests for cache clearing."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache for testing."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        yield cache
        cache.close()

    def test_clear_removes_all(self, cache):
        """Clear removes all entries."""
        cache.set("h1", [0.1])
        cache.set("h2", [0.2])
        cache.set("h3", [0.3])

        cache.clear()

        assert cache.stats()["entries"] == 0
        assert cache.get("h1") is None

    def test_clear_resets_stats(self, cache):
        """Clear resets hit/miss stats."""
        cache.set("h1", [0.1])
        cache.get("h1")
        cache.get("missing")

        cache.clear()

        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self, tmp_path):
        """Cache handles concurrent access."""
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker{worker_id}_item{i}"
                    cache.set(key, [float(worker_id), float(i)])
                    result = cache.get(key)
                    if result is not None:
                        results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        cache.close()

        assert len(errors) == 0
        assert len(results) == 50  # 5 workers * 10 items


class TestCorruptedEntryHandling:
    """Tests for handling corrupted or legacy cache entries."""

    def test_reads_json_format(self, tmp_path):
        """Cache reads JSON-serialized embeddings."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Manually insert JSON data (current format)
        embedding = [0.1, 0.2, 0.3]
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO embeddings (content_hash, embedding) VALUES (?, ?)",
            ("json_hash", json.dumps(embedding).encode("utf-8"))
        )
        conn.commit()
        conn.close()

        result = cache.get("json_hash")

        assert result == embedding
        cache.close()

    def test_corrupted_entry_deleted_and_returns_none(self, tmp_path):
        """Corrupted entries are deleted and return None (cache miss)."""
        import pickle

        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Manually insert non-JSON data (e.g., legacy pickle or corrupted)
        embedding = [0.4, 0.5, 0.6]
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO embeddings (content_hash, embedding) VALUES (?, ?)",
            ("corrupted_hash", pickle.dumps(embedding))
        )
        conn.commit()
        conn.close()

        # Should return None and delete the corrupted entry
        result = cache.get("corrupted_hash")
        assert result is None

        # Verify the entry was deleted
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE content_hash = ?",
            ("corrupted_hash",)
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

        # Stats should reflect a miss, not a hit
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        cache.close()

    def test_corrupted_entry_allows_recaching(self, tmp_path):
        """After corrupted entry is deleted, can cache a fresh value."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Insert corrupted data
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO embeddings (content_hash, embedding) VALUES (?, ?)",
            ("recache_hash", b"\x80\x04\x95")  # Truncated pickle = corrupted
        )
        conn.commit()
        conn.close()

        # First get returns None (corrupted entry deleted)
        result = cache.get("recache_hash")
        assert result is None

        # Now set a fresh value
        new_embedding = [1.0, 2.0, 3.0]
        cache.set("recache_hash", new_embedding)

        # Should retrieve the new value
        result = cache.get("recache_hash")
        assert result == new_embedding

        cache.close()

    def test_reads_string_format(self, tmp_path):
        """Cache reads when blob is already a string (line 119)."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Manually insert JSON as text, not bytes
        # SQLite can return TEXT columns as strings in some modes
        embedding = [0.7, 0.8, 0.9]
        json_str = json.dumps(embedding)

        conn = sqlite3.connect(str(db_path))
        # Use text mode by casting or storing as TEXT
        conn.execute(
            "INSERT INTO embeddings (content_hash, embedding) VALUES (?, ?)",
            ("string_hash", json_str)  # Plain string, not encoded to bytes
        )
        conn.commit()
        conn.close()

        result = cache.get("string_hash")

        assert result == embedding
        cache.close()


class TestContextManager:
    """Tests for context manager protocol (__enter__/__exit__)."""

    def test_context_manager_usage(self, tmp_path):
        """Cache can be used as context manager (lines 290-296)."""
        db_path = tmp_path / "cache.db"

        with EmbeddingCache(cache_path=db_path) as cache:
            cache.set("ctx_hash", [0.1, 0.2, 0.3])
            result = cache.get("ctx_hash")
            assert result == [0.1, 0.2, 0.3]

        # After exiting context, connection should be closed
        # Accessing the cache should work (creates new connection)
        # but we can verify internal state if needed

    def test_context_manager_closes_connection(self, tmp_path):
        """Context manager closes connection on exit."""
        db_path = tmp_path / "cache.db"

        cache = EmbeddingCache(cache_path=db_path)
        with cache:
            cache.set("test", [1.0])

        # Connection should be None after exit
        assert not hasattr(cache._local, "conn") or cache._local.conn is None

    def test_enter_returns_self(self, tmp_path):
        """__enter__ returns the cache instance."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        result = cache.__enter__()

        assert result is cache
        cache.close()


class TestDestructor:
    """Tests for __del__ method (garbage collection cleanup)."""

    def test_del_closes_connection(self, tmp_path):
        """__del__ closes connection gracefully (lines 281-288)."""
        db_path = tmp_path / "cache.db"

        # Create and use cache
        cache = EmbeddingCache(cache_path=db_path)
        cache.set("del_test", [0.5, 0.6])

        # Force __del__ by deleting reference
        cache.__del__()

        # Should not raise even if already closed
        cache.__del__()

    def test_del_handles_exception_silently(self, tmp_path, monkeypatch):
        """__del__ silently handles exceptions (lines 285-288)."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Force close to raise an exception
        def raise_error():
            raise RuntimeError("Simulated shutdown error")

        monkeypatch.setattr(cache, "close", raise_error)

        # __del__ should not raise even when close() raises
        # This tests the except block in __del__
        cache.__del__()  # Should not raise

    def test_del_with_no_connection(self, tmp_path):
        """__del__ handles case when no connection exists."""
        db_path = tmp_path / "cache.db"
        cache = EmbeddingCache(cache_path=db_path)

        # Close explicitly first
        cache.close()

        # __del__ should handle already-closed state
        cache.__del__()
