"""Persistent SQLite-backed embedding cache."""

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from vector_core.settings import settings
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore


class EmbeddingCache(ThreadSafeSQLiteStore):
    """
    Persistent disk-backed embedding cache using SQLite.

    Features:
    - Thread-safe (SQLite with proper locking via ThreadSafeSQLiteStore)
    - LRU eviction when cache exceeds size limit
    - Cache statistics (hits, misses)
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        max_entries: int | None = None,
    ):
        """
        Initialize embedding cache.

        Args:
            cache_path: Path to SQLite database file. Default: ~/.cache/vector-core/embeddings.db
            max_entries: Maximum number of entries before LRU eviction. Default from settings.
        """
        db_path = cache_path or (settings.cache_dir / "embeddings.db")
        super().__init__(db_path, config=SQLiteConfig())
        self.max_entries = max_entries or settings.cache_max_entries
        self._stats_lock = threading.Lock()  # Only protects in-memory stats dict
        self._stats = {"hits": 0, "misses": 0}
        self._ensure_parent_dir()
        self._init_db()

    # Alias for backward compatibility (cache_path was the public name)
    @property
    def cache_path(self) -> Path:
        """Path to the cache database file."""
        return self.db_path

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB,
                model TEXT,
                dim INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON embeddings(accessed_at)
        """)
        conn.commit()

    @staticmethod
    def hash_content(text: str) -> str:
        """SHA256 hash of text content."""
        from vector_core.utils.hashing import hash_content
        return hash_content(text)

    def get(self, content_hash: str) -> list[float] | None:
        """
        Get cached embedding by content hash.

        Args:
            content_hash: SHA256 hash of the text content

        Returns:
            Embedding vector or None if not cached
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT embedding FROM embeddings WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()

        if row is None:
            with self._stats_lock:
                self._stats["misses"] += 1
            return None

        with self._stats_lock:
            self._stats["hits"] += 1

        # Update accessed_at for LRU
        conn.execute(
            "UPDATE embeddings SET accessed_at = ? WHERE content_hash = ?",
            (datetime.now(UTC).isoformat(), content_hash)
        )
        conn.commit()

        # Deserialize JSON (pickle support removed for security)
        blob = row[0]
        try:
            if isinstance(blob, bytes):
                return cast(list[float], json.loads(blob.decode("utf-8")))
            else:
                # Already a string (shouldn't happen but handle gracefully)
                return cast(list[float], json.loads(blob))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Corrupted or legacy pickle entry - treat as cache miss
            # Delete the corrupted entry to allow re-caching
            conn.execute(
                "DELETE FROM embeddings WHERE content_hash = ?",
                (content_hash,)
            )
            conn.commit()
            with self._stats_lock:
                self._stats["hits"] -= 1  # Undo the hit count
                self._stats["misses"] += 1
            return None

    def set(
        self,
        content_hash: str,
        embedding: list[float],
        model: str | None = None,
    ) -> None:
        """
        Store embedding with optional metadata.

        Args:
            content_hash: SHA256 hash of the text content
            embedding: Embedding vector
            model: Model name used to generate embedding
        """
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (content_hash, embedding, model, dim, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                json.dumps(embedding).encode("utf-8"),  # JSON serialization (secure)
                model or settings.embedding_model,
                len(embedding),
                now,
                now,
            )
        )
        conn.commit()

        # Evict old entries if over limit
        self._maybe_evict()

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], list[float]],
    ) -> list[float]:
        """
        Get from cache or compute and cache.

        Args:
            text: Text to embed
            compute_fn: Function to compute embedding if not cached

        Returns:
            Embedding vector (from cache or freshly computed)
        """
        content_hash = self.hash_content(text)

        # Try cache first
        cached = self.get(content_hash)
        if cached is not None:
            return cached

        # Compute and cache
        embedding = compute_fn(text)
        self.set(content_hash, embedding)
        return embedding

    async def get_or_compute_async(
        self,
        text: str,
        compute_fn: Callable,
    ) -> list[float]:
        """
        Async version of get_or_compute.

        Args:
            text: Text to embed
            compute_fn: Async function to compute embedding if not cached

        Returns:
            Embedding vector (from cache or freshly computed)
        """
        content_hash = self.hash_content(text)

        # Try cache first
        cached = self.get(content_hash)
        if cached is not None:
            return cached

        # Compute and cache
        embedding: list[float] = await compute_fn(text)
        self.set(content_hash, embedding)
        return embedding

    def _maybe_evict(self) -> None:
        """Evict oldest entries if cache exceeds limit."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]

        if count > self.max_entries:
            # Delete oldest 10% to avoid frequent evictions
            to_delete = max(1, int(self.max_entries * 0.1))
            conn.execute(
                """
                DELETE FROM embeddings WHERE content_hash IN (
                    SELECT content_hash FROM embeddings
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
                """,
                (to_delete,)
            )
            conn.commit()

    def stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings")
        row = cursor.fetchone()
        count = row[0] or 0
        size_bytes = row[1] or 0

        with self._stats_lock:
            hits = self._stats["hits"]
            misses = self._stats["misses"]

        return {
            "entries": count,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "hits": hits,
            "misses": misses,
            "hit_rate": (
                round(hits / (hits + misses), 3)
                if (hits + misses) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        conn = self._get_conn()
        conn.execute("DELETE FROM embeddings")
        conn.commit()
        with self._stats_lock:
            self._stats = {"hits": 0, "misses": 0}
