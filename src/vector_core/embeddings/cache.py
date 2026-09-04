"""Persistent SQLite-backed embedding cache."""

import json
import logging
import math
import sqlite3
import struct
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from vector_core.settings import settings
from vector_core.utils.hashing import hash_content
from vector_core.utils.locking import file_lock
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore

EMBEDDING_CACHE_SCHEMA_VERSION = "binary-f32-v1"
EMBEDDING_PREPROCESSING_VERSION = "truncate-chars-v1"
logger = logging.getLogger(__name__)


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
        # Several MCP processes commonly start together. Serialize PRAGMA/DDL
        # initialization so a harmless first-use race cannot disable caching in
        # every process except the winner.
        with file_lock(db_path, timeout=30.0):
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache_v2 (
                cache_key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_v2_accessed_at
            ON embedding_cache_v2(accessed_at)
        """)
        conn.commit()

    @staticmethod
    def hash_content(text: str) -> str:
        """SHA256 hash of text content."""
        return hash_content(text)

    @classmethod
    def make_key(
        cls,
        text: str,
        *,
        namespace: str,
        model: str,
        dim: int,
        preprocessing_version: str = EMBEDDING_PREPROCESSING_VERSION,
    ) -> str:
        """Build a model-safe key for an effective, already-preprocessed input."""
        if not namespace:
            raise ValueError("embedding cache namespace must be non-empty")
        if dim <= 0:
            raise ValueError("embedding cache dimension must be positive")
        key_data = {
            "cache_schema": EMBEDDING_CACHE_SCHEMA_VERSION,
            "content_hash": cls.hash_content(text),
            "dim": dim,
            "model": model,
            "namespace": namespace,
            "preprocessing": preprocessing_version,
        }
        return cls.hash_content(json.dumps(key_data, sort_keys=True, separators=(",", ":")))

    def get_many(
        self,
        cache_keys: Sequence[str],
        *,
        expected_dim: int,
    ) -> dict[str, list[float]]:
        """Read and validate many binary float32 vectors in one transaction."""
        unique_keys = list(dict.fromkeys(cache_keys))
        if not unique_keys:
            return {}
        if expected_dim <= 0:
            raise ValueError("expected_dim must be positive")

        conn = self._get_conn()
        rows: list[tuple[str, bytes, int]] = []
        now = datetime.now(UTC).isoformat()
        with conn:
            conn.execute("BEGIN")
            # Stay below SQLite builds that retain the historical 999-variable limit.
            for start in range(0, len(unique_keys), 900):
                chunk = unique_keys[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        "SELECT cache_key, embedding, dim FROM embedding_cache_v2 "
                        f"WHERE cache_key IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )

        results: dict[str, list[float]] = {}
        invalid: list[tuple[str]] = []
        expected_bytes = expected_dim * 4
        for cache_key, blob, stored_dim in rows:
            if (
                stored_dim != expected_dim
                or not isinstance(blob, bytes)
                or len(blob) != expected_bytes
            ):
                invalid.append((cache_key,))
                continue
            vector = list(struct.unpack(f"<{expected_dim}f", blob))
            if not all(math.isfinite(value) for value in vector):
                invalid.append((cache_key,))
                continue
            results[cache_key] = vector

        # Access timestamps and corrupt-entry cleanup are maintenance only. Do
        # not turn a valid read into a miss if another process wins this write.
        try:
            with conn:
                if invalid:
                    conn.executemany("DELETE FROM embedding_cache_v2 WHERE cache_key = ?", invalid)
                if results:
                    conn.executemany(
                        "UPDATE embedding_cache_v2 SET accessed_at = ? WHERE cache_key = ?",
                        ((now, key) for key in results),
                    )
        except sqlite3.Error:
            logger.debug("Could not update embedding cache access metadata", exc_info=True)

        with self._stats_lock:
            self._stats["hits"] += len(results)
            self._stats["misses"] += len(unique_keys) - len(results)
        return results

    def set_many(
        self,
        embeddings: Mapping[str, Sequence[float]],
        *,
        expected_dim: int,
    ) -> None:
        """Validate and write many vectors as little-endian float32 atomically."""
        if not embeddings:
            return
        if expected_dim <= 0:
            raise ValueError("expected_dim must be positive")

        now = datetime.now(UTC).isoformat()
        rows: list[tuple[str, bytes, int, str, str]] = []
        for cache_key, vector in embeddings.items():
            if len(vector) != expected_dim or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"invalid embedding for {cache_key}: expected {expected_dim} finite values"
                )
            blob = struct.pack(f"<{expected_dim}f", *vector)
            rows.append((cache_key, blob, expected_dim, now, now))

        conn = self._get_conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO embedding_cache_v2
                    (cache_key, embedding, dim, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    accessed_at = excluded.accessed_at
                """,
                rows,
            )
            self._maybe_evict_combined(conn)

    def _maybe_evict_combined(self, conn: sqlite3.Connection) -> None:
        """Enforce one entry limit across legacy and deployment-safe caches."""
        legacy_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        v2_count = conn.execute("SELECT COUNT(*) FROM embedding_cache_v2").fetchone()[0]
        if legacy_count + v2_count <= self.max_entries:
            return
        to_delete = max(
            legacy_count + v2_count - self.max_entries,
            int(self.max_entries * 0.1),
            1,
        )
        # Prefer retiring legacy entries because their keys do not identify an
        # embedding deployment. Preserve their own LRU order while doing so.
        legacy_delete = min(legacy_count, to_delete)
        if legacy_delete:
            conn.execute(
                """
                DELETE FROM embeddings WHERE content_hash IN (
                    SELECT content_hash FROM embeddings
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
                """,
                (legacy_delete,),
            )
        v2_delete = to_delete - legacy_delete
        if v2_delete:
            conn.execute(
                """
                DELETE FROM embedding_cache_v2 WHERE cache_key IN (
                    SELECT cache_key FROM embedding_cache_v2
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
                """,
                (v2_delete,),
            )

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
            "SELECT embedding FROM embeddings WHERE content_hash = ?", (content_hash,)
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
            (datetime.now(UTC).isoformat(), content_hash),
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
            conn.execute("DELETE FROM embeddings WHERE content_hash = ?", (content_hash,))
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

        with conn:
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
                ),
            )
            self._maybe_evict_combined(conn)

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
        with conn:
            self._maybe_evict_combined(conn)

    def stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT COUNT(*), SUM(size) FROM (
                SELECT LENGTH(embedding) AS size FROM embeddings
                UNION ALL
                SELECT LENGTH(embedding) AS size FROM embedding_cache_v2
            )
        """)
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
            "hit_rate": (round(hits / (hits + misses), 3) if (hits + misses) > 0 else 0.0),
        }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        conn = self._get_conn()
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM embedding_cache_v2")
        conn.commit()
        with self._stats_lock:
            self._stats = {"hits": 0, "misses": 0}
