"""Global vocabulary for cross-codebase sparse vector search.

Enables comparable sparse scores across codebases by:
- Maintaining a single global vocabulary (token -> index mapping)
- Storing TF-only vectors in documents
- Computing IDF at query time from global corpus statistics
"""

import logging
import math
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vector_core.embeddings.sparse import SparseVector
from vector_core.embeddings.tokenization import (
    CAMEL_CASE_PATTERN,
    IDENTIFIER_PATTERN,
    DEFAULT_STOP_TOKENS,
    levenshtein_similarity,
)
from vector_core.settings import settings
from vector_core.utils.locking import file_lock
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1


class GlobalVocabulary(ThreadSafeSQLiteStore):
    """
    Thread-safe global vocabulary with SQLite persistence.

    Key design principles:
    - Append-only vocabulary indices (tokens never change index once assigned)
    - Per-codebase contribution tracking for clean removal
    - TF-only document vectors, IDF computed at query time
    - Thread-safe with per-thread connections (via ThreadSafeSQLiteStore)

    Usage:
        # Preferred: Use singleton for shared vocabulary
        vocab = GlobalVocabulary.get_instance()

        # Or create standalone instance
        vocab = GlobalVocabulary()

        # During indexing
        tokens_per_doc = [set(tokenize(doc)) for doc in documents]
        vocab.register_codebase("my_codebase", tokens_per_doc)

        for doc in documents:
            sparse_vec = vocab.vectorize_document(doc)
            # Store sparse_vec in Qdrant

        # During search
        query_vec = vocab.vectorize_query(query_text)
        # Use query_vec for sparse search
    """

    # Singleton instance for shared vocabulary across components
    _instance: "GlobalVocabulary | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GlobalVocabulary":
        """
        Get or create the singleton GlobalVocabulary instance.

        This ensures all components (indexers, search engines) share the same
        vocabulary and IDF statistics. The instance is thread-safe.

        Uses lock-first pattern to prevent race conditions. The performance
        overhead is negligible since get_instance() is called infrequently
        (once per component initialization).

        Returns:
            Shared GlobalVocabulary instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        Closes the existing instance and clears the singleton reference.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    def __init__(
        self,
        db_path: Path | None = None,
        min_token_length: int = 2,
        stop_tokens: set[str] | None = None,
        cache_ttl: float | None = None,
    ):
        """
        Initialize global vocabulary.

        Args:
            db_path: Path to SQLite database. Default: ~/.cache/vector-core/global_vocabulary.db
            min_token_length: Minimum token length to include
            stop_tokens: Set of stop tokens to filter. Default: common English words.
            cache_ttl: Cache TTL in seconds for multi-server consistency.
                       Default: settings.global_vocab_cache_ttl (5s).
                       Lower values improve consistency when multiple servers share the DB.
        """
        db_path = db_path or (settings.cache_dir / "global_vocabulary.db")
        super().__init__(db_path, config=SQLiteConfig())
        self.min_token_length = min_token_length
        self.stop_tokens = stop_tokens if stop_tokens is not None else DEFAULT_STOP_TOKENS

        # In-memory caches (refreshed from DB as needed)
        self._vocab_cache: dict[str, int] | None = None
        self._doc_freq_cache: dict[str, int] | None = None
        self._total_docs_cache: int | None = None
        self._cache_lock = threading.Lock()
        # Cache TTL for multi-server consistency (configurable via settings)
        self._cache_ttl = cache_ttl if cache_ttl is not None else settings.global_vocab_cache_ttl
        self._cache_timestamp: float = 0.0

        self._ensure_parent_dir()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()

        # Core vocabulary mapping (append-only)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                token TEXT PRIMARY KEY,
                idx INTEGER UNIQUE NOT NULL,
                doc_freq INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vocab_idx ON vocabulary(idx)")

        # Per-codebase doc_freq contributions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS codebase_contributions (
                codebase_id TEXT NOT NULL,
                token TEXT NOT NULL,
                doc_freq INTEGER NOT NULL,
                PRIMARY KEY (codebase_id, token)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_contrib_codebase ON codebase_contributions(codebase_id)"
        )

        # Codebase document counts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS codebase_doc_counts (
                codebase_id TEXT PRIMARY KEY,
                doc_count INTEGER NOT NULL,
                last_updated REAL NOT NULL
            )
        """)

        # Metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()

        # Check/set schema version
        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
        row = cursor.fetchone()
        if row is None:
            # New database - set schema version
            conn.execute(
                "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),)
            )
            conn.commit()
        else:
            db_version = int(row[0])
            if db_version > SCHEMA_VERSION:
                raise ValueError(
                    f"Database schema version ({db_version}) is newer than supported "
                    f"({SCHEMA_VERSION}). Please update vector-core."
                )
            # Future: add migration logic here if db_version < SCHEMA_VERSION

    def _invalidate_cache(self) -> None:
        """Invalidate all in-memory caches."""
        with self._cache_lock:
            self._vocab_cache = None
            self._doc_freq_cache = None
            self._total_docs_cache = None
            self._cache_timestamp = 0.0

    def _is_cache_expired(self) -> bool:
        """Check if cache has expired based on TTL."""
        return (time.time() - self._cache_timestamp) > self._cache_ttl

    def _refresh_cache_if_expired(self) -> None:
        """Invalidate cache if TTL has expired (with double-checked locking).

        Uses double-checked locking to ensure only one thread invalidates
        the cache when TTL expires, preventing unnecessary DB queries.

        Safety note: This pattern is safe in CPython due to the GIL (Global
        Interpreter Lock), which ensures atomic reads of the simple boolean
        check in _is_cache_expired(). The GIL prevents data races on the
        individual cache attributes. For alternative Python implementations
        without a GIL (e.g., future no-GIL Python), this would need a lock
        on the first check as well. Documented as intentional design decision.
        """
        if self._is_cache_expired():
            with self._cache_lock:
                # Double-check after acquiring lock
                if self._is_cache_expired():
                    self._vocab_cache = None
                    self._doc_freq_cache = None
                    self._total_docs_cache = None
                    self._cache_timestamp = 0.0

    def _get_vocab(self) -> dict[str, int]:
        """Get vocabulary mapping (token -> index), using cache with TTL."""
        self._refresh_cache_if_expired()

        with self._cache_lock:
            if self._vocab_cache is not None:
                return self._vocab_cache

        conn = self._get_conn()
        cursor = conn.execute("SELECT token, idx FROM vocabulary")
        vocab = {row[0]: row[1] for row in cursor.fetchall()}

        with self._cache_lock:
            self._vocab_cache = vocab
            if self._cache_timestamp == 0.0:
                self._cache_timestamp = time.time()
        return vocab

    def _get_doc_freq(self) -> dict[str, int]:
        """Get document frequency mapping (token -> count), using cache with TTL."""
        self._refresh_cache_if_expired()

        with self._cache_lock:
            if self._doc_freq_cache is not None:
                return self._doc_freq_cache

        conn = self._get_conn()
        cursor = conn.execute("SELECT token, doc_freq FROM vocabulary")
        doc_freq = {row[0]: row[1] for row in cursor.fetchall()}

        with self._cache_lock:
            self._doc_freq_cache = doc_freq
            if self._cache_timestamp == 0.0:
                self._cache_timestamp = time.time()
        return doc_freq

    @property
    def total_docs(self) -> int:
        """Total document count across all codebases."""
        self._refresh_cache_if_expired()

        with self._cache_lock:
            if self._total_docs_cache is not None:
                return self._total_docs_cache

        conn = self._get_conn()
        cursor = conn.execute("SELECT SUM(doc_count) FROM codebase_doc_counts")
        row = cursor.fetchone()
        total = row[0] or 0

        with self._cache_lock:
            self._total_docs_cache = total
            if self._cache_timestamp == 0.0:
                self._cache_timestamp = time.time()
        return total

    @property
    def vocab_size(self) -> int:
        """Total unique tokens in vocabulary."""
        return len(self._get_vocab())

    def get_codebase_doc_count(self, codebase_id: str) -> int:
        """Get document count for a specific codebase."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT doc_count FROM codebase_doc_counts WHERE codebase_id = ?",
            (codebase_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for sparse vectorization.

        Handles:
        - camelCase splitting (getUserData -> get, user, data)
        - snake_case splitting (get_user_data -> get, user, data)
        - Stop word filtering
        - Minimum length filtering

        Args:
            text: Text to tokenize

        Returns:
            List of normalized tokens
        """
        # Split on non-alphanumeric, keeping underscores
        raw_tokens = IDENTIFIER_PATTERN.findall(text)

        tokens = []
        for token in raw_tokens:
            # Split camelCase: getUserData -> get, User, Data
            parts = CAMEL_CASE_PATTERN.findall(token)
            if parts:
                tokens.extend(parts)
            else:
                # snake_case or single word
                tokens.extend(token.split("_"))

        # Normalize and filter
        result = []
        for token in tokens:
            normalized = token.lower()
            if (
                len(normalized) >= self.min_token_length
                and normalized not in self.stop_tokens
                and not normalized.isdigit()
            ):
                result.append(normalized)

        return result

    def _get_next_index(self, conn: sqlite3.Connection) -> int:
        """Get the next available vocabulary index."""
        cursor = conn.execute("SELECT MAX(idx) FROM vocabulary")
        row = cursor.fetchone()
        max_idx = row[0]
        return 0 if max_idx is None else max_idx + 1

    def register_codebase(
        self,
        codebase_id: str,
        tokens_per_doc: list[set[str]],
    ) -> int:
        """
        Register or update a codebase's vocabulary contribution.

        Call this during indexing with all document tokens.
        For updates, removes old contribution before adding new.

        Uses cross-process file locking to safely coordinate between
        multiple MCP servers sharing the same vocabulary database.

        Args:
            codebase_id: Unique identifier for the codebase
            tokens_per_doc: List of token sets, one per document

        Returns:
            Number of new tokens added to vocabulary
        """
        # Calculate new doc frequencies (outside lock for performance)
        new_doc_freq: Counter[str] = Counter()
        for doc_tokens in tokens_per_doc:
            for token in doc_tokens:
                new_doc_freq[token] += 1

        doc_count = len(tokens_per_doc)
        new_tokens_count = 0

        # Cross-process file lock for multi-server coordination
        # (registration is infrequent, so lock overhead is acceptable)
        with file_lock(self.db_path, timeout=60.0):
            conn = self._get_conn()
            now = time.time()

            # Thread lock for within-process safety
            with self._conn_lock:
                # Begin transaction BEFORE any writes
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Remove old contribution if exists (within transaction)
                    self._remove_codebase_contribution(conn, codebase_id)

                    # Get current max index (must be inside transaction for consistency)
                    next_idx = self._get_next_index(conn)

                    # Get existing vocab (fresh read within transaction)
                    cursor = conn.execute("SELECT token FROM vocabulary")
                    existing_tokens = {row[0] for row in cursor.fetchall()}

                    # Add new tokens to vocabulary
                    for token in new_doc_freq:
                        if token not in existing_tokens:
                            conn.execute(
                                """INSERT INTO vocabulary (token, idx, doc_freq, created_at)
                                VALUES (?, ?, 0, ?)""",
                                (token, next_idx, now),
                            )
                            next_idx += 1
                            new_tokens_count += 1

                    # Update doc_freq for all tokens
                    for token, freq in new_doc_freq.items():
                        conn.execute(
                            "UPDATE vocabulary SET doc_freq = doc_freq + ? WHERE token = ?",
                            (freq, token),
                        )

                    # Store per-codebase contributions
                    conn.executemany(
                        """INSERT INTO codebase_contributions (codebase_id, token, doc_freq)
                        VALUES (?, ?, ?)""",
                        [(codebase_id, token, freq) for token, freq in new_doc_freq.items()],
                    )

                    # Update doc count
                    conn.execute(
                        """INSERT OR REPLACE INTO codebase_doc_counts
                        (codebase_id, doc_count, last_updated) VALUES (?, ?, ?)""",
                        (codebase_id, doc_count, now),
                    )

                    conn.commit()
                except sqlite3.IntegrityError as e:
                    conn.rollback()
                    logger.error(f"Integrity error registering codebase {codebase_id}: {e}")
                    raise
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    error_msg = str(e).lower()
                    if "locked" in error_msg or "busy" in error_msg:
                        logger.warning(f"Database locked during codebase registration: {e}")
                    logger.error(f"Operational error registering codebase {codebase_id}: {e}")
                    raise
                except sqlite3.Error as e:
                    conn.rollback()
                    logger.error(f"SQLite error registering codebase {codebase_id}: {e}")
                    raise

        self._invalidate_cache()
        return new_tokens_count

    def _remove_codebase_contribution(
        self,
        conn: sqlite3.Connection,
        codebase_id: str,
    ) -> None:
        """Remove a codebase's contribution from global doc_freq."""
        # Get old contributions
        cursor = conn.execute(
            "SELECT token, doc_freq FROM codebase_contributions WHERE codebase_id = ?",
            (codebase_id,),
        )
        old_contributions = cursor.fetchall()

        if not old_contributions:
            return

        # Decrement global doc_freq
        for token, freq in old_contributions:
            conn.execute(
                "UPDATE vocabulary SET doc_freq = doc_freq - ? WHERE token = ?",
                (freq, token),
            )

        # Remove contribution records
        conn.execute(
            "DELETE FROM codebase_contributions WHERE codebase_id = ?",
            (codebase_id,),
        )

        # Remove doc count
        conn.execute(
            "DELETE FROM codebase_doc_counts WHERE codebase_id = ?",
            (codebase_id,),
        )

    def unregister_codebase(self, codebase_id: str) -> None:
        """
        Remove a codebase's vocabulary contribution.

        Call when deleting a codebase from the index.

        Uses cross-process file locking to safely coordinate between
        multiple MCP servers sharing the same vocabulary database.

        Args:
            codebase_id: Unique identifier for the codebase
        """
        # Cross-process file lock for multi-server coordination
        with file_lock(self.db_path, timeout=60.0):
            conn = self._get_conn()

            with self._conn_lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    self._remove_codebase_contribution(conn, codebase_id)
                    conn.commit()
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    error_msg = str(e).lower()
                    if "locked" in error_msg or "busy" in error_msg:
                        logger.warning(f"Database locked during codebase unregistration: {e}")
                    logger.error(f"Operational error unregistering codebase {codebase_id}: {e}")
                    raise
                except sqlite3.Error as e:
                    conn.rollback()
                    logger.error(f"SQLite error unregistering codebase {codebase_id}: {e}")
                    raise

        self._invalidate_cache()

    def update_codebase_incremental(
        self,
        codebase_id: str,
        added_tokens: list[set[str]],
        removed_tokens: list[set[str]],
        net_doc_change: int,
    ) -> int:
        """
        Incrementally update a codebase's vocabulary contribution.

        More efficient than full re-registration for small changes.

        Uses cross-process file locking to safely coordinate between
        multiple MCP servers sharing the same vocabulary database.

        Args:
            codebase_id: Unique identifier for the codebase
            added_tokens: Token sets from added/modified documents
            removed_tokens: Token sets from removed/modified documents
            net_doc_change: Net change in document count (can be negative)

        Returns:
            Number of new tokens added to vocabulary
        """
        # Calculate doc_freq deltas (outside lock for performance)
        added_freq: Counter[str] = Counter()
        for doc_tokens in added_tokens:
            for token in doc_tokens:
                added_freq[token] += 1

        removed_freq: Counter[str] = Counter()
        for doc_tokens in removed_tokens:
            for token in doc_tokens:
                removed_freq[token] += 1

        new_tokens_count = 0

        # Cross-process file lock for multi-server coordination
        with file_lock(self.db_path, timeout=60.0):
            conn = self._get_conn()
            now = time.time()

            with self._conn_lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Get current max index (inside transaction for consistency)
                    next_idx = self._get_next_index(conn)

                    # Get existing vocab (fresh read within transaction)
                    cursor = conn.execute("SELECT token FROM vocabulary")
                    existing_tokens = {row[0] for row in cursor.fetchall()}

                    # Add new tokens to vocabulary
                    for token in added_freq:
                        if token not in existing_tokens:
                            conn.execute(
                                """INSERT INTO vocabulary (token, idx, doc_freq, created_at)
                                VALUES (?, ?, 0, ?)""",
                                (token, next_idx, now),
                            )
                            next_idx += 1
                            new_tokens_count += 1

                    # Update global doc_freq
                    all_tokens = set(added_freq.keys()) | set(removed_freq.keys())
                    for token in all_tokens:
                        delta = added_freq.get(token, 0) - removed_freq.get(token, 0)
                        if delta != 0:
                            conn.execute(
                                "UPDATE vocabulary SET doc_freq = doc_freq + ? WHERE token = ?",
                                (delta, token),
                            )

                    # Update per-codebase contributions
                    for token, freq in removed_freq.items():
                        conn.execute(
                            """
                            UPDATE codebase_contributions
                            SET doc_freq = doc_freq - ?
                            WHERE codebase_id = ? AND token = ?
                            """,
                            (freq, codebase_id, token),
                        )

                    for token, freq in added_freq.items():
                        conn.execute(
                            """
                            INSERT INTO codebase_contributions (codebase_id, token, doc_freq)
                            VALUES (?, ?, ?)
                            ON CONFLICT(codebase_id, token) DO UPDATE SET doc_freq = doc_freq + ?
                            """,
                            (codebase_id, token, freq, freq),
                        )

                    # Clean up zero-count contributions
                    conn.execute(
                        "DELETE FROM codebase_contributions WHERE doc_freq <= 0",
                    )

                    # Update doc count
                    conn.execute(
                        """
                        UPDATE codebase_doc_counts
                        SET doc_count = doc_count + ?, last_updated = ?
                        WHERE codebase_id = ?
                        """,
                        (net_doc_change, now, codebase_id),
                    )

                    conn.commit()
                except sqlite3.IntegrityError as e:
                    conn.rollback()
                    logger.error(f"Integrity error updating codebase {codebase_id}: {e}")
                    raise
                except sqlite3.OperationalError as e:
                    conn.rollback()
                    error_msg = str(e).lower()
                    if "locked" in error_msg or "busy" in error_msg:
                        logger.warning(f"Database locked during incremental update: {e}")
                    logger.error(f"Operational error updating codebase {codebase_id}: {e}")
                    raise
                except sqlite3.Error as e:
                    conn.rollback()
                    logger.error(f"SQLite error updating codebase {codebase_id}: {e}")
                    raise

        self._invalidate_cache()
        return new_tokens_count

    def vectorize_document(self, text: str) -> SparseVector:
        """
        Create TF-only sparse vector for a document.

        IMPORTANT: Call register_codebase() BEFORE vectorizing documents to ensure
        all tokens are in the vocabulary. Tokens not in vocabulary are ignored.

        Documents store term frequency only - IDF applied at query time.

        Args:
            text: Document text to vectorize

        Returns:
            SparseVector with TF weights (empty if no tokens match vocabulary)
        """
        tokens = self.tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        vocab = self._get_vocab()
        tf = Counter(tokens)

        indices = []
        values = []

        for token, count in tf.items():
            if token in vocab:
                idx = vocab[token]
                # TF with log normalization: 1 + log(count)
                tf_weight = 1 + math.log(count) if count > 0 else 0
                indices.append(idx)
                values.append(tf_weight)

        # Sort by index for Qdrant
        if indices:
            pairs = sorted(zip(indices, values, strict=True))
            sorted_indices, sorted_values = zip(*pairs, strict=True)
            indices = list(sorted_indices)
            values = list(sorted_values)

        return SparseVector(indices=indices, values=values)

    def vectorize_query(
        self,
        query: str,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.75,
    ) -> SparseVector:
        """
        Create IDF-only sparse vector for a query.

        Query vectors use IDF weights computed from global corpus statistics.
        This enables comparable scoring across all indexed codebases.

        Args:
            query: Query text
            fuzzy: Whether to use fuzzy matching for unknown tokens
            fuzzy_threshold: Minimum similarity for fuzzy matches (0-1)

        Returns:
            SparseVector with IDF weights
        """
        tokens = self.tokenize(query)
        if not tokens:
            return SparseVector(indices=[], values=[])

        vocab = self._get_vocab()
        doc_freq = self._get_doc_freq()
        total = self.total_docs

        # Deduplicate by both token AND index
        seen_tokens: set[str] = set()
        seen_indices: set[int] = set()
        indices: list[int] = []
        values: list[float] = []

        for token in tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)

            matched_token = None
            similarity = 1.0

            if token in vocab:
                matched_token = token
            elif fuzzy:
                matched_token, similarity = self._find_fuzzy_match(
                    token, vocab, fuzzy_threshold
                )

            if matched_token:
                idx = vocab[matched_token]
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                # IDF with smoothing: log((N + 1) / (df + 1)) + 1
                df = doc_freq.get(matched_token, 0)
                idf = math.log((total + 1) / (df + 1)) + 1
                weight = idf * similarity  # Discount by fuzzy match quality

                indices.append(idx)
                values.append(weight)

        # Sort by index (required by Qdrant)
        if indices:
            pairs = sorted(zip(indices, values, strict=True))
            sorted_indices, sorted_values = zip(*pairs, strict=True)
            indices = list(sorted_indices)
            values = list(sorted_values)

        return SparseVector(indices=indices, values=values)

    def _find_fuzzy_match(
        self,
        token: str,
        vocab: dict[str, int],
        threshold: float = 0.75,
        max_candidates: int = 500,
    ) -> tuple[str | None, float]:
        """
        Find closest vocabulary token using Levenshtein similarity.

        Args:
            token: Token to match
            vocab: Vocabulary mapping
            threshold: Minimum similarity threshold (0-1)
            max_candidates: Max vocabulary terms to check

        Returns:
            Tuple of (best_match, similarity) or (None, 0.0) if no match
        """
        if not vocab:
            return None, 0.0

        target_len = len(token)
        best_match = None
        best_score = 0.0

        # Only check tokens of similar length for performance
        candidates = [t for t in vocab if abs(len(t) - target_len) <= 2][:max_candidates]

        for candidate in candidates:
            score = levenshtein_similarity(token, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate

        return best_match, best_score

    def get_codebase_ids(self) -> list[str]:
        """Get list of registered codebase IDs."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT codebase_id FROM codebase_doc_counts")
        return [row[0] for row in cursor.fetchall()]

    def get_codebase_stats(self, codebase_id: str) -> dict[str, Any] | None:
        """Get statistics for a specific codebase."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT doc_count, last_updated FROM codebase_doc_counts WHERE codebase_id = ?",
            (codebase_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        cursor = conn.execute(
            "SELECT COUNT(*) FROM codebase_contributions WHERE codebase_id = ?",
            (codebase_id,),
        )
        token_count = cursor.fetchone()[0]

        return {
            "codebase_id": codebase_id,
            "doc_count": row[0],
            "token_count": token_count,
            "last_updated": row[1],
        }

    def stats(self) -> dict[str, Any]:
        """Get global vocabulary statistics."""
        conn = self._get_conn()

        cursor = conn.execute("SELECT COUNT(*) FROM vocabulary")
        vocab_size = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM codebase_doc_counts")
        codebase_count = cursor.fetchone()[0]

        return {
            "vocab_size": vocab_size,
            "total_docs": self.total_docs,
            "codebase_count": codebase_count,
            "db_path": str(self.db_path),
        }
