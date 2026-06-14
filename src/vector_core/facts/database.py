"""SQLite-based fact storage with thread-safe operations."""

import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from vector_core.settings import settings
from vector_core.utils.sentinel import UNSET, UnsetType, is_set
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore

from vector_core.facts.models import (
    DuplicateFactError,
    Fact,
    FactNotFoundError,
    FactSource,
    FactSummary,
    SourceStatus,
    SourceType,
    compute_spo_hash,
)

logger = logging.getLogger(__name__)

# BFS traversal limit to prevent DoS/resource exhaustion
MAX_BFS_VISITED = 10000  # Maximum nodes visited during graph traversal


def _require_non_blank(value: str, field: str) -> None:
    """Raise ValueError if a required text field is not a non-blank string.

    The store validates but does not normalize: values are stored exactly
    as given. Blank fields would corrupt spo_hash-based deduplication and
    entity adjacency, which are built from these strings.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_confidence_in_range(confidence: float) -> None:
    """Raise ValueError if confidence is outside the documented 0.0-1.0 range."""
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence must be between 0.0 and 1.0, got {confidence!r}"
        )


def _require_valid_date_range(valid_from: date | None, valid_to: date | None) -> None:
    """Raise ValueError if both bounds are set and valid_from is after valid_to.

    An inverted validity interval is never meaningful and silently makes the
    fact unmatchable by any ``valid_at`` query, since no date can satisfy both
    ``valid_from <= d`` and ``d <= valid_to``.
    """
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ValueError(
            f"valid_from ({valid_from.isoformat()}) must not be after "
            f"valid_to ({valid_to.isoformat()})"
        )


class FactStore(ThreadSafeSQLiteStore):
    """
    SQLite-based fact storage with thread-safe operations.

    Uses vector-core's shared_data_dir for storage location.
    Shared by all MCP servers using vector-core.

    Thread-safety:
    - Uses per-thread connections (via ThreadSafeSQLiteStore)
    - WAL mode for concurrent readers + single writer
    - 5-second timeout for lock acquisition

    Tables:
    - facts: Core fact triples with metadata
    - fact_sources: Source references (one-to-many with facts)
    - entity_adjacency: Optimized entity graph for BFS traversal
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize fact store.

        Args:
            db_path: Path to SQLite database. Default: ~/.local/share/vector-core/facts.db
        """
        db_path = db_path or (settings.shared_data_dir / "facts.db")
        super().__init__(
            db_path,
            config=SQLiteConfig(
                foreign_keys=True,  # Enable FK constraints for cascading deletes
                busy_timeout_ms=5000,
                connect_timeout=5.0,
            ),
        )
        self._ensure_parent_dir()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()

        # Core facts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                object_type TEXT NOT NULL,
                context TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                valid_from TEXT,
                valid_to TEXT,
                spo_hash TEXT UNIQUE NOT NULL,
                created TEXT NOT NULL,
                modified TEXT NOT NULL
            )
        """)

        # Fact sources (one-to-many with facts)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                source_id TEXT,
                source_path TEXT,
                content_hash TEXT,
                location TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                extracted_at TEXT,
                verified_at TEXT
            )
        """)

        # Entity adjacency for BFS graph traversal
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_adjacency (
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                PRIMARY KEY (entity_name, entity_type, fact_id, role)
            )
        """)

        # Indexes for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject "
            "ON facts(subject)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_object "
            "ON facts(object)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_predicate "
            "ON facts(predicate)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject_type "
            "ON facts(subject_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_object_type "
            "ON facts(object_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_confidence "
            "ON facts(confidence)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_valid_from "
            "ON facts(valid_from)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_spo_hash "
            "ON facts(spo_hash)"
        )

        # Indexes for fact_sources
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_fact "
            "ON fact_sources(fact_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_hash "
            "ON fact_sources(content_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_status "
            "ON fact_sources(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_type_id "
            "ON fact_sources(source_type, source_id)"
        )

        # Index for entity adjacency
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_adjacency_entity "
            "ON entity_adjacency(entity_name, entity_type)"
        )

        conn.commit()

    def create(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        subject_type: str = "entity",
        object_type: str = "entity",
        context: str | None = None,
        confidence: float = 1.0,
        valid_from: date | None = None,
        valid_to: date | None = None,
        source: FactSource | None = None,
    ) -> Fact:
        """
        Create a new fact.

        Args:
            subject: Subject entity name
            predicate: Relationship predicate
            object_value: Object entity name
            subject_type: Type of subject (default: "entity")
            object_type: Type of object (default: "entity")
            context: Optional context description
            confidence: Confidence level 0.0-1.0 (default: 1.0)
            valid_from: Start date of validity
            valid_to: End date of validity
            source: Optional source reference

        Returns:
            Created Fact

        Raises:
            DuplicateFactError: If fact with same SPO already exists
            ValueError: If any SPO/type field is blank, confidence is outside
                0.0-1.0, or valid_from is after valid_to. Raised before any
                database access.
        """
        for field, value in (
            ("subject", subject),
            ("predicate", predicate),
            ("object_value", object_value),
            ("subject_type", subject_type),
            ("object_type", object_type),
        ):
            _require_non_blank(value, field)
        _require_confidence_in_range(confidence)
        _require_valid_date_range(valid_from, valid_to)

        conn = self._get_conn()
        now = datetime.now(UTC)
        fact_id = uuid4()

        # Compute SPO hash for duplicate detection
        spo_hash = compute_spo_hash(
            subject, subject_type, predicate, object_value, object_type
        )

        # Check for duplicates
        cursor = conn.execute(
            "SELECT id FROM facts WHERE spo_hash = ?",
            (spo_hash,),
        )
        existing = cursor.fetchone()
        if existing:
            raise DuplicateFactError(spo_hash, existing[0])

        # Create fact object
        sources = [source] if source else []
        fact = Fact(
            id=fact_id,
            subject=subject,
            subject_type=subject_type,
            predicate=predicate,
            object_value=object_value,
            object_type=object_type,
            context=context,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
            sources=sources,
            spo_hash=spo_hash,
            created=now,
            modified=now,
        )

        # Insert fact with transaction safety
        try:
            conn.execute(
                """
                INSERT INTO facts
                (id, subject, subject_type, predicate, object, object_type,
                 context, confidence, valid_from, valid_to, spo_hash, created, modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(fact_id),
                    subject,
                    subject_type,
                    predicate,
                    object_value,
                    object_type,
                    context,
                    confidence,
                    valid_from.isoformat() if valid_from else None,
                    valid_to.isoformat() if valid_to else None,
                    spo_hash,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

            # Insert source if provided
            if source:
                self._insert_source(conn, fact_id, source)

            # Update entity adjacency
            self._update_adjacency(conn, fact_id, subject, subject_type, "subject")
            self._update_adjacency(conn, fact_id, object_value, object_type, "object")

            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback()
            logger.error(f"Integrity error creating fact: {e}")
            raise DuplicateFactError(spo_hash, "unknown") from e
        except sqlite3.OperationalError as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "locked" in error_msg or "busy" in error_msg:
                logger.warning(f"Database locked during fact creation: {e}")
                from vector_core.errors import DatabaseLockError
                raise DatabaseLockError(str(self.db_path), 5.0) from e
            logger.error(f"Operational error creating fact: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Failed to create fact: {e}") from e
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"SQLite error creating fact: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Database error: {e}") from e
        return fact

    def _insert_source(
        self,
        conn: sqlite3.Connection,
        fact_id: UUID,
        source: FactSource,
    ) -> None:
        """Insert a source record."""
        conn.execute(
            """
            INSERT INTO fact_sources
            (fact_id, source_type, source_id, source_path, content_hash,
             location, status, extracted_at, verified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(fact_id),
                source.source_type.value,
                str(source.source_id) if source.source_id else None,
                source.source_path,
                source.content_hash,
                source.location,
                source.status.value,
                source.extracted_at.isoformat() if source.extracted_at else None,
                source.verified_at.isoformat() if source.verified_at else None,
            ),
        )

    def _update_adjacency(
        self,
        conn: sqlite3.Connection,
        fact_id: UUID,
        entity_name: str,
        entity_type: str,
        role: str,
    ) -> None:
        """Update entity adjacency table."""
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_adjacency
            (entity_name, entity_type, fact_id, role)
            VALUES (?, ?, ?, ?)
            """,
            (entity_name.lower(), entity_type.lower(), str(fact_id), role),
        )

    def read(self, fact_id: UUID) -> Fact:
        """
        Read a fact by ID.

        Args:
            fact_id: UUID of the fact

        Returns:
            Fact

        Raises:
            FactNotFoundError: If fact not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, subject, subject_type, predicate, object, object_type,
                   context, confidence, valid_from, valid_to, spo_hash, created, modified
            FROM facts WHERE id = ?
            """,
            (str(fact_id),),
        )
        row = cursor.fetchone()
        if row is None:
            raise FactNotFoundError(str(fact_id))

        # Get sources
        sources = self._get_sources(conn, fact_id)

        return Fact(
            id=UUID(row[0]),
            subject=row[1],
            subject_type=row[2],
            predicate=row[3],
            object_value=row[4],
            object_type=row[5],
            context=row[6],
            confidence=row[7],
            valid_from=date.fromisoformat(row[8]) if row[8] else None,
            valid_to=date.fromisoformat(row[9]) if row[9] else None,
            spo_hash=row[10],
            created=datetime.fromisoformat(row[11]),
            modified=datetime.fromisoformat(row[12]),
            sources=sources,
        )

    def _get_sources(self, conn: sqlite3.Connection, fact_id: UUID) -> list[FactSource]:
        """Get all sources for a fact."""
        cursor = conn.execute(
            """
            SELECT source_type, source_id, source_path, content_hash,
                   location, status, extracted_at, verified_at
            FROM fact_sources WHERE fact_id = ?
            """,
            (str(fact_id),),
        )
        return [
            FactSource(
                source_type=SourceType(row[0]),
                source_id=UUID(row[1]) if row[1] else None,
                source_path=row[2],
                content_hash=row[3],
                location=row[4],
                status=SourceStatus(row[5]),
                extracted_at=datetime.fromisoformat(row[6]) if row[6] else None,
                verified_at=datetime.fromisoformat(row[7]) if row[7] else None,
            )
            for row in cursor.fetchall()
        ]

    def _read_batch(self, conn: sqlite3.Connection, fact_ids: list[UUID]) -> list[Fact]:
        """
        Read multiple facts with their sources in minimal queries.

        Uses a single query to fetch all fact data and a single query
        for all sources, avoiding N+1 query pattern.

        Args:
            conn: Database connection
            fact_ids: List of fact UUIDs to read

        Returns:
            List of Facts in the same order as fact_ids (IDs not found
            in the database are skipped). Callers select IDs with their
            own ORDER BY and rely on that order surviving the batch read.
        """
        if not fact_ids:
            return []

        # Convert to strings for SQL
        id_strs = [str(fid) for fid in fact_ids]
        placeholders = ",".join("?" * len(id_strs))

        # Batch fetch all facts
        cursor = conn.execute(
            f"""
            SELECT id, subject, subject_type, predicate, object, object_type,
                   context, confidence, valid_from, valid_to, spo_hash, created, modified
            FROM facts WHERE id IN ({placeholders})
            """,
            id_strs,
        )

        # Build fact objects without sources first
        facts_by_id: dict[str, Fact] = {}
        for row in cursor.fetchall():
            fact = Fact(
                id=UUID(row[0]),
                subject=row[1],
                subject_type=row[2],
                predicate=row[3],
                object_value=row[4],
                object_type=row[5],
                context=row[6],
                confidence=row[7],
                valid_from=date.fromisoformat(row[8]) if row[8] else None,
                valid_to=date.fromisoformat(row[9]) if row[9] else None,
                spo_hash=row[10],
                created=datetime.fromisoformat(row[11]),
                modified=datetime.fromisoformat(row[12]),
                sources=[],
            )
            facts_by_id[row[0]] = fact

        # Batch fetch all sources
        source_cursor = conn.execute(
            f"""
            SELECT fact_id, source_type, source_id, source_path, content_hash,
                   location, status, extracted_at, verified_at
            FROM fact_sources WHERE fact_id IN ({placeholders})
            """,
            id_strs,
        )

        # Attach sources to their facts
        for row in source_cursor.fetchall():
            fact_id_str = row[0]
            if fact_id_str in facts_by_id:
                source = FactSource(
                    source_type=SourceType(row[1]),
                    source_id=UUID(row[2]) if row[2] else None,
                    source_path=row[3],
                    content_hash=row[4],
                    location=row[5],
                    status=SourceStatus(row[6]),
                    extracted_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    verified_at=datetime.fromisoformat(row[8]) if row[8] else None,
                )
                facts_by_id[fact_id_str].sources.append(source)

        # SQL `IN (...)` returns rows in arbitrary order; restore the
        # caller's order (typically from an ORDER BY in the ID query).
        return [facts_by_id[id_str] for id_str in id_strs if id_str in facts_by_id]

    def update(
        self,
        fact_id: UUID,
        context: str | None | UnsetType = UNSET,
        confidence: float | None = None,
        valid_from: date | None | UnsetType = UNSET,
        valid_to: date | None | UnsetType = UNSET,
    ) -> Fact:
        """
        Update an existing fact.

        Only metadata fields can be updated. SPO triple is immutable.
        Use UNSET (the default) to leave field unchanged. Use None to clear.

        Args:
            fact_id: UUID of the fact
            context: New context (None to clear, UNSET to leave unchanged)
            confidence: New confidence level
            valid_from: New start date (None to clear, UNSET to leave unchanged)
            valid_to: New end date (None to clear, UNSET to leave unchanged)

        Returns:
            Updated Fact

        Raises:
            FactNotFoundError: If fact not found
            ValueError: If confidence is outside 0.0-1.0, or the resulting
                valid_from/valid_to range is inverted (new value where
                provided, otherwise the existing one). Raised before any row
                is written.
        """
        if confidence is not None:
            _require_confidence_in_range(confidence)

        conn = self._get_conn()

        # Load fact for update (raises FactNotFoundError if not found)
        fact = self.read(fact_id)

        # Validate the effective validity range (new value where provided,
        # otherwise the fact's existing value) before writing anything.
        effective_from = valid_from if is_set(valid_from) else fact.valid_from
        effective_to = valid_to if is_set(valid_to) else fact.valid_to
        _require_valid_date_range(effective_from, effective_to)

        now = datetime.now(UTC)

        # Build updates
        updates = []
        params: list[Any] = []

        if is_set(context):
            updates.append("context = ?")
            params.append(context)
            fact.context = context

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
            fact.confidence = confidence

        if is_set(valid_from):
            updates.append("valid_from = ?")
            params.append(valid_from.isoformat() if valid_from else None)
            fact.valid_from = valid_from

        if is_set(valid_to):
            updates.append("valid_to = ?")
            params.append(valid_to.isoformat() if valid_to else None)
            fact.valid_to = valid_to

        # Always update modified timestamp
        updates.append("modified = ?")
        params.append(now.isoformat())
        fact.modified = now

        params.append(str(fact_id))

        if updates:
            conn.execute(
                f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        return fact

    def delete(self, fact_id: UUID) -> bool:
        """
        Delete a fact and its sources.

        Args:
            fact_id: UUID of the fact

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()

        try:
            # Delete from adjacency first (CASCADE handles sources)
            conn.execute(
                "DELETE FROM entity_adjacency WHERE fact_id = ?",
                (str(fact_id),),
            )

            cursor = conn.execute(
                "DELETE FROM facts WHERE id = ?",
                (str(fact_id),),
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "locked" in error_msg or "busy" in error_msg:
                logger.warning(f"Database locked during fact deletion: {e}")
                from vector_core.errors import DatabaseLockError
                raise DatabaseLockError(str(self.db_path), 5.0) from e
            logger.error(f"Operational error deleting fact: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Failed to delete fact: {e}") from e
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"SQLite error deleting fact: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Database error: {e}") from e
        return cursor.rowcount > 0

    def add_source(self, fact_id: UUID, source: FactSource) -> Fact:
        """
        Add a source to an existing fact.

        Args:
            fact_id: UUID of the fact
            source: Source to add

        Returns:
            Updated Fact with new source

        Raises:
            FactNotFoundError: If fact not found
        """
        conn = self._get_conn()

        # Verify fact exists (raises FactNotFoundError if not)
        _ = self.read(fact_id)

        try:
            # Insert source
            self._insert_source(conn, fact_id, source)

            # Update modified timestamp
            now = datetime.now(UTC)
            conn.execute(
                "UPDATE facts SET modified = ? WHERE id = ?",
                (now.isoformat(), str(fact_id)),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback()
            logger.error(f"Integrity error adding source to fact: {e}")
            from vector_core.errors import DatabaseIntegrityError
            raise DatabaseIntegrityError(f"Failed to add source: {e}") from e
        except sqlite3.OperationalError as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "locked" in error_msg or "busy" in error_msg:
                logger.warning(f"Database locked during add_source: {e}")
                from vector_core.errors import DatabaseLockError
                raise DatabaseLockError(str(self.db_path), 5.0) from e
            logger.error(f"Operational error adding source: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Failed to add source: {e}") from e
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"SQLite error adding source: {e}")
            from vector_core.errors import DatabaseError
            raise DatabaseError(f"Database error: {e}") from e

        # Refresh and return
        return self.read(fact_id)

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        subject_type: str | None = None,
        object_type: str | None = None,
        min_confidence: float | None = None,
        valid_at: date | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """
        Query facts by various criteria.

        Args:
            subject: Filter by subject (case-insensitive)
            predicate: Filter by predicate (case-insensitive)
            object_value: Filter by object (case-insensitive)
            subject_type: Filter by subject type
            object_type: Filter by object type
            min_confidence: Minimum confidence threshold
            valid_at: Filter by validity date (within valid_from/valid_to range)
            limit: Maximum results to return

        Returns:
            List of matching Facts
        """
        conn = self._get_conn()

        query = """
            SELECT id FROM facts WHERE 1=1
        """
        params: list[Any] = []

        if subject:
            query += " AND LOWER(subject) = LOWER(?)"
            params.append(subject)

        if predicate:
            query += " AND LOWER(predicate) = LOWER(?)"
            params.append(predicate)

        if object_value:
            query += " AND LOWER(object) = LOWER(?)"
            params.append(object_value)

        if subject_type:
            query += " AND LOWER(subject_type) = LOWER(?)"
            params.append(subject_type)

        if object_type:
            query += " AND LOWER(object_type) = LOWER(?)"
            params.append(object_type)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if valid_at:
            # valid_at must be within [valid_from, valid_to]
            # NULL means unbounded on that side
            query += """
                AND (valid_from IS NULL OR valid_from <= ?)
                AND (valid_to IS NULL OR valid_to >= ?)
            """
            date_str = valid_at.isoformat()
            params.extend([date_str, date_str])

        query += " ORDER BY modified DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        fact_ids = [UUID(row[0]) for row in cursor.fetchall()]
        return self._read_batch(conn, fact_ids)

    def get_entity_facts(
        self,
        entity_name: str,
        entity_type: str | None = None,
    ) -> list[Fact]:
        """
        Get all facts involving an entity (as subject or object).

        Args:
            entity_name: Entity name (case-insensitive)
            entity_type: Optional entity type filter

        Returns:
            List of Facts involving the entity
        """
        conn = self._get_conn()

        if entity_type:
            cursor = conn.execute(
                """
                SELECT DISTINCT fact_id FROM entity_adjacency
                WHERE entity_name = ? AND entity_type = ?
                """,
                (entity_name.lower(), entity_type.lower()),
            )
        else:
            cursor = conn.execute(
                """
                SELECT DISTINCT fact_id FROM entity_adjacency
                WHERE entity_name = ?
                """,
                (entity_name.lower(),),
            )

        fact_ids = [UUID(row[0]) for row in cursor.fetchall()]
        return self._read_batch(conn, fact_ids)

    def list_summaries(
        self,
        subject_type: str | None = None,
        object_type: str | None = None,
        predicate: str | None = None,
        limit: int = 50,
    ) -> list[FactSummary]:
        """
        List facts as lightweight summaries.

        Args:
            subject_type: Filter by subject type
            object_type: Filter by object type
            predicate: Filter by predicate
            limit: Maximum results

        Returns:
            List of FactSummary
        """
        conn = self._get_conn()

        query = """
            SELECT f.id, f.subject, f.subject_type, f.predicate, f.object,
                   f.object_type, f.confidence,
                   (SELECT COUNT(*) FROM fact_sources WHERE fact_id = f.id) as source_count
            FROM facts f WHERE 1=1
        """
        params: list[Any] = []

        if subject_type:
            query += " AND LOWER(f.subject_type) = LOWER(?)"
            params.append(subject_type)

        if object_type:
            query += " AND LOWER(f.object_type) = LOWER(?)"
            params.append(object_type)

        if predicate:
            query += " AND LOWER(f.predicate) = LOWER(?)"
            params.append(predicate)

        query += " ORDER BY f.modified DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [
            FactSummary(
                id=UUID(row[0]),
                subject=row[1],
                subject_type=row[2],
                predicate=row[3],
                object_value=row[4],
                object_type=row[5],
                confidence=row[6],
                source_count=row[7],
            )
            for row in cursor.fetchall()
        ]

    def find_by_spo_hash(self, spo_hash: str) -> Fact | None:
        """
        Find fact by SPO hash.

        Args:
            spo_hash: The SPO hash to look up

        Returns:
            Fact if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id FROM facts WHERE spo_hash = ?",
            (spo_hash,),
        )
        row = cursor.fetchone()
        if row:
            return self.read(UUID(row[0]))
        return None

    def get_facts_by_source(
        self,
        source_type: SourceType | None = None,
        source_id: UUID | None = None,
        content_hash: str | None = None,
    ) -> list[Fact]:
        """
        Get facts by source criteria.

        Args:
            source_type: Filter by source type
            source_id: Filter by source UUID
            content_hash: Filter by content hash

        Returns:
            List of matching Facts
        """
        conn = self._get_conn()

        query = """
            SELECT DISTINCT f.id
            FROM facts f
            JOIN fact_sources s ON f.id = s.fact_id
            WHERE 1=1
        """
        params: list[Any] = []

        if source_type:
            query += " AND s.source_type = ?"
            params.append(source_type.value)

        if source_id:
            query += " AND s.source_id = ?"
            params.append(str(source_id))

        if content_hash:
            query += " AND s.content_hash = ?"
            params.append(content_hash)

        cursor = conn.execute(query, params)
        fact_ids = [UUID(row[0]) for row in cursor.fetchall()]
        return self._read_batch(conn, fact_ids)

    def update_source_status(
        self,
        source_type: SourceType | None = None,
        source_id: UUID | None = None,
        content_hash: str | None = None,
        new_status: SourceStatus = SourceStatus.DELETED,
    ) -> int:
        """
        Update status of sources matching criteria.

        Used for integrity tracking when source documents are modified/deleted.

        Args:
            source_type: Source type to match (None = all types)
            source_id: Source UUID to match (for notes/glossary)
            content_hash: Content hash to match (for documents)
            new_status: New status to set

        Returns:
            Number of sources updated
        """
        conn = self._get_conn()
        now = datetime.now(UTC)

        query = "UPDATE fact_sources SET status = ?, verified_at = ?"
        params: list[Any] = [new_status.value, now.isoformat()]

        conditions: list[str] = []
        if source_type is not None:
            conditions.append("source_type = ?")
            params.append(source_type.value)

        if source_id:
            conditions.append("source_id = ?")
            params.append(str(source_id))

        if content_hash:
            conditions.append("content_hash = ?")
            params.append(content_hash)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Get total fact count."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM facts")
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def find_connections(
        self,
        source_entity: str,
        target_entity: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        max_depth: int = 3,
        limit: int = 10,
    ) -> list[list[Fact]]:
        """
        Find connections between entities using BFS graph traversal.

        Uses the entity_adjacency table for efficient graph traversal.

        Args:
            source_entity: Starting entity name (case-insensitive)
            target_entity: Target entity to find path to (optional)
                          If None, returns all reachable entities up to max_depth
            source_type: Type of source entity (optional, for disambiguation)
            target_type: Type of target entity (optional, for disambiguation)
            max_depth: Maximum path length (default 3, max 10)
            limit: Maximum paths to return (default 10)

        Returns:
            List of paths, where each path is a list of Facts connecting entities.
            For target_entity=None, returns single-fact paths to all reachable entities.
        """
        from collections import deque

        max_depth = min(max(1, max_depth), 10)
        conn = self._get_conn()

        # Normalize entity names and type filters the same way adjacency
        # writes do (_update_adjacency stores both lowercased), so callers can
        # pass types in any case.
        source_entity = source_entity.lower()
        target_lower = target_entity.lower() if target_entity else None
        source_type = source_type.lower() if source_type else None
        target_type = target_type.lower() if target_type else None

        # BFS state
        # Each queue item: (current_entity, current_type, path_so_far)
        queue: deque[tuple[str, str | None, list[UUID]]] = deque()
        visited: set[tuple[str, str | None]] = set()
        paths: list[list[Fact]] = []

        # Find starting points
        start_query = """
            SELECT DISTINCT entity_name, entity_type, fact_id
            FROM entity_adjacency
            WHERE LOWER(entity_name) = ?
        """
        start_params: list[Any] = [source_entity]

        if source_type:
            start_query += " AND entity_type = ?"
            start_params.append(source_type)

        cursor = conn.execute(start_query, start_params)
        for row in cursor.fetchall():
            ent_name, ent_type, fact_id = row
            key = (ent_name.lower(), ent_type)
            if key not in visited:
                visited.add(key)
                queue.append((ent_name.lower(), ent_type, []))

        # BFS traversal with visited limit to prevent resource exhaustion
        while queue and len(paths) < limit:
            # Safety check: prevent unbounded traversal
            if len(visited) > MAX_BFS_VISITED:
                logger.warning(
                    f"BFS traversal terminated: visited {len(visited)} nodes "
                    f"(exceeds MAX_BFS_VISITED={MAX_BFS_VISITED})"
                )
                break

            current_entity, current_type, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            # Find all facts involving this entity
            adj_query = """
                SELECT fact_id, role
                FROM entity_adjacency
                WHERE LOWER(entity_name) = ?
            """
            adj_params: list[Any] = [current_entity]

            if current_type:
                adj_query += " AND entity_type = ?"
                adj_params.append(current_type)

            adj_cursor = conn.execute(adj_query, adj_params)

            for row in adj_cursor.fetchall():
                fact_id, role = row
                fact_uuid = UUID(fact_id)

                # Skip if this fact is already in path
                if fact_uuid in path:
                    continue

                # Get the other entity from this fact
                other_query = """
                    SELECT entity_name, entity_type, role
                    FROM entity_adjacency
                    WHERE fact_id = ? AND NOT (LOWER(entity_name) = ? AND role = ?)
                """
                other_cursor = conn.execute(
                    other_query,
                    (fact_id, current_entity, role),
                )

                for other_row in other_cursor.fetchall():
                    other_name, other_type, other_role = other_row
                    other_key = (other_name.lower(), other_type)
                    new_path = path + [fact_uuid]

                    # Check if we found target
                    if target_lower:
                        if other_name.lower() == target_lower:
                            if target_type is None or other_type == target_type:
                                # Found a path to target
                                facts = [self.read(fid) for fid in new_path]
                                paths.append(facts)
                                if len(paths) >= limit:
                                    break
                    # No target - collect all reachable entities
                    elif other_key not in visited:
                        facts = [self.read(fid) for fid in new_path]
                        paths.append(facts)
                        if len(paths) >= limit:
                            break

                    # Continue BFS if not at max depth
                    if len(new_path) < max_depth and other_key not in visited:
                        visited.add(other_key)
                        queue.append((other_name.lower(), other_type, new_path))

                if len(paths) >= limit:
                    break

        return paths

    def get_neighbors(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """
        Get immediate neighbors of an entity.

        Args:
            entity: Entity name (case-insensitive)
            entity_type: Type filter (optional)

        Returns:
            List of dicts with neighbor info:
            - entity: Neighbor entity name (original case)
            - type: Neighbor entity type (original case)
            - predicate: Relationship predicate
            - direction: 'outgoing' or 'incoming'
            - fact_id: UUID of connecting fact
        """
        conn = self._get_conn()
        entity_lower = entity.lower()

        # Find all facts involving this entity, get original-case names from facts table
        query = """
            SELECT a1.fact_id, a1.role, a2.role,
                   f.subject, f.subject_type, f.object, f.object_type, f.predicate
            FROM entity_adjacency a1
            JOIN entity_adjacency a2 ON a1.fact_id = a2.fact_id
            JOIN facts f ON a1.fact_id = f.id
            WHERE a1.entity_name = ?
            AND NOT (a2.entity_name = ? AND a2.role = a1.role)
        """
        params: list[Any] = [entity_lower, entity_lower]

        if entity_type:
            query += " AND a1.entity_type = ?"
            params.append(entity_type.lower())

        cursor = conn.execute(query, params)
        neighbors = []

        for row in cursor.fetchall():
            (fact_id, my_role, neighbor_role,
             subject, subject_type, object_val, object_type, predicate) = row

            # Get original-case neighbor name from facts table
            if neighbor_role == "subject":
                neighbor_name = subject
                neighbor_type_val = subject_type
            else:
                neighbor_name = object_val
                neighbor_type_val = object_type

            # Determine direction
            # If I'm subject and neighbor is object -> outgoing
            # If I'm object and neighbor is subject -> incoming
            direction = "outgoing" if my_role == "subject" else "incoming"

            neighbors.append({
                "entity": neighbor_name,
                "type": neighbor_type_val,
                "predicate": predicate,
                "direction": direction,
                "fact_id": fact_id,
            })

        return neighbors

    def iter_all(self) -> Iterator[Fact]:
        """
        Iterate over all facts.

        The id list is snapshotted up front and each fact is read lazily. A
        single fact that cannot be loaded — deleted by another writer between
        the snapshot and its read, or a malformed stored row — is skipped
        rather than aborting iteration over the rest of the corpus. A failure
        of the initial id query itself is *not* swallowed: it propagates, so a
        systemic read failure stays loud.

        Yields:
            Fact for each readable fact
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT id FROM facts ORDER BY modified DESC")
        for row in cursor.fetchall():
            try:
                yield self.read(UUID(row[0]))
            except FactNotFoundError:
                # Deleted between the id snapshot and this read; skip it.
                logger.debug("Fact %s vanished during iter_all, skipping", row[0])
                continue
            except sqlite3.Error:
                # Systemic DB failure (e.g. database is locked). Must NOT be
                # swallowed as a single bad row: that would yield an empty or
                # partial corpus and let a force reindex delete every point.
                # Fail loud.
                raise
            except (ValueError, KeyError, TypeError):
                # Malformed stored row (bad date/enum/uuid) — skip just this
                # fact rather than aborting iteration over the rest.
                logger.warning(
                    "Skipping unreadable fact %s during iter_all", row[0], exc_info=True
                )
                continue

    def get_facts_by_source_status(
        self,
        status: SourceStatus,
        limit: int = 100,
    ) -> list[Fact]:
        """
        Get facts that have sources with a specific status.

        Useful for finding facts with deleted or modified sources.

        Args:
            status: Source status to filter by
            limit: Maximum facts to return

        Returns:
            List of facts with at least one source of the given status
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT DISTINCT f.id
            FROM facts f
            JOIN fact_sources s ON f.id = s.fact_id
            WHERE s.status = ?
            ORDER BY f.modified DESC
            LIMIT ?
            """,
            (status.value, limit),
        )
        fact_ids = [UUID(row[0]) for row in cursor.fetchall()]
        return self._read_batch(conn, fact_ids)

    def get_source_statistics(self) -> dict:
        """
        Get statistics about source statuses across all facts.

        Returns:
            Dict with counts by status and source type
        """
        conn = self._get_conn()

        # Count by status
        status_cursor = conn.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM fact_sources
            GROUP BY status
            """
        )
        by_status = {row[0]: row[1] for row in status_cursor.fetchall()}

        # Count by source type
        type_cursor = conn.execute(
            """
            SELECT source_type, COUNT(*) as cnt
            FROM fact_sources
            GROUP BY source_type
            """
        )
        by_type = {row[0]: row[1] for row in type_cursor.fetchall()}

        # Total
        total_cursor = conn.execute("SELECT COUNT(*) FROM fact_sources")
        total = total_cursor.fetchone()[0]

        return {
            "total_sources": total,
            "by_status": by_status,
            "by_type": by_type,
        }
