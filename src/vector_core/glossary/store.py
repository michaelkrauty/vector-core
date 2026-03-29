"""SQLite-based glossary storage."""

import hashlib
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from vector_core.glossary.models import (
    GlossaryEntry,
    GlossaryEntrySummary,
    GlossaryNotFoundError,
    TermExistsError,
)
from vector_core.settings import settings
from vector_core.utils.sentinel import UNSET, UnsetType, is_set
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore

logger = logging.getLogger(__name__)


def _compute_entry_hash(entry: GlossaryEntry) -> str:
    """Compute hash of entry content for change detection."""
    aliases_str = ",".join(sorted(entry.aliases))
    content = (
        f"{entry.term}|{entry.expansion}|{entry.definition}|"
        f"{entry.domain or ''}|{aliases_str}"
    )
    return hashlib.sha256(content.encode()).hexdigest()


class GlossaryStore(ThreadSafeSQLiteStore):
    """
    SQLite-based glossary storage with thread-safe operations.

    Uses vector-core's shared_data_dir for storage location.
    Shared by all MCP servers using vector-core.

    Thread-safety:
    - Uses per-thread connections (via ThreadSafeSQLiteStore)
    - WAL mode for concurrent readers + single writer
    - 5-second timeout for lock acquisition
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize glossary store.

        Args:
            db_path: Path to SQLite database. Default: ~/.local/share/vector-core/glossary.db
        """
        db_path = db_path or (settings.shared_data_dir / "glossary.db")
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

        # Core entries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_entries (
                id TEXT PRIMARY KEY,
                term TEXT UNIQUE NOT NULL,
                term_normalized TEXT NOT NULL,
                expansion TEXT NOT NULL,
                definition TEXT NOT NULL,
                domain TEXT,
                created TEXT NOT NULL,
                modified TEXT NOT NULL,
                entry_hash TEXT NOT NULL
            )
        """)

        # Aliases table (many-to-one with entries)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_aliases (
                alias_normalized TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL REFERENCES glossary_entries(id) ON DELETE CASCADE,
                alias_original TEXT NOT NULL
            )
        """)

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_term_normalized "
            "ON glossary_entries(term_normalized)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_domain "
            "ON glossary_entries(domain)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_modified "
            "ON glossary_entries(modified)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aliases_entry "
            "ON glossary_aliases(entry_id)"
        )
        conn.commit()

    def create(
        self,
        term: str,
        expansion: str,
        definition: str,
        domain: str | None = None,
        aliases: list[str] | None = None,
    ) -> GlossaryEntry:
        """
        Create a new glossary entry.

        Args:
            term: Canonical term (e.g., "USAF")
            expansion: Full expansion (e.g., "United States Air Force")
            definition: Detailed definition
            domain: Optional category (e.g., "military", "tech")
            aliases: Optional alternative terms

        Returns:
            Created GlossaryEntry

        Raises:
            TermExistsError: If term or any alias already exists
        """
        conn = self._get_conn()
        now = datetime.now(UTC)
        entry_id = uuid4()
        aliases = aliases or []

        # Check if term exists
        if self.exists_term(term):
            raise TermExistsError(term)

        # Check if any alias exists
        for alias in aliases:
            if self.exists_term(alias):
                raise TermExistsError(alias)

        entry = GlossaryEntry(
            id=entry_id,
            term=term,
            expansion=expansion,
            definition=definition,
            domain=domain,
            aliases=aliases,
            created=now,
            modified=now,
        )
        entry_hash = _compute_entry_hash(entry)

        conn.execute(
            """
            INSERT INTO glossary_entries
            (id, term, term_normalized, expansion, definition,
             domain, created, modified, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry_id),
                term,
                term.lower(),
                expansion,
                definition,
                domain,
                now.isoformat(),
                now.isoformat(),
                entry_hash,
            ),
        )

        # Insert aliases
        for alias in aliases:
            conn.execute(
                """
                INSERT INTO glossary_aliases (alias_normalized, entry_id, alias_original)
                VALUES (?, ?, ?)
                """,
                (alias.lower(), str(entry_id), alias),
            )

        conn.commit()
        return entry

    def read(self, entry_id: UUID) -> GlossaryEntry:
        """
        Read a glossary entry by ID.

        Args:
            entry_id: UUID of the entry

        Returns:
            GlossaryEntry

        Raises:
            GlossaryNotFoundError: If entry not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, term, expansion, definition, domain, created, modified
            FROM glossary_entries WHERE id = ?
            """,
            (str(entry_id),),
        )
        row = cursor.fetchone()
        if row is None:
            raise GlossaryNotFoundError(str(entry_id))

        # Get aliases
        alias_cursor = conn.execute(
            "SELECT alias_original FROM glossary_aliases WHERE entry_id = ?",
            (str(entry_id),),
        )
        aliases = [r[0] for r in alias_cursor.fetchall()]

        return GlossaryEntry(
            id=UUID(row[0]),
            term=row[1],
            expansion=row[2],
            definition=row[3],
            domain=row[4],
            aliases=aliases,
            created=datetime.fromisoformat(row[5]),
            modified=datetime.fromisoformat(row[6]),
        )

    def update(
        self,
        entry_id: UUID,
        term: str | None = None,
        expansion: str | None = None,
        definition: str | None = None,
        domain: str | None | UnsetType = UNSET,
        aliases: list[str] | None | UnsetType = UNSET,
    ) -> GlossaryEntry:
        """
        Update an existing glossary entry.

        Only provided fields are updated. Use None to clear domain/aliases.
        Use UNSET (the default) to leave field unchanged.

        Args:
            entry_id: UUID of the entry
            term: New canonical term
            expansion: New expansion
            definition: New definition
            domain: New domain (None to clear, UNSET to leave unchanged)
            aliases: New aliases (replaces existing, None/[] to clear, UNSET to leave unchanged)

        Returns:
            Updated GlossaryEntry

        Raises:
            GlossaryNotFoundError: If entry not found
            TermExistsError: If new term already exists
        """
        conn = self._get_conn()

        # Get existing entry
        entry = self.read(entry_id)
        now = datetime.now(UTC)

        # Build updates
        updates: list[str] = []
        params: list[str | None] = []

        if term is not None and term != entry.term:
            # Check if new term exists
            if self.exists_term(term):
                raise TermExistsError(term)
            updates.append("term = ?")
            updates.append("term_normalized = ?")
            params.extend([term, term.lower()])
            entry.term = term

        if expansion is not None:
            updates.append("expansion = ?")
            params.append(expansion)
            entry.expansion = expansion

        if definition is not None:
            updates.append("definition = ?")
            params.append(definition)
            entry.definition = definition

        if is_set(domain):  # Explicitly provided (including None)
            updates.append("domain = ?")
            params.append(domain)
            entry.domain = domain

        # Always update modified timestamp and hash
        entry.modified = now
        entry_hash = _compute_entry_hash(entry)
        updates.append("modified = ?")
        updates.append("entry_hash = ?")
        params.extend([now.isoformat(), entry_hash])

        params.append(str(entry_id))

        if updates:
            conn.execute(
                f"UPDATE glossary_entries SET {', '.join(updates)} WHERE id = ?",
                params,
            )

        # Handle aliases
        if is_set(aliases):  # Explicitly provided
            # Delete existing aliases
            conn.execute(
                "DELETE FROM glossary_aliases WHERE entry_id = ?",
                (str(entry_id),),
            )
            # Insert new aliases
            new_aliases = aliases or []
            for alias in new_aliases:
                if self._alias_exists_for_other(alias, entry_id):
                    raise TermExistsError(alias)
                conn.execute(
                    """
                    INSERT INTO glossary_aliases (alias_normalized, entry_id, alias_original)
                    VALUES (?, ?, ?)
                    """,
                    (alias.lower(), str(entry_id), alias),
                )
            entry.aliases = new_aliases

        conn.commit()
        return entry

    def _alias_exists_for_other(self, alias: str, exclude_entry_id: UUID) -> bool:
        """Check if alias exists for a different entry."""
        conn = self._get_conn()
        # Check as term
        cursor = conn.execute(
            "SELECT id FROM glossary_entries WHERE term_normalized = ? AND id != ?",
            (alias.lower(), str(exclude_entry_id)),
        )
        if cursor.fetchone():
            return True
        # Check as alias
        cursor = conn.execute(
            "SELECT entry_id FROM glossary_aliases WHERE alias_normalized = ? AND entry_id != ?",
            (alias.lower(), str(exclude_entry_id)),
        )
        return cursor.fetchone() is not None

    def delete(self, entry_id: UUID) -> bool:
        """
        Delete a glossary entry.

        Args:
            entry_id: UUID of the entry

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM glossary_entries WHERE id = ?",
            (str(entry_id),),
        )
        conn.commit()
        return cursor.rowcount > 0

    def lookup(self, term: str) -> GlossaryEntry | None:
        """
        Look up entry by term or alias (case-insensitive).

        Args:
            term: Term or alias to look up

        Returns:
            GlossaryEntry or None if not found
        """
        conn = self._get_conn()
        normalized = term.lower()

        # Try as term first
        cursor = conn.execute(
            "SELECT id FROM glossary_entries WHERE term_normalized = ?",
            (normalized,),
        )
        row = cursor.fetchone()
        if row:
            return self.read(UUID(row[0]))

        # Try as alias
        cursor = conn.execute(
            "SELECT entry_id FROM glossary_aliases WHERE alias_normalized = ?",
            (normalized,),
        )
        row = cursor.fetchone()
        if row:
            return self.read(UUID(row[0]))

        return None

    def find_by_term_or_id(self, term_or_id: str) -> GlossaryEntry | None:
        """
        Find entry by term, alias, or UUID.

        Args:
            term_or_id: Term, alias, or UUID string

        Returns:
            GlossaryEntry or None if not found
        """
        # Try as UUID first
        try:
            entry_id = UUID(term_or_id)
            return self.read(entry_id)
        except (ValueError, GlossaryNotFoundError):
            pass

        # Try as term/alias
        return self.lookup(term_or_id)

    def exists_term(self, term: str) -> bool:
        """
        Check if term or alias exists.

        Args:
            term: Term to check

        Returns:
            True if exists as term or alias
        """
        conn = self._get_conn()
        normalized = term.lower()

        cursor = conn.execute(
            "SELECT 1 FROM glossary_entries WHERE term_normalized = ?",
            (normalized,),
        )
        if cursor.fetchone():
            return True

        cursor = conn.execute(
            "SELECT 1 FROM glossary_aliases WHERE alias_normalized = ?",
            (normalized,),
        )
        return cursor.fetchone() is not None

    def list_all(
        self,
        domain: str | None = None,
        limit: int | None = None,
    ) -> list[GlossaryEntrySummary]:
        """
        List all entries as summaries.

        Args:
            domain: Optional domain filter
            limit: Maximum entries to return

        Returns:
            List of GlossaryEntrySummary
        """
        conn = self._get_conn()
        query = "SELECT id, term, expansion, domain FROM glossary_entries"
        params: list = []

        if domain:
            query += " WHERE domain = ?"
            params.append(domain)

        query += " ORDER BY term"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        return [
            GlossaryEntrySummary(
                id=UUID(row[0]),
                term=row[1],
                expansion=row[2],
                domain=row[3],
            )
            for row in cursor.fetchall()
        ]

    def iter_all(self) -> Iterator[GlossaryEntry]:
        """
        Iterate over all entries (full objects).

        Yields:
            GlossaryEntry for each entry
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id FROM glossary_entries ORDER BY term"
        )
        for row in cursor.fetchall():
            yield self.read(UUID(row[0]))

    def count(self) -> int:
        """Get total entry count."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM glossary_entries")
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def get_domains(self) -> list[str]:
        """Get list of all unique domains."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT DISTINCT domain FROM glossary_entries WHERE domain IS NOT NULL ORDER BY domain"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_entry_hash(self, entry_id: UUID) -> str | None:
        """Get the hash of an entry for change detection."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT entry_hash FROM glossary_entries WHERE id = ?",
            (str(entry_id),),
        )
        row = cursor.fetchone()
        return row[0] if row else None
