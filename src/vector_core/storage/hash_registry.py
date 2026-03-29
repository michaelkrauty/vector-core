"""SQLite-based hash→UUID registry for document tracking."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from vector_core.settings import settings
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore


@dataclass
class RegistryEntry:
    """Entry in the hash registry."""

    content_hash: str
    uuid: UUID
    path: str
    doc_type: str
    status: str
    registered_at: datetime
    last_verified: datetime | None


class HashRegistry(ThreadSafeSQLiteStore):
    """
    SQLite-based hash→UUID registry with thread-safe operations.

    Used to track document identity across moves/renames:
    - Content hash is the stable identifier (survives moves)
    - UUID is the MCP-visible ID
    - Path is a hint that can become stale

    Thread-safety:
    - Uses per-thread connections (no sharing across threads)
    - WAL mode for concurrent readers + single writer
    - 5-second timeout for lock acquisition
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize hash registry.

        Args:
            db_path: Path to SQLite database. Default: ~/.cache/vector-core/hash_registry.db
        """
        super().__init__(
            db_path=db_path or (settings.cache_dir / "hash_registry.db"),
            config=SQLiteConfig(connect_timeout=5.0),
        )
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._ensure_parent_dir()
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hash_registry (
                content_hash TEXT PRIMARY KEY,
                uuid TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                registered_at TEXT NOT NULL,
                last_verified TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_registry_uuid ON hash_registry(uuid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_registry_status ON hash_registry(status)"
        )
        conn.commit()

    def register(
        self,
        content_hash: str,
        uuid: UUID,
        path: str,
        doc_type: str,
    ) -> None:
        """
        Register a new hash→UUID mapping.

        Args:
            content_hash: SHA256 hash of document content
            uuid: UUID for the document
            path: Current file path
            doc_type: Document type (e.g., "pdf", "docx", "note")
        """
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO hash_registry
            (content_hash, uuid, path, doc_type, status, registered_at, last_verified)
            VALUES (?, ?, ?, ?, 'active', ?, ?)
            """,
            (content_hash, str(uuid), path, doc_type, now, now),
        )
        conn.commit()

    def lookup_by_hash(self, content_hash: str) -> RegistryEntry | None:
        """
        Look up entry by content hash.

        Args:
            content_hash: SHA256 hash to look up

        Returns:
            RegistryEntry or None if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT content_hash, uuid, path, doc_type, status, registered_at, last_verified
            FROM hash_registry WHERE content_hash = ?
            """,
            (content_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def lookup_by_uuid(self, uuid: UUID) -> RegistryEntry | None:
        """
        Look up entry by UUID.

        Args:
            uuid: UUID to look up

        Returns:
            RegistryEntry or None if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT content_hash, uuid, path, doc_type, status, registered_at, last_verified
            FROM hash_registry WHERE uuid = ?
            """,
            (str(uuid),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def update_path(self, content_hash: str, new_path: str) -> bool:
        """
        Update the path for an existing entry.

        Args:
            content_hash: SHA256 hash of document
            new_path: New file path

        Returns:
            True if entry was updated, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE hash_registry SET path = ? WHERE content_hash = ?",
            (new_path, content_hash),
        )
        conn.commit()
        return cursor.rowcount > 0

    def update_status(self, content_hash: str, status: str) -> bool:
        """
        Update the status for an existing entry.

        Valid statuses: 'active', 'deleted', 'modified', 'relocated'

        Args:
            content_hash: SHA256 hash of document
            status: New status

        Returns:
            True if entry was updated, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE hash_registry SET status = ? WHERE content_hash = ?",
            (status, content_hash),
        )
        conn.commit()
        return cursor.rowcount > 0

    def mark_verified(self, content_hash: str) -> bool:
        """
        Mark an entry as recently verified.

        Args:
            content_hash: SHA256 hash of document

        Returns:
            True if entry was updated, False if not found
        """
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        cursor = conn.execute(
            "UPDATE hash_registry SET last_verified = ? WHERE content_hash = ?",
            (now, content_hash),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete(self, content_hash: str) -> bool:
        """
        Delete an entry from the registry.

        Args:
            content_hash: SHA256 hash of document

        Returns:
            True if entry was deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM hash_registry WHERE content_hash = ?",
            (content_hash,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_by_status(
        self,
        status: str,
        limit: int | None = None,
    ) -> list[RegistryEntry]:
        """
        List entries by status.

        Args:
            status: Status to filter by
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        conn = self._get_conn()
        query = """
            SELECT content_hash, uuid, path, doc_type, status, registered_at, last_verified
            FROM hash_registry WHERE status = ?
        """
        params: list = [status]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def count(self, status: str | None = None) -> int:
        """
        Count entries, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            Count of entries
        """
        conn = self._get_conn()
        if status:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM hash_registry WHERE status = ?",
                (status,),
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM hash_registry")
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def _row_to_entry(self, row: tuple) -> RegistryEntry:
        """Convert database row to RegistryEntry."""
        return RegistryEntry(
            content_hash=row[0],
            uuid=UUID(row[1]),
            path=row[2],
            doc_type=row[3],
            status=row[4],
            registered_at=datetime.fromisoformat(row[5]),
            last_verified=(
                datetime.fromisoformat(row[6]) if row[6] else None
            ),
        )
