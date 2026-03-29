"""Facts source integrity verification.

This module handles the integrity tracking of fact sources.
When notes, glossary entries, or documents are modified or deleted,
this module marks the corresponding fact sources appropriately.

Workflow:
1. Note deleted -> mark all note sources with that UUID as deleted
2. Note modified -> mark all note sources with that UUID as modified
3. Glossary deleted -> mark all glossary sources with that UUID as deleted
4. Document modified -> mark all document sources with that hash as modified
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from vector_core.facts.database import FactStore
from vector_core.facts.models import Fact, SourceStatus, SourceType

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheckResult:
    """Result of a source integrity check."""

    total_sources: int
    active_sources: int
    deleted_sources: int
    modified_sources: int
    relocated_sources: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_sources": self.total_sources,
            "active_sources": self.active_sources,
            "deleted_sources": self.deleted_sources,
            "modified_sources": self.modified_sources,
            "relocated_sources": self.relocated_sources,
            "integrity_score": self.active_sources / self.total_sources
            if self.total_sources > 0
            else 1.0,
        }


class SourceIntegrityManager:
    """
    Manages fact source integrity.

    Provides methods to:
    - Mark sources when content is modified/deleted
    - Query sources by status
    - Verify source integrity
    """

    def __init__(self, fact_store: FactStore):
        """
        Initialize integrity manager.

        Args:
            fact_store: FactStore instance
        """
        self.fact_store = fact_store

    def mark_note_deleted(self, note_id: UUID) -> int:
        """
        Mark all sources referencing a note as deleted.

        Called when a note is deleted.

        Args:
            note_id: UUID of the deleted note

        Returns:
            Number of sources marked as deleted
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.NOTE,
            source_id=note_id,
            new_status=SourceStatus.DELETED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as deleted for note {note_id}")
        return count

    def mark_note_modified(self, note_id: UUID) -> int:
        """
        Mark all sources referencing a note as modified.

        Called when a note is updated.

        Args:
            note_id: UUID of the modified note

        Returns:
            Number of sources marked as modified
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.NOTE,
            source_id=note_id,
            new_status=SourceStatus.MODIFIED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as modified for note {note_id}")
        return count

    def mark_glossary_deleted(self, entry_id: UUID) -> int:
        """
        Mark all sources referencing a glossary entry as deleted.

        Called when a glossary entry is deleted.

        Args:
            entry_id: UUID of the deleted glossary entry

        Returns:
            Number of sources marked as deleted
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.GLOSSARY,
            source_id=entry_id,
            new_status=SourceStatus.DELETED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as deleted for glossary {entry_id}")
        return count

    def mark_glossary_modified(self, entry_id: UUID) -> int:
        """
        Mark all sources referencing a glossary entry as modified.

        Called when a glossary entry is updated.

        Args:
            entry_id: UUID of the modified glossary entry

        Returns:
            Number of sources marked as modified
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.GLOSSARY,
            source_id=entry_id,
            new_status=SourceStatus.MODIFIED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as modified for glossary {entry_id}")
        return count

    def mark_document_modified(self, content_hash: str) -> int:
        """
        Mark all sources referencing a document by hash as modified.

        Called when a document file is modified.

        Args:
            content_hash: SHA-256 hash of the document

        Returns:
            Number of sources marked as modified
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.DOCUMENT,
            content_hash=content_hash,
            new_status=SourceStatus.MODIFIED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as modified for document {content_hash[:16]}...")
        return count

    def mark_document_deleted(self, content_hash: str) -> int:
        """
        Mark all sources referencing a document by hash as deleted.

        Called when a document file is deleted.

        Args:
            content_hash: SHA-256 hash of the document

        Returns:
            Number of sources marked as deleted
        """
        count = self.fact_store.update_source_status(
            source_type=SourceType.DOCUMENT,
            content_hash=content_hash,
            new_status=SourceStatus.DELETED,
        )
        if count > 0:
            logger.info(f"Marked {count} sources as deleted for document {content_hash[:16]}...")
        return count

    def get_facts_with_deleted_sources(self, limit: int = 100) -> list[Fact]:
        """
        Get facts that have deleted sources.

        Useful for identifying facts that may need review.

        Args:
            limit: Maximum facts to return

        Returns:
            List of facts with at least one deleted source
        """
        return self.fact_store.get_facts_by_source_status(
            status=SourceStatus.DELETED,
            limit=limit,
        )

    def get_facts_with_modified_sources(self, limit: int = 100) -> list[Fact]:
        """
        Get facts that have modified sources.

        These facts may need re-verification.

        Args:
            limit: Maximum facts to return

        Returns:
            List of facts with at least one modified source
        """
        return self.fact_store.get_facts_by_source_status(
            status=SourceStatus.MODIFIED,
            limit=limit,
        )

    def check_fact_integrity(self, fact_id: UUID) -> IntegrityCheckResult:
        """
        Check integrity of a single fact's sources.

        Args:
            fact_id: Fact UUID

        Returns:
            IntegrityCheckResult with source status breakdown
        """
        try:
            fact = self.fact_store.read(fact_id)
        except Exception as e:
            logger.debug(f"Could not read fact {fact_id} for integrity check: {e}")
            return IntegrityCheckResult(
                total_sources=0,
                active_sources=0,
                deleted_sources=0,
                modified_sources=0,
                relocated_sources=0,
            )

        if fact is None:
            return IntegrityCheckResult(
                total_sources=0,
                active_sources=0,
                deleted_sources=0,
                modified_sources=0,
                relocated_sources=0,
            )

        total = len(fact.sources)
        active = sum(1 for s in fact.sources if s.status == SourceStatus.ACTIVE)
        deleted = sum(1 for s in fact.sources if s.status == SourceStatus.DELETED)
        modified = sum(1 for s in fact.sources if s.status == SourceStatus.MODIFIED)
        relocated = sum(1 for s in fact.sources if s.status == SourceStatus.RELOCATED)

        return IntegrityCheckResult(
            total_sources=total,
            active_sources=active,
            deleted_sources=deleted,
            modified_sources=modified,
            relocated_sources=relocated,
        )

    def get_source_statistics(self) -> dict:
        """
        Get overall source statistics across all facts.

        Returns:
            Dict with source counts by status
        """
        return self.fact_store.get_source_statistics()

    def revalidate_sources(
        self,
        source_type: SourceType | None = None,
        source_id: UUID | None = None,
    ) -> int:
        """
        Reset modified/deleted sources back to active.

        Used after re-verifying that sources are still valid.

        Args:
            source_type: Filter by source type
            source_id: Filter by source UUID

        Returns:
            Number of sources reset to active
        """
        return self.fact_store.update_source_status(
            source_type=source_type,
            source_id=source_id,
            new_status=SourceStatus.ACTIVE,
        )
