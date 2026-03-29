"""Facts data models and exceptions.

This module provides shared fact models for knowledge graph storage.
Used by mcp-notes for personal knowledge and mcp-docs for document facts.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from uuid import UUID


class FactError(Exception):
    """Base exception for facts operations."""

    pass


class FactNotFoundError(FactError):
    """Raised when a fact is not found."""

    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"Fact not found: {identifier}")


class DuplicateFactError(FactError):
    """Raised when attempting to create a fact that already exists."""

    def __init__(self, spo_hash: str, existing_id: str):
        self.spo_hash = spo_hash
        self.existing_id = existing_id
        super().__init__(
            f"Fact with same subject/predicate/object already exists (id={existing_id})"
        )


class SourceType(str, Enum):
    """Source types for facts."""

    NOTE = "note"
    DOCUMENT = "document"
    GLOSSARY = "glossary"
    MANUAL = "manual"


class SourceStatus(str, Enum):
    """Status of a fact source."""

    ACTIVE = "active"
    DELETED = "deleted"
    MODIFIED = "modified"
    RELOCATED = "relocated"


@dataclass
class FactSource:
    """Source reference for a fact.

    Different source types use different identification fields:
    - note/glossary: source_id (UUID) as primary, path as secondary
    - document: content_hash as primary (survives file moves), path as hint
    - manual: no identifiers (user-entered)

    Attributes:
        source_type: Type of source (note, document, glossary, manual)
        source_id: UUID for notes/glossary entries
        source_path: Path hint for documents (can become stale)
        content_hash: SHA-256 hash for documents (primary identifier)
        location: Location within source (e.g., "page 3", "section: History")
        status: Integrity status (active, deleted, modified, relocated)
        extracted_at: When the fact was extracted from this source
        verified_at: When the source was last verified to still exist
    """

    source_type: SourceType
    source_id: UUID | None = None
    source_path: str | None = None
    content_hash: str | None = None
    location: str | None = None
    status: SourceStatus = SourceStatus.ACTIVE
    extracted_at: datetime | None = None
    verified_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "source_id": str(self.source_id) if self.source_id else None,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "location": self.location,
            "status": self.status.value,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
        }


def compute_spo_hash(
    subject: str,
    subject_type: str,
    predicate: str,
    object_value: str,
    object_type: str,
) -> str:
    """
    Compute hash for duplicate detection.

    Includes types because "Python (language)" and "Python (snake)" are different facts.
    All values are lowercased for case-insensitive matching.

    Args:
        subject: Subject of the fact
        subject_type: Type of subject entity
        predicate: Relationship predicate
        object_value: Object of the fact
        object_type: Type of object entity

    Returns:
        SHA-256 hash string
    """
    normalized = (
        f"{subject.lower()}|{subject_type.lower()}|"
        f"{predicate.lower()}|{object_value.lower()}|{object_type.lower()}"
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class Fact:
    """A knowledge fact representing a subject-predicate-object triple.

    Facts capture structured knowledge like:
    - ("John Smith", person, "served_in", "101st Airborne", military_unit)
    - ("Python", language, "created_by", "Guido van Rossum", person)

    Attributes:
        id: Unique identifier (UUID)
        subject: Subject entity name
        subject_type: Type/category of subject (e.g., "person", "organization")
        predicate: Relationship type (e.g., "served_in", "works_at")
        object_value: Object entity name (named 'object' in storage)
        object_type: Type/category of object
        context: Optional contextual information (e.g., "as squad leader")
        confidence: Confidence level 0.0-1.0 (1.0 = verified/manual)
        valid_from: Start date of fact validity (e.g., employment start)
        valid_to: End date of fact validity (e.g., employment end)
        sources: List of sources that support this fact
        spo_hash: Hash of subject/predicate/object for duplicate detection
        created: Creation timestamp
        modified: Last modification timestamp
    """

    id: UUID
    subject: str
    subject_type: str
    predicate: str
    object_value: str  # Named 'object' in storage
    object_type: str
    spo_hash: str
    created: datetime
    modified: datetime
    context: str | None = None
    confidence: float = 1.0
    valid_from: date | None = None
    valid_to: date | None = None
    sources: list[FactSource] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "subject": self.subject,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object": self.object_value,
            "object_type": self.object_type,
            "context": self.context,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "sources": [s.to_dict() for s in self.sources],
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
        }


@dataclass
class FactSummary:
    """Lightweight fact summary for listing."""

    id: UUID
    subject: str
    subject_type: str
    predicate: str
    object_value: str
    object_type: str
    confidence: float
    source_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "subject": self.subject,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object": self.object_value,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "source_count": self.source_count,
        }
