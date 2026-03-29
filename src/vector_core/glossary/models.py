"""Glossary data models and exceptions."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


class GlossaryError(Exception):
    """Base exception for glossary operations."""

    pass


class GlossaryNotFoundError(GlossaryError):
    """Raised when a glossary entry is not found."""

    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"Glossary entry not found: {identifier}")


class TermExistsError(GlossaryError):
    """Raised when trying to create an entry with a term that already exists."""

    def __init__(self, term: str):
        self.term = term
        super().__init__(f"Term already exists: {term}")


@dataclass
class GlossaryEntry:
    """Full glossary entry with all fields."""

    id: UUID
    term: str
    expansion: str
    definition: str
    domain: str | None
    aliases: list[str]
    created: datetime
    modified: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "term": self.term,
            "expansion": self.expansion,
            "definition": self.definition,
            "domain": self.domain,
            "aliases": self.aliases,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
        }


@dataclass
class GlossaryEntrySummary:
    """Lightweight summary for listing."""

    id: UUID
    term: str
    expansion: str
    domain: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "term": self.term,
            "expansion": self.expansion,
            "domain": self.domain,
        }
