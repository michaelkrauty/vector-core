"""Shared glossary system for MCP servers.

Provides:
- GlossaryStore: SQLite-based storage for glossary entries
- GlossaryIndexer: Qdrant indexing with GlobalVocabulary sparse vectors
- GlossaryToolHelper: Shared logic for MCP tool implementations
- GlossaryEntry: Data model for entries
"""

from vector_core.glossary.indexer import GlossaryIndexer
from vector_core.glossary.models import (
    GlossaryEntry,
    GlossaryEntrySummary,
    GlossaryError,
    GlossaryNotFoundError,
    TermExistsError,
)
from vector_core.glossary.store import GlossaryStore
from vector_core.glossary.tools import GlossaryToolHelper

__all__ = [
    "GlossaryStore",
    "GlossaryIndexer",
    "GlossaryToolHelper",
    "GlossaryEntry",
    "GlossaryEntrySummary",
    "GlossaryError",
    "GlossaryNotFoundError",
    "TermExistsError",
]
