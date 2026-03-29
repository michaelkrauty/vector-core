"""Facts module for knowledge graph storage.

Provides shared fact storage, integrity management, and semantic search
indexing for MCP servers using vector-core.
"""

from vector_core.facts.database import FactStore
from vector_core.facts.indexer import FACTS_CODEBASE_ID, FactIndexer, generate_fact_text
from vector_core.facts.integrity import IntegrityCheckResult, SourceIntegrityManager
from vector_core.facts.models import (
    DuplicateFactError,
    Fact,
    FactError,
    FactNotFoundError,
    FactSource,
    FactSummary,
    SourceStatus,
    SourceType,
    compute_spo_hash,
)

__all__ = [
    # Database
    "FactStore",
    # Indexer
    "FactIndexer",
    "FACTS_CODEBASE_ID",
    "generate_fact_text",
    # Integrity
    "SourceIntegrityManager",
    "IntegrityCheckResult",
    # Models
    "Fact",
    "FactSource",
    "FactSummary",
    "SourceType",
    "SourceStatus",
    "compute_spo_hash",
    # Errors
    "FactError",
    "FactNotFoundError",
    "DuplicateFactError",
]
