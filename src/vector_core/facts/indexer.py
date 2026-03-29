"""Fact indexer for Qdrant-based semantic search.

This module provides semantic search indexing for facts (SPO triples).
It's designed to be shared across MCP servers, with the collection name
provided at construction time to allow integration with existing collections.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from qdrant_client.models import (
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from vector_core.embeddings.client import EmbeddingClient, EmbeddingServiceError
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.facts.models import Fact, FactNotFoundError, SourceStatus
from vector_core.storage.qdrant import (
    QdrantConnectionError,
    QdrantStorage,
    generate_collection_name,
    generate_point_id,
)

logger = logging.getLogger(__name__)

# Codebase ID for GlobalVocabulary registration
FACTS_CODEBASE_ID = "facts"


def generate_fact_text(fact: Fact) -> str:
    """
    Generate searchable text from a fact.

    Creates natural language representation of the SPO triple plus context.

    Args:
        fact: Fact to generate text for

    Returns:
        Searchable text string
    """
    # Build natural language representation
    parts = [
        fact.subject,
        fact.predicate.replace("_", " "),
        fact.object_value,
    ]

    # Add types if specified
    if fact.subject_type:
        parts.append(f"({fact.subject_type})")
    if fact.object_type:
        parts.append(f"({fact.object_type})")

    # Add context if present
    if fact.context:
        parts.append(fact.context)

    return " ".join(parts)


class FactIndexer:
    """
    Indexes facts into Qdrant for hybrid search.

    Uses vector-core components:
    - EmbeddingClient for dense vectors
    - GlobalVocabulary for cross-codebase sparse vectors (two-pass indexing)
    - QdrantStorage for vector storage

    This indexer is designed to share a collection with other content types
    (like notes or documents), using the "type" field to distinguish facts.
    """

    def __init__(
        self,
        fact_store: FactStore | None = None,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
        *,
        collection_name: str | None = None,
        base_dir: Path | str | None = None,
        collection_prefix: str = "notes",
    ):
        """
        Initialize indexer.

        The collection can be specified in three ways (in order of precedence):
        1. Explicit collection_name parameter
        2. Generated from base_dir + collection_prefix
        3. Raises ValueError if neither is provided

        Args:
            fact_store: FactStore instance (created if not provided)
            storage: QdrantStorage instance (created if not provided)
            embedder: EmbeddingClient instance (created if not provided)
            global_vocab: GlobalVocabulary instance (uses singleton if not provided)
            collection_name: Explicit Qdrant collection name
            base_dir: Base directory for collection name generation
            collection_prefix: Prefix for generated collection name
        """
        self.fact_store = fact_store or FactStore()
        self._storage = storage
        self._embedder = embedder
        self._global_vocab = global_vocab

        # Determine collection name
        if collection_name:
            self._collection_name = collection_name
        elif base_dir:
            self._collection_name = generate_collection_name(
                str(base_dir),
                prefix=collection_prefix,
            )
        else:
            self._collection_name = None  # Will raise on first use

    @property
    def storage(self) -> QdrantStorage:
        """Get QdrantStorage instance, creating if needed."""
        if self._storage is None:
            self._storage = QdrantStorage()
        return self._storage

    @property
    def embedder(self) -> EmbeddingClient:
        """Get EmbeddingClient instance, creating if needed."""
        if self._embedder is None:
            self._embedder = EmbeddingClient()
        return self._embedder

    @property
    def global_vocab(self) -> GlobalVocabulary:
        """Get GlobalVocabulary instance.

        Note: Returns the instance passed to __init__, or the singleton after
        _ensure_global_vocab() is called. May be None before async initialization.
        """
        if self._global_vocab is None:
            # Synchronous fallback - create new instance
            # (async callers should use _ensure_global_vocab for singleton)
            self._global_vocab = GlobalVocabulary()
        return self._global_vocab

    async def _ensure_global_vocab(self) -> None:
        """Ensure GlobalVocabulary is initialized using singleton."""
        if self._global_vocab is None:
            self._global_vocab = GlobalVocabulary.get_instance()

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        if self._collection_name is None:
            raise ValueError(
                "collection_name not set. Provide collection_name or base_dir to constructor."
            )
        return self._collection_name

    async def ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with payload indexes."""
        if not await self.storage.collection_exists(self.collection_name):
            await self.storage.create_collection(self.collection_name)
            logger.info(f"Created collection: {self.collection_name}")

        # Ensure payload indexes for efficient filtering (idempotent)
        # These match the rich payload fields for "enhanced hybrid" strategy
        await self.storage.ensure_payload_indexes(
            self.collection_name,
            [
                ("type", PayloadSchemaType.KEYWORD),
                ("fact_id", PayloadSchemaType.KEYWORD),
                ("subject_normalized", PayloadSchemaType.KEYWORD),
                ("object_normalized", PayloadSchemaType.KEYWORD),
                ("subject_type", PayloadSchemaType.KEYWORD),
                ("object_type", PayloadSchemaType.KEYWORD),
                ("predicate", PayloadSchemaType.KEYWORD),
                ("confidence", PayloadSchemaType.FLOAT),
                ("source_types", PayloadSchemaType.KEYWORD),
                ("has_deleted_source", PayloadSchemaType.BOOL),
                ("has_modified_source", PayloadSchemaType.BOOL),
                ("has_relocated_source", PayloadSchemaType.BOOL),
            ],
        )

    async def index_all(self, force: bool = False) -> dict:
        """
        Index all facts using two-pass GlobalVocabulary pattern.

        Pass 1: Collect tokens from all facts and register with GlobalVocabulary
        Pass 2: Generate embeddings and sparse vectors, upsert to Qdrant

        Args:
            force: If True, reindex all facts. If False, incremental update.

        Returns:
            dict with indexing results
        """
        await self._ensure_global_vocab()
        await self.ensure_collection()

        # Get all facts
        all_facts = self.fact_store.list_summaries()

        if not all_facts:
            logger.info("No facts to index")
            return {
                "total": 0,
                "indexed": 0,
                "last_indexed": datetime.now(UTC).isoformat(),
            }

        # Get existing indexed facts
        if not force:
            indexed_ids = await self._get_indexed_fact_ids()
        else:
            indexed_ids = set()
            # Delete all fact points
            await self._delete_all_fact_points()

        # Filter to new facts only
        facts_to_index = []
        for summary in all_facts:
            if force or str(summary.id) not in indexed_ids:
                # Load full fact
                try:
                    fact = self.fact_store.read(summary.id)
                    facts_to_index.append(fact)
                except FactNotFoundError:
                    # Fact was deleted between list and read - skip silently
                    logger.debug(f"Fact {summary.id} not found, skipping")
                    continue
                except Exception as e:
                    # Unexpected error - log with details and continue
                    logger.warning(
                        f"Unexpected error reading fact {summary.id}: {e}",
                        exc_info=True,
                    )
                    continue

        if not facts_to_index:
            logger.info("No new facts to index")
            return {
                "total": len(all_facts),
                "indexed": 0,
                "last_indexed": datetime.now(UTC).isoformat(),
            }

        # Pass 1: Collect tokens for GlobalVocabulary registration
        tokens_per_doc: list[set[str]] = []
        for fact in facts_to_index:
            text = generate_fact_text(fact)
            tokens_per_doc.append(set(self.global_vocab.tokenize(text)))

        # Register facts vocabulary
        self.global_vocab.register_codebase(FACTS_CODEBASE_ID, tokens_per_doc)

        # Pass 2: Index facts with embeddings and sparse vectors
        indexed_count = 0
        for fact in facts_to_index:
            try:
                await self._index_fact(fact)
                indexed_count += 1
            except EmbeddingServiceError as e:
                # Embedding service unavailable - log and continue with remaining facts
                logger.error(f"Embedding failed for fact {fact.id}: {e}")
            except QdrantConnectionError as e:
                # Storage unavailable - abort indexing to avoid partial state
                logger.error(f"Storage unavailable, aborting indexing: {e}")
                raise
            except Exception as e:
                # Unexpected error - log with full traceback and continue
                logger.error(
                    f"Unexpected error indexing fact {fact.id}: {e}",
                    exc_info=True,
                )

        logger.info(f"Indexed {indexed_count}/{len(facts_to_index)} facts")

        return {
            "total": len(all_facts),
            "indexed": indexed_count,
            "last_indexed": datetime.now(UTC).isoformat(),
        }

    async def index_fact(self, fact: Fact) -> None:
        """
        Index a single fact.

        Ensures GlobalVocabulary is trained before indexing.

        Args:
            fact: Fact to index
        """
        await self._ensure_global_vocab()
        await self.ensure_collection()

        # Ensure GlobalVocabulary is trained
        if self.global_vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 0:
            await self._train_vocabulary()

        # Delete existing point for this fact
        await self._delete_fact_point(fact.id)

        # Index
        await self._index_fact(fact)

    async def delete_fact_index(self, fact_id: UUID) -> None:
        """
        Remove a fact from the index.

        Args:
            fact_id: Fact UUID to remove
        """
        await self._delete_fact_point(fact_id)

    async def _index_fact(self, fact: Fact) -> None:
        """Index a single fact."""
        if self.global_vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 0:
            raise RuntimeError("GlobalVocabulary not initialized for facts codebase")

        # Generate searchable text
        text = generate_fact_text(fact)

        # Get embedding
        embedding = await self.embedder.embed_single_cached(text)

        # Generate sparse vector
        sparse = self.global_vocab.vectorize_document(text)

        # Generate point ID
        point_id = generate_point_id(f"fact:{fact.id}")

        # Build rich payload for "enhanced hybrid" strategy
        # Includes normalized fields for filtering and integrity flags
        source_types = list({s.source_type.value for s in fact.sources})
        has_deleted = any(s.status == SourceStatus.DELETED for s in fact.sources)
        has_modified = any(s.status == SourceStatus.MODIFIED for s in fact.sources)
        has_relocated = any(s.status == SourceStatus.RELOCATED for s in fact.sources)

        payload = {
            "type": "fact",
            "fact_id": str(fact.id),
            # Core triple
            "subject": fact.subject,
            "subject_type": fact.subject_type,
            "predicate": fact.predicate,
            "object": fact.object_value,
            "object_type": fact.object_type,
            # Normalized for filtering (indexed)
            "subject_normalized": fact.subject.lower(),
            "object_normalized": fact.object_value.lower(),
            # Full metadata
            "context": fact.context,
            "confidence": fact.confidence,
            "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            # Sources - types for filtering, count for display
            "source_types": source_types,
            "source_count": len(fact.sources),
            # Integrity flags (indexed)
            "has_deleted_source": has_deleted,
            "has_modified_source": has_modified,
            "has_relocated_source": has_relocated,
            # Timestamps
            "created": fact.created.isoformat(),
            "modified": fact.modified.isoformat(),
            "content": text,  # For highlight extraction
        }

        # Create point
        point = PointStruct(
            id=point_id,
            vector={
                "dense": embedding,
                "sparse": QdrantSparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                ),
            },
            payload=payload,
        )

        # Upsert
        await self.storage.upsert_batch(self.collection_name, [point])
        logger.debug(f"Indexed fact {fact.id}")

    async def _delete_fact_point(self, fact_id: UUID) -> None:
        """Delete point for a fact."""
        await self.storage.delete_by_filter(
            self.collection_name,
            field="fact_id",
            value=str(fact_id),
        )

    async def _delete_all_fact_points(self) -> None:
        """Delete all fact points from collection."""
        await self.storage.delete_by_filter(
            self.collection_name,
            field="type",
            value="fact",
        )

    async def _get_indexed_fact_ids(self) -> set[str]:
        """Get set of indexed fact IDs."""
        try:
            points = await self.storage.scroll_points(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="type", match=MatchValue(value="fact")),
                ],
                payload_fields=["fact_id"],
            )

            return {p.get("fact_id", "") for p in points if p.get("fact_id")}
        except QdrantConnectionError:
            # Collection may not exist yet - return empty set for fresh indexing
            logger.debug("Collection not found, starting fresh index")
            return set()
        except Exception as e:
            # Unexpected error - log but allow indexing to proceed
            logger.warning(
                f"Could not retrieve indexed fact IDs: {e}",
                exc_info=True,
            )
            return set()

    async def _train_vocabulary(self) -> None:
        """Train GlobalVocabulary on all facts for sparse vector generation."""
        tokens_per_doc: list[set[str]] = []

        all_facts = self.fact_store.list_summaries()
        for summary in all_facts:
            try:
                fact = self.fact_store.read(summary.id)
                text = generate_fact_text(fact)
                tokens_per_doc.append(set(self.global_vocab.tokenize(text)))
            except FactNotFoundError:
                # Fact was deleted between list and read - skip silently
                logger.debug(f"Fact {summary.id} not found during vocabulary training")
                continue
            except Exception as e:
                # Unexpected error - log with details and continue
                logger.warning(
                    f"Failed to tokenize fact {summary.id} for vocabulary: {e}",
                    exc_info=True,
                )
                continue

        # Register vocabulary
        self.global_vocab.register_codebase(FACTS_CODEBASE_ID, tokens_per_doc)

    async def close(self) -> None:
        """Close connections safely."""
        if self._storage is not None:
            await self._storage.close()
            self._storage = None
        if self._embedder is not None:
            await self._embedder.close()
            self._embedder = None
