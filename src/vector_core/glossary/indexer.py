"""Glossary indexer using GlobalVocabulary for sparse vectors."""

import logging
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue, PayloadSchemaType

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.glossary.models import GlossaryEntry
from vector_core.glossary.store import GlossaryStore
from vector_core.storage.hybrid import HybridSearcher
from vector_core.storage.qdrant import QdrantStorage, generate_point_id
from vector_core.utils.hashing import hash_content

logger = logging.getLogger(__name__)

# Codebase ID for GlobalVocabulary registration
GLOSSARY_CODEBASE_ID = "glossary"

# Payload indexes for glossary entries
GLOSSARY_PAYLOAD_INDEXES = [
    ("type", PayloadSchemaType.KEYWORD),
    ("term_normalized", PayloadSchemaType.KEYWORD),
    ("domain", PayloadSchemaType.KEYWORD),
]


def _generate_embedding_content(entry: GlossaryEntry) -> str:
    """Generate content string for embedding."""
    parts = [entry.term, entry.expansion, entry.definition]
    if entry.domain:
        parts.append(entry.domain)
    if entry.aliases:
        parts.extend(entry.aliases)
    return " ".join(parts)


class GlossaryIndexer:
    """
    Indexes glossary entries into Qdrant for semantic search.

    Uses type="glossary" in payload for filtering.
    Requires GlobalVocabulary for sparse vectors.

    Two-pass indexing pattern:
    1. Collect tokens from all entries and register with GlobalVocabulary
    2. Generate embeddings and sparse vectors, upsert to Qdrant
    """

    def __init__(
        self,
        collection_name: str,
        glossary_store: GlossaryStore | None = None,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
    ):
        """
        Initialize glossary indexer.

        Args:
            collection_name: Qdrant collection name
            glossary_store: GlossaryStore instance (creates default if None)
            storage: QdrantStorage instance (creates default if None)
            embedder: EmbeddingClient instance (creates default if None)
            global_vocab: GlobalVocabulary instance (creates default if None)
        """
        self.collection_name = collection_name
        self.glossary_store = glossary_store or GlossaryStore()
        self.storage = storage or QdrantStorage()
        self.embedder = embedder or EmbeddingClient()
        self.global_vocab = global_vocab or GlobalVocabulary()
        self.hybrid_searcher = HybridSearcher(self.storage)

    async def ensure_collection(self) -> bool:
        """
        Ensure collection exists with required indexes.

        Returns:
            True if collection was created, False if existed
        """
        return await self.storage.ensure_collection_with_indexes(
            self.collection_name,
            GLOSSARY_PAYLOAD_INDEXES,
        )

    async def index_all(self, force: bool = False) -> int:
        """
        Index all glossary entries.

        Uses two-pass pattern:
        1. Collect tokens from all entries
        2. Register codebase with GlobalVocabulary
        3. Generate embeddings and sparse vectors
        4. Upsert to Qdrant

        Args:
            force: If True, reindex all entries even if unchanged

        Returns:
            Number of entries indexed
        """
        entries = list(self.glossary_store.iter_all())
        if not entries:
            return 0

        # Pass 1: Collect tokens for GlobalVocabulary registration
        tokens_per_entry: list[set[str]] = []
        for entry in entries:
            content = _generate_embedding_content(entry)
            tokens_per_entry.append(set(self.global_vocab.tokenize(content)))

        # Register this codebase's vocabulary
        self.global_vocab.register_codebase(GLOSSARY_CODEBASE_ID, tokens_per_entry)

        # Pass 2: Generate embeddings + sparse vectors, upsert
        points = []
        for entry in entries:
            content = _generate_embedding_content(entry)
            dense = await self.embedder.embed_single_cached(content)
            sparse = self.global_vocab.vectorize_document(content)

            point_id = generate_point_id(f"glossary:{entry.id}")
            point = self.storage.create_point(
                point_id=point_id,
                dense_vector=dense,
                sparse_vector=sparse,
                payload=self._create_payload(entry),
            )
            points.append(point)

        if points:
            await self.storage.upsert_batch(self.collection_name, points)

        logger.info(f"Indexed {len(points)} glossary entries")
        return len(points)

    async def index_entry(self, entry_id: UUID) -> None:
        """
        Index a single entry.

        For glossary (small corpus), always re-registers all vocabulary
        to ensure IDF is accurate.

        Args:
            entry_id: UUID of the entry to index
        """
        entry = self.glossary_store.read(entry_id)

        # Re-register vocabulary (glossary is small, this is fast)
        entries = list(self.glossary_store.iter_all())
        tokens_per_entry: list[set[str]] = []
        for e in entries:
            content = _generate_embedding_content(e)
            tokens_per_entry.append(set(self.global_vocab.tokenize(content)))
        self.global_vocab.register_codebase(GLOSSARY_CODEBASE_ID, tokens_per_entry)

        # Generate vectors for this entry
        content = _generate_embedding_content(entry)
        dense = await self.embedder.embed_single_cached(content)
        sparse = self.global_vocab.vectorize_document(content)

        point_id = generate_point_id(f"glossary:{entry.id}")
        await self.storage.upsert_point(
            collection=self.collection_name,
            point_id=point_id,
            dense_vector=dense,
            sparse_vector=sparse,
            payload=self._create_payload(entry),
        )

    async def delete_entry_index(self, entry_id: UUID) -> None:
        """
        Delete an entry from the index.

        Args:
            entry_id: UUID of the entry to delete
        """
        await self.storage.delete_by_filter(
            collection=self.collection_name,
            field="glossary_id",
            value=str(entry_id),
        )

    async def search(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Semantic search for glossary entries.

        Args:
            query: Search query
            domain: Optional domain filter
            limit: Maximum results

        Returns:
            List of matching entries with scores
        """
        # Generate query vectors
        dense_query = await self.embedder.embed_single_cached(query)
        sparse_query = self.global_vocab.vectorize_query(query)

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="type", match=MatchValue(value="glossary"))
        ]
        if domain:
            filter_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )

        # Query using hybrid search with RRF fusion
        results = await self.hybrid_searcher.search(
            collection=self.collection_name,
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=limit,
            filter_conditions=filter_conditions,
        )

        return [
            {
                **dict(r.payload or {}),
                "score": r.score,
            }
            for r in results
        ]

    def _create_payload(self, entry: GlossaryEntry) -> dict:
        """Create Qdrant payload for an entry."""
        return {
            "type": "glossary",
            "glossary_id": str(entry.id),
            "term": entry.term,
            "term_normalized": entry.term.lower(),
            "expansion": entry.expansion,
            "definition": entry.definition[:2000],  # Truncate for storage
            "domain": entry.domain,
            "aliases": entry.aliases,
            "created": entry.created.isoformat(),
            "modified": entry.modified.isoformat(),
            "entry_hash": hash_content(
                f"{entry.term}|{entry.expansion}|{entry.definition}"
            ),
        }

    async def close(self) -> None:
        """Close resources safely."""
        if self.storage is not None:
            await self.storage.close()
        if self.global_vocab is not None:
            self.global_vocab.close()
        if self.glossary_store is not None:
            self.glossary_store.close()
