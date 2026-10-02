"""Glossary indexer using GlobalVocabulary for sparse vectors."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue, PayloadSchemaType

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.glossary.models import GlossaryEntry
from vector_core.glossary.store import GlossaryStore
from vector_core.storage.embedding_fragments import (
    FRAGMENT_KEY,
    fragment_marker,
    fragment_point,
    upsert_fragment_group,
)
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    active_embedding_collection,
    embedding_collection_lock,
    ensure_embedding_collection,
)
from vector_core.storage.embedding_sources import resolve_shared_embedding_text
from vector_core.storage.hybrid import HybridSearcher
from vector_core.storage.qdrant import QdrantStorage, generate_point_id
from vector_core.utils.hashing import hash_content

logger = logging.getLogger(__name__)

# Codebase ID for GlobalVocabulary registration
GLOSSARY_CODEBASE_ID = "glossary"

# Payload indexes for glossary entries
GLOSSARY_PAYLOAD_INDEXES = [
    ("type", PayloadSchemaType.KEYWORD),
    ("glossary_id", PayloadSchemaType.KEYWORD),
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
        *,
        generation: CollectionGeneration | None = None,
        text_resolver: Callable[[dict[str, Any]], Awaitable[str | None]] | None = None,
        finalize_candidate: Callable[[str], Awaitable[None]] | None = None,
    ):
        """
        Initialize glossary indexer.

        Args:
            collection_name: Qdrant collection name
            glossary_store: GlossaryStore instance (creates default if None)
            storage: QdrantStorage instance (creates default if None)
            embedder: EmbeddingClient instance (creates default if None)
            global_vocab: GlobalVocabulary instance (creates default if None)
            generation: Operation-scoped target. The caller must hold the logical
                collection lock for this indexer's entire lifetime.
            text_resolver: Optional resolver for all types sharing the collection.
            finalize_candidate: Optional candidate rebuild before publication.
        """
        self.collection_name = collection_name
        self.logical_name = collection_name
        self._generation = generation
        self._text_resolver = text_resolver or self._resolve_embedding_text
        self._finalize_candidate = finalize_candidate
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
        if self._generation is not None:
            return False
        async with embedding_collection_lock(self.storage, self.logical_name):
            previous = await active_embedding_collection(self.storage, self.logical_name)
            generation = await self._ensure_generation(lock_held=True)
            return generation.physical_name != previous

    async def _resolve_embedding_text(self, payload: dict[str, Any]) -> str:
        return await resolve_shared_embedding_text(payload, glossary_store=self.glossary_store)

    async def _ensure_generation(self, *, lock_held: bool = False) -> CollectionGeneration:
        if self._generation is not None:
            return self._generation
        return await ensure_embedding_collection(
            self.storage,
            self.logical_name,
            self.embedder,
            self._text_resolver,
            payload_indexes=GLOSSARY_PAYLOAD_INDEXES,
            lock_held=lock_held,
            finalize_candidate=self._finalize_candidate,
            vectorize=self.global_vocab.vectorize_document,
        )

    @asynccontextmanager
    async def collection_operation(self) -> AsyncIterator[CollectionGeneration]:
        """Resolve before source mutation and retain the lock through vector writes."""
        if self._generation is not None:
            yield self._generation
        else:
            async with embedding_collection_lock(self.storage, self.logical_name):
                generation = await self._ensure_generation(lock_held=True)
                yield generation

    @asynccontextmanager
    async def _mutation_target(self) -> AsyncIterator[str]:
        async with self.collection_operation() as generation:
            yield generation.physical_name

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
        async with self._mutation_target() as collection:
            return await self._index_all(collection, force=force)

    async def _index_all(self, collection: str, *, force: bool = False) -> int:
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
        for entry in entries:
            content = _generate_embedding_content(entry)
            sparse = self.global_vocab.vectorize_document(content)

            point_id = generate_point_id(f"glossary:{entry.id}")
            points = await fragment_point(
                self.embedder,
                point_id=point_id,
                payload=self._create_payload(entry),
                sparse=sparse,
                text=content,
                vectorize=self.global_vocab.vectorize_document,
            )
            await upsert_fragment_group(self.storage, collection, points)

        logger.info(f"Indexed {len(entries)} glossary entries")
        return len(entries)

    async def index_entry(self, entry_id: UUID) -> None:
        """
        Index a single entry.

        For glossary (small corpus), always re-registers all vocabulary
        to ensure IDF is accurate.

        Args:
            entry_id: UUID of the entry to index
        """
        async with self._mutation_target() as collection:
            await self._index_entry(entry_id, collection)

    async def _index_entry(self, entry_id: UUID, collection: str) -> None:
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
        sparse = self.global_vocab.vectorize_document(content)

        point_id = generate_point_id(f"glossary:{entry.id}")
        points = await fragment_point(
            self.embedder,
            point_id=point_id,
            payload=self._create_payload(entry),
            sparse=sparse,
            text=content,
            vectorize=self.global_vocab.vectorize_document,
        )
        await upsert_fragment_group(self.storage, collection, points)

    async def delete_entry_index(self, entry_id: UUID) -> None:
        """
        Delete an entry from the index.

        Args:
            entry_id: UUID of the entry to delete
        """
        async with self._mutation_target() as collection:
            await self.storage.delete_by_filter(
                collection=collection,
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
        generation = await self._ensure_generation()
        # Generate query vectors
        dense_query = await self.embedder.embed_single_cached(query, role="query")
        sparse_query = self.global_vocab.vectorize_query(query)

        # Build filter conditions
        filter_conditions = [FieldCondition(key="type", match=MatchValue(value="glossary"))]
        if domain:
            filter_conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        # Query using hybrid search with RRF fusion
        results = await self.hybrid_searcher.search(
            collection=generation.physical_name,
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=limit,
            filter_conditions=filter_conditions,
            group_by="glossary_id",
        )

        markers = [fragment_marker(result.payload or {}) for result in results]
        parent_ids = list(
            dict.fromkeys(
                marker["parent_id"] for marker in markers if marker and marker["index"] > 0
            )
        )
        parents = {}
        if parent_ids:
            client = await self.storage.get_client()
            # Hydrate final winners only, without fetching their complete raw
            # embedding source or duplicating display fields onto stored children.
            records = await client.retrieve(
                generation.physical_name,
                ids=parent_ids,
                with_payload=["type", "glossary_id", "expansion", "definition", FRAGMENT_KEY],
                with_vectors=False,
            )
            parents = {record.id: record.payload or {} for record in records}

        output = []
        for result, marker in zip(results, markers, strict=True):
            payload = dict(result.payload or {})
            if marker and marker["index"] > 0:
                parent = parents.get(marker["parent_id"], {})
                parent_marker = fragment_marker(parent)
                if (
                    parent.get("type") != "glossary"
                    or parent.get("glossary_id") != payload.get("glossary_id")
                    or parent_marker is None
                    or parent_marker["index"] != 0
                    or parent_marker["parent_id"] != marker["parent_id"]
                    or parent_marker["source_hash"] != marker["source_hash"]
                ):
                    raise ValueError("Glossary fragment has no matching canonical source")
                for field in ("expansion", "definition"):
                    if not isinstance(parent.get(field), str):
                        raise ValueError(f"Canonical glossary source has no {field}")
                    payload[field] = parent[field]
            output.append({**payload, "score": result.score})
        return output

    @staticmethod
    def _create_payload(entry: GlossaryEntry) -> dict:
        """Create Qdrant payload for an entry."""
        return {
            "type": "glossary",
            "glossary_id": str(entry.id),
            "term": entry.term,
            "term_normalized": entry.term.lower(),
            "expansion": entry.expansion,
            "definition": entry.definition[
                :2000
            ],  # Compact presentation; raw input is retained below.
            "domain": entry.domain,
            "aliases": entry.aliases,
            "embedding_text": _generate_embedding_content(entry),
            "created": entry.created.isoformat(),
            "modified": entry.modified.isoformat(),
            "entry_hash": hash_content(f"{entry.term}|{entry.expansion}|{entry.definition}"),
        }

    async def close(self) -> None:
        """Close resources safely."""
        if self.storage is not None:
            await self.storage.close()
        if self.global_vocab is not None:
            self.global_vocab.close()
        if self.glossary_store is not None:
            self.glossary_store.close()
