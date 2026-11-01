"""Fact indexer for Qdrant-based semantic search.

This module provides semantic search indexing for facts (SPO triples).
It's designed to be shared across MCP servers, with the collection name
provided at construction time to allow integration with existing collections.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from qdrant_client.models import (
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from vector_core.embeddings.client import EmbeddingClient, EmbeddingServiceError
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.facts.models import Fact, SourceStatus
from vector_core.storage.embedding_fragments import (
    FRAGMENT_KEY,
    fragment_point,
    is_derived_fragment,
    upsert_fragment_group,
)
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    embedding_collection_lock,
    ensure_embedding_collection,
)
from vector_core.storage.embedding_sources import resolve_shared_embedding_text
from vector_core.storage.fragment_recovery import FragmentGroupRecoveryError
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
        generation: CollectionGeneration | None = None,
        text_resolver: Callable[[dict[str, Any]], Awaitable[str | None]] | None = None,
        finalize_candidate: Callable[[str], Awaitable[None]] | None = None,
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
            generation: Operation-scoped target. The caller must hold the logical
                collection lock for this indexer's entire lifetime.
            text_resolver: Optional resolver for all types sharing the collection.
            finalize_candidate: Optional candidate rebuild before publication.
        """
        self.fact_store = fact_store or FactStore()
        self._storage = storage
        self._embedder = embedder
        self._global_vocab = global_vocab
        self._generation = generation
        self._text_resolver = text_resolver or resolve_shared_embedding_text
        self._finalize_candidate = finalize_candidate

        # Determine collection name
        self._collection_name: str | None
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
        await self._ensure_generation()

    @property
    def logical_name(self) -> str:
        """Stable collection name used to resolve each operation's generation."""
        return self.collection_name

    async def _ensure_generation(self, *, lock_held: bool = False) -> CollectionGeneration:
        if self._generation is not None:
            return self._generation

        # Ensure payload indexes for efficient filtering (idempotent)
        # These match the rich payload fields for "enhanced hybrid" strategy
        return await ensure_embedding_collection(
            self.storage,
            self.logical_name,
            self.embedder,
            self._text_resolver,
            lock_held=lock_held,
            finalize_candidate=self._finalize_candidate,
            vectorize=self.global_vocab.vectorize_document,
            payload_indexes=[
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

    @asynccontextmanager
    async def _mutation_target(self) -> AsyncIterator[str]:
        if self._generation is not None:
            yield self._generation.physical_name
        else:
            async with embedding_collection_lock(self.storage, self.logical_name):
                generation = await self._ensure_generation(lock_held=True)
                yield generation.physical_name

    def _iter_fact_tokens(self) -> Iterator[tuple[Fact, set[str]]]:
        """Yield ``(fact, token_set)`` for every readable fact in the store.

        ``FactStore.iter_all()`` already skips facts that fail to load (deleted
        or malformed rows); this additionally skips a fact whose text fails to
        tokenize, so a single bad fact cannot abort indexing of the rest. A
        failure of the underlying fact query itself still propagates.
        """
        for fact in self.fact_store.iter_all():
            try:
                tokens = set(self.global_vocab.tokenize(generate_fact_text(fact)))
            except Exception:
                logger.warning("Skipping fact %s: tokenization failed", fact.id, exc_info=True)
                continue
            yield fact, tokens

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
        async with self._mutation_target() as collection:
            return await self._index_all(collection, force=force)

    async def _index_all(self, collection: str, *, force: bool = False) -> dict:
        await self._ensure_global_vocab()

        # Incremental mode needs the facts already stored, with the token sets
        # their points hold; force mode reindexes every fact. None means the
        # scan failed: every fact is then reindexed, and which of them already
        # had points is unknown.
        stored = {} if force else await self._scan_stored_facts(collection)

        # Read and tokenize the COMPLETE fact corpus BEFORE any destructive
        # delete. Two things must span every fact, not just the ones upserted:
        #   * iter_all() — the previous list_summaries() call defaulted to
        #     limit=50, so "index all facts" silently indexed only the 50
        #     most-recently-modified facts and left every older fact
        #     unsearchable by semantic fact search.
        #   * tokens_per_doc — register_codebase() replaces the "facts"
        #     codebase's entire vocabulary contribution, so it needs every
        #     fact's tokens for correct IDF statistics. Collecting tokens from
        #     only the incremental batch corrupted the vocabulary, dropping the
        #     codebase document count to the size of that batch. Facts already
        #     stored contribute the tokens their points hold, which a failed
        #     earlier write can leave behind the store's current text.
        # Reading before deleting means a corpus read failure leaves the
        # existing index intact instead of emptying it. Mirrors
        # NoteIndexer.index_all's two-pass pattern.
        to_index: list[tuple[Fact, set[str]]] = []
        total_facts = 0

        for fact, tokens in self._iter_fact_tokens():
            total_facts += 1
            if stored is None or str(fact.id) not in stored:
                to_index.append((fact, tokens))

        # The corpus read succeeded; in force mode clear stale points now. This
        # runs even when the scan yielded nothing (the store may have been fully
        # emptied) so a force rebuild never leaves deleted facts searchable, but
        # only after the read so a read failure can't empty the index.
        if force:
            await self._delete_all_fact_points(collection)

        # Register the facts vocabulary from the complete corpus. This runs even
        # for an empty corpus, so deleting the last fact clears the stale facts
        # contribution — otherwise GlobalVocabulary keeps a nonzero doc count and
        # skewed IDF, and index_fact() would not rebuild it on the next add.
        tokens_per_doc = [tokens for _fact, tokens in to_index]
        if stored:
            tokens_per_doc.extend(stored.values())
        self.global_vocab.register_codebase(FACTS_CODEBASE_ID, tokens_per_doc)

        if total_facts == 0:
            logger.info("No facts to index")
            return {
                "total": 0,
                "indexed": 0,
                "last_indexed": datetime.now(UTC).isoformat(),
            }

        if not to_index:
            logger.info("No new facts to index")
            return {
                "total": total_facts,
                "indexed": 0,
                "last_indexed": datetime.now(UTC).isoformat(),
            }

        # Pass 2: Index facts with embeddings and sparse vectors.
        #
        # Every fact above is registered, but only the ones written below gain
        # a point. Each fact leaves ``unwritten`` the moment its group is
        # stored; whatever remains when the pass ends, however it ends (a
        # skipped failure, an abort, cancellation), is retired again so the
        # contribution describes exactly the facts with stored points.
        unwritten = {str(fact.id): tokens for fact, tokens in to_index}
        indexed_count = 0
        try:
            for fact, _tokens in to_index:
                try:
                    await self._index_fact(
                        fact, collection, on_commit=partial(unwritten.pop, str(fact.id), None)
                    )
                    indexed_count += 1
                except EmbeddingServiceError as e:
                    # Embedding service unavailable - log and continue with remaining facts
                    logger.error(f"Embedding failed for fact {fact.id}: {e}")
                except QdrantConnectionError as e:
                    # Storage unavailable - abort indexing to avoid partial state
                    logger.error(f"Storage unavailable, aborting indexing: {e}")
                    raise
                except FragmentGroupRecoveryError:
                    # The prior group is journaled, but storage requires explicit
                    # recovery. Do not bury this among per-fact indexing failures.
                    raise
                except Exception as e:
                    # Unexpected error - log with full traceback and continue
                    logger.error(
                        f"Unexpected error indexing fact {fact.id}: {e}",
                        exc_info=True,
                    )
        finally:
            await self._retire_unwritten(collection, unwritten, stored)

        logger.info(f"Indexed {indexed_count}/{len(to_index)} facts")

        return {
            "total": total_facts,
            "indexed": indexed_count,
            "last_indexed": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # Sparse vocabulary accounting
    #
    # Single-fact writes keep the facts contribution to GlobalVocabulary
    # current with incremental deltas instead of re-registering the whole
    # corpus on every write. Two invariants keep sparse retrieval honest:
    #
    # 1. A fact's terms are in the vocabulary before the fact is vectorized.
    #    ``vectorize_document`` silently drops unknown tokens, so a term
    #    registered late never reaches the sparse vector of the fact that
    #    introduced it.
    # 2. The contribution describes exactly the indexed facts. Its document
    #    count and document frequencies feed the IDF weights of every codebase
    #    sharing the vocabulary database.
    #
    # Each fact is one vocabulary document, tokenized from
    # ``generate_fact_text(fact)``: the text ``_index_fact`` stores in the
    # canonical point's ``content``. The removal side reads that stored text
    # back, so adding and retiring a fact tokenize the same text and repeated
    # edits cannot leak document frequency. Derived fragments are dense slices
    # of the same fact, not further documents.
    #
    # Hence the contribution describes the facts with a stored canonical point,
    # not the facts in the store: a fact whose write failed is not counted.
    # ------------------------------------------------------------------

    async def _stored_fact_tokens(
        self, collection: str, fact_id: UUID | None = None
    ) -> dict[str, set[str]]:
        """Token set of every fact with a stored canonical point, by fact id.

        Restricted to one fact when ``fact_id`` is given. A canonical point that
        predates the current payload layout and holds no usable text still
        counts as a stored fact, just one without tokens.
        """
        if fact_id is None:
            condition = FieldCondition(key="type", match=MatchValue(value="fact"))
        else:
            condition = FieldCondition(key="fact_id", match=MatchValue(value=str(fact_id)))
        payloads = await self.storage.scroll_points(
            collection,
            filter_conditions=[condition],
            # fact_id keeps the projection non-empty for a point without
            # content; scroll_points drops points whose payload comes back empty.
            payload_fields=["fact_id", "content", FRAGMENT_KEY],
            max_results=0,
        )
        stored: dict[str, set[str]] = {}
        for payload in payloads:
            stored_id = payload.get("fact_id")
            try:
                derived = is_derived_fragment(payload)
            except ValueError:
                # Broken lineage is never source text; one damaged point must
                # not block accounting for every other fact.
                logger.warning(f"Ignoring fact point with a malformed fragment marker: {stored_id}")
                continue
            if derived or not isinstance(stored_id, str):
                continue
            tokens = stored.setdefault(stored_id, set())
            text = payload.get("content")
            if isinstance(text, str):
                tokens.update(self.global_vocab.tokenize(text))
        return stored

    async def _indexed_token_set(self, fact_id: UUID, collection: str) -> tuple[bool, set[str]]:
        """Tokens currently registered for a fact, read back from its stored point.

        Returns ``(is_indexed, tokens)``. An unindexed fact has no canonical
        point and contributes nothing.
        """
        tokens = (await self._stored_fact_tokens(collection, fact_id)).get(str(fact_id))
        return tokens is not None, tokens or set()

    async def _rebuild_vocabulary(self, collection: str) -> None:
        """Register the facts contribution from the facts stored in ``collection``.

        Training from the store instead would count facts that have no point
        yet, so a later write of one of them would count it a second time. A
        failed registration is logged, not raised, like any vocabulary write.
        """
        stored = await self._stored_fact_tokens(collection)
        try:
            self.global_vocab.register_codebase(FACTS_CODEBASE_ID, list(stored.values()))
        except Exception as e:
            logger.warning(f"Failed to rebuild the facts vocabulary contribution: {e}")

    async def _scan_stored_facts(self, collection: str) -> dict[str, set[str]] | None:
        """``_stored_fact_tokens`` for index_all: None if the points could not be read."""
        try:
            return await self._stored_fact_tokens(collection)
        except QdrantConnectionError:
            # Collection may not exist yet - index every fact
            logger.debug("Collection not found, starting fresh index")
            return None
        except Exception as e:
            # Unexpected error - log but allow indexing to proceed
            logger.warning(
                f"Could not retrieve indexed facts: {e}",
                exc_info=True,
            )
            return None

    async def _retire_unwritten(
        self,
        collection: str,
        unwritten: dict[str, set[str]],
        stored_before: dict[str, set[str]] | None,
    ) -> None:
        """Retire facts ``index_all`` registered but left without a stored point.

        A fact that was not written keeps what it had before the pass, as a
        failed write restores the previous group. Force mode deleted every fact
        point first and incremental mode only writes facts without one, so
        ``stored_before`` already shows none of these has a point. Only when
        the incremental scan failed (None) are the points read back. A failed
        read or vocabulary write is logged, not raised.
        """
        if not unwritten:
            return
        stored = stored_before
        if stored is None:
            stored = await self._scan_stored_facts(collection)
            if stored is None:
                logger.warning(
                    f"{len(unwritten)} facts that were not indexed stay in the facts "
                    "vocabulary contribution until index_all(force=True)"
                )
                return
        retired = [tokens for fact_id, tokens in unwritten.items() if fact_id not in stored]
        if retired:
            self._update_vocabulary([], retired, -len(retired))

    def _update_vocabulary(
        self,
        added_tokens: list[set[str]],
        removed_tokens: list[set[str]],
        net_doc_change: int,
    ) -> bool:
        """Apply a facts vocabulary delta, logging rather than raising on failure.

        Returns whether the delta reached the database. Callers must not
        compensate for a delta that was never applied: inverting one that
        failed would subtract tokens and a document that were never added.

        A fact write must not fail because the shared vocabulary database was
        busy: a missed delta degrades sparse ranking, while a raised error would
        lose the fact. ``index_all(force=True)`` rebuilds the contribution from
        scratch and repairs any drift.
        """
        try:
            self.global_vocab.update_codebase_incremental(
                FACTS_CODEBASE_ID,
                added_tokens=added_tokens,
                removed_tokens=removed_tokens,
                net_doc_change=net_doc_change,
            )
        except Exception as e:
            logger.warning(f"Failed to update the facts vocabulary contribution: {e}")
            return False
        return True

    async def index_fact(self, fact: Fact) -> None:
        """
        Index a single fact.

        The facts vocabulary contribution is updated before the fact is
        vectorized, so terms it introduces reach its own sparse vector. A
        re-index swaps the fact's stored token set for its new one; a new fact
        adds one document. An empty contribution is first rebuilt from the
        stored facts. A failed vocabulary write is logged and indexing
        continues; a failed embed or upsert reverts the delta.

        Args:
            fact: Fact to index
        """
        async with self._mutation_target() as collection:
            await self._index_single_fact(fact, collection)

    async def _index_single_fact(self, fact: Fact, collection: str) -> None:
        await self._ensure_global_vocab()

        if self.global_vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 0:
            # Possibly never trained, or reset while facts are stored: rebuild
            # from the stored points, then apply this fact's delta as usual.
            await self._rebuild_vocabulary(collection)

        # Register the delta before _index_fact vectorizes (invariant 1).
        added = set(self.global_vocab.tokenize(generate_fact_text(fact)))
        was_indexed, previous = await self._indexed_token_set(fact.id, collection)
        removed = [previous] if was_indexed else []
        doc_change = 0 if was_indexed else 1
        applied = self._update_vocabulary([added], removed, doc_change)

        # Index. The point id is stable per fact, so the upsert overwrites any
        # existing point in place; pre-deleting first would only open a window
        # where a failed embed/upsert leaves the fact missing from search.
        stored = False

        def mark_stored() -> None:
            nonlocal stored
            stored = True

        try:
            await self._index_fact(fact, collection, on_commit=mark_stored)
        except BaseException:
            # Until the new group is stored the fact is unchanged: a failed
            # embed writes nothing, and a failed upsert restores (or journals
            # for recovery) the previous group, so its contribution must be
            # unchanged too. Cancellation counts. Only a cancellation during the
            # post-commit cleanup follows a stored write, and that delta stands.
            if applied and not stored:
                self._update_vocabulary(removed, [added], -doc_change)
            raise

    async def delete_fact_index(self, fact_id: UUID) -> None:
        """
        Remove a fact from the index.

        The fact's vocabulary contribution is retired only once its points are
        gone. Subtracting after a failed delete would leave the fact searchable
        while its terms had been written off.

        Args:
            fact_id: Fact UUID to remove
        """
        async with self._mutation_target() as collection:
            await self._delete_single_fact(fact_id, collection)

    async def _delete_single_fact(self, fact_id: UUID, collection: str) -> None:
        await self._ensure_global_vocab()
        try:
            was_indexed, tokens = await self._indexed_token_set(fact_id, collection)
        except Exception as e:
            # Deletion must proceed even if the stored text cannot be read; the
            # cost is a stale contribution that the next force rebuild repairs.
            logger.warning(
                f"Could not read fact {fact_id} back before deleting it from the "
                f"index; its vocabulary contribution is left in place: {e}"
            )
            was_indexed, tokens = False, set()
        await self._delete_fact_point(fact_id, collection)
        if was_indexed:
            self._update_vocabulary([], [tokens], -1)

    async def _index_fact(
        self,
        fact: Fact,
        collection: str,
        *,
        on_commit: Callable[[], object] | None = None,
    ) -> None:
        """Index a single fact; ``on_commit`` runs once its group is stored.

        Callers register the fact's terms first. A contribution that is still
        empty here means that registration failed, which is logged and does
        not stop the write; the sparse vector then lacks unregistered terms.
        """
        # Generate searchable text
        text = generate_fact_text(fact)

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
            "embedding_text_field": "content",
        }

        points = await fragment_point(
            self.embedder,
            point_id=point_id,
            payload=payload,
            sparse=sparse,
            text=text,
            vectorize=self.global_vocab.vectorize_document,
        )
        await upsert_fragment_group(self.storage, collection, points, on_commit=on_commit)
        logger.debug(f"Indexed fact {fact.id}")

    async def _delete_fact_point(self, fact_id: UUID, collection: str) -> None:
        """Delete point for a fact."""
        await self.storage.delete_by_filter(
            collection,
            field="fact_id",
            value=str(fact_id),
        )

    async def _delete_all_fact_points(self, collection: str) -> None:
        """Delete all fact points from collection."""
        await self.storage.delete_by_filter(
            collection,
            field="type",
            value="fact",
        )

    async def close(self) -> None:
        """Close connections safely."""
        if self._storage is not None:
            await self._storage.close()
            self._storage = None
        if self._embedder is not None:
            await self._embedder.close()
            self._embedder = None
