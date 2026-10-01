"""Reversible dense-index migration with immutable physical collection targets.

    async with embedding_collection_lock(storage, logical_name):
        generation = await ensure_embedding_collection(
            storage, logical_name, embedder, resolve_text, lock_held=True,
        )
        # Keep this lock through writes; never send vectors to the discovery alias.
        await storage.upsert_batch(generation.physical_name, points)

All cooperating writers must share the lock directory. Legacy binaries and
writers on other machines do not participate in this local locking protocol.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4
from weakref import WeakKeyDictionary

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    WriteOrdering,
)

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import settings
from vector_core.storage.embedding_sources import resolve_shared_embedding_text as _resolve_shared
from vector_core.storage.qdrant import QdrantStorage
from vector_core.utils.locking import async_file_lock

EMBEDDING_TEXT_KEY = "embedding_text"
GENERATION_METADATA_KEY = "embedding_generation"
TextResolver = Callable[[dict[str, Any]], Awaitable[str | None]]
CandidateFinalizer = Callable[[str], Awaitable[None]]


class EmbeddingMigrationError(RuntimeError):
    """An index cannot safely be bound to the requested embedding space."""


@dataclass(frozen=True)
class CollectionGeneration:
    logical_name: str
    physical_name: str
    identity: EmbeddingIdentity
    migrated: bool


_bindings: WeakKeyDictionary[EmbeddingClient, dict[tuple[str, str], CollectionGeneration]] = (
    WeakKeyDictionary()
)
_held_locks: ContextVar[tuple[tuple[object, str], ...]] = ContextVar("embedding_locks", default=())


def _alias_name(logical_name: str) -> str:
    return f"{logical_name}__active"


def _storage_scope(url: str) -> str:
    parsed = urlsplit(url.rstrip("/"))
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "loopback"
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{scheme}://{host}:{port}{parsed.path.rstrip('/')}"


@asynccontextmanager
async def embedding_collection_lock(
    storage: QdrantStorage, logical_name: str
) -> AsyncIterator[None]:
    """Serialize migration and complete write operations for a logical index."""
    scope = hashlib.sha256(f"{_storage_scope(storage.url)}\0{logical_name}".encode()).hexdigest()
    owner = (asyncio.current_task(), scope)
    if owner in _held_locks.get():
        yield
        return
    async with async_file_lock(
        scope,
        timeout=max(settings.file_lock_timeout, 3600.0),
        lock_dir=settings.cache_dir / "embedding-migration-locks",
    ):
        token = _held_locks.set((*_held_locks.get(), owner))
        try:
            yield
        finally:
            _held_locks.reset(token)


async def active_embedding_collection(storage: QdrantStorage, logical_name: str) -> str | None:
    """Read the discovery pointer, falling back to an untouched legacy collection."""
    client = await storage.get_client()
    aliases = await client.get_aliases()
    for alias in aliases.aliases:
        if alias.alias_name == _alias_name(logical_name):
            return alias.collection_name
    return logical_name if await storage.collection_exists(logical_name) else None


def _manifest_identity(metadata: dict[str, Any] | None) -> EmbeddingIdentity | None:
    manifest = (metadata or {}).get(GENERATION_METADATA_KEY)
    if not isinstance(manifest, dict) or manifest.get("state") != "ready":
        return None
    try:
        return EmbeddingIdentity.from_dict(manifest["identity"])
    except (KeyError, TypeError, ValueError):
        return None


async def _verify_generation(storage: QdrantStorage, generation: CollectionGeneration) -> None:
    client = await storage.get_client()
    info = await client.get_collection(generation.physical_name)
    vectors = info.config.params.vectors
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    if dense is None or dense.size != generation.identity.dimension:
        raise EmbeddingMigrationError(
            "Collection generation has an incompatible dense vector schema"
        )
    metadata = await storage.get_metadata(generation.physical_name)
    manifest = (metadata or {}).get(GENERATION_METADATA_KEY)
    if (
        _manifest_identity(metadata) != generation.identity
        or not isinstance(manifest, dict)
        or manifest.get("logical_name") != generation.logical_name
    ):
        raise EmbeddingMigrationError("Collection generation is incomplete or has invalid identity")


async def _write_manifest(
    storage: QdrantStorage,
    generation: CollectionGeneration,
    source: str | None,
    metadata: dict[str, Any],
    state: str,
) -> None:
    client = await storage.get_client()
    payload = dict(metadata)
    payload.update(
        type="__metadata__",
        embedding_generation={
            "schema": 1,
            "logical_name": generation.logical_name,
            "identity": generation.identity.to_dict(),
            "source": source,
            "state": state,
        },
    )
    await client.upsert(
        generation.physical_name,
        [
            PointStruct(
                id=0,
                vector={
                    "dense": [0.0] * generation.identity.dimension,
                    "sparse": SparseVector(indices=[], values=[]),
                },
                payload=payload,
            )
        ],
        wait=True,
        ordering=WriteOrdering.STRONG,
    )


async def _copy_points(
    storage: QdrantStorage,
    source: str,
    generation: CollectionGeneration,
    embedder: EmbeddingClient,
    text_resolver: TextResolver,
    *,
    allow_skip: bool,
) -> int:
    client = await storage.get_client()
    offset = None
    copied = 0
    while True:
        records, offset = await client.scroll(
            source,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        pending = []
        texts = []
        for record in records:
            payload = dict(record.payload or {})
            if record.id == 0:
                if payload.get("type") != "__metadata__":
                    raise EmbeddingMigrationError("Reserved point ID 0 contains source data")
                continue
            if payload.get("type") == "__metadata__":
                raise EmbeddingMigrationError("Unexpected metadata point outside reserved ID 0")
            text = payload.get(EMBEDDING_TEXT_KEY)
            if text is None:
                text = await text_resolver(payload)
                payload["embedding_text_source"] = (
                    "legacy-note-metadata"
                    if payload.get("type") == "note"
                    else "legacy-reconstruction"
                )
            if text is None:
                if not allow_skip:
                    raise EmbeddingMigrationError(
                        f"Point {record.id!r} cannot be omitted without a source finalizer"
                    )
                continue
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingMigrationError(f"Point {record.id!r} has no usable embedding text")
            sparse = record.vector.get("sparse") if isinstance(record.vector, dict) else None
            if not isinstance(sparse, (SparseVector, dict)):
                raise EmbeddingMigrationError(f"Point {record.id!r} has no retained sparse vector")
            payload[EMBEDDING_TEXT_KEY] = text
            pending.append((record.id, payload, sparse))
            texts.append(text)
        embeddings = await embedder.embed_all(texts, role="document")
        points = [
            PointStruct(id=point_id, payload=payload, vector={"dense": dense, "sparse": sparse})
            for (point_id, payload, sparse), dense in zip(pending, embeddings, strict=True)
        ]
        if points:
            await client.upsert(
                generation.physical_name,
                points,
                wait=True,
                ordering=WriteOrdering.STRONG,
            )
        copied += len(points)
        if offset is None:
            return copied


async def _publish(storage: QdrantStorage, logical_name: str, physical_name: str) -> None:
    client = await storage.get_client()
    aliases = await client.get_aliases()
    alias_name = _alias_name(logical_name)
    operations: list[CreateAliasOperation | DeleteAliasOperation] = []
    if any(alias.alias_name == alias_name for alias in aliases.aliases):
        operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name)))
    operations.append(
        CreateAliasOperation(
            create_alias=CreateAlias(
                collection_name=physical_name,
                alias_name=alias_name,
            )
        )
    )
    # Cancellation must not release the operation lock while a publication can
    # still commit in the background. After completion, propagate cancellation.
    publishing = asyncio.create_task(
        client.update_collection_aliases(change_aliases_operations=operations)
    )
    cancelled = False
    while True:
        try:
            await asyncio.shield(publishing)
            break
        except asyncio.CancelledError:
            if publishing.done():
                publishing.result()
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


async def _copy_indexes(
    storage: QdrantStorage,
    source: str | None,
    target: str,
    payload_indexes: Sequence[tuple[str, PayloadSchemaType]],
) -> None:
    indexes: dict[str, Any] = dict(payload_indexes)
    client = await storage.get_client()
    if source:
        info = await client.get_collection(source)
        for key, schema in info.payload_schema.items():
            indexes[key] = schema.params or schema.data_type
    for key, schema in indexes.items():
        await client.create_payload_index(
            collection_name=target,
            field_name=key,
            field_schema=schema,
            wait=True,
        )


async def ensure_embedding_collection(
    storage: QdrantStorage,
    logical_name: str,
    embedder: EmbeddingClient,
    text_resolver: TextResolver,
    *,
    payload_indexes: Sequence[tuple[str, PayloadSchemaType]] = (),
    lock_held: bool = False,
    finalize_candidate: CandidateFinalizer | None = None,
) -> CollectionGeneration:
    """Bind to a complete compatible generation, rebuilding reversibly if needed.

    Unknown legacy identity always triggers re-embedding. All point types are
    copied; a resolver must fail rather than fabricate unavailable source text.
    Returning None omits a point for an optional source-backed finalizer, which
    must raise on any incomplete rebuild. Failed candidates are retained and
    never published. Retrying starts a fresh candidate, so deletes during an
    interrupted build cannot strand stale points.

    A client already bound before another client publishes a replacement fails
    closed on its next operation. Construct a new client for an intentional
    configuration change. The discovery alias is never a vector I/O target.
    """
    if not lock_held:
        async with embedding_collection_lock(storage, logical_name):
            return await ensure_embedding_collection(
                storage,
                logical_name,
                embedder,
                text_resolver,
                payload_indexes=payload_indexes,
                lock_held=True,
                finalize_candidate=finalize_candidate,
            )

    identity = await embedder.resolve_identity()
    storage.embedding_dim = identity.dimension
    source = await active_embedding_collection(storage, logical_name)
    binding_key = (_storage_scope(storage.url), logical_name)
    bindings = _bindings.setdefault(embedder, {})
    bound = bindings.get(binding_key)
    if bound is not None:
        if source != bound.physical_name and source:
            current_metadata = await storage.get_metadata(source)
            if _manifest_identity(current_metadata) == identity:
                bound = CollectionGeneration(logical_name, source, identity, False)
        if source != bound.physical_name or identity != bound.identity:
            raise EmbeddingMigrationError(
                "Embedding collection was superseded; restart the client before further operations"
            )
        await _verify_generation(storage, bound)
        if payload_indexes:
            await storage.ensure_payload_indexes(bound.physical_name, list(payload_indexes))
        bindings[binding_key] = bound
        return bound

    metadata = await storage.get_metadata(source) if source else None
    if source and _manifest_identity(metadata) == identity:
        generation = CollectionGeneration(logical_name, source, identity, False)
        await _verify_generation(storage, generation)
        if payload_indexes:
            await storage.ensure_payload_indexes(source, list(payload_indexes))
        bindings[binding_key] = generation
        return generation

    # Never reuse a previously built target: A -> B -> A must start from B, and
    # an interrupted build's source may have changed before a retry.
    lineage = hashlib.sha256(
        f"{logical_name}\0{source}\0{identity.fingerprint}".encode()
    ).hexdigest()
    target = f"vcgen_{lineage[:24]}_{uuid4().hex}"
    generation = CollectionGeneration(logical_name, target, identity, source is not None)
    await storage.create_collection(target, dense_dim=identity.dimension)
    await _write_manifest(storage, generation, source, metadata or {}, "building")
    client = await storage.get_client()
    try:
        copied = (
            await _copy_points(
                storage,
                source,
                generation,
                embedder,
                text_resolver,
                allow_skip=finalize_candidate is not None,
            )
            if source
            else 0
        )
        if finalize_candidate is not None:
            await finalize_candidate(target)
        else:
            count = await client.count(target, exact=True)
            if count.count != copied + 1:
                raise EmbeddingMigrationError(
                    "Candidate point count does not match the completed copy"
                )
        await _copy_indexes(storage, source, target, payload_indexes)
        candidate_metadata = await storage.get_metadata(target)
        await _write_manifest(storage, generation, source, candidate_metadata or {}, "ready")
        await _verify_generation(storage, generation)
        await _publish(storage, logical_name, target)
    except Exception as error:
        raise EmbeddingMigrationError(
            f"Embedding migration did not complete; source {source!r} is retained and "
            f"candidate {target!r} requires an active-pointer check before reuse: {error}"
        ) from error
    bindings[binding_key] = generation
    return generation


async def resolve_shared_embedding_text(
    payload: dict[str, Any],
    *,
    glossary_store: Any = None,
) -> str:
    """Resolve retained fact text or verified source-backed shared content."""
    try:
        return await _resolve_shared(payload, glossary_store=glossary_store)
    except Exception as error:
        raise EmbeddingMigrationError(str(error)) from error
