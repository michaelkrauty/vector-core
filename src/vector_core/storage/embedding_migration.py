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
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4
from weakref import WeakKeyDictionary

from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointsList,
    PointStruct,
    Range,
    SparseVector,
    WriteOrdering,
)

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import settings
from vector_core.storage.embedding_fragments import (
    FRAGMENT_KEY,
    ChildPayload,
    Vectorize,
    fragment_id,
    fragment_marker,
    fragment_point,
    source_hash,
)
from vector_core.storage.embedding_sources import resolve_shared_embedding_text as _resolve_shared
from vector_core.storage.embedding_sources import stored_embedding_text
from vector_core.storage.qdrant import QdrantStorage
from vector_core.utils.locking import async_file_lock

EMBEDDING_TEXT_KEY = "embedding_text"
EMBEDDING_TEXT_FIELD_KEY = "embedding_text_field"
GENERATION_METADATA_KEY = "embedding_generation"
_MAX_UPSERT_BYTES = 30 * 1024 * 1024
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


async def _copy_points(  # noqa: PLR0912 - source-preserving migration rejects ambiguous lineage
    storage: QdrantStorage,
    source: str,
    generation: CollectionGeneration,
    embedder: EmbeddingClient,
    text_resolver: TextResolver,
    *,
    allow_skip: bool,
    child_payload: ChildPayload | None = None,
    vectorize: Vectorize | None = None,
) -> int:
    client = await storage.get_client()
    offset = None
    copied = 0
    while True:
        records, offset = await client.scroll(
            source,
            # A source can retain 24 MB of raw text. Do not accumulate a page
            # of expanded sources or duplicate their raw payloads in children.
            limit=1,
            offset=offset,
            scroll_filter=Filter(
                must_not=[
                    FieldCondition(key=f"{FRAGMENT_KEY}.index", range=Range(gt=0)),
                ]
            ),
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            payload = dict(record.payload or {})
            if record.id == 0:
                if payload.get("type") != "__metadata__":
                    raise EmbeddingMigrationError("Reserved point ID 0 contains source data")
                continue
            if payload.get("type") == "__metadata__":
                raise EmbeddingMigrationError("Unexpected metadata point outside reserved ID 0")
            marker = fragment_marker(payload)
            if marker is not None:
                await _verify_fragment_group(
                    client,
                    source,
                    cast(int | str, record.id),
                    payload,
                    marker,
                )
            text = await _prepare_copy_payload(payload, text_resolver)
            if text is None:
                if not allow_skip:
                    raise EmbeddingMigrationError(
                        f"Point {record.id!r} cannot be omitted without a source finalizer"
                    )
                continue
            sparse = record.vector.get("sparse") if isinstance(record.vector, dict) else None
            if not isinstance(sparse, (SparseVector, dict)):
                raise EmbeddingMigrationError(f"Point {record.id!r} has no retained sparse vector")
            points = await fragment_point(
                embedder,
                point_id=cast(int | str, record.id),
                payload=payload,
                sparse=sparse,
                text=text,
                child_payload=child_payload,
                vectorize=vectorize,
            )
            for start in range(1, len(points), 128):
                existing = await client.retrieve(
                    source,
                    ids=[point.id for point in points[start : start + 128]],
                    with_payload=[FRAGMENT_KEY],
                    with_vectors=False,
                )
                for collision in existing:
                    old_marker = fragment_marker(collision.payload or {})
                    if old_marker is None or old_marker["index"] == 0:
                        raise EmbeddingMigrationError(
                            f"Derived fragment ID {collision.id!r} collides with source data"
                        )
                    if old_marker["parent_id"] != record.id:
                        raise EmbeddingMigrationError("Derived fragment IDs collide across sources")
            await _upsert_copy_points(client, generation.physical_name, points)
            copied += len(points)
        if offset is None:
            await _verify_fragment_orphans(client, source)
            return copied


async def _verify_fragment_group(
    client: Any,
    source: str,
    point_id: int | str,
    payload: dict[str, Any],
    marker: dict[str, Any],
) -> None:
    """Authenticate a whole group against one loaded and hashed canonical source."""
    raw = stored_embedding_text(payload)
    if (
        marker["index"] != 0
        or marker["parent_id"] != point_id
        or raw is None
        or marker["source_hash"] != source_hash(raw)
        or marker["end"] > len(raw)
    ):
        raise EmbeddingMigrationError("Canonical fragment source does not match lineage")
    intervals: dict[int, tuple[int, int]] = {0: (0, marker["end"])}
    offset = None
    while True:
        children, offset = await client.scroll(
            source,
            limit=128,
            offset=offset,
            with_vectors=False,
            with_payload=[FRAGMENT_KEY, "embedding_text", "embedding_text_field", "content"],
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=f"{FRAGMENT_KEY}.parent_id", match=MatchValue(value=point_id)
                    ),
                    FieldCondition(key=f"{FRAGMENT_KEY}.index", range=Range(gt=0)),
                ]
            ),
        )
        for child in children:
            child_payload = child.payload or {}
            child_marker = fragment_marker(child_payload)
            if child_marker is None or (
                child_marker["source_hash"] != marker["source_hash"]
                or child_marker["count"] != marker["count"]
                or child_marker["index"] in intervals
                or child_marker["end"] > len(raw)
                or stored_embedding_text(child_payload)
                != raw[child_marker["start"] : child_marker["end"]]
                or child.id
                != fragment_id(
                    point_id,
                    marker["source_hash"],
                    child_marker["start"],
                    child_marker["end"],
                    child_marker["index"],
                )
            ):
                raise EmbeddingMigrationError(
                    "Derived embedding fragment has invalid parent lineage"
                )
            intervals[child_marker["index"]] = (child_marker["start"], child_marker["end"])
        if offset is None:
            break
    if len(intervals) != marker["count"]:
        raise EmbeddingMigrationError("Embedding fragment group is incomplete")
    cursor = 0
    for index in range(marker["count"]):
        start, end = intervals[index]
        if start != cursor:
            raise EmbeddingMigrationError(
                "Embedding fragment intervals overlap or leave source uncovered"
            )
        cursor = end
    if cursor != len(raw):
        raise EmbeddingMigrationError("Embedding fragment group leaves source uncovered")


async def _verify_fragment_orphans(client: Any, source: str) -> None:
    """Detect orphan groups with small marker-only lookups and bounded caching."""
    parents: OrderedDict[int | str, dict[str, Any]] = OrderedDict()
    offset = None
    while True:
        records, offset = await client.scroll(
            source,
            limit=128,
            offset=offset,
            with_vectors=False,
            with_payload=[FRAGMENT_KEY],
            scroll_filter=Filter(
                must=[
                    FieldCondition(key=f"{FRAGMENT_KEY}.index", range=Range(gt=0)),
                ]
            ),
        )
        for record in records:
            marker = fragment_marker(record.payload or {})
            if marker is None:
                raise EmbeddingMigrationError("Malformed derived embedding fragment")
            parent_id = marker["parent_id"]
            parent_marker = parents.get(parent_id)
            if parent_marker is None:
                found = await client.retrieve(
                    source,
                    ids=[parent_id],
                    with_vectors=False,
                    with_payload=[FRAGMENT_KEY],
                )
                parent_marker = fragment_marker(found[0].payload or {}) if len(found) == 1 else None
                if (
                    parent_marker is None
                    or parent_marker["index"] != 0
                    or (parent_marker["parent_id"] != parent_id)
                ):
                    raise EmbeddingMigrationError("Orphaned derived embedding fragment")
                parents[parent_id] = parent_marker
                if len(parents) > 128:
                    parents.popitem(last=False)
            parents.move_to_end(parent_id)
            if marker["source_hash"] != parent_marker["source_hash"] or (
                marker["count"] != parent_marker["count"]
            ):
                raise EmbeddingMigrationError(
                    "Derived embedding fragment has invalid parent lineage"
                )
        if offset is None:
            return


async def _verify_candidate_fragments(client: Any, target: str) -> None:
    """A finalizer may replace groups, but every retained group must be complete."""
    offset = None
    while True:
        records, offset = await client.scroll(
            target,
            limit=1,
            offset=offset,
            scroll_filter=Filter(
                must_not=[
                    FieldCondition(key=f"{FRAGMENT_KEY}.index", range=Range(gt=0)),
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            marker = fragment_marker(payload)
            if marker is None:
                continue
            await _verify_fragment_group(
                client, target, cast(int | str, record.id), payload, marker
            )
        if offset is None:
            await _verify_fragment_orphans(client, target)
            return


async def _prepare_copy_payload(payload: dict[str, Any], text_resolver: TextResolver) -> str | None:
    text = stored_embedding_text(payload)
    if text is not None:
        return text
    text = await text_resolver(payload)
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        raise EmbeddingMigrationError("Source point has no usable embedding text")
    payload["embedding_text_source"] = (
        "legacy-note-metadata" if payload.get("type") == "note" else "legacy-reconstruction"
    )
    # Preserve exact source input once. A retained document chunk can be larger
    # than the embedding context; duplicating it can exceed the request limit.
    if payload.get("content") == text:
        payload[EMBEDDING_TEXT_FIELD_KEY] = "content"
    else:
        payload[EMBEDDING_TEXT_KEY] = text
    return text


async def _upsert_copy_points(client: Any, collection: str, points: list[PointStruct]) -> None:
    """Bound actual UTF-8 REST bodies, retaining server-confirmed write ordering."""
    envelope_bytes = len(
        PointsList(points=[])
        .model_dump_json(
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        )
        .encode()
    )
    batch: list[PointStruct] = []
    batch_bytes = envelope_bytes
    for point in points:
        # Match qdrant-client's REST serializer, including escaping, omitted
        # fields, Unicode and float representation; payload character counts
        # and fixed point-count batches cannot bound request size.
        point_bytes = len(
            point.model_dump_json(
                by_alias=True,
                exclude_none=True,
                exclude_unset=True,
            ).encode()
        )
        if point_bytes + envelope_bytes > _MAX_UPSERT_BYTES:
            raise EmbeddingMigrationError(
                f"Point {point.id!r} needs {point_bytes + envelope_bytes} serialized bytes, "
                f"exceeding the {_MAX_UPSERT_BYTES}-byte migration request budget"
            )
        if batch and batch_bytes + 1 + point_bytes > _MAX_UPSERT_BYTES:
            await client.upsert(collection, batch, wait=True, ordering=WriteOrdering.STRONG)
            batch = []
            batch_bytes = envelope_bytes
        batch_bytes += point_bytes + bool(batch)
        batch.append(point)
    if batch:
        await client.upsert(collection, batch, wait=True, ordering=WriteOrdering.STRONG)


def _exception_description(error: Exception) -> str:
    """Retain wrapped transport types when their ordinary message is empty."""
    descriptions = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        descriptions.append(f"{type(current).__name__}: {current!s}".rstrip(": "))
        cause = current.__cause__ or current.__context__
        if cause is None and isinstance(current, ResponseHandlingException):
            cause = current.source
        current = cause
    return " caused by ".join(descriptions)


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
    child_payload: ChildPayload | None = None,
    vectorize: Vectorize | None = None,
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
                child_payload=child_payload,
                vectorize=vectorize,
            )

    identity = await embedder.resolve_identity()
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
                child_payload=child_payload,
                vectorize=vectorize,
            )
            if source
            else 0
        )
        # Check every generated point even when a finalizer will add source-backed
        # groups. A finalizer must never mask an incomplete transport write.
        count = await client.count(target, exact=True)
        if count.count != copied + 1:
            raise EmbeddingMigrationError("Candidate point count does not match the completed copy")
        if finalize_candidate is not None:
            await finalize_candidate(target)
            await _verify_candidate_fragments(client, target)
        await _copy_indexes(storage, source, target, payload_indexes)
        candidate_metadata = await storage.get_metadata(target)
        await _write_manifest(storage, generation, source, candidate_metadata or {}, "ready")
        await _verify_generation(storage, generation)
        await _publish(storage, logical_name, target)
    except Exception as error:
        raise EmbeddingMigrationError(
            f"Embedding migration did not complete; source {source!r} is retained and "
            f"candidate {target!r} requires an active-pointer check before reuse: "
            f"{_exception_description(error)}"
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
