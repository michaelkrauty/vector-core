"""Source-preserving dense fragments with stable lineage and bounded writes."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    SparseVector,
    WriteOrdering,
)

from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.tokenization import default_tokenize
from vector_core.settings import settings
from vector_core.storage.embedding_sources import stored_embedding_text

if TYPE_CHECKING:
    from vector_core.embeddings.client import EmbeddingClient, EmbeddingSpan
    from vector_core.storage.qdrant import QdrantStorage

FRAGMENT_KEY = "embedding_fragment"
_FRAGMENT_NAMESPACE = UUID("f5c375ee-5ff4-47b4-a383-56c4cf3a5c4c")
ChildPayload = Callable[[dict[str, Any], "EmbeddingSpan"], dict[str, Any]]
Vectorize = Callable[[str], Any]


def _qdrant_sparse(vector: Any) -> SparseVector:
    if isinstance(vector, SparseVector):
        return vector
    if isinstance(vector, dict):
        return SparseVector.model_validate(vector)
    return SparseVector(indices=vector.indices, values=vector.values)


# Retain entity identity, filters and compact presentation metadata, never bodies
# or arbitrary source fields. Custom writers can supply their own projection.
_CHILD_FIELDS = {
    "type",
    "note_id",
    "title",
    "tags",
    "category",
    "created",
    "modified",
    "created_at",
    "updated_at",
    "hash",
    "content_hash",
    "note_hash",
    "document_id",
    "filename",
    "path",
    "doc_type",
    "chunk_index",
    "section_title",
    "glossary_id",
    "domain",
    "term",
    "term_normalized",
    "aliases",
    "entry_hash",
    "fact_id",
    "subject",
    "subject_type",
    "predicate",
    "object_type",
    "subject_normalized",
    "object_normalized",
    "confidence",
    "valid_from",
    "valid_to",
    "source_types",
    "source_count",
    "has_deleted_source",
    "has_modified_source",
    "has_relocated_source",
    "embedding_text_source",
}


def source_hash(text: str) -> str:
    """Hash the complete, unformatted raw embedding source."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fragment_id(parent_id: int | str, digest: str, start: int, end: int, index: int) -> str:
    """A namespaced UUID with type-preserving parent identity and source range."""
    name = json.dumps([parent_id, digest, start, end, index], ensure_ascii=False)
    return str(uuid5(_FRAGMENT_NAMESPACE, name))


def fragment_marker(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Validate marker structure; a malformed claimed lineage is never source data."""
    if FRAGMENT_KEY not in payload:
        return None
    marker = payload[FRAGMENT_KEY]
    if not isinstance(marker, dict):
        raise ValueError("Malformed embedding fragment marker")
    integers = ("schema", "start", "end", "index", "count")
    if any(type(marker.get(key)) is not int for key in integers):
        raise ValueError("Malformed embedding fragment range")
    digest = marker.get("source_hash")
    parent_id = marker.get("parent_id")
    if (
        marker["schema"] != 1
        or type(parent_id) not in (int, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or marker["start"] < 0
        or marker["end"] <= marker["start"]
        or not 0 <= marker["index"] < marker["count"]
        or (marker["index"] == 0 and marker["start"] != 0)
    ):
        raise ValueError("Invalid embedding fragment lineage")
    return marker


def is_derived_fragment(payload: dict[str, Any]) -> bool:
    marker = fragment_marker(payload)
    return marker is not None and marker["index"] > 0


def fragment_text(payload: dict[str, Any]) -> str:
    """Return the actual dense snippet, including canonical index zero's range."""
    text = stored_embedding_text(payload)
    if text is None:
        text = payload.get("content")
    if not isinstance(text, str) or not text:
        raise ValueError("Fragment has no retained source text")
    marker = fragment_marker(payload)
    if marker is not None and marker["index"] == 0:
        if source_hash(text) != marker["source_hash"] or marker["end"] > len(text):
            raise ValueError("Canonical fragment source does not match lineage")
        return text[marker["start"] : marker["end"]]
    return text


def knowledge_child_payload(payload: dict[str, Any], span: EmbeddingSpan) -> dict[str, Any]:
    child = {key: value for key, value in payload.items() if key in _CHILD_FIELDS}
    if type(payload.get("char_start")) is int:
        child["char_start"] = payload["char_start"] + span.start
        child["char_end"] = payload["char_start"] + span.end
    return child


def _existing_global_vectorize(text: str) -> SparseVector:
    """Read existing global indices without registrations or SQLite initialization."""
    if GlobalVocabulary._instance is not None:
        vector = GlobalVocabulary._instance.vectorize_document(text)
        return SparseVector(indices=vector.indices, values=vector.values)
    counts = Counter(default_tokenize(text))
    if not counts:
        return SparseVector(indices=[], values=[])
    found: dict[str, int] = {}
    path = (settings.cache_dir / "global_vocabulary.db").resolve()
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
        tokens = list(counts)
        for start in range(0, len(tokens), 900):
            batch = tokens[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            found.update(
                connection.execute(
                    f"SELECT token, idx FROM vocabulary WHERE token IN ({placeholders})",  # noqa: S608
                    batch,
                )
            )
    pairs = sorted((index, 1 + math.log(counts[token])) for token, index in found.items())
    return SparseVector(indices=[index for index, _ in pairs], values=[value for _, value in pairs])


async def fragment_point(  # noqa: PLR0912 - exact source preservation and adapter validation
    embedder: EmbeddingClient,
    *,
    point_id: int | str,
    payload: dict[str, Any],
    sparse: Any,
    text: str | None = None,
    child_payload: ChildPayload | None = None,
    vectorize: Vectorize | None = None,
) -> list[PointStruct]:
    """Expand one source while retaining its original ID, raw fields and sparse vector."""
    canonical = dict(payload)
    retained = stored_embedding_text(canonical)
    if text is None:
        text = retained
    elif retained is not None and retained != text:
        raise ValueError("Explicit embedding input differs from retained source")
    if not isinstance(text, str) or not text:
        raise ValueError("Source point has no usable embedding text")
    if retained is None:
        if canonical.get("content") == text:
            canonical["embedding_text_field"] = "content"
        else:
            canonical["embedding_text"] = text
    spans = embedder.split_text(text, role="document")
    cursor = 0
    for span in spans:
        if (
            span.start != cursor
            or span.end <= span.start
            or span.text != text[span.start : span.end]
        ):
            raise ValueError("Embedding spans do not partition exact source")
        cursor = span.end
    if cursor != len(text):
        raise ValueError("Embedding spans leave source uncovered")
    embeddings = await embedder.embed_all([span.text for span in spans], role="document")
    digest = source_hash(text)
    adapter = child_payload or knowledge_child_payload
    vectorizer = vectorize or _existing_global_vectorize
    points = []
    for index, (span, dense) in enumerate(zip(spans, embeddings, strict=True)):
        if index:
            projected = dict(adapter(payload, span))
            # Never inherit an authoritative whole-source reference from a child adapter.
            for key in ("embedding_text", "embedding_text_field", "content", "body", "raw"):
                projected.pop(key, None)
            reference = payload.get("embedding_text_field")
            if isinstance(reference, str):
                projected.pop(reference, None)
            projected["content"] = span.text
            projected["embedding_text_field"] = "content"
            child_sparse = _qdrant_sparse(vectorizer(span.text))
        else:
            projected = canonical
            child_sparse = _qdrant_sparse(sparse)
        projected[FRAGMENT_KEY] = {
            "schema": 1,
            "parent_id": point_id,
            "source_hash": digest,
            "start": span.start,
            "end": span.end,
            "index": index,
            "count": len(spans),
        }
        points.append(
            PointStruct(
                id=point_id
                if index == 0
                else fragment_id(point_id, digest, span.start, span.end, index),
                payload=projected,
                vector={"dense": dense, "sparse": child_sparse},
            )
        )
    return points


async def upsert_fragment_group(  # noqa: PLR0912, PLR0915 - validate before any destructive write
    storage: QdrantStorage, collection: str, points: list[PointStruct]
) -> None:
    """Write a complete group before retiring stale children, under the caller's lock."""
    from vector_core.storage.embedding_migration import _upsert_copy_points  # noqa: PLC0415

    if not points:
        raise ValueError("Cannot upsert an empty fragment group")
    marker = fragment_marker(points[0].payload or {})
    if marker is None and len(points) == 1:
        payload = dict(points[0].payload or {})
        raw = stored_embedding_text(payload)
        if raw is None:
            raw = payload.get("content")
            if isinstance(raw, str) and raw:
                payload["embedding_text_field"] = "content"
        if not isinstance(raw, str) or not raw:
            raise ValueError("Singleton source has no retained embedding input")
        parent_id = str(points[0].id) if isinstance(points[0].id, UUID) else points[0].id
        payload[FRAGMENT_KEY] = {
            "schema": 1,
            "parent_id": parent_id,
            "source_hash": source_hash(raw),
            "start": 0,
            "end": len(raw),
            "index": 0,
            "count": 1,
        }
        points = [points[0].model_copy(update={"payload": payload})]
        marker = payload[FRAGMENT_KEY]
    canonical_id = str(points[0].id) if isinstance(points[0].id, UUID) else points[0].id
    if marker is None or marker["index"] != 0 or marker["parent_id"] != canonical_id:
        raise ValueError("Fragment group has no canonical source")
    if marker["count"] != len(points):
        raise ValueError("Incomplete fragment group")
    parent_id = marker["parent_id"]
    keep_ids = {point.id for point in points}
    if len(keep_ids) != len(points):
        raise ValueError("Fragment IDs collide within source group")
    raw = stored_embedding_text(points[0].payload or {})
    if raw is None or source_hash(raw) != marker["source_hash"]:
        raise ValueError("Canonical source does not match fragment hash")
    cursor = 0
    for index, point in enumerate(points):
        current = fragment_marker(point.payload or {})
        if (
            current is None
            or current["parent_id"] != parent_id
            or current["source_hash"] != marker["source_hash"]
            or current["count"] != len(points)
            or current["index"] != index
            or current["start"] != cursor
            or current["end"] > len(raw)
            or (
                index
                and (
                    point.id
                    != fragment_id(
                        parent_id, marker["source_hash"], current["start"], current["end"], index
                    )
                    or stored_embedding_text(point.payload or {})
                    != raw[current["start"] : current["end"]]
                )
            )
        ):
            raise ValueError("Invalid fragment group lineage or source coverage")
        cursor = current["end"]
    if cursor != len(raw):
        raise ValueError("Incomplete fragment group source coverage")
    client = await storage.get_client()
    # UUID namespaces make collisions unlikely, not impossible; reject canonical
    # source IDs before a write can overwrite unrelated user data.
    for start in range(0, len(points), 128):
        existing = await client.retrieve(
            collection,
            ids=[point.id for point in points[start : start + 128]],
            with_payload=True,
        )
        for record in existing:
            existing_marker = fragment_marker(record.payload or {})
            if record.id == parent_id:
                if existing_marker is not None and (
                    existing_marker["index"] != 0 or existing_marker["parent_id"] != parent_id
                ):
                    raise ValueError("Canonical source ID collides with derived fragment")
            elif (
                existing_marker is None
                or existing_marker["index"] == 0
                or (existing_marker["parent_id"] != parent_id)
            ):
                raise ValueError("Derived fragment ID collides with an existing source")
    await _upsert_copy_points(client, collection, points)
    stale_filter = Filter(
        must=[
            FieldCondition(key=f"{FRAGMENT_KEY}.schema", match=MatchValue(value=1)),
            FieldCondition(key=f"{FRAGMENT_KEY}.parent_id", match=MatchValue(value=parent_id)),
            FieldCondition(key=f"{FRAGMENT_KEY}.index", range=Range(gt=0)),
        ]
    )
    offset = None
    while True:
        records, offset = await client.scroll(
            collection,
            scroll_filter=stale_filter,
            offset=offset,
            limit=128,
            with_payload=False,
            with_vectors=False,
        )
        stale = [record.id for record in records if record.id not in keep_ids]
        if stale:
            await client.delete(
                collection,
                points_selector=PointIdsList(points=stale),
                wait=True,
                ordering=WriteOrdering.STRONG,
            )
        if offset is None:
            break
