"""Hybrid point creation utilities for Qdrant indexers.

Provides shared functionality for creating Qdrant points with dense + sparse
vectors, used by MCP server indexers (mcp-notes, mcp-docs, mcp-codesearch).
"""

from typing import Any

from qdrant_client.models import PointStruct
from qdrant_client.models import SparseVector as QdrantSparseVector

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage import generate_point_id


def create_hybrid_point(
    point_id: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    payload: dict[str, Any],
) -> PointStruct:
    """
    Create a Qdrant point with both dense and sparse vectors.

    This is the standard pattern for hybrid search across all MCP indexers:
    - Dense vectors from embeddings (semantic similarity)
    - Sparse vectors from GlobalVocabulary (keyword matching)

    Args:
        point_id: Unique point identifier (UUID string)
        dense_vector: Dense embedding vector from EmbeddingClient
        sparse_vector: Sparse vector from GlobalVocabulary.vectorize_document()
        payload: Metadata payload dict

    Returns:
        PointStruct ready for upsert to Qdrant
    """
    return PointStruct(
        id=point_id,
        vector={
            "dense": dense_vector,
            "sparse": QdrantSparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
        },
        payload=payload,
    )


def create_hybrid_point_with_key(
    key: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    payload: dict[str, Any],
) -> PointStruct:
    """
    Create a hybrid point with deterministic ID from a key string.

    Combines generate_point_id() with create_hybrid_point() for convenience.
    Use when you have a deterministic key (e.g., "chunk:uuid:0") instead of
    a pre-generated point ID.

    Args:
        key: Deterministic key string (e.g., "file:/path/to/file.py")
        dense_vector: Dense embedding vector
        sparse_vector: Sparse vector from GlobalVocabulary
        payload: Metadata payload dict

    Returns:
        PointStruct with deterministic UUID based on key
    """
    point_id = generate_point_id(key)
    return create_hybrid_point(point_id, dense_vector, sparse_vector, payload)


def sparse_to_qdrant(sparse: SparseVector) -> QdrantSparseVector:
    """
    Convert vector-core SparseVector to Qdrant's SparseVector format.

    Args:
        sparse: SparseVector from GlobalVocabulary

    Returns:
        QdrantSparseVector for use in PointStruct
    """
    return QdrantSparseVector(
        indices=sparse.indices,
        values=sparse.values,
    )
