"""Tests for indexing point creation utilities."""

import uuid

import pytest
from qdrant_client.models import PointStruct

from vector_core import SparseVector
from vector_core.indexing.points import (
    create_hybrid_point,
    create_hybrid_point_with_key,
    sparse_to_qdrant,
)


@pytest.fixture
def sample_dense_vector() -> list[float]:
    """Sample dense embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_sparse_vector() -> SparseVector:
    """Sample sparse vector."""
    return SparseVector(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])


@pytest.fixture
def sample_payload() -> dict:
    """Sample payload dict."""
    return {
        "type": "chunk",
        "path": "src/example.py",
        "content": "def hello(): pass",
    }


class TestSparseToQdrant:
    """Tests for sparse_to_qdrant conversion."""

    def test_converts_sparse_vector(self, sample_sparse_vector: SparseVector) -> None:
        """sparse_to_qdrant converts SparseVector to Qdrant format."""
        result = sparse_to_qdrant(sample_sparse_vector)

        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.3, 0.2]

    def test_empty_sparse_vector(self) -> None:
        """sparse_to_qdrant handles empty vectors."""
        empty = SparseVector(indices=[], values=[])
        result = sparse_to_qdrant(empty)

        assert result.indices == []
        assert result.values == []


class TestCreateHybridPoint:
    """Tests for create_hybrid_point function."""

    def test_creates_point_with_vectors(
        self,
        sample_dense_vector: list[float],
        sample_sparse_vector: SparseVector,
        sample_payload: dict,
    ) -> None:
        """create_hybrid_point creates PointStruct with dense and sparse vectors."""
        point_id = str(uuid.uuid4())

        result = create_hybrid_point(
            point_id=point_id,
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        assert isinstance(result, PointStruct)
        assert result.id == point_id
        assert result.vector["dense"] == sample_dense_vector
        assert result.vector["sparse"].indices == sample_sparse_vector.indices
        assert result.vector["sparse"].values == sample_sparse_vector.values
        assert result.payload == sample_payload

    def test_payload_preserved(
        self,
        sample_dense_vector: list[float],
        sample_sparse_vector: SparseVector,
    ) -> None:
        """create_hybrid_point preserves all payload fields."""
        payload = {
            "type": "document",
            "path": "/path/to/doc.pdf",
            "title": "Test Document",
            "tags": ["test", "example"],
            "nested": {"key": "value"},
        }

        result = create_hybrid_point(
            point_id=str(uuid.uuid4()),
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=payload,
        )

        assert result.payload["type"] == "document"
        assert result.payload["tags"] == ["test", "example"]
        assert result.payload["nested"]["key"] == "value"


class TestCreateHybridPointWithKey:
    """Tests for create_hybrid_point_with_key function."""

    def test_generates_deterministic_id(
        self,
        sample_dense_vector: list[float],
        sample_sparse_vector: SparseVector,
        sample_payload: dict,
    ) -> None:
        """create_hybrid_point_with_key generates deterministic UUID from key."""
        key = "chunk:abc123:0"

        result1 = create_hybrid_point_with_key(
            key=key,
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        result2 = create_hybrid_point_with_key(
            key=key,
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        # Same key produces same ID
        assert result1.id == result2.id

    def test_different_keys_different_ids(
        self,
        sample_dense_vector: list[float],
        sample_sparse_vector: SparseVector,
        sample_payload: dict,
    ) -> None:
        """create_hybrid_point_with_key produces different IDs for different keys."""
        result1 = create_hybrid_point_with_key(
            key="chunk:abc:0",
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        result2 = create_hybrid_point_with_key(
            key="chunk:abc:1",
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        assert result1.id != result2.id

    def test_returns_valid_uuid(
        self,
        sample_dense_vector: list[float],
        sample_sparse_vector: SparseVector,
        sample_payload: dict,
    ) -> None:
        """create_hybrid_point_with_key returns valid UUID string."""
        result = create_hybrid_point_with_key(
            key="test:key",
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload=sample_payload,
        )

        # Should be a valid UUID string
        parsed = uuid.UUID(result.id)
        assert str(parsed) == result.id
