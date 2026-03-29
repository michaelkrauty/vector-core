"""Fixtures for integration tests."""

import hashlib
from uuid import uuid4

import pytest

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.qdrant import QdrantStorage


def generate_uuid_point_id(key: str) -> str:
    """Generate deterministic UUID from key for Qdrant compatibility."""
    # Create a UUID from the SHA256 hash (use first 32 hex chars for UUID)
    hash_val = hashlib.sha256(key.encode()).hexdigest()[:32]
    # Format as UUID
    return f"{hash_val[:8]}-{hash_val[8:12]}-{hash_val[12:16]}-{hash_val[16:20]}-{hash_val[20:32]}"


# Check if Qdrant is available
def qdrant_available() -> bool:
    """Check if Qdrant is running."""
    import httpx
    try:
        response = httpx.get("http://localhost:6333/collections", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip if Qdrant not available
requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at localhost:6333"
)


@pytest.fixture
def test_collection_name():
    """Generate unique test collection name."""
    return f"test_{uuid4().hex[:12]}"


@pytest.fixture
async def qdrant_storage():
    """Create QdrantStorage instance for testing."""
    storage = QdrantStorage(
        url="http://localhost:6333",
        embedding_dim=128,  # Small dimension for fast tests
    )
    yield storage
    await storage.close()


@pytest.fixture
async def test_collection(qdrant_storage, test_collection_name):
    """Create a test collection and clean up after."""
    await qdrant_storage.create_collection(test_collection_name, dense_dim=128)
    yield test_collection_name
    # Cleanup
    try:
        await qdrant_storage.delete_collection(test_collection_name)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def sample_dense_vector():
    """Sample dense vector for testing."""
    return [0.1] * 128


@pytest.fixture
def sample_sparse_vector():
    """Sample sparse vector for testing."""
    return SparseVector(
        indices=[0, 5, 10, 15],
        values=[0.5, 0.3, 0.2, 0.1],
    )


@pytest.fixture
def sample_payload():
    """Sample payload for testing."""
    return {
        "type": "test",
        "path": "test/file.py",
        "content": "Test content for testing",
    }
