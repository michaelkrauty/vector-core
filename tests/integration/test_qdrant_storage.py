"""Integration tests for QdrantStorage with real Qdrant."""

from uuid import uuid4

import pytest

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.qdrant import (
    QdrantConnectionError,
    QdrantStorage,
    generate_collection_name,
    generate_point_id,
)

from .conftest import generate_uuid_point_id, requires_qdrant


@requires_qdrant
class TestQdrantStorageConnection:
    """Tests for Qdrant connection handling."""

    async def test_connection(self, qdrant_storage):
        """Can connect to Qdrant."""
        client = await qdrant_storage.get_client()
        assert client is not None

    async def test_connection_cached(self, qdrant_storage):
        """Client is cached."""
        client1 = await qdrant_storage.get_client()
        client2 = await qdrant_storage.get_client()
        assert client1 is client2

    async def test_close(self, qdrant_storage):
        """Close clears client."""
        await qdrant_storage.get_client()
        await qdrant_storage.close()
        assert qdrant_storage._client is None

    async def test_connection_error(self):
        """Connection error on invalid URL."""
        storage = QdrantStorage(url="http://localhost:9999")
        with pytest.raises(QdrantConnectionError):
            await storage.collection_exists("test")
        await storage.close()


@requires_qdrant
class TestCollectionManagement:
    """Tests for collection CRUD operations."""

    async def test_create_collection(self, qdrant_storage, test_collection_name):
        """Create a collection."""
        await qdrant_storage.create_collection(test_collection_name, dense_dim=128)

        exists = await qdrant_storage.collection_exists(test_collection_name)
        assert exists is True

        # Cleanup
        await qdrant_storage.delete_collection(test_collection_name)

    async def test_delete_collection(self, qdrant_storage, test_collection_name):
        """Delete a collection."""
        await qdrant_storage.create_collection(test_collection_name)
        await qdrant_storage.delete_collection(test_collection_name)

        exists = await qdrant_storage.collection_exists(test_collection_name)
        assert exists is False

    async def test_collection_exists_false(self, qdrant_storage):
        """Non-existent collection returns False."""
        exists = await qdrant_storage.collection_exists("nonexistent_xyz123")
        assert exists is False

    async def test_list_collections(self, qdrant_storage, test_collection_name):
        """List collections."""
        await qdrant_storage.create_collection(test_collection_name)

        collections = await qdrant_storage.list_collections()
        assert test_collection_name in collections

        # Cleanup
        await qdrant_storage.delete_collection(test_collection_name)

    async def test_list_collections_with_prefix(self, qdrant_storage):
        """List collections with prefix filter."""
        name1 = f"prefix_a_{uuid4().hex[:8]}"
        name2 = f"prefix_b_{uuid4().hex[:8]}"

        await qdrant_storage.create_collection(name1)
        await qdrant_storage.create_collection(name2)

        filtered = await qdrant_storage.list_collections(prefix="prefix_a")
        assert name1 in filtered
        assert name2 not in filtered

        # Cleanup
        await qdrant_storage.delete_collection(name1)
        await qdrant_storage.delete_collection(name2)


@requires_qdrant
class TestPointOperations:
    """Tests for point CRUD operations."""

    async def test_upsert_point(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Upsert a single point."""
        point_id = generate_uuid_point_id("test:file.py")

        await qdrant_storage.upsert_point(
            test_collection,
            point_id=point_id,
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload={"type": "file", "path": "file.py"},
        )

        # Verify point exists
        points = await qdrant_storage.retrieve_points(test_collection, [point_id])
        assert len(points) == 1
        assert points[0].payload["path"] == "file.py"

    async def test_upsert_batch(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Batch upsert multiple points."""
        points = []
        for i in range(10):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"test:file{i}.py"),
                dense_vector=sample_dense_vector,
                sparse_vector=sample_sparse_vector,
                payload={"type": "file", "index": i},
            )
            points.append(point)

        await qdrant_storage.upsert_batch(test_collection, points)

        # Verify all points exist
        all_payloads = await qdrant_storage.scroll_points(test_collection)
        assert len(all_payloads) == 10

    async def test_upsert_batch_empty(self, qdrant_storage, test_collection):
        """Batch upsert with empty list."""
        await qdrant_storage.upsert_batch(test_collection, [])
        # Should not raise

    async def test_delete_by_filter(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Delete points by filter."""
        # Insert points with different types
        for i, ptype in enumerate(["file", "file", "chunk"]):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"test:{i}"),
                dense_vector=sample_dense_vector,
                sparse_vector=sample_sparse_vector,
                payload={"type": ptype, "index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Delete all "file" type points
        await qdrant_storage.delete_by_filter(test_collection, "type", "file")

        # Only chunk should remain
        payloads = await qdrant_storage.scroll_points(test_collection)
        assert len(payloads) == 1
        assert payloads[0]["type"] == "chunk"

    async def test_scroll_points(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Scroll through points."""
        points = []
        for i in range(25):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"scroll:file{i}"),
                dense_vector=sample_dense_vector,
                sparse_vector=sample_sparse_vector,
                payload={"type": "file", "index": i},
            )
            points.append(point)

        await qdrant_storage.upsert_batch(test_collection, points)

        # Scroll with limit
        payloads = await qdrant_storage.scroll_points(test_collection, limit=10)
        assert len(payloads) == 25  # Gets all despite limit per request

    async def test_scroll_with_filter(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Scroll with filter conditions."""
        from qdrant_client.models import FieldCondition, MatchValue

        # Insert mixed types
        for i, ptype in enumerate(["a", "b", "a", "b", "a"]):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"filter:{i}"),
                dense_vector=sample_dense_vector,
                sparse_vector=sample_sparse_vector,
                payload={"type": ptype, "index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Filter by type "a"
        filter_cond = [FieldCondition(key="type", match=MatchValue(value="a"))]
        payloads = await qdrant_storage.scroll_points(
            test_collection, filter_conditions=filter_cond
        )

        assert len(payloads) == 3
        assert all(p["type"] == "a" for p in payloads)

    async def test_retrieve_points(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Retrieve specific points by ID."""
        ids = [generate_uuid_point_id(f"retrieve:{i}") for i in range(3)]

        for i, point_id in enumerate(ids):
            point = qdrant_storage.create_point(
                point_id=point_id,
                dense_vector=sample_dense_vector,
                sparse_vector=sample_sparse_vector,
                payload={"index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Retrieve two specific points
        points = await qdrant_storage.retrieve_points(test_collection, ids[:2])
        assert len(points) == 2

    async def test_retrieve_with_vectors(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Retrieve points with vectors included."""
        point_id = generate_uuid_point_id("vectors:test")
        point = qdrant_storage.create_point(
            point_id=point_id,
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload={"test": True},
        )
        await qdrant_storage.upsert_batch(test_collection, [point])

        points = await qdrant_storage.retrieve_points(
            test_collection, [point_id], with_vectors=True
        )
        assert len(points) == 1
        assert points[0].vector is not None


@requires_qdrant
class TestMetadataStorage:
    """Tests for collection metadata storage."""

    async def test_store_metadata(self, qdrant_storage, test_collection):
        """Store metadata at ID=0."""
        metadata = {
            "codebase_path": "/path/to/code",
            "file_count": 42,
            "vocab_data": {"word1": 0, "word2": 1},
        }

        await qdrant_storage.store_metadata(test_collection, metadata)

        # Retrieve and verify
        stored = await qdrant_storage.get_metadata(test_collection)
        assert stored is not None
        assert stored["codebase_path"] == "/path/to/code"
        assert stored["file_count"] == 42
        assert stored["vocab_data"] == {"word1": 0, "word2": 1}
        assert "updated_at" in stored

    async def test_get_metadata_none(self, qdrant_storage, test_collection):
        """Get metadata returns None when not set."""
        result = await qdrant_storage.get_metadata(test_collection)
        assert result is None

    async def test_update_metadata(self, qdrant_storage, test_collection):
        """Update existing metadata."""
        await qdrant_storage.store_metadata(test_collection, {"version": 1})
        await qdrant_storage.store_metadata(test_collection, {"version": 2})

        stored = await qdrant_storage.get_metadata(test_collection)
        assert stored["version"] == 2


@requires_qdrant
class TestQueryOperations:
    """Tests for vector query operations."""

    async def test_query_dense(
        self, qdrant_storage, test_collection, sample_sparse_vector
    ):
        """Query using dense vectors."""
        # Insert points with different dense vectors
        vectors = [
            [1.0] + [0.0] * 127,  # First element = 1
            [0.0, 1.0] + [0.0] * 126,  # Second element = 1
            [0.0] * 128,  # All zeros
        ]

        for i, vec in enumerate(vectors):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"dense:{i}"),
                dense_vector=vec,
                sparse_vector=sample_sparse_vector,
                payload={"index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Query with vector similar to first
        query = [1.0] + [0.0] * 127
        results = await qdrant_storage.query_dense(test_collection, query, limit=2)

        assert len(results) >= 1
        # First result should be most similar (index=0)
        assert results[0].payload["index"] == 0

    async def test_query_sparse(
        self, qdrant_storage, test_collection, sample_dense_vector
    ):
        """Query using sparse vectors."""
        # Insert points with different sparse vectors
        sparse_vecs = [
            SparseVector(indices=[0, 1], values=[1.0, 0.5]),
            SparseVector(indices=[2, 3], values=[1.0, 0.5]),
            SparseVector(indices=[0], values=[0.1]),
        ]

        for i, sparse in enumerate(sparse_vecs):
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"sparse:{i}"),
                dense_vector=sample_dense_vector,
                sparse_vector=sparse,
                payload={"index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Query with vector similar to first
        query = SparseVector(indices=[0, 1], values=[1.0, 0.5])
        results = await qdrant_storage.query_sparse(test_collection, query, limit=2)

        assert len(results) >= 1
        # First result should be most similar (index=0)
        assert results[0].payload["index"] == 0

    async def test_query_with_filter(
        self, qdrant_storage, test_collection, sample_dense_vector, sample_sparse_vector
    ):
        """Query with filter conditions."""
        from qdrant_client.models import FieldCondition, MatchValue

        # Insert points with different types
        for i, ptype in enumerate(["include", "exclude"]):
            vec = [float(i)] + [0.0] * 127
            point = qdrant_storage.create_point(
                point_id=generate_uuid_point_id(f"qfilter:{i}"),
                dense_vector=vec,
                sparse_vector=sample_sparse_vector,
                payload={"type": ptype, "index": i},
            )
            await qdrant_storage.upsert_batch(test_collection, [point])

        # Query with filter
        filter_cond = [FieldCondition(key="type", match=MatchValue(value="include"))]
        results = await qdrant_storage.query_dense(
            test_collection,
            [1.0] + [0.0] * 127,
            limit=10,
            filter_conditions=filter_cond,
        )

        assert all(r.payload["type"] == "include" for r in results)


@requires_qdrant
class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_collection_name(self):
        """Generate deterministic collection name."""
        name1 = generate_collection_name("/path/to/code", prefix="test")
        name2 = generate_collection_name("/path/to/code", prefix="test")

        assert name1 == name2
        assert name1.startswith("test_")
        assert len(name1) == 17  # "test_" + 12 chars

    def test_generate_collection_name_trailing_slash(self):
        """Trailing slashes normalized."""
        name1 = generate_collection_name("/path/to/code")
        name2 = generate_collection_name("/path/to/code/")

        assert name1 == name2

    def test_generate_point_id(self):
        """Generate deterministic point ID."""
        id1 = generate_point_id("chunk:file.py:42")
        id2 = generate_point_id("chunk:file.py:42")

        assert id1 == id2
        assert len(id1) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_generate_point_id_different(self):
        """Different inputs produce different IDs."""
        id1 = generate_point_id("chunk:file.py:42")
        id2 = generate_point_id("chunk:file.py:43")

        assert id1 != id2

    def test_create_point(self, qdrant_storage, sample_dense_vector, sample_sparse_vector):
        """Create point struct."""
        point = qdrant_storage.create_point(
            point_id="test_id",
            dense_vector=sample_dense_vector,
            sparse_vector=sample_sparse_vector,
            payload={"test": True},
        )

        assert point.id == "test_id"
        assert point.payload["test"] is True
        assert "dense" in point.vector
        assert "sparse" in point.vector
