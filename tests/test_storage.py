"""Tests for QdrantStorage."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.models import (
    PointIdsList,
    PointStruct,
    ScoredPoint,
)

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.qdrant import (
    QdrantConnectionError,
    QdrantStorage,
)


class TestQdrantStorageInit:
    """Tests for QdrantStorage initialization."""

    def test_default_settings(self):
        """Storage uses default settings when not specified."""
        storage = QdrantStorage()
        assert storage.url is not None
        assert storage.embedding_dim >= 0  # 0 = auto-detect at runtime

    def test_custom_settings(self):
        """Storage accepts custom settings."""
        storage = QdrantStorage(
            url="http://custom:6333",
            api_key="secret",
            embedding_dim=384,
        )
        assert storage.url == "http://custom:6333"
        assert storage.api_key == "secret"
        assert storage.embedding_dim == 384


class TestCollectionManagement:
    """Tests for collection management methods."""

    @pytest.mark.asyncio
    async def test_collection_exists_true(self):
        """collection_exists returns True when collection exists."""
        storage = QdrantStorage()

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=mock_collections)
            mock_get_client.return_value = mock_client

            result = await storage.collection_exists("test_collection")

        assert result is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self):
        """collection_exists returns False when collection doesn't exist."""
        storage = QdrantStorage()

        mock_collections = MagicMock()
        mock_collections.collections = []

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=mock_collections)
            mock_get_client.return_value = mock_client

            result = await storage.collection_exists("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_exists_connection_error(self):
        """collection_exists raises QdrantConnectionError on connect failure."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(QdrantConnectionError):
                await storage.collection_exists("test")

    @pytest.mark.asyncio
    async def test_collection_exists_non_connection_error(self):
        """collection_exists re-raises non-connection errors (line 143)."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            # Use an exception that doesn't contain "connect" or "refused"
            mock_client.get_collections = AsyncMock(
                side_effect=ValueError("Some other error")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="Some other error"):
                await storage.collection_exists("test")

    @pytest.mark.asyncio
    async def test_create_collection(self):
        """create_collection creates hybrid collection."""
        storage = QdrantStorage(embedding_dim=384)

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.create_collection = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.create_collection("test_collection")

            mock_client.create_collection.assert_called_once()
            call_kwargs = mock_client.create_collection.call_args[1]
            assert call_kwargs["collection_name"] == "test_collection"
            assert "dense" in call_kwargs["vectors_config"]
            assert "sparse" in call_kwargs["sparse_vectors_config"]

    @pytest.mark.asyncio
    async def test_delete_collection(self):
        """delete_collection deletes the collection."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.delete_collection = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.delete_collection("test_collection")

            mock_client.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_list_collections(self):
        """list_collections returns collection names."""
        storage = QdrantStorage()

        # Create mocks with properly configured .name attribute
        mock_col1 = MagicMock()
        mock_col1.name = "vc_abc123"
        mock_col2 = MagicMock()
        mock_col2.name = "vc_def456"
        mock_col3 = MagicMock()
        mock_col3.name = "other_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_col1, mock_col2, mock_col3]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=mock_collections)
            mock_get_client.return_value = mock_client

            result = await storage.list_collections()

        assert len(result) == 3
        assert "vc_abc123" in result

    @pytest.mark.asyncio
    async def test_list_collections_with_prefix(self):
        """list_collections filters by prefix."""
        storage = QdrantStorage()

        # Create mocks with properly configured .name attribute
        mock_col1 = MagicMock()
        mock_col1.name = "vc_abc123"
        mock_col2 = MagicMock()
        mock_col2.name = "vc_def456"
        mock_col3 = MagicMock()
        mock_col3.name = "other_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_col1, mock_col2, mock_col3]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=mock_collections)
            mock_get_client.return_value = mock_client

            result = await storage.list_collections(prefix="vc_")

        assert len(result) == 2
        assert all(n.startswith("vc_") for n in result)


class TestPointOperations:
    """Tests for point operations."""

    @pytest.mark.asyncio
    async def test_upsert_point(self):
        """upsert_point upserts a single point."""
        storage = QdrantStorage()

        sparse = SparseVector(indices=[0, 1], values=[0.5, 0.5])

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.upsert_point(
                collection="test",
                point_id=12345,
                dense_vector=[0.1, 0.2, 0.3],
                sparse_vector=sparse,
                payload={"key": "value"},
            )

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            assert call_args[0][0] == "test"  # collection name

    @pytest.mark.asyncio
    async def test_upsert_batch_empty(self):
        """upsert_batch does nothing for empty input."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.upsert_batch("test", [])

            mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_batch_multiple_points(self):
        """upsert_batch upserts multiple points."""
        storage = QdrantStorage()

        points = [
            PointStruct(
                id=i,
                vector={"dense": [0.1] * 3, "sparse": {"indices": [], "values": []}},
                payload={"idx": i},
            )
            for i in range(5)
        ]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.upsert_batch("test", points)

            # Should have called upsert
            assert mock_client.upsert.called

    @pytest.mark.asyncio
    async def test_upsert_batch_retry_on_failure(self):
        """upsert_batch retries on failure with exponential backoff (lines 269-275)."""
        storage = QdrantStorage()

        points = [
            PointStruct(
                id=i,
                vector={"dense": [0.1] * 3, "sparse": {"indices": [], "values": []}},
                payload={"idx": i},
            )
            for i in range(3)
        ]

        call_count = 0

        async def mock_upsert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise Exception("Transient error")
            # Third attempt succeeds

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = mock_upsert
            mock_get_client.return_value = mock_client

            # Patch asyncio.sleep to speed up test
            with patch('asyncio.sleep', new_callable=AsyncMock):
                await storage.upsert_batch("test", points, max_retries=3)

        # Should have retried
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_batch_exhausted_retries(self):
        """upsert_batch raises after exhausting retries (lines 274-275)."""
        storage = QdrantStorage()

        points = [
            PointStruct(
                id=1,
                vector={"dense": [0.1] * 3, "sparse": {"indices": [], "values": []}},
                payload={"idx": 1},
            )
        ]

        async def mock_upsert_always_fails(*args, **kwargs):
            raise Exception("Persistent error")

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = mock_upsert_always_fails
            mock_get_client.return_value = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(Exception, match="Persistent error"):
                    await storage.upsert_batch("test", points, max_retries=3)

    @pytest.mark.asyncio
    async def test_upsert_batch_timeout_has_informative_message(self):
        """A bare TimeoutError from asyncio.timeout() must be re-raised with
        an operation-specific message, otherwise it propagates to the MCP tool
        layer as an empty "Error executing tool X: " at the client.
        """
        storage = QdrantStorage()

        points = [
            PointStruct(
                id=i,
                vector={"dense": [0.1] * 3, "sparse": {"indices": [], "values": []}},
                payload={"idx": i},
            )
            for i in range(3)
        ]

        async def slow_upsert(*args, **kwargs):
            # Sleep well past the patched timeout so asyncio.timeout fires
            await asyncio.sleep(5)

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = slow_upsert
            mock_get_client.return_value = mock_client

            with patch("vector_core.storage.qdrant.settings") as mock_settings:
                mock_settings.qdrant_operation_timeout = 0.01
                with pytest.raises(TimeoutError) as exc_info:
                    await storage.upsert_batch(
                        "test_collection", points, max_retries=1
                    )

        msg = str(exc_info.value)
        assert msg, "TimeoutError must not have an empty message"
        assert "upsert_batch" in msg
        assert "test_collection" in msg
        assert "3 points" in msg
        # Chain preserved for debuggers
        assert isinstance(exc_info.value.__cause__, TimeoutError)

    @pytest.mark.asyncio
    async def test_delete_by_filter(self):
        """delete_by_filter deletes matching points."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.delete = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.delete_by_filter("test", "file_path", "/path/to/file.py")

            mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_points(self):
        """delete_points deletes the given IDs via a PointIdsList selector."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.delete = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.delete_points("test", ["id-1", "id-2", 3])

            mock_client.delete.assert_called_once()
            args, kwargs = mock_client.delete.call_args
            assert args[0] == "test"
            selector = kwargs["points_selector"]
            assert isinstance(selector, PointIdsList)
            assert selector.points == ["id-1", "id-2", 3]

    @pytest.mark.asyncio
    async def test_delete_points_empty_is_noop(self):
        """delete_points with no IDs is a no-op and never touches the client."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            await storage.delete_points("test", [])

            mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_scroll_points(self):
        """scroll_points returns point payloads."""
        storage = QdrantStorage()

        mock_point = MagicMock()
        mock_point.payload = {"key": "value"}

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.scroll = AsyncMock(return_value=([mock_point], None))
            mock_get_client.return_value = mock_client

            result = await storage.scroll_points("test")

        assert len(result) == 1
        assert result[0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_scroll_points_pagination(self):
        """scroll_points handles pagination."""
        storage = QdrantStorage()

        mock_point1 = MagicMock()
        mock_point1.payload = {"page": 1}
        mock_point2 = MagicMock()
        mock_point2.payload = {"page": 2}

        call_count = 0

        async def mock_scroll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ([mock_point1], "next_offset")
            else:
                return ([mock_point2], None)

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.scroll = AsyncMock(side_effect=mock_scroll)
            mock_get_client.return_value = mock_client

            result = await storage.scroll_points("test")

        assert len(result) == 2
        assert call_count == 2


class TestMetadataStorage:
    """Tests for metadata storage at point ID=0."""

    @pytest.mark.asyncio
    async def test_store_metadata(self):
        """store_metadata stores at point ID=0."""
        storage = QdrantStorage(embedding_dim=384)

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_get_client.return_value = mock_client

            await storage.store_metadata("test", {"key": "value"})

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            points = call_args[0][1]
            assert points[0].id == 0  # Reserved ID for metadata

    @pytest.mark.asyncio
    async def test_get_metadata_exists(self):
        """get_metadata retrieves metadata from point ID=0."""
        storage = QdrantStorage()

        mock_point = MagicMock()
        mock_point.payload = {"type": "__metadata__", "key": "value"}

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.retrieve = AsyncMock(return_value=[mock_point])
            mock_get_client.return_value = mock_client

            result = await storage.get_metadata("test")

        assert result is not None
        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_metadata_not_exists(self):
        """get_metadata returns None when no metadata."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.retrieve = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_client

            result = await storage.get_metadata("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_metadata_json_deserialization(self):
        """get_metadata deserializes JSON values."""
        storage = QdrantStorage()

        mock_point = MagicMock()
        mock_point.payload = {
            "type": "__metadata__",
            "vocab_data": '{"hello": 1}',  # JSON string
        }

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.retrieve = AsyncMock(return_value=[mock_point])
            mock_get_client.return_value = mock_client

            result = await storage.get_metadata("test")

        # JSON should be deserialized
        assert isinstance(result["vocab_data"], dict)
        assert result["vocab_data"]["hello"] == 1


class TestQueryOperations:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_query_dense(self):
        """query_dense queries using dense vector."""
        storage = QdrantStorage()

        mock_response = MagicMock()
        mock_response.points = [
            ScoredPoint(id=1, version=1, score=0.9, payload={"key": "value"}, vector=None),
        ]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.query_points = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await storage.query_dense(
                "test",
                query_vector=[0.1, 0.2, 0.3],
                limit=10,
            )

        assert len(result) == 1
        assert result[0].score == 0.9

    @pytest.mark.asyncio
    async def test_query_sparse(self):
        """query_sparse queries using sparse vector."""
        storage = QdrantStorage()

        sparse = SparseVector(indices=[0, 1], values=[0.5, 0.5])

        mock_response = MagicMock()
        mock_response.points = [
            ScoredPoint(id=1, version=1, score=0.8, payload={"key": "value"}, vector=None),
        ]

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.query_points = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await storage.query_sparse(
                "test",
                query_vector=sparse,
                limit=10,
            )

        assert len(result) == 1
        assert result[0].score == 0.8


class TestCreatePoint:
    """Tests for create_point helper."""

    def test_create_point_structure(self):
        """create_point creates valid PointStruct."""
        storage = QdrantStorage()

        sparse = SparseVector(indices=[0, 1, 2], values=[0.3, 0.5, 0.2])

        point = storage.create_point(
            point_id=12345,
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse_vector=sparse,
            payload={"file_path": "/test.py"},
        )

        assert isinstance(point, PointStruct)
        assert point.id == 12345
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert point.payload["file_path"] == "/test.py"


class TestRetrievePoints:
    """Tests for retrieve_points method."""

    @pytest.mark.asyncio
    async def test_retrieve_points(self):
        """retrieve_points retrieves specific points."""
        storage = QdrantStorage()

        mock_point = MagicMock()
        mock_point.id = 12345
        mock_point.payload = {"key": "value"}

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.retrieve = AsyncMock(return_value=[mock_point])
            mock_get_client.return_value = mock_client

            result = await storage.retrieve_points("test", [12345])

        assert len(result) == 1
        assert result[0].id == 12345


class TestGetClient:
    """Tests for get_client public method."""

    @pytest.mark.asyncio
    async def test_get_client_returns_async_client(self):
        """get_client returns the underlying Qdrant client."""
        storage = QdrantStorage()

        with patch.object(storage, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await storage.get_client()

        assert result is mock_client


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """close cleans up the client."""
        storage = QdrantStorage()

        mock_client = AsyncMock()
        storage._client = mock_client

        await storage.close()

        mock_client.close.assert_called_once()
        assert storage._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_created(self):
        """close is safe when client not created."""
        storage = QdrantStorage()
        assert storage._client is None

        await storage.close()  # Should not raise

        assert storage._client is None


class TestClientCreation:
    """Tests for _get_client lazy client creation (lines 116-121)."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client_when_none(self):
        """_get_client creates client when _client is None (lines 116-121)."""
        storage = QdrantStorage(url="http://localhost:6333")
        assert storage._client is None

        # Call _get_client
        client = await storage._get_client()

        # Client should now be set
        assert client is not None
        assert storage._client is client

        # Clean up
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_client(self):
        """_get_client returns existing client on subsequent calls."""
        storage = QdrantStorage()

        mock_client = AsyncMock()
        storage._client = mock_client

        # Call _get_client
        result = await storage._get_client()

        # Should return the existing client
        assert result is mock_client

    @pytest.mark.asyncio
    async def test_get_client_uses_api_key_when_set(self):
        """_get_client passes api_key to client constructor."""
        storage = QdrantStorage(url="http://localhost:6333", api_key="test_key")

        with patch("vector_core.storage.qdrant.AsyncQdrantClient") as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance

            await storage._get_client()

            mock_class.assert_called_once_with(
                url="http://localhost:6333",
                api_key="test_key",
            )


class TestHealthCheck:
    """Tests for Qdrant health check functionality."""

    @pytest.mark.asyncio
    async def test_check_health_returns_true_when_healthy(self):
        """check_health returns True when Qdrant responds."""
        storage = QdrantStorage(health_check_interval=30.0)

        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=MagicMock())
        storage._client = mock_client

        result = await storage.check_health()

        assert result is True
        assert storage.is_healthy is True
        mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_health_returns_false_when_unhealthy(self):
        """check_health returns False when Qdrant fails."""
        storage = QdrantStorage()

        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(side_effect=Exception("Connection refused"))
        storage._client = mock_client

        result = await storage.check_health()

        assert result is False
        assert storage.is_healthy is False

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_timeout(self):
        """check_health returns False on timeout."""
        import asyncio

        storage = QdrantStorage()

        async def slow_response():
            await asyncio.sleep(10)  # Very slow
            return MagicMock()

        mock_client = AsyncMock()
        mock_client.get_collections = slow_response
        storage._client = mock_client

        # Should timeout quickly (0.1s)
        result = await storage.check_health(timeout=0.1)

        assert result is False
        assert storage.is_healthy is False

    @pytest.mark.asyncio
    async def test_check_health_without_client_returns_false(self):
        """check_health returns False when no client exists."""
        storage = QdrantStorage()
        assert storage._client is None

        result = await storage.check_health()

        assert result is False

    def test_is_healthy_property(self):
        """is_healthy property reflects health state."""
        storage = QdrantStorage()
        assert storage.is_healthy is False

        storage._healthy = True
        assert storage.is_healthy is True

    @pytest.mark.asyncio
    async def test_close_resets_health_state(self):
        """close() resets health check state."""
        storage = QdrantStorage()
        storage._healthy = True
        storage._last_health_check = 12345.0
        storage._client = AsyncMock()

        await storage.close()

        assert storage._healthy is False
        assert storage._last_health_check == 0.0

    @pytest.mark.asyncio
    async def test_get_client_triggers_health_check_periodically(self):
        """_get_client triggers health check when interval elapsed."""
        import time

        storage = QdrantStorage(health_check_interval=0.0)  # Immediate health check

        with patch("vector_core.storage.qdrant.AsyncQdrantClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock())
            mock_class.return_value = mock_client

            # First call creates client and checks health
            await storage._get_client()

            # Should have called get_collections for health check
            assert mock_client.get_collections.called

    @pytest.mark.asyncio
    async def test_reconnect_creates_new_client(self):
        """_reconnect closes old client and creates new one."""
        storage = QdrantStorage()

        old_client = AsyncMock()
        storage._client = old_client
        storage._healthy = True

        with patch("vector_core.storage.qdrant.AsyncQdrantClient") as mock_class:
            new_client = AsyncMock()
            mock_class.return_value = new_client

            await storage._reconnect()

            old_client.close.assert_called_once()
            assert storage._client is new_client
            assert storage._healthy is False
