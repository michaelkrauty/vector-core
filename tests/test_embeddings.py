"""Tests for EmbeddingClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingServiceError,
    SyncEmbeddingClient,
)


class TestEmbeddingClientInit:
    """Tests for EmbeddingClient initialization."""

    def test_default_settings(self):
        """Client uses default settings when not specified."""
        client = EmbeddingClient()
        assert client.base_url is not None
        assert client.model is not None
        assert client.batch_size > 0
        assert client.timeout > 0
        assert client.dim >= 0  # 0 = auto-detect at runtime

    def test_custom_settings(self):
        """Client accepts custom settings."""
        client = EmbeddingClient(
            base_url="http://custom:8080",
            model="custom-model",
            batch_size=10,
            timeout=30.0,
            dim=384,
        )
        assert client.base_url == "http://custom:8080"
        assert client.model == "custom-model"
        assert client.batch_size == 10
        assert client.timeout == 30.0
        assert client.dim == 384

    def test_trailing_slash_stripped(self):
        """Trailing slash is stripped from base_url."""
        client = EmbeddingClient(base_url="http://example.com/")
        assert client.base_url == "http://example.com"


class TestSyncEmbeddingClient:
    """Tests for the sync wrapper used by non-async consumers."""

    def test_sync_embed_batch_uses_persistent_bridge(self):
        client = SyncEmbeddingClient(base_url="http://example.com", model="test", dim=2)
        calls = []

        async def fake_embed_batch(texts):
            calls.append(list(texts))
            return [[float(len(text)), 1.0] for text in texts]

        client._client.embed_batch = fake_embed_batch

        assert client.embed_batch(["a", "bbb"]) == [[1.0, 1.0], [3.0, 1.0]]
        assert client.embed_single("zz") == [2.0, 1.0]
        assert calls == [["a", "bbb"], ["zz"]]
        assert client._bridge._closed is False
        client.close()
        assert client._bridge._closed is True

    def test_sync_context_manager_closes_bridge(self):
        with SyncEmbeddingClient(base_url="http://example.com", model="test", dim=2) as client:
            assert client._bridge._closed is False
        assert client._bridge._closed is True

    def test_sync_close_calls_async_client_close_on_bridge_loop(self):
        client = SyncEmbeddingClient(base_url="http://example.com", model="test", dim=2)
        closed = {"value": False}

        async def fake_close():
            closed["value"] = True

        client._client.close = fake_close
        client.close()

        assert closed["value"] is True


class TestEmbedBatch:
    """Tests for embed_batch method."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty input returns empty list without API call."""
        client = EmbeddingClient()
        result = await client.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_embedding(self):
        """Successful API call returns embeddings in order."""
        client = EmbeddingClient(dim=4)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
                {"index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            result = await client.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]

    @pytest.mark.asyncio
    async def test_reorders_by_index(self):
        """Results are reordered by index if returned out of order."""
        client = EmbeddingClient(dim=4)

        # API returns out of order
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            result = await client.embed_batch(["hello", "world"])

        # Should be reordered by index
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]

    @pytest.mark.asyncio
    async def test_truncates_long_text(self):
        """Very long texts are truncated to avoid API errors."""
        client = EmbeddingClient(dim=4)

        long_text = "x" * 10000  # Longer than max_chars

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            await client.embed_batch([long_text])

            # Verify the text was truncated
            call_args = mock_http.post.call_args
            sent_texts = call_args[1]["json"]["input"]
            assert len(sent_texts[0]) <= 8000

    @pytest.mark.asyncio
    async def test_connection_error_raises_embedding_error(self):
        """Connection error raises EmbeddingServiceError."""
        client = EmbeddingClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["hello"])

            assert "Cannot connect to embedding service" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_embedding_error(self):
        """Timeout raises EmbeddingServiceError."""
        client = EmbeddingClient()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["hello"])

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_503_raises_embedding_error(self):
        """503 response raises EmbeddingServiceError."""
        client = EmbeddingClient()

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "503", request=mock_request, response=mock_response
                )
            )
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["hello"])

            assert "unavailable" in str(exc_info.value).lower()


class TestEmbedSingle:
    """Tests for embed_single method."""

    @pytest.mark.asyncio
    async def test_delegates_to_batch(self):
        """embed_single delegates to embed_batch."""
        client = EmbeddingClient(dim=4)

        with patch.object(client, 'embed_batch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [[0.1, 0.2, 0.3, 0.4]]

            result = await client.embed_single("hello")

            mock_batch.assert_called_once_with(["hello"])
            assert result == [0.1, 0.2, 0.3, 0.4]


class TestEmbedSingleCached:
    """Tests for embed_single_cached method."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Cached values are returned without API call."""
        client = EmbeddingClient(dim=4)

        with patch.object(client, 'embed_single', new_callable=AsyncMock) as mock_single:
            mock_single.return_value = [0.1, 0.2, 0.3, 0.4]

            # First call - not cached
            result1 = await client.embed_single_cached("hello")
            assert mock_single.call_count == 1

            # Second call - should be cached
            result2 = await client.embed_single_cached("hello")
            assert mock_single.call_count == 1  # No additional call

            assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """LRU cache evicts oldest entries."""
        client = EmbeddingClient(dim=4)
        client._cache_max_size = 2

        with patch.object(client, 'embed_single', new_callable=AsyncMock) as mock_single:
            mock_single.side_effect = lambda t: [float(ord(t[0]))] * 4

            # Fill cache
            await client.embed_single_cached("a")
            await client.embed_single_cached("b")
            assert mock_single.call_count == 2

            # Evict oldest
            await client.embed_single_cached("c")
            assert mock_single.call_count == 3

            # "a" should have been evicted
            await client.embed_single_cached("a")
            assert mock_single.call_count == 4  # Had to recompute


class TestEmbedAll:
    """Tests for embed_all method."""

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty input returns empty list."""
        client = EmbeddingClient()
        result = await client.embed_all([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batching(self):
        """Texts are batched correctly."""
        client = EmbeddingClient(batch_size=2, dim=4)

        call_count = 0

        async def mock_embed_batch(texts):
            nonlocal call_count
            call_count += 1
            return [[float(i)] * 4 for i in range(len(texts))]

        with patch.object(client, 'embed_batch', side_effect=mock_embed_batch):
            result = await client.embed_all(["a", "b", "c", "d", "e"])

        # Should have made 3 batches: [a,b], [c,d], [e]
        assert call_count == 3
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        """Results are returned in same order as input."""
        client = EmbeddingClient(batch_size=2, dim=4)

        async def mock_embed_batch(texts):
            # Return embeddings that encode the text position
            return [[float(ord(t[0]))] * 4 for t in texts]

        with patch.object(client, 'embed_batch', side_effect=mock_embed_batch):
            result = await client.embed_all(["a", "b", "c"])

        assert result[0][0] == float(ord("a"))
        assert result[1][0] == float(ord("b"))
        assert result[2][0] == float(ord("c"))

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Progress callback is called."""
        client = EmbeddingClient(batch_size=2, dim=4)
        progress_calls = []

        async def mock_embed_batch(texts):
            return [[0.0] * 4 for _ in texts]

        with patch.object(client, 'embed_batch', side_effect=mock_embed_batch):
            await client.embed_all(
                ["a", "b", "c"],
                progress_cb=lambda completed, total: progress_calls.append((completed, total))
            )

        assert len(progress_calls) > 0
        assert progress_calls[-1] == (3, 3)


class TestClientLifecycle:
    """Tests for client lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """close() cleans up HTTP client."""
        import asyncio
        client = EmbeddingClient()

        # Create the client and set the loop (simulating proper initialization)
        mock_http = AsyncMock()
        client._client = mock_http
        client._client_loop = asyncio.get_running_loop()  # Same loop = will close

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._client is None
        assert client._client_loop is None

    @pytest.mark.asyncio
    async def test_close_when_not_created(self):
        """close() is safe when client not created."""
        client = EmbeddingClient()
        assert client._client is None

        await client.close()  # Should not raise

        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """_get_client creates httpx client on first call (lines 64-69)."""
        client = EmbeddingClient(timeout=30.0)
        assert client._client is None

        # Call the method
        http_client = await client._get_client()

        # Should have created an httpx client
        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)
        assert client._client is http_client

        # Clean up
        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_returns_cached(self):
        """_get_client returns cached client on subsequent calls."""
        client = EmbeddingClient()

        # First call creates client
        http_client1 = await client._get_client()
        # Second call returns same client
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

        await client.close()


class TestBatchRetryLogic:
    """Tests for batch retry logic (lines 122-139)."""

    @pytest.mark.asyncio
    async def test_non_http_status_error_retry_individual(self):
        """Non-HTTPStatusError triggers individual retry (lines 122-124)."""
        client = EmbeddingClient(dim=4)

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "400", request=mock_request, response=mock_response
                )
            )
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["hello"])

            assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_exception_retry_one_at_a_time(self):
        """Generic exception triggers individual retry (lines 125-138)."""
        client = EmbeddingClient(dim=4)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First batch call fails with generic exception
                raise ValueError("Some unexpected error")

            # Individual retries succeed
            input_texts = kwargs.get("json", {}).get("input", [])
            response = MagicMock()
            response.json.return_value = {
                "data": [
                    {"index": i, "embedding": [float(call_count)] * 4}
                    for i in range(len(input_texts))
                ]
            }
            response.raise_for_status = MagicMock()
            return response

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            result = await client.embed_batch(["hello", "world"])

            # Should have made 3 calls: 1 batch + 2 individual retries
            assert call_count == 3
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_generic_exception_single_text_raises_embedding_error(self):
        """Generic exception with single text raises EmbeddingServiceError."""
        client = EmbeddingClient(dim=4)

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_get_client.return_value = mock_http

            # Now wraps the exception in EmbeddingServiceError
            with pytest.raises(EmbeddingServiceError, match="Embedding failed"):
                await client.embed_batch(["single"])

    @pytest.mark.asyncio
    async def test_retry_individual_with_embedding_service_error(self):
        """EmbeddingServiceError during retry is re-raised (lines 133-134)."""
        client = EmbeddingClient(dim=4)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First batch call fails with generic exception
                raise ValueError("Batch failed")

            # Individual retry raises ConnectError -> EmbeddingServiceError
            raise httpx.ConnectError("Connection refused")

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError):
                await client.embed_batch(["hello", "world"])

    @pytest.mark.asyncio
    async def test_retry_individual_with_other_exception_raises_error(self):
        """Other exception during individual retry raises EmbeddingServiceError.

        Previously this would silently return a zero vector, which corrupts
        search accuracy. Now it properly raises an error.
        """
        client = EmbeddingClient(dim=4)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First batch call fails
                raise ValueError("Batch failed")

            if call_count == 2:
                # First individual retry succeeds
                response = MagicMock()
                response.json.return_value = {
                    "data": [{"index": 0, "embedding": [1.0, 1.0, 1.0, 1.0]}]
                }
                response.raise_for_status = MagicMock()
                return response

            # Second individual retry fails with non-EmbeddingServiceError
            raise RuntimeError("Some other error")

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            # Should raise EmbeddingServiceError instead of returning zero vector
            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["hello", "world"])

            # Error message should include text preview for debugging
            # The recursive single-text call raises "Embedding failed: ..."
            assert "Embedding failed" in str(exc_info.value)
            assert "world" in str(exc_info.value)


class TestEmbedSingleCachedRaceCondition:
    """Tests for cache race condition handling (line 207)."""

    @pytest.mark.asyncio
    async def test_double_check_after_lock_acquisition(self):
        """Tests the double-check pattern after acquiring lock (line 206-207).

        This simulates a race condition where another task populates the cache
        while we're waiting for the lock.
        """
        client = EmbeddingClient(dim=4)
        call_count = 0

        async def mock_embed_single(text):
            nonlocal call_count
            call_count += 1
            return [float(call_count)] * 4

        with patch.object(client, 'embed_single', side_effect=mock_embed_single):
            # Call twice with same text - second should use cache
            import asyncio
            result1, result2 = await asyncio.gather(
                client.embed_single_cached("same_text"),
                client.embed_single_cached("same_text"),
            )

        # Should only have called embed_single once due to locking
        assert call_count == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_hit_after_lock_returns_cached_value(self):
        """Tests that cached value found after lock acquisition is returned (line 207)."""
        import asyncio
        client = EmbeddingClient(dim=4)

        # Pre-populate the cache by calling once
        with patch.object(client, 'embed_single', new_callable=AsyncMock) as mock:
            mock.return_value = [1.0, 2.0, 3.0, 4.0]
            result1 = await client.embed_single_cached("test_text")
            assert mock.call_count == 1

            # Second call should hit cache without calling embed_single
            result2 = await client.embed_single_cached("test_text")
            assert mock.call_count == 1  # No additional call

        assert result1 == result2 == [1.0, 2.0, 3.0, 4.0]


class TestRetryExceptionPaths:
    """Tests for exception paths in retry logic (lines 140-148)."""

    @pytest.mark.asyncio
    async def test_individual_retry_exception_includes_text_preview(self):
        """Exception in individual retry includes text preview (lines 143-148)."""
        client = EmbeddingClient(dim=4)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Batch fails
                raise ValueError("Batch failed")

            # Individual retry also fails with non-embedding error
            raise RuntimeError("Individual text failed")

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch(["short", "another"])

            # Should include text preview in error
            error_msg = str(exc_info.value)
            assert "short" in error_msg or "another" in error_msg

    @pytest.mark.asyncio
    async def test_long_text_preview_truncated_in_error(self):
        """Long text is truncated in error message (lines 143, 154)."""
        client = EmbeddingClient(dim=4)

        long_text = "x" * 200  # Longer than 100 chars

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Failed")

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await client.embed_batch([long_text])

            # Text preview should be truncated with "..."
            error_msg = str(exc_info.value)
            assert "..." in error_msg
            # Should not contain full 200 chars
            assert len(error_msg) < 300


class TestCircuitBreaker:
    """Tests for circuit breaker pattern in EmbeddingClient."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Circuit opens after 5 consecutive failures."""
        from vector_core.embeddings.client import CircuitBreakerOpenError

        client = EmbeddingClient(dim=4)
        client._circuit_threshold = 3  # Lower threshold for faster testing

        async def mock_post(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=mock_post)
            mock_get_client.return_value = mock_http

            # First 3 failures should raise EmbeddingServiceError
            for i in range(3):
                with pytest.raises(EmbeddingServiceError):
                    await client.embed_batch(["test"])

            # Circuit should now be open
            assert client._circuit_open_until is not None
            assert client._circuit_failure_count == 3

            # Next request should immediately raise CircuitBreakerOpenError
            with pytest.raises(CircuitBreakerOpenError) as exc_info:
                await client.embed_batch(["test"])

            assert client.base_url in str(exc_info.value)
            assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self):
        """Successful request resets circuit breaker."""
        client = EmbeddingClient(dim=4)

        # Simulate some failures
        client._circuit_failure_count = 3
        client._circuit_open_until = None  # Not open yet

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]
        }

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            await client.embed_batch(["test"])

        # Circuit should be reset
        assert client._circuit_failure_count == 0
        assert client._circuit_open_until is None

    @pytest.mark.asyncio
    async def test_circuit_half_open_allows_request(self):
        """After reset time, circuit enters half-open state allowing one request."""
        import time as time_module

        client = EmbeddingClient(dim=4)
        client._circuit_threshold = 3
        client._circuit_reset_time = 0.1  # Short reset for testing

        # Manually open the circuit with expired timeout
        client._circuit_failure_count = 5
        client._circuit_open_until = time_module.monotonic() - 1.0  # Already expired

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]
        }

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            # Should succeed and reset circuit
            result = await client.embed_batch(["test"])

        assert result == [[1.0, 2.0, 3.0, 4.0]]
        assert client._circuit_failure_count == 0
        assert client._circuit_open_until is None

    @pytest.mark.asyncio
    async def test_close_resets_circuit_breaker(self):
        """close() resets circuit breaker state."""
        client = EmbeddingClient(dim=4)

        # Set up failure state
        client._circuit_failure_count = 5
        client._circuit_open_until = 12345.0

        await client.close()

        assert client._circuit_failure_count == 0
        assert client._circuit_open_until is None

    @pytest.mark.asyncio
    async def test_5xx_errors_count_toward_circuit(self):
        """5xx errors are counted toward circuit breaker threshold."""
        client = EmbeddingClient(dim=4)
        client._circuit_threshold = 2

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "500", request=mock_request, response=mock_response
                )
            )
            mock_get_client.return_value = mock_http

            # First failure
            with pytest.raises(EmbeddingServiceError):
                await client.embed_batch(["test"])
            assert client._circuit_failure_count == 1

            # Second failure opens circuit
            with pytest.raises(EmbeddingServiceError):
                await client.embed_batch(["test"])
            assert client._circuit_failure_count == 2
            assert client._circuit_open_until is not None

    @pytest.mark.asyncio
    async def test_4xx_errors_do_not_count_toward_circuit(self):
        """4xx errors (client errors) don't count toward circuit breaker."""
        client = EmbeddingClient(dim=4)

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "400", request=mock_request, response=mock_response
                )
            )
            mock_get_client.return_value = mock_http

            with pytest.raises(EmbeddingServiceError):
                await client.embed_batch(["test"])

        # 4xx errors shouldn't count as failures
        assert client._circuit_failure_count == 0
