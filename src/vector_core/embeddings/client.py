"""Embedding client for OpenAI-compatible APIs (llama.cpp, vLLM, etc.)."""

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Callable

import httpx

from vector_core.settings import settings
from vector_core.utils.retry import retry_operation

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when the embedding service is unavailable or returns an error."""
    pass


class CircuitBreakerOpenError(EmbeddingServiceError):
    """Raised when circuit breaker is open and requests are blocked."""

    def __init__(self, service_url: str, retry_after: float):
        self.service_url = service_url
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open: embedding service at {service_url} is unavailable. "
            f"Will retry in {retry_after:.1f}s"
        )


class EmbeddingClient:
    """
    Generate dense embeddings via OpenAI-compatible /v1/embeddings endpoint.

    Works with:
    - llama.cpp server
    - vLLM
    - OpenAI API
    - Any OpenAI-compatible embedding API
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
        concurrency: int | None = None,
        dim: int | None = None,
    ):
        """
        Initialize embedding client.

        Args:
            base_url: Base URL of embedding API. Default from settings.
            model: Model name. Default from settings.
            batch_size: Max texts per batch. Default from settings.
            timeout: Request timeout in seconds. Default from settings.
            concurrency: Max concurrent batch requests. Default from settings.
            dim: Embedding dimension. Default from settings.
        """
        self.base_url = (base_url or settings.embedding_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.timeout = float(timeout or settings.embedding_timeout)
        self.concurrency = concurrency or settings.embedding_concurrency
        self.dim = dim or settings.embedding_dim

        # Persistent HTTP client (reuse connections)
        self._client: httpx.AsyncClient | None = None
        # Track the event loop where client was created (for safe cleanup)
        self._client_loop: asyncio.AbstractEventLoop | None = None
        # Query embedding cache (LRU-style, in-memory) with thread-safe access
        self._query_cache: dict[str, list[float]] = {}
        self._cache_max_size = 100
        self._cache_lock: asyncio.Lock | None = None  # Created lazily per event loop
        self._cache_lock_init = threading.Lock()  # Thread-safe lock for creating async lock
        # Semaphore for concurrent batch limiting (created per-call to avoid event loop issues)
        # Note: Not cached because async primitives are bound to the event loop they're created in

        # Circuit breaker state (protects against repeated calls to unavailable service)
        self._circuit_failure_count = 0
        self._circuit_open_until: float | None = None
        self._circuit_threshold = settings.circuit_breaker_threshold
        self._circuit_reset_time = settings.circuit_breaker_reset_seconds
        self._circuit_lock = threading.Lock()  # Thread-safe circuit state access

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            # Track the loop where client was created for safe cleanup
            self._client_loop = asyncio.get_running_loop()
        return self._client

    async def close(self) -> None:
        """
        Close the HTTP client and reset circuit breaker. Call on shutdown.

        SAFETY: If called from a different event loop than where the client was
        created (e.g., during atexit cleanup via asyncio.run()), we skip the
        async close to avoid "Event loop is closed" errors. The client will be
        GC'd and connections will timeout naturally.
        """
        if self._client:
            try:
                current_loop = asyncio.get_running_loop()
                if self._client_loop is current_loop:
                    # Same loop - safe to close properly
                    await self._client.aclose()
                else:
                    # Different loop - cannot safely close httpx client
                    # httpx.AsyncClient has internal locks bound to creation loop
                    logger.debug(
                        "Skipping async httpx client close (called from different event loop)"
                    )
            except RuntimeError:
                # No running loop - cannot close async resources
                logger.debug("Skipping async httpx client close (no running event loop)")
            finally:
                self._client = None
                self._client_loop = None
        # Reset circuit breaker state
        with self._circuit_lock:
            self._circuit_failure_count = 0
            self._circuit_open_until = None

    async def __aenter__(self) -> "EmbeddingClient":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures client is closed."""
        await self.close()

    def _check_circuit(self) -> None:
        """
        Check circuit breaker state. Raises CircuitBreakerOpenError if open.

        When the circuit is open, this fast-fails requests instead of waiting
        for the service that we know is down. After the reset time passes,
        the circuit enters "half-open" state and allows one request through.
        """
        with self._circuit_lock:
            if self._circuit_open_until is None:
                return  # Circuit is closed, allow request

            now = time.monotonic()
            if now < self._circuit_open_until:
                # Circuit is still open
                retry_after = self._circuit_open_until - now
                raise CircuitBreakerOpenError(self.base_url, retry_after)

            # Circuit timeout expired - enter half-open state
            # Allow request through, will reset or re-open based on result
            logger.info(
                f"Circuit breaker half-open: allowing request to {self.base_url}"
            )

    def _record_success(self) -> None:
        """Record a successful request. Resets failure count and closes circuit."""
        with self._circuit_lock:
            if self._circuit_failure_count > 0 or self._circuit_open_until is not None:
                logger.info(f"Circuit breaker closed: {self.base_url} is healthy")
            self._circuit_failure_count = 0
            self._circuit_open_until = None

    def _record_failure(self) -> None:
        """
        Record a failed request. May open the circuit if threshold is reached.

        Called after all retries are exhausted for a single request.
        """
        with self._circuit_lock:
            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self._circuit_threshold:
                if self._circuit_open_until is None:
                    self._circuit_open_until = time.monotonic() + self._circuit_reset_time
                    logger.warning(
                        f"Circuit breaker opened: {self.base_url} failed "
                        f"{self._circuit_failure_count} times. "
                        f"Blocking requests for {self._circuit_reset_time}s"
                    )
                else:
                    # Already open, extend the timeout
                    self._circuit_open_until = time.monotonic() + self._circuit_reset_time

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors in same order as input

        Raises:
            CircuitBreakerOpenError: If service is known to be unavailable
            EmbeddingServiceError: If embedding fails after retries
        """
        if not texts:
            return []

        # Check circuit breaker before attempting request
        self._check_circuit()

        # Truncate very long texts that might cause 400 errors
        max_chars = settings.embedding_max_text_chars
        truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        client = await self._get_client()

        # Transient errors worth retrying
        transient_exceptions = (httpx.ConnectError, httpx.TimeoutException)

        async def make_request() -> httpx.Response:
            """Make the embedding request (can be retried on transient errors)."""
            resp = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "input": truncated,
                    "model": self.model,
                    "encoding_format": "float",
                },
            )
            # 503 is transient - re-raise for retry
            if resp.status_code == 503:
                raise httpx.ConnectError(
                    f"Service unavailable (503) at {self.base_url}"
                )
            resp.raise_for_status()
            return resp

        try:
            resp = await retry_operation(
                make_request,
                max_retries=2,  # Total 3 attempts (1 + 2 retries)
                retry_exceptions=transient_exceptions,
                initial_delay=1.0,
                max_delay=4.0,
                operation_name=f"embed_batch({len(texts)} texts)",
            )
        except httpx.ConnectError as e:
            # Connection failed after all retries - record failure for circuit breaker
            self._record_failure()
            raise EmbeddingServiceError(
                f"Cannot connect to embedding service at {self.base_url} after retries. "
                f"Ensure the server is running with an embedding model loaded. "
                f"Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            # Timeout after all retries - record failure for circuit breaker
            self._record_failure()
            raise EmbeddingServiceError(
                f"Embedding service timed out after {self.timeout}s (with retries). "
                f"The server may be overloaded or the batch is too large."
            ) from e
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx/5xx) - may or may not be transient
            # Only count server errors (5xx) as circuit breaker failures
            if e.response.status_code >= 500:
                self._record_failure()
            raise EmbeddingServiceError(
                f"Embedding service error: {e.response.status_code} - {e.response.text[:200]}"
            ) from e
        except Exception as batch_error:
            # If batch fails, try one at a time to identify the problematic text
            if len(texts) > 1:
                results = []
                for i, text in enumerate(truncated):
                    try:
                        single_result = await self.embed_batch([text])
                        results.extend(single_result)
                    except EmbeddingServiceError:
                        raise  # Re-raise service errors, don't mask them
                    except Exception as e:
                        # Log and raise - don't silently use zero vectors
                        # Zero vectors corrupt search accuracy by never matching anything
                        preview = text[:100] + "..." if len(text) > 100 else text
                        logger.error(
                            f"Failed to embed text {i + 1}/{len(texts)}: {e}. "
                            f"Text preview: {preview!r}"
                        )
                        raise EmbeddingServiceError(
                            f"Failed to embed text: {e}. "
                            f"Text preview: {preview!r}"
                        ) from e
                return results
            # Single text failed - include preview for debugging
            preview = truncated[0][:100] + "..." if len(truncated[0]) > 100 else truncated[0]
            raise EmbeddingServiceError(
                f"Embedding failed: {batch_error}. Text preview: {preview!r}"
            ) from batch_error

        # Success - reset circuit breaker
        self._record_success()
        data = resp.json()
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        results = await self.embed_batch([text])
        return results[0]

    async def _get_cache_lock(self) -> asyncio.Lock:
        """Get or create cache lock for the current event loop (thread-safe)."""
        if self._cache_lock is None:
            # Use threading lock for double-checked locking to prevent race condition
            # where multiple coroutines could create separate asyncio.Locks
            with self._cache_lock_init:
                if self._cache_lock is None:
                    self._cache_lock = asyncio.Lock()
        return self._cache_lock

    async def embed_single_cached(self, text: str) -> list[float]:
        """
        Embed a single text with in-memory caching (for query use).

        Thread-safe via asyncio.Lock to prevent duplicate API calls.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (from cache or freshly computed)
        """
        # Use SHA256 for better collision resistance (full 64 chars for consistency
        # with EmbeddingCache which uses vector_core.utils.hashing.hash_content)
        cache_key = hashlib.sha256(text.encode()).hexdigest()

        # Lock-free check first (common case - cache hit)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        # Lock for cache miss to prevent duplicate API calls
        lock = await self._get_cache_lock()
        async with lock:
            # Double-check after acquiring lock
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

            result = await self.embed_single(text)

            # LRU-style eviction
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry (first key in dict)
                oldest = next(iter(self._query_cache))
                del self._query_cache[oldest]

            self._query_cache[cache_key] = result
            return result

    async def _embed_batch_with_semaphore(
        self,
        batch_idx: int,
        batch: list[str],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, list[list[float]]]:
        """Embed a batch with semaphore limiting. Returns (batch_idx, embeddings)."""
        async with semaphore:
            embeddings = await self.embed_batch(batch)
            return (batch_idx, embeddings)

    async def embed_all(
        self,
        texts: list[str],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """
        Embed all texts with concurrent batching.

        Args:
            texts: List of texts to embed
            progress_cb: Optional callback(completed, total) for progress reporting

        Returns:
            List of embedding vectors in same order as input texts
        """
        if not texts:
            return []

        total = len(texts)

        # Create batches with their indices
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batches.append((i // self.batch_size, batch))

        # Create semaphore for this call (not cached to avoid event loop binding issues)
        semaphore = asyncio.Semaphore(self.concurrency)

        # Process batches concurrently with semaphore limiting
        tasks = [
            self._embed_batch_with_semaphore(batch_idx, batch, semaphore)
            for batch_idx, batch in batches
        ]

        # Gather results (concurrent execution up to semaphore limit)
        results = await asyncio.gather(*tasks)

        # Sort by batch index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])

        # Flatten embeddings in correct order
        embeddings: list[list[float]] = []
        for _, batch_embeddings in sorted_results:
            embeddings.extend(batch_embeddings)

        if progress_cb:
            progress_cb(len(embeddings), total)

        return embeddings
