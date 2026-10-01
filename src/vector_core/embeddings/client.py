"""Embedding client for OpenAI-compatible APIs (llama.cpp, vLLM, etc.)."""

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

import httpx

from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.embeddings.limiter import GlobalRequestLimiter
from vector_core.settings import settings
from vector_core.utils.retry import retry_operation

logger = logging.getLogger(__name__)
EmbeddingRole = Literal["query", "document"]
T = TypeVar("T")


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

    def __init__(  # noqa: PLR0915, PLR0917
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
        concurrency: int | None = None,
        dim: int | None = None,
        cache_namespace: str | None = None,
        cache_path: Path | None = None,
        global_concurrency: int | None = None,
        limiter_dir: Path | None = None,
        *,
        profile: str | None = None,
        query_instruction: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        max_input_bytes: int | None = None,
        tokenizer_path: Path | None = None,
        max_input_tokens: int | None = None,
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
            cache_namespace: Stable model/deployment identity enabling persistent reuse.
            cache_path: Persistent cache database path. Default under cache_dir.
            global_concurrency: Cross-process HTTP request capacity. 0 disables it.
            limiter_dir: Directory containing stable request-slot lock files.
        """
        self.base_url = (base_url or settings.embedding_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.timeout = float(timeout or settings.embedding_timeout)
        self.concurrency = concurrency or settings.embedding_concurrency
        self.dim = dim if dim is not None else settings.embedding_dim
        self._max_text_chars = settings.embedding_max_text_chars
        self.profile = profile if profile is not None else settings.embedding_profile
        if self.profile == "auto":
            alias = self.model.lower().replace("_", "-")
            self.profile = (
                "nemotron3"
                if "nemotron-3-embed" in alias
                else "qwen3"
                if "qwen3-embedding" in alias
                else "raw"
            )
        if self.profile not in {"raw", "qwen3", "nemotron3"}:
            raise ValueError(f"Unknown embedding profile: {self.profile}")
        instruction = (
            query_instruction
            if query_instruction is not None
            else settings.embedding_query_instruction
        )
        default_query = {
            "raw": "",
            "qwen3": f"Instruct: {instruction}\nQuery:",
            "nemotron3": "query: ",
        }
        query_prefix = query_prefix if query_prefix is not None else settings.embedding_query_prefix
        document_prefix = (
            document_prefix if document_prefix is not None else settings.embedding_document_prefix
        )
        self.query_prefix = default_query[self.profile] if query_prefix is None else query_prefix
        self.document_prefix = (
            ("passage: " if self.profile == "nemotron3" else "")
            if document_prefix is None
            else document_prefix
        )
        self.max_input_tokens = (
            max_input_tokens
            if max_input_tokens is not None
            else settings.embedding_max_input_tokens
        )
        if self.max_input_tokens is None:
            self.max_input_tokens = {"nemotron3": 4096, "qwen3": 32768}.get(self.profile)
        tokenizer_path = tokenizer_path or settings.embedding_tokenizer_path
        self._tokenizer = None
        self.tokenizer_fingerprint = None
        if tokenizer_path is not None:
            try:
                from tokenizers import Tokenizer  # noqa: PLC0415 - optional dependency
            except ImportError as error:
                raise ValueError(
                    "Local embedding tokenizer requires vector-core[tokenizer]"
                ) from error
            tokenizer_json = Path(tokenizer_path).read_bytes()
            self._tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
            self.tokenizer_fingerprint = hashlib.sha256(tokenizer_json).hexdigest()
            if self.max_input_tokens is None:
                raise ValueError("A local tokenizer requires embedding_max_input_tokens")
        byte_limit = (
            max_input_bytes if max_input_bytes is not None else settings.embedding_max_input_bytes
        )
        # Known byte-level BPE profiles need no tokenizer dependency for safe operation.
        # Qwen reserves eight tokens for backend special tokens; Nemotron inserts none.
        self.max_input_bytes = (
            byte_limit
            if byte_limit is not None
            else (
                {"nemotron3": 4096, "qwen3": 32760}.get(self.profile, 0)
                if self._tokenizer is None
                else 0
            )
        )
        if self.max_input_bytes < 0 or (
            self.max_input_tokens is not None and self.max_input_tokens <= 0
        ):
            raise ValueError("Embedding input limits must be positive (bytes also permits 0)")
        if self._tokenizer is None and self.max_input_tokens is not None:
            if self.profile == "raw":
                raise ValueError("A token limit for a raw profile requires a local tokenizer")
            safe_bytes = self.max_input_tokens - (8 if self.profile == "qwen3" else 0)
            if safe_bytes <= 0:
                raise ValueError("Embedding token limit leaves no room for input")
            self.max_input_bytes = min(self.max_input_bytes or safe_bytes, safe_bytes)
        self._identity: EmbeddingIdentity | None = None
        self._identity_lock = asyncio.Lock()
        self.cache_namespace = (
            cache_namespace if cache_namespace is not None else settings.embedding_cache_namespace
        )
        self._cache_path = cache_path or (settings.cache_dir / "embeddings.db")
        self._embedding_cache: EmbeddingCache | None = None
        self._embedding_cache_init_lock = asyncio.Lock()
        self._persistent_cache_failed = False
        capacity = (
            global_concurrency
            if global_concurrency is not None
            else settings.embedding_global_concurrency
        )
        limiter_scope = "\0".join((self.base_url, self.model))
        self._request_limiter = GlobalRequestLimiter(
            capacity,
            limiter_scope,
            limiter_dir or (settings.cache_dir / "embedding-request-locks"),
        )

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
        try:
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
            cache = self._embedding_cache
            self._embedding_cache = None
            if cache is not None:
                try:
                    cache.close()
                except Exception:
                    logger.warning("Could not close persistent embedding cache", exc_info=True)
            # Reset circuit breaker state even when HTTP shutdown is cancelled.
            with self._circuit_lock:
                self._circuit_failure_count = 0
                self._circuit_open_until = None

    async def __aenter__(self) -> "EmbeddingClient":
        """Context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
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
            logger.info(f"Circuit breaker half-open: allowing request to {self.base_url}")

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

    async def embed_batch(
        self, texts: list[str], *, role: EmbeddingRole = "document"
    ) -> list[list[float]]:
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
        self._check_identity()
        return await self._embed_prepared_batch(self._prepare_texts(texts, role=role))

    async def _embed_prepared_batch(self, texts: list[str]) -> list[list[float]]:
        """Send already formatted inputs; retries and fallback must not format again."""
        if not texts:
            return []

        # Check circuit breaker before attempting request
        self._check_circuit()

        client = await self._get_client()

        # Transient errors worth retrying
        transient_exceptions = (httpx.ConnectError, httpx.TimeoutException)

        async def make_request() -> httpx.Response:
            """Make the embedding request (can be retried on transient errors)."""
            async with self._request_limiter.acquire():
                resp = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    json={
                        "input": texts,
                        "model": self.model,
                        "encoding_format": "float",
                    },
                )
            # 503 is transient - re-raise for retry
            if resp.status_code == 503:
                raise httpx.ConnectError(f"Service unavailable (503) at {self.base_url}")
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
                for i, text in enumerate(texts):
                    try:
                        single_result = await self._embed_prepared_batch([text])
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
                            f"Failed to embed text: {e}. Text preview: {preview!r}"
                        ) from e
                return results
            # Single text failed - include preview for debugging
            preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            raise EmbeddingServiceError(
                f"Embedding failed: {batch_error}. Text preview: {preview!r}"
            ) from batch_error

        # Success - reset circuit breaker
        self._record_success()
        try:
            data = resp.json()
            return self._validate_response_embeddings(data, len(texts))
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise EmbeddingServiceError(
                f"Embedding service returned an invalid response: {error}"
            ) from error

    @staticmethod
    def _validate_vector(vector: object, expected_dim: int) -> list[float]:
        if not isinstance(vector, list) or len(vector) != expected_dim:
            raise ValueError(f"expected embedding dimension {expected_dim}")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector
        ):
            raise ValueError("embedding contains a non-numeric value")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains a non-finite value")
        return values

    def _validate_response_embeddings(
        self, payload: object, expected_count: int
    ) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("embedding response has no data list")
        items = payload["data"]
        if len(items) != expected_count:
            raise ValueError(
                f"embedding response count {len(items)} does not match {expected_count} inputs"
            )
        by_index: dict[int, object] = {}
        for item in items:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("index"), int)
                or isinstance(item.get("index"), bool)
            ):
                raise ValueError("embedding response item has no integer index")
            index = item["index"]
            if index in by_index or not 0 <= index < expected_count:
                raise ValueError(f"invalid or duplicate embedding index {index}")
            by_index[index] = item.get("embedding")

        inferred_dim = self.dim
        if inferred_dim == 0:
            first = by_index[0]
            if not isinstance(first, list) or not first:
                raise ValueError("cannot infer embedding dimension from response")
            inferred_dim = len(first)
        values = [
            self._validate_vector(by_index[index], inferred_dim) for index in range(expected_count)
        ]
        if self.dim == 0:
            self.dim = inferred_dim
        return values

    def _prepare_texts(self, texts: list[str], *, role: EmbeddingRole) -> list[str]:
        """Format raw inputs exactly once, preserving the prefix when bounding content."""
        if role not in {"query", "document"}:
            raise ValueError(f"Unknown embedding role: {role}")
        prefix = self.query_prefix if role == "query" else self.document_prefix
        char_budget = self._max_text_chars - len(prefix)
        byte_budget = self.max_input_bytes - len(prefix.encode("utf-8"))
        if char_budget <= 0 or (self.max_input_bytes and byte_budget <= 0):
            raise ValueError("Embedding input limit leaves no room after the role prefix")
        tokenizer = self._tokenizer
        # Qwen servers may append EOS. Nemotron's contract adds no special tokens.
        budget = (self.max_input_tokens or 0) - (8 if self.profile == "qwen3" else 0)

        def fits(value: str) -> bool:
            assert tokenizer is not None
            return len(tokenizer.encode(prefix + value, add_special_tokens=False).ids) <= budget

        prepared = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding input must be a non-blank string")
            body = text[:char_budget]
            if self.max_input_bytes:
                body = body.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore")
            if tokenizer is not None:
                if not fits(body):
                    low, high = 0, len(body)
                    while low < high:
                        middle = (low + high + 1) // 2
                        if fits(body[:middle]):
                            low = middle
                        else:
                            high = middle - 1
                    body = body[:low]
                if not fits(body):
                    raise ValueError("Embedding role prefix exceeds the token budget")
            if not body.strip():
                raise ValueError("Embedding input limit leaves no non-blank content after prefix")
            prepared.append(prefix + body)
        return prepared

    def _formatting_config(self) -> dict:
        return {
            "profile": self.profile,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "max_input_bytes": self.max_input_bytes,
            "max_input_tokens": self.max_input_tokens,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    def _configured_identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            model=self.model,
            namespace=self.cache_namespace,
            endpoint=self.base_url,
            dimension=self.dim,
            max_text_chars=self._max_text_chars,
            **self._formatting_config(),
        )

    def configured_identity(self) -> EmbeddingIdentity:
        """Describe an explicitly resolved configuration without contacting the backend.

        Index binding must use resolve_identity() to verify the actual output dimension.
        """
        self._check_identity()
        return self._configured_identity()

    def _check_identity(self) -> None:
        if self._identity is not None and self._configured_identity() != self._identity:
            raise EmbeddingServiceError("Embedding configuration changed on a bound client")

    async def resolve_identity(self) -> EmbeddingIdentity:
        """Probe the backend once before binding an index to this client.

        The uncached request also validates an explicitly configured dimension.
        Subsequent calls retain the same immutable identity and reject mutation.
        """
        async with self._identity_lock:
            if self._identity is None:
                await self.embed_single("Embedding dimension verification")
                self._identity = self._configured_identity()
            self._check_identity()
            return self._identity

    async def _get_embedding_cache(self) -> EmbeddingCache | None:
        """Open the opt-in persistent cache, permanently failing open per client."""
        if not self.cache_namespace or self._persistent_cache_failed:
            return None
        if self._embedding_cache is None:
            async with self._embedding_cache_init_lock:
                if self._embedding_cache is not None:
                    return self._embedding_cache
                try:
                    self._embedding_cache = await asyncio.to_thread(
                        EmbeddingCache,
                        cache_path=self._cache_path,
                    )
                except Exception:
                    self._persistent_cache_failed = True
                    logger.warning(
                        "Persistent embedding cache unavailable; continuing without it",
                        exc_info=True,
                    )
        return self._embedding_cache

    async def embed_single(self, text: str, *, role: EmbeddingRole = "document") -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        results = await self.embed_batch([text], role=role)
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

    async def embed_single_cached(
        self, text: str, *, role: EmbeddingRole = "document"
    ) -> list[float]:
        """
        Embed a single text with in-memory caching (for query use).

        Thread-safe via asyncio.Lock to prevent duplicate API calls.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (from cache or freshly computed)
        """
        self._check_identity()
        # Use SHA256 for better collision resistance (full 64 chars for consistency
        # with EmbeddingCache which uses vector_core.utils.hashing.hash_content)
        effective = self._prepare_texts([text], role=role)[0]
        cache_key = self._memory_cache_key(effective, role=role)

        # Lock-free check first (common case - cache hit)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        # Lock for cache miss to prevent duplicate API calls
        lock = await self._get_cache_lock()
        async with lock:
            # Double-check after acquiring lock
            cache_key = self._memory_cache_key(effective, role=role)
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

            result = await self.embed_single(text, role=role)
            # Auto-detection may have resolved the dimension during this request.
            cache_key = self._memory_cache_key(effective, role=role)

            # LRU-style eviction
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry (first key in dict)
                oldest = next(iter(self._query_cache))
                del self._query_cache[oldest]

            self._query_cache[cache_key] = result
            return result

    def _memory_cache_key(self, text: str, *, role: EmbeddingRole) -> str:
        data = {
            **self._formatting_config(),
            "max_text_chars": self._max_text_chars,
            "endpoint": self.base_url,
            "model": self.model,
            "namespace": self.cache_namespace,
            "dim": self.dim,
            "role": role,
            "input": text,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    async def _embed_batch_with_semaphore(
        self,
        batch_idx: int,
        batch: list[str],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, list[list[float]]]:
        """Embed a batch with semaphore limiting. Returns (batch_idx, embeddings)."""
        async with semaphore:
            embeddings = await self._embed_prepared_batch(batch)
            return (batch_idx, embeddings)

    async def embed_all(
        self,
        texts: list[str],
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        role: EmbeddingRole = "document",
    ) -> list[list[float]]:
        """
        Embed all texts with concurrent batching.

        Args:
            texts: List of texts to embed
            progress_cb: Optional callback(completed, total) invoked after each
                batch finishes embedding, with the running count of embedded
                texts. Batches complete in arbitrary order, so `completed`
                counts texts done so far, not a prefix of the input. The final
                invocation is always (total, total).

        Returns:
            List of embedding vectors in same order as input texts
        """
        if not texts:
            return []

        self._check_identity()
        effective_texts = self._prepare_texts(texts, role=role)
        cache = await self._get_embedding_cache()
        if cache is not None:
            if self.dim > 0:
                return await self._embed_all_cached(
                    effective_texts, cache, progress_cb=progress_cb, role=role
                )
            unique_texts = list(dict.fromkeys(effective_texts))
            text_counts = Counter(effective_texts)
            unique_embeddings = await self._embed_all_uncached(
                unique_texts,
                progress_cb=progress_cb,
                progress_weights=[text_counts[text] for text in unique_texts],
                progress_total=len(effective_texts),
            )
            by_text = dict(zip(unique_texts, unique_embeddings, strict=True))
            embeddings = [by_text[text] for text in effective_texts]
            try:
                await self._write_cache_entries(cache, unique_texts, unique_embeddings, role=role)
            except Exception:
                logger.warning(
                    "Persistent embedding cache write failed; continuing without it",
                    exc_info=True,
                )
                self._persistent_cache_failed = True
                cache.close()
                self._embedding_cache = None
            return embeddings

        return await self._embed_all_uncached(effective_texts, progress_cb=progress_cb)

    async def _embed_all_uncached(
        self,
        texts: list[str],
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        progress_weights: list[int] | None = None,
        initial_completed: int = 0,
        progress_total: int | None = None,
    ) -> list[list[float]]:
        """Embed all supplied effective texts with the existing batch scheduler."""
        total = len(texts)
        weights = progress_weights or [1] * total
        completed = initial_completed
        callback_total = progress_total if progress_total is not None else sum(weights)

        async def embed_with_progress(
            batch_idx: int,
            batch: list[str],
            semaphore: asyncio.Semaphore,
        ) -> tuple[int, list[list[float]]]:
            nonlocal completed
            result = await self._embed_batch_with_semaphore(batch_idx, batch, semaphore)
            batch_start = batch_idx * self.batch_size
            completed += sum(weights[batch_start : batch_start + len(batch)])
            if progress_cb:
                progress_cb(completed, callback_total)
            return result

        # Create batches with their indices
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batches.append((i // self.batch_size, batch))

        # Create semaphore for this call (not cached to avoid event loop binding issues)
        semaphore = asyncio.Semaphore(self.concurrency)

        # Process batches concurrently with semaphore limiting
        tasks = [embed_with_progress(batch_idx, batch, semaphore) for batch_idx, batch in batches]

        # Gather results (concurrent execution up to semaphore limit)
        results = await asyncio.gather(*tasks)

        # Sort by batch index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])

        # Flatten embeddings in correct order
        embeddings: list[list[float]] = []
        for _, batch_embeddings in sorted_results:
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _write_cache_entries(
        self,
        cache: EmbeddingCache,
        effective_texts: list[str],
        embeddings: list[list[float]],
        *,
        role: EmbeddingRole = "document",
    ) -> None:
        if not self.cache_namespace or self.dim <= 0:
            return
        entries = {
            self._persistent_cache_key(text, role=role): embedding
            for text, embedding in zip(effective_texts, embeddings, strict=True)
        }
        await asyncio.to_thread(cache.set_many, entries, expected_dim=self.dim)

    def _persistent_cache_key(
        self, effective_text: str, *, role: EmbeddingRole = "document"
    ) -> str:
        """Key one effective input to an explicit deployment and endpoint."""
        assert self.cache_namespace is not None
        return EmbeddingCache.make_key(
            effective_text,
            namespace=f"{self.cache_namespace}\0{self.base_url}",
            model=self.model,
            dim=self.dim,
            role=role,
            profile=self._configured_identity().fingerprint,
        )

    async def _embed_all_cached(
        self,
        effective_texts: list[str],
        cache: EmbeddingCache,
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        role: EmbeddingRole = "document",
    ) -> list[list[float]]:
        """Resolve cache hits and scatter each unique miss back to every input position."""
        assert self.cache_namespace is not None
        keys = [self._persistent_cache_key(text, role=role) for text in effective_texts]
        unique_keys = list(dict.fromkeys(keys))
        try:
            cached = await asyncio.to_thread(cache.get_many, unique_keys, expected_dim=self.dim)
        except Exception:
            logger.warning(
                "Persistent embedding cache read failed; continuing without it",
                exc_info=True,
            )
            self._persistent_cache_failed = True
            cache.close()
            self._embedding_cache = None
            return await self._embed_all_uncached(effective_texts, progress_cb=progress_cb)
        missing_keys = [key for key in unique_keys if key not in cached]
        key_to_text = dict(zip(keys, effective_texts, strict=True))
        key_counts = Counter(keys)
        cached_count = sum(key_counts[key] for key in cached)

        if missing_keys:
            if progress_cb and cached_count:
                progress_cb(cached_count, len(keys))
            missing_vectors = await self._embed_all_uncached(
                [key_to_text[key] for key in missing_keys],
                progress_cb=progress_cb,
                progress_weights=[key_counts[key] for key in missing_keys],
                initial_completed=cached_count,
                progress_total=len(keys),
            )
            fresh = dict(zip(missing_keys, missing_vectors, strict=True))
            try:
                await asyncio.to_thread(cache.set_many, fresh, expected_dim=self.dim)
            except Exception:
                logger.warning(
                    "Persistent embedding cache write failed; continuing without it",
                    exc_info=True,
                )
                self._persistent_cache_failed = True
                cache.close()
                self._embedding_cache = None
            cached.update(fresh)

        result = [cached[key] for key in keys]
        if progress_cb and not missing_keys:
            progress_cb(len(result), len(result))
        return result


class _SyncAsyncBridge:
    """Run async embedding coroutines from synchronous code on one persistent loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vector-core-sync-embedding",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            raise RuntimeError("sync embedding bridge is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()


class SyncEmbeddingClient:
    """Synchronous facade for :class:`EmbeddingClient`.

    `EmbeddingClient` owns an `httpx.AsyncClient`, so repeatedly wrapping calls
    with `asyncio.run()` can bind internal async resources to short-lived event
    loops. This facade keeps a single background event loop for the lifetime of
    the client and closes the async client on that same loop.
    """

    def __init__(  # noqa: PLR0917
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
        concurrency: int | None = None,
        dim: int | None = None,
        cache_namespace: str | None = None,
        cache_path: Path | None = None,
        global_concurrency: int | None = None,
        limiter_dir: Path | None = None,
        *,
        profile: str | None = None,
        query_instruction: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        max_input_bytes: int | None = None,
        tokenizer_path: Path | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        self._client = EmbeddingClient(
            base_url=base_url,
            model=model,
            batch_size=batch_size,
            timeout=timeout,
            concurrency=concurrency,
            dim=dim,
            cache_namespace=cache_namespace,
            cache_path=cache_path,
            global_concurrency=global_concurrency,
            limiter_dir=limiter_dir,
            profile=profile,
            query_instruction=query_instruction,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            max_input_bytes=max_input_bytes,
            tokenizer_path=tokenizer_path,
            max_input_tokens=max_input_tokens,
        )
        self._bridge = _SyncAsyncBridge()

    @property
    def base_url(self) -> str:
        return self._client.base_url

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def dim(self) -> int:
        return self._client.dim

    def configured_identity(self) -> EmbeddingIdentity:
        return self._client.configured_identity()

    def resolve_identity(self) -> EmbeddingIdentity:
        return self._bridge.run(self._client.resolve_identity())

    def embed_batch(
        self, texts: list[str], *, role: EmbeddingRole = "document"
    ) -> list[list[float]]:
        return self._bridge.run(self._client.embed_batch(texts, role=role))

    def embed_single(self, text: str, *, role: EmbeddingRole = "document") -> list[float]:
        return self._bridge.run(self._client.embed_single(text, role=role))

    def embed_single_cached(self, text: str, *, role: EmbeddingRole = "document") -> list[float]:
        return self._bridge.run(self._client.embed_single_cached(text, role=role))

    def embed_all(
        self,
        texts: list[str],
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        role: EmbeddingRole = "document",
    ) -> list[list[float]]:
        return self._bridge.run(self._client.embed_all(texts, progress_cb=progress_cb, role=role))

    def close(self) -> None:
        if self._bridge._closed:
            return
        self._bridge.run(self._client.close())
        self._bridge.close()

    def __enter__(self) -> "SyncEmbeddingClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
