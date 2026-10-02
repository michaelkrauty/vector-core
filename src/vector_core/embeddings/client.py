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
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, NoReturn, TypeVar

import httpx

from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.embeddings.limiter import GlobalRequestLimiter
from vector_core.settings import settings
from vector_core.utils.retry import retry_operation

logger = logging.getLogger(__name__)
EmbeddingRole = Literal["query", "document"]
T = TypeVar("T")


@dataclass(frozen=True)
class EmbeddingSpan:
    """One exact half-open character range in an unmodified source string."""

    start: int
    end: int
    text: str


class EmbeddingInputTooLongError(ValueError):
    """A complete formatted input exceeds an explicitly configured limit."""

    def __init__(self, index: int, role: EmbeddingRole, measured: int, limit: int, unit: str):
        self.index = index
        self.role = role
        self.measured = measured
        self.limit = limit
        self.unit = unit
        super().__init__(
            f"Embedding {role} input {index} needs {measured} {unit}; limit is {limit}. "
            "Split the complete document into searchable spans or shorten the query explicitly."
        )


class EmbeddingRequestTooLargeError(ValueError):
    """One complete input cannot fit the configured HTTP request body budget."""


class EmbeddingInputRejectedError(ValueError):
    """The backend rejected complete input; this is not a service outage."""

    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        self.reason = reason
        super().__init__(f"Embedding backend rejected input ({status_code}): {reason}")


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
        max_text_chars: int | None = None,
        tokenizer_add_special_tokens: bool | None = None,
        reserved_tokens: int | None = None,
        max_request_bytes: int | None = None,
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
        self._max_text_chars = (
            max_text_chars if max_text_chars is not None else settings.embedding_max_text_chars
        )
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
        self.tokenizer_add_special_tokens = (
            tokenizer_add_special_tokens
            if tokenizer_add_special_tokens is not None
            else settings.embedding_tokenizer_add_special_tokens
        )
        self.reserved_tokens = (
            reserved_tokens if reserved_tokens is not None else settings.embedding_reserved_tokens
        )
        self.max_request_bytes = (
            max_request_bytes
            if max_request_bytes is not None
            else settings.embedding_max_request_bytes
        )
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
        byte_limit = (
            max_input_bytes if max_input_bytes is not None else settings.embedding_max_input_bytes
        )
        self.max_input_bytes = byte_limit or 0
        if self.max_input_bytes < 0 or (
            self.max_input_tokens is not None and self.max_input_tokens <= 0
        ):
            raise ValueError("Embedding input limits must be positive (bytes also permits 0)")
        if self._tokenizer is None and self.max_input_tokens is not None:
            raise ValueError("An embedding token limit requires a local tokenizer")
        if self._max_text_chars is not None and self._max_text_chars <= 0:
            raise ValueError("Embedding character limit must be positive")
        if self.reserved_tokens < 0:
            raise ValueError("Embedding reserved token count must be non-negative")
        if self.max_input_tokens is not None and self.reserved_tokens >= self.max_input_tokens:
            raise ValueError("Embedding token limit leaves no room for input")
        if self.max_request_bytes is not None and self.max_request_bytes <= 0:
            raise ValueError("Embedding request byte limit must be positive")
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
        prepared = self._prepare_texts(texts, role=role)
        results = []
        for _, batch in self._request_batches(prepared):
            results.extend(await self._embed_prepared_batch(batch))
        return results

    def _request_payload(self, texts: list[str]) -> dict[str, Any]:
        return {"input": texts, "model": self.model, "encoding_format": "float"}

    def _request_batches(self, texts: list[str]) -> list[tuple[int, list[str]]]:
        """Pack complete inputs using the same JSON serializer as the HTTP transport."""
        batches: list[tuple[int, list[str]]] = []
        start = 0
        batch: list[str] = []
        for index, text in enumerate(texts):
            candidate = [*batch, text]
            if self.max_request_bytes is not None:
                size = len(
                    httpx.Request(
                        "POST", self.base_url, json=self._request_payload(candidate)
                    ).content
                )
                if size > self.max_request_bytes:
                    if batch:
                        batches.append((start, batch))
                        start, batch = index, []
                    size = len(
                        httpx.Request(
                            "POST", self.base_url, json=self._request_payload([text])
                        ).content
                    )
                    if size > self.max_request_bytes:
                        raise EmbeddingRequestTooLargeError(
                            f"Embedding input {index} needs a {size}-byte HTTP body; "
                            f"request limit is {self.max_request_bytes} bytes. "
                            "Increase the transport budget or explicitly split the document."
                        )
            batch.append(text)
            if len(batch) == self.batch_size:
                batches.append((start, batch))
                start, batch = index + 1, []
        if batch:
            batches.append((start, batch))
        return batches

    def _raise_http_error(self, error: httpx.HTTPStatusError) -> NoReturn:
        status = error.response.status_code
        reason = error.response.text[:200]
        if status == 413:
            raise EmbeddingRequestTooLargeError(
                f"Embedding backend rejected the request body (413): {reason}. "
                "Configure a smaller HTTP batch budget or explicitly split the document."
            ) from error
        if status in {400, 422}:
            raise EmbeddingInputRejectedError(status, reason) from error
        if status >= 500:
            self._record_failure()
        raise EmbeddingServiceError(f"Embedding service error: {status} - {reason}") from error

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
                    json=self._request_payload(texts),
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
            self._raise_http_error(e)
        except Exception as batch_error:
            # If batch fails, try one at a time to identify the problematic text
            if len(texts) > 1:
                results = []
                for i, text in enumerate(texts):
                    try:
                        single_result = await self._embed_prepared_batch([text])
                        results.extend(single_result)
                    except (
                        EmbeddingServiceError,
                        EmbeddingInputRejectedError,
                        EmbeddingRequestTooLargeError,
                    ):
                        raise  # Preserve input rejection even in diagnostic batch fallback.
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
        """Format and validate complete inputs. No accepted source text is discarded."""
        if role not in {"query", "document"}:
            raise ValueError(f"Unknown embedding role: {role}")
        prefix = self.query_prefix if role == "query" else self.document_prefix
        prepared = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                raise ValueError("Embedding input must be a non-empty string")
            formatted = prefix + text
            self._validate_formatted(formatted, index=index, role=role)
            prepared.append(formatted)
        return prepared

    def _validate_formatted(self, text: str, *, index: int, role: EmbeddingRole) -> None:
        if self._max_text_chars is not None and len(text) > self._max_text_chars:
            raise EmbeddingInputTooLongError(
                index, role, len(text), self._max_text_chars, "characters"
            )
        if self.max_input_bytes:
            size = len(text.encode("utf-8"))
            if size > self.max_input_bytes:
                raise EmbeddingInputTooLongError(
                    index, role, size, self.max_input_bytes, "UTF-8 bytes"
                )
        if self.max_input_tokens is not None:
            assert self._tokenizer is not None
            size = (
                len(
                    self._tokenizer.encode(
                        text,
                        add_special_tokens=self.tokenizer_add_special_tokens,
                    )
                )
                + self.reserved_tokens
            )
            if size > self.max_input_tokens:
                raise EmbeddingInputTooLongError(index, role, size, self.max_input_tokens, "tokens")

    @staticmethod
    def _token_boundary(encoding: Any, prefix_length: int, body_length: int, budget: int) -> int:
        """Locate a source boundary without allocating the complete token ID list."""
        mask = encoding.special_tokens_mask
        available = max(0, budget - sum(mask))
        end = 0
        for (_, token_end), special in zip(encoding.offsets, mask, strict=True):
            if special:
                continue
            if not available:
                break
            available -= 1
            if prefix_length < token_end < prefix_length + body_length:
                end = max(end, token_end - prefix_length)
        return end

    def split_text(
        self,
        text: str,
        *,
        role: EmbeddingRole = "document",
        context_prefix: str = "",
    ) -> list[EmbeddingSpan]:
        """Partition raw text into exact, independently embeddable source spans.

        Embed each ``context_prefix + span.text`` with the same role. Character
        offsets refer only to ``text``. The optional context repeats for each
        span and is included, along with the role prefix, in every limit check.
        Tokenization uses bounded windows; this guarantees coverage and valid
        spans, not the globally largest possible span for every tokenizer.
        """
        self._check_identity()
        if not isinstance(text, str) or not text:
            raise ValueError("Embedding input must be a non-empty string")
        if role not in {"query", "document"}:
            raise ValueError(f"Unknown embedding role: {role}")
        if not isinstance(context_prefix, str):
            raise ValueError("Embedding context prefix must be a string")
        prefix = (self.query_prefix if role == "query" else self.document_prefix) + context_prefix
        char_budget = (
            self._max_text_chars - len(prefix) if self._max_text_chars is not None else len(text)
        )
        byte_budget = self.max_input_bytes - len(prefix.encode("utf-8"))
        if char_budget <= 0 or (self.max_input_bytes and byte_budget <= 0):
            self._validate_formatted(prefix + text[:1], index=0, role=role)
            raise ValueError("Embedding limits leave no room after the prefix")
        # A bounded work window prevents a giant retained document from creating
        # millions of tokenizer objects at once. It is not an acceptance limit.
        max_window = min(
            char_budget, (self.max_input_tokens * 8) if self.max_input_tokens else len(text)
        )
        window = max_window
        prefix_tokens = 0
        if self.max_input_tokens is not None:
            assert self._tokenizer is not None
            prefix_tokens = len(
                self._tokenizer.encode(
                    prefix,
                    add_special_tokens=self.tokenizer_add_special_tokens,
                )
            )
        spans: list[EmbeddingSpan] = []
        start = 0
        while start < len(text):
            body = text[start : start + window]
            if self.max_input_bytes:
                body = body.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore")
            while body and self.max_input_tokens is not None:
                assert self._tokenizer is not None
                encoding = self._tokenizer.encode(
                    prefix + body,
                    add_special_tokens=self.tokenizer_add_special_tokens,
                )
                budget = self.max_input_tokens - self.reserved_tokens
                count = len(encoding)
                if count <= budget:
                    # Reuse this exact validation, and adapt the next work
                    # window to observed density. Dense inputs should not pay
                    # for eight contexts of tokenization on every emitted span.
                    body_tokens = max(1, count - prefix_tokens)
                    usable = max(1, budget - prefix_tokens)
                    window = min(max_window, max(1, len(body) * usable // body_tokens))
                    break
                end = self._token_boundary(encoding, len(prefix), len(body), budget)
                # Never binary-search token counts: boundary merges can make
                # them nonmonotonic. Re-encode actual token-boundary slices.
                body = body[:end] if end else body[: len(body) // 2]
            if not body:
                self._validate_formatted(
                    prefix + text[start : start + 1], index=len(spans), role=role
                )
                raise ValueError("Embedding limits cannot fit a source character after the prefix")
            # Character and byte constraints bounded the slice above; the
            # successful encoding already checked token limits exactly.
            end = start + len(body)
            spans.append(EmbeddingSpan(start, end, body))
            start = end
        return spans

    def _formatting_config(self) -> dict:
        return {
            "profile": self.profile,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "max_input_bytes": self.max_input_bytes,
            "max_input_tokens": self.max_input_tokens,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tokenizer_add_special_tokens": self.tokenizer_add_special_tokens,
            "reserved_tokens": self.reserved_tokens,
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
                # A diagnostic sentence can exceed an intentionally small input
                # limit even when ordinary short documents are perfectly valid.
                await self.embed_single("x")
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
            batch_start: int,
            batch: list[str],
            semaphore: asyncio.Semaphore,
        ) -> tuple[int, list[list[float]]]:
            nonlocal completed
            result = await self._embed_batch_with_semaphore(batch_start, batch, semaphore)
            completed += sum(weights[batch_start : batch_start + len(batch)])
            if progress_cb:
                progress_cb(completed, callback_total)
            return result

        # Create batches with their indices
        batches = self._request_batches(texts)

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
        max_text_chars: int | None = None,
        tokenizer_add_special_tokens: bool | None = None,
        reserved_tokens: int | None = None,
        max_request_bytes: int | None = None,
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
            max_text_chars=max_text_chars,
            tokenizer_add_special_tokens=tokenizer_add_special_tokens,
            reserved_tokens=reserved_tokens,
            max_request_bytes=max_request_bytes,
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

    def split_text(
        self,
        text: str,
        *,
        role: EmbeddingRole = "document",
        context_prefix: str = "",
    ) -> list[EmbeddingSpan]:
        return self._client.split_text(text, role=role, context_prefix=context_prefix)

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
