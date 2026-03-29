"""Structured logging with correlation ID support.

This module provides utilities for consistent logging across MCP servers:

1. Correlation IDs - Track requests across async operations
2. Operation timing - Performance metrics at DEBUG level
3. MCP request logging - Decorator for tool functions

Usage:
    from vector_core.logging import get_logger, mcp_logged, operation_timer

    logger = get_logger(__name__)

    @mcp.tool()
    @mcp_logged
    async def my_tool(query: str) -> dict:
        with operation_timer(logger, "database query"):
            result = await db.query(query)
        return result
"""

import contextvars
import logging
import time
import uuid
from collections.abc import Awaitable
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

# Context variable for correlation ID (async/thread safe)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(cid: str | None = None) -> str:
    """Set correlation ID in context.

    Args:
        cid: Correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = uuid.uuid4().hex[:12]  # Short form for readability
    _correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Clear correlation ID from context."""
    _correlation_id.set(None)


class CorrelationLogAdapter(logging.LoggerAdapter):
    """Logger adapter that prepends correlation ID to all messages."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Add correlation ID prefix to log message."""
        cid = get_correlation_id()
        if cid:
            msg = f"[{cid}] {msg}"
        return msg, kwargs


def get_logger(name: str) -> CorrelationLogAdapter:
    """Get a logger with correlation ID support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        CorrelationLogAdapter wrapping the named logger
    """
    return CorrelationLogAdapter(logging.getLogger(name), {})


@contextmanager
def operation_timer(
    logger: logging.Logger | CorrelationLogAdapter,
    operation: str,
    level: int = logging.DEBUG,
):
    """Context manager for timing operations.

    Logs operation duration at the specified level (default DEBUG).

    Args:
        logger: Logger to use for output
        operation: Description of the operation being timed
        level: Logging level (default DEBUG)

    Usage:
        with operation_timer(logger, "embedding generation"):
            embeddings = await embed(texts)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log(level, f"{operation} completed in {elapsed_ms:.1f}ms")


class OperationMetrics:
    """Collect timing metrics for multi-step operations.

    Useful for breaking down latency across pipeline stages.

    Usage:
        metrics = OperationMetrics("code_search")
        metrics.start_step("embedding")
        embeddings = await embed(query)
        metrics.start_step("search")
        results = await search(embeddings)
        metrics.end_step()
        logger.debug(f"Metrics: {metrics.to_dict()}")
    """

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.timings: dict[str, float] = {}
        self._start: float | None = None
        self._current_step: str | None = None

    def start_step(self, step: str) -> None:
        """Start timing a new step (ends previous step if any)."""
        if self._start is not None and self._current_step:
            self.timings[self._current_step] = (
                time.perf_counter() - self._start
            ) * 1000
        self._current_step = step
        self._start = time.perf_counter()

    def end_step(self) -> None:
        """End the current step."""
        if self._start is not None and self._current_step:
            self.timings[self._current_step] = (
                time.perf_counter() - self._start
            ) * 1000
        self._start = None
        self._current_step = None

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a dictionary."""
        return {
            "operation": self.operation_name,
            "timings_ms": self.timings.copy(),
            "total_ms": sum(self.timings.values()),
        }


P = ParamSpec("P")
T = TypeVar("T")


def mcp_logged(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T]]:
    """Decorator to log MCP tool requests and responses.

    Adds:
    - Correlation ID generation
    - Request logging with truncated args
    - Response logging with success/error status
    - Duration timing

    Usage:
        @mcp.tool()
        @mcp_logged
        async def search(query: str) -> dict:
            ...
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        set_correlation_id()  # Sets context for CorrelationLogAdapter
        logger = get_logger(func.__module__)

        # Truncate large args for logging (avoid spamming logs)
        safe_args: dict[str, Any] = {}
        for k, v in kwargs.items():
            str_v = str(v)
            if len(str_v) > 100:
                safe_args[k] = str_v[:100] + "..."
            else:
                safe_args[k] = v

        logger.info(f"MCP request: {func.__name__} args={safe_args}")

        try:
            result = await func(*args, **kwargs)

            # Check if result indicates error (dict with "error_code" key)
            success = not (isinstance(result, dict) and "error_code" in result)

            elapsed = (time.perf_counter() - start) * 1000
            status = "ok" if success else "error"
            logger.info(
                f"MCP response: {func.__name__} "
                f"status={status} duration={elapsed:.1f}ms"
            )
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                f"MCP error: {func.__name__} "
                f"duration={elapsed:.1f}ms error={type(e).__name__}: {e}"
            )
            raise

        finally:
            # Don't clear correlation ID - it might be used by nested calls
            pass

    return wrapper


def log_mcp_request(
    logger: logging.Logger | CorrelationLogAdapter,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """Log MCP tool request and set correlation ID.

    For manual logging when decorator can't be used.

    Args:
        logger: Logger to use
        tool_name: Name of the MCP tool
        args: Tool arguments

    Returns:
        The correlation ID that was set
    """
    cid = set_correlation_id()
    # Truncate large args
    safe_args = {
        k: str(v)[:100] + "..." if len(str(v)) > 100 else v
        for k, v in args.items()
    }
    logger.info(f"[{cid}] MCP request: {tool_name} args={safe_args}")
    return cid


def log_mcp_response(
    logger: logging.Logger | CorrelationLogAdapter,
    tool_name: str,
    success: bool,
    elapsed_ms: float,
) -> None:
    """Log MCP tool response.

    For manual logging when decorator can't be used.

    Args:
        logger: Logger to use
        tool_name: Name of the MCP tool
        success: Whether the operation succeeded
        elapsed_ms: Duration in milliseconds
    """
    cid = get_correlation_id()
    status = "ok" if success else "error"
    logger.info(
        f"[{cid}] MCP response: {tool_name} "
        f"status={status} duration={elapsed_ms:.1f}ms"
    )
