"""Retry utilities with exponential backoff.

Provides decorators and helpers for retrying operations that may fail
due to transient errors (network issues, service unavailability, etc.).

Usage:
    @async_retry(
        max_retries=3,
        retry_exceptions=(ConnectionError, TimeoutError),
        initial_delay=1.0,
        max_delay=8.0,
    )
    async def fetch_data():
        ...
"""

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def async_retry(
    max_retries: int = 3,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    initial_delay: float = 1.0,
    max_delay: float = 8.0,
    backoff_factor: float = 2.0,
    log_retries: bool = True,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator for async functions to retry on specified exceptions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (total calls = max_retries + 1)
        retry_exceptions: Tuple of exception types to retry on
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries (caps exponential growth)
        backoff_factor: Multiplier for delay after each retry
        log_retries: Whether to log retry attempts

    Returns:
        Decorated async function that retries on specified exceptions

    Example:
        @async_retry(max_retries=3, retry_exceptions=(ConnectionError,))
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com/data")
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: Exception | None = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        if log_retries:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} for {func.__name__}: "
                                f"{type(e).__name__}: {e}. Waiting {delay:.1f}s..."
                            )
                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Last attempt failed
                        if log_retries:
                            logger.error(
                                f"All {max_retries} retries exhausted for {func.__name__}: "
                                f"{type(e).__name__}: {e}"
                            )
                        raise

            # Should not reach here, but for type safety
            if last_error:
                raise last_error
            raise RetryExhaustedError(f"Unexpected: no error captured for {func.__name__}")

        return wrapper

    return decorator


async def retry_operation(
    operation: Callable[[], Awaitable[R]],
    max_retries: int = 3,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    initial_delay: float = 1.0,
    max_delay: float = 8.0,
    backoff_factor: float = 2.0,
    operation_name: str = "operation",
) -> R:
    """
    Execute an async operation with retry logic and exponential backoff.

    Args:
        operation: Async callable to execute
        max_retries: Maximum number of retry attempts
        retry_exceptions: Tuple of exception types to retry on
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay after each retry
        operation_name: Name for logging purposes

    Returns:
        Result of the operation

    Raises:
        The last exception if all retries are exhausted

    Example:
        result = await retry_operation(
            lambda: client.fetch_data(),
            max_retries=3,
            retry_exceptions=(ConnectionError, TimeoutError),
            operation_name="fetch_data"
        )
    """
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retry_exceptions as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {operation_name}: "
                    f"{type(e).__name__}: {e}. Waiting {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(
                    f"All {max_retries} retries exhausted for {operation_name}: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    # Should not reach here
    if last_error:
        raise last_error
    raise RetryExhaustedError(f"Unexpected: no error captured for {operation_name}")
