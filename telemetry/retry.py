from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Awaitable, Callable, Iterable, Optional, Type, TypeVar

T = TypeVar("T")


def _compute_backoff(attempt: int, base_delay: float, factor: float, jitter: float) -> float:
    delay = base_delay * (factor ** attempt)
    if jitter:
        delay += random.uniform(0, jitter)
    return delay


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
    jitter: float = 0.1,
    retry_exceptions: Iterable[Type[BaseException]] = (Exception,),
) -> T:
    for attempt in range(retries):
        try:
            return fn()
        except tuple(retry_exceptions):
            if attempt >= retries - 1:
                raise
            time.sleep(_compute_backoff(attempt, base_delay, factor, jitter))
    return fn()


async def retry_async_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
    jitter: float = 0.1,
    retry_exceptions: Iterable[Type[BaseException]] = (Exception,),
) -> T:
    for attempt in range(retries):
        try:
            return await fn()
        except tuple(retry_exceptions):
            if attempt >= retries - 1:
                raise
            await asyncio.sleep(_compute_backoff(attempt, base_delay, factor, jitter))
    return await fn()
