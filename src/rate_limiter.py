"""Client-side request pacing and event-loop-safe asyncio primitives."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class LoopLocal(Generic[T]):
    """Lazily create one object per running event loop.

    asyncio locks/semaphores bind to the loop they are first used on. The sync
    wrappers call ``asyncio.run`` repeatedly (CLI, Streamlit reruns), so each
    new loop gets a fresh primitive while plain state (e.g. the rate limiter's
    next-allowed timestamp) is kept across calls.
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._value: T | None = None

    def get(self) -> T:
        """Return the object bound to the currently running loop."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._value is None:
            self._loop = loop
            self._value = self._factory()
        return self._value


class RateLimiter:
    """Evenly spaces requests to at most ``rpm`` per minute (0 disables).

    Uses a "next allowed slot" schedule: each acquire reserves the next slot
    ``60 / rpm`` seconds after the previous one, which guarantees spacing even
    under concurrency. ``clock`` and ``sleep`` are injectable for tests.
    """

    def __init__(self, rpm: int, clock: Clock = time.monotonic, sleep: Sleeper = asyncio.sleep) -> None:
        if rpm < 0:
            raise ValueError("rpm must be >= 0")
        self.interval: float = 60.0 / rpm if rpm > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_slot: float | None = None
        self._lock: LoopLocal[asyncio.Lock] = LoopLocal(asyncio.Lock)

    async def acquire(self) -> float:
        """Wait for the next free slot; return the number of seconds waited."""
        if self.interval == 0.0:
            return 0.0
        async with self._lock.get():
            now = self._clock()
            slot = now if self._next_slot is None else max(now, self._next_slot)
            wait = slot - now
            self._next_slot = slot + self.interval
        if wait > 0:
            await self._sleep(wait)
        return wait
