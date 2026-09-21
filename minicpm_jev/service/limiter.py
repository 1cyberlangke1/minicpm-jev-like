"""请求队列上限: 同时在飞的请求数超过 ``queue_max`` 立刻 429, 不排队等待.

用 ``asyncio.Lock`` + 计数器实现 (几十行, 不值得引第三方队列库): 判断与自增在锁里做,
``slot()`` 是异步上下文管理器, 异常路径也一定归还名额。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .errors import RateLimitedError

__all__ = ["RequestLimiter"]


class RequestLimiter:
    """同时在飞请求数上限."""

    def __init__(self, limit: int, *, retry_after: int = 1) -> None:
        """输入: limit -- 上限 (>= 1); retry_after -- 429 里建议的重试秒数."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        self._limit = limit
        self._retry_after = retry_after
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        """上限."""
        return self._limit

    @property
    def active(self) -> int:
        """当前在飞请求数."""
        return self._active

    async def acquire(self) -> None:
        """占一个名额.

        输入/输出: 无;
        预期: 已满时抛 RateLimitedError (429 + Retry-After), 不阻塞等待。
        """
        async with self._lock:
            if self._active >= self._limit:
                raise RateLimitedError(
                    f"Too many requests in flight (limit {self._limit}).",
                    retry_after=self._retry_after,
                )
            self._active += 1

    async def release(self) -> None:
        """归还一个名额 (多还也不会把计数压成负数)."""
        async with self._lock:
            if self._active > 0:
                self._active -= 1

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """占名额的上下文管理器; 退出时必归还 (异常也归还)."""
        await self.acquire()
        try:
            yield
        finally:
            await self.release()
