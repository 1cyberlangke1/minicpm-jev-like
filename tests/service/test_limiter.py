"""在飞请求数上限: 超了立刻 429, 不排队等待."""

import asyncio

import pytest

from minicpm_jev.service import RateLimitedError, RequestLimiter


def test_limit_must_be_positive() -> None:
    """上限至少为 1."""
    with pytest.raises(ValueError):
        RequestLimiter(0)
    with pytest.raises(ValueError):
        RequestLimiter(-1)


def test_slot_counts_and_releases() -> None:
    """slot 进入 +1, 退出归零; 嵌套也算数."""

    async def scenario() -> None:
        limiter = RequestLimiter(2)
        assert limiter.active == 0
        async with limiter.slot():
            assert limiter.active == 1
            async with limiter.slot():
                assert limiter.active == 2
            assert limiter.active == 1
        assert limiter.active == 0

    asyncio.run(scenario())


def test_full_queue_raises_rate_limited() -> None:
    """满了立刻 429 (带 retry_after), 归还后又能占."""

    async def scenario() -> None:
        limiter = RequestLimiter(1)
        await limiter.acquire()
        with pytest.raises(RateLimitedError) as error:
            await limiter.acquire()
        assert error.value.status_code == 429
        assert error.value.retry_after >= 1
        await limiter.release()
        await limiter.acquire()
        await limiter.release()
        assert limiter.active == 0

    asyncio.run(scenario())


def test_slot_releases_when_body_raises() -> None:
    """请求体抛异常也必须归还名额, 否则名额会漏光."""

    async def scenario() -> None:
        limiter = RequestLimiter(1)
        with pytest.raises(RuntimeError):
            async with limiter.slot():
                raise RuntimeError("boom")
        assert limiter.active == 0

    asyncio.run(scenario())


def test_concurrent_acquires_are_capped() -> None:
    """并发抢名额时恰好 limit 个成功, 其余都是 429."""

    async def scenario() -> None:
        limiter = RequestLimiter(3)
        results = await asyncio.gather(
            *[limiter.acquire() for _ in range(10)], return_exceptions=True
        )
        granted = [item for item in results if item is None]
        refused = [item for item in results if isinstance(item, RateLimitedError)]
        assert len(granted) == 3
        assert len(refused) == 7
        assert limiter.active == 3
        for _ in range(3):
            await limiter.release()
        assert limiter.active == 0

    asyncio.run(scenario())


def test_release_is_idempotent() -> None:
    """多归还也不会把计数压成负数."""

    async def scenario() -> None:
        limiter = RequestLimiter(1)
        await limiter.release()
        await limiter.release()
        assert limiter.active == 0

    asyncio.run(scenario())
