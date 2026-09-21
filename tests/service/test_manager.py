"""单模型驻留: 复用 / 切换 / 过渡态标记 / 失败后不留旧模型."""

import asyncio
from pathlib import Path

import pytest

from minicpm_jev.config import Settings
from minicpm_jev.service import EngineManager, ModelRegistry, UsageError


class _StubEngine:
    """引擎替身: 管理器只用 close 这一个接口."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _registry(tmp_path: Path) -> ModelRegistry:
    """造两个假模型. 输入: 临时目录; 输出: 清单."""
    (tmp_path / "model-a.gguf").write_bytes(b"a")
    (tmp_path / "model-b.gguf").write_bytes(b"b")
    return ModelRegistry(tmp_path)


def test_same_model_is_loaded_once(tmp_path: Path) -> None:
    """同一模型重复 ensure 只构造一次."""
    created: list[str] = []
    registry = _registry(tmp_path)

    def factory(config: object) -> _StubEngine:
        name = Path(config.model_path).stem  # type: ignore[attr-defined]
        created.append(name)
        return _StubEngine(name)

    async def scenario() -> None:
        manager = EngineManager(Settings(), registry, engine_factory=factory)
        first = await manager.ensure("model-a")
        second = await manager.ensure("model-a")
        assert first is second
        assert manager.loaded_model == "model-a"
        await manager.aclose()

    asyncio.run(scenario())
    assert created == ["model-a"]


def test_switching_closes_old_engine(tmp_path: Path) -> None:
    """换模型时先卸旧的, 且新的不被顺手关掉."""
    engines: list[_StubEngine] = []
    registry = _registry(tmp_path)

    def factory(config: object) -> _StubEngine:
        name = Path(config.model_path).stem  # type: ignore[attr-defined]
        engine = _StubEngine(name)
        engines.append(engine)
        return engine

    async def scenario() -> None:
        manager = EngineManager(Settings(), registry, engine_factory=factory)
        await manager.ensure("model-a")
        await manager.ensure("model-b")
        assert manager.loaded_model == "model-b"
        assert engines[0].closed is True
        assert engines[1].closed is False
        await manager.aclose()

    asyncio.run(scenario())
    assert engines[1].closed is True




def test_failed_load_leaves_no_model(tmp_path: Path) -> None:
    """装载失败就停在无模型状态, 不偷偷退回旧模型."""
    registry = _registry(tmp_path)
    calls: list[str] = []

    def factory(config: object) -> _StubEngine:
        name = Path(config.model_path).stem  # type: ignore[attr-defined]
        calls.append(name)
        if name == "model-b":
            raise RuntimeError("boom")
        return _StubEngine(name)

    async def scenario() -> None:
        manager = EngineManager(Settings(), registry, engine_factory=factory)
        await manager.ensure("model-a")
        with pytest.raises(RuntimeError):
            await manager.ensure("model-b")
        assert manager.loaded_model is None
        assert manager.is_switching is False
        await manager.aclose()

    asyncio.run(scenario())
    assert calls == ["model-a", "model-b"]




def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    """清单里没有的模型不许加载."""
    registry = _registry(tmp_path)

    async def scenario() -> None:
        manager = EngineManager(
            Settings(), registry, engine_factory=lambda config: _StubEngine("x")
        )
        with pytest.raises(UsageError):
            await manager.ensure("nope")
        await manager.aclose()

    asyncio.run(scenario())
