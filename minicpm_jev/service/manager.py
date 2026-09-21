"""单模型驻留的引擎管理器.

同一时刻只加载一个模型推理: 请求的模型与当前不一致时, 先卸载再加载, 全程串行化
(``asyncio.Lock``); 装载丢到线程里做, 因为 llama.cpp 会阻塞事件循环。切换期间
``is_switching`` 为真, 服务层据此回 529。

引擎工厂是唯一的测试缝: 默认就是真的 ``BatchEngine``, 测试可以注入「慢一点但仍
构造真引擎」的包装来验证并发语义。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from ..config import Settings
from ..runtime import BatchEngine, EngineConfig
from .registry import ModelRegistry

__all__ = ["EngineFactory", "EngineLike", "EngineManager"]


class EngineLike(Protocol):
    """管理器用到的引擎最小接口 (只有 close)."""

    def close(self) -> None:
        """释放底层资源."""


#: 引擎工厂: EngineConfig -> 引擎 (默认是真 BatchEngine)
EngineFactory = Callable[[EngineConfig], EngineLike]


class EngineManager:
    """同一时刻只驻留一个模型."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """输入: settings -- 启动配置; registry -- 模型清单; engine_factory -- 可替换的工厂."""
        self._settings = settings
        self._registry = registry
        self._factory: EngineFactory = engine_factory or BatchEngine
        self._lock = asyncio.Lock()
        self._engine: EngineLike | None = None
        self._loaded: str | None = None
        self._switching = False

    @property
    def loaded_model(self) -> str | None:
        """当前驻留的模型名 (没加载就是 None)."""
        return self._loaded

    @property
    def is_switching(self) -> bool:
        """是否正在卸载 / 装载."""
        return self._switching

    def _close_current(self) -> None:
        """关掉当前引擎并清状态 (线程里调用)."""
        engine = self._engine
        self._engine = None
        self._loaded = None
        if engine is not None:
            engine.close()

    def _load(self, model: str) -> EngineLike:
        """构造引擎 (线程里调用). 输入: 真实模型名; 输出: 新引擎."""
        config = self._settings.to_engine_config(self._registry.path_of(model))
        return self._factory(config)

    async def ensure(self, model: str) -> EngineLike:
        """确保请求的模型处于加载状态, 返回可用引擎.

        输入: model -- 真实模型名 (已过别名解析);
        输出: 引擎;
        预期: 已是目标模型就直接复用; 否则卸载旧模型 -> 装载新模型 (串行, 不并发切换);
              装载失败时保持「无模型」状态, 不偷偷退回旧模型。
        """
        if self._engine is not None and self._loaded == model:
            return self._engine
        async with self._lock:
            if self._engine is not None and self._loaded == model:
                return self._engine
            self._switching = True
            try:
                await asyncio.to_thread(self._close_current)
                engine = await asyncio.to_thread(self._load, model)
            finally:
                self._switching = False
            self._engine = engine
            self._loaded = model
            return engine

    async def preload(self) -> None:
        """启动时预加载 settings.model_path 指定的模型 (没配就什么都不做)."""
        path = self._settings.model_path
        if path is None:
            return
        name = path.stem
        if name in self._registry.names():
            await self.ensure(name)

    async def aclose(self) -> None:
        """关闭当前引擎 (幂等, 可在 lifespan 收尾时反复调)."""
        async with self._lock:
            self._switching = True
            try:
                await asyncio.to_thread(self._close_current)
            finally:
                self._switching = False
