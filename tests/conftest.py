"""共享测试夹具: 一个会话级的批量推理引擎 (模型只加载一次)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from minicpm_jev import BatchEngine, EngineConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"


@pytest.fixture(scope="session")
def engine() -> Iterator[BatchEngine]:
    """会话级 GPU 引擎: 加载 1.5GB GGUF 有点慢, 全测试共用一份."""
    with BatchEngine(EngineConfig(model_path=MODEL_PATH)) as instance:
        yield instance
