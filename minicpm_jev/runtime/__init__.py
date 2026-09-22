"""推理运行时: 设备档位与批量引擎.

- device: 设备档位 (GPU 主力 / CPU 兜底) 与 n_ctx 边界;
- engine: 一次 llama_decode 处理整批序列, 只在决策位读 logits.
"""

from .device import (
    DEFAULT_N_CTX,
    MAX_N_CTX,
    Device,
    EngineError,
    acquire_device,
    reset_device_lock,
    validate_n_ctx,
)
from .engine import BatchEngine, EngineConfig, ScoringEngine

__all__ = [
    "DEFAULT_N_CTX",
    "MAX_N_CTX",
    "BatchEngine",
    "Device",
    "EngineConfig",
    "EngineError",
    "ScoringEngine",
    "acquire_device",
    "reset_device_lock",
    "validate_n_ctx",
]
