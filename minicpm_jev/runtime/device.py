"""设备档位与上下文长度契约.

- GPU 主力: ``n_gpu_layers=-1``;
- CPU 兜底: ``n_gpu_layers=0`` + 进程级 ``CUDA_VISIBLE_DEVICES=-1`` —— Windows 下
  空串 ``""`` 会被环境机制整个丢弃, 等于没屏蔽, 必须写 ``-1``;
- 同一进程只允许一种设备档位, 混用直接抛 ``EngineError``, 不静默降级;
- ``n_ctx`` 默认 32768, 上限 131072, 超上限报错并把上限写进文案.

进程级屏蔽的理由: GPU 可见时本构建即使 0 层 offload 仍会把部分计算调度到 CUDA0,
跨调用不再可复现, 所以 CPU 档必须在加载模型前就把设备藏起来。
"""

from __future__ import annotations

import os
from enum import Enum

__all__ = [
    "DEFAULT_N_CTX",
    "MAX_N_CTX",
    "Device",
    "EngineError",
    "acquire_device",
    "reset_device_lock",
    "validate_n_ctx",
]

#: 默认上下文长度 (KV cell 总数)
DEFAULT_N_CTX = 32768
#: 上下文长度上限 (128K)
MAX_N_CTX = 131072


class EngineError(RuntimeError):
    """引擎契约被违反: 批次超限 / 设备冲突 / 标签数量不匹配 / 取不到 logits / 上下文越界."""


class Device(str, Enum):
    """推理设备档位."""

    GPU = "gpu"
    CPU = "cpu"


_ACTIVE_DEVICE: Device | None = None


def acquire_device(device: Device) -> None:
    """锁定进程级设备档位, 一个进程只允许一种.

    输入: device -- 目标设备;
    输出: 无;
    预期: CPU 档在加载模型前先把 ``CUDA_VISIBLE_DEVICES`` 置 ``-1``; 已锁定别的档位
          时抛 EngineError, 消息用英文 (可能冒泡到 HTTP body).
    """
    global _ACTIVE_DEVICE
    if _ACTIVE_DEVICE is None:
        if device is Device.CPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        _ACTIVE_DEVICE = device
        return
    if _ACTIVE_DEVICE is not device:
        raise EngineError(
            f"engine already locked to {_ACTIVE_DEVICE.value} in this process; "
            f"cannot create a {device.value} engine"
        )


def reset_device_lock() -> None:
    """释放进程级档位锁 (并把 CPU 档设过的环境变量还原).

    输入/输出: 无;
    预期: 仅供测试使用 —— 生产路径不允许一个进程中途换档位. 测试用完必须把锁恢复成
          进入前的档位, 否则同进程后续用例会被误伤.
    """
    global _ACTIVE_DEVICE
    if _ACTIVE_DEVICE is Device.CPU:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _ACTIVE_DEVICE = None


def validate_n_ctx(n_ctx: int) -> int:
    """校验上下文长度.

    输入: n_ctx -- 期望的 KV cell 总数;
    输出: 原值 (便于链式赋值);
    预期: 非整数 / 小于 1 / 超过上限抛 EngineError; 超上限的文案带上收到的值与上限值.
    """
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int):
        raise EngineError(f"n_ctx must be an integer, got {type(n_ctx).__name__}")
    if n_ctx < 1:
        raise EngineError(f"n_ctx must be >= 1, got {n_ctx}")
    if n_ctx > MAX_N_CTX:
        raise EngineError(f"n_ctx={n_ctx} 超过上限 {MAX_N_CTX}")
    return n_ctx
