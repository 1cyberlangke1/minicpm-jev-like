"""服务启动配置 (纯数据 + 构造期校验).

字段与 PLAN 第 7 节的配置表一一对应。校验全部在构造时做完, 非法值不进运行期:

- ``n_ctx`` 走 runtime.device 的边界 (默认 32k, 上限 131072);
- ``chunk_size`` 走 chunking 的边界 ((0, 128]);
- 端口 / 队列长度 / 批大小必须为正, 别名表必须是 别名 -> 真实模型名 的字符串映射。

语言约定: 启动期错误用中文 (永远到不了 HTTP); 请求期错误用英文 (见 decision /
runtime)。这样既能照抄 PLAN 决策记录 4 的 ``n_ctx=200000 超过上限 131072``,
又守住了「对外错误消息一律英文」。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE
from ..runtime import DEFAULT_N_CTX, Device, EngineConfig, validate_n_ctx

__all__ = ["ConfigError", "Settings"]

#: 合法端口区间
MAX_PORT = 65535
#: 默认监听地址 (只听本机)
DEFAULT_HOST = "127.0.0.1"
#: 默认监听端口
DEFAULT_PORT = 8000
#: 默认请求队列上限
DEFAULT_QUEUE_MAX = 30


class ConfigError(ValueError):
    """启动配置不合法 (缺字段 / 类型错 / 越界 / 未知键)。"""


def _require_positive_int(value: object, name: str) -> int:
    """校验一个正整数.

    输入: value -- 待校验值; name -- 字段名 (出错文案用);
    输出: 原值;
    预期: 非整数 (含 bool) 或 <= 0 抛 ConfigError, 不做隐式转换。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} 必须是整数, 收到 {type(value).__name__}")
    if value <= 0:
        raise ConfigError(f"{name} 必须为正, 收到 {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """服务启动配置.

    ``model_path`` 为 None 表示启动时不预加载模型, 等第一个请求按名字加载。
    ``api_key`` 为 None / 空串表示不鉴权。
    """

    model_path: Path | None = None
    device: Device = Device.GPU
    n_ctx: int = DEFAULT_N_CTX
    chunk_size: int = DEFAULT_CHUNK_SIZE
    n_seq_max: int = 32
    n_batch: int = 2048
    n_ubatch: int = 512
    n_threads: int | None = None
    queue_max: int = DEFAULT_QUEUE_MAX
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    api_key: str | None = None
    model_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path))

        if not isinstance(self.device, Device):
            try:
                object.__setattr__(self, "device", Device(self.device))
            except ValueError as error:
                raise ConfigError(
                    f"device 只能是 'gpu' 或 'cpu', 收到 {self.device!r}"
                ) from error

        validate_n_ctx(self.n_ctx)
        if not 1 <= self.chunk_size <= MAX_CHUNK_SIZE:
            raise ConfigError(
                f"chunk_size 必须在 (0, {MAX_CHUNK_SIZE}] 内, 收到 {self.chunk_size}"
            )
        for name in ("n_seq_max", "n_batch", "n_ubatch", "queue_max"):
            _require_positive_int(getattr(self, name), name)
        if self.n_ubatch > self.n_batch:
            raise ConfigError(
                f"n_ubatch({self.n_ubatch}) 不能大于 n_batch({self.n_batch})"
            )
        if self.n_threads is not None:
            _require_positive_int(self.n_threads, "n_threads")

        if not isinstance(self.host, str) or not self.host:
            raise ConfigError(f"host 必须是非空字符串, 收到 {self.host!r}")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ConfigError(f"port 必须是整数, 收到 {type(self.port).__name__}")
        if not 1 <= self.port <= MAX_PORT:
            raise ConfigError(f"port 必须在 1~{MAX_PORT} 内, 收到 {self.port}")

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise ConfigError(
                    f"api_key 必须是字符串或 null, 收到 {type(self.api_key).__name__}"
                )
            if not self.api_key:
                object.__setattr__(self, "api_key", None)

        object.__setattr__(self, "model_aliases", _validated_aliases(self.model_aliases))

    def to_engine_config(self, model_path: Path | None = None) -> EngineConfig:
        """转成引擎构造参数.

        输入: model_path -- 覆盖启动默认模型 (按请求切模型时用);
        输出: EngineConfig;
        预期: 两者都没给模型路径时抛 ConfigError, 不去猜 weights 目录。
        """
        resolved = model_path if model_path is not None else self.model_path
        if resolved is None:
            raise ConfigError("未指定模型路径, 无法构造引擎")
        return EngineConfig(
            model_path=Path(resolved),
            device=self.device,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_ubatch=self.n_ubatch,
            n_seq_max=self.n_seq_max,
            n_threads=self.n_threads,
        )


def _validated_aliases(aliases: object) -> dict[str, str]:
    """校验别名表: 别名 -> 真实模型名, 两边都必须是非空字符串.

    输入: aliases -- 待校验的映射;
    输出: 普通 dict (冻结数据类里存可变对象, 这里至少保证形状正确);
    预期: 非映射 / 空键 / 空值 / 自己映射自己 都抛 ConfigError; 存在性由服务层查。
    """
    if aliases is None:
        return {}
    if not isinstance(aliases, Mapping):
        raise ConfigError("model_aliases 必须是 别名 -> 真实模型名 的对象")
    result: dict[str, str] = {}
    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias:
            raise ConfigError("model_aliases 的别名必须是非空字符串")
        if not isinstance(target, str) or not target:
            raise ConfigError(f"model_aliases[{alias!r}] 的目标必须是非空字符串")
        if alias == target:
            raise ConfigError(f"model_aliases[{alias!r}] 指向自己, 没有意义")
        result[alias] = target
    return result
