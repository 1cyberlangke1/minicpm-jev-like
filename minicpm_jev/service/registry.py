"""本地模型清单: 扫 weights/ 下的 .gguf, 名字就是文件主干.

- ``/v1/models`` 直接列这份清单 (名称 / 描述 / 发布日期), 不联网、不列远端;
- 别名表在这里校验: 目标必须真实存在, 别名不能与真实模型名撞名 —— 不合法启动即报错;
- 请求里的 ``model`` 先过别名表再精确匹配, 未知模型抛 UsageError(400)。

注意: 这里**没有别名层**的默认行为 —— 请求什么模型名就是什么模型, 别名只有配置里
显式写了才生效。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import ConfigError
from .errors import UsageError

__all__ = ["ModelInfo", "ModelRegistry"]

#: 量化标记的形状: Q4_K_M / F16 之类
QUANT_PATTERN = re.compile(r"^[QF][0-9][A-Z0-9_]*$")


@dataclass(frozen=True)
class ModelInfo:
    """一个本地模型."""

    name: str
    path: Path
    size_bytes: int
    release_date: str

    @property
    def quant(self) -> str | None:
        """从文件名里认量化标记 (认不出就是 None)."""
        for part in self.name.split("-"):
            if QUANT_PATTERN.match(part):
                return part
        return None

    @property
    def description(self) -> str:
        """给 /v1/models 用的描述 (量化 + 体积), 与官方一样是英文."""
        quant = self.quant or "unknown quant"
        mebibytes = self.size_bytes / 1024 / 1024
        return f"Local GGUF model ({quant}), {mebibytes:.0f} MiB"

    def to_dict(self) -> dict[str, str]:
        """官方 /v1/models 的条目形状."""
        return {
            "name": self.name,
            "description": self.description,
            "release_date": self.release_date,
        }


class ModelRegistry:
    """本地模型清单 + 别名解析."""

    def __init__(self, weights_dir: Path, aliases: Mapping[str, str] | None = None) -> None:
        """输入: weights_dir -- 权重目录; aliases -- 可选的 别名 -> 真实模型名 映射."""
        self._weights_dir = Path(weights_dir)
        self._models = self._scan()
        self._by_name = {info.name: info for info in self._models}
        self._aliases = self._validate_aliases(dict(aliases or {}))

    def _scan(self) -> tuple[ModelInfo, ...]:
        """扫目录里的 .gguf. 输出: 按文件名排序的模型信息; 目录不存在就给空表."""
        if not self._weights_dir.is_dir():
            return ()
        infos: list[ModelInfo] = []
        for path in sorted(self._weights_dir.glob("*.gguf")):
            stat = path.stat()
            release = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            infos.append(
                ModelInfo(
                    name=path.stem,
                    path=path,
                    size_bytes=stat.st_size,
                    release_date=release,
                )
            )
        return tuple(infos)

    def _validate_aliases(self, aliases: dict[str, str]) -> dict[str, str]:
        """校验别名表 (存在性 + 撞名). 预期: 不合法抛 ConfigError, 启动期就炸."""
        for alias, target in aliases.items():
            if target not in self._by_name:
                raise ConfigError(f"别名 {alias!r} 指向不存在的模型: {target}")
            if alias in self._by_name:
                raise ConfigError(f"别名 {alias!r} 与真实模型名撞名")
        return aliases

    @property
    def weights_dir(self) -> Path:
        """权重目录."""
        return self._weights_dir

    def names(self) -> tuple[str, ...]:
        """本地真实模型名 (不含别名)."""
        return tuple(info.name for info in self._models)

    def list(self) -> list[ModelInfo]:
        """清单快照."""
        return list(self._models)

    def resolve(self, requested: str) -> str:
        """把请求里的 model 解析成真实模型名.

        输入: requested -- 请求里的名字 (可能是别名);
        输出: 真实模型名;
        预期: 未知模型抛 UsageError(400); 不静默换成别的模型。
        """
        target = self._aliases.get(requested, requested)
        if target not in self._by_name:
            raise UsageError(f"Unknown model: {requested}")
        return target

    def path_of(self, name: str) -> Path:
        """真实模型名 -> 权重路径. 预期: 未知名字抛 UsageError."""
        info = self._by_name.get(name)
        if info is None:
            raise UsageError(f"Unknown model: {name}")
        return info.path
