"""配置加载: 内置默认 < config.json < 命令行参数.

- ``--config <path>`` 显式指定文件; 不指定时工作目录里有 config.json 就加载;
- 显式指定的文件不存在 / 不是合法 JSON / 不是对象 / 有未知键 -> 启动即报错;
- 命令行参数名与配置键同名 (``--model-path`` -> ``model_path``), 只写要覆盖的那几个;
- 一个参数都不传、也没有 config.json 时, 全部走内置默认 (服务能直接起来)。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..chunking import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE
from ..runtime import DEFAULT_N_CTX, MAX_N_CTX, Device
from .settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_QUEUE_MAX,
    ConfigError,
    Settings,
)

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "build_parser",
    "build_settings",
    "load_config_file",
    "load_settings",
    "settings_from_cli",
]

#: 未显式指定 --config 时, 工作目录里找这个文件名
DEFAULT_CONFIG_NAME = "config.json"

#: 配置文件允许出现的键 (与 Settings 字段一致, 外加 model_aliases)
FIELD_NAMES = (
    "model_path",
    "device",
    "n_ctx",
    "chunk_size",
    "n_seq_max",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "queue_max",
    "host",
    "port",
    "api_key",
    "model_aliases",
)


def load_config_file(path: Path) -> dict[str, Any]:
    """读一个 JSON 配置文件.

    输入: path -- 配置文件路径;
    输出: 解析后的键值表;
    预期: 文件不存在 / 非法 JSON / 顶层不是对象 / 出现未知键 -> ConfigError,
          不静默忽略任何东西。
    """
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"配置文件不是合法 JSON: {path} ({error})") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是对象: {path}")
    unknown = sorted(set(raw) - set(FIELD_NAMES))
    if unknown:
        raise ConfigError(f"配置文件出现未知键: {', '.join(unknown)}")
    return raw


def build_settings(
    config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """按「默认 < 配置文件 < 覆盖」拼出 Settings.

    输入: config -- 配置文件内容; overrides -- 命令行覆盖;
    输出: 校验过的 Settings;
    预期: 未知键 / 类型错 / 越界值抛 ConfigError (来自 Settings 构造), 不吞不补。
    """
    merged: dict[str, Any] = dict(config or {})
    for key, value in dict(overrides or {}).items():
        if key not in FIELD_NAMES:
            raise ConfigError(f"未知配置项: {key}")
        if value is not None:
            merged[key] = value
    if "model_path" in merged and merged["model_path"] is not None:
        merged["model_path"] = Path(str(merged["model_path"]))
    return Settings(**merged)


def load_settings(
    *,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
) -> Settings:
    """完整加载流程 (给服务入口用).

    输入: config_path -- 显式配置文件 (None 则看 cwd/config.json); overrides -- 命令行覆盖;
          cwd -- 找默认配置文件的位置 (默认当前目录);
    输出: Settings;
    预期: 显式给的路径一定要存在且合法; 没给且默认文件不存在时安静走默认值。
    """
    # 没给路径且默认文件不存在时 config 就是 None, 类型上要先把这条支路写全
    config: dict[str, Any] | None
    if config_path is not None:
        config = load_config_file(Path(config_path))
    else:
        candidate = (cwd or Path.cwd()) / DEFAULT_CONFIG_NAME
        config = load_config_file(candidate) if candidate.is_file() else None
    return build_settings(config, overrides)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器 (默认 SUPPRESS: 只记录真的传了的参数).

    输入/输出: argparse 解析器;
    预期: 所有配置键都有同名短横线长选项, 例如 --chunk-size -> chunk_size。
    """
    parser = argparse.ArgumentParser(
        prog="minicpm_jev",
        description="MiniCPM5 JEV 风格本地受限决策服务",
    )
    parser.add_argument("--config", default=None, help="JSON 配置文件路径")
    parser.add_argument("--model-path", default=argparse.SUPPRESS,
                        help="启动默认模型 (GGUF 路径)")
    parser.add_argument("--device", choices=[item.value for item in Device],
                        default=argparse.SUPPRESS)
    parser.add_argument("--n-ctx", type=int, default=argparse.SUPPRESS,
                        help=f"上下文长度, 默认 {DEFAULT_N_CTX}, 上限 {MAX_N_CTX}")
    parser.add_argument("--chunk-size", type=int, default=argparse.SUPPRESS,
                        help=f"候选分块窗口, (0, {MAX_CHUNK_SIZE}], 默认 {DEFAULT_CHUNK_SIZE}")
    parser.add_argument("--n-seq-max", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-batch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-ubatch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-threads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--queue-max", type=int, default=argparse.SUPPRESS,
                        help=f"请求队列上限, 默认 {DEFAULT_QUEUE_MAX}")
    parser.add_argument("--host", default=argparse.SUPPRESS,
                        help=f"监听地址, 默认 {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=argparse.SUPPRESS,
                        help=f"监听端口, 默认 {DEFAULT_PORT}")
    parser.add_argument("--api-key", default=argparse.SUPPRESS,
                        help="本地鉴权 key, 不传 = 不鉴权")
    return parser


def settings_from_cli(argv: Sequence[str] | None = None) -> Settings:
    """从命令行 (含可选 --config) 得到最终配置.

    输入: argv -- 参数序列 (None 取 sys.argv[1:]);
    输出: Settings;
    预期: --config 指定但文件不合法时直接报错; 其它参数按名覆盖配置文件。
    """
    namespace = build_parser().parse_args(argv)
    values = vars(namespace)
    config_path = values.pop("config", None)
    return load_settings(
        config_path=Path(config_path) if config_path else None,
        overrides=values,
    )
