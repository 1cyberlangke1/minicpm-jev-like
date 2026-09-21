"""启动配置: 数据类 + 加载优先级.

- settings: 全部启动字段与构造期校验, 附带转成 EngineConfig 的出口;
- loader: 内置默认 < config.json < 命令行参数, 未知键/越界值一律启动即报错。
"""

from .loader import (
    DEFAULT_CONFIG_NAME,
    build_parser,
    build_settings,
    load_config_file,
    load_settings,
    settings_from_cli,
)
from .settings import ConfigError, Settings

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "ConfigError",
    "Settings",
    "build_parser",
    "build_settings",
    "load_config_file",
    "load_settings",
    "settings_from_cli",
]
