"""``python -m minicpm_jev.service`` 启动本地服务.

配置来源: 内置默认 < config.json < 命令行参数 (见 minicpm_jev.config.loader)。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import uvicorn

from ..config import settings_from_cli
from .app import create_app

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """按命令行配置起服务.

    输入: argv -- 参数序列 (None 取 sys.argv[1:]);
    输出: 进程退出码 (uvicorn 正常退出给 0);
    预期: 配置不合法直接抛错, 不会带着坏配置半启动。
    """
    settings = settings_from_cli(argv)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
