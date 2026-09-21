"""服务入口: 命令行配置 -> uvicorn 接线 (不真起服务)."""

from pathlib import Path

import pytest

from minicpm_jev.config import ConfigError
from minicpm_jev.service.__main__ import main


def _patch_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """把 uvicorn.run 换成记录参数的假函数, 返回记录表."""
    recorded: dict[str, object] = {}

    def fake_run(
        app: object,
        host: str | None = None,
        port: int | None = None,
        log_level: str | None = None,
    ) -> None:
        recorded.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr("minicpm_jev.service.__main__.uvicorn.run", fake_run)
    return recorded


def test_main_wires_cli_into_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """命令行给的 host / port 必须原样交给 uvicorn, 且返回 0."""
    recorded = _patch_uvicorn(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert main(["--host", "127.0.0.1", "--port", "9123", "--queue-max", "7"]) == 0
    assert recorded["host"] == "127.0.0.1"
    assert recorded["port"] == 9123
    assert recorded["log_level"] == "info"
    assert recorded["app"] is not None




def test_main_rejects_bad_value_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """越界配置在起服务之前就报错, 不会半启动."""
    _patch_uvicorn(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        main(["--chunk-size", "999"])
