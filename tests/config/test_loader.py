"""配置加载: 默认 < 文件 < 命令行, 以及各种非法输入."""

import json
from pathlib import Path

import pytest

from minicpm_jev.config import (
    DEFAULT_CONFIG_NAME,
    ConfigError,
    build_settings,
    load_config_file,
    load_settings,
    settings_from_cli,
)


def _write(tmp_path: Path, payload: object, name: str = DEFAULT_CONFIG_NAME) -> Path:
    """把 payload 写成 JSON 配置文件, 返回路径."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_config_file_reads_object(tmp_path: Path) -> None:
    """正常文件读成 dict."""
    path = _write(tmp_path, {"port": 9000, "chunk_size": 8})
    assert load_config_file(path) == {"port": 9000, "chunk_size": 8}




def test_load_config_file_rejects_bad_json(tmp_path: Path) -> None:
    """非法 JSON 直接报错, 不回退默认."""
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config_file(path)
    assert "JSON" in str(error.value)




def test_load_config_file_rejects_unknown_key(tmp_path: Path) -> None:
    """未知键说明写错了配置, 直接报错并列出键名."""
    path = _write(tmp_path, {"port": 8000, "chunk_sise": 128})
    with pytest.raises(ConfigError) as error:
        load_config_file(path)
    assert "chunk_sise" in str(error.value)




def test_build_settings_file_then_overrides() -> None:
    """覆盖优先于文件, 文件优先于默认."""
    settings = build_settings(
        {"port": 9000, "chunk_size": 8, "host": "0.0.0.0"},
        {"port": 9100},
    )
    assert settings.port == 9100
    assert settings.chunk_size == 8
    assert settings.host == "0.0.0.0"






def test_load_settings_uses_cwd_config_json(tmp_path: Path) -> None:
    """没给 --config 时, 工作目录里的 config.json 自动生效."""
    _write(tmp_path, {"port": 9200})
    settings = load_settings(cwd=tmp_path)
    assert settings.port == 9200












def test_settings_from_cli_config_file_and_override(tmp_path: Path) -> None:
    """--config 指定文件, 命令行再覆盖单个字段."""
    path = _write(tmp_path, {"port": 9600, "chunk_size": 32})
    settings = settings_from_cli(["--config", str(path), "--chunk-size", "64"])
    assert settings.port == 9600
    assert settings.chunk_size == 64




