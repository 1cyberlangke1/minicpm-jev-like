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
from minicpm_jev.runtime import Device


def _write(tmp_path: Path, payload: object, name: str = DEFAULT_CONFIG_NAME) -> Path:
    """把 payload 写成 JSON 配置文件, 返回路径."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_config_file_reads_object(tmp_path: Path) -> None:
    """正常文件读成 dict."""
    path = _write(tmp_path, {"port": 9000, "chunk_size": 8})
    assert load_config_file(path) == {"port": 9000, "chunk_size": 8}


def test_load_config_file_rejects_missing_file(tmp_path: Path) -> None:
    """显式指定却不存在的文件直接报错."""
    with pytest.raises(ConfigError) as error:
        load_config_file(tmp_path / "nope.json")
    assert "不存在" in str(error.value)


def test_load_config_file_rejects_bad_json(tmp_path: Path) -> None:
    """非法 JSON 直接报错, 不回退默认."""
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config_file(path)
    assert "JSON" in str(error.value)


def test_load_config_file_rejects_non_object(tmp_path: Path) -> None:
    """顶层必须是对象."""
    path = _write(tmp_path, ["port", 8000])
    with pytest.raises(ConfigError):
        load_config_file(path)


def test_load_config_file_rejects_unknown_key(tmp_path: Path) -> None:
    """未知键说明写错了配置, 直接报错并列出键名."""
    path = _write(tmp_path, {"port": 8000, "chunk_sise": 128})
    with pytest.raises(ConfigError) as error:
        load_config_file(path)
    assert "chunk_sise" in str(error.value)


def test_build_settings_defaults_when_nothing_given() -> None:
    """什么都不给 = 内置默认."""
    settings = build_settings()
    assert settings.port == 8000
    assert settings.device is Device.GPU


def test_build_settings_file_then_overrides() -> None:
    """覆盖优先于文件, 文件优先于默认."""
    settings = build_settings(
        {"port": 9000, "chunk_size": 8, "host": "0.0.0.0"},
        {"port": 9100},
    )
    assert settings.port == 9100
    assert settings.chunk_size == 8
    assert settings.host == "0.0.0.0"


def test_build_settings_ignores_none_overrides() -> None:
    """命令行没传的参数是 None, 不能把文件里的值抹掉."""
    settings = build_settings({"port": 9000}, {"port": None, "host": None})
    assert settings.port == 9000


def test_build_settings_rejects_unknown_override() -> None:
    """未知覆盖键报错 (调用方拼错字段名要立刻发现)."""
    with pytest.raises(ConfigError) as error:
        build_settings({}, {"chunk_sise": 8})
    assert "chunk_sise" in str(error.value)


def test_load_settings_uses_cwd_config_json(tmp_path: Path) -> None:
    """没给 --config 时, 工作目录里的 config.json 自动生效."""
    _write(tmp_path, {"port": 9200})
    settings = load_settings(cwd=tmp_path)
    assert settings.port == 9200


def test_load_settings_falls_back_to_defaults(tmp_path: Path) -> None:
    """工作目录没有 config.json 时安静走默认 (一个参数都不传也能起服务)."""
    settings = load_settings(cwd=tmp_path)
    assert settings.port == 8000
    assert not (tmp_path / DEFAULT_CONFIG_NAME).exists()


def test_load_settings_explicit_path_must_exist(tmp_path: Path) -> None:
    """显式路径不存在时不能悄悄退回默认."""
    with pytest.raises(ConfigError):
        load_settings(config_path=tmp_path / "missing.json")


def test_load_settings_explicit_path_wins_over_cwd(tmp_path: Path) -> None:
    """显式路径优先于工作目录里的同名文件."""
    _write(tmp_path, {"port": 9300})
    explicit = _write(tmp_path, {"port": 9400}, name="other.json")
    settings = load_settings(config_path=explicit, cwd=tmp_path)
    assert settings.port == 9400


def test_settings_from_cli_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 一个参数都不传: 走默认 (把 cwd 挪到空目录避免读到真 config.json)."""
    monkeypatch.chdir(tmp_path)
    settings = settings_from_cli([])
    assert settings.port == 8000
    assert settings.host == "127.0.0.1"


def test_settings_from_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 参数按名覆盖, 短横线自动映射成下划线."""
    monkeypatch.chdir(tmp_path)
    settings = settings_from_cli(
        ["--port", "9500", "--chunk-size", "16", "--device", "cpu",
         "--n-ctx", "4096", "--api-key", "sekret"]
    )
    assert settings.port == 9500
    assert settings.chunk_size == 16
    assert settings.device is Device.CPU
    assert settings.n_ctx == 4096
    assert settings.api_key == "sekret"


def test_settings_from_cli_config_file_and_override(tmp_path: Path) -> None:
    """--config 指定文件, 命令行再覆盖单个字段."""
    path = _write(tmp_path, {"port": 9600, "chunk_size": 32})
    settings = settings_from_cli(["--config", str(path), "--chunk-size", "64"])
    assert settings.port == 9600
    assert settings.chunk_size == 64


def test_settings_from_cli_rejects_missing_config(tmp_path: Path) -> None:
    """--config 指向不存在的文件时报错."""
    with pytest.raises(ConfigError):
        settings_from_cli(["--config", str(tmp_path / "missing.json")])


def test_settings_from_cli_rejects_bad_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """越界值在构造期就被拦下."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        settings_from_cli(["--chunk-size", "999"])
