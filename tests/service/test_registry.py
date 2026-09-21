"""本地模型清单: 扫描 / 描述 / 别名解析与校验."""

from pathlib import Path

import pytest

from minicpm_jev.config import ConfigError
from minicpm_jev.service import ModelRegistry, UsageError


def _touch(path: Path, size: int = 1024) -> Path:
    """造一个假权重文件 (内容无所谓, 只看名字与体积)."""
    path.write_bytes(b"x" * size)
    return path


def test_scan_finds_only_gguf_files(tmp_path: Path) -> None:
    """只认目录里的 .gguf, 不收子目录、不收别的后缀."""
    _touch(tmp_path / "MiniCPM5-2B-Q4_K_M.gguf")
    _touch(tmp_path / "readme.txt")
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "sub" / "nested.gguf")
    registry = ModelRegistry(tmp_path)
    assert registry.names() == ("MiniCPM5-2B-Q4_K_M",)


def test_missing_directory_gives_empty_registry(tmp_path: Path) -> None:
    """目录不存在就空清单, 任何解析都报未知模型."""
    registry = ModelRegistry(tmp_path / "nope")
    assert registry.names() == ()
    assert registry.list() == []
    with pytest.raises(UsageError):
        registry.resolve("anything")


def test_description_and_release_date(tmp_path: Path) -> None:
    """描述里带量化与体积, 发布日期是 ISO 字符串."""
    _touch(tmp_path / "MiniCPM5-2B-Q4_K_M.gguf", size=2 * 1024 * 1024)
    info = ModelRegistry(tmp_path).list()[0]
    assert info.quant == "Q4_K_M"
    assert "Q4_K_M" in info.description
    assert "2 MiB" in info.description
    assert "T" in info.release_date
    entry = info.to_dict()
    assert set(entry) == {"name", "description", "release_date"}


def test_unknown_quant_is_labelled(tmp_path: Path) -> None:
    """认不出量化标记时给明确说明, 不是空字符串."""
    _touch(tmp_path / "mystery.gguf")
    assert "unknown quant" in ModelRegistry(tmp_path).list()[0].description


def test_resolve_without_aliases_is_exact_match(tmp_path: Path) -> None:
    """没有别名层: 请求什么就是什么, 大小写 / 分隔符不同都算未知."""
    _touch(tmp_path / "model-a.gguf")
    registry = ModelRegistry(tmp_path)
    assert registry.resolve("model-a") == "model-a"
    with pytest.raises(UsageError) as error:
        registry.resolve("model_a")
    assert "Unknown model" in str(error.value)


def test_alias_resolves_to_real_model(tmp_path: Path) -> None:
    """配了别名才做映射, 清单里仍然只列真实模型."""
    _touch(tmp_path / "model-a.gguf")
    registry = ModelRegistry(tmp_path, {"latest": "model-a"})
    assert registry.resolve("latest") == "model-a"
    assert registry.resolve("model-a") == "model-a"
    assert registry.names() == ("model-a",)


def test_alias_to_missing_model_is_rejected(tmp_path: Path) -> None:
    """别名指向不存在的模型 -> 启动即报错."""
    _touch(tmp_path / "model-a.gguf")
    with pytest.raises(ConfigError) as error:
        ModelRegistry(tmp_path, {"latest": "nope"})
    assert "latest" in str(error.value)


def test_alias_colliding_with_real_name_is_rejected(tmp_path: Path) -> None:
    """别名与真实模型名撞名 -> 启动即报错."""
    _touch(tmp_path / "model-a.gguf")
    _touch(tmp_path / "model-b.gguf")
    with pytest.raises(ConfigError) as error:
        ModelRegistry(tmp_path, {"model-a": "model-b"})
    assert "撞名" in str(error.value)


def test_path_of_returns_weight_file(tmp_path: Path) -> None:
    """真实模型名 -> 权重路径; 未知名字报错."""
    path = _touch(tmp_path / "model-a.gguf")
    registry = ModelRegistry(tmp_path)
    assert registry.path_of("model-a") == path
    with pytest.raises(UsageError):
        registry.path_of("nope")
