"""Settings: 默认值 / 边界 / 别名表校验 (不加载模型)."""

import pytest

from minicpm_jev.chunking import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE
from minicpm_jev.config import ConfigError, Settings
from minicpm_jev.runtime import DEFAULT_N_CTX, MAX_N_CTX, Device, EngineError, EngineConfig


def test_defaults_match_documented_values() -> None:
    """默认值就是 PLAN 配置表里的那一套."""
    settings = Settings()
    assert settings.model_path is None
    assert settings.device is Device.GPU
    assert settings.n_ctx == DEFAULT_N_CTX == 32768
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE == 128
    assert settings.queue_max == 30
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.api_key is None
    assert dict(settings.model_aliases) == {}


def test_model_path_accepts_string_and_becomes_path() -> None:
    """字符串路径自动转成 Path."""
    settings = Settings(model_path="weights/model.gguf")
    assert settings.model_path is not None
    assert settings.model_path.name == "model.gguf"


def test_device_accepts_string_value() -> None:
    """device 可以给 'cpu' / 'gpu' 字符串."""
    assert Settings(device="cpu").device is Device.CPU
    assert Settings(device="gpu").device is Device.GPU
    with pytest.raises(ConfigError) as error:
        Settings(device="tpu")
    assert "device" in str(error.value)


def test_n_ctx_upper_bound_is_enforced() -> None:
    """n_ctx 上限走 runtime 的边界, 文案带收到的值与上限."""
    assert Settings(n_ctx=MAX_N_CTX).n_ctx == MAX_N_CTX
    with pytest.raises(EngineError) as error:
        Settings(n_ctx=MAX_N_CTX + 1)
    assert str(MAX_N_CTX) in str(error.value)


@pytest.mark.parametrize("value", [0, -1])
def test_n_ctx_must_be_positive(value: int) -> None:
    """n_ctx 必须为正."""
    with pytest.raises(EngineError):
        Settings(n_ctx=value)


@pytest.mark.parametrize("value", [0, -1, MAX_CHUNK_SIZE + 1])
def test_chunk_size_range_is_enforced(value: int) -> None:
    """chunk_size 在 (0, 128] 内."""
    with pytest.raises(ConfigError) as error:
        Settings(chunk_size=value)
    assert "chunk_size" in str(error.value)


def test_chunk_size_upper_bound_is_legal() -> None:
    """上限 128 合法 (闭区间)."""
    assert Settings(chunk_size=MAX_CHUNK_SIZE).chunk_size == MAX_CHUNK_SIZE


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("n_seq_max", {}),
        ("n_batch", {"n_ubatch": 1}),
        ("n_ubatch", {}),
        ("queue_max", {}),
    ],
)
def test_positive_int_fields(name: str, extra: dict) -> None:
    """批大小 / 序列上限 / 队列长度都必须为正整数."""
    assert getattr(Settings(**{name: 1}, **extra), name) == 1
    with pytest.raises(ConfigError):
        Settings(**{name: 0}, **extra)
    with pytest.raises(ConfigError):
        Settings(**{name: -5}, **extra)


def test_n_ubatch_must_not_exceed_n_batch() -> None:
    """微批不能大于批."""
    with pytest.raises(ConfigError):
        Settings(n_batch=16, n_ubatch=32)
    assert Settings(n_batch=32, n_ubatch=16).n_ubatch == 16


@pytest.mark.parametrize("value", [0, -1])
def test_n_threads_must_be_positive_when_given(value: int) -> None:
    """给了 n_threads 就必须为正."""
    with pytest.raises(ConfigError):
        Settings(n_threads=value)


@pytest.mark.parametrize("value", [0, 65536, -1])
def test_port_range_is_enforced(value: int) -> None:
    """端口必须在 1~65535."""
    with pytest.raises(ConfigError) as error:
        Settings(port=value)
    assert "port" in str(error.value)


def test_host_must_be_non_empty_string() -> None:
    """监听地址不能是空串."""
    with pytest.raises(ConfigError):
        Settings(host="")
    with pytest.raises(ConfigError):
        Settings(host=123)  # type: ignore[arg-type]


def test_empty_api_key_means_no_auth() -> None:
    """空串等于不鉴权 (官方语义: 不配 key 就不校验)."""
    assert Settings(api_key="").api_key is None
    assert Settings(api_key="secret").api_key == "secret"
    with pytest.raises(ConfigError):
        Settings(api_key=123)  # type: ignore[arg-type]


def test_model_aliases_shape_is_validated() -> None:
    """别名表必须是 别名 -> 真实模型名 的非空字符串映射."""
    settings = Settings(model_aliases={"latest": "MiniCPM5-2B-Q4_K_M"})
    assert dict(settings.model_aliases) == {"latest": "MiniCPM5-2B-Q4_K_M"}
    assert dict(Settings(model_aliases=None).model_aliases) == {}

    with pytest.raises(ConfigError):
        Settings(model_aliases=["latest"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        Settings(model_aliases={"": "model"})
    with pytest.raises(ConfigError):
        Settings(model_aliases={"latest": ""})
    with pytest.raises(ConfigError):
        Settings(model_aliases={"same": "same"})


def test_to_engine_config_requires_a_model_path() -> None:
    """没模型路径就不构造引擎, 不去猜 weights 目录."""
    with pytest.raises(ConfigError):
        Settings().to_engine_config()


def test_to_engine_config_carries_fields_over() -> None:
    """引擎参数逐字段带过去, 并允许按请求覆盖模型路径."""
    settings = Settings(
        model_path="weights/a.gguf",
        n_ctx=2048,
        chunk_size=16,
        n_batch=256,
        n_ubatch=128,
        n_seq_max=4,
        n_threads=3,
    )
    config = settings.to_engine_config()
    assert isinstance(config, EngineConfig)
    assert config.model_path.name == "a.gguf"
    assert config.n_ctx == 2048
    assert config.n_batch == 256
    assert config.n_ubatch == 128
    assert config.n_seq_max == 4
    assert config.n_threads == 3

    switched = settings.to_engine_config("weights/b.gguf")
    assert switched.model_path.name == "b.gguf"
    assert switched.n_ctx == 2048
