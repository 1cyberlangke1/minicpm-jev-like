"""Settings: 默认值 / 边界 / 别名表校验 (不加载模型)."""

import pytest

from minicpm_jev.chunking import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE
from minicpm_jev.config import ConfigError, Settings
from minicpm_jev.runtime import DEFAULT_N_CTX, MAX_N_CTX, Device, EngineError


def test_defaults_match_documented_values() -> None:
    """默认值就是 PLAN 配置表里的那一套."""
    settings = Settings()
    assert settings.model_path is None
    assert settings.device is Device.GPU
    assert settings.n_ctx == DEFAULT_N_CTX == 32768
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE == 128
    assert settings.n_seq_max == 16
    assert settings.queue_max == 30
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.api_key is None
    assert dict(settings.model_aliases) == {}






def test_n_ctx_upper_bound_is_enforced() -> None:
    """n_ctx 上限走 runtime 的边界, 文案带收到的值与上限."""
    assert Settings(n_ctx=MAX_N_CTX).n_ctx == MAX_N_CTX
    with pytest.raises(EngineError) as error:
        Settings(n_ctx=MAX_N_CTX + 1)
    assert str(MAX_N_CTX) in str(error.value)




@pytest.mark.parametrize("value", [0, -1, MAX_CHUNK_SIZE + 1])
def test_chunk_size_range_is_enforced(value: int) -> None:
    """chunk_size 在 (0, 128] 内."""
    with pytest.raises(ConfigError) as error:
        Settings(chunk_size=value)
    assert "chunk_size" in str(error.value)






def test_n_ubatch_must_not_exceed_n_batch() -> None:
    """微批不能大于批."""
    with pytest.raises(ConfigError):
        Settings(n_batch=16, n_ubatch=32)
    assert Settings(n_batch=32, n_ubatch=16).n_ubatch == 16








def test_empty_api_key_means_no_auth() -> None:
    """空串等于不鉴权 (官方语义: 不配 key 就不校验)."""
    assert Settings(api_key="").api_key is None
    assert Settings(api_key="secret").api_key == "secret"
    with pytest.raises(ConfigError):
        Settings(api_key=123)  # type: ignore[arg-type]




def test_to_engine_config_requires_a_model_path() -> None:
    """没模型路径就不构造引擎, 不去猜 weights 目录."""
    with pytest.raises(ConfigError):
        Settings().to_engine_config()


