"""设备契约: 主力是 GPU, CPU 档只要能在本机跑起来就行; 外加 n_ctx 边界."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from minicpm_jev import (
    DEFAULT_N_CTX,
    MAX_N_CTX,
    Device,
    EngineConfig,
    EngineError,
    acquire_device,
    reset_device_lock,
    validate_n_ctx,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"


def test_cpu_device_can_run() -> None:
    """CPU 纯档在独立进程里能加载并出分: 自己要把 CUDA_VISIBLE_DEVICES 置 -1.

    CPU 档与 GPU 档互斥, 所以必须单开进程跑, 不能在已有 GPU 引擎的进程里建.
    """
    lines = [
        "import os",
        "from pathlib import Path",
        "from minicpm_jev import BatchEngine, Device, EngineConfig",
        "from minicpm_jev.labels import BOOL_LABEL_SPECS, resolve_labels",
        f"model = Path(r'{MODEL_PATH}')",
        "assert 'CUDA_VISIBLE_DEVICES' not in os.environ",
        "config = EngineConfig(model_path=model, device=Device.CPU, n_ctx=1024)",
        "with BatchEngine(config) as engine:",
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'",
        "    labels = resolve_labels(",
        "        lambda text: engine.tokenize(text, add_bos=False), BOOL_LABEL_SPECS",
        "    )",
        "    tokens = engine.render_tokens([",
        "        {'role': 'system', 'content': '只回答 yes 或 no。'},",
        "        {'role': 'user', 'content': '问题: 1+1=2 吗?'},",
        "    ])",
        "    scores = engine.score([tokens], labels)[0]",
        "    assert abs(sum(scores.values()) - 1.0) < 1e-6",
        "print('CPU_OK')",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "CPU_OK" in completed.stdout


def test_default_n_ctx_is_32k() -> None:
    """默认上下文 32768, 一个参数都不传就该是这个值."""
    assert DEFAULT_N_CTX == 32768
    config = EngineConfig(model_path=MODEL_PATH)
    assert config.n_ctx == DEFAULT_N_CTX




def test_n_ctx_over_upper_bound_reports_limit() -> None:
    """超上限报错, 文案里同时给出收到的值和上限值."""
    with pytest.raises(EngineError) as error:
        validate_n_ctx(MAX_N_CTX + 1)
    message = str(error.value)
    assert str(MAX_N_CTX) in message
    assert str(MAX_N_CTX + 1) in message

    with pytest.raises(EngineError):
        EngineConfig(model_path=MODEL_PATH, n_ctx=200000)






def test_device_lock_is_process_wide_then_restored() -> None:
    """同进程只允许一种档位; 用完必须把锁恢复成 GPU, 不误伤后续用例."""
    reset_device_lock()
    acquire_device(Device.CPU)
    try:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
        with pytest.raises(EngineError) as error:
            acquire_device(Device.GPU)
        message = str(error.value)
        assert "cpu" in message
        assert "gpu" in message
    finally:
        reset_device_lock()
        acquire_device(Device.GPU)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


