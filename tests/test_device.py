"""设备契约: 主力是 GPU, CPU 档只要能在本机跑起来就行."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
