"""批量条数甜点标定: 扫 n_seq_max, 同时量「首字延迟」与「显存占用」.

输入: --model <gguf>; --max <最大条数>; --n-ctx <上下文>; --state-tokens <公共前缀规模>;
输出: 控制台表格 (每档: 去重序列数 / 批延迟 ms / 单条 ms / 吞吐 seq/s / 显存 MiB);
预期: 纯只读基准, 只在自己进程里加载模型, 不改任何外部状态; 结果用来定 Settings.n_seq_max。

读法: 延迟还在线性区、显存还留有余量的最大档, 就是甜点。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicpm_jev import BatchEngine, EngineConfig  # noqa: E402


def vram_used_mib() -> int | None:
    """当前 GPU 已用显存 (MiB); 取不到 (没显卡 / 没 nvidia-smi) 就返回 None."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return int(completed.stdout.strip().splitlines()[0])
    except ValueError:
        return None


def build_state(words: int) -> str:
    """造一段固定规模的公共前缀 (state), 让各档的 token 量可比."""
    filler = "顾客反馈订单延迟, 客服需要判断优先级并给出处理建议。"
    return filler * max(1, words // len(filler))


def measure(engine: BatchEngine, sequences: list[list[int]], repeats: int) -> float:
    """重复跑同一批, 返回平均批延迟 (秒)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        engine.decision_logits(sequences)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    """跑完整套扫描. 输出: 退出码."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GGUF 路径")
    parser.add_argument("--max", type=int, default=32, help="最大去重序列数")
    parser.add_argument("--n-ctx", type=int, default=8192, help="上下文长度")
    parser.add_argument("--state-tokens", type=int, default=600, help="公共前缀规模")
    parser.add_argument("--repeats", type=int, default=3, help="每档重复次数 (取最快)")
    args = parser.parse_args()

    state = build_state(args.state_tokens)
    print(f"model={Path(args.model).name} n_ctx={args.n_ctx} state≈{args.state_tokens} 字")
    print(f"{'seq':>4} {'batch_ms':>10} {'per_seq_ms':>11} {'seq_per_s':>10} {'vram_mib':>9}")

    counts = [value for value in (1, 2, 4, 8, 16, 32, 64) if value <= args.max]
    for count in counts:
        config = EngineConfig(
            model_path=Path(args.model),
            n_ctx=args.n_ctx,
            n_seq_max=max(counts),
            n_batch=2048,
            n_ubatch=512,
        )
        with BatchEngine(config) as engine:
            sequences = [
                engine.render_tokens(
                    [
                        {"role": "system", "content": state},
                        {"role": "user", "content": f"问题 {index}: 这条反馈紧急吗?"},
                    ],
                    assistant_prefix="</think>\n",
                )
                for index in range(count)
            ]
            total_tokens = sum(len(tokens) for tokens in sequences)
            if total_tokens > args.n_ctx:
                # 超预算就跳过这一档并说明原因, 不硬塞也不静默缩小规模
                print(
                    f"{count:>4} {'skip':>10}  "
                    f"(batch {total_tokens} tokens > n_ctx={args.n_ctx})"
                )
                continue
            elapsed = measure(engine, sequences, args.repeats)
            vram = vram_used_mib()
        batch_ms = elapsed * 1000
        print(
            f"{count:>4} {batch_ms:>10.1f} {batch_ms / count:>11.2f} "
            f"{count / elapsed:>10.1f} {vram if vram is not None else -1:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
