# -*- coding: utf-8 -*-
"""
模型下载脚本（GGUF 单文件/分片，按量化等级挑选）。

输入:
    --source {openbmb, abliterated}   选仓库：原版 或 heretic-abliterated 去审查版
    --quant  <QUANT>                  量化等级，如 Q4_K_M；--quant list 打印该仓库全部可用等级
    --dest   <目录>                   落盘目录，默认 weights/（已被 .gitignore 排除，不入库）
    --force                           已存在同名文件时强制重新下载（默认跳过）
    --parts  <N>                      并发分段数，默认 1（走 huggingface_hub 单连接）；
                                      N>1 时走多线程 HTTP Range 下载（见"预期行为"第三条）

输出:
    下载到 <dest>/<原文件名>，stdout 打印每个文件的最终路径；退出码 0 成功 / 1 失败。

预期行为:
    - 走 HfApi 在线列仓库文件清单，量化名按大小写与分隔符不敏感匹配（q4_k_m == Q4-K-M），
      同等级若有分片（-00001-of-0000N）会全部下载；
    - 默认下载走 huggingface_hub 缓存 + 断点续传，endpoint 跟随环境变量 HF_ENDPOINT
      （本机已配 hf-mirror.com 加速）；
    - --parts N（N>1）：诊断发现单连接被限速到 ~0.23MB/s，而多连接 Range 聚合可达 ~1.9MB/s
      （8 路近似线性）。故改用 requests 直连官方 resolve 接口，按 N 段并行 Range 拉取，
      每段落盘为 <name>.part<i>（各自可断点续传），全部就绪后按序拼接成最终 .gguf。
      此路径不依赖 xet，按 HF_ENDPOINT（默认 hf-mirror.com）直连 resolve→CDN。
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import requests
from huggingface_hub import HfApi, hf_hub_download

# 多段并行下载用的 resolve 基址：跟随 HF_ENDPOINT，默认直连 hf-mirror
RESOLVE_BASE = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"

# 两个支持的仓库
SOURCES = {
    "openbmb": "openbmb/MiniCPM5-2B-GGUF",
    "abliterated": "Abiray/MiniCPM5-2B-heretic-abliterated-GGUF",
}

# 文件名里可识别的量化等级标记（按 GGUF 社区惯例）
QUANT_PATTERN = re.compile(
    r"(F16|BF16|Q2_K|Q3_K_[SML]|Q4_K_[SML]|Q4_0|Q4_1|Q5_K_[SML]|Q5_0|Q5_1"
    r"|Q6_K|Q8_0|IQ[0-9]_[A-Za-z0-9_]+)"
)


def norm(s: str) -> str:
    """归一化：大写、分隔符统一为下划线，做不敏感比较用。"""
    return re.sub(r"[-.]+", "_", s).upper()


def _get_total_size(url: str) -> int:
    """用 Range bytes=0-0 探一次，从 Content-Range 读文件总字节数。输出 int，失败抛异常。"""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
    cr = r.headers.get("content-range")
    r.close()
    if r.status_code == 206 and cr:
        return int(cr.rsplit("/", 1)[1])
    raise RuntimeError(f"服务器不支持 Range 下载（HTTP {r.status_code}）")


def _fetch_segment(url: str, start: int, end: int, part_path: Path, idx: int) -> int:
    """下载 [start,end] 单段并落盘 part_path（已有一部分则从断点续）。输出该段最终字节数。
    预期：网络异常自动重试 ≤20 次；每收 64MB 打一行进度。"""
    have = part_path.stat().st_size if part_path.exists() else 0
    seg_len = end - start + 1
    if have >= seg_len:
        return have
    for attempt in range(20):
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part_path, "ab") as f:
                    f.seek(have)
                    since_log = 0
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                        have += len(chunk)
                        since_log += len(chunk)
                        if since_log >= (64 << 20):
                            print(f"  part{idx}: {have/1048576:.0f}/{seg_len/1048576:.0f} MB",
                                  flush=True)
                            since_log = 0
            if have >= seg_len:
                return have
        except Exception as e:
            print(f"  part{idx} 第{attempt+1}次重试（{type(e).__name__}），已下 {have/1048576:.0f} MB",
                  flush=True)
    raise RuntimeError(f"part{idx} 重试 20 次仍失败")


def download_multipart(repo_id: str, name: str, dest: Path, parts: int,
                       force: bool = False) -> Path:
    """输入：仓库/文件名/目录/并发数；输出：合并后的最终文件路径。
    预期：N 段并行 Range 拉取 → .part 文件 → 按序拼接 → 校验总大小后删除 .part。"""
    from urllib.parse import quote
    url = f"{RESOLVE_BASE}/{quote(repo_id)}/resolve/main/{quote(name)}"
    final = dest / Path(name).name
    if final.exists() and not force:
        print(f"⏭ 已存在，跳过: {final}", flush=True)
        return final
    total = _get_total_size(url)
    part_dir = dest / ".parts"
    part_dir.mkdir(exist_ok=True)
    print(f"⬇ {name}: {total/1073741824:.2f} GiB, {parts} 段并行", flush=True)
    bounds = [(i * total // parts, (i + 1) * total // parts - 1) for i in range(parts)]
    jobs = [(idx, s, e, part_dir / f"{Path(name).name}.part{idx}")
            for idx, (s, e) in enumerate(bounds)]
    with ThreadPoolExecutor(max_workers=parts) as pool:
        got = list(pool.map(lambda j: _fetch_segment(url, j[1], j[2], j[3], j[0]), jobs))
    if sum(got) != total:
        raise RuntimeError(f"分段字节数对不上: {sum(got)} != {total}")
    tmp = dest / (Path(name).name + ".assembling")
    with open(tmp, "wb") as out:
        for _, _, _, pf in jobs:
            with open(pf, "rb") as f:
                while chunk := f.read(1 << 20):
                    out.write(chunk)
            pf.unlink()
    os.replace(tmp, final)
    return final


class GgufFile(TypedDict):
    """仓库里一个 GGUF 文件的摘要：路径 / 量化档 / 是否分片。"""

    rfilename: str
    quant: str
    shard: bool


def list_gguf_files(repo_id: str) -> list[GgufFile]:
    """拉仓库文件清单，只保留 .gguf。输出 [{rfilename, quant, shard}] 字典列表。"""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=False)
    entries: list[GgufFile] = []
    for sib in info.siblings or []:
        name = sib.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        base = Path(name).stem
        m = QUANT_PATTERN.search(base.upper().replace("-", "_"))
        entries.append({
            "rfilename": name,
            "quant": m.group(1) if m else "(未识别)",
            "shard": "of-" in name,
        })
    return entries


def main() -> int:
    p = argparse.ArgumentParser(description="下载 MiniCPM5-2B GGUF 模型（模型文件不入库）")
    p.add_argument("--source", choices=SOURCES, default="openbmb",
                   help="openbmb=原版(默认), abliterated=heretic 去审查版")
    p.add_argument("--quant", default=None,
                   help="量化等级，如 Q4_K_M；填 list 只列可用等级")
    p.add_argument("--dest", default="weights", help="落盘目录，默认 weights/")
    p.add_argument("--force", action="store_true", help="同名文件已存在也重下")
    p.add_argument("--parts", type=int, default=1,
                   help="并发分段数（>1 走多线程 Range 下载，绕过单连接限速），默认 1")
    args = p.parse_args()

    repo_id = SOURCES[args.source]
    try:
        files = list_gguf_files(repo_id)
    except Exception as e:  # 网络/仓库不存在等人话报错，不裸抛栈
        print(f"❌ 获取仓库文件清单失败（{repo_id}）：{e}", file=sys.stderr)
        return 1

    if not files:
        print(f"❌ 仓库 {repo_id} 里没有 .gguf 文件", file=sys.stderr)
        return 1

    quants = sorted({f["quant"] for f in files})
    if args.quant is None or args.quant.lower() == "list":
        print(f"仓库 {repo_id} 可用量化等级:")
        for q in quants:
            print(f"  {q}")
        print(f"\n用法: python {Path(__file__).name} --source {args.source} --quant <等级>")
        return 0

    want = norm(args.quant)
    picked = [f["rfilename"] for f in files if norm(f["quant"]) == want]
    if not picked:
        print(f"❌ 仓库 {repo_id} 没有量化等级 {args.quant}，可用: {', '.join(quants)}",
              file=sys.stderr)
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"仓库: {repo_id} | 等级: {args.quant} | 文件数: {len(picked)}")
    try:
        for name in sorted(picked):
            if args.parts > 1:
                path = str(download_multipart(repo_id, name, dest, args.parts, args.force))
            else:
                local = dest / Path(name).name
                if local.exists() and not args.force:
                    print(f"⏭ 已存在，跳过: {local}")
                    continue
                print(f"⬇ 下载 {name} ...", flush=True)
                path = hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(dest))
            print(f"✅ {path}")
    except Exception as e:
        print(f"❌ 下载失败：{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
