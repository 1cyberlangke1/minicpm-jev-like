"""批量推理引擎: 一次 llama_decode 处理整批序列, 只在决策位读 logits.

批由 llama-cpp-python 自带的 LlamaBatch 摆 (add_sequence / set_batch), 引擎只声明
"这条序列是哪个 seq_id", 不碰任何 ctypes 结构体字段.

设计要点 (都是踩过坑的):
- 通用批 (GPU): 哈希去重后每条序列一个 seq_id, 用 add_sequence 一次塞几条, 受
  n_batch 限制; 单条就超过 n_batch 的长序列退回 set_batch 分块.
- CPU 档只做单序列推理, 不拼多序列批: CPU 是给没有 GPU 的机器兜底的, 一次喂一条
  就够用, 不需要批.
- prefill-only: 不 decode 任何 token, 只在每条序列最后一个 token 上打开 logits.
- KV 重置必须用 llama_memory_clear: kv_unified 下 llama_memory_seq_rm(mem,-1,-1,-1)
  清不干净, 实测第二次 decode 直接报「序列位置不连续」.
- 设备契约: gpu = n_gpu_layers=-1; cpu 纯档 = n_gpu_layers=0 且进程级
  CUDA_VISIBLE_DEVICES=-1 (空串会被 Windows 环境机制整个丢弃, 必须写 -1).
  同一进程只允许一种设备档位, 混用直接报错, 不静默降级.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import llama_cpp
import numpy as np
from llama_cpp import _internals as _internals

from .labels import LabelSet
from .template import get_chat_template, render_chat

__all__ = ["BatchEngine", "Device", "EngineConfig", "EngineError"]

Message = Mapping[str, str]


class EngineError(RuntimeError):
    """引擎契约被违反: 批次超限 / 设备冲突 / 标签数量不匹配 / 取不到 logits."""


class Device(str, Enum):
    """推理设备档位."""

    GPU = "gpu"
    CPU = "cpu"


_ACTIVE_DEVICE: Device | None = None


def _acquire_device(device: Device) -> None:
    """锁定进程级设备档位, 一个进程只允许一种.

    输入: device -- 目标设备
    输出: 无
    预期: CPU 档在加载模型前先把 CUDA_VISIBLE_DEVICES 置 -1; 已锁定别的档位时抛
          EngineError. 之所以要进程级屏蔽: GPU 可见时本构建即使 0 层 offload 仍会把
          部分计算调度到 CUDA0, 跨调用不再可复现.
    """
    global _ACTIVE_DEVICE
    if _ACTIVE_DEVICE is None:
        if device is Device.CPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        _ACTIVE_DEVICE = device
        return
    if _ACTIVE_DEVICE is not device:
        raise EngineError(
            f"本进程已锁定 {_ACTIVE_DEVICE.value} 设备, 不能再创建 {device.value} 引擎"
        )


@dataclass(frozen=True)
class EngineConfig:
    """引擎构造参数.

    n_seq_max 是 batch 能挂的 seq_id 上限 (也即去重后的序列条数上限);
    n_ctx 是统一 KV 的 cell 总数; n_batch 是单次 llama_decode 的 token 上限.
    """

    model_path: Path
    device: Device = Device.GPU
    n_ctx: int = 4096
    n_batch: int = 2048
    n_ubatch: int = 512
    n_seq_max: int = 32
    n_threads: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        for field_name in ("n_ctx", "n_batch", "n_ubatch", "n_seq_max"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} 必须为正, 收到 {value}")
        if self.n_ubatch > self.n_batch:
            raise ValueError(f"n_ubatch({self.n_ubatch}) 不能大于 n_batch({self.n_batch})")
        if self.n_threads is not None and self.n_threads <= 0:
            raise ValueError(f"n_threads 必须为正, 收到 {self.n_threads}")


def _clear_kv(memory: object) -> None:
    """整体清空 KV 缓存.

    kv_unified 下 llama_memory_seq_rm(mem, -1, -1, -1) 清不干净 (实测第二次 decode
    直接报「序列位置不连续」), 必须走 llama_memory_clear.
    """
    llama_cpp.llama_memory_clear(memory, True)


def _read_logits(context: object, index: int, n_vocab: int) -> np.ndarray:
    """读批内第 index 个 token 的全 vocab logits, 拷成 float32.

    输入: context -- 已 decode 的上下文; index -- 批内 token 下标 (不是输出序号)
    输出: 长度 n_vocab 的 float32 数组
    预期: 该位置没打开 logits 时抛 EngineError, 不返回垃圾数据
    """
    pointer = context.get_logits_ith(index)  # type: ignore[attr-defined]
    if not pointer:
        raise EngineError(f"批内第 {index} 个 token 没打开 logits, 取不到分布")
    return np.ctypeslib.as_array(pointer, shape=(n_vocab,)).astype(np.float32, copy=True)


def _normalize_labels(labels: LabelSet | Sequence[LabelSet], count: int) -> list[LabelSet]:
    """把「一个标签集」或「每条序列一个标签集」统一成逐条列表."""
    if isinstance(labels, LabelSet):
        return [labels] * count
    resolved = list(labels)
    if len(resolved) != count:
        raise EngineError(f"标签集数量 {len(resolved)} 与序列数 {count} 不一致")
    return resolved


class BatchEngine:
    """一次 decode 处理整批序列的受限决策打分引擎."""

    def __init__(self, config: EngineConfig) -> None:
        _acquire_device(config.device)
        self._config = config

        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = -1 if config.device is Device.GPU else 0
        self._model = _internals.LlamaModel(
            path_model=str(config.model_path), params=model_params, verbose=False
        )
        # 模板缺失就该在这里炸, 不能拖到打分时才发现
        self._chat_template = get_chat_template(self._model)

        context_params = llama_cpp.llama_context_default_params()
        context_params.n_ctx = config.n_ctx
        context_params.n_batch = config.n_batch
        context_params.n_ubatch = config.n_ubatch
        context_params.n_seq_max = config.n_seq_max
        context_params.n_gpu_layers = model_params.n_gpu_layers
        context_params.kv_unified = True
        threads = config.n_threads or (os.cpu_count() or 4)
        context_params.n_threads = threads
        context_params.n_threads_batch = threads
        self._context = _internals.LlamaContext(
            model=self._model, params=context_params, verbose=False
        )
        # 批容器用库自带的 LlamaBatch: seq_id / n_seq_id / 位置全由 add_sequence /
        # set_batch 填好, 引擎不碰 ctypes 字段
        self._batch = _internals.LlamaBatch(
            n_tokens=config.n_batch, embd=0, n_seq_max=config.n_seq_max, verbose=False
        )
        self._closed = False

    @property
    def device(self) -> Device:
        return self._config.device

    @property
    def n_vocab(self) -> int:
        return self._model.n_vocab()

    @property
    def n_ctx(self) -> int:
        return self._config.n_ctx

    @property
    def n_batch(self) -> int:
        return self._config.n_batch

    @property
    def n_seq_max(self) -> int:
        return self._config.n_seq_max

    def close(self) -> None:
        """释放 batch / context / model."""
        if self._closed:
            return
        self._closed = True
        self._batch.close()
        self._context.close()
        self._model.close()

    def __enter__(self) -> "BatchEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def tokenize(self, text: str, *, add_bos: bool = True, special: bool = False) -> list[int]:
        """把文本切成 token id.

        输入: text -- 待切文本; add_bos -- 是否带 BOS; special -- 是否解析特殊标记
        输出: token id 列表
        """
        return list(
            self._model.tokenize(text.encode("utf-8"), add_bos=add_bos, special=special)
        )

    def render(
        self,
        messages: Sequence[Message],
        *,
        add_assistant: bool = True,
        assistant_prefix: str = "",
    ) -> bytes:
        """按官方模板渲染消息列表 (可带 assistant 预填前缀)."""
        return render_chat(
            self._model,
            messages,
            add_assistant=add_assistant,
            assistant_prefix=assistant_prefix,
            template=self._chat_template,
        )

    def render_tokens(
        self,
        messages: Sequence[Message],
        *,
        add_assistant: bool = True,
        assistant_prefix: str = "",
    ) -> list[int]:
        """渲染 + tokenize 一把梭, 得到可直接入批的 prompt token."""
        prompt = self.render(
            messages, add_assistant=add_assistant, assistant_prefix=assistant_prefix
        )
        return list(self._model.tokenize(prompt, add_bos=True, special=True))

    def decision_logits(self, sequences: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """一次批量 prefill, 取每条序列决策位的全 vocab logits.

        输入: sequences -- token id 序列列表, 每条的最后一位就是决策位
        输出: 与输入等长的 float32 logits 数组 (长度 n_vocab)
        预期: 完全相同的序列由哈希去重共享一次计算并返回逐位相同的结果;
              去重后超过 n_seq_max 或整批 token 超过 n_ctx 直接报错, 不截断
        """
        if not sequences:
            return []
        unique: dict[tuple[int, ...], int] = {}
        mapping: list[int] = []
        for tokens in sequences:
            key = tuple(tokens)
            index = unique.get(key)
            if index is None:
                index = len(unique)
                unique[key] = index
            mapping.append(index)
        if len(unique) > self._config.n_seq_max:
            raise EngineError(
                f"去重后 {len(unique)} 条序列超过 n_seq_max={self._config.n_seq_max}"
            )
        unique_sequences: list[list[int]] = [[] for _ in unique]
        for key, index in unique.items():
            if not key:
                raise EngineError(f"第 {index} 条序列为空, 没有决策位可读")
            unique_sequences[index] = list(key)
        total_tokens = sum(len(tokens) for tokens in unique_sequences)
        if total_tokens > self._config.n_ctx:
            raise EngineError(
                f"整批 {total_tokens} 个 token 超过 n_ctx={self._config.n_ctx}"
            )

        collected: list[np.ndarray | None] = [None] * len(unique_sequences)
        n_vocab = self.n_vocab
        n_batch = self._config.n_batch
        # kv_unified 下 seq_rm(-1,...) 清不干净, 必须整体 clear
        _clear_kv(self._context.memory)

        index = 0
        while index < len(unique_sequences):
            tokens = unique_sequences[index]
            if len(tokens) > n_batch:
                # 单条就超过 n_batch: 没有并行余地, 退回 set_batch 分块
                collected[index] = self._eval_chunked_sequence(tokens)
                index += 1
                continue
            self._batch.reset()
            packed: list[tuple[int, int]] = []
            filled = 0
            while index < len(unique_sequences):
                candidate = unique_sequences[index]
                if len(candidate) > n_batch or filled + len(candidate) > n_batch:
                    break
                first_index = self._batch.n_tokens()
                # 库自己填位置与 seq_id, 顺手把末位 logits 打开
                self._batch.add_sequence(candidate, index, False)
                packed.append((index, first_index + len(candidate) - 1))
                filled += len(candidate)
                index += 1
                if self._config.device is Device.CPU:
                    # CPU 档不做多序列批: 一次只喂一条
                    break
            self._context.decode(self._batch)
            for target, batch_index in packed:
                collected[target] = _read_logits(self._context, batch_index, n_vocab)

        missing = [seq_index for seq_index, value in enumerate(collected) if value is None]
        if missing:
            raise EngineError(f"这些序列没拿到决策位 logits: {missing}")
        return [collected[seq_index] for seq_index in mapping]  # type: ignore[index]

    def _eval_chunked_sequence(self, tokens: Sequence[int]) -> np.ndarray:
        """一条装不进 n_batch 的长序列: 用 set_batch 分块喂完, 取末位 logits.

        走 set_batch 是因为它按 (n_past + i) 填位置, 天然支持续接; add_sequence 的
        位置从 0 起算, 只能整条一次塞进去.
        """
        _clear_kv(self._context.memory)
        n_batch = self._config.n_batch
        start = 0
        while start < len(tokens):
            chunk = list(tokens[start : start + n_batch])
            self._batch.reset()
            self._batch.set_batch(chunk, start, False)
            self._context.decode(self._batch)
            start += len(chunk)
        return _read_logits(self._context, self._batch.n_tokens() - 1, self.n_vocab)

    def score(
        self,
        sequences: Sequence[Sequence[int]],
        labels: LabelSet | Sequence[LabelSet],
    ) -> list[dict[str, float]]:
        """批量打分: 决策位 logits -> 每个标签组的受限概率.

        输入: sequences -- token id 序列; labels -- 一个标签集 (全批共用) 或逐条标签集
        输出: 逐条的 {组名: 概率}
        """
        logits = self.decision_logits(sequences)
        label_sets = _normalize_labels(labels, len(sequences))
        return [
            label_set.probabilities(vector) for label_set, vector in zip(label_sets, logits)
        ]
