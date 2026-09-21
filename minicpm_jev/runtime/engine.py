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
- 设备档位与 n_ctx 边界见 runtime.device: gpu = n_gpu_layers=-1; cpu 纯档 =
  n_gpu_layers=0 且进程级 CUDA_VISIBLE_DEVICES=-1; 同一进程只允许一种档位.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import llama_cpp
import numpy as np
from llama_cpp import _internals as _internals

from ..labels import LabelSet, NumericLabels
from ..template import get_chat_template, render_chat
from .device import DEFAULT_N_CTX, Device, EngineError, acquire_device, validate_n_ctx

__all__ = ["BatchEngine", "EngineConfig"]

Message = Mapping[str, str]


@dataclass(frozen=True)
class EngineConfig:
    """引擎构造参数.

    n_seq_max 是 batch 能挂的 seq_id 上限 (也即去重后的序列条数上限);
    n_ctx 是统一 KV 的 cell 总数; n_batch 是单次 llama_decode 的 token 上限.
    """

    model_path: Path
    device: Device = Device.GPU
    n_ctx: int = DEFAULT_N_CTX
    n_batch: int = 2048
    n_ubatch: int = 512
    n_seq_max: int = 32
    n_threads: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        validate_n_ctx(self.n_ctx)
        for field_name in ("n_batch", "n_ubatch", "n_seq_max"):
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
        raise EngineError(f"logits are not enabled for batch token {index}")
    return np.ctypeslib.as_array(pointer, shape=(n_vocab,)).astype(np.float32, copy=True)


def _normalize_labels(labels: LabelSet | Sequence[LabelSet], count: int) -> list[LabelSet]:
    """把「一个标签集」或「每条序列一个标签集」统一成逐条列表."""
    if isinstance(labels, LabelSet):
        return [labels] * count
    resolved = list(labels)
    if len(resolved) != count:
        raise EngineError(f"got {len(resolved)} label sets for {count} sequences")
    return resolved


class BatchEngine:
    """一次 decode 处理整批序列的受限决策打分引擎."""

    def __init__(self, config: EngineConfig) -> None:
        acquire_device(config.device)
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
        self._numeric_labels: NumericLabels | None = None

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

    def tokenize_label(self, text: str) -> list[int]:
        """按标签口径切文本: 不带 BOS, 不解析特殊标记.

        输入: text -- 标签文本 (例如 "12" 或 "None");
        输出: token id 列表;
        预期: 与标签编译走同一口径, 保证「编出来的单 token」就是「打分时读的那个
              token」; 两者一旦不一致, 打分位就会静默错位.
        """
        return self.tokenize(text, add_bos=False, special=False)

    @property
    def numeric_labels(self) -> NumericLabels:
        """数字标签工厂 (进程内缓存), 供分块打分复用."""
        if self._numeric_labels is None:
            self._numeric_labels = NumericLabels(self.tokenize_label)
        return self._numeric_labels

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
                f"deduplicated {len(unique)} sequences exceed "
                f"n_seq_max={self._config.n_seq_max}"
            )
        unique_sequences: list[list[int]] = [[] for _ in unique]
        for key, index in unique.items():
            if not key:
                raise EngineError(f"sequence {index} is empty; no decision position")
            unique_sequences[index] = list(key)
        total_tokens = sum(len(tokens) for tokens in unique_sequences)
        if total_tokens > self._config.n_ctx:
            raise EngineError(
                f"batch of {total_tokens} tokens exceeds n_ctx={self._config.n_ctx}"
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
            raise EngineError(f"no decision logits for sequences {missing}")
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

    def _fill(
        self, entries: Sequence[tuple[int, int, int]], logits_all: bool
    ) -> None:
        """手填批槽位. 输入: [(seq_id, token, pos)]; logits_all -- 是否每个槽位都要 logits.

        直接写原生字段, 因为库自带的两个助手都撑不起「多序列各自续接」:
        set_batch 把 seq_id 写死成 0 并且覆盖式重置 n_tokens, add_sequence 的
        pos 从 0 起算、只能整条一次塞进去。这里只填 token/pos/seq_id/logits
        四个数组, 其余字段不动。
        """
        if len(entries) > self._config.n_batch:
            raise EngineError(
                f"batch of {len(entries)} tokens exceeds n_batch={self._config.n_batch}"
            )
        batch = self._batch.batch
        for index, (seq_id, token, pos) in enumerate(entries):
            batch.token[index] = token
            batch.pos[index] = pos
            batch.seq_id[index][0] = seq_id
            batch.n_seq_id[index] = 1
            batch.logits[index] = 1 if logits_all else 0
        if entries and not logits_all:
            batch.logits[len(entries) - 1] = 1
        batch.n_tokens = len(entries)

    def reasoned_logits(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        think_tokens: int,
        closure: str = "\n[",
    ) -> list[np.ndarray]:
        """带推理预算的决策位 logits (锁步批量).

        输入: sequences -- prompt 序列 (assistant 头收尾, 不带预填前缀);
              think_tokens -- 每条允许自由推理的 token 数;
              closure -- 推理结束后强制接上的闭合标记;
        输出: 与输入等长的决策位 logits;
        预期: think_tokens <= 0 时等价于把闭合标记直接拼进 prompt 再读决策位;
              大于 0 时每条序列各占一个 seq_id, 每步一次 decode 全体前进一步,
              谁吐 EOS 谁退出活跃集; 全停之后各自喂闭合标记读决策位。
              超 n_ctx 或超 n_seq_max 直接报错, 不截断也不静默丢。
        """
        if not sequences:
            return []
        closure_tokens = self.tokenize(closure, add_bos=False, special=False)
        if think_tokens <= 0:
            return self.decision_logits(
                [list(tokens) + closure_tokens for tokens in sequences]
            )
        if len(sequences) > self._config.n_seq_max:
            raise EngineError(
                f"{len(sequences)} sequences exceed n_seq_max={self._config.n_seq_max}"
            )
        for index, tokens in enumerate(sequences):
            if not tokens:
                raise EngineError(f"sequence {index} is empty; nothing to prefill")
            total = len(tokens) + think_tokens + len(closure_tokens)
            if total > self._config.n_ctx:
                raise EngineError(
                    f"sequence {index}: prompt {len(tokens)} + think {think_tokens}"
                    f" + closure {len(closure_tokens)} exceeds n_ctx={self._config.n_ctx}"
                )

        _clear_kv(self._context.memory)
        n_batch = self._config.n_batch
        eos = self._model.token_eos()
        n_past = [0] * len(sequences)
        pending: dict[int, int] = {}

        # 1) 各序列按自己的 seq_id 分块喂完 prompt; 最后一块打开 logits, 顺手取首步
        for seq_index, tokens in enumerate(sequences):
            for start in range(0, len(tokens), n_batch):
                chunk = list(tokens[start : start + n_batch])
                is_last = start + len(chunk) == len(tokens)
                self._fill(
                    [
                        (seq_index, token, start + offset)
                        for offset, token in enumerate(chunk)
                    ],
                    logits_all=is_last,
                )
                self._context.decode(self._batch)
                n_past[seq_index] = start + len(chunk)
                if is_last:
                    logits = _read_logits(
                        self._context, len(chunk) - 1, self.n_vocab
                    )
                    pending[seq_index] = int(np.argmax(logits))

        # 2) 锁步生成: 每步一次 decode, 每条还活着的序列各前进一个 token
        for _ in range(think_tokens):
            active = [index for index, token in pending.items() if token != eos]
            if not active:
                break
            self._fill(
                [(index, pending[index], n_past[index]) for index in active],
                logits_all=True,
            )
            self._context.decode(self._batch)
            following: dict[int, int] = {}
            for slot, index in enumerate(active):
                n_past[index] += 1
                logits = _read_logits(self._context, slot, self.n_vocab)
                following[index] = int(np.argmax(logits))
            pending = following

        # 3) 各自喂闭合标记并读决策位 (闭合标记很短, 逐条喂最省心)
        results: list[np.ndarray] = []
        for seq_index in range(len(sequences)):
            self._fill(
                [
                    (seq_index, token, n_past[seq_index] + offset)
                    for offset, token in enumerate(closure_tokens)
                ],
                logits_all=False,
            )
            self._context.decode(self._batch)
            results.append(
                _read_logits(self._context, len(closure_tokens) - 1, self.n_vocab)
            )
        return results

    def score(
        self,
        sequences: Sequence[Sequence[int]],
        labels: LabelSet | Sequence[LabelSet],
        *,
        think_tokens: int = 0,
        closure: str = "</think>\n[",
    ) -> list[dict[str, float]]:
        """批量打分: 决策位 logits -> 每个标签组的受限概率.

        输入: sequences -- token id 序列; labels -- 一个标签集 (全批共用) 或逐条标签集;
              think_tokens -- 大于 0 时先让模型自由推理这么多 token 再闭合读决策位;
              closure -- 闭合标记文本, 必须与本题型的正常预填前缀一致
                        (noul 用 "\n", choice/score 用 "\n[")
        输出: 逐条的 {组名: 概率}
        """
        if think_tokens > 0:
            logits = self.reasoned_logits(
                sequences, think_tokens=think_tokens, closure=closure
            )
        else:
            logits = self.decision_logits(sequences)
        label_sets = _normalize_labels(labels, len(sequences))
        return [
            label_set.probabilities(vector) for label_set, vector in zip(label_sets, logits)
        ]
