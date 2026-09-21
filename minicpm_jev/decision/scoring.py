"""三原语打分: 把 (state, question) 变成官方形状的答案.

prompt 结构 (铁律, 不许手搓标记):

- system = state (结构化就原样 JSON 序列化);
- user = instructions + 候选/档位清单 + 回答要求;
- assistant 预填 = 思考段截断 + 标签引导 (choice / score 用 ``[``, noul 不加),
  走库函数的 ``assistant_prefix``, 不碰模板标记。

分块 (choice 的候选数不受限):

- ``N <= chunk_size`` 单块, 标签 ``0`` ~ ``N-1``, **不带锚点**;
- ``N > chunk_size`` 每块 ``0`` ~ ``m-1`` 再补 ``None``, 跨块用 log-odds 对齐。

``score`` 档位最多 10 个, 天然单块; 分是档位上的概率加权值 ``Σ p_i · i``。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..chunking import DEFAULT_CHUNK_SIZE, Chunk, align_chunks, plan_chunks
from ..labels import BOOL_LABEL_SPECS, resolve_labels
from ..runtime import BatchEngine
from ..template import render_entry
from .confidence import normalized_peak
from .primitives import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)

__all__ = [
    "Usage",
    "answer",
    "answer_choice",
    "answer_noul",
    "answer_score",
    "weighted_level_score",
]

#: 思考段截断: 让模型跳过内部推理, 直接在决策位吐标签
THINK_PREFIX = "</think>\n"
#: choice / score 的标签引导: 收尾在 "[", 模型接着吐编号 token
INDEX_PREFIX = THINK_PREFIX + "["
#: 分块时锚点选项的说明 (与白皮书措辞一致)
NONE_HINT = "[None] 以上所有选项均不合适、错误或存在严重缺陷"


@dataclass
class Usage:
    """一次决策的 token 统计.

    ``input_tokens`` 按官方语义统计「本次全部序列的 token 数 (去重前)」;
    ``output_tokens`` 恒为 1 —— prefill-only 确实在决策位读了一个 token 的分布,
    只是不把它的文本吐出来。
    """

    input_tokens: int = 0
    output_tokens: int = 1

    def add_input(self, tokens: int) -> None:
        """累加输入 token 数. 输入: tokens -- 本次新增的 token 数; 输出: 无."""
        self.input_tokens += tokens

    def to_dict(self) -> dict[str, int]:
        """官方 usage 形状."""
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def _record_usage(usage: Usage | None, sequences: Sequence[Sequence[int]]) -> None:
    """把本批序列的 token 数记进 usage (没给 usage 就什么都不做)."""
    if usage is not None:
        usage.add_input(sum(len(sequence) for sequence in sequences))


def weighted_level_score(probabilities: Sequence[float]) -> float:
    """档位概率 -> 概率加权分 ``Σ p_i · i``.

    输入: probabilities -- 按档位编号升序的概率 (长度 = 档位数, >= 1);
    输出: 加权分, 落在 [0, len-1];
    预期: 空输入抛 ValueError; 只做加权, 不负责归一 (调用方给的就是受限分布)。
    """
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return sum(
        index * float(probability) for index, probability in enumerate(probabilities)
    )


def _system_message(state: Any) -> dict[str, str]:
    """state -> system 消息. 输入: 任意 EntryType; 输出: role/content 映射."""
    return {"role": "system", "content": render_entry(state)}


def _noul_user(question: Noul) -> str:
    """noul 的 user 消息: 问题 + true/false 释义 + 回答要求."""
    parts = [render_entry(question.instructions)]
    criteria = question.criteria or {}
    true_hint = render_entry(criteria.get("true"))
    false_hint = render_entry(criteria.get("false"))
    if true_hint:
        parts.append(f"回答 yes 表示: {true_hint}")
    if false_hint:
        parts.append(f"回答 no 表示: {false_hint}")
    parts.append("只回答 yes 或 no。")
    return "\n\n".join(parts)


def _choice_user(question: Choice, chunk: Chunk) -> str:
    """choice 的 user 消息: 本块的候选清单 (块内局部编号) + 回答要求."""
    items = list(question.criteria.items())[chunk.start : chunk.stop]
    lines = [render_entry(question.instructions), "", "候选列表:"]
    for local_index, (name, description) in enumerate(items):
        text = render_entry(description)
        lines.append(f"[{local_index}] {name}" + (f": {text}" if text else ""))
    if chunk.with_none:
        lines.append(NONE_HINT)
    lines += ["", "只回答编号本身。"]
    return "\n".join(lines)


def _score_user(question: Score) -> str:
    """score 的 user 消息: 有序档位清单 + 回答要求."""
    lines = [render_entry(question.instructions), "", "档位列表:"]
    for index, level in enumerate(question.criteria):
        lines.append(f"[{index}] {render_entry(level)}")
    lines += ["", "只回答档位编号本身。"]
    return "\n".join(lines)


def answer_noul(
    engine: BatchEngine,
    state: Any,
    question: Noul,
    *,
    usage: Usage | None = None,
) -> NoulAnswer:
    """noul 决策.

    输入: engine -- 批量引擎; state -- 请求级背景; question -- Noul;
    输出: NoulAnswer (noul = P(yes));
    预期: 走官方模板 + 思考段截断; 标签是 yes/no 的全书写变体组。
    """
    labels = resolve_labels(engine.tokenize_label, BOOL_LABEL_SPECS)
    tokens = engine.render_tokens(
        [_system_message(state), {"role": "user", "content": _noul_user(question)}],
        assistant_prefix=THINK_PREFIX,
    )
    _record_usage(usage, [tokens])
    probabilities = engine.score([tokens], labels)[0]
    return NoulAnswer(noul=probabilities["true"])


def answer_choice(
    engine: BatchEngine,
    state: Any,
    question: Choice,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usage: Usage | None = None,
) -> ChoiceAnswer:
    """choice 决策 (候选数不受限).

    输入: engine; state; question -- Choice; chunk_size -- 分块窗口 (0, 128];
    输出: ChoiceAnswer (choice = 全局概率最高的选项);
    预期: 单块不加锚点, 分块每块补 None 并用 log-odds 拉平; 概率和恒为 1。
    """
    plan = plan_chunks(len(question.criteria), chunk_size)
    sequences: list[list[int]] = []
    label_sets = []
    for chunk in plan.chunks:
        label_sets.append(
            engine.numeric_labels.get(chunk.count, with_none=chunk.with_none)
        )
        sequences.append(
            engine.render_tokens(
                [
                    _system_message(state),
                    {"role": "user", "content": _choice_user(question, chunk)},
                ],
                assistant_prefix=INDEX_PREFIX,
            )
        )
    _record_usage(usage, sequences)
    aligned = align_chunks(engine.score(sequences, label_sets))
    names = list(question.criteria)
    probabilities = {name: value for name, value in zip(names, aligned)}
    best = max(probabilities, key=lambda name: probabilities[name])
    return ChoiceAnswer(
        choice=best,
        probabilities=probabilities,
        confidence=normalized_peak(list(probabilities.values())),
    )


def answer_score(
    engine: BatchEngine,
    state: Any,
    question: Score,
    *,
    usage: Usage | None = None,
) -> ScoreAnswer:
    """score 决策: 档位上的概率加权值.

    输入: engine; state; question -- Score (2~10 档);
    输出: ScoreAnswer (score = Σ p_i · i, legend / probabilities 按官方形状回填);
    预期: 档位编号就是档位值, 不做百分制映射; 天然单块, 不带锚点。
    """
    level_count = len(question.criteria)
    labels = engine.numeric_labels.get(level_count, with_none=False)
    tokens = engine.render_tokens(
        [_system_message(state), {"role": "user", "content": _score_user(question)}],
        assistant_prefix=INDEX_PREFIX,
    )
    _record_usage(usage, [tokens])
    probabilities = engine.score([tokens], labels)[0]
    ordered = [probabilities[str(index)] for index in range(level_count)]
    score = weighted_level_score(ordered)
    return ScoreAnswer(
        score=score,
        legend={
            str(index): render_entry(level)
            for index, level in enumerate(question.criteria)
        },
        probabilities={
            str(index): probability for index, probability in enumerate(ordered)
        },
        confidence=normalized_peak(ordered),
    )


def answer(
    engine: BatchEngine,
    state: Any,
    question: Question,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usage: Usage | None = None,
) -> Answer:
    """按题型分发.

    输入: engine; state; question -- 三原语之一; chunk_size -- 仅 choice 用;
    输出: 对应的 Answer;
    预期: 未知类型直接抛 TypeError (调用方应先 parse_question 校验过).
    """
    if isinstance(question, Noul):
        return answer_noul(engine, state, question, usage=usage)
    if isinstance(question, Choice):
        return answer_choice(engine, state, question, chunk_size=chunk_size, usage=usage)
    if isinstance(question, Score):
        return answer_score(engine, state, question, usage=usage)
    raise TypeError(f"unsupported question type: {type(question).__name__}")
