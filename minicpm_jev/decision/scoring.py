"""三原语打分: 把 (state, question) 变成官方形状的答案.

prompt 结构 (铁律, 不许手搓标记):

- system = state (结构化就原样 JSON 序列化);
- user = instructions + 候选/档位清单 + 回答要求;
- assistant 预填 = 思考段截断 + 标签引导 (choice / score 用 ``[``, noul 不加),
  走库函数的 ``assistant_prefix``, 不碰模板标记。

批 (一次请求一次 decode):

- 先把每道题**规划**成若干条序列 + 每条的标签集 + 一个解码器;
- 所有题目的序列拼成**一个**批交给引擎, 只调一次 ``engine.score``;
- 再按切片把结果还回各题 —— 这样同一个 state 前缀只在一批里算, 不逐题单发。

分块 (choice 的候选数不受限):

- ``N <= chunk_size`` 单块, 标签 ``0`` ~ ``N-1``, **不带锚点**;
- ``N > chunk_size`` 每块 ``0`` ~ ``m-1`` 再补 ``None``, 跨块用 log-odds 对齐。

``score`` 档位最多 10 个, 天然单块; 分是档位上的概率加权值 ``Σ p_i · i``。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..chunking import DEFAULT_CHUNK_SIZE, Chunk, align_chunks, plan_chunks
from ..labels import BOOL_LABEL_SPECS, LabelSet, resolve_labels
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
    "answer_all",
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

# 句式约定: 背景与候选同处一条 user 消息, 结尾给【要求】, 让模型在同一段上下文里
# 同时看到判定标准与候选范围。注意 [None] 只在**真的带锚点**时才能出现在提示里 ——
# 单块不带锚点时标签集里没有 None, 提示却许诺 None 会让模型把票投给一个不存在的标签。
NOUL_TASK = "【任务】根据以下背景回答问题。只允许回答 yes 或 no。"
SCORE_TASK = "【任务】根据以下背景与档位列表，选出最符合的一个档位。"
CONTEXT_HEADER = "【场景背景】"
QUESTION_HEADER = "【问题】"
CANDIDATE_HEADER = "【候选列表】"
LEVEL_HEADER = "【档位列表】"
REQUIREMENT_SCORE = "【要求】仅输出档位标签（如 [0]）。"


def _choice_task(with_none: bool) -> str:
    """choice 任务句. 输入: 本块是否带 None 锚点; 输出: 与标签集一致的措辞."""
    if with_none:
        return (
            "【任务】从以下候选中选出最正确的一项。"
            "如果所有候选都不合适，选择 [None]。"
        )
    return "【任务】从以下候选中选出最正确的一项。"


def _choice_requirement(with_none: bool) -> str:
    """choice 要求句. 输入: 本块是否带 None 锚点; 输出: 与标签集一致的措辞."""
    if with_none:
        return "【要求】仅输出选项标签（如 [0] 或 [None]）。"
    return "【要求】仅输出选项标签（如 [0]）。"

#: 每道题解出来的概率 (标签名 -> 概率) 喂给解码器
PerSequence = Sequence[Mapping[str, float]]
#: 把「属于自己那道题的那几条概率」解成答案
Decoder = Callable[[PerSequence], Answer]


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


def _context_block(state: Any) -> str:
    """背景块. 输入: 任意 EntryType; 输出: 【场景背景】+ 渲染后的文本."""
    return f"{CONTEXT_HEADER}\n{render_entry(state)}"


def _user_message(prompt: str) -> list[dict[str, str]]:
    """整条 prompt 只作一条 user 消息 (与实验脚本一致)."""
    return [{"role": "user", "content": prompt}]


def _noul_prompt(state: Any, question: Noul) -> str:
    """noul 的完整 prompt: 任务 + 背景 + 问题 (+ true/false 释义).

    任务句已声明「只允许回答 yes 或 no」, 结尾不再补一条同义要求 —— 决策位前
    若压着一条元指令, 模型会去续写指令本身, 而不是回答问句。
    """
    parts = [
        NOUL_TASK,
        _context_block(state),
        f"{QUESTION_HEADER}\n{render_entry(question.instructions)}",
    ]
    criteria = question.criteria or {}
    true_hint = render_entry(criteria.get("true"))
    false_hint = render_entry(criteria.get("false"))
    if true_hint or false_hint:
        lines = []
        if true_hint:
            lines.append(f"yes 表示：{true_hint}")
        if false_hint:
            lines.append(f"no 表示：{false_hint}")
        parts.append("【判定标准】\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _choice_prompt(state: Any, question: Choice, chunk: Chunk) -> str:
    """choice 的完整 prompt: 任务 + 背景 + 问题 + 本块候选 (块内局部编号) + 要求."""
    items = list(question.criteria.items())[chunk.start : chunk.stop]
    options: list[str] = []
    for local_index, (name, description) in enumerate(items):
        text = render_entry(description)
        options.append(f"[{local_index}] {name}" + (f"：{text}" if text else ""))
    if chunk.with_none:
        options.append(NONE_HINT)
    return "\n\n".join(
        [
            _choice_task(chunk.with_none),
            _context_block(state),
            f"{QUESTION_HEADER}\n{render_entry(question.instructions)}",
            CANDIDATE_HEADER + "\n" + "\n".join(options),
            _choice_requirement(chunk.with_none),
        ]
    )


def _score_prompt(state: Any, question: Score) -> str:
    """score 的完整 prompt: 任务 + 背景 + 问题 + 档位表 + 要求."""
    levels = [
        f"[{index}] {render_entry(level)}"
        for index, level in enumerate(question.criteria)
    ]
    return "\n\n".join(
        [
            SCORE_TASK,
            _context_block(state),
            f"{QUESTION_HEADER}\n{render_entry(question.instructions)}",
            LEVEL_HEADER + "\n" + "\n".join(levels),
            REQUIREMENT_SCORE,
        ]
    )


@dataclass(frozen=True)
class _Plan:
    """一道题在整批里的切片与解码方式."""

    sequences: list[list[int]]
    label_sets: list[LabelSet]
    decode: Decoder


def _plan_noul(engine: BatchEngine, state: Any, question: Noul) -> _Plan:
    """noul 规划: 一条序列, 标签是 yes/no 全书写变体组."""
    labels = resolve_labels(engine.tokenize_label, BOOL_LABEL_SPECS)
    tokens = engine.render_tokens(
        _user_message(_noul_prompt(state, question)),
        assistant_prefix=THINK_PREFIX,
    )

    def decode(probabilities: PerSequence) -> Answer:
        return NoulAnswer(noul=probabilities[0]["true"])

    return _Plan([tokens], [labels], decode)


def _plan_choice(
    engine: BatchEngine,
    state: Any,
    question: Choice,
    chunk_size: int,
) -> _Plan:
    """choice 规划: 每个候选块一条序列, 块内局部编号 (+ 分块时的 None 锚点)."""
    plan = plan_chunks(len(question.criteria), chunk_size)
    sequences: list[list[int]] = []
    label_sets: list[LabelSet] = []
    for chunk in plan.chunks:
        label_sets.append(
            engine.numeric_labels.get(chunk.count, with_none=chunk.with_none)
        )
        sequences.append(
            engine.render_tokens(
                _user_message(_choice_prompt(state, question, chunk)),
                assistant_prefix=INDEX_PREFIX,
            )
        )
    names = list(question.criteria)

    def decode(probabilities: PerSequence) -> Answer:
        aligned = align_chunks(probabilities)
        combined = {name: value for name, value in zip(names, aligned)}
        best = max(combined, key=lambda name: combined[name])
        return ChoiceAnswer(
            choice=best,
            probabilities=combined,
            confidence=normalized_peak(list(combined.values())),
        )

    return _Plan(sequences, label_sets, decode)


def _plan_score(engine: BatchEngine, state: Any, question: Score) -> _Plan:
    """score 规划: 一条序列, 档位编号 0~M-1 天然单块 (不带锚点)."""
    level_count = len(question.criteria)
    labels = engine.numeric_labels.get(level_count, with_none=False)
    tokens = engine.render_tokens(
        _user_message(_score_prompt(state, question)),
        assistant_prefix=INDEX_PREFIX,
    )

    def decode(probabilities: PerSequence) -> Answer:
        single = probabilities[0]
        ordered = [single[str(index)] for index in range(level_count)]
        return ScoreAnswer(
            score=weighted_level_score(ordered),
            legend={
                str(index): render_entry(level)
                for index, level in enumerate(question.criteria)
            },
            probabilities={
                str(index): probability for index, probability in enumerate(ordered)
            },
            confidence=normalized_peak(ordered),
        )

    return _Plan([tokens], [labels], decode)


def _plan(
    engine: BatchEngine,
    state: Any,
    question: Question,
    chunk_size: int,
) -> _Plan:
    """按题型规划. 预期: 未知类型抛 TypeError, 不静默跳过."""
    if isinstance(question, Noul):
        return _plan_noul(engine, state, question)
    if isinstance(question, Choice):
        return _plan_choice(engine, state, question, chunk_size)
    if isinstance(question, Score):
        return _plan_score(engine, state, question)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def answer_all(
    engine: BatchEngine,
    state: Any,
    questions: Mapping[str, Question],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usage: Usage | None = None,
) -> dict[str, Answer]:
    """一次请求里的所有问题走**一个**批.

    输入: engine; state; questions -- 题名 -> 题目; chunk_size -- 候选窗口;
          usage -- 可选统计累加器;
    输出: 题名 -> 答案 (顺序与输入一致);
    预期: 只调一次 engine.score (同批序列由引擎按哈希去重并并行 prefill);
          questions 为空抛 ValueError, 不返回空结果。
    """
    if not questions:
        raise ValueError("questions must not be empty")

    plans = {
        question_id: _plan(engine, state, question, chunk_size)
        for question_id, question in questions.items()
    }
    sequences: list[list[int]] = []
    label_sets: list[LabelSet] = []
    for plan in plans.values():
        sequences.extend(plan.sequences)
        label_sets.extend(plan.label_sets)

    _record_usage(usage, sequences)
    scored = engine.score(sequences, label_sets)

    results: dict[str, Answer] = {}
    cursor = 0
    for question_id, plan in plans.items():
        width = len(plan.sequences)
        results[question_id] = plan.decode(scored[cursor : cursor + width])
        cursor += width
    return results


def answer(
    engine: BatchEngine,
    state: Any,
    question: Question,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usage: Usage | None = None,
) -> Answer:
    """单题决策 (answer_all 的薄包装, 方便按题调用与测试).

    输入: engine; state; question -- 三原语之一; chunk_size; usage;
    输出: 对应答案;
    预期: 走与批量完全相同的代码路径, 不另起一套。
    """
    return answer_all(
        engine,
        state,
        {"question": question},
        chunk_size=chunk_size,
        usage=usage,
    )["question"]


def answer_noul(engine: BatchEngine, state: Any, question: Noul) -> NoulAnswer:
    """noul 单题便捷入口. 输出: NoulAnswer (noul = P(yes))."""
    result = answer(engine, state, question)
    if not isinstance(result, NoulAnswer):  # pragma: no cover - 类型收窄
        raise TypeError("expected NoulAnswer")
    return result


def answer_choice(
    engine: BatchEngine,
    state: Any,
    question: Choice,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ChoiceAnswer:
    """choice 单题便捷入口. 输出: ChoiceAnswer (choice = 全局概率最高项)."""
    result = answer(engine, state, question, chunk_size=chunk_size)
    if not isinstance(result, ChoiceAnswer):  # pragma: no cover - 类型收窄
        raise TypeError("expected ChoiceAnswer")
    return result


def answer_score(engine: BatchEngine, state: Any, question: Score) -> ScoreAnswer:
    """score 单题便捷入口. 输出: ScoreAnswer (score = Σ p_i · i)."""
    result = answer(engine, state, question)
    if not isinstance(result, ScoreAnswer):  # pragma: no cover - 类型收窄
        raise TypeError("expected ScoreAnswer")
    return result
