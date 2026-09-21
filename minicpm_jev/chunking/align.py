"""跨块对齐: 把各块的受限分布拉到同一个零点上.

第 c 块第 j 个候选:

    Score(c, j) = ln P(候选) - ln P(None_c)

减掉本块的 ``P(None)`` 就消掉了该块自己的对数配分常数, 不同块的分才可比。
单块没有锚点, 直接取 ``ln P(候选)`` —— 它本来就是全局分布, 是这条公式在「只有一块
且锚点权重为 1」时的特例。

汇总所有候选后剔除 ``None`` 做全局 softmax:

    P_global(i) = exp(Score(i)) / Σ_j exp(Score(j))

本模块只做数学, 不碰 token / 模型; 输入是各块的受限概率, 输出是全局概率。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..labels.numeric import NONE_LABEL

__all__ = ["ChunkAlignError", "align_chunks", "chunk_log_odds", "global_softmax"]


class ChunkAlignError(ValueError):
    """对齐前提被破坏: 缺锚点 / 锚点概率为 0 / 全体候选概率为 0."""


def _log(probability: float) -> float:
    """概率 -> 对数; 0 取 -inf (表示该候选全局概率为 0), 负数由调用方拦下."""
    return math.log(probability) if probability > 0.0 else -math.inf


def _real_labels(probabilities: Mapping[str, float]) -> list[tuple[int, float]]:
    """挑出真实候选并按其编号升序排好.

    输入: 标签文本 -> 概率的映射 (可能含 None 锚点);
    输出: [(编号, 概率), ...], 编号升序;
    预期: 负概率抛 ValueError; 非数字标签抛 ChunkAlignError (契约要求数字标签).
    """
    labels: list[tuple[int, float]] = []
    for name, probability in probabilities.items():
        if probability < 0.0:
            raise ValueError(f"label {name!r} has negative probability {probability}")
        if name == NONE_LABEL:
            continue
        try:
            index = int(name)
        except (TypeError, ValueError) as error:
            raise ChunkAlignError(
                f"label {name!r} is not a numeric candidate index"
            ) from error
        labels.append((index, probability))
    labels.sort(key=lambda item: item[0])
    return labels


def chunk_log_odds(probabilities: Mapping[str, float]) -> list[float]:
    """把一块的受限分布换算成逐候选对数几率.

    输入: 标签文本 -> 概率 (带 None 锚点就减掉 ln P(None));
    输出: 对数几率列表, 顺序 = 候选编号升序 (不含锚点);
    预期: 锚点概率为 0 时抛 ChunkAlignError —— 零点没了, 跨块比较无意义, 不糊弄.
    """
    labels = _real_labels(probabilities)
    anchor = probabilities.get(NONE_LABEL)
    if anchor is not None and anchor <= 0.0:
        raise ChunkAlignError(
            f"chunk anchor {NONE_LABEL!r} has zero probability; cannot align"
        )
    anchor_log = _log(anchor) if anchor is not None else 0.0
    return [_log(probability) - anchor_log for _index, probability in labels]


def global_softmax(scores: Sequence[float]) -> list[float]:
    """对数几率 -> 全局概率 (数值稳定).

    输入: 逐候选对数几率 (可含 -inf, 表示该候选概率为 0);
    输出: 概率列表, 和为 1, 与输入等长;
    预期: 空输入抛 ValueError; 全体 -inf 抛 ChunkAlignError.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    best = max(scores)
    if best == -math.inf:
        raise ChunkAlignError("all candidates have zero probability")
    weights = [
        math.exp(score - best) if score != -math.inf else 0.0 for score in scores
    ]
    total = sum(weights)
    if total <= 0.0:
        raise ChunkAlignError("softmax weights collapsed to zero")
    return [weight / total for weight in weights]


def align_chunks(chunks: Sequence[Mapping[str, float]]) -> list[float]:
    """多块受限分布 -> 全局概率分布.

    输入: 每块的 {标签: 概率}, 块序 = 候选切片顺序;
    输出: 全局概率列表, 顺序 = 块序 × 块内编号升序, 和为 1;
    预期: 多于一块时任何一块缺 None 锚点都抛 ChunkAlignError (说明分块没按铁律走),
          不偷偷当成单块处理.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) > 1:
        for index, chunk in enumerate(chunks):
            if NONE_LABEL not in chunk:
                raise ChunkAlignError(
                    f"chunk {index} is missing the {NONE_LABEL!r} anchor"
                )
    scores = [score for chunk in chunks for score in chunk_log_odds(chunk)]
    return global_softmax(scores)
