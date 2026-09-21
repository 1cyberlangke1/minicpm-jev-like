"""置信度: 把概率分布的集中程度压成一个 0~1 的数.

官方只说 ``confidence`` 是「由概率分布导出、0 到 1」, 公式没公开。这里用**归一化
峰值**:

    confidence = clamp((K · p_max - 1) / (K - 1), 0, 1)

``K`` 是分布里的标签数。它的性质: 均匀分布得 0, 独热得 1, 单标签 (K = 1) 给 1,
且对同一分布单调 —— 两个原语共用同一口径, 阈值经验可以直接迁移。

本模块只做这一个统计量, 不碰 token / 模型。
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["normalized_peak"]


def normalized_peak(probabilities: Sequence[float]) -> float:
    """归一化峰值置信度.

    输入: probabilities -- 某个受限分布的各标签概率 (长度 >= 1);
    输出: [0, 1] 区间的置信度;
    预期: 空输入 / 负概率 / 超过 1 的概率都抛 ValueError, 不静默截断数据。
    """
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"probability out of range: {value}")

    count = len(values)
    if count == 1:
        return 1.0
    peak = max(values)
    raw = (count * peak - 1.0) / (count - 1.0)
    return min(1.0, max(0.0, raw))
