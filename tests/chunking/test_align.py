"""跨块对齐: log-odds 零点 + 全局 softmax."""

import math

import pytest

from minicpm_jev.chunking import (
    ChunkAlignError,
    align_chunks,
    chunk_log_odds,
    global_softmax,
)
from minicpm_jev.labels import NONE_LABEL


def test_single_chunk_without_anchor_is_identity() -> None:
    """单块没有锚点: 全局概率就是块内受限分布本身."""
    result = align_chunks([{"0": 0.6, "1": 0.4}])
    assert result == pytest.approx([0.6, 0.4])


def test_single_chunk_with_anchor_keeps_ratios() -> None:
    """带锚点的单块: 减去同一个零点后比值不变, 仍归一到候选上."""
    result = align_chunks([{"0": 0.3, "1": 0.3, NONE_LABEL: 0.4}])
    assert result == pytest.approx([0.5, 0.5])


def test_two_chunks_are_aligned_by_anchor() -> None:
    """跨块: 各块减掉自己的 P(None) 之后才能比较."""
    chunk0 = {"0": 0.6, "1": 0.2, NONE_LABEL: 0.2}
    chunk1 = {"0": 0.1, NONE_LABEL: 0.9}
    result = align_chunks([chunk0, chunk1])
    # 对数几率: ln3, 0, ln(1/9)
    assert result == pytest.approx([0.7297, 0.2432, 0.0270], abs=1e-4)
    assert sum(result) == pytest.approx(1.0)


def test_anchor_flips_ranking_across_chunks() -> None:
    """块内看起来弱的候选, 只要该块的 None 更强, 全局反而可能领先."""
    weak_local = {"0": 0.2, NONE_LABEL: 0.8}
    strong_local = {"0": 0.5, NONE_LABEL: 0.01}
    result = align_chunks([weak_local, strong_local])
    assert result[1] > result[0]


def test_candidate_order_follows_numeric_labels_not_mapping_order() -> None:
    """候选顺序按标签数字升序, 不受 dict 插入顺序影响."""
    shuffled = {"2": 0.2, "0": 0.3, "10": 0.1, NONE_LABEL: 0.4}
    scores = chunk_log_odds(shuffled)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(math.log(0.3 / 0.4))
    assert scores[1] == pytest.approx(math.log(0.2 / 0.4))
    assert scores[2] == pytest.approx(math.log(0.1 / 0.4))


def test_zero_probability_candidate_gets_zero_globally() -> None:
    """块内概率为 0 的候选, 全局概率也是 0 (不是极小值)."""
    result = align_chunks([{"0": 1.0, "1": 0.0, NONE_LABEL: 1e-12}])
    assert result[1] == 0.0
    assert result[0] == pytest.approx(1.0)


def test_tiny_probabilities_do_not_overflow() -> None:
    """极小的概率走对数空间, 不会下溢成 nan."""
    result = align_chunks([{"0": 1e-300, "1": 1e-300, NONE_LABEL: 1.0}])
    assert result == pytest.approx([0.5, 0.5])


def test_multi_chunk_missing_anchor_is_rejected() -> None:
    """分块时缺锚点说明分块没按铁律走, 直接报错不猜."""
    with pytest.raises(ChunkAlignError) as error:
        align_chunks([{"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}])
    assert "anchor" in str(error.value)
    assert "chunk 0" in str(error.value)


def test_zero_anchor_is_rejected() -> None:
    """锚点概率为 0 时零点不存在, 不能假装能对齐."""
    with pytest.raises(ChunkAlignError) as error:
        chunk_log_odds({"0": 0.5, "1": 0.5, NONE_LABEL: 0.0})
    assert "zero probability" in str(error.value)


def test_all_candidates_zero_is_rejected() -> None:
    """全体候选概率为 0 时全局分布无定义."""
    with pytest.raises(ChunkAlignError):
        align_chunks([{"0": 0.0, "1": 0.0, NONE_LABEL: 1.0}])


def test_negative_probability_is_rejected() -> None:
    """负概率不是概率, 直接报错."""
    with pytest.raises(ValueError):
        chunk_log_odds({"0": -0.1, NONE_LABEL: 0.5})


def test_non_numeric_label_is_rejected() -> None:
    """候选标签必须是数字编号, 其它文本说明调用方用错了接口."""
    with pytest.raises(ChunkAlignError) as error:
        chunk_log_odds({"blue": 0.5, NONE_LABEL: 0.5})
    assert "blue" in str(error.value)


def test_empty_inputs_are_rejected() -> None:
    """空块列表 / 空对数几率都没有意义."""
    with pytest.raises(ValueError):
        align_chunks([])
    with pytest.raises(ValueError):
        global_softmax([])


def test_global_softmax_handles_negative_infinity() -> None:
    """-inf 表示确定不可能, 不参与归一."""
    result = global_softmax([0.0, -math.inf, 0.0])
    assert result == pytest.approx([0.5, 0.0, 0.5])


def test_alignment_is_scale_invariant_per_chunk() -> None:
    """同一块内所有概率同乘一个正数, 对齐结果不变 (锚点消掉公共因子)."""
    base = {"0": 0.3, "1": 0.2, NONE_LABEL: 0.5}
    scaled = {name: value * 7.5 for name, value in base.items()}
    assert align_chunks([base]) == pytest.approx(align_chunks([scaled]))
