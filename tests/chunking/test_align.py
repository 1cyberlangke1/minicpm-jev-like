"""跨块对齐: log-odds 零点 + 全局 softmax."""

import pytest

from minicpm_jev.chunking import (
    ChunkAlignError,
    align_chunks,
    chunk_log_odds,
)
from minicpm_jev.labels import NONE_LABEL


def test_single_chunk_without_anchor_is_identity() -> None:
    """单块没有锚点: 全局概率就是块内受限分布本身."""
    result = align_chunks([{"0": 0.6, "1": 0.4}])
    assert result == pytest.approx([0.6, 0.4])




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












