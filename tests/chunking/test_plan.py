"""分块规划: 单块不加锚点, 多块每块带锚点."""

import pytest

from minicpm_jev.chunking import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    Chunk,
    plan_chunks,
)


def test_single_chunk_when_total_fits() -> None:
    """N <= chunk_size 时单块, 且不带锚点."""
    plan = plan_chunks(1)
    assert plan.chunks == (Chunk(start=0, count=1, with_none=False),)
    assert plan.is_chunked is False
    assert plan.max_slots == 1




def test_one_over_chunk_size_splits_with_anchor() -> None:
    """129 个候选 -> 两块, 每块都带锚点."""
    plan = plan_chunks(129)
    assert [chunk.count for chunk in plan.chunks] == [128, 1]
    assert all(chunk.with_none for chunk in plan.chunks)
    assert plan.is_chunked is True
    assert plan.max_slots == 129




def test_tail_chunk_holds_remainder() -> None:
    """257 个候选 -> 三块, 尾巴只有 1 个."""
    plan = plan_chunks(257)
    assert [chunk.count for chunk in plan.chunks] == [128, 128, 1]
    assert plan.chunks[-1].start == 256






def test_chunks_are_contiguous_and_cover_all_candidates() -> None:
    """切片必须无缝覆盖 [0, total): 不重不漏."""
    plan = plan_chunks(1000)
    assert plan.chunks[0].start == 0
    for previous, current in zip(plan.chunks, plan.chunks[1:]):
        assert previous.stop == current.start
    assert plan.chunks[-1].stop == 1000
    assert sum(chunk.count for chunk in plan.chunks) == 1000






@pytest.mark.parametrize("chunk_size", [0, -1, MAX_CHUNK_SIZE + 1, 1000])
def test_rejects_out_of_range_chunk_size(chunk_size: int) -> None:
    """窗口必须在 (0, 128] 内, 越界直接报错不截断."""
    with pytest.raises(ValueError):
        plan_chunks(10, chunk_size=chunk_size)






