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


def test_exactly_chunk_size_stays_single_chunk() -> None:
    """刚好等于窗口大小仍是单块 (边界不能切)."""
    plan = plan_chunks(128)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].with_none is False
    assert plan.chunks[0].count == 128
    assert plan.max_slots == 128


def test_one_over_chunk_size_splits_with_anchor() -> None:
    """129 个候选 -> 两块, 每块都带锚点."""
    plan = plan_chunks(129)
    assert [chunk.count for chunk in plan.chunks] == [128, 1]
    assert all(chunk.with_none for chunk in plan.chunks)
    assert plan.is_chunked is True
    assert plan.max_slots == 129


def test_two_full_chunks() -> None:
    """256 个候选正好两块满块."""
    plan = plan_chunks(256)
    assert [chunk.start for chunk in plan.chunks] == [0, 128]
    assert [chunk.count for chunk in plan.chunks] == [128, 128]


def test_tail_chunk_holds_remainder() -> None:
    """257 个候选 -> 三块, 尾巴只有 1 个."""
    plan = plan_chunks(257)
    assert [chunk.count for chunk in plan.chunks] == [128, 128, 1]
    assert plan.chunks[-1].start == 256


def test_custom_chunk_size() -> None:
    """窗口可调: 10 个候选按 4 切 -> 4/4/2."""
    plan = plan_chunks(10, chunk_size=4)
    assert [chunk.count for chunk in plan.chunks] == [4, 4, 2]
    assert [chunk.start for chunk in plan.chunks] == [0, 4, 8]
    assert all(chunk.with_none for chunk in plan.chunks)


def test_custom_chunk_size_single_chunk() -> None:
    """窗口调大后可能又变回单块 (且不带锚点)."""
    plan = plan_chunks(10, chunk_size=10)
    assert plan.is_chunked is False
    assert plan.chunks[0].with_none is False


def test_chunks_are_contiguous_and_cover_all_candidates() -> None:
    """切片必须无缝覆盖 [0, total): 不重不漏."""
    plan = plan_chunks(1000)
    assert plan.chunks[0].start == 0
    for previous, current in zip(plan.chunks, plan.chunks[1:]):
        assert previous.stop == current.start
    assert plan.chunks[-1].stop == 1000
    assert sum(chunk.count for chunk in plan.chunks) == 1000


def test_offsets_give_global_index_of_each_chunk() -> None:
    """offsets 就是每块第一个候选的全局下标."""
    assert plan_chunks(300, chunk_size=100).offsets == (0, 100, 200)
    assert plan_chunks(50, chunk_size=100).offsets == (0,)


@pytest.mark.parametrize("total", [0, -1])
def test_rejects_non_positive_total(total: int) -> None:
    """没有候选就不该分块."""
    with pytest.raises(ValueError):
        plan_chunks(total)


@pytest.mark.parametrize("chunk_size", [0, -1, MAX_CHUNK_SIZE + 1, 1000])
def test_rejects_out_of_range_chunk_size(chunk_size: int) -> None:
    """窗口必须在 (0, 128] 内, 越界直接报错不截断."""
    with pytest.raises(ValueError):
        plan_chunks(10, chunk_size=chunk_size)


def test_default_chunk_size_is_documented_value() -> None:
    """默认窗口 128 (甜点区间上沿), 常量对外可见."""
    assert DEFAULT_CHUNK_SIZE == 128
    assert MAX_CHUNK_SIZE == 128
    assert plan_chunks(500).chunk_size == 128


def test_chunk_rejects_bad_slice() -> None:
    """Chunk 自身也守住边界: 负数起点 / 空块都不允许."""
    with pytest.raises(ValueError):
        Chunk(start=-1, count=1, with_none=False)
    with pytest.raises(ValueError):
        Chunk(start=0, count=0, with_none=False)


def test_chunk_slots_include_anchor() -> None:
    """带锚点的块槽数 = 候选数 + 1."""
    assert Chunk(start=0, count=128, with_none=True).slots == 129
    assert Chunk(start=0, count=128, with_none=False).slots == 128
