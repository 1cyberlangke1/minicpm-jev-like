"""候选分块规划.

输入候选总数 N 与窗口大小 chunk_size, 输出每块的切片与是否带 ``None`` 锚点:

- ``N <= chunk_size``: 单块, **不带** 锚点;
- ``N > chunk_size``: ``ceil(N / chunk_size)`` 块, 每块末尾补一个 ``None`` 锚点
  (即每块 ``count + 1`` 个标签槽).

窗口上限 128 的理由: 实测候选数到 390~400 会出现注意力断崖, 而数字单 token 只到
999, 128 是「注意力甜点区间」与「编号仍是三位数」的交集.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "Chunk",
    "ChunkPlan",
    "plan_chunks",
]

#: 窗口上限: 再大就吃注意力断崖, 也没有单 token 编号可用
MAX_CHUNK_SIZE = 128
#: 默认窗口大小
DEFAULT_CHUNK_SIZE = 128


@dataclass(frozen=True)
class Chunk:
    """一个候选块: 全局切片 ``[start, start + count)``, 以及是否带 None 锚点."""

    start: int
    count: int
    with_none: bool

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"chunk start must be >= 0, got {self.start}")
        if self.count < 1:
            raise ValueError(f"chunk count must be >= 1, got {self.count}")

    @property
    def stop(self) -> int:
        """本块切片的右开边界 (全局下标)."""
        return self.start + self.count

    @property
    def slots(self) -> int:
        """本块占用的标签槽数 (真实候选 + 可能的锚点)."""
        return self.count + (1 if self.with_none else 0)


@dataclass(frozen=True)
class ChunkPlan:
    """一次分块的完整结果: 总数 / 窗口 / 各块切片."""

    total: int
    chunk_size: int
    chunks: tuple[Chunk, ...]

    @property
    def is_chunked(self) -> bool:
        """是否真的切了多块 (单块 = 不分块)."""
        return len(self.chunks) > 1

    @property
    def offsets(self) -> tuple[int, ...]:
        """每块第一个候选的全局下标, 用来把块内编号还原成全局编号."""
        return tuple(chunk.start for chunk in self.chunks)

    @property
    def max_slots(self) -> int:
        """单块最大标签槽数 (分块时是 chunk_size + 1)."""
        return max(chunk.slots for chunk in self.chunks)


def plan_chunks(total: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ChunkPlan:
    """把 total 个候选按 chunk_size 切片.

    输入: total -- 候选总数 (>= 1); chunk_size -- 窗口大小 (1 ~ MAX_CHUNK_SIZE);
    输出: ChunkPlan (单块不带锚点, 多块每块带锚点);
    预期: total < 1 或 chunk_size 越界直接抛 ValueError, 不静默修正.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if chunk_size < 1 or chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be in [1, {MAX_CHUNK_SIZE}], got {chunk_size}"
        )

    if total <= chunk_size:
        return ChunkPlan(
            total=total,
            chunk_size=chunk_size,
            chunks=(Chunk(start=0, count=total, with_none=False),),
        )

    chunks: list[Chunk] = []
    start = 0
    while start < total:
        count = min(chunk_size, total - start)
        chunks.append(Chunk(start=start, count=count, with_none=True))
        start += count
    return ChunkPlan(total=total, chunk_size=chunk_size, chunks=tuple(chunks))
