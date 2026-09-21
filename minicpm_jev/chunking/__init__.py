"""候选分块与跨块对齐.

两块:
- plan: 按 chunk_size 切片, 单块不加锚点, 分块时每块末尾补 None;
- align: 用 ln P(候选) - ln P(None) 把各块拉到同一零点, 再全局 softmax.
"""

from .align import ChunkAlignError, align_chunks, chunk_log_odds, global_softmax
from .plan import DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, Chunk, ChunkPlan, plan_chunks

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "Chunk",
    "ChunkAlignError",
    "ChunkPlan",
    "align_chunks",
    "chunk_log_odds",
    "global_softmax",
    "plan_chunks",
]
