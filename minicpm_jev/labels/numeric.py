"""数字标签编译: ``0`` ~ ``n-1`` 与 ``None`` 锚点.

分块打分用的标签就是**数字本身**: 决策位前缀 (助手预填) 收尾在 ``[``, 模型接着吐的
就是编号 token. 词表实测 ``0``~``999`` 全是原生单 token, 所以编号能一对一落到 token
上; 注意 token id 不连续 (不是 ``id("0") + n``), 必须逐个查表.

两种形态 (与分块铁律一致):

- 单块 (``N <= chunk_size``): 只要 ``0`` ~ ``N-1``, **不带 ``None``**;
- 分块 (``N > chunk_size``): 每块 ``0`` ~ ``m-1``, 末尾补一个 ``None`` 当跨块对齐
  锚点.

``NumericLabels`` 是带缓存的工厂: 同一个 ``(count, with_none)`` 只编译一次, 编译结果
不可变, 可以安全复用.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .specs import LabelResolutionError, LabelSet

__all__ = ["NONE_LABEL", "NumericLabels", "label_texts"]

#: 跨块对齐锚点的标签文本
NONE_LABEL = "None"


def label_texts(count: int, *, with_none: bool = False) -> tuple[str, ...]:
    """给出该块的标签文本序列.

    输入: count -- 本块真实候选数 (>= 1); with_none -- 是否补 None 锚点;
    输出: ("0", "1", ..., "count-1"), 需要锚点时末尾追加 "None";
    预期: count < 1 直接抛 ValueError, 不返回空标签集.
    """
    if count < 1:
        raise ValueError(f"label count must be >= 1, got {count}")
    texts = [str(index) for index in range(count)]
    if with_none:
        texts.append(NONE_LABEL)
    return tuple(texts)


class NumericLabels:
    """数字标签工厂: 把标签文本编译成单 token 标签集, 并缓存结果."""

    def __init__(self, tokenize: Callable[[str], Sequence[int]]) -> None:
        """输入: tokenize -- 文本 -> token id 序列 (add_bos=False, special=False);
        输出: 无; 预期: 不做任何 IO, 编译推迟到第一次 get."""
        self._tokenize = tokenize
        self._cache: dict[tuple[int, bool], LabelSet] = {}

    @property
    def cached_entries(self) -> int:
        """当前缓存条目数 (测试与监控用)."""
        return len(self._cache)

    def get(self, count: int, *, with_none: bool = False) -> LabelSet:
        """取 (并缓存) 一组数字标签.

        输入: count -- 真实候选数; with_none -- 是否带 None 锚点;
        输出: 每个标签一个组的 LabelSet, 组名就是标签文本;
        预期: 任何标签切不成单 token 抛 LabelResolutionError; count < 1 抛 ValueError.
        """
        key = (count, with_none)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        names = label_texts(count, with_none=with_none)
        groups: list[tuple[int, ...]] = []
        for name in names:
            tokens = list(self._tokenize(name))
            if len(tokens) != 1:
                raise LabelResolutionError(
                    f"numeric label {name!r} tokenizes to {len(tokens)} tokens, "
                    "expected exactly 1"
                )
            groups.append((tokens[0],))
        label_set = LabelSet(names=names, token_ids=tuple(groups))
        self._cache[key] = label_set
        return label_set
