"""受限决策标签: 标签书写变体 -> 单 token 组, 组间做受限 softmax.

决策位的 logits 只允许落在给定标签 token 上: 同一极性的全部书写变体
(yes/Yes/YES/...) 的概率质量求和, 再在组间归一. 非标签 token 的质量直接丢弃,
不参与归一 —— 这正是「受限决策」与自由生成的分野.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BOOL_LABEL_SPECS",
    "LabelResolutionError",
    "LabelSet",
    "LabelSpec",
    "resolve_labels",
]


class LabelResolutionError(ValueError):
    """标签变体切不成单个 token, 或同一个 token 落在两个组里."""


@dataclass(frozen=True)
class LabelSpec:
    """一个标签组: 组名 + 该组全部书写变体."""

    name: str
    variants: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("标签组名不能为空")
        if not self.variants:
            raise ValueError(f"标签组 {self.name} 至少需要一个变体")


@dataclass(frozen=True)
class LabelSet:
    """解析完成的标签集: 组名 + 每组的单 token id."""

    names: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("标签集至少需要一个组")
        if len(self.names) != len(self.token_ids):
            raise ValueError("names 与 token_ids 长度不一致")
        owner: dict[int, str] = {}
        for name, ids in zip(self.names, self.token_ids):
            if not ids:
                raise ValueError(f"标签组 {name} 没有任何 token")
            for token_id in ids:
                previous = owner.get(token_id)
                if previous is not None:
                    raise LabelResolutionError(
                        f"token {token_id} 同时属于标签组 {previous} 与 {name}"
                    )
                owner[token_id] = name

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """全部标签 token, 顺序与 probabilities 的权重切分一致."""
        return tuple(token_id for ids in self.token_ids for token_id in ids)

    def probabilities(self, logits: np.ndarray) -> dict[str, float]:
        """在决策位 logits 上做组间受限 softmax.

        输入: logits -- 长度 n_vocab 的决策位 logits
        输出: 组名 -> 概率, 各组之和为 1
        预期: 数值稳定 (先减标签内最大值), 非标签 token 完全不参与
        """
        values = np.asarray(logits, dtype=np.float64)
        picked = values[np.asarray(self.all_token_ids, dtype=np.int64)]
        weights = np.exp(picked - picked.max())
        total = float(weights.sum())
        result: dict[str, float] = {}
        cursor = 0
        for name, ids in zip(self.names, self.token_ids):
            result[name] = float(weights[cursor : cursor + len(ids)].sum()) / total
            cursor += len(ids)
        return result


def resolve_labels(
    tokenize: Callable[[str], Sequence[int]], specs: Sequence[LabelSpec]
) -> LabelSet:
    """把标签变体解析成单 token id.

    输入: tokenize -- 字符串 -> token id 序列 (add_bos=False, special=False 的切法);
          specs -- 标签组定义
    输出: 可直接用于打分的 LabelSet
    预期: 任何多 token 变体直接抛 LabelResolutionError, 不截断也不取首 token
    """
    names: list[str] = []
    groups: list[tuple[int, ...]] = []
    for spec in specs:
        ids: list[int] = []
        for variant in spec.variants:
            tokens = list(tokenize(variant))
            if len(tokens) != 1:
                raise LabelResolutionError(
                    f"标签组 {spec.name} 的变体 {variant!r} 被切成 {len(tokens)} 个 token, "
                    "无法参与决策位受限打分"
                )
            if tokens[0] not in ids:
                ids.append(tokens[0])
        names.append(spec.name)
        groups.append(tuple(ids))
    return LabelSet(tuple(names), tuple(groups))




# 已验证 (MiniCPM5-2B Q4_K_M, n_vocab=130560):
#   真组 12 个变体全单 token; 假组 11 个变体全单 token.
#   唯一的例外是裸 "FALSE" (无前导空格), 它被切成 [59, 44794] 两个 token,
#   所以不在下面的清单里 —— 带上它 resolve_labels 会直接报错.
BOOL_LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(
        "true",
        (
            " yes",
            " Yes",
            " YES",
            "yes",
            "Yes",
            "YES",
            " true",
            " True",
            " TRUE",
            "true",
            "True",
            "TRUE",
        ),
    ),
    LabelSpec(
        "false",
        (
            " no",
            " No",
            " NO",
            "no",
            "No",
            "NO",
            " false",
            " False",
            " FALSE",
            "false",
            "False",
        ),
    ),
)
