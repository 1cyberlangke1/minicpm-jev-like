"""受限决策标签.

两块:
- specs: 标签组 / 标签集 / 组间受限 softmax, 以及 bool 的书写变体表;
- numeric: 数字标签 ([0] ~ [n-1] 与 [None] 锚点) 的编译与缓存.
"""

from .numeric import NONE_LABEL, NumericLabels, label_texts
from .specs import (
    BOOL_LABEL_SPECS,
    LabelResolutionError,
    LabelSet,
    LabelSpec,
    resolve_labels,
)

__all__ = [
    "BOOL_LABEL_SPECS",
    "NONE_LABEL",
    "LabelResolutionError",
    "LabelSet",
    "LabelSpec",
    "NumericLabels",
    "label_texts",
    "resolve_labels",
]
