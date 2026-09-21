"""MiniCPM5 的 JEV/TypeSafe 风格 System One 受限决策引擎 (批量推理核心).

对外只暴露三块:
- template: 官方对话模板渲染 (绝不手搓 ChatML / 裸文本) 与 EntryType 序列化;
- labels: 受限决策标签组与组间受限 softmax;
- engine: 批量推理引擎, 一次 llama_decode 处理整批序列, 只在决策位读 logits.
"""

from .engine import BatchEngine, Device, EngineConfig, EngineError
from .labels import (
    BOOL_LABEL_SPECS,
    NONE_LABEL,
    LabelResolutionError,
    LabelSet,
    LabelSpec,
    NumericLabels,
    label_texts,
    resolve_labels,
)
from .template import (
    ChatTemplateError,
    EntryTypeError,
    get_chat_template,
    is_entry,
    render_chat,
    render_entry,
)

__all__ = [
    "BOOL_LABEL_SPECS",
    "BatchEngine",
    "ChatTemplateError",
    "Device",
    "EngineConfig",
    "EngineError",
    "EntryTypeError",
    "LabelResolutionError",
    "LabelSet",
    "LabelSpec",
    "NONE_LABEL",
    "NumericLabels",
    "get_chat_template",
    "is_entry",
    "label_texts",
    "render_chat",
    "render_entry",
    "resolve_labels",
]
