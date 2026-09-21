"""MiniCPM5 的 JEV/TypeSafe 风格 System One 受限决策引擎 (批量推理核心).

对外只暴露三块:
- chat_template: 官方对话模板渲染 (绝不手搓 ChatML / 裸文本);
- labels: 受限决策标签组与组间受限 softmax;
- engine: 批量推理引擎, 一次 llama_decode 处理整批序列, 只在决策位读 logits.
"""

from .chat_template import ChatTemplateError, get_chat_template, render_chat
from .engine import BatchEngine, Device, EngineConfig, EngineError
from .labels import (
    BOOL_LABEL_SPECS,
    LabelResolutionError,
    LabelSet,
    LabelSpec,
    resolve_labels,
)

__all__ = [
    "BOOL_LABEL_SPECS",
    "BatchEngine",
    "ChatTemplateError",
    "Device",
    "EngineConfig",
    "EngineError",
    "LabelResolutionError",
    "LabelSet",
    "LabelSpec",
    "get_chat_template",
    "render_chat",
    "resolve_labels",
]
