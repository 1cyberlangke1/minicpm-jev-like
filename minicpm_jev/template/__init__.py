"""对话模板与结构化输入.

两块:
- chat_template: 走 GGUF 官方对话模板渲染 (绝不手搓 ChatML / 裸文本);
- entry: 官方 EntryType (str / object / list / null) 的序列化.
"""

from .chat_template import ChatTemplateError, get_chat_template, render_chat
from .entry import ENTRY_TYPES, EntryTypeError, is_entry, render_entry

__all__ = [
    "ENTRY_TYPES",
    "ChatTemplateError",
    "EntryTypeError",
    "get_chat_template",
    "is_entry",
    "render_chat",
    "render_entry",
]
