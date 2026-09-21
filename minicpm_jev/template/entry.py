"""EntryType 结构化输入的序列化.

官方契约里 ``instructions`` 与 ``criteria`` 的"值"可以是四种形状 (官方叫
EntryType): ``str`` / ``object`` / ``list`` / ``null``. 引擎把它们**原样**序列化
成 prompt 文本, 不自己拼字符串 —— 手搓会丢结构, 也会让模型看到非官方的形状.

约定 (逐条对应官方 advanced 页):

- ``str``  -> 原样返回;
- ``null`` -> 空串 (choice 的 criteria 值为 null 时表示"该选项没有额外描述");
- ``object`` / ``list`` -> JSON 文本 (``ensure_ascii=False``, ``indent=2``,
  键序保持插入序, 与官方示例的排版一致).

本模块只管"值 -> 文本"; prompt 怎么组装是上层的事.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["ENTRY_TYPES", "EntryTypeError", "is_entry", "render_entry"]

#: 官方 EntryType 的四种形状
ENTRY_TYPES: tuple[type, ...] = (str, dict, list, type(None))


class EntryTypeError(TypeError):
    """值不是 EntryType (str / object / list / null).

    消息用英文: 这类错误会一路冒泡到 HTTP body, 与官方错误文案同一语言.
    """


def is_entry(value: Any) -> bool:
    """判断一个值是不是合法 EntryType.

    输入: 任意 Python 值;
    输出: True 表示 str / dict / list / None 之一;
    预期: 只做类型判断, 不递归校验 dict 的键 (那是 render_entry 的职责).
    """
    return isinstance(value, ENTRY_TYPES)


def _check_object_keys(value: Any, path: str) -> None:
    """递归校验 JSON object 的键必须是字符串.

    输入: 待校验的值 + 出错时用于定位的路径;
    输出: 无 (通过即返回);
    预期: 发现非字符串键抛 EntryTypeError, 消息里带路径, 不静默转换.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EntryTypeError(
                    f"{path}: object key must be a string, got {type(key).__name__}"
                )
            _check_object_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_object_keys(item, f"{path}[{index}]")


def render_entry(value: Any) -> str:
    """把 EntryType 渲染成 prompt 文本.

    输入: str / dict / list / None;
    输出: 文本 (None -> 空串, str 原样, dict/list -> JSON 文本);
    预期: 其它类型抛 EntryTypeError; 不 str() 兜底、不静默丢弃、不截断.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        _check_object_keys(value, "entry")
        return json.dumps(value, ensure_ascii=False, indent=2)
    raise EntryTypeError(
        f"entry must be str, object, list or null, got {type(value).__name__}"
    )
