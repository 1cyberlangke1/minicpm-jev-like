"""EntryType 序列化: str 原样 / null 空串 / object 与 list 走 JSON."""

import pytest

from minicpm_jev.template import EntryTypeError, render_entry


def test_string_passes_through_unchanged() -> None:
    """字符串原样返回, 不转义、不加工."""
    text = "Does `extracted_value` match the `field`?"
    assert render_entry(text) == text
    assert render_entry("多行\n文本\t带制表") == "多行\n文本\t带制表"
    assert render_entry("") == ""


def test_none_renders_as_empty_string() -> None:
    """null 表示"没有额外描述", 渲染成空串而不是字面量 'null'."""
    assert render_entry(None) == ""


def test_object_renders_as_pretty_json_in_insertion_order() -> None:
    """object 渲染成 indent=2 的 JSON, 键序与插入序一致."""
    value = {"question": "Which option?", "focus": "pick one"}
    assert render_entry(value) == (
        '{\n'
        '  "question": "Which option?",\n'
        '  "focus": "pick one"\n'
        '}'
    )




def test_array_renders_as_json() -> None:
    """list 渲染成 JSON 数组, 元素顺序不变."""
    assert render_entry(["Calm", "Frustrated", "Very angry"]) == (
        '[\n'
        '  "Calm",\n'
        '  "Frustrated",\n'
        '  "Very angry"\n'
        ']'
    )








def test_non_string_object_key_is_rejected() -> None:
    """JSON object 的键必须是字符串, 整数键直接报错 (英文消息)."""
    with pytest.raises(EntryTypeError) as error:
        render_entry({1: "one"})
    message = str(error.value)
    assert "key must be a string" in message
    assert "int" in message
    assert "entry" in message




@pytest.mark.parametrize("value", [1, 1.5, True, b"bytes", object()])
def test_other_types_are_rejected(value: object) -> None:
    """int/float/bool/bytes/对象都不是 EntryType, 一律报错不兜底."""
    with pytest.raises(EntryTypeError) as error:
        render_entry(value)
    assert "must be str, object, list or null" in str(error.value)




