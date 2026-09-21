"""EntryType 序列化: str 原样 / null 空串 / object 与 list 走 JSON."""

import pytest

from minicpm_jev.template import ENTRY_TYPES, EntryTypeError, is_entry, render_entry


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


def test_chinese_is_not_escaped() -> None:
    """ensure_ascii=False: 中文原样留在 JSON 里, 不变成 \\uXXXX."""
    assert render_entry({"售后": "产品质量与退换"}) == (
        '{\n'
        '  "售后": "产品质量与退换"\n'
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


def test_nested_structure_is_rendered_recursively() -> None:
    """嵌套 object/list 一路保留结构, 不被压平成字符串."""
    value = {
        "field": {"name": "amount_due", "type": "number", "unit": "USD"},
        "candidates": [{"k": "a"}, {"k": "b"}],
    }
    rendered = render_entry(value)
    assert '"name": "amount_due"' in rendered
    assert '"unit": "USD"' in rendered
    assert rendered.count('"k":') == 2
    assert rendered.startswith("{\n")
    assert rendered.endswith("\n}")


def test_empty_object_and_array() -> None:
    """空容器给出合法 JSON 文本, 不是空串."""
    assert render_entry({}) == "{}"
    assert render_entry([]) == "[]"


def test_null_inside_container_is_kept_as_json_null() -> None:
    """容器里的 null 是 JSON null (区别于顶层 null -> 空串)."""
    assert render_entry({"Beaver": None}) == '{\n  "Beaver": null\n}'
    assert render_entry([None, "x"]) == '[\n  null,\n  "x"\n]'


def test_non_string_object_key_is_rejected() -> None:
    """JSON object 的键必须是字符串, 整数键直接报错 (英文消息)."""
    with pytest.raises(EntryTypeError) as error:
        render_entry({1: "one"})
    message = str(error.value)
    assert "key must be a string" in message
    assert "int" in message
    assert "entry" in message


def test_nested_non_string_key_reports_path() -> None:
    """嵌套里的非法键要能定位到具体路径."""
    with pytest.raises(EntryTypeError) as error:
        render_entry({"outer": [{"ok": 1}, {2: "bad"}]})
    assert "entry.outer[1]" in str(error.value)


@pytest.mark.parametrize("value", [1, 1.5, True, b"bytes", object()])
def test_other_types_are_rejected(value: object) -> None:
    """int/float/bool/bytes/对象都不是 EntryType, 一律报错不兜底."""
    with pytest.raises(EntryTypeError) as error:
        render_entry(value)
    assert "must be str, object, list or null" in str(error.value)


def test_is_entry_truth_table() -> None:
    """is_entry 只认真四种形状."""
    assert is_entry("x") and is_entry({"a": 1}) and is_entry([1]) and is_entry(None)
    assert not is_entry(1) and not is_entry(1.0) and not is_entry(True)
    assert not is_entry(b"x") and not is_entry(object())
    assert ENTRY_TYPES == (str, dict, list, type(None))


def test_render_is_deterministic() -> None:
    """同一个值渲染两次必须逐字符相同 (可做缓存键)."""
    value = {"a": [1, 2, {"b": "中文"}], "c": None}
    assert render_entry(value) == render_entry(value)
