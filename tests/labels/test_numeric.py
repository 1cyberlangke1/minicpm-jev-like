"""数字标签编译: 单 token 约束 / None 锚点 / 缓存."""

import numpy as np
import pytest

from minicpm_jev.labels import NONE_LABEL, LabelResolutionError, NumericLabels, label_texts


def test_label_texts_without_none() -> None:
    """不带锚点时只有 0 ~ n-1."""
    assert label_texts(1) == ("0",)
    assert label_texts(3) == ("0", "1", "2")
    assert label_texts(128)[-1] == "127"
    assert len(label_texts(128)) == 128


def test_label_texts_with_none_appends_anchor() -> None:
    """带锚点时末尾多一个 None, 且总数是 n+1."""
    assert label_texts(2, with_none=True) == ("0", "1", NONE_LABEL)
    assert len(label_texts(128, with_none=True)) == 129


@pytest.mark.parametrize("count", [0, -1])
def test_label_texts_rejects_non_positive(count: int) -> None:
    """候选数为 0 或负数直接报错, 不返回空标签集."""
    with pytest.raises(ValueError):
        label_texts(count)


def test_numeric_labels_compile_to_single_tokens() -> None:
    """每个标签文本都落到恰好一个 token 上, 组序与标签序一致."""
    fake = NumericLabels(lambda text: [100 + int(text)])
    label_set = fake.get(3)
    assert label_set.names == ("0", "1", "2")
    assert label_set.token_ids == ((100,), (101,), (102,))


def test_numeric_labels_with_none_uses_last_group() -> None:
    """None 锚点排在最后一组, 便于按索引切分."""
    fake = NumericLabels(lambda text: [7] if text == NONE_LABEL else [int(text)])
    label_set = fake.get(2, with_none=True)
    assert label_set.names == ("0", "1", NONE_LABEL)
    assert label_set.token_ids[-1] == (7,)


def test_numeric_labels_cache_returns_same_object() -> None:
    """同一个 (count, with_none) 只编译一次, 缓存命中返回同一对象."""
    calls: list[str] = []

    def tokenize(text: str) -> list[int]:
        calls.append(text)
        return [7] if text == NONE_LABEL else [int(text)]

    fake = NumericLabels(tokenize)
    first = fake.get(4)
    second = fake.get(4)
    assert first is second
    assert len(calls) == 4
    assert fake.cached_entries == 1

    fake.get(4, with_none=True)
    assert fake.cached_entries == 2


def test_numeric_labels_rejects_multi_token_label() -> None:
    """标签切不成单 token 直接报错 (不截断、不取首 token)."""
    fake = NumericLabels(lambda text: [1, 2] if text == "1" else [0])
    with pytest.raises(LabelResolutionError) as error:
        fake.get(3)
    assert "'1'" in str(error.value)
    assert fake.cached_entries == 0


def test_numeric_labels_reject_colliding_tokens() -> None:
    """两个标签撞到同一 token 时报错 (否则概率会双记)."""
    fake = NumericLabels(lambda text: [42])
    with pytest.raises(LabelResolutionError):
        fake.get(2)


def test_numeric_labels_probabilities_follow_logits() -> None:
    """在合成 logits 上, 概率排序与 logits 排序一致."""
    fake = NumericLabels(lambda text: [int(text)] if text != NONE_LABEL else [7])
    label_set = fake.get(3, with_none=True)
    logits = np.zeros(16, dtype=np.float32)
    logits[0], logits[1], logits[2], logits[7] = 1.0, 2.0, 3.0, -10.0
    result = label_set.probabilities(logits)
    assert result["2"] > result["1"] > result["0"] > result[NONE_LABEL]
    assert sum(result.values()) == pytest.approx(1.0)


def test_numeric_labels_accepts_real_tokenizer(engine) -> None:
    """真模型词表: 0~127 与 None 全部单 token (实测约束, 不能靠假设)."""
    factory = NumericLabels(lambda text: engine.tokenize(text, add_bos=False))
    label_set = factory.get(128, with_none=True)
    assert len(label_set.names) == 129
    assert len({ids[0] for ids in label_set.token_ids}) == 129
