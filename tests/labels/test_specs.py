"""标签组 / 标签集 / 受限 softmax 的单元测试 (不加载模型)."""

import numpy as np
import pytest

from minicpm_jev.labels import (
    BOOL_LABEL_SPECS,
    LabelResolutionError,
    LabelSet,
    LabelSpec,
    resolve_labels,
)


def test_label_spec_rejects_empty_name() -> None:
    """组名不能为空."""
    with pytest.raises(ValueError):
        LabelSpec("", (" yes",))


def test_label_spec_rejects_empty_variants() -> None:
    """每组至少要有一个变体."""
    with pytest.raises(ValueError):
        LabelSpec("true", ())


def test_label_set_rejects_length_mismatch() -> None:
    """组名与 token 组数量必须一致."""
    with pytest.raises(ValueError):
        LabelSet(names=("a", "b"), token_ids=((1,),))


def test_label_set_rejects_empty_group() -> None:
    """任何一组都不能没有 token."""
    with pytest.raises(ValueError):
        LabelSet(names=("a",), token_ids=((),))


def test_label_set_rejects_token_shared_by_two_groups() -> None:
    """同一个 token 落在两组里直接报错 (否则概率会双记)."""
    with pytest.raises(LabelResolutionError) as error:
        LabelSet(names=("a", "b"), token_ids=((7,), (7,)))
    assert "7" in str(error.value)


def test_all_token_ids_keep_group_order() -> None:
    """all_token_ids 按组序展开, 与 probabilities 的权重切分一致."""
    label_set = LabelSet(names=("a", "b"), token_ids=((3, 1), (9,)))
    assert label_set.all_token_ids == (3, 1, 9)


def test_probabilities_uniform_over_two_groups() -> None:
    """两个单 token 组等 logits -> 各 0.5, 且和为 1."""
    label_set = LabelSet(names=("true", "false"), token_ids=((10,), (20,)))
    logits = np.full(64, -5.0, dtype=np.float32)
    logits[10] = 0.0
    logits[20] = 0.0
    result = label_set.probabilities(logits)
    assert result == pytest.approx({"true": 0.5, "false": 0.5})
    assert sum(result.values()) == pytest.approx(1.0)


def test_probabilities_ignore_non_label_mass() -> None:
    """非标签 token 的 logits 再高也不参与归一 (受限决策的定义)."""
    label_set = LabelSet(names=("yes", "no"), token_ids=((10,), (20,)))
    logits = np.full(64, -5.0, dtype=np.float32)
    logits[10] = 0.0
    logits[20] = 0.0
    logits[63] = 500.0
    assert label_set.probabilities(logits) == pytest.approx({"yes": 0.5, "no": 0.5})


def test_probabilities_sum_variants_within_group() -> None:
    """同组多书写变体的概率质量求和后再归一."""
    label_set = LabelSet(names=("true", "false"), token_ids=((10, 11), (20,)))
    logits = np.full(64, -5.0, dtype=np.float32)
    logits[10] = 0.0
    logits[11] = 0.0
    logits[20] = 0.0
    result = label_set.probabilities(logits)
    assert result["true"] == pytest.approx(2.0 / 3.0)
    assert result["false"] == pytest.approx(1.0 / 3.0)


def test_probabilities_are_numerically_stable_for_large_logits() -> None:
    """logits 很大时不溢出 (先减最大值)."""
    label_set = LabelSet(names=("a", "b"), token_ids=((1,), (2,)))
    logits = np.zeros(8, dtype=np.float32)
    logits[1] = 10_000.0
    logits[2] = 9_000.0
    result = label_set.probabilities(logits)
    assert result["a"] > 0.999
    assert sum(result.values()) == pytest.approx(1.0)


def test_probabilities_order_follows_logits() -> None:
    """接近的 logits 给出接近的概率, 排序与 logits 一致."""
    label_set = LabelSet(names=("a", "b", "c"), token_ids=((1,), (2,), (3,)))
    logits = np.zeros(8, dtype=np.float32)
    logits[1], logits[2], logits[3] = 0.0, 0.5, 1.0
    result = label_set.probabilities(logits)
    assert result["c"] > result["b"] > result["a"]


def test_resolve_labels_maps_single_token_variants() -> None:
    """变体 -> 单 token: 同组变体归并, 组序与 specs 一致."""
    table = {" yes": 11, " Yes": 12, " no": 21}
    specs = (LabelSpec("true", (" yes", " Yes")), LabelSpec("false", (" no",)))
    label_set = resolve_labels(lambda text: [table[text]], specs)
    assert label_set.names == ("true", "false")
    assert label_set.token_ids == ((11, 12), (21,))


def test_resolve_labels_dedupes_repeated_variants() -> None:
    """同组里重复的变体只算一次 token."""
    specs = (LabelSpec("true", (" yes", " yes")),)
    label_set = resolve_labels(lambda text: [11], specs)
    assert label_set.token_ids == ((11,),)


def test_resolve_labels_rejects_multi_token_variant() -> None:
    """多 token 变体直接报错, 不截断也不取首 token."""
    specs = (LabelSpec("true", ("FALSE",)),)
    with pytest.raises(LabelResolutionError) as error:
        resolve_labels(lambda text: [59, 44794], specs)
    assert "FALSE" in str(error.value)
    assert "2" in str(error.value)


def test_bool_specs_match_documented_variant_counts() -> None:
    """bool 标签表按实测维护: 真组 12 个变体, 假组 11 个 (裸 FALSE 除外)."""
    assert [spec.name for spec in BOOL_LABEL_SPECS] == ["true", "false"]
    counts = {spec.name: len(spec.variants) for spec in BOOL_LABEL_SPECS}
    assert counts == {"true": 12, "false": 11}
    assert "FALSE" not in BOOL_LABEL_SPECS[1].variants
