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






def test_resolve_labels_maps_single_token_variants() -> None:
    """变体 -> 单 token: 同组变体归并, 组序与 specs 一致."""
    table = {" yes": 11, " Yes": 12, " no": 21}
    specs = (LabelSpec("true", (" yes", " Yes")), LabelSpec("false", (" no",)))
    label_set = resolve_labels(lambda text: [table[text]], specs)
    assert label_set.names == ("true", "false")
    assert label_set.token_ids == ((11, 12), (21,))




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
