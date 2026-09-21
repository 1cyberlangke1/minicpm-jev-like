"""置信度: 归一化峰值的性质与边界 (不加载模型)."""

import pytest

from minicpm_jev.decision import normalized_peak


def test_uniform_distribution_gives_zero() -> None:
    """均匀分布 = 完全没把握 -> 0."""
    assert normalized_peak([0.5, 0.5]) == pytest.approx(0.0)
    assert normalized_peak([0.2] * 5) == pytest.approx(0.0)
    assert normalized_peak([1 / 128] * 128) == pytest.approx(0.0, abs=1e-9)


def test_one_hot_gives_one() -> None:
    """独热 = 完全确定 -> 1."""
    assert normalized_peak([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert normalized_peak([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_single_label_is_trivially_certain() -> None:
    """只有一个标签时没有不确定性可言, 给 1 (与 K-1 分母的约定一致)."""
    assert normalized_peak([1.0]) == 1.0


def test_two_label_formula() -> None:
    """两个标签时退化成 (2p - 1)."""
    assert normalized_peak([0.8, 0.2]) == pytest.approx(0.6)
    assert normalized_peak([0.5, 0.5]) == pytest.approx(0.0)


def test_multi_label_formula() -> None:
    """K 个标签: (K·p_max - 1) / (K - 1)."""
    assert normalized_peak([0.5, 0.5, 0.0, 0.0, 0.0]) == pytest.approx(0.375)
    assert normalized_peak([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_below_uniform_is_clamped_to_zero() -> None:
    """比均匀还平的分布数值上不可能, 但浮点误差可能造出来 -> clamp 到 0."""
    assert normalized_peak([0.3, 0.3, 0.3]) == 0.0


def test_confidence_is_monotone_in_peak() -> None:
    """同一分布形状下, 峰值越大置信度越高."""
    peaks = [0.5, 0.6, 0.7, 0.9]
    scores = [normalized_peak([peak, 1 - peak]) for peak in peaks]
    assert scores == sorted(scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] == pytest.approx(0.0)
    assert scores[-1] == pytest.approx(0.8)


def test_empty_input_is_rejected() -> None:
    """没有概率就没有置信度."""
    with pytest.raises(ValueError):
        normalized_peak([])


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_out_of_range_probability_is_rejected(value: float) -> None:
    """概率必须在 [0, 1] 内, 越界直接报错不夹取."""
    with pytest.raises(ValueError):
        normalized_peak([value, 0.5])
