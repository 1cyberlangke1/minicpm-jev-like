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


