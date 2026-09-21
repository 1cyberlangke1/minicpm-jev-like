"""概率加权分: 公式性质与单调性 (不加载模型)."""

import pytest

from minicpm_jev.decision import weighted_level_score


def test_single_level_is_zero() -> None:
    """只有一个档位时分数恒为 0 (档位编号就是档位值)."""
    assert weighted_level_score([1.0]) == 0.0


def test_one_hot_on_first_level_is_zero() -> None:
    """全概率在第一档 -> 0 分."""
    assert weighted_level_score([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_one_hot_on_last_level_is_max() -> None:
    """全概率在最后一档 -> 上界 len-1."""
    assert weighted_level_score([0.0, 0.0, 1.0]) == pytest.approx(2.0)
    assert weighted_level_score([0.0] * 4 + [1.0]) == pytest.approx(4.0)


def test_uniform_distribution_is_middle() -> None:
    """均匀分布 -> 中间值 (len-1)/2."""
    assert weighted_level_score([0.5, 0.5]) == pytest.approx(0.5)
    assert weighted_level_score([0.25] * 4) == pytest.approx(1.5)
    assert weighted_level_score([0.2] * 5) == pytest.approx(2.0)


def test_score_stays_inside_level_range() -> None:
    """任何归一分布的分数都落在 [0, len-1]."""
    distributions = (
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25, 0.25],
    )
    for probabilities in distributions:
        score = weighted_level_score(probabilities)
        assert 0.0 <= score <= len(probabilities) - 1


def test_score_is_monotone_when_mass_shifts_right() -> None:
    """把概率质量往高档位挪, 分数必须单调上升 (PLAN 验收项)."""
    levels = 4
    scores: list[float] = []
    for shift in range(levels):
        probabilities = [0.0] * levels
        probabilities[shift] = 0.6
        probabilities[min(shift + 1, levels - 1)] += 0.4
        scores.append(weighted_level_score(probabilities))
    assert scores == sorted(scores)
    assert scores[0] < scores[1] < scores[2] < scores[3]


def test_score_is_linear_in_probabilities() -> None:
    """分数对概率线性: 两个分布各取一半混合, 分数就是两者平均."""
    low = [1.0, 0.0, 0.0]
    high = [0.0, 0.0, 1.0]
    mixed = [(a + b) / 2 for a, b in zip(low, high)]
    assert weighted_level_score(mixed) == pytest.approx(
        (weighted_level_score(low) + weighted_level_score(high)) / 2
    )


def test_empty_input_is_rejected() -> None:
    """没有档位就没有分数."""
    with pytest.raises(ValueError):
        weighted_level_score([])
