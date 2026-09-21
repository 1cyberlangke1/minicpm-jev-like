"""概率加权分: 公式性质与单调性 (不加载模型)."""

import pytest

from minicpm_jev.decision import weighted_level_score










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




def test_empty_input_is_rejected() -> None:
    """没有档位就没有分数."""
    with pytest.raises(ValueError):
        weighted_level_score([])
