"""三原语打分: 真模型端到端 (含分块候选)."""

import pytest

from minicpm_jev.decision import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    answer,
    answer_choice,
    answer_noul,
    answer_score,
)


def test_noul_true_question_is_confident_yes(engine) -> None:
    """常识为真的问题: noul 概率应明显偏向 yes."""
    result = answer_noul(engine, "General common sense.", Noul(
        instructions="Is the sky blue on a clear day?"
    ))
    assert isinstance(result, NoulAnswer)
    assert 0.0 <= result.noul <= 1.0
    assert result.noul > 0.5


def test_noul_false_question_is_confident_no(engine) -> None:
    """常识为假的问题: noul 概率应明显偏向 no."""
    result = answer_noul(engine, "General common sense.", Noul(
        instructions="Is fire cold to the touch?"
    ))
    assert result.noul < 0.5


def test_choice_single_chunk_picks_expected_option(engine) -> None:
    """单块 (3 个候选): 概率和为 1, 命中语义正确项, 置信度在 [0, 1]."""
    question = Choice(
        instructions="On a clear day, what color does the sky appear?",
        criteria={
            "blue": "the clear daytime sky",
            "green": "grass and leaves",
            "red": "fresh blood",
        },
    )
    result = answer_choice(engine, "Answer with common sense.", question)
    assert isinstance(result, ChoiceAnswer)
    assert result.choice == "blue"
    assert set(result.probabilities) == {"blue", "green", "red"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= result.confidence <= 1.0
    assert result.probabilities["blue"] == max(result.probabilities.values())


def test_choice_chunked_candidates_still_pick_expected_option(engine) -> None:
    """10 个候选按 chunk_size=4 分块 (3 块, 每块带 None): 仍命中正确项且概率和为 1."""
    criteria = {
        "blue": "the clear daytime sky",
        "green": "grass and leaves",
        "red": "fresh blood",
        "yellow": "ripe bananas",
        "black": "coal",
        "white": "fresh snow",
        "orange": "a ripe orange fruit",
        "purple": "eggplant skin",
        "brown": "tree bark",
        "grey": "storm clouds",
    }
    question = Choice(
        instructions="On a clear day, what color does the sky appear?",
        criteria=criteria,
    )
    result = answer_choice(
        engine, "Answer with common sense.", question, chunk_size=4
    )
    assert set(result.probabilities) == set(criteria)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= result.confidence <= 1.0
    assert result.choice == "blue"


def test_choice_single_chunk_matches_chunked_ranking(engine) -> None:
    """同一批候选, 单块与分块的 top-1 必须一致 (分块只是手段, 不能改结论)."""
    criteria = {
        "blue": "the clear daytime sky",
        "green": "grass and leaves",
        "red": "fresh blood",
        "yellow": "ripe bananas",
        "black": "coal",
    }
    question = Choice(
        instructions="On a clear day, what color does the sky appear?",
        criteria=criteria,
    )
    single = answer_choice(engine, "Answer with common sense.", question)
    chunked = answer_choice(
        engine, "Answer with common sense.", question, chunk_size=2
    )
    assert single.choice == chunked.choice


def test_score_is_probability_weighted_average(engine) -> None:
    """score = Σ p_i · i, 且 legend / probabilities 键是档位编号字符串."""
    question = Score(
        instructions="How angry is the customer?",
        criteria=["calm", "mildly annoyed", "angry", "furious"],
    )
    state = "Customer: This is the third time my order was wrong. I am furious and want a refund."
    result = answer_score(engine, state, question)
    assert isinstance(result, ScoreAnswer)
    assert set(result.probabilities) == {"0", "1", "2", "3"}
    assert set(result.legend) == {"0", "1", "2", "3"}
    assert result.legend["3"] == "furious"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    expected = sum(
        index * result.probabilities[str(index)] for index in range(4)
    )
    assert result.score == pytest.approx(expected)
    assert 0.0 <= result.score <= 3.0
    assert 0.0 <= result.confidence <= 1.0


def test_score_leans_angry_for_angry_state(engine) -> None:
    """明显愤怒的工单不该落在最平静那档."""
    question = Score(
        instructions="How angry is the customer?",
        criteria=["calm", "mildly annoyed", "angry", "furious"],
    )
    state = "Customer: I am furious! Third failed delivery and nobody answers my emails."
    result = answer_score(engine, state, question)
    assert result.score > 1.0


def test_answer_dispatches_by_question_type(engine) -> None:
    """answer() 按题型分发, 返回对应答案类型."""
    state = "General common sense."
    noul = answer(engine, state, Noul(instructions="Is water wet?"))
    choice = answer(
        engine,
        state,
        Choice(instructions="Pick one", criteria={"a": "first", "b": "second"}),
    )
    score = answer(
        engine,
        state,
        Score(instructions="Rate", criteria=["low", "high"]),
    )
    assert isinstance(noul, NoulAnswer)
    assert isinstance(choice, ChoiceAnswer)
    assert isinstance(score, ScoreAnswer)


def test_answer_rejects_unknown_type(engine) -> None:
    """非三原语对象直接 TypeError, 不静默返回空答案."""
    with pytest.raises(TypeError):
        answer(engine, "x", "not a question")  # type: ignore[arg-type]
