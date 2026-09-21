"""三原语类型与校验: 官方契约的边界 (不加载模型)."""

import pytest

from minicpm_jev.decision import (
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    QuestionError,
    Score,
    ScoreAnswer,
    parse_question,
)


def test_noul_accepts_all_entry_shapes() -> None:
    """instructions 四种形状都合法: str / object / list / null."""
    assert Noul(instructions="Is the sky blue?").instructions == "Is the sky blue?"
    assert Noul(instructions={"question": "Is it blue?"}).criteria is None
    assert Noul(instructions=["a", "b"]).criteria is None
    assert Noul(instructions=None).criteria is None






def test_noul_criteria_rejects_unknown_key() -> None:
    """只认 true / false 两个键, 别的键报错."""
    with pytest.raises(QuestionError) as error:
        Noul(instructions="x", criteria={"maybe": "??"})
    assert "maybe" in str(error.value)




def test_choice_requires_non_empty_mapping() -> None:
    """选项表必须是非空对象."""
    with pytest.raises(QuestionError):
        Choice(instructions="pick", criteria={})
    with pytest.raises(QuestionError):
        Choice(instructions="pick", criteria=["a", "b"])




def test_choice_accepts_null_and_structured_descriptions() -> None:
    """选项描述可以是 null / 对象 / 数组 —— 官方 EntryType."""
    question = Choice(
        instructions="pick",
        criteria={
            "a": None,
            "b": {"what": "b 覆盖的场景", "not_for": "b 不覆盖的场景"},
            "c": ["c1", "c2"],
        },
    )
    assert list(question.criteria) == ["a", "b", "c"]




def test_score_accepts_two_to_ten_levels() -> None:
    """官方区间闭区间 [2, 10]."""
    assert len(Score(instructions="rate", criteria=["a", "b"]).criteria) == 2
    levels = [f"level {index}" for index in range(MAX_SCORE_LEVELS)]
    assert len(Score(instructions="rate", criteria=levels).criteria) == MAX_SCORE_LEVELS








def test_parse_question_rejects_unknown_type() -> None:
    """未知 type 不猜、不默认."""
    with pytest.raises(QuestionError) as error:
        parse_question({"type": "sort", "instructions": "?"})
    assert "sort" in str(error.value)




def test_parse_question_requires_criteria_for_choice_and_score() -> None:
    """choice / score 的 criteria 必填."""
    with pytest.raises(QuestionError):
        parse_question({"type": "choice", "instructions": "?"})
    with pytest.raises(QuestionError):
        parse_question({"type": "score", "instructions": "?"})




