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


def test_noul_rejects_non_entry_instructions() -> None:
    """数字不是 EntryType."""
    with pytest.raises(QuestionError) as error:
        Noul(instructions=7)
    assert "noul.instructions" in str(error.value)
    assert "int" in str(error.value)


def test_noul_criteria_accepts_true_false() -> None:
    """criteria 是 true/false 两侧的释义, 值也是 EntryType."""
    question = Noul(
        instructions="Does it ask for credentials?",
        criteria={"true": {"what": "asks for a password"}, "false": None},
    )
    assert set(question.criteria or {}) == {"true", "false"}


def test_noul_criteria_rejects_unknown_key() -> None:
    """只认 true / false 两个键, 别的键报错."""
    with pytest.raises(QuestionError) as error:
        Noul(instructions="x", criteria={"maybe": "??"})
    assert "maybe" in str(error.value)


def test_noul_criteria_rejects_non_mapping() -> None:
    """criteria 不是对象时报错."""
    with pytest.raises(QuestionError):
        Noul(instructions="x", criteria=["true"])


def test_choice_requires_non_empty_mapping() -> None:
    """选项表必须是非空对象."""
    with pytest.raises(QuestionError):
        Choice(instructions="pick", criteria={})
    with pytest.raises(QuestionError):
        Choice(instructions="pick", criteria=["a", "b"])


def test_choice_rejects_empty_option_name() -> None:
    """选项名不能是空串 (官方用它回填 choice 字段)."""
    with pytest.raises(QuestionError) as error:
        Choice(instructions="pick", criteria={"": "no name"})
    assert "non-empty" in str(error.value)


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


def test_choice_allows_many_options() -> None:
    """本地不设 255 上限 (靠分块支持无限候选), 只要求名字非空."""
    criteria = {f"option_{index}": None for index in range(300)}
    assert len(Choice(instructions="pick", criteria=criteria).criteria) == 300


def test_score_accepts_two_to_ten_levels() -> None:
    """官方区间闭区间 [2, 10]."""
    assert len(Score(instructions="rate", criteria=["a", "b"]).criteria) == 2
    levels = [f"level {index}" for index in range(MAX_SCORE_LEVELS)]
    assert len(Score(instructions="rate", criteria=levels).criteria) == MAX_SCORE_LEVELS


@pytest.mark.parametrize("levels", [["only"], list(range(11))])
def test_score_rejects_out_of_range_level_count(levels: list) -> None:
    """档位数不在 2~10 内直接报错, 消息里带上下界."""
    with pytest.raises(QuestionError) as error:
        Score(instructions="rate", criteria=levels)
    message = str(error.value)
    assert str(MIN_SCORE_LEVELS) in message
    assert str(MAX_SCORE_LEVELS) in message


def test_score_rejects_non_list_criteria() -> None:
    """score 的 criteria 是有序数组, 对象不行 (顺序即档位值)."""
    with pytest.raises(QuestionError):
        Score(instructions="rate", criteria={"0": "a", "1": "b"})


def test_parse_question_dispatches_by_type() -> None:
    """三种 type 各自解析成对应类型."""
    noul = parse_question({"type": "noul", "instructions": "?"})
    choice = parse_question({"type": "choice", "instructions": "?", "criteria": {"a": None}})
    score = parse_question({"type": "score", "instructions": "?", "criteria": ["a", "b"]})
    assert isinstance(noul, Noul)
    assert isinstance(choice, Choice)
    assert isinstance(score, Score)


def test_parse_question_rejects_unknown_type() -> None:
    """未知 type 不猜、不默认."""
    with pytest.raises(QuestionError) as error:
        parse_question({"type": "sort", "instructions": "?"})
    assert "sort" in str(error.value)


def test_parse_question_requires_instructions() -> None:
    """instructions 必填."""
    with pytest.raises(QuestionError) as error:
        parse_question({"type": "noul"})
    assert "instructions" in str(error.value)


def test_parse_question_requires_criteria_for_choice_and_score() -> None:
    """choice / score 的 criteria 必填."""
    with pytest.raises(QuestionError):
        parse_question({"type": "choice", "instructions": "?"})
    with pytest.raises(QuestionError):
        parse_question({"type": "score", "instructions": "?"})


def test_parse_question_rejects_non_mapping() -> None:
    """题目必须是对象."""
    with pytest.raises(QuestionError):
        parse_question(["noul"])  # type: ignore[arg-type]


def test_answer_dicts_match_official_shape() -> None:
    """答案序列化形状: noul 不带 confidence, choice / score 带."""
    noul = NoulAnswer(noul=0.91).to_dict()
    assert noul == {"type": "noul", "noul": 0.91}
    assert "confidence" not in noul

    choice = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 1.0, "tech": 0.0},
        confidence=1.0,
    ).to_dict()
    assert choice["type"] == "choice"
    assert choice["choice"] == "billing"
    assert choice["probabilities"] == {"billing": 1.0, "tech": 0.0}
    assert choice["confidence"] == 1.0

    score = ScoreAnswer(
        score=1.4,
        legend={"0": "calm", "1": "frustrated", "2": "furious"},
        probabilities={"0": 0.1, "1": 0.4, "2": 0.5},
        confidence=0.4,
    ).to_dict()
    assert score["type"] == "score"
    assert score["score"] == 1.4
    assert score["legend"]["2"] == "furious"
    assert score["probabilities"]["1"] == 0.4
    assert score["confidence"] == 0.4
