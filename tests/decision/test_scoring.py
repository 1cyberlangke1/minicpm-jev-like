"""三原语打分: 真模型端到端 (含分块候选)."""

import pytest

from minicpm_jev.chunking import plan_chunks
from minicpm_jev.decision import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Usage,
    answer,
    answer_all,
    answer_choice,
    answer_noul,
    answer_score,
)
from minicpm_jev.decision.scoring import _choice_prompt, _score_prompt


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


def test_noul_water_is_wet(engine) -> None:
    """水是湿的: 单发路径下 noul 概率偏向 yes (只测单发, 不跨批比差)."""
    result = answer_noul(
        engine, "General common sense.", Noul(instructions="Is water wet?")
    )
    assert isinstance(result, NoulAnswer)
    assert result.noul > 0.5


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


def test_choice_without_descriptions_still_decides(engine) -> None:
    """官方允许 criteria 的值为 null (选项只有名字): 没描述也要能选出正确项."""
    question = Choice(
        instructions="Which animal says meow?",
        criteria={"cat": None, "dog": None, "cow": None},
    )
    result = answer_choice(engine, "Answer with common sense.", question)
    assert result.choice == "cat"
    assert set(result.probabilities) == {"cat", "dog", "cow"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_prompt_omits_empty_sections() -> None:
    """null 字段渲染成空串时整段省略: 不留空标题, 也不留尾随空格."""
    chunk = plan_chunks(2, 128).chunks[0]
    prompt = _choice_prompt(
        Choice(instructions=None, criteria={"a": None, "b": None}), chunk
    )
    assert "Question:" not in prompt
    assert "[0] a" in prompt
    assert "[1] b" in prompt
    assert all(line == line.rstrip() for line in prompt.splitlines())

    score_prompt = _score_prompt(
        Score(instructions="Rate it.", criteria=["low", None])
    )
    assert "[1]" in score_prompt
    assert "[1] " not in score_prompt


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


def _mixed_questions() -> dict:
    """三原语各一道, 用来验批处理路径."""
    return {
        "urgent": Noul(instructions="Is this message urgent?"),
        "team": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "payments and invoices", "tech": "technical issues"},
        ),
        "anger": Score(
            instructions="How angry is the customer?",
            criteria=["calm", "mildly annoyed", "angry"],
        ),
    }


def test_answer_all_calls_engine_once(engine, monkeypatch) -> None:
    """一次请求里的多个问题只准调一次 engine.score (同批并行 prefill)."""
    calls: list[int] = []
    original = engine.score

    def spy(sequences, labels):
        calls.append(len(sequences))
        return original(sequences, labels)

    monkeypatch.setattr(engine, "score", spy)
    answers = answer_all(
        engine,
        "Ticket: my order is late again and nobody answers my emails.",
        _mixed_questions(),
    )
    assert set(answers) == {"urgent", "team", "anger"}
    assert len(calls) == 1, "多问题必须合成一批, 不能逐题单发"
    # noul 1 条 + choice 1 条 (2 个候选 = 单块) + score 1 条
    assert calls[0] == 3


def test_every_request_field_reaches_the_model(engine, monkeypatch) -> None:
    """回归: state / instructions / 选项名与描述 / 档位 / 判定标准都要真的进 prompt.

    之前 choice 整段漏掉 instructions (问句没进上下文), 只断言「选出正确项」是发现
    不了的, 所以这里把真正喂给模型的 token 序列解回文本, 逐字段核对。
    """
    seen: list[str] = []
    original = engine.score

    def spy(sequences, labels):
        seen.extend(
            engine._model.detokenize(list(sequence)).decode("utf-8", "replace")
            for sequence in sequences
        )
        return original(sequences, labels)

    monkeypatch.setattr(engine, "score", spy)
    state = "STATE_MARKER_a1"

    answer_noul(
        engine,
        state,
        Noul(
            instructions="NOUL_QUESTION_b2",
            criteria={"true": "TRUE_HINT_c3", "false": "FALSE_HINT_d4"},
        ),
    )
    for marker in (state, "NOUL_QUESTION_b2", "TRUE_HINT_c3", "FALSE_HINT_d4"):
        assert marker in seen[-1], f"noul 的 {marker} 没进 prompt"

    seen.clear()
    answer_choice(
        engine,
        state,
        Choice(
            instructions="CHOICE_QUESTION_e5",
            criteria={"option_one": "DESC_ONE_f6", "option_two": None},
        ),
    )
    for marker in (
        state,
        "CHOICE_QUESTION_e5",
        "option_one",
        "DESC_ONE_f6",
        "option_two",
    ):
        assert marker in seen[-1], f"choice 的 {marker} 没进 prompt"

    seen.clear()
    answer_choice(
        engine,
        state,
        Choice(
            instructions="CHUNKED_QUESTION_j0",
            criteria={
                "first_option": "FIRST_DESC_k1",
                "second_option": "SECOND_DESC_l2",
            },
        ),
        chunk_size=1,
    )
    joined = "\n".join(seen)
    for marker in (
        "first_option",
        "FIRST_DESC_k1",
        "second_option",
        "SECOND_DESC_l2",
        "[None]",
    ):
        assert marker in joined, f"分块时 {marker} 没进 prompt"

    seen.clear()
    answer_score(
        engine,
        state,
        Score(
            instructions="SCORE_QUESTION_g7",
            criteria=["LEVEL_ONE_h8", "LEVEL_TWO_i9"],
        ),
    )
    for marker in (state, "SCORE_QUESTION_g7", "LEVEL_ONE_h8", "LEVEL_TWO_i9"):
        assert marker in seen[-1], f"score 的 {marker} 没进 prompt"


def test_answer_all_usage_counts_every_sequence(engine) -> None:
    """usage 按官方语义统计「去重前全部序列的 token 数」."""
    state = "General common sense."
    questions = _mixed_questions()

    batch_usage = Usage()
    answer_all(engine, state, questions, usage=batch_usage)

    total = 0
    for question in questions.values():
        single_usage = Usage()
        answer(engine, state, question, usage=single_usage)
        total += single_usage.input_tokens

    assert batch_usage.input_tokens == total
    assert batch_usage.output_tokens == 1


def test_answer_all_rejects_empty_questions(engine) -> None:
    """没有题目就没有答案, 直接报错."""
    with pytest.raises(ValueError):
        answer_all(engine, "x", {})
