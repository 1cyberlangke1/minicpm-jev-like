"""官方三原语的类型与校验.

请求侧 (Question):

- ``Noul``: ``instructions`` 必填; ``criteria`` 可选, 形状 ``{"true": ..., "false": ...}``;
- ``Choice``: ``instructions`` + ``criteria`` (选项名 -> 描述), 选项名不能为空;
- ``Score``: ``instructions`` + ``criteria`` (有序档位数组), 官方要求 2~10 档。

``instructions`` 与 ``criteria`` 的值都是 EntryType (str / object / list / null),
形状交给 template.entry 渲染。

响应侧 (Answer): ``to_dict`` 直接给出官方形状 (noul / choice / score 三种),
``noul`` 不带 ``confidence`` (官方语义如此)。

所有校验消息用英文 —— 它们会冒泡到 HTTP body。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..template.entry import EntryTypeError, is_entry

__all__ = [
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "Question",
    "QuestionError",
    "Score",
    "ScoreAnswer",
    "parse_question",
]

#: 官方 score 的档位数区间
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class QuestionError(ValueError):
    """问题不满足官方契约 (服务层映射成 422)."""


def _require_entry(value: Any, path: str) -> None:
    """校验一个值是不是 EntryType.

    输入: value -- 待校验值; path -- 出错定位 (例如 "choice.criteria['a']");
    输出: 无;
    预期: 非法类型抛 QuestionError, 消息里带路径; 不静默转字符串。
    """
    if not is_entry(value):
        raise QuestionError(
            f"{path} must be a string, object, array or null, "
            f"got {type(value).__name__}"
        )


@dataclass(frozen=True)
class Noul:
    """是/否问题: 返回回答 yes 的概率."""

    instructions: Any
    criteria: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_entry(self.instructions, "noul.instructions")
        if self.criteria is None:
            return
        if not isinstance(self.criteria, Mapping):
            raise QuestionError("noul.criteria must be an object with true/false keys")
        for key, value in self.criteria.items():
            if key not in ("true", "false"):
                raise QuestionError(
                    f"noul.criteria has unknown key {key!r}; expected 'true' or 'false'"
                )
            _require_entry(value, f"noul.criteria[{key!r}]")


@dataclass(frozen=True)
class Choice:
    """从若干选项里挑一个: criteria 是 选项名 -> 描述."""

    instructions: Any
    criteria: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_entry(self.instructions, "choice.instructions")
        if not isinstance(self.criteria, Mapping):
            raise QuestionError("choice.criteria must be an object of option -> description")
        if not self.criteria:
            raise QuestionError("choice.criteria must contain at least one option")
        for key, value in self.criteria.items():
            if not isinstance(key, str) or not key:
                raise QuestionError("choice.criteria keys must be non-empty strings")
            _require_entry(value, f"choice.criteria[{key!r}]")


@dataclass(frozen=True)
class Score:
    """按有序档位打分: criteria 是档位描述数组 (2~10 档)."""

    instructions: Any
    criteria: list[Any]

    def __post_init__(self) -> None:
        _require_entry(self.instructions, "score.instructions")
        if not isinstance(self.criteria, list):
            raise QuestionError("score.criteria must be an ordered array of level descriptions")
        if not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise QuestionError(
                f"score.criteria must have {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, "
                f"got {len(self.criteria)}"
            )
        for index, level in enumerate(self.criteria):
            _require_entry(level, f"score.criteria[{index}]")


#: 三种问题类型
Question = Noul | Choice | Score


@dataclass(frozen=True)
class NoulAnswer:
    """noul 答案: 回答 yes 的概率."""

    noul: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "noul": self.noul}


@dataclass(frozen=True)
class ChoiceAnswer:
    """choice 答案: 选中的选项 + 全量概率 + 置信度."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    """score 答案: 概率加权分 + 档位图例 + 全量概率 + 置信度."""

    score: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "confidence": self.confidence,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
        }


#: 三种答案类型
Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def parse_question(raw: Mapping[str, Any]) -> Question:
    """把请求体里的一个题目解析成类型化 Question.

    输入: raw -- {"type", "instructions", "criteria"?};
    输出: Noul / Choice / Score;
    预期: 缺 instructions / 未知 type / criteria 形状不对都抛 QuestionError (英文消息),
          绝不猜类型、不补默认值。
    """
    if not isinstance(raw, Mapping):
        raise QuestionError("question must be an object")
    if "instructions" not in raw:
        raise QuestionError("question is missing required field 'instructions'")

    instructions = raw["instructions"]
    criteria = raw.get("criteria")
    question_type = raw.get("type")
    if question_type == "noul":
        return Noul(instructions=instructions, criteria=criteria)
    if question_type == "choice":
        if criteria is None:
            raise QuestionError("choice question is missing required field 'criteria'")
        return Choice(instructions=instructions, criteria=criteria)
    if question_type == "score":
        if criteria is None:
            raise QuestionError("score question is missing required field 'criteria'")
        return Score(instructions=instructions, criteria=list(criteria) if isinstance(criteria, list) else criteria)
    raise QuestionError(
        f"unknown question type {question_type!r}; expected 'noul', 'choice' or 'score'"
    )
