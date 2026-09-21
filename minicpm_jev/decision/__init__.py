"""三原语决策: 问题类型 / 答案类型 / 置信度 / 打分编排.

- primitives: 官方契约的 Question 与 Answer 类型 (含校验);
- confidence: 归一化峰值口径;
- scoring: 把 (state, question) 变成答案 —— 渲染 prompt、分批打分、跨块对齐。
"""

from .confidence import normalized_peak
from .primitives import (
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    QuestionError,
    Score,
    ScoreAnswer,
    parse_question,
)
from .scoring import (
    Usage,
    answer,
    answer_all,
    answer_choice,
    answer_noul,
    answer_score,
    weighted_level_score,
)

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
    "Usage",
    "answer",
    "answer_all",
    "answer_choice",
    "answer_noul",
    "answer_score",
    "normalized_peak",
    "parse_question",
    "weighted_level_score",
]
