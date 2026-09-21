"""HTTP 请求 / 响应模型 (pydantic v2, 逐字段对齐官方契约).

请求三字段全必填: ``state`` (str / object / array)、``model`` (非空字符串)、
``questions`` (非空映射)。校验失败由 FastAPI 原生处理器转成 422 + 字段错误数组,
与官方真机的形状一致。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "ModelInfoModel",
    "ModelsResponse",
    "SystemOneRequest",
    "SystemOneResponse",
    "UsageModel",
]


class SystemOneRequest(BaseModel):
    """POST /v1/systemone 请求体."""

    state: Any
    model: str = Field(min_length=1)
    questions: dict[str, dict[str, Any]] = Field(min_length=1)

    @field_validator("state")
    @classmethod
    def _state_shape(cls, value: Any) -> Any:
        """state 只能是字符串 / 对象 / 数组 (官方契约), 其余直接 422."""
        if not isinstance(value, (str, dict, list)):
            raise ValueError("state must be a string, object or array")
        return value


class ModelInfoModel(BaseModel):
    """GET /v1/models 的单个条目."""

    name: str
    description: str
    release_date: str


class ModelsResponse(BaseModel):
    """GET /v1/models 响应."""

    models: list[ModelInfoModel]


class UsageModel(BaseModel):
    """usage 字段."""

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    """POST /v1/systemone 响应: model 回真实模型名, answers 按题名索引."""

    model: str
    answers: dict[str, Any]
    usage: UsageModel
