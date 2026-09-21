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
    """POST /v1/systemone 请求体.

    ``think_tokens`` 是本地扩展字段 (官方契约没有): 大于 0 时先让模型自由推理这么多
    token, 再把闭合标记接回决策位。缺省 0 表示纯 prefill 决策, 与官方语义一致。
    这里不设固定上限 —— 真正的约束是上下文窗口, 由引擎按
    prompt + think + closure 是否超过 n_ctx 校验, 超了直接报错不截断;
    正常模型想完会自己吐 EOS, 只有死循环才会一直走到窗口边界。
    闭合标记本身不对外暴露, 由题型在内部派生。
    """

    state: Any
    model: str = Field(min_length=1)
    questions: dict[str, dict[str, Any]] = Field(min_length=1)
    think_tokens: int = Field(default=0, ge=0)

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
