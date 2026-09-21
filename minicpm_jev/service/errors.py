"""服务层错误: 每条都带 HTTP 状态码与官方 error_type.

错误体形状 (照官方真机):

- 非 422: ``{"detail": {"error_type": "...", "message": "..."}}``;
- 422: ``detail`` 是字段错误数组 (与 FastAPI / pydantic 的原生形状一致)。

消息一律英文 —— 它们就是 HTTP body 的内容。
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "AuthenticationError",
    "MissingKeyError",
    "OverloadedError",
    "RateLimitedError",
    "ServiceError",
    "UnprocessableError",
    "UsageError",
]


class ServiceError(Exception):
    """服务层可预期错误的基类 (统一被 app 的错误处理器转成 HTTP)."""

    status_code = 500
    error_type = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def detail(self) -> dict[str, object]:
        """官方错误体: ``{"detail": {"error_type", "message"}}``."""
        return {"detail": {"error_type": self.error_type, "message": self.message}}


class AuthenticationError(ServiceError):
    """带了 key 但不对 -> 401."""

    status_code = 401
    error_type = "authentication_error"


class MissingKeyError(ServiceError):
    """完全没带 key -> 403."""

    status_code = 403
    error_type = "authentication_error"


class UsageError(ServiceError):
    """业务校验失败 (未知模型 / 候选超限等) -> 400 api_usage_error."""

    status_code = 400
    error_type = "api_usage_error"


class RateLimitedError(ServiceError):
    """在飞请求数超过上限 -> 429 (带 Retry-After)."""

    status_code = 429
    error_type = "rate_limited_error"

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OverloadedError(ServiceError):
    """过渡态 (装载 / 换模型) 无法受理 -> 529."""

    status_code = 529
    error_type = "overloaded_error"


class UnprocessableError(ServiceError):
    """题目级校验失败 -> 422, detail 是字段错误数组."""

    status_code = 422
    error_type = "validation_error"

    def __init__(self, message: str, loc: Sequence[str] = ("body",)) -> None:
        super().__init__(message)
        self.loc = tuple(loc)

    def entry(self) -> dict[str, object]:
        """单条字段错误 (形状与 pydantic v2 对齐, 好让官方 SDK 照样解析)."""
        return {"type": "value_error", "loc": list(self.loc), "msg": self.message}

    def detail(self) -> dict[str, object]:
        return {"detail": [self.entry()]}
