"""HTTP 服务层: 纯 JEV 契约 (POST /v1/systemone + GET /v1/models).

- app: FastAPI 工厂 (鉴权 / 队列 / 单模型切换 / 错误体);
- errors: 带 HTTP 码与官方 error_type 的错误族;
- auth: Bearer 鉴权 (默认关闭);
- limiter: 在飞请求数上限;
- registry: 本地模型清单与别名解析;
- manager: 单模型驻留的引擎管理;
- schemas: pydantic 请求 / 响应模型。
"""

from .app import DEFAULT_WEIGHTS_DIR, create_app
from .auth import AuthGuard
from .errors import (
    AuthenticationError,
    MissingKeyError,
    OverloadedError,
    RateLimitedError,
    ServiceError,
    UnprocessableError,
    UsageError,
)
from .limiter import RequestLimiter
from .manager import EngineFactory, EngineLike, EngineManager
from .registry import ModelInfo, ModelRegistry
from .schemas import (
    ModelInfoModel,
    ModelsResponse,
    SystemOneRequest,
    SystemOneResponse,
    UsageModel,
)

__all__ = [
    "DEFAULT_WEIGHTS_DIR",
    "AuthenticationError",
    "AuthGuard",
    "EngineFactory",
    "EngineLike",
    "EngineManager",
    "MissingKeyError",
    "ModelInfo",
    "ModelInfoModel",
    "ModelRegistry",
    "ModelsResponse",
    "OverloadedError",
    "RateLimitedError",
    "RequestLimiter",
    "ServiceError",
    "SystemOneRequest",
    "SystemOneResponse",
    "UnprocessableError",
    "UsageError",
    "UsageModel",
    "create_app",
]
