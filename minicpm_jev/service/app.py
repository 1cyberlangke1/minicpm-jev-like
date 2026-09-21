"""FastAPI 应用: POST /v1/systemone + GET /v1/models.

契约照官方真机: 错误体 ``{"detail": {...}}`` (422 是字段数组), 缺 key 403 / 错 key 401,
未知模型 400 api_usage_error, 队列满 429 + Retry-After, 换模型期间 529。

并发模型: 引擎不是线程安全的, 所以 (1) 用 RequestLimiter 卡住同时在飞的请求数,
(2) 模型装载 / 切换在线程里串行执行, (3) 请求路径本身不共享可变状态。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import Settings
from ..decision import QuestionError, Usage, answer, parse_question
from .auth import AuthGuard
from .errors import (
    AuthenticationError,
    MissingKeyError,
    OverloadedError,
    RateLimitedError,
    ServiceError,
    UnprocessableError,
)
from .limiter import RequestLimiter
from .manager import EngineFactory, EngineManager
from .registry import ModelRegistry
from .schemas import ModelsResponse, SystemOneRequest, SystemOneResponse

__all__ = ["DEFAULT_WEIGHTS_DIR", "create_app"]

#: 默认权重目录 (仓库根的 weights/)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


def create_app(
    settings: Settings | None = None,
    *,
    weights_dir: Path | None = None,
    registry: ModelRegistry | None = None,
    engine_factory: EngineFactory | None = None,
) -> FastAPI:
    """构造服务应用.

    输入: settings -- 启动配置 (默认全默认值); weights_dir -- 权重目录;
          registry -- 直接给一份清单 (测试用); engine_factory -- 引擎工厂 (默认真引擎);
    输出: FastAPI 应用;
    预期: 别名非法 / 权重目录缺模型都不会在这里报错 (未知模型按请求报 400),
          只有别名表本身不合法才启动即炸。
    """
    resolved = settings or Settings()
    model_registry = registry or ModelRegistry(
        weights_dir or DEFAULT_WEIGHTS_DIR, resolved.model_aliases
    )
    guard = AuthGuard(resolved.api_key)
    limiter = RequestLimiter(resolved.queue_max)
    manager = EngineManager(resolved, model_registry, engine_factory=engine_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await manager.preload()
        try:
            yield
        finally:
            await manager.aclose()

    app = FastAPI(title="minicpm_jev", lifespan=lifespan)
    app.state.settings = resolved
    app.state.registry = model_registry
    app.state.manager = manager
    app.state.limiter = limiter

    @app.exception_handler(ServiceError)
    async def _on_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = str(exc.retry_after)
        if isinstance(exc, (AuthenticationError, MissingKeyError)):
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            status_code=exc.status_code, content=exc.detail(), headers=headers
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> ModelsResponse:
        """列本地模型 (名称 / 描述 / 发布日期)."""
        guard.check(request.headers.get("authorization"))
        return ModelsResponse(models=[info.to_dict() for info in model_registry.list()])

    @app.post("/v1/systemone")
    async def systemone(request: Request, body: SystemOneRequest) -> SystemOneResponse:
        """三原语决策 (一次请求里的多个问题共享同一个 state)."""
        guard.check(request.headers.get("authorization"))
        async with limiter.slot():
            model_name = model_registry.resolve(body.model)
            if manager.is_switching:
                raise OverloadedError(
                    "Service is busy loading a model. Please retry shortly."
                )
            engine = await manager.ensure(model_name)
            usage = Usage()
            answers: dict[str, object] = {}
            for question_id, raw in body.questions.items():
                try:
                    question = parse_question(raw)
                except QuestionError as error:
                    raise UnprocessableError(
                        str(error), loc=("body", "questions", question_id)
                    ) from error
                result = answer(
                    engine,
                    body.state,
                    question,
                    chunk_size=resolved.chunk_size,
                    usage=usage,
                )
                answers[question_id] = result.to_dict()
            return SystemOneResponse(
                model=model_name, answers=answers, usage=usage.to_dict()
            )

    return app
