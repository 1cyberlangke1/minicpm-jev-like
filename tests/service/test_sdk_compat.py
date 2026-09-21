"""官方 SDK 兼容性: 让 typesafe-sdk 真的打我们的本地服务.

这是最硬的兼容性证据 —— 官方客户端会把响应解析成它自己的类型化对象, 字段名 / 层级 /
类型 / 错误码只要有一处不对就会抛异常。

做法: 起一个真 uvicorn 服务 (真端口, 真 HTTP), 再让 SDK 用 base_url 指过来。
SDK 没装就整体 skip, 不让测试挂。
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from minicpm_jev.config import Settings
from minicpm_jev.service import ModelRegistry, create_app

typesafe_sdk = pytest.importorskip("typesafe_sdk")

Choice = typesafe_sdk.Choice
Noul = typesafe_sdk.Noul
Score = typesafe_sdk.Score
TypeSafeClient = typesafe_sdk.TypeSafeClient

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"
MODEL_NAME = MODEL_PATH.stem
API_KEY = "sdk-sekret"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="需要本地 GGUF 权重"
)


def _free_port() -> int:
    """找一个空闲端口 (先绑 0 再读回来)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    """起真服务 (带 key + 真模型), 等它 ready 再把地址交出去."""
    port = _free_port()
    settings = Settings(
        model_path=MODEL_PATH,
        n_ctx=2048,
        chunk_size=8,
        host="127.0.0.1",
        port=port,
        api_key=API_KEY,
    )
    app = create_app(settings, registry=ModelRegistry(MODEL_PATH.parent))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("uvicorn 没能在 60s 内启动")
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=60)


def _client(url: str, key: str = API_KEY) -> object:
    """造一个指向本地服务的官方客户端."""
    return TypeSafeClient(api_key=key, base_url=url, timeout=120.0)


def test_sdk_lists_our_models(base_url: str) -> None:
    """models.list() 能把我们的 /v1/models 解析成类型化对象."""
    response = _client(base_url).models.list()
    names = [model.name for model in response.models]
    assert MODEL_NAME in names
    entry = next(model for model in response.models if model.name == MODEL_NAME)
    assert entry.description
    assert entry.release_date


def test_sdk_parses_noul(base_url: str) -> None:
    """noul 答案能被 SDK 解析, 且概率在 [0, 1]."""
    response = _client(base_url).system_one(
        state="General common sense.",
        questions={"sky": Noul(instructions="Is the sky blue on a clear day?")},
        model=MODEL_NAME,
    )
    answer = response.answers["sky"]
    assert answer.type == "noul"
    assert 0.0 <= answer.noul <= 1.0
    assert answer.noul > 0.5
    assert response.model == MODEL_NAME
    assert response.usage.output_tokens == 1
    assert response.usage.input_tokens > 0


def test_sdk_parses_choice(base_url: str) -> None:
    """choice 答案: 选中项 / 概率表 / 置信度都能解析."""
    response = _client(base_url).system_one(
        state="Answer with common sense.",
        questions={
            "sky": Choice(
                instructions="On a clear day, what color does the sky appear?",
                criteria={
                    "blue": "the clear daytime sky",
                    "green": "grass and leaves",
                    "red": "fresh blood",
                },
            )
        },
        model=MODEL_NAME,
    )
    answer = response.answers["sky"]
    assert answer.type == "choice"
    assert answer.choice == "blue"
    assert set(answer.probabilities) == {"blue", "green", "red"}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= answer.confidence <= 1.0


def test_sdk_parses_score(base_url: str) -> None:
    """score 答案: 分数 / 图例 / 概率表都能解析."""
    response = _client(base_url).system_one(
        state="Customer: I am furious! Third failed delivery and nobody answers.",
        questions={
            "anger": Score(
                instructions="How angry is the customer?",
                criteria=["calm", "mildly annoyed", "angry", "furious"],
            )
        },
        model=MODEL_NAME,
    )
    answer = response.answers["anger"]
    assert answer.type == "score"
    # 我们按官方形状回字符串键, SDK 侧会把它转成整数键
    assert {int(key) for key in answer.legend} == {0, 1, 2, 3}
    assert 0.0 <= answer.score <= 3.0
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= answer.confidence <= 1.0


def test_sdk_maps_wrong_key_to_authentication_error(base_url: str) -> None:
    """错 key -> 401, 被 SDK 映射成它自己的鉴权异常."""
    with pytest.raises(typesafe_sdk.TypeSafeAuthenticationError):
        _client(base_url, key="wrong-key").models.list()


def test_sdk_maps_unknown_model_to_bad_request(base_url: str) -> None:
    """未知模型 -> 400, 被 SDK 映射成 BadRequest."""
    with pytest.raises(typesafe_sdk.TypeSafeBadRequestError):
        _client(base_url).system_one(
            state="x",
            questions={"q": Noul(instructions="Is this fine?")},
            model="no-such-model",
        )


def test_sdk_maps_bad_question_to_unprocessable(base_url: str) -> None:
    """题目非法 -> 422, 被 SDK 映射成 UnprocessableEntity."""
    with pytest.raises(typesafe_sdk.TypeSafeUnprocessableEntityError):
        _client(base_url).system_one(
            state="x",
            questions={"bad": Score(instructions="Rate", criteria=["only-one"])},
            model=MODEL_NAME,
        )


