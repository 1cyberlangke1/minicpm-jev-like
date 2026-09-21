"""HTTP 端到端: 形状 / 校验 / 鉴权 / 队列 / 真引擎决策.

拆成两个应用: 不带模型的 (只验形状与错误码, 不加载权重) 和带模型的
(真跑 noul / choice / score)。这样大部分用例是秒级的, 只有决策用例付加载成本。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minicpm_jev.config import Settings
from minicpm_jev.runtime import BatchEngine
from minicpm_jev.service import ModelRegistry, create_app

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"
MODEL_NAME = MODEL_PATH.stem

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="需要本地 GGUF 权重"
)

NOUL_BODY = {
    "state": "General common sense.",
    "model": MODEL_NAME,
    "questions": {
        "sky": {"type": "noul", "instructions": "Is the sky blue on a clear day?"}
    },
}


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    """扫真实 weights 目录的清单 (不加载模型)."""
    return ModelRegistry(MODEL_PATH.parent)


@pytest.fixture(scope="module")
def plain_client(registry: ModelRegistry) -> Iterator[TestClient]:
    """不带模型的应用: 只测形状 / 校验 / 鉴权, 不触发引擎加载."""
    app = create_app(Settings(n_ctx=1024), registry=registry)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def decision_client(registry: ModelRegistry) -> Iterator[TestClient]:
    """带模型的应用: 启动时预加载一次, 全模块共用同一份引擎."""
    app = create_app(
        Settings(model_path=MODEL_PATH, n_ctx=2048, chunk_size=8),
        registry=registry,
    )
    with TestClient(app) as client:
        yield client


def test_models_lists_local_gguf(plain_client: TestClient) -> None:
    """GET /v1/models 逐条给 name / description / release_date."""
    response = plain_client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    names = [entry["name"] for entry in body["models"]]
    assert MODEL_NAME in names
    # 清单按文件名排序 (Windows 上 Path 比较忽略大小写), 只断言确定性
    assert names == sorted(names, key=str.lower)
    assert len(names) == len(set(names))
    for entry in body["models"]:
        assert set(entry) == {"name", "description", "release_date"}
        assert entry["description"]


def test_unknown_model_is_400(plain_client: TestClient) -> None:
    """未知模型 -> 400 api_usage_error (照官方), 消息里带名字."""
    body = {
        "state": "x",
        "model": "no-such-model",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    response = plain_client.post("/v1/systemone", json=body)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error_type"] == "api_usage_error"
    assert "no-such-model" in detail["message"]


def test_missing_field_is_422_array(plain_client: TestClient) -> None:
    """缺 model -> 422, 且 detail 是字段错误数组 (与官方同形)."""
    response = plain_client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any("model" in entry["loc"] for entry in detail)


def test_bad_state_shape_is_422(plain_client: TestClient) -> None:
    """state 必须是字符串 / 对象 / 数组."""
    response = plain_client.post(
        "/v1/systemone",
        json={
            "state": 42,
            "model": MODEL_NAME,
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        },
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_empty_questions_is_422(plain_client: TestClient) -> None:
    """questions 不能为空."""
    response = plain_client.post(
        "/v1/systemone",
        json={"state": "x", "model": MODEL_NAME, "questions": {}},
    )
    assert response.status_code == 422


def test_bad_question_points_at_question_id(plain_client: TestClient) -> None:
    """题目本身不合法 -> 422, loc 指到具体题名, 消息里说明原因."""
    response = plain_client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "model": MODEL_NAME,
            "questions": {"broken": {"type": "sort", "instructions": "?"}},
        },
    )
    assert response.status_code == 422
    entry = response.json()["detail"][0]
    assert entry["loc"][:2] == ["body", "questions"]
    assert "broken" in entry["loc"]
    assert "sort" in entry["msg"]


def test_noul_end_to_end(decision_client: TestClient) -> None:
    """noul: 真模型出概率, 形状 / usage / model 字段全对."""
    response = decision_client.post("/v1/systemone", json=NOUL_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_NAME
    answer = body["answers"]["sky"]
    assert set(answer) == {"type", "noul"}
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0
    assert answer["noul"] > 0.5
    assert body["usage"]["output_tokens"] == 1
    assert body["usage"]["input_tokens"] > 0


def test_choice_and_score_end_to_end(decision_client: TestClient) -> None:
    """choice / score: 概率和为 1, 档位图例与分数都在值域内."""
    body = {
        "state": "Answer with common sense.",
        "model": MODEL_NAME,
        "questions": {
            "sky": {
                "type": "choice",
                "instructions": "On a clear day, what color does the sky appear?",
                "criteria": {
                    "blue": "the clear daytime sky",
                    "green": "grass and leaves",
                    "red": "fresh blood",
                },
            },
            "anger": {
                "type": "score",
                "instructions": "How angry is the customer?",
                "criteria": ["calm", "mildly annoyed", "angry", "furious"],
            },
        },
    }
    response = decision_client.post("/v1/systemone", json=body)
    assert response.status_code == 200
    answers = response.json()["answers"]

    choice = answers["sky"]
    assert choice["type"] == "choice"
    assert choice["choice"] == "blue"
    assert set(choice["probabilities"]) == {"blue", "green", "red"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= choice["confidence"] <= 1.0

    score = answers["anger"]
    assert score["type"] == "score"
    assert set(score["legend"]) == {"0", "1", "2", "3"}
    assert score["legend"]["3"] == "furious"
    assert 0.0 <= score["score"] <= 3.0
    assert sum(score["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_auth_codes_and_headers(registry: ModelRegistry) -> None:
    """配了 key: 没带 -> 403, 不对 -> 401, 正确 -> 200; 都带 WWW-Authenticate."""
    app = create_app(Settings(n_ctx=1024, api_key="sekret"), registry=registry)
    with TestClient(app) as client:
        missing = client.get("/v1/models")
        assert missing.status_code == 403
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        assert missing.json()["detail"]["error_type"] == "authentication_error"

        wrong = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert wrong.headers["WWW-Authenticate"] == "Bearer"

        ok = client.get("/v1/models", headers={"Authorization": "Bearer sekret"})
        assert ok.status_code == 200

        blocked = client.post("/v1/systemone", json=NOUL_BODY)
        assert blocked.status_code == 403


def test_queue_full_returns_429(registry: ModelRegistry) -> None:
    """队列上限 1: 第一个请求还在装载时, 第二个立刻 429 + Retry-After."""
    started = threading.Event()
    release = threading.Event()

    def slow_factory(config: object) -> BatchEngine:
        """先卡住 (让第一个请求占着名额), 放行后照常构造真引擎."""
        started.set()
        release.wait(timeout=30)
        return BatchEngine(config)  # type: ignore[arg-type]

    app = create_app(
        Settings(n_ctx=1024, chunk_size=8, queue_max=1),
        registry=registry,
        engine_factory=slow_factory,
    )
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/v1/systemone", json=NOUL_BODY)
            assert started.wait(timeout=30), "第一个请求没有进入装载"
            second = client.post("/v1/systemone", json=NOUL_BODY)
            assert second.status_code == 429
            assert second.headers["Retry-After"] == "1"
            assert second.json()["detail"]["error_type"] == "rate_limited_error"
            release.set()
            done = first.result(timeout=120)
            assert done.status_code == 200
            assert done.json()["answers"]["sky"]["type"] == "noul"


def test_alias_end_to_end(registry: ModelRegistry) -> None:
    """配了别名: 请求可以用别名, 响应回真实模型名."""
    aliased = ModelRegistry(MODEL_PATH.parent, {"latest": MODEL_NAME})
    app = create_app(
        Settings(model_path=MODEL_PATH, n_ctx=1024, chunk_size=8),
        registry=aliased,
    )
    with TestClient(app) as client:
        body = dict(NOUL_BODY)
        body["model"] = "latest"
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200
        assert response.json()["model"] == MODEL_NAME


def test_switching_model_returns_529(
    plain_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换模型 / 装载的过渡态: 新请求立刻 529, 不排队等权重加载."""
    from minicpm_jev.service import EngineManager

    monkeypatch.setattr(EngineManager, "is_switching", property(lambda self: True))
    response = plain_client.post("/v1/systemone", json=NOUL_BODY)
    assert response.status_code == 529
    assert response.json()["detail"]["error_type"] == "overloaded_error"
