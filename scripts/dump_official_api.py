"""把官方 TypeSafe API 的真实响应 dump 到 tests/fixtures/official/.

case 为通用演示 case (三原语各一份 + 组合 + 结构化输入 + 前端预置 6 例),
外加 /v1/models 与错误形态探针; dump 一次之后测试就不再需要联网和密钥.

密钥只从环境变量 TYPESAFE_API_KEY 读, 不落盘、不入库; dump 文件里也绝不含密钥.
用法 (密钥只在这一次进程里可见):

    $env:HTTPS_PROXY='http://127.0.0.1:7890'
    $env:TYPESAFE_API_KEY='...'
    .venv\\Scripts\\python.exe scripts/dump_official_api.py

输入: 环境变量 TYPESAFE_API_KEY (缺失直接报错, 不静默跳过);
输出: tests/fixtures/official/*.json, 形如
      {"endpoint", "request", "status", "body", "notes"} —— 测试可直接复读回放;
预期: 只做只读调用 (GET /v1/models + POST /v1/systemone), 不改远端任何状态.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.typesafe.ai"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures" / "official"
MODEL = "jev-latest"
TIMEOUT = 60.0

# 前端预置 5 的 state 是 JSON.stringify(chat log, null, 2), 这里等价重建
PRESET5_STATE = json.dumps(
    [
        {"role": "user", "text": "hey, did my refund arrive yet?"},
        {"role": "agent", "text": "let me check the payment system"},
    ],
    ensure_ascii=False,
    indent=2,
)

# ---------------- 前端 6 个预置 case (web/static/app.js PRESETS 原样) ----------------

PRESETS = (
    (
        "preset1_hotdog",
        "preset1: 热狗退款 (单值判断)",
        "Customer: I ordered a hot dog at noon and it arrived stone cold at 1:30 PM. "
        "I want my money back.",
        {
            "refund": {
                "type": "noul",
                "instructions": "Does the customer explicitly request a refund?",
            }
        },
    ),
    (
        "preset2_sky_color",
        "preset2: 天空颜色 (单项选择, state 为空串)",
        "",
        {
            "sky_color": {
                "type": "choice",
                "instructions": "What color does the sky appear?",
                "criteria": {
                    "blue": "Rayleigh scattering of short wavelengths",
                    "green": "filtered by clouds",
                    "red": "only near sunset",
                },
            }
        },
    ),
    (
        "preset3_monkey",
        "preset3: 猴子作画 (等级评分)",
        "An unattended monkey smacked paint onto this canvas. "
        "Critics are debating its artistic merit.",
        {
            "quality": {
                "type": "score",
                "instructions": "Rate the artistic quality of the painting.",
                "criteria": ["amateur", "decent", "good", "great", "masterpiece"],
            }
        },
    ),
    (
        "preset4_ticket",
        "preset4: 工单三连问 (复合决策)",
        "Ticket #4021: My payments have been failing for 3 days, I was charged twice, "
        "and I have been on hold for an hour. Fix it today or I am leaving.",
        {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this message convey real urgency?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should own this ticket?",
                "criteria": {
                    "billing": "payments and invoices",
                    "tech": "technical issues",
                    "ops": "logistics",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["calm", "mildly annoyed", "frustrated",
                             "very frustrated", "furious"],
            },
        },
    ),
    (
        "preset5_structured",
        "preset5: 结构化上下文与复杂标准 (chat log + 字典 criteria)",
        PRESET5_STATE,
        {
            "intent": {
                "type": "choice",
                "instructions": "Classify the latest user intent.",
                "criteria": {
                    "refund_status": {
                        "topic": "money",
                        "detail": "asking about a pending refund",
                    },
                    "small_talk": {
                        "topic": "other",
                        "detail": "greeting or chatter",
                    },
                },
            }
        },
    ),
    (
        "preset6_chinese_ticket",
        "preset6: 中文工单三连问",
        "客户：我昨天在你们店买的蓝牙耳机只有一个耳朵有声音，客服让我等三天了还没回复，"
        "今天必须给我解决，不然我就投诉到消费者协会！",
        {
            "is_urgent": {
                "type": "noul",
                "instructions": "这条消息是否表达了真实的紧急情绪？",
            },
            "department": {
                "type": "choice",
                "instructions": "这个问题应该由哪个部门处理？",
                "criteria": {
                    "售后": "产品质量与退换",
                    "技术": "软件与网络故障",
                    "物流": "发货与配送",
                },
            },
            "anger": {
                "type": "score",
                "instructions": "客户的愤怒程度打几分？",
                "criteria": ["平静", "有点不满", "生气", "很生气", "暴怒"],
            },
        },
    ),
    (
        "advanced_invoice",
        "官方 advanced 页: 结构化 state + 结构化 instructions/criteria",
        {
            "source_text": "Invoice #4471 issued March 3, 2026 to Beaver Dam Logistics "
            "for $12,840.00, net 30."
        },
        {
            "invoice_number_is_correct": {
                "type": "noul",
                "instructions": {
                    "field": {
                        "name": "invoice_number",
                        "type": "string",
                        "description": "The identifier printed on the invoice.",
                    },
                    "extracted_value": "4471",
                    "question": "Does `extracted_value` match the `field` as it appears "
                    "in `source_text`?",
                },
            },
            "customer_name": {
                "type": "choice",
                "instructions": {
                    "field": {
                        "name": "customer_name",
                        "type": "string",
                        "description": "The organization the invoice was issued to.",
                    },
                    "question": "Which option is the value of `field` in `source_text`?",
                },
                "criteria": {
                    "Beaver Logistics": None,
                    "Dam Logistics": None,
                    "Beaver Dam Logistics": None,
                    "Beaver": None,
                    "Dam": None,
                },
            },
            "amount_due": {
                "type": "score",
                "instructions": {
                    "field": {
                        "name": "amount_due",
                        "type": "number",
                        "unit": "USD",
                        "description": "The total the invoice asks to be paid.",
                    },
                    "question": "How large is the `field` value in `source_text`?",
                },
                "criteria": [
                    "Under $1,000",
                    "$1,000 to $10,000",
                    "$10,000 to $100,000",
                    "$100,000 to $1,000,000",
                    "Over $1,000,000",
                ],
            },
            "payment_terms": {
                "type": "score",
                "instructions": {
                    "field": {
                        "name": "payment_terms",
                        "type": "integer",
                        "unit": "days",
                        "description": 'Days allowed for payment, from terms such as "net 30".',
                    },
                    "question": "How many days does the `field` in `source_text` allow "
                    "for payment?",
                },
                "criteria": [
                    "Due on receipt",
                    "Net 10",
                    "Net 30",
                    "Net 60",
                    "Net 90",
                ],
            },
        },
    ),
)


def _oversized_questions() -> dict:
    """256 个候选: 探官方 255 上限的报错形态."""
    return {
        "too_many": {
            "type": "choice",
            "instructions": "在 256 个里挑一个?",
            "criteria": {f"option_{i}": f"候选 {i}" for i in range(256)},
        }
    }


def _call(method: str, path: str, *, key: str | None,
          body: dict | None = None) -> tuple[int, dict]:
    """发一次请求, 返回 (HTTP 状态码, 响应体).

    输入: method/path/key/body; key=None 表示不带 Authorization 头;
    输出: (status, parsed_json) —— 响应不是 JSON 时包成 {"_raw": 文本};
    预期: 任何 HTTP 错误码都照实返回, 不抛异常 (dump 就是要记录错误形态).
    """
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if key is not None:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        status = error.code
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_raw": raw}
    return status, parsed


def _save(name: str, endpoint: str, request_body: dict | None, status: int,
          body: dict, notes: dict | None = None) -> None:
    """写一个 dump 文件 (含请求与响应, 不含任何密钥)."""
    payload: dict = {
        "endpoint": endpoint,
        "request": request_body,
        "status": status,
        "body": body,
    }
    if notes:
        payload["notes"] = notes
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"{status:>3}  {name}.json")


def main() -> int:
    """跑完整套 dump. 输出: 进程退出码 (缺密钥 -> 2)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("缺少环境变量 TYPESAFE_API_KEY, 拒绝空跑", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    status, body = _call("GET", "/v1/models", key=key)
    _save("models", "GET /v1/models", None, status, body)

    for name, label, state, questions in PRESETS:
        request_body = {"state": state, "model": MODEL, "questions": questions}
        status, body = _call("POST", "/v1/systemone", key=key, body=request_body)
        _save(f"systemone_{name}", "POST /v1/systemone", request_body, status, body,
              {"source": label})

    request_body = {"state": "x", "model": MODEL, "questions": _oversized_questions()}
    status, body = _call("POST", "/v1/systemone", key=key, body=request_body)
    _save("systemone_too_many_options", "POST /v1/systemone", request_body, status, body,
          {"probe": "官方 criteria 上限 255 的报错形态"})

    # 错误形态: 错 key / 缺 key / 非法请求体 / 未知模型
    status, body = _call("GET", "/v1/models", key="definitely-not-a-valid-key")
    _save("err_bad_key", "GET /v1/models", None, status, body)

    status, body = _call("GET", "/v1/models", key=None)
    _save("err_missing_key", "GET /v1/models", None, status, body)

    bad_body = {"state": "x", "questions": {}}
    status, body = _call("POST", "/v1/systemone", key=key, body=bad_body)
    _save("err_invalid_body", "POST /v1/systemone", bad_body, status, body)

    unknown = {"state": "x", "model": "no-such-model-xyz",
               "questions": {"q": {"type": "noul", "instructions": "Is this fine?"}}}
    status, body = _call("POST", "/v1/systemone", key=key, body=unknown)
    _save("err_unknown_model", "POST /v1/systemone", unknown, status, body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
