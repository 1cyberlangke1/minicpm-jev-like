"""走 HTTP 打 /v1/systemone: 官方契约的请求与响应.

输入: 无 (发票 state 写在下面); 输出: 每题的类型/答案/置信度;
预期: 服务没起会直接抛连接错误 —— 先跑 `python -m minicpm_jev.service`。
"""

from __future__ import annotations

import httpx

URL = "http://127.0.0.1:8000/v1/systemone"

BODY = {
    "state": {
        "source_text": (
            "Invoice #4471 issued March 3, 2026 to Beaver Dam Logistics "
            "for $12,840.00, net 30."
        )
    },
    "model": "MiniCPM5-2B-Q4_K_M",
    "questions": {
        "invoice_number_is_correct": {
            "type": "noul",
            "instructions": {
                "field": {
                    "name": "invoice_number",
                    "type": "string",
                    "description": "The identifier printed on the invoice.",
                },
                "extracted_value": "4471",
                "question": "Does `extracted_value` match the `field` as it appears in `source_text`?",
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
    },
}


def main() -> int:
    """发一次请求, 打印每题结果与 usage."""
    response = httpx.post(URL, json=BODY, timeout=60.0)
    response.raise_for_status()
    payload = response.json()
    for name, answer in payload["answers"].items():
        detail = {
            key: value for key, value in answer.items() if key != "type"
        }
        print(f"{name:<26} {answer['type']:<7} {detail}")
    print("usage:", payload["usage"])
    print("model:", payload["model"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
