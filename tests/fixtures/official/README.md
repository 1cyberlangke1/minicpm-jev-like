# 官方 TypeSafe API 夹具

这里存的是官方 `https://api.typesafe.ai` 的**真实响应**, 供测试离线回放:
跑测试不需要联网, 也不需要任何密钥。

## 文件

| 文件 | 来源 |
| --- | --- |
| `models.json` | `GET /v1/models` |
| `systemone_preset1_hotdog.json` | 演示 case 1 (热狗退款, noul) |
| `systemone_preset2_sky_color.json` | 演示 case 2 (天空颜色, choice, state 为空串) |
| `systemone_preset3_monkey.json` | 演示 case 3 (猴子作画, score 5 档) |
| `systemone_preset4_ticket.json` | 演示 case 4 (工单三连问, 三原语混合) |
| `systemone_preset5_structured.json` | 演示 case 5 (chat log state + 字典 criteria) |
| `systemone_preset6_chinese_ticket.json` | 演示 case 6 (中文工单三连问) |
| `systemone_advanced_invoice.json` | 官方 advanced 页的发票案例 (结构化 state + 结构化 instructions) |
| `systemone_too_many_options.json` | 256 个候选 -> 400 的报错形态 |
| `err_bad_key.json` / `err_missing_key.json` | 401 / 403 鉴权错误 |
| `err_invalid_body.json` / `err_unknown_model.json` | 422 校验失败 / 400 未知模型 |

每个文件的形状是 `{"endpoint", "request", "status", "body", "notes"}`:
`request` 原样保留请求体 (便于回放), `status` 是 HTTP 状态码, `body` 是响应体。

## 重新生成

```powershell
$env:HTTPS_PROXY='http://127.0.0.1:7890'
$env:TYPESAFE_API_KEY='<临时传入, 不要写进任何文件>'
.venv\Scripts\python.exe scripts\dump_official_api.py
```

密钥只从环境变量读, 不落盘、不入库; 夹具文件里也不含密钥。
没有密钥时脚本直接报错退出, 不会静默跳过。

## 用途

- 回放官方行为, 校验本服务的响应形状与官方逐字段一致 (字段名、层级、错误体);
- 拿同一批 case 对比本地引擎与官方的分布差异;
- 官方改版时重新 dump, diff 出契约变化。
