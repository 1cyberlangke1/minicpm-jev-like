# minicpm_jev_like

用 MiniCPM5-2B (GGUF) 跑 **JEV / TypeSafe 风格的 System One 受限决策**：模型不做自由
生成，只在给定候选里出概率。三原语 `noul`（是/否）、`choice`（选一个）、
`score`（档位加权分），对外是官方的 `POST /v1/systemone` 契约，本地 CPU / GPU 双档。

![走迷宫](img/maze_walk.gif)

上面这段是 `examples/maze_walk.py` 的真实终端输出（12×12 迷宫、seed 42、18 步最优解）。
GIF 里每一步的停留时间就是实机那一步的决策耗时，所以播放速度和终端里跑一遍一样。

## 它做什么

传统的分类/抽取模型给一段文本出一个标签，这里换个做法：把「候选」直接写进题面，
让一个 2B 的对话模型在决策位只对候选标签做受限打分，取概率最高的那个。

- **不做自由生成**：不吐文本，只在决策位读一次 logits，对标签组做受限 softmax；
- **候选数不受限**：`choice` 超过分块窗口（默认 128）就分块，跨块用 log-odds 对齐，
  块间靠 `[None]` 锚点定零点；
- **一次请求一次 decode**：同一个 `state` 下的所有问题合成一批，公共前缀只算一遍；
- **契约对齐官方**：字段与错误码逐条对齐，非法输入直接 `422`，不降级不换路。

## 感谢

权重来自这两个仓库，感谢作者开放：

- 原版：[openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
- 去审查对比版：[Abiray/MiniCPM5-2B-heretic-abliterated-GGUF](https://huggingface.co/Abiray/MiniCPM5-2B-heretic-abliterated-GGUF)

思路参考 [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev)：把「决策」从自由生成里
拆出来，变成对固定候选的受限打分。

## 环境要求

- Python 3.11，Windows / Linux 均可；
- 纯 CPU 能跑（兜底档）；推荐 NVIDIA GPU + CUDA 12.x；
- 2B Q4_K_M 全层 offload 约 3GB 显存，8GB 卡可以把 `n_ctx` 开到 32k；
- 权重约 1.5GB/个，不入库，由脚本下载。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyPI 上的 `llama-cpp-python` wheel 是纯 CPU 版，装完把 `config.json` 的 `device`
设成 `cpu` 就能跑。要用 GPU，跑 `bash scripts/build_llama_cuda.sh` 源码编译
（需要 VS2022 + CUDA Toolkit + cmake + ninja），脚本会把产物换装进 `llama_cpp/lib/`，
首次十几分钟、之后走增量。

## 下载模型

```powershell
.venv\Scripts\python.exe scripts\download_model.py --quant list
.venv\Scripts\python.exe scripts\download_model.py --quant Q4_K_M --parts 12
```

默认落到 `weights/`（已在 `.gitignore` 里）。`--source abliterated` 可以下
heretic 去审查版，用来做对比。

## 启动服务

```powershell
.venv\Scripts\python.exe -m minicpm_jev.service
```

默认读仓库根目录的 `config.json`；命令行参数优先级更高：

```powershell
.venv\Scripts\python.exe -m minicpm_jev.service --device cpu --n-ctx 8192
```

常用配置项（完整见 `config.json`）：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `host` / `port` | `127.0.0.1` / `8000` | 监听地址 |
| `api_key` | `null` | 不填 = 不鉴权；填了要带 `Authorization: Bearer <key>` |
| `model_path` | `weights/MiniCPM5-2B-Q4_K_M.gguf` | 启动默认模型 |
| `model_aliases` | `{}` | 请求里的模型名 -> 本地路径 |
| `device` | `gpu` | `gpu` / `cpu`，CPU 档会屏蔽 CUDA 设备 |
| `n_ctx` | `32768` | 上下文长度，上限 131072，超了启动报错 |
| `chunk_size` | `128` | 候选分块窗口 (0, 128] |
| `n_seq_max` | `16` | 一批能挂的去重序列条数上限 |
| `queue_max` | `30` | 请求队列上限，超了返回 429 |

## 调用

```powershell
curl.exe -s http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d "{\"state\":\"Invoice #4471 issued March 3, 2026 for $12,840.00, net 30.\",\"model\":\"MiniCPM5-2B-Q4_K_M\",\"questions\":{\"amount_due\":{\"type\":\"score\",\"instructions\":{\"field\":{\"name\":\"amount_due\",\"type\":\"number\",\"unit\":\"USD\",\"description\":\"The total the invoice asks to be paid.\"},\"question\":\"How large is the field value?\"},\"criteria\":[\"Under $1,000\",\"$1,000 to $10,000\",\"$10,000 to $100,000\"]}}}"
```

三原语写在 `questions` 里，`state` 是共用背景。完整的请求与响应形状见
`examples/systemone_http.py`。

## 示例

`examples/` 下四个可直接运行的脚本：

| 脚本 | 演示什么 |
| --- | --- |
| `maze_walk.py` | 随机墙迷宫实走：每步一次 `choice`，终端就地刷新、带配色与单步耗时 |
| `dnd_checks.py` | 跑团判定：命中/豁免走 `noul`，先攻/伤害期望走 `choice` |
| `text_checks.py` | 文本判别三个场景：垃圾邮件/钓鱼（`noul`）、情感（`choice`）、优先级标签（`score`） |
| `systemone_http.py` | 走 HTTP 打 `/v1/systemone`，官方契约的请求与响应 |

## 测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

## 目录结构

```
minicpm_jev/
  template/   官方对话模板渲染 (不手搓标记)
  labels/     受限标签组与组内受限 softmax
  chunking/   候选分块规划与跨块 log-odds 对齐
  decision/   三原语打分与答案解码
  runtime/    批量推理引擎 (一次 decode 打完整批, 公共前缀只算一遍)
  service/    FastAPI 服务 (官方 /v1/systemone 契约)
  config/     config.json 与命令行装载
scripts/      模型下载、CUDA 编译、批大小甜点测量
tests/        pytest 矩阵与官方接口 dump
examples/     场景示例
img/          效果图
```

## 局限性

实测出来的边界，不是猜的：

- **`score` 档位基本不可用**。档位能从文本直接读出来时还行（优先级标签 7/8），
  一旦要模型自己把量映射到区间就崩：金额落在哪档 5/12、数几件 3/10、几颗星 2/8、
  工单多急 4/12、命令多危险 4/12。换成 `choice` 问也救不回来，给题面补一句
  「档位从低到高排列」更是逐条一字未变 —— 缺的是序数量级判断能力，不是题面没写清。
- **`noul` 有 token 级先验**。垃圾邮件二分类 9/16，模型在 yes/no 之间偏向某一极，
  换成 true/false 措辞只是把偏置翻个面，先验校正也救不回来；同样这批题用 `choice`
  双选能到 14/16。
- **`choice` 最稳，但也怕细粒度**。情感三分类 9/14，错的六条全是 neutral 被判成
  positive —— 中性档是它最弱的一档。
- **延迟随 state 线性增长**。默认（不思考）下 state 约 500 token 以内能压在 100ms，
  再长就是 prefill 的物理限制。
- **迷宫对题面极度敏感**。同一张图，四邻探测当选项 + 明确当前坐标能到 3/3，
  把探测挪进 state 或只给全景就掉到 2/3 甚至 0/3。

不过也别小看它：纯推理类任务（多步算术、星期推算、单位换算、空间关系、反事实）
两个权重都能满分，受限决策这条路本身是通的。

## 许可证

[MIT](LICENSE)

---

这是一个**实验项目**，用来验证「2B 量级的小模型 + 受限决策」能走到哪一步，
不承诺生产可用性。

代码由 LLM 编写。

项目地址：https://github.com/1cyberlangke1/minicpm_jev_like
