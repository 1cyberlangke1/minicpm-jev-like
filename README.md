# minicpm_jev_like

基于 MiniCPM5-2B (GGUF) 实现的 **JEV / TypeSafe 风格快速非自回归决策引擎**：模型不执行自由文本生成，仅对预设的候选选项计算受限概率分布。完整支持 `noul`（布尔判断）、`choice`（多选一）与 `score`（离散档位加权期望分）三种决策原语，接口兼容官方 `POST /v1/systemone` 规范，支持本地 CPU 与 GPU 双后端。

![走迷宫](img/maze_walk.gif)

上图为 `examples/maze_walk.py` 在终端环境的实时录制效果（12×12 迷宫，随机种子 42，耗费 18 步达成最短路径）。动画中每一步画面的停顿即为本地实机单次决策的真实耗时。

## 运行环境

- Python 3.11（支持 Windows 及 Linux 操作系统）；
- 支持纯 CPU 运行（内置基线兜底）；推荐搭载 NVIDIA GPU 及 CUDA 12.x 驱动环境；
- 2B Q4_K_M 量化版本完全卸载至 GPU 显存约占用 3GB；在 8GB 显存设备上可将上下文长度 `n_ctx` 扩展至 32k；
- 模型权重体积约 1.5GB/个（未纳入代码仓库，可通过内置脚本拉取）。

## 环境安装

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyPI 默认分发的 `llama-cpp-python` 预编译包通常为 CPU 版本，安装完成后将配置文件 `config.json` 中的 `device` 设为 `cpu` 即可启动。如需启用 GPU 加速，请执行 `bash scripts/build_llama_cuda.sh` 从源码编译（需预先配置 Visual Studio 2022、CUDA Toolkit、CMake 与 Ninja 工具链），编译产物将自动部署至 `llama_cpp/lib/` 路径下。

## 下载模型权重

```powershell
.venv\Scripts\python.exe scripts\download_model.py --quant list
.venv\Scripts\python.exe scripts\download_model.py --quant Q4_K_M --parts 12
```

模型默认下载并保存至 `weights/` 目录（该路径已被 `.gitignore` 排除）。添加参数 `--source abliterated` 可下载去审查对照版本。

## 启动服务

```powershell
.venv\Scripts\python.exe -m minicpm_jev.service
```

服务启动时默认加载仓库根目录下的 `config.json` 配置文件；亦可通过命令行参数覆盖相应配置：

```powershell
.venv\Scripts\python.exe -m minicpm_jev.service --device cpu --n-ctx 8192
```

核心配置参数说明（完整参数参见 `config.json`）：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `host` / `port` | `127.0.0.1` / `8000` | 服务监听 IP 与端口号 |
| `api_key` | `null` | API 密钥（为空时不鉴权；配置后须携带 `Authorization: Bearer <key>` 鉴权头） |
| `model_path` | `weights/MiniCPM5-2B-Q4_K_M.gguf` | 默认加载的模型文件路径 |
| `model_aliases` | `{}` | 外部请求模型标识与本地实际文件路径的别名映射 |
| `device` | `gpu` | 运行设备，支持 `gpu` 或 `cpu`（选 CPU 时将屏蔽 CUDA 设备） |
| `n_ctx` | `32768` | 模型上下文长度（上限 131072，超出限制将在启动时拦截报错） |
| `chunk_size` | `128` | 多选候选分块窗口大小 (0, 128] |
| `n_seq_max` | `16` | 单批次允许挂载的去重序列上限 |
| `queue_max` | `30` | 待处理请求并发队列容量上限（超出返回 429） |

## 接口调用示例

```powershell
curl.exe -s http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d "{\"state\":\"Invoice #4471 issued March 3, 2026 for $12,840.00, net 30.\",\"model\":\"MiniCPM5-2B-Q4_K_M\",\"questions\":{\"amount_due\":{\"type\":\"score\",\"instructions\":{\"field\":{\"name\":\"amount_due\",\"type\":\"number\",\"unit\":\"USD\",\"description\":\"The total the invoice asks to be paid.\"},\"question\":\"How large is the field value?\"},\"criteria\":[\"Under $1,000\",\"$1,000 to $10,000\",\"$10,000 to $100,000\"]}}}"
```

多个决策问题可统一写入 `questions` 映射表，`state` 作为共用前置背景信息。端到端请求与响应结构详见 `examples/systemone_http.py`。

## 场景示例

`examples/` 目录下提供以下场景的可执行验证脚本：

| 脚本文件 | 演示内容 |
| --- | --- |
| `maze_walk.py` | 随机墙迷宫路径规划：每步执行一次 `choice` 决策，支持控制台实时重绘、高亮提示与单步延迟统计 |
| `dnd_checks.py` | 桌面角色扮演判定模拟：命中与豁免判定使用 `noul`，先攻与伤害期望评估使用 `choice` |
| `text_checks.py` | 文本分析三口径对比：垃圾邮件与钓鱼检测（`noul`）、情感极性分类（`choice`）、优先级标签提取（`score`） |
| `systemone_http.py` | 基于标准 HTTP 请求调用 `/v1/systemone` 接口，演示官方契约下的请求组装与答案解析 |

## 核心机制

传统的文本分类或信息抽取任务通常依赖生成式模型输出一段文本再行解析。本项目采用受限解码方案：直接将待选候选项嵌入输入上下文末尾，让 2B 参数量级的轻量模型在决策位置执行单步前向传播，仅对指定候选词计算受限概率分布，并提取置信度最高的结果。

- **非自回归单步推理**：跳过逐字生成的自回归循环，仅在决策位读取一次未归一化概率（logits），并在候选词集合内完成受限 Softmax 计算。
- **无上限候选分块处理**：当 `choice` 选项数量超出单批分块窗口（默认 128）时，系统自动切分数据块，并通过跨块的 `[None]` 锚点与对数几率差完成全局概率对齐。
- **共享前缀单次解码**：归属于同一个上下文环境（`state`）下的多个并发决策问题会自动合并至同一计算批次，公共提示词前缀仅计算一次缓存。
- **对齐官方接口契约**：入参格式校验与错误码与官方规范一致，遇到非法请求体直接返回 `422` 状态码。

## 自动化测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

## 工程目录结构

```
minicpm_jev/
  template/   模型输入对话模板渲染引擎（严格遵循官方标记规范）
  labels/     候选标签映射组与组内受限 Softmax 计算逻辑
  chunking/   超长候选集分块规划器与跨块对数几率差对齐模块
  decision/   三原语逻辑判定引擎与响应结构组装解码
  runtime/    底层高性能批处理推理引擎（批内共享公共前缀单次解码）
  service/    基于 FastAPI 实现的服务网关（实现 /v1/systemone 规范）
  config/     JSON 配置加载与命令行参数解析组件
scripts/      模型权重获取、CUDA 动态库编译脚本与批处理基准测试工具
tests/        自动化单元测试矩阵与官方接口契约对照验证
examples/     完整应用场景演示代码
img/          控制台运行实况动态演示图
```

## 实测局限性与认知边界

以下为基于通用 2B 模型在未执行针对性强化微调（RLCD/DPO）前提下的客观实测数据：

- **`score` 档位离散量级判断能力有限**：当档位描述完全存在于原始文本中时，模型能够较好地完成文本复制对应（例如工单优先级标签命中率为 7/8）；但当任务需要模型自主理解连续数值并将其映射至抽象区间时，准确率下降明显：金额区间判断为 5/12、数量计数判断为 3/10、星级评分转换为 2/8、工单紧急程度判断为 4/12、危险命令评级为 4/12。即便在提示词中追加显式顺序提示（如“档位按由低到高严格排列”），预测输出分布亦无显著变化，表明未微调小模型在单步前向传播下缺少对序数量级的连续空间感知。
- **`noul` 布尔输出存在词元级先验偏差**：在垃圾邮件二分类测试中准确率为 9/16，模型在 `yes`/`no` 之间存在固有先验倾斜；将其替换为 `true`/`false` 仅改变了偏差方向，静态先验校准难以完全抹平该效应。相比之下，若将其重构为语义具象的 `choice` 双候选对抗（如 `normal` vs `spam`），准确率可提升至 14/16。
- **`choice` 语义对齐最稳定，但对细粒度中性分类敏感**：在标准情感三分类任务中准确率为 9/14，误判样本主要集中在将“中性（neutral）”文本归类为“积极（positive）”。
- **单步延迟受输入长度线性约束**：在无思考链（零思考步）模式下，当输入上下文 `state` 处于 500 token 以内时，推理耗时可稳定在 100ms 级别；更长文本的延迟主要受制于预填充（Prefill）阶段的物理计算量。
- **空间规划任务高度依赖局部观察特征输入**：在迷宫实走实验中，若将四向障碍探测作为结构化选项并显式提供当前坐标，连续走通率为 3/3；若仅在 `state` 中提供全局迷宫字符画，成功率则下降至 0/3 ~ 2/3。

尽管在连续量级量化任务上存在局限，但在基础逻辑推理任务（如多步算术运算、日期推算、单位换算、相对空间关系与反事实判别）中，模型表现出良好的受限决策闭环能力。

## 致谢与引用

指令微调（Instruct）权重来源于以下开源仓库，感谢原作者的开放贡献：

- 官方基础版：[openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
- 去审查对照版：[Abiray/MiniCPM5-2B-heretic-abliterated-GGUF](https://huggingface.co/Abiray/MiniCPM5-2B-heretic-abliterated-GGUF)

- 思路借鉴自 [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev)。

## 开源许可证

本项目遵循 [MIT License](LICENSE) 协议发布。

---

这是一个实验项目，用来验证 2B 模型受限决策的能力，不承诺生产可用性。

项目代码借助大语言模型协作编写。
