"""文本判别的三个场景: 二分类 (noul) / 多选一 (choice) / 档位 (score).

用例中英混合, 都是「给一段文本, 出一个受限标签」的活。三个口径的实测准确率
如实写在下面, 用来划清这套引擎的边界 —— 能做什么、哪一环最弱, 看数字最清楚:

- ``noul`` 二分类 (垃圾邮件/钓鱼 9/16): 它只有一个出口, 模型容易无脑答某一极,
  换成 true/false 措辞只是把偏置翻个面, 先验校正也救不回来;
- ``choice`` 多选一 (情感 9/14): 错的六条全是 neutral 被判成 positive,
  中性档是这个模型最弱的一档;
- ``score`` 档位 (优先级标签 7/8): 这是 ``score`` 唯一像样的成绩, 但**好看不等于
  有用** —— 它的档位在文本里明写 (P0~P3), 模型只要把看到的标签抄成编号, 属于
  查表而不是估计; 同样的事用正则提取能拿满分, 业务上没有价值。

``score`` 的真实边界在于「档位能不能从文本直接读出」: 能读出来 (上面的优先级)
就 7/8; 要模型自己把量映射到区间就全崩 —— 金额落在哪档 5/12、数几件 3/10、
几颗星 2/8、工单多急 4/12、命令多危险 4/12。而且把同一批档位题改用 ``choice``
问也救不回来 (金额 5/12 -> 7/12、数量 3/10 -> 4/10、星级 2/8 -> 2/8),
给题面补一句「档位从低到高排列」更是逐条一字未变 —— 所以瓶颈不是原语、不是
编号标签、也不是档位描述写得清不清楚, 而是模型缺少序数量级判断能力。

输入: 无 (用例写在 TASKS 里); 输出: 每题的 模型答案 / 金标 / 置信度 与分任务准确率;
预期: 二分类走 noul, 多选一走 choice, 档位走 score, 三个都带中文用例。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicpm_jev import (  # noqa: E402
    BatchEngine,
    Choice,
    EngineConfig,
    Noul,
    Score,
    answer_choice,
    answer_noul,
    answer_score,
)

DEFAULT_MODEL = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"

SPAM_CRITERIA = {
    "true": "unsolicited advertising, a scam, or phishing",
    "false": "a normal personal or work message",
}
SENTIMENT_CRITERIA = {
    "positive": "the writer is pleased or approving",
    "neutral": "the writer reports a fact without feeling either way",
    "negative": "the writer is unhappy, complaining, or critical",
}
PRIORITY_LEVELS = [
    "P0 - drop everything",
    "P1 - high",
    "P2 - normal",
    "P3 - low",
]

# (任务名, 题型, 问题, 候选/档位, [(输入, 金标)])
# 金标随题型变: noul 是 bool, choice 是选项名, score 是档位下标
Gold = bool | str | int
Task = tuple[
    str, str, str, "Mapping[str, Any] | Sequence[str]", tuple[tuple[str, Gold], ...]
]
TASKS: tuple[Task, ...] = (
    (
        "垃圾邮件/钓鱼",
        "noul",
        "Is this message spam?",
        SPAM_CRITERIA,
        (
            ("Congratulations! You've won a $1000 gift card. Click here to claim now!", True),
            ("Meeting moved to 3pm in room 402.", False),
            ("URGENT: verify your account within 24 hours or it will be suspended", True),
            ("Your package has shipped, tracking number 1Z999AA10123456784.", False),
            ("Cheap meds, 90% off, no prescription needed!!!", True),
            ("Can you review my pull request when you have time?", False),
            ("Lunch tomorrow? Let me know.", False),
            ("You have been selected as a winner, reply with your bank details", True),
            ("恭喜您获得一等奖，点击链接立即领取奖品", True),
            ("明天下午三点在 402 会议室开会", False),
            ("您的账户存在异常，请立即点击链接验证身份", True),
            ("您的快递已发出，单号 SF1234567890", False),
            ("低价代购名牌手表，加微信详聊", True),
            ("这份文档麻烦你有空帮我看一下", False),
            ("周末一起打球吗？", False),
            ("您已被抽中幸运用户，请提供银行卡号领取奖金", True),
        ),
    ),
    (
        "情感倾向",
        "choice",
        "What is the sentiment of this text?",
        SENTIMENT_CRITERIA,
        (
            ("The update broke everything and support never replied.", "negative"),
            ("Works exactly as advertised and shipped fast.", "positive"),
            ("The package arrived on Tuesday.", "neutral"),
            ("It is okay, nothing special.", "neutral"),
            ("Worst purchase I have ever made.", "negative"),
            ("The team did a fantastic job on this release.", "positive"),
            ("发货很快，质量比想象中好。", "positive"),
            ("客服三天没回复，太失望了。", "negative"),
            ("包裹周二到了。", "neutral"),
            ("还行吧，没什么特别的。", "neutral"),
            ("这是我买过最差的东西。", "negative"),
            ("这次版本做得非常棒。", "positive"),
            ("说明书上写的是三号电池。", "neutral"),
            ("用了一周就坏了，客服还推卸责任。", "negative"),
        ),
    ),
    (
        "优先级标签",
        "score",
        "Which priority label does this ticket carry?",
        PRIORITY_LEVELS,
        (
            ("Ticket priority: P2. Customer cannot export reports.", 2),
            ("Priority: P0. Production is down.", 0),
            ("This one is P3, no rush at all.", 3),
            ("Marked as P1 by the on-call engineer.", 1),
            ("优先级：P2，报表导出失败。", 2),
            ("优先级：P0，生产环境宕机。", 0),
            ("工单等级 P3，不急。", 3),
            ("已标为 P1，需要尽快处理。", 1),
        ),
    ),
)


def run_noul(
    engine: BatchEngine,
    question: str,
    criteria: Mapping[str, Any] | None,
    text: str,
) -> tuple[bool, float]:
    """是/否题. 输入: 引擎 + 问题 + 正反释义 + 文本; 输出: (判定, P(yes))."""
    answer = answer_noul(
        engine, text, Noul(instructions=question, criteria=criteria)
    )
    return answer.noul >= 0.5, answer.noul


def run_choice(
    engine: BatchEngine,
    question: str,
    criteria: Mapping[str, Any],
    text: str,
) -> tuple[str, float]:
    """选择题. 输入: 引擎 + 问题 + 选项 + 文本; 输出: (选项名, 置信度)."""
    answer = answer_choice(
        engine, text, Choice(instructions=question, criteria=criteria)
    )
    return answer.choice, answer.confidence


def run_score(
    engine: BatchEngine,
    question: str,
    levels: Sequence[str],
    text: str,
) -> tuple[int, float, float]:
    """档位题. 输入: 引擎 + 问题 + 档位 + 文本; 输出: (档位下标, 置信度, 加权分)."""
    answer = answer_score(
        engine, text, Score(instructions=question, criteria=list(levels))
    )
    probabilities = [answer.probabilities[str(index)] for index in range(len(levels))]
    picked = max(range(len(levels)), key=lambda index: probabilities[index])
    return picked, answer.confidence, answer.score


def main() -> int:
    """跑完三个任务并打印逐题结果与准确率."""
    with BatchEngine(EngineConfig(model_path=DEFAULT_MODEL, n_ctx=4096)) as engine:
        for name, kind, question, extra, cases in TASKS:
            hits = 0
            print(f"\n== {name} ({kind}) ==")
            for text, expected in cases:
                # 三个口径的「模型答案」类型不同, 分开命名, 只在对齐时取并集
                result: Gold
                # 表里 extra 的实际形状由 kind 决定, 元组表达不了这种联动,
                # 所以每个分支按自己的题型收窄一次 (运行时数据就是这么配的)。
                if kind == "noul":
                    verdict, probability = run_noul(
                        engine, question, cast("Mapping[str, Any]", extra), text
                    )
                    result = verdict
                    confidence = probability if verdict else 1 - probability
                    detail = (
                        f"={'yes' if verdict else 'no '} "
                        f"金标={'yes' if expected else 'no '} P(yes)={probability:.3f}"
                    )
                elif kind == "score":
                    level, confidence, weighted = run_score(
                        engine, question, cast("Sequence[str]", extra), text
                    )
                    result = level
                    detail = f"档位={level} 金标={expected} 加权={weighted:.2f}"
                else:
                    option, confidence = run_choice(
                        engine, question, cast("Mapping[str, Any]", extra), text
                    )
                    result = option
                    detail = f"={option:<8} 金标={expected:<8}"
                hits += int(result == expected)
                print(f"  {detail} 置信={confidence:.2f} | {text[:48]}")
            print(f"  准确率 {hits}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
