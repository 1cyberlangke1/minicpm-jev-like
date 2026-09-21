"""跑团判定: 命中/豁免走 noul, 先攻/伤害期望走 choice.

输入: 无 (题目写在表里); 输出: 每题 模型答案 与 金标 的对比;
预期: 命中与豁免是纯算术, 两个模型都对; 伤害期望那题是共同短板 (见下)。
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
    answer_choice,
    answer_noul,
)

MODEL = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"

# (名字, 问题, 期望 yes)
CHECKS = (
    ("hit", "Attack roll: d20 shows 14, attack bonus +3, target AC 15. Does the attack hit?", True),
    ("miss", "Attack roll: d20 shows 8, attack bonus +2, target AC 16. Does the attack hit?", False),
    ("save_fail", "Dexterity save: d20 shows 11, DEX modifier +2, DC 14. Does the save succeed?", False),
    ("save_pass", "Constitution save: d20 shows 15, CON modifier +4, DC 18. Does the save succeed?", True),
    ("advantage", "Attack with advantage: rolls are 8 and 16, take the higher, bonus +2, target AC 15. Does it hit?", True),
    ("crit", "Attack roll: d20 shows 20, attack bonus +5, target AC 22. Does the attack hit?", True),
)

# (名字, 问题, 候选, 金标)
PICKS = (
    (
        "initiative",
        "Initiative order: the rogue has DEX +4, the wizard +2, the fighter +1. Who acts first?",
        {"rogue": "DEX +4", "wizard": "DEX +2", "fighter": "DEX +1"},
        "rogue",
    ),
    (
        # 期望值 d4+3=5.5 > d8=4.5, 但两个模型都倾向选面数大的骰子, 是已知短板
        "best_damage",
        "Which option has the highest average damage?",
        {"d4+3": "one four-sided die plus 3", "d6+1": "one six-sided die plus 1", "d8": "one eight-sided die"},
        "d4+3",
    ),
)


def main() -> int:
    """跑两类判定."""
    with BatchEngine(EngineConfig(model_path=MODEL, n_ctx=4096)) as engine:
        hits = 0
        for name, question, expected in CHECKS:
            answer = answer_noul(engine, "", Noul(instructions=question))
            got = answer.noul >= 0.5
            hits += int(got == expected)
            print(
                f"{name:<10} 模型={'yes' if got else 'no ':<3} "
                f"金标={'yes' if expected else 'no ':<3} "
                f"P(yes)={answer.noul:.3f}"
            )
        print(f"判定 {hits}/{len(CHECKS)}")
        for name, question, criteria, expected in PICKS:
            answer = answer_choice(
                engine, "", Choice(instructions=question, criteria=criteria)
            )
            print(
                f"{name:<12} 模型={answer.choice:<8} 金标={expected:<8} "
                f"confidence={answer.confidence:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
