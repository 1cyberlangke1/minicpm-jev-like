"""迷宫实走: 固定地图, 每走一步就地刷新终端.

地图是「随机撒墙 + 兜底通路」: 内部格子按 WALL_RATIO 随机立墙, 再检查 S 到 G 通
不通, 不通就沿「最少破墙」的路线补开几个口子。不要求全连通 —— 允许有走不到的死角,
只要 S 到 G 至少有一条路就行; 这样地图天然带环, 同一位置往往有好几条路可走,
才谈得上靠历史判断哪条更靠近终点。
seed 默认 42 (命令行可改)。

题面要给三样东西: 四邻探测当编号选项、当前坐标、最近几次决策。少了坐标与历史,
模型看不到自己在哪、刚走过哪, 只会在两格之间来回横跳。

输入: --seed (默认 42) / --model-path / --device / --n-ctx;
输出: 终端就地刷新的地图 + 四个方向选项的概率进度条与数字, 别的都不打;
      重画走「光标回原点 + 逐行清行尾 + 整屏一次 write」, 不做整屏清空, 所以不闪;
预期: 撞墙或到达终点即结束; 走满最短步数 3 倍还没到也结束 (说明在打转)。
"""

from __future__ import annotations

import argparse
import atexit
import random
import sys
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minicpm_jev import (  # noqa: E402
    BatchEngine,
    Choice,
    Device,
    EngineConfig,
    answer_choice,
)

DEFAULT_MODEL = ROOT / "weights" / "MiniCPM5-2B-Q4_K_M.gguf"

SIZE = 12
#: 内部格子随机立墙的比例: 越大越挤, 越小越空旷
WALL_RATIO = 0.35
DEFAULT_SEED = 42
BAR_WIDTH = 8
#: 题面里回看的最近决策条数
HISTORY = 5
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: 每个方向在坐标上意味着什么: 要写进题面, 否则模型不知道 up 是往哪走
HINTS = {
    "up": "row - 1",
    "down": "row + 1",
    "left": "column - 1",
    "right": "column + 1",
}
#: 地图字符的读法: 直接写在每个选项里, 省掉「回 legend 查一遍」那一跳
CHAR_NAMES = {
    "#": "wall",
    ".": "open",
    "*": "visited",
    "S": "start",
    "G": "goal",
}
ASK = (
    "Which single step moves the explorer toward the goal 'G' without entering a wall?"
)
WALL = "#"

#: 迷宫示例内部的几个形状: 字符网格 / 一格坐标 / 逐方向概率表
Grid = list[list[str]]
Cell = tuple[int, int]
Probs = Mapping[str, float]

#: 光标回左上角 (不清屏)
HOME = "\033[H"
#: 整屏清空: 只在第一帧用一次, 抹掉启动前终端里残留的内容
CLEAR_SCREEN = "\033[2J"
#: 从光标清到行尾, 覆盖写时用来抹掉上一帧残留
CLEAR_TO_END = "\033[K"
#: 藏光标 / 显光标: 覆盖写时光标乱跳本身就是一种闪
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
#: 终端配色: 当前格亮绿, 走过的轨迹青色, 本步概率最高的选项绿色
YOU_COLOR = "\033[1;32m"
TRAIL_COLOR = "\033[36m"
PICK_COLOR = "\033[32m"
RESET = "\033[0m"


def generate(size: int, rng: random.Random) -> list[list[str]]:
    """随机撒墙, 再保证 S 到 G 至少有一条路.

    输入: size -- 边长 (>=5); rng -- 随机源;
    输出: 字符网格, '#' 墙, '.' 通路, 'S' 起点 (左上), 'G' 终点 (右下);
    预期: 内部格子按 WALL_RATIO 随机立墙 (外圈永远是墙); 若 S 到 G 不连通, 就沿
          「最少破墙」的路线补开几个口子。不保证全连通, 允许存在走不到的死角。
    """
    if size < 5:
        raise ValueError(f"size 必须 >=5, 收到 {size}")
    grid = [[WALL] * size for _ in range(size)]
    for row in range(1, size - 1):
        for column in range(1, size - 1):
            grid[row][column] = WALL if rng.random() < WALL_RATIO else "."
    grid[1][1] = "."
    grid[size - 2][size - 2] = "."
    carve_path(grid, (1, 1), (size - 2, size - 2))
    grid[1][1] = "S"
    grid[size - 2][size - 2] = "G"
    return grid


def carve_path(grid: Grid, start: Cell, goal: Cell) -> None:
    """S 到 G 不通时, 沿破墙最少的路线把墙打通.

    输入: grid -- 网格 (原地改); start / goal -- 起点与终点坐标;
    输出: 无;
    预期: 0-1 BFS —— 踩通路代价 0、踩墙代价 1, 所以找出来的是「最少破几面墙」的
          路线; 回溯时把沿途的墙改成通路。本来就连通时一个格子都不会动。
    """
    size = len(grid)
    distance: list[list[int | None]] = [[None] * size for _ in range(size)]
    previous: list[list[tuple[int, int] | None]] = [[None] * size for _ in range(size)]
    distance[start[0]][start[1]] = 0
    queue = deque([start])
    while queue:
        row, column = queue.popleft()
        here = distance[row][column]
        # 只有已经定下距离的格才会被入队, 这里只是把 Optional 收窄成 int
        assert here is not None
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + delta_row
            next_column = column + delta_column
            if not (0 <= next_row < size and 0 <= next_column < size):
                continue
            cost = 1 if grid[next_row][next_column] == WALL else 0
            known = distance[next_row][next_column]
            if known is not None and known <= here + cost:
                continue
            distance[next_row][next_column] = here + cost
            previous[next_row][next_column] = (row, column)
            if cost == 0:
                queue.appendleft((next_row, next_column))
            else:
                queue.append((next_row, next_column))
    row, column = goal
    parent = previous[row][column]
    while parent is not None:
        grid[row][column] = "."
        row, column = parent
        parent = previous[row][column]
    grid[start[0]][start[1]] = "."


def shortest_steps(grid: Grid, start: Cell, goal: Cell) -> int:
    """BFS 最短步数. 输入: 网格 + 起点终点; 输出: 步数 (到不了返回 -1)."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) == goal:
            return distance
        for delta_row, delta_column in DELTAS.values():
            next_row = row + delta_row
            next_column = column + delta_column
            if grid[next_row][next_column] == WALL:
                continue
            if (next_row, next_column) in seen:
                continue
            seen.add((next_row, next_column))
            queue.append(((next_row, next_column), distance + 1))
    return -1


def neighbors(grid: Grid, row: int, column: int) -> dict[str, str]:
    """四邻探测. 输入: 网格 + 坐标; 输出: 方向 -> 方向语义 + 那一格是什么.

    描述写成 "row - 1: '#' wall" 这种形式: "row - 1" 是把方向名和字符网格对上的
    那座桥 (实测去掉后五个 seed 全败), 而坐标 "(5,4)" 是多余的第三份信息
    (地图上已经有了), 留着只是噪声。
    """
    found: dict[str, str] = {}
    for name, (delta_row, delta_column) in DELTAS.items():
        char = grid[row + delta_row][column + delta_column]
        found[name] = f"{HINTS[name]}: '{char}' {CHAR_NAMES[char]}"
    return found


def render_map(board: Grid, row: int, column: int, *, color: bool = False) -> str:
    """地图画成带边框的方块, 当前格显示成 '@'.

    输入: board -- 字符网格; row / column -- 当前位置; color -- 是否上色;
    输出: 多行文本;
    预期: color=True 时当前格亮绿、走过的格子青色; 给模型的题面必须 color=False,
          否则 ANSI 转义码会混进提示词污染输入。
    """
    lines = ["+" + "-" * len(board[0]) + "+"]
    for index, line in enumerate(board):
        cells = []
        for cell_index, char in enumerate(line):
            if index == row and cell_index == column:
                cells.append(f"{YOU_COLOR}@{RESET}" if color else "@")
            elif color and char == "*":
                cells.append(f"{TRAIL_COLOR}*{RESET}")
            else:
                cells.append(char)
        lines.append("|" + "".join(cells) + "|")
    lines.append("+" + "-" * len(board[0]) + "+")
    return "\n".join(lines)


def build_state(board: Grid, row: int, column: int, history: list[str]) -> str:
    """拼给模型的题面背景 (走 system 位).

    输入: board -- 当前网格 (含已走过的 '*'); row / column -- 当前位置;
          history -- 逐条决策记录;
    输出: 多行纯文本;
    预期: 不含 ANSI 转义码 (给模型的输入不能被终端配色污染)。
    """
    return (
        "Maze:\n"
        + render_map(board, row, column)
        + f"\n\nCurrent position: row {row}, column {column}."
        + "\nRecent moves: "
        + ("; ".join(history[-HISTORY:]) if history and HISTORY else "(none)")
    )


def render_probe(probabilities: Probs) -> str:
    """四个方向选项的概率进度条 + 数字, 排成两行: 上/下 一行, 左/右 一行.

    本步概率最高的那一格整格标绿, 一眼能看出模型选了哪个方向。
    """
    best = (
        max(probabilities, key=lambda name: probabilities[name])
        if probabilities
        else None
    )
    cells = []
    for name in ("up", "down", "left", "right"):
        value = probabilities.get(name, 0.0)
        filled = round(value * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        text = f"{name:<6}{bar} {value:.3f}"
        cells.append(f"{PICK_COLOR}{text}{RESET}" if name == best else text)
    return f"  {cells[0]}    {cells[1]}\n  {cells[2]}    {cells[3]}"


def paint(text: str, *, clear: bool = False) -> None:
    """整屏一次覆盖写.

    输入: text -- 整屏文本 (可含换行); clear -- 是否先整屏清一次;
    输出: 无;
    预期: 光标先回左上角, 每行写完补一个清行尾, 最后一次性 write 出去。
          平时不用 \\033[2J —— 整屏清空会让终端先黑一帧再重画, 看着就是闪;
          只有第一帧 clear=True 清一次, 之后全靠覆盖写。
    """
    body = "\n".join(f"{line}{CLEAR_TO_END}" for line in text.split("\n"))
    prefix = CLEAR_SCREEN if clear else ""
    sys.stdout.write(prefix + HOME + body + "\n")
    sys.stdout.flush()


def draw(
    board: Grid,
    row: int,
    column: int,
    probabilities: Probs,
    trail: list[str],
    note: str,
    model: str,
    *,
    clear: bool = False,
) -> None:
    """覆盖重画整屏: 模型名 + 地图 + 概率 + 移动轨迹 + 结果 + 底部留白."""
    if trail:
        recent = trail[-HISTORY:]
        moves = ("... " if len(trail) > len(recent) else "") + " ".join(recent)
    else:
        moves = "(start)"
    paint(
        f"model: {model}\n\n"
        f"{render_map(board, row, column, color=True)}\n\n"
        f"{render_probe(probabilities)}\n\n"
        f"{TRAIL_COLOR}moves: {moves}{RESET}\n\n{note}\n\n\n",
        clear=clear,
    )


def walk(
    engine: BatchEngine,
    grid: Grid,
    start: Cell,
    goal: Cell,
    limit: int,
    model: str = "",
) -> tuple[bool, int]:
    """走迷宫, 每步就地刷新. 输出: (是否通关, 步数)."""
    board = [list(line) for line in grid]
    row, column = start
    steps = 0
    history: list[str] = []
    trail: list[str] = []
    drawn = False

    def show(probabilities: Probs, note: str) -> None:
        """画一帧: 第一帧整屏清一次, 之后都是覆盖写."""
        nonlocal drawn
        draw(
            board, row, column, probabilities, trail, note, model,
            clear=not drawn,
        )
        drawn = True

    while steps < limit:
        steps += 1
        question = Choice(
            instructions=ASK, criteria=neighbors(board, row, column)
        )
        state = build_state(board, row, column, history)
        started = time.perf_counter()
        answer = answer_choice(engine, state, question)
        elapsed = (time.perf_counter() - started) * 1000.0
        picked = answer.choice
        delta_row, delta_column = DELTAS[picked]
        next_row = row + delta_row
        next_column = column + delta_column
        note = f"step {steps}: {picked} ({elapsed:.0f}ms)"
        if board[next_row][next_column] == WALL:
            show(answer.probabilities, note + " hit a wall, stop")
            return False, steps
        board[row][column] = "*"
        row, column = next_row, next_column
        history.append(f"{picked} -> ({row},{column})")
        trail.append(picked)
        if (row, column) == goal:
            show(answer.probabilities, note + " reached the goal")
            return True, steps
        show(answer.probabilities, note)
    return False, steps


def main() -> int:
    """生成默认边长的迷宫并实走."""
    parser = argparse.ArgumentParser(description="迷宫实走")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="随机种子, 默认 42"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args()

    grid = generate(SIZE, random.Random(args.seed))
    start = (1, 1)
    goal = (SIZE - 2, SIZE - 2)
    limit = shortest_steps(grid, start, goal) * 3
    sys.stdout.write(HIDE_CURSOR)
    atexit.register(lambda: sys.stdout.write(SHOW_CURSOR + "\n"))
    with BatchEngine(
        EngineConfig(
            model_path=args.model_path,
            device=Device(args.device),
            n_ctx=args.n_ctx,
        )
    ) as engine:
        walk(engine, grid, start, goal, limit, args.model_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
