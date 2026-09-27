from __future__ import annotations

# examples/run_all_examples.py —— 一把跑完所有示例的离线冒烟测试
#
# 它做一件很朴素的事：**依次以子进程方式**跑
#
#     01_quickstart.py / 03_react_text_mode.py / 04_memory.py
#     05_multiagent_sequential.py / 06_multiagent_hierarchical.py / 07_code_assistant.py
#
# 每个都加 `--offline`（离线：`ScriptedLLM` / echo 驱动，不联网、不需要 API key），
# 汇总退出码与耗时，最后打印一张 PASS/FAIL 表；有任何失败就返回非 0 退出码。
#
# 三个刻意的设计（也是"为什么不用 unittest 干这件事"的答案）：
#
#   1. **子进程而不是 import**：示例顶层就有 `sys.path.insert`、argparse、`raise
#      SystemExit(main())` 这类"脚本味"的代码，import 进来会互相污染
#      （`__name__ != "__main__"` 时行为还不一样）。子进程跑的是**真实入口**，
#      和用户手敲命令完全同构。
#
#   2. **超时**：任何一个示例挂住（等网络、死循环）都会把 CI 卡死。
#      这里给每个示例一个硬超时（默认 180 秒），超时算 FAIL 并杀掉进程。
#
#   3. **cwd + PYTHONPATH 双保险**：以仓库根为 cwd、并把仓库根塞进子进程的
#      PYTHONPATH —— 示例自己也会 `sys.path.insert`，两条路都通；
#      这里显式设置是为了让"某个示例忘了加那两行"也能跑起来。
#
# 跑法：`python3 examples/run_all_examples.py`
#       `python3 examples/run_all_examples.py --only 05,06`
#       `python3 examples/run_all_examples.py --verbose`

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent

#: 要跑的示例（顺序即执行顺序）。02 需要真实网络，故意不在列表里。
TARGETS: tuple[str, ...] = (
    "01_quickstart.py",
    "03_react_text_mode.py",
    "04_memory.py",
    "05_multiagent_sequential.py",
    "06_multiagent_hierarchical.py",
    "07_code_assistant.py",
)

WIDTH = 78


def title(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").split("\n"):
        print(f"  {line}".rstrip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依次离线跑 liteagent 的示例，汇总成一张 PASS/FAIL 表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="确认以离线模式运行（**默认就是**；保留这个开关是为了让所有示例的 CLI 形状一致）",
    )
    parser.add_argument(
        "--only", default="",
        help="只跑名字里含这些片段（逗号分隔）的示例，例如 --only 05,06",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="单个示例的超时秒数（默认 180；超时算 FAIL 并杀掉进程）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="把每个示例的完整 stdout / stderr 也打出来",
    )
    parser.add_argument(
        "--tail", type=int, default=4,
        help="失败时打印 stderr 的最后几行（默认 4）",
    )
    return parser.parse_args(argv)


def select_targets(only: str) -> list[str]:
    """按 `--only` 过滤（子串匹配，逗号分隔）；空串表示全跑。

    只按**顺序**返回 TARGETS 里的项，`--only` 不能凭空造出一个不存在的示例 ——
    想跑清单外的东西请改 TARGETS，而不是让这里猜。
    """
    fragments = [item.strip() for item in (only or "").split(",") if item.strip()]
    if not fragments:
        return list(TARGETS)
    return [name for name in TARGETS if any(frag in name for frag in fragments)]


def tail_lines(text: str, limit: int) -> list[str]:
    """取最后 limit 行非空行（失败时用来指路）。"""
    lines = [line for line in (text or "").strip().split("\n") if line.strip()]
    return lines[-limit:] if limit > 0 else []


#: 纯装饰行（分隔线）不配当"摘要"：示例的横幅是 `====` / `----` 拼出来的。
DECORATION = set("=-~*_# ")


def last_non_empty(text: str) -> str:
    """stdout 最后一行**有内容**的行（成功时当作一句话摘要）。

    跳过空行与纯装饰行（全由 `= - ~ * _ # 空格` 组成），否则标题横幅会被当成摘要。
    """
    for line in reversed((text or "").strip().split("\n")):
        stripped = line.strip()
        if stripped and not set(stripped) <= DECORATION:
            return stripped
    return ""


def run_one(name: str, *, timeout: float, verbose: bool, tail: int) -> dict[str, object]:
    """跑一个示例，返回 `{name, status, exit_code, seconds, summary, stderr_tail}`。

    `status` 取 `"PASS"` / `"FAIL"` / `"MISSING"` / `"TIMEOUT"`。
    **绝不让异常穿出去**：一个示例起不来不应该让整张表打不出来。
    """
    path = EXAMPLES_DIR / name
    if not path.is_file():
        return {"name": name, "status": "MISSING", "exit_code": None, "seconds": 0.0,
                "summary": "文件不存在（可能还没创建）", "stderr_tail": [], "stdout": "",
                "stderr": ""}

    env = dict(os.environ)
    # 把仓库根塞进 PYTHONPATH：示例自己也会 sys.path.insert，这里是第二道保险。
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    # 离线示例不需要任何 API key；显式清掉常见的几个，避免"本机有 key 就偷偷联网"。
    for key in ("LITEAGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                "ANTHROPIC_API_KEY"):
        env.pop(key, None)

    command = [sys.executable, str(path), "--offline"]
    print(f"  $ {' '.join(command)}")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {"name": name, "status": "TIMEOUT", "exit_code": None,
                "seconds": seconds, "summary": f"超过 {timeout:g}s 被强杀",
                "stderr_tail": tail_lines(stderr, tail), "stdout": stdout, "stderr": stderr}
    except OSError as exc:  # 解释器都起不来这种极端情况
        seconds = time.perf_counter() - started
        return {"name": name, "status": "FAIL", "exit_code": None, "seconds": seconds,
                "summary": f"无法启动子进程：{exc}", "stderr_tail": [], "stdout": "",
                "stderr": str(exc)}

    seconds = time.perf_counter() - started
    stdout, stderr = completed.stdout or "", completed.stderr or ""
    if verbose:
        print("  --- stdout ---")
        for line in stdout.rstrip().split("\n"):
            print(f"    {line}")
        if stderr.strip():
            print("  --- stderr ---")
            for line in stderr.rstrip().split("\n"):
                print(f"    {line}")

    passed = completed.returncode == 0
    # 只认**最后一行**做摘要：示例最后一行通常是小结，比第一行长。
    summary = last_non_empty(stdout) if passed else f"退出码 {completed.returncode}"
    if verbose or not passed:
        if stderr.strip() and not verbose:
            print("  --- stderr（最后几行）---")
            for line in tail_lines(stderr, tail):
                print(f"    {line}")
    return {"name": name, "status": "PASS" if passed else "FAIL",
            "exit_code": completed.returncode, "seconds": seconds, "summary": summary,
            "stderr_tail": tail_lines(stderr, tail), "stdout": stdout, "stderr": stderr}


def print_table(rows: Sequence[dict[str, object]]) -> None:
    """打印 PASS/FAIL 表。列宽固定，中文名照样对齐（按显示宽度算）。"""
    print()
    print(f"  {'示例':<34}{'结果':<10}{'退出码':<10}{'耗时':<12}备注")
    print("  " + "-" * (WIDTH - 2))
    for row in rows:
        exit_code = row["exit_code"]
        exit_text = "-" if exit_code is None else str(exit_code)
        seconds = float(row["seconds"])  # type: ignore[arg-type]
        print(f"  {str(row['name']):<34}{str(row['status']):<10}{exit_text:<10}"
              f"{seconds:>7.2f}s    {str(row['summary'])[:40]}")


def main(argv: Sequence[str] | None = None) -> int:
    """跑完所有示例，返回 0（全 PASS）或 1（有 FAIL/MISSING/TIMEOUT）。"""
    args = parse_args(argv)
    targets = select_targets(args.only)

    title("liteagent 示例离线冒烟：run_all_examples.py")
    note("""
每个示例都以 `python3 examples/<name> --offline` 的形式跑在**子进程**里：
离线（ScriptedLLM / echo）、零网络、零 API key。退出码 0 = PASS。
`--offline` 是默认行为，写在这里只是为了让命令长得和 README 一致。
    """)
    if args.only:
        note(f"--only {args.only!r} -> 实际要跑 {len(targets)} 个：{targets}")

    rows: list[dict[str, object]] = []
    for name in targets:
        title(f"跑 {name}")
        rows.append(run_one(name, timeout=args.timeout, verbose=args.verbose,
                            tail=args.tail))

    title("汇总")
    print_table(rows)

    passed = [row for row in rows if row["status"] == "PASS"]
    failed = [row for row in rows if row["status"] == "FAIL"]
    missing = [row for row in rows if row["status"] == "MISSING"]
    timed_out = [row for row in rows if row["status"] == "TIMEOUT"]

    total = sum(float(row["seconds"]) for row in rows)  # type: ignore[arg-type]
    print()
    print(f"  共 {len(rows)} 个示例：PASS {len(passed)}，FAIL {len(failed)}，"
          f"TIMEOUT {len(timed_out)}，MISSING {len(missing)}；总耗时 {total:.2f}s")

    if missing:
        note("缺失的示例（文件不存在）：" + ", ".join(str(row["name"]) for row in missing))
    if failed or timed_out:
        note("失败的示例（stderr 最后几行见上）："
             + ", ".join(str(row["name"]) for row in failed + timed_out))

    # 只要有一个不是 PASS 就返回非 0：MISSING 也算失败 ——
    # "示例清单不完整"和"示例跑挂了"都是这份交付物的真实缺陷。
    if failed or timed_out or missing:
        print()
        print("  ==> 结果：FAIL")
        return 1
    print()
    print("  ==> 结果：ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
