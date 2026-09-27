from __future__ import annotations

# =============================================================================
# examples/07_code_assistant.py —— 代码助手场景的端到端验证（旗舰示例）
# =============================================================================
#
# 这个示例回答：**liteagent 能不能真的干活**。
#
# 它在一个 `tempfile.TemporaryDirectory()` 沙箱里造出一个"有 bug 的 Python 项目"，
# 然后让一个 Agent 用内置工具完成一条真实的任务链：
#
#     read_file（读源码） -> read_file（读测试） -> write_file（改文件）
#       -> python_eval（算式快检） -> run_tests（子进程跑测试验证）
#
# 关键点（也是面试可以讲的）：
#
#   1. **工具是真的在执行**，不是 mock：write_file 真的写进沙箱，run_tests 真的
#      起一个子进程跑 unittest。结尾断言"文件内容确实变了"且"测试确实通过"。
#   2. **沙箱是硬约束**：所有路径都过 `PathSandbox`，`..` / 绝对路径 / symlink 逃逸
#      一律抛 `SandboxViolationError`。写路径永远不许越界，即使开了只读放宽。
#   3. **离线可复现**：默认用 `ScriptedLLM` 把"模型该做的决策"脚本化，整条链路确定性
#      可重放，不需要 API key、不联网。在线模式（`--provider openai`）从环境变量读 key，
#      没有 key 就打印提示并**优雅退回离线**，绝不崩。
#   4. **可观测性**：全程用 `TraceRecorder` 收集事件，结尾用 `trace_stats` 打印
#      步数 / 工具调用次数 / token 用量 / 延迟。这是"框架在生产里能被运维"的证据。
#
# 跑法：
#   python3 examples/07_code_assistant.py --offline          # 默认，离线确定性
#   python3 examples/07_code_assistant.py                    # 同上（provider 默认 echo）
#   python3 examples/07_code_assistant.py --provider openai  # 有 KEY 时走真实模型
#
# 注意：每个 .py 的第一行必须是 `from __future__ import annotations`（仓库冻结约定），
# 所以本文件用 `#` 注释而不是模块 docstring 来写说明。

import argparse
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 让示例在"源码树里直接跑"与"pip install -e . 之后跑"两种情况下都能 import
# ---------------------------------------------------------------------------
# `python3 examples/07_code_assistant.py` 的 sys.path[0] 是 examples/ 而非仓库根，
# 不补这一行 `import liteagent` 会直接 ModuleNotFoundError。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent import (  # noqa: E402  （必须在 sys.path 调整之后再 import）
    Agent,
    AgentConfig,
    LLMConfig,
    ScriptedLLM,
    ScriptedResponse,
    ToolRegistry,
    TraceRecorder,
    build_llm,
    trace_stats,
)
from liteagent.errors import ConfigError  # noqa: E402
# 内置工具的 factory 与沙箱类不在 `liteagent` 顶层 `__all__` 里（顶层只导出装配入口
# `register_all`），所以按"它们定义在哪个模块"精确 import —— 这也是读示例的人需要知道
# 的源码地图。
from liteagent.tools.builtin.code import make_code_tools  # noqa: E402
from liteagent.tools.builtin.files import PathSandbox, make_file_tools  # noqa: E402
from liteagent.tools.builtin.shell import make_shell_tools  # noqa: E402
# `as_llm_callback` 是"低层字符串事件 -> TraceEvent"的**唯一**适配器（§2.7）。
from liteagent.agent import as_llm_callback  # noqa: E402

WIDTH = 78


def hr(title: str) -> None:
    """分节标题。"""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


# ===========================================================================
# 1. 沙箱：一个"有 bug 的 Python 小项目"
# ===========================================================================

# 仓库里要放两个文件：一个待修的模块 + 一个测试。测试文件先写死"正确行为"，
# 这样它既是修复目标（跑不过就是 bug），也是验收标准（改完必须跑过）。
TARGET_FILE = "calculator.py"
TEST_FILE = "tests/test_calculator.py"

#: 修复前的版本：`add` 把加法写成了减法（一个真实、单点、可验证的 bug）。
#: 另外 `mean` 对空列表会 ZeroDivisionError —— 这次只修 add，保持任务聚焦。
BUGGY_CALCULATOR = '''\
"""A tiny calculator module used by the code-assistant example."""


def add(a, b):
    # BUG: 加法被写成了减法。
    return a - b


def mean(values):
    return sum(values) / len(values)
'''

#: 修复后的版本。这也是 ScriptedLLM 在 `write_file` 里要"生成"的内容 ——
#: 离线模式下模型不会真的写代码，但**工具会真的把这段内容写进磁盘**，链路是真的。
FIXED_CALCULATOR = '''\
"""A tiny calculator module used by the code-assistant example."""


def add(a, b):
    return a + b


def mean(values):
    return sum(values) / len(values)


def safe_divide(a, b):
    """返回 a / b；b 为 0 时返回 None，而不是抛 ZeroDivisionError。"""
    if b == 0:
        return None
    return a / b
'''

#: 验收测试。注意 `safe_divide` 在测试方法**内部** import：它在修复前还不存在，
#: 放在模块顶部会让整个测试模块 ImportError（报错噪音大），放方法里则得到一条
#: 干净的"这条测试失败"的信号。
TEST_SOURCE = '''\
import unittest

from calculator import add, mean


class CalculatorTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)

    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_safe_divide(self):
        # 这是本次要新增的功能；还没实现时这条会失败（而不是让整个模块 import 崩掉）。
        from calculator import safe_divide

        self.assertEqual(safe_divide(6, 3), 2)
        self.assertIsNone(safe_divide(1, 0))


if __name__ == "__main__":
    unittest.main()
'''


def seed_sandbox(root: Path) -> None:
    """在沙箱里造出"修复前"的项目（这一步是测试夹具，不是 Agent 干的活）。"""
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / TARGET_FILE).write_text(BUGGY_CALCULATOR, encoding="utf-8")
    (root / TEST_FILE).write_text(TEST_SOURCE, encoding="utf-8")


# ===========================================================================
# 2. 工具装配：内置工具是真函数，不是 mock
# ===========================================================================

def build_registry(sandbox: PathSandbox) -> ToolRegistry:
    """把内置工具注册进一个全新的 ToolRegistry。

    这里刻意**直接调用各 factory**（而不是 `register_all(include=[...])`），
    因为示例要说清"组是哪个 factory 造的"：

      - `make_file_tools(sandbox)`  -> read_file / write_file / list_dir / search_files / delete_file
      - `make_shell_tools(allow_shell=False, sandbox=...)` -> run_shell（默认禁用）
      - `make_code_tools(sandbox)`  -> python_eval / python_exec / run_tests

    `sandbox` 通过**闭包**注入到每个工具里：模型看到的 schema 里**没有**这个参数，
    它也就无法把路径指到沙箱外（详见 liteagent/tools/builtin/files.py 的 PathSandbox）。
    """
    registry = ToolRegistry()
    for tool in make_file_tools(sandbox):
        registry.register(tool)
    # allow_shell=False：工具**依然可见**（模型知道有这个能力），但调用只会返回
    # 'shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)'。这是刻意的：
    # "看不见"会让模型反复猜"为什么没有 shell 工具"，"看得见但被拒"是一句可读的反馈。
    for tool in make_shell_tools(False, sandbox):
        registry.register(tool)
    for tool in make_code_tools(sandbox):
        registry.register(tool)
    return registry


def tool_text(tool: object, args: dict[str, object]) -> str:
    """同步调用一个工具并拿到文本结果。

    为什么需要它：内置工具的返回值有两种形态 —— 文件类返回 `ToolResult`（带
    `metadata['sandbox']`），而 `run_tests` / `python_eval` 直接返回 `str`。
    Agent 走 executor 时由 `executor._stringify` 统一，我们手动调用就要自己归一化。
    """
    out = tool.run(args)  # type: ignore[attr-defined]
    return out.content if hasattr(out, "content") else str(out)


# ===========================================================================
# 3. LLM 装配：离线脚本 / 在线真实 provider
# ===========================================================================

#: 各 provider 的默认模型（只是让示例"开箱能跑"，不追求最新）。
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-3-5-sonnet-latest",
}


def build_agent_prompt() -> str:
    """给 Agent 的指令：它必须自己决定用哪些工具、按什么顺序。"""
    return (
        f"{TARGET_FILE} 里的 add 计算结果不对，请读取源码定位并修复它；"
        f"顺便新增一个 safe_divide(a, b)：b 为 0 时返回 None。"
        f"改完用 run_tests 跑一下 {TEST_FILE} 所在的测试目录，确认全部通过。"
    )


def build_scripted_llm() -> ScriptedLLM:
    """离线：把"模型的决策"脚本化成 5 条响应，整条链路确定性可重放。

    每条 `ScriptedResponse.tool(name, arguments)` 就是模型的一次"我要调用这个工具"。
    注意 `arguments` 里没有任何沙箱路径魔法 —— 全是相对沙箱根的普通相对路径。
    """
    return ScriptedLLM(
        [
            # 第 1 步：读源码。content= 是模型的"思考"，会被打印成 Thought。
            # `delay_s`：模拟"模型推理耗时"，会走 sleep_fn，且 ScriptedLLM 把它
            # 如实记成 latency_ms = delay_s * 1000（这是 §2.8 唯一允许断言的延迟值）。
            ScriptedResponse.tool(
                "read_file",
                {"path": TARGET_FILE},
                content="先读源码，定位 add 的实现。",
                delay_s=0.02,
            ),
            # 第 2 步：读测试，确认"正确行为"的定义（测试即规格）。
            ScriptedResponse.tool(
                "read_file",
                {"path": TEST_FILE},
                content="再看测试期望：add(2,3) 应为 5，且需要一个 safe_divide。",
                delay_s=0.02,
            ),
            # 第 3 步：写回修复后的文件。这是这条链路上唯一有副作用的动作。
            ScriptedResponse.tool(
                "write_file",
                {"path": TARGET_FILE, "content": FIXED_CALCULATOR},
                content="add 的符号写反了（a-b 应为 a+b）；同时补上 safe_divide，写回原文件。",
                delay_s=0.02,
            ),
            # 第 4 步：先用 python_eval 做一次**受限表达式**快检。
            # 它是 AST 白名单求值（不是 eval、不是安全沙箱），适合"算一下看看对不对"
            # 这类无副作用的验证，不需要审批、也不起子进程。
            ScriptedResponse.tool(
                "python_eval",
                {"expression": "a + b", "variables_json": "{\"a\": 2, \"b\": 3}"},
                content="顺手用受限表达式求值确认加法语义：2 + 3 应当等于 5。",
                delay_s=0.02,
            ),
            # 第 5 步：跑测试验证（run_tests 会起子进程真的执行 unittest）。
            ScriptedResponse.tool(
                "run_tests",
                {"path": "tests"},
                content="改完必须验证：跑 tests/ 目录下的测试套件。",
                delay_s=0.02,
            ),
            # 第 6 步：收尾。没有 tool_calls -> finish_reason 自动变成 "stop" -> 循环终结。
            ScriptedResponse.text(
                "已修复 add（a - b -> a + b）并新增 safe_divide（除零返回 None）；"
                "tests/ 全部通过。",
                delay_s=0.02,
            ),
        ],
        model="scripted-code-assistant",
    )


def build_llm_for(args: argparse.Namespace) -> tuple[object, str]:
    """返回 `(llm, mode_label)`。

    优先级：`--offline` / `--provider echo` -> 离线；否则尝试真实 provider，
    但没有 API key（或缺 provider）时**打印清晰提示并退回离线** —— 示例永远不该崩。
    """
    provider = (args.provider or "echo").strip().lower()
    if args.offline or provider == "echo":
        return build_scripted_llm(), "offline (ScriptedLLM)"

    options: dict[str, object] = {"provider": provider}
    if args.model:
        options["model"] = args.model
    elif provider in DEFAULT_MODELS:
        options["model"] = DEFAULT_MODELS[provider]
    try:
        config = LLMConfig(**options)  # type: ignore[arg-type]
    except (TypeError, ConfigError) as exc:
        print(f"[fallback] 无法构造 LLMConfig(provider={provider!r})：{exc}")
        print("[fallback] 退回离线模式（ScriptedLLM）。")
        return build_scripted_llm(), "offline (fallback)"

    # ===== 关键的"优雅降级"：没有 key 就明说，然后继续跑离线 =====
    if not config.resolve_api_key():
        env_name = f"{provider.upper().replace('-', '_')}_API_KEY"
        print(f"[fallback] provider={provider!r} 没有可用的 API key"
              f"（查过 LITEAGENT_API_KEY 与 {env_name}）。")
        print("[fallback] 设置环境变量后重试，或直接加 --offline。")
        print("[fallback] 本次退回离线模式（ScriptedLLM），功能演示不受影响。")
        return build_scripted_llm(), "offline (fallback: no api key)"

    try:
        llm = build_llm(config)
    except Exception as exc:  # noqa: BLE001 - provider 未注册/构造失败都不能让示例崩
        print(f"[fallback] 构造 provider={provider!r} 失败：{exc!r}")
        print("[fallback] 退回离线模式（ScriptedLLM）。")
        return build_scripted_llm(), "offline (fallback)"

    print(f"[online] 使用真实 provider={provider!r} model={config.model!r}"
          "（这一步会联网；失败会编码进 AgentResult，不会抛异常）")
    return llm, f"online ({provider})"


# ===========================================================================
# 4. 轨迹打印：把 Thought / Action / Observation 讲清楚
# ===========================================================================

def _one_line(text: str, limit: int = 160) -> str:
    """压成一行并截断，便于在终端里纵向阅读。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def print_trajectory(result: object) -> None:
    """从 `AgentResult.state.messages` 还原"每一步想了什么、做了什么、看到什么"。

    为什么从 state.messages 而不是从 trace 事件里取：native 模式下 LLM_RESPONSE
    事件只带 `content_len`（事件 data 要 JSON 可序列化，放整段文本会让 trace 膨胀），
    而 state.messages 保留了**完整的对话转录**，正好是教学要展示的东西。
    """
    hr("Agent 轨迹：Thought -> Action -> Observation")
    state = getattr(result, "state", None)
    if state is None:  # pragma: no cover - AgentResult 一定会带 state
        print("(没有 state)")
        return

    step = 0
    for message in state.messages:
        role = message.role.value
        if role == "system":
            continue
        if role == "user":
            print(f"[User]        {_one_line(message.content)}")
            continue
        if role == "assistant":
            step += 1
            print()
            print(f"--- 第 {step} 步 ---")
            print(f"  Thought:    {_one_line(message.content) or '(无)'}")
            for call in message.tool_calls:
                args_text = json.dumps(call.arguments, ensure_ascii=False)
                print(f"  Action:     {call.name}({_one_line(args_text, 120)})")
            continue
        if role == "tool":
            print(f"  Observation: {_one_line(message.content, 200)}")
            continue
        print(f"[{role}] {_one_line(message.content)}")


# ===========================================================================
# 5. 主流程
# ===========================================================================

def run_demo(args: argparse.Namespace, root: Path) -> int:
    hr("步骤 0 / 造一个有 bug 的沙箱项目")
    sandbox = PathSandbox(root)
    seed_sandbox(root)
    print(f"沙箱根（PathSandbox.root，realpath）: {sandbox.root}")
    print(f"  写入 {TARGET_FILE}（含 bug）与 {TEST_FILE}（验收测试）")

    registry = build_registry(sandbox)
    print()
    print(f"注册了 {len(registry)} 个内置工具：")
    for name in registry.names():
        tool = registry.get(name)
        flags = []
        if getattr(tool.spec, "dangerous", False):
            flags.append("dangerous")
        if getattr(tool.spec, "requires_approval", False):
            flags.append("requires_approval")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  - {name}{suffix}")

    # ---- 先用工具直接跑一次测试：证明"现在确实是坏的" ----
    print()
    print("先手动跑一次测试（修复前应当失败）……")
    before_tests = tool_text(registry.get("run_tests"), {"path": "tests"})
    before_status = before_tests.splitlines()[0] if before_tests else ""
    print(f"  修复前: {before_status}")

    # ---- 装配 Agent ----
    trace_path = root / "trace.jsonl"
    # TraceRecorder 自带一个 CallbackManager + 内存收集器 + 可选 JSONL 落盘。
    # 它的 `.manager.emit` 本身是个普通可调用对象，正好能作为 Agent 的 callback 订阅进去
    # （回调协议接受"带 on_event 的对象"或"普通函数"两种形态）。
    recorder = TraceRecorder(trace_path)

    llm, mode_label = build_llm_for(args)
    # 【关键的一行】Agent 会替它自己造的 memory/executor 挂 on_event，但**不会**动
    # 调用方传进来的 llm —— llm 的所有权是调用方的（红线 11/15 的同一精神）。
    # 所以我们自己把"低层字符串事件"接上唯一适配器，否则 LLM_REQUEST / LLM_RESPONSE
    # 永远进不了 trace（trace_stats 的 llm_calls / llm_latency_ms 会一直是 0，
    # usage 只能靠 RUN_FINISHED 的汇总兜底）。
    llm.on_event = as_llm_callback(recorder.manager)  # type: ignore[attr-defined]

    agent = Agent(
        llm=llm,  # type: ignore[arg-type]
        tools=registry,
        config=AgentConfig(
            name="code-assistant",
            max_steps=max(1, int(args.max_steps)),
            temperature=0.0,       # 可复现：贪心解码
            max_total_tokens=200_000,
        ),
        callbacks=[recorder.manager.emit],
        description="在沙箱里读代码、改代码、跑测试的代码助手",
    )

    hr(f"步骤 1 / 让 Agent 干活（mode: {mode_label}）")
    prompt = build_agent_prompt()
    print(f"任务: {prompt}")
    print()
    started = time.perf_counter()
    result = agent.run(prompt)
    elapsed_s = time.perf_counter() - started

    print_trajectory(result)
    print()
    print(f"最终输出: {_one_line(result.output, 200)}")
    print(f"status={result.status.value}  steps={result.steps}  "
          f"tool_calls={len(result.tool_calls)}  wall_clock={elapsed_s:.3f}s")

    # ---- 验收：文件真的被改了，测试真的过了 ----
    hr("步骤 2 / 验收（硬断言：文件真的变了、测试真的过了）")
    final_source = (root / TARGET_FILE).read_text(encoding="utf-8")
    changed = final_source != BUGGY_CALCULATOR
    has_fix = "return a + b" in final_source and "return a - b" not in final_source
    has_feature = "def safe_divide" in final_source
    after_tests = tool_text(registry.get("run_tests"), {"path": "tests"})
    after_status = after_tests.splitlines()[0] if after_tests else ""
    tests_ok = "status: OK" in after_tests

    print(f"文件内容已改变        : {changed}")
    print(f"bug 已修复(a-b -> a+b): {has_fix}")
    print(f"新功能 safe_divide 存在: {has_feature}")
    print(f"修复前测试状态        : {before_status}")
    print(f"修复后测试状态        : {after_status}")
    print()
    print(f"修复后 {TARGET_FILE} 的最终内容：")
    for line in final_source.splitlines():
        print(f"  | {line}")

    if mode_label.startswith("offline"):
        # 离线模式是**确定性**的：脚本化的动作 + 真实的工具执行，结果必须成立。
        # 用 unittest 风格断言（而不是裸 assert）—— `python -O` 不会把它们抹掉。
        case = unittest.TestCase()
        case.assertTrue(changed, "沙箱里的文件内容应当真的变了")
        case.assertTrue(has_fix, "add 的 bug 应当被修复")
        case.assertTrue(has_feature, "safe_divide 应当被新增")
        case.assertTrue(tests_ok, f"修复后测试应当通过，实际: {after_status!r}")
        print()
        print("离线验收：4 条断言全部通过 ✓（文件真被改写、测试真在子进程里跑过）")
    else:
        # 在线模式的动作由真实模型决定，不能硬断言；如实报告即可。
        print()
        print("在线模式：动作由真实模型决定，这里只报告结果、不做硬断言。")
        if not tests_ok:
            print("  提示：模型没有让测试全绿，可加 --offline 看确定性版本。")

    # ---- 可观测性：trace 统计 ----
    hr("步骤 3 / 可观测性：TraceRecorder -> trace_stats(recorder.events)")
    stats = trace_stats(recorder.events)
    latency = stats.get("llm_latency_ms") or {}
    print(f"事件总数          : {len(recorder.events)}")
    print(f"runs              : {stats['runs']}")
    print(f"steps             : {stats['steps']}")
    print(f"llm_calls         : {stats['llm_calls']}")
    print(f"tool_calls        : {stats['tool_calls']}   (按 call_id 去重)")
    print(f"tool_failures     : {stats['tool_failures']}")
    print(f"retries           : {stats['retries']}")
    print(f"parse_errors      : {stats['parse_errors']}")
    print(f"nudges            : {stats['nudges']}")
    print(f"usage             : {stats['usage']}")
    print(f"cost_usd          : {stats['cost_usd']}")
    print(f"llm_latency_ms    : total={latency.get('total', 0.0):.1f}, "
          f"mean={latency.get('mean', 0.0):.1f}, "
          f"p50={latency.get('p50', 0.0):.1f}, p95={latency.get('p95', 0.0):.1f}")
    print(f"tool_latency_ms   : {stats['tool_latency_ms']}")
    print(f"errors            : {stats['errors']}")
    print()
    print(f"Agent 自身统计    : steps={result.steps}, "
          f"usage={result.usage.to_dict()}, duration={result.duration_ms:.1f}ms")
    print(f"trace 文件        : {trace_path}"
          f"（JSONL，{len(recorder.events)} 行，可 `liteagent trace` 直接读）")

    recorder.__exit__()  # 关掉 JSONL 句柄（不是 with 块时手动收尾）
    print()
    print("沙箱是 tempfile.TemporaryDirectory()，main() 返回后会被自动清理 ——")
    print("所以这个示例永远碰不到仓库或用户 home 下的真实文件（PathSandbox 是第二重保险）。")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行。

    `--offline` 是**默认模式**的显式开关：守门测试与读者都靠它确认"这次没有联网"。
    """
    parser = argparse.ArgumentParser(
        description="liteagent 代码助手示例：在沙箱里读代码 -> 改代码 -> 跑测试",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="强制离线（默认行为；用 ScriptedLLM 脚本化决策，无需 API key、不联网）",
    )
    parser.add_argument(
        "--provider", default="echo",
        help="LLM provider：echo（默认，离线）| openai | anthropic | deepseek | ...",
    )
    parser.add_argument("--model", default=None, help="覆盖模型名（在线模式用）")
    parser.add_argument("--max-steps", type=int, default=8, help="Agent 最大轮数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("liteagent 代码助手示例 —— 沙箱 / 真实工具 / 可观测性")
    print(f"cwd={os.getcwd()}")
    # TemporaryDirectory：沙箱根是一次性的临时目录，退出即清理，
    # 绝不会碰到仓库或用户 home 下的真实文件（配合 PathSandbox 的双重保险）。
    with tempfile.TemporaryDirectory(prefix="liteagent-code-assistant-") as tmp:
        return run_demo(args, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
