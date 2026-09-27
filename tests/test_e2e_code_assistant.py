from __future__ import annotations

# =============================================================================
# tests/test_e2e_code_assistant.py —— 端到端：代码助手场景（§12 line 5135）
# =============================================================================
#
# 这个文件回答的是"框架能不能真的干活"，而不是"某个函数返回值对不对"。它把一个
# **代码助手**的最小闭环拼起来跑一遍：
#
#     读文件（read_file） -> 搜索（search_files） -> 写文件（write_file）
#
# 三条刻意的选择（面试可讲）：
#
#   1. **不 mock 工具**：`write_file` 真的写进 `tempfile.TemporaryDirectory()` 里的磁盘。
#      断言是"把文件重新读回来逐字比较"，而不是"检查工具被调用过" —— 后者只能证明
#      "我们调了一个叫 write_file 的东西"。同理 `search_files` 的 Observation 里必须
#      真的出现磁盘上那行源码（证明它真的扫了文件，不是回了句空话）。
#   2. **不用真模型**：决策由 `ScriptedLLM` 脚本化，因此整条链路确定性可重放、零网络、
#      零 API key。换真模型就断不了言了 —— "离线可复现"是端到端测试能进 CI 的前提。
#   3. **trace 事件用 `collect_events` 收全**：它订阅的是 `CallbackManager`，因此
#      Agent 自己发的事件与低层（llm/tools/memory）经 `as_llm_callback` 适配的事件
#      都在同一个列表里，能断言**顺序**与**完整性**，而不只是"某事件出现过"。
#
# 最后再加两条对 `examples/07_code_assistant.py` 的检查：核心流程可被 import（拿到
# `build_registry` / `build_scripted_llm` 等入口真装配一次），以及子进程 `--offline`
# 退出码为 0（那是"读者照着 README 跑"的那条路径）。

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

from liteagent import Agent, AgentConfig, ScriptedLLM, ScriptedResponse, ToolRegistry
from liteagent.agent import EventType, as_llm_callback
from liteagent.agent.state import AgentStatus
from liteagent.tools.builtin.files import PathSandbox, make_file_tools
from tests.helpers import assert_no_error_events, collect_events

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_07 = REPO_ROOT / "examples" / "07_code_assistant.py"

#: 沙箱里的"有 bug 的模块"。`add` 写成了减法；这正是要被修掉的那一行。
BUGGY_SOURCE = (
    "def add(a, b):\n"
    "    return a - b\n"
    "\n"
    "\n"
    "def mean(values):\n"
    "    return sum(values) / len(values)\n"
)

#: Agent 要写回的"修复后"源码。逐字比较用的就是它。
FIXED_SOURCE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def mean(values):\n"
    "    return sum(values) / len(values)\n"
)

TARGET_FILE = "calculator.py"

#: 给 Agent 的任务描述（内容不影响脚本化的决策，但让测试读起来是一个真实任务）。
TASK_PROMPT = (
    f"{TARGET_FILE} 里的 add 结果不对：读源码定位，搜索确认调用点，然后写回修复版。"
)

#: `ScriptedLLM` 在没给 usage 时用的固定常量（§6.6 规则 3）：每次调用 10 + 5 = 15。
USAGE_PER_CALL = 15
SCRIPTED_CALLS = 4

#: 子进程跑示例的超时（与 `test_examples_offline.py` 保持一致）。
SUBPROCESS_TIMEOUT_S = 60


def _clean_env() -> dict[str, str]:
    """剥掉 API key 并钉死 UTF-8（理由见 `test_examples_offline._clean_env`）。"""
    env = {
        key: value
        for key, value in os.environ.items()
        if not (key == "LITEAGENT_API_KEY" or key.endswith("_API_KEY"))
    }
    env.pop("LITEAGENT_ALLOW_SHELL", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _seed_sandbox(root: pathlib.Path) -> None:
    """造出"修复前"的项目：一个待修模块 + 一个待搜索的调用点。"""
    (root / TARGET_FILE).write_text(BUGGY_SOURCE, encoding="utf-8")
    # 第二个文件不是摆设：它让 `search_files` 有跨文件的搜索空间，"搜索"这一步
    # 才不是"只在一个文件里找"的退化情形。
    (root / "main.py").write_text(
        "from calculator import add\n\nprint(add(2, 40))\n", encoding="utf-8"
    )


def _build_registry(sandbox: PathSandbox) -> ToolRegistry:
    """注册内置文件工具（**真工具**，带闭包注入的沙箱）。"""
    registry = ToolRegistry()
    for tool_obj in make_file_tools(sandbox):
        registry.register(tool_obj)
    return registry


def _build_scripted_llm() -> ScriptedLLM:
    """把"模型的决策"脚本化：读文件 -> 搜索 -> 写文件 -> 收尾。"""
    return ScriptedLLM(
        [
            ScriptedResponse.tool(
                "read_file", {"path": TARGET_FILE}, content="先读源码看看 add 怎么写。"
            ),
            ScriptedResponse.tool(
                "search_files",
                {"pattern": r"def add", "path": ".", "glob": "**/*"},
                content="搜索确认 add 在哪些文件里被引用。",
            ),
            ScriptedResponse.tool(
                "write_file",
                {"path": TARGET_FILE, "content": FIXED_SOURCE},
                content="把 a - b 改成 a + b，写回原文件。",
            ),
            ScriptedResponse.text("已修复 add：a - b -> a + b。", delay_s=0.0),
        ],
        model="scripted-code-assistant",
    )


class CodeAssistantEndToEndTests(unittest.TestCase):
    """一个 setUp 跑完整条链路，各用例断言它的不同侧面。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-e2e-code-")
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        _seed_sandbox(self.root)

        self.sandbox = PathSandbox(self.root)
        self.registry = _build_registry(self.sandbox)

        # ---- 事件收集：Agent 事件 + 低层字符串事件（经 as_llm_callback 适配）----
        self.manager, self.events = collect_events()
        self.llm = _build_scripted_llm()
        # Agent **不会**改调用方传入的 llm 的 on_event（llm 的所有权是调用方的），
        # 所以自己把低层事件接上唯一适配器，否则 LLM_REQUEST / LLM_RESPONSE 进不了 trace。
        self.llm.on_event = as_llm_callback(self.manager)

        self.agent = Agent(
            llm=self.llm,
            tools=self.registry,
            config=AgentConfig(name="code-assistant", max_steps=8, temperature=0.0),
            callbacks=[self.manager.emit],
            description="在沙箱里读代码、搜索、改代码的代码助手",
        )
        self.result = self.agent.run(TASK_PROMPT)

    # ------------------------------------------------------------------ 结果
    def test_run_finishes_and_rewrites_the_file_on_disk(self) -> None:
        """文件**真的**被写出且内容逐字正确 —— 这是端到端测试的核心断言。"""
        self.assertEqual(
            AgentStatus.FINISHED,
            self.result.status,
            f"应当 FINISHED，实际 {self.result.status.value}；"
            f"error={self.result.error!r}",
        )
        self.assertTrue(self.result.ok)
        self.assertIsNone(self.result.error)

        # 不看工具返回值，直接把磁盘上的文件重新读回来比较。
        written = (self.root / TARGET_FILE).read_text(encoding="utf-8")
        self.assertEqual(FIXED_SOURCE, written, "写出的文件内容必须与脚本给定的修复版逐字一致")
        self.assertNotIn("return a - b", written, "bug 那一行必须被替换掉")

        # 未被改动的兄弟文件保持原样（写操作没有误伤）。
        self.assertIn(
            "print(add(2, 40))", (self.root / "main.py").read_text(encoding="utf-8")
        )

    def test_flow_is_read_then_search_then_write(self) -> None:
        """动作顺序与"读 -> 搜 -> 写"的脚本一致，且每个工具都真的执行成功了。"""
        self.assertEqual(
            ["read_file", "search_files", "write_file"],
            [call.name for call in self.result.tool_calls],
        )
        self.assertEqual(3, len(self.result.tool_results))
        for tool_result in self.result.tool_results:
            self.assertTrue(tool_result.ok, f"工具执行失败：{tool_result.content!r}")

        # Observation 里必须出现磁盘上的真实内容 —— 证明工具真的干活了。
        read_observation = self.result.tool_results[0].content
        self.assertIn("return a - b", read_observation, "read_file 必须读到磁盘上的 bug 行")
        search_observation = self.result.tool_results[1].content
        self.assertIn(f"{TARGET_FILE}:1:", search_observation, "search_files 必须命中源码行")
        self.assertIn("searched", search_observation, "search_files 必须报告扫描了多少文件")
        self.assertEqual(
            str(self.sandbox.root), self.result.tool_results[2].metadata.get("sandbox")
        )

    def test_trace_events_are_complete_and_ordered(self) -> None:
        """`collect_events` 收到的 trace 事件齐全、首尾正确、无错误事件。"""
        types = [event.type for event in self.events]
        self.assertGreater(len(types), 0, "事件列表不应为空（collect_events 没接上？）")
        self.assertEqual(EventType.RUN_STARTED, types[0], "第一条事件必须是 run_started")
        self.assertEqual(EventType.RUN_FINISHED, types[-1], "最后一条事件必须是 run_finished")

        required = {
            EventType.RUN_STARTED,
            EventType.RUN_FINISHED,
            EventType.STEP_STARTED,
            EventType.STEP_FINISHED,
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
            EventType.TOOL_STARTED,
            EventType.TOOL_FINISHED,
        }
        missing = sorted(event_type.value for event_type in required - set(types))
        self.assertEqual([], missing, f"trace 里缺少这些事件：{missing}")

        # 计数与脚本对得上：4 次 LLM 调用、3 次工具调用、4 个 step。
        counts: dict[EventType, int] = {}
        for event in self.events:
            counts[event.type] = counts.get(event.type, 0) + 1
        self.assertEqual(SCRIPTED_CALLS, counts.get(EventType.LLM_REQUEST, 0))
        self.assertEqual(SCRIPTED_CALLS, counts.get(EventType.LLM_RESPONSE, 0))
        self.assertEqual(3, counts.get(EventType.TOOL_STARTED, 0))
        self.assertEqual(3, counts.get(EventType.TOOL_FINISHED, 0))

        finished_names = [
            event.data.get("tool_name")
            for event in self.events
            if event.type is EventType.TOOL_FINISHED
        ]
        self.assertEqual(["read_file", "search_files", "write_file"], finished_names)

        # 干得成的路径上不该有任何错误事件（错误事件集合见 §12.1 的 ERROR_EVENT_TYPES）。
        assert_no_error_events(self, self.events)

    def test_usage_is_accumulated_across_the_loop(self) -> None:
        """usage 在循环里逐次累计：4 次调用 × 每次 15 tokens。"""
        expected_total = USAGE_PER_CALL * SCRIPTED_CALLS
        self.assertEqual(SCRIPTED_CALLS, self.llm.call_count)
        self.assertEqual(expected_total, self.result.usage.total_tokens)
        self.assertEqual(
            expected_total,
            self.result.usage.prompt_tokens + self.result.usage.completion_tokens,
        )
        # 结果里的 usage 与状态机里的 usage 是同一份账（不是两条各记一半的账）。
        self.assertIsNotNone(self.result.state)
        self.assertEqual(
            self.result.usage.total_tokens, self.result.state.usage.total_tokens
        )
        self.assertEqual(SCRIPTED_CALLS, self.result.steps)
        self.assertEqual("已修复 add：a - b -> a + b。", self.result.output)

    def test_scripted_queue_is_fully_consumed(self) -> None:
        """脚本被恰好消费完：多调一次会抛 ScriptedExhaustedError，少调一次会在这里失败。"""
        self.llm.assert_exhausted()
        self.assertEqual(0, self.llm.remaining)


class Example07IntegrationTests(unittest.TestCase):
    """`examples/07_code_assistant.py` 的核心流程：能 import、能装配、能跑。"""

    def _load_example_module(self) -> types.ModuleType:
        """用 `importlib` 按**路径**加载示例模块（不执行 main，不碰 argv）。"""
        self.assertTrue(EXAMPLE_07.is_file(), f"示例不存在：{EXAMPLE_07}")
        spec = importlib.util.spec_from_file_location(
            "liteagent_example_07_code_assistant", EXAMPLE_07
        )
        self.assertIsNotNone(spec, "spec_from_file_location 返回了 None")
        self.assertIsNotNone(spec.loader, "示例没有可用的 loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_example_module_exposes_the_core_entrypoints(self) -> None:
        module = self._load_example_module()
        for name in (
            "main",
            "run_demo",
            "build_registry",
            "build_scripted_llm",
            "build_llm_for",
            "seed_sandbox",
            "build_agent_prompt",
            "TARGET_FILE",
            "TEST_FILE",
        ):
            self.assertTrue(
                hasattr(module, name), f"示例 07 缺少入口 {name!r}（核心流程不可复用）"
            )

    def test_example_core_flow_can_be_assembled_from_imported_pieces(self) -> None:
        """不跑 `main`，只用示例自己的零件装配一次：证明"核心流程"真的可被复用。"""
        module = self._load_example_module()
        with tempfile.TemporaryDirectory(prefix="liteagent-ex07-") as tmp:
            root = pathlib.Path(tmp)
            module.seed_sandbox(root)
            sandbox = PathSandbox(root)
            registry = module.build_registry(sandbox)

            names = set(registry.names())
            self.assertTrue(
                {"read_file", "write_file", "search_files", "run_tests", "python_eval"}
                <= names,
                f"示例 07 注册的内置工具不全：{sorted(names)}",
            )
            self.assertTrue(
                (root / module.TARGET_FILE).is_file(), "seed_sandbox 必须落盘待修文件"
            )

            scripted = module.build_scripted_llm()
            self.assertIsInstance(scripted, ScriptedLLM)
            # 示例的剧本是"5 个工具调用 + 1 条收尾文本"（见文件内注释）。
            self.assertEqual(6, scripted.remaining)

            agent = Agent(llm=scripted, tools=registry, config=AgentConfig(max_steps=8))
            self.assertIn("read_file", agent.tools.names())

    def test_example_runs_offline_in_a_subprocess(self) -> None:
        """真跑 `python3 examples/07_code_assistant.py --offline`，退出码 0。

        断言不止于退出码：示例自己会打印"离线验收"那段自证，这里钉住它，确保我们
        不是把"验收被 skip 掉"当成通过。
        """
        completed = subprocess.run(
            [sys.executable, str(EXAMPLE_07), "--offline"],
            cwd=str(REPO_ROOT),
            env=_clean_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        self.assertEqual(
            0,
            completed.returncode,
            f"示例 07 --offline 退出码应为 0，实际 {completed.returncode}\n"
            f"--- stderr 尾部 ---\n{completed.stderr[-2000:]}",
        )
        self.assertIn("离线验收：4 条断言全部通过", completed.stdout)
        self.assertIn("status=FINISHED", completed.stdout)
        self.assertNotIn("Traceback (most recent call last)", completed.stderr)


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
