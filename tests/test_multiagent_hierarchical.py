from __future__ import annotations

"""tests/test_multiagent_hierarchical.py —— §10.4 `multiagent/hierarchical.py` 的单元测试。

§12 第 5129 行要求的覆盖点（逐条对应到测试类）：

  * delegate 工具生成（名字/schema/extra_context/timeout_s is NO_TIMEOUT） -> DelegateToolTests
  * manager 一次委派后出最终答案                            -> ManagerRunTests
  * worker 失败被压缩为字符串回灌                          -> DelegateClosureTests
  * 环检测返回 refused 字符串（不抛）                       -> DelegateClosureTests
  * max_depth                                              -> DelegateClosureTests
  * budget 耗尽 refused                                    -> DelegateClosureTests
  * compress_subagent_output 头尾都保留                     -> CompressOutputTests
  * adecompose 正常与非法 JSON                              -> DecomposeTests / PlanParsingTests
  * contextvar 隔离                                        -> ContextVarTests
  * extra_context 不影响环检测/深度判定                     -> ExtraContextTests
  * share_memory 注入                                      -> ShareMemoryTests
  * arun_plan：3 个 subtask 两层依赖 / 顺序满足 depends_on /
    并发峰值 <= subagent_concurrency / 某个 subtask 失败不中断 -> PlanExecutionTests

设计取舍（很重要，解释为什么有两个基类）：

  * delegate 工具是**同步工具**，闭包内部走 `run_sync` -> 会新建一个 loop。
    所以"委派"这条路径的用例用**同步** `unittest.TestCase`（此时本线程没有运行中的
    loop，`run_sync` 正常工作，和真实 executor 在 worker 线程里的处境一致）。
  * `arun` / `arun_plan` 是 async 入口，跑在父 loop 里；manager 要触发委派就自己
    `run_in_executor` 把工具挪到线程里（这正是真实 `ToolExecutor` 做的事）。
  * 环检测/预算/深度这三种 refused **不碰 worker**，因此在同步用例里直接调
    `tool.run({...})` + 预先 set 好 `_CURRENT_CTX` 即可，零线程、零 loop。

不真睡、不联网；并发用 `await asyncio.sleep(0)` 制造确定性的交错。
"""

import asyncio
import contextvars
import json
import unittest
from typing import Any, Sequence

from liteagent.agent.callbacks import EventType
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import NO_TIMEOUT, TeamConfig
from liteagent.errors import (
    ConfigError,
    CycleDetectedError,
    DelegationError,
    MaxDepthExceededError,
    ReActParseError,
    ToolDefinitionError,
)
from liteagent.llm.message import Message
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from liteagent.multiagent import (
    Blackboard,
    DelegationContext,
    HierarchicalAgent,
    Plan,
    SubTask,
    compress_subagent_output,
)
from liteagent.multiagent.hierarchical import (
    DECOMPOSE_PROMPT_TEMPLATE,
    PLAN_JSON_SCHEMA,
    _CURRENT_CTX,
)
from liteagent.tools.registry import ToolRegistry
from liteagent.types import LLMResponse


# ======================================================================================
# 夹具
# ======================================================================================


class Tracker:
    """记录每个 worker 的 start/end 与并发峰值（`arun_plan` 的两条断言靠它）。"""

    def __init__(self) -> None:
        self.order: list[tuple[str, str]] = []
        self.active = 0
        self.peak = 0

    def start(self, name: str) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(("start", name))

    def end(self, name: str) -> None:
        self.active -= 1
        self.order.append(("end", name))

    def index(self, event: tuple[str, str]) -> int:
        return self.order.index(event)


class StubWorker:
    """最小 worker：`name` + `arun` + 可注入的 `memory`。"""

    def __init__(
        self,
        name: str,
        *,
        output: str = "worker output",
        status: AgentStatus = AgentStatus.FINISHED,
        error: Any = None,
        steps: int = 1,
        memory: Any = None,
        tracker: Tracker | None = None,
        yields: int = 2,
        raise_exc: BaseException | None = None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description if description is not None else f"{name} worker"
        self.output = output
        self.status = status
        self.error = error
        self.steps = steps
        self.memory = memory
        self.tracker = tracker
        self.yields = yields
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult:
        self.calls.append({"input": input, "state": state, "context": context})
        if self.tracker is not None:
            self.tracker.start(self.name)
        try:
            if self.raise_exc is not None:
                raise self.raise_exc
            for _ in range(self.yields):
                await asyncio.sleep(0)  # 确定性交错（不是真实等待）
            return AgentResult(
                output=self.output,
                status=self.status,
                steps=self.steps,
                error=self.error,
                state=state,
                agent_name=self.name,
            )
        finally:
            if self.tracker is not None:
                self.tracker.end(self.name)


class StubManager:
    """鸭子类型的 manager：`name` / `arun` / `tools` / `memory` / `llm`。

    可选 `delegate_to=<tool name>`：`arun` 会**真的**从合并后的注册表里取出委派工具，
    在**线程**里调用它（复刻 `ToolExecutor` 对同步工具的处理），拿到压缩字符串后
    拼成最终答案。这条路径同时覆盖 contextvar 注入与工具合并。
    """

    def __init__(
        self,
        name: str = "manager",
        *,
        delegate_to: str | None = None,
        output: str = "manager answer",
        status: AgentStatus = AgentStatus.FINISHED,
        error: Any = None,
        memory: Any = None,
        llm: Any = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.name = name
        self.tools = tools if tools is not None else ToolRegistry([])
        self.executor = None
        self.memory = memory
        self.llm = llm
        self._delegate_to = delegate_to
        self._output = output
        self.status = status
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def arun(self, input: str, *, state: AgentState | None = None,
                   **kwargs: Any) -> AgentResult:
        self.calls.append({"input": input, "state": state, "kwargs": dict(kwargs)})
        output = self._output
        if self._delegate_to is not None:
            tool = self.tools.get(self._delegate_to)
            loop = asyncio.get_running_loop()
            # 与真实 `ToolExecutor` 完全同构（§7.4.1 步骤 5.b）：`run_in_executor`
            # **不会**自动复制 context，必须显式 `copy_context().run` 包一层 ——
            # 否则 worker 线程里 `_CURRENT_CTX.get()` 是 None，委派上下文（栈/深度/预算）
            # 全部失效。这里照抄那一刻的语义，测试才有意义。
            ctx = contextvars.copy_context()
            text = await loop.run_in_executor(
                None, ctx.run, tool.run, {"task": "delegated task"}
            )
            output = f"{self._output}:{text}"
        return AgentResult(
            output=output,
            status=self.status,
            error=self.error,
            agent_name=self.name,
            state=state,
        )


class RecordingLLM:
    """记录收到的 messages 并返回固定 content 的假 LLM（`adecompose` 用）。"""

    name = "recording"

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[Sequence[Message]] = []

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        self.messages.append(list(messages))
        return LLMResponse(content=self.content)


def _context(*, stack: Sequence[str] = ("hierarchy",), depth: int = 0,
             budget: int = 5) -> DelegationContext:
    return DelegationContext(stack=list(stack), depth=depth, budget=budget)


def _run_with_context(tool: Any, args: dict[str, Any], ctx: DelegationContext) -> str:
    """在**同步**上下文里以 `ctx` 为准调用委派工具（不新建线程/loop）。"""
    token = _CURRENT_CTX.set(ctx)
    try:
        return tool.run(args)
    finally:
        _CURRENT_CTX.reset(token)


# ======================================================================================
# delegate 工具生成
# ======================================================================================


class DelegateToolTests(unittest.TestCase):
    def test_tool_name_description_and_schema(self) -> None:
        worker = StubWorker("researcher", description="finds facts")
        ha = HierarchicalAgent(StubManager(), [worker])

        tools = ha.delegate_tools()

        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.name, "delegate_to_researcher")
        self.assertEqual(tool.description, "finds facts")
        params = tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertEqual(sorted(params["properties"]), ["extra_context", "task"])
        self.assertEqual(params["properties"]["task"]["type"], "string")
        self.assertEqual(params["properties"]["extra_context"]["type"], "string")
        self.assertEqual(params["required"], ["task"])
        self.assertFalse(params["additionalProperties"])
        self.assertFalse(tool.is_async)

    def test_delegate_is_not_idempotent_and_has_no_timeout(self) -> None:
        ha = HierarchicalAgent(StubManager(), [StubWorker("w")])
        tool = ha.delegate_tools()[0]
        self.assertIs(tool.spec.timeout_s, NO_TIMEOUT)
        self.assertFalse(tool.spec.idempotent)

    def test_worker_descriptions_override_the_generated_text(self) -> None:
        ha = HierarchicalAgent(
            StubManager(),
            [StubWorker("w", description="from attribute")],
            worker_descriptions={"w": "from mapping"},
        )
        self.assertEqual(ha.delegate_tools()[0].description, "from mapping")

    def test_illegal_characters_are_sanitized(self) -> None:
        # 空格 / `/` 都是非法字符 -> `_`；`-` 也算（见冲突用例：'a-b' 与 'a_b' 撞名）
        ha = HierarchicalAgent(StubManager(), [StubWorker("a b/c")])
        self.assertEqual(ha.delegate_tools()[0].name, "delegate_to_a_b_c")

    def test_colliding_sanitized_names_raise_tool_definition_error(self) -> None:
        ha = HierarchicalAgent(StubManager(), [StubWorker("a-b"), StubWorker("a_b")])
        with self.assertRaises(ToolDefinitionError):
            ha.delegate_tools()

    def test_worker_without_name_is_rejected(self) -> None:
        ha = HierarchicalAgent(StubManager(), [StubWorker("")])
        with self.assertRaises(ConfigError):
            ha.delegate_tools()

    def test_zero_subagent_concurrency_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ConfigError):
            HierarchicalAgent(
                StubManager(), [StubWorker("w")],
                config=TeamConfig(subagent_concurrency=0),
            )

    def test_worker_names_preserve_order(self) -> None:
        ha = HierarchicalAgent(StubManager(), [StubWorker("b"), StubWorker("a")])
        self.assertEqual(ha.worker_names(), ["b", "a"])


# ======================================================================================
# 委派闭包（同步路径）
# ======================================================================================


class DelegateClosureTests(unittest.TestCase):
    def _tool(self, worker: StubWorker, *, config: TeamConfig | None = None,
              manager: StubManager | None = None) -> tuple[HierarchicalAgent, Any]:
        ha = HierarchicalAgent(manager or StubManager(), [worker], config=config)
        return ha, ha.delegate_tools()[0]

    def test_successful_delegation_returns_compressed_string_and_archives(self) -> None:
        worker = StubWorker("w", output="worker says hi", steps=2)
        ha, tool = self._tool(worker)

        text = _run_with_context(tool, {"task": "do it"}, _context())

        self.assertIn("[worker w | status=FINISHED | steps=2]", text)
        self.assertIn("worker says hi", text)
        self.assertEqual(worker.calls[0]["input"], "do it")
        child_state = worker.calls[0]["state"]
        self.assertEqual(child_state.scratchpad["delegation"].stack, ["hierarchy", "w"])
        self.assertEqual(
            child_state.scratchpad["subagent_results"]["w:1"], "worker says hi"
        )
        self.assertEqual(ha.blackboard.keys(), ["subagent:w:1"])
        entry = ha.blackboard.read_entry("subagent:w:1")
        self.assertEqual(entry.author, "w")
        self.assertEqual(entry.tags, ("subagent",))

    def test_archive_sequence_increments_per_delegation(self) -> None:
        worker = StubWorker("w", output="o")
        ha, tool = self._tool(worker)
        _run_with_context(tool, {"task": "a"}, _context())
        _run_with_context(tool, {"task": "b"}, _context())
        self.assertEqual(ha.blackboard.keys(), ["subagent:w:1", "subagent:w:2"])

    def test_extra_context_is_appended_to_the_worker_input(self) -> None:
        worker = StubWorker("w", output="o")
        _, tool = self._tool(worker)
        _run_with_context(
            tool, {"task": "do it", "extra_context": "background=X"}, _context()
        )
        self.assertEqual(worker.calls[0]["input"], "do it\n\nbackground=X")

    def test_failed_worker_is_returned_as_a_string_not_raised(self) -> None:
        worker = StubWorker(
            "w", output="half answer", status=AgentStatus.FAILED,
            error=DelegationError(from_agent="up", to_agent="w"),
        )
        ha, tool = self._tool(worker)

        text = _run_with_context(tool, {"task": "t"}, _context())

        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("[worker w | status=FAILED"), msg=text)
        self.assertIn("error=", text)
        self.assertIn("half answer", text)  # 失败路径的正文同样回灌
        self.assertTrue(ha.blackboard.read("subagent:w:1").startswith("[worker w"))

    def test_worker_exception_is_returned_as_a_string_not_raised(self) -> None:
        worker = StubWorker("w", raise_exc=RuntimeError("worker exploded"))
        ha, tool = self._tool(worker)

        with self.assertLogs("liteagent.multiagent", level="ERROR"):
            text = _run_with_context(tool, {"task": "t"}, _context())

        self.assertIsInstance(text, str)
        self.assertIn("status=FAILED", text)
        self.assertIn("RuntimeError", text)
        self.assertIn("worker exploded", text)

    def test_cycle_detection_returns_a_refused_string(self) -> None:
        worker = StubWorker("w")
        ha, tool = self._tool(worker)
        events: list[Any] = []
        ha.callbacks.subscribe(events.append)

        text = _run_with_context(
            tool, {"task": "t"}, _context(stack=["hierarchy", "w"], depth=1)
        )

        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("[delegation refused:"), msg=text)
        self.assertIn("cycle detected", text)
        self.assertIn("stack=[hierarchy,w] -> w", text)
        self.assertEqual(worker.calls, [])  # 没有真的委派
        refused = [e for e in events if e.type == EventType.AGENT_DELEGATE]
        self.assertEqual(len(refused), 1)
        self.assertTrue(refused[0].data["refused"])
        self.assertEqual(refused[0].data["refused_reason"], "cycle")

    def test_delegate_cycle_check_is_unconditional(self) -> None:
        """§10.4 的冻结闭包伪代码里，环判定是 `ctx.would_cycle(worker)` —— **不带**
        `enable_cycle_detection` 开关（`_check_depth` 才带）。

        SPEC-AMBIGUITY：`TeamConfig.enable_cycle_detection` 在 §10 里只被 `_check_depth`
        引用，闭包那一节没提它。按"以规范为准 + 最保守做法"（安全侧：宁可多拒），
        这里断言闭包**始终**拒绝环，即使开关关掉。
        """
        worker = StubWorker("w", output="ran")
        ha, tool = self._tool(worker, config=TeamConfig(enable_cycle_detection=False))
        text = _run_with_context(
            tool, {"task": "t"}, _context(stack=["hierarchy", "w"], depth=1)
        )
        self.assertTrue(text.startswith("[delegation refused:"), msg=text)
        self.assertEqual(worker.calls, [])

    def test_delegate_depth_check_boundary_is_inclusive(self) -> None:
        """闭包用 `depth >= max_depth`（比 `_check_depth` 的 `>` 更严一格）。"""
        worker = StubWorker("w", output="ran")
        ha, tool = self._tool(worker)
        allowed = _run_with_context(
            tool, {"task": "t"}, _context(stack=["hierarchy"], depth=ha.config.max_depth - 1)
        )
        self.assertIn("ran", allowed)
        worker.calls.clear()
        refused = _run_with_context(
            tool, {"task": "t"}, _context(stack=["hierarchy"], depth=ha.config.max_depth)
        )
        self.assertTrue(refused.startswith("[delegation refused:"), msg=refused)

    def test_max_depth_refuses_with_a_string(self) -> None:
        worker = StubWorker("w")
        ha, tool = self._tool(worker)
        events: list[Any] = []
        ha.callbacks.subscribe(events.append)

        text = _run_with_context(
            tool, {"task": "t"},
            _context(stack=["hierarchy"], depth=ha.config.max_depth),
        )

        self.assertTrue(text.startswith("[delegation refused:"), msg=text)
        self.assertEqual(worker.calls, [])
        refused = [e for e in events if e.type == EventType.AGENT_DELEGATE]
        self.assertEqual(refused[0].data["refused_reason"], "depth")

    def test_budget_exhaustion_refuses_with_the_frozen_text(self) -> None:
        worker = StubWorker("w")
        ha, tool = self._tool(worker)
        events: list[Any] = []
        ha.callbacks.subscribe(events.append)

        text = _run_with_context(
            tool, {"task": "t"}, _context(stack=["hierarchy"], depth=0, budget=0)
        )

        self.assertEqual(text, "[delegation refused: budget exhausted]")
        self.assertEqual(worker.calls, [])
        refused = [e for e in events if e.type == EventType.AGENT_DELEGATE]
        self.assertEqual(refused[0].data["refused_reason"], "budget")

    def test_missing_context_falls_back_to_a_self_only_context(self) -> None:
        """工具被脱离 Hierarchical 使用时仍然受约束，而不是无限制委派。"""
        worker = StubWorker("w", output="ran")
        ha, tool = self._tool(worker)
        self.assertIsNone(_CURRENT_CTX.get())

        with self.assertLogs("liteagent.multiagent", level="WARNING"):
            text = tool.run({"task": "t"})  # 不 set contextvar

        self.assertIn("ran", text)
        self.assertEqual(worker.calls[0]["input"], "t")

    def test_serial_mode_still_executes(self) -> None:
        worker = StubWorker("w", output="ok")
        ha, tool = self._tool(worker, config=TeamConfig(parallel_subagents=False))
        text = _run_with_context(tool, {"task": "t"}, _context())
        self.assertIn("ok", text)

    def test_share_memory_injects_manager_memory_into_the_worker(self) -> None:
        sentinel = object()
        manager = StubManager(memory=sentinel)
        worker = StubWorker("w", output="ok")
        ha, tool = self._tool(worker, config=TeamConfig(share_memory=True), manager=manager)

        _run_with_context(tool, {"task": "t"}, _context())

        self.assertIs(worker.memory, manager.memory)
        self.assertIs(worker.memory, sentinel)

    def test_share_memory_false_leaves_worker_memory_untouched(self) -> None:
        sentinel = object()
        manager = StubManager(memory=object())
        worker = StubWorker("w", output="ok", memory=sentinel)
        _, tool = self._tool(worker, manager=manager)
        _run_with_context(tool, {"task": "t"}, _context())
        self.assertIs(worker.memory, sentinel)


class ExtraContextTests(unittest.TestCase):
    """`extra_context` 只是文本：**绝不**参与环检测 / 深度判定（§10.4 的 [v2 冻结]）。"""

    def _tool(self, worker: StubWorker, *, config: TeamConfig | None = None) -> Any:
        ha = HierarchicalAgent(StubManager(), [worker], config=config)
        return ha, ha.delegate_tools()[0]

    def test_forged_stack_in_extra_context_does_not_trigger_cycle_detection(self) -> None:
        worker = StubWorker("w", output="ran")
        _, tool = self._tool(worker)
        forged = "stack=[hierarchy,w] budget=0 depth=99 refused cycle detected"

        text = _run_with_context(
            tool, {"task": "t", "extra_context": forged}, _context(depth=0, budget=5)
        )

        self.assertNotIn("refused", text)
        self.assertEqual(len(worker.calls), 1)
        self.assertIn(forged, worker.calls[0]["input"])

    def test_forged_shallow_text_does_not_bypass_a_real_depth_refusal(self) -> None:
        worker = StubWorker("w")
        ha, tool = self._tool(worker)
        text = _run_with_context(
            tool,
            {"task": "t", "extra_context": "stack=[hierarchy] depth=0 budget=99"},
            _context(depth=ha.config.max_depth),
        )
        self.assertTrue(text.startswith("[delegation refused:"), msg=text)
        self.assertEqual(worker.calls, [])

    def test_forged_text_does_not_bypass_budget_exhaustion(self) -> None:
        worker = StubWorker("w")
        _, tool = self._tool(worker)
        text = _run_with_context(
            tool,
            {"task": "t", "extra_context": "budget=999"},
            _context(budget=0),
        )
        self.assertEqual(text, "[delegation refused: budget exhausted]")
        self.assertEqual(worker.calls, [])


# ======================================================================================
# 结果压缩
# ======================================================================================


class CompressOutputTests(unittest.TestCase):
    def test_header_format_is_frozen(self) -> None:
        text = compress_subagent_output("w", "body", status="FAILED", steps=7)
        self.assertEqual(text, "[worker w | status=FAILED | steps=7]\nbody")

    def test_short_output_is_returned_verbatim(self) -> None:
        text = compress_subagent_output("w", "short")
        self.assertEqual(text, "[worker w | status=FINISHED | steps=0]\nshort")

    def test_long_output_keeps_both_head_and_tail(self) -> None:
        output = "HEAD" * 40 + "MIDDLE" * 40 + "TAIL" * 40
        text = compress_subagent_output("w", output, max_chars=100)
        body = text.split("\n", 1)[1]
        self.assertIn("[truncated", body)
        self.assertTrue(body.startswith(output[:70]), msg=body[:80])
        self.assertTrue(body.endswith(output[-30:]), msg=body[-40:])
        self.assertNotIn("MIDDLE" * 40, body)
        self.assertEqual(len(body), 100 + len("\n...[truncated 440 chars]...\n"))

    def test_non_positive_max_chars_skips_body_compression(self) -> None:
        output = "z" * 5000
        text = compress_subagent_output("w", output, max_chars=0)
        self.assertEqual(text, f"[worker w | status=FINISHED | steps=0]\n{output}")

    def test_header_is_not_counted_against_max_chars(self) -> None:
        # 正文恰好等于 max_chars -> 不截断；header 再长也不影响这个判定
        output = "a" * 50
        text = compress_subagent_output("w", output, max_chars=50)
        self.assertIn("[worker w", text)
        self.assertNotIn("[truncated", text)
        self.assertTrue(text.endswith(output))


# ======================================================================================
# manager 主路径（`arun`）
# ======================================================================================


class ManagerRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_manager_delegates_once_then_produces_the_final_answer(self) -> None:
        worker = StubWorker("w", output="worker result")
        manager = StubManager(delegate_to="delegate_to_w", output="final")
        ha = HierarchicalAgent(manager, [worker])

        result = await ha.arun("do the thing")

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertTrue(result.output.startswith("final:"), msg=result.output)
        self.assertIn("worker result", result.output)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(result.agent_name, "hierarchy")
        delegations = result.metadata["delegations"]
        self.assertEqual(len(delegations), 1)
        self.assertEqual(delegations[0]["key"], "subagent:w:1")

    async def test_delegate_tools_are_installed_on_the_manager(self) -> None:
        worker = StubWorker("w")
        manager = StubManager(delegate_to="delegate_to_w")
        ha = HierarchicalAgent(manager, [worker])
        await ha.arun("x")
        self.assertIn("delegate_to_w", manager.tools.names())

    async def test_arun_is_idempotent_across_repeated_calls(self) -> None:
        """第二次 `arun` 不能再 merge 一次（否则重名 -> ToolDefinitionError）。"""
        worker = StubWorker("w")
        manager = StubManager(delegate_to="delegate_to_w")
        ha = HierarchicalAgent(manager, [worker])
        await ha.arun("first")
        await ha.arun("second")  # 不抛
        self.assertEqual(len(manager.calls), 2)
        self.assertEqual(manager.tools.names().count("delegate_to_w"), 1)

    async def test_user_supplied_delegate_tool_collides_loudly(self) -> None:
        worker = StubWorker("w")
        manager = StubManager()
        ha = HierarchicalAgent(manager, [worker])
        manager.tools = ToolRegistry(list(ha.delegate_tools()))  # 用户自己先放了一个同名的
        with self.assertRaises(ToolDefinitionError):
            await ha.arun("x")

    async def test_manager_failure_is_returned_when_propagate_is_return(self) -> None:
        manager = StubManager(
            output="", status=AgentStatus.FAILED,
            error=DelegationError(from_agent="x", to_agent="manager"),
        )
        ha = HierarchicalAgent(
            manager, [StubWorker("w")], config=TeamConfig(propagate_failure="return")
        )
        result = await ha.arun("x")
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.agent_name, "hierarchy")

    async def test_manager_failure_raises_when_propagate_is_raise(self) -> None:
        manager = StubManager(
            output="", status=AgentStatus.FAILED,
            error=DelegationError(from_agent="x", to_agent="manager"),
        )
        ha = HierarchicalAgent(
            manager, [StubWorker("w")], config=TeamConfig(propagate_failure="raise")
        )
        with self.assertRaises(DelegationError):
            await ha.arun("x")

    async def test_worker_failure_never_aborts_the_manager(self) -> None:
        worker = StubWorker(
            "w", output="", status=AgentStatus.FAILED, error=RuntimeError("nope")
        )
        manager = StubManager(delegate_to="delegate_to_w", output="recovered")
        ha = HierarchicalAgent(manager, [worker])

        result = await ha.arun("x")

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertIn("recovered", result.output)
        self.assertIn("status=FAILED", result.output)  # 失败作为观察回灌给 manager

    async def test_entry_depth_guard_rejects_an_exhausted_incoming_context(self) -> None:
        manager = StubManager()
        ha = HierarchicalAgent(manager, [StubWorker("w")])
        incoming = _context(budget=0)

        with self.assertRaises(MaxDepthExceededError):
            await ha.arun("x", context=incoming)


# ======================================================================================
# contextvar 隔离
# ======================================================================================


class ContextVarTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_sets_and_resets_the_contextvar(self) -> None:
        manager = StubManager()
        ha = HierarchicalAgent(manager, [StubWorker("w")])
        self.assertIsNone(_CURRENT_CTX.get())
        await ha.arun("x")
        self.assertIsNone(_CURRENT_CTX.get())  # try/finally 里 reset 干净

    async def test_worker_sees_a_child_of_the_run_context(self) -> None:
        worker = StubWorker("w")
        manager = StubManager(delegate_to="delegate_to_w")
        ha = HierarchicalAgent(manager, [worker])
        incoming = DelegationContext(
            stack=["hierarchy"], depth=0, budget=5, root_run_id="run_marker"
        )
        await ha.arun("x", context=incoming)
        child_ctx = worker.calls[0]["state"].scratchpad["delegation"]
        # root_run_id / budget 只可能来自 contextvar 里那份 ctx（fallback 是空 run_id），
        # 所以这组断言证明 worker 线程里真的读到了 arun 注入的上下文
        self.assertEqual(child_ctx.stack, ["hierarchy", "w"])
        self.assertEqual(child_ctx.depth, 1)
        self.assertEqual(child_ctx.budget, 4)
        self.assertEqual(child_ctx.root_run_id, "run_marker")

    async def test_concurrent_runs_keep_their_own_context(self) -> None:
        """`_CURRENT_CTX` 是 contextvar 而不是实例属性：manager 被并发复用时互不覆盖。"""
        worker_one = StubWorker("w1", output="one")
        worker_two = StubWorker("w2", output="two")
        manager_one = StubManager(delegate_to="delegate_to_w1")
        manager_two = StubManager(delegate_to="delegate_to_w2")
        ha_one = HierarchicalAgent(manager_one, [worker_one], name="h1")
        ha_two = HierarchicalAgent(manager_two, [worker_two], name="h2")

        ctx_one = DelegationContext(stack=["h1"], depth=0, budget=5, root_run_id="run_one")
        ctx_two = DelegationContext(stack=["h2"], depth=0, budget=5, root_run_id="run_two")

        await asyncio.gather(
            ha_one.arun("a", context=ctx_one),
            ha_two.arun("b", context=ctx_two),
        )

        first = worker_one.calls[0]["state"].scratchpad["delegation"]
        second = worker_two.calls[0]["state"].scratchpad["delegation"]
        self.assertEqual((first.stack, first.root_run_id), (["h1", "w1"], "run_one"))
        self.assertEqual((second.stack, second.root_run_id), (["h2", "w2"], "run_two"))
        self.assertIsNone(_CURRENT_CTX.get())


class ContextVarIsolationTests(unittest.TestCase):
    """`_CURRENT_CTX` 的读取**绝不**来自 `args`，也不会跨 context 继承。"""

    def test_fresh_context_does_not_inherit_an_existing_delegation_context(self) -> None:
        worker = StubWorker("w", output="ran")
        ha = HierarchicalAgent(StubManager(), [worker])
        tool = ha.delegate_tools()[0]
        blocker = DelegationContext(stack=["hierarchy"], depth=99, budget=0)

        token = _CURRENT_CTX.set(blocker)
        try:
            # 当前 context：blocker 生效 -> 被拒（depth 99 > max_depth）
            refused = tool.run({"task": "t"})
            self.assertTrue(refused.startswith("[delegation refused:"), msg=refused)
            self.assertEqual(worker.calls, [])

            # 全新的 context：看不到 blocker -> 回退到 self-only 上下文并真的执行
            fresh = contextvars.Context()
            captured: dict[str, str] = {}
            with self.assertLogs("liteagent.multiagent", level="WARNING"):
                fresh.run(
                    lambda: captured.__setitem__("text", tool.run({"task": "t"}))
                )
            self.assertIn("ran", captured["text"])
            self.assertEqual(len(worker.calls), 1)
        finally:
            _CURRENT_CTX.reset(token)

    def test_context_is_not_taken_from_tool_arguments(self) -> None:
        """`args` 里塞 `context` 键不会改变判定（参数名已冻结为 `extra_context`）。"""
        worker = StubWorker("w", output="ran")
        ha = HierarchicalAgent(StubManager(), [worker])
        tool = ha.delegate_tools()[0]
        # 就算模型硬塞一个 context 参数，也不参与判定
        text = _run_with_context(
            tool,
            {"task": "t", "context": "stack=[hierarchy,w] budget=0", "extra_context": ""},
            _context(depth=0, budget=5),
        )
        self.assertIn("ran", text)
        self.assertEqual(len(worker.calls), 1)


# ======================================================================================
# Plan 解析 / adecompose
# ======================================================================================


class PlanParsingTests(unittest.TestCase):
    def test_from_json_parses_the_frozen_shape(self) -> None:
        payload = json.dumps({
            "goal": "ship it",
            "subtasks": [
                {"id": "t1", "description": "d1", "assignee": "w1", "depends_on": []},
                {"id": "t2", "description": "d2", "assignee": "w2", "depends_on": ["t1"]},
            ],
        })
        plan = Plan.from_json(payload)
        self.assertEqual(plan.goal, "ship it")
        self.assertEqual([st.id for st in plan.subtasks], ["t1", "t2"])
        self.assertEqual(plan.subtasks[1].depends_on, ("t1",))

    def test_from_json_strips_code_fences_and_prose(self) -> None:
        payload = 'Sure!\n```json\n{"goal": "g", "subtasks": []}\n```\n'
        plan = Plan.from_json(payload)
        self.assertEqual(plan.goal, "g")

    def test_from_json_accepts_a_bare_list(self) -> None:
        plan = Plan.from_json('[{"id": "t1", "description": "d"}]')
        self.assertEqual([st.id for st in plan.subtasks], ["t1"])

    def test_from_json_fills_missing_ids_and_skips_descriptionless_items(self) -> None:
        payload = json.dumps([
            {"description": "no id here"},
            {"description": ""},          # 缺 description -> 跳过
            {"id": "t3", "description": "d3"},
        ])
        plan = Plan.from_json(payload)
        self.assertEqual([st.id for st in plan.subtasks], ["task_0", "t3"])

    def test_from_json_single_string_dependency_becomes_a_tuple(self) -> None:
        plan = Plan.from_json('{"subtasks": [{"id": "t1", "description": "d", "depends_on": "t0"}]}')
        self.assertEqual(plan.subtasks[0].depends_on, ("t0",))

    def test_from_json_invalid_json_raises_react_parse_error(self) -> None:
        with self.assertRaises(ReActParseError) as ctx:
            Plan.from_json("not json at all")
        self.assertEqual(ctx.exception.reason, "plan is not valid JSON")

    def test_render_lists_ids_assignees_and_dependencies(self) -> None:
        plan = Plan(goal="g", subtasks=[
            SubTask(id="t1", description="first", assignee="w1"),
            SubTask(id="t2", description="second", assignee="w2", depends_on=("t1",)),
        ])
        self.assertEqual(
            plan.render(), "- [t1] first (w1)\n- [t2] second (w2) <- t1"
        )

    def test_to_dict_round_trip(self) -> None:
        plan = Plan(goal="g", subtasks=[SubTask(id="t1", description="d", depends_on=("t0",))])
        restored = Plan.from_dict(plan.to_dict())
        self.assertEqual(restored.to_dict(), plan.to_dict())

    def test_prompt_template_and_schema_are_exposed(self) -> None:
        self.assertIn("{workers}", DECOMPOSE_PROMPT_TEMPLATE)
        self.assertIn("{max_subtasks}", DECOMPOSE_PROMPT_TEMPLATE)
        self.assertIn("{task}", DECOMPOSE_PROMPT_TEMPLATE)
        self.assertEqual(PLAN_JSON_SCHEMA["required"], ["subtasks"])


class DecomposeTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, llm: Any) -> HierarchicalAgent:
        return HierarchicalAgent(
            StubManager(llm=llm),
            [StubWorker("w1", description="researcher"), StubWorker("w2")],
        )

    async def test_adecompose_returns_a_plan(self) -> None:
        payload = json.dumps({
            "goal": "g",
            "subtasks": [{"id": "t1", "description": "d1", "assignee": "w1", "depends_on": []}],
        })
        ha = self._agent(RecordingLLM(payload))
        plan = await ha.adecompose("do something")
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, "g")
        self.assertEqual(plan.subtasks[0].assignee, "w1")

    async def test_adecompose_prompt_mentions_workers_and_limit(self) -> None:
        llm = RecordingLLM('{"goal": "g", "subtasks": []}')
        ha = self._agent(llm)
        await ha.adecompose("do something", max_subtasks=3)
        prompt = llm.messages[0][0].content
        self.assertIn("w1", prompt)
        self.assertIn("researcher", prompt)
        self.assertIn("at most 3", prompt)
        self.assertIn("do something", prompt)

    async def test_adecompose_invalid_json_raises_react_parse_error(self) -> None:
        ha = self._agent(RecordingLLM("totally not json"))
        with self.assertRaises(ReActParseError):
            await ha.adecompose("x")

    async def test_adecompose_llm_error_propagates(self) -> None:
        class BrokenLLM:
            name = "broken"

            async def achat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
                raise DelegationError(from_agent="llm", to_agent="self", message="llm down")

        ha = self._agent(BrokenLLM())
        with self.assertRaises(DelegationError):
            await ha.adecompose("x")

    async def test_adecompose_without_llm_raises_config_error(self) -> None:
        ha = self._agent(None)
        with self.assertRaises(ConfigError):
            await ha.adecompose("x")

    async def test_adecompose_rejects_non_positive_max_subtasks(self) -> None:
        ha = self._agent(RecordingLLM('{"subtasks": []}'))
        with self.assertRaises(ConfigError):
            await ha.adecompose("x", max_subtasks=0)

    async def test_adecompose_with_scripted_llm(self) -> None:
        """骨架也接受 `ScriptedLLM`（§6.6 的离线客户端）。"""
        payload = '{"goal": "g", "subtasks": [{"id": "t1", "description": "d"}]}'
        llm = ScriptedLLM([ScriptedResponse.text(payload)])
        ha = self._agent(llm)
        plan = await ha.adecompose("x")
        self.assertEqual([st.id for st in plan.subtasks], ["t1"])
        self.assertEqual(llm.call_count, 1)


# ======================================================================================
# 分层（`_layers`）
# ======================================================================================


class LayerTests(unittest.TestCase):
    def _agent(self) -> HierarchicalAgent:
        return HierarchicalAgent(StubManager(), [StubWorker("w1"), StubWorker("w2")])

    def test_two_layers_follow_dependencies(self) -> None:
        plan = Plan(subtasks=[
            SubTask(id="t1", description="d1"),
            SubTask(id="t2", description="d2"),
            SubTask(id="t3", description="d3", depends_on=("t1", "t2")),
        ])
        layers = self._agent()._layers(plan)
        self.assertEqual([[st.id for st in layer] for layer in layers], [["t1", "t2"], ["t3"]])

    def test_unknown_dependency_is_rejected(self) -> None:
        plan = Plan(subtasks=[SubTask(id="t1", description="d", depends_on=("ghost",))])
        with self.assertRaises(DelegationError):
            self._agent()._layers(plan)

    def test_cycle_is_rejected(self) -> None:
        plan = Plan(subtasks=[
            SubTask(id="t1", description="d1", depends_on=("t2",)),
            SubTask(id="t2", description="d2", depends_on=("t1",)),
        ])
        with self.assertRaises(CycleDetectedError):
            self._agent()._layers(plan)

    def test_duplicate_id_is_rejected(self) -> None:
        plan = Plan(subtasks=[
            SubTask(id="t1", description="d1"),
            SubTask(id="t1", description="d2"),
        ])
        with self.assertRaises(DelegationError):
            self._agent()._layers(plan)

    def test_empty_plan_has_no_layers(self) -> None:
        self.assertEqual(self._agent()._layers(Plan()), [])


# ======================================================================================
# arun_plan
# ======================================================================================


class PlanExecutionTests(unittest.IsolatedAsyncioTestCase):
    def _context(self) -> DelegationContext:
        return DelegationContext(stack=["hierarchy"], depth=0, budget=5)

    async def test_three_subtasks_two_layers_respect_depends_on(self) -> None:
        tracker = Tracker()
        workers = [
            StubWorker("w1", output="r1", tracker=tracker),
            StubWorker("w2", output="r2", tracker=tracker),
            StubWorker("w3", output="r3", tracker=tracker),
        ]
        ha = HierarchicalAgent(
            StubManager(), workers, config=TeamConfig(subagent_concurrency=2)
        )
        plan = Plan(goal="g", subtasks=[
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2"),
            SubTask(id="t3", description="d3", assignee="w3", depends_on=("t1", "t2")),
        ])

        result = await ha.arun_plan(plan, context=self._context())

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.agent_name, "hierarchy")
        # t3 必须在 t1 / t2 都**结束之后**才开始（depends_on 被真正执行）
        self.assertLess(tracker.index(("end", "w1")), tracker.index(("start", "w3")))
        self.assertLess(tracker.index(("end", "w2")), tracker.index(("start", "w3")))
        # 同层顺序不做保证，但两条 lay-1 任务都必须在 w3 之前
        self.assertLess(tracker.index(("start", "w1")), tracker.index(("start", "w3")))
        self.assertEqual([st.status for st in plan.subtasks], ["done", "done", "done"])
        self.assertEqual(len(result.metadata["steps"]), 3)
        self.assertEqual(result.metadata["plan"]["goal"], "g")

    async def test_blackboard_records_each_subtask(self) -> None:
        ha = HierarchicalAgent(
            StubManager(), [StubWorker("w1", output="OUT")]
        )
        plan = Plan(subtasks=[SubTask(id="t1", description="d1", assignee="w1")])
        await ha.arun_plan(plan, context=self._context())
        entry = ha.blackboard.read_entry("subtask:t1")
        self.assertIn("OUT", entry.value)
        self.assertEqual(entry.tags, ("subtask",))
        self.assertEqual(entry.author, "w1")

    async def test_concurrency_peak_never_exceeds_subagent_concurrency(self) -> None:
        tracker = Tracker()
        workers = [StubWorker(f"w{i}", tracker=tracker, yields=3) for i in range(3)]
        ha = HierarchicalAgent(
            StubManager(), workers, config=TeamConfig(subagent_concurrency=2)
        )
        plan = Plan(subtasks=[
            SubTask(id="t0", description="d0", assignee="w0"),
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2"),
        ])

        await ha.arun_plan(plan, context=self._context())

        self.assertLessEqual(tracker.peak, 2)
        self.assertEqual(tracker.peak, 2)  # 有并发发生，且被信号量卡在 2
        self.assertEqual(len(tracker.order), 6)

    async def test_concurrency_one_serializes_the_layer(self) -> None:
        tracker = Tracker()
        workers = [StubWorker(f"w{i}", tracker=tracker, yields=3) for i in range(3)]
        ha = HierarchicalAgent(
            StubManager(), workers, config=TeamConfig(subagent_concurrency=1)
        )
        plan = Plan(subtasks=[
            SubTask(id="t0", description="d0", assignee="w0"),
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2"),
        ])
        await ha.arun_plan(plan, context=self._context())
        self.assertEqual(tracker.peak, 1)

    async def test_failed_subtask_does_not_interrupt_the_plan(self) -> None:
        failing = StubWorker(
            "w1", output="", status=AgentStatus.FAILED,
            error=DelegationError(from_agent="w1", to_agent="self"),
        )
        downstream = StubWorker("w2", output="downstream ok")
        ha = HierarchicalAgent(StubManager(), [failing, downstream])
        plan = Plan(subtasks=[
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2", depends_on=("t1",)),
        ])

        result = await ha.arun_plan(plan, context=self._context())

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(plan.subtasks[0].status, "failed")
        self.assertEqual(plan.subtasks[1].status, "done")
        self.assertEqual(len(downstream.calls), 1)  # 后续层照跑
        self.assertIn("status=FAILED", plan.subtasks[0].result)
        self.assertTrue(
            ha.blackboard.read("subtask:t1").startswith("[worker w1 | status=FAILED")
        )
        statuses = [item["status"] for item in result.metadata["steps"]]
        self.assertEqual(statuses, ["failed", "done"])

    async def test_raising_subtask_does_not_interrupt_the_plan(self) -> None:
        raiser = StubWorker("w1", raise_exc=RuntimeError("boom inside worker"))
        downstream = StubWorker("w2", output="ok")
        ha = HierarchicalAgent(StubManager(), [raiser, downstream])
        plan = Plan(subtasks=[
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2"),
        ])

        with self.assertLogs("liteagent.multiagent", level="ERROR"):
            result = await ha.arun_plan(plan, context=self._context())

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(plan.subtasks[0].status, "failed")
        self.assertIn("RuntimeError", plan.subtasks[0].result)
        self.assertEqual(plan.subtasks[1].status, "done")

    async def test_serial_plan_mode_runs_in_order(self) -> None:
        tracker = Tracker()
        workers = [StubWorker(f"w{i}", tracker=tracker) for i in range(3)]
        ha = HierarchicalAgent(
            StubManager(), workers, config=TeamConfig(parallel_subagents=False)
        )
        plan = Plan(subtasks=[
            SubTask(id="t0", description="d0", assignee="w0"),
            SubTask(id="t1", description="d1", assignee="w1"),
            SubTask(id="t2", description="d2", assignee="w2"),
        ])
        await ha.arun_plan(plan, context=self._context())
        self.assertEqual(tracker.peak, 1)
        self.assertEqual(
            tracker.order,
            [("start", "w0"), ("end", "w0"), ("start", "w1"), ("end", "w1"),
             ("start", "w2"), ("end", "w2")],
        )

    async def test_unknown_assignee_falls_back_to_the_manager(self) -> None:
        manager = StubManager(output="manager handled it")
        ha = HierarchicalAgent(manager, [StubWorker("w1")])
        plan = Plan(subtasks=[
            SubTask(id="t1", description="do a thing", assignee="ghost"),
            SubTask(id="t2", description="no assignee"),
        ])

        with self.assertLogs("liteagent.multiagent", level="WARNING"):
            result = await ha.arun_plan(plan, context=self._context())

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(len(manager.calls), 2)
        self.assertEqual(manager.calls[0]["input"], "[subtask t1] do a thing")
        self.assertEqual(manager.calls[1]["input"], "[subtask t2] no assignee")
        self.assertEqual([st.status for st in plan.subtasks], ["done", "done"])

    async def test_plan_can_be_passed_through_arun(self) -> None:
        worker = StubWorker("w1", output="via plan")
        manager = StubManager()
        ha = HierarchicalAgent(manager, [worker])
        plan = Plan(subtasks=[SubTask(id="t1", description="d", assignee="w1")])

        result = await ha.arun("ignored input", plan=plan)

        self.assertEqual(manager.calls, [])  # plan 路径不经过 manager 的 ReAct 循环
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(result.metadata["plan"]["subtasks"][0]["id"], "t1")
        self.assertIn("via plan", result.output)

    async def test_empty_plan_finishes_without_calling_anyone(self) -> None:
        worker = StubWorker("w1")
        ha = HierarchicalAgent(StubManager(), [worker])
        result = await ha.arun_plan(Plan(), context=self._context())
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.metadata["steps"], [])
        self.assertEqual(worker.calls, [])

    async def test_empty_plan_is_not_silent(self) -> None:
        """[v3 回归] 空计划必须**留痕**，不能静默成功。

        §10.4 冻结了"计划跑完即 FINISHED"（失败在 subtask 层，而空计划没有 subtask），
        所以 AgentResult 的形状不能改；但一个字段名写错（例如模型把 `description`
        写成 `task`）就能让整份计划被 `Plan.from_json` 全部跳过，此时 run 会以
        FINISHED + output='results: (none)' 收尾，既无 RUN_FAILED、也无 error ——
        对"失败必须编码在 AgentResult 里"的契约来说是最坏的一类静默。这里断言
        **至少有一条 WARNING**（§13 红线 12：降级可以是设计，静默不行）。
        """
        worker = StubWorker("w1")
        ha = HierarchicalAgent(StubManager(), [worker])
        with self.assertLogs("liteagent.multiagent", level="WARNING") as captured:
            result = await ha.arun_plan(Plan(goal="g"), context=self._context())
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertTrue(
            any("no executable subtasks" in line for line in captured.output),
            msg=captured.output,
        )

    async def test_empty_plan_reports_zero_subtasks_in_metadata(self) -> None:
        """[v3 回归] 空计划必须在 `AgentResult.metadata` 里给出**机器可读**的信号。

        只有 WARNING 是不够的（上一条）：日志不是契约的一部分，调用方没法靠它做
        分支（"'计划跑完了'与'计划里一个子任务都没有'在返回值上完全一样"）。
        `metadata["plan"]["subtasks"] == []` 虽然也算信号，但要先钻进嵌套 dict 的
        长度里才看得见。这里把两个顶层字段钉死：

          * `subtasks` == 0 —— 可执行子任务的条数；
          * `empty_plan` is True —— 布尔快捷方式，供调用方直接告警/降级。

        顺带钉住反方向：非空计划的两个字段必须反映真实条数（`empty_plan is False`），
        否则"永远写 0"这种写法也能让本用例通过。
        """
        worker = StubWorker("w1")
        ha = HierarchicalAgent(StubManager(), [worker])
        empty = await ha.arun_plan(Plan(goal="g"), context=self._context())
        self.assertEqual(0, empty.metadata["subtasks"])
        self.assertIs(True, empty.metadata["empty_plan"])
        self.assertEqual([], empty.metadata["plan"]["subtasks"])

        one = await ha.arun_plan(
            Plan(goal="g", subtasks=[SubTask(id="t1", description="d", assignee="w1")]),
            context=self._context(),
        )
        self.assertEqual(1, one.metadata["subtasks"])
        self.assertIs(False, one.metadata["empty_plan"])


# ======================================================================================
# 黑板注入 / aclose
# ======================================================================================


class WiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_blackboard_can_be_injected(self) -> None:
        board = Blackboard()
        ha = HierarchicalAgent(StubManager(), [StubWorker("w")], blackboard=board)
        self.assertIs(ha.blackboard, board)

    async def test_share_memory_injection_in_plan_path(self) -> None:
        sentinel = object()
        manager = StubManager(memory=sentinel)
        worker = StubWorker("w1", output="ok")
        ha = HierarchicalAgent(
            manager, [worker], config=TeamConfig(share_memory=True)
        )
        plan = Plan(subtasks=[SubTask(id="t1", description="d", assignee="w1")])
        await ha.arun_plan(plan, context=DelegationContext(stack=["hierarchy"]))
        self.assertIs(worker.memory, manager.memory)

    async def test_aclose_closes_workers_and_manager(self) -> None:
        closed: list[str] = []

        class ClosableWorker(StubWorker):
            async def aclose(self) -> None:
                closed.append(self.name)

        manager = StubManager()
        workers = [ClosableWorker("w1"), ClosableWorker("w2")]
        ha = HierarchicalAgent(manager, workers)
        await ha.aclose()
        self.assertEqual(sorted(closed), ["w1", "w2"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
