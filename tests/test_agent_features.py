from __future__ import annotations

"""``Agent`` 的特性面测试：循环防护、回灌、流式、生命周期（§9.4）。

覆盖 §12 对 ``test_agent_features.py`` 冻结的覆盖点：

* 重复动作**四种策略**（off / nudge / fail / nudge_then_fail）；
* nudge 文本内容（§9.4.4 的冻结字面量）；
* 验证错误回灌；
* 未知工具名回灌含**可用工具列表**；
* 参数每次略变的重复调用被 nudge 或终止；
* ``astream`` 产出事件顺序 + **消费者提前 break 不泄漏 task**；
* ``describe`` / ``reset``；
* ``callbacks`` 抛错不影响主流程；
* **arun 不可重入**（并发两次 -> 一次 FAILED 且 error 提到 "already has a run in flight"）。

全部离线确定性：``ScriptedLLM`` 驱动，零网络、零真实 sleep。
"""

import asyncio
import unittest

from liteagent.agent.agent import Agent
from liteagent.agent.callbacks import EventType, TraceEvent, as_llm_callback
from liteagent.agent.state import AgentStatus
from liteagent.config import AgentConfig
from liteagent.errors import AgentError, MaxStepsExceededError, RepeatedActionError
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from liteagent.memory.manager import MemoryManager
from liteagent.memory.base import MemoryConfig
from tests.helpers import add, close_loop_bound_pools, echo, make_registry


def _build(
    llm: ScriptedLLM,
    *tools: object,
    config: AgentConfig | None = None,
) -> tuple[Agent, list[TraceEvent]]:
    agent = Agent(llm=llm, tools=make_registry(*tools), config=config)
    events: list[TraceEvent] = []
    agent.callbacks.subscribe(events.append)
    llm.on_event = as_llm_callback(agent.callbacks)
    return agent, events


def _repeating_llm() -> ScriptedLLM:
    """每次都要求调用**同一个** add(1,2) 的 LLM。"""
    return ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)


class RepeatActionPolicyTests(unittest.IsolatedAsyncioTestCase):
    """重复/无进展动作的四种策略（§9.4.1 步骤 4）。"""

    async def test_policy_off_runs_until_max_steps(self) -> None:
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=4, repeat_action_policy="off"),
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, MaxStepsExceededError)
        self.assertEqual(4, result.steps)
        self.assertEqual(0, len(agent.state.nudges))
        types = [e.type for e in events]
        self.assertNotIn(EventType.REPEAT_DETECTED, types)

    async def test_policy_fail_stops_without_nudging(self) -> None:
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="fail"),
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, RepeatedActionError)
        self.assertEqual(2, result.error.count)  # 第 2 次就超阈值
        self.assertEqual(0, len(agent.state.nudges))
        types = [e.type for e in events]
        self.assertEqual(1, types.count(EventType.REPEAT_DETECTED))
        self.assertNotIn(EventType.NUDGE, types)

    async def test_policy_nudge_then_fail(self) -> None:
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="nudge_then_fail"),
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, RepeatedActionError)
        types = [e.type for e in events]
        self.assertEqual(1, types.count(EventType.NUDGE))
        self.assertEqual(2, types.count(EventType.REPEAT_DETECTED))
        self.assertEqual(1, len(agent.state.nudges))
        # NUDGE 事件带 {text}（§2.7）
        nudge_events = [e for e in events if e.type is EventType.NUDGE]
        self.assertEqual({"text"}, set(nudge_events[0].data))
        self.assertEqual(agent.state.nudges[0], nudge_events[0].data["text"])

    async def test_policy_nudge_also_stops_after_second_detection(self) -> None:
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="nudge"),
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, RepeatedActionError)
        self.assertEqual(1, len(agent.state.nudges))

    async def test_repeat_detected_event_data_keys(self) -> None:
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="fail"),
        )
        await agent.arun("go")
        detected = [e for e in events if e.type is EventType.REPEAT_DETECTED][0]
        self.assertEqual({"action_key", "count"}, set(detected.data))
        self.assertEqual(2, detected.data["count"])
        self.assertTrue(detected.data["action_key"].startswith("add:"))


class NudgeTextTests(unittest.IsolatedAsyncioTestCase):
    """§9.4.4 的 nudge 字面量（冻结）。"""

    async def test_nudge_text_is_byte_exact(self) -> None:
        agent, _ = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="nudge_then_fail"),
        )
        await agent.arun("go")
        self.assertEqual(1, len(agent.state.nudges))
        self.assertEqual(
            "You already called add with these exact arguments 2 times.\n"
            "The previous result was:\n"
            "3\n"
            "Do not repeat it. Either use a different tool/arguments, or give your "
            "final answer now\n"
            'as "Thought: ...\\nFinal Answer: ...".',
            agent.state.nudges[0],
        )

    async def test_nudge_is_injected_as_user_nudge_message(self) -> None:
        agent, _ = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="nudge_then_fail"),
        )
        await agent.arun("go")
        nudges = [
            message
            for message in agent.state.messages
            if message.metadata.get("kind") == "nudge"
        ]
        self.assertEqual(1, len(nudges))
        self.assertEqual("user", nudges[0].role.value)
        self.assertEqual(agent.state.nudges[0], nudges[0].content)


class ToolFeedbackTests(unittest.IsolatedAsyncioTestCase):
    """工具失败时的回灌文案（验证错误 / 未知工具名）。"""

    async def test_validation_error_is_fed_back_to_the_model(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": "not-an-int", "b": "nope"}),
                ScriptedResponse.text("fixed"),
            ]
        )
        agent, events = _build(llm, add)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        failure = result.tool_results[0]
        self.assertFalse(failure.ok)
        self.assertEqual("ToolValidationError", failure.error_type)
        self.assertEqual("recoverable", failure.metadata["feedback_kind"])
        self.assertTrue(failure.content.startswith("ERROR("))
        # 回灌给模型的消息（native 模式 -> role=tool）
        tool_messages = [
            message for message in agent.state.messages if message.role.value == "tool"
        ]
        self.assertEqual(1, len(tool_messages))
        self.assertIn("ERROR(", tool_messages[0].content)
        self.assertIn("invalid arguments", tool_messages[0].content)
        # 工具层事件归属 §2.7：失败走 TOOL_ERROR，且带上 error_type 供观测
        tool_errors = [e for e in events if e.type is EventType.TOOL_ERROR]
        self.assertEqual(1, len(tool_errors))
        self.assertEqual("ToolValidationError", tool_errors[0].data["error_type"])

    async def test_unknown_tool_feedback_lists_available_tools(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("no_such_tool", {"x": 1}),
                ScriptedResponse.text("understood"),
            ]
        )
        agent, events = _build(llm, add, echo)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        failure = result.tool_results[0]
        self.assertFalse(failure.ok)
        self.assertEqual("ToolNotFoundError", failure.error_type)
        self.assertIn("no_such_tool", failure.content)
        self.assertIn("add", failure.content)
        self.assertIn("echo", failure.content)
        self.assertEqual(1, len([e for e in events if e.type is EventType.TOOL_ERROR]))


class VaryingArgumentRepeatTests(unittest.IsolatedAsyncioTestCase):
    """参数每次略变但观察结果完全相同 -> 无进展检测（§9.4.1 的 hit_digest）。"""

    async def test_identical_results_with_varying_args_get_nudged(self) -> None:
        # 四组不同参数都返回 3：canonical_key 每次都不同（命中不了 hit_key），
        # 但 observation 摘要完全相同 -> 只能靠 hit_digest 兜住。
        pairs = [(1, 2), (2, 1), (3, 0), (0, 3)]
        llm = ScriptedLLM(
            [ScriptedResponse.tool("add", {"a": a, "b": b}) for a, b in pairs], loop=True
        )
        agent, events = _build(
            llm,
            add,
            config=AgentConfig(max_steps=4, repeat_action_policy="nudge_then_fail"),
        )
        result = await agent.arun("go")

        types = [e.type for e in events]
        self.assertGreaterEqual(types.count(EventType.REPEAT_DETECTED), 2)
        self.assertGreaterEqual(types.count(EventType.NUDGE), 1)
        self.assertTrue(agent.state.nudges)
        self.assertIn(
            "returned identical results", agent.state.nudges[0]
        )
        # 终止必须发生（要么 RepeatedActionError，要么步数用尽）——不允许无限空转
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertLessEqual(result.steps, 4)

    async def test_same_args_repeat_hits_the_canonical_key_layer(self) -> None:
        """对照组：参数完全相同 -> 命中 hit_key（canonical_key 计数）。"""
        agent, events = _build(
            _repeating_llm(),
            add,
            config=AgentConfig(max_steps=5, repeat_action_policy="nudge_then_fail"),
        )
        await agent.arun("go")
        detected = [e for e in events if e.type is EventType.REPEAT_DETECTED][0]
        self.assertEqual(2, detected.data["count"])
        # 无进展后缀不该出现：命中的是 hit_key 而不是 hit_digest
        self.assertNotIn("identical results", agent.state.nudges[0])


class DescribeAndResetTests(unittest.IsolatedAsyncioTestCase):
    """自述与复位（§9.4）。"""

    async def test_describe_shape(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        agent, _ = _build(llm, add, echo)
        described = agent.describe()
        self.assertEqual(
            {"name", "description", "model", "tools", "mode", "max_steps"},
            set(described),
        )
        self.assertEqual("agent", described["name"])
        self.assertEqual("scripted-1", described["model"])
        self.assertEqual(["add", "echo"], described["tools"])
        self.assertEqual("native", described["mode"])
        self.assertEqual(10, described["max_steps"])

    async def test_describe_on_fresh_agent_reports_idle_state(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        agent, _ = _build(llm, add)
        self.assertIs(AgentStatus.IDLE, agent.state.status)
        self.assertEqual(0, agent.state.step)
        self.assertIsInstance(agent.describe(), dict)

    async def test_reset_clears_state_and_optionally_memory(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")], loop=True)
        agent, _ = _build(llm, add)
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertIs(agent.state, result.state)

        agent.reset()
        self.assertIs(AgentStatus.IDLE, agent.state.status)
        self.assertEqual(0, agent.state.step)
        self.assertEqual([], agent.state.messages)
        self.assertGreater(len(agent.memory.buffer), 0)  # 短期记忆还在

        agent.reset(clear_memory=True)  # 运行中的 loop 里退化为同步清 buffer，不抛
        self.assertIs(AgentStatus.IDLE, agent.state.status)
        self.assertEqual(0, len(agent.memory.buffer))


class AstreamTests(unittest.IsolatedAsyncioTestCase):
    """``astream`` 的事件顺序与"提前 break 不泄漏 task"（§9.4）。"""

    async def test_astream_yields_events_in_order(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("done"),
            ]
        )
        agent, _ = _build(llm, add)
        types: list[EventType] = []
        async for event in agent.astream("go"):
            types.append(event.type)

        self.assertIs(EventType.RUN_STARTED, types[0])
        self.assertIs(EventType.RUN_FINISHED, types[-1])
        self.assertIn(EventType.STEP_STARTED, types)
        self.assertIn(EventType.TOOL_STARTED, types)
        self.assertIn(EventType.TOOL_FINISHED, types)
        self.assertIn(EventType.LLM_REQUEST, types)
        self.assertLess(types.index(EventType.TOOL_STARTED), types.index(EventType.TOOL_FINISHED))

    async def test_early_break_does_not_leak_the_run_task(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")], loop=True)
        agent, _ = _build(llm, add, config=AgentConfig(max_steps=1))
        stream = agent.astream("go")
        seen: list[EventType] = []
        async for event in stream:
            seen.append(event.type)
            if event.type is EventType.STEP_STARTED:
                break
        self.assertEqual(
            [EventType.RUN_STARTED, EventType.MEMORY_WRITE, EventType.STEP_STARTED], seen
        )
        await stream.aclose()  # 触发 GeneratorExit -> astream 的 finally 取消 task

        # 守卫必须已释放：否则这里会返回 "already has a run in flight"
        self.assertFalse(agent._run_guard.locked())
        result = await agent.arun("again")
        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("done", result.output)

    async def test_early_break_stops_a_tool_using_run(self) -> None:
        """[v3 回归] 带**工具调用**的 run 在消费者 break 后必须真的停下。

        v2 的洞：``ToolExecutor._invoke`` 用 ``asyncio.wait_for`` 包超时，而 3.10 的
        实现在"内层 future 与调用方取消落在同一个 tick"时会 ``return fut.result()``，
        把取消**静默吞掉**（GH-86296 / bpo-42130，3.12 才重写）。于是 astream 的
        ``task.cancel()`` 发出去了、返回 True、却毫无效果 —— arun 继续跑完整个 ReAct
        循环、继续烧 LLM 调用、继续写记忆。默认 ``default_timeout_s=30`` 下必现，
        因为任何带工具调用的 agent 都走这条 wait_for 路径。

        这里用 ``max_steps=4`` + ``loop=True`` 把"没停下来"变成一个可判定的数字：
        在 ``TOOL_STARTED``（此刻 arun 正挂在 ``_invoke`` 的 ``wait_for`` 上，正好落在
        那个同 tick 窗口里）break，修好后只发生过 1 次 LLM 调用；吞掉取消则继续跑到 3~4 次。
        """
        llm = ScriptedLLM(
            [ScriptedResponse.tool("add", {"a": 1, "b": 2})],
            loop=True,
        )
        agent, _ = _build(llm, add, config=AgentConfig(max_steps=4))
        stream = agent.astream("go")
        async for event in stream:
            if event.type is EventType.TOOL_STARTED:
                break
        await stream.aclose()  # 触发 astream 的 finally：task.cancel() + gather
        self.assertLessEqual(
            len(llm.calls),
            2,
            msg=f"break 之后 run 仍在继续：LLM 调用 {len(llm.calls)} 次",
        )
        self.assertFalse(agent._run_guard.locked())

    async def test_astream_completes_normally_when_fully_consumed(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")], loop=True)
        agent, _ = _build(llm, add, config=AgentConfig(max_steps=1))
        types = [event.type async for event in agent.astream("go")]
        self.assertIs(EventType.RUN_STARTED, types[0])
        self.assertIs(EventType.RUN_FINISHED, types[-1])
        self.assertFalse(agent._run_guard.locked())


class PromptAndBufferDeduplicationTests(unittest.IsolatedAsyncioTestCase):
    """[v3 回归] 每轮 prompt 里当前用户消息只出现一次；同一条响应只占一条 assistant 消息。

    v2 的两处重复写入叠加：Agent 在 run 开始时 `memory.aadd(Message.user(input))`
    （§9.4.1 步骤 0），首轮又用 `append_user_input=True` 组装 prompt，而 `abuild_prompt`
    第 4 段已经把整个 window 倒了出来、第 6 段又追加一次同一个 user_input ——
    首轮 prompt = `[system, user, user]`。assistant 侧同理：步骤 2 写入原始响应，
    终结分支再 `aadd(Message.assistant(answer))` 追加一条。
    后果不只是白付 token：窗口以 1.5 倍速度膨胀，多轮对话很快被截断到错误的对话边界。
    """

    async def test_current_user_message_appears_once_in_every_prompt(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("The answer is 3."),
                ScriptedResponse.text("The answer is 4."),
            ]
        )
        memory = MemoryManager.from_config(MemoryConfig(), llm=llm)
        agent = Agent(llm=llm, memory=memory, tools=make_registry(add), name="m")

        first = await agent.arun("add 1 and 2")
        self.assertIs(AgentStatus.FINISHED, first.status)
        # 一轮 = 一次工具调用请求 + 一次终结请求
        self.assertEqual(2, len(llm.calls))
        first_prompt = [m.content for m in llm.calls[0].messages]
        self.assertEqual(1, first_prompt.count("add 1 and 2"),
                         msg=f"首轮 prompt 里用户消息出现了多次: {first_prompt}")

        await agent.arun("add 1 and 2 again")
        second_prompt = [m.content for m in llm.calls[2].messages]
        self.assertEqual(1, second_prompt.count("add 1 and 2 again"),
                         msg=f"第二轮 prompt 里用户消息出现了多次: {second_prompt}")
        # 上一轮的用户消息也只应保留一份
        self.assertEqual(1, second_prompt.count("add 1 and 2"),
                         msg=f"历史用户消息重复: {second_prompt}")

    async def test_one_response_leaves_exactly_one_assistant_message(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("The answer is 3.")])
        memory = MemoryManager.from_config(MemoryConfig(), llm=llm)
        agent = Agent(llm=llm, memory=memory, config=AgentConfig(mode="native"), name="m")

        await agent.arun("what is 1+2")
        contents = [m.content for m in memory.buffer.messages()]
        self.assertEqual(1, contents.count("The answer is 3."),
                         msg=f"同一条响应在短期记忆里出现了多次: {contents}")
        # 一轮未使用工具的最小对话 = user + assistant 各一条（不是 3 条）
        self.assertEqual(2, len(contents), msg=f"短期窗口条数不对: {contents}")


class CallbackAndReentrancyTests(unittest.IsolatedAsyncioTestCase):
    """回调隔离与不可重入守卫（§9.4）。"""

    async def test_raising_callback_does_not_affect_main_flow(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("still fine"),
            ]
        )
        agent, events = _build(llm, add)

        def _boom(event: TraceEvent) -> None:
            raise RuntimeError("subscriber exploded")

        agent.callbacks.subscribe(_boom)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("still fine", result.output)
        self.assertTrue(agent.callbacks.errors)
        self.assertIsInstance(agent.callbacks.errors[0][1], RuntimeError)
        # 其它订阅者照常收到事件
        self.assertIn(EventType.RUN_FINISHED, [e.type for e in events])

    async def test_arun_is_not_reentrant(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")], loop=True)
        agent, _ = _build(llm, add)

        first = asyncio.create_task(agent.arun("first"))
        await asyncio.sleep(0)  # 让 first 跑到守卫获取之后（守卫在任何 await 之前获取）
        second = await agent.arun("second")

        self.assertIs(AgentStatus.FAILED, second.status)
        self.assertIsInstance(second.error, AgentError)
        self.assertIn("already has a run in flight", str(second.error))
        self.assertEqual("", second.output)

        first_result = await first
        self.assertIs(AgentStatus.FINISHED, first_result.status)
        self.assertEqual("done", first_result.output)

    async def test_guard_is_released_after_a_failed_run(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)
        agent, _ = _build(
            llm, add, config=AgentConfig(max_steps=1, repeat_action_policy="off")
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertFalse(agent._run_guard.locked())
        # 失败之后仍可再跑
        again = await agent.arun("again")
        self.assertIs(AgentStatus.FAILED, again.status)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    """取消传播（§9.4.7）：CancelledError 必须继续抛出，状态标 ABORTED。"""

    async def test_cancelled_run_marks_aborted_and_reraises(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)
        agent, events = _build(llm, add, config=AgentConfig(max_steps=100))
        task = asyncio.create_task(agent.arun("go"))
        for _ in range(100):  # 确定性地等到守卫被获取（守卫在任何 await 之前）
            if agent._run_guard.locked():
                break
            await asyncio.sleep(0)
        self.assertTrue(agent._run_guard.locked())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIs(AgentStatus.ABORTED, agent.state.status)
        self.assertIsNotNone(agent.state.finished_at)  # finally 里的补救
        self.assertFalse(agent._run_guard.locked())
        run_failed = [e for e in events if e.type is EventType.RUN_FAILED]
        self.assertTrue(run_failed)
        self.assertTrue(run_failed[-1].data["aborted"])


class CustomExecutorContractTests(unittest.IsolatedAsyncioTestCase):
    """红线 6：Agent 必须防御性捕获自定义 executor 抛出的异常（编码成失败结果）。"""

    async def test_custom_executor_exception_becomes_a_failed_tool_result(self) -> None:
        class _ExplodingExecutor:
            config = None

            async def execute(self, call):  # noqa: ANN001
                raise RuntimeError("kaboom")

            async def execute_many(self, calls):  # noqa: ANN001
                raise RuntimeError("kaboom")

            async def aclose(self) -> None:
                return None

        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("handled"),
            ]
        )
        agent = Agent(
            llm=llm,
            tools=make_registry(add),
            config=AgentConfig(),
            executor=_ExplodingExecutor(),
        )
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        failure = result.tool_results[0]
        self.assertFalse(failure.ok)
        self.assertEqual("ToolExecutionError", failure.error_type)
        self.assertTrue(failure.content.startswith("ERROR("))


def tearDownModule() -> None:  # pragma: no cover - 测试卫生
    """[v3] 模块收尾：关掉本模块（含各用例自建 Agent）留下的 per-loop 私有线程池。

    本文件的 async 用例继承 `IsolatedAsyncioTestCase` —— 基类直接关闭每个用例的
    event loop，**不走** `config._run_and_cleanup`，所以 `LoopBoundPool.release_loop`
    从来没被调用过，池里的 worker（非 daemon 线程）会一直活到进程结束。

    手动变异验证过这条分支的**载荷**：把 `test_agent_features` /
    `test_agent_react_native` / `test_agent_react_text` 三处的本函数体改成 `pass`，
    跑完整套件后进程里残留 **33** 个 `liteagent-exec_*` 工作线程；
    三处都在时归零。

    收尾与断言都在 `tests.helpers.close_loop_bound_pools()`（唯一入口）：它
    `aclose_all()` 之后**断言**没有 `liteagent-*` 线程残留 —— 新增的 Agent 测试
    模块忘了收尾时会变红，而不是静默累积。
    """
    close_loop_bound_pools()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
