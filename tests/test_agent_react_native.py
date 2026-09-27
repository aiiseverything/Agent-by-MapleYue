from __future__ import annotations

"""``liteagent/agent/agent.py`` 的原生（function calling）模式测试（§9.4）。

覆盖 §12 对 ``test_agent_react_native.py`` 冻结的覆盖点：

* 单工具一轮、多工具并发顺序、多轮；
* ``finish_reason=stop`` 即终结；
* ``max_steps`` 用尽 -> ``FAILED`` + ``MaxStepsExceededError``；
* usage 累加；
* 事件序列断言（**按 §2.7 的归属矩阵**）；
* ``raise_on_error=True`` 时抛；
* ``finish_reason=length`` 续写一次；
* ``content_filter`` 立即失败；
* ``tool_calls`` 但空 -> 自纠正；
* ``max_total_tokens`` 用尽 -> ``BudgetExceededError`` + ``BUDGET_EXCEEDED`` 事件；
* ``max_wall_clock_s`` -> ``RunTimeoutError``；
* 非法 ``**overrides`` 键 -> ``ConfigError``。

全部离线确定性：``ScriptedLLM`` 驱动，零网络、零真实 sleep。
"""

import unittest

from liteagent.agent.agent import Agent
from liteagent.agent.callbacks import EventType, TraceEvent, as_llm_callback
from liteagent.agent.state import AgentStatus
from liteagent.config import AgentConfig, utc_now
from liteagent.errors import (
    AgentError,
    BudgetExceededError,
    ConfigError,
    MaxStepsExceededError,
    RunTimeoutError,
)
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from tests.helpers import add, close_loop_bound_pools, echo, make_registry


def _build(
    llm: ScriptedLLM,
    *tools: object,
    config: AgentConfig | None = None,
) -> tuple[Agent, list[TraceEvent]]:
    """造一个 Agent，并把 **Agent 层与 LLM 层**的事件都收进同一个列表。

    LLM 事件（LLM_REQUEST/LLM_RESPONSE/LLM_ERROR）的唯一发射者是 ``BaseLLMClient._emit``
    （§2.7），``Agent`` 不替它转发，所以测试必须自己把 ``as_llm_callback`` 接到 llm 上，
    否则"按归属矩阵断言事件序列"会漏掉半边。
    """
    agent = Agent(llm=llm, tools=make_registry(*tools), config=config)
    events: list[TraceEvent] = []
    agent.callbacks.subscribe(events.append)
    llm.on_event = as_llm_callback(agent.callbacks)
    return agent, events


def _types(events: list[TraceEvent]) -> list[EventType]:
    return [event.type for event in events]


class NativeRunTests(unittest.IsolatedAsyncioTestCase):
    """native 模式的常规路径。"""

    async def test_single_tool_round(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("the sum is 3"),
            ]
        )
        agent, _ = _build(llm, add, echo)
        result = await agent.arun("add 1 and 2")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertTrue(result.ok)
        self.assertEqual("the sum is 3", result.output)
        self.assertEqual(2, result.steps)
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("add", result.tool_calls[0].name)
        self.assertEqual(1, len(result.tool_results))
        self.assertEqual("3", result.tool_results[0].content)
        self.assertTrue(result.tool_results[0].ok)
        # native 模式把 schema 结构化地交给模型；文本模式才会是 None。
        self.assertIsNotNone(llm.calls[0].tools)
        self.assertEqual(["add", "echo"], llm.tool_names_seen(0))

    async def test_parallel_tool_calls_preserve_call_order(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tools(
                    ("add", {"a": 1, "b": 2}),
                    ("echo", {"text": "hi"}),
                    ("add", {"a": 10, "b": 20}),
                ),
                ScriptedResponse.text("all done"),
            ]
        )
        agent, _ = _build(llm, add, echo)
        result = await agent.arun("do three things")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual(["add", "echo", "add"], [c.name for c in result.tool_calls])
        # execute_many 的结果**顺序与 calls 严格一致**（§7.4.2）
        self.assertEqual(["add", "echo", "add"], [r.name for r in result.tool_results])
        self.assertEqual(["3", "hi", "30"], [r.content for r in result.tool_results])
        self.assertEqual(
            ["call_0", "call_1", "call_2"], [r.call_id for r in result.tool_results]
        )

    async def test_multi_round_accumulates_state(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 1}),
                ScriptedResponse.tool("echo", {"text": "next"}),
                ScriptedResponse.text("finished"),
            ]
        )
        agent, _ = _build(llm, add, echo)
        result = await agent.arun("two rounds")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual(3, result.steps)
        self.assertEqual(2, len(result.tool_calls))
        self.assertEqual(2, len(result.tool_results))
        self.assertEqual(3, llm.call_count)
        # transcript 里有：2 条 assistant + 2 条 tool 消息（observation 也回灌了）
        roles = [message.role.value for message in agent.state.messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)
        self.assertIs(AgentStatus.FINISHED, agent.state.status)
        self.assertIsNotNone(agent.state.finished_at)

    async def test_finish_reason_stop_terminates_immediately(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done", finish_reason="stop")])
        agent, events = _build(llm, add)
        result = await agent.arun("just answer")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("done", result.output)
        self.assertEqual(1, result.steps)
        self.assertEqual(1, llm.call_count)
        self.assertEqual([], result.tool_calls)
        self.assertEqual(EventType.RUN_FINISHED, events[-1].type)

    async def test_blank_content_falls_back_to_non_empty_output(self) -> None:
        """兜底：绝不返回空字符串（空 output 的 FINISHED 与 FAILED 无法区分）。"""
        llm = ScriptedLLM([ScriptedResponse.text("   ")])
        agent, _ = _build(llm, add)
        result = await agent.arun("x")
        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("", result.output)  # resp.content.strip() 也是空 -> 就是空串
        self.assertTrue(result.ok)


class NativeFailureTests(unittest.IsolatedAsyncioTestCase):
    """失败分支：max_steps / budget / timeout / 非法参数。"""

    async def test_max_steps_exhausted_reports_failure(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)
        agent, events = _build(
            llm, add, config=AgentConfig(max_steps=3, repeat_action_policy="off")
        )
        result = await agent.arun("never finishes")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, MaxStepsExceededError)
        self.assertEqual(3, result.error.max_steps)
        self.assertEqual(3, result.steps)
        run_failed = [e for e in events if e.type is EventType.RUN_FAILED]
        self.assertEqual(1, len(run_failed))
        self.assertEqual(
            {"error_type", "message", "aborted"},
            set(run_failed[0].data),
        )
        self.assertEqual("MaxStepsExceededError", run_failed[0].data["error_type"])
        self.assertFalse(run_failed[0].data["aborted"])

    async def test_raise_on_error_true_reraises(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)
        agent, _ = _build(
            llm,
            add,
            config=AgentConfig(
                max_steps=2, repeat_action_policy="off", raise_on_error=True
            ),
        )
        with self.assertRaises(MaxStepsExceededError):
            await agent.arun("boom")

    async def test_raise_on_error_false_encodes_failure_in_result(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1, "b": 2})], loop=True)
        agent, _ = _build(
            llm, add, config=AgentConfig(max_steps=2, repeat_action_policy="off")
        )
        result = await agent.arun("boom")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsNotNone(result.error)

    async def test_max_total_tokens_emits_budget_exceeded(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("never reached"),
            ]
        )
        agent, events = _build(llm, add, config=AgentConfig(max_total_tokens=10))
        result = await agent.arun("budget")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, BudgetExceededError)
        self.assertEqual("total_tokens", result.error.kind)
        self.assertEqual(10, result.error.limit)
        self.assertEqual(15, result.error.used)  # ScriptedLLM 单次固定 15 tokens
        budget = [e for e in events if e.type is EventType.BUDGET_EXCEEDED]
        self.assertEqual(1, len(budget))
        self.assertEqual({"kind", "limit", "used"}, set(budget[0].data))
        self.assertEqual("total_tokens", budget[0].data["kind"])
        self.assertEqual(10, budget[0].data["limit"])
        self.assertEqual(15, budget[0].data["used"])
        self.assertEqual(1, llm.call_count)  # 第 2 条响应从未被消费

    async def test_max_wall_clock_s_times_out(self) -> None:
        from liteagent.agent.state import AgentState

        llm = ScriptedLLM([ScriptedResponse.text("done")])
        agent, events = _build(llm, add)
        injected = AgentState.create("slow", agent_name="agent", run_id="run_injected")
        # 把 started_at 拨到 100 秒前 -> 真实墙钟判定必然触发，不依赖 sleep。
        injected.started_at = utc_now() - 100.0
        result = await agent.arun("slow", state=injected, max_wall_clock_s=1.0)

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, RunTimeoutError)
        self.assertEqual(1.0, result.error.timeout_s)
        self.assertGreater(result.error.elapsed_s, 1.0)
        self.assertEqual(0, llm.call_count)  # 第 0.5 步就终止了，没走到 LLM
        self.assertEqual("run_injected", result.state.run_id)  # 注入的 run_id 被保留
        self.assertIn(EventType.RUN_FAILED, _types(events))

    async def test_invalid_override_key_raises_config_error(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")])
        agent, _ = _build(llm, add)
        with self.assertRaises(ConfigError) as ctx:
            await agent.arun("x", botched_key=1)
        self.assertIn("botched_key", str(ctx.exception))
        # 参数错误不该消耗任何 LLM 调用
        self.assertEqual(0, llm.call_count)

    async def test_invalid_mode_override_raises_config_error(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")])
        agent, _ = _build(llm, add)
        with self.assertRaises(ConfigError):
            await agent.arun("x", mode="quantum")

    async def test_native_mode_requires_tool_calling_llm(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")])
        llm.supports_tool_calling = False
        agent, _ = _build(llm, add)
        with self.assertRaises(ConfigError):
            await agent.arun("x", mode="native")

    async def test_valid_overrides_reach_the_llm_call(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("done")])
        agent, _ = _build(llm, add)
        result = await agent.arun(
            "x", max_steps=1, temperature=0.0, max_tokens=7, tool_choice="none"
        )
        self.assertIs(AgentStatus.FINISHED, result.status)
        kwargs = llm.calls[0].kwargs
        self.assertEqual(0.0, kwargs["temperature"])
        self.assertEqual(7, kwargs["max_tokens"])
        self.assertEqual("none", kwargs["tool_choice"])


class NativeFinishReasonTests(unittest.IsolatedAsyncioTestCase):
    """finish_reason 参与控制流的四个分支（§9.4.6）。"""

    async def test_length_triggers_exactly_one_continuation(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text("half an answer", finish_reason="length"),
                ScriptedResponse.text("rest of the answer", finish_reason="stop"),
            ]
        )
        agent, events = _build(llm, add)
        result = await agent.arun("truncate me")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("rest of the answer", result.output)
        self.assertEqual(1, agent.state.truncation_errors)
        self.assertEqual(2, llm.call_count)
        self.assertEqual(1, len(agent.state.nudges))
        self.assertIn(
            "truncated before it finished", agent.state.nudges[0]
        )
        self.assertEqual(1, _types(events).count(EventType.NUDGE))

    async def test_length_beyond_retry_budget_fails(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text("a", finish_reason="length"),
                ScriptedResponse.text("b", finish_reason="length"),
            ]
        )
        agent, events = _build(
            llm, add, config=AgentConfig(max_truncation_retries=1)
        )
        result = await agent.arun("truncate forever")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, AgentError)
        self.assertIn("truncated", str(result.error))
        self.assertEqual(2, agent.state.truncation_errors)
        self.assertIn(EventType.RUN_FAILED, _types(events))

    async def test_content_filter_fails_immediately(self) -> None:
        llm = ScriptedLLM(
            [ScriptedResponse.text("blocked", finish_reason="content_filter")]
        )
        agent, events = _build(llm, add)
        result = await agent.arun("say something bad")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, AgentError)
        self.assertIn("content filter", str(result.error))
        self.assertEqual("content_filter", result.metadata["finish_reason"])
        self.assertEqual(1, llm.call_count)
        self.assertIn(EventType.RUN_FAILED, _types(events))

    async def test_empty_tool_calls_trigger_self_correction(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text("", finish_reason="tool_calls"),
                ScriptedResponse.text("recovered answer"),
            ]
        )
        agent, events = _build(llm, add, echo)
        before = len(agent.state.messages)
        result = await agent.arun("call a tool")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("recovered answer", result.output)
        self.assertEqual(1, agent.state.parse_errors)
        # 一次自纠正恰好新增 2 条消息（assistant 原文 + nudge 反馈）
        self.assertEqual(before + 3, len(agent.state.messages))
        self.assertEqual(1, _types(events).count(EventType.PARSE_ERROR))
        parse_errors = [e for e in events if e.type is EventType.PARSE_ERROR]
        self.assertEqual(
            {"reason", "offset", "raw_len", "attempt"}, set(parse_errors[0].data)
        )
        self.assertEqual(1, parse_errors[0].data["attempt"])
        self.assertEqual(1, _types(events).count(EventType.NUDGE))

    async def test_empty_tool_calls_exhaust_parse_retries(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text("", finish_reason="tool_calls"),
                ScriptedResponse.text("", finish_reason="tool_calls"),
            ]
        )
        agent, _ = _build(llm, add, config=AgentConfig(max_parse_retries=1))
        result = await agent.arun("keep lying about tools")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, AgentError)
        self.assertIn("could not be parsed", str(result.error))
        self.assertEqual(2, agent.state.parse_errors)


class NativeEventMatrixTests(unittest.IsolatedAsyncioTestCase):
    """事件序列与 §2.7 归属矩阵（Agent 不得重复发低层事件）。"""

    async def test_event_sequence_for_one_tool_round(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("3"),
            ]
        )
        agent, events = _build(llm, add)
        result = await agent.arun("add")
        run_id = result.state.run_id

        types = _types(events)
        self.assertIs(EventType.RUN_STARTED, types[0])
        self.assertIs(EventType.RUN_FINISHED, types[-1])
        # Agent 侧事件必须带正确的归属信息（低层的 memory_* 事件由 MemoryManager 发，
        # §2.7 不要求它们携带 run_id —— 只有 Agent 知道 run_id）。
        agent_owned = {
            EventType.RUN_STARTED,
            EventType.RUN_FINISHED,
            EventType.RUN_FAILED,
            EventType.STEP_STARTED,
            EventType.STEP_FINISHED,
        }
        for event in events:
            if event.type in agent_owned:
                self.assertEqual(run_id, event.run_id, f"{event.type} missing run_id")
                self.assertEqual("agent", event.agent_name)

        # Agent 侧事件：2 步 -> 2 个 STEP_STARTED，中间 1 个 STEP_FINISHED
        self.assertEqual(2, types.count(EventType.STEP_STARTED))
        self.assertEqual(1, types.count(EventType.STEP_FINISHED))
        step_starts = [e for e in events if e.type is EventType.STEP_STARTED]
        self.assertEqual([1, 2], [e.step for e in step_starts])

        # 低层事件：LLM 层 2 次请求 2 次响应（两个 step）；工具层 1 开始 1 完成
        self.assertEqual(2, types.count(EventType.LLM_REQUEST))
        self.assertEqual(2, types.count(EventType.LLM_RESPONSE))
        self.assertEqual(1, types.count(EventType.TOOL_STARTED))
        self.assertEqual(1, types.count(EventType.TOOL_FINISHED))
        self.assertEqual(0, types.count(EventType.TOOL_ERROR))
        # MemoryManager 的两个事件也必须出现（§2.7 里它们的唯一发射者是 MemoryManager）：
        # 每个 step 组装一次 prompt -> 2 条 MEMORY_RETRIEVE；写入至少 3 条
        # （用户输入 / assistant 响应 / 工具观察回灌）。
        self.assertGreaterEqual(types.count(EventType.MEMORY_WRITE), 3)
        self.assertEqual(2, types.count(EventType.MEMORY_RETRIEVE))

        # 顺序：step_started < llm_request < llm_response < tool_started < tool_finished
        self.assertLess(
            types.index(EventType.STEP_STARTED), types.index(EventType.LLM_REQUEST)
        )
        self.assertLess(
            types.index(EventType.LLM_REQUEST), types.index(EventType.LLM_RESPONSE)
        )
        self.assertLess(
            types.index(EventType.LLM_RESPONSE), types.index(EventType.TOOL_STARTED)
        )
        self.assertLess(
            types.index(EventType.TOOL_STARTED), types.index(EventType.TOOL_FINISHED)
        )

    async def test_run_started_and_finished_data_keys(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        agent, events = _build(llm, add, echo)
        await agent.arun("hello")

        started = [e for e in events if e.type is EventType.RUN_STARTED][0]
        self.assertEqual({"input", "mode", "tools"}, set(started.data))
        self.assertEqual("hello", started.data["input"])
        self.assertEqual("native", started.data["mode"])
        self.assertEqual(["add", "echo"], started.data["tools"])

        finished = [e for e in events if e.type is EventType.RUN_FINISHED][0]
        # §2.7：RUN_FINISHED 的 data 是 {output_len, steps, usage}（step 是公共字段，不在 data 里）
        self.assertEqual({"output_len", "steps", "usage"}, set(finished.data))
        self.assertEqual(2, finished.data["output_len"])
        self.assertEqual(1, finished.data["steps"])
        self.assertEqual(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finished.data["usage"],
        )

    async def test_usage_accumulates_across_calls(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("add", {"a": 1, "b": 2}),
                ScriptedResponse.text("done"),
            ]
        )
        agent, events = _build(llm, add)
        result = await agent.arun("x")

        self.assertEqual(30, result.usage.total_tokens)
        self.assertEqual(20, result.usage.prompt_tokens)
        self.assertEqual(10, result.usage.completion_tokens)
        self.assertEqual(result.usage, agent.state.usage)
        # trace 里两条 LLM_RESPONSE 的 usage 之和与 result 一致
        llm_responses = [e for e in events if e.type is EventType.LLM_RESPONSE]
        self.assertEqual(2, len(llm_responses))
        self.assertEqual(
            30,
            sum(e.data["usage"]["total_tokens"] for e in llm_responses),
        )

    async def test_agent_does_not_duplicate_tool_events(self) -> None:
        """TOOL_* 的唯一发射者是 ToolExecutor（§2.7）—— 3 个调用就是 3 条，不是 6 条。"""
        llm = ScriptedLLM(
            [
                ScriptedResponse.tools(
                    ("add", {"a": 1, "b": 2}),
                    ("echo", {"text": "hi"}),
                ),
                ScriptedResponse.text("done"),
            ]
        )
        agent, events = _build(llm, add, echo)
        await agent.arun("x")
        types = _types(events)
        self.assertEqual(2, types.count(EventType.TOOL_STARTED))
        self.assertEqual(2, types.count(EventType.TOOL_FINISHED))


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
