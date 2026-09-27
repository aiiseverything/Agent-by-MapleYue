from __future__ import annotations

"""``liteagent/agent/parser.py`` + 文本 ReAct 模式（§9.3 / §9.4）的测试。

覆盖 §12 对 ``test_agent_react_text.py`` 冻结的覆盖点：

* ``ScriptedResponse.react`` 的 action / final 两种载荷；
* 跨行 JSON、``\`\`\`json`` 围栏、单参数容错、全角冒号、中文 marker、Markdown 加粗 marker；
* ``Observation:`` 回灌内容；
* parse error 自纠正（**断言消息增量恰好 2**）；
* ``max_parse_retries`` 用尽；
* ``mode="text"`` 强制；
* **Action 与 Final Answer 同时出现时执行 Action 且不终结**（§9.3 步骤 4）。

全部离线确定性：``ScriptedLLM`` 驱动，零网络、零真实 sleep。
"""

import unittest

from liteagent.agent.agent import Agent
from liteagent.agent.callbacks import EventType, TraceEvent, as_llm_callback
from liteagent.agent.parser import ParsedAction, ReActParser
from liteagent.agent.state import AgentStatus
from liteagent.config import AgentConfig
from liteagent.errors import AgentError, ReActParseError
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from tests.helpers import add, close_loop_bound_pools, echo, make_registry


def _build(
    llm: ScriptedLLM,
    *tools: object,
    config: AgentConfig | None = None,
) -> tuple[Agent, list[TraceEvent]]:
    """造一个**文本模式**的 Agent 并收集 Agent 层 + LLM 层事件。"""
    agent = Agent(
        llm=llm,
        tools=make_registry(*tools),
        config=config if config is not None else AgentConfig(mode="text"),
    )
    events: list[TraceEvent] = []
    agent.callbacks.subscribe(events.append)
    llm.on_event = as_llm_callback(agent.callbacks)
    return agent, events


class ReActFixtureTests(unittest.TestCase):
    """``ScriptedResponse.react`` 的两种载荷形态（§6.6）。"""

    def test_react_action_payload(self) -> None:
        response = ScriptedResponse.react(
            "I should add", action="add", action_input={"a": 1, "b": 2}
        )
        self.assertEqual(
            'Thought: I should add\nAction: add\nAction Input: {"a": 1, "b": 2}',
            response.content,
        )
        self.assertEqual([], response.tool_calls)
        parsed = ReActParser(tool_names=("add",)).parse(response.content)
        self.assertTrue(parsed.is_action())
        self.assertEqual("add", parsed.action)
        self.assertEqual({"a": 1, "b": 2}, parsed.action_input)

    def test_react_final_payload(self) -> None:
        response = ScriptedResponse.react("done thinking", final="42")
        self.assertEqual("Thought: done thinking\nFinal Answer: 42", response.content)
        parsed = ReActParser(tool_names=("add",)).parse(response.content)
        self.assertTrue(parsed.is_final())
        self.assertEqual("42", parsed.final_answer)
        self.assertFalse(parsed.is_action())

    def test_action_wins_over_final_when_both_present(self) -> None:
        """§9.3 步骤 4 的不变量：``action`` 与 ``final_answer`` 至多一个非 None。"""
        content = (
            'Thought: try it\nAction: echo\nAction Input: {"text": "hi"}\n'
            "Final Answer: premature"
        )
        parsed = ReActParser(tool_names=("echo",)).parse(content)
        self.assertTrue(parsed.is_action())
        self.assertFalse(parsed.is_final())
        self.assertIsNone(parsed.final_answer)


class TextAgentRunTests(unittest.IsolatedAsyncioTestCase):
    """文本模式的 ReAct 循环（§9.4.5 的 text 列）。"""

    async def test_action_then_final_round_trip(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.react("need echo", action="echo", action_input={"text": "hi"}),
                ScriptedResponse.react("got it", final="the echo said hi"),
            ]
        )
        agent, events = _build(llm, add, echo)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("the echo said hi", result.output)
        self.assertEqual(2, result.steps)
        self.assertEqual(["echo"], [c.name for c in result.tool_calls])
        self.assertTrue(result.tool_calls[0].id.startswith("call_text_"))
        self.assertTrue(result.tool_results[0].ok)
        self.assertEqual("hi", result.tool_results[0].content)
        self.assertIn(EventType.ACTION_PARSED, [e.type for e in events])

    async def test_observation_is_written_back_to_transcript(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.react("x", action="echo", action_input={"text": "hi"}),
                ScriptedResponse.react("x", final="done"),
            ]
        )
        agent, _ = _build(llm, echo)
        await agent.arun("go")

        observations = [
            message
            for message in agent.state.messages
            if message.metadata.get("kind") == "observation"
        ]
        self.assertEqual(1, len(observations))
        observation = observations[0]
        self.assertEqual("user", observation.role.value)
        self.assertEqual("Observation: hi", observation.content)
        self.assertEqual(1, observation.metadata["step"])

    async def test_mode_text_forces_text_path_even_with_tool_calling_llm(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.react("x", final="ok")])
        agent, events = _build(llm, add, echo)
        self.assertTrue(llm.supports_tool_calling)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        # 文本模式不把 schema 结构化地交给模型（tools 参数为 None）
        self.assertIsNone(llm.calls[0].tools)
        started = [e for e in events if e.type is EventType.RUN_STARTED][0]
        self.assertEqual("text", started.data["mode"])
        self.assertEqual("text", agent.describe()["mode"])
        # system prompt 里仍渲染了工具清单（文本模式靠 prompt 暴露工具）
        system = llm.calls[0].messages[0]
        self.assertEqual("system", system.role.value)
        self.assertIn("echo", system.content)

    async def test_unknown_tool_name_is_not_a_parse_error(self) -> None:
        """§9.3 步骤 8：未知工具名不抛异常，交给 Agent 自纠正。"""
        llm = ScriptedLLM(
            [
                ScriptedResponse.react("x", action="nope", action_input={}),
                ScriptedResponse.react("x", final="ok"),
            ]
        )
        agent, _ = _build(llm, echo)
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("nope", result.tool_calls[0].name)
        self.assertFalse(result.tool_results[0].ok)
        self.assertEqual(0, agent.state.parse_errors)


class TextSyntaxTests(unittest.IsolatedAsyncioTestCase):
    """容错语法逐条走完 Agent（§9.3 步骤 3/5）。"""

    async def _run_single_tool(self, content: str) -> Agent:
        llm = ScriptedLLM([ScriptedResponse.text(content), ScriptedResponse.react("x", final="ok")])
        agent, _ = _build(llm, add, echo)
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FINISHED, result.status)
        return agent

    async def test_cross_line_json_action_input(self) -> None:
        agent = await self._run_single_tool(
            'Thought: add them\nAction: add\nAction Input: {\n  "a": 1,\n  "b": 2\n}'
        )
        self.assertEqual({"a": 1, "b": 2}, agent.state.tool_calls[0].arguments)
        self.assertEqual("3", agent.state.tool_results[0].content)

    async def test_json_fence_action_input(self) -> None:
        agent = await self._run_single_tool(
            "Thought: add\nAction: add\nAction Input:\n```json\n{\"a\": 3, \"b\": 4}\n```"
        )
        self.assertEqual({"a": 3, "b": 4}, agent.state.tool_calls[0].arguments)
        self.assertEqual("7", agent.state.tool_results[0].content)

    async def test_plain_fence_action_input(self) -> None:
        agent = await self._run_single_tool(
            "Thought: add\nAction: add\nAction Input:\n```\n{\"a\": 5, \"b\": 6}\n```"
        )
        self.assertEqual({"a": 5, "b": 6}, agent.state.tool_calls[0].arguments)

    async def test_single_parameter_bare_value_tolerance(self) -> None:
        """单参数工具的裸值 -> ``{<参数名>: 文本}``（§9.3 步骤 5.c）。"""
        agent = await self._run_single_tool(
            "Thought: echo it\nAction: echo\nAction Input: hello world"
        )
        self.assertEqual({"text": "hello world"}, agent.state.tool_calls[0].arguments)
        self.assertEqual("hello world", agent.state.tool_results[0].content)

    async def test_full_width_colon_markers(self) -> None:
        agent = await self._run_single_tool(
            'Thought：全角\nAction：echo\nAction Input：{"text": "hi"}'
        )
        self.assertEqual("echo", agent.state.tool_calls[0].name)
        self.assertEqual({"text": "hi"}, agent.state.tool_calls[0].arguments)

    async def test_cjk_markers(self) -> None:
        agent = await self._run_single_tool(
            '思考: 需要回显\n行动: echo\n行动输入: {"text": "hi"}'
        )
        self.assertEqual("echo", agent.state.tool_calls[0].name)
        self.assertEqual({"text": "hi"}, agent.state.tool_calls[0].arguments)

    async def test_markdown_bold_markers(self) -> None:
        agent = await self._run_single_tool(
            '**Thought:** bold\n**Action:** echo\n**Action Input:** {"text": "hi"}'
        )
        self.assertEqual("echo", agent.state.tool_calls[0].name)
        self.assertEqual({"text": "hi"}, agent.state.tool_calls[0].arguments)

    async def test_action_and_final_answer_together_executes_action(self) -> None:
        """§9.3 步骤 4：同段落里既有 Action 又有 Final Answer 时**执行 Action 且不终结**。"""
        llm = ScriptedLLM(
            [
                ScriptedResponse.text(
                    'Thought: try it\nAction: echo\nAction Input: {"text": "hi"}\n'
                    "Final Answer: premature"
                ),
                ScriptedResponse.react("ok", final="real answer"),
            ]
        )
        agent, events = _build(llm, echo)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("real answer", result.output)
        # 第一步没有终结：还需要第二步才拿到最终答案
        self.assertEqual(2, result.steps)
        self.assertEqual("echo", result.tool_calls[0].name)
        self.assertEqual(1, len(agent.state.tool_results))
        action_events = [e for e in events if e.type is EventType.ACTION_PARSED]
        self.assertEqual(1, len(action_events))
        self.assertEqual("echo", action_events[0].data["action"])
        self.assertEqual({"text": "hi"}, action_events[0].data["arguments"])


class TextParseErrorTests(unittest.IsolatedAsyncioTestCase):
    """parse-error 自纠正分支（§9.4.2）。"""

    async def test_parse_error_exactly_two_new_messages(self) -> None:
        raw = "this is definitely not ReAct syntax"
        llm = ScriptedLLM(
            [ScriptedResponse.text(raw), ScriptedResponse.react("x", final="recovered")]
        )
        agent, events = _build(llm, echo)
        # 空 transcript 起步（Agent 不会把 memory 的 prompt 塞进 state.messages）
        self.assertEqual([], agent.state.messages)
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual("recovered", result.output)
        self.assertEqual(1, agent.state.parse_errors)
        # ★ 可断言不变式：一次 parse error 恰好新增 2 条（1 assistant + 1 NUDGE），
        #   再加第 2 步的 assistant 答案 = 3 条。v1 的二次 add 会让总数变成 4。
        self.assertEqual(3, len(agent.state.messages))
        self.assertEqual(
            1,
            sum(1 for message in agent.state.messages if message.content == raw),
            "assistant 原文被写入了两次（v1 的回归）",
        )
        nudge_messages = [
            message
            for message in agent.state.messages
            if message.metadata.get("kind") == "nudge"
        ]
        self.assertEqual(1, len(nudge_messages))
        self.assertIn("could not be parsed", nudge_messages[0].content)
        self.assertIn("Action:", nudge_messages[0].content)

        parse_events = [e for e in events if e.type is EventType.PARSE_ERROR]
        self.assertEqual(1, len(parse_events))
        self.assertEqual(
            {"reason", "offset", "raw_len", "attempt"}, set(parse_events[0].data)
        )
        self.assertEqual("no ReAct structure found", parse_events[0].data["reason"])
        self.assertEqual(len(raw), parse_events[0].data["raw_len"])
        self.assertEqual(1, parse_events[0].data["attempt"])
        self.assertEqual(1, len(agent.state.nudges))

    async def test_max_parse_retries_exhausted(self) -> None:
        llm = ScriptedLLM(
            [ScriptedResponse.text("garbage one"), ScriptedResponse.text("garbage two")]
        )
        agent, events = _build(llm, echo, config=AgentConfig(mode="text", max_parse_retries=1))
        result = await agent.arun("go")

        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertIsInstance(result.error, AgentError)
        self.assertIn("could not be parsed", str(result.error))
        self.assertEqual(2, agent.state.parse_errors)
        # §3.1：cause 只存属性（不写 self.__cause__）
        self.assertIsInstance(result.error.cause, ReActParseError)
        self.assertEqual(1, len([e for e in events if e.type is EventType.RUN_FAILED]))
        # 每次 parse error 都新增 2 条消息（先注入反馈再判上限）
        self.assertEqual(4, len(agent.state.messages))

    async def test_zero_parse_retries_fails_on_first_error(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("garbage")])
        agent, _ = _build(llm, echo, config=AgentConfig(mode="text", max_parse_retries=0))
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FAILED, result.status)
        self.assertEqual(1, agent.state.parse_errors)
        self.assertEqual(1, llm.call_count)

    async def test_parse_error_consumes_a_step_without_retrying_llm(self) -> None:
        """D-10：parse error 消耗 step，但不重试 LLM 调用本身。"""
        llm = ScriptedLLM(
            [ScriptedResponse.text("garbage"), ScriptedResponse.react("x", final="ok")]
        )
        agent, _ = _build(
            llm, echo, config=AgentConfig(mode="text", max_steps=2, max_parse_retries=2)
        )
        result = await agent.arun("go")
        self.assertIs(AgentStatus.FINISHED, result.status)
        self.assertEqual(2, llm.call_count)  # 两步两次调用，没有额外重试
        self.assertEqual(2, result.steps)


class JsonObjectParserTests(unittest.TestCase):
    """§9.3 步骤 2：整段响应就是一个 JSON 对象时直接解析成 action / final_answer。

    `[v3 补测]` 这条路径由 `allow_json_object`（默认 True）控制、被 §9.3 冻结为步骤 2、
    被 §9.4.1 的真实产品路径启用（`Agent._build_parser()` 不传该参数），但 v2 的 1618
    个测试里**没有任何一条提到 allow_json_object、也没有构造过"整段 JSON 对象"的输入** ——
    把这一整段短路掉（`if self.allow_json_object:` -> `if False:`）测试套件照样全绿。
    这里把步骤 2 的每个可观测分支钉死（顺序按冻结伪代码：final_answer 优先于 action）。
    """

    def test_json_object_action(self) -> None:
        parsed = ReActParser(tool_names=("search",)).parse(
            '{"action": "search", "action_input": {"q": "x"}}'
        )
        self.assertTrue(parsed.is_action())
        self.assertEqual("search", parsed.action)
        self.assertEqual({"q": "x"}, parsed.action_input)
        self.assertIsNone(parsed.final_answer)
        self.assertTrue(parsed.json_mode)  # 冻结字段：v2 全项目无人断言过它

    def test_json_object_final_answer(self) -> None:
        parsed = ReActParser(tool_names=("search",)).parse('{"final_answer": "done"}')
        self.assertTrue(parsed.is_final())
        self.assertEqual("done", parsed.final_answer)
        self.assertTrue(parsed.json_mode)

    def test_json_object_tool_aliases(self) -> None:
        """`tool` / `tool_name` 都是 `action` 的容错别名（§9.3 步骤 2 冻结）。"""
        for raw in ('{"tool_name": "search", "action_input": {}}',
                    '{"tool": "search"}'):
            with self.subTest(raw=raw):
                parsed = ReActParser(tool_names=("search",)).parse(raw)
                self.assertTrue(parsed.is_action())
                self.assertEqual("search", parsed.action)

    def test_json_object_argument_aliases(self) -> None:
        for key in ("arguments", "tool_input"):
            with self.subTest(key=key):
                parsed = ReActParser(tool_names=("search",)).parse(
                    '{"action": "search", "%s": {"q": "x"}}' % key
                )
                self.assertTrue(parsed.is_action())
                self.assertEqual({"q": "x"}, parsed.action_input)

    def test_json_object_final_answer_wins_over_action(self) -> None:
        """整段 JSON 里同时有 action 与 final_answer 时，**final 优先**（§9.3 步骤 2 冻结顺序）。

        注意这与步骤 4 的"Action 优先于 Final Answer"（逐行 marker 扫描）相反 ——
        这是规范自己写死的两条相反规则，各自正交，不能互相套用。
        """
        parsed = ReActParser(tool_names=("search",)).parse(
            '{"action": "search", "final_answer": "x"}'
        )
        self.assertTrue(parsed.is_final())
        self.assertEqual("x", parsed.final_answer)

    def test_json_object_without_actionable_structure_falls_through(self) -> None:
        """步骤 2 认不出可执行结构时退回步骤 3/4 —— 没有 marker 就按 strict 语义报错。"""
        for raw in ('{"thought": "t"}', '[1, 2, 3]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ReActParseError):
                    ReActParser(tool_names=("search",)).parse(raw)

    def test_flag_off_disables_the_json_object_shortcut(self) -> None:
        """`allow_json_object=False` 时同一输入不再走步骤 2。"""
        strict = ReActParser(tool_names=("search",), allow_json_object=False)
        with self.assertRaises(ReActParseError):
            strict.parse('{"action": "search", "action_input": {"q": "x"}}')
        lax = ReActParser(tool_names=("search",), allow_json_object=False, strict=False)
        self.assertTrue(lax.parse('{"action": "search", "action_input": {"q": "x"}}').is_final())


class TextParserDirectTests(unittest.TestCase):
    """解析器层的边界（规范没在 §12 单列，但支撑上面的容错断言）。"""

    def test_thought_only_is_not_a_parse_error(self) -> None:
        parsed = ReActParser(tool_names=("echo",)).parse("Thought: just thinking")
        self.assertIsInstance(parsed, ParsedAction)
        self.assertIsNone(parsed.action)
        self.assertIsNone(parsed.final_answer)
        self.assertEqual("just thinking", parsed.thought)

    def test_strict_mode_raises_on_no_structure(self) -> None:
        with self.assertRaises(ReActParseError) as ctx:
            ReActParser(tool_names=("echo",)).parse("free text without markers")
        self.assertEqual("no ReAct structure found", ctx.exception.reason)
        self.assertEqual(0, ctx.exception.offset)

    def test_non_strict_mode_falls_back_to_final_answer(self) -> None:
        parsed = ReActParser(tool_names=("echo",), strict=False).parse("plain answer")
        self.assertTrue(parsed.is_final())
        self.assertEqual("plain answer", parsed.final_answer)

    def test_build_parse_error_feedback_lists_tool_names(self) -> None:
        parser = ReActParser(tool_names=("echo", "add"))
        error = ReActParseError(raw="x", offset=0, reason="no ReAct structure found")
        feedback = parser.build_parse_error_feedback(error)
        self.assertTrue(feedback.startswith("Your previous output could not be parsed:"))
        self.assertIn("Action: <one of [echo, add]>", feedback)
        self.assertIn("Final Answer:", feedback)


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
