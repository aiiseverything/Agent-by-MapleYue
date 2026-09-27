from __future__ import annotations

"""``liteagent/llm/scripted.py`` 的单元测试（§6.6）。

覆盖 §12 对 ``test_scripted.py`` 冻结的覆盖点：

* 队列消费；
* ``loop=True`` 优先于 ``strict``；
* ``loop=False + strict=True`` 耗尽抛 ``ScriptedExhaustedError``；
* ``error`` 响应被记为一次调用；
* ``assert_exhausted`` 在未耗尽时失败；
* **``call_id`` 占位分配**（``ScriptedResponse.tool("t")`` 的 ``id == ""``，消费后变 ``call_0``）；
* ``tool_raw``；
* ``tool_names_seen``（``tools=None`` 时 ``[]``）；
* ``calls[i].messages`` **不被后续轮次污染**；
* ``kwargs`` 含 ``temperature=None``。

零网络、零真实等待（``latency_s`` / ``delay_s`` 都经由注入的 ``RecordingSleep``）。
"""

import unittest
import warnings

import asyncio

from liteagent.config import LLMConfig
from liteagent.errors import (
    ConfigError,
    LLMRateLimitError,
    LLMTimeoutError,
    ScriptedExhaustedError,
)
from liteagent.llm.message import Message
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from liteagent.types import TokenUsage, ToolCall
from tests.helpers import RecordingSleep


def _user(text: str = "hi") -> list[Message]:
    return [Message.user(text)]


class ScriptedResponseConstructorTests(unittest.TestCase):
    """``ScriptedResponse`` 的构造器与序列化（§6.6 规则 8）。"""

    def test_text_and_tool_constructors(self) -> None:
        text = ScriptedResponse.text("hi")
        self.assertEqual("hi", text.content)
        self.assertEqual([], text.tool_calls)
        self.assertIsNone(text.finish_reason)

        tool = ScriptedResponse.tool("add", {"a": 1}, content="thinking")
        self.assertEqual("thinking", tool.content)
        self.assertEqual(1, len(tool.tool_calls))
        self.assertEqual("add", tool.tool_calls[0].name)
        self.assertEqual({"a": 1}, tool.tool_calls[0].arguments)

    def test_tool_call_id_is_a_placeholder_before_consumption(self) -> None:
        """``call_id=None`` 是**占位**：真正的 id 由 ScriptedLLM 在消费时分配。"""
        self.assertEqual("", ScriptedResponse.tool("t").tool_calls[0].id)
        self.assertEqual("given", ScriptedResponse.tool("t", call_id="given").tool_calls[0].id)

    def test_tool_raw_records_both_raw_and_dunder_raw_arguments(self) -> None:
        response = ScriptedResponse.tool_raw("add", "{not json")
        call = response.tool_calls[0]
        self.assertEqual("{not json", call.raw_arguments)
        self.assertEqual({"__raw__": "{not json"}, call.arguments)

    def test_tools_accepts_tuples_and_tool_call_objects(self) -> None:
        existing = ToolCall(id="explicit", name="c", arguments={})
        response = ScriptedResponse.tools(("a", {"x": 1}), existing, content="go")
        self.assertEqual(["a", "c"], [call.name for call in response.tool_calls])
        self.assertEqual("explicit", response.tool_calls[1].id)
        self.assertEqual("go", response.content)

    def test_react_renders_action_final_and_thought_only_payloads(self) -> None:
        action = ScriptedResponse.react("think", action="add", action_input={"a": 1})
        self.assertEqual(
            'Thought: think\nAction: add\nAction Input: {"a": 1}', action.content
        )
        default_input = ScriptedResponse.react("think", action="add")
        self.assertEqual(
            'Thought: think\nAction: add\nAction Input: {}', default_input.content
        )
        final = ScriptedResponse.react("think", final="42")
        self.assertEqual("Thought: think\nFinal Answer: 42", final.content)
        self.assertEqual("Thought: think", ScriptedResponse.react("think").content)

    def test_react_action_wins_over_final(self) -> None:
        response = ScriptedResponse.react("t", action="a", final="f")
        self.assertIn("Action: a", response.content)
        self.assertNotIn("Final Answer", response.content)

    def test_error_constructor_sets_the_error_field(self) -> None:
        response = ScriptedResponse.error(RuntimeError("boom"))
        self.assertIsInstance(response.error, RuntimeError)
        # 字段与构造器同名共存：实例属性拿到字段值，类属性拿到构造器。
        self.assertTrue(callable(ScriptedResponse.error))
        self.assertIsNone(ScriptedResponse.text("x").error)

    def test_unknown_kwargs_raise_type_error(self) -> None:
        """§6.6 规则 8：未知键**抛 ``TypeError``**，不静默丢弃。"""
        with self.assertRaises(TypeError):
            ScriptedResponse.text("x", unknown_key=True)

    def test_to_dict_is_field_complete(self) -> None:
        payload = ScriptedResponse.text("x").to_dict()
        self.assertEqual(
            {
                "content",
                "tool_calls",
                "finish_reason",
                "usage",
                "error",
                "delay_s",
                "stream_chunks",
                "stream_error",
            },
            set(payload),
        )
        self.assertIsNone(payload["error"])
        self.assertEqual("x", payload["content"])


class QueueConsumptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_are_consumed_fifo(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("one"), ScriptedResponse.text("two")])
        self.assertEqual(2, llm.remaining)
        self.assertEqual("one", (await llm.achat(_user())).content)
        self.assertEqual(1, llm.remaining)
        self.assertEqual("two", (await llm.achat(_user())).content)
        self.assertEqual(0, llm.remaining)
        llm.assert_exhausted()

    async def test_string_entries_are_promoted_to_text_responses(self) -> None:
        llm = ScriptedLLM(["plain string"])
        self.assertEqual("plain string", (await llm.achat(_user())).content)

    async def test_callable_entries_receive_a_copy_of_the_messages(self) -> None:
        seen: list[int] = []

        def respond(messages):
            seen.append(len(messages))
            messages.append(Message.user("mutated"))  # 不能污染调用方的 list
            return ScriptedResponse.text("from callable")

        llm = ScriptedLLM([respond])
        history = _user()
        response = await llm.achat(history)
        self.assertEqual("from callable", response.content)
        self.assertEqual([1], seen)
        self.assertEqual(1, len(history))

    async def test_push_and_extend_append_to_the_tail(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a")])
        llm.push(ScriptedResponse.text("b"))
        llm.extend([ScriptedResponse.text("c")])
        self.assertEqual(3, llm.remaining)
        contents = [(await llm.achat(_user())).content for _ in range(3)]
        self.assertEqual(["a", "b", "c"], contents)

    async def test_reset_restores_the_queue_and_clears_history(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a")])
        await llm.achat(_user())
        self.assertEqual(1, llm.call_count)
        llm.reset()
        self.assertEqual(0, llm.call_count)
        self.assertEqual(0, llm.exhausted_count)
        self.assertEqual(1, llm.remaining)
        self.assertEqual("a", (await llm.achat(_user())).content)

    async def test_unknown_response_type_raises_config_error(self) -> None:
        llm = ScriptedLLM()
        llm._queue.append(object())  # 直接塞一个非法元素，绕过构造器的类型提示
        with self.assertRaises(ConfigError) as ctx:
            await llm.achat(_user())
        self.assertIn("scripted response", str(ctx.exception))


class StrictAndLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_exhaustion_raises_scripted_exhausted_error(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("only")], strict=True, loop=False)
        await llm.achat(_user())
        with self.assertRaises(ScriptedExhaustedError) as ctx:
            await llm.achat(_user())
        self.assertEqual(1, ctx.exception.consumed)
        self.assertEqual(1, llm.exhausted_count)
        # 耗尽的这次调用也被记账（response 为 None）。
        self.assertEqual(2, llm.call_count)
        self.assertIsNone(llm.calls[-1].response)

    async def test_loop_wins_over_strict(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("again")], loop=True, strict=True)
        self.assertEqual(0, llm.remaining)  # loop=True 时 remaining 恒为 0
        contents = [(await llm.achat(_user())).content for _ in range(3)]
        self.assertEqual(["again", "again", "again"], contents)
        self.assertEqual(0, llm.exhausted_count)  # loop 下重复消费不算"耗尽"
        self.assertEqual(3, llm.call_count)
        llm.assert_exhausted()

    async def test_non_strict_exhaustion_returns_an_empty_response_with_a_warning(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("only")], strict=False)
        await llm.achat(_user())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = await llm.achat(_user())
        self.assertEqual("", response.content)
        self.assertEqual([], response.tool_calls)
        self.assertEqual("stop", response.finish_reason)
        self.assertEqual(1, len(caught))
        self.assertTrue(issubclass(caught[0].category, RuntimeWarning))

    async def test_loop_on_an_empty_script_falls_back_to_a_warning(self) -> None:
        llm = ScriptedLLM([], loop=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = await llm.achat(_user())
        self.assertEqual("", response.content)
        self.assertEqual(1, len(caught))

    async def test_assert_exhausted_fails_while_items_remain(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        await llm.achat(_user())
        with self.assertRaises(AssertionError) as ctx:
            llm.assert_exhausted()
        self.assertIn("not exhausted", str(ctx.exception))

    async def test_assert_exhausted_fails_when_a_response_was_never_consumed(self) -> None:
        """队列空了但脚本里还有没被消费的条目（例如被 push 之后又被丢弃）。"""
        llm = ScriptedLLM([ScriptedResponse.text("a")])
        llm._queue.clear()
        llm._consumed = 0
        with self.assertRaises(AssertionError):
            llm.assert_exhausted()


class ErrorResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_response_is_recorded_as_one_call(self) -> None:
        error = LLMRateLimitError(status_code=429, message="slow down")
        llm = ScriptedLLM([ScriptedResponse.error(error), ScriptedResponse.text("ok")])
        with self.assertRaises(LLMRateLimitError):
            await llm.achat(_user())
        self.assertEqual(1, llm.call_count)
        self.assertIs(error, llm.calls[0].response.error)
        # 下一次调用消费第二条，说明错误响应确实从队列里出队了。
        self.assertEqual("ok", (await llm.achat(_user())).content)
        self.assertEqual(2, llm.call_count)

    async def test_error_response_emits_an_llm_error_event(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.error(RuntimeError("boom"))])
        with self.assertRaises(RuntimeError):
            await llm.achat(_user())
        errors = [data for name, data in llm.events if name == "llm_error"]
        self.assertEqual(1, len(errors))
        self.assertEqual("RuntimeError", errors[0]["error_type"])
        self.assertEqual(0, errors[0]["retry"])

    async def test_last_call_raises_when_nothing_was_recorded(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a")])
        with self.assertRaises(AssertionError):
            llm.last_call()


class CallIdAssignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_placeholder_ids_are_assigned_in_consumption_order(self) -> None:
        response = ScriptedResponse.tools(("a", {}), ("b", {}))
        llm = ScriptedLLM([response])
        result = await llm.achat(_user())
        self.assertEqual(["call_0", "call_1"], [call.id for call in result.tool_calls])
        # 就地改写：响应对象上的占位 id 也被换成真 id（测试可回看）。
        self.assertEqual(["call_0", "call_1"], [call.id for call in response.tool_calls])
        # 序号是实例属性、跨调用继续递增。
        second = await self._second(llm)
        self.assertEqual("call_2", second.tool_calls[0].id)

    async def _second(self, llm: ScriptedLLM):
        llm.push(ScriptedResponse.tool("c"))
        return await llm.achat(_user())

    async def test_explicit_call_id_is_kept_and_does_not_consume_the_sequence(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.tool("a", call_id="explicit-id"),
                ScriptedResponse.tool("b"),
            ]
        )
        first = await llm.achat(_user())
        second = await llm.achat(_user())
        self.assertEqual("explicit-id", first.tool_calls[0].id)
        self.assertEqual("call_0", second.tool_calls[0].id)

    async def test_sequence_starts_at_zero_for_each_instance(self) -> None:
        first = ScriptedLLM([ScriptedResponse.tool("a")])
        second = ScriptedLLM([ScriptedResponse.tool("a")])
        self.assertEqual("call_0", (await first.achat(_user())).tool_calls[0].id)
        self.assertEqual("call_0", (await second.achat(_user())).tool_calls[0].id)


class CallAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_kwargs_contains_the_four_frozen_keys_even_when_none(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user())
        self.assertEqual(
            {"temperature": None, "max_tokens": None, "tool_choice": None, "stop": None},
            llm.calls[0].kwargs,
        )
        self.assertEqual("temperature" in llm.calls[0].kwargs, True)
        self.assertIsNone(llm.calls[0].kwargs["temperature"])

    async def test_kwargs_records_explicit_values(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(
            _user(), temperature=0.2, max_tokens=64, tool_choice="auto", stop=["END"]
        )
        self.assertEqual(
            {"temperature": 0.2, "max_tokens": 64, "tool_choice": "auto", "stop": ["END"]},
            llm.calls[0].kwargs,
        )

    async def test_unknown_kwargs_are_tolerated_and_not_recorded(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user(), top_p=0.9)
        self.assertNotIn("top_p", llm.calls[0].kwargs)

    async def test_calls_i_messages_are_not_polluted_by_later_rounds(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        history = _user("first")
        await llm.achat(history)
        history.append(Message.assistant("second"))
        self.assertEqual(1, len(llm.calls[0].messages))
        self.assertEqual("first", llm.calls[0].messages[0].content)

        await llm.achat(history)
        self.assertEqual(1, len(llm.calls[0].messages))
        self.assertEqual(2, len(llm.calls[1].messages))

    async def test_tools_list_is_shallow_copied_into_the_record(self) -> None:
        """§6.6 规则 7 冻结为**逐项浅拷贝**：外层 list 与每个 dict 都是新的，
        因此调用方之后替换顶层键 / 追加元素都不会污染已记录的 ``calls[i].tools``
        （嵌套值仍共享 —— 那是浅拷贝的既定语义，不要在这里断言深拷贝）。"""
        tools = [{"name": "add"}]
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user(), tools=tools)
        tools[0]["name"] = "tampered"
        tools.append({"name": "extra"})
        self.assertEqual(["add"], [spec["name"] for spec in llm.calls[0].tools])
        self.assertIsNot(tools, llm.calls[0].tools)

    async def test_last_messages_is_a_copy(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user("hello"))
        messages = llm.last_messages()
        self.assertEqual("hello", messages[0].content)
        messages.append(Message.user("extra"))
        self.assertEqual(1, len(llm.last_messages()))

    async def test_call_index_is_sequential_and_count_matches(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        await llm.achat(_user())
        await llm.achat(_user())
        self.assertEqual([0, 1], [call.index for call in llm.calls])
        self.assertEqual(2, llm.call_count)


class ToolNamesSeenTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_none_yields_an_empty_list(self) -> None:
        """§6.6：``tools is None``（文本模式）时返回 ``[]``，不去解析 system prompt。"""
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user(), tools=None)
        self.assertIsNone(llm.calls[0].tools)
        self.assertEqual([], llm.tool_names_seen())

    async def test_openai_and_anthropic_tool_shapes_both_work(self) -> None:
        openai_tools = [
            {"type": "function", "function": {"name": "add", "parameters": {}}},
            {"type": "function", "function": {"name": "echo", "parameters": {}}},
        ]
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user(), tools=openai_tools)
        self.assertEqual(["add", "echo"], llm.tool_names_seen())

        anthropic_tools = [
            {"name": "add", "input_schema": {}},
            {"name": "search", "input_schema": {}},
        ]
        other = ScriptedLLM([ScriptedResponse.text("x")])
        await other.achat(_user(), tools=anthropic_tools)
        self.assertEqual(["add", "search"], other.tool_names_seen())

    async def test_index_selects_a_specific_call_and_negative_index_is_supported(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        await llm.achat(_user(), tools=[{"function": {"name": "first"}}])
        await llm.achat(_user(), tools=[{"function": {"name": "second"}}])
        self.assertEqual(["first"], llm.tool_names_seen(0))
        self.assertEqual(["second"], llm.tool_names_seen(-1))

    async def test_empty_tool_list_yields_no_names(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        await llm.achat(_user(), tools=[])
        self.assertEqual([], llm.tool_names_seen())


class UsageAndLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_usage_is_the_frozen_constant(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("x")])
        response = await llm.achat(_user())
        self.assertEqual(TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15), response.usage)

    async def test_instance_default_usage_is_used_when_the_response_has_none(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        llm = ScriptedLLM([ScriptedResponse.text("x")], default_usage=usage)
        self.assertEqual(usage, (await llm.achat(_user())).usage)

    async def test_per_response_usage_wins(self) -> None:
        usage = TokenUsage(prompt_tokens=7, completion_tokens=8, total_tokens=15)
        llm = ScriptedLLM(
            [ScriptedResponse.text("x", usage=usage)], default_usage=TokenUsage(1, 1, 2)
        )
        self.assertEqual(usage, (await llm.achat(_user())).usage)

    async def test_latency_ms_is_delay_s_times_1000(self) -> None:
        sleep = RecordingSleep()
        llm = ScriptedLLM(
            [ScriptedResponse.text("x", delay_s=0.5)],
            config=LLMConfig(provider="echo", sleep_fn=sleep),
        )
        response = await llm.achat(_user())
        self.assertEqual(500.0, response.latency_ms)
        # 每次调用依次走两条时间线：latency_s（这里取默认 0.0）再 delay_s。
        self.assertEqual([0.0, 0.5], sleep.delays)

    async def test_latency_s_is_charged_on_every_call(self) -> None:
        sleep = RecordingSleep()
        llm = ScriptedLLM(
            [ScriptedResponse.text("a"), ScriptedResponse.text("b")],
            latency_s=0.25,
            config=LLMConfig(provider="echo", sleep_fn=sleep),
        )
        await llm.achat(_user())
        await llm.achat(_user())
        # 每次：latency_s(0.25) + delay_s(0.0)。
        self.assertEqual([0.25, 0.0, 0.25, 0.0], sleep.delays)

    async def test_model_defaults_to_scripted_1_and_can_be_overridden(self) -> None:
        self.assertEqual("scripted-1", ScriptedLLM([ScriptedResponse.text("x")]).model)
        llm = ScriptedLLM([ScriptedResponse.text("x")], model="my-model")
        self.assertEqual("my-model", (await llm.achat(_user())).model)

    async def test_an_explicit_config_wins_over_the_model_argument(self) -> None:
        """§6.6：显式传 config 时以 config 为准（调用方给了更具体的信息）。"""
        llm = ScriptedLLM(
            [ScriptedResponse.text("x")],
            model="ignored",
            config=LLMConfig(provider="echo", model="from-config"),
        )
        self.assertEqual("from-config", (await llm.achat(_user())).model)

    async def test_finish_reason_defaults_follow_the_tool_calls(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add")])
        response = await llm.achat(_user())
        self.assertEqual("tool_calls", response.finish_reason)
        self.assertTrue(response.has_tool_calls)

    async def test_explicit_finish_reason_is_preserved(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("cut", finish_reason="length")])
        self.assertEqual("length", (await llm.achat(_user())).finish_reason)

    async def test_class_flags(self) -> None:
        llm = ScriptedLLM()
        self.assertTrue(llm.supports_tool_calling)
        self.assertFalse(llm.requires_api_key)
        self.assertEqual("scripted", llm.name)
        self.assertEqual("native", llm.resolve_mode(has_tools=True))
        self.assertEqual("text", llm.resolve_mode(has_tools=False))


class DelayTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """§6.6 规则 5：``delay_s`` 受 ``config.timeout_s`` 约束，``latency_s`` **不受**。"""

    async def test_delay_longer_than_the_timeout_raises_llm_timeout_error(self) -> None:
        async def never(seconds: float) -> None:
            # 正数延迟永不返回（只有 wait_for 能打断它）；0.0 直接返回，否则
            # 无条件的 latency 睡眠（规则 5）会把本来不睡的分支也一起挂住。
            if seconds <= 0:
                return
            await asyncio.Event().wait()

        llm = ScriptedLLM(
            [ScriptedResponse.text("slow", delay_s=10.0)],
            config=LLMConfig(provider="echo", timeout_s=0.01, sleep_fn=never),
        )
        with self.assertRaises(LLMTimeoutError) as ctx:
            await llm.achat(_user())
        self.assertEqual(0.01, ctx.exception.timeout_s)

    async def test_delay_within_the_timeout_does_not_raise(self) -> None:
        sleep = RecordingSleep()
        llm = ScriptedLLM(
            [ScriptedResponse.text("ok", delay_s=0.5)],
            config=LLMConfig(provider="echo", timeout_s=10.0, sleep_fn=sleep),
        )
        self.assertEqual("ok", (await llm.achat(_user())).content)
        self.assertEqual([0.0, 0.5], sleep.delays)

    async def test_latency_s_is_not_subject_to_the_timeout(self) -> None:
        """规则 5 明写：``latency_s`` **无条件**走 sleep_fn，不受 timeout 约束。"""
        sleep = RecordingSleep()
        llm = ScriptedLLM(
            [ScriptedResponse.text("ok")],
            latency_s=99.0,
            config=LLMConfig(provider="echo", timeout_s=0.001, sleep_fn=sleep),
        )
        self.assertEqual("ok", (await llm.achat(_user())).content)


class RecordingShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_object_is_recorded_on_the_call(self) -> None:
        scripted = ScriptedResponse.text("x")
        llm = ScriptedLLM([scripted])
        await llm.achat(_user())
        self.assertIs(scripted, llm.calls[0].response)

    async def test_events_are_recorded_for_a_successful_call(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        await llm.achat(_user())
        self.assertEqual(
            ["llm_request", "llm_response"], [name for name, _data in llm.events]
        )
        request = [data for name, data in llm.events if name == "llm_request"][0]
        self.assertEqual(1, request["messages_count"])
        self.assertEqual(0, request["tools_count"])
        self.assertEqual(0, request["retry"])
        self.assertEqual("scripted-1", request["model"])

    async def test_on_event_callback_receives_the_same_events(self) -> None:
        seen: list[tuple[str, dict]] = []
        llm = ScriptedLLM(
            [ScriptedResponse.text("hi")], on_event=lambda name, data: seen.append((name, data))
        )
        await llm.achat(_user())
        self.assertEqual(["llm_request", "llm_response"], [name for name, _ in seen])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
