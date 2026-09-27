from __future__ import annotations

"""``ScriptedLLM.astream_chat`` 的单元测试（§6.6，``[v2 新增]``）。

覆盖 §12 对 ``test_llm_streaming.py`` 冻结的覆盖点：

* 单 chunk 全量（``stream_chunks is None`` -> ``[content]``）；
* 多 chunk 顺序与 ``index``；
* 最后一个 chunk 的 ``finish_reason``（含从 tool_calls 推导）；
* ``stream_error`` 中断（在最后一个 chunk **之后**抛）；
* ``astream_chat`` 也计入 ``calls``。

零网络、零真实等待；需要断言等待时用 ``tests.helpers.RecordingSleep``。
"""

import unittest

from liteagent.config import LLMConfig
from liteagent.errors import LLMRateLimitError
from liteagent.llm.message import Message
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from tests.helpers import RecordingSleep


class StreamChunkShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_chunk_carries_the_whole_content(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hello world")])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual(1, len(chunks))
        self.assertEqual("hello world", chunks[0].delta)
        self.assertEqual(0, chunks[0].index)
        self.assertEqual("stop", chunks[0].finish_reason)
        self.assertIsNone(chunks[0].tool_call_delta)

    async def test_explicit_single_chunk_behaves_like_the_default(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi", stream_chunks=["hi"])])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual(["hi"], [chunk.delta for chunk in chunks])

    async def test_multiple_chunks_preserve_order_and_indexes(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("abc", stream_chunks=["a", "b", "c"])])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual(["a", "b", "c"], [chunk.delta for chunk in chunks])
        self.assertEqual([0, 1, 2], [chunk.index for chunk in chunks])

    async def test_only_the_last_chunk_carries_the_finish_reason(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("ab", stream_chunks=["a", "b"])])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual([None, "stop"], [chunk.finish_reason for chunk in chunks])

    async def test_finish_reason_from_the_response_is_used_verbatim(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text(
                    "truncated", stream_chunks=["trunc"], finish_reason="length"
                )
            ]
        )
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual("length", chunks[-1].finish_reason)

    async def test_finish_reason_is_derived_from_tool_calls(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.tool("add", {"a": 1}, content="calling")])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual(["calling"], [chunk.delta for chunk in chunks])
        self.assertEqual("tool_calls", chunks[-1].finish_reason)

    async def test_an_explicit_empty_chunk_list_yields_nothing(self) -> None:
        """``stream_chunks=[]`` 与 ``None`` 语义不同（显式表示"一个 chunk 都不发"）。"""
        llm = ScriptedLLM([ScriptedResponse.text("ignored", stream_chunks=[])])
        chunks = [chunk async for chunk in llm.astream_chat([Message.user("hi")])]
        self.assertEqual([], chunks)

    async def test_multiple_queued_responses_stream_in_order(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text("first", stream_chunks=["f1", "f2"]),
                ScriptedResponse.text("second", stream_chunks=["s1"]),
            ]
        )
        first = [chunk async for chunk in llm.astream_chat([Message.user("1")])]
        second = [chunk async for chunk in llm.astream_chat([Message.user("2")])]
        self.assertEqual(["f1", "f2"], [chunk.delta for chunk in first])
        self.assertEqual(["s1"], [chunk.delta for chunk in second])
        self.assertEqual(0, llm.remaining)


class StreamErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_error_is_raised_after_the_last_chunk(self) -> None:
        llm = ScriptedLLM(
            [
                ScriptedResponse.text(
                    "abc",
                    stream_chunks=["a", "b", "c"],
                    stream_error=LLMRateLimitError(status_code=429, message="stream cut"),
                )
            ]
        )
        received = []
        with self.assertRaises(LLMRateLimitError):
            async for chunk in llm.astream_chat([Message.user("hi")]):
                received.append(chunk)
        self.assertEqual(["a", "b", "c"], [chunk.delta for chunk in received])
        self.assertEqual(1, len(llm.calls))

    async def test_stream_error_is_recorded_as_an_event(self) -> None:
        llm = ScriptedLLM(
            [ScriptedResponse.text("x", stream_error=RuntimeError("cut"))]
        )
        with self.assertRaises(RuntimeError):
            async for _chunk in llm.astream_chat([Message.user("hi")]):
                pass
        names = [name for name, _data in llm.events]
        self.assertIn("llm_error", names)
        error_events = [data for name, data in llm.events if name == "llm_error"]
        self.assertEqual("RuntimeError", error_events[0]["error_type"])

    async def test_error_response_raises_before_any_chunk_is_yielded(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.error(RuntimeError("boom"))])
        received = []
        with self.assertRaises(RuntimeError):
            async for chunk in llm.astream_chat([Message.user("hi")]):
                received.append(chunk)
        self.assertEqual([], received)
        self.assertEqual(1, len(llm.calls))

    async def test_an_ordinary_error_on_the_first_of_two_responses_stops_the_stream(self) -> None:
        llm = ScriptedLLM(
            [ScriptedResponse.error(RuntimeError("boom")), ScriptedResponse.text("later")]
        )
        with self.assertRaises(RuntimeError):
            async for _chunk in llm.astream_chat([Message.user("hi")]):
                pass
        self.assertEqual(1, len(llm.calls))
        self.assertEqual(1, llm.remaining)


class StreamCallAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_astream_chat_is_counted_in_calls(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        async for _chunk in llm.astream_chat([Message.user("one")]):
            pass
        self.assertEqual(1, llm.call_count)
        self.assertEqual(0, llm.calls[0].index)
        self.assertEqual(1, len(llm.calls[0].messages))
        self.assertEqual("one", llm.calls[0].messages[0].content)

    async def test_streaming_kwargs_use_the_same_frozen_key_set(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        async for _chunk in llm.astream_chat(
            [Message.user("hi")], temperature=0.5, max_tokens=32
        ):
            pass
        self.assertEqual(
            {"temperature": 0.5, "max_tokens": 32, "tool_choice": None, "stop": None},
            llm.calls[0].kwargs,
        )

    async def test_streaming_records_tools_and_supports_tool_names_seen(self) -> None:
        tools = [{"function": {"name": "add", "parameters": {}}}]
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        async for _chunk in llm.astream_chat([Message.user("hi")], tools=tools):
            pass
        self.assertEqual(tools, llm.calls[0].tools)
        self.assertEqual(["add"], llm.tool_names_seen())

    async def test_streaming_without_tools_reports_no_tool_names(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hi")])
        async for _chunk in llm.astream_chat([Message.user("hi")]):
            pass
        self.assertIsNone(llm.calls[0].tools)
        self.assertEqual([], llm.tool_names_seen())

    async def test_streaming_records_calls_so_assert_exhausted_passes(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        for _ in range(2):
            async for _chunk in llm.astream_chat([Message.user("hi")]):
                pass
        self.assertEqual(2, llm.call_count)
        llm.assert_exhausted()

    async def test_streaming_emits_request_and_response_events(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("hello")])
        async for _chunk in llm.astream_chat([Message.user("hi")]):
            pass
        self.assertEqual(
            ["llm_request", "llm_response"], [name for name, _data in llm.events]
        )
        response_event = [data for name, data in llm.events if name == "llm_response"][0]
        self.assertEqual(5, response_event["content_len"])
        self.assertEqual("stop", response_event["finish_reason"])

    async def test_latency_goes_through_the_injected_sleep_fn(self) -> None:
        """§5.5 规则 1：``latency_s`` 必须经由 ``sleep_fn``，不得直接 ``asyncio.sleep``。"""
        sleep = RecordingSleep()
        llm = ScriptedLLM(
            [ScriptedResponse.text("hi", delay_s=0.25)],
            latency_s=0.125,
            config=LLMConfig(provider="echo", sleep_fn=sleep),
        )
        async for _chunk in llm.astream_chat([Message.user("hi")]):
            pass
        self.assertEqual([0.125, 0.25], sleep.delays)

    async def test_calls_i_messages_are_not_polluted_by_later_streams(self) -> None:
        llm = ScriptedLLM([ScriptedResponse.text("a"), ScriptedResponse.text("b")])
        history = [Message.user("first")]

        async for _chunk in llm.astream_chat(history):
            pass
        history.append(Message.assistant("second"))

        self.assertEqual(1, len(llm.calls[0].messages))
        self.assertEqual("first", llm.calls[0].messages[0].content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
