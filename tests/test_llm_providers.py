from __future__ import annotations

"""``liteagent/llm/providers.py`` 的单元测试（§6.4）。

覆盖 §12 对 ``test_llm_providers.py`` 冻结的覆盖点：

* ``OpenAIChatClient`` 请求体（tools 形态、消息转换）与响应解析
  （**含 arguments 非法 JSON -> ``__raw__`` 且不抛异常**）；
* ``AnthropicChatClient``：system 被提取、tool_result 合并进**同一 user 消息**、
  block 解析、stop_reason 映射（含 ``max_tokens`` -> ``length``）；
* ``DeepSeekChatClient`` 的 ``default_base_url``；
* ``EchoLLM`` 的 ``use:add {"a":1,"b":2}`` == 3。

全程零网络：真实 provider 的 ``achat`` 走 ``tests.helpers.FakeTransport``。
"""

import json
import os
import unittest
from unittest import mock

from liteagent.config import LLMConfig
from liteagent.errors import ConfigError, LLMAuthError, LLMResponseFormatError
from liteagent.llm.message import Message, Role
from liteagent.llm.providers import (
    PROVIDER_CLASSES,
    AnthropicChatClient,
    DeepSeekChatClient,
    EchoLLM,
    HTTPChatClient,
    OpenAICompatibleClient,
    OpenAIChatClient,
)
from liteagent.llm.transport import HTTPResponse
from liteagent.types import ToolCall
from tests.helpers import FakeTransport, add

# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


def _openai_response(content: str = "hi", *, tool_calls=None, finish_reason=None) -> str:
    message: dict = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps(
        {
            "model": "gpt-4o-mini",
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )


def _openai_client(responses=(), **config_kw) -> tuple[OpenAIChatClient, FakeTransport]:
    transport = FakeTransport(list(responses))
    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test", **config_kw)
    return OpenAIChatClient(config, transport=transport), transport


def _anthropic_client(responses=(), **config_kw) -> tuple[AnthropicChatClient, FakeTransport]:
    transport = FakeTransport(list(responses))
    config = LLMConfig(
        provider="anthropic", model="claude-3-5-haiku", api_key="sk-ant", **config_kw
    )
    return AnthropicChatClient(config, transport=transport), transport


def _anthropic_ok(content_blocks, *, stop_reason="end_turn", usage=None) -> str:
    return json.dumps(
        {
            "model": "claude-3-5-haiku",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 4, "output_tokens": 6},
        }
    )


# --------------------------------------------------------------------------------------
# OpenAIChatClient —— 请求体
# --------------------------------------------------------------------------------------


class OpenAIRequestTests(unittest.TestCase):
    def test_endpoint_and_headers(self) -> None:
        client, _ = _openai_client()
        self.assertEqual("https://api.openai.com/v1/chat/completions", client._endpoint())
        headers = client._headers("sk-test")
        self.assertEqual("application/json", headers["Content-Type"])
        self.assertEqual("Bearer sk-test", headers["Authorization"])

    def test_extra_headers_are_merged(self) -> None:
        client, _ = _openai_client(extra_headers={"X-Trace": "1"})
        self.assertEqual("1", client._headers("k")["X-Trace"])

    def test_message_conversion_for_tool_and_assistant_messages(self) -> None:
        client, _ = _openai_client()
        messages = [
            Message.system("sys"),
            Message.user("hi"),
            Message.assistant("", tool_calls=[ToolCall(id="call_1", name="add", arguments={"b": 2, "a": 1})]),
            Message(role=Role.TOOL, content="3", name="add", tool_call_id="call_1"),
        ]
        body = client._build_body(
            messages,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                # arguments 必须是**字符串**，且键序稳定（sort_keys）。
                                "arguments": json.dumps(
                                    {"b": 2, "a": 1}, ensure_ascii=False, sort_keys=True
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ],
            body["messages"],
        )
        self.assertEqual("gpt-4o-mini", body["model"])

    def test_tools_are_normalized_to_the_nested_function_form(self) -> None:
        client, _ = _openai_client()
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        body = client._build_body(
            [Message.user("hi")],
            tools=[{"name": "add", "description": "add two ints", "parameters": schema}],
            tool_choice="auto",
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": schema,
                        "description": "add two ints",
                    },
                }
            ],
            body["tools"],
        )
        self.assertEqual("auto", body["tool_choice"])

    def test_tool_normalization_is_idempotent(self) -> None:
        client, _ = _openai_client()
        already = {"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}
        body = client._build_body(
            [Message.user("hi")],
            tools=[already],
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual([already], body["tools"])

    def test_tool_without_parameters_gets_an_empty_object_schema(self) -> None:
        client, _ = _openai_client()
        body = client._build_body(
            [Message.user("hi")],
            tools=[{"name": "ping"}],
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(
            {"type": "object", "properties": {}},
            body["tools"][0]["function"]["parameters"],
        )
        self.assertNotIn("description", body["tools"][0]["function"])

    def test_tool_choice_is_only_sent_alongside_tools(self) -> None:
        client, _ = _openai_client()
        body = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice="auto",
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)

    def test_optional_fields_are_only_included_when_set(self) -> None:
        client, _ = _openai_client()
        minimal = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual({"model", "messages"}, set(minimal))

        full = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice=None,
            temperature=0.25,
            max_tokens=128,
            stop=["END"],
            stream=True,
        )
        self.assertEqual(0.25, full["temperature"])
        self.assertEqual(128, full["max_tokens"])
        self.assertEqual(["END"], full["stop"])
        self.assertTrue(full["stream"])

    def test_extra_body_is_merged_last(self) -> None:
        client, _ = _openai_client(extra_body={"top_p": 0.9})
        body = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(0.9, body["top_p"])


# --------------------------------------------------------------------------------------
# OpenAIChatClient —— 响应解析
# --------------------------------------------------------------------------------------


class OpenAIResponseTests(unittest.TestCase):
    def test_parses_content_usage_and_finish_reason(self) -> None:
        client, _ = _openai_client()
        response = client._parse_response(json.loads(_openai_response("hello")), latency_ms=12.5)
        self.assertEqual("hello", response.content)
        self.assertEqual([], response.tool_calls)
        self.assertEqual("stop", response.finish_reason)
        self.assertEqual(11, response.usage.prompt_tokens)
        self.assertEqual(7, response.usage.completion_tokens)
        self.assertEqual(18, response.usage.total_tokens)
        self.assertEqual("gpt-4o-mini", response.model)
        self.assertEqual("gpt-4o-mini", response.usage.model_hint)
        self.assertEqual(12.5, response.latency_ms)

    def test_parses_tool_calls_with_json_string_arguments(self) -> None:
        client, _ = _openai_client()
        payload = json.loads(
            _openai_response(
                None,
                tool_calls=[
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    }
                ],
                finish_reason=None,
            )
        )
        response = client._parse_response(payload, latency_ms=0.0)
        self.assertEqual(1, len(response.tool_calls))
        call = response.tool_calls[0]
        self.assertEqual("call_42", call.id)
        self.assertEqual("add", call.name)
        self.assertEqual({"a": 1, "b": 2}, call.arguments)
        self.assertNotIn("parse_error", call.metadata)
        # finish_reason 缺失但有 tool_calls -> 自动推断为 "tool_calls"。
        self.assertEqual("tool_calls", response.finish_reason)
        self.assertEqual("", response.content)

    def test_illegal_json_arguments_become_raw_without_raising(self) -> None:
        """§6.4 冻结：**绝不在这里抛异常**（否则模型失去自纠正机会）。"""
        client, _ = _openai_client()
        payload = json.loads(
            _openai_response(
                None,
                tool_calls=[
                    {
                        "id": "call_7",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{not json}"},
                    }
                ],
            )
        )
        response = client._parse_response(payload, latency_ms=0.0)
        call = response.tool_calls[0]
        self.assertEqual({"__raw__": "{not json}"}, call.arguments)
        self.assertEqual("{not json}", call.raw_arguments)
        self.assertIn("parse_error", call.metadata)
        self.assertIsInstance(call.metadata["parse_error"], str)
        self.assertTrue(call.metadata["parse_error"])

    def test_json_array_arguments_also_become_raw(self) -> None:
        """合法 JSON 但不是对象（"[1,2]"）同样走 ``__raw__`` 路径。"""
        client, _ = _openai_client()
        payload = json.loads(
            _openai_response(
                None,
                tool_calls=[
                    {"id": "c", "type": "function", "function": {"name": "add", "arguments": "[1, 2]"}}
                ],
            )
        )
        call = client._parse_response(payload, latency_ms=0.0).tool_calls[0]
        self.assertEqual({"__raw__": "[1, 2]"}, call.arguments)
        self.assertIn("parse_error", call.metadata)

    def test_arguments_already_a_mapping_is_accepted(self) -> None:
        """少数兼容实现直接回 dict 而不是字符串（§6.4 一并接受）。"""
        client, _ = _openai_client()
        payload = json.loads(
            _openai_response(
                None,
                tool_calls=[
                    {"id": "c", "type": "function", "function": {"name": "add", "arguments": {"a": 1}}}
                ],
            )
        )
        call = client._parse_response(payload, latency_ms=0.0).tool_calls[0]
        self.assertEqual({"a": 1}, call.arguments)
        self.assertNotIn("parse_error", call.metadata)

    def test_missing_choices_raises_response_format_error(self) -> None:
        client, _ = _openai_client()
        with self.assertRaises(LLMResponseFormatError):
            client._parse_response({"model": "x"}, latency_ms=0.0)
        with self.assertRaises(LLMResponseFormatError):
            client._parse_response({"choices": []}, latency_ms=0.0)

    def test_missing_message_object_raises_response_format_error(self) -> None:
        client, _ = _openai_client()
        with self.assertRaises(LLMResponseFormatError):
            client._parse_response({"choices": [{"finish_reason": "stop"}]}, latency_ms=0.0)

    def test_null_content_is_normalized_to_empty_string(self) -> None:
        client, _ = _openai_client()
        response = client._parse_response(json.loads(_openai_response(None)), latency_ms=0.0)
        self.assertEqual("", response.content)


# --------------------------------------------------------------------------------------
# OpenAIChatClient —— 端到端 achat（FakeTransport）
# --------------------------------------------------------------------------------------


class OpenAIAchatTests(unittest.IsolatedAsyncioTestCase):
    async def test_achat_sends_the_built_body_and_returns_the_parsed_response(self) -> None:
        client, transport = _openai_client(
            [HTTPResponse(status_code=200, text=_openai_response("pong"))]
        )
        response = await client.achat([Message.user("ping")], tools=None, temperature=0.1)

        self.assertEqual("pong", response.content)
        self.assertEqual(1, len(transport.requests))
        request = transport.requests[0]
        self.assertEqual("https://api.openai.com/v1/chat/completions", request.url)
        self.assertEqual("POST", request.method)
        self.assertEqual("Bearer sk-test", request.headers["Authorization"])
        self.assertEqual("gpt-4o-mini", request.json_body["model"])
        # temperature 参数 > config.temperature（§6.3 `_resolve_params`）。
        self.assertEqual(0.1, request.json_body["temperature"])

    async def test_unknown_kwargs_are_passed_through_into_the_body(self) -> None:
        client, transport = _openai_client([HTTPResponse(status_code=200, text=_openai_response("ok"))])
        await client.achat([Message.user("hi")], top_p=0.5, response_format={"type": "json_object"})
        body = transport.requests[0].json_body
        self.assertEqual(0.5, body["top_p"])
        self.assertEqual({"type": "json_object"}, body["response_format"])

    async def test_missing_api_key_raises_auth_error_before_any_request(self) -> None:
        transport = FakeTransport([HTTPResponse(status_code=200, text="{}")])
        client = OpenAIChatClient(
            LLMConfig(provider="openai", api_key=None), transport=transport
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMAuthError) as ctx:
                await client.achat([Message.user("hi")])
        self.assertIn("missing API key", str(ctx.exception))
        self.assertEqual([], transport.requests)

    async def test_retry_policy_defaults_do_not_send_a_temperature(self) -> None:
        client, transport = _openai_client([HTTPResponse(status_code=200, text=_openai_response("ok"))])
        await client.achat([Message.user("hi")])
        self.assertNotIn("temperature", transport.requests[0].json_body)

    async def test_aclose_closes_only_the_injected_transport(self) -> None:
        client, transport = _openai_client()
        with mock.patch.object(transport, "close") as close:
            await client.aclose()
        close.assert_called_once_with()
        # 注入的 transport 由 client 拥有；进程级默认 transport 不在这里被关（注释见实现）。
        self.assertIs(transport, client.transport)

    async def test_events_are_emitted_for_a_successful_call(self) -> None:
        events: list[tuple[str, dict]] = []
        transport = FakeTransport([HTTPResponse(status_code=200, text=_openai_response("ok"))])
        client = OpenAIChatClient(
            LLMConfig(provider="openai", api_key="k"),
            transport=transport,
            on_event=lambda name, data: events.append((name, data)),
        )
        await client.achat([Message.user("hi")])
        names = [name for name, _ in events]
        self.assertEqual(["llm_request", "llm_response"], names)
        response_data = events[1][1]
        self.assertEqual(2, response_data["content_len"])
        self.assertEqual(0, response_data["tool_calls"])
        self.assertEqual("stop", response_data["finish_reason"])
        # 事件 data 里不得出现 dataclass 实例（必须已 to_jsonable）。
        self.assertEqual(
            {"prompt_tokens", "completion_tokens", "total_tokens"},
            set(response_data["usage"]),
        )
        self.assertEqual(1, events[0][1]["messages_count"])


# --------------------------------------------------------------------------------------
# OpenAICompatibleClient / DeepSeekChatClient
# --------------------------------------------------------------------------------------


class BaseClientBehaviourTests(unittest.IsolatedAsyncioTestCase):
    """``BaseLLMClient`` / ``HTTPChatClient`` 的横切行为（§6.3）。"""

    def test_count_tokens_handles_empty_and_cjk(self) -> None:
        client, _ = _openai_client()
        self.assertEqual(0, client.count_tokens(""))
        self.assertEqual(2, client.count_tokens("中文"))  # CJK 每字 1 token
        self.assertEqual(5, client.count_tokens("中文字中文"))  # 5 个 CJK 字符
        self.assertEqual(2, client.count_tokens("abcdefgh"))  # ASCII 4 字符 1 token

    def test_count_tokens_is_cached_per_text(self) -> None:
        """同一条文本只算一次（ReAct 里 system prompt 与历史前缀每轮几乎不变）。"""
        client, _ = _openai_client()
        with mock.patch("liteagent.llm.base._heuristic_tokens", return_value=7) as patched:
            self.assertEqual(7, client.count_tokens("some long prompt"))
            self.assertEqual(7, client.count_tokens("some long prompt"))
            self.assertEqual(1, patched.call_count)
        # 空串不进缓存、直接短路（不调用估算函数）。
        with mock.patch("liteagent.llm.base._heuristic_tokens") as patched:
            self.assertEqual(0, client.count_tokens(""))
            patched.assert_not_called()

    def test_resolve_mode_depends_on_tool_support_and_has_tools(self) -> None:
        client, _ = _openai_client()
        self.assertEqual("native", client.resolve_mode(has_tools=True))
        self.assertEqual("text", client.resolve_mode(has_tools=False))

    async def test_chat_inside_a_running_loop_raises_config_error(self) -> None:
        """§5.2：同步入口在运行中的 loop 内必须抛 ``ConfigError``，不得套娃。"""
        client, _ = _openai_client()
        with self.assertRaises(ConfigError):
            client.chat([Message.user("hi")])

    async def test_non_object_json_payload_raises_response_format_error(self) -> None:
        client, _ = _openai_client([HTTPResponse(status_code=200, text="[]")])
        with self.assertRaises(LLMResponseFormatError):
            await client.achat([Message.user("hi")])

    async def test_usage_defaults_to_zero_when_absent(self) -> None:
        client, _ = _openai_client()
        payload = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        usage = client._parse_response(payload, latency_ms=0.0).usage
        self.assertEqual(0, usage.prompt_tokens)
        self.assertEqual(0, usage.completion_tokens)
        self.assertEqual(0, usage.total_tokens)
        self.assertTrue(usage.is_empty())

    async def test_config_temperature_is_used_when_the_caller_passes_none(self) -> None:
        client, transport = _openai_client(
            [HTTPResponse(status_code=200, text=_openai_response("ok"))], temperature=0.7
        )
        await client.achat([Message.user("hi")])
        self.assertEqual(0.7, transport.requests[0].json_body["temperature"])

    async def test_config_stream_is_ignored_with_a_warning(self) -> None:
        """降级必须留可观测痕迹（§13 红线 12），而不是静默按阻塞式返回。"""
        client, transport = _openai_client(
            [HTTPResponse(status_code=200, text=_openai_response("ok"))], stream=True
        )
        with self.assertWarns(RuntimeWarning):
            response = await client.achat([Message.user("hi")])
        self.assertEqual("ok", response.content)
        self.assertNotIn("stream", transport.requests[0].json_body)

    def test_resolve_params_prefers_explicit_values_over_config(self) -> None:
        client, _ = _openai_client(temperature=0.1, max_tokens=8)
        self.assertEqual((0.5, 32), client._resolve_params(0.5, 32))
        self.assertEqual((0.1, 8), client._resolve_params(None, None))


class CompatibleAndDeepSeekTests(unittest.TestCase):
    def test_openai_compatible_requires_an_explicit_base_url(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            OpenAICompatibleClient(LLMConfig(provider="openai-compatible", api_key="k"))
        self.assertIn("requires an explicit base_url", str(ctx.exception))

    def test_openai_compatible_does_not_inherit_the_openai_default(self) -> None:
        self.assertEqual("", OpenAICompatibleClient.default_base_url)

    def test_openai_compatible_uses_the_given_base_url(self) -> None:
        client = OpenAICompatibleClient(
            LLMConfig(
                provider="openai-compatible", api_key="k", base_url="http://localhost:8000/v1"
            )
        )
        self.assertEqual("http://localhost:8000/v1/chat/completions", client._endpoint())

    def test_deepseek_default_base_url(self) -> None:
        self.assertEqual("https://api.deepseek.com/v1", DeepSeekChatClient.default_base_url)
        self.assertEqual("deepseek", DeepSeekChatClient.name)

    def test_deepseek_needs_no_explicit_base_url(self) -> None:
        client = DeepSeekChatClient(LLMConfig(provider="deepseek", api_key="k"))
        self.assertEqual("https://api.deepseek.com/v1/chat/completions", client._endpoint())

    def test_deepseek_reuses_the_openai_codec(self) -> None:
        self.assertTrue(issubclass(DeepSeekChatClient, OpenAIChatClient))
        client = DeepSeekChatClient(LLMConfig(provider="deepseek", api_key="k"))
        response = client._parse_response(json.loads(_openai_response("hi")), latency_ms=0.0)
        self.assertEqual("hi", response.content)


# --------------------------------------------------------------------------------------
# AnthropicChatClient —— 请求体
# --------------------------------------------------------------------------------------


class AnthropicRequestTests(unittest.TestCase):
    def test_endpoint_and_headers(self) -> None:
        client, _ = _anthropic_client()
        self.assertEqual("https://api.anthropic.com/messages", client._endpoint())
        headers = client._headers("sk-ant")
        self.assertEqual("sk-ant", headers["x-api-key"])
        self.assertEqual("2023-06-01", headers["anthropic-version"])
        self.assertEqual("application/json", headers["Content-Type"])
        # Anthropic 不用 Bearer 前缀（api_key_prefix 为空串）。
        self.assertEqual("", client.api_key_prefix)

    def test_system_messages_are_extracted_to_the_top_level_field(self) -> None:
        client, _ = _anthropic_client()
        body = client._build_body(
            [
                Message.system("first"),
                Message.system("second"),
                Message.user("hi"),
            ],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual("first\n\nsecond", body["system"])
        self.assertEqual([{"role": "user", "content": "hi"}], body["messages"])
        for message in body["messages"]:
            self.assertNotEqual("system", message["role"])

    def test_system_key_is_absent_when_there_is_no_system_message(self) -> None:
        client, _ = _anthropic_client()
        body = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertNotIn("system", body)

    def test_consecutive_tool_results_merge_into_one_user_message(self) -> None:
        client, _ = _anthropic_client()
        messages = [
            Message.user("do it"),
            Message.assistant(
                "",
                tool_calls=[
                    ToolCall(id="tu_1", name="add", arguments={"a": 1}),
                    ToolCall(id="tu_2", name="add", arguments={"a": 2}),
                ],
            ),
            Message(role=Role.TOOL, content="1", name="add", tool_call_id="tu_1"),
            Message(role=Role.TOOL, content="2", name="add", tool_call_id="tu_2"),
        ]
        body = client._build_body(
            messages,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(3, len(body["messages"]))
        merged = body["messages"][-1]
        self.assertEqual("user", merged["role"])
        self.assertEqual(
            [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "1"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "2"},
            ],
            merged["content"],
        )

    def test_a_non_tool_message_breaks_the_tool_result_merge(self) -> None:
        client, _ = _anthropic_client()
        messages = [
            Message(role=Role.TOOL, content="1", name="add", tool_call_id="tu_1"),
            Message.user("interrupt"),
            Message(role=Role.TOOL, content="2", name="add", tool_call_id="tu_2"),
        ]
        body = client._build_body(
            messages,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        roles = [message["role"] for message in body["messages"]]
        self.assertEqual(["user", "user", "user"], roles)
        self.assertEqual(3, len(body["messages"]))

    def test_assistant_tool_calls_become_tool_use_blocks(self) -> None:
        client, _ = _anthropic_client()
        body = client._build_body(
            [
                Message.assistant(
                    "thinking",
                    tool_calls=[ToolCall(id="tu_1", name="add", arguments={"a": 1, "b": 2})],
                )
            ],
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "thinking"},
                        {"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 1, "b": 2}},
                    ],
                }
            ],
            body["messages"],
        )

    def test_tools_use_input_schema_shape(self) -> None:
        client, _ = _anthropic_client()
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        body = client._build_body(
            [Message.user("hi")],
            tools=[{"type": "function", "function": {"name": "add", "description": "d", "parameters": schema}}],
            tool_choice="required",
            temperature=None,
            max_tokens=None,
            stop=None,
            stream=False,
        )
        self.assertEqual(
            [{"name": "add", "input_schema": schema, "description": "d"}], body["tools"]
        )
        self.assertEqual({"type": "any"}, body["tool_choice"])

    def test_tool_choice_none_and_auto_conversion(self) -> None:
        client, _ = _anthropic_client()
        for value, expected in (("none", None), ("auto", {"type": "auto"}), (None, None)):
            with self.subTest(value=value):
                body = client._build_body(
                    [Message.user("hi")],
                    tools=[{"name": "add"}],
                    tool_choice=value,
                    temperature=None,
                    max_tokens=None,
                    stop=None,
                    stream=False,
                )
                if expected is None:
                    self.assertNotIn("tool_choice", body)
                else:
                    self.assertEqual(expected, body["tool_choice"])

    def test_max_tokens_is_required_so_a_default_is_supplied(self) -> None:
        client, _ = _anthropic_client()
        messages = [Message.user("hi")]
        default_body = client._build_body(
            messages, tools=None, tool_choice=None, temperature=None,
            max_tokens=None, stop=None, stream=False,
        )
        self.assertEqual(1024, default_body["max_tokens"])
        explicit_body = client._build_body(
            messages, tools=None, tool_choice=None, temperature=None,
            max_tokens=64, stop=None, stream=False,
        )
        self.assertEqual(64, explicit_body["max_tokens"])

    def test_stop_sequences_and_stream_keys(self) -> None:
        client, _ = _anthropic_client()
        body = client._build_body(
            [Message.user("hi")],
            tools=None,
            tool_choice=None,
            temperature=0.3,
            max_tokens=None,
            stop=["END"],
            stream=True,
        )
        self.assertEqual(["END"], body["stop_sequences"])
        self.assertTrue(body["stream"])
        self.assertEqual(0.3, body["temperature"])


# --------------------------------------------------------------------------------------
# AnthropicChatClient —— 响应解析
# --------------------------------------------------------------------------------------


class AnthropicResponseTests(unittest.TestCase):
    def test_text_blocks_are_concatenated(self) -> None:
        client, _ = _anthropic_client()
        payload = json.loads(
            _anthropic_ok(
                [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ]
            )
        )
        response = client._parse_response(payload, latency_ms=1.0)
        self.assertEqual("hello world", response.content)
        self.assertEqual([], response.tool_calls)
        self.assertEqual("stop", response.finish_reason)

    def test_tool_use_blocks_become_tool_calls(self) -> None:
        client, _ = _anthropic_client()
        payload = json.loads(
            _anthropic_ok(
                [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "tu_9", "name": "add", "input": {"a": 1, "b": 2}},
                ],
                stop_reason="tool_use",
            )
        )
        response = client._parse_response(payload, latency_ms=0.0)
        self.assertEqual("calling", response.content)
        self.assertEqual("tool_calls", response.finish_reason)
        self.assertEqual(1, len(response.tool_calls))
        self.assertEqual("tu_9", response.tool_calls[0].id)
        self.assertEqual({"a": 1, "b": 2}, response.tool_calls[0].arguments)

    def test_tool_use_with_non_object_input_falls_back_to_raw(self) -> None:
        client, _ = _anthropic_client()
        payload = json.loads(
            _anthropic_ok(
                [{"type": "tool_use", "id": "tu_1", "name": "add", "input": "{oops"}],
                stop_reason="tool_use",
            )
        )
        call = client._parse_response(payload, latency_ms=0.0).tool_calls[0]
        self.assertEqual({"__raw__": "{oops"}, call.arguments)
        self.assertIn("parse_error", call.metadata)

    def test_stop_reason_mapping_table(self) -> None:
        client, _ = _anthropic_client()
        cases = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "pause_turn": "stop",
            "something_new": "stop",
            None: "stop",
        }
        for stop_reason, expected in cases.items():
            with self.subTest(stop_reason=stop_reason):
                payload = json.loads(_anthropic_ok([{"type": "text", "text": "x"}], stop_reason=stop_reason))
                response = client._parse_response(payload, latency_ms=0.0)
                self.assertEqual(expected, response.finish_reason)

    def test_max_tokens_with_empty_content_still_reports_length(self) -> None:
        """§6.4 规则 8 的 MUST：否则 §9.4.6 的截断续写分支永远不会被触发。"""
        client, _ = _anthropic_client()
        payload = json.loads(_anthropic_ok([], stop_reason="max_tokens"))
        response = client._parse_response(payload, latency_ms=0.0)
        self.assertEqual("", response.content)
        self.assertEqual("length", response.finish_reason)

    def test_tool_use_wins_over_other_stop_reasons(self) -> None:
        client, _ = _anthropic_client()
        payload = json.loads(
            _anthropic_ok(
                [{"type": "tool_use", "id": "tu_1", "name": "add", "input": {}}],
                stop_reason="end_turn",
            )
        )
        self.assertEqual("tool_calls", client._parse_response(payload, latency_ms=0.0).finish_reason)

    def test_usage_input_output_maps_to_prompt_completion(self) -> None:
        client, _ = _anthropic_client()
        payload = json.loads(
            _anthropic_ok([{"type": "text", "text": "x"}], usage={"input_tokens": 30, "output_tokens": 12})
        )
        usage = client._parse_response(payload, latency_ms=0.0).usage
        self.assertEqual(30, usage.prompt_tokens)
        self.assertEqual(12, usage.completion_tokens)
        self.assertEqual(42, usage.total_tokens)
        self.assertEqual("claude-3-5-haiku", usage.model_hint)

    def test_missing_content_blocks_raises_response_format_error(self) -> None:
        client, _ = _anthropic_client()
        with self.assertRaises(LLMResponseFormatError):
            client._parse_response({"stop_reason": "end_turn"}, latency_ms=0.0)


class AnthropicAchatTests(unittest.IsolatedAsyncioTestCase):
    async def test_achat_extracts_system_and_sends_the_body(self) -> None:
        client, transport = _anthropic_client(
            [HTTPResponse(status_code=200, text=_anthropic_ok([{"type": "text", "text": "hi there"}]))]
        )
        response = await client.achat(
            [Message.system("be brief"), Message.user("hello")]
        )
        self.assertEqual("hi there", response.content)
        body = transport.requests[0].json_body
        self.assertEqual("be brief", body["system"])
        self.assertEqual([{"role": "user", "content": "hello"}], body["messages"])
        self.assertEqual("sk-ant", transport.requests[0].headers["x-api-key"])
        self.assertEqual("2023-06-01", transport.requests[0].headers["anthropic-version"])


# --------------------------------------------------------------------------------------
# EchoLLM
# --------------------------------------------------------------------------------------


class EchoLLMTests(unittest.IsolatedAsyncioTestCase):
    def test_class_flags(self) -> None:
        self.assertEqual("echo", EchoLLM.name)
        self.assertTrue(EchoLLM.supports_tool_calling)
        self.assertFalse(EchoLLM.requires_api_key)

    def _tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                },
            }
        ]

    async def test_use_directive_produces_a_tool_call_that_executes_to_three(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo"))
        response = await llm.achat(
            [Message.user('use:add {"a":1,"b":2}')], tools=self._tools()
        )
        self.assertEqual("tool_calls", response.finish_reason)
        self.assertEqual(1, len(response.tool_calls))
        call = response.tool_calls[0]
        self.assertEqual("add", call.name)
        self.assertEqual({"a": 1, "b": 2}, call.arguments)
        # 用真实工具执行一次，证明参数确实可被 function calling 消费。
        self.assertEqual(3, add(**call.arguments))

    async def test_use_without_arguments_yields_an_empty_argument_dict(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo"))
        response = await llm.achat([Message.user("use:add")], tools=self._tools())
        self.assertEqual({}, response.tool_calls[0].arguments)
        self.assertNotIn("parse_error", response.tool_calls[0].metadata)

    async def test_use_with_illegal_json_keeps_raw_and_records_the_error(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo"))
        response = await llm.achat([Message.user("use:add {not json")], tools=self._tools())
        call = response.tool_calls[0]
        self.assertEqual({}, call.arguments)
        self.assertIn("parse_error", call.metadata)
        self.assertEqual("{not json", call.raw_arguments)

    async def test_use_is_ignored_when_no_tools_are_given(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo"))
        response = await llm.achat([Message.user("use:add")], tools=None)
        self.assertEqual([], response.tool_calls)
        self.assertEqual("stop", response.finish_reason)

    async def test_content_reports_message_count_and_last_user_prefix(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo"))
        response = await llm.achat(
            [Message.system("sys"), Message.user("x" * 300)]
        )
        self.assertIn("received 2 message(s)", response.content)
        self.assertIn("x" * 200, response.content)
        self.assertNotIn("x" * 201, response.content)
        self.assertEqual("stop", response.finish_reason)

    async def test_usage_is_non_zero_and_model_hint_matches(self) -> None:
        llm = EchoLLM(LLMConfig(provider="echo", model="echo-1"))
        response = await llm.achat([Message.user("hello world")])
        self.assertEqual("echo-1", response.model)
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)
        self.assertEqual(
            response.usage.prompt_tokens + response.usage.completion_tokens,
            response.usage.total_tokens,
        )


# --------------------------------------------------------------------------------------
# 模块级注册表（冻结）
# --------------------------------------------------------------------------------------


class ProviderClassesTests(unittest.TestCase):
    def test_frozen_provider_class_table(self) -> None:
        self.assertEqual(
            {"openai", "openai-compatible", "deepseek", "anthropic", "echo"},
            set(PROVIDER_CLASSES),
        )
        self.assertIs(OpenAIChatClient, PROVIDER_CLASSES["openai"])
        self.assertIs(DeepSeekChatClient, PROVIDER_CLASSES["deepseek"])
        self.assertIs(AnthropicChatClient, PROVIDER_CLASSES["anthropic"])
        self.assertIs(EchoLLM, PROVIDER_CLASSES["echo"])

    def test_http_clients_share_the_common_base(self) -> None:
        for client_type in (OpenAIChatClient, OpenAICompatibleClient, DeepSeekChatClient, AnthropicChatClient):
            with self.subTest(client_type=client_type.__name__):
                self.assertTrue(issubclass(client_type, HTTPChatClient))
        # §6.4 [v2 新增]：真实 HTTP client 超时不重发。
        self.assertFalse(HTTPChatClient.retry_on_timeout)
        self.assertTrue(HTTPChatClient.supports_tool_calling)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
