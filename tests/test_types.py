from __future__ import annotations

"""``liteagent/types.py`` 的单元测试（§12 清单行：test_types.py）。

覆盖重点（§12 冻结清单，逐条对应到用例）：
* ``TokenUsage.__add__`` / ``total_tokens`` 自动求和
* ``ToolCall.canonical_key`` 稳定性
* ``ToolCall.try_from_arguments_json`` **不抛**且失败时带 ``__raw__``
* ``ToolResult.success`` / ``failure``（断言 ``failure().content.startswith("ERROR(")``）
* ``ToolResult.to_observation`` 在 ``ok=False`` 时**不带重复前缀**
* ``LLMResponse.to_message``
* ``from_dict`` 缺字段抛 ``SerializationError``

规范真值源：``docs/INTERFACES.md`` §4（694-876 行）、§3（错误字段表）。
硬性要求：只用 stdlib ``unittest``、禁用裸 ``assert``、不联网、不睡时钟。
本文件不需要任何 ``auto_register`` 夹具，因此不触碰默认注册表（§12.1 规则 1）。
"""

import inspect
import json
import unittest

from liteagent import config
from liteagent.errors import (
    LiteAgentError,
    ReActParseError,
    SerializationError,
    ToolExecutionError,
)
from liteagent.llm.message import Message, Role
from liteagent.types import LLMResponse, ScriptedCall, TokenUsage, ToolCall, ToolResult

# --------------------------------------------------------------------------------------
# 小夹具
# --------------------------------------------------------------------------------------


def _call(name: str = "read_file", arguments: dict | None = None) -> ToolCall:
    return ToolCall.create(name, arguments or {"path": "a.txt"})


# ======================================================================================
# 4.1 TokenUsage
# ======================================================================================


class TokenUsageTests(unittest.TestCase):
    """§4.1：``__post_init__`` 自动求和、``__add__`` / ``__iadd__`` 的语义。"""

    def test_default_construction_is_empty(self) -> None:
        usage = TokenUsage()
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)
        self.assertTrue(usage.is_empty())

    def test_post_init_autosums_total_when_only_parts_given(self) -> None:
        # §4.1：只给 prompt/completion 时 total 必须自动补齐，否则 max_total_tokens 预算永远看不到消耗。
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        self.assertEqual(usage.total_tokens, 15)

    def test_post_init_keeps_explicit_total(self) -> None:
        # 显式给的 total 不能被覆盖（provider 的 total 口径可能含缓存命中）。
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=99)
        self.assertEqual(usage.total_tokens, 99)

    def test_is_empty_false_when_any_counter_set(self) -> None:
        self.assertFalse(TokenUsage(prompt_tokens=1).is_empty())
        self.assertFalse(TokenUsage(completion_tokens=1).is_empty())
        # total_tokens=1 但 parts 都是 0：不是空（is_empty 判三个字段）
        self.assertFalse(TokenUsage(total_tokens=1).is_empty())

    def test_add_returns_new_object_and_sums_each_field(self) -> None:
        left = TokenUsage(prompt_tokens=10, completion_tokens=5)  # total 自动 = 15
        right = TokenUsage(prompt_tokens=3, completion_tokens=7)  # total 自动 = 10
        merged = left + right
        self.assertIsInstance(merged, TokenUsage)
        self.assertIsNot(merged, left)
        self.assertEqual(merged.prompt_tokens, 13)
        self.assertEqual(merged.completion_tokens, 12)
        # 逐项相加：15 + 10，而**不是**让 __post_init__ 重算成 25（此处恰好相等，见下一条）
        self.assertEqual(merged.total_tokens, 25)
        # 原对象未被修改（__add__ 不是 __iadd__）
        self.assertEqual(left.total_tokens, 15)

    def test_add_sums_totals_even_when_inconsistent_with_parts(self) -> None:
        # total 与 parts 不一致时（provider 口径差异），相加的仍然是 total 本身。
        left = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=100)
        right = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=7)
        merged = left + right
        self.assertEqual(merged.prompt_tokens, 2)
        self.assertEqual(merged.total_tokens, 107)

    def test_add_carries_model_hint(self) -> None:
        left = TokenUsage(prompt_tokens=1)
        right = TokenUsage(prompt_tokens=1, model_hint="gpt-4o-mini")
        self.assertEqual((left + right).model_hint, "gpt-4o-mini")
        # 左操作数已经有 hint 时以它为准（它是本次累加的上下文）
        self.assertEqual((right + left).model_hint, "gpt-4o-mini")

    def test_add_rejects_non_tokenusage(self) -> None:
        with self.assertRaises(TypeError):
            TokenUsage(prompt_tokens=1) + 5  # type: ignore[operator]

    def test_iadd_mutates_in_place(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)  # total 自动 = 15
        original_id = id(usage)
        usage += TokenUsage(prompt_tokens=1, completion_tokens=2)  # total 自动 = 3
        self.assertEqual(id(usage), original_id)
        self.assertEqual(usage.prompt_tokens, 11)
        self.assertEqual(usage.completion_tokens, 7)
        # 逐项相加：15 + 3（不是让 __post_init__ 重算成 18 之外的任何值）
        self.assertEqual(usage.total_tokens, 18)

    def test_iadd_fills_model_hint_once(self) -> None:
        usage = TokenUsage(prompt_tokens=1)
        usage += TokenUsage(prompt_tokens=1, model_hint="deepseek-chat")
        usage += TokenUsage(prompt_tokens=1, model_hint="other")
        self.assertEqual(usage.model_hint, "deepseek-chat")

    def test_to_dict_has_exactly_the_three_int_keys(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, model_hint="gpt-4o-mini")
        payload = usage.to_dict()
        self.assertEqual(
            set(payload), {"prompt_tokens", "completion_tokens", "total_tokens"}
        )
        # model_hint 刻意不输出（§4.1）
        self.assertNotIn("model_hint", payload)
        for value in payload.values():
            self.assertIsInstance(value, int)

    def test_to_dict_is_json_serializable(self) -> None:
        json.dumps(TokenUsage(prompt_tokens=1).to_dict())

    def test_from_dict_round_trip(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        self.assertEqual(TokenUsage.from_dict(usage.to_dict()), usage)

    def test_to_dict_drops_model_hint_so_round_trip_loses_it(self) -> None:
        # §4.1 冻结：to_dict 只输出三个 int，model_hint 不参与往返（这是刻意的 opt-out）。
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, model_hint="gpt-4o-mini")
        self.assertNotIn("model_hint", usage.to_dict())
        self.assertEqual(TokenUsage.from_dict(usage.to_dict()).model_hint, "")

    def test_from_dict_accepts_missing_model_hint(self) -> None:
        restored = TokenUsage.from_dict(
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        )
        self.assertEqual(restored.model_hint, "")

    def test_from_dict_missing_field_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError) as ctx:
            TokenUsage.from_dict({"prompt_tokens": 1, "completion_tokens": 2})
        self.assertIn("total_tokens", str(ctx.exception))
        self.assertIn("TokenUsage", ctx.exception.target)

    def test_from_dict_wrong_type_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError):
            TokenUsage.from_dict(
                {"prompt_tokens": "10", "completion_tokens": 2, "total_tokens": 12}
            )

    def test_from_dict_non_mapping_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError):
            TokenUsage.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_estimated_cost_usd_delegates_to_config_price_table(self) -> None:
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, model_hint="gpt-4o-mini")
        expected = config.estimate_cost_usd(usage, model="gpt-4o-mini")
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(usage.estimated_cost_usd, expected, places=12)

    def test_estimated_cost_usd_none_for_unknown_model(self) -> None:
        usage = TokenUsage(prompt_tokens=1000, model_hint="no-such-model")
        self.assertIsNone(usage.estimated_cost_usd)

    def test_estimated_cost_usd_none_without_model_hint(self) -> None:
        self.assertIsNone(TokenUsage(prompt_tokens=1000).estimated_cost_usd)


# ======================================================================================
# 4.2 ToolCall
# ======================================================================================


class ToolCallTests(unittest.TestCase):
    """§4.2：create / from_arguments_json / try_from_arguments_json / canonical_key / to_dict。"""

    def test_create_generates_call_prefix_id(self) -> None:
        call = ToolCall.create("read_file", {"path": "a.txt"})
        self.assertTrue(call.id.startswith("call_"), call.id)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, {"path": "a.txt"})
        self.assertEqual(call.raw_arguments, "")
        self.assertEqual(call.metadata, {})

    def test_create_honours_explicit_call_id(self) -> None:
        call = ToolCall.create("read_file", None, call_id="call_fixed")
        self.assertEqual(call.id, "call_fixed")
        self.assertEqual(call.arguments, {})

    def test_create_generates_distinct_ids(self) -> None:
        self.assertNotEqual(ToolCall.create("f").id, ToolCall.create("f").id)

    def test_create_does_not_share_argument_mapping(self) -> None:
        source = {"path": "a.txt"}
        call = ToolCall.create("read_file", source)
        source["path"] = "mutated"
        self.assertEqual(call.arguments["path"], "a.txt")

    def test_create_rejects_non_json_serializable_argument(self) -> None:
        with self.assertRaises(SerializationError) as ctx:
            ToolCall.create("f", {"bad": object()})
        self.assertIn("bad", str(ctx.exception))

    def test_create_rejects_non_str_argument_name(self) -> None:
        with self.assertRaises(SerializationError):
            ToolCall.create("f", {1: "x"})  # type: ignore[dict-item]

    def test_canonical_key_is_stable_under_key_order(self) -> None:
        left = ToolCall.create("read_file", {"b": 2, "a": 1})
        right = ToolCall.create("read_file", {"a": 1, "b": 2})
        self.assertEqual(left.canonical_key(), right.canonical_key())

    def test_canonical_key_is_stable_across_instances(self) -> None:
        args = {"path": "a.txt", "n": 3}
        first = ToolCall.create("read_file", dict(args)).canonical_key()
        second = ToolCall.create("read_file", dict(args)).canonical_key()
        self.assertEqual(first, second)

    def test_canonical_key_is_stable_under_nested_key_order(self) -> None:
        left = ToolCall.create("f", {"o": {"y": 1, "x": 2}})
        right = ToolCall.create("f", {"o": {"x": 2, "y": 1}})
        self.assertEqual(left.canonical_key(), right.canonical_key())

    def test_canonical_key_exact_format(self) -> None:
        call = ToolCall.create("read_file", {"a": 1, "b": 2})
        expected = "read_file:" + json.dumps(
            {"a": 1, "b": 2}, sort_keys=True, ensure_ascii=False, default=str
        )
        self.assertEqual(call.canonical_key(), expected)

    def test_canonical_key_differs_on_name_or_arguments(self) -> None:
        base = ToolCall.create("read_file", {"path": "a.txt"}).canonical_key()
        self.assertNotEqual(base, ToolCall.create("write_file", {"path": "a.txt"}).canonical_key())
        self.assertNotEqual(base, ToolCall.create("read_file", {"path": "b.txt"}).canonical_key())

    def test_canonical_key_keeps_non_ascii_verbatim(self) -> None:
        call = ToolCall.create("f", {"q": "读取"})
        self.assertIn("读取", call.canonical_key())

    def test_from_arguments_json_parses_object_and_keeps_raw(self) -> None:
        call = ToolCall.from_arguments_json("read_file", '{"path": "a.txt"}')
        self.assertEqual(call.arguments, {"path": "a.txt"})
        self.assertEqual(call.raw_arguments, '{"path": "a.txt"}')

    def test_from_arguments_json_invalid_json_raises_with_offset(self) -> None:
        with self.assertRaises(ReActParseError) as ctx:
            ToolCall.from_arguments_json("read_file", '{"path": ')
        self.assertIsInstance(ctx.exception, LiteAgentError)
        self.assertEqual(ctx.exception.raw, '{"path": ')
        self.assertGreaterEqual(ctx.exception.offset, 0)

    def test_from_arguments_json_non_object_raises(self) -> None:
        for raw in ("[1, 2]", "42", '"abc"'):
            with self.subTest(raw=raw):
                with self.assertRaises(ReActParseError):
                    ToolCall.from_arguments_json("f", raw)

    def test_try_from_arguments_json_returns_error_instead_of_raising(self) -> None:
        # §4.2 [v2]：provider 只用 try_ 版本 —— 解析失败绝不能抛，否则模型失去自纠正机会。
        call, error = ToolCall.try_from_arguments_json("read_file", '{"path": ')
        self.assertIsNotNone(error)
        self.assertIsInstance(error, str)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, {"__raw__": '{"path": '})
        self.assertEqual(call.raw_arguments, '{"path": ')

    def test_try_from_arguments_json_never_raises_for_garbage(self) -> None:
        for raw in ("", "not json", "[1,2]", "null", "42", "{'single': 1}", "\ud800"):
            with self.subTest(raw=raw):
                try:
                    call, error = ToolCall.try_from_arguments_json("f", raw)
                except Exception as exc:  # pragma: no cover - 失败时给出具体类型便于定位
                    self.fail(f"try_from_arguments_json raised {type(exc).__name__}: {exc}")
                self.assertIsInstance(call, ToolCall)
                self.assertIsInstance(error, str)

    def test_try_from_arguments_json_success_returns_none_error(self) -> None:
        call, error = ToolCall.try_from_arguments_json("read_file", '{"path": "a.txt"}')
        self.assertIsNone(error)
        self.assertEqual(call.arguments, {"path": "a.txt"})
        self.assertEqual(call.raw_arguments, '{"path": "a.txt"}')

    def test_try_from_arguments_json_honours_call_id_on_both_paths(self) -> None:
        ok_call, ok_error = ToolCall.try_from_arguments_json("f", "{}", call_id="call_fixed")
        self.assertIsNone(ok_error)
        self.assertEqual(ok_call.id, "call_fixed")
        bad_call, bad_error = ToolCall.try_from_arguments_json("f", "{bad", call_id="call_fixed")
        self.assertIsNotNone(bad_error)
        self.assertEqual(bad_call.id, "call_fixed")

    def test_to_dict_has_the_five_frozen_keys(self) -> None:
        call = ToolCall.create("read_file", {"path": "a.txt"})
        payload = call.to_dict()
        self.assertEqual(
            set(payload), {"id", "name", "arguments", "raw_arguments", "metadata"}
        )
        json.dumps(payload)

    def test_from_dict_round_trip(self) -> None:
        call = ToolCall.from_arguments_json("read_file", '{"path": "a.txt"}')
        restored = ToolCall.from_dict(call.to_dict())
        self.assertEqual(restored, call)

    def test_from_dict_missing_field_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError) as ctx:
            ToolCall.from_dict({"id": "call_1", "name": "f", "arguments": {}})
        self.assertIn("raw_arguments", str(ctx.exception))

    def test_from_dict_wrong_type_raises_serialization_error(self) -> None:
        payload = ToolCall.create("f").to_dict()
        payload["arguments"] = "not-a-mapping"
        with self.assertRaises(SerializationError):
            ToolCall.from_dict(payload)


# ======================================================================================
# 4.3 ToolResult
# ======================================================================================


class ToolResultTests(unittest.TestCase):
    """§4.3：success/failure、error_text 幂等、to_observation 不重复前缀、to_message。"""

    def test_success_fields(self) -> None:
        call = _call()
        result = ToolResult.success(call, "file body", duration_ms=1.5, attempts=2)
        self.assertEqual(result.call_id, call.id)
        self.assertEqual(result.name, call.name)
        self.assertEqual(result.content, "file body")
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_type)
        self.assertEqual(result.duration_ms, 1.5)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.metadata, {})

    def test_success_coerces_content_to_str(self) -> None:
        result = ToolResult.success(_call(), 42)  # type: ignore[arg-type]
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, "42")

    def test_success_copies_metadata(self) -> None:
        metadata = {"k": "v"}
        result = ToolResult.success(_call(), "x", metadata=metadata)
        metadata["k"] = "mutated"
        self.assertEqual(result.metadata, {"k": "v"})

    def test_failure_content_starts_with_error_prefix(self) -> None:
        # §12 明确要求的断言：failure().content 必须以 "ERROR(" 开头（模型靠它判断失败）。
        result = ToolResult.failure(_call(), ValueError("boom"))
        self.assertTrue(result.content.startswith("ERROR("), result.content)
        self.assertNotEqual(result.content, "")

    def test_failure_records_error_type_and_message(self) -> None:
        result = ToolResult.failure(_call(), ToolExecutionError("read_file", "call_1"))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolExecutionError")
        self.assertIsInstance(result.error, str)
        self.assertEqual(result.content, result.error_text())

    def test_failure_merges_liteagent_error_context(self) -> None:
        error = ToolExecutionError("read_file", "call_1", context={"step": 3})
        result = ToolResult.failure(_call(), error)
        self.assertEqual(result.metadata["error_context"], {"step": 3})
        # 必须是副本：不能让 trace 与异常对象共享同一个 dict（§13 红线 11）
        error.context["step"] = 99
        self.assertEqual(result.metadata["error_context"], {"step": 3})

    def test_failure_without_context_has_no_error_context_key(self) -> None:
        result = ToolResult.failure(_call(), ValueError("boom"))
        self.assertNotIn("error_context", result.metadata)
        result2 = ToolResult.failure(_call(), ToolExecutionError("f", "c"))
        self.assertNotIn("error_context", result2.metadata)

    def test_failure_preserves_caller_metadata(self) -> None:
        result = ToolResult.failure(
            _call(), ValueError("boom"), metadata={"attempts_left": 0}
        )
        self.assertEqual(result.metadata["attempts_left"], 0)

    def test_error_text_success_returns_content_verbatim(self) -> None:
        result = ToolResult.success(_call(), "ok body")
        self.assertEqual(result.error_text(), "ok body")

    def test_error_text_failure_is_idempotent(self) -> None:
        # 反例保护：第二次调用不能产出两遍前缀（§4.3 的 "否则错误文本会出现两遍"）。
        result = ToolResult.failure(_call(), ValueError("boom"))
        self.assertEqual(result.error_text(), result.content)
        self.assertEqual(result.error_text().count("ERROR("), 1)

    def test_error_text_failure_appends_extra_content_once(self) -> None:
        result = ToolResult(
            call_id="call_1",
            name="read_file",
            content="hint for the model",
            ok=False,
            error="boom",
            error_type="ToolExecutionError",
        )
        text = result.error_text()
        self.assertTrue(text.startswith("ERROR(ToolExecutionError): boom"))
        self.assertIn("hint for the model", text)
        self.assertEqual(text.count("ERROR("), 1)
        # 幂等：多次调用结果相同
        self.assertEqual(result.error_text(), text)

    def test_to_observation_failure_has_no_duplicate_prefix(self) -> None:
        # §4.3 [v2] 冻结：ok=False 时 to_observation **直接返回 content**。
        result = ToolResult.failure(_call(), ValueError("boom"))
        observation = result.to_observation()
        self.assertEqual(observation, result.content)
        self.assertEqual(observation.count("ERROR("), 1)
        self.assertEqual(observation.count("boom"), 1)

    def test_to_observation_success_short_text_untouched(self) -> None:
        result = ToolResult.success(_call(), "small")
        self.assertEqual(result.to_observation(max_chars=8000), "small")

    def test_to_observation_success_truncates_head_and_tail(self) -> None:
        content = "H" * 100 + "-" * 100 + "T" * 100  # 300 字符
        result = ToolResult.success(_call(), content)
        observation = result.to_observation(max_chars=100)
        head_chars = int(100 * 0.7)
        tail_chars = 100 - head_chars
        self.assertTrue(observation.startswith("H" * head_chars))
        self.assertTrue(observation.endswith("T" * tail_chars))
        self.assertIn(f"truncated {len(content) - 100} chars", observation)
        # 中间被丢弃
        self.assertNotIn("-" * 100, observation)

    def test_to_observation_success_zero_max_chars_returns_content(self) -> None:
        result = ToolResult.success(_call(), "x" * 100)
        self.assertEqual(result.to_observation(max_chars=0), "x" * 100)

    def test_to_observation_default_max_chars_is_synced_with_config(self) -> None:
        # §4.5：types.py 不能 import config，默认值用字面量 8000 + 注释标注同步位置。
        # 这条断言就是那份"注释即契约"的可执行版本。
        default = inspect.signature(ToolResult.to_observation).parameters["max_chars"].default
        self.assertEqual(default, config.DEFAULT_MAX_OBSERVATION_CHARS)

    def test_to_message_is_tool_role_with_tool_call_id(self) -> None:
        call = _call("read_file")
        result = ToolResult.success(call, "body")
        message = result.to_message()
        self.assertIsInstance(message, Message)
        self.assertEqual(message.role, Role.TOOL)
        self.assertEqual(message.content, "body")
        self.assertEqual(message.name, "read_file")
        self.assertEqual(message.tool_call_id, call.id)
        self.assertEqual(message.metadata, {"tool_name": "read_file"})

    def test_to_message_carries_error_text_on_failure(self) -> None:
        result = ToolResult.failure(_call(), ValueError("boom"))
        message = result.to_message()
        self.assertEqual(message.content, result.content)
        self.assertTrue(message.content.startswith("ERROR("))

    def test_to_dict_and_from_dict_round_trip(self) -> None:
        result = ToolResult.failure(
            _call(), ToolExecutionError("read_file", "call_1", context={"a": 1}), attempts=3
        )
        payload = result.to_dict()
        self.assertEqual(
            set(payload),
            {
                "call_id",
                "name",
                "content",
                "ok",
                "error",
                "error_type",
                "duration_ms",
                "attempts",
                "metadata",
            },
        )
        json.dumps(payload)
        self.assertEqual(ToolResult.from_dict(payload), result)

    def test_from_dict_requires_nullable_error_keys(self) -> None:
        # error / error_type 是"必需但可为 None"；漏写键与值为 None 不是一回事（§4.5 的守卫）。
        payload = ToolResult.success(_call(), "x").to_dict()
        del payload["error"]
        with self.assertRaises(SerializationError):
            ToolResult.from_dict(payload)

    def test_from_dict_missing_field_raises_serialization_error(self) -> None:
        payload = ToolResult.success(_call(), "x").to_dict()
        del payload["content"]
        with self.assertRaises(SerializationError) as ctx:
            ToolResult.from_dict(payload)
        self.assertIn("content", str(ctx.exception))

    def test_from_dict_accepts_null_error_fields(self) -> None:
        payload = ToolResult.success(_call(), "x").to_dict()
        self.assertIsNone(payload["error"])
        restored = ToolResult.from_dict(payload)
        self.assertIsNone(restored.error)
        self.assertIsNone(restored.error_type)


# ======================================================================================
# 4.4 LLMResponse
# ======================================================================================


class LLMResponseTests(unittest.TestCase):
    """§4.4：has_tool_calls / to_message / to_dict(include_raw) / from_dict。"""

    def test_defaults(self) -> None:
        response = LLMResponse()
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.finish_reason, "stop")
        self.assertTrue(response.usage.is_empty())
        self.assertEqual(response.model, "")
        self.assertIsNone(response.raw)
        self.assertEqual(response.latency_ms, 0.0)
        self.assertFalse(response.has_tool_calls)

    def test_has_tool_calls_true_when_tool_calls_present(self) -> None:
        response = LLMResponse(tool_calls=[_call()])
        self.assertTrue(response.has_tool_calls)

    def test_to_message_is_assistant_with_content_and_tool_calls(self) -> None:
        call = _call()
        response = LLMResponse(content="thinking", tool_calls=[call], finish_reason="tool_calls")
        message = response.to_message()
        self.assertIsInstance(message, Message)
        self.assertEqual(message.role, Role.ASSISTANT)
        self.assertEqual(message.content, "thinking")
        self.assertEqual(message.tool_calls, [call])

    def test_to_message_copies_the_tool_calls_list(self) -> None:
        # Message 会被长期持有；共享同一个 list 会让"谁改了它"不可追踪。
        response = LLMResponse(tool_calls=[_call()])
        message = response.to_message()
        message.tool_calls.append(_call("extra"))
        self.assertEqual(len(response.tool_calls), 1)

    def test_to_dict_hides_raw_by_default(self) -> None:
        response = LLMResponse(content="x", raw={"provider": "openai"})
        payload = response.to_dict()
        self.assertIn("raw", payload)
        self.assertIsNone(payload["raw"])
        self.assertEqual(payload["content"], "x")

    def test_to_dict_include_raw_true_keeps_body(self) -> None:
        response = LLMResponse(raw={"provider": "openai"})
        self.assertEqual(response.to_dict(include_raw=True)["raw"], {"provider": "openai"})

    def test_to_dict_is_json_serializable(self) -> None:
        response = LLMResponse(
            content="x",
            tool_calls=[_call()],
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
        )
        json.dumps(response.to_dict())

    def test_from_dict_round_trip(self) -> None:
        response = LLMResponse(
            content="x",
            tool_calls=[ToolCall.from_arguments_json("f", '{"a": 1}')],
            finish_reason="tool_calls",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            model="gpt-4o-mini",
            latency_ms=12.5,
        )
        restored = LLMResponse.from_dict(response.to_dict())
        self.assertEqual(restored, response)

    def test_from_dict_accepts_token_usage_instance(self) -> None:
        payload = LLMResponse().to_dict()
        payload["usage"] = TokenUsage(prompt_tokens=7)
        restored = LLMResponse.from_dict(payload)
        self.assertEqual(restored.usage.prompt_tokens, 7)

    def test_from_dict_missing_field_raises_serialization_error(self) -> None:
        payload = LLMResponse().to_dict()
        del payload["usage"]
        with self.assertRaises(SerializationError) as ctx:
            LLMResponse.from_dict(payload)
        self.assertIn("usage", str(ctx.exception))

    def test_from_dict_missing_content_raises_serialization_error(self) -> None:
        payload = LLMResponse().to_dict()
        del payload["content"]
        with self.assertRaises(SerializationError):
            LLMResponse.from_dict(payload)

    def test_from_dict_bad_tool_call_entry_raises_serialization_error(self) -> None:
        payload = LLMResponse().to_dict()
        payload["tool_calls"] = [{"name": "f"}]  # 缺 id / arguments / ...
        with self.assertRaises(SerializationError):
            LLMResponse.from_dict(payload)


# ======================================================================================
# 4.5 ScriptedCall
# ======================================================================================


class ScriptedCallTests(unittest.TestCase):
    """§4.5：to_dict 必须对任意 kwargs / tools 内容都可序列化（永不抛 TypeError）。"""

    def test_to_dict_shape(self) -> None:
        message = Message(role=Role.USER, content="hi")
        call = ScriptedCall(index=0, messages=[message], tools=None, kwargs={})
        payload = call.to_dict()
        self.assertEqual(
            set(payload), {"index", "messages", "tools", "kwargs", "response"}
        )
        self.assertEqual(payload["index"], 0)
        self.assertIsNone(payload["tools"])
        self.assertIsNone(payload["response"])
        self.assertEqual(payload["messages"][0]["content"], "hi")

    def test_to_dict_makes_exotic_kwargs_json_serializable(self) -> None:
        call = ScriptedCall(
            index=1,
            messages=[],
            tools=[{"type": "function", "extra": {"nested"}}],
            kwargs={"callable": lambda: None},
        )
        payload = call.to_dict()
        json.dumps(payload)  # 未知类型退化为 repr，绝不抛
        self.assertEqual(payload["index"], 1)


# ======================================================================================
# 跨数据类的公共约束
# ======================================================================================


class TypesModuleContractTests(unittest.TestCase):
    """§4 的模块级冻结约定（slots / 公开面 / 抛错类型）。"""

    def test_leaf_dataclasses_use_slots(self) -> None:
        # §4："slots=True 只用于叶子数据类" —— 临时挂属性会被立即拒绝（保护手误）。
        instances = (
            TokenUsage(prompt_tokens=1),
            ToolCall.create("f"),
            ToolResult.success(ToolCall.create("f"), "x"),
            ScriptedCall(index=0, messages=[], tools=None, kwargs={}),
        )
        for instance in instances:
            with self.subTest(cls=type(instance).__name__):
                self.assertFalse(hasattr(instance, "__dict__"))
                with self.assertRaises(AttributeError):
                    instance.temporary = 1  # type: ignore[attr-defined]

    def test_llm_response_does_not_use_slots(self) -> None:
        # §4：LLMResponse 刻意不用 slots（含 dict 默认值，provider 会补挂信息）。
        response = LLMResponse()
        response.temporary = 1  # type: ignore[attr-defined]
        self.assertEqual(response.temporary, 1)

    def test_public_exports(self) -> None:
        import liteagent.types as types_module

        self.assertEqual(
            set(types_module.__all__),
            {"TokenUsage", "ToolCall", "ToolResult", "LLMResponse", "ScriptedCall"},
        )

    def test_react_parse_error_is_a_liteagent_error(self) -> None:
        # types.py 只用 LiteAgentError 家族抛错（§3：任何层都能 import 它）。
        self.assertTrue(issubclass(ReActParseError, LiteAgentError))

    def test_serialization_error_carries_target(self) -> None:
        with self.assertRaises(SerializationError) as ctx:
            ToolCall.from_dict({"id": 1})
        self.assertTrue(ctx.exception.target.startswith("ToolCall"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
