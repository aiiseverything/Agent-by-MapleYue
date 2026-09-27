from __future__ import annotations

"""``liteagent/llm/message.py`` 的单元测试（§6.1）。

覆盖 §12 对 ``test_message.py`` 冻结的四个覆盖点：

1. ``Role.coerce``（大小写不敏感、未知值抛 ``SerializationError``、非 str/Role 也抛）；
2. ``to_dict`` / ``from_dict`` 往返（含 tool_calls / metadata / None 字段全量输出）；
3. ``drop_orphan_tool_messages`` 的**两种**修补（孤儿 tool 消息删除；无结果的
   assistant.tool_calls 剥离但消息保留）；
4. ``messages_tokens("") == 0``（空文本短路，与 ``HeuristicTokenizer("") == 0`` 对齐）。

另有若干条只依赖纯函数的补充断言（构造便捷方法、``text()``、``render_transcript``、
``is_tool_pair_start``）—— 它们都是 §6.1 明文冻结的行为，不引入任何不确定字段。
"""

import json
import unittest

from liteagent.errors import SerializationError
from liteagent.llm.message import (
    Message,
    Role,
    drop_orphan_tool_messages,
    messages_tokens,
    render_transcript,
)
from liteagent.types import ToolCall, ToolResult


def _tool_call(name: str = "add", arguments=None, *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments or {}))


class RoleCoerceTests(unittest.TestCase):
    """§6.1 ``Role.coerce`` 的冻结语义。"""

    def test_coerce_accepts_enum_and_mixed_case_strings(self) -> None:
        self.assertIs(Role.SYSTEM, Role.coerce(Role.SYSTEM))
        self.assertIs(Role.USER, Role.coerce("user"))
        self.assertIs(Role.USER, Role.coerce("User"))
        self.assertIs(Role.ASSISTANT, Role.coerce("  ASSISTANT  "))
        self.assertIs(Role.TOOL, Role.coerce(Role.TOOL.value))

    def test_coerce_rejects_unknown_string_with_serialization_error(self) -> None:
        with self.assertRaises(SerializationError) as ctx:
            Role.coerce("robot")
        error = ctx.exception
        self.assertEqual("Role", error.target)
        self.assertIn("robot", str(error))

    def test_coerce_rejects_non_string_types(self) -> None:
        for value in (None, 123, ["user"], {"role": "user"}):
            with self.subTest(value=value):
                with self.assertRaises(SerializationError) as ctx:
                    Role.coerce(value)
                self.assertEqual("Role", ctx.exception.target)


class MessageRoundTripTests(unittest.TestCase):
    """``to_dict`` / ``from_dict`` 往返（§6.1 / §2.2）。"""

    def test_from_dict_of_to_dict_is_identity(self) -> None:
        message = Message(
            role=Role.ASSISTANT,
            content="thinking",
            name="planner",
            tool_calls=[_tool_call("add", {"a": 1, "b": 2}, call_id="call_9")],
            tool_call_id=None,
            metadata={"kind": "assistant", "step": 2},
        )
        restored = Message.from_dict(message.to_dict())
        self.assertEqual(message, restored)
        self.assertEqual(Role.ASSISTANT, restored.role)
        self.assertEqual("call_9", restored.tool_calls[0].id)
        self.assertEqual({"a": 1, "b": 2}, restored.tool_calls[0].arguments)

    def test_to_dict_outputs_all_six_keys_and_never_omits_none(self) -> None:
        payload = Message.user("hi").to_dict()
        self.assertEqual(
            {"role", "content", "name", "tool_calls", "tool_call_id", "metadata"},
            set(payload),
        )
        self.assertEqual("user", payload["role"])
        self.assertIsNone(payload["name"])
        self.assertIsNone(payload["tool_call_id"])
        self.assertEqual([], payload["tool_calls"])

    def test_from_dict_tolerates_missing_optional_keys(self) -> None:
        restored = Message.from_dict({"role": "system", "content": "be nice"})
        self.assertEqual(Role.SYSTEM, restored.role)
        self.assertEqual([], restored.tool_calls)
        self.assertEqual({}, restored.metadata)
        self.assertEqual("be nice", restored.content)
        self.assertEqual([], restored.tool_calls)
        self.assertIsNone(restored.tool_call_id)

    def test_from_dict_rejects_wrong_types(self) -> None:
        with self.assertRaises(SerializationError):
            Message.from_dict({"role": "user", "content": 42})
        with self.assertRaises(SerializationError):
            Message.from_dict({"role": "user", "tool_calls": "not-a-list"})
        with self.assertRaises(SerializationError):
            Message.from_dict({"role": "user", "metadata": ["not", "a", "mapping"]})

    def test_from_dict_normalizes_null_content_to_empty_string(self) -> None:
        """provider 的 tool_calls-only 响应会把 content 写成 null（§6.1 注释）。"""
        restored = Message.from_dict({"role": "assistant", "content": None})
        self.assertEqual("", restored.content)


class MessageConstructorTests(unittest.TestCase):
    """构造便捷方法与 ``text()`` / ``is_tool_pair_start``。"""

    def test_factories_set_role_and_metadata(self) -> None:
        system = Message.system("sys", kind="system")
        self.assertEqual(Role.SYSTEM, system.role)
        self.assertEqual({"kind": "system"}, system.metadata)

        user = Message.user("hello")
        self.assertEqual(Role.USER, user.role)

        assistant = Message.assistant("ok", tool_calls=[_tool_call()])
        self.assertEqual(Role.ASSISTANT, assistant.role)
        self.assertEqual(1, len(assistant.tool_calls))

    def test_assistant_normalizes_none_tool_calls_to_empty_list(self) -> None:
        self.assertEqual([], Message.assistant("", tool_calls=None).tool_calls)
        self.assertFalse(Message.assistant("", tool_calls=None).is_tool_pair_start())

    def test_observation_is_a_user_message_with_kind_marker(self) -> None:
        observation = Message.observation("result", step=3, tool_name="add")
        self.assertEqual(Role.USER, observation.role)
        self.assertEqual(
            {"kind": "observation", "step": 3, "tool_name": "add"}, observation.metadata
        )
        # 缺省时 step / tool_name 不出现在 metadata 里（避免 trace 里出现 "step": null）。
        bare = Message.observation("result")
        self.assertEqual({"kind": "observation"}, bare.metadata)

    def test_message_tool_wraps_tool_result_to_message(self) -> None:
        result = ToolResult.success(_tool_call(), "5")
        message = Message.tool(result)
        self.assertEqual(Role.TOOL, message.role)
        self.assertEqual("call_1", message.tool_call_id)
        self.assertEqual("add", message.name)
        self.assertEqual("5", message.content)

    def test_text_concatenates_content_and_tool_calls(self) -> None:
        message = Message.assistant(
            "let me check", tool_calls=[_tool_call("add", {"b": 2, "a": 1})]
        )
        # canonical json 用 sort_keys=True，键序稳定 -> 可精确断言。
        self.assertEqual('let me check\nadd({"a": 1, "b": 2})', message.text())
        self.assertEqual("", Message.assistant("").text())

    def test_copy_replaces_fields_without_mutating_the_original(self) -> None:
        original = Message.user("a")
        copied = original.copy(content="b")
        self.assertEqual("a", original.content)
        self.assertEqual("b", copied.content)


class MessagesTokensTests(unittest.TestCase):
    """``messages_tokens``（§6.1 [v2 变更]：空文本 -> 0）。"""

    def test_empty_text_messages_count_as_zero(self) -> None:
        self.assertEqual(0, messages_tokens([]))
        self.assertEqual(0, messages_tokens([Message.assistant("")]))
        self.assertEqual(
            0,
            messages_tokens([Message.assistant("", tool_calls=[]), Message.user("")]),
        )

    def test_empty_text_short_circuits_even_with_an_external_counter(self) -> None:
        """空文本短路必须发生在**两条分支**上，否则外部 counter 会贡献非零值。"""
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return 7

        self.assertEqual(0, messages_tokens([Message.user("")], counter))
        self.assertEqual([], calls)

    def test_non_empty_text_uses_counter_or_ascii_approximation(self) -> None:
        messages = [Message.user("abcdefgh")]  # 8 字符
        self.assertEqual(2, messages_tokens(messages))  # ceil(8/4)
        self.assertEqual(99, messages_tokens(messages, lambda text: 99))

    def test_tool_call_only_message_is_counted(self) -> None:
        """tool_call 是模型可见的思考内容，只数 content 会低估。"""
        plain = messages_tokens([Message.assistant("")])
        with_call = messages_tokens([Message.assistant("", tool_calls=[_tool_call()])])
        self.assertEqual(0, plain)
        self.assertGreater(with_call, 0)


class DropOrphanToolMessagesTests(unittest.TestCase):
    """``drop_orphan_tool_messages`` 的两种修补（§6.1）。"""

    def test_removes_tool_message_without_matching_assistant_call(self) -> None:
        assistant = Message.assistant("", tool_calls=[_tool_call(call_id="call_1")])
        orphan = Message(
            role=Role.TOOL, content="ghost", name="add", tool_call_id="call_999"
        )
        kept = Message(role=Role.TOOL, content="5", name="add", tool_call_id="call_1")

        result = drop_orphan_tool_messages([assistant, orphan, kept])
        self.assertEqual([assistant, kept], result)

    def test_strips_tool_calls_without_matching_tool_result_but_keeps_message(self) -> None:
        assistant = Message.assistant(
            "thinking",
            tool_calls=[_tool_call(call_id="call_1"), _tool_call(call_id="call_2")],
        )
        only_first = Message(role=Role.TOOL, content="1", name="add", tool_call_id="call_1")

        result = drop_orphan_tool_messages([assistant, only_first])
        self.assertEqual(2, len(result))
        repaired = result[0]
        self.assertEqual("thinking", repaired.content)  # 消息本身保留
        self.assertEqual(["call_1"], [call.id for call in repaired.tool_calls])
        # 未改动的 tool 消息按原对象返回（BufferMemory 依赖 dataclass 相等性）。
        self.assertIs(only_first, result[1])

    def test_tool_message_without_any_id_is_treated_as_orphan(self) -> None:
        """无 id 的 tool 消息在任何 API 里都非法 -> 删除；同时它的缺席也会让
        assistant 的 tool_calls 变成"无结果"，因此那部分也一并剥离（两种修补同时生效）。"""
        assistant = Message.assistant("", tool_calls=[_tool_call(call_id="call_1")])
        idless = Message(role=Role.TOOL, content="x", name="add", tool_call_id=None)
        result = drop_orphan_tool_messages([assistant, idless])
        self.assertEqual(1, len(result))
        self.assertEqual([], result[0].tool_calls)

    def test_leaves_a_consistent_history_untouched(self) -> None:
        assistant = Message.assistant("", tool_calls=[_tool_call(call_id="call_1")])
        tool = Message(role=Role.TOOL, content="1", name="add", tool_call_id="call_1")
        messages = [Message.system("s"), Message.user("u"), assistant, tool]
        self.assertEqual(messages, drop_orphan_tool_messages(messages))

    def test_strips_all_calls_when_no_tool_results_exist(self) -> None:
        assistant = Message.assistant(
            "want to call", tool_calls=[_tool_call(call_id="call_1")]
        )
        result = drop_orphan_tool_messages([assistant])
        self.assertEqual(1, len(result))
        self.assertEqual([], result[0].tool_calls)
        self.assertEqual("want to call", result[0].content)


class RenderTranscriptTests(unittest.TestCase):
    """``render_transcript``（§6.1）：给摘要器用的纯文本渲染。"""

    def test_renders_roles_tool_prefixes_and_tool_call_lines(self) -> None:
        messages = [
            Message.system("sys"),
            Message.user("hello"),
            Message.assistant("", tool_calls=[_tool_call("add", {"a": 1})]),
            Message(role=Role.TOOL, content="1", name="add", tool_call_id="call_1"),
        ]
        rendered = render_transcript(messages)
        self.assertEqual(
            "\n".join(
                [
                    "[system] sys",
                    "[user] hello",
                    "[assistant]",
                    '  -> add({"a": 1})',
                    "[tool:add] 1",
                ]
            ),
            rendered,
        )

    def test_include_tool_calls_false_drops_the_arrow_lines(self) -> None:
        assistant = Message.assistant("", tool_calls=[_tool_call("add", {"a": 1})])
        self.assertNotIn("->", render_transcript([assistant], include_tool_calls=False))
        self.assertIn("->", render_transcript([assistant]))

    def test_tool_message_falls_back_to_metadata_tool_name(self) -> None:
        message = Message(
            role=Role.TOOL,
            content="x",
            tool_call_id="call_1",
            metadata={"tool_name": "search"},
        )
        self.assertTrue(render_transcript([message]).startswith("[tool:search]"))


class CanonicalJsonTests(unittest.TestCase):
    """``text()`` 用的 canonical json 参数必须与 ``ToolCall.canonical_key`` 一致。"""

    def test_argument_order_does_not_change_rendering(self) -> None:
        first = Message.assistant("", tool_calls=[_tool_call("f", {"a": 1, "b": 2})])
        second = Message.assistant("", tool_calls=[_tool_call("f", {"b": 2, "a": 1})])
        self.assertEqual(first.text(), second.text())
        # ensure_ascii=False：中文参数不被转义（否则同一份 arguments 有两种写法）。
        chinese = Message.assistant("", tool_calls=[_tool_call("f", {"城市": "北京"})])
        self.assertIn("北京", chinese.text())
        self.assertEqual(
            json.dumps({"城市": "北京"}, sort_keys=True, ensure_ascii=False, default=str),
            chinese.text().split("(", 1)[1][:-1],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
