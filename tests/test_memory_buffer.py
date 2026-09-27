from __future__ import annotations

"""tests/test_memory_buffer.py —— §8.3 `memory/buffer.py` 的单元测试。

§12 第 5118 行要求的覆盖点：
  * 双约束裁剪（条数 + token 预算）
  * `keep_last_n` 保证
  * `keep_system`
  * `_repair_tool_pairs`（直接 import 模块级函数）
  * `evicted`/`drain_evicted` 幂等
  * token 预算
  * **多线程 add 与 window 并发不抛 `RuntimeError`**

token 估算全部走 `HeuristicTokenizer`（默认）：4 个 ASCII 字符 = 1 token，
所以下面用 `"aaaa"` 表示 1 token、`"a" * 16` 表示 4 token。
"""

import threading
import unittest

from liteagent.llm.message import Message, Role, drop_orphan_tool_messages
from liteagent.memory.buffer import BufferConfig, BufferMemory, _repair_tool_pairs
from liteagent.types import ToolCall


def _msg(text: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=text)


def _tokens(text: str) -> int:
    """与实现同源的估算（单条 ASCII 文本的期望值）。"""
    return BufferMemory().tokenizer.estimate(text)


def _tool_call(call_id: str, name: str = "t") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


class DualConstraintTests(unittest.TestCase):
    """§8.3 第 3 步：`max_messages` 与 `max_tokens` 谁先触发听谁的。"""

    def test_max_messages_triggers_first(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=3, keep_last_n=0))
        for index in range(5):
            buf.add(_msg(f"m{index}"))
        window = buf.window()
        self.assertEqual([m.content for m in window], ["m2", "m3", "m4"])
        self.assertEqual(len(buf), 5)  # buffer 全量不受裁剪影响

    def test_token_budget_triggers_first(self) -> None:
        # 每条 16 ASCII 字符 = 4 token；预算 8 -> 只能装 2 条。
        buf = BufferMemory(BufferConfig(max_tokens=8, max_messages=50, keep_last_n=0))
        for index in range(5):
            buf.add(_msg("a" * 16))
        window = buf.window()
        self.assertEqual(len(window), 2)
        self.assertLessEqual(buf.window_tokens(), 8)

    def test_window_is_recomputed_and_never_exceeds_budget(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=8, max_messages=50, keep_last_n=0))
        buf.add(_msg("a" * 16))
        self.assertEqual(len(buf.window()), 1)
        buf.add(_msg("a" * 16))
        buf.add(_msg("a" * 16))
        window = buf.window()
        self.assertEqual(len(window), 2)
        self.assertLessEqual(buf.window_tokens(), 8)
        self.assertEqual(buf.estimated_tokens(), 12)

    def test_zero_max_messages_yields_empty_window(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=100, max_messages=0, keep_last_n=0))
        buf.add(_msg("hello"))
        self.assertEqual(buf.window(), [])

    def test_estimated_tokens_of_empty_buffer_is_zero(self) -> None:
        buf = BufferMemory()
        self.assertEqual(buf.estimated_tokens(), 0)
        self.assertEqual(buf.window_tokens(), 0)
        self.assertEqual(buf.window(), [])
        self.assertEqual(len(buf), 0)


class KeepLastNTests(unittest.TestCase):
    """§8.3：`keep_last_n` 是"允许超预算的最小可用上下文"。"""

    def test_keep_last_n_survives_over_budget(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=1, max_messages=50, keep_last_n=2))
        for index in range(5):
            buf.add(_msg(f"{index}" + "a" * 15))
        window = buf.window()
        self.assertEqual(len(window), 2)
        self.assertEqual([m.content for m in window], ["3" + "a" * 15, "4" + "a" * 15])
        # 明确承认"超预算"：这是冻结语义，不是 bug。
        self.assertGreater(buf.window_tokens(), 1)

    def test_keep_last_n_zero_frees_the_budget(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=1, max_messages=50, keep_last_n=0))
        for index in range(5):
            buf.add(_msg("a" * 16))
        self.assertEqual(buf.window(), [])


class KeepSystemTests(unittest.TestCase):
    """§8.3 第 2 步：`keep_system` 把前导 SYSTEM 消息钉在窗口头部。"""

    def test_leading_system_is_pinned_even_over_budget(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=1, max_messages=50, keep_last_n=0))
        buf.add(_msg("sys!", Role.SYSTEM))
        buf.add(_msg("a" * 16, Role.USER))
        window = buf.window()
        self.assertEqual([m.role for m in window], [Role.SYSTEM])
        self.assertEqual(window[0].content, "sys!")

    def test_keep_system_false_lets_the_budget_drop_it(self) -> None:
        buf = BufferMemory(
            BufferConfig(max_tokens=1, max_messages=50, keep_last_n=0, keep_system=False)
        )
        buf.add(_msg("sys!", Role.SYSTEM))
        buf.add(_msg("a" * 16, Role.USER))
        # 同一条 SYSTEM 消息在 keep_system=True 时是 pinned（上一条用例），关掉之后
        # 它跟普通消息一样受预算约束 —— 这里连它带 user 一起被裁掉。
        self.assertEqual(buf.window(), [])

    def test_only_leading_system_messages_are_pinned(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=1, max_messages=50, keep_last_n=1))
        buf.add(_msg("user1", Role.USER))
        buf.add(_msg("sys!", Role.SYSTEM))
        buf.add(_msg("a" * 16, Role.USER))
        window = buf.window()
        # 前导不是 SYSTEM，因此没有任何消息被 pin；keep_last_n=1 只保住了最后一条，
        # 夹在中间的 SYSTEM 被预算裁掉。
        self.assertEqual([m.content for m in window], ["a" * 16])
        # 清空时同理：只有"前导连续"的 SYSTEM 被保留（此处一条都没有）。
        buf.clear()
        self.assertEqual(buf.messages(), [])

    def test_clear_keep_system_true_keeps_leading_system_only(self) -> None:
        buf = BufferMemory()
        buf.add(_msg("sys1", Role.SYSTEM))
        buf.add(_msg("sys2", Role.SYSTEM))
        buf.add(_msg("user", Role.USER))
        buf.add(_msg("sys3", Role.SYSTEM))
        buf.clear()
        self.assertEqual([m.content for m in buf.messages()], ["sys1", "sys2"])

    def test_clear_keep_system_false_removes_everything(self) -> None:
        buf = BufferMemory()
        buf.add(_msg("sys1", Role.SYSTEM))
        buf.add(_msg("user", Role.USER))
        buf.clear(keep_system=False)
        self.assertEqual(buf.messages(), [])
        self.assertEqual(len(buf), 0)
        self.assertIsNone(buf.last())


class RepairToolPairsTests(unittest.TestCase):
    """§8.3 的模块级函数（直接 import）：半截工具对整体丢弃。"""

    def test_complete_pair_is_kept(self) -> None:
        assistant = Message.assistant("call", [_tool_call("c1")])
        result = Message(role=Role.TOOL, content="ok", tool_call_id="c1")
        out = _repair_tool_pairs([Message.user("u"), assistant, result])
        self.assertEqual(out, [Message.user("u"), assistant, result])

    def test_incomplete_pair_drops_assistant_and_its_results(self) -> None:
        assistant = Message.assistant("call", [_tool_call("c1"), _tool_call("c2")])
        only_c1 = Message(role=Role.TOOL, content="ok", tool_call_id="c1")
        out = _repair_tool_pairs([assistant, only_c1])
        self.assertEqual(out, [])

    def test_call_without_id_makes_the_whole_assistant_incomplete(self) -> None:
        """无 id 的 tool_call 视为"无法配对"（等价于结果缺失）：整条 assistant 丢弃。"""
        assistant = Message.assistant("call", [_tool_call("c1"), _tool_call("")])
        result = Message(role=Role.TOOL, content="ok", tool_call_id="c1")
        self.assertEqual(_repair_tool_pairs([assistant, result]), [])

    def test_empty_call_id_pair_is_cleaned_by_the_combined_pipeline(self) -> None:
        """组合口径（window() 第 5 步）：粗筛删掉空 id 的 tool 消息并剥掉工具调用。"""
        assistant = Message.assistant("call", [_tool_call("")])
        result = Message(role=Role.TOOL, content="ok", tool_call_id="")
        repaired = _repair_tool_pairs([assistant, result])
        self.assertEqual([m for m in repaired if m.is_tool_pair_start()], [])
        combined = _repair_tool_pairs(drop_orphan_tool_messages([assistant, result]))
        # 粗筛保留 assistant 本体（content 是模型可见的思考），但剥掉它的 tool_calls；
        # 空 id 的 tool 消息被视为孤儿，整条删除。对 API 而言这是合法的消息序列。
        self.assertEqual([m.role for m in combined], [Role.ASSISTANT])
        self.assertEqual(combined[0].tool_calls, [])
        self.assertFalse(combined[0].is_tool_pair_start())

    def test_orphan_tool_message_is_left_alone_by_this_stage(self) -> None:
        """粗筛（drop_orphan_tool_messages）与本函数分工明确：本函数不动孤儿 tool 消息。"""
        orphan = Message(role=Role.TOOL, content="ok", tool_call_id="nope")
        self.assertEqual(_repair_tool_pairs([orphan]), [orphan])

    def test_untouched_messages_are_returned_by_identity(self) -> None:
        plain = Message.user("plain")
        assistant = Message.assistant("call", [_tool_call("c1")])
        result = Message(role=Role.TOOL, content="ok", tool_call_id="c1")
        out = _repair_tool_pairs([plain, assistant, result])
        self.assertIs(out[0], plain)
        self.assertIs(out[1], assistant)

    def test_window_applies_repair_at_the_trim_boundary(self) -> None:
        """裁剪把工具对切散时，窗口里不能留下孤儿 tool 消息或半截工具对。"""
        messages = [
            Message.assistant("thinking", [_tool_call("c1")]),
            Message(role=Role.TOOL, content="ok", tool_call_id="c1"),
            _msg("later"),
        ]

        complete = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=3, keep_last_n=0))
        complete.extend(messages)
        self.assertEqual([m.role for m in complete.window()], [Role.ASSISTANT, Role.TOOL, Role.USER])

        # 窗口只装得下最后 2 条：assistant 被切掉，它的 tool 结果成了孤儿，必须一起清掉。
        trimmed = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=2, keep_last_n=0))
        trimmed.extend(messages)
        window = trimmed.window()
        self.assertEqual([m.role for m in window], [Role.USER])
        for message in window:
            self.assertFalse(message.is_tool_pair_start())
            self.assertIsNone(message.tool_call_id)


class EvictedTests(unittest.TestCase):
    """`evicted` / `drain_evicted` 的幂等与顺序（§8.3 第 6 步）。"""

    def setUp(self) -> None:
        self.buf = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=2, keep_last_n=0))
        self.messages = [_msg(f"m{index}") for index in range(5)]
        for message in self.messages:
            self.buf.add(message)

    def test_evicted_is_original_order_and_a_copy(self) -> None:
        self.buf.window()
        evicted = self.buf.evicted()
        self.assertEqual([m.content for m in evicted], ["m0", "m1", "m2"])
        evicted.append(_msg("intruder"))
        self.assertEqual(len(self.buf.evicted()), 3)

    def test_window_is_idempotent_before_drain(self) -> None:
        self.buf.window()
        first = [m.content for m in self.buf.evicted()]
        self.buf.window()
        self.assertEqual([m.content for m in self.buf.evicted()], first)

    def test_drain_evicted_is_idempotent(self) -> None:
        self.buf.window()
        drained = self.buf.drain_evicted()
        self.assertEqual([m.content for m in drained], ["m0", "m1", "m2"])
        self.assertEqual(self.buf.drain_evicted(), [])
        self.assertEqual(self.buf.evicted(), [])

    def test_drained_messages_never_reappear_after_more_windows(self) -> None:
        self.buf.window()
        self.buf.drain_evicted()
        self.buf.window()
        self.assertEqual(self.buf.evicted(), [])

    def test_clear_resets_drain_tracking(self) -> None:
        self.buf.window()
        self.buf.drain_evicted()
        self.buf.clear()
        for message in self.messages:
            self.buf.add(message)
        self.buf.window()
        self.assertEqual(len(self.buf.evicted()), 3)

    def test_on_evict_only_reports_fresh_evictions(self) -> None:
        seen: list[list[str]] = []
        buf = BufferMemory(
            BufferConfig(max_tokens=10_000, max_messages=2, keep_last_n=0),
            on_evict=lambda msgs: seen.append([m.content for m in msgs]),
        )
        for message in self.messages:
            buf.add(message)
        buf.window()
        buf.window()
        self.assertEqual(seen, [["m0", "m1", "m2"]])
        # 回调抛异常不影响 window() 的返回（旁路观测必须降级）。
        buf2 = BufferMemory(
            BufferConfig(max_tokens=10_000, max_messages=1, keep_last_n=0),
            on_evict=lambda msgs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        for message in self.messages:
            buf2.add(message)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(len(buf2.window()), 1)


class OtherBufferApiTests(unittest.TestCase):
    """`add`/`extend`/`messages`/`last`/`__len__` 的基本契约。"""

    def test_extend_equals_repeated_add(self) -> None:
        a = BufferMemory()
        b = BufferMemory()
        for message in [_msg("1"), _msg("2")]:
            a.add(message)
        b.extend([_msg("1"), _msg("2")])
        self.assertEqual([m.content for m in a.messages()], [m.content for m in b.messages()])

    def test_last_returns_full_buffer_tail(self) -> None:
        buf = BufferMemory()
        self.assertIsNone(buf.last())
        buf.add(_msg("1"))
        buf.add(_msg("2"))
        self.assertIsNotNone(buf.last())
        self.assertEqual(buf.last().content, "2")  # type: ignore[union-attr]

    def test_messages_returns_copy_but_same_instances(self) -> None:
        buf = BufferMemory()
        message = _msg("1")
        buf.add(message)
        snapshot = buf.messages()
        snapshot.append(_msg("2"))
        self.assertEqual(len(buf.messages()), 1)
        self.assertIs(buf.messages()[0], message)


class ConcurrencyTests(unittest.TestCase):
    """§8.1 线程模型：`add` 与 `window` 并发不得抛 `RuntimeError`。"""

    def test_concurrent_add_and_window_does_not_raise(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=64, max_messages=8, keep_last_n=2))
        failures: list[BaseException] = []
        barrier = threading.Barrier(6)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                for step in range(60):
                    buf.add(_msg(f"t{index}-{step}"))
                    buf.window()
                    buf.evicted()
                    buf.drain_evicted()
                    buf.messages()
                    buf.estimated_tokens()
                    buf.window_tokens()
            except BaseException as exc:  # noqa: BLE001 - 收集后再断言，便于看到原始堆栈
                failures.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(failures, [])
        self.assertEqual(len(buf), 6 * 60)
        self.assertLessEqual(len(buf.window()), 8)

    def test_concurrent_clear_and_window_does_not_raise(self) -> None:
        buf = BufferMemory(BufferConfig(max_tokens=16, max_messages=4, keep_last_n=1))
        failures: list[BaseException] = []
        barrier = threading.Barrier(4)

        def adder() -> None:
            barrier.wait()
            try:
                for step in range(80):
                    buf.add(_msg(f"x{step}"))
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        def reader() -> None:
            barrier.wait()
            try:
                for _ in range(80):
                    buf.window()
                    buf.drain_evicted()
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        def clearer() -> None:
            barrier.wait()
            try:
                for _ in range(20):
                    buf.clear(keep_system=False)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=adder),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(failures, [])
        self.assertGreaterEqual(len(buf.messages()), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
