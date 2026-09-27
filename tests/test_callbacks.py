from __future__ import annotations

"""``liteagent/agent/callbacks.py`` 的单元测试（§9.2）。

覆盖 §12 对本文件冻结的覆盖点：

* ``EventType`` 成员集合快照（**含 v2 新增的三个**：TOOL_APPROVAL / BUDGET_EXCEEDED /
  CONTEXT_TRUNCATED）；
* ``TraceEvent.to_dict`` 的**平铺**形态与保留键；
* ``to_json`` / ``from_json`` 往返（**含 ``ts`` -> ``timestamp`` 映射**）；
* ``data`` 含保留键时**构造即抛** ``ConfigError``；
* ``JsonlTraceCallback`` 写读往返 + **多线程写不丢行**；
* ``load_trace`` 跳过坏行；
* ``TokenCounterCallback`` 聚合 + ``cost_usd``；
* ``TraceRecorder`` 上下文；
* ``render_trace`` 非空；
* ``trace_stats`` 对固定事件序列的**整个 dict 精确定义**。

零网络、零真实等待：本文件不触发任何 LLM / 工具调用，只测事件与统计。
"""

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from liteagent.agent.callbacks import (
    RESERVED_KEYS,
    CallbackManager,
    EventType,
    FunctionCallback,
    JsonlTraceCallback,
    TokenCounterCallback,
    TraceEvent,
    TraceRecorder,
    events_of_type,
    load_trace,
    render_trace,
    total_usage,
    trace_stats,
)
from liteagent.config import estimate_cost_usd
from liteagent.errors import ConfigError, SerializationError
from liteagent.types import TokenUsage


def _event(event_type: EventType, *, step: int = 0, **data: object) -> TraceEvent:
    """造一条固定 run_id 的事件（便于断言 dict 逐字相等）。"""
    return TraceEvent(type=event_type, run_id="r1", agent_name="a1", step=step, data=dict(data))


class EventTypeTests(unittest.TestCase):
    """``EventType`` 的成员集合与 coerce（§9.2）。"""

    #: 冻结的成员名快照（§9.2 的 27 个成员；带 [v2 新增] 的三个在末尾单列断言）。
    FROZEN_MEMBER_NAMES = frozenset(
        {
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_FAILED",
            "STEP_STARTED",
            "STEP_FINISHED",
            "LLM_REQUEST",
            "LLM_RESPONSE",
            "LLM_ERROR",
            "THOUGHT",
            "ACTION_PARSED",
            "PARSE_ERROR",
            "REPEAT_DETECTED",
            "NUDGE",
            "TOOL_STARTED",
            "TOOL_RETRY",
            "TOOL_FINISHED",
            "TOOL_ERROR",
            "TOOL_APPROVAL",
            "MEMORY_WRITE",
            "MEMORY_RETRIEVE",
            "MEMORY_COMPRESS",
            "BUDGET_EXCEEDED",
            "CONTEXT_TRUNCATED",
            "AGENT_DELEGATE",
            "AGENT_RETURN",
            "BLACKBOARD_WRITE",
            "BLACKBOARD_READ",
        }
    )

    def test_member_name_set_snapshot(self) -> None:
        self.assertEqual(self.FROZEN_MEMBER_NAMES, {m.name for m in EventType})
        self.assertEqual(27, len(EventType))

    def test_v2_new_members_present_with_frozen_values(self) -> None:
        """三个 v2 新增成员不能漏，且 value 是冻结的 snake_case。"""
        self.assertEqual("tool_approval", EventType.TOOL_APPROVAL.value)
        self.assertEqual("budget_exceeded", EventType.BUDGET_EXCEEDED.value)
        self.assertEqual("context_truncated", EventType.CONTEXT_TRUNCATED.value)

    def test_coerce_accepts_member_value_and_name(self) -> None:
        self.assertIs(EventType.TOOL_FINISHED, EventType.coerce("tool_finished"))
        self.assertIs(EventType.TOOL_FINISHED, EventType.coerce("TOOL_FINISHED"))
        self.assertIs(EventType.TOOL_FINISHED, EventType.coerce(EventType.TOOL_FINISHED))

    def test_coerce_unknown_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            EventType.coerce("no_such_event")


class TraceEventSerializationTests(unittest.TestCase):
    """``TraceEvent`` 的 to_dict / from_dict / json 往返（§9.2）。"""

    def test_to_dict_flattens_data_and_always_emits_ts(self) -> None:
        event = _event(EventType.TOOL_FINISHED, step=3, tool_name="add", duration_ms=1.5)
        payload = event.to_dict()
        self.assertEqual(
            {"type", "run_id", "agent_name", "step", "ts", "tool_name", "duration_ms"},
            set(payload),
        )
        self.assertEqual("tool_finished", payload["type"])
        self.assertEqual("r1", payload["run_id"])
        self.assertEqual("a1", payload["agent_name"])
        self.assertEqual(3, payload["step"])
        self.assertEqual("add", payload["tool_name"])
        # 恒输出 `ts`，绝不输出 `timestamp`。
        self.assertIn("ts", payload)
        self.assertNotIn("timestamp", payload)
        self.assertEqual(event.timestamp, payload["ts"])

    def test_data_containing_reserved_key_raises_on_construction(self) -> None:
        for reserved in sorted(RESERVED_KEYS):
            with self.subTest(reserved=reserved):
                with self.assertRaises(ConfigError):
                    TraceEvent(type=EventType.RUN_STARTED, data={reserved: 1})

    def test_emit_type_rejects_reserved_key_in_data(self) -> None:
        manager = CallbackManager()
        with self.assertRaises(ConfigError):
            manager.emit_type(EventType.RUN_STARTED, step=1, ts=1.0)

    def test_json_round_trip_maps_ts_to_timestamp(self) -> None:
        event = _event(EventType.ACTION_PARSED, step=2, action="echo", arguments={"text": "hi"})
        restored = TraceEvent.from_json(event.to_json())
        self.assertIs(EventType.ACTION_PARSED, restored.type)
        self.assertEqual("r1", restored.run_id)
        self.assertEqual("a1", restored.agent_name)
        self.assertEqual(2, restored.step)
        self.assertEqual(event.timestamp, restored.timestamp)
        self.assertEqual({"action": "echo", "arguments": {"text": "hi"}}, restored.data)

    def test_from_dict_accepts_timestamp_alias_and_ts_wins(self) -> None:
        only_timestamp = TraceEvent.from_dict({"type": "run_started", "timestamp": 5.0, "extra": 2})
        self.assertEqual(5.0, only_timestamp.timestamp)
        self.assertEqual({"extra": 2}, only_timestamp.data)

        both = TraceEvent.from_dict({"type": "run_started", "ts": 7.0, "timestamp": 5.0})
        self.assertEqual(7.0, both.timestamp)

    def test_from_dict_accepts_nested_data_field(self) -> None:
        restored = TraceEvent.from_dict({"type": "nudge", "step": 4, "data": {"text": "stop"}})
        self.assertEqual(4, restored.step)
        self.assertEqual({"text": "stop"}, restored.data)

    def test_from_dict_missing_type_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError):
            TraceEvent.from_dict({"run_id": "r"})

    def test_summary_is_single_line_with_tool_name_and_ms(self) -> None:
        event = _event(
            EventType.TOOL_FINISHED, step=3, tool_name="read_file", ok=True, duration_ms=12.3
        )
        summary = event.summary()
        self.assertEqual("[3] tool_finished read_file ok=True 12.3ms", summary)
        self.assertNotIn("\n", summary)


class CallbackManagerTests(unittest.TestCase):
    """分发 / 订阅 / 回调抛错不影响主流程（§9.2）。"""

    def test_emit_routes_to_both_function_and_object_callbacks(self) -> None:
        seen: list[str] = []
        manager = CallbackManager()
        manager.subscribe(lambda event: seen.append(event.type.value))
        manager.add(FunctionCallback(lambda et, data: seen.append(f"2:{et}")))
        manager.emit_type(EventType.THOUGHT, text="hi")
        self.assertEqual(["thought", "2:thought"], seen)

    def test_emit_never_raises_and_records_errors(self) -> None:
        manager = CallbackManager()
        good: list[EventType] = []

        def _boom(event: TraceEvent) -> None:
            raise ValueError("callback exploded")

        manager.subscribe(_boom)
        manager.subscribe(good.append)
        manager.emit_type(EventType.STEP_STARTED, step=1)  # 不抛
        self.assertEqual([EventType.STEP_STARTED], [e.type for e in good])
        self.assertEqual(1, len(manager.errors))
        self.assertIsInstance(manager.errors[0][1], ValueError)

    def test_subscribe_returns_working_unsubscribe(self) -> None:
        seen: list[EventType] = []
        manager = CallbackManager()
        unsubscribe = manager.subscribe(seen.append)
        manager.emit_type(EventType.NUDGE, text="a")
        unsubscribe()
        unsubscribe()  # 幂等
        manager.emit_type(EventType.NUDGE, text="b")
        self.assertEqual([EventType.NUDGE], [e.type for e in seen])
        self.assertEqual(0, len(manager))

    def test_add_and_remove_and_clear(self) -> None:
        manager = CallbackManager()
        fn = lambda event: None  # noqa: E731
        manager.add(fn)
        self.assertEqual(1, len(manager))
        manager.remove(fn)
        self.assertEqual(0, len(manager))
        manager.remove(fn)  # 不存在时忽略
        manager.subscribe(lambda event: None)
        manager.clear()
        self.assertEqual(0, len(manager))


class JsonlTraceCallbackTests(unittest.TestCase):
    """JSONL 写读往返与多线程写不丢行（§9.2 v2 变更）。"""

    def test_write_read_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            callback = JsonlTraceCallback(path)
            events = [
                _event(EventType.RUN_STARTED, input="x", mode="native"),
                _event(EventType.STEP_STARTED, step=1),
                _event(EventType.RUN_FINISHED, output_len=3, steps=1),
            ]
            for event in events:
                callback.on_event(event)
            callback.close()
            loaded = callback.load()
            self.assertEqual(3, len(loaded))
            self.assertEqual([e.type for e in events], [e.type for e in loaded])
            self.assertEqual(events[0].data, loaded[0].data)
            self.assertEqual(events[1].step, loaded[1].step)
            # 文件是真 JSONL：逐行都能 json.loads。
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    json.loads(line)

    def test_load_includes_lines_written_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_event(EventType.RUN_STARTED).to_json() + "\n")
            callback = JsonlTraceCallback(path)
            callback.on_event(_event(EventType.RUN_FINISHED))
            loaded = callback.load()
            callback.close()
            self.assertEqual(2, len(loaded))

    def test_multithreaded_writes_do_not_lose_lines(self) -> None:
        """多线程并发写：行数与内容都完整（线程安全锁的直接效果）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            callback = JsonlTraceCallback(path)
            threads_count = 8
            per_thread = 50
            barrier = threading.Barrier(threads_count)

            def _worker(worker_id: int) -> None:
                barrier.wait()  # 尽量让写入真的并发
                for index in range(per_thread):
                    callback.on_event(
                        _event(EventType.NUDGE, text=f"{worker_id}:{index}")
                    )

            with ThreadPoolExecutor(max_workers=threads_count) as pool:
                list(pool.map(_worker, range(threads_count)))
            callback.close()

            loaded = load_trace(path)
            self.assertEqual(threads_count * per_thread, len(loaded))
            texts = {event.data["text"] for event in loaded}
            self.assertEqual(
                {
                    f"{worker_id}:{index}"
                    for worker_id in range(threads_count)
                    for index in range(per_thread)
                },
                texts,
            )

    def test_close_is_idempotent_and_drops_late_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            callback = JsonlTraceCallback(path)
            callback.on_event(_event(EventType.RUN_STARTED))
            callback.close()
            callback.close()
            callback.on_event(_event(EventType.RUN_FINISHED))  # 关闭后丢弃，不抛
            self.assertEqual(1, len(load_trace(path)))


class LoadTraceTests(unittest.TestCase):
    """``load_trace`` 的容错（§9.2）。"""

    def test_skips_blank_and_unparsable_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            good_a = _event(EventType.RUN_STARTED).to_json()
            good_b = _event(EventType.RUN_FINISHED).to_json()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(good_a + "\n")
                handle.write("\n")
                handle.write("   \n")
                handle.write("{not json at all\n")
                handle.write('"a bare string"\n')
                handle.write(good_b + "\n")
            loaded = load_trace(path)
            self.assertEqual([EventType.RUN_STARTED, EventType.RUN_FINISHED],
                             [event.type for event in loaded])

    def test_missing_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual([], load_trace(os.path.join(tmp, "does-not-exist.jsonl")))


class TokenCounterCallbackTests(unittest.TestCase):
    """usage 聚合与 cost_usd（§9.2）。"""

    def test_aggregates_usage_and_calls(self) -> None:
        counter = TokenCounterCallback()
        counter.on_event(TraceEvent(type=EventType.LLM_REQUEST, data={"model": "gpt-4o-mini"}))
        counter.on_event(
            TraceEvent(
                type=EventType.LLM_RESPONSE,
                data={"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
            )
        )
        counter.on_event(
            TraceEvent(
                type=EventType.LLM_RESPONSE,
                data={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            )
        )
        self.assertEqual(2, counter.calls)
        self.assertEqual(
            TokenUsage(prompt_tokens=110, completion_tokens=55, total_tokens=165),
            counter.usage,
        )

    def test_cost_usd_uses_price_table_and_resets(self) -> None:
        counter = TokenCounterCallback()
        counter.on_event(TraceEvent(type=EventType.LLM_REQUEST, data={"model": "gpt-4o-mini"}))
        counter.on_event(
            TraceEvent(
                type=EventType.LLM_RESPONSE,
                data={"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
            )
        )
        expected = estimate_cost_usd(
            TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4o-mini",
        )
        self.assertAlmostEqual(expected, counter.cost_usd)
        counter.reset()
        self.assertEqual(0, counter.calls)
        self.assertEqual(TokenUsage(), counter.usage)
        self.assertIsNone(counter.cost_usd)

    def test_cost_usd_is_none_when_model_unpriced(self) -> None:
        counter = TokenCounterCallback()
        counter.on_event(
            TraceEvent(
                type=EventType.LLM_RESPONSE,
                data={
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "model": "no-such-model",
                },
            )
        )
        self.assertIsNone(counter.cost_usd)
        self.assertEqual(1, counter.calls)

    def test_ignores_non_llm_events(self) -> None:
        counter = TokenCounterCallback()
        counter.on_event(TraceEvent(type=EventType.TOOL_FINISHED, data={"tool_name": "add"}))
        self.assertEqual(0, counter.calls)
        self.assertEqual(TokenUsage(), counter.usage)


class TraceRecorderTests(unittest.TestCase):
    """上下文管理器：内存事件 + JSONL + 退出后解绑（§9.2）。"""

    def test_records_events_in_memory_and_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            with TraceRecorder(path, run_id="run_fixed") as recorder:
                self.assertEqual("run_fixed", recorder.run_id)
                recorder.manager.emit_type(EventType.RUN_STARTED, run_id="run_fixed")
                recorder.manager.emit_type(EventType.RUN_FINISHED, run_id="run_fixed")
                self.assertEqual(2, len(recorder.events))
            self.assertEqual(2, len(recorder.events))
            # 退出后 manager 不再收集 / 不再写文件
            recorder.manager.emit_type(EventType.NUDGE)
            self.assertEqual(2, len(recorder.events))
            self.assertEqual(2, len(load_trace(path)))

    def test_recorder_without_path_only_collects(self) -> None:
        with TraceRecorder() as recorder:
            self.assertTrue(recorder.run_id.startswith("run_"))
            recorder.manager.emit_type(EventType.THOUGHT, text="x")
        self.assertEqual([EventType.THOUGHT], [e.type for e in recorder.events])


class DisplayAndStatsTests(unittest.TestCase):
    """render_trace / total_usage / events_of_type / trace_stats（§9.2）。"""

    #: 固定的 trace（每个字段都可手算），供 trace_stats 的整 dict 断言使用。
    FIXED_EVENTS = (
        TraceEvent(
            type=EventType.RUN_STARTED,
            run_id="run_fixed",
            step=0,
            data={"input": "x", "mode": "native", "tools": ["add"]},
        ),
        TraceEvent(type=EventType.STEP_STARTED, run_id="run_fixed", step=1),
        TraceEvent(
            type=EventType.LLM_REQUEST,
            run_id="run_fixed",
            step=1,
            data={"model": "gpt-4o-mini", "messages_count": 2, "tools_count": 1, "retry": 0},
        ),
        TraceEvent(
            type=EventType.LLM_RESPONSE,
            run_id="run_fixed",
            step=1,
            data={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "latency_ms": 100.0,
                "finish_reason": "tool_calls",
                "model": "gpt-4o-mini",
            },
        ),
        TraceEvent(
            type=EventType.TOOL_STARTED,
            run_id="run_fixed",
            step=1,
            data={"tool_name": "add", "call_id": "call_0", "attempt": 0},
        ),
        TraceEvent(
            type=EventType.TOOL_FINISHED,
            run_id="run_fixed",
            step=1,
            data={"tool_name": "add", "call_id": "call_0", "ok": True, "duration_ms": 12.0},
        ),
        TraceEvent(type=EventType.STEP_FINISHED, run_id="run_fixed", step=1, data={"status": "OBSERVING"}),
    )

    def test_render_trace_never_empty(self) -> None:
        empty = render_trace([])
        self.assertTrue(empty)
        self.assertTrue(empty.startswith("trace:"))

        rendered = render_trace(list(self.FIXED_EVENTS))
        self.assertTrue(rendered)
        self.assertIn("run_started", rendered)
        self.assertIn("step 1", rendered)
        self.assertIn("tool_finished: add ok=True 12.0ms", rendered)

        flat = render_trace(list(self.FIXED_EVENTS), indent=False)
        self.assertIn("tool_finished add ok=True 12.0ms", flat)

    def test_total_usage_and_events_of_type(self) -> None:
        self.assertEqual(
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            total_usage(list(self.FIXED_EVENTS)),
        )
        self.assertEqual(
            ["add"],
            [e.data["tool_name"] for e in events_of_type(self.FIXED_EVENTS, "tool_finished")],
        )
        with self.assertRaises(ConfigError):
            events_of_type(self.FIXED_EVENTS, "nope")

    def test_trace_stats_full_dict_for_fixed_sequence(self) -> None:
        expected_cost = estimate_cost_usd(
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o-mini",
        )
        self.assertEqual(
            {
                "runs": 1,
                "steps": 1,
                "llm_calls": 1,
                "llm_latency_ms": {"total": 100.0, "mean": 100.0, "p50": 100.0, "p95": 100.0},
                "tool_calls": 1,
                "tool_failures": 0,
                "tool_latency_ms": {"add": {"count": 1, "mean": 12.0, "p95": 12.0}},
                "retries": 0,
                "parse_errors": 0,
                "nudges": 0,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "cost_usd": expected_cost,
                "errors": {},
            },
            trace_stats(list(self.FIXED_EVENTS)),
        )

    def test_trace_stats_empty_sequence_and_error_counts(self) -> None:
        self.assertEqual(
            {
                "runs": 0,
                "steps": 0,
                "llm_calls": 0,
                "llm_latency_ms": {"total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0},
                "tool_calls": 0,
                "tool_failures": 0,
                "tool_latency_ms": {},
                "retries": 0,
                "parse_errors": 0,
                "nudges": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost_usd": None,
                "errors": {},
            },
            trace_stats([]),
        )

        with_errors = trace_stats(
            [
                TraceEvent(type=EventType.PARSE_ERROR, data={"reason": "x", "offset": 0, "raw_len": 3, "attempt": 1}),
                TraceEvent(type=EventType.NUDGE, data={"text": "n"}),
                TraceEvent(type=EventType.RUN_FAILED, data={"error_type": "AgentError", "aborted": False}),
                TraceEvent(type=EventType.LLM_ERROR, data={"error_type": "LLMTimeoutError", "retry": 0}),
            ]
        )
        self.assertEqual(1, with_errors["parse_errors"])
        self.assertEqual(1, with_errors["nudges"])
        self.assertEqual({"AgentError": 1, "LLMTimeoutError": 1}, with_errors["errors"])

    def test_trace_stats_p95_nearest_lower_no_interpolation(self) -> None:
        """4 个耗时 -> 下标 int(0.95*(4-1)) == 2（**不插值**）。"""
        events = [
            TraceEvent(
                type=EventType.LLM_RESPONSE,
                data={
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "latency_ms": value,
                    "model": "gpt-4o-mini",
                },
            )
            for value in (10.0, 20.0, 30.0, 40.0)
        ]
        stats = trace_stats(events)
        self.assertEqual(100.0, stats["llm_latency_ms"]["total"])
        self.assertEqual(25.0, stats["llm_latency_ms"]["mean"])
        self.assertEqual(20.0, stats["llm_latency_ms"]["p50"])  # int(0.5*3) == 1
        self.assertEqual(30.0, stats["llm_latency_ms"]["p95"])  # int(0.95*3) == 2


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
