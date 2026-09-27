from __future__ import annotations

"""tests/test_blackboard.py —— §10.2 / §10.2.1 `multiagent/blackboard.py` 的单元测试。

§12 第 5127 行要求的覆盖点（逐条对应到下面的测试类）：

  * 读写                         -> ReadWriteTests / AsyncFacadeTests
  * 版本递增                     -> VersioningTests
  * `if_version` 冲突            -> IfVersionTests
  * `increment`                  -> IncrementUpdateTests
  * `update`                     -> IncrementUpdateTests
  * `ttl` 惰性过期               -> TtlTests
  * `snapshot` 是副本            -> SnapshotHistoryTests
  * `history`                    -> SnapshotHistoryTests
  * `max_entries` 淘汰           -> MaxEntriesTests
  * `subscribe` 取消             -> SubscribeTests
  * 多线程并发写 100 次后计数正确 -> ConcurrencyTests
  * `awatch` 收到写入事件（异步） -> WatchTests
  * 写入方在 worker 线程写完、loop 已关闭时不抛异常（实测 M-4） -> ClosedLoopTests

测试卫生（§12.1 / §2.8）：
  * 时钟一律用 `mock.patch.object(liteagent.multiagent.blackboard, "utc_now", ...)`。
    **不能**用 `tests.helpers.frozen_time`：`blackboard.py` 是
    `from liteagent.config import utc_now` 的模块级裸名形态，patch `config.utc_now`
    够不到它（helpers.py 的 docstring 明文列出了这个例外）。
  * 不断言 `BlackboardEntry.created_at`（§2.8 的禁止清单里有它）。
  * 不真睡：`awatch` 的超时用 `timeout_s=0.0`（实现里 remaining==0 会立即抛）。
    唯一的例外是 `[v3]` 的丢唤醒回归用例 —— 它必须真的把消费者**钉在** `async for`
    的 body 里再写第二次，用一次 50ms 的 `asyncio.sleep` 确保"唤醒已经落空"成为事实，
    而不是"调度还没轮到"。没有这个真实等待就无法确定性复现那个窗口。
"""

import asyncio
import contextlib
import threading
import unittest
from typing import Any, Callable
from unittest import mock

from liteagent.errors import ConfigError, VersionConflictError
from liteagent.multiagent import blackboard as blackboard_module
from liteagent.multiagent.blackboard import Blackboard, BlackboardEntry


# ======================================================================================
# 夹具
# ======================================================================================


class _Clock:
    """可手动推进的假时钟（当作 `blackboard.utc_now` 的替身）。

    为什么要自己造而不是用 `frozen_time`：见模块 docstring —— blackboard 用的是
    模块级裸名 import，patch `liteagent.config.utc_now` 对它无效。
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _patch_clock(clock: Callable[[], float]) -> Any:
    return mock.patch.object(blackboard_module, "utc_now", clock)


async def _wait_for(predicate: Callable[[], bool], *, rounds: int = 500) -> bool:
    """反复 `await asyncio.sleep(0)` 直到 predicate 为真（确定性的"等注册完成"）。

    **不是轮询 sleep**：`sleep(0)` 只是让出一次调度权，不消耗真实时间。
    """
    for _ in range(rounds):
        if predicate():
            return True
        await asyncio.sleep(0)
    return bool(predicate())


# ======================================================================================
# 读写 / 基本容器语义
# ======================================================================================


class ReadWriteTests(unittest.TestCase):
    """§10.2：`write` / `read` / `read_entry` / `delete` / `__len__` / `__contains__`。"""

    def test_write_then_read_roundtrip(self) -> None:
        bb = Blackboard()
        entry = bb.write("answer", 42, author="alice", tags=("x", "y"))
        self.assertIsInstance(entry, BlackboardEntry)
        self.assertEqual(entry.key, "answer")
        self.assertEqual(entry.value, 42)
        self.assertEqual(entry.author, "alice")
        self.assertEqual(entry.version, 1)
        self.assertEqual(entry.tags, ("x", "y"))
        self.assertIsNone(entry.expires_at)
        self.assertEqual(bb.read("answer"), 42)
        self.assertIn("answer", bb)
        self.assertEqual(len(bb), 1)

    def test_read_missing_returns_default(self) -> None:
        bb = Blackboard()
        self.assertIsNone(bb.read("nope"))
        self.assertEqual(bb.read("nope", default="fallback"), "fallback")
        self.assertIsNone(bb.read_entry("nope"))
        self.assertNotIn("nope", bb)
        self.assertEqual(len(bb), 0)

    def test_write_rejects_empty_or_non_string_key(self) -> None:
        bb = Blackboard()
        for bad in ("", None, 7, b"bytes"):
            with self.subTest(key=bad):
                with self.assertRaises(ConfigError):
                    bb.write(bad, 1)  # type: ignore[arg-type]

    def test_delete_reports_whether_a_live_entry_was_removed(self) -> None:
        bb = Blackboard()
        bb.write("k", 1)
        self.assertTrue(bb.delete("k"))
        self.assertFalse(bb.delete("k"))  # 已经没了
        self.assertNotIn("k", bb)
        self.assertEqual(len(bb), 0)

    def test_keys_prefix_and_tags_filter(self) -> None:
        bb = Blackboard()
        bb.write("stage:a", 1, tags=("stage",))
        bb.write("stage:b", 2, tags=("stage", "extra"))
        bb.write("other", 3)
        self.assertEqual(sorted(bb.keys()), ["other", "stage:a", "stage:b"])
        self.assertEqual(bb.keys(prefix="stage:"), ["stage:a", "stage:b"])
        self.assertEqual(bb.keys(tags=("stage",)), ["stage:a", "stage:b"])
        self.assertEqual(bb.keys(tags=("extra",)), ["stage:b"])

    def test_list_is_sorted_by_updated_at(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("a", 1)
            clock.advance(1)
            bb.write("b", 2)
            clock.advance(1)
            bb.write("c", 3)
        self.assertEqual([e.key for e in bb.list()], ["a", "b", "c"])

    def test_clear_empties_entries_and_history(self) -> None:
        bb = Blackboard()
        bb.write("a", 1)
        bb.write("b", 2)
        bb.clear()
        self.assertEqual(len(bb), 0)
        self.assertEqual(bb.history(), [])
        self.assertEqual(bb.snapshot(), {})

    def test_to_dict_shape(self) -> None:
        bb = Blackboard()
        bb.write("a", {"nested": (1, 2)})
        data = bb.to_dict()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["max_entries"], 1000)
        # tags 走 to_jsonable -> list；value 里的 tuple 也被转成 list
        self.assertEqual(data["entries"]["a"]["value"], {"nested": [1, 2]})
        self.assertEqual(data["entries"]["a"]["tags"], [])


# ======================================================================================
# 版本
# ======================================================================================


class VersioningTests(unittest.TestCase):
    """§10.2：已存在 -> version+1、updated_at 刷新、created_at 保留、author 覆盖。"""

    def test_version_increments_and_created_at_is_preserved(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            first = bb.write("k", "v1", author="alice")
            self.assertEqual(first.version, 1)
            clock.advance(5)
            second = bb.write("k", "v2", author="bob")
        self.assertEqual(second.version, 2)
        self.assertEqual(second.created_at, first.created_at)  # 创建时间不刷新
        self.assertGreater(second.updated_at, first.updated_at)  # 更新时间刷新
        self.assertEqual(second.author, "bob")  # author 覆盖
        self.assertEqual(bb.read("k"), "v2")

    def test_history_holds_each_version(self) -> None:
        bb = Blackboard()
        for value in range(4):
            bb.write("k", value)
        self.assertEqual([e.version for e in bb.history()], [1, 2, 3, 4])


class IfVersionTests(unittest.TestCase):
    """§10.2：`if_version` 非 None 且与当前版本不符 -> `VersionConflictError`。"""

    def test_matching_if_version_succeeds(self) -> None:
        bb = Blackboard()
        bb.write("k", 1)
        entry = bb.write("k", 2, if_version=1)
        self.assertEqual(entry.version, 2)

    def test_conflict_raises_with_key_expected_actual(self) -> None:
        bb = Blackboard()
        bb.write("k", 1)
        bb.write("k", 2)
        with self.assertRaises(VersionConflictError) as ctx:
            bb.write("k", 3, if_version=1)
        error = ctx.exception
        self.assertEqual(error.key, "k")
        self.assertEqual(error.expected, 1)
        self.assertEqual(error.actual, 2)

    def test_if_version_zero_means_create_only(self) -> None:
        bb = Blackboard()
        entry = bb.write("k", "created", if_version=0)
        self.assertEqual(entry.version, 1)
        with self.assertRaises(VersionConflictError):
            bb.write("k", "again", if_version=0)

    def test_conflict_leaves_state_untouched(self) -> None:
        bb = Blackboard()
        calls: list[str] = []
        bb.subscribe(lambda key, entry: calls.append(key))
        bb.write("k", "original")
        calls.clear()
        with self.assertRaises(VersionConflictError):
            bb.write("k", "rejected", if_version=99)
        self.assertEqual(bb.read("k"), "original")
        self.assertEqual(bb.read_entry("k").version, 1)
        self.assertEqual(calls, [])  # 冲突时订阅回调不被触发


# ======================================================================================
# increment / update
# ======================================================================================


class IncrementUpdateTests(unittest.TestCase):
    """§10.2：`increment` 原子自增；`update` 对 dict 值浅合并。"""

    def test_increment_starts_from_zero(self) -> None:
        bb = Blackboard()
        self.assertEqual(bb.increment("counter"), 1.0)
        self.assertEqual(bb.increment("counter"), 2.0)
        self.assertEqual(bb.read("counter"), 2)
        self.assertEqual(bb.read_entry("counter").version, 2)

    def test_increment_honours_amount(self) -> None:
        bb = Blackboard()
        self.assertEqual(bb.increment("c", amount=2.5), 2.5)
        self.assertEqual(bb.increment("c", amount=-0.5), 2.0)

    def test_increment_rejects_non_numeric_value(self) -> None:
        bb = Blackboard()
        bb.write("s", "not a number")
        with self.assertRaises(ConfigError):
            bb.increment("s")

    def test_increment_rejects_boolean_amount(self) -> None:
        bb = Blackboard()
        with self.assertRaises(ConfigError):
            bb.increment("c", amount=True)  # bool 是 int 的子类，但这里是 bug

    def test_update_shallow_merges_dict_value(self) -> None:
        bb = Blackboard()
        bb.write("cfg", {"a": 1, "b": 2}, author="alice", tags=("cfg",))
        entry = bb.update("cfg", {"b": 3, "c": 4}, author="bob")
        self.assertEqual(entry.value, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(entry.version, 2)
        self.assertEqual(entry.author, "bob")
        self.assertEqual(entry.tags, ("cfg",))  # tags 沿袭

    def test_update_on_missing_key_starts_from_empty_dict(self) -> None:
        bb = Blackboard()
        entry = bb.update("fresh", {"a": 1})
        self.assertEqual(entry.value, {"a": 1})
        self.assertEqual(entry.version, 1)

    def test_update_rejects_non_dict_value(self) -> None:
        bb = Blackboard()
        bb.write("n", 5)
        with self.assertRaises(ConfigError):
            bb.update("n", {"a": 1})

    def test_update_rejects_non_mapping_patch(self) -> None:
        bb = Blackboard()
        with self.assertRaises(ConfigError):
            bb.update("k", ["not", "a", "mapping"])  # type: ignore[arg-type]


# ======================================================================================
# TTL / 惰性过期
# ======================================================================================


class TtlTests(unittest.TestCase):
    """§10.2：过期条目按不存在处理（**惰性过期**）。"""

    def test_entry_is_expired_uses_ge_semantics(self) -> None:
        entry = BlackboardEntry(key="k", value=1, expires_at=100.0)
        self.assertFalse(entry.is_expired(now=99.999))
        self.assertTrue(entry.is_expired(now=100.0))  # 到期那一刻即失效
        self.assertTrue(entry.is_expired(now=101.0))

    def test_entry_without_ttl_never_expires(self) -> None:
        entry = BlackboardEntry(key="k", value=1)
        self.assertFalse(entry.is_expired(now=10**12))

    def test_read_returns_default_after_ttl(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("k", "alive", ttl_s=10)
            self.assertEqual(bb.read("k"), "alive")
            clock.advance(9)
            self.assertEqual(bb.read("k"), "alive")
            clock.advance(1)  # 正好到期
            self.assertIsNone(bb.read("k"))
            self.assertIsNone(bb.read_entry("k"))
            self.assertNotIn("k", bb)
            self.assertEqual(len(bb), 0)
            self.assertEqual(bb.keys(), [])
            self.assertEqual(bb.snapshot(), {})

    def test_write_after_expiry_resets_version(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("k", "v1", ttl_s=5)
            clock.advance(6)
            entry = bb.write("k", "v2")
        self.assertEqual(entry.version, 1)  # 过期条目被当作不存在

    def test_list_can_include_expired_entries(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("k", 1, ttl_s=5)
            clock.advance(6)
            self.assertEqual(bb.list(), [])
            self.assertEqual([e.key for e in bb.list(include_expired=True)], ["k"])

    def test_ttl_zero_is_immediately_expired(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("k", 1, ttl_s=0)
            self.assertIsNone(bb.read("k"))

    def test_delete_of_expired_entry_returns_false(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("k", 1, ttl_s=1)
            clock.advance(2)
            self.assertFalse(bb.delete("k"))


# ======================================================================================
# snapshot / history
# ======================================================================================


class SnapshotHistoryTests(unittest.TestCase):
    """§10.2.1：`snapshot()` / `keys()` / `list()` 一律返回快照副本。"""

    def test_snapshot_is_a_copy_not_the_internal_dict(self) -> None:
        bb = Blackboard()
        bb.write("a", 1)
        snap = bb.snapshot()
        self.assertEqual(snap, {"a": 1})
        snap["injected"] = True
        snap["a"] = "mutated"
        self.assertNotIn("injected", bb)
        self.assertEqual(bb.read("a"), 1)  # 改副本不影响黑板

    def test_list_and_keys_are_copies(self) -> None:
        """返回的是**新的 list**（不与内部 dict 共享容器），就像 `MemoryStore.all()`。

        注意 frozen 的取舍（§10.2.1 "value 不做深拷贝（性能）"）：容器是副本，
        但 **entry 对象本身是共享的** —— 所以这里只断言"改动返回的 list 不影响黑板"，
        不去改 entry 的字段（那会改到内部 state，是文档明令禁止的用法）。
        """
        bb = Blackboard()
        bb.write("a", 1)
        entries = bb.list()
        entries.append(BlackboardEntry(key="zzz", value=0))
        self.assertEqual(len(bb.list()), 1)
        keys = bb.keys()
        keys.append("injected")
        self.assertEqual(bb.keys(), ["a"])

    def test_snapshot_excludes_expired(self) -> None:
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            bb.write("keep", 1)
            bb.write("gone", 2, ttl_s=1)
            clock.advance(2)
            self.assertEqual(bb.snapshot(), {"keep": 1})

    def test_history_records_writes_in_ascending_time_order(self) -> None:
        bb = Blackboard()
        bb.write("a", 1)
        bb.write("b", 2)
        bb.write("a", 3)
        history = bb.history()
        self.assertEqual([(e.key, e.version) for e in history], [("a", 1), ("b", 1), ("a", 2)])

    def test_history_limit_takes_the_newest(self) -> None:
        bb = Blackboard()
        for value in range(5):
            bb.write("k", value)
        self.assertEqual([e.value for e in bb.history(limit=2)], [3, 4])

    def test_history_non_positive_limit_returns_empty(self) -> None:
        bb = Blackboard()
        bb.write("k", 1)
        self.assertEqual(bb.history(limit=0), [])
        self.assertEqual(bb.history(limit=-5), [])


# ======================================================================================
# max_entries 淘汰
# ======================================================================================


class MaxEntriesTests(unittest.TestCase):
    """§10.2：超过 max_entries -> 淘汰最旧的（按 updated_at）并记 WARNING。"""

    def test_eviction_drops_the_oldest_updated_at(self) -> None:
        clock = _Clock()
        bb = Blackboard(max_entries=2)
        with _patch_clock(clock):
            bb.write("a", 1)
            clock.advance(1)
            bb.write("b", 2)
            clock.advance(1)
            bb.write("c", 3)
        self.assertEqual(len(bb), 2)
        self.assertEqual(bb.keys(), ["b", "c"])
        self.assertNotIn("a", bb)

    def test_lru_refresh_protects_a_key_from_eviction(self) -> None:
        clock = _Clock()
        bb = Blackboard(max_entries=2)
        with _patch_clock(clock):
            bb.write("a", 1)
            clock.advance(1)
            bb.write("b", 2)
            clock.advance(1)
            bb.write("a", 3)  # 刷新 a 的 updated_at
            clock.advance(1)
            bb.write("c", 4)  # 现在最旧的是 b
        self.assertEqual(bb.keys(), ["a", "c"])

    def test_eviction_logs_a_warning(self) -> None:
        bb = Blackboard(max_entries=1)
        bb.write("a", 1)
        with self.assertLogs("liteagent.multiagent", level="WARNING") as captured:
            bb.write("b", 2)
        self.assertTrue(
            any("max_entries" in line for line in captured.output),
            msg=f"expected an eviction warning, got {captured.output}",
        )

    def test_non_positive_max_entries_is_rejected_loudly(self) -> None:
        for bad in (0, -1):
            with self.subTest(max_entries=bad):
                with self.assertRaises(ConfigError):
                    Blackboard(max_entries=bad)


# ======================================================================================
# subscribe
# ======================================================================================


class SubscribeTests(unittest.TestCase):
    """§10.2：`subscribe` 返回**幂等**的取消函数；回调在锁外调用。"""

    def test_subscriber_receives_every_write(self) -> None:
        bb = Blackboard()
        seen: list[tuple[str, int]] = []
        bb.subscribe(lambda key, entry: seen.append((key, entry.version)))
        bb.write("a", 1)
        bb.write("a", 2)
        bb.write("b", 1)
        self.assertEqual(seen, [("a", 1), ("a", 2), ("b", 1)])

    def test_unsubscribe_stops_callbacks_and_is_idempotent(self) -> None:
        bb = Blackboard()
        seen: list[str] = []
        unsubscribe = bb.subscribe(lambda key, entry: seen.append(key))
        bb.write("a", 1)
        unsubscribe()
        bb.write("b", 2)
        self.assertEqual(seen, ["a"])
        unsubscribe()  # 第二次是 no-op，不抛
        bb.write("c", 3)
        self.assertEqual(seen, ["a"])

    def test_only_the_unsubscribed_callback_stops(self) -> None:
        bb = Blackboard()
        first: list[str] = []
        second: list[str] = []
        cancel_first = bb.subscribe(lambda key, entry: first.append(key))
        bb.subscribe(lambda key, entry: second.append(key))
        cancel_first()
        bb.write("k", 1)
        self.assertEqual(first, [])
        self.assertEqual(second, ["k"])

    def test_subscriber_error_does_not_break_the_write(self) -> None:
        bb = Blackboard()

        def explode(key: str, entry: BlackboardEntry) -> None:
            raise RuntimeError("subscriber blew up")

        bb.subscribe(explode)
        seen: list[str] = []
        bb.subscribe(lambda key, entry: seen.append(key))
        with self.assertLogs("liteagent.multiagent", level="ERROR"):
            entry = bb.write("k", 1)  # 不抛
        self.assertEqual(entry.version, 1)
        self.assertEqual(seen, ["k"])  # 后续订阅者仍然收到
        self.assertEqual(bb.read("k"), 1)

    def test_subscribe_rejects_non_callable(self) -> None:
        bb = Blackboard()
        with self.assertRaises(ConfigError):
            bb.subscribe("not callable")  # type: ignore[arg-type]


# ======================================================================================
# 低层事件（§2.7：Blackboard 是 blackboard_write / blackboard_read 的唯一发射者）
# ======================================================================================


class EventTests(unittest.TestCase):
    def test_write_emits_blackboard_write(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        bb = Blackboard(on_event=lambda name, data: events.append((name, data)))
        bb.write("k", 1, author="alice")
        self.assertEqual(len(events), 1)
        name, data = events[0]
        self.assertEqual(name, "blackboard_write")
        self.assertEqual(data["key"], "k")
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["author"], "alice")

    def test_read_emits_blackboard_read_with_hit_flag(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        bb = Blackboard(on_event=lambda name, data: events.append((name, data)))
        bb.write("k", 1)
        events.clear()
        bb.read("k")
        bb.read("missing")
        self.assertEqual([name for name, _ in events], ["blackboard_read", "blackboard_read"])
        self.assertEqual([data["hit"] for _, data in events], [True, False])

    def test_on_event_failure_does_not_break_write(self) -> None:
        def boom(name: str, data: dict[str, Any]) -> None:
            raise RuntimeError("observer is broken")

        bb = Blackboard(on_event=boom)
        with self.assertLogs("liteagent.multiagent", level="ERROR"):
            entry = bb.write("k", 1)
        self.assertEqual(entry.value, 1)


# ======================================================================================
# 并发
# ======================================================================================


class ConcurrencyTests(unittest.TestCase):
    """§10.2.1 的并发安全：唯一 RLock 让"读-改-写"不丢更新。"""

    def test_multithreaded_increments_total_exactly_100(self) -> None:
        bb = Blackboard()
        threads_count = 10
        per_thread = 10
        barrier = threading.Barrier(threads_count)
        failures: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(per_thread):
                    bb.increment("counter")
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                failures.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        self.assertEqual(failures, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        # 100 次并发自增一次都不能丢（VersionConflictError 之外的竞态会在这里露馅）
        self.assertEqual(bb.read("counter"), 100)
        self.assertEqual(bb.read_entry("counter").version, 100)

    def test_multithreaded_distinct_writes_do_not_lose_entries(self) -> None:
        bb = Blackboard(max_entries=1000)
        threads_count = 4
        per_thread = 25
        barrier = threading.Barrier(threads_count)

        def worker(offset: int) -> None:
            barrier.wait(timeout=10)
            for index in range(per_thread):
                bb.write(f"k{offset}-{index}", index)

        threads = [
            threading.Thread(target=worker, args=(offset,)) for offset in range(threads_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertEqual(len(bb), threads_count * per_thread)
        self.assertEqual(len(bb.keys()), 100)

    def test_concurrent_snapshot_and_write_never_raise_runtime_error(self) -> None:
        """§10.2.1：`list()` / `keys()` / `snapshot()` 边写边读不抛

        `RuntimeError: dict changed size during iteration`（它们的迭代必须在锁内完成）。
        """
        bb = Blackboard(max_entries=50)
        errors: list[BaseException] = []
        rounds = 200
        barrier = threading.Barrier(4)

        def writer(seed: int) -> None:
            try:
                barrier.wait(timeout=10)
                for index in range(rounds):
                    bb.write(f"k{(seed + index) % 80}", index)
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                errors.append(exc)

        def reader() -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(rounds):
                    bb.snapshot()
                    bb.list()
                    bb.keys()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(7,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertLessEqual(len(bb), 50)


# ======================================================================================
# awatch（异步）
# ======================================================================================


class WatchTests(unittest.IsolatedAsyncioTestCase):
    """§10.2：`awatch` 每次写入 yield 最新 entry；合并语义；总超时。"""

    async def test_awatch_yields_entry_after_write(self) -> None:
        bb = Blackboard()
        received: list[BlackboardEntry] = []
        done = asyncio.Event()

        async def watcher() -> None:
            async for entry in bb.awatch("k", timeout_s=5.0):
                received.append(entry)
                done.set()
                break

        task = asyncio.ensure_future(watcher())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
        bb.write("k", "hello", author="alice")
        await asyncio.wait_for(done.wait(), timeout=5.0)
        await asyncio.wait_for(task, timeout=5.0)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].value, "hello")
        self.assertEqual(received[0].author, "alice")
        self.assertEqual(received[0].version, 1)

    async def test_awatch_can_be_created_before_the_key_exists(self) -> None:
        bb = Blackboard()
        agen = bb.awatch("later", timeout_s=5.0)
        pending = asyncio.ensure_future(agen.__anext__())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("later"))))
        bb.write("later", 7)
        entry = await asyncio.wait_for(pending, timeout=5.0)
        self.assertEqual(entry.value, 7)
        await agen.aclose()

    async def test_awatch_merges_burst_writes_into_the_latest(self) -> None:
        """合并语义（§10.2 [v2 冻结]）：多次写入快于消费时只 yield 最新一条。"""
        bb = Blackboard()
        agen = bb.awatch("k", timeout_s=5.0)
        pending = asyncio.ensure_future(agen.__anext__())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
        bb.write("k", "a")
        bb.write("k", "b")  # 两次同步写，中间不让出调度
        entry = await asyncio.wait_for(pending, timeout=5.0)
        self.assertEqual(entry.value, "b")
        self.assertEqual(entry.version, 2)
        await agen.aclose()

    async def test_awatch_deregisters_on_aclose(self) -> None:
        bb = Blackboard()
        agen = bb.awatch("k", timeout_s=5.0)
        pending = asyncio.ensure_future(agen.__anext__())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
        bb.write("k", 1)
        await asyncio.wait_for(pending, timeout=5.0)
        await agen.aclose()
        self.assertTrue(await _wait_for(lambda: not bb._watch_loops.get("k")))

    async def test_awatch_timeout_raises_timeout_error(self) -> None:
        """`timeout_s=0.0` 在实现里等价于"立刻到点"（remaining==0 -> 抛），不依赖真实时钟。"""
        bb = Blackboard()
        agen = bb.awatch("k", timeout_s=0.0)
        with self.assertRaises(asyncio.TimeoutError):
            await agen.__anext__()

    async def test_awatch_ignores_other_keys(self) -> None:
        bb = Blackboard()
        agen = bb.awatch("k", timeout_s=5.0)
        pending = asyncio.ensure_future(agen.__anext__())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
        bb.write("other", 1)  # 不该唤醒
        self.assertFalse(pending.done())
        bb.write("k", 2)
        entry = await asyncio.wait_for(pending, timeout=5.0)
        self.assertEqual(entry.value, 2)
        await agen.aclose()

    # ----------------------------------------------------------------------------------
    # [v3 回归] 丢唤醒 / 版本回退 / 登记顺序 —— 这三条都曾确定性复现
    # ----------------------------------------------------------------------------------

    async def test_awatch_delivers_write_that_lands_while_consumer_handles_previous(self) -> None:
        """[v3 回归] 消费者**正在处理上一条 entry** 时发生的写入不得丢失。

        v2 的缺陷：唤醒是提示式的（`notify_all` 在没有等待者时是空操作），而版本比对
        只在 `cond.wait()` **返回之后**才执行 —— 于是写入落在"生成器挂在 yield 上"到
        "重新进入 wait()"之间时，这次唤醒被丢弃，版本比对再也不会执行，消费者永久停在
        旧值（`timeout_s=None` 时就是永久挂起）。这里用 Event 把消费者钉在 body 里，
        再写第二次并让出调度（确保唤醒任务已跑过、丢弃已成事实），最后放行消费者。
        """
        bb = Blackboard()
        received: list[Any] = []
        in_body = asyncio.Event()
        release = asyncio.Event()

        async def watcher() -> None:
            async for entry in bb.awatch("k", timeout_s=5.0):
                received.append(entry.value)
                if entry.value == "v1":
                    in_body.set()
                    await release.wait()  # "消费者正在处理上一条 entry"

        task = asyncio.ensure_future(watcher())
        self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
        bb.write("k", "v1")
        await asyncio.wait_for(in_body.wait(), timeout=5.0)
        # 此刻生成器挂在 yield 上：这次写入的 notify_all 没有任何等待者，会被丢弃
        bb.write("k", "v2")
        await asyncio.sleep(0.05)
        release.set()
        self.assertTrue(await _wait_for(lambda: len(received) >= 2, rounds=200))
        self.assertEqual(received, ["v1", "v2"])
        self.assertEqual(bb.read("k"), "v2")  # 值确实在黑板里，不是"没写进去"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_awatch_delivers_rewrite_after_ttl_expiry(self) -> None:
        """[v3 回归] 条目 TTL 过期后重写（version 从 1 重新开始）不得被静默吞掉。"""
        clock = _Clock()
        bb = Blackboard()
        with _patch_clock(clock):
            agen = bb.awatch("k", timeout_s=5.0)
            pending = asyncio.ensure_future(agen.__anext__())
            self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
            bb.write("k", "v1", ttl_s=5)
            first = await asyncio.wait_for(pending, timeout=5.0)
            self.assertEqual(first.value, "v1")
            self.assertEqual(first.version, 1)
            clock.advance(6)  # v1 过期：读侧把版本视为 0，重写又会得到 version == 1
            pending2 = asyncio.ensure_future(agen.__anext__())
            second_write = bb.write("k", "v2", ttl_s=5)
            self.assertEqual(second_write.version, 1)  # 版本号确实回退了
            second = await asyncio.wait_for(pending2, timeout=5.0)
            self.assertEqual(second.value, "v2")
            await agen.aclose()

    async def test_awatch_delivers_rewrite_after_delete_and_clear(self) -> None:
        """[v3 回归] delete / clear 之后的重写同样不得被静默吞掉（版本号撞回旧值）。"""
        for reset in ("delete", "clear"):
            with self.subTest(reset=reset):
                bb = Blackboard()
                received: list[Any] = []
                agen = bb.awatch("k", timeout_s=5.0)
                pending = asyncio.ensure_future(agen.__anext__())
                self.assertTrue(await _wait_for(lambda: bool(bb._watch_loops.get("k"))))
                bb.write("k", "v1")
                received.append((await asyncio.wait_for(pending, timeout=5.0)).value)
                if reset == "delete":
                    self.assertTrue(bb.delete("k"))
                else:
                    bb.clear()
                pending2 = asyncio.ensure_future(agen.__anext__())
                bb.write("k", "v2")  # version 又是 1
                received.append((await asyncio.wait_for(pending2, timeout=5.0)).value)
                self.assertEqual(received, ["v1", "v2"])
                await agen.aclose()

    async def test_awatch_registers_watcher_before_reading_baseline(self) -> None:
        """§10.2.1 第 1 步冻结的顺序是"先登记 watcher、再读基线"。

        反过来（先读基线、再登记）会制造一段"既不转发通知、又不在基线里"的空洞：
        落在两步之间的写入永久丢失。这里直接在基线读取的那一刻观测 watcher 是否已登记。
        """
        bb = Blackboard()
        registered_at_read: list[bool] = []
        real = Blackboard._write_seq_of

        def hooked(key: str) -> int:
            registered_at_read.append(bool(bb._watch_loops.get(key)))
            return real(bb, key)

        with mock.patch.object(bb, "_write_seq_of", side_effect=hooked):
            agen = bb.awatch("k", timeout_s=0.0)
            with self.assertRaises(asyncio.TimeoutError):
                await agen.__anext__()
            await agen.aclose()
        self.assertTrue(registered_at_read, "基线读取根本没有发生")
        self.assertTrue(all(registered_at_read),
                        f"读基线时 watcher 尚未登记：{registered_at_read}")

    async def test_async_facade_mirrors_sync_api(self) -> None:
        bb = Blackboard()
        entry = await bb.awrite("k", 1, author="async")
        self.assertEqual(entry.version, 1)
        self.assertEqual(await bb.aread("k"), 1)
        self.assertEqual((await bb.aread_entry("k")).author, "async")
        self.assertEqual(await bb.asnapshot(), {"k": 1})
        self.assertEqual([e.key for e in await bb.alist()], ["k"])
        self.assertTrue(await bb.adelete("k"))
        await bb.aclear()
        self.assertEqual(len(bb), 0)


# ======================================================================================
# 实测 M-4：写入方在 worker 线程写完、loop 已关闭时不抛异常
# ======================================================================================


class _DeadLoopStub:
    """一个"看起来没关、其实是死的" loop：`is_closed()` 撒谎，`call_soon_threadsafe` 抛。

    用来覆盖 §10.2.1 第 2 步里 `except RuntimeError` 的那条分支（`is_closed()`
    判定与真正的 `RuntimeError` 之间永远存在竞态窗口）。
    """

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Event loop is closed")


class ClosedLoopTests(unittest.TestCase):
    """实测 M-4：`loop.call_soon_threadsafe` 对已关闭的 loop 抛 RuntimeError。

    约束（§13 红线 6/10）：一次普通的同步 `write` **绝不能**把异常抛给调用方，
    但必须留下日志痕迹。
    """

    def test_write_from_worker_thread_after_watcher_loop_closed(self) -> None:
        bb = Blackboard()
        loop = asyncio.new_event_loop()
        try:
            async def prime() -> "asyncio.Future[Any]":
                agen = bb.awatch("k", timeout_s=None)
                pending = asyncio.ensure_future(agen.__anext__())
                # 让生成器跑到 cond.wait()（注册已完成）
                for _ in range(50):
                    if bb._watch_loops.get("k"):
                        break
                    await asyncio.sleep(0)
                self.assertTrue(
                    bb._watch_loops.get("k"),
                    msg="awatch 未能在 loop 上登记 watcher，测试前提不成立",
                )
                return pending

            pending = loop.run_until_complete(prime())
            self.assertFalse(pending.done())
        finally:
            loop.close()
        self.assertTrue(loop.is_closed())
        # 关键前提：watcher 仍然登记在那个已关闭的 loop 上（loop 关闭不会跑生成器的
        # finally），所以这次 write 一定会走到 `_forward_notify` 的 is_closed() 分支。
        self.assertTrue(bb._watch_loops.get("k"))

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                bb.write("k", 42)
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [], "写入方在 loop 已关闭时不该收到异常（实测 M-4）")
        self.assertEqual(bb.read("k"), 42)

    def test_write_swallows_runtime_error_and_logs(self) -> None:
        bb = Blackboard()
        bb._watch_loops["k"] = [(_DeadLoopStub(), object())]  # type: ignore[list-item]
        with self.assertLogs("liteagent.multiagent", level="INFO") as captured:
            entry = bb.write("k", 1)
        self.assertEqual(entry.value, 1)
        self.assertTrue(
            any("skip notify" in line for line in captured.output),
            msg=f"expected a 'skip notify' log line, got {captured.output}",
        )

    def test_dead_loop_does_not_block_other_watchers(self) -> None:
        """一个死 loop 只影响它自己那条 watcher 记录，不拖累真正的写入路径。"""
        bb = Blackboard()
        bb._watch_loops["k"] = [(_DeadLoopStub(), object())]  # type: ignore[list-item]
        seen: list[str] = []
        bb.subscribe(lambda key, entry: seen.append(key))
        with self.assertLogs("liteagent.multiagent", level="INFO"):
            bb.write("k", 1)
        self.assertEqual(seen, ["k"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
