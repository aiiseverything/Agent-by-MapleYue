from __future__ import annotations

"""tests/test_memory_vector.py —— §8.5 `memory/vector.py` 的单元测试。

§12 第 5120 行要求的覆盖点：
  * `upsert` 去重
  * `should_auto_write` 三种策略与 marker（**含 `role="assistant"` 恒 False**）
  * **注入 dim=4 的 `CallableEmbedder` 后 `vm.dim == 4` 且 add 成功**
  * 混合打分公式手算对照（**传 `now`**）
  * 排序三级稳定
  * MMR 去冗
  * `min_score` 过滤
  * `metadata_filter`
  * `max_items` FIFO 淘汰
  * `access_count` 递增
  * `score_breakdown`

所有相似度都用 `tests.helpers.det_embedder` 的查找表控制（零随机、零网络）。
"""

import unittest

from liteagent.errors import MemoryStoreError
from liteagent.llm.message import Message, Role
from liteagent.memory.base import MemoryItem
from liteagent.memory.embeddings import CallableEmbedder, HashingEmbedder
from liteagent.memory.vector import (
    AUTO_WRITE_MARKERS,
    VectorConfig,
    VectorMemory,
)
from tests.helpers import det_embedder

#: 一个 7 天的秒数（`recency` 手算时用得上）。
_WEEK = 7.0 * 86400.0

#: 手算用的一组单位向量（det_embedder 会做 L2 归一化，这几个已经是单位向量）。
_UNIT_X = [1.0, 0.0, 0.0, 0.0]
_UNIT_Y = [0.0, 1.0, 0.0, 0.0]
_UNIT_SKEW = [1.0, 0.5, 0.0, 0.0]


def _memory(table: dict, **config_kwargs) -> VectorMemory:
    return VectorMemory(embedder=det_embedder(table), config=VectorConfig(**config_kwargs))


def _item(item_id: str, content: str, *, created_at: float = 1000.0, importance: float = 0.5,
          metadata: dict | None = None) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        content=content,
        created_at=created_at,
        last_access_at=created_at,
        importance=importance,
        metadata=dict(metadata) if metadata else {},
    )


class DimensionAuthorityTests(unittest.TestCase):
    """§8.5 的维度权威性：注入的 embedder 的 dim 是权威。"""

    def test_injected_dim4_embedder_is_authoritative(self) -> None:
        table = {"q": _UNIT_X, "alpha": _UNIT_X}
        vm = _memory(table)
        self.assertEqual(vm.dim, 4)
        self.assertEqual(vm.config.dim, 4)
        self.assertIsInstance(vm.embedder, CallableEmbedder)
        # add 成功（若维度校验还是按 config.dim=256 走，这里会抛 MemoryStoreError）。
        vm.add(_item("m1", "alpha"))
        item = vm.get("m1")
        self.assertIsNotNone(item)
        self.assertEqual(len(item.embedding), 4)  # type: ignore[arg-type]
        self.assertEqual(len(vm), 1)

    def test_injected_embedder_beats_config_dim(self) -> None:
        vm = _memory({"alpha": _UNIT_X}, dim=64)
        self.assertEqual(vm.dim, 4)
        self.assertEqual(vm.config.dim, 4)

    def test_default_embedder_uses_config_dim(self) -> None:
        vm = VectorMemory(config=VectorConfig(dim=32))
        self.assertEqual(vm.dim, 32)
        self.assertIsInstance(vm.embedder, HashingEmbedder)
        vm.add(MemoryItem(id="m", content="hello", created_at=1.0))
        self.assertEqual(len(vm.get("m").embedding), 32)  # type: ignore[arg-type]

    def test_explicit_wrong_dim_embedding_raises(self) -> None:
        vm = _memory({"alpha": _UNIT_X})
        item = _item("m1", "alpha")
        item.embedding = [0.0] * 8
        with self.assertRaises(MemoryStoreError):
            vm.add(item)
        self.assertEqual(len(vm), 0)

    def test_embedder_returning_wrong_dim_raises_at_write_time(self) -> None:
        """注入的 embedder 撒谎（dim 声明 4、实际给 3 维）必须在写入时就炸出来。"""
        vm = VectorMemory(embedder=CallableEmbedder(lambda texts: [[0.0, 0.0, 0.0]], dim=4))
        with self.assertRaises(MemoryStoreError):
            vm.add(_item("m1", "whatever"))

    def test_missing_embedding_is_computed_once(self) -> None:
        calls: list[list[str]] = []

        def _fn(texts):
            calls.append(list(texts))
            return [_UNIT_X for _ in texts]

        vm = VectorMemory(embedder=CallableEmbedder(_fn, dim=4))
        vm.add(_item("m1", "alpha"))
        self.assertEqual(calls, [["alpha"]])
        # 已经有 embedding 的条目不会再调 embedder。
        item = _item("m2", "alpha")
        item.embedding = list(_UNIT_X)
        vm.add(item)
        self.assertEqual(len(calls), 1)


class UpsertDedupTests(unittest.TestCase):
    """§8.5 `upsert`：同 id 与近似内容两条去重规则。"""

    def setUp(self) -> None:
        self.vm = _memory({"alpha": _UNIT_X, "alpha2": _UNIT_X, "beta": _UNIT_Y}, max_items=100)

    def test_same_id_updates_in_place(self) -> None:
        first = _item("m1", "alpha")
        first.access_count = 5
        self.vm.add(first)
        created_before = first.created_at
        incoming = _item("m1", "alpha2", created_at=9999.0, importance=0.9)
        final, created = self.vm.upsert(incoming)
        self.assertFalse(created)
        self.assertIs(final, first)
        self.assertEqual(len(self.vm), 1)
        self.assertEqual(final.content, "alpha2")
        self.assertEqual(final.access_count, 5)          # 保留历史使用情况
        self.assertEqual(final.importance, 0.9)          # 取 max
        self.assertGreater(final.created_at, created_before)  # created_at 刷新

    def test_near_duplicate_content_merges_into_existing(self) -> None:
        self.vm.add(_item("m1", "alpha"))
        incoming = _item("m2", "alpha2")  # 同一向量 -> sim == 1.0 >= dedup_threshold
        final, created = self.vm.upsert(incoming)
        self.assertFalse(created)
        self.assertEqual(final.id, "m1")
        self.assertEqual(len(self.vm), 1)
        self.assertEqual(final.content, "alpha2")

    def test_new_content_creates_a_new_entry(self) -> None:
        self.vm.add(_item("m1", "alpha"))
        final, created = self.vm.upsert(_item("m2", "beta"))
        self.assertTrue(created)
        self.assertEqual(final.id, "m2")
        self.assertEqual(len(self.vm), 2)

    def test_threshold_boundary_is_respected(self) -> None:
        # 把门槛抬到 1.01 -> 任何内容都不再合并（含完全相同的向量）。
        vm = _memory({"alpha": _UNIT_X, "alpha2": _UNIT_X}, dedup_threshold=1.01)
        vm.add(_item("m1", "alpha"))
        _, created = vm.upsert(_item("m2", "alpha2"))
        self.assertTrue(created)
        self.assertEqual(len(vm), 2)

    def test_merge_keeps_metadata_union_and_drops_stale_score(self) -> None:
        target = _item("m1", "alpha", metadata={"tag": "old"})
        self.vm.add(target)
        target.score = 0.5
        target.score_breakdown = {"sim": 0.5}
        incoming = _item("m2", "alpha2", metadata={"extra": "new"})
        final, _ = self.vm.upsert(incoming)
        self.assertEqual(final.metadata, {"tag": "old", "extra": "new"})
        self.assertIsNone(final.score)
        self.assertIsNone(final.score_breakdown)

    def test_upsert_does_not_recompute_a_provided_embedding(self) -> None:
        """条目的 embedding 已就绪时不得再调 embedder（内容可以不在查找表里）。"""
        vm = VectorMemory(
            embedder=det_embedder({"alpha": _UNIT_X}), config=VectorConfig(dedup_threshold=1.01)
        )
        item = _item("m1", "not-in-the-table")
        item.embedding = list(_UNIT_X)
        final, created = vm.upsert(item)
        self.assertTrue(created)
        self.assertIs(final, item)


class ShouldAutoWriteTests(unittest.TestCase):
    """§8.5 `should_auto_write`：三种策略 + marker + **role 硬约束不可豁免**。"""

    def test_manual_policy_never_writes(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="manual")
        self.assertFalse(vm.should_auto_write(Message.user("记住我喜欢用 uv 管理依赖")))
        self.assertFalse(vm.should_auto_write(Message.user("x" * 200)))

    def test_turn_policy_writes_every_user_message(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="turn")
        self.assertTrue(vm.should_auto_write(Message.user("hi")))
        self.assertTrue(vm.should_auto_write(Message.user("!!!")))  # turn 不走启发式

    def test_selective_policy_requires_length_or_marker(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="selective", auto_write_min_chars=40)
        self.assertTrue(vm.should_auto_write(Message.user("x" * 40)))
        self.assertFalse(vm.should_auto_write(Message.user("hello there")))
        self.assertTrue(vm.should_auto_write(Message.user("记住我喜欢用 uv")))

    def test_every_frozen_marker_hits(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="selective", auto_write_min_chars=10_000)
        for marker in AUTO_WRITE_MARKERS:
            self.assertTrue(
                vm.should_auto_write(Message.user(f"please {marker} this")), marker
            )

    def test_markers_are_case_insensitive(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="selective", auto_write_min_chars=10_000)
        self.assertTrue(vm.should_auto_write(Message.user("Remember my preference")))

    def test_symbol_only_content_is_rejected(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="selective", auto_write_min_chars=10)
        self.assertFalse(vm.should_auto_write(Message.user("!" * 50)))
        self.assertFalse(vm.should_auto_write(Message.user("   " * 20)))

    def test_role_assistant_is_always_false(self) -> None:
        for policy in ("selective", "turn", "manual"):
            vm = _memory({"x": _UNIT_X}, write_policy=policy, auto_write_min_chars=10)
            message = Message.assistant("remember my name is claude " + "x" * 100)
            self.assertFalse(vm.should_auto_write(message), policy)

    def test_other_roles_are_always_false(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="turn")
        self.assertFalse(vm.should_auto_write(Message(role=Role.SYSTEM, content="x" * 100)))
        self.assertFalse(vm.should_auto_write(Message(role=Role.TOOL, content="x" * 100)))

    def test_unknown_policy_warns_and_refuses(self) -> None:
        vm = _memory({"x": _UNIT_X}, write_policy="sometimes")
        with self.assertWarns(RuntimeWarning):
            self.assertFalse(vm.should_auto_write(Message.user("x" * 100)))


class HybridScoreTests(unittest.TestCase):
    """§8.5.1 的混合打分公式：手算对照（必须传 `now`）。"""

    def test_manual_computation_of_score(self) -> None:
        now = 2_000_000.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        item = _item("m1", "alpha", created_at=now - _WEEK, importance=0.8)
        vm.add(item)

        # sim = 1.0（同向量）；recency = 2 ** (-7/7) = 0.5；importance = 0.8
        # score = 1.0*1.0 + 0.15*0.5 + 0.1*0.8 = 1.155
        expected = 1.0 * 1.0 + 0.15 * 0.5 + 0.1 * 0.8
        self.assertAlmostEqual(vm.score_item(item, _UNIT_X, now=now), expected, places=12)
        self.assertAlmostEqual(vm.score_item(item, None, now=now), 0.15 * 0.5 + 0.1 * 0.8, places=12)

        results = vm.search("q", limit=1, now=now)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, expected, places=12)
        self.assertEqual(
            results[0].score_breakdown,
            {"sim": 1.0, "recency": 0.5, "importance": 0.8, "score": expected},
        )

    def test_recency_halves_after_one_half_life(self) -> None:
        now = 5_000_000.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        vm.add(_item("m1", "alpha", created_at=now - _WEEK, importance=0.0))
        self.assertAlmostEqual(
            vm.score_item(vm.get("m1"), _UNIT_X, now=now), 1.0 + 0.15 * 0.5, places=12
        )

    def test_importance_is_clamped(self) -> None:
        now = 1.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        vm.add(_item("m1", "alpha", created_at=now, importance=5.0))
        # importance 被 clamp 到 1.0 -> score = 1 + 0.15*1 + 0.1*1
        self.assertAlmostEqual(vm.score_item(vm.get("m1"), _UNIT_X, now=now), 1.25, places=12)

    def test_negative_similarity_is_filtered_out(self) -> None:
        now = 1.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X, "neg": [-1.0, 0.0, 0.0, 0.0]})
        vm.add(_item("m1", "alpha", created_at=now))
        vm.add(_item("m2", "neg", created_at=now))
        results = vm.search("q", limit=5, use_mmr=False, now=now)
        self.assertEqual([i.id for i in results], ["m1"])

    def test_empty_query_degrades_to_recency_and_importance(self) -> None:
        """`query_vec is None` 时 sim 项为 0（spec §8.5 明文）。空查询仍应排序返回。"""
        now = 1.0
        vm = _memory({"alpha": _UNIT_X})
        vm.add(_item("m1", "alpha", created_at=now, importance=0.5))
        results = vm.search("", limit=1, now=now)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 0.15 * 1.0 + 0.1 * 0.5, places=12)

    def test_weights_come_from_config(self) -> None:
        now = 1_000.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X}, w_sim=2.0, w_recency=0.0, w_importance=0.0)
        vm.add(_item("m1", "alpha", created_at=now))
        self.assertAlmostEqual(vm.score_item(vm.get("m1"), _UNIT_X, now=now), 2.0, places=12)


class OrderingAndFilterTests(unittest.TestCase):
    """排序三级稳定 / MMR / min_score / metadata_filter。"""

    def setUp(self) -> None:
        self.now = 10_000.0

    def test_three_level_stable_sort(self) -> None:
        # 关掉 recency/importance 权重 -> 所有人的 score 都是 sim，逼出后两级排序键。
        vm = _memory(
            {"q": _UNIT_X, "x": _UNIT_X, "y": _UNIT_X, "z": _UNIT_X},
            w_recency=0.0,
            w_importance=0.0,
        )
        vm.add(_item("b", "x", created_at=200.0))
        vm.add(_item("a", "y", created_at=200.0))
        vm.add(_item("c", "z", created_at=300.0))
        vm.add(_item("d", "x", created_at=100.0))
        results = vm.search("q", limit=4, use_mmr=False, now=self.now)
        # 先按 -score（全等），再按 -created_at（300 > 200 > 100），最后按 id 升序。
        self.assertEqual([i.id for i in results], ["c", "a", "b", "d"])

    def test_mmr_dedupes_near_identical_results(self) -> None:
        table = {"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_X, "c": _UNIT_SKEW}
        vm = _memory(table, mmr_lambda=0.3)
        vm.add(_item("a", "a", created_at=self.now))
        vm.add(_item("b", "b", created_at=self.now))
        vm.add(_item("c", "c", created_at=self.now))

        without_mmr = vm.search("q", limit=2, use_mmr=False, now=self.now)
        self.assertEqual([i.id for i in without_mmr], ["a", "b"])  # 两条几乎一样 -> 浪费一个名额

        with_mmr = vm.search("q", limit=2, use_mmr=True, now=self.now)
        self.assertEqual([i.id for i in with_mmr], ["a", "c"])

    def test_mmr_lambda_one_degrades_to_top_k(self) -> None:
        table = {"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_X, "c": _UNIT_SKEW}
        vm = _memory(table, mmr_lambda=1.0)
        for item_id, content in (("a", "a"), ("b", "b"), ("c", "c")):
            vm.add(_item(item_id, content, created_at=self.now))
        results = vm.search("q", limit=2, use_mmr=True, now=self.now)
        self.assertEqual([i.id for i in results], ["a", "b"])

    def test_min_score_filters_by_score(self) -> None:
        vm = _memory(
            {"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_Y},
            w_recency=0.0,
            w_importance=0.0,
        )
        vm.add(_item("a", "a", created_at=self.now))
        vm.add(_item("b", "b", created_at=self.now))
        self.assertEqual([i.id for i in vm.search("q", now=self.now, use_mmr=False)], ["a", "b"])
        filtered = vm.search("q", min_score=0.5, now=self.now, use_mmr=False)
        self.assertEqual([i.id for i in filtered], ["a"])
        # 参数优先于 config 的默认门槛。
        vm2 = _memory(
            {"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_Y},
            w_recency=0.0,
            w_importance=0.0,
            retrieve_min_score=0.5,
        )
        vm2.add(_item("a", "a", created_at=self.now))
        vm2.add(_item("b", "b", created_at=self.now))
        self.assertEqual([i.id for i in vm2.search("q", now=self.now, use_mmr=False)], ["a"])

    def test_metadata_filter_matches_metadata_only(self) -> None:
        vm = _memory(
            {"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_X},
            w_recency=0.0,
            w_importance=0.0,
        )
        vm.add(_item("a", "a", created_at=self.now, metadata={"tag": "keep", "role": "user"}))
        vm.add(_item("b", "b", created_at=self.now, metadata={"tag": "drop"}))
        kept = vm.search("q", metadata_filter={"tag": "keep"}, now=self.now, use_mmr=False)
        self.assertEqual([i.id for i in kept], ["a"])
        self.assertEqual(
            [i.id for i in vm.search("q", metadata_filter={}, now=self.now, use_mmr=False)],
            ["a", "b"],
        )
        self.assertEqual(
            [i.id for i in vm.search("q", metadata_filter={"role": "user"}, now=self.now)],
            ["a"],
        )
        self.assertEqual(
            vm.search("q", metadata_filter={"nope": 1}, now=self.now), []
        )

    def test_limit_zero_and_empty_store_return_empty(self) -> None:
        vm = _memory({"q": _UNIT_X, "a": _UNIT_X})
        self.assertEqual(vm.search("q", limit=0), [])
        vm.add(_item("a", "a"))
        self.assertEqual(vm.search("q", limit=0), [])
        empty = _memory({"q": _UNIT_X})
        self.assertEqual(empty.search("q"), [])

    def test_limit_is_respected(self) -> None:
        table = {"q": _UNIT_X}
        for index in range(6):
            table[f"t{index}"] = [1.0, 0.1 * index, 0.0, 0.0]
        vm = _memory(table)
        for index in range(6):
            vm.add(_item(f"m{index}", f"t{index}", created_at=self.now))
        self.assertEqual(len(vm.search("q", limit=3, now=self.now, use_mmr=False)), 3)


class SideEffectTests(unittest.TestCase):
    """§8.5.1 第 6 步：命中的条目会被写回 `score` / `access_count` / `last_access_at`。"""

    def test_access_count_and_last_access_at_are_updated(self) -> None:
        vm = _memory({"q": _UNIT_X, "a": _UNIT_X})
        item = _item("a", "a", created_at=100.0)
        vm.add(item)
        self.assertEqual(item.access_count, 0)
        vm.search("q", limit=1, now=500.0)
        self.assertEqual(item.access_count, 1)
        self.assertEqual(item.last_access_at, 500.0)
        vm.search("q", limit=1, now=900.0)
        self.assertEqual(item.access_count, 2)
        self.assertEqual(item.last_access_at, 900.0)

    def test_non_hits_are_not_touched(self) -> None:
        vm = _memory({"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_Y},
                     w_recency=0.0, w_importance=0.0)
        hit = _item("a", "a", created_at=100.0)
        miss = _item("b", "b", created_at=100.0)
        vm.add(hit)
        vm.add(miss)
        vm.search("q", min_score=0.5, limit=1, now=500.0)
        self.assertEqual(hit.access_count, 1)
        self.assertEqual(miss.access_count, 0)
        self.assertIsNone(miss.score)
        self.assertIsNone(miss.score_breakdown)


class CapacityAndStoreApiTests(unittest.TestCase):
    """`max_items` FIFO 淘汰与 MemoryStore 的基本契约。"""

    def test_fifo_eviction_keeps_newest(self) -> None:
        vm = _memory(
            {"a": _UNIT_X, "b": _UNIT_X, "c": _UNIT_X}, max_items=2
        )
        vm.add(_item("a", "a", created_at=1.0))
        vm.add(_item("b", "b", created_at=2.0))
        with self.assertWarns(RuntimeWarning):
            vm.add(_item("c", "c", created_at=3.0))
        self.assertEqual(len(vm), 2)
        self.assertIsNone(vm.get("a"))
        self.assertEqual([i.id for i in vm.all()], ["b", "c"])

    def test_re_adding_same_id_does_not_evict(self) -> None:
        vm = _memory({"a": _UNIT_X}, max_items=1)
        vm.add(_item("a", "a", created_at=1.0))
        vm.add(_item("a", "a", created_at=2.0))
        self.assertEqual(len(vm), 1)
        self.assertEqual(vm.get("a").created_at, 2.0)  # type: ignore[union-attr]

    def test_all_is_created_at_ascending_and_shares_instances(self) -> None:
        vm = _memory({"a": _UNIT_X, "b": _UNIT_X}, max_items=10)
        first = _item("a", "a", created_at=20.0)
        second = _item("b", "b", created_at=10.0)
        vm.add(first)
        vm.add(second)
        ordered = vm.all()
        self.assertEqual([i.id for i in ordered], ["b", "a"])
        self.assertIs(ordered[0], second)
        ordered.append(_item("z", "a"))
        self.assertEqual(len(vm.all()), 2)

    def test_get_delete_clear_and_len(self) -> None:
        vm = _memory({"a": _UNIT_X, "b": _UNIT_X})
        vm.add(_item("a", "a"))
        vm.add(_item("b", "b"))
        self.assertIsNotNone(vm.get("a"))
        self.assertIsNone(vm.get("missing"))
        self.assertTrue(vm.delete("a"))
        self.assertFalse(vm.delete("a"))
        self.assertEqual(len(vm), 1)
        vm.clear()
        self.assertEqual(len(vm), 0)
        self.assertEqual(vm.all(), [])

    def test_delete_invalidates_the_matrix_cache(self) -> None:
        vm = _memory({"q": _UNIT_X, "a": _UNIT_X, "b": _UNIT_X})
        vm.add(_item("a", "a"))
        vm.add(_item("b", "b"))
        vm.delete("a")
        results = vm.search("q", limit=5, now=1.0, use_mmr=False)
        self.assertEqual([i.id for i in results], ["b"])

    def test_add_many(self) -> None:
        vm = _memory({"a": _UNIT_X, "b": _UNIT_X}, max_items=10)
        vm.add_many([_item("a", "a"), _item("b", "b")])
        self.assertEqual(len(vm), 2)

    def test_stats_reports_effective_settings(self) -> None:
        vm = _memory({"a": _UNIT_X}, write_policy="turn", max_items=7, dedup_threshold=0.5)
        stats = vm.stats()
        self.assertEqual(stats["name"], "vector")
        self.assertEqual(stats["items"], 0)
        self.assertEqual(stats["dim"], 4)
        self.assertEqual(stats["embedder"], "det")
        self.assertEqual(stats["write_policy"], "turn")
        self.assertEqual(stats["max_items"], 7)
        self.assertEqual(stats["dedup_threshold"], 0.5)
        self.assertEqual(
            stats["weights"], {"sim": 1.0, "recency": 0.15, "importance": 0.1}
        )
        self.assertIn("use_numpy", stats)


class FutureTimestampClampTests(unittest.TestCase):
    """回归：`now < item.created_at`（时钟回拨 / 从磁盘 load 出未来时间戳）时的 age 钳制。

    修复前 `_score_parts` 直接算 `2.0 ** (-age_days / half_life_days)`：负 age 让**指数为正**，
    轻则 `recency > 1.0` 违反 §8.5.1 冻结的「recency ∈ (0, 1]」，重则 `2.0 ** 大正数`
    抛 `OverflowError` 并冲出 `VectorMemory.search`。规范没写负 age 的钳制（值域空缺），
    实现裁决：age 钳到 >= 0 —— "来自未来"等价于"此刻还没开始变旧"。
    """

    def test_now_before_created_at_does_not_raise_and_clamps_recency(self) -> None:
        # 仅未来 10 天：修复前不会溢出，但 recency = 2 ** (10/7) ≈ 2.69 > 1.0（违反值域）。
        now = 1000.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        vm.add(_item("m1", "alpha", created_at=now + 10.0 * 86400.0, importance=0.5))

        results = vm.search("q", limit=1, use_mmr=False, now=now)

        self.assertEqual([i.id for i in results], ["m1"])
        breakdown = results[0].score_breakdown
        self.assertIsNotNone(breakdown)
        recency = breakdown["recency"]  # type: ignore[index]
        self.assertEqual(recency, 1.0)  # 钳到"刚刚写入"，而不是 2.69
        self.assertLessEqual(recency, 1.0)
        self.assertGreater(recency, 0.0)
        # sim=1.0, recency=1.0, importance=0.5 -> score = 1 + 0.15 + 0.05
        self.assertAlmostEqual(breakdown["score"], 1.2, places=12)  # type: ignore[index]

    def test_far_future_timestamp_does_not_overflow(self) -> None:
        # 未来约 1e12 秒（≈ 3.2 万年，restore 出脏数据/时钟大幅回拨的极端形态）：
        # 修复前指数为 +1e12/7/86400 量级 -> 必抛 OverflowError: (34, 'Numerical result out of range')。
        now = 1000.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        vm.add(_item("m1", "alpha", created_at=now + 1.0e12))

        results = vm.search("q", limit=1, use_mmr=False, now=now)  # 不得抛

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score_breakdown["recency"], 1.0)  # type: ignore[index]

    def test_score_item_is_finite_for_future_items(self) -> None:
        now = 42.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        item = _item("m1", "alpha", created_at=now + 5.0e11, importance=0.0)
        vm.add(item)

        score = vm.score_item(item, _UNIT_X, now=now)  # 不得抛

        # sim=1.0, recency=1.0, importance=0.0 -> 1.0 + 0.15
        self.assertAlmostEqual(score, 1.15, places=12)
        self.assertLess(score, float("inf"))

    def test_every_future_offset_keeps_recency_in_range(self) -> None:
        now = 7.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X})
        # 覆盖"略微未来 -> 极端未来"整条路径，值域不变量必须恒成立。
        for offset in (1.0, 86400.0, 10.0 * 86400.0, 1.0e6 * 86400.0, 1.0e10 * 86400.0):
            vm.clear()
            vm.add(_item("m1", "alpha", created_at=now + offset, importance=0.3))
            results = vm.search("q", limit=1, use_mmr=False, now=now)
            self.assertEqual(len(results), 1, offset)
            recency = results[0].score_breakdown["recency"]  # type: ignore[index]
            self.assertTrue(0.0 < recency <= 1.0, (offset, recency))

    def test_degenerate_half_life_still_treats_future_as_not_aged(self) -> None:
        # half_life_days <= 0 的阶梯分支：未来时间戳同样钳到 age=0 -> recency 1.0。
        now = 3.0
        vm = _memory({"q": _UNIT_X, "alpha": _UNIT_X}, half_life_days=0.0)
        vm.add(_item("m1", "alpha", created_at=now + 1.0e9, importance=0.0))
        results = vm.search("q", limit=1, use_mmr=False, now=now)
        self.assertEqual(results[0].score_breakdown["recency"], 1.0)  # type: ignore[index]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
