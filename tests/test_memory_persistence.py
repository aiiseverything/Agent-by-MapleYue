from __future__ import annotations

"""tests/test_memory_persistence.py —— §8.5/§8.6 的持久化（`[v2 新增]`）。

§12 第 5121 行要求的覆盖点：
  * `save`/`load` 往返后 `len` 与 `search` 顺序一致
  * embedding 逐元素相等
  * 坏行跳过
  * 维度不符抛 `MemoryStoreError`
  * `MemoryManager.persist`/`restore`
"""

import json
import os
import tempfile
import unittest

from liteagent.errors import ConfigError, MemoryStoreError
from liteagent.memory.base import MemoryConfig, MemoryItem
from liteagent.memory.manager import MemoryManager
from liteagent.memory.vector import VectorConfig, VectorMemory
from tests.helpers import det_embedder

_NOW = 1_700_000_000.0

#: dim=4 的确定性查找表（所有向量都是单位向量，检索顺序完全可预测）。
_TABLE = {
    "q": [1.0, 0.0, 0.0, 0.0],
    "alpha": [1.0, 0.0, 0.0, 0.0],
    "beta": [0.0, 1.0, 0.0, 0.0],
    "gamma": [0.0, 0.0, 1.0, 0.0],
    "delta": [1.0, 0.5, 0.0, 0.0],
}


def _memory(table: dict | None = None, **kwargs) -> VectorMemory:
    return VectorMemory(
        embedder=det_embedder(_TABLE if table is None else table),
        config=VectorConfig(**kwargs),
    )


def _item(item_id: str, content: str, *, created_at: float = _NOW - 86400.0) -> MemoryItem:
    return MemoryItem(
        id=item_id, content=content, created_at=created_at, last_access_at=created_at
    )


class SaveLoadRoundtripTests(unittest.TestCase):
    """save -> load 的往返必须保内容、保顺序、保向量。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-persist-")
        self.path = os.path.join(self._tmp.name, "nested", "vector.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip_preserves_len_search_order_and_embeddings(self) -> None:
        source = _memory()
        source.add(_item("m1", "alpha", created_at=_NOW - 3 * 86400.0))
        source.add(_item("m2", "beta", created_at=_NOW - 2 * 86400.0))
        source.add(_item("m3", "gamma", created_at=_NOW - 1 * 86400.0))

        written = source.save(self.path)
        self.assertEqual(written, 3)
        self.assertTrue(os.path.exists(self.path))  # 父目录被自动创建

        restored = _memory()
        loaded = restored.load(self.path)
        self.assertEqual(loaded, 3)
        self.assertEqual(len(restored), len(source))

        # embedding 逐元素相等（不重算、不丢精度）。
        for original in source.all():
            copy = restored.get(original.id)
            self.assertIsNotNone(copy)
            self.assertEqual(copy.embedding, original.embedding)
            self.assertEqual(len(copy.embedding), 4)
            self.assertEqual(
                copy.to_dict(include_embedding=True),
                original.to_dict(include_embedding=True),
            )

        # search 顺序一致（关掉 MMR 以固定为纯分数序）。
        before = [i.id for i in source.search("q", limit=5, now=_NOW, use_mmr=False)]
        after = [i.id for i in restored.search("q", limit=5, now=_NOW, use_mmr=False)]
        self.assertEqual(before, after)
        self.assertTrue(before)

    def test_roundtrip_without_embedding_recomputes_on_load(self) -> None:
        source = _memory()
        source.add(_item("m1", "alpha"))
        self.assertEqual(source.save(self.path, include_embedding=False), 1)
        with open(self.path, encoding="utf-8") as handle:
            payload = json.loads(handle.readline())
        self.assertNotIn("embedding", payload)

        restored = _memory()
        self.assertEqual(restored.load(self.path), 1)
        self.assertEqual(
            restored.get("m1").embedding, source.get("m1").embedding  # type: ignore[union-attr]
        )

    def test_save_overwrites_previous_content(self) -> None:
        first = _memory()
        first.add(_item("m1", "alpha"))
        first.save(self.path)

        second = _memory()
        second.add(_item("m2", "beta"))
        self.assertEqual(second.save(self.path), 1)

        reader = _memory()
        self.assertEqual(reader.load(self.path), 1)
        self.assertEqual(len(reader), 1)
        self.assertIsNotNone(reader.get("m2"))
        self.assertIsNone(reader.get("m1"))

    def test_load_missing_file_returns_zero_without_raising(self) -> None:
        memory = _memory()
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(memory.load(os.path.join(self._tmp.name, "nope.jsonl")), 0)

    def test_loaded_items_are_searchable_by_content(self) -> None:
        source = _memory()
        source.add(_item("m1", "delta"))
        source.save(self.path)
        restored = _memory()
        restored.load(self.path)
        hits = restored.search("q", limit=1, now=_NOW)
        self.assertEqual([i.id for i in hits], ["m1"])


class BadLineTests(unittest.TestCase):
    """坏行跳过并记 WARNING（不抛）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-persist-")
        self.path = os.path.join(self._tmp.name, "vector.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, lines: list[str]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    def test_bad_lines_are_skipped(self) -> None:
        good = _item("m1", "alpha").to_dict(include_embedding=True)
        self._write(
            [
                json.dumps(good),
                "{ not json at all",
                "",  # 空行直接跳过（不占 warning）
                json.dumps({"content": "no id"}),
                json.dumps(["not", "a", "mapping"]),
                json.dumps({"id": "m3", "content": "beta", "created_at": "oops"}),
            ]
        )
        memory = _memory()
        with self.assertWarns(RuntimeWarning):
            loaded = memory.load(self.path)
        self.assertEqual(loaded, 1)
        self.assertEqual(len(memory), 1)
        self.assertIsNotNone(memory.get("m1"))

    def test_valid_lines_after_bad_lines_still_load(self) -> None:
        good_a = _item("m1", "alpha").to_dict(include_embedding=True)
        good_b = _item("m2", "beta").to_dict(include_embedding=True)
        self._write(["garbage", json.dumps(good_a), "```", json.dumps(good_b)])
        memory = _memory()
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(memory.load(self.path), 2)
        self.assertEqual(sorted(i.id for i in memory.all()), ["m1", "m2"])


class DimensionMismatchTests(unittest.TestCase):
    """embedding 维度与 `self.dim` 不符 -> `MemoryStoreError`（宁可失败也不半脏）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-persist-")
        self.path = os.path.join(self._tmp.name, "vector.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_with_mismatched_embedding_dim_raises(self) -> None:
        line = _item("m1", "alpha").to_dict(include_embedding=True)
        line["embedding"] = [0.0] * 4
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

        memory = VectorMemory(config=VectorConfig(dim=8))
        self.assertEqual(memory.dim, 8)
        with self.assertRaises(MemoryStoreError) as ctx:
            memory.load(self.path)
        self.assertIn("dim", str(ctx.exception))

    def test_load_with_matching_dim_succeeds(self) -> None:
        line = _item("m1", "alpha").to_dict(include_embedding=True)
        line["embedding"] = [0.0] * 8
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        memory = VectorMemory(config=VectorConfig(dim=8))
        self.assertEqual(memory.load(self.path), 1)


class ManagerPersistRestoreTests(unittest.IsolatedAsyncioTestCase):
    """§8.6 `MemoryManager.persist` / `restore`。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-persist-")
        self.path = os.path.join(self._tmp.name, "long_term.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_persist_then_restore_keeps_long_term_items(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig(persist_path=self.path))
        self.assertEqual(len(manager.long_term), 0)  # 首次启动：文件不存在 -> 空库
        await manager.aremember("记住我用 uv 管理依赖", importance=0.9, source="manual")
        await manager.aremember("我喜欢用 ruff 做 lint", importance=0.7)

        self.assertEqual(manager.persist(), 2)
        self.assertTrue(os.path.exists(self.path))

        restored = MemoryManager.from_config(MemoryConfig(persist_path=self.path))
        self.assertEqual(len(restored.long_term), 2)
        self.assertEqual(
            {item.content for item in restored.long_term.all()},
            {"记住我用 uv 管理依赖", "我喜欢用 ruff 做 lint"},
        )
        # 显式 restore 是幂等的加性操作（再读一遍会重复 add 同 id 条目，条数不变）。
        before = len(restored.long_term)
        self.assertEqual(restored.long_term.load(self.path), 2)
        self.assertEqual(len(restored.long_term), before)

    async def test_persist_without_path_raises_config_error(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig())
        with self.assertRaises(ConfigError):
            manager.persist()
        with self.assertRaises(ConfigError):
            manager.restore()

    async def test_persist_returns_zero_when_long_term_is_disabled(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig(long_term_enabled=False))
        self.assertIsNone(manager.long_term)
        self.assertEqual(manager.persist(self.path), 0)
        self.assertEqual(manager.restore(self.path), 0)

    async def test_restore_missing_file_returns_zero(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig())
        self.assertEqual(manager.restore(self.path), 0)
        self.assertEqual(len(manager.long_term), 0)

    async def test_explicit_persist_and_restore_paths_override_config(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig())
        await manager.aremember("remember this", importance=0.5)
        explicit = os.path.join(self._tmp.name, "explicit.jsonl")
        self.assertEqual(manager.persist(explicit), 1)

        other = MemoryManager.from_config(MemoryConfig())
        self.assertEqual(other.restore(explicit), 1)
        self.assertEqual(len(other.long_term), 1)

    async def test_from_config_swallows_restore_failures_with_a_warning(self) -> None:
        """启动期恢复失败不能把 Agent 拦在门外（降级必须留痕）。"""
        line = _item("m1", "alpha").to_dict(include_embedding=True)
        line["embedding"] = [0.0] * 4  # 默认 embedder 是 256 维 -> 读进来必然抛
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        with self.assertWarns(RuntimeWarning):
            manager = MemoryManager.from_config(MemoryConfig(persist_path=self.path))
        self.assertEqual(len(manager.long_term), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
