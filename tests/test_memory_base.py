from __future__ import annotations

"""tests/test_memory_base.py —— §8.1 `memory/base.py` 的单元测试。

§12 第 5116 行要求的覆盖点：
  * `MemoryItem` 序列化（`include_embedding` 两种）
  * `from_dict` 缺 embedding 不重算
  * `HeuristicTokenizer`（中英混排、空串 -> 0）
  * `cosine_similarity`（零向量、维度不符抛 `MemoryStoreError`）
"""

import math
import unittest

from liteagent.config import DEFAULT_CJK_CHAR_COST, DEFAULT_TOKEN_CHAR_RATIO
from liteagent.errors import MemoryStoreError, SerializationError
from liteagent.memory.base import (
    CallableTokenizer,
    HeuristicTokenizer,
    MemoryConfig,
    MemoryItem,
    Tokenizer,
    get_default_tokenizer,
    is_cjk_char,
)
from liteagent.memory.embeddings import cosine_similarity, cosine_similarity_matrix


class MemoryItemSerializationTests(unittest.TestCase):
    """§8.1 `to_dict` / `from_dict` 的冻结语义（含 embedding 的两种模式）。"""

    def test_to_dict_default_omits_embedding_key(self) -> None:
        item = MemoryItem.create("hello", item_id="m1")
        data = item.to_dict()
        self.assertNotIn("embedding", data)
        # 其余字段一律全量输出（含 None）。
        self.assertEqual(
            set(data),
            {
                "id",
                "content",
                "role",
                "metadata",
                "created_at",
                "last_access_at",
                "importance",
                "access_count",
                "score",
                "score_breakdown",
                "source",
            },
        )
        self.assertIsNone(data["score"])
        self.assertIsNone(data["score_breakdown"])
        self.assertEqual(data["id"], "m1")
        self.assertEqual(data["content"], "hello")

    def test_to_dict_include_embedding_true_emits_list(self) -> None:
        item = MemoryItem.create("hello", item_id="m1")
        item.embedding = [0.1, 0.2, 0.3]
        data = item.to_dict(include_embedding=True)
        self.assertIn("embedding", data)
        self.assertEqual(data["embedding"], [0.1, 0.2, 0.3])
        # 显式 None 也要输出该键（§2.2 的"字段全量输出"对 opt-in 字段同样适用）。
        item.embedding = None
        self.assertIsNone(item.to_dict(include_embedding=True)["embedding"])

    def test_to_dict_returns_copies_of_mutable_fields(self) -> None:
        item = MemoryItem.create("hello", metadata={"a": 1}, item_id="m1")
        item.embedding = [1.0, 2.0]
        item.score_breakdown = {"sim": 0.5}
        data = item.to_dict(include_embedding=True)
        data["metadata"]["a"] = 999
        data["embedding"].append(3.0)
        data["score_breakdown"]["sim"] = 0.0
        self.assertEqual(item.metadata, {"a": 1})
        self.assertEqual(item.embedding, [1.0, 2.0])
        self.assertEqual(item.score_breakdown, {"sim": 0.5})

    def test_from_dict_roundtrip_with_embedding(self) -> None:
        item = MemoryItem.create(
            "你好 world",
            role="assistant",
            importance=0.75,
            metadata={"kind": "note"},
            source="agent:main",
            item_id="m7",
        )
        item.embedding = [0.25, -0.5]
        item.access_count = 3
        restored = MemoryItem.from_dict(item.to_dict(include_embedding=True))
        self.assertEqual(restored.to_dict(include_embedding=True), item.to_dict(include_embedding=True))

    def test_from_dict_missing_embedding_stays_none_and_is_not_recomputed(self) -> None:
        """§8.1：未提供 embedding 时置 None，**不重算**。

        判据：`from_dict` 里没有任何 embedder 可用（这是个纯数据类），所以"不重算"的
        可观测形式就是 `embedding is None` —— 若哪天有人塞了个默认 embedder 进去，
        这条断言会立刻失败。
        """
        payload = {"id": "m2", "content": "记住我用 uv", "role": "user"}
        item = MemoryItem.from_dict(payload)
        self.assertIsNone(item.embedding)
        # 键在但值是 null 也一样（save/load 往返的形态）。
        item2 = MemoryItem.from_dict({**payload, "embedding": None})
        self.assertIsNone(item2.embedding)

    def test_from_dict_missing_required_fields_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError):
            MemoryItem.from_dict({"content": "no id"})
        with self.assertRaises(SerializationError):
            MemoryItem.from_dict({"id": "x"})
        with self.assertRaises(SerializationError):
            MemoryItem.from_dict("not a mapping")  # type: ignore[arg-type]

    def test_from_dict_bad_field_type_raises_serialization_error(self) -> None:
        with self.assertRaises(SerializationError):
            MemoryItem.from_dict({"id": "x", "content": "c", "created_at": "not-a-number"})

    def test_create_default_id_prefix_and_age_days(self) -> None:
        item = MemoryItem.create("hi")
        self.assertTrue(item.id.startswith("mem_"), item.id)
        item.created_at = 1_000_000.0
        self.assertEqual(item.age_days(now=1_000_000.0), 0.0)
        self.assertEqual(item.age_days(now=1_000_000.0 + 86400.0 * 3), 3.0)

    def test_memory_config_defaults_and_from_dict_ignores_unknown_keys(self) -> None:
        cfg = MemoryConfig()
        self.assertTrue(cfg.long_term_enabled)
        self.assertEqual(cfg.write_policy, "selective")
        parsed = MemoryConfig.from_dict({"write_policy": "turn", "unknown_key": 1})
        self.assertEqual(parsed.write_policy, "turn")
        self.assertEqual(parsed.buffer_max_tokens, cfg.buffer_max_tokens)


class HeuristicTokenizerTests(unittest.TestCase):
    """§8.1 的混合估算（D-06）：CJK 每字 1 token，其余 4 字符 1 token。"""

    def setUp(self) -> None:
        self.tok = HeuristicTokenizer()

    def test_empty_string_is_zero(self) -> None:
        self.assertEqual(self.tok.estimate(""), 0)
        self.assertEqual(self.tok.estimate_many([]), 0)

    def test_pure_ascii_uses_char_ratio(self) -> None:
        # ceil(4/4) == 1；ceil(5/4) == 2；ceil(40/4) == 10
        self.assertEqual(self.tok.estimate("abcd"), 1)
        self.assertEqual(self.tok.estimate("abcde"), 2)
        self.assertEqual(self.tok.estimate("a" * 40), 10)

    def test_pure_cjk_counts_one_token_per_char(self) -> None:
        self.assertEqual(self.tok.estimate("你好"), 2)
        self.assertEqual(self.tok.estimate("你好世界"), 4)

    def test_mixed_cjk_and_ascii(self) -> None:
        # "你好hello" -> n_cjk=2, n_other=5 -> ceil(2*1.0 + 5/4.0) == ceil(3.25) == 4
        got = self.tok.estimate("你好hello")
        self.assertEqual(got, math.ceil(2 * DEFAULT_CJK_CHAR_COST + 5 / DEFAULT_TOKEN_CHAR_RATIO))
        self.assertEqual(got, 4)
        # 中英混排的估算必须严格大于"只看 ASCII"的同一段 ASCII 部分（中文不被当成 4 字符）。
        self.assertGreater(self.tok.estimate("你好hello"), self.tok.estimate("hello"))

    def test_cjk_punctuation_and_fullwidth_count_as_cjk(self) -> None:
        # 0x3000-0x303F（CJK 标点）与 0xFF00-0xFFEF（全角）都在冻结区间里。
        self.assertTrue(is_cjk_char("，"))
        self.assertTrue(is_cjk_char("。"))
        self.assertTrue(is_cjk_char("！"))
        self.assertFalse(is_cjk_char("a"))
        self.assertFalse(is_cjk_char("!"))
        self.assertEqual(self.tok.estimate("你好，"), 3)

    def test_estimate_many_sums(self) -> None:
        self.assertEqual(self.tok.estimate_many(["abcd", "你好"]), 1 + 2)
        self.assertEqual(self.tok.estimate_many(["", ""]), 0)

    def test_custom_parameters_are_honoured(self) -> None:
        tok = HeuristicTokenizer(char_ratio=2.0, cjk_char_cost=3.0)
        self.assertEqual(tok.estimate("abcd"), 2)
        self.assertEqual(tok.estimate("你"), 3)

    def test_non_empty_text_never_estimates_below_one(self) -> None:
        """§8.1：非空且估算结果 < 1 -> 返回 1（最小 1 token 的保守假设）。"""
        tok = HeuristicTokenizer(char_ratio=1000.0, cjk_char_cost=0.0)
        self.assertEqual(tok.estimate("abcd"), 1)
        self.assertEqual(tok.estimate("你"), 1)

    def test_degenerate_char_ratio_degrades_instead_of_dividing_by_zero(self) -> None:
        """`char_ratio <= 0` 是非法配置；实现退化成"每字符 1 token"而不是崩掉。"""
        tok = HeuristicTokenizer(char_ratio=0.0)
        self.assertEqual(tok.estimate("abcd"), 4)
        self.assertEqual(tok.estimate("你好hello"), 7)

    def test_lru_cache_returns_same_value_across_instances(self) -> None:
        a = HeuristicTokenizer()
        b = HeuristicTokenizer()
        self.assertEqual(a.estimate("你好hello world"), b.estimate("你好hello world"))

    def test_callable_tokenizer_and_default_factory(self) -> None:
        tok = CallableTokenizer(lambda text: len(text), name="len")
        self.assertEqual(tok.name, "len")
        self.assertEqual(tok.estimate("abcd"), 4)
        default = get_default_tokenizer()
        self.assertIsInstance(default, Tokenizer)
        # 进程内只有一份默认 tokenizer（functools.cache）。
        self.assertIs(default, get_default_tokenizer())


class CosineSimilarityTests(unittest.TestCase):
    """§8.2 的相似度契约：零向量 -> 0.0（不抛），维度不符 -> `MemoryStoreError`。"""

    def test_identical_and_orthogonal(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, places=12)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=12)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, places=12)

    def test_unnormalized_vectors_are_divided_by_norms(self) -> None:
        self.assertAlmostEqual(cosine_similarity([3.0, 0.0], [7.0, 0.0]), 1.0, places=12)

    def test_zero_vector_returns_zero_without_raising(self) -> None:
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0, 1.0], [0.0, 0.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_dimension_mismatch_raises_memory_store_error(self) -> None:
        with self.assertRaises(MemoryStoreError) as ctx:
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        self.assertIn("dimension mismatch", str(ctx.exception))

    def test_cosine_similarity_matrix_matches_elementwise(self) -> None:
        query = [1.0, 0.0]
        matrix = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        got = cosine_similarity_matrix(query, matrix)
        expected = [cosine_similarity(query, row) for row in matrix]
        self.assertEqual(got, expected)
        self.assertAlmostEqual(got[2], 0.5 / math.sqrt(0.5), places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
