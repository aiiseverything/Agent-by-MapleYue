from __future__ import annotations

"""tests/test_memory_embeddings.py —— §8.2 `memory/embeddings.py` 的单元测试。

§12 第 5117 行要求的覆盖点：
  * `HashingEmbedder` 确定性（跨实例一致、**用 hashlib 手算 idx 做数学断言**）
  * 归一化
  * 相似文本相似度 > 不相似
  * 空文本零向量
  * `NumpyHashingEmbedder` 一致（`NUMPY_AVAILABLE` 时）
  * `RandomProjectionEmbedder` 可复现

"手算 idx" 的做法：下面的 `_EXPECTED_*` 常量是用
`hashlib.blake2b(token.encode("utf-8"), digest_size=8)` 离线算出来的
（idx = 前 4 字节大端整数 % dim，sign = 第 5 字节最低位），
它们同时钉住了三件事：用 blake2b 而不是内置 `hash()`、大端序、以及 `% dim` 的取模方式。
"""

import math
import unittest

from liteagent.config import DEFAULT_HASHING_EMBED_DIM
from liteagent.errors import ConfigError, LLMAuthError, MemoryStoreError
from liteagent.memory.embeddings import (
    NUMPY_AVAILABLE,
    CallableEmbedder,
    HashingEmbedder,
    RemoteEmbedder,
    RandomProjectionEmbedder,
    cosine_similarity,
    default_embedder,
    tokenize,
)

if NUMPY_AVAILABLE:  # pragma: no branch
    from liteagent.memory.embeddings import NumpyHashingEmbedder

#: 离线手算出来的 (token, dim) -> (idx, sign)（blake2b-8 前 4 字节大端 % dim，第 5 字节最低位）。
_EXPECTED_INDEX = {
    (8, "hello"): (0, -1.0),
    (8, "world"): (7, -1.0),
    (8, "你"): (5, -1.0),
    (16, "hello"): (8, -1.0),
    (16, "world"): (15, -1.0),
    (256, "你好"): (118, 1.0),
}

#: 每个 token 的权重是 `1.0 + log1p(count)`，单次出现即 `1 + ln(2)`。
_W1 = 1.0 + math.log1p(1)


class TokenizeTests(unittest.TestCase):
    """§8.2 步骤 1 的冻结分词。"""

    def test_ascii_words_lowered_and_punctuation_dropped(self) -> None:
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])
        self.assertEqual(tokenize("a_b c1"), ["a_b", "c1"])

    def test_cjk_emits_unigrams_and_bigrams(self) -> None:
        self.assertEqual(tokenize("你好世界"), ["你", "好", "世", "界", "你好", "好世", "世界"])
        # 单字 CJK 片段没有 bigram。
        self.assertEqual(tokenize("你"), ["你"])

    def test_empty_and_symbol_only_return_no_tokens(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])
        self.assertEqual(tokenize("!!!...!!!"), [])


class HashingEmbedderDeterminismTests(unittest.TestCase):
    """§8.2 的冻结算法：确定性 + 数学可复算。"""

    def test_hand_computed_vector_for_hello_world(self) -> None:
        """dim=8 时 "hello world" 的向量完全可手算（两个 token 都落在 idx 0/7 且符号为负）。"""
        vec = HashingEmbedder(dim=8).embed(["hello world"])[0]
        idx_h, sign_h = _EXPECTED_INDEX[(8, "hello")]
        idx_w, sign_w = _EXPECTED_INDEX[(8, "world")]
        raw = [0.0] * 8
        raw[idx_h] += sign_h * _W1
        raw[idx_w] += sign_w * _W1
        norm = math.sqrt(sum(x * x for x in raw))
        expected = [x / norm for x in raw]
        self.assertEqual(len(vec), 8)
        for got, want in zip(vec, expected):
            self.assertAlmostEqual(got, want, places=12)
        # 最直观的一条：两个 token 分别落在 0 与 7，其余位置必须是 0。
        self.assertAlmostEqual(vec[0], -math.sqrt(0.5), places=12)
        self.assertAlmostEqual(vec[7], -math.sqrt(0.5), places=12)
        for index in (1, 2, 3, 4, 5, 6):
            self.assertEqual(vec[index], 0.0)

    def test_repeated_token_weight_is_log_counted(self) -> None:
        """`w = 1 + log1p(count)`：重复 token 不是简单线性叠加（手算 dim=16 的整条向量）。"""
        vec = HashingEmbedder(dim=16).embed(["hello hello world"])[0]
        idx_h, sign_h = _EXPECTED_INDEX[(16, "hello")]
        idx_w, sign_w = _EXPECTED_INDEX[(16, "world")]
        self.assertNotEqual(idx_h, idx_w)  # 手算前提：两个 token 不撞桶
        raw = [0.0] * 16
        raw[idx_h] += sign_h * (1.0 + math.log1p(2))
        raw[idx_w] += sign_w * _W1
        norm = math.sqrt(sum(x * x for x in raw))
        for got, want in zip(vec, (x / norm for x in raw)):
            self.assertAlmostEqual(got, want, places=12)
        # 若实现误用 `count` 而不是 `1 + log1p(count)`，重复词会压过 world 从而改变比例。
        self.assertGreater(abs(vec[idx_h]), abs(vec[idx_w]))

    def test_same_input_same_output_across_instances(self) -> None:
        a = HashingEmbedder(dim=64).embed(["你好 world", "hello"])
        b = HashingEmbedder(dim=64).embed(["你好 world", "hello"])
        self.assertEqual(a, b)

    def test_embed_one_matches_embed(self) -> None:
        embedder = HashingEmbedder(dim=32)
        self.assertEqual(embedder.embed_one("你好 world"), embedder.embed(["你好 world"])[0])

    def test_vectors_are_l2_normalized(self) -> None:
        for text in ("hello world", "你好世界", "the quick brown fox"):
            vec = HashingEmbedder(dim=64).embed([text])[0]
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(norm, 1.0, places=12, msg=text)

    def test_empty_text_is_zero_vector_not_nan(self) -> None:
        vec = HashingEmbedder(dim=16).embed([""])[0]
        self.assertEqual(vec, [0.0] * 16)
        for value in vec:
            self.assertFalse(math.isnan(value))
        # 纯标点也一样（分词后没有 token）。
        self.assertEqual(HashingEmbedder(dim=16).embed(["!!! ..."])[0], [0.0] * 16)

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        embedder = HashingEmbedder(dim=256)
        base, close, far = embedder.embed(
            [
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumps over the lazy cat",
                "quantum chromodynamics lattice gauge theory",
            ]
        )
        self.assertGreater(cosine_similarity(base, close), cosine_similarity(base, far))
        self.assertGreater(cosine_similarity(base, close), 0.5)

    def test_invalid_dim_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            HashingEmbedder(dim=0)
        with self.assertRaises(ConfigError):
            HashingEmbedder(dim=-4)


class NumpyHashingEmbedderConsistencyTests(unittest.TestCase):
    """§8.2：两个实现在 1e-9 内一致（只在 numpy 可用时跑）。"""

    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not available in this environment")
    def test_numpy_matches_pure_python_within_1e_9(self) -> None:
        texts = ["hello world", "你好世界 hello", "", "!!!", "a b c d e " * 3]
        pure = HashingEmbedder(dim=128).embed(texts)
        fast = NumpyHashingEmbedder(dim=128).embed(texts)
        self.assertEqual(len(pure), len(fast))
        for left, right in zip(pure, fast):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertLess(abs(a - b), 1e-9)

    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not available in this environment")
    def test_numpy_empty_text_is_zero_vector(self) -> None:
        self.assertEqual(NumpyHashingEmbedder(dim=8).embed([""])[0], [0.0] * 8)


class RandomProjectionEmbedderTests(unittest.TestCase):
    """§8.2：由 seed 决定的可复现伪随机向量。"""

    def test_same_seed_reproducible_across_instances(self) -> None:
        a = RandomProjectionEmbedder(dim=8, seed=3).embed(["hello world"])[0]
        b = RandomProjectionEmbedder(dim=8, seed=3).embed(["hello world"])[0]
        self.assertEqual(a, b)

    def test_different_seed_gives_different_vector(self) -> None:
        a = RandomProjectionEmbedder(dim=8, seed=3).embed(["hello world"])[0]
        b = RandomProjectionEmbedder(dim=8, seed=4).embed(["hello world"])[0]
        self.assertNotEqual(a, b)

    def test_normalized_and_zero_for_empty(self) -> None:
        vec = RandomProjectionEmbedder(dim=8, seed=0).embed(["hello world"])[0]
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vec)), 1.0, places=12)
        self.assertEqual(RandomProjectionEmbedder(dim=8, seed=0).embed([""])[0], [0.0] * 8)


class OtherEmbedderTests(unittest.TestCase):
    """`CallableEmbedder` / `RemoteEmbedder` / `default_embedder` 的契约边角。"""

    def test_callable_embedder_normalizes_and_validates_shape(self) -> None:
        embedder = CallableEmbedder(lambda texts: [[3.0, 4.0] for _ in texts], dim=2)
        vec = embedder.embed_one("x")
        self.assertAlmostEqual(vec[0], 0.6, places=12)
        self.assertAlmostEqual(vec[1], 0.8, places=12)

    def test_callable_embedder_rejects_wrong_dim(self) -> None:
        embedder = CallableEmbedder(lambda texts: [[1.0, 2.0, 3.0]], dim=2)
        with self.assertRaises(MemoryStoreError):
            embedder.embed(["x"])

    def test_callable_embedder_rejects_wrong_row_count(self) -> None:
        embedder = CallableEmbedder(lambda texts: [[1.0, 2.0]], dim=2)
        with self.assertRaises(MemoryStoreError):
            embedder.embed(["x", "y"])

    def test_default_embedder_is_hashing_regardless_of_numpy(self) -> None:
        embedder = default_embedder()
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(embedder.dim, DEFAULT_HASHING_EMBED_DIM)
        self.assertEqual(default_embedder(16).dim, 16)

    def test_remote_embedder_without_key_raises_auth_error(self) -> None:
        embedder = RemoteEmbedder(model="text-embedding-3-small", api_key=None)
        # 环境里没有凭证时 is_available() 必须如实返回 False（不假装可用）。
        self.assertFalse(embedder.is_available() or bool(embedder.api_key))
        if not embedder.api_key:
            with self.assertRaises(LLMAuthError):
                embedder.embed(["x"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
