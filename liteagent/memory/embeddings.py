from __future__ import annotations

# 向量化抽象（§8.2）。冻结决策 D-04：默认的 HashingEmbedder 是**纯 stdlib、确定性、
# 无需网络**的，它是**词面相似**而不是语义相似 —— 生产环境应当注入 RemoteEmbedder
# 或用户自己的 CallableEmbedder。这一点必须在文档里诚实说明，不要声称"语义检索"。
#
# 本文件是 memory 包里唯一允许出现 numpy 的地方（§1.3），且只能写成 try/except 形态。
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"。

import collections
import functools
import hashlib
import math
import os
import random
import re
import zlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

from liteagent.config import DEFAULT_HASHING_EMBED_DIM, DEFAULT_LLM_TIMEOUT_S, to_jsonable
from liteagent.errors import (
    ConfigError,
    LLMAuthError,
    LLMResponseFormatError,
    LiteAgentError,
    MemoryStoreError,
)
from liteagent.llm.transport import (
    HTTPRequest,
    Transport,
    default_transport,
    map_http_error,
    wrap_transport_exception,
)
from liteagent.memory.base import CJK_RANGES

try:  # pragma: no cover - 环境相关
    import numpy as _np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _np = None
    NUMPY_AVAILABLE = False

# §2.7 的冻结定义式 `LowLevelEvent = Callable[[str, dict[str, Any]], None]`。
# SPEC-AMBIGUITY: §2.7 只给了定义式、没有写它的归属模块，而当前 config.py 里也没有它
# （§6.3/§7.4/§10.2 都只写 `on_event: LowLevelEvent | None`）。裁决：按**字面**在本文件
# 定义一次 —— 类型别名是结构等价的，将来 config.py 补上同名别名也不会冲突，
# 而"从 config import"在今天会直接 ImportError 打死整个模块。
LowLevelEvent = Callable[[str, dict[str, Any]], None]

# ---------------------------------------------------------------------------
# 分词与哈希（HashingEmbedder / NumpyHashingEmbedder / RandomProjectionEmbedder 共用）
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9_]+")

# CJK 连续片段的正则由 base.CJK_RANGES 生成 —— "什么算 CJK" 只有一处真值源，
# 保证 HeuristicTokenizer 与这里的 tokenizer 不会对同一段中文给出不一致的判定。
_CJK_RUN_RE = re.compile(
    "[" + "".join("\\u%04x-\\u%04x" % (lo, hi) for lo, hi in CJK_RANGES) + "]+"
)


def tokenize(text: str) -> list[str]:
    """冻结的分词（§8.2 步骤 1）：

    1. 转小写
    2. 英文/数字：正则 `[a-z0-9_]+` 提取，保留长度 >= 1 的 token
    3. CJK：对每个 CJK 连续片段取 **单字** 与 **相邻双字 bigram** 两种 token
    4. 丢弃纯空白与单字符 ASCII 标点

    第 4 条是第 2 条的推论：`[a-z0-9_]+` 天然不匹配空白与标点，所以不需要额外的过滤步骤。
    空串 -> `[]`（进而是零向量）。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = [m.group(0) for m in _WORD_RE.finditer(lowered)]
    for run in _CJK_RUN_RE.findall(lowered):
        # 单字：中文没有空格分词，逐字 hash 才有召回
        tokens.extend(run)
        # 相邻双字：补一点词序信息（"上海" != "海上"）
        if len(run) >= 2:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _hash_pairs(text: str, dim: int) -> list[tuple[int, float]]:
    """把文本映射成 `(向量下标, 带符号权重)` 列表（冻结算法）。

    - `h = hashlib.blake2b(token, digest_size=8)`：**禁止用内置 `hash()`** —— Python 的
      str hash 受 PYTHONHASHSEED 随机化影响，同一段文本在两个进程里会得到不同的向量，
      于是"持久化再读回来"就再也检索不到（这是必须写进文档的坑）。
    - `idx = int.from_bytes(h[:4], "big") % dim`；`sign` 取第 5 个字节的最低位；
    - `w = 1.0 + log1p(count)`：词频的对数加权，压住高频词又不至于让重复词失权。

    返回顺序 = token 首次出现顺序（Counter 保序），这样纯 Python 实现与 numpy 实现的
    累加顺序完全一致，两者可以做到逐位一致（§8.2 要求 1e-9 内一致）。
    """
    counts = collections.Counter(tokenize(text))
    pairs: list[tuple[int, float]] = []
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        pairs.append((idx, sign * (1.0 + math.log1p(count))))
    return pairs


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化。**零向量原样返回全 0，绝不返回 NaN**（空文本走的就是这条路）。"""
    norm_sq = 0.0
    for x in vec:
        norm_sq += x * x
    if norm_sq <= 0.0:
        return vec
    norm = math.sqrt(norm_sq)
    return [x / norm for x in vec]


@functools.lru_cache(maxsize=4096)
def _pseudo_vector(seed: int, crc: int, dim: int) -> tuple[float, ...]:
    """`random.Random(seed ^ crc32(token))` 生成的确定性伪随机向量（带缓存）。

    用 `random.Random` 实例而不是全局 `random.*`：全局 RNG 的调用顺序会被同进程的其它
    代码改变，测试就不可复现了（§5.5 的注入原则）。缓存按值做键（不持有 embedder），
    且 `lru_cache` 有上界，不会随语料无限增长。
    """
    rng = random.Random(seed ^ crc)
    return tuple(rng.uniform(-1.0, 1.0) for _ in range(dim))


def _check_dim(dim: int, owner: str) -> None:
    """维度必须是正整数：`x % dim` / 矩阵形状都靠它，dim=0 会变成 ZeroDivisionError。"""
    if not isinstance(dim, int) or dim <= 0:
        raise ConfigError(message=f"{owner}: dim must be a positive int, got {dim!r}")


# ---------------------------------------------------------------------------
# Embedder 家族
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """向量化契约。

    `embed` 返回 **L2 归一化**后的向量（归一化是契约的一部分，检索时可以直接用点积
    当余弦）。实现者如果返回未归一化向量，`VectorMemory` 的混合打分仍会走
    `cosine_similarity` 的除法兜底，但"点积=余弦"的性能假设就不成立了。
    """

    name: str = "abstract"
    dim: int = 0

    # §2.7 冻结：所有低层类的 on_event 一律是 LowLevelEvent | None（即
    # Callable[[str, dict[str, Any]], None]），只接收 (事件名, data) 二元组。
    # 这里做成**类属性**而不是构造参数，因为 §8.2 冻结的五个构造函数签名里都没有它
    # （§2.7 也只写 `Embedder` 而没有写 `Embedder.__init__`）。用户/子类直接赋值即可。
    on_event: LowLevelEvent | None = None

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """返回 L2 归一化后的向量（**归一化是契约的一部分**，检索时直接用点积当余弦）。"""
        ...

    def embed_one(self, text: str) -> list[float]:
        """单条便捷入口（内部就是 `embed([text])[0]`）。"""
        vectors = self.embed([text])
        if not vectors:
            raise MemoryStoreError(
                message=f"{type(self).__name__}.embed returned no vector for a single input",
                store=type(self).__name__,
            )
        return [float(x) for x in vectors[0]]

    def is_available(self) -> bool:
        """默认 True。需要外部凭据/服务的实现（RemoteEmbedder）应覆写它，
        好让调用方在"必然失败"之前就降级，而不是每次都撞一次网络错误。"""
        return True

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """低层事件出口（实现与 §6.3 的 `BaseLLMClient._emit` 一致）。

        `event_type` 必须是 §9.2 `EventType` 的 `.value`（memory 层不得 import agent 层）。
        当前**没有任何 embedder 发事件**：§2.7/红线 18 规定每条事件只有一个发射者，而
        memory 的三条事件都归 `MemoryManager`。保留此出口是为了自定义子类复用同一形态。
        """
        if self.on_event is not None:
            self.on_event(event_type, to_jsonable(dict(data)))


class HashingEmbedder(Embedder):
    """纯 stdlib 的 hashing trick + 特征哈希（冻结算法，测试会断言确定性）。

    1. 分词 tokenize(text) -> list[str]（冻结）：
       - 转小写
       - 英文/数字：正则 r"[a-z0-9_]+" 提取，保留长度 >= 1 的 token
       - CJK：对每个 CJK 连续片段取 **单字** 与 **相邻双字 bigram** 两种 token
       - 丢弃纯空白与单字符 ASCII 标点
    2. 每个 token t:
          h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
          idx = int.from_bytes(h[:4], "big") % dim
          sign = 1.0 if (h[4] & 1) == 0 else -1.0
          w = 1.0 + math.log1p(count_of_t)
          vec[idx] += sign * w
       **禁止使用内置 hash()**：Python 的 str hash 受 PYTHONHASHSEED 随机化影响，
       会导致同一进程外不可复现 —— 这是必须写进文档的坑。
    3. L2 归一化；零向量（空文本）返回全 0（不要 NaN）。

    **能力边界（必须如实说明）**：这是词面哈希，不是语义 embedding。"你好" 与
    "hi" 的相似度是 0。离线跑测试够用，生产要语义就把 `RemoteEmbedder` 注入 `VectorMemory`。
    """

    name = "hashing"

    def __init__(self, dim: int = DEFAULT_HASHING_EMBED_DIM) -> None:
        _check_dim(dim, "HashingEmbedder")
        self.dim = dim

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """冻结分词（静态方法，`HashingEmbedder.tokenize(s)` 与实例调用都可用）。"""
        return tokenize(text)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for idx, value in _hash_pairs(text, self.dim):
                vec[idx] += value
            out.append(_l2_normalize(vec))
        return out


if NUMPY_AVAILABLE:

    class NumpyHashingEmbedder(Embedder):
        """与 HashingEmbedder 数学等价但用 numpy 加速。仅当 NUMPY_AVAILABLE。
        **测试必须断言两者对同一输入的输出在 1e-9 内一致**（双实现一致性）。

        "数学等价"在实现上的保证：两者共用 `tokenize` / `_hash_pairs`（同一份 "
        (下标, 权重) 序列、同一个顺序），numpy 侧只是把累加换成 `np.add.at`
        —— 它是 unbuffered 的，按给定顺序逐元素累加，因此重复下标也按同一顺序求和。
        """

        name = "numpy_hashing"

        def __init__(self, dim: int = DEFAULT_HASHING_EMBED_DIM) -> None:
            _check_dim(dim, "NumpyHashingEmbedder")
            self.dim = dim

        @staticmethod
        def tokenize(text: str) -> list[str]:
            return tokenize(text)

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for text in texts:
                pairs = _hash_pairs(text, self.dim)
                if not pairs:
                    # 空文本 / 全标点 -> 零向量（与纯 Python 实现一致）
                    out.append([0.0] * self.dim)
                    continue
                idxs = _np.fromiter((p[0] for p in pairs), dtype=_np.intp, count=len(pairs))
                vals = _np.fromiter((p[1] for p in pairs), dtype=_np.float64, count=len(pairs))
                vec = _np.zeros(self.dim, dtype=_np.float64)
                _np.add.at(vec, idxs, vals)
                norm = float(_np.linalg.norm(vec))
                if norm > 0.0:
                    vec = vec / norm
                out.append([float(x) for x in vec])
            return out


class RandomProjectionEmbedder(Embedder):
    """测试用：由 seed 决定的可复现伪随机向量。
    每个 token 的向量由 random.Random(seed ^ crc32(token)) 生成后累加，再归一化。
    用途：断言"给定构造好的相似度顺序时，VectorMemory 的排序正确"。
    """

    name = "random_projection"

    def __init__(self, dim: int = 32, *, seed: int = 0) -> None:
        _check_dim(dim, "RandomProjectionEmbedder")
        self.dim = dim
        self.seed = seed

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            # 按出现顺序累加（重复 token 累加多次 = 词频即权重），顺序固定 -> 可复现
            for token in tokenize(text):
                pseudo = _pseudo_vector(self.seed, zlib.crc32(token.encode("utf-8")), self.dim)
                for i, x in enumerate(pseudo):
                    vec[i] += x
            out.append(_l2_normalize(vec))
        return out


class CallableEmbedder(Embedder):
    """把外部函数（真实模型 / 测试查找表）适配成 `Embedder`。

    注入的 fn 返回的向量会**照契约归一化**：即使 fn 给的是未归一化向量，
    检索侧的"点积 = 余弦"依然成立。零向量保持全 0。
    """

    def __init__(
        self,
        fn: Callable[[Sequence[str]], list[list[float]]],
        *,
        dim: int,
        name: str = "callable",
    ) -> None:
        _check_dim(dim, "CallableEmbedder")
        self._fn = fn
        self.dim = dim
        self.name = name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raw = self._fn(texts)
        # 形状校验：注入函数的错形状必须在**这一层**炸出来，否则会变成矩阵里
        # 一列长度不齐的向量，等到检索时才以更难查的形态出错。
        rows: list[list[float]] = []
        for i, row in enumerate(raw):
            vec = [float(x) for x in row]
            if len(vec) != self.dim:
                raise MemoryStoreError(
                    message=(
                        f"CallableEmbedder: vector {i} has dim {len(vec)}, expected {self.dim}"
                    ),
                    store=self.name,
                )
            rows.append(vec)
        if len(rows) != len(texts):
            raise MemoryStoreError(
                message=(
                    f"CallableEmbedder: fn returned {len(rows)} vectors for {len(texts)} texts"
                ),
                store=self.name,
            )
        return [_l2_normalize(vec) for vec in rows]


class RemoteEmbedder(Embedder):
    """用 Transport 调 OpenAI 兼容 /embeddings 端点（无需 openai SDK）。
    需要 api_key；无网络/无 key 时 embed 抛 LLMError，由 VectorMemory 决定是否降级。
    """

    name = "remote"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dim: int = 1536,
        transport: Transport | None = None,
        timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    ) -> None:
        _check_dim(dim, "RemoteEmbedder")
        self.model = model
        # 凭据解析优先级与 §6.3 的 LLMConfig.resolve_api_key 保持一致，避免用户
        # 在 LLM 那边配好了 key、embedder 这边却"没读到"。
        self.api_key = (
            api_key
            or os.environ.get("LITEAGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved = base_url or os.environ.get("LITEAGENT_BASE_URL") or self.default_base_url
        self.base_url = resolved.rstrip("/")
        self.dim = dim
        self.transport = transport if transport is not None else default_transport()
        self.timeout_s = timeout_s

    def is_available(self) -> bool:
        """没有 key 就不可能在离线/无凭据环境下成功 —— 如实返回 False。"""
        return bool(self.api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if not self.api_key:
            raise LLMAuthError(
                message=(
                    "RemoteEmbedder requires an api_key "
                    "(pass api_key= or set LITEAGENT_API_KEY / OPENAI_API_KEY)"
                )
            )
        request = HTTPRequest(
            method="POST",
            url=self.base_url + "/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json_body={
                "model": self.model,
                "input": items,
                "encoding_format": "float",
            },
            timeout_s=self.timeout_s,
        )
        try:
            response = self.transport.send(request)
        except LiteAgentError:
            # 已经是框架异常（LLMTimeoutError / LLMConnectionError ...）：原样上抛，
            # 二次包装会让调用方的 except LLMTimeoutError 失配。
            raise
        except Exception as exc:
            raise wrap_transport_exception(exc, timeout_s=self.timeout_s) from exc
        if not response.ok:
            # map_http_error 返回（不抛）异常对象：401 -> LLMAuthError，429 -> rate limit ...
            raise map_http_error(
                response.status_code,
                response.text,
                url=request.url,
                headers=response.headers,
                timeout_s=self.timeout_s,
            )
        rows = self._parse_rows(response)
        if len(rows) != len(items):
            raise LLMResponseFormatError(
                message=(
                    f"RemoteEmbedder: got {len(rows)} embeddings for {len(items)} inputs"
                ),
                body=response.text[:2000],
            )
        out: list[list[float]] = []
        for vec in rows:
            if len(vec) != self.dim:
                # 维度不符是"存储层"意义上的错误（§3.3），不是 JSON 结构问题
                raise MemoryStoreError(
                    message=(
                        f"RemoteEmbedder: model {self.model!r} returned dim {len(vec)}, "
                        f"expected {self.dim}; pass dim=... or fix the model"
                    ),
                    store=self.name,
                )
            # 远端向量通常是归一化的，但不保证；按契约再归一化一次（幂等）。
            out.append(_l2_normalize(vec))
        return out

    @staticmethod
    def _parse_rows(response: Any) -> list[list[float]]:
        """解析 OpenAI 兼容响应 `{"data": [{"embedding": [...], "index": 0}, ...]}`。

        `response.json()` 自己会在解析失败时抛 LLMResponseFormatError（§6.2），
        这里只需要处理"JSON 合法但形状不对"的情况。
        """
        payload = response.json()
        if not isinstance(payload, Mapping) or "data" not in payload:
            raise LLMResponseFormatError(
                message="RemoteEmbedder: response has no 'data' field",
                body=response.text[:2000],
            )
        entries = payload["data"]
        if not isinstance(entries, list):
            raise LLMResponseFormatError(
                message="RemoteEmbedder: response 'data' is not a list",
                body=response.text[:2000],
            )
        rows: list[list[float]] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or not isinstance(entry.get("embedding"), list):
                raise LLMResponseFormatError(
                    message=f"RemoteEmbedder: response data[{i}] has no 'embedding' list",
                    body=response.text[:2000],
                )
            try:
                rows.append([float(x) for x in entry["embedding"]])
            except (TypeError, ValueError) as exc:
                raise LLMResponseFormatError(
                    message=f"RemoteEmbedder: response data[{i}] is not numeric: {exc}",
                    body=response.text[:2000],
                ) from exc
        # 服务端不保证按 index 排序（规范只说 data 是数组）；有合法 index 时按它排序，
        # 否则保持原序（比"猜"更安全）。
        indexes = [e.get("index") for e in entries]
        if all(isinstance(idx, int) for idx in indexes):
            order = sorted(range(len(rows)), key=lambda i: (indexes[i], i))
            rows = [rows[i] for i in order]
        return rows


def default_embedder(dim: int = DEFAULT_HASHING_EMBED_DIM) -> Embedder:
    """永远返回 **HashingEmbedder**（不依赖 numpy，保证跨实现者行为一致）。
    想用 numpy 加速需显式传 NumpyHashingEmbedder()。
    """
    return HashingEmbedder(dim=dim)


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------


def _cosine_impl(a: Sequence[float], b: Sequence[float]) -> float:
    """核心实现。numpy 可用时用 numpy，否则纯 Python（两者结果在浮点误差内一致）。"""
    if NUMPY_AVAILABLE:
        va = _np.asarray(a, dtype=_np.float64)
        vb = _np.asarray(b, dtype=_np.float64)
        na = float(_np.linalg.norm(va))
        nb = float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(va, vb) / (na * nb))
    dot = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        sum_a += fx * fx
        sum_b += fy * fy
    if sum_a == 0.0 or sum_b == 0.0:
        return 0.0
    return dot / math.sqrt(sum_a * sum_b)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """两向量已归一化时 = 点积；否则除以模长乘积。零向量 -> 0.0（不抛异常）。
    维度不等 -> MemoryStoreError。
    """
    if len(a) != len(b):
        raise MemoryStoreError(
            message=f"cosine_similarity: dimension mismatch ({len(a)} != {len(b)})",
            store="cosine_similarity",
        )
    return _cosine_impl(a, b)


def cosine_similarity_matrix(
    query: Sequence[float], matrix: Sequence[Sequence[float]]
) -> list[float]:
    """逐行算余弦。**O(n) 线性扫描**，没有 ANN 索引（§13 的能力边界：不要声称"向量数据库"）。

    逐行调用 `cosine_similarity`（而不是一次性 `matrix @ query`），是为了让本函数与
    单条 API 的结果**逐位一致** —— 两者被混用在断言里时，不会出现 1e-16 级别的差异。
    """
    return [cosine_similarity(query, row) for row in matrix]
