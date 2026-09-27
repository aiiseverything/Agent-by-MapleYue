from __future__ import annotations

# 记忆层基元：条目（MemoryItem）、存储契约（MemoryStore）、配置（MemoryConfig）与
# token 估算家族（Tokenizer）。本模块是 memory 包的根依赖，只依赖 errors / config
# 两个下层模块（§1.1 的 L3 行）。
#
# 为什么 MemoryStore 是同步的（§8.1 / D-12）：纯内存操作是微秒级的，async 只会引入
# 无谓的事件循环耦合；并且工具大多跑在 worker 线程里，那些线程没有运行中的事件循环，
# 纯 async API 会在那里直接抛 RuntimeError: no running event loop。需要 LLM 的摘要
# 走 MemoryManager 的 async API。
#
# 线程模型（冻结小节，与 §10.2.1 同构）见 MemoryStore 的类 docstring。
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import functools
import math
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

from liteagent import config
from liteagent.config import (
    DEFAULT_BUFFER_MAX_MESSAGES,
    DEFAULT_BUFFER_MAX_TOKENS,
    DEFAULT_CJK_CHAR_COST,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_HASHING_EMBED_DIM,
    DEFAULT_MAX_MEMORY_ITEMS,
    DEFAULT_MAX_SUMMARY_CHARS,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_RECENCY_HALF_LIFE_D,
    DEFAULT_RESERVE_COMPLETION_TOKENS,
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_SUMMARY_MIN_EVICT,
    DEFAULT_SUMMARY_TRIGGER_RATIO,
    DEFAULT_TOKEN_CHAR_RATIO,
    DEFAULT_W_IMPORTANCE,
    DEFAULT_W_RECENCY,
    DEFAULT_W_SIM,
)
from liteagent.errors import SerializationError

# tiktoken 是**可选**依赖，且只能写成 try/except 形态（§1.3 的零依赖红线：
# 顶层裸 import 三方库会被 test_zero_dependency.py 抓出来）。它不可用时
# TiktokenTokenizer 这个类根本不会被定义，所以模块仍然可以 import。
try:  # pragma: no cover - 环境相关
    import tiktoken as _tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _tiktoken = None
    TIKTOKEN_AVAILABLE = False

# SPEC-AMBIGUITY: §1.3 把三方库的允许位置列举为 llm/transport.py、memory/embeddings.py、
# config.py、cli.py 四处，但 §8.1 又要求在 memory/base.py 里定义 TiktokenTokenizer 并提供
# 模块级常量 TIKTOKEN_AVAILABLE。裁决：服从 §8.1（tiktoken 只出现在 try/except 体内，
# test_zero_dependency 的 AST 判定明确放行该形态），并在本文件顶层声明 TIKTOKEN_AVAILABLE。

CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # 中日韩统一表意文字
    (0x3400, 0x4DBF),  # 扩展 A
    (0x3000, 0x303F),  # CJK 标点
    (0xFF00, 0xFFEF),  # 全角字符
    (0xAC00, 0xD7AF),  # 韩文音节
    (0x3040, 0x30FF),  # 平假名 / 片假名
)
"""冻结的 CJK 区间（§8.1）。逐条硬编码而不是用 `unicodedata.name()` 匹配名字 ——
后者依赖 Unicode 数据库版本，同一段文本在不同解释器上会得到不同的估算结果。
embeddings.py 的分词复用本常量，保证两处对"CJK 字符"的定义永远一致。"""


def is_cjk_char(ch: str) -> bool:
    """`ch` 的首个码点是否落在冻结的 CJK 区间内（多字符输入只看首字符）。"""
    if not ch:
        return False
    cp = ord(ch[0])
    for lo, hi in CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


@dataclass
class MemoryItem:
    """一条记忆（短期 buffer 与长期向量库共用同一个载体）。

    时间字段一律是 Unix 秒的 `float`（§2.2），`created_at` / `last_access_at` 都取自
    唯一时钟 `config.utc_now()`。注意 default_factory 写成 `lambda: config.utc_now()`
    而**不是** `default_factory=utc_now`：后者在模块导入时就把函数对象固定下来了，
    `tests.helpers.frozen_time` 的 `mock.patch("liteagent.config.utc_now")` 会失效（§12.1）。
    """

    id: str
    content: str
    role: str = "user"  # "user" | "assistant" | "system" | "tool"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: config.utc_now())
    last_access_at: float = field(default_factory=lambda: config.utc_now())
    importance: float = 0.5  # 0..1
    access_count: int = 0
    embedding: list[float] | None = None
    score: float | None = None  # 检索时填充（混合分）
    score_breakdown: dict[str, float] | None = None  # {"sim":..,"recency":..,"importance":..}
    source: str = ""  # 来源标记，如 "agent:main" / "manual"

    @classmethod
    def create(
        cls,
        content: str,
        *,
        role: str = "user",
        importance: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
        source: str = "",
        item_id: str | None = None,
    ) -> "MemoryItem":
        """构造入口。`item_id` 默认 `'mem_' + uuid4().hex[:12]`
        （存储层 ID 允许随机，测试断言请用内容而非 ID，见 §2.8）。
        """
        return cls(
            id=item_id if item_id is not None else "mem_" + uuid.uuid4().hex[:12],
            content=content,
            role=role,
            metadata=dict(metadata) if metadata is not None else {},
            importance=importance,
            source=source,
        )

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """[v2 变更] **本方法只有一个定义**（v1 在同一处写了两行互斥签名）。
        冻结签名与语义：
          - include_embedding=False（**默认**）-> 省略 "embedding" 键
          - include_embedding=True -> 输出 embedding（list[float]，可能很长）
        其余字段一律全量输出（含 None）。理由见 §2.2：embedding 会让 trace/CLI 膨胀。
        持久化往返（§8.5 的 save/load）与 to_dict 的精确比对测试用 True。
        """
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "last_access_at": self.last_access_at,
            "importance": self.importance,
            "access_count": self.access_count,
            "score": self.score,
            "score_breakdown": (
                dict(self.score_breakdown) if self.score_breakdown is not None else None
            ),
            "source": self.source,
        }
        if include_embedding:
            # 显式 None 也要输出该键：§2.2 的"字段全量输出"对 opt-in 字段同样适用，
            # 否则 save/load 往返的精确 dict 比对会因键缺失而失败。
            data["embedding"] = (
                [float(x) for x in self.embedding] if self.embedding is not None else None
            )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryItem":
        """未提供 embedding 时置 None，**不重算**。

        只有 `id` / `content` 缺失才抛 `SerializationError`；其余字段缺失时沿用
        dataclass 的默认值（含 `created_at` 的默认时钟）。类型转换失败同样抛
        `SerializationError`（§3.3：from_dict 收到缺字段/错类型）。
        """
        if not isinstance(data, Mapping):
            raise SerializationError(
                message="MemoryItem.from_dict expects a mapping", target="MemoryItem"
            )
        if data.get("id") is None or data.get("content") is None:
            missing = [k for k in ("id", "content") if data.get(k) is None]
            raise SerializationError(
                message=f"MemoryItem.from_dict: missing required field(s) {missing}",
                target="MemoryItem",
            )
        try:
            kwargs: dict[str, Any] = {
                "id": str(data["id"]),
                "content": str(data["content"]),
            }
            if data.get("role") is not None:
                kwargs["role"] = str(data["role"])
            if data.get("metadata") is not None:
                kwargs["metadata"] = dict(data["metadata"])
            if data.get("created_at") is not None:
                kwargs["created_at"] = float(data["created_at"])
            if data.get("last_access_at") is not None:
                kwargs["last_access_at"] = float(data["last_access_at"])
            if data.get("importance") is not None:
                kwargs["importance"] = float(data["importance"])
            if data.get("access_count") is not None:
                kwargs["access_count"] = int(data["access_count"])
            # embedding / score / score_breakdown 是"允许为 None"的字段：
            # 键存在就照抄（含 None），键不存在就沿用默认 None。
            if "embedding" in data:
                raw = data["embedding"]
                kwargs["embedding"] = None if raw is None else [float(x) for x in raw]
            if "score" in data:
                raw_score = data["score"]
                kwargs["score"] = None if raw_score is None else float(raw_score)
            if "score_breakdown" in data:
                raw_bd = data["score_breakdown"]
                kwargs["score_breakdown"] = (
                    None if raw_bd is None else {str(k): float(v) for k, v in raw_bd.items()}
                )
            if data.get("source") is not None:
                kwargs["source"] = str(data["source"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise SerializationError(
                message=f"MemoryItem.from_dict: bad field type: {exc}", target="MemoryItem"
            ) from exc
        return cls(**kwargs)

    def text(self) -> str:
        """用于 token 估算的载荷文本（即 `content`）。"""
        return self.content

    def age_days(self, *, now: float | None = None) -> float:
        """`(now - created_at) / 86400`（§8.5.1 的冻结公式）。`now=None` 时取唯一时钟。"""
        current = config.utc_now() if now is None else now
        return (current - self.created_at) / 86400.0


@dataclass
class MemoryConfig:
    """记忆层的总配置（字段与默认值冻结于 §5.3），由 `MemoryManager.from_config`
    翻译成各层的 `BufferConfig` / `SummaryConfig` / `VectorConfig`。

    全部字段都有默认值，保证 `MemoryConfig()` 可无参构造（§2.3）。
    """

    buffer_max_tokens: int = DEFAULT_BUFFER_MAX_TOKENS
    buffer_max_messages: int = DEFAULT_BUFFER_MAX_MESSAGES
    buffer_keep_last_n: int = 2
    summary_enabled: bool = True
    summary_trigger_ratio: float = DEFAULT_SUMMARY_TRIGGER_RATIO
    summary_min_evict: int = DEFAULT_SUMMARY_MIN_EVICT
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    long_term_enabled: bool = True
    embedder_dim: int = DEFAULT_HASHING_EMBED_DIM
    write_policy: str = "selective"  # "selective" | "turn" | "manual"
    auto_write_min_chars: int = 40
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    max_items: int = DEFAULT_MAX_MEMORY_ITEMS
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    retrieve_min_score: float = 0.0
    w_sim: float = DEFAULT_W_SIM
    w_recency: float = DEFAULT_W_RECENCY
    w_importance: float = DEFAULT_W_IMPORTANCE
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_D
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    token_char_ratio: float = DEFAULT_TOKEN_CHAR_RATIO
    cjk_char_cost: float = DEFAULT_CJK_CHAR_COST
    # [v2 新增] 上下文窗口反推预算（D-15）
    context_window_tokens: int | None = None  # 非 None 时反推 buffer_max_tokens
    reserve_completion_tokens: int = DEFAULT_RESERVE_COMPLETION_TOKENS
    tools_schema_tokens_reserve: int = 0
    persist_path: str | None = None  # 非 None 时 MemoryManager 启动即 restore

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出（含 None），可直接过 `config.to_jsonable`（§2.2）。"""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryConfig":
        """宽松解析：只取已知字段名，未知键忽略（向前兼容 —— 上层配置字典里可能有
        属于 `AppConfig` 别的段的键），缺失字段用默认值。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class MemoryStore(ABC):
    """同步内存存储契约。**同步**的理由：纯内存操作是微秒级，async 只会引入无谓的
    事件循环耦合；并且工具大多跑在 worker 线程里，那些线程没有运行中的事件循环，
    纯 async API 会在那里直接抛异常。需要 LLM 的摘要走 MemoryManager 的 async API。
    完整论证见 `DESIGN_DECISIONS.md` D-12。

    [v2 变更] **线程模型（冻结小节，与 §10.2.1 同构）**：
      1. 每个具体实现（BufferMemory / VectorMemory）各持有
         `self._lock = threading.RLock()`，在 **`__init__`** 创建
         （threading 原语不受 §0.4 的 R-LOOP 约束 —— 这正是选它的理由）。
      2. `add/add_many/get/search/all/delete/clear/upsert/window/drain_evicted`
         **全部**在锁内完成，且临界区内**没有 await、没有 I/O**。
      3. `search` 的副作用（`access_count` / `last_access_at` / `score`）**必须在同一把锁内**更新。
      4. 向量矩阵（numpy 数组）的读也走同一把锁；**不允许裸露的 `_matrix`**。
      背景：`TeamConfig.share_memory=True` 时多个子 Agent 共享一个 MemoryManager；
      Hierarchical 的 worker 在各自线程 + 各自新建的 loop 里跑，主 loop 同时可能在检索 ——
      无锁下会得到 `RuntimeError: list changed size during iteration` 或读到一半的淘汰状态。

    本 ABC 提供一个默认的 `self._lock`（同样是在 `__init__` 里创建的 `threading.RLock`），
    子类可以直接复用；子类若在自己的 `__init__` 里重建一把锁同样合规 —— 契约只要求
    "最终 `self._lock` 是一个可重入的 threading 锁，且所有公开方法都在它里面完成"。
    """

    name: str = "abstract"

    def __init__(self) -> None:
        # §8.1 规则 1：threading 原语与事件循环无关，这是它们被大量使用的理由（D-12）。
        # 类型标注写 Any 而非 threading.RLock：后者是工厂函数而不是类，标注它会让类型
        # 检查器误判（`threading.RLock` 的实例类型是 `_thread.RLock`）。
        self._lock: Any = threading.RLock()

    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...

    def add_many(self, items: Sequence[MemoryItem]) -> None:
        """默认实现逐条 `add`。

        拿得到 `self._lock` 时整批在同一把锁内完成（§8.1 规则 2 要求 `add_many` 也在锁内）
        —— 锁是 `RLock`，所以内层 `self.add` 的重入是安全的。子类若在自己的 `__init__`
        里重建锁、或压根没走 `MemoryStore.__init__`，这里退化成"逐条 add"：每一条仍然由
        子类的 `add` 自己保证原子性，只是整批不再是一个原子步骤（对 `load()` 这类调用足够）。
        """
        lock = getattr(self, "_lock", None)
        if lock is None:
            for item in items:
                self.add(item)
            return
        with lock:
            for item in items:
                self.add(item)

    @abstractmethod
    def get(self, item_id: str) -> MemoryItem | None: ...

    @abstractmethod
    def search(
        self, query: str, *, limit: int = DEFAULT_RETRIEVE_LIMIT, **kwargs: Any
    ) -> list[MemoryItem]: ...

    @abstractmethod
    def all(self) -> list[MemoryItem]:
        """按 created_at 升序。**冻结**：只复制外层 list，**元素是同一批 MemoryItem 实例**
        （共享可变），测试可以用 `is` 断言身份。文档写明这一点。
        """
        ...

    @abstractmethod
    def delete(self, item_id: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __len__(self) -> int:
        """默认按条数实现。注意：`all()` 会复制外层 list，不要在大 store 上高频调用。"""
        return len(self.all())


# ---------------------------------------------------------------------------
# Tokenizer 家族（冻结）
# ---------------------------------------------------------------------------


class Tokenizer(ABC):
    """token 估算契约。

    估算结果**必须**同时用于摘要触发判定与向量记忆的截断（§8.1 D-06：唯一估算函数），
    因此任何实现都要满足"空串 -> 0"这条最小约定，否则窗口预算会算歪。
    """

    name: str = "abstract"

    @abstractmethod
    def estimate(self, text: str) -> int: ...

    def estimate_many(self, texts: Sequence[str]) -> int:
        """逐条求和。空序列 -> 0。"""
        return sum(self.estimate(text) for text in texts)


class HeuristicTokenizer(Tokenizer):
    """默认实现（零依赖）。est = ceil(n_cjk * cjk_char_cost + n_other / char_ratio)
    - n_cjk：码点落在 CJK 区间的字符数。冻结的区间判定（逐条实现，不要用 unicodedata 名字匹配，
      那在不同 Unicode 版本下不稳定）：
        0x4E00-0x9FFF   中日韩统一表意
        0x3400-0x4DBF   扩展 A
        0x3000-0x303F   CJK 标点
        0xFF00-0xFFEF   全角字符
        0xAC00-0xD7AF   韩文音节
        0x3040-0x30FF   平假名/片假名
    - n_other：其余码点数（含空格与 ASCII 标点）
    - 参数默认 char_ratio=DEFAULT_TOKEN_CHAR_RATIO(4.0), cjk_char_cost=1.0
    - 结果用 functools.lru_cache(maxsize=4096) 缓存（按 (text, ratio, cost) 元组）
    - **空文本 -> 0**；非空且估算结果 < 1 -> 返回 1（最小 1 token 的保守假设）

    冻结决策 D-06：**混合估算** —— CJK 按 1 字 1 token，其余按 4 字符 1 token。
    纯 `len/4` 对中文严重低估（会让窗口超预算），引入 tiktoken 会违反零依赖；
    代价是 ±20% 误差，靠预算留 20% 余量 + 允许注入 `CallableTokenizer` 缓解。
    """

    name = "heuristic"

    def __init__(
        self,
        char_ratio: float = DEFAULT_TOKEN_CHAR_RATIO,
        cjk_char_cost: float = DEFAULT_CJK_CHAR_COST,
    ) -> None:
        self.char_ratio = char_ratio
        self.cjk_char_cost = cjk_char_cost

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _estimate_cached(text: str, char_ratio: float, cjk_char_cost: float) -> int:
        """缓存的纯函数实现。

        缓存键刻意用 (text, ratio, cost) 而不是 `self`：`lru_cache` 挂在实例方法上会让
        缓存项强引用 `self`（实例无法被回收），而这里的 tokenizer 是无状态的，用值做键
        既正确又不会泄漏；也正因如此它必须是 `staticmethod`（`staticmethod` 自 3.10 起
        可以直接调用，`HeuristicTokenizer._estimate_cached(...)` 依然可用）。
        """
        if text == "":
            return 0
        n_cjk = 0
        for ch in text:
            if is_cjk_char(ch):
                n_cjk += 1
        n_other = len(text) - n_cjk
        if char_ratio > 0:
            est = math.ceil(n_cjk * cjk_char_cost + n_other / char_ratio)
        else:
            # char_ratio <= 0 是非法配置（会 ZeroDivisionError），这里退化成"每字符 1 token"
            # 的保守估算而不是崩溃 —— token 估算不应该成为压垮 agent 轮次的原因。
            est = math.ceil(n_cjk * cjk_char_cost) + n_other
        return est if est >= 1 else 1

    def estimate(self, text: str) -> int:
        return self._estimate_cached(text, self.char_ratio, self.cjk_char_cost)


class CallableTokenizer(Tokenizer):
    """把任意 `fn(text) -> int` 注入成本 tokenizer（如用户自带 BPE 计数函数）。"""

    def __init__(self, fn: Callable[[str], int], *, name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def estimate(self, text: str) -> int:
        # 不做 except：注入函数抛异常时应当直接暴露（吞掉会让"估算错"变成"窗口悄悄超预算"）。
        return int(self._fn(text))


if TIKTOKEN_AVAILABLE:

    class TiktokenTokenizer(Tokenizer):
        """仅当 tiktoken 可 import 时定义；用 cl100k_base 编码。
        encode 失败时回退到 HeuristicTokenizer（并记 warning）。
        """

        name = "tiktoken"

        def __init__(self, encoding: str = "cl100k_base") -> None:
            self.encoding = encoding
            self._enc = _tiktoken.get_encoding(encoding)
            self._fallback = HeuristicTokenizer()

        def estimate(self, text: str) -> int:
            if text == "":
                return 0
            try:
                return max(1, len(self._enc.encode(text)))
            except Exception as exc:  # tiktoken 对畸形码点/BPE 边界会抛各种异常
                # 降级必须留痕（§13 红线 12）：warning 只报一次（warnings 默认按位置去重），
                # 不会把日志刷爆。
                warnings.warn(
                    f"TiktokenTokenizer({self.encoding!r}) encode failed ({exc!r}); "
                    "falling back to HeuristicTokenizer for this text",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._fallback.estimate(text)


@functools.cache
def get_default_tokenizer() -> Tokenizer:
    """优先级：TIKTOKEN_AVAILABLE -> TiktokenTokenizer("cl100k_base") -> HeuristicTokenizer()
    结果缓存（functools.cache）。

    为什么缓存：BufferMemory / SummaryMemory / VectorMemory 会频繁取它，而 tokenizer
    是无状态的，共享一个实例既省对象也顺带复用 HeuristicTokenizer 的 lru_cache。
    因为缓存结果，测试若要模拟"tiktoken 可用"，得先 `get_default_tokenizer.cache_clear()`
    —— 这也是进程内只有一份默认 tokenizer 的直接后果。
    """
    if TIKTOKEN_AVAILABLE:  # pragma: no cover - 本环境没有 tiktoken
        try:
            return TiktokenTokenizer("cl100k_base")
        except Exception as exc:  # 编码表下载失败 / 版本不兼容等
            # 降级必须留痕（§13 红线 12）：不能静默退回启发式估算。
            warnings.warn(
                f"TiktokenTokenizer unavailable ({exc!r}); falling back to HeuristicTokenizer",
                RuntimeWarning,
                stacklevel=2,
            )
    return HeuristicTokenizer()
