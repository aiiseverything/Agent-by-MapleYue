from __future__ import annotations

# 长期向量记忆（§8.5）。三个"面试可讲"的设计点：
#
# 1) **写入是有策略的**：不是每轮对话都往长期库里灌，`should_auto_write` 用
#    「长度阈值 / 命中 marker 关键字」的启发式 + **role 硬约束**决定一条消息值不值得
#    成为"用户事实"。
# 2) **检索是混合打分**：纯相似度排序会让"三天前说过的偏好"输给"刚提到的一个无关词"，
#    所以 score = w_sim*sim + w_recency*recency + w_importance*importance。
# 3) **结果是去冗的**：MMR 在"相关性"与"多样性"之间权衡，避免 5 条几乎一样的记忆塞满
#    prompt —— 那等于只检索到 1 条，还白白花掉 4 条的上下文预算。
#
# 能力边界（必须如实说明，别吹成"向量数据库"）：
#   - 检索是 **O(n) 线性扫描**，没有 ANN 索引（faiss 不可用），也没有 rerank 模型；
#   - 默认 `HashingEmbedder` 是**词面**哈希而不是语义 embedding（"你好" 与 "hi" 的相似度是 0），
#     生产环境应注入 `RemoteEmbedder`。默认实现的唯一好处是：零依赖、确定性、离线可跑。
#
# 线程模型（§8.1 冻结小节）：所有公开方法在 `self._lock = threading.RLock()` 内完成
# （锁由 `MemoryStore.__init__` 建好），临界区内**没有 await、没有 I/O** —— 因此
# `save()` 是"锁内做快照 + 锁外写文件"，`load()` 是"逐行解析后逐条 add（各自进锁）"。
# 为什么用 threading 而不是 asyncio.Lock：见 base.py 的 MemoryStore docstring（D-12）。
#
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# 别名成 _config：本模块的构造函数有名为 `config` 的形参（§8.5 冻结签名），
# 它会遮蔽同名的模块对象。取别名后 `_config.utc_now()` 在任何作用域里都指向模块，
# 而 tests.helpers.frozen_time 的 `patch("liteagent.config.utc_now")` 依旧生效
# （patch 的是模块对象上的属性，与本地别名无关）。
from liteagent import config as _config
from liteagent.config import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_HASHING_EMBED_DIM,
    DEFAULT_MAX_MEMORY_ITEMS,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_PERSIST_BATCH,
    DEFAULT_RECENCY_HALF_LIFE_D,
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_W_IMPORTANCE,
    DEFAULT_W_RECENCY,
    DEFAULT_W_SIM,
    to_jsonable,
)
from liteagent.errors import MemoryStoreError
from liteagent.llm.message import Message, Role
from liteagent.memory.base import (
    MemoryItem,
    MemoryStore,
    Tokenizer,
    get_default_tokenizer,
)
from liteagent.memory.embeddings import (
    NUMPY_AVAILABLE,
    Embedder,
    HashingEmbedder,
    LowLevelEvent,
    cosine_similarity,
    cosine_similarity_matrix,
)

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

AUTO_WRITE_MARKERS: tuple[str, ...] = (
    "记住",
    "请记",
    "记下",
    "我是",
    "我的名字",
    "我偏好",
    "我喜欢",
    "以后",
    "下次",
    "remember",
    "my name is",
    "i prefer",
    "i like",
    "note that",
    "keep in mind",
    "always use",
    "never use",
    "don't forget",
)
"""自动写入的**冻结关键字**（§8.5）：小写匹配，命中即自动写入。

为什么两种语言各留一半：中文没有空格分词，用户表达"记住这件事"的高频词是固定的几个
二字词；英文则用短语（`my name is` 而不是 `name`，否则 "the name of the file" 也会命中，
误报比漏报更贵 —— 误报会把噪声写进长期库并在之后每一轮被召回）。"""

# `<relevant_memories>` 块的边界标签（渲染格式是冻结的，见 §8.6）。
_MEMORIES_OPEN = "<relevant_memories>"
_MEMORIES_CLOSE = "</relevant_memories>"


@dataclass
class VectorConfig:
    """长期向量库的配置（字段名与默认值逐字冻结于 §8.5）。

    全部字段都有默认值，保证 `VectorConfig()` 可无参构造（§2.3）。
    注意 `dim` 是**可被 embedder 覆盖**的：构造 `VectorMemory(embedder=...)` 时
    注入的 embedder 的维度是权威（见 `VectorMemory.__init__`）。
    """

    dim: int = DEFAULT_HASHING_EMBED_DIM
    write_policy: str = "selective"  # "selective" | "turn" | "manual"
    auto_write_min_chars: int = 40
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    max_items: int = DEFAULT_MAX_MEMORY_ITEMS
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    retrieve_min_score: float = 0.0
    w_sim: float = DEFAULT_W_SIM
    w_recency: float = DEFAULT_W_RECENCY
    w_importance: float = DEFAULT_W_IMPORTANCE
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_D
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    use_numpy: bool = True  # NUMPY_AVAILABLE 时是否用矩阵加速


def _role_value(message: Message) -> str:
    """取消息的 role 字符串（小写）。

    为什么不直接 `message.role == Role.USER`：`should_auto_write` 的入参是协议对象，
    测试与第三方可能塞 `Message` 的鸭子类型替身、或一个 capitalize 过的 `"User"`
    （`Role.coerce` 明确要容忍这种输入）。统一走一次归一化，判定就只看一个字符串。
    """
    role = getattr(message, "role", None)
    if isinstance(role, Role):
        return role.value
    return str(role or "").strip().lower()


def _is_meaningful(content: str) -> bool:
    """内容是否"有点东西"：非空白，且至少含一个字母/数字/CJK 字符。

    `str.isalnum()` 对 CJK 返回 True，所以"纯符号"（`"。。。"`、`"!?!"`）会被判定为
    **无意义** —— 这类消息写进长期库只会污染后续每一轮的召回结果。
    """
    if not content.strip():
        return False
    return any(ch.isalnum() for ch in content)


def _matches_metadata_filter(
    item: MemoryItem, metadata_filter: Mapping[str, Any] | None
) -> bool:
    """`metadata_filter` 的判定：每个键值都必须与 `item.metadata` 逐值相等。

    SPEC-AMBIGUITY: §8.5 只给了 `metadata_filter: Mapping[str, Any]` 这一个名字，没说
    匹配范围（metadata 还是含 role/importance 等字段）。裁决：**只匹配 metadata** ——
    名字里写了 metadata，按字面实现可预测性最高；需要按角色过滤的调用方请把 role 写进
    metadata（`MemoryManager.aadd` 就是这么做的）。空字典/None 视为不过滤。
    """
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if item.metadata.get(key) != expected:
            return False
    return True


class VectorMemory(MemoryStore):
    """长期记忆：向量检索 + 近因/重要度混合打分 + MMR 去冗。
    [v2] **线程安全**：所有公开方法在 `self._lock = threading.RLock()` 内完成（§8.1）。
    """

    name = "vector"

    def __init__(
        self,
        embedder: Embedder | None = None,
        config: VectorConfig | None = None,
        *,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        """[v2 变更] **维度权威性冻结**：

        - `embedder is not None` -> `self.dim = embedder.dim` 并写回 `self.config.dim = self.dim`
          （注入的 embedder 的维度**是权威**）。没有这条，测试注入 dim=4 的
          `CallableEmbedder` 后每次 add 都会被 `MemoryStoreError` 打死。
        - `embedder is None` -> 用 `config.dim` 造 `HashingEmbedder(dim=config.dim)`。
        - 所有维度校验一律对 `self.dim`。

        为什么写回 config 而不是只在 self 上放一个维度：`VectorConfig` 是会被序列化/展示
        的配置对象，如果 `config.dim` 与实际向量的维度不一致，排查"检索返回空"这类问题时
        会先被一个假的 256 误导半天。写回之后 config 就是可信的单一真值。
        """
        # MemoryStore.__init__ 里建 self._lock（threading.RLock，与事件循环无关）。
        super().__init__()
        cfg = config if config is not None else VectorConfig()
        self.config: VectorConfig = cfg

        if embedder is not None:
            self.embedder: Embedder = embedder
            self.dim: int = int(embedder.dim)
        else:
            # HashingEmbedder 的构造函数会校验 dim >= 1（dim=0 会让 % dim 抛 ZeroDivisionError）。
            self.embedder = HashingEmbedder(dim=cfg.dim)
            self.dim = int(self.embedder.dim)
        # 维度权威性：无论哪条分支，config.dim 最终等于真正生效的维度。
        cfg.dim = self.dim

        # §2.7：低层类的 on_event 一律是 LowLevelEvent | None，只接收 (事件名, data)。
        self.on_event: LowLevelEvent | None = on_event
        # 词法：dict 保序（3.7+）就是我们的插入序 —— FIFO 淘汰"最旧"直接取第一个键；
        # 同 id 覆盖赋值不会改变它在 dict 里的位置，因此"更新"不会被误当成"新写入"而
        # 影响淘汰顺序。
        self._items: dict[str, MemoryItem] = {}
        # 矩阵缓存：与 _items 同序的 (id, 向量行) 快照，避免每次 search 重新收集一遍。
        self._matrix_ids: list[str] = []
        self._matrix: list[list[float]] = []
        self._matrix_dirty = True
        # tokenizer 只用于"三层统一 set"（§8.6 from_config）；向量层的检索不需要它，
        # 但保持与 BufferMemory / SummaryMemory 同形，便于 MemoryManager 统一注入。
        self._tokenizer: Tokenizer | None = None

    # ---- tokenizer（与另两层同形的注入点）----

    @property
    def tokenizer(self) -> Tokenizer:
        """当前生效的 tokenizer（覆盖值 > 默认）。"""
        return self._tokenizer if self._tokenizer is not None else get_default_tokenizer()

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer) -> None:
        with self._lock:
            self._tokenizer = value

    # ---- 内部工具（调用方必须已持有 self._lock）----

    def _embed(self, text: str) -> list[float]:
        """把文本编码成向量，并**在这一层**校验维度。

        维度必须在写入前就炸出来：一旦让一个 dim 不符的向量进了 `_matrix`，
        之后每次 search 都会在 `cosine_similarity` 里以更难定位的形态失败
        （甚至 numpy 路径下抛出形状错误而不是框架异常）。
        """
        vec = self.embedder.embed_one(text)
        if len(vec) != self.dim:
            raise MemoryStoreError(
                store=self.name,
                message=(
                    f"{type(self.embedder).__name__} returned dim {len(vec)}, "
                    f"expected {self.dim} (embedder={self.embedder.name!r})"
                ),
            )
        return vec

    def _prepare_embedding(self, item: MemoryItem) -> None:
        """确保 `item.embedding` 就绪：缺则补算，有但维度不符则抛。调用方必须已持锁。"""
        if item.embedding is None:
            item.embedding = self._embed(item.content)
            return
        if len(item.embedding) != self.dim:
            raise MemoryStoreError(
                store=self.name,
                message=(
                    f"item {item.id} embedding dim {len(item.embedding)} != store dim {self.dim}"
                ),
            )

    def _matrix_locked(self) -> tuple[list[str], list[list[float]]]:
        """返回与 `self._items` 同序的 (ids, 向量矩阵)。调用方必须已持锁。

        为什么缓存：`DEFAULT_MAX_MEMORY_ITEMS=10000` 时，没有缓存的话每次 search 都要
        重新收集 10000 个 list 引用；缓存之后只有写入路径才付这份成本（写少读多）。
        防御性地再校验一次长度：任何绕过公开方法的写入（子类/测试直接改 `_items`）
        都会让缓存失效并被这里纠正，而不是让矩阵与条目错位。
        """
        if self._matrix_dirty or len(self._matrix_ids) != len(self._items):
            ids: list[str] = []
            rows: list[list[float]] = []
            for item_id, item in self._items.items():
                ids.append(item_id)
                if item.embedding is None:  # pragma: no cover - 公开路径保证不可能
                    warnings.warn(
                        f"VectorMemory: item {item_id} has no embedding; "
                        "treating it as a zero vector during retrieval",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    rows.append([0.0] * self.dim)
                else:
                    rows.append(item.embedding)
            self._matrix_ids = ids
            self._matrix = rows
            self._matrix_dirty = False
        return self._matrix_ids, self._matrix

    def _similarities(
        self, query_vec: Sequence[float] | None, rows: Sequence[Sequence[float]]
    ) -> list[float]:
        """一次算完"查询 vs 全部候选行"的余弦。`query_vec is None` -> 全 0。

        `use_numpy=False`（或环境里没有 numpy）时逐行走 `cosine_similarity`；否则走
        `cosine_similarity_matrix`（它内部用 numpy，且实现上与单条 API **逐位一致**，
        两条路径切换不会改变断言结果）。
        """
        if query_vec is None:
            return [0.0] * len(rows)
        if self.config.use_numpy and NUMPY_AVAILABLE and len(rows) > 1:
            return cosine_similarity_matrix(query_vec, rows)
        return [cosine_similarity(query_vec, row) for row in rows]

    def _score_parts(
        self, item: MemoryItem, sim: float, now: float
    ) -> tuple[float, float, float, float]:
        """§8.5.1 的冻结公式，返回 `(sim, recency, importance, score)`。

          sim        = cosine_similarity(query_vec, item.embedding)  范围 [-1, 1]
          recency    = 2 ** (-clamp(age_days, 0, +inf) / half_life_days)  范围 (0, 1]
          importance = clamp(item.importance, 0.0, 1.0)

        `half_life_days <= 0` 会让指数除零：把它退化成"阶梯"（未过期 1.0 / 过期 0.0）
        而不是抛异常 —— 一个手滑的配置不该让整个检索崩掉，而且这个退化是单调的，
        排序语义仍然成立。

        SPEC-AMBIGUITY（规范空缺，本实现裁决）: §8.5.1 只写了 `recency` 的**值域**是
        `(0, 1]`，却把 `age_days = (now - created_at)/86400` 定义成可以为负 —— 当
        `now < item.created_at`（时钟回拨，或从磁盘 `load` 出带"未来"时间戳的条目）时，
        指数变成**正**数：轻则 `recency > 1.0` 违反冻结值域，重则 `2.0 ** 大正数` 抛
        `OverflowError` 并冲出 `search`。裁决：**把 age 钳到 >= 0**，即"未来时间戳等价于
        此刻，这条记忆还没有开始变旧"。钳制方向是单调安全的：越老的记忆 recency 越小，
        "来自未来"的条目不会比刚写入的条目更受偏爱，排序语义不被破坏。

        为什么不在 `MemoryItem.age_days` 里钳：那是 §8.5.1 逐字冻结的公开公式
        （返回 `(now - created_at)/86400`，负年龄是它字面上的、可解释的输出），
        把钳制下沉到公式里会让一个公开方法撒谎；打分才是那个对值域有承诺的地方。
        """
        half_life = float(self.config.half_life_days)
        age_days = item.age_days(now=now)
        # 钳制发生在**开方之前**：不钳的话 `-age_days/half_life` 是正指数，
        # `2.0 ** 1000` 之类的表达式直接 OverflowError（非纯理论：时钟回拨即可命中）。
        if age_days < 0.0:
            age_days = 0.0
        if half_life > 0.0:
            recency = 2.0 ** (-age_days / half_life)
        else:
            recency = 1.0 if age_days <= 0.0 else 0.0
        importance = min(1.0, max(0.0, float(item.importance)))
        score = (
            float(self.config.w_sim) * float(sim)
            + float(self.config.w_recency) * recency
            + float(self.config.w_importance) * importance
        )
        return (float(sim), recency, importance, score)

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """低层事件出口（形态与 §6.3 的 `BaseLLMClient._emit` 一致）。

        SPEC-AMBIGUITY: §8.5 在 FIFO 淘汰处写了"记 WARNING 并发事件"，但 §2.7 的事件归属
        矩阵把 memory 的三条事件（MEMORY_WRITE / MEMORY_RETRIEVE / MEMORY_COMPRESS）
        全部划给 `MemoryManager`，§13 红线 18 又规定"同一事件只能由唯一发射者发出"。
        裁决：本层**只发 warning**（可观测，且不会与 MemoryManager 重复记账）；
        `on_event` 出口保留给自定义子类复用同一形态。
        """
        if self.on_event is not None:
            self.on_event(event_type, to_jsonable(dict(data)))

    @staticmethod
    def _warn(message: str) -> None:
        """淘汰/降级必须留痕（§13 红线 12），warnings 默认按位置去重不会刷爆日志。"""
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    # ---- MemoryStore 实现 ----

    def add(self, item: MemoryItem) -> None:
        """未算 embedding 时补算；维度不符 -> MemoryStoreError；
        超过 max_items -> FIFO 淘汰最旧（记 WARNING）。

        "最旧"用的是**插入序**（dict 的键顺序）而不是 created_at：FIFO 的字面语义是
        "先来的先走"，而 upsert 会刷新 created_at —— 若按 created_at 淘汰，一条被反复
        确认的老记忆会永远压在库里、把新记忆挤走，这不是 FIFO 想表达的意思。
        """
        with self._lock:
            self._prepare_embedding(item)
            is_new = item.id not in self._items
            self._items[item.id] = item
            self._matrix_dirty = True
            if is_new:
                self._evict_overflow_locked()

    def _evict_overflow_locked(self) -> None:
        """把条数压回 `max_items` 之内（FIFO）。调用方必须已持锁。

        `max_items <= 0` 时每次 add 都会被立刻淘汰 —— 这是字面语义（配置说"最多存 0 条"），
        但必须留痕，否则用户会看到"写进去了又搜不到"而毫无线索。
        """
        limit = int(self.config.max_items)
        if limit < 0:
            limit = 0
        while len(self._items) > limit:
            oldest_id = next(iter(self._items))
            self._items.pop(oldest_id, None)
            self._matrix_dirty = True
            self._warn(
                f"VectorMemory: max_items={limit} exceeded; evicting oldest item {oldest_id} (FIFO)"
            )

    def get(self, item_id: str) -> MemoryItem | None:
        """按 id 取条目。返回的是**库里的实例**（与 `all()` 同一约定，见 §8.1），
        因此调用方不要就地改它；检索侧的状态（score/access_count）本来就是共享的。"""
        with self._lock:
            return self._items.get(item_id)

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RETRIEVE_LIMIT,
        min_score: float | None = None,
        now: float | None = None,
        use_mmr: bool | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """混合打分检索（§8.5.1 的 6 步冻结流程）。

        **这不是只读操作**：命中的条目会被写回 `score` / `score_breakdown` /
        `access_count` / `last_access_at`（同一把锁内完成）。调用方要清楚这一点 ——
        重复 search 会污染"访问频次"这类统计。
        """
        with self._lock:
            # now 的注入点：传了就完全确定（测试可手算 recency），没传才取唯一时钟。
            current = _config.utc_now() if now is None else float(now)
            # [v2 冻结] 唯一一处刻意改名：config 里叫 retrieve_min_score。
            threshold = (
                float(self.config.retrieve_min_score) if min_score is None else float(min_score)
            )
            top_k = max(0, int(limit))
            if top_k == 0 or not self._items:
                # 空库/零条请求：不要白算一次 query 的 embedding（注入的 embedder 可能
                # 对未知文本直接抛错，而"空库检索"本不该碰它）。
                return []

            query_vec: list[float] | None = None
            if query:
                query_vec = self._embed(query)

            ids, matrix = self._matrix_locked()
            sims = self._similarities(query_vec, matrix)

            scored: list[MemoryItem] = []
            for item_id, sim in zip(ids, sims):
                # 第 1 步：过滤。负相关明显不相关 —— 留着它只会靠近因/重要度混进结果。
                if sim < 0.0:
                    continue
                item = self._items[item_id]
                if not _matches_metadata_filter(item, metadata_filter):
                    continue
                sim_value, recency, importance, score = self._score_parts(item, sim, current)
                if score < threshold:
                    continue
                # 第 2 步：把打分结果写回条目（observability：面试时可解释"为什么这条排第一"）。
                item.score = score
                item.score_breakdown = {
                    "sim": sim_value,
                    "recency": recency,
                    "importance": importance,
                    "score": score,
                }
                scored.append(item)

            # 第 3 步：三级稳定排序 (-score, -created_at, id)。第三级用 id 是为了让
            # "同分同秒"的条目也有确定顺序（测试可以无条件断言顺序）。
            scored.sort(key=lambda entry: (-(entry.score or 0.0), -entry.created_at, entry.id))

            # 第 4 步：候选池取 3 倍（MMR 需要比 limit 更多的候选才有"选择"可言）。
            pool_size = max(top_k * 3, top_k)
            candidates = scored[:pool_size]

            # 第 5 步：MMR。默认 limit > 1 时开启（limit=1 时 MMR 退化成直接取最高分）。
            mmr_enabled = use_mmr if use_mmr is not None else top_k > 1
            if mmr_enabled:
                selected = self._mmr_select(candidates, top_k)
            else:
                selected = candidates[:top_k]

            # 第 6 步：命中条目的持久化副作用（同一把锁内，且与检索结果一一对应）。
            for item in selected:
                item.access_count += 1
                item.last_access_at = current
            return selected

    def _mmr_select(self, pool: Sequence[MemoryItem], limit: int) -> list[MemoryItem]:
        """Maximal Marginal Relevance 重排（冻结公式与并列规则）。调用方必须已持锁。

            mmr_i = mmr_lambda * score_i - (1 - mmr_lambda) * max_{j in selected} sim(i, j)

        每轮选 argmax(mmr)，直到选满 limit 或无候选；并列时取 id 字典序最小者。
        λ=1 退化成"按分数取前 k"，λ=0 退化成"只挑与已选最不像的"。
        """
        lam = float(self.config.mmr_lambda)
        selected: list[MemoryItem] = []
        remaining = list(pool)
        while remaining and len(selected) < limit:
            best_index = 0
            best_key: tuple[float, str] | None = None
            for index, item in enumerate(remaining):
                if selected:
                    redundancy = max(
                        cosine_similarity(item.embedding, chosen.embedding)
                        for chosen in selected
                    )
                else:
                    redundancy = 0.0
                mmr = lam * (item.score or 0.0) - (1.0 - lam) * redundancy
                # key 的元组比较天然实现"并列取 id 字典序最小"（确定性）。
                key = (-mmr, item.id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def all(self) -> list[MemoryItem]:
        """按 `created_at` 升序（**冻结**：只复制外层 list，元素是同一批实例）。
        同 `created_at` 时保持插入序（Python 的 sorted 是稳定排序）。"""
        with self._lock:
            return sorted(self._items.values(), key=lambda item: item.created_at)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(item_id, None) is not None
            if existed:
                self._matrix_dirty = True
            return existed

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._matrix_ids = []
            self._matrix = []
            self._matrix_dirty = True

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    # ---- 向量层专有 ----

    def upsert(self, item: MemoryItem) -> tuple[MemoryItem, bool]:
        """去重写入。返回 `(最终条目, 是否新建)`。

        两条去重规则（冻结）：
        - **同 id**：`dict` 里已经有这条 -> 视为更新而非新建（不触发 FIFO，不新增条目）；
        - **近似内容**：与已有条目的最大余弦相似度 `>= dedup_threshold` -> 更新那一条
          （content 替换、created_at 刷新、importance 取 max、access_count 保留）。

        为什么 access_count 要保留：它是"这条记忆被证明有用过几次"的证据，去重合并时丢掉
        会让"经常被召回的老事实"看起来像刚写进来的新记忆，后面的重要度排序就没有依据了。
        为什么 created_at 要刷新：内容刚被用户重新确认过，近因项应当反映"最近一次确认"，
        否则一条被复述的偏好会因为写得早而在排序里吃亏。
        """
        with self._lock:
            # 先补齐 embedding：新建分支要拿它跟库里比对，合并分支也要用新内容的向量。
            self._prepare_embedding(item)

            existing = self._items.get(item.id)
            if existing is not None:
                self._merge_locked(existing, item)
                self._matrix_dirty = True
                return existing, False

            best: MemoryItem | None = None
            best_sim = -1.0
            ids, matrix = self._matrix_locked()
            if matrix:
                for item_id, sim in zip(ids, self._similarities(item.embedding, matrix)):
                    if sim > best_sim:
                        best = self._items[item_id]
                        best_sim = sim
            if best is not None and best_sim >= float(self.config.dedup_threshold):
                self._merge_locked(best, item)
                self._matrix_dirty = True
                return best, False

            # 走 add 的公共路径（同一把 RLock 可重入），FIFO 与矩阵失效都在那里统一处理。
            self.add(item)
            return item, True

    def _merge_locked(self, target: MemoryItem, incoming: MemoryItem) -> None:
        """把 `incoming` 合并进 `target`（去重命中时的更新）。调用方必须已持锁。

        只更新"这条记忆是什么"相关的字段（content/role/metadata/embedding/source）与
        冻结要求刷新的 created_at、取 max 的 importance；**保留** access_count 与
        last_access_at（它们的语义是"这条记忆的历史使用情况"，与内容替换无关）。
        """
        target.content = incoming.content
        target.created_at = _config.utc_now()
        target.importance = max(float(target.importance), float(incoming.importance))
        if incoming.role:
            target.role = incoming.role
        if incoming.source:
            target.source = incoming.source
        # metadata 做并集（incoming 优先）：调用方可能只想补一个键，整体替换会丢掉
        # 之前写进去的标签（如 kind / role）。
        merged = dict(target.metadata)
        merged.update(incoming.metadata)
        target.metadata = merged
        # 内容换了，向量必须跟着换 —— 用 incoming 已经算好的那份，避免对同一段文本
        # 重复调用 embedder（注入型 embedder 的调用次数也是测试可观测的）。
        target.embedding = incoming.embedding
        # score/score_breakdown 是"上一次检索的产物"，内容已变，旧分数不再成立。
        target.score = None
        target.score_breakdown = None

    def score_item(
        self,
        item: MemoryItem,
        query_vec: Sequence[float] | None,
        *,
        now: float | None = None,
    ) -> float:
        """见 §8.5.1 的冻结公式。`query_vec=None` 时 sim 项为 0。

        **纯函数式**：只返回分数，不写回 `item.score` / `item.score_breakdown`
        （写回是 `search` 第 2 步的职责）。这样测试可以自由地拿它做手算对照，
        而不用担心"算一次就改了对象状态"。
        """
        current = _config.utc_now() if now is None else float(now)
        if query_vec is None or item.embedding is None:
            sim = 0.0
        else:
            sim = cosine_similarity(query_vec, item.embedding)
        return self._score_parts(item, sim, current)[3]

    def should_auto_write(self, message: Message) -> bool:
        """按 write_policy 判定：

        - `"manual"`    -> 恒 False（只有显式 remember 才写）
        - `"turn"`      -> `role == "user"` 时恒 True
        - `"selective"` -> `role == "user"` 且 (长度 >= `auto_write_min_chars` 或命中
          `AUTO_WRITE_MARKERS`) 且内容非空、不是纯符号

        [v2 冻结] **role 约束不可豁免**：`role != "user"` 时恒 False（任何策略下）。
        `MemoryManager.aadd` 传 `auto_write=True` 只豁免长度/marker 启发式，
        **不得**豁免 role 约束。

        理由（这是这套设计最值得讲的一点）：assistant 的答案**不该**被当成"用户事实"写进
        长期库。否则模型自己编的一句话会被下一轮当成用户说过的事实召回，形成"自我确认循环"
        —— 幻觉被写进长期记忆后，连纠正的机会都没有了。
        """
        role = _role_value(message)
        if role != "user":
            # 任何策略下的第一条硬约束，先判它，后面的启发式都不参与。
            return False

        policy = self.config.write_policy
        if policy == "manual":
            return False
        if policy == "turn":
            # "恒 True"：这是"每轮都记"的激进模式，调用方自己为噪声负责。
            return True
        if policy != "selective":
            # 未知策略 -> 不写 + 留痕。宁可少写（用户能从 warning 立刻发现拼写错误），
            # 也不要按某个默认策略悄悄往长期库里灌数据。
            self._warn(
                f"VectorMemory: unknown write_policy {policy!r}; "
                "no automatic long-term write (expected one of selective/turn/manual)"
            )
            return False

        content = message.content or ""
        if not _is_meaningful(content):
            return False
        if len(content) >= int(self.config.auto_write_min_chars):
            return True
        lowered = content.lower()
        return any(marker in lowered for marker in AUTO_WRITE_MARKERS)

    def stats(self) -> dict[str, Any]:
        """长期库的自述信息（供 `MemoryManager.stats()` 与 CLI `/memory` 使用）。"""
        with self._lock:
            return {
                "name": self.name,
                "items": len(self._items),
                "dim": self.dim,
                "embedder": self.embedder.name,
                "write_policy": self.config.write_policy,
                "max_items": int(self.config.max_items),
                "retrieve_limit": int(self.config.retrieve_limit),
                "retrieve_min_score": float(self.config.retrieve_min_score),
                "dedup_threshold": float(self.config.dedup_threshold),
                "half_life_days": float(self.config.half_life_days),
                "mmr_lambda": float(self.config.mmr_lambda),
                "weights": {
                    "sim": float(self.config.w_sim),
                    "recency": float(self.config.w_recency),
                    "importance": float(self.config.w_importance),
                },
                # 报"真正生效"的加速开关：装了 numpy 但 use_numpy=False 时这里是 False。
                "use_numpy": bool(self.config.use_numpy and NUMPY_AVAILABLE),
            }

    # ---- [v2 新增] 持久化：让"长期存储"真的跨进程 ----

    def save(self, path: str | os.PathLike[str], *, include_embedding: bool = True) -> int:
        """JSONL 追加写（'w' 覆盖），每行 `MemoryItem.to_dict(include_embedding=...)`。
        每 `DEFAULT_PERSIST_BATCH` 条 flush 一次。返回写入条数。
        父目录不存在则创建；写失败 -> MemoryStoreError。

        为什么默认 `include_embedding=True`：不存向量的"持久化"没有意义 —— 读回来之后
        要么重算（换 embedder 就检索不到了），要么维度不符直接报错。
        """
        target = os.fspath(path)
        with self._lock:
            # 先快照、后写盘：§8.1 要求临界区内没有 I/O，而 10000 条的 to_dict 也在锁内
            # 会很贵 —— 快照是纯内存操作，写文件放到锁外。
            snapshot = [
                item.to_dict(include_embedding=include_embedding)
                for item in self._items.values()
            ]
        count = 0
        try:
            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                for record in snapshot:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    count += 1
                    if count % DEFAULT_PERSIST_BATCH == 0:
                        handle.flush()
                handle.flush()
        except OSError as exc:
            raise MemoryStoreError(
                store=self.name,
                message=f"failed to save vector memory to {target!r}: {exc}",
                cause=exc,
            ) from exc
        return count

    def load(self, path: str | os.PathLike[str]) -> int:
        """读 JSONL 并 add 进内存。返回加载条数。

        - 坏行（JSON 非法 / 缺 id 与 content / 字段类型不对）**跳过并记 WARNING**，不抛；
        - `embedding` 维度与 `self.dim` 不符 -> `MemoryStoreError`（这是"配置错了"，
          继续加载只会让库处于半脏状态，明确失败比静默降级好）；
        - 文件不存在 -> 返回 0 并记 WARNING（`MemoryManager.restore` 的契约要求不抛，
          见 §8.6，因此这里不能抛）。

        没有 embedding 的行会由 `add` 触发一次重算（`from_dict` 不重算，是 `add` 补的）；
        跨进程持久化时请用 `save(include_embedding=True)` 避免这份额外的计算。
        """
        target = os.fspath(path)
        if not os.path.exists(target):
            self._warn(
                f"VectorMemory.load: file {target!r} does not exist; nothing loaded"
            )
            return 0
        loaded = 0
        try:
            handle = open(target, "r", encoding="utf-8")
        except OSError as exc:
            raise MemoryStoreError(
                store=self.name,
                message=f"failed to read vector memory from {target!r}: {exc}",
                cause=exc,
            ) from exc

        with handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, Mapping):
                        raise ValueError("JSONL line is not an object")
                    item = MemoryItem.from_dict(payload)
                except Exception as exc:  # noqa: BLE001 - 坏行的形态不可枚举（JSON/字段/类型）
                    self._warn(
                        f"VectorMemory.load: skipping bad line {lineno} of {target!r}: {exc!r}"
                    )
                    continue
                if item.embedding is not None and len(item.embedding) != self.dim:
                    raise MemoryStoreError(
                        store=self.name,
                        message=(
                            f"loaded item {item.id} has embedding dim {len(item.embedding)}, "
                            f"expected {self.dim}; refusing to load a mixed-dimension store"
                        ),
                    )
                self.add(item)
                loaded += 1
        return loaded



