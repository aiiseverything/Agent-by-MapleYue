from __future__ import annotations

# 记忆编排层（§8.6）：Agent 只与本类交互，不直接碰 Buffer / Vector / Summary。
#
# 三层协作的分工（面试讲这一张图就够）：
#   BufferMemory   —— 短期窗口。送进模型的**当前上下文**，有界、可裁剪、O(1) 追加。
#   SummaryMemory  —— 压缩。把"被窗口裁掉的历史"压成一段可携带的摘要，永不抛异常。
#   VectorMemory   —— 长期。跨 run / 跨进程的事实库，按混合打分检索。
# 本类的价值就在于**把三者的时序固定下来**：写（aadd）只写短期 + 有策略地写长期；
# 压缩只在每轮末尾做一次；组装 prompt 时摘要在前、长期记忆次之、窗口再次、本轮增量输入最后。
#
# 为什么所有检索入口都要有 `now` 参数（[v2 变更]）：`recency = 2 ** (-age_days/half_life)`
# 依赖"现在"。如果只在最底层读时钟，上层就无法构造确定性断言（测试只能断言"大概"）。
# 因此 `abuild_prompt(now=...)` -> `aretrieve(now=...)` -> `retrieve(now=...)` ->
# `VectorMemory.search(now=...)` 逐级透传，任何一层都不会中途换成 `utc_now()`。
#
# 事件（§2.7）：MEMORY_WRITE / MEMORY_RETRIEVE / MEMORY_COMPRESS / CONTEXT_TRUNCATED
# 的唯一发射者都是本类 —— memory 层不得 import agent 层，所以事件类型用 §9.2 EventType
# 的 `.value` 字符串，由 `agent.callbacks.as_llm_callback` 适配成 TraceEvent。
#
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import os
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

# 别名成 _config：本类的构造函数有名为 `config` 的形参（§8.6 冻结签名），会遮蔽模块对象。
# 取别名后 `_config.utc_now()` 在任何作用域里都指向模块，而 tests.helpers.frozen_time 的
# `patch("liteagent.config.utc_now")` 依旧生效（patch 的是模块对象上的属性）。
from liteagent import config as _config
from liteagent.config import (
    DEFAULT_MEMORY_BLOCK_CHARS,
    DEFAULT_MEMORY_QUERY_CHARS,
    DEFAULT_REACT_SYSTEM_PROMPT,
    DEFAULT_TOKEN_CHAR_RATIO,
    render_template,
    to_jsonable,
    truncate_head_tail,
)
from liteagent.errors import ConfigError, MemoryStoreError
from liteagent.llm.message import Message, Role, drop_orphan_tool_messages
from liteagent.memory.base import (
    MemoryConfig,
    MemoryItem,
    Tokenizer,
    get_default_tokenizer,
)
from liteagent.memory.buffer import BufferConfig, BufferMemory, _repair_tool_pairs
from liteagent.memory.embeddings import Embedder, LowLevelEvent, default_embedder
from liteagent.memory.summary import SummaryConfig, SummaryMemory
from liteagent.memory.vector import VectorConfig, VectorMemory

if TYPE_CHECKING:
    # 只为注解存在（PEP 563 下注解不求值）。memory -> llm 是允许方向，但**不在运行期**
    # import llm.base：避免把整个 provider 家族拖进 memory 包的 import 图，
    # 也让本文件能脱离 LLM 实现单独 import / 冒烟测试。
    from liteagent.llm.base import LLMClient

# 摘要块的标签（§8.4 冻结的渲染格式）。
_SUMMARY_OPEN = "<conversation_summary>"
_SUMMARY_CLOSE = "</conversation_summary>"

# 长期记忆块的标签（§8.6 冻结的渲染格式）。与 vector.py 用的是同一对字面量 ——
# 两处都写死而不是互相 import：渲染发生在哪一层，哪一层就持有自己的字面量，
# 这样 vector.py 不依赖 manager，反过来也一样。
_MEMORIES_OPEN = "<relevant_memories>"
_MEMORIES_CLOSE = "</relevant_memories>"

# `to_dict()` 里最多带多少条长期记忆（CLI `/memory` 用）。长期库可能有上万条，
# 全量塞进 trace/CLI 输出既慢又没人看；按 created_at 取最新的若干条足够说明问题。
_TO_DICT_LONG_TERM_LIMIT = 20


def _buffer_config_from(cfg: MemoryConfig) -> BufferConfig:
    """`MemoryConfig` -> `BufferConfig`（§8.6 的冻结映射表的逐字段实现）。

    改名三处（`max_*` / `keep_last_n`）是历史遗留，不许"顺手统一"—— 映射表是冻结的。
    `keep_system=True` 固定：系统提示必须一直钉在窗口头部，否则被裁掉后模型会失去行为约束。
    """
    return BufferConfig(
        max_tokens=cfg.buffer_max_tokens,
        max_messages=cfg.buffer_max_messages,
        keep_last_n=cfg.buffer_keep_last_n,
        keep_system=True,
    )


def _summary_config_from(cfg: MemoryConfig) -> SummaryConfig:
    """`MemoryConfig` -> `SummaryConfig`（§8.6 映射表）。

    `max_input_chars=16000` / `update_existing=True` 是固定的：前者限制一次摘要喂给模型的
    对话量（避免摘要调用自己就超上下文），后者是滚动更新（新摘要合并旧摘要，而不是从零重写）。
    """
    return SummaryConfig(
        enabled=cfg.summary_enabled,
        trigger_ratio=cfg.summary_trigger_ratio,
        min_evict_batch=cfg.summary_min_evict,
        max_summary_chars=cfg.max_summary_chars,
        max_input_chars=16000,
        update_existing=True,
    )


def _vector_config_from(cfg: MemoryConfig) -> VectorConfig:
    """`MemoryConfig` -> `VectorConfig`（§8.6 映射表）。

    唯一一处**刻意改名**：`retrieve_min_score`（config）-> `retrieve_min_score`（VectorConfig）
    与 `VectorMemory.search(min_score=...)` 的分工 —— 配置项与检索参数名字不同是有意的：
    一个是"默认门槛"，一个是"这次检索的门槛"。
    """
    return VectorConfig(
        dim=cfg.embedder_dim,  # 仅在 embedder is None 时生效（VectorMemory 会按注入的 embedder 覆盖）
        write_policy=cfg.write_policy,
        auto_write_min_chars=cfg.auto_write_min_chars,
        dedup_threshold=cfg.dedup_threshold,
        max_items=cfg.max_items,
        retrieve_limit=cfg.retrieve_limit,
        retrieve_min_score=cfg.retrieve_min_score,
        w_sim=cfg.w_sim,
        w_recency=cfg.w_recency,
        w_importance=cfg.w_importance,
        half_life_days=cfg.recency_half_life_days,
        mmr_lambda=cfg.mmr_lambda,
        use_numpy=True,
    )


def _role_str(message: Message) -> str:
    """取消息 role 的规范字符串（小写）。

    与 `vector._role_value` 同构（两个模块各自持有一份 4 行的工具函数，比为这点小事
    引入一条"manager -> vector 的私有名 import"更清晰）。`Role` 是 str-mixin 枚举，
    但鸭子类型的替身可能给出 `"User"` 这种 capitalize 过的值，统一归一化一次。
    """
    role = getattr(message, "role", None)
    if isinstance(role, Role):
        return role.value
    return str(role or "").strip().lower()


def _is_framework_message(message: Message) -> bool:
    """是否是**框架自己生成**、但 role 恰好是 USER 的消息（见 `aadd` 里的裁决）。"""
    kind = (message.metadata or {}).get("kind")
    return kind in ("observation", "nudge")


class MemoryManager:
    """三层协作的唯一入口。Agent 只与本类交互，不直接碰 Buffer/Vector/Summary。"""

    def __init__(
        self,
        *,
        buffer: BufferMemory | None = None,
        long_term: VectorMemory | None = None,
        summarizer: SummaryMemory | None = None,
        config: MemoryConfig | None = None,
        tokenizer: Tokenizer | None = None,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        """按配置装配三层。

        - 显式传入的层优先（注入点，测试用它替换 embedder / 假摘要器）；
        - `long_term is None` 且 `config.long_term_enabled` 为假 -> `self.long_term = None`
          （"禁用长期记忆"是一条真实路径：省掉 embedding 计算与 O(n) 检索）；
        - `summarizer is None` -> 造一个 `llm=None` 的摘要器（走抽取式兜底）。
          本构造函数**没有** `llm` 形参（§8.6 冻结签名），需要真 LLM 摘要请用
          `MemoryManager.from_config(..., llm=...)` —— `Agent` 内部走的正是那条路径。
        - `tokenizer` 非 None 时**统一 set 到三层**（构造后 set 而不是构造参数，
          因为冻结的三个层各自有自己的 tokenizer 构造参数）。
        """
        cfg = config if config is not None else MemoryConfig()
        self.config: MemoryConfig = cfg
        self.on_event: LowLevelEvent | None = on_event

        self.buffer: BufferMemory = (
            buffer if buffer is not None else BufferMemory(_buffer_config_from(cfg))
        )
        if long_term is not None:
            self.long_term: VectorMemory | None = long_term
        elif cfg.long_term_enabled:
            # E12：from_config 造默认 embedder 是允许的同层边；这里同构（直接构造也用默认 embedder）。
            self.long_term = VectorMemory(
                embedder=default_embedder(cfg.embedder_dim),
                config=_vector_config_from(cfg),
            )
        else:
            self.long_term = None
        self.summarizer: SummaryMemory = (
            summarizer
            if summarizer is not None
            else SummaryMemory(llm=None, config=_summary_config_from(cfg))
        )

        self._tokenizer: Tokenizer | None = tokenizer
        if tokenizer is not None:
            self._apply_tokenizer(tokenizer)

        # [v2 新增] 最近一次 abuild_prompt 检索到的条目（只读属性，见下）。
        # 存在的理由是**可观测性**：§9.4.1 要用 `len(memory.last_retrieved)` 发事件，
        # 而重新 search 一次会污染 access_count（search 是带副作用的）。
        self._last_retrieved: list[MemoryItem] = []

        # D-15：上下文窗口反推短期窗口预算（config.context_window_tokens 为 None 时不动）。
        self._apply_context_budget()

    # ---- 只读属性 ----

    @property
    def tokenizer(self) -> Tokenizer:
        """当前生效的 token 估算器（显式注入 > 进程默认）。

        "估算函数必须唯一"（§8.1 D-06）：窗口裁剪、摘要触发、观察值字符预算、
        记忆块截断都走这一个实例。
        """
        return self._tokenizer if self._tokenizer is not None else get_default_tokenizer()

    @property
    def last_retrieved(self) -> list[MemoryItem]:
        """最近一次 `abuild_prompt` 检索到的条目（**只读属性**，元素是库里的实例）。

        返回外层 list 的**副本**：调用方不能通过 append/clear 影响管理器状态
        （元素本身仍是共享实例 —— 与 `MemoryStore.all()` 的冻结约定一致）。
        """
        return list(self._last_retrieved)

    @property
    def buffer_budget_tokens(self) -> int:
        """短期窗口当前生效的 token 预算。

        直接读 `buffer.config.max_tokens`（而不是缓存一份）：D-15 的反推预算、以及调用方
        构造后对 `buffer.config.max_tokens` 的任何调整，都必须立刻在这两个入口上体现。
        """
        return int(self.buffer.config.max_tokens)

    # ---- 内部：装配 ----

    def _apply_tokenizer(self, tokenizer: Tokenizer) -> None:
        """三层统一 set tokenizer。缺属性/只读属性的层跳过并留痕（鸭子类型容错）。

        为什么不用 hasattr 判一次就赋值：hasattr 只能证明"读得到"，读得到不等于写得了
        （只读 property 会在赋值时抛 AttributeError）。注入自定义层是公开用法，
        这里不能让它把整个 MemoryManager 的构造搞崩。
        """
        for layer in (self.buffer, self.summarizer, self.long_term):
            if layer is None:
                continue
            try:
                layer.tokenizer = tokenizer
            except AttributeError:
                warnings.warn(
                    f"MemoryManager: {type(layer).__name__} does not accept a tokenizer "
                    "override; that layer keeps its own tokenizer",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _apply_context_budget(self) -> None:
        """[v2 新增] D-15 的上下文窗口反推预算。

            budget = context_window_tokens - reserve_completion_tokens
                     - tools_schema_tokens_reserve - tokenizer.estimate(system_prompt)
            buffer.config.max_tokens = max(512, budget)     # 下限 512

        为什么要反推：用户配的是"我的模型有 8k 上下文"，而框架需要的是"短期窗口能占多少"。
        让用户自己减出 3000 是反人类的，也让"换模型"变成一次全量配置重算。

        SPEC-AMBIGUITY: §8.6 说 system_prompt "由调用方提供，无则用
        `DEFAULT_REACT_SYSTEM_PROMPT` 渲染后的长度"，但 `from_config` 的冻结签名里
        **没有** system_prompt 形参。裁决：一律按后者（渲染默认模板，`tools` 用空串）——
        工具 schema 的份量由 `tools_schema_tokens_reserve` 单独预留，不重复计。
        """
        window = self.config.context_window_tokens
        if window is None:
            return
        system_prompt = render_template(
            DEFAULT_REACT_SYSTEM_PROMPT,
            {"name": "agent", "tools": "", "tool_names": ""},
        )
        budget = (
            int(window)
            - int(self.config.reserve_completion_tokens)
            - int(self.config.tools_schema_tokens_reserve)
            - self.tokenizer.estimate(system_prompt)
        )
        if budget < 512:
            # 反推结果是负数（窗口比"预留 + 系统提示"还小）：取 512 下限，但必须留痕 ——
            # 这类配置错误直接表现为"上下文被裁得只剩两三条消息"，没有 warning 会查很久。
            warnings.warn(
                f"MemoryManager: context_window_tokens={window} leaves no room for the "
                f"short-term window (computed budget={budget}); clamping to 512",
                RuntimeWarning,
                stacklevel=2,
            )
        self.buffer.config.max_tokens = max(512, int(budget))

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """低层事件出口（形态与 §6.3 的 `BaseLLMClient._emit` 一致）。

        §2.7 要求进 `data` 的值必须已过 `to_jsonable`，这样 `TraceEvent.to_json()` 里的
        `json.dumps` 永不因类型而抛 TypeError。
        """
        if self.on_event is not None:
            self.on_event(event_type, to_jsonable(dict(data)))

    def _estimate_messages(self, messages: Sequence[Message]) -> int:
        """整段消息的 token 估算（走唯一的 tokenizer）。"""
        return self.tokenizer.estimate_many([message.text() for message in messages])

    # ---- 写 ----

    async def aadd(self, message: Message, *, auto_write: bool | None = None) -> MemoryItem | None:
        """1) `buffer.add(message)`
           2) 若 long_term 启用且 `should_auto_write(message)`（`auto_write` 参数可覆盖启发式判定）
              -> `await awrite_long_term(...)`（走 upsert 去重）
           3) emit MEMORY_WRITE `{kind: message.metadata.get("kind"), count}`
           返回新写入的长期条目或 None。**不做**压缩检查（压缩在轮次末尾统一做）。

        `auto_write` 的三态语义（[v2 冻结]）：
        - `None`（默认）：走启发式判定；
        - `True`：**只豁免**长度/marker 启发式，**不豁免 role 约束** —— 对 `role != "user"`
          的消息传 True 会记 WARNING 并拒绝写入（否则 assistant 的答案会被当成用户事实）；
        - `False`：显式关闭本次长期写入（调用方想精确控制何时写）。
        """
        self.buffer.add(message)
        written: MemoryItem | None = None
        role = _role_str(message)

        if self.long_term is not None:
            if auto_write is True and role != "user":
                # 这是调用方的编码错误，不是数据错误：明确拒绝 + 留痕（§13 红线 10/12）。
                warnings.warn(
                    f"MemoryManager.aadd: auto_write=True ignored for role={role!r} "
                    "(only user messages may become long-term facts)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif auto_write is False:
                pass
            elif role == "user" and (
                auto_write is True or self.long_term.should_auto_write(message)
            ):
                if _is_framework_message(message):
                    # SPEC-AMBIGUITY: §8.5 的 role 判据只说"role != user 恒 False"，
                    # 没有提到"框架自己生成、但 role=USER"的消息（文本模式的 Observation、
                    # nudge 纠偏提示，§6.1 / §9.4.4 都用 role=USER）。若照字面放行，
                    # 一段 40 字以上的工具观察或纠偏文案会被当成"用户事实"写进长期库、
                    # 并在之后每一轮被召回 —— 与 role 约束的立法意图（别让非用户输入变事实）
                    # 直接矛盾。裁决：本层跳过这些 kind，并留一次可观测的 warning。
                    warnings.warn(
                        "MemoryManager.aadd: skipping long-term auto-write for framework "
                        f"message kind={message.metadata.get('kind')!r} (only real user "
                        "messages become long-term facts)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    written = await self.awrite_long_term(self._auto_item(message, role))

        # count 计的是"这条消息进入了记忆"这件事（短期写入是无条件的，发生在上面第一行）。
        self._emit("memory_write", {"kind": message.metadata.get("kind"), "count": 1})
        return written

    @staticmethod
    def _auto_item(message: Message, role: str) -> MemoryItem:
        """把一条消息转成长期条目。

        `role` 写进 metadata 是有意的：`VectorMemory.search(metadata_filter=...)` 只匹配
        metadata（见 vector.py 的裁决），把 role 放进去，调用方就能做
        `metadata_filter={"role": "user"}` 这类过滤。importance 用 0.5 的中位默认值 ——
        自动写入的条目没有"用户显式声明重要性"的信息，抬高它反而会压过手动 remember。
        """
        metadata = dict(message.metadata or {})
        metadata.setdefault("role", role)
        return MemoryItem.create(
            message.content or "",
            role=role,
            importance=0.5,
            metadata=metadata,
            source="auto",
        )

    def replace_last_in_buffer(self, message: Message) -> bool:
        """把短期窗口末尾那条**同角色**消息原地替换成 `message`（[v3 新增]）。

        终结分支专用：Agent 的步骤 2 已把模型这一轮的**原始**响应写进 buffer，
        终结时只需把原始文本规范化成 `strip_markers` 之后的最终答案 —— 而不是**再追加
        一条**（v2 会追加，于是同一条响应在窗口里出现两遍：text 模式是
        `Final Answer: X` 与 `X`，native 模式是逐字相同的两条；多轮对话的窗口因此以
        1.5 倍速度膨胀、模型也会看到自己的答案两次）。

        不发射 MEMORY_WRITE：这条消息在步骤 2 就已经进过记忆，替换不算新的写入。
        返回是否真的替换了；调用方据此决定是否退化为 `aadd`。
        """
        return self.buffer.replace_last(message)

    async def aadd_turn(
        self, user_message: Message, assistant_message: Message | None = None
    ) -> list[MemoryItem]:
        """保留 API（当前没有调用点）。文档明确标注"保留 API"，避免实现者猜它是否有副作用。

        语义：把一轮对话（用户 + 可选 assistant）都写进短期记忆，并返回**这一轮真正写进
        长期库**的条目列表。assistant 侧不会有长期写入（role 约束），因此返回值通常只含
        用户那一条 —— 这也是它现在没有调用点的原因：`aadd` 已经能表达同样的意思。
        """
        written: list[MemoryItem] = []
        item = await self.aadd(user_message)
        if item is not None:
            written.append(item)
        if assistant_message is not None:
            extra = await self.aadd(assistant_message)
            if extra is not None:
                written.append(extra)
        return written

    async def aremember(
        self,
        content: str,
        *,
        importance: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
        source: str = "",
    ) -> MemoryItem:
        """显式长期写入（`memory_tools.remember` / Agent 调用）。走 upsert 去重。
        `long_term` 为 None 时抛 `MemoryStoreError`。

        为什么显式写入也要走去重：用户可能在不同轮次说"记住我用 uv 管理依赖"，
        没有去重就会积累十几条近义记忆，检索时它们还会互相挤占 MMR 的额度。
        """
        if self.long_term is None:
            raise MemoryStoreError(
                store="memory_manager",
                message="aremember requires long-term memory (long_term_enabled=False)",
            )
        item_metadata = dict(metadata) if metadata is not None else {}
        item_metadata.setdefault("role", "user")
        item = MemoryItem.create(
            content,
            role="user",
            importance=float(importance),
            metadata=item_metadata,
            source=source or "manual",
        )
        final = await self.awrite_long_term(item)
        self._emit("memory_write", {"kind": final.metadata.get("kind"), "count": 1})
        return final

    async def awrite_long_term(self, item: MemoryItem) -> MemoryItem:
        """把条目写进长期库（upsert 去重），返回**库里真正保存的那一条**。

        为什么是 async：`VectorMemory` 的写入要算 embedding，而生产环境的 embedder 可能是
        `RemoteEmbedder`（一次网络调用）。现在它内部是同步调用（纯内存 + 本地哈希），
        保留 async 形态是为了将来换成 `asyncio.to_thread` 时不破坏 API。
        """
        if self.long_term is None:
            raise MemoryStoreError(
                store="memory_manager",
                message="awrite_long_term requires long-term memory (long_term_enabled=False)",
            )
        final, _created = self.long_term.upsert(item)
        return final

    # ---- 读（[v2 变更] 全部支持 now 注入）----

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        now: float | None = None,
        min_score: float | None = None,
        use_mmr: bool | None = None,
    ) -> list[MemoryItem]:
        """同步检索（纯内存，无 LLM）。**逐级透传 now 给 `VectorMemory.search`**。

        `now=None` 时让最底层去取唯一时钟（只有真正需要"现在"时才读时钟，传了就完全确定）。
        `limit` / `min_score` 为 None 时取 `config` 的默认值（`retrieve_limit` /
        `retrieve_min_score`）—— "默认值只在一处定义"。
        """
        if self.long_term is None:
            return []
        return self.long_term.search(
            query,
            limit=self.config.retrieve_limit if limit is None else int(limit),
            min_score=self.config.retrieve_min_score if min_score is None else min_score,
            now=now,
            use_mmr=use_mmr,
        )

    async def aretrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        now: float | None = None,
        min_score: float | None = None,
        use_mmr: bool | None = None,
    ) -> list[MemoryItem]:
        """异步镜像（当前实现只是调用同一份 sync 逻辑）。

        保留异步形态是为了未来支持 `RemoteEmbedder` 的网络调用而不破坏 API。
        现在**刻意不**改成 `asyncio.to_thread(self.retrieve, ...)`：那会让"测试用同步 stub
        替换 `retrieve`"这条常见路径踩到 M-2（`to_thread` 遇到 async 函数会原样返回
        coroutine 对象，调用方拿到的是 coroutine 而不是结果）。
        """
        return self.retrieve(query, limit=limit, now=now, min_score=min_score, use_mmr=use_mmr)

    async def abuild_prompt(
        self,
        *,
        system: str,
        user_input: str,
        extra: Sequence[Message] = (),
        retrieve: bool = True,
        memory_query: str | None = None,
        now: float | None = None,
        append_user_input: bool = True,
    ) -> list[Message]:
        """**组装顺序（冻结）**：

           1. `Message(SYSTEM, system)`
           2. [摘要消息]  若 `summarizer.summary` 非空 -> `Message(SYSTEM, '<conversation_summary>...')`
              metadata={"kind": "summary"}
           3. [长期记忆块] 若 retrieve 且检索有结果 -> `Message(SYSTEM, '<relevant_memories>...')`
              metadata={"kind": "memories", "memory_count": n}
           4. `buffer.window()` 的全部消息
           5. `extra` 里的消息（如 multiagent 的上下文注入）
           6. `append_user_input=True` 时追加 `Message(USER, user_input)`

        [v2 新增] 检索结果的可观测性：本方法把检索到的条目存进 `self.last_retrieved`
        （**不再触发第二次 search** —— `search` 会写 `access_count`，重复调用会污染数据）。
        [v2 新增] 末尾的最终预算校验见 `_enforce_context_budget()`。
        注意：第 2/3 段是**每轮重新生成**的，不写回 buffer，避免重复累积。
        """
        messages: list[Message] = [Message.system(system)]

        # 第 2 段：摘要。放在系统提示之后、长期记忆之前（§8.4 冻结的位置）：
        # 先给"全局背景"，再给"相关细节"，最后才是逐字的历史。
        summary = self.summarizer.summary if self.summarizer is not None else ""
        if summary:
            messages.append(
                Message.system(
                    f"{_SUMMARY_OPEN}\n{summary}\n{_SUMMARY_CLOSE}", kind="summary"
                )
            )

        # 第 3 段：长期记忆。query 缺省用本轮用户输入。
        #
        # SPEC-AMBIGUITY: §9.4.1 只在 `state.step == 1` 传 user_input，后续轮传 ""
        # （`append_user_input=False`），而 `memory_query` 默认就是 user_input —— 若把
        # "空 query" 当成"不检索"，长期记忆块就只会出现在第一轮，第 2 轮起模型再也看不到
        # 自己刚刚召回的事实（`MEMORY_RETRIEVE` 也会一直是 0）。
        # 裁决：**空 query 仍然检索**，让 `VectorMemory.search` 走 `query_vec=None`
        # 分支（§8.5 明确写了"`query_vec=None` 时 sim 项为 0"），退化成
        # "按近因 + 重要度排序"，这样每轮的长期记忆块都存在且语义稳定。
        retrieved: list[MemoryItem] = []
        if retrieve and self.long_term is not None:
            query = user_input if memory_query is None else memory_query
            retrieved = await self.aretrieve(query, now=now)
            self._emit("memory_retrieve", {"count": len(retrieved), "query_len": len(query)})
        # 无论检索与否都覆盖：`last_retrieved` 的语义是"最近一次组装用了什么"，
        # 留着上一轮的条目会让调用方读到过期的可观测数据。
        self._last_retrieved = list(retrieved)

        memories_index: int | None = None
        if retrieved:
            block, included = self._render_memories(retrieved)
            if block:
                memories_index = len(messages)
                messages.append(
                    Message.system(block, kind="memories", memory_count=len(included))
                )

        # 第 4 段：短期窗口（已经过 tool-pair 修复的、真正能送进模型的那批）。
        window_messages = self.buffer.window()
        window_indices = list(range(len(messages), len(messages) + len(window_messages)))
        messages.extend(window_messages)

        # 第 5 段：外部注入（如 multiagent 的共享上下文）。
        messages.extend(list(extra))

        # 第 6 段：[v2 变更] 当前轮的**增量**用户指令。空串不追加（避免造出空消息）；
        # 历史用户消息已经在 window 里了（Agent 在 run 开始时 aadd 过一次，§9.4.1）。
        #
        # [v3 修正] 上面那句注释描述的正是这个洞：既然 §9.4.1 步骤 0 已经把它写进 buffer，
        # 第 4 段又把整个 window 倒了出来，那么这里的无条件追加会让**每轮的当前用户消息
        # 在 prompt 里出现两次**（首轮 = [system, user, user]），白付一份 token，
        # 多轮对话里模型也会看到用户同一句话说了两遍。去重守卫：窗口末尾已经是同一条
        # 用户消息时跳过追加。`buffer_max_messages=0`（窗口为空）之类的退化配置下
        # 守卫不成立，仍会追加 —— 语义与 v2 一致，不会丢输入。
        already_in_window = bool(window_messages) and window_messages[-1].content == user_input
        if append_user_input and user_input and not already_in_window:
            messages.append(Message.user(user_input))

        return self._enforce_context_budget(
            messages,
            memories_index=memories_index,
            window_indices=window_indices,
        )

    def _render_memories(
        self, items: Sequence[MemoryItem]
    ) -> tuple[str, list[MemoryItem]]:
        """渲染 `<relevant_memories>` 块（格式冻结），返回 `(块文本, 实际收录的条目)`。

            <relevant_memories>
            - (score=0.83, 2026-09-27) <content>
            ...
            </relevant_memories>

        两级截断：单条 content 超过 `DEFAULT_MEMORY_QUERY_CHARS` 用 head+tail 截（保留结论），
        整块超过 `DEFAULT_MEMORY_BLOCK_CHARS` 时**按分数截条数**（items 已按分数降序，
        从尾部丢就是丢最不相关的）。
        """
        lines: list[str] = []
        included: list[MemoryItem] = []
        for item in items:
            content = item.content
            if len(content) > DEFAULT_MEMORY_QUERY_CHARS:
                content = truncate_head_tail(content, DEFAULT_MEMORY_QUERY_CHARS)
            score = 0.0 if item.score is None else float(item.score)
            # 日期用 UTC 渲染（§2.2 的时钟语义），只到"天" —— 记忆的粒度是"哪天说的"。
            date_text = _config.format_ts(item.created_at, fmt="%Y-%m-%d")
            lines.append(f"- (score={score:.2f}, {date_text}) {content}")
            included.append(item)

        # 整块预算：从尾部（最低分）逐条丢，至少保留 1 条 —— 一条都不给等于"检索白做了"，
        # 而模型至少该看到最相关的那条。
        total = len(_MEMORIES_OPEN) + len(_MEMORIES_CLOSE) + 2
        for line in lines:
            total += len(line) + 1
        while len(lines) > 1 and total > DEFAULT_MEMORY_BLOCK_CHARS:
            dropped_line = lines.pop()
            included.pop()
            total -= len(dropped_line) + 1

        if not included:
            return "", []
        block = _MEMORIES_OPEN + "\n" + "\n".join(lines) + "\n" + _MEMORIES_CLOSE
        return block, included

    def _enforce_context_budget(
        self,
        messages: list[Message],
        *,
        memories_index: int | None = None,
        window_indices: Sequence[int] = (),
    ) -> list[Message]:
        """最终预算校验（D-15）：整段估算超过 `context_window_tokens` 就按序裁剪。

        裁剪顺序（冻结）：**长期记忆块 -> 窗口最旧消息**；不动 pinned system、不动最后一条
        user（那个被裁掉就等于"这一轮没有输入"，模型会开始自由发挥）。

        为什么要有最后这道闸：窗口预算是"消息条数/token 的估算"，而摘要块、记忆块、系统提示
        都是**额外**拼上去的；只有把整段拼完再量一次，才能真正守住模型的上下文上限
        （超了就是 provider 侧 400，用户看到的是一次莫名其妙的失败）。
        """
        limit = self.config.context_window_tokens
        if limit is None:
            return messages
        before = self._estimate_messages(messages)
        if before <= int(limit):
            return messages

        working = list(messages)
        mem_index = memories_index
        candidates = [index for index in window_indices if 0 <= index < len(working)]

        # 第 1 刀：整块丢掉长期记忆。它是最"可有可无"的一段 —— 记忆检索是概率性的补充信息，
        # 而系统提示与当前对话是任务本身。
        if mem_index is not None and 0 <= mem_index < len(working):
            working.pop(mem_index)
            candidates = [index - 1 if index > mem_index else index for index in candidates]

        # 第 2 刀：从窗口**最旧**的一条开始丢（candidates 是按原序的，第一条就是最旧的）。
        while self._estimate_messages(working) > int(limit):
            last_user = max(
                (index for index, message in enumerate(working) if message.role == Role.USER),
                default=None,
            )
            pick: int | None = None
            for index in candidates:
                if index >= len(working):
                    continue
                message = working[index]
                if message.role == Role.SYSTEM:  # 不动 pinned system
                    continue
                if last_user is not None and index == last_user:  # 不动最后一条 user
                    continue
                pick = index
                break
            if pick is None:
                # 没有可丢的了（只剩 pinned system + 最后一条 user）：带着超预算的上下文
                # 继续跑。这里不再留 warning —— 上面那个 warning 已经在"反推预算不足"时
                # 报过一次，重复报同一件事只会淹没真正的新信息。
                break
            working.pop(pick)
            candidates = [index - 1 if index > pick else index for index in candidates]

        if len(working) == len(messages):
            return working

        # 裁剪可能切断工具对（assistant 带 tool_calls 但结果被丢了）—— 送进 API 会直接 400，
        # 所以复用 buffer 的同一套修复函数（§8.3 第 5 步的口径，不另造一份）。
        working = _repair_tool_pairs(drop_orphan_tool_messages(working))
        after = self._estimate_messages(working)
        self._emit(
            "context_truncated",
            {
                "before": before,
                "after": after,
                "dropped_messages": len(messages) - len(working),
            },
        )
        return working

    def build_prompt(self, **kwargs: Any) -> list[Message]:
        """`run_sync(lambda: self.abuild_prompt(**kwargs))`；在运行中的 loop 内抛 `ConfigError`
        （Agent 内部一律用 `abuild_prompt`）。

        为什么要把"在 loop 里调用同步版"变成硬错误：同步版会在当前线程新建一个 loop
        并阻塞等待，而外层 loop 还在跑 —— 两个 loop 争用同一批 asyncio 原语（R-LOOP / M-3）
        与同一个 threading 锁，症状是极难复现的死锁/串号。
        """
        return _config.run_sync(lambda: self.abuild_prompt(**kwargs))

    # ---- 压缩 ----

    async def acompress_if_needed(self, *, force: bool = False) -> str | None:
        """判定 + 执行 + 写回：取 `buffer.drain_evicted()`，若满足 `should_compress` 则
        `await summarizer.acompress(evicted, previous_summary=self.summarizer.summary)`，
        结果存入 summarizer，emit MEMORY_COMPRESS `{before_tokens, after_tokens, compressed}`。
        返回新摘要或 None。

        `summarizer` 为 None 时：若 evicted 非空则直接丢弃（记 WARNING）。

        **调用契约（冻结）**：Agent 只在每轮末尾调用一次（§9.4.1 步骤 7），**禁止**在单轮内
        为每条 observation 都调一次 —— 否则一轮 N 个工具会触发 N 次 LLM 摘要判定
        （N 次网络调用 + N 份 token 账单，而摘要质量并不会更好）。

        为什么先 drain 再判定（顺序照抄 §8.6）：`should_compress` 要的是"本次待压缩批次的
        规模"，而且 drain 有幂等保证 —— 取走的消息不会被反复喂给摘要器。代价是被取走但
        这轮没触发压缩的那几条不会再进摘要（它们已经不在窗口里了），这是冻结顺序的直接后果。
        """
        evicted = self.buffer.drain_evicted()
        if self.summarizer is None:
            if evicted:
                warnings.warn(
                    f"MemoryManager: dropping {len(evicted)} evicted message(s) because no "
                    "summarizer is configured",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        if not evicted:
            # 没有待压缩的消息：不调用摘要器（否则 compression_count 会被空转调用推高，
            # 而 compression_count 是"真的压过几次"的证据）。
            return None

        if not force and not self.summarizer.should_compress(
            current_tokens=self.buffer.estimated_tokens(),
            max_tokens=self.buffer.config.max_tokens,
            pending_evicted=len(evicted),
        ):
            return None

        before_tokens = self.tokenizer.estimate_many([message.text() for message in evicted])
        new_summary = await self.summarizer.acompress(
            evicted, previous_summary=self.summarizer.summary
        )
        self._emit(
            "memory_compress",
            {
                "before_tokens": before_tokens,
                "after_tokens": self.summarizer.summary_tokens,
                "compressed": bool(new_summary),
            },
        )
        return new_summary or None

    async def acompress(self, *, force: bool = False) -> str | None:
        """`acompress_if_needed` 的显式别名（§8.6 冻结：同上的显式别名）。"""
        return await self.acompress_if_needed(force=force)

    # ---- 维护 ----

    async def aclear(self, *, long_term: bool = False, summary: bool = True) -> None:
        """清空记忆。

        - 短期窗口总是清空（`BufferMemory.clear(keep_system=True)`：系统提示不是"记忆"，
          清掉它下次调用就少了一段行为约束）；
        - `long_term=True` 才清长期库（默认保留 —— 那是跨 run 最有价值的部分）；
        - `summary=True` 才丢摘要。

        `last_retrieved` 一律清空：它是"上次组装用了什么"的缓存，清空记忆后继续暴露
        旧检索结果只会误导调用方。
        """
        self.buffer.clear()
        if long_term and self.long_term is not None:
            self.long_term.clear()
        if summary and self.summarizer is not None:
            # SummaryMemory 没有公开的 clear()（§8.4 的接口清单里没有）：有就用，
            # 没有就退回直接重置内部字段 —— 两层都是本包内实现，字段名是同一份契约的一部分。
            clear = getattr(self.summarizer, "clear", None)
            if callable(clear):
                clear()
            else:
                self.summarizer._summary = ""
        self._last_retrieved = []

    def stats(self) -> dict[str, Any]:
        """{"short_term_messages","short_term_tokens","window_messages","window_tokens",
            "evicted_pending","long_term_items","has_summary","summary_tokens",
            "compressions","embedder","buffer_budget_tokens","context_window_tokens"}

        [v2 变更] 末尾两个键是 §8.6 的预算反推结果（`context_window_tokens` 为 None 时
        `buffer_budget_tokens` 就等于 `config.buffer_max_tokens`）。

        SPEC-AMBIGUITY: D-15 说"四个数放进 stats()"，但 §8.6 又给了上面这份**逐字的键清单**。
        裁决：以键清单为准（它是被冻结的返回结构，测试会按它断言），`reserve_completion_tokens`
        与 `tools_schema_tokens_reserve` 仍可从 `manager.config` 直接读到。
        """
        summary = self.summarizer.summary if self.summarizer is not None else ""
        return {
            "short_term_messages": len(self.buffer),
            "short_term_tokens": self.buffer.estimated_tokens(),
            "window_messages": len(self.buffer.window()),
            "window_tokens": self.buffer.window_tokens(),
            "evicted_pending": len(self.buffer.evicted()),
            "long_term_items": len(self.long_term) if self.long_term is not None else 0,
            "has_summary": bool(summary),
            "summary_tokens": self.summarizer.summary_tokens if self.summarizer is not None else 0,
            "compressions": self.summarizer.compression_count if self.summarizer is not None else 0,
            "embedder": self.long_term.embedder.name if self.long_term is not None else None,
            "buffer_budget_tokens": self.buffer_budget_tokens,
            "context_window_tokens": self.config.context_window_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        """装配信息（trace / CLI `/memory` 用）。**不含** embedding（§2.2 的 opt-out：
        256 维浮点列表会让输出膨胀数十倍而没人看）。"""
        long_term_items: list[dict[str, Any]] = []
        if self.long_term is not None:
            recent = sorted(
                self.long_term.all(), key=lambda item: item.created_at, reverse=True
            )[:_TO_DICT_LONG_TERM_LIMIT]
            long_term_items = [item.to_dict(include_embedding=False) for item in recent]
        return {
            "stats": self.stats(),
            "summary": self.summarizer.summary if self.summarizer is not None else "",
            "last_retrieved": [
                item.to_dict(include_embedding=False) for item in self._last_retrieved
            ],
            "long_term": long_term_items,
        }

    @classmethod
    def from_config(
        cls,
        config: MemoryConfig | None = None,
        *,
        llm: "LLMClient | None" = None,
        embedder: Embedder | None = None,
    ) -> "MemoryManager":
        """按配置装配三层；`llm` 为 None 时摘要器用抽取式兜底。

        [v2 冻结] **必须把 embedder 透传给 `VectorMemory`**；
        `MemoryConfig.embedder_dim` 仅在 `embedder is None` 时生效。
        `persist_path` 非 None 时在构造末尾执行 `restore`（失败记 WARNING 不抛）。

        SPEC-AMBIGUITY: §8.6 提到 "`tokenizer` 参数覆盖三层内部各自的 tokenizer（构造后统一 set）"，
        但 `from_config` 的冻结签名里没有 `tokenizer` 形参。裁决：这句话描述的是 `__init__`
        的 `tokenizer` 参数（本方法一律传 None，让三层各自取 `get_default_tokenizer()` ——
        它本身是 `functools.cache` 的单例，三层拿到的仍是同一个实例）。
        """
        cfg = config if config is not None else MemoryConfig()

        vector_embedder = embedder if embedder is not None else default_embedder(cfg.embedder_dim)
        long_term = (
            VectorMemory(embedder=vector_embedder, config=_vector_config_from(cfg))
            if cfg.long_term_enabled
            else None
        )
        manager = cls(
            buffer=BufferMemory(_buffer_config_from(cfg)),
            long_term=long_term,
            summarizer=SummaryMemory(llm=llm, config=_summary_config_from(cfg)),
            config=cfg,
        )
        if cfg.persist_path is not None:
            try:
                manager.restore(cfg.persist_path)
            except Exception as exc:  # noqa: BLE001 - 启动期恢复失败不该让 Agent 起不来
                # 降级必须留痕（§13 红线 12）：用户会看到"长期记忆是空的"，
                # 而 warning 告诉他文件在哪、为什么没读进来。
                warnings.warn(
                    f"MemoryManager.from_config: restore from {cfg.persist_path!r} failed "
                    f"({exc!r}); starting with empty long-term memory",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return manager

    # ---- [v2 新增] 持久化 ----

    def persist(self, path: str | os.PathLike[str] | None = None) -> int:
        """把长期记忆写到磁盘（默认用 `config.persist_path`，为 None 时 `ConfigError`）。
        返回写入条数。`long_term` 为 None 时返回 0。
        """
        target = path if path is not None else self.config.persist_path
        if target is None:
            raise ConfigError(
                "MemoryManager.persist requires a path (pass path= or set MemoryConfig.persist_path)"
            )
        if self.long_term is None:
            return 0
        return self.long_term.save(target)

    def restore(self, path: str | os.PathLike[str] | None = None) -> int:
        """从磁盘恢复长期记忆。文件不存在 -> 返回 0（不抛）。

        与 `persist` 不同：路径为 None 时才抛 `ConfigError`（"没给路径"是调用错误），
        而"文件不存在"是正常的首次启动（`VectorMemory.load` 对此返回 0 并留 warning）。
        """
        target = path if path is not None else self.config.persist_path
        if target is None:
            raise ConfigError(
                "MemoryManager.restore requires a path (pass path= or set MemoryConfig.persist_path)"
            )
        if self.long_term is None:
            return 0
        if not os.path.exists(target):
            return 0
        return self.long_term.load(target)

    def tokenizer_chars_budget(self) -> int:
        """[v2 新增] 给 executor 用的字符预算：`buffer_budget_tokens * DEFAULT_TOKEN_CHAR_RATIO`。

        §7.4.1 步骤 5.d 的 `max_observation_chars` 用它取 min，把 token 预算与字符上限联动：
        观察值的截断口径必须与"窗口能装多少"同源，否则会出现"单条观察占满半个上下文"。
        """
        return int(self.buffer_budget_tokens * DEFAULT_TOKEN_CHAR_RATIO)




