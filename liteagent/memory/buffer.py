from __future__ import annotations

# 短期记忆：有界消息窗口（§8.3）。
#
# 为什么裁剪推迟到 window()：add() 在每个 observation 之后都会被调用，必须是 O(1)；
# 而裁剪要对整段 buffer 做 token 估算 + 工具对完整性修复。把"算窗口"推迟到真正要把
# 消息送进模型的那一刻，add 保持便宜，而送出去的上下文永远是 buffer 的最新视图。
#
# 术语（§0.5，别混）：
#   buffer  = 全量历史（self._messages，跨 run 累积、未裁剪）
#   window  = 裁剪后真正送进模型的那批（window() 每次重算）
#   evicted = 在 buffer 里但不在 window 里、等待摘要的那些
#
# 线程模型（§8.1 冻结小节）：所有公开方法在 `self._lock = threading.RLock()` 内完成，
# 临界区内没有 await、没有 I/O。为什么用 threading 而不是 asyncio.Lock —— 工具线程
# （没有运行中的事件循环）会直接调 BufferMemory，而 §0.4 的 R-LOOP 又禁止在 __init__
# 里创建 asyncio 原语。threading 原语与事件循环无关，这正是选它的理由（D-12）。
# 背景：`TeamConfig.share_memory=True` 时多个子 Agent 共享一个 MemoryManager，
# Hierarchical 的 worker 在各自线程 + 各自新建的 loop 里跑，主 loop 同时可能在读窗口，
# 无锁下会得到 "list changed size during iteration" 或读到裁剪到一半的状态。
#
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import threading
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

from liteagent.config import (
    DEFAULT_BUFFER_MAX_MESSAGES,
    DEFAULT_BUFFER_MAX_TOKENS,
)
from liteagent.llm.message import Message, Role, drop_orphan_tool_messages
from liteagent.memory.base import Tokenizer, get_default_tokenizer


@dataclass
class BufferConfig:
    """短期窗口的预算与保留策略（§8.3，字段名与默认值冻结）。

    双约束 = 条数（`max_messages`）+ token 预算（`max_tokens`），裁剪时谁先触发听谁的。
    `keep_last_n` 是"最小可用上下文"的下限：宁可超预算，也要保证最近 N 条在窗口里，
    否则模型会在一段没有近因的上下文上瞎猜。
    """

    max_tokens: int = DEFAULT_BUFFER_MAX_TOKENS
    max_messages: int = DEFAULT_BUFFER_MAX_MESSAGES
    keep_last_n: int = 2                # 至少保留最近 N 条消息（即使超预算）
    keep_system: bool = True
    tokenizer: Tokenizer | None = None  # None -> get_default_tokenizer()


def _repair_tool_pairs(messages: Sequence[Message]) -> list[Message]:
    """工具对完整性修复（§8.3 第 5 步调用的**模块级**函数，测试直接 import 它）。

    规则（冻结）：若保留的 assistant 消息带 `tool_calls`，则它的**全部** tool 结果必须
    也在列表里，否则把该 assistant 消息整体丢弃（连同它的工具结果）。

    与 `llm.message.drop_orphan_tool_messages` 的分工：后者先跑，按"tool_call_id 能否
    找到配对"做粗筛（孤立 tool 消息删掉、assistant 的无结果 tool_calls 剥掉，但**保留
    assistant 消息本体**）；本函数是第二道闸，专治"assistant 的半截工具对"——当它的结果
    已经不在窗口里时，宁可整条 assistant 丢掉，也不要把残缺的工具对送进 API
    （OpenAI / Anthropic 收到 tool_calls 但没有对应 tool 结果会直接 400）。

    为什么这里是"整体丢弃"而不是像 drop_orphan 那样只剥 tool_calls：本函数的场景是窗口
    裁剪边界，被裁掉的 assistant 往往正在"等结果"，留下它的思考文本会让模型看到一个它
    无法自洽的中间状态。两种粒度都是冻结判据（§8.3 / §6.1），不要互相替代。

    无 id 的 tool_call 视为**无法配对**（等价于结果缺失）：`drop_orphan_tool_messages`
    同样把空 `tool_call_id` 当孤儿，两个函数对"什么算合法配对"的口径必须一致。

    未改动的消息**原对象返回**（不复制）：`BufferMemory` 的 evicted 计算依赖消息相等性，
    少造副本能减少"窗口里的对象不在 buffer 里"这类边界情况（与 §6.1 同一考虑）。
    """
    items = list(messages)

    # 声明侧：哪些 tool_call id 真的有结果消息与之配对。空 id 不入集合 —— 没有 id
    # 就没有可核对的配对关系，保守地当作"结果缺失"。
    resolved: set[str] = {
        message.tool_call_id
        for message in items
        if message.role == Role.TOOL and message.tool_call_id
    }

    # 第一遍：找出要整体丢弃的 assistant，并记下它们的 tool_call id。
    dropped_call_ids: set[str] = set()
    kept: list[Message] = []
    for message in items:
        if message.is_tool_pair_start():
            call_ids = [call.id for call in message.tool_calls if call.id]
            complete = (
                len(call_ids) == len(message.tool_calls)
                and all(call_id in resolved for call_id in call_ids)
            )
            if not complete:
                dropped_call_ids.update(call_ids)
                continue
            kept.append(message)
            continue
        kept.append(message)

    if not dropped_call_ids:
        # 绝大多数情况下不需要改动任何消息，连列表推导都省掉（这个是热路径：
        # 每一轮 LLM 调用前都会算一次窗口）。
        return kept

    # 第二遍：连同被丢弃 assistant 的工具结果一起删掉（"连同它的工具结果"）。
    return [
        message
        for message in kept
        if not (message.role == Role.TOOL and message.tool_call_id in dropped_call_ids)
    ]


class BufferMemory:
    """短期记忆：有界消息窗口，按 token 预算 + 条数双约束裁剪。

    [v2] **线程安全**：所有公开方法在 `self._lock = threading.RLock()` 内完成（§8.1）。

    两个"消息集合"别混：`messages()` 是全量 buffer（只增不减，`clear` 除外），
    `window()` 是每次重算出来的、真正送进模型的那批。`evicted()` 就是"在 buffer 里
    但不在 window 里"的那些。
    """

    def __init__(self, config: BufferConfig | None = None,
                 *, on_evict: Callable[[list[Message]], None] | None = None) -> None:
        """`on_evict`：旁路观测回调，`window()` 每次算出**新**被裁掉的消息时调用一次
        （已裁过的不会重复通知）；回调抛异常只记 warning，不影响 `window()` 的返回。
        想要"取出并消费"应当用 `drain_evicted()` —— 那是给摘要器用的语义，有幂等保证。
        """
        # **不复制** config：§8.6 的上下文窗口反推预算会在构造后写
        # `buffer.config.max_tokens = max(512, budget)`，复制会让那次调整静默失效。
        self.config: BufferConfig = config if config is not None else BufferConfig()
        self._on_evict = on_evict
        # R-LOOP（§0.4）：__init__ 里只允许 threading 原语（它不绑事件循环）。
        self._lock = threading.RLock()
        self._messages: list[Message] = []
        self._evicted: list[Message] = []
        # 已经被 drain_evicted 取走的消息对象 id()：取走即视为"已交给摘要器"。
        # 下一次 window() 重算时必须把它们排除，否则同一批消息会被反复摘要
        # —— 每轮开头 window() 一次、每轮末尾 drain 一次的循环会让摘要无限重算。
        # 用 id() 是安全的：这些消息始终被 self._messages 强引用，对象不会被回收，
        # id 也就不会被复用（clear() 会连同 _drained_ids 一起重置）。
        self._drained_ids: set[int] = set()
        # 构造后统一 set 的 tokenizer 覆盖（§8.6 from_config）；None 表示从 config 取。
        self._tokenizer_override: Tokenizer | None = None

    # ---- tokenizer 注入 ----

    @property
    def tokenizer(self) -> Tokenizer:
        """当前生效的 tokenizer（解析顺序：构造后 set 的覆盖 > `config.tokenizer` > 默认）。

        为什么允许构造后 set：`MemoryManager.from_config(tokenizer=...)` 要用一个
        参数统一覆盖三层的 tokenizer（§8.6），而 BufferConfig 里的 tokenizer 只是
        它的持久化落点；两条路径都要能生效。
        """
        return self._tokenizer_override or self.config.tokenizer or get_default_tokenizer()

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer) -> None:
        with self._lock:
            self._tokenizer_override = value

    def _estimate_tokens(self, messages: Sequence[Message]) -> int:
        """消息 -> token 估算。走 `Tokenizer.estimate_many`，保证与摘要触发判定、
        向量记忆截断用的是同一套估算函数（§8.1 D-06：估算函数必须唯一）。

        每次调用重新解析 tokenizer（而不是在 __init__ 里缓存）：估算函数被换掉时
        必须立刻生效，否则"注入 CallableTokenizer"只对之后新建的 buffer 有效。
        """
        return self.tokenizer.estimate_many([message.text() for message in messages])

    # ---- 写 ----

    def add(self, message: Message) -> None:
        """追加；不在此处裁剪（裁剪在 window/trim 时做），保证 add 是 O(1)。"""
        with self._lock:
            self._messages.append(message)

    def extend(self, messages: Sequence[Message]) -> None:
        """批量追加。与逐条 `add` 等价，只是少拿几把锁。"""
        with self._lock:
            self._messages.extend(messages)

    # ---- 读 ----

    def messages(self) -> list[Message]:
        """全量（未裁剪）副本。返回新 list，元素仍是同一批 Message 实例。"""
        with self._lock:
            return list(self._messages)

    def window(self) -> list[Message]:
        """裁剪后的窗口（每次调用重算）。实现 §8.3 的 6 步冻结算法。"""
        with self._lock:
            msgs = list(self._messages)
            result = self._compute_window(msgs)
            # 第 6 步：evicted = 在 buffer 里但不在 window 里的消息，按原序返回。
            # 再扣掉已经 drain 过的（避免同一批消息被反复喂给摘要器）。
            pending = [
                message
                for message in self._compute_evicted(msgs, result)
                if id(message) not in self._drained_ids
            ]
            previous_ids = {id(message) for message in self._evicted}
            self._evicted = pending
            fresh = [message for message in pending if id(message) not in previous_ids]

        if self._on_evict is not None and fresh:
            # 回调在锁**外**触发：它是用户的代码，可能做 I/O 甚至回调回本对象；
            # 持锁调用会把整个 buffer 卡住（RLock 虽然可重入，但别的线程进不来）。
            self._fire_on_evict(fresh)
        return result

    def estimated_tokens(self) -> int:
        """全量（buffer）的估算。窗口的估算用 `window_tokens()`。"""
        with self._lock:
            return self._estimate_tokens(self._messages)

    def window_tokens(self) -> int:
        """窗口的估算。**重算一次窗口**而不是读缓存：窗口是"每次调用重算"的语义，
        缓存会让 `window_tokens()` 在 `max_tokens` 被动态调整后给出过期数字。"""
        with self._lock:
            return self._estimate_tokens(self._compute_window(list(self._messages)))

    def evicted(self) -> list[Message]:
        """被裁掉的消息（供摘要使用）。返回副本；已 `drain_evicted` 取走的不再出现。"""
        with self._lock:
            return list(self._evicted)

    def drain_evicted(self) -> list[Message]:
        """取出并清空 evicted（摘要器调用；取走后不会重复摘要）。

        幂等：第二次调用返回 `[]`。取走的消息 id 会被记进 `_drained_ids`，因此即使
        之后再调 `window()`，它们也不会重新出现在 `evicted()` 里。
        """
        with self._lock:
            pending = list(self._evicted)
            self._drained_ids.update(id(message) for message in pending)
            self._evicted = []
            return pending

    def last(self) -> Message | None:
        """最近追加的一条（**全量 buffer** 的末尾，不是窗口的末尾）；空 buffer -> None。

        SPEC-AMBIGUITY: §8.3 只给了 `last()` 的签名，没说取全量还是取窗口。裁决：取全量
        —— 名字没加限定词时，"最后一条"指最后被 `add` 进来的那条；默认配置下它与窗口
        末尾一致（`keep_last_n>=1` 保证最近的消息一定在窗口里），只在 `keep_last_n=0`
        之类的极端配置下才分叉。
        """
        with self._lock:
            return self._messages[-1] if self._messages else None

    def replace_last(self, message: Message) -> bool:
        """把末尾那条消息**原地替换**成 `message`（角色必须相同）。

        [v3 新增] 给 Agent 的终结分支用：步骤 2 已经把模型这一轮的**原始**响应写进窗口，
        终结时又需要把 `strip_markers` 之后的答案落库 —— 若直接再 `add` 一次，同一条响应
        会在短期窗口里出现两遍（text 模式是 `Final Answer: X` 与 `X` 两条；native 模式
        则是逐字相同的两条），多轮对话的窗口以 1.5 倍速度膨胀、模型也会看到自己的答案
        两次。替换语义比追加更贴合"同一条响应只有一个版本"。

        返回是否真的替换了（末尾不存在、角色不同、或内容完全相同 -> False，调用方据此
        决定要不要退化为 `add`）。
        """
        with self._lock:
            if not self._messages:
                return False
            previous = self._messages[-1]
            if previous.role != message.role:
                return False
            if previous is message:
                return False
            self._messages[-1] = message
            return True

    def clear(self, *, keep_system: bool = True) -> None:
        """清空。`keep_system=True` 时保留**前导连续**的 SYSTEM 消息（与窗口的 pinned
        前缀同一口径），因为系统提示通常由调用方在 run 之间复用。"""
        with self._lock:
            if keep_system:
                index = 0
                while index < len(self._messages) and self._messages[index].role == Role.SYSTEM:
                    index += 1
                del self._messages[index:]
            else:
                self._messages.clear()
            self._evicted = []
            # 被删掉的消息对象可能已被回收，它们的 id 可能被新对象复用 —— 必须清空，
            # 否则"新造的同 id 消息"会被误判成已摘要。
            self._drained_ids.clear()

    def __len__(self) -> int:
        """全量条数（**不是**窗口条数）—— `MemoryManager.stats()` 的
        `short_term_messages` 用它。"""
        with self._lock:
            return len(self._messages)

    # ---- 内部：冻结的裁剪算法 ----

    def _compute_window(self, msgs: list[Message]) -> list[Message]:
        """§8.3 冻结算法的 1-5 步。调用方必须已持有 `self._lock`。"""
        if not msgs:
            return []

        config = self.config

        # 第 2 步：把前导连续的 SYSTEM 消息单独取出为 pinned。
        pinned: list[Message] = []
        rest: list[Message] = msgs
        if config.keep_system:
            index = 0
            while index < len(msgs) and msgs[index].role == Role.SYSTEM:
                index += 1
            pinned = msgs[:index]
            rest = msgs[index:]

        max_messages = max(0, config.max_messages)
        keep_last_n = max(0, config.keep_last_n)
        pinned_tokens = self._estimate_tokens(pinned)

        # 第 3 步：从尾部向前累加，直到任一约束先触发。
        kept: list[Message] = []
        kept_tokens = 0
        for message in reversed(rest):
            # 条数约束排在最前：它是**无条件**的上限（"len(kept) == max_messages -> 停"），
            # 因此窗口长度永远 <= max_messages，即使 keep_last_n 比它更大。
            if len(kept) >= max_messages:
                break
            message_tokens = self._estimate_tokens([message])
            # 预算约束：越界就"不保留该条并停止"。但 len(kept) < keep_last_n 时豁免
            # （允许超预算）—— 这是"最小可用上下文"，没有它模型会在一条消息都没有的
            # 情况下被调用。
            if len(kept) >= keep_last_n and pinned_tokens + kept_tokens + message_tokens > config.max_tokens:
                break
            kept.append(message)
            kept_tokens += message_tokens

        # 第 4 步：pinned 在前，其余按原序（kept 是倒着攒的，要翻回来）。
        result = pinned + list(reversed(kept))

        # 第 5 步：工具对完整性修复（先粗筛孤儿，再清半截工具对）。
        result = drop_orphan_tool_messages(result)
        return _repair_tool_pairs(result)

    @staticmethod
    def _compute_evicted(msgs: list[Message], window: list[Message]) -> list[Message]:
        """§8.3 第 6 步：被裁掉的消息，按原序返回。调用方必须已持有 `self._lock`。

        SPEC-AMBIGUITY: §8.3 第 6 步把它写成 `[m for m in msgs if m not in window]`，
        那是**集合**差（`in` 走 dataclass 相等性，不是身份）。当 buffer 里存在两条内容
        完全相同的消息（模板化提问、重试、脚本化测试都极易撞上）时，被裁掉的那条会因为
        "窗口里有一条和它相等的消息"而被判定为"没被裁掉"：于是 evicted 返回空，被裁掉的
        历史既不进窗口也不进摘要，凭空消失 —— 这与 `evicted()` 的文档承诺
        （"被裁掉的消息，供摘要使用"）直接冲突。
        裁决：按**多重集合差**实现（窗口里每条消息最多抵消 buffer 里一条相等消息），
        与字面写法在"消息互不相同"时逐字等价，有重复时给出文档承诺的语义。
        """
        remaining = list(window)
        evicted: list[Message] = []
        for message in msgs:
            for index, candidate in enumerate(remaining):
                if candidate is message or candidate == message:
                    del remaining[index]
                    break
            else:
                evicted.append(message)
        return evicted

    def _fire_on_evict(self, messages: list[Message]) -> None:
        """触发用户的 `on_evict` 回调。

        为什么把异常吞在这里：`on_evict` 是纯旁路观测（把被裁掉的消息抄送给外部做审计/
        统计），它抛异常不该让 `window()` 失败、进而让整个 agent 轮次崩掉。但 §13 红线
        10/12 不允许静默 —— 必须留下 warning。`CancelledError` 继承 BaseException，
        不会被 `except Exception` 吞掉（M-5）。
        """
        callback = self._on_evict
        if callback is None:  # pragma: no cover - 调用方已判空，防御性
            return
        try:
            callback(list(messages))
        except Exception as exc:  # noqa: BLE001 - 见 docstring：观测回调必须降级而非放大
            warnings.warn(
                f"BufferMemory.on_evict callback raised {exc!r}; eviction continues",
                RuntimeWarning,
                stacklevel=2,
            )
