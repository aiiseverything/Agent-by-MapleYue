from __future__ import annotations

# 摘要压缩（§8.4）：把"被窗口裁掉的那批消息"压成一段可携带的自然语言摘要。
#
# 为什么摘要必须是**独立于窗口**的一条消息：短期窗口是有界的，被裁掉的历史如果直接丢失，
# 模型就会忘记用户最初的目标与已经试错的路径；把摘要作为单独一段拼进 prompt
# （§8.4 冻结：位置在系统提示之后、长期记忆之前），既省 token 又保住关键事实。
#
# 本类的第一设计约束是"**永不抛异常**"：摘要是上下文优化，不是关键路径。LLM 挂掉、
# 没有配 LLM、返回空串 —— 一律退到零 LLM 的抽取式兜底，并在降级处留 warning
# （§13 红线 10/12）。调用方（MemoryManager.acompress_if_needed）因此可以完全不写
# try/except。
#
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import asyncio
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from liteagent.config import (
    DEFAULT_MAX_SUMMARY_CHARS,
    DEFAULT_SUMMARY_MIN_EVICT,
    DEFAULT_SUMMARY_TRIGGER_RATIO,
    DEFAULT_TEMPERATURE,
    render_template,
    truncate_head_tail,
)
from liteagent.llm.message import Message, Role, render_transcript
from liteagent.memory.base import Tokenizer, get_default_tokenizer

if TYPE_CHECKING:
    # 只为注解存在（annotations 已被 PEP 563 变成字符串，运行期不需要真类型）。
    # memory -> llm 是允许方向，但**不在运行期** import llm.base：一是避免把整个
    # provider 家族拖进 memory 包的 import 图，二是 llm/base.py 由另一个实现者负责，
    # 运行期依赖它会让本文件无法被单独 import 与冒烟测试。
    from liteagent.llm.base import LLMClient


# 给摘要模型看的提示词（冻结字面量，`{variables}` 用 `config.render_template` 渲染）。
# 为什么把"必须逐字保留什么"写得这么啰嗦：摘要是唯一会把细节抹平的环节，而 agent
# 场景里最贵的信息恰恰是路径/命令/失败原因这类"看着像噪声"的碎片。
SUMMARY_PROMPT_TEMPLATE: str = """\
You are compressing a conversation history for an AI agent.
Write a concise summary in the SAME LANGUAGE as the conversation.

It MUST preserve, verbatim where possible:
- the user's original goal and any explicit constraints
- file paths, function/class names, commands, URLs, numbers
- decisions already made and the reasons
- what has been tried and FAILED (so it is not retried)
- the current state of the task and what remains

Do NOT add commentary, do NOT invent facts. Use bullet points. Max {max_chars} characters.

{previous_block}
Conversation to compress:
{transcript}

Summary:
"""

# 抽取式兜底的字符上限（§8.4）。与 SummaryConfig.max_summary_chars 同源；当配置被显式
# 置成非正数（"不限长"的手滑写法）时用它兜底，避免摘要无限膨胀。
FALLBACK_SUMMARY_MAX_CHARS: int = DEFAULT_MAX_SUMMARY_CHARS

# 抽取式兜底里每条消息保留的 content 字符数（§8.4 冻结：前 80 字符）。
_FALLBACK_LINE_CHARS = 80


@dataclass
class SummaryConfig:
    """摘要器配置（§8.4，字段名与默认值冻结）。

    `update_existing=True` 是滚动更新模式：新摘要会把上一版摘要一起喂给模型做合并，
    而不是每轮从零重写（从零重写会丢掉早期目标 —— 那正是摘要存在的理由）。
    """

    enabled: bool = True
    trigger_ratio: float = DEFAULT_SUMMARY_TRIGGER_RATIO
    min_evict_batch: int = DEFAULT_SUMMARY_MIN_EVICT
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    max_input_chars: int = 16000        # 喂给摘要模型的对话文本上限
    update_existing: bool = True        # True -> 滚动更新（把旧摘要一起喂进去）


class SummaryMemory:
    """短期记忆的压缩器：触发判定 + LLM 摘要 + 抽取式兜底。

    `llm=None` 是合法配置（`MemoryManager.from_config(llm=None)`）：此时所有摘要都走
    `fallback_extractive_summary`，零 LLM、零网络、完全确定性 —— 这是离线测试与
    "没有 API key 也要能跑"的保证（§13 红线 12 要求这种降级必须留痕）。
    """

    def __init__(self, llm: "LLMClient | None" = None,
                 config: SummaryConfig | None = None,
                 *, tokenizer: Tokenizer | None = None) -> None:
        self._llm = llm
        # 与 BufferMemory 一致：不复制 config，允许调用方在构造后调整阈值。
        self.config: SummaryConfig = config if config is not None else SummaryConfig()
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else get_default_tokenizer()
        self._summary: str = ""
        self._compression_count: int = 0

    # ---- 只读属性 ----

    @property
    def summary(self) -> str:
        """当前摘要（空串表示还没有摘要）。"""
        return self._summary

    @property
    def summary_tokens(self) -> int:
        """摘要占用的 token 估算。与窗口/触发判定共用同一个 tokenizer（§8.1 D-06）。"""
        return self._tokenizer.estimate(self._summary)

    @property
    def compression_count(self) -> int:
        """`acompress` 被调用并产出结果的次数（单调递增，只增不减）。"""
        return self._compression_count

    @property
    def tokenizer(self) -> Tokenizer:
        """当前生效的 tokenizer。可构造后赋值，供 `MemoryManager.from_config` 统一覆盖三层。"""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer) -> None:
        self._tokenizer = value

    # ---- 触发判定 ----

    def should_compress(self, *, current_tokens: int, max_tokens: int,
                        pending_evicted: int) -> bool:
        """冻结判定（OR 关系，任一成立即压缩）：

          1. enabled 为 False -> 永不压缩
          2. current_tokens >= ceil(max_tokens * trigger_ratio)
          3. pending_evicted >= min_evict_batch

        为什么是 OR 而不是 AND：两个条件是两条独立的成本曲线 —— (2) 是"再不放摘要，
        下一次调用就要超上下文"，(3) 是"积压的 evicted 已经够多，再攒下去只能丢掉"。
        任意一条触发都说明"现在压缩比不压缩划算"。
        """
        if not self.config.enabled:
            return False
        if current_tokens >= math.ceil(max_tokens * self.config.trigger_ratio):
            return True
        return pending_evicted >= self.config.min_evict_batch

    # ---- 生成 ----

    async def acompress(self, messages: Sequence[Message], *,
                        previous_summary: str | None = None) -> str:
        """生成新摘要。**永不抛异常**：

        - llm 为 None 或调用失败 -> 走 `fallback_extractive_summary`
        - 结果超过 max_summary_chars -> `truncate_head_tail`
        - 每次都递增 `compression_count`

        返回新的摘要文本（空字符串表示无可摘要内容，调用方**不应**写入空摘要）。

        `previous_summary`：调用方已有的摘要。为 None 时回退到本对象已存的摘要
        （受 `config.update_existing` 控制，见 `_resolve_previous`）。
        """
        msgs = list(messages)
        if not msgs:
            # 无内容可摘要：按契约返回空串，让调用方跳过"写入摘要"这一步。
            # SPEC-AMBIGUITY: §8.4 说"每次都递增 compression_count"没有排除空输入，
            # 这里按字面执行（空输入也计数），保证"调用次数"与计数严格对应。
            self._compression_count += 1
            return ""

        text = await self._asummarize(msgs, previous_summary=previous_summary)
        text = self._truncate(text)

        self._compression_count += 1
        if text:
            # 绝不写入空摘要：空的返回值是"本次没压出东西"，不该覆盖上一版有效摘要。
            self._summary = text
        return text

    async def _asummarize(self, messages: list[Message], *,
                          previous_summary: str | None) -> str:
        """真正干活的那一步：LLM 走通就用它的输出，任何失败都降级到抽取式兜底。"""
        llm = self._llm
        if llm is None:
            # 这是配置性降级（没有 LLM 可用），不是异常 —— 但同样必须留痕（§13 红线 12）。
            # warnings 默认按 (message, category, module, lineno) 去重，只会提示一次。
            warnings.warn(
                "SummaryMemory has no LLM configured; using extractive fallback summary",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fallback_extractive_summary(messages)

        prompt = self._build_prompt(messages, previous_summary=previous_summary)
        try:
            response = await llm.achat(
                [Message.user(prompt)],
                temperature=DEFAULT_TEMPERATURE,
            )
        except asyncio.CancelledError:
            # M-5：CancelledError 继承 BaseException，必须原样上抛（取消不是"LLM 失败"，
            # 把它降级成兜底摘要会让取消语义失效）。
            raise
        except Exception as exc:  # noqa: BLE001 - 见 docstring：摘要永不抛
            warnings.warn(
                f"summary LLM call failed ({exc!r}); using extractive fallback summary",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fallback_extractive_summary(messages)

        # 不假设 response 一定是 LLMResponse（测试常用鸭子类型的假 LLM）；
        # 拿不到非空文本就等价于"调用失败"，走兜底而不是返回空串。
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            warnings.warn(
                "summary LLM returned empty content; using extractive fallback summary",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fallback_extractive_summary(messages)
        return content.strip()

    def fallback_extractive_summary(self, messages: Sequence[Message]) -> str:
        """抽取式兜底（零 LLM）：逐条取 `'[role] ' + content 前 80 字符`，按行拼接，
        最后由 stats 行收尾：`'(N messages summarized)'`。截断到 `max_summary_chars`。

        为什么保留这个兜底而不是"没 LLM 就返回空"：摘要的**存在性**比质量更重要 ——
        窗口里的消息已经被裁掉了，返回空等于那段历史彻底消失；抽取式至少留下
        每条消息的开头，模型仍能看出"发生过什么"。
        """
        msgs = list(messages)
        lines: list[str] = []
        for message in msgs:
            # 用 `.value` 而不是 f"{message.role}"：str-mixin 枚举的 __format__ 在不同
            # Python 版本下给出 "Role.USER" 或 "user"，显式取值才能跨版本稳定。
            role = message.role.value if isinstance(message.role, Role) else str(message.role)
            lines.append(f"[{role}] {message.content[:_FALLBACK_LINE_CHARS]}")
        lines.append(f"({len(msgs)} messages summarized)")
        text = "\n".join(lines)
        limit = self._char_limit()
        if limit > 0:
            text = truncate_head_tail(text, limit)
        return text

    # ---- 内部 ----

    def _char_limit(self) -> int:
        """摘要的字符上限。配置被写成非正数时用 FALLBACK_SUMMARY_MAX_CHARS 兜底 ——
        `truncate_head_tail` 对 `max_chars<=0` 是"原样返回"，直接传 0 等于关闭截断，
        会让摘要无限膨胀。"""
        configured = self.config.max_summary_chars
        return configured if configured > 0 else FALLBACK_SUMMARY_MAX_CHARS

    def _resolve_previous(self, previous_summary: str | None) -> str:
        """决定"旧摘要"是否参与本次合并。

        `update_existing=False` 是总开关：它表示"每轮从零重写摘要"，因此无论旧摘要来自
        显式参数还是本对象，都不并入 prompt（否则这个配置项在 MemoryManager 的路径上
        永远不生效 —— 管理器的契约是**总是**显式传 `previous_summary=self.summary`）。
        """
        if not self.config.update_existing:
            return ""
        previous = previous_summary if previous_summary is not None else self._summary
        return (previous or "").strip()

    def _build_prompt(self, messages: list[Message], *,
                      previous_summary: str | None) -> str:
        """渲染 `SUMMARY_PROMPT_TEMPLATE`。

        `render_template` 用 `format_map(_SafeDict)`，因此 transcript 里的 `{}`（JSON
        工具参数里到处都是）不会被二次解析 —— 这是模板渲染最经典的坑。
        """
        previous = self._resolve_previous(previous_summary)
        if previous:
            # §8.4 冻结的渲染格式（含结尾的两个换行）。
            previous_block = f"Existing summary so far (merge, do not repeat):\n{previous}\n\n"
        else:
            previous_block = ""

        transcript = render_transcript(messages)
        max_input = self.config.max_input_chars
        if max_input > 0:
            # 头多尾少：头部通常是任务目标与约束，尾部是刚发生的事（§5.2 的同一考虑）。
            transcript = truncate_head_tail(transcript, max_input)

        return render_template(SUMMARY_PROMPT_TEMPLATE, {
            "max_chars": self._char_limit(),
            "previous_block": previous_block,
            "transcript": transcript,
        })

    def _truncate(self, text: str) -> str:
        """超长截断（§8.4）。头尾都保留，避免把结论行切掉。"""
        limit = self._char_limit()
        if limit > 0:
            return truncate_head_tail(text, limit)
        return text
