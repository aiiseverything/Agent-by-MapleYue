from __future__ import annotations

# 消息模型与消息级纯函数（§6.1）。
#
# 设计要点（为什么这样写）：
# - `Role` 用 `class Role(str, Enum)` 而不是 `enum.StrEnum`：3.10 没有 `StrEnum`（§0.1），
#   而 str-mixin 让 `Role.USER == "user"` 成立，消息可以直接与 provider 的 JSON 字符串比对。
# - `Message` 是**可变** dataclass 且**不用** `slots=True`：它是 ReAct 循环里创建/替换最频繁的
#   对象，且带 `dict` 默认值（§4 决策 D-01 明确要求核心结构用 stdlib dataclass）。
# - 本模块只依赖 `errors` 与 `types`（§1.1 的 L2 边），**不得** import `config` ——
#   因此 token 估算的除数只能写成字面量 `4`（同步位置：§2.5 的 `DEFAULT_TOKEN_CHAR_RATIO`，
#   两者都是"ASCII 4 字符 1 token"语义）。
#
# 为什么头部写注释而不是模块 docstring：§2.1 冻结"每个 .py 文件**第一行**必须是
# `from __future__ import annotations`"，而 docstring 只有作为第一条语句时才有效，
# 两者不可兼得 —— 规范优先。

import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from liteagent.errors import SerializationError
from liteagent.types import ToolCall, ToolResult

# 异常构造约定（与 liteagent/errors.py 逐字对齐，别写反）：
#   `errors.py` 把 §3.3 表里的**额外字段排在 `message` 之前**（位置或关键字皆可），
#   而 `message` 是 keyword-only：`SerializationError(target=..., message=..., cause=...)`。
#   因此本模块一律**全关键字**构造异常 —— 对"message 在前/在后"两种约定都成立，
#   也不依赖参数顺序（`SerializationError("Role")` 会把 "Role" 当成 target 而不是 message）。


#: `text()` / `render_transcript` 渲染 tool_call 参数用的 json 参数。
#: 与 `ToolCall.canonical_key()` 的 json 参数保持一致（sort_keys + ensure_ascii=False），
#: 否则同一份 arguments 会在"重复检测键"与"渲染文本"里出现两种写法。
_CANONICAL_JSON_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "default": str,
}

#: ASCII 近似的除数（= config.DEFAULT_TOKEN_CHAR_RATIO，见模块 docstring 的说明）。
_ASCII_CHARS_PER_TOKEN = 4


class Role(str, Enum):
    """消息角色。取值与 OpenAI/Anthropic 的 role 字符串逐字一致。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def coerce(cls, value: "Role | str") -> "Role":
        """接受 `'system'` / `'System'` / `Role.SYSTEM`；未知值抛 `SerializationError`。

        大小写不敏感是有意为之：不同 provider 的回流数据里出现过 `"User"`（首字母大写），
        这种输入必须被归一化而不是让整条 trace 崩在 `Role("User")` 上。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError as exc:
                raise SerializationError(
                    target="Role",
                    message=f"unknown role {value!r}",
                    cause=exc,
                ) from exc
        raise SerializationError(
            target="Role",
            message=f"role must be Role or str, got {type(value).__name__}",
        )


@dataclass
class Message:
    """一条对话消息。字段与 `to_dict()` 的六个键一一对应（§6.1）。"""

    role: Role = Role.USER
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- 构造便捷方法（全部为 classmethod，返回 Message）----

    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message":
        """系统提示。`**metadata` 直接进 `metadata` 字段（§2.4 的预留 key 由此写入）。"""
        return cls(role=Role.SYSTEM, content=content, metadata=dict(metadata))

    @classmethod
    def user(cls, content: str, **metadata: Any) -> "Message":
        """用户输入（文本 ReAct 的 Observation 也走这里，见 `observation`）。"""
        return cls(role=Role.USER, content=content, metadata=dict(metadata))

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Sequence[ToolCall] | None = None,
        **metadata: Any,
    ) -> "Message":
        """模型输出。`tool_calls=None` 归一化为 `[]`（不是 `None`），
        保证 `bool(msg.tool_calls)` / 序列化两条路径行为一致。"""
        calls = list(tool_calls) if tool_calls else []
        return cls(role=Role.ASSISTANT, content=content, tool_calls=calls, metadata=dict(metadata))

    @classmethod
    def tool(cls, result: "ToolResult") -> "Message":
        """等价 `result.to_message()`。保留这一层是为了让调用方读起来像"造一条 tool 消息"。"""
        return result.to_message()

    @classmethod
    def observation(
        cls,
        content: str,
        *,
        step: int | None = None,
        tool_name: str | None = None,
    ) -> "Message":
        """文本 ReAct 模式的注释消息：`role=USER`，`metadata['kind']='observation'`。

        为什么要 `role=USER`：文本 ReAct 没有原生 tool 角色，Observation 必须以"用户侧输入"
        的形式回灌，这是与多数 chat API 兼容的唯一做法（§6.1 冻结理由）。
        `step` / `tool_name` 只在非 None 时写入 —— 缺省不出现在 metadata 里，
        避免 trace 里出现 `"step": null` 这种噪声（§2.4 的键语义要求）。
        """
        metadata: dict[str, Any] = {"kind": "observation"}
        if step is not None:
            metadata["step"] = step
        if tool_name is not None:
            metadata["tool_name"] = tool_name
        return cls(role=Role.USER, content=content, metadata=metadata)

    # ---- 变换 ----

    def copy(self, **changes: Any) -> "Message":
        """`dataclasses.replace` 的薄包装：不可变式修改，避免调用方就地改别人的消息。"""
        return replace(self, **changes)

    def text(self) -> str:
        """用于 token 估算与摘要：`content` + 每个 tool_call 的 `'name(canonical_args)'`。

        为什么不是简单拼接：tool_call 是模型可见的"思考内容"的一部分，
        只数 `content` 会低估 assistant 消息的 token 占用。
        没有 content 也没有 tool_calls 时返回 `""`（调用方据此算 0 token）。
        """
        parts: list[str] = []
        if self.content:
            parts.append(self.content)
        for call in self.tool_calls:
            parts.append(f"{call.name}({_canonical_arguments(call.arguments)})")
        return "\n".join(parts)

    def is_tool_pair_start(self) -> bool:
        """是否是"工具对"的前半：assistant 且带 tool_calls。"""
        return self.role == Role.ASSISTANT and bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """`{"role","content","name","tool_calls","tool_call_id","metadata"}`。

        role 输出 `.value` 字符串；tool_calls 为 `list[dict]`；**不用 None 省略字段**
        （§2.2 字段全量输出，测试要能 `assertEqual` 精确比对）。
        role 走 `Role.coerce` 是为了容忍"手工塞了字符串 role"的消息，
        未知值会抛 `SerializationError` 而不是产出非法 JSON。
        """
        return {
            "role": Role.coerce(self.role).value,
            "content": self.content,
            "name": self.name,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message":
        """反序列化。缺失的可选键用字段默认值补齐（§2.2：`from_dict` 要容忍缺失字段），
        但**类型错误**一律抛 `SerializationError`（§3.3：from_dict 收到错类型即抛）。"""
        if not isinstance(data, Mapping):
            raise SerializationError(
                target="Message",
                message=f"Message.from_dict expects a mapping, got {type(data).__name__}",
            )

        content = data.get("content", "")
        if content is None:
            # provider 的 tool_calls-only 响应会把 content 写成 null，这里归一化为 ""。
            content = ""
        if not isinstance(content, str):
            raise SerializationError(
                target="Message.content",
                message=f"Message.content must be str, got {type(content).__name__}",
            )

        raw_calls = data.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
            raise SerializationError(
                target="Message.tool_calls",
                message=f"Message.tool_calls must be a sequence, got {type(raw_calls).__name__}",
            )
        calls: list[ToolCall] = []
        for index, item in enumerate(raw_calls):
            if isinstance(item, ToolCall):
                calls.append(item)
            elif isinstance(item, Mapping):
                calls.append(ToolCall.from_dict(item))
            else:
                raise SerializationError(
                    target="Message.tool_calls",
                    message=f"Message.tool_calls[{index}] must be a mapping, got {type(item).__name__}",
                )

        metadata = data.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise SerializationError(
                target="Message.metadata",
                message=f"Message.metadata must be a mapping, got {type(metadata).__name__}",
            )

        name = data.get("name")
        tool_call_id = data.get("tool_call_id")
        return cls(
            role=Role.coerce(data.get("role", Role.USER)),
            content=content,
            name=None if name is None else str(name),
            tool_calls=calls,
            tool_call_id=None if tool_call_id is None else str(tool_call_id),
            metadata=dict(metadata),
        )


def _canonical_arguments(arguments: Mapping[str, Any]) -> str:
    """把 tool_call 的 arguments 渲染成稳定字符串（与 `ToolCall.canonical_key` 同参数）。"""
    return json.dumps(dict(arguments), **_CANONICAL_JSON_KWARGS)


def _ascii_estimate(text: str) -> int:
    """内置 ASCII 近似：`max(1, ceil(len(text)/4))`（仅用于展示，不用于预算判定）。"""
    return max(1, math.ceil(len(text) / _ASCII_CHARS_PER_TOKEN))


def messages_tokens(
    messages: Sequence[Message],
    counter: Callable[[str], int] | None = None,
) -> int:
    """估算一批消息的 token 数。

    **注意**：这里收的是 counter 函数而不是 Tokenizer 对象 —— llm 层不能 import memory 层
    （反向依赖）。memory 侧调用时传 `tokenizer.estimate` 即可。
    `counter` 为 None 时用内置的 ASCII 近似 `max(1, ceil(len(text)/4))`
    （仅用于展示，不用于预算判定）。

    [v2 变更] 空文本（`text() == ""` 的全部消息）返回 **0**，与 `HeuristicTokenizer("") == 0` 对齐；
    这里对"空文本"统一短路（两条分支都短路），保证默认近似与外部 counter 的结果口径一致。
    """
    total = 0
    for message in messages:
        text = message.text()
        if not text:
            continue
        total += counter(text) if counter is not None else _ascii_estimate(text)
    return total


def render_transcript(
    messages: Sequence[Message],
    *,
    include_tool_calls: bool = True,
) -> str:
    """把消息渲染成纯文本（给摘要器 / LLM 摘要 prompt 用）。

    格式：`'[role] content'`，tool 消息带 `'[tool:<name>]'`，assistant 的 tool_calls 渲染为
    `'  -> <name>(<canonical json>)'`（每行一个，缩进两格便于人读）。
    每行做 `rstrip`：content 为空时输出 `'[assistant]'` 而不是带尾随空格的行，
    否则 golden-string 断言与 diff 都会很难看。
    """
    lines: list[str] = []
    for message in messages:
        role = message.role.value if isinstance(message.role, Role) else str(message.role)
        if role == Role.TOOL.value:
            # tool 消息的 name 由 ToolResult.to_message 写入；手工构造的消息可能只有
            # metadata["tool_name"]（§2.4 的冗余键），再兜底 "unknown" 保证行格式可读。
            name = message.name or str(message.metadata.get("tool_name") or "unknown")
            prefix = f"[tool:{name}]"
        else:
            prefix = f"[{role}]"
        lines.append(f"{prefix} {message.content}".rstrip())
        if include_tool_calls and message.is_tool_pair_start():
            for call in message.tool_calls:
                lines.append(f"  -> {call.name}({_canonical_arguments(call.arguments)})")
    return "\n".join(lines)


def drop_orphan_tool_messages(messages: Sequence[Message]) -> list[Message]:
    """删除孤儿 tool 消息，并剥离没有结果的 assistant.tool_calls。

    1. `tool_call_id` 找不到对应 `assistant.tool_calls` 的 TOOL 消息 —— 整条删除；
    2. `assistant.tool_calls` 里没有匹配 TOOL 结果的那部分 —— 从 `tool_calls` 中剥离
       （消息本身保留，content 是模型可见的思考内容，不能一起丢）。

    作用：裁剪窗口后保持 OpenAI/Anthropic 的消息合法性（否则 API 会 400）。

    为什么这里是"修复"而不是"降级"：它是窗口裁剪流程的既定一步
    （§8.3 `window()` 第 5 步）而不是异常兜底，所以不额外发 warning；
    真正需要可观测性的是上游的裁剪事件（`CONTEXT_TRUNCATED`，§2.7）。
    未改动的消息**原对象返回**（不复制）：BufferMemory 的 `evicted = [m for m in msgs
    if m not in window]` 依赖 dataclass 相等性，尽量少造副本能减少"窗口里的对象不在 buffer 里"
    这类边界情况。
    """
    items = list(messages)

    declared: set[str] = set()
    for message in items:
        if message.is_tool_pair_start():
            for call in message.tool_calls:
                if call.id:
                    declared.add(call.id)
    resolved: set[str] = {
        message.tool_call_id
        for message in items
        if message.role == Role.TOOL and message.tool_call_id
    }

    result: list[Message] = []
    for message in items:
        if message.role == Role.TOOL:
            # tool_call_id 为空视为孤儿：没有 id 的 tool 消息在任何 API 里都非法。
            if message.tool_call_id and message.tool_call_id in declared:
                result.append(message)
            continue
        if message.is_tool_pair_start():
            kept = [call for call in message.tool_calls if call.id in resolved]
            if len(kept) != len(message.tool_calls):
                result.append(message.copy(tool_calls=kept))
                continue
        result.append(message)
    return result
