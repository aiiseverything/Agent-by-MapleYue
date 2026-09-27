from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from liteagent import errors as _errors

# §9.1 冻结：AgentConfig 的唯一归属地是 config.py，本模块只做 re-export
# （agent/__init__.py 的 __all__ 里有 "AgentConfig"，它从这里或 config 取都行）。
from liteagent.config import AgentConfig  # noqa: F401  (re-export)
from liteagent.config import to_jsonable, utc_now
from liteagent.errors import AgentError, LiteAgentError, SerializationError
from liteagent.llm.message import Message, Role, drop_orphan_tool_messages
from liteagent.types import TokenUsage, ToolCall, ToolResult

__all__ = [
    "AgentStatus",
    "AgentState",
    "AgentResult",
    "AgentConfig",  # re-export（见上方说明）
    "TERMINAL_STATUSES",
]

_LOGGER = logging.getLogger("liteagent.agent")


class AgentStatus(str, Enum):
    """一次 run 的状态机取值（§9.1 冻结，成员名与值逐字一致）。"""

    IDLE = "IDLE"
    THINKING = "THINKING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

    @classmethod
    def coerce(cls, value: "AgentStatus | str") -> "AgentStatus":
        """把字符串转成成员；未知值抛 ``SerializationError``。

        规范只冻结了 ``Role.coerce`` 与 ``EventType.coerce``，没给 ``AgentStatus.coerce``，
        但 ``from_dict`` 必须把 ``"FAILED"`` 变回成员 —— 与其在反序列化处写一段只会用一次的
        私有转换，不如把这条规则挂在类型自己身上（`Role`/`EventType` 已是同一形状）。
        大小写不敏感地接受成员名与成员值（两者在本枚举里恰好同形，但规则写清楚）。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.strip()
            for candidate in (text, text.upper(), text.lower()):
                try:
                    return cls(candidate)
                except ValueError:
                    continue
            try:
                return cls[text.upper()]
            except KeyError:
                pass
        valid = ", ".join(member.value for member in cls)
        raise SerializationError(
            target="AgentStatus",
            message=f"invalid AgentStatus value {value!r}; expected one of: {valid}",
        )


#: 终态（§9.1 的 `mark_finished` 幂等规则依赖它）。集合语义而非 list：查一次是 O(1)。
TERMINAL_STATUSES = frozenset(
    {AgentStatus.FINISHED, AgentStatus.FAILED, AgentStatus.ABORTED}
)


def _new_run_id() -> str:
    """`'run_' + uuid4().hex[:12]`（§9.1 冻结的字面格式，测试只断言前缀）。"""
    return "run_" + uuid.uuid4().hex[:12]


@dataclass
class AgentState:
    """单次 run 的全部状态（术语表定义的 "state"，§0.5）。

    只有 `run_id` 没有默认值：状态必须能回答"这是哪一次 run"，让它在构造时就被显式决定
    （`AgentState.create()` 负责生成），比给个空串默认值更容易在 trace 里定位问题。
    为了省内存与避免 `default_factory` 的坑，本类**不使用** `slots=True`（§4 的决策 D-01）。
    """

    run_id: str
    input: str = ""
    agent_name: str = "agent"
    #: 未压缩的完整 transcript（含被 trim_transcript 裁掉的历史；被裁掉的消息仍活在 trace 里）
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    status: AgentStatus = AgentStatus.IDLE
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    action_counts: dict[str, int] = field(default_factory=dict)  # canonical_key -> 次数
    # [v2 新增] 循环防护的第二/三层（D-16）
    tool_name_counts: dict[str, int] = field(default_factory=dict)  # 只按 name 计数
    observation_digests: dict[str, int] = field(default_factory=dict)  # blake2b 摘要 -> 次数
    tool_failure_counts: dict[str, int] = field(default_factory=dict)  # infrastructure 失败次数
    parse_errors: int = 0
    llm_errors: int = 0
    truncation_errors: int = 0  # [v2] length 截断续写次数
    nudges: list[str] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    error: LiteAgentError | None = None
    started_at: float = field(default_factory=utc_now)  # [v2] 唯一时钟 config.utc_now
    finished_at: float | None = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def create(
        cls, input: str, *, agent_name: str = "agent", run_id: str | None = None
    ) -> "AgentState":
        """造一个 IDLE 状态；``run_id`` 默认 ``'run_' + uuid4().hex[:12]``。

        `agent.py` 在注入外部 state 时**必须保留调用方的 run_id**（§9.4），
        所以 `run_id` 是可选注入而不是每次生成。
        """
        return cls(
            run_id=run_id if run_id else _new_run_id(),
            input=input,
            agent_name=agent_name,
        )

    # ------------------------------------------------------- transcript 记账
    def add_message(self, message: Message) -> Message:
        """追加到 ``messages`` 并返回它（便于 `state.add_message(resp.to_message())` 链式用法）。

        上限**不在**这里管：由 ``Agent`` 在每步末尾统一调
        ``trim_transcript(config.max_transcript_messages)``（§9.4.1 步骤 7）。
        把裁剪集中在唯一一处，是为了让"窗口裁剪"与"工具对修复"永远成对发生。
        """
        self.messages.append(message)
        return message

    def trim_transcript(self, max_messages: int) -> int:
        """保留前导 SYSTEM 消息 + 最近 ``max_messages`` 条，返回**实际丢弃**的条数。

        两步都做工具对修复（§9.1 要求"先修工具对"，但只在裁剪**之后**修一次是不够的：
        裁剪点本身可能把 assistant 与其 tool 结果切开，所以：
        1) 先对整个 transcript 修一次（一般 transcript 是健康的，这一步通常是恒等变换）；
        2) 再切出 `SYSTEM 前缀 + 尾部 max_messages 条`；
        3) 对切出来的片段**再修一次**，把切点造成的孤儿 tool 消息 / 无结果 tool_calls 修掉。
        这两次修复都是幂等的，多修一次的代价远小于把非法消息对发给 provider（API 会 400）。

        被丢弃的消息**仍然保留在 trace 事件里**（事件在 emit 时已经发出，与 state 无关）。
        """
        if max_messages < 0:
            # 负数没有语义，按"不保留"处理（比静默当成 0 更不容易掩盖调用方的笔误）。
            _LOGGER.warning(
                "trim_transcript got negative max_messages=%s; clamping to 0", max_messages
            )
            max_messages = 0

        original_len = len(self.messages)
        repaired = drop_orphan_tool_messages(self.messages)

        head_len = 0
        while head_len < len(repaired) and repaired[head_len].role == Role.SYSTEM:
            head_len += 1

        tail = repaired[head_len:]
        if max_messages <= 0:
            # 边界：`tail[-0:]` 会退化成 `tail[0:]`（把整段都留下），必须显式处理。
            kept = repaired[:head_len]
        elif len(tail) <= max_messages:
            kept = repaired  # 不需要裁剪；但如果第 1 步修过东西，把修复结果落盘
        else:
            kept = repaired[:head_len] + tail[-max_messages:]
            kept = drop_orphan_tool_messages(kept)

        self.messages[:] = kept  # 原地替换：外部持有的 list 引用（如 Agent 的局部变量）保持有效
        dropped = original_len - len(self.messages)
        if dropped:
            # 裁剪是"降级"的一种（§13 红线 12）：窗口变短会让模型看不到早期历史，
            # 必须在日志里留痕。事件 CONTEXT_TRUNCATED 的发射者是 MemoryManager（§2.7），
            # 状态层不得越权发事件，所以这里只记日志。
            _LOGGER.debug(
                "trim_transcript dropped %d message(s): %d -> %d (max_messages=%d)",
                dropped,
                original_len,
                len(self.messages),
                max_messages,
            )
        return dropped

    # ------------------------------------------------------ 循环防护计数器
    def record_tool_call(self, call: ToolCall) -> int:
        """记一次工具调用请求，返回 ``canonical_key`` 的**累计**次数。

        冻结顺序（§9.1）：先 `action_counts[canonical_key]`，再 `tool_name_counts[name]`。
        两个计数器并存是有意的：`action_counts` 精确到参数（同参数重试），
        `tool_name_counts` 只看名字（同一工具换参数刷屏）。
        """
        key = call.canonical_key()
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        self.tool_name_counts[call.name] = self.tool_name_counts.get(call.name, 0) + 1
        return self.action_counts[key]

    def repeat_count(self, call: ToolCall) -> int:
        """该 canonical_key 已经出现过几次（0 表示没记过）。"""
        return self.action_counts.get(call.canonical_key(), 0)

    def record_observation(self, result: ToolResult) -> int:
        """记一次观察结果，返回该内容摘要出现的次数（无进展检测用，§9.4.1 步骤 4）。

        用 blake2b(digest_size=8) 而不是 `content` 本身：观察文本可能上万字符、
        且基本不重复，按内容存 dict key 会让 state 的内存随步数线性膨胀。
        8 字节摘要（64 bit）在"一次 run 内几十条观察"的规模下碰撞概率可以忽略。

        注意：按 §9.1 的口径摘要的是 `result.content` 而**不是** `error_text()`；
        失败结果的 content 由 `ToolResult.failure` 保证非空（§4.3），所以失败也能被计数。
        """
        text = result.content or ""
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        self.observation_digests[digest] = self.observation_digests.get(digest, 0) + 1
        return self.observation_digests[digest]

    def record_tool_failure(self, name: str) -> int:
        """记一次工具失败，返回该工具累计失败次数（熔断判定用）。"""
        self.tool_failure_counts[name] = self.tool_failure_counts.get(name, 0) + 1
        return self.tool_failure_counts[name]

    # ------------------------------------------------------------ 查询
    def last_message(self) -> Message | None:
        """transcript 的最后一条消息；空 transcript 返回 ``None``（不去发明一条空消息）。"""
        return self.messages[-1] if self.messages else None

    def last_tool_results(self, n: int = 1) -> list[ToolResult]:
        """最近 ``n`` 条工具结果（用于 nudge 文本里回放"你上一次拿到的结果"）。

        `n <= 0` 返回空列表：调用方拿它当"要几条"，负数没有意义。
        """
        if n <= 0:
            return []
        return list(self.tool_results[-n:])

    # ------------------------------------------------------------ 状态机
    def mark_finished(
        self, status: AgentStatus, *, error: LiteAgentError | None = None
    ) -> None:
        """设置 status/error，并把 ``finished_at`` 置为当前时间（唯一时钟）。

        **幂等**：已经是终态（FINISHED/FAILED/ABORTED）时只补 `finished_at`，
        不改写 `status` 与 `error`。理由：`arun` 的 `finally` 里还有一次补救调用，
        若它能把 FAILED 改回 FINISHED，"失败却 ok=True" 就会真的发生（§9.1 的冻结说明）。
        """
        if self.status in TERMINAL_STATUSES:
            if self.finished_at is None:
                self.finished_at = utc_now()
            return
        self.status = status
        if error is not None:
            self.error = error
        self.finished_at = utc_now()

    # ------------------------------------------------------------ 派生
    def clone(self) -> "AgentState":
        """给 multiagent 传状态的浅拷贝：容器换新对象，Message 对象本身共享。

        "浅拷贝容器、深拷贝 messages 列表"的含义是**列表对象**新建
        （`list(self.messages)`），而不是深拷贝 Message 内部 —— Message 在本项目里
        按不可变对象使用（`.copy(**changes)` 才产生新对象），共享它们是安全的，
        也让 BufferMemory 的 `m not in window` 这类相等性判断继续成立。
        `usage` 单独复制：它是 `+=` 就地累加的可变对象，共享会让父子 Agent 互相污染预算。

        `run_id` **保持不变**：clone 的语义是"同一次 run 的状态副本"
        （子 Agent 需要新 run_id 时会自己调 `AgentState.create`）。
        """
        return AgentState(
            run_id=self.run_id,
            input=self.input,
            agent_name=self.agent_name,
            messages=list(self.messages),
            step=self.step,
            status=self.status,
            tool_calls=list(self.tool_calls),
            tool_results=list(self.tool_results),
            usage=TokenUsage(
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
                total_tokens=self.usage.total_tokens,
                model_hint=self.usage.model_hint,
            ),
            action_counts=dict(self.action_counts),
            tool_name_counts=dict(self.tool_name_counts),
            observation_digests=dict(self.observation_digests),
            tool_failure_counts=dict(self.tool_failure_counts),
            parse_errors=self.parse_errors,
            llm_errors=self.llm_errors,
            truncation_errors=self.truncation_errors,
            nudges=list(self.nudges),
            scratchpad=dict(self.scratchpad),
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    @property
    def duration_ms(self) -> float:
        """从 ``started_at`` 到 ``finished_at``（未结束时到"现在"）的毫秒数。

        永远 >= 0：`frozen_time` 之类的测试夹具可能把时钟往回拨，
        负的耗时在日志里只会让人以为出了 bug。
        """
        end = self.finished_at if self.finished_at is not None else utc_now()
        return max(0.0, (end - self.started_at) * 1000.0)

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        """字段**全量**输出（§2.2，含 None），保证 `assertEqual` 可以精确比对。

        嵌套对象用各自的 `to_dict()`（而不是让 `to_jsonable` 反射 dataclass）：
        `TokenUsage.to_dict()` 只输出 3 个计数字段，而 `to_jsonable` 会把
        `model_hint` 也带出来 —— 那会污染 trace 的稳定结构（§4.1）。
        `scratchpad` 例外：里面放的是任意对象（如 DelegationContext），必须过 `to_jsonable`。
        """
        return {
            "run_id": self.run_id,
            "input": self.input,
            "agent_name": self.agent_name,
            "messages": [m.to_dict() for m in self.messages],
            "step": self.step,
            "status": self.status.value,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "usage": self.usage.to_dict(),
            "action_counts": dict(self.action_counts),
            "tool_name_counts": dict(self.tool_name_counts),
            "observation_digests": dict(self.observation_digests),
            "tool_failure_counts": dict(self.tool_failure_counts),
            "parse_errors": self.parse_errors,
            "llm_errors": self.llm_errors,
            "truncation_errors": self.truncation_errors,
            "nudges": list(self.nudges),
            "scratchpad": to_jsonable(dict(self.scratchpad)),
            "error": None if self.error is None else self.error.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentState":
        """`to_dict()` 的逆操作；缺字段一律回落到该字段的默认值（§2.2 容忍原则）。

        结构错（不是 Mapping、``messages`` 不是列表）抛 ``SerializationError``；
        单条消息解析失败让 `Message.from_dict` 自己的异常透出（同样是 ``SerializationError``，
        带上 "Message.字段名" 的 target，比在这里吞掉更有用）。
        """
        if not isinstance(data, Mapping):
            raise SerializationError(
                target="AgentState", message=f"expected a mapping, got {type(data).__name__}"
            )

        raw_messages = data.get("messages") or []
        if not isinstance(raw_messages, list):
            raise SerializationError(target="AgentState.messages", message="expected a list")

        raw_calls = data.get("tool_calls") or []
        raw_results = data.get("tool_results") or []
        if not isinstance(raw_calls, list):
            raise SerializationError(target="AgentState.tool_calls", message="expected a list")
        if not isinstance(raw_results, list):
            raise SerializationError(target="AgentState.tool_results", message="expected a list")

        usage_raw = data.get("usage")
        usage = (
            TokenUsage.from_dict(usage_raw)
            if isinstance(usage_raw, Mapping)
            else TokenUsage()
        )

        return cls(
            run_id=str(data.get("run_id") or ""),
            input=str(data.get("input") or ""),
            agent_name=str(data.get("agent_name") or "agent"),
            messages=[_message_from_dict(m) for m in raw_messages],
            step=int(data.get("step") or 0),
            status=AgentStatus.coerce(data.get("status", AgentStatus.IDLE)),
            tool_calls=[ToolCall.from_dict(c) for c in raw_calls],
            tool_results=[ToolResult.from_dict(r) for r in raw_results],
            usage=usage,
            action_counts=_int_map(data.get("action_counts")),
            tool_name_counts=_int_map(data.get("tool_name_counts")),
            observation_digests=_int_map(data.get("observation_digests")),
            tool_failure_counts=_int_map(data.get("tool_failure_counts")),
            parse_errors=int(data.get("parse_errors") or 0),
            llm_errors=int(data.get("llm_errors") or 0),
            truncation_errors=int(data.get("truncation_errors") or 0),
            nudges=[str(x) for x in (data.get("nudges") or [])],
            scratchpad=dict(data.get("scratchpad") or {}),
            error=_error_from_dict(data.get("error")),
            started_at=float(data.get("started_at") or utc_now()),
            finished_at=_optional_float(data.get("finished_at")),
        )


@dataclass
class AgentResult:
    """一次 run 的结果（对外返回物，`Agent.arun` 的唯一产物）。

    **字段顺序是冻结的：`output` 第一、`status` 第二**，因此禁止位置参数构造
    （`AgentResult(FAILED, error=e)` 会把状态塞进 output，得到"失败却 ok=True"的结果）。
    所有构造点必须写关键字参数；`Agent` 内部统一走 `Agent._result(...)`（§9.1）。
    """

    output: str = ""
    status: AgentStatus = AgentStatus.FINISHED
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: LiteAgentError | None = None
    state: AgentState | None = None
    duration_ms: float = 0.0
    agent_name: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """只有 FINISHED 算成功；FAILED/ABORTED/其它一律 False（保守判定）。"""
        return self.status == AgentStatus.FINISHED

    def to_dict(self, *, include_state: bool = False) -> dict[str, Any]:
        """字段全量输出；``state`` 是三个显式 opt-out 之一（§2.2）。

        与 `MemoryItem.to_dict(include_embedding=False)` 一致：**关闭时键不出现**，
        而不是出现一个 `None` —— opt-out 的意义就是别让几百条消息把 JSON 撑爆。
        """
        out: dict[str, Any] = {
            "output": self.output,
            "status": self.status.value,
            "steps": self.steps,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_results": [r.to_dict() for r in self.tool_results],
            "usage": self.usage.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
            "duration_ms": self.duration_ms,
            "agent_name": self.agent_name,
            "metadata": to_jsonable(dict(self.metadata)),
        }
        if include_state:
            out["state"] = None if self.state is None else self.state.to_dict()
        return out

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        *,
        output: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentResult":
        """从 state 快照构造结果（**全关键字**，见类 docstring）。

        `status` 直接取 `state.status`：调用方在调本方法前应该已经
        `state.mark_finished(...)`，这样"结果里的状态"与"状态机里的状态"不会出现两个版本。
        """
        return cls(
            output=output,
            status=state.status,
            steps=state.step,
            tool_calls=list(state.tool_calls),
            tool_results=list(state.tool_results),
            usage=state.usage,
            error=state.error,
            state=state,
            duration_ms=state.duration_ms,
            agent_name=state.agent_name,
            metadata=dict(metadata or {}),
        )

    def raise_for_status(self) -> None:
        """非 FINISHED 时抛错：优先抛携带根因的 ``self.error``。

        没有 error 对象也不能静默返回（那等于把失败当成功），
        此时自己造一个 ``AgentError`` —— 与 `Agent._result` 的处理一致。
        """
        if self.status == AgentStatus.FINISHED:
            return
        if self.error is not None:
            raise self.error
        raise AgentError(
            f"agent {self.agent_name} finished with status {self.status.value} "
            "but carries no error object"
        )


# --------------------------------------------------------------------------------------
# from_dict 的内部辅助（模块级私有函数：不污染类的公开面）
# --------------------------------------------------------------------------------------


def _message_from_dict(raw: Any) -> Message:
    """单条消息反序列化 + 类型校验（错误信息集中在一处，`from_dict` 里保持一行）。"""
    if not isinstance(raw, Mapping):
        raise SerializationError(
            target="AgentState.messages", message=f"expected a mapping, got {type(raw).__name__}"
        )
    return Message.from_dict(raw)


def _int_map(raw: Any) -> dict[str, int]:
    """把 ``{"k": 3}`` 形态的计数表读回来；非 Mapping 或非数值一律当空表并留 WARNING。

    这里选择"降级 + 警告"而不是抛错：计数器是**防护性**数据（重复检测用），
    丢了一部分只会让防护变松，不至于让整个 run 的结果无法还原；
    但静默丢就是 §13 红线 12 说的隐性降级，所以必须留痕。
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _LOGGER.warning("AgentState.from_dict: expected a mapping for counters, got %s", type(raw))
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            _LOGGER.warning("AgentState.from_dict: dropping non-numeric counter %r=%r", key, value)
    return out


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        _LOGGER.warning("AgentState.from_dict: ignoring non-numeric finished_at=%r", raw)
        return None


def _error_from_dict(raw: Any) -> LiteAgentError | None:
    """把 ``error.to_dict()`` 还原成异常对象（同名类 + message + context）。

    为什么值得做：state 往返后 `error_type` 是 trace/结果里最有诊断价值的字段，
    统一降级成 `LiteAgentError` 会让"这次是 MaxStepsExceeded 还是 RepeatedAction"消失。
    子类额外字段（如 `max_steps`）**不还原**：它们的构造参数各不相同（§3.3 的字段表），
    而 message 里已经带了同样的信息（`_default_message` 会把字段写进 message）。
    还原失败（类型名不在 errors 里）时降级为 `LiteAgentError` 并记 WARNING。
    """
    if raw is None:
        return None
    if isinstance(raw, LiteAgentError):  # 容忍调用方直接塞异常对象
        return raw
    if not isinstance(raw, Mapping):
        _LOGGER.warning("AgentState.from_dict: expected a mapping for error, got %s", type(raw))
        return None

    message = str(raw.get("message") or "")
    context = raw.get("context") if isinstance(raw.get("context"), Mapping) else None
    type_name = raw.get("type")

    candidate: Any = None
    if isinstance(type_name, str) and type_name:
        candidate = getattr(_errors, type_name, None)
    if isinstance(candidate, type) and issubclass(candidate, LiteAgentError):
        try:
            return candidate(message=message, context=dict(context) if context else None)
        except TypeError as exc:  # pragma: no cover - 防御：未来的子类改了构造签名
            _LOGGER.warning(
                "AgentState.from_dict: cannot rebuild %s (%s); degrading to LiteAgentError",
                type_name,
                exc,
            )
    elif type_name:
        _LOGGER.warning(
            "AgentState.from_dict: unknown error type %r; degrading to LiteAgentError", type_name
        )
    return LiteAgentError(message=message, context=dict(context) if context else None)
