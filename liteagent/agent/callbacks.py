from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, Union

from liteagent.config import estimate_cost_usd, to_jsonable, utc_now
from liteagent.errors import ConfigError, SerializationError
from liteagent.llm.base import LowLevelEvent
from liteagent.types import TokenUsage

# rich 是**可选**依赖（§9.2 冻结：RichCallback 必须两种环境都能 import 与运行）。
# 按 §1.3 的写法放进 try/except —— AST 守门用例把 try 内的三方 import 视为可选依赖。
try:  # pragma: no cover - 环境相关
    import rich.console as _rich_console

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rich_console = None
    RICH_AVAILABLE = False

__all__ = [
    "EventType",
    "TraceEvent",
    "Callback",
    "CallbackLike",
    "CallbackManager",
    "FunctionCallback",
    "LoggingCallback",
    "JsonlTraceCallback",
    "RichCallback",
    "TokenCounterCallback",
    "MemoryTraceCallback",
    "TraceRecorder",
    "load_trace",
    "total_usage",
    "events_of_type",
    "render_trace",
    "trace_stats",
    "as_llm_callback",
    "LowLevelEvent",
    "RESERVED_KEYS",
    "RICH_AVAILABLE",
]

#: 本模块自己的日志器（§9.2 冻结的 logger 名）。
_LOGGER = logging.getLogger("liteagent.callbacks")
#: LoggingCallback 的默认 logger 名：与 `Agent.astream` 的告警出口保持一致，
#: `cli.py -v` 只要把根 logger 打开就能同时看到两者。
_AGENT_LOGGER_NAME = "liteagent.agent"

#: TraceEvent 的保留键（§9.2 冻结集合）。data 里禁止出现它们，
#: 否则 to_dict 平铺时会静默覆盖事件自身的元数据。
RESERVED_KEYS = frozenset({"type", "run_id", "agent_name", "step", "ts"})

#: 摘要里"裸露值"（不写成 k=v）的键：工具名是人读 trace 时第一眼要找的东西。
_BARE_KEYS = frozenset({"tool_name"})


class EventType(str, Enum):
    """trace 事件类型（v2 冻结，共 27 个成员；带 `[v2 新增]` 的三个不能漏）。

    事件的**发射者**由 §2.7 唯一指定：本模块只提供机制，任何内置回调都不得 emit。
    """

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    THOUGHT = "thought"
    ACTION_PARSED = "action_parsed"
    PARSE_ERROR = "parse_error"
    REPEAT_DETECTED = "repeat_detected"
    NUDGE = "nudge"
    TOOL_STARTED = "tool_started"
    TOOL_RETRY = "tool_retry"
    TOOL_FINISHED = "tool_finished"
    TOOL_ERROR = "tool_error"
    TOOL_APPROVAL = "tool_approval"  # [v2 新增] HITL 审批结果
    MEMORY_WRITE = "memory_write"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_COMPRESS = "memory_compress"
    BUDGET_EXCEEDED = "budget_exceeded"  # [v2 新增] token/成本预算超限
    CONTEXT_TRUNCATED = "context_truncated"  # [v2 新增] 上下文裁剪（降级可观测）
    AGENT_DELEGATE = "agent_delegate"
    AGENT_RETURN = "agent_return"
    BLACKBOARD_WRITE = "blackboard_write"
    BLACKBOARD_READ = "blackboard_read"

    @classmethod
    def coerce(cls, value: "EventType | str") -> "EventType":
        """接受成员本身、成员值（``"tool_finished"``）与成员名（``"TOOL_FINISHED"``）。

        未知字符串抛 ``ConfigError``（**不是** SerializationError：事件类型来自代码常量
        与低层字符串，出错属于"用错了常量/名字"，是配置级错误）。
        `as_llm_callback` 依赖这个异常来做"未知事件忽略"的判定。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.strip()
            try:
                return cls(text)
            except ValueError:
                pass
            try:
                return cls[text.upper()]
            except KeyError:
                pass
        valid = ", ".join(member.value for member in cls)
        raise ConfigError(f"unknown event type {value!r}; valid values: {valid}")


# --------------------------------------------------------------------------------------
# 事件
# --------------------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """一条 trace 事件。**构造即校验**（§9.2）—— 校验在构造点而不是序列化点，
    是为了让"什么时候错的"与"哪里错的"是同一个位置（traceback 直接指向调用方）。
    """

    type: EventType
    run_id: str = ""
    agent_name: str = ""
    step: int = 0
    timestamp: float = field(default_factory=utc_now)  # [v2] 唯一时钟
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 允许 `TraceEvent(type="tool_finished")`：字符串事件名在低层到处都是（§2.7），
        # 在这里统一 coerce 比要求每个调用点自己转更不容易漏（转换失败同样是 ConfigError）。
        if not isinstance(self.type, EventType):
            self.type = EventType.coerce(self.type)

        if not isinstance(self.data, Mapping):
            raise ConfigError(
                f"TraceEvent.data must be a mapping, got {type(self.data).__name__}"
            )
        reserved = RESERVED_KEYS.intersection(self.data)
        if reserved:
            # 冻结在这里抛（§9.2）：推迟到 to_dict 只能是"静默覆盖"，那会让 trace 里
            # 的 step 与事件构造时的 step 不一致，排查时极其难查。
            raise ConfigError(
                "TraceEvent.data must not contain reserved keys "
                f"{sorted(reserved)}; reserved keys are {sorted(RESERVED_KEYS)}"
            )
        # §2.7 的类型约束：data 里的值必须是 JSON 可序列化的。在**唯一入口**兜底，
        # 保证 to_json / JSONL 写入永不因类型抛 TypeError（低层可能忘记过 to_jsonable）。
        normalized = to_jsonable(dict(self.data))
        if isinstance(normalized, dict):
            self.data = normalized
        else:  # pragma: no cover - to_jsonable(dict) 必然返回 dict，纯防御
            _LOGGER.warning(
                "to_jsonable() returned %s for TraceEvent.data; keeping it empty",
                type(normalized).__name__,
            )
            self.data = {}

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        """``{"type": <value>, "run_id", "agent_name", "step", "ts", **data}``。

        **恒输出键名 ``ts``**（不是 timestamp）：JSONL 是给人 grep 的，短名字更省空间，
        也避免与 `ToolResult.duration_ms` 之类的字段在视觉上混淆。data 平铺到顶层，
        于是 `grep tool_finished trace.jsonl` 能直接看到工具名与耗时。
        """
        out: dict[str, Any] = {
            "type": self.type.value,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "step": self.step,
            "ts": self.timestamp,
        }
        out.update(self.data)  # data 已在 __post_init__ 里保证不含保留键，不会覆盖上面的键
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceEvent":
        """反序列化：同时接受**平铺**（to_dict 的形态）与**嵌套**（``data`` 字段）两种布局。

        - 平铺：``ts`` -> ``timestamp``；同时接受 ``timestamp`` 键，二者都在时**以 ts 为准**
          （to_dict 恒输出 ts，旧文件里可能残留 timestamp）。
        - 嵌套：``data`` 字段里的键整体进 ``.data``。
        - 其余非保留键一并进 ``.data``（平铺形态的剩余字段）。
        保留键集合固定为 {type, run_id, agent_name, step, ts}（+ timestamp 别名）。
        """
        if not isinstance(data, Mapping):
            raise SerializationError(
                target="TraceEvent",
                message=f"expected a mapping, got {type(data).__name__}",
            )
        if "type" not in data or data.get("type") is None:
            raise SerializationError(target="TraceEvent.type", message="missing required field")

        payload: dict[str, Any] = {}
        nested = data.get("data")
        if isinstance(nested, Mapping):
            payload.update(nested)

        ts = data.get("ts")
        if ts is None:
            ts = data.get("timestamp")

        for key, value in data.items():
            if key in RESERVED_KEYS or key in ("data", "timestamp"):
                continue
            payload[key] = value  # 平铺键后写：与 to_dict 的"data 覆盖在最后"顺序相反地生效

        step_raw = data.get("step", 0)
        try:
            step = int(step_raw or 0)
        except (TypeError, ValueError):
            _LOGGER.warning("TraceEvent.from_dict: non-numeric step=%r; using 0", step_raw)
            step = 0

        return cls(
            type=EventType.coerce(data.get("type")),
            run_id=str(data.get("run_id") or ""),
            agent_name=str(data.get("agent_name") or ""),
            step=step,
            timestamp=float(ts) if isinstance(ts, (int, float)) else utc_now(),
            data=payload,
        )

    def to_json(self) -> str:
        """单行 JSON（`ensure_ascii=False`，中文 debug 信息保持可读）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "TraceEvent":
        """从一行 JSON 还原；解析失败时抛 ``SerializationError``（load_trace 会跳过它）。"""
        try:
            raw = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                target="TraceEvent", message=f"invalid JSON line: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise SerializationError(
                target="TraceEvent", message=f"expected a JSON object, got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    # ------------------------------------------------------------ 展示
    def summary(self) -> str:
        """单行人类可读摘要（CLI `trace` 与 LoggingCallback 共用）。

        冻结样例：``'[3] tool_finished read_file ok=True 12.3ms'`` —— 即
        ``[step] <type> <裸工具名> <k=v ...>``，其中 ``*_ms`` 字段带单位、工具名裸写。
        未登记的类型把 data 全部按键名排序铺开，保证新增事件类型也能被看到（可观测优先）。
        """
        parts = [f"[{self.step}]", self.type.value]
        known = _SUMMARY_KEYS.get(self.type.value, ())
        for key in known:
            if key in self.data:
                parts.append(_format_field(key, self.data[key]))
        for key in sorted(self.data):
            if key not in known:
                parts.append(_format_field(key, self.data[key]))
        return " ".join(parts)


#: 各事件类型在 summary 里优先展示的 data 键（顺序即展示顺序）。
#: 没登记的键不会被丢掉，只是在排序后追加（见 `summary`）。
_SUMMARY_KEYS: dict[str, tuple[str, ...]] = {
    "run_started": ("mode", "input"),
    "run_finished": ("steps", "output_len"),
    "run_failed": ("error_type", "aborted", "message"),
    "step_finished": ("status",),
    "llm_request": ("model", "messages_count", "retry"),
    "llm_response": ("finish_reason", "tool_calls", "latency_ms"),
    "llm_error": ("error_type", "retry"),
    "thought": ("text",),
    "action_parsed": ("action", "arguments"),
    "parse_error": ("reason", "attempt"),
    "repeat_detected": ("action_key", "count"),
    "nudge": ("text",),
    "tool_started": ("tool_name", "attempt"),
    "tool_retry": ("tool_name", "attempt", "delay_s"),
    # 与 §9.2 的样例逐字对齐：tool_name 裸写、ok=True、12.3ms
    "tool_finished": ("tool_name", "ok", "duration_ms"),
    "tool_error": ("tool_name", "error_type", "disabled"),
    "tool_approval": ("tool_name", "approved"),
    "memory_write": ("kind", "count"),
    "memory_retrieve": ("count", "query_len"),
    "memory_compress": ("before_tokens", "after_tokens"),
    "budget_exceeded": ("kind", "limit", "used"),
    "context_truncated": ("before", "after", "dropped_messages"),
    "agent_delegate": ("to", "depth", "refused"),
    "agent_return": ("to", "status", "failed"),
    "blackboard_write": ("key", "version"),
    "blackboard_read": ("key", "hit"),
}


def _format_field(key: str, value: Any) -> str:
    """summary 里单个 data 键的渲染规则（耗时带 ms 单位、工具名裸写、长文本折叠）。"""
    if key in ("duration_ms", "latency_ms") and isinstance(value, (int, float)):
        return f"{float(value):.1f}ms"
    if isinstance(value, bool):  # 必须在 int 之前判：bool 是 int 的子类
        return f"{key}={value}"
    if key in _BARE_KEYS and isinstance(value, str):
        return _shorten(value)
    return f"{key}={_shorten(value)}"


def _shorten(value: Any, limit: int = 80) -> str:
    """把任意值渲染成**单行**短文本：摘要必须是单行，否则日志会被一次 thought 刷屏。"""
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        try:
            text = json.dumps(to_jsonable(value), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):  # pragma: no cover - to_jsonable 已兜底
            text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------------------
# 回调机制
# --------------------------------------------------------------------------------------


class Callback(Protocol):
    """事件订阅者协议：任何带 ``on_event(event)`` 的对象都能被订阅。"""

    def on_event(self, event: TraceEvent) -> None: ...


#: 允许的订阅形态（§9.2 冻结）：要么是 Callback 协议对象，要么是 `fn(event)` 函数。
CallbackLike = Union[Callback, Callable[[TraceEvent], None]]


class CallbackManager:
    """回调分发器。**永不因为回调出错而影响主流程**（§13 红线 10 的机制保障）。

    线程安全用 `threading.Lock`（§0.4 R-LOOP：asyncio 原语不能进 `__init__`），
    理由很实际：`delegate_to_*` 是同步工具，会在 worker 线程里跑 Agent，
    而 trace 事件正是从那些线程里 emit 出来的。
    """

    def __init__(self, callbacks: Sequence[CallbackLike] = ()) -> None:
        # list(...) 复制：调用方传进来的序列不能被我们就地改动（红线 11 的同一精神）。
        self._callbacks: list[CallbackLike] = list(callbacks)
        self._lock = threading.Lock()
        #: 单个回调抛出的异常（不强推给调用方，但必须可观测）
        self.errors: list[tuple[CallbackLike, BaseException]] = []

    # ------------------------------------------------------------ 订阅管理
    def add(self, callback: CallbackLike) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def remove(self, callback: CallbackLike) -> None:
        """移除回调；不存在时记 debug 日志后忽略（`astream` 的 finally 里会无条件 remove）。"""
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                _LOGGER.debug("remove(): callback %r is not subscribed", callback)

    def subscribe(self, fn: Callable[[TraceEvent], None]) -> Callable[[], None]:
        """订阅并返回一个 unsubscribe 闭包（闭包可重复调用，第二次是 no-op）。"""
        self.add(fn)

        def _unsubscribe() -> None:
            self.remove(fn)

        return _unsubscribe

    def clear(self) -> None:
        """清空回调列表与错误记录（把 manager 复位）。

        SPEC-AMBIGUITY: §9.2 只写了 `clear(self) -> None`，没说清的是"清回调还是清错误"。
        裁决：两者都清 —— 名字叫 clear 的复位操作如果留下上一次的错误记录，
        会让"这次运行有没有回调出错"无法判断。
        """
        with self._lock:
            self._callbacks.clear()
            self.errors.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._callbacks)

    @property
    def callbacks(self) -> list[CallbackLike]:
        """回调列表的**副本**（调用方不该绕过 add/remove 直接改内部列表）。"""
        with self._lock:
            return list(self._callbacks)

    # ------------------------------------------------------------ 分发
    def emit(self, event: TraceEvent) -> None:
        """把事件分发给所有回调。**永不抛异常**。

        分发前先在锁内复制列表：回调里再 `add/remove`（`astream` 的 finally 就会）
        不会让迭代中的列表变化，也不会死锁。
        单个回调抛错 -> 记进 `self.errors` + 记日志，继续调用其余回调 ——
        一个只负责写 JSONL 的回调磁盘满掉，不该把整次 run 干掉。
        """
        with self._lock:
            targets = list(self._callbacks)

        for callback in targets:
            try:
                handler = getattr(callback, "on_event", None)
                if callable(handler):
                    handler(event)
                elif callable(callback):
                    callback(event)  # type: ignore[operator]
                else:
                    raise TypeError(
                        f"callback {callback!r} is neither a Callback nor callable"
                    )
            except (KeyboardInterrupt, SystemExit):
                # 这两个是"用户按了 Ctrl-C / 进程要退出"，不能当成回调故障咽下去。
                _LOGGER.exception("callback %r raised %s; aborting emit", callback, "BaseException")
                raise
            except BaseException as exc:
                # 含 asyncio.CancelledError：它继承 BaseException。这里记下来但**不**上抛 ——
                # emit 的契约是"永不抛异常"，取消信号会在 await 边界被重新观察到，
                # 而回调是在同步调用栈里跑的，吞掉它不会打断真正的取消传播（§9.4.7）。
                with self._lock:
                    self.errors.append((callback, exc))
                _LOGGER.exception("callback %r failed on event %s", callback, event.type.value)

    def emit_type(
        self,
        event_type: EventType | str,
        *,
        run_id: str = "",
        agent_name: str = "",
        step: int = 0,
        **data: Any,
    ) -> TraceEvent:
        """构造 + 分发，返回构造出的事件（便于调用方顺手断言/复用）。

        保留键校验**提前到这里**（§9.2 v2 变更）：`TraceEvent.__post_init__` 也会抛，
        但提前抛的好处是错误信息里能带上"是 emit_type 的 **data 传错了"这一层调用语义。
        """
        reserved = RESERVED_KEYS.intersection(data)
        if reserved:
            raise ConfigError(
                "emit_type() data must not contain reserved keys "
                f"{sorted(reserved)}; pass run_id/agent_name/step as keyword arguments instead"
            )
        event = TraceEvent(
            type=EventType.coerce(event_type),
            run_id=run_id,
            agent_name=agent_name,
            step=step,
            data=dict(data),
        )
        self.emit(event)
        return event


# --------------------------------------------------------------------------------------
# 内置回调（全部线程安全）
# --------------------------------------------------------------------------------------


class FunctionCallback:
    """把普通函数包成 Callback。

    支持两种签名（用 `inspect.signature` 的**位置参数个数**判定）：
    ``fn(event)`` 或 ``fn(event_type: str, data: dict)``。
    判定而不是 try/except TypeError 调用，是因为后者会吞掉函数**内部**真正的 TypeError。
    """

    def __init__(
        self,
        fn: Callable[..., None],
        *,
        event_types: Sequence[EventType] | None = None,
        name: str = "",
    ) -> None:
        if not callable(fn):
            raise ConfigError(f"FunctionCallback expects a callable, got {type(fn).__name__}")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "") or repr(fn)
        # 只保留 EventType：`event.type not in self.event_types` 是集合查找，热路径更快。
        self.event_types: frozenset[EventType] | None = (
            None if event_types is None else frozenset(EventType.coerce(t) for t in event_types)
        )
        self.takes_two_args = _positional_arity(fn) >= 2

    def __call__(self, event: TraceEvent) -> None:
        self.on_event(event)

    def on_event(self, event: TraceEvent) -> None:
        if self.event_types is not None and event.type not in self.event_types:
            return
        if self.takes_two_args:
            self.fn(event.type.value, event.data)
        else:
            self.fn(event)

    def __repr__(self) -> str:  # pragma: no cover - 只为日志可读
        return f"FunctionCallback({self.name})"


def _positional_arity(fn: Callable[..., Any]) -> int:
    """函数能接受的位置参数个数（用于区分 1 参 / 2 参回调）。

    取不到签名（C 实现的内建函数等）时保守返回 1 —— 单参形态是本项目里的主流用法
    （`Agent.astream` 的 `_put(event)`）。
    """
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        _LOGGER.debug("cannot inspect signature of %r; assuming single-argument callback", fn)
        return 1
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            # *args 形态：无法确定，按"能接受两个"处理（发送方多给一个参数不会炸）。
            return 2
    return count


class LoggingCallback:
    """把事件写进 logging：默认 INFO，ERROR 级事件（run_failed/llm_error/tool_error）用 WARNING。

    为什么三类"错误事件"只报 WARNING 而不是 ERROR：框架本身把它们编码进了 `AgentResult`
    （FAILED 是正常返回值的一种），用 ERROR 会让日志系统的告警规则全部误报。
    """

    #: 需要提级的错误类事件（§9.2 冻结的三类）
    ERROR_EVENTS = frozenset({"run_failed", "llm_error", "tool_error"})

    def __init__(
        self,
        logger: Any = None,
        *,
        level: int = logging.INFO,
        event_types: Sequence[EventType] | None = None,
    ) -> None:
        self.logger = logger if logger is not None else logging.getLogger(_AGENT_LOGGER_NAME)
        self.level = level
        self.event_types: frozenset[EventType] | None = (
            None if event_types is None else frozenset(EventType.coerce(t) for t in event_types)
        )
        # 事件可能来自 worker 线程；logging 本身线程安全，这里的锁只保护"级别选择 + 输出"
        # 这一步不会被别的线程插进半行。
        self._lock = threading.Lock()

    def on_event(self, event: TraceEvent) -> None:
        if self.event_types is not None and event.type not in self.event_types:
            return
        level = logging.WARNING if event.type.value in self.ERROR_EVENTS else self.level
        with self._lock:
            self.logger.log(level, "%s", event.summary())

    def __repr__(self) -> str:  # pragma: no cover
        return f"LoggingCallback({getattr(self.logger, 'name', self.logger)})"


class JsonlTraceCallback:
    """把事件按行追加写入 JSONL（`'a'` 模式），`close()` 关闭，`load()` 读回。

    **必须加锁**（§9.2 v2 变更）：文本模式的文件写入不是原子的，
    跨线程 emit（delegate 路径）会让两行交错成一行、JSON 解析失败。
    互斥 + 每次写完 flush，保证"不丢行"这条断言在并发下也成立。
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(self.path))
        # 目录不存在时自动创建：CLI 的 --trace out/trace.jsonl 是常见用法，
        # 让它因为"目录不存在"失败没有任何信息量（写文件本身的错误仍会抛）。
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._fh: Any = open(self.path, "a", encoding="utf-8")
        self._closed = False

    def on_event(self, event: TraceEvent) -> None:
        line = event.to_json() + "\n"
        with self._lock:
            if self._closed:
                # 关闭后再来事件（例如 TraceRecorder 退出后 manager 还被复用）：
                # 记 debug 而不是抛异常，否则会变成 CallbackManager.errors 里的一堆噪音。
                _LOGGER.debug("dropping event after close(): %s", event.type.value)
                return
            self._fh.write(line)
            self._fh.flush()  # 不 flush 的话 load()/tail -f 会看不到刚写的行

    def close(self) -> None:
        """关闭文件；幂等（`TraceRecorder.__exit__` 与用户代码可能都调一次）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.flush()
            finally:
                self._fh.close()

    def load(self) -> list[TraceEvent]:
        """读回文件里的全部事件（含本回调打开之前就已存在的行）。"""
        with self._lock:
            if not self._closed:
                self._fh.flush()  # 让"写进去还没落盘"的行也能被读到
        return load_trace(self.path)

    def __repr__(self) -> str:  # pragma: no cover
        return f"JsonlTraceCallback({self.path!r})"


class RichCallback:
    """彩色终端输出；**无 rich 时退化为 `print`**（两种环境都必须能 import 与运行）。

    `console` 显式传入时优先用它（CLI/测试可以塞一个 `Console(file=StringIO())` 抓输出）。
    """

    #: 事件类型 -> rich 样式（未登记的类型用默认样式）
    STYLES: dict[str, str] = {
        "run_started": "bold green",
        "run_finished": "bold green",
        "run_failed": "bold red",
        "llm_error": "red",
        "tool_error": "red",
        "parse_error": "red",
        "repeat_detected": "yellow",
        "budget_exceeded": "bold yellow",
        "context_truncated": "yellow",
        "thought": "cyan",
        "action_parsed": "blue",
        "tool_finished": "white",
        "tool_started": "dim",
        "nudge": "magenta",
    }

    def __init__(self, *, show_tokens: bool = True, console: Any = None) -> None:
        self.show_tokens = show_tokens
        self.console = console
        if self.console is None and RICH_AVAILABLE:  # pragma: no cover - 环境相关
            self.console = _rich_console.Console()
        # 判定标准是"这个对象能不能 print"，而不是"rich 装没装"：
        # 用户显式传入的 console 永远优先（哪怕在无 rich 环境里传入自己的富打印器）。
        self.use_rich = self.console is not None and hasattr(self.console, "print")
        self._lock = threading.Lock()

    def _write(self, text: str, style: str = "") -> None:
        if self.use_rich:
            self.console.print(text, style=style, highlight=False)
        else:  # 无 rich 的降级路径：行为可见（打出来的还是同一行摘要）
            print(text)

    def on_event(self, event: TraceEvent) -> None:
        style = self.STYLES.get(event.type.value, "")
        with self._lock:  # 多线程 emit 时不让两行交错
            self._write(event.summary(), style)
            if self.show_tokens and event.type is EventType.LLM_RESPONSE:
                usage = event.data.get("usage")
                if isinstance(usage, Mapping):
                    self._write(
                        "    tokens: prompt={} completion={} total={}".format(
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            usage.get("total_tokens", 0),
                        ),
                        "dim",
                    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"RichCallback(rich={self.use_rich})"


class TokenCounterCallback:
    """累计 `llm_response` 事件里的 usage，并按价格表估算成本。

    usage 在 event.data 里已经是 **dict**（§2.7：LLM_RESPONSE 的 usage 必须是
    `resp.usage.to_dict()`），所以用 `TokenUsage.from_dict` 读回来。
    模型名的来源按优先级：usage 字典自带的 `model_hint`/`model` -> 事件 data 的 `model`
    -> 最近一次 `llm_request` 的 `model`（LLM_RESPONSE 的冻结键里没有 model，
    但单次 run 内模型不会变，记住上一条请求的模型是安全的近似）。
    """

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self.calls = 0
        self._cost_total = 0.0
        self._priced_calls = 0
        self._model = ""
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float | None:
        """累计成本（美元）；**所有** usage 事件都没命中价格表时返回 None（诚实 > 猜）。"""
        with self._lock:
            return self._cost_total if self._priced_calls else None

    def on_event(self, event: TraceEvent) -> None:
        if event.type is EventType.LLM_REQUEST:
            model = event.data.get("model")
            if isinstance(model, str) and model:
                with self._lock:
                    self._model = model
            return
        if event.type is not EventType.LLM_RESPONSE:
            return

        raw = event.data.get("usage")
        usage = _usage_from_dict(raw) if isinstance(raw, Mapping) else TokenUsage()
        model = _resolve_event_model(event, fallback=self._model)
        cost = estimate_cost_usd(usage, model=model) if model else None
        if cost is None and model:
            # 降级可观测（§13 红线 12）：这个模型没有价格表条目，cost_usd 会保持 None。
            _LOGGER.debug("no price entry for model %r; cost_usd stays None", model)

        with self._lock:
            self.calls += 1
            self.usage = self.usage + usage  # __add__ 产生新对象：不与其他引用共享可变状态
            if cost is not None:
                self._cost_total += cost
                self._priced_calls += 1

    def reset(self) -> None:
        with self._lock:
            self.usage = TokenUsage()
            self.calls = 0
            self._cost_total = 0.0
            self._priced_calls = 0
            self._model = ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"TokenCounterCallback(calls={self.calls}, cost_usd={self.cost_usd})"


def _resolve_event_model(event: TraceEvent, *, fallback: str = "") -> str:
    """从 LLM 事件里挖出模型名（usage 字典 -> data['model'] -> 上一次请求的模型）。"""
    raw = event.data.get("usage")
    if isinstance(raw, Mapping):
        for key in ("model_hint", "model"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
    value = event.data.get("model")
    if isinstance(value, str) and value:
        return value
    return fallback


class MemoryTraceCallback:
    """收集 `memory_*` 事件（测试用）。`list.append` 是原子的，因此无需加锁。"""

    #: 三类记忆事件（§2.7 里 MemoryManager 唯一发射的三个）
    MEMORY_EVENTS = frozenset({"memory_write", "memory_retrieve", "memory_compress"})

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def on_event(self, event: TraceEvent) -> None:
        if event.type.value in self.MEMORY_EVENTS:
            self.events.append(event)

    def __len__(self) -> int:  # pragma: no cover - 顺手的小便利
        return len(self.events)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryTraceCallback({len(self.events)} event(s))"


# --------------------------------------------------------------------------------------
# 记录器
# --------------------------------------------------------------------------------------


class TraceRecorder:
    """上下文管理器：建 run_id、挂 JSONL 回调、结束时关闭。

        with TraceRecorder("trace.jsonl") as rec:
            rec.manager.emit_type(EventType.RUN_STARTED)
            rec.events   # 内存中的全部事件

    **它自己不发任何事件**（§2.7：发射者唯一）。它只是"把 manager + 文件 + 内存列表"
    打包成一个 with 块，让测试与 CLI 少写三行样板。
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        callbacks: Sequence[CallbackLike] = (),
        run_id: str | None = None,
    ) -> None:
        # 与 AgentState.create 同一格式（'run_' + 12 位 hex），两条路径的 trace 能对上号。
        self.run_id = run_id if run_id else "run_" + uuid.uuid4().hex[:12]
        self.path = None if path is None else os.fspath(path)
        #: list.append 是原子的，多线程 emit 下无需加锁（§9.2 冻结说明）
        self.events: list[TraceEvent] = []
        self._jsonl = JsonlTraceCallback(path) if path is not None else None
        initial: list[CallbackLike] = list(callbacks)
        if self._jsonl is not None:
            initial.insert(0, self._jsonl)  # 文件回调放最前：磁盘失败也不会挡住后续回调
        self.manager = CallbackManager(initial)
        self.manager.add(self._collect)

    def _collect(self, event: TraceEvent) -> None:
        """内存收集器（绑定方法，`remove` 时同一个对象可被移除）。"""
        self.events.append(event)

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        # 先把收集器摘掉再关文件：退出之后 manager 再被 emit 不会写进已关闭的文件。
        self.manager.remove(self._collect)
        if self._jsonl is not None:
            self.manager.remove(self._jsonl)
            self._jsonl.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"TraceRecorder(run_id={self.run_id!r}, path={self.path!r}, events={len(self.events)})"


# --------------------------------------------------------------------------------------
# 读取与统计
# --------------------------------------------------------------------------------------


def load_trace(path: str | os.PathLike[str]) -> list[TraceEvent]:
    """读 JSONL；跳过空行与解析不了的行（记 WARNING），**不抛异常**。

    为什么宽容：trace 文件是"进程被 kill 时也要能读"的产物，最后一行残缺是常态。
    读不动整个文件来报错，等于把已经跑完的 run 的观测数据一起丢掉。
    """
    resolved = os.fspath(path)
    events: list[TraceEvent] = []
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(TraceEvent.from_json(line))
                except Exception as exc:  # 含 ConfigError / SerializationError / JSONDecodeError
                    _LOGGER.warning(
                        "skipping unparsable trace line %s:%d (%s)", resolved, lineno, exc
                    )
    except FileNotFoundError:
        _LOGGER.warning("trace file not found: %s", resolved)
        return []
    except OSError as exc:
        _LOGGER.warning("cannot read trace file %s (%s)", resolved, exc)
        return []
    return events


def _usage_from_dict(raw: Mapping[str, Any]) -> TokenUsage:
    """容错版 `TokenUsage.from_dict`：缺字段/类型不对时按 0 补齐并记 WARNING。

    为什么不直接抛：usage 的来源是 trace（可能是手写的事件序列、旧版本文件、
    或 provider 少报了一个字段）。统计功能因为一个残缺的 counting dict 整体失败，
    等于把整份观测数据丢掉 —— 降级 + 留痕（§13 红线 12）比抛异常有用。
    """
    try:
        return TokenUsage.from_dict(raw)
    except (SerializationError, TypeError, ValueError) as exc:
        _LOGGER.warning("malformed usage dict %r (%s); falling back to zero padding", raw, exc)

    def _as_int(key: str) -> int:
        try:
            return int(raw.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return TokenUsage(
        prompt_tokens=_as_int("prompt_tokens"),
        completion_tokens=_as_int("completion_tokens"),
        total_tokens=_as_int("total_tokens"),
    )


def total_usage(events: Sequence[TraceEvent]) -> TokenUsage:
    """把全部 `llm_response` 事件的 usage 加起来（其余事件不带 usage）。"""
    total = TokenUsage()
    for event in events:
        if event.type is not EventType.LLM_RESPONSE:
            continue
        raw = event.data.get("usage")
        if isinstance(raw, Mapping):
            total = total + _usage_from_dict(raw)
    return total


def events_of_type(
    events: Sequence[TraceEvent], event_type: EventType | str
) -> list[TraceEvent]:
    """按类型过滤；未知类型字符串抛 ``ConfigError``（`EventType.coerce` 的语义）。"""
    wanted = EventType.coerce(event_type)
    return [event for event in events if event.type is wanted]


def render_trace(events: Sequence[TraceEvent], *, indent: bool = True) -> str:
    """CLI `trace` 的渲染：按 step 分组，缩进显示 thought/action/tool。

    `indent=False` 时退化成"每行一条 `summary()`"的平铺输出（便于 grep / diff）。
    输出**永不为空**：空 trace 也会打一行表头，调用方不必为"没事件"写特例。
    """
    runs = sum(1 for event in events if event.type is EventType.RUN_STARTED)
    lines = [f"trace: {len(events)} event(s), {runs} run(s)"]

    run_level = (EventType.RUN_STARTED, EventType.RUN_FINISHED, EventType.RUN_FAILED)
    for event in events:
        if event.type in run_level:
            lines.append(event.summary())

    seen: set[int] = set()
    ordered_steps: list[int] = []
    for event in events:
        if event.type in run_level or event.step in seen:
            continue
        seen.add(event.step)
        ordered_steps.append(event.step)

    for step in ordered_steps:
        if indent:
            lines.append(f"step {step}")
        for event in events:
            if event.type in run_level or event.step != step:
                continue
            lines.append(f"  {_event_detail(event)}" if indent else event.summary())
    return "\n".join(lines)


def _event_detail(event: TraceEvent) -> str:
    """缩进树里一行的内容：``<type>: <主载荷>``（主载荷取不到时退化为 summary 的尾部）。"""
    data = event.data
    if event.type is EventType.THOUGHT:
        detail = _shorten(data.get("text", ""))
    elif event.type is EventType.ACTION_PARSED:
        detail = f"{data.get('action', '?')}({_shorten(data.get('arguments', {}))})"
    elif event.type is EventType.NUDGE:
        detail = _shorten(data.get("text", ""))
    elif event.type in (EventType.TOOL_STARTED, EventType.TOOL_FINISHED, EventType.TOOL_ERROR):
        detail = " ".join(
            _format_field(key, data[key])
            for key in ("tool_name", "ok", "duration_ms", "error_type")
            if key in data
        )
    else:
        # 其余事件：沿用 summary 里 `[step] type ` 之后的部分，避免两套渲染规则漂移。
        prefix = f"[{event.step}] {event.type.value}"
        detail = event.summary()[len(prefix) :].strip()
    return f"{event.type.value}: {detail}" if detail else event.type.value


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """nearest-lower 分位数（§9.2 冻结：下标 `int(q * (n - 1))`，**不做插值**）。

    小样本（trace 里通常只有几十个耗时）下插值出来的值没有意义，
    取实际观测到的那一个更诚实。
    """
    count = len(sorted_values)
    if count == 0:
        return 0.0
    index = int(quantile * (count - 1))
    index = max(0, min(index, count - 1))  # quantile 传入越界值时的保护
    return float(sorted_values[index])


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    """`{"total","mean","p50","p95"}`（n == 0 时四个数都是 0.0）。"""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    total = sum(ordered)
    return {
        "total": total,
        "mean": total / len(ordered),
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
    }


#: 计入 `trace_stats["errors"]` 的三类错误事件及其 error_type 键。
_ERROR_EVENTS = (EventType.RUN_FAILED, EventType.LLM_ERROR, EventType.TOOL_ERROR)


def trace_stats(events: Sequence[TraceEvent]) -> dict[str, Any]:
    """trace 的统计摘要（CLI `trace --stats --json` 的输出就是本函数的返回值）。

    字段表与算法见 §9.2（p50/p95 用 nearest-lower，不插值）。

    事件到字段的映射（§9.2 只冻结了字段名，映射关系由本节补全并冻结：
    每个字段都取自"语义上唯一对应"的事件）：
    - runs/steps/llm_calls = RUN_STARTED / STEP_STARTED / LLM_REQUEST 的条数；
    - llm_latency_ms <- LLM_RESPONSE 的 `latency_ms`；
    - tool_calls = 见过的**不同 call_id** 数（TOOL_STARTED 是 attempt 级的，
      重试会产生多条，所以按 call_id 去重；事件里没有 call_id 时退化为 TOOL_FINISHED 条数，
      再退化为 TOOL_ERROR 条数）；
    - tool_failures = TOOL_ERROR 条数 + `ok=False` 的 TOOL_FINISHED 条数
      （同一个 call_id 两种都出现过时只算一次）；
    - tool_latency_ms <- TOOL_FINISHED 的 `duration_ms`，按 `tool_name` 分组；
    - retries = TOOL_RETRY 条数 + 带 `retry > 0` 的 LLM_REQUEST 条数
      （LLM 层不为重试单独发事件，而是给 LLM_REQUEST 带 retry 字段，§6.3）；
    - parse_errors / nudges = PARSE_ERROR / NUDGE 的条数；
    - usage = 全部 LLM_RESPONSE 的 usage 之和；
    - cost_usd = 有价格表命中的模型成本之和，全都没命中时为 None；
    - errors = RUN_FAILED / LLM_ERROR / TOOL_ERROR 的 `error_type` 计数。
    """
    counts = Counter(event.type for event in events)

    llm_latencies = [
        float(event.data["latency_ms"])
        for event in events
        if event.type is EventType.LLM_RESPONSE
        and isinstance(event.data.get("latency_ms"), (int, float))
    ]

    tool_finished = [event for event in events if event.type is EventType.TOOL_FINISHED]
    tool_errors = [event for event in events if event.type is EventType.TOOL_ERROR]

    # --- tool_calls：优先按 call_id 去重（重试的多次 attempt 共享同一个 call_id） ---
    call_ids: set[str] = set()
    for event in events:
        if event.type in (EventType.TOOL_STARTED, EventType.TOOL_FINISHED, EventType.TOOL_ERROR):
            call_id = event.data.get("call_id")
            if isinstance(call_id, str) and call_id:
                call_ids.add(call_id)
    if call_ids:
        tool_calls = len(call_ids)
    elif tool_finished:
        tool_calls = len(tool_finished)
    else:
        tool_calls = len(tool_errors)

    # --- tool_failures：TOOL_ERROR 与 ok=False 的 TOOL_FINISHED 合并去重 ---
    error_call_ids = {
        event.data["call_id"]
        for event in tool_errors
        if isinstance(event.data.get("call_id"), str) and event.data.get("call_id")
    }
    tool_failures = len(tool_errors)
    for event in tool_finished:
        if event.data.get("ok") is not False:
            continue
        call_id = event.data.get("call_id")
        if isinstance(call_id, str) and call_id and call_id in error_call_ids:
            continue  # 同一次调用已经由 TOOL_ERROR 计过数
        tool_failures += 1

    # --- tool_latency_ms：每次完成一次调用记一条（失败的调用也有耗时，同样计入） ---
    tool_counts: dict[str, int] = {}
    tool_durations: dict[str, list[float]] = {}
    for event in tool_finished:
        name = event.data.get("tool_name") or event.data.get("name")
        key = name if isinstance(name, str) and name else "?"
        tool_counts[key] = tool_counts.get(key, 0) + 1
        duration = event.data.get("duration_ms")
        if isinstance(duration, (int, float)):
            tool_durations.setdefault(key, []).append(float(duration))
    tool_latency_ms: dict[str, dict[str, Any]] = {}
    for key, count in tool_counts.items():
        durations = sorted(tool_durations.get(key, ()))
        tool_latency_ms[key] = {
            "count": count,  # 冻结类型：count 是 int
            "mean": (sum(durations) / len(durations)) if durations else 0.0,
            "p95": _percentile(durations, 0.95),
        }

    # --- 成本 ---
    cost_total = 0.0
    priced_calls = 0
    current_model = ""
    for event in events:
        if event.type is EventType.LLM_REQUEST:
            model = event.data.get("model")
            if isinstance(model, str) and model:
                current_model = model
            continue
        if event.type is not EventType.LLM_RESPONSE:
            continue
        raw_usage = event.data.get("usage")
        if not isinstance(raw_usage, Mapping):
            continue
        model = _resolve_event_model(event, fallback=current_model)
        if not model:
            continue
        cost = estimate_cost_usd(_usage_from_dict(raw_usage), model=model)
        if cost is not None:
            cost_total += cost
            priced_calls += 1

    # --- 错误类型计数 ---
    errors: Counter[str] = Counter()
    for event in events:
        if event.type in _ERROR_EVENTS:
            error_type = event.data.get("error_type")
            if isinstance(error_type, str) and error_type:
                errors[error_type] += 1

    retries = counts[EventType.TOOL_RETRY] + sum(
        1
        for event in events
        if event.type is EventType.LLM_REQUEST
        and isinstance(event.data.get("retry"), (int, float))
        and event.data["retry"] > 0
    )

    # --- usage：以 LLM_RESPONSE 为准；trace 里完全没有逐次 usage 时才退化为 RUN_FINISHED 的汇总 ---
    # （RUN_FINISHED 的 data 里也带 usage，§2.7；两者都算会重复计数，所以只在缺前者时兜底。）
    usage_total = total_usage(events)
    if usage_total.is_empty():
        for event in events:
            if event.type is EventType.RUN_FINISHED and isinstance(event.data.get("usage"), Mapping):
                usage_total = usage_total + _usage_from_dict(event.data["usage"])

    return {
        "runs": counts[EventType.RUN_STARTED],
        "steps": counts[EventType.STEP_STARTED],
        "llm_calls": counts[EventType.LLM_REQUEST],
        "llm_latency_ms": _latency_summary(llm_latencies),
        "tool_calls": tool_calls,
        "tool_failures": tool_failures,
        "tool_latency_ms": tool_latency_ms,
        "retries": retries,
        "parse_errors": counts[EventType.PARSE_ERROR],
        "nudges": counts[EventType.NUDGE],
        "usage": usage_total.to_dict(),
        "cost_usd": cost_total if priced_calls else None,
        "errors": dict(errors),
    }


# --------------------------------------------------------------------------------------
# 低层 -> TraceEvent 的唯一适配器
# --------------------------------------------------------------------------------------


def as_llm_callback(manager: CallbackManager) -> LowLevelEvent:
    """把 LLM/tools/memory 层的 ``(event_type_str, data)`` 轻量回调转成 ``TraceEvent`` 并 emit。

    这是 §2.7 冻结的**唯一**适配器（低层不得 import agent 层，所以事件名在低层是字符串）。
    两条硬约束：
    1. 对未知字符串**忽略**（记 debug）——低层将来新增事件类型不应该让老版本的高层崩；
    2. 整个函数**永不抛异常** —— 它在 `BaseLLMClient._emit` / `ToolExecutor` 的主流程里
       被同步调用，抛出去会把一次正常的 LLM 调用变成失败。
    附带能力：低层若在 data 里塞了 `run_id`/`agent_name`/`step`/`ts`（§2.4 的元数据），
    这里把它们**提升**成事件字段（而不是触发保留键校验），因为低层确实可能比高层更早知道
    run 的归属（例如 worker 线程里的 executor）。
    """

    def _on_event(event_type: str, data: dict[str, Any] | None = None) -> None:
        try:
            try:
                coerced = EventType.coerce(event_type)
            except ConfigError:
                _LOGGER.debug("ignoring unknown low-level event %r", event_type)
                return

            payload: dict[str, Any] = dict(data) if isinstance(data, Mapping) else {}
            run_id = payload.pop("run_id", "")
            agent_name = payload.pop("agent_name", "")
            step = payload.pop("step", 0)
            payload.pop("type", None)  # 事件类型以位置参数为准
            ts = payload.pop("ts", None)

            event = TraceEvent(
                type=coerced,
                run_id=str(run_id or ""),
                agent_name=str(agent_name or ""),
                step=int(step or 0),
                timestamp=float(ts) if isinstance(ts, (int, float)) else utc_now(),
                data=payload,
            )
            manager.emit(event)
        except Exception:  # noqa: BLE001 - 适配器契约：永不向上抛
            _LOGGER.warning(
                "as_llm_callback failed to convert event %r; ignored", event_type, exc_info=True
            )

    return _on_event
