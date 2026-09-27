from __future__ import annotations

"""``liteagent/tools/executor.py`` —— 审批、并发、超时、重试、取消（§7.4）。

执行器的职责可以用一句话概括：**把"一次工具调用"这件事的全部不确定性收敛成
一个 ``ToolResult``**。模型侧只认文本，所以无论底下发生了什么（找不到工具、参数
不合法、审批被拒、超时、重试耗尽、熔断、被兄弟调用取消），出口永远是一个
``ok=False`` 且 ``content`` 以 ``"ERROR("`` 开头的结构化结果 —— 唯一允许逃逸的
是 ``asyncio.CancelledError``（调用方取消，属于控制流而不是业务失败）。

三个必须记住的冻结点（面试高频）：

1. **获取顺序**：先并发信号量、后 ``seq`` 锁（§13 红线 14）。反序会在 N 个占满
   信号量的调用者之间形成死锁。
2. **同步工具走 ``run_in_executor``，绝不 ``to_thread(async_fn)``**（M-2：``to_thread``
   拿到 async 函数只会返回一个没人 await 的协程对象，静默失效）。
3. **同步工具超时 = 孤儿线程**：``asyncio.wait_for`` 只能取消 await 层，worker
   线程仍在跑。标记 ``metadata["orphan_thread"]`` 并**不重试**（§3.4 的 v2 例外），
   否则两个线程会同时写同一份资源。
"""

import asyncio
import base64
import functools
import inspect
import json
import logging
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import copy_context
from dataclasses import is_dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from liteagent.config import (
    ExecutorConfig,
    LoopBoundPool,
    NO_TIMEOUT,
    compute_backoff,
    default_sleep,
    run_sync,
    to_jsonable,
    truncate_head_tail,
)
from liteagent.errors import (
    AgentAbortedError,
    LiteAgentError,
    MemoryStoreError,
    SandboxViolationError,
    ToolApprovalDeniedError,
    ToolDefinitionError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRetryExhaustedError,
    ToolSkippedError,
    ToolTimeoutError,
    ToolValidationError,
)
from liteagent.tools.base import Tool, cancel_scope
from liteagent.tools.registry import ToolRegistry
from liteagent.tools.schema import unvalidatable_parameters, validate_instance
from liteagent.types import ToolCall, ToolResult

__all__ = ["ToolExecutor", "ExecutorConfig"]

logger = logging.getLogger("liteagent.tools.executor")

# §2.7 的低层事件回调签名。agent 层不得被 tools 层 import，所以事件名是**字符串**
# （取值即 agent/callbacks.py 里 EventType 的 .value）；as_llm_callback 是唯一适配器。
LowLevelEvent = Callable[[str, dict[str, Any]], None]

# 事件名常量：与 §9.2 的 EventType 逐字对应（工具层不得 import agent 层，所以这里写字符串）。
_EV_TOOL_STARTED = "tool_started"
_EV_TOOL_RETRY = "tool_retry"
_EV_TOOL_FINISHED = "tool_finished"
_EV_TOOL_ERROR = "tool_error"
_EV_TOOL_APPROVAL = "tool_approval"


# --------------------------------------------------------------------------------------
# 回灌文案（§3.4 的第二层分类：模型能自纠正 vs 环境/实现故障）
# --------------------------------------------------------------------------------------

#: infrastructure 类失败的**冻结文案**（§3.4 表格原文）。
FEEDBACK_INFRASTRUCTURE = (
    "do not retry this tool with the same arguments; "
    "try a different approach or give your final answer"
)
#: recoverable 类失败：必须告诉模型"怎么改"，而不是只说"你错了"。
FEEDBACK_RECOVERABLE = (
    "fix the arguments to match the tool's JSON schema and call this tool again, "
    "or pick one of the available tools"
)

_RECOVERABLE_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ToolValidationError,
    ToolNotFoundError,
    ToolDefinitionError,
)
#: §3.4 表格里列出 feedback_kind 的异常；这里只用来做**可读性**归类，
#: 真正的判定在 :func:`feedback_kind_for`（表之外的异常一律按 infrastructure 处理 ——
#: "劝模型换策略"永远比"让模型原样重试"安全）。
_INFRASTRUCTURE_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ToolExecutionError,
    SandboxViolationError,
    ToolTimeoutError,
    ToolRetryExhaustedError,
    ToolSkippedError,
    MemoryStoreError,
    ToolApprovalDeniedError,
)


def feedback_kind_for(exc: BaseException) -> str:
    """把异常映射成 ``"recoverable"`` / ``"infrastructure"``（§3.4 冻结映射表）。

    表外的 ``LiteAgentError``（如 ``ConfigError``/``SerializationError``）归为
    ``"infrastructure"``：这两类都是"模型改参数也救不了"的故障，而按 recoverable
    回灌会让模型反复重试同一条死路。

    SPEC-AMBIGUITY: §3.4 只枚举了两类异常，没说表外怎么办。裁决如上（fail-safe 方向）。
    """
    if isinstance(exc, _RECOVERABLE_ERROR_TYPES):
        return "recoverable"
    if isinstance(exc, _INFRASTRUCTURE_ERROR_TYPES):
        return "infrastructure"
    return "infrastructure"


def _append_feedback(exc: LiteAgentError, kind: str) -> None:
    """把回灌文案追加到异常 ``message`` 末尾（**原地修改**，见下方理由）。

    为什么改 message 而不是改 ``ToolResult.content``：``ToolResult.error_text()``
    的实现是 ``"ERROR(T): msg" + ("\\n" + content if content != 前缀)``。若把文案直接
    追加到 content 上，content 就不再等于前缀，``to_message()`` 会再拼一次前缀 ——
    错误文本出现两遍（native 模式下模型真的会看到两遍）。把文案并进 message 后
    ``content == error_text()`` 恒成立，任何调用序都不会退化。

    异常对象是执行器**自己刚造出来的**（尚未交给任何调用方），原地改 message 不会
    影响共享状态；``LiteAgentError.__str__`` 读的是 ``self.message``，因此
    ``ToolResult.failure`` 里生成的 content 自然带上文案。
    """
    if not kind:
        return
    suffix = FEEDBACK_INFRASTRUCTURE if kind == "infrastructure" else FEEDBACK_RECOVERABLE
    if not suffix or suffix in exc.message:
        return
    exc.message = f"{exc.message}\n{suffix}"
    # Exception.args 是 str(exc) 在未覆盖 __str__ 时的来源；这里显式同步一份，
    # 避免"message 改了但 args 没改"的两种真值。
    exc.args = (exc.message,)


def _default_is_retryable(exc: BaseException) -> bool:
    """§7.4.1 步骤 5.f 的默认重试判定：只认白名单（``exc.retryable``，§3.4）。"""
    return isinstance(exc, LiteAgentError) and bool(exc.retryable)


class _CancelSafeAcquire:
    """在 worker 线程里等一把 ``threading.Lock``，且**取消安全**（§7.4.1 步骤 5.b）。

    [v3 修正] 朴素写法 ``await asyncio.to_thread(lock.acquire)`` 有一个洞：调用方被取消时
    asyncio 只能取消 ``await`` 层，worker 线程仍会把 ``lock.acquire()`` 跑完；而释放锁的
    ``finally`` 属于**已经被取消的协程**，永远不会执行 —— 锁被一个没有逻辑归属者的线程
    永久持有，该工具此后每次调用都卡在 ``lock.acquire()`` 上。

    后果比"该工具卡住"更重：卡死的是**默认执行器**里的线程（非 daemon），
    ``asyncio.run`` / ``config._run_and_cleanup`` 收尾时会 ``shutdown_default_executor()``
    去 join 它，于是同步 API（``execute_sync`` / ``Agent.run``）会**无异常、无超时地**
    永久挂起。

    本类把"取消已经发生"这一事实传回给线程：线程拿到锁后如果发现已取消，立刻原地归还，
    绝不让锁变成孤儿。``abort()`` 与 ``acquire()`` 通过 ``_release_lock`` 串行化，
    保证锁恰好归还一次（``threading.Lock.release()`` 对未持有的锁会抛 RuntimeError）。
    """

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._aborted = threading.Event()
        self._release_lock = threading.Lock()
        self._acquired = False
        self._released = False

    def acquire(self) -> bool:
        """在 worker 线程里执行。返回 False 表示"已取消，锁已原地归还"。"""
        self._lock.acquire()
        with self._release_lock:
            self._acquired = True
            if self._aborted.is_set() and not self._released:
                self._released = True
                self._lock.release()
                return False
        return True

    def abort(self) -> None:
        """协程被取消时调用：把锁的归属还给系统（要么此刻归还，要么等 acquire 归还）。"""
        self._aborted.set()
        with self._release_lock:
            if self._acquired and not self._released:
                self._released = True
                self._lock.release()


async def _await_with_deadline(awaitable: Any, timeout: float) -> Any:
    """``asyncio.wait_for`` 的**取消安全**替代（§7.4.1 步骤 5.c 的取消语义）。

    [v3 修正] 3.10 的 ``asyncio.wait_for`` 有一个已知缺陷（GH-86296 / bpo-42130，
    3.12 才用 ``asyncio.timeouts`` 重写）：当被等待的 future 与调用方的取消落在
    **同一个 tick**（future 已经 done）时，它的 ``except CancelledError: if fut.done():
    return fut.result()`` 会把**调用方的取消静默吞掉**。

    后果不是"超时不准"，而是"取消失效"：``Agent.astream`` 提前 break 时的
    ``task.cancel()`` 发出去了、也返回 True，却毫无效果 —— arun 会跑完整个 ReAct 循环，
    继续烧 LLM 调用、继续写记忆。默认 ``default_timeout_s=30`` 下必现，因为**任何带工具
    调用的 agent** 都走这条 ``wait_for`` 路径（纯文本、没有工具调用的 agent 不走）。

    这里手写超时：``loop.call_later`` 到点取消内层任务，并在 ``CancelledError`` 里
    **区分"自己的超时取消"与"调用方的取消"**：后者一律原样上抛，绝不 return。
    超时仍折算成 ``asyncio.TimeoutError``（与 ``wait_for`` 的可观测形态一致，
    调用方的 ``except asyncio.TimeoutError`` 分支不用改）。
    """
    loop = asyncio.get_running_loop()
    fut = asyncio.ensure_future(awaitable, loop=loop)
    timed_out = False

    def _expire() -> None:
        nonlocal timed_out
        timed_out = True
        fut.cancel()

    handle = loop.call_later(timeout, _expire)
    try:
        return await fut
    except asyncio.CancelledError:
        if timed_out:
            raise asyncio.TimeoutError() from None
        # 调用方取消：把内层也取消掉，然后**原样上抛**（绝不 return fut.result()）。
        if not fut.done():
            fut.cancel()
        raise
    finally:
        handle.cancel()


def _coerce_int(value: Any, default: int) -> int:
    """宽容地把配置/元数据里的值转成 int；失败则用默认值（不抛）。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("executor: cannot coerce %r to int; using %r", value, default)
        return default


def _coerce_float(value: Any, default: float) -> float:
    """宽容地把配置/元数据里的值转成 float；失败则用默认值（不抛）。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("executor: cannot coerce %r to float; using %r", value, default)
        return default


# --------------------------------------------------------------------------------------
# §7.4.3 _stringify
# --------------------------------------------------------------------------------------

_KIND_NONE = "none"
_KIND_STR = "str"
_KIND_BYTES = "bytes"
_KIND_TOOL_RESULT = "tool_result"
_KIND_JSON = "json"
_KIND_DATACLASS = "dataclass"
_KIND_OTHER = "other"


def _stringify_kind(raw_return: Any) -> str:
    """§7.4.3 的类型分派。**单独抽出来**是因为 ``_stringify`` 与 ``_stringify_extras``
    必须对同一个值给出同一个判定 —— 两处各写一份 ``isinstance`` 迟早会漂移。

    顺序敏感：``ToolResult`` 是 dataclass，``bool`` 是 ``int``，都必须先判定更具体的那个。
    """
    if raw_return is None:
        return _KIND_NONE
    if isinstance(raw_return, ToolResult):
        return _KIND_TOOL_RESULT
    if isinstance(raw_return, str):
        return _KIND_STR
    if isinstance(raw_return, (bytes, bytearray)):
        return _KIND_BYTES
    if isinstance(raw_return, (dict, list, int, float, bool)):
        return _KIND_JSON
    if is_dataclass(raw_return) and not isinstance(raw_return, type):
        return _KIND_DATACLASS
    return _KIND_OTHER


def _stringify(raw_return: Any) -> str:
    """把工具返回值渲染成回灌给模型的文本（§7.4.3 冻结规则，逐条实现）。

    - ``None`` -> ``""``
    - ``str`` -> 原样（不转义、不裁剪：截断是调用方的事）
    - ``bytes`` -> base64 单行
    - ``ToolResult`` -> 取 ``raw.content``（metadata/ok 由 :func:`_stringify_extras` 处理）
    - ``dict``/``list``/``int``/``float``/``bool`` -> 缩进 JSON
    - dataclass 实例 -> ``to_jsonable`` 后再 JSON
    - 其它 -> ``str(obj)``（原始类型名进 metadata["stringified"]，§13 红线 12）

    ``default=str`` 是最后一道保险：JSON 里混进 datetime/Decimal 也不能让整次
    工具调用因为"渲染失败"而失败。
    """
    kind = _stringify_kind(raw_return)
    if kind == _KIND_NONE:
        return ""
    if kind == _KIND_STR:
        return raw_return
    if kind == _KIND_BYTES:
        return base64.b64encode(bytes(raw_return)).decode("ascii")
    if kind == _KIND_TOOL_RESULT:
        content = raw_return.content
        return content if isinstance(content, str) else str(content)
    if kind == _KIND_JSON:
        return json.dumps(raw_return, ensure_ascii=False, indent=2, default=str)
    if kind == _KIND_DATACLASS:
        return json.dumps(to_jsonable(raw_return), ensure_ascii=False, indent=2, default=str)
    return str(raw_return)


def _stringify_extras(raw_return: Any) -> dict[str, Any]:
    """``_stringify`` 的伴随信息（写进 ``ToolResult.metadata``）。

    - ``ToolResult`` -> 合并它的 metadata 与 ok/error 状态（§7.4.3 原文："取 raw.content
      并合并其 metadata 与 ok 状态"）；
    - 其它兜底类型 -> ``stringified=<原始类型名>``，让"这个结果是 str() 转出来的"
      在 trace 里可观测（§13 红线 12：降级不得静默）。
    """
    kind = _stringify_kind(raw_return)
    if kind == _KIND_TOOL_RESULT:
        # 只合并它的 metadata：ok / error / error_type 已经落在外层 ToolResult 的
        # **同名字段**上（见 _success），再往 metadata 里抄一份是 §2.4 禁止的同义 key。
        return dict(raw_return.metadata)
    if kind == _KIND_OTHER:
        return {"stringified": type(raw_return).__name__}
    return {}


# --------------------------------------------------------------------------------------
# ToolExecutor
# --------------------------------------------------------------------------------------


class ToolExecutor:
    """工具执行器：审批 -> 校验 -> 并发 -> 超时 -> 重试 -> 截断 -> 取消 -> 熔断。

    ``ExecutorConfig`` 的归属地是 ``config.py``（§5.3），本模块只做 re-export ——
    这样 ``from liteagent.tools.executor import ExecutorConfig`` 与
    ``from liteagent.config import ExecutorConfig`` 拿到的是**同一个类**。
    """

    registry: ToolRegistry
    config: ExecutorConfig

    def __init__(
        self,
        registry: ToolRegistry,
        config: ExecutorConfig | None = None,
        *,
        thread_pool: "ThreadPoolExecutor | None" = None,
        is_retryable: Callable[[BaseException], bool] | None = None,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        """[v2 变更] ``__init__`` 里冻结创建（全部同步原语，不受 R-LOOP 约束）。

        - ``self._pool = LoopBoundPool()``：per-loop 的 semaphore/thread_pool。**绝不在
          这里造 ``asyncio`` 原语**（§0.4 R-LOOP + §13 红线 5）：``asyncio.Semaphore``
          一旦争用就把 loop 存进自己身上，第二次 ``asyncio.run`` 必炸。
        - ``self._rng``：**一次**创建、全程复用。每次重试重建会让退避序列不可复现，
          与 D-07 的可复现断言直接冲突（§5.5 规则 3）。
        - ``self._seq_locks``：``threading.Lock`` 而非 loop-bound ``asyncio.Lock``。
          ``sequential_tools``（如 ``run_shell``）存在的唯一理由是**跨调用互斥**，而
          同步工具会被 ``execute_many`` 丢进不同 worker 线程、每个线程里可能各自
          ``asyncio.run`` —— loop-bound 锁跨 loop/跨线程完全不串行（§7.4.1 步骤 5.b）。

        ``thread_pool`` 的语义（冻结）：非 None 时**所有**同步工具都在它里面跑，
        ``aclose`` **不**关它（所有权属于调用方）；为 None 时按 loop 懒建私有池
        （``_pool.thread_pool("exec", config.thread_pool_size)``），``aclose`` 全部 shutdown。
        **不使用 ``loop.set_default_executor``**：v1 的写法在 3.10 无法可靠探测"用户是否
        已设过"，且状态放在实例上会让第二次 ``asyncio.run`` 的新 loop 静默跳过设置 ——
        而"跨两次 ``asyncio.run`` 复用同一 executor"正是 §12 要守的场景。
        """
        self.registry = registry
        self.config = config if config is not None else ExecutorConfig()
        self.on_event = on_event
        self._is_retryable = is_retryable

        self._pool = LoopBoundPool()
        self._rng: random.Random = random.Random(self.config.retry_policy.rng_seed)
        # 解析一次就固定（§5.5 规则 2）：运行中换 sleep_fn 不该悄悄改变已有实例的行为。
        self._sleep = self.config.resolved_sleep()
        self._seq_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self.config.sequential_tools
        }
        self._failure_counts: dict[str, int] = {}  # 熔断计数（按工具名，§7.4.1 步骤 4.6）
        # 保护 _failure_counts：execute_many 的同步工具会跨线程跑，而 delegate 类工具
        # 会在自己的线程里再起一个 loop 调本 executor（§7.4.1 步骤 5.b 的场景）。
        # threading 原语不受 R-LOOP 限制（§0.4），成本可忽略。
        self._counts_lock = threading.Lock()
        self._external_pool = thread_pool
        # [v3] "参数 schema 降级成 {} 所以无法在执行前校验"是**工具的结构属性**（常量），
        # 逐次刷日志只会把真正的告警淹没 —— 每个工具名只发一次 WARNING。
        # 与 _failure_counts 共用 _counts_lock（它同样被跨线程访问）。
        self._coverage_warned: set[str] = set()
        # [v3 修正] fail_fast 的"本批次是否触发过"是**批次级信息**，绝不能存在实例上：
        # 同一个 executor 可能被多个并发调用者共享（`Agent(executor=...)` 是公开参数），
        # 批次 B 开头会把实例标志重置，于是批次 A 收尾时把自己 fail_fast 亲手取消的
        # 兄弟任务误判成"调用方取消了 execute_many"而 raise CancelledError，破坏
        # §7.4.2 规则 1 的"fail_fast 仍返回与 calls 等长的列表"。
        # 现在由 `_watch_fail_fast` 返回一个**局部**布尔（见 execute_many）。

    # ----------------------------------------------------------------------------------
    # 事件
    # ----------------------------------------------------------------------------------

    def _emit(self, event_type: str, **data: Any) -> None:
        """低层事件发射（§2.7：TOOL_* 的**唯一**发射者是 ToolExecutor）。

        ``data`` 一律过 ``to_jsonable``（§2.7 的类型约束：进 data 的值必须已可序列化，
        否则 ``TraceEvent.to_json()`` 会因类型抛 ``TypeError``）。回调自身的异常不能
        打死执行器，但必须留痕（§13 红线 10）—— 记 WARNING，并用 ``except Exception``
        而非 ``except BaseException``，保证 ``CancelledError`` 不被吞（M-5）。
        """
        if self.on_event is None:
            return
        try:
            self.on_event(event_type, to_jsonable(dict(data)))
        except Exception as exc:  # noqa: BLE001 - 回调是用户代码，任何异常都不能外溢
            logger.warning("executor: on_event callback failed for %s: %r", event_type, exc)

    # ----------------------------------------------------------------------------------
    # 配置解析（§7.4.1 步骤 2/3）
    # ----------------------------------------------------------------------------------

    def _resolve_timeout(
        self, tool: Tool, call: ToolCall, timeout_s: float | None
    ) -> float | None:
        """超时解析（§7.4.1 步骤 2 冻结）。

        ``timeout_s == NO_TIMEOUT`` 或 ``config.default_timeout_s is None`` -> 不设超时；
        否则在 ``[参数, call.metadata["timeout_s"], tool.spec.timeout_s,
        config.default_timeout_s]`` 里取**最小有效值**。

        ``NO_TIMEOUT`` 哨兵之所以必要：v1 的超时链里**没有任何值能表达"不超时"**
        （``None`` 已经被"未指定"占用）。
        """
        if timeout_s == NO_TIMEOUT or self.config.default_timeout_s is None:
            return None
        candidates: list[float] = []
        for value in (
            timeout_s,
            call.metadata.get("timeout_s"),
            tool.spec.timeout_s,
            self.config.default_timeout_s,
        ):
            if value is None or value == NO_TIMEOUT:
                continue
            candidates.append(_coerce_float(value, self.config.default_timeout_s))
        if not candidates:
            return None
        return min(candidates)

    def _resolve_max_retries(self, tool: Tool, call: ToolCall, max_retries: int | None) -> int:
        """重试次数解析（§7.4.1 步骤 3）：参数 > ``call.metadata`` > ``spec`` > 配置。"""
        for candidate in (
            max_retries,
            call.metadata.get("max_retries"),
            tool.spec.max_retries,
            self.config.retry_policy.max_retries,
        ):
            if candidate is None:
                continue
            return max(0, _coerce_int(candidate, self.config.retry_policy.max_retries))
        return 0

    # ----------------------------------------------------------------------------------
    # 熔断计数（§7.4.1 步骤 4.6）
    # ----------------------------------------------------------------------------------

    def _failure_count(self, tool_name: str) -> int:
        with self._counts_lock:
            return self._failure_counts.get(tool_name, 0)

    def _record_failure(self, tool_name: str, feedback_kind: str) -> None:
        """只在 **infrastructure** 类失败时累加。

        为什么不累加 recoverable（校验错误/工具名写错）：那是模型的输入问题，模型改一次
        就好了；把它算进熔断会让"模型打错一次名字"直接禁掉一个健康工具。
        为什么不在这里清零 recoverable：清零会让"失败-恢复-失败"的间歇性故障永远
        触发不了熔断（§7.4.1 冻结的是"**连续** infrastructure 失败"）。
        """
        if feedback_kind != "infrastructure":
            return
        with self._counts_lock:
            self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1

    def _record_success(self, tool_name: str) -> None:
        """成功后清零（§7.4.1 步骤 4.6："成功后清零"）。"""
        with self._counts_lock:
            self._failure_counts.pop(tool_name, None)

    # ----------------------------------------------------------------------------------
    # 结果构造
    # ----------------------------------------------------------------------------------

    def _failure(
        self,
        call: ToolCall,
        exc: BaseException,
        *,
        attempts: int,
        duration_ms: float,
        metadata: Mapping[str, Any] | None = None,
        feedback: bool = True,
    ) -> ToolResult:
        """所有失败出口的唯一构造点，保证 §7.4.1 步骤 7 的返回不变式。

        ``feedback=False`` 只用于审批拒绝：那一条的回灌文案是**冻结字面量**
        ``ERROR(ToolApprovalDeniedError): this tool requires human approval``
        （§7.4.1 步骤 4.5），追加任何后缀都会破坏逐字断言。

        SPEC-AMBIGUITY: §3.4 把 ``ToolApprovalDeniedError`` 也列入需要"劝模型换策略"
        的 infrastructure 类，与 §7.4.1 的冻结字面量冲突。裁决：具体条款优先 ——
        审批拒绝不加后缀。其余 infrastructure 失败一律追加 ``FEEDBACK_INFRASTRUCTURE``。
        """
        md: dict[str, Any] = dict(metadata) if metadata else {}
        if feedback and isinstance(exc, LiteAgentError):
            _append_feedback(exc, str(md.get("feedback_kind") or ""))
        # §2.4：metadata["attempts"] 是 ToolResult.attempts 的冗余副本（executor 写）。
        md.setdefault("attempts", attempts)
        return ToolResult.failure(call, exc, duration_ms=duration_ms, attempts=attempts, metadata=md)

    def _success(
        self,
        tool: Tool,
        call: ToolCall,
        raw_return: Any,
        *,
        attempts: int,
        duration_ms: float,
    ) -> ToolResult:
        """成功出口：序列化 -> 截断（**只做一次**）-> 组装 metadata（§7.4.1 步骤 5.d）。

        "截断只做一次"的含义：``to_observation(max_chars=...)`` 由调用方（Agent）传，
        截断器**不得**再写 ``metadata["truncated"]``，否则两处互相覆盖、trace 里的
        数值取决于谁后跑。
        """
        text = _stringify(raw_return)
        md = _stringify_extras(raw_return)
        original_chars = len(text)
        max_chars = self.config.max_result_chars
        if max_chars and max_chars > 0 and original_chars > max_chars:
            text = truncate_head_tail(text, max_chars)
            md["truncated"] = True
        else:
            md["truncated"] = False
        # 即使没截断也写 original_chars：它是"模型看到的文本有多长"的权威口径，
        # 有它在，截断与否都能一眼对齐（§2.2 的字段全量输出精神）。
        md["original_chars"] = original_chars
        md["attempts"] = attempts

        if isinstance(raw_return, ToolResult) and not raw_return.ok:
            # 工具自己返回了一个失败结果（§7.4.3："合并其 metadata 与 ok 状态"）。
            # 直接字段构造而不是走 failure()：raw.content 已经是 "ERROR(...)" 文本，
            # 再套一层 error_text() 会把错误文本拼两遍。
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=text,
                ok=False,
                error=raw_return.error,
                error_type=raw_return.error_type,
                duration_ms=duration_ms,
                attempts=attempts,
                metadata=md,
            )
        return ToolResult.success(
            call, text, duration_ms=duration_ms, attempts=attempts, metadata=md
        )

    def _finish(
        self,
        tool: Tool,
        call: ToolCall,
        result: ToolResult,
        *,
        attempts: int,
        duration_ms: float,
    ) -> ToolResult:
        """统一的收尾：熔断计数 + 校验覆盖率缺口留痕 + TOOL_FINISHED / TOOL_ERROR 事件。"""
        self._note_validation_gaps(tool, call, result)
        if result.ok:
            self._record_success(call.name)
            self._emit(
                _EV_TOOL_FINISHED,
                tool_name=tool.name,
                call_id=call.id,
                ok=True,
                attempts=attempts,
                duration_ms=duration_ms,
                content_len=len(result.content),
                truncated=bool(result.metadata.get("truncated", False)),
            )
        else:
            # 缺 feedback_kind 的失败只有一种来源：工具自己返回了一个 ok=False 的
            # ToolResult（§7.4.3 分支）。它是**工具实现**的失败，按 infrastructure 计。
            self._record_failure(
                call.name, str(result.metadata.get("feedback_kind") or "infrastructure")
            )
            self._emit(
                _EV_TOOL_ERROR,
                tool_name=tool.name,
                call_id=call.id,
                error_type=result.error_type,
                message=result.error,
                attempts=attempts,
                duration_ms=duration_ms,
                **(
                    {"orphan_thread": True}
                    if result.metadata.get("orphan_thread")
                    else {}
                ),
            )
        return result

    # ----------------------------------------------------------------------------------
    # 核心：单次调用
    # ----------------------------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        *,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> ToolResult:
        """单次调用。**永不向外抛工具异常**（唯一例外：``asyncio.CancelledError`` /
        ``KeyboardInterrupt``，见 §7.4.1 5.a 之前的冻结规则）。

        步骤 1-7 的顺序是冻结的：查工具 -> 解析超时/重试 -> 校验 -> 审批 -> 熔断 ->
        尝试循环 -> 返回不变式。
        """
        started = time.monotonic()

        def elapsed_ms() -> float:
            return (time.monotonic() - started) * 1000.0

        # --- 步骤 1：查工具 ---
        try:
            tool = self.registry.get(call.name)
        except ToolNotFoundError as exc:
            # 必须把可用工具名回灌给模型（截到 20 个）：只说"找不到"模型只能靠猜，
            # 列出名字它下一轮就能改对（§3.4 的 recoverable 要求"怎么改"）。
            available = [str(n) for n in exc.available][:20]
            if not available:
                available = list(self.registry.names())[:20]
            err = ToolNotFoundError(call.name, available)
            result = self._failure(
                call,
                err,
                attempts=1,
                duration_ms=elapsed_ms(),
                metadata={"feedback_kind": "recoverable"},
            )
            # 工具都没找到，没有 Tool 对象可用于事件里的 tool_name —— 用 call.name。
            self._record_failure(call.name, str(result.metadata.get("feedback_kind", "")))
            self._emit(
                _EV_TOOL_ERROR,
                tool_name=call.name,
                call_id=call.id,
                error_type=result.error_type,
                message=result.error,
                attempts=1,
                duration_ms=result.duration_ms,
            )
            return result

        # --- 步骤 2/3：超时与重试 ---
        effective_timeout = self._resolve_timeout(tool, call, timeout_s)
        max_retries_eff = self._resolve_max_retries(tool, call, max_retries)

        # --- 步骤 4：参数校验（仅此处）。失败**不消耗重试预算** ---
        validation_metadata = self._validate_arguments(tool, call)
        if validation_metadata is not None:
            result = self._failure(
                call,
                ToolValidationError(
                    errors=list(validation_metadata["validation_errors"]),
                    tool_name=tool.name,
                ),
                attempts=1,
                duration_ms=elapsed_ms(),
                metadata={
                    "validation_errors": list(validation_metadata["validation_errors"]),
                    "feedback_kind": "recoverable",
                },
            )
            return self._finish(
                tool, call, result, attempts=1, duration_ms=result.duration_ms
            )

        # --- 步骤 4.5：审批（HITL，位于校验之后、执行之前）---
        approval = await self._approve(tool, call)
        if approval is not None:
            return self._finish(
                tool, call, approval, attempts=approval.attempts, duration_ms=approval.duration_ms
            )
        approved = bool(tool.spec.requires_approval)

        # --- 步骤 4.6：熔断 ---
        limit = self.config.disable_tool_after_failures
        if limit > 0 and self._failure_count(tool.name) >= limit:
            err = ToolExecutionError(
                tool_name=tool.name,
                call_id=call.id,
                message=(
                    f"tool {tool.name!r} is disabled after {limit} consecutive "
                    "infrastructure failures"
                ),
            )
            self._emit(_EV_TOOL_ERROR, tool_name=tool.name, call_id=call.id, disabled=True)
            result = self._failure(
                call,
                err,
                attempts=1,
                duration_ms=elapsed_ms(),
                metadata={"disabled": True, "feedback_kind": "infrastructure"},
            )
            # 刻意**不**累加计数：熔断拦下的这次不是"工具的又一次失败"，而是同一次故障
            # 的持续状态（§7.4.1 步骤 4.6 的伪代码里也没有增行）。让计数停在触发值
            # 才是"连续失败了几次"的正确读法；"被拦了多少次"由上面那条 disabled 事件观测。
            return result

        # --- 步骤 5：尝试循环（总尝试 = 1 + max_retries）---
        total = 1 + max_retries_eff
        if (not tool.spec.idempotent) and (not self.config.allow_retry_on_non_idempotent):
            # 非幂等工具重试 = 可能重复产生副作用（下单、发信）。默认不许。
            total = 1
        delay = 0.0
        last_exc: LiteAgentError | None = None
        exhausted = False
        orphan_thread = False
        result: ToolResult | None = None
        attempts = 0

        for attempt in range(total):
            attempts = attempt + 1
            self._emit(
                _EV_TOOL_STARTED,
                tool_name=tool.name,
                call_id=call.id,
                attempt=attempt,
                arguments=dict(call.arguments),
            )
            exc: LiteAgentError | None = None
            # 顺序写死：**先并发信号量、后 seq 锁**（§13 红线 14）。反序会在 N 个占满
            # 信号量的调用者之间形成死锁 —— 每个都握着 seq 锁等信号量，谁都不放。
            async with self._pool.semaphore("exec", self.config.max_concurrency):
                async with self._seq_guard(tool.name, tool):
                    # cancel_scope 每个 attempt 都要重新进一次（禁止提到重试循环外）：
                    # 否则第 2 次尝试拿到的是已 set 的同一个 Event，工具一进循环就自杀，
                    # 重试全部秒失败。
                    with cancel_scope() as flag:
                        try:
                            raw_return = await self._invoke(tool, call, effective_timeout, flag)
                        except asyncio.CancelledError:
                            # 规则 1（M-5）：宽 except 的首行必须是它 —— CancelledError
                            # 继承 BaseException，但**禁止**写 except BaseException 再判类型。
                            raise
                        except asyncio.TimeoutError:
                            # 规则 2：超时判定只认 asyncio.TimeoutError
                            # （asyncio.TimeoutError is not builtins.TimeoutError，3.10）。
                            # flag.set() 必须在 with 块**内部**完成，否则 worker 线程
                            # 可能永远看不到取消信号（M-5：contextvar 写不回传，但 Event
                            # 对象本身是活的，set 之后 worker 读到的 is_set() 会变 True）。
                            flag.set()
                            timeout_exc = ToolTimeoutError(
                                tool_name=tool.name,
                                timeout_s=effective_timeout or 0.0,
                            )
                            timeout_exc.retryable = True
                            exc = timeout_exc
                            if not tool.spec.is_async:
                                # 同步工具的实际线程不可中断（D-09）：绝不假装线程停了。
                                orphan_thread = True
                                logger.warning(
                                    "executor: tool %s timed out after %ss but its worker "
                                    "thread is still running (orphan_thread=True); the result "
                                    "is marked and this attempt will NOT be retried",
                                    tool.name,
                                    effective_timeout,
                                )
                        except LiteAgentError as e:
                            exc = e
                        except KeyboardInterrupt:
                            # 比 `except BaseException` 更具体，所以顺序上必须先写：
                            # Ctrl-C 是控制流而不是"工具失败"，把它包成 ToolExecutionError
                            # 会让长耗时的同步工具彻底无法中断（§13 红线 6 明确列出的
                            # 两个可逃逸异常之一）。
                            raise
                        except subprocess.TimeoutExpired as e:
                            # 同步工具内部用 subprocess.run(timeout=...) 自己抛出来的。
                            # 一律折算成 ToolTimeoutError，让"超时"只有一个类型出口。
                            exc = ToolTimeoutError(
                                tool_name=tool.name,
                                timeout_s=effective_timeout or 0.0,
                                cause=e,
                            )
                        except BaseException as e:  # noqa: BLE001 - CancelledError 已在上面拦掉
                            exc = ToolExecutionError(
                                tool_name=tool.name, call_id=call.id, cause=e
                            )
                        else:
                            result = self._success(
                                tool, call, raw_return, attempts=attempts, duration_ms=elapsed_ms()
                            )
            # 信号量已释放：退避在**信号量之外**执行（§7.4.1 步骤 5.b 的范围冻结）。
            # 包住整个重试循环的话，一个正在退避的任务会占着并发槽，max_concurrency=4
            # 时 4 个重试中的工具就能把整个执行器饿死。
            if result is not None:
                break
            if exc is None:  # pragma: no cover - 防御：上面的分支必须给 exc 赋值
                exc = ToolExecutionError(
                    tool_name=tool.name, call_id=call.id, message="tool produced no result"
                )
            last_exc = exc

            # --- 步骤 5.f：判定是否重试 ---
            retryable = (self._is_retryable or _default_is_retryable)(exc)
            if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async:
                # [v2 例外，§3.4] 同步工具超时 -> 线程仍存活，重试会**再起一个线程**；
                # 对 write_file / run_shell 这类工具意味着两个线程同时写同一份资源 -> 数据破坏。
                retryable = False
            if (not retryable) or attempts >= total:
                exhausted = bool(retryable) and attempts >= total
                break
            delay = compute_backoff(
                attempt,
                base_s=self.config.retry_policy.backoff_base_s,
                max_s=self.config.retry_policy.backoff_max_s,
                jitter=self.config.retry_policy.jitter,
                rng=self._rng,
            )
            if exc.retry_after_s:
                # 服务端明确给了"多久之后再来"，退避取两者的较大值。
                delay = max(delay, float(exc.retry_after_s))
            self._emit(
                _EV_TOOL_RETRY,
                tool_name=tool.name,
                call_id=call.id,
                attempt=attempt,
                delay_s=delay,
                error_type=type(exc).__name__,
            )
            await self._sleep(delay)

        if result is not None:
            if orphan_thread:
                # 理论不可达（成功与超时互斥），但若实现漂移了就必须可见。
                logger.warning("executor: tool %s reported success AND orphan_thread", tool.name)
            if approved:
                # §2.4：requires_approval 的工具，结果里**必有** approved。
                # 少了它，"这个动作是人批过的吗"在 trace 里就查不到了（审计刚需）。
                result.metadata["approved"] = True
            return self._finish(
                tool, call, result, attempts=attempts, duration_ms=result.duration_ms
            )

        # --- 步骤 5.g：失败收尾 ---
        final_exc: LiteAgentError = last_exc or ToolExecutionError(
            tool_name=tool.name, call_id=call.id, message="tool execution failed"
        )
        metadata: dict[str, Any] = {}
        if exhausted and attempts >= 1 and attempts > 1:
            # "若用尽且重试过"：attempts > 1 才谈得上"重试过"。
            metadata["last_error_type"] = type(final_exc).__name__
            final_exc = ToolRetryExhaustedError(
                attempts=attempts, last_error=final_exc, tool_name=tool.name
            )
        metadata["feedback_kind"] = feedback_kind_for(final_exc)
        if orphan_thread:
            metadata["orphan_thread"] = True
        result = self._failure(
            call,
            final_exc,
            attempts=attempts,
            duration_ms=elapsed_ms(),
            metadata=metadata,
        )
        if approved:
            result.metadata["approved"] = True
        return self._finish(tool, call, result, attempts=attempts, duration_ms=result.duration_ms)

    # ----------------------------------------------------------------------------------
    # 单次调用的子步骤
    # ----------------------------------------------------------------------------------

    def _note_validation_gaps(self, tool: Tool, call: ToolCall, result: ToolResult) -> None:
        """[v3] 步骤 4 的"覆盖率缺口"留痕：**哪些参数根本没被执行前校验覆盖**。

        两条都是 `{}` 降级，都会让"执行前入参校验"这个卖点静默失效（§13 红线 12：
        降级可以是设计，静默不行）：

        1. **参数 schema 降级成 `{}`(any)**：注解无法精确表达时（`Union[A, B]` /
           `Mapping[...]` / `Callable[...]` / `get_type_hints` 失败后的字符串注解…）
           schema 生成阶段的兜底产物，原文在 `ToolSpec.warnings` 里
           （`python3 -m liteagent tools show <name>` 可见）。`validate_instance` 对它
           **恒真** —— 该参数的类型与取值永远不会在校验阶段被拦下。
        2. **arguments 降级成 `{}`**：provider 侧 JSON 解析失败时
           （`ToolCall.try_from_arguments_json` 会留下 `{"__raw__": ...}` 而被步骤 4 拦下，
           但 `liteagent/llm/providers.py::_parse_inline_arguments` 这条路径只留
           `metadata["parse_error"]` 并把整份参数丢成 `{}`）。required 字段还有机会拦，
           **全 optional 的工具就会以空参数执行**，而校验层看不见这件事 ——
           这正是"降级成 `{}` 的参数绕过执行前校验"的那条路径。

        为什么落在 `_finish` 而不是 `_validate_arguments`：`_finish` 是成功 / 校验失败 /
        审批 / 失败四条出口的唯一汇合点，一处写就全覆盖。两个**不进** `_finish` 的早退
        分支（工具名找不到、熔断 disabled）刻意不写：前者拿不到 `Tool`，后者是
        "这个工具的连续故障状态"，与本次参数无关。

        WARNING 的频率：schema 缺口是**结构常量**（每个工具名只发一次）；
        arguments 降级是**本次调用**的事实，按调用发（模型偶发写错 JSON 少见，
        且这条日志是排查"为什么工具拿到空参数"的唯一线索）。
        """
        unvalidated = unvalidatable_parameters(tool.parameters)
        degraded = call.metadata.get("parse_error") if isinstance(call.metadata, Mapping) else None
        if not unvalidated and degraded is None:
            return
        gaps: dict[str, Any] = {}
        if unvalidated:
            gaps["unvalidated_parameters"] = list(unvalidated)
            gaps["note"] = (
                "parameters whose JSON Schema degraded to '{}' (any) cannot be "
                "validated before execution"
            )
            with self._counts_lock:
                first_time = tool.name not in self._coverage_warned
                self._coverage_warned.add(tool.name)
            if first_time:
                logger.warning(
                    "executor: tool %s declares parameter(s) %s whose JSON Schema "
                    "degraded to '{}' (any); these parameters CANNOT be validated "
                    "before execution (see ToolSpec.warnings / `tools show`)",
                    tool.name,
                    unvalidated,
                )
        if degraded is not None:
            # provider 侧解析失败：参数被整体丢成 {}，只有 required 还能拦住。
            gaps["arguments_degraded"] = True
            gaps["arguments_degraded_reason"] = str(degraded)
            logger.warning(
                "executor: arguments for tool %r degraded to {} (provider parse "
                "error: %s); the pre-execution validation cannot check them — "
                "the tool may run with empty arguments",
                tool.name,
                str(degraded)[:200],
            )
        result.metadata["validation_gaps"] = gaps

    def _validate_arguments(self, tool: Tool, call: ToolCall) -> dict[str, Any] | None:
        """步骤 4：参数校验。通过 -> ``None``；失败 -> 含 ``validation_errors`` 的 dict。

        ``call.arguments`` 里出现 ``"__raw__"`` 表示模型给的 arguments **根本不是合法
        JSON**（``ToolCall.try_from_arguments_json`` 的产物）。这时不能拿它去跑 schema
        校验（会产出一堆"缺字段"的噪声错误，掩盖真正的根因），而是直接把
        "must be a valid JSON object" + 原始串前 200 字符回灌 —— 模型看到自己写的东西
        才知道往哪儿改。
        """
        raw = call.arguments.get("__raw__") if isinstance(call.arguments, Mapping) else None
        if raw is not None:
            message = (
                f"tool call arguments for {call.name!r} must be a valid JSON object; "
                f"got: {str(raw)[:200]}"
            )
            return {"validation_errors": [message]}
        errors = validate_instance(dict(call.arguments), tool.parameters)
        if errors:
            return {"validation_errors": list(errors)}
        return None

    async def _approve(self, tool: Tool, call: ToolCall) -> ToolResult | None:
        """步骤 4.5：审批（HITL）。返回 ``None`` 表示放行；返回结果表示该直接返回。

        ``policy is None`` 且 ``requires_approval=True`` 时**一律拒绝**（fail-closed）：
        "没有审批人"绝不等于"审批通过"。这是安全默认值，不是可用性取舍。

        抛 ``AgentAbortedError`` 走**取消**路径（转成 ``asyncio.CancelledError``，由 Agent
        转 ``ABORTED``），而不是返回一个失败结果 —— 用户主动中止是一次控制流事件。
        """
        if not tool.spec.requires_approval:
            return None
        started = time.monotonic()
        policy = self.config.approval_policy
        approved = False
        if policy is not None:
            try:
                verdict = policy(call, tool)
                if inspect.isawaitable(verdict):
                    # ApprovalPolicy 的冻结签名是同步的，但真实项目里审批回调经常是
                    # async 的。若不 await 就直接 bool(coro)，协程对象恒为真值 ->
                    # "默认放行"，这是最危险的一种静默失效（安全方向）。
                    verdict = await verdict
                approved = bool(verdict)
            except AgentAbortedError:
                self._emit(
                    _EV_TOOL_APPROVAL,
                    tool_name=tool.name,
                    call_id=call.id,
                    approved=False,
                    aborted=True,
                    policy_present=True,
                )
                raise asyncio.CancelledError() from None
            except LiteAgentError as e:
                # 审批策略自身故障：**拒绝**（fail-closed）并把原因回灌。
                result = self._failure(
                    call,
                    e,
                    attempts=1,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    metadata={"approved": False, "feedback_kind": "infrastructure"},
                )
                return result
        self._emit(
            _EV_TOOL_APPROVAL,
            tool_name=tool.name,
            call_id=call.id,
            approved=approved,
            policy_present=policy is not None,
        )
        if not approved:
            err = ToolApprovalDeniedError(
                tool_name=tool.name,
                reason="no approval policy configured" if policy is None else "denied by policy",
            )
            return self._failure(
                call,
                err,
                attempts=1,
                duration_ms=(time.monotonic() - started) * 1000.0,
                metadata={"approved": False, "feedback_kind": "infrastructure"},
                feedback=False,  # 回灌文案是冻结字面量，不加后缀（见 _failure 的说明）
            )
        return None

    async def _invoke(
        self,
        tool: Tool,
        call: ToolCall,
        timeout: float | None,
        flag: threading.Event,
    ) -> Any:
        """真正把工具跑起来（§7.4.1 步骤 5.c，v2 修掉 M-2 与线程池归属）。

        - **异步工具**走 ``tool.arun``，与本函数同 loop。
        - **同步工具**走 ``tool.run``（**同步**入口，不是 ``arun``）提交到线程池。
          M-2 实测：``asyncio.to_thread(tool.arun, args)`` 不会执行 async 函数，
          它只会返回一个没人 await 的协程对象 —— 静默失效。

        SPEC-AMBIGUITY: §7.4.1 的伪代码是 ``loop.run_in_executor(tp, tool.run, args)``，
        并注释"contextvars 在此复制（含 cancel_scope 的 Event）"。实测（3.10.12）**不成立**：
        ``run_in_executor`` 不复制调用方的 context，worker 线程里
        ``current_cancel_flag()`` 恒为 None，协作式取消形同虚设。裁决：用
        ``contextvars.copy_context().run`` 包一层提交 —— 语义与伪代码完全一致，
        只是把注释里声称的行为**真正实现**出来（``tool.run`` 仍在 worker 线程里执行）。
        """
        args = dict(call.arguments)
        if tool.spec.is_async:
            coro: Any = tool.arun(args)
        else:
            tp = self._external_pool or self._pool.thread_pool(
                "exec", self.config.thread_pool_size
            )
            # 调用方线程提交；contextvars 在此复制（含 cancel_scope 的 Event）。
            ctx = copy_context()
            coro = asyncio.get_running_loop().run_in_executor(tp, ctx.run, tool.run, args)
        if timeout is None:
            return await coro
        # 超时只能取消 await 层（同步工具的 worker 线程仍在跑 -> orphan_thread）。
        # [v3] 不用 asyncio.wait_for：3.10 的它会在"内层 future 同 tick 已完成"时
        # 吞掉调用方的取消（GH-86296），让 astream 的 task.cancel() 变成空操作。
        return await _await_with_deadline(coro, timeout)

    @asynccontextmanager
    async def _seq_guard(self, name: str, tool: Tool) -> AsyncIterator[None]:
        """``sequential_tools`` 的跨调用互斥（§7.4.1 步骤 5.b）。

        **锁的临界区是"调用方可见的整段调用"**（这里到 ``finally``）：同步工具超时后
        orphan 线程可能仍持有底层资源（比如半个写了一半的文件），而锁已在调用方释放
        —— 这是**已知限制**，用结果的 ``orphan_thread=True`` 标记使其可见，而不是
        假装锁能保护到线程结束。

        用 ``threading.Lock`` 而不是 loop-bound ``asyncio.Lock``：``sequential_tools``
        的唯一理由是跨调用互斥（经典例子 ``run_shell``），而同步工具会被
        ``execute_many`` 丢进不同 worker 线程、每个线程里的 ``asyncio.run`` 都是**新 loop**
        —— loop-bound 锁跨 loop/跨线程完全不串行。

        等待时用 ``asyncio.to_thread`` 把阻塞的 ``lock.acquire()`` 丢进 worker 线程：
        直接 ``lock.acquire()`` 会阻塞事件循环，把"某个工具在跑 shell"变成"整个 loop 停摆"。

        `[v3 修正]` 取消安全：取消等锁者**不会**再让锁变成孤儿。线程通过
        ``_CancelSafeAcquire`` 拿到锁后若发现已取消，会立刻原地归还；因此"等待中被取消"
        之后，该工具的下一次调用仍能在毫秒级拿到锁，进程也能正常收尾（v2 会永久挂起）。
        触发取消的路径不止 fail_fast：外层 ``asyncio.wait_for``、``Agent.astream``
        提前 break 的 ``task.cancel()``、Ctrl-C、父任务取消都会命中。
        """
        lock = self._seq_locks.get(name)
        if lock is None:
            yield  # 不在 sequential_tools 里 -> 无锁
            return
        guard = _CancelSafeAcquire(lock)
        try:
            acquired = await asyncio.to_thread(guard.acquire)
        except asyncio.CancelledError:
            # 取消只取消了 await 层：worker 线程仍可能拿到锁 —— 交给 abort() 归还。
            guard.abort()
            raise
        if not acquired:  # pragma: no cover - abort() 路径必然先抛 CancelledError
            raise asyncio.CancelledError()
        try:
            yield
        finally:
            lock.release()

    # ----------------------------------------------------------------------------------
    # execute_many
    # ----------------------------------------------------------------------------------

    async def execute_many(
        self,
        calls: Sequence[ToolCall],
        *,
        concurrency: int | None = None,
        timeout_s: float | None = None,
    ) -> list[ToolResult]:
        """并发执行；**返回顺序与 calls 严格一致**（不看完成顺序）。

        ``concurrency=None`` -> 用 ``config.max_concurrency``；每个调用**同时**受全局
        信号量（步骤 5.b 的 ``"exec"``）约束。``timeout_s`` 是**逐调用**超时，与
        ``execute`` 的同名参数语义一致（"整批总预算"不在本参数范围内）。

        ``fail_fast=True`` 时：首个 ``ok=False`` 之后取消其余任务，**仍然返回与 calls
        等长的列表**，被取消的位置填 ``ToolSkippedError``（``metadata["skipped_by_fail_fast"]``）。

        **永不抛业务异常**（§13 红线 6 / §7.4.2）；唯一例外是 ``asyncio.CancelledError``
        与 ``KeyboardInterrupt``。
        """
        call_list = list(calls)
        if not call_list:
            return []

        limit = (
            self.config.max_concurrency
            if concurrency is None
            else _coerce_int(concurrency, self.config.max_concurrency)
        )
        if limit <= 0:
            # 参数非法不抛异常（红线 6），但**必须留痕**（红线 12）：otherwise
            # 调用方会以为自己拿到了一个无限的并发额度。
            logger.warning(
                "execute_many: concurrency=%r is not positive; falling back to "
                "max_concurrency=%d",
                concurrency,
                self.config.max_concurrency,
            )
            limit = self.config.max_concurrency
        # 批次级限流只在本函数存活期内有效（在运行中的 loop 里创建，不触 R-LOOP）；
        # 全局信号量仍然由 execute 内部持有，两层是**叠加**而不是替代。
        guard: asyncio.Semaphore | None = None
        if limit != self.config.max_concurrency:
            guard = asyncio.Semaphore(limit)

        # 批次级状态一律用局部变量（不要挂 self）：同一 executor 可能被并发共享。
        fail_fast_triggered = False

        async def _run_one(call: ToolCall) -> ToolResult:
            if guard is None:
                return await self.execute(call, timeout_s=timeout_s)
            async with guard:
                return await self.execute(call, timeout_s=timeout_s)

        futures = [asyncio.ensure_future(_run_one(c)) for c in call_list]
        try:
            if self.config.fail_fast:
                fail_fast_triggered = await self._watch_fail_fast(futures, call_list)
            # 必须 return_exceptions=True：用 False 时任一子任务异常会让其余结果
            # **全部丢失**、无法按下标对齐（§7.4.2 规则 1）。
            results = await asyncio.gather(*futures, return_exceptions=True)
        except asyncio.CancelledError:
            # 调用方取消了 execute_many 本身：把取消传播给所有子任务并收尾，
            # 避免留下仍在跑的孤儿任务（§7.4.2 规则 4 的"必须收尾"）。
            for fut in futures:
                if not fut.done():
                    fut.cancel()
            try:
                # return_exceptions=True：子任务自身的异常不会在这里再炸一次
                # （它们已经被编码成结果）。这里只可能被**第二次**取消打断，
                # 那是正常现象 —— 记一条 debug 后原样上抛，绝不静默吞掉（红线 10）。
                await asyncio.gather(*futures, return_exceptions=True)
            except asyncio.CancelledError:
                logger.debug(
                    "execute_many: second cancellation arrived during cleanup; "
                    "all sibling tasks are already cancelled"
                )
                raise
            raise
        finally:
            for fut in futures:
                if not fut.done():
                    fut.cancel()

        out: list[ToolResult] = []
        for index, item in enumerate(results):
            call = call_list[index]
            if isinstance(item, ToolResult):
                out.append(item)
            elif isinstance(item, asyncio.CancelledError):
                # fail_fast 取消的兄弟调用：占位，保证下标一一对应。
                out.append(
                    ToolResult.failure(
                        call,
                        ToolSkippedError(tool_name=call.name, reason="cancelled_by_fail_fast"),
                        metadata={"skipped_by_fail_fast": True, "attempts": 1},
                    )
                )
            else:
                # 防御性分支：execute 已保证"除 CancelledError 外一切异常都编码为
                # ToolResult"，所以走到这里只可能是实现有 bug。
                logger.error(
                    "executor: execute() leaked %r for call %s; this is a bug",
                    type(item).__name__,
                    call.name,
                )
                out.append(
                    ToolResult.failure(call, item, metadata={"attempts": 1})  # type: ignore[arg-type]
                )

        # 规则 3：若结果里有 CancelledError 而**不是** fail_fast 引起的（即调用方取消了
        # execute_many 本身），取消传播优先于结果收集 —— 重新抛出第一条。
        if not fail_fast_triggered:
            for item in results:
                if isinstance(item, asyncio.CancelledError):
                    raise item
        return out

    async def _watch_fail_fast(
        self, futures: list["asyncio.Future[ToolResult]"], calls: list[ToolCall]
    ) -> bool:
        """fail_fast 的监视器：首个 ``ok=False`` 出现 -> 取消其余未完成任务（§7.4.2 规则 4）。

        返回值是**本批次**是否触发过 fail_fast（调用方用局部变量接收，见 execute_many；
        绝不能存在实例属性上，否则并发的两个批次会互相污染）。
        完成顺序用 ``asyncio.wait(FIRST_COMPLETED)`` 观察；同一批唤醒里多个任务都完成时，
        按 **calls 的下标**取胜者（确定性的 tie-break，否则"首个失败"在 trace 里会抖）。
        取消之后**必须**由调用方的 ``gather`` 收尾，这里不做（否则会 await 两次）。
        """
        pending: set[Any] = set(futures)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            hit: Any = None
            for fut in sorted(done, key=futures.index):
                if fut.cancelled():
                    continue
                if fut.exception() is not None:
                    continue  # 只可能是 CancelledError（execute 内部已归一化）
                res = fut.result()
                if isinstance(res, ToolResult) and not res.ok:
                    hit = fut
                    break
            if hit is None:
                continue
            index = futures.index(hit)
            skipped = 0
            for fut in pending:
                if fut.cancel():
                    skipped += 1
            self._emit(
                _EV_TOOL_ERROR,
                tool_name=calls[index].name,
                call_id=calls[index].id,
                fail_fast_first_index=index,
                skipped=skipped,
            )
            return True
        return False

    # ----------------------------------------------------------------------------------
    # 同步入口与资源释放
    # ----------------------------------------------------------------------------------

    def execute_sync(self, call: ToolCall, **kwargs: Any) -> ToolResult:
        """``execute`` 的同步镜像。

        走 ``config.run_sync``：它在 ``asyncio.run`` 返回后的 ``finally`` 里调用
        ``LoopBoundPool.release_loop(loop)`` —— 这就是本文件"在 ``finally`` 里调用
        ``self._pool.release(loop)``"的落点（§5.2 已代办，不必也不能在这里重复做：
        重复 release 会把别的实例刚建好的池也关掉）。在运行中的 loop 内调用抛
        ``ConfigError``（run_sync 的第一条契约），绝不套娃 ``run_until_complete``。
        """
        return run_sync(functools.partial(self.execute, call, **kwargs))

    async def aclose(self) -> None:
        """``self._pool.release(None)``：shutdown 全部私有线程池
        （``wait=False, cancel_futures=True``）。外部传入的 ``thread_pool`` 不关闭
        —— 所有权是调用方的，替别人关池会让后续调用静默失败。

        **不** shutdown loop 的默认 executor（``_seq_guard`` 的取消安全等锁在用它，
        但由 ``asyncio.run`` 自己的收尾负责；v2 的注释"已不再使用它"是错的）。
        """
        self._pool.release(None)

    def close(self) -> None:
        """``aclose`` 的同步镜像。不需要运行中的 loop（release 是纯同步操作），
        因此这里直接调用而**不**经 ``run_sync`` —— 后者在 loop 内会抛 ``ConfigError``，
        让"在 finally 里清理"变得不可用。"""
        self._pool.release(None)

    async def __aenter__(self) -> "ToolExecutor":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __enter__(self) -> "ToolExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<ToolExecutor tools={len(self.registry)} "
            f"max_concurrency={self.config.max_concurrency} "
            f"fail_fast={self.config.fail_fast}>"
        )
