from __future__ import annotations

# ``tests/test_tools_executor.py`` —— ``liteagent/tools/executor.py`` 的单元测试。
#
# 覆盖重点逐字来自 ``docs/INTERFACES.md`` §12 第 5115 行（工具层两个最重的文件之一）：
#
# * 并发顺序保持、``max_concurrency`` 上限（计数工具断言峰值）、同步工具**真的执行了**
#   （``tool.run`` 被调用，而不是返回一个没人 await 的 coroutine，M-2）；
# * **同步工具超时**：``attempts == 1`` 且 ``metadata["orphan_thread"] is True``
#   且用 ``threading.enumerate()`` 快照证明 worker 线程**仍然活着**；
# * **异步工具超时仍可重试**（与同步工具区别对待，§3.4 的 v2 例外）；
# * 重试次数与退避（``jitter=0`` + ``RecordingSleep`` 断言 ``delays`` 序列）、
#   ``retryable=False`` 不重试、``idempotent=False`` 不重试、验证失败不消耗重试次数；
# * ``ToolNotFoundError`` 返回可用工具名、结果截断、``_stringify`` 各类型；
# * ``sequential_tools`` 在两个嵌套 loop（普通调用 + delegate 式内层 loop）下仍真正串行；
# * ``execute_many`` 顺序、``fail_fast`` 取消兄弟任务后结果列表仍与 calls 对齐且不抛异常；
# * ``execute_sync`` 在 loop 内抛 ``ConfigError``；
# * **跨两次 ``asyncio.run`` 复用同一 executor（守 §0.4 R-LOOP）且 ``len(executor._pool) == 0``
#   不泄漏**；
# * 审批三例（无 policy 拒绝 / policy 返回 False / policy 返回 True 正常执行）；
# * **熔断**（连续 3 次 infrastructure 失败后第 4 次不执行且 ``disabled=True``）；
# * 两类回灌文案（recoverable / infrastructure）。
#
# 测试卫生（§12.1 冻结规则）：
# 1. 零网络（没有任何工具发起真实 I/O）、零真实时钟依赖、零 ``asyncio.sleep`` 式的
#    "等一会儿看看"；退避一律走 ``tests.helpers.RecordingSleep``。
#    工具函数内部的 ``time.sleep(0.02)`` / ``await asyncio.sleep(0.02)`` 是**被测行为**
#    （"这个工具要跑 20ms"），不是测试侧的等待 —— 没有它就无法观测并发峰值。
# 2. 本文件不碰默认注册表（没有 ``auto_register``），因此不需要
#    ``reset_default_registry()``。

import asyncio
import base64
import contextlib
import dataclasses
import json
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from unittest import mock

from liteagent.config import ExecutorConfig, NO_TIMEOUT, RetryPolicy
from liteagent.errors import (
    AgentAbortedError,
    ConfigError,
    LiteAgentError,
    SandboxViolationError,
    ToolExecutionError,
)
from liteagent.tools.base import Tool, tool
from liteagent.tools.executor import ToolExecutor, _stringify, _stringify_extras
from liteagent.tools.registry import ToolRegistry
from liteagent.tools.schema import unvalidatable_parameters, validate_instance
from liteagent.types import ToolCall, ToolResult
from tests.helpers import (
    RecordingSleep,
    add as add_tool,
    make_registry,
)


# ======================================================================================
# 测试夹具状态（每个用例在 setUp 里重置）
# ======================================================================================

_SYNC_CALLS: list[str] = []
_ASYNC_CALLS: list[str] = []
_FAIL_CALLS: list[str] = []
_PEAK: dict[str, int] = {"current": 0, "peak": 0}
_PEAK_LOCK = threading.Lock()
_SCRIPT: dict[str, list[str]] = {}

#: 同步工具的"闸门"：只有测试放开它，阻塞中的同步工具才会返回。
#: 用它替代 ``time.sleep(30)`` —— 既能让超时必然发生，又不会真的拖慢测试。
_GATE = threading.Event()


def _reset_fixtures() -> None:
    _SYNC_CALLS.clear()
    _ASYNC_CALLS.clear()
    _FAIL_CALLS.clear()
    _SCRIPT.clear()
    _PEAK["current"] = 0
    _PEAK["peak"] = 0
    _GATE.clear()


def _enter_peak() -> None:
    with _PEAK_LOCK:
        _PEAK["current"] += 1
        _PEAK["peak"] = max(_PEAK["peak"], _PEAK["current"])


def _leave_peak() -> None:
    with _PEAK_LOCK:
        _PEAK["current"] -= 1


# ======================================================================================
# 测试专用异常（**不是** liteagent 的新异常类型）
# ======================================================================================


class _RetryableTestError(LiteAgentError):
    """把 ``retryable`` 显式置 True，用于驱动"可重试失败"这条分支。

    liteagent 自己的异常里，可重试的工具类异常只有 ``ToolTimeoutError``，
    而它同时被 §3.4 的 v2 例外（同步工具不重试）拦着。要单独观察退避序列、
    ``retry_after_s``、重试耗尽，就必须有一个可重试的非超时异常。
    """

    retryable = True


# ======================================================================================
# 被测工具（模块级定义：装饰器在 import 期一次性完成反射，import 零副作用）
# ======================================================================================


@tool
def record_sync(tag: str = "t") -> str:
    """Record a synchronous execution and return the tag."""
    _SYNC_CALLS.append(tag)
    return tag


@tool
def return_thread_id(tag: str = "t") -> str:
    """Return the id of the thread this sync tool really ran in."""
    _SYNC_CALLS.append(tag)
    return str(threading.get_ident())


@tool
async def record_async(tag: str = "t", delay: float = 0.0) -> str:
    """Record an asynchronous execution and return the tag."""
    _ASYNC_CALLS.append(tag)
    if delay:
        await asyncio.sleep(delay)
    return tag


@tool
def block_on_gate(tag: str = "t") -> str:
    """Block until the test releases ``_GATE`` (orphan-thread fixture)."""
    _GATE.wait(20.0)
    return tag


@tool
async def async_timeout_tool(delay: float = 1.0) -> str:
    """Sleep longer than the configured timeout (async timeout fixture)."""
    await asyncio.sleep(delay)
    return "late"


@tool(timeout_s=0.05)
async def spec_timeout_tool(delay: float = 1.0) -> str:
    """A tool that carries its own ``timeout_s`` in its spec."""
    await asyncio.sleep(delay)
    return "late"


@tool
def counted_sync(tag: str = "t", delay: float = 0.02) -> str:
    """Track the concurrency peak of sync executions (worker threads)."""
    _enter_peak()
    try:
        time.sleep(delay)
    finally:
        _leave_peak()
    return tag


@tool
async def counted_async(tag: str = "t", delay: float = 0.02) -> str:
    """Track the concurrency peak of async executions."""
    _enter_peak()
    try:
        await asyncio.sleep(delay)
    finally:
        _leave_peak()
    return tag


@tool
def nested_loop_tool(tag: str = "t") -> str:
    """A delegate-style sync tool: it runs its **own** ``asyncio.run`` inside."""
    _enter_peak()
    try:

        async def _inner() -> None:
            await asyncio.sleep(0.05)

        asyncio.run(_inner())
    finally:
        _leave_peak()
    return tag


@tool
def raise_typed_error(kind: str = "execution") -> str:
    """Raise a specific exception kind (exception-path fixture)."""
    if kind == "value":
        raise ValueError("plain python failure")
    if kind == "sandbox":
        raise SandboxViolationError(path="/etc/shadow", root="/tmp/sandbox")
    if kind == "subprocess":
        raise subprocess.TimeoutExpired(cmd="sleep 5", timeout=1)
    raise ToolExecutionError(tool_name="raise_typed_error", call_id="c")


@tool
def flaky(fail_times: int = 2, retry_after_s: float = 0.0) -> str:
    """Fail with a retryable error until it has failed ``fail_times`` times."""
    _FAIL_CALLS.append("flaky")
    if len(_FAIL_CALLS) <= fail_times:
        exc = _RetryableTestError("flaky failure")
        if retry_after_s:
            exc.retry_after_s = retry_after_s
        raise exc
    return "recovered"


@tool(idempotent=False, max_retries=3)
def non_idempotent_tool(stub: int = 0) -> str:
    """Always fails with a retryable error (non-idempotent fixture)."""
    _FAIL_CALLS.append("non_idempotent")
    raise _RetryableTestError("non-idempotent failure")


@tool(max_retries=2)
def spec_retries_tool(stub: int = 0) -> str:
    """Always fails with a retryable error (spec-level max_retries fixture)."""
    _FAIL_CALLS.append("spec_retries")
    raise _RetryableTestError("spec-level retries failure")


def _scripted_outcome(script_key: str, tool_name: str) -> str:
    """弹出 ``_SCRIPT[script_key]`` 的下一个结果："ok" / "fail"。"""
    outcomes = _SCRIPT.get(script_key) or []
    outcome = outcomes.pop(0) if outcomes else "ok"
    if outcome == "fail":
        _FAIL_CALLS.append(script_key)
        raise ToolExecutionError(tool_name=tool_name)
    return "ok"


@tool
def scripted_tool(script_key: str = "default") -> str:
    """Pop the next outcome from ``_SCRIPT[script_key]`` ("ok" / "fail")."""
    return _scripted_outcome(script_key, "scripted_tool")


@tool
def scripted_tool_b(script_key: str = "default") -> str:
    """A second scripted tool (per-tool电路-breaker isolation fixture)."""
    return _scripted_outcome(script_key, "scripted_tool_b")


@tool
def long_result(size: int = 200) -> str:
    """Return a long deterministic string (truncation fixture)."""
    return ("abcdefghij" * (size // 10 + 1))[:size]


@tool
def return_tool_result(tag: str = "t") -> ToolResult:
    """Return a ToolResult directly (the §7.4.3 merging branch)."""
    inner = ToolCall(id="inner-call", name="inner-name")
    return ToolResult.success(inner, "inner-content", metadata={"inner_key": tag})


@tool(requires_approval=True)
def needs_approval(x: int = 0) -> int:
    """A tool that requires human approval before it runs."""
    _SYNC_CALLS.append("needs_approval")
    return x


@tool
def echo_args(a: int = 0) -> int:
    """Echo the integer argument back (validation fixture)."""
    return a


@tool
def degraded_param(value: Union[int, str] = 1) -> str:
    """Echo a parameter whose schema cannot be described (validation-coverage fixture).

    Args:
        value: a union type — `annotation_to_schema` degrades it to `{}` (any),
            and `_function_to_schema` still pastes this docstring onto the fragment.
    """
    return f"{value!r}"


@tool
def only_optional(first: str = "x") -> str:
    """All parameters have defaults (empty-arguments-bypass fixture)."""
    return first


@tool
async def cancel_lingering(tag: str = "t", linger: float = 0.3) -> str:
    """A tool that lingers in its cancellation cleanup (fail_fast window fixture).

    真实的 async 工具在 ``finally: await cleanup()`` 里天然会延长取消响应时间，
    因此"被 fail_fast 取消后还要 0.3s 才真正结束"是合法行为，不是伪造的时序。
    它把 fail_fast 的竞争窗口从"一两个 event-loop tick"拉宽到可稳定复现的量级。
    """
    try:
        await asyncio.sleep(10.0)
    except asyncio.CancelledError:
        await asyncio.sleep(linger)
        raise
    return tag


@tool
def block_on_gate_seq(tag: str = "t") -> str:
    """Block until the test releases ``_GATE``（sequential_tools 取消用例的持有者）。"""
    _GATE.wait(20.0)
    return tag


_ALL_TOOLS: tuple[Tool, ...] = (
    record_sync,
    return_thread_id,
    record_async,
    block_on_gate,
    async_timeout_tool,
    spec_timeout_tool,
    counted_sync,
    counted_async,
    nested_loop_tool,
    raise_typed_error,
    flaky,
    non_idempotent_tool,
    spec_retries_tool,
    scripted_tool,
    scripted_tool_b,
    long_result,
    return_tool_result,
    needs_approval,
    echo_args,
    cancel_lingering,
    block_on_gate_seq,
)

#: 一个永远有内容的注册表（每个用例自己造一个新的，避免互相污染）。
def _registry() -> ToolRegistry:
    return ToolRegistry(_ALL_TOOLS)


def _make_executor(registry: ToolRegistry | None = None, **config_kwargs: Any) -> ToolExecutor:
    """构造执行器；默认 ``max_retries=0`` + ``jitter=0``（确定性优先）。"""
    retry = config_kwargs.pop("retry_policy", None)
    if retry is None:
        retry = RetryPolicy(max_retries=0, jitter=0.0, backoff_base_s=0.1)
    return ToolExecutor(
        registry if registry is not None else _registry(),
        ExecutorConfig(retry_policy=retry, **config_kwargs),
    )


class _ExecutorTestCase(unittest.TestCase):
    """同步用例基类：隔离夹具 + 保证闸门一定被放开（否则 worker 线程要等 20s）。"""

    def setUp(self) -> None:
        super().setUp()
        _reset_fixtures()

    def tearDown(self) -> None:
        _GATE.set()
        super().tearDown()


class _AsyncExecutorTestCase(unittest.IsolatedAsyncioTestCase):
    """异步用例基类（§12 硬性要求：异步用例继承 ``IsolatedAsyncioTestCase``）。"""

    def setUp(self) -> None:
        super().setUp()
        _reset_fixtures()

    async def asyncTearDown(self) -> None:
        _GATE.set()
        await super().asyncTearDown()


# ======================================================================================
# 基础：成功路径、返回不变式、同步工具真的执行了
# ======================================================================================


class ExecuteBasicsTests(_AsyncExecutorTestCase):
    """``execute`` 的返回不变式（§7.4.1 步骤 7）。"""

    async def test_success_result_invariants(self) -> None:
        call = ToolCall.create("record_sync", {"tag": "hello"})
        async with _make_executor() as executor:
            result = await executor.execute(call)

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.call_id, call.id)
        self.assertEqual(result.name, call.name)
        self.assertGreaterEqual(result.attempts, 1)
        self.assertEqual(result.content, "hello")
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_type)
        self.assertIn("hello", _SYNC_CALLS)

    async def test_failure_content_starts_with_error_prefix(self) -> None:
        """`ok is False` -> content 非空且以 "ERROR(" 开头（冻结不变式）。"""
        call = ToolCall.create("raise_typed_error", {"kind": "value"})
        async with _make_executor() as executor:
            result = await executor.execute(call)

        self.assertFalse(result.ok)
        self.assertNotEqual(result.content, "")
        self.assertTrue(result.content.startswith("ERROR("), msg=result.content)
        self.assertEqual(result.call_id, call.id)
        self.assertEqual(result.name, call.name)
        self.assertGreaterEqual(result.attempts, 1)

    async def test_plain_python_exception_is_wrapped(self) -> None:
        """工具内部的普通异常 -> ``ToolExecutionError``（不向外抛）。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "value"})
            )
        self.assertEqual(result.error_type, "ToolExecutionError")
        self.assertEqual(result.metadata["feedback_kind"], "infrastructure")
        self.assertIn("plain python failure", result.error or "")

    async def test_sandbox_violation_keeps_its_type(self) -> None:
        """``SandboxViolationError`` 原样保留（§7.4.1 步骤 5.e）。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "sandbox"})
            )
        self.assertEqual(result.error_type, "SandboxViolationError")

    async def test_subprocess_timeout_maps_to_tool_timeout(self) -> None:
        """``subprocess.TimeoutExpired`` -> ``ToolTimeoutError``（超时只有一个类型出口）。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "subprocess"})
            )
        self.assertEqual(result.error_type, "ToolTimeoutError")

    async def test_duration_ms_is_non_negative(self) -> None:
        """§2.8：不取具体值，只断言它是个合法的非负数。"""
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": 1}))
        self.assertGreaterEqual(result.duration_ms, 0.0)

    async def test_sync_tool_really_runs_via_sync_entry(self) -> None:
        """M-2：同步工具必须走 ``tool.run``（同步入口），**不是**返回一个 coroutine。"""
        call = ToolCall.create("record_sync", {"tag": "ran"})
        async with _make_executor() as executor:
            with mock.patch.object(record_sync, "run", wraps=record_sync.run) as spy:
                result = await executor.execute(call)

        self.assertEqual(spy.call_count, 1)
        self.assertEqual(result.content, "ran")
        self.assertNotIn("coroutine", result.content)
        self.assertFalse(asyncio.iscoroutine(result.content))

    async def test_sync_tool_runs_in_a_worker_thread(self) -> None:
        """同步工具跑在 worker 线程里：返回的线程 id 不等于**发起调用**的线程 id。"""
        loop_thread_id = str(threading.get_ident())
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("return_thread_id", {"tag": "x"}))
        self.assertTrue(result.ok)
        self.assertNotEqual(result.content, loop_thread_id)

    async def test_async_tool_runs_on_the_same_loop(self) -> None:
        """异步工具与调用方同 loop（因此不需要线程池）。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("record_async", {"tag": "async-ran"})
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "async-ran")
        self.assertIn("async-ran", _ASYNC_CALLS)

    async def test_tool_returning_tool_result_merges_metadata(self) -> None:
        """§7.4.3：返回 ToolResult 时取它的 content 并合并 metadata。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("return_tool_result", {"tag": "merged"})
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "inner-content")
        self.assertEqual(result.metadata["inner_key"], "merged")

    async def test_aclose_is_idempotent(self) -> None:
        executor = _make_executor()
        await executor.execute(ToolCall.create("record_sync", {"tag": "x"}))
        await executor.aclose()
        await executor.aclose()  # 第二次不该炸
        self.assertEqual(len(executor._pool), 0)


# ======================================================================================
# 找不到工具 / 参数校验 / 回灌文案
# ======================================================================================


class NotFoundAndValidationTests(_AsyncExecutorTestCase):
    """§7.4.1 步骤 1（找不到工具）与步骤 4（校验失败）。"""

    async def test_unknown_tool_lists_available_names(self) -> None:
        """content 里必须列出可用工具名（模型靠它自纠正），metadata 标 recoverable。"""
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("does_not_exist", {}))

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolNotFoundError")
        self.assertEqual(result.metadata["feedback_kind"], "recoverable")
        self.assertGreaterEqual(result.attempts, 1)
        self.assertIn("record_sync", result.content)
        self.assertIn("does_not_exist", result.content)

    async def test_unknown_tool_is_not_retried(self) -> None:
        """名字写错不消耗重试预算（重试 3 次也还是同一个错名）。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3)
        ) as executor:
            result = await executor.execute(ToolCall.create("does_not_exist", {}))
        self.assertEqual(result.attempts, 1)
        self.assertEqual(sleep.delays, [])

    async def test_validation_failure_reports_errors(self) -> None:
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": "bad"}))

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolValidationError")
        self.assertEqual(result.metadata["feedback_kind"], "recoverable")
        self.assertTrue(result.metadata["validation_errors"])
        self.assertIn("expected integer", result.metadata["validation_errors"][0])

    async def test_validation_failure_does_not_consume_retries(self) -> None:
        """冻结：校验失败 ``attempts=1``，**不消耗重试预算**。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3)
        ) as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": "bad"}))
        self.assertEqual(result.attempts, 1)
        self.assertEqual(sleep.delays, [])

    async def test_raw_arguments_feedback_mentions_json_object(self) -> None:
        """模型给的 arguments 不是合法 JSON -> 回灌原文前 200 字符 + 固定提示串。"""
        call = ToolCall(
            id="call_raw",
            name="echo_args",
            arguments={"__raw__": "{not json"},
            raw_arguments="{not json",
        )
        async with _make_executor() as executor:
            result = await executor.execute(call)

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolValidationError")
        self.assertIn("must be a valid JSON object", result.content)
        self.assertIn("{not json", result.content)

    async def test_recoverable_feedback_text(self) -> None:
        """recoverable 类失败的回灌文案必须告诉模型**怎么改**（§3.4 第二层分类）。"""
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": "bad"}))
        self.assertIn(
            "fix the arguments to match the tool's JSON schema", result.content
        )
        self.assertNotIn("do not retry this tool with the same arguments", result.content)

    async def test_infrastructure_feedback_text(self) -> None:
        """infrastructure 类失败的回灌文案必须劝模型**换策略**。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "value"})
            )
        self.assertIn(
            "do not retry this tool with the same arguments", result.content
        )
        self.assertIn("try a different approach or give your final answer", result.content)
        self.assertNotIn(
            "fix the arguments to match the tool's JSON schema", result.content
        )

    async def test_validation_error_type_is_recoverable_not_infrastructure(self) -> None:
        """`feedback_kind` 决定回灌文案，也决定熔断计数是否累加（见熔断用例）。"""
        async with _make_executor() as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": "bad"}))
        self.assertEqual(result.metadata["feedback_kind"], "recoverable")

    async def test_extra_argument_is_rejected_by_additional_properties(self) -> None:
        """生成的 schema 带 ``additionalProperties: false`` -> 多余参数会被拦下。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("echo_args", {"a": 1, "b": 2})
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolValidationError")


# ======================================================================================
# [v3] 执行前校验的**覆盖率缺口**必须留痕
# ======================================================================================


class ValidationCoverageGapTests(_AsyncExecutorTestCase):
    """[v3 回归] "这个参数/这次调用根本没被执行前校验覆盖"必须可观测（§13 红线 12）。

    审计发现的路径（两条都是 `{}` 降级，都会让"执行前入参校验"这个卖点**静默失效**）：

    1. **参数 schema 降级成 `{}`(any)**：`annotation_to_schema` 对无法精确表达的注解
       （`Union[A, B]` / `Mapping[...]` / `Callable[...]` …）的兜底产物
       （README 的"边界"一节写明了它恒真，但**只写在文档里**，运行时无信号）。
    2. **arguments 降级成 `{}`**：provider 侧 JSON 解析失败时
       （`providers.py::_parse_inline_arguments` 返回 `{}` + `metadata["parse_error"]`），
       整份参数被丢成空 dict；required 字段还有机会拦，**全 optional 的工具会以空参数执行**。

    修复把两者钉成 `ToolResult.metadata["validation_gaps"]` + WARNING。本组用例同时
    钉住**反方向**（没有降级就不能写这个 key），否则"无条件写一份 gap"也能通过。
    """

    async def test_degraded_parameter_is_flagged_in_metadata(self) -> None:
        async with _make_executor(ToolRegistry([degraded_param])) as executor:
            result = await executor.execute(
                ToolCall.create("degraded_param", {"value": {"anything": True}})
            )
        self.assertTrue(result.ok, msg=result.content)
        gaps = result.metadata["validation_gaps"]
        self.assertEqual(["value"], gaps["unvalidated_parameters"])
        self.assertIn("cannot be validated before execution", gaps["note"])

    async def test_degraded_parameter_really_skips_validation(self) -> None:
        """前提自证：那个参数确实**恒真** —— 否则上一条断言只是在描述一个假设。

        `degraded_param` 的片段里只有 `description`（docstring 贴上去的），
        一个校验关键字都没有：`unvalidatable_parameters` 认出来，`validate_instance`
        对它放行任何值。这条用例把"判据"和"后果"一起钉死。
        """
        fragment = degraded_param.parameters["properties"]["value"]
        self.assertEqual({"description": fragment.get("description")}, fragment)
        self.assertEqual(["value"], unvalidatable_parameters(degraded_param.parameters))
        self.assertEqual(
            [],
            validate_instance({"value": {"totally": "wrong type"}},
                              degraded_param.parameters),
        )

    async def test_gap_warning_is_emitted_once_per_tool(self) -> None:
        """结构性缺口是常量：每个工具只发一次 WARNING，不能逐次刷屏。"""
        with self.assertLogs("liteagent.tools.executor", level="WARNING") as captured:
            async with _make_executor(ToolRegistry([degraded_param])) as executor:
                await executor.execute(ToolCall.create("degraded_param", {"value": 1}))
                await executor.execute(ToolCall.create("degraded_param", {"value": 2}))
        hits = [line for line in captured.output if "CANNOT be validated" in line]
        self.assertEqual(1, len(hits), msg=captured.output)

    async def test_healthy_tool_gets_no_validation_gaps_key(self) -> None:
        """反方向：schema 完好时**不得**写 `validation_gaps`。"""
        async with _make_executor(ToolRegistry([echo_args])) as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": 3}))
        self.assertTrue(result.ok, msg=result.content)
        self.assertNotIn("validation_gaps", result.metadata)

    async def test_provider_degraded_arguments_bypass_validation_but_are_flagged(self) -> None:
        """**审计点名的那条路径**：arguments 降级成 `{}`，全 optional 的工具照跑。

        `only_optional` 的所有参数都有默认值 -> `validate_instance({}, ...) == []`
        -> 校验通过 -> 工具以**空参数**执行。这就是"绕过执行前入参校验"的现场：
        本用例先钉住"它确实跑了"（bug 的后果），再钉住"降级在 metadata 与日志里可见"
        （修复）。两者缺一，这条用例就退化成"只测了个布尔量"。
        """
        call = ToolCall.create("only_optional", {})
        call.metadata["parse_error"] = (
            "no JSON object found after 'use:': 'use:only_optional {oops'"
        )
        with self.assertLogs("liteagent.tools.executor", level="WARNING") as captured:
            async with _make_executor(ToolRegistry([only_optional])) as executor:
                result = await executor.execute(call)
        self.assertTrue(result.ok, msg=result.content)          # 空参数确实执行了
        self.assertEqual("x", result.content)                    # 用的是默认值
        gaps = result.metadata["validation_gaps"]
        self.assertIs(True, gaps["arguments_degraded"])
        self.assertIn("no JSON object found", gaps["arguments_degraded_reason"])
        self.assertTrue(
            any("cannot check them" in line for line in captured.output),
            msg=captured.output,
        )

    async def test_provider_degraded_arguments_with_required_field_fail_validation(self) -> None:
        """同一条路径上 required 字段仍然能拦 —— 缺口是"可见"而不是"放弃校验"。"""
        call = ToolCall.create("add", {})
        call.metadata["parse_error"] = "expected a JSON object, got str"
        async with _make_executor(ToolRegistry([add_tool])) as executor:
            result = await executor.execute(call)
        self.assertFalse(result.ok)
        self.assertEqual("ToolValidationError", result.error_type)
        self.assertEqual("recoverable", result.metadata["feedback_kind"])
        self.assertIs(True, result.metadata["validation_gaps"]["arguments_degraded"])


# ======================================================================================
# 超时：同步孤儿线程 vs 异步可重试
# ======================================================================================


class TimeoutTests(_AsyncExecutorTestCase):
    """§7.4.1 步骤 5.c 的取消语义与 §3.4 的 v2 例外。"""

    async def test_sync_timeout_is_orphan_thread_and_not_retried(self) -> None:
        """**全项目最容易出 bug 的一条**：同步工具超时 = 孤儿线程 + 不重试。

        理由（§3.4）：``asyncio.wait_for`` 无法中断 worker 线程里的同步代码；
        重试会**再起一个线程**，对 ``write_file`` / ``run_shell`` 意味着两个线程
        同时写同一份资源。
        """
        sleep = RecordingSleep()
        async with _make_executor(
            default_timeout_s=0.05, sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3)
        ) as executor:
            result = await executor.execute(ToolCall.create("block_on_gate", {"tag": "x"}))

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolTimeoutError")
        # attempts == 1：即使 max_retries=3 也只尝试一次
        self.assertEqual(result.attempts, 1)
        # 没有退避 -> 证明真的没重试
        self.assertEqual(sleep.delays, [])
        self.assertIs(result.metadata["orphan_thread"], True)
        self.assertEqual(result.metadata["feedback_kind"], "infrastructure")

    async def test_orphan_thread_is_still_alive_after_timeout(self) -> None:
        """用 ``threading.enumerate()`` 快照证明线程**仍然活着**（绝不假装它停了）。"""
        async with _make_executor(default_timeout_s=0.05) as executor:
            before = {thread.ident for thread in threading.enumerate()}
            result = await executor.execute(ToolCall.create("block_on_gate", {"tag": "y"}))
            after_snapshot = list(threading.enumerate())

        self.assertIs(result.metadata["orphan_thread"], True)
        self.assertFalse(_GATE.is_set())

        newcomers = [t for t in after_snapshot if t.ident not in before]
        self.assertTrue(newcomers, "expected the orphaned worker thread to still exist")
        self.assertTrue(all(t.is_alive() for t in newcomers))
        self.assertTrue(
            any("liteagent-exec" in (t.name or "") for t in newcomers),
            msg=[t.name for t in newcomers],
        )

    async def test_async_tool_timeout_is_retried(self) -> None:
        """异步工具无孤儿线程，超时**允许重试**（与同步工具区别对待，§3.4）。"""
        sleep = RecordingSleep()
        async with _make_executor(
            default_timeout_s=0.05,
            sleep_fn=sleep,
            retry_policy=RetryPolicy(max_retries=2, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(
                ToolCall.create("async_timeout_tool", {"delay": 1.0})
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.error_type, "ToolRetryExhaustedError")
        self.assertEqual(result.metadata["last_error_type"], "ToolTimeoutError")
        self.assertNotIn("orphan_thread", result.metadata)
        self.assertEqual(sleep.delays, [0.1, 0.2])

    async def test_async_timeout_success_after_retry(self) -> None:
        """异步工具第一次超时、第二次成功 -> 整体 ok=True，且 attempts=2。"""

        @tool
        async def slow_then_fast(delay: float = 1.0) -> str:
            """Timeout on the first call, succeed on the second."""
            _FAIL_CALLS.append("slow_then_fast")
            if len(_FAIL_CALLS) == 1:
                await asyncio.sleep(delay)
            return "fast"

        registry = ToolRegistry([slow_then_fast])
        async with _make_executor(
            registry,
            default_timeout_s=0.05,
            sleep_fn=RecordingSleep(),
            retry_policy=RetryPolicy(max_retries=2, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(
                ToolCall.create("slow_then_fast", {"delay": 1.0})
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "fast")
        self.assertEqual(result.attempts, 2)

    async def test_no_timeout_sentinel_disables_the_timeout(self) -> None:
        """``NO_TIMEOUT``：v1 的超时链里没有任何值能表达"不超时"。"""
        async with _make_executor(default_timeout_s=0.05) as executor:
            result = await executor.execute(
                ToolCall.create("record_async", {"tag": "no-timeout", "delay": 0.05}),
                timeout_s=NO_TIMEOUT,
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "no-timeout")

    async def test_config_none_disables_the_timeout_globally(self) -> None:
        async with _make_executor(default_timeout_s=None) as executor:
            result = await executor.execute(
                ToolCall.create("record_async", {"tag": "global", "delay": 0.02})
            )
        self.assertTrue(result.ok)

    async def test_timeout_resolution_takes_the_minimum(self) -> None:
        """参数 > call.metadata > spec > config，取**最小有效值**（§7.4.1 步骤 2）。"""
        # spec.timeout_s=0.05 比 config.default_timeout_s=30.0 小 -> 用 spec 的
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("spec_timeout_tool", {"delay": 1.0})
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolTimeoutError")

    async def test_call_metadata_timeout_wins_when_smaller(self) -> None:
        call = ToolCall(
            id="call_meta_timeout",
            name="record_async",
            arguments={"tag": "meta", "delay": 1.0},
            metadata={"timeout_s": 0.05},
        )
        async with _make_executor() as executor:
            result = await executor.execute(call)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolTimeoutError")

    async def test_explicit_timeout_argument_wins_when_smaller(self) -> None:
        async with _make_executor(default_timeout_s=30.0) as executor:
            result = await executor.execute(
                ToolCall.create("record_async", {"tag": "arg", "delay": 1.0}),
                timeout_s=0.05,
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolTimeoutError")

    async def test_cancel_flag_is_visible_inside_the_sync_tool(self) -> None:
        """协作式取消：executor 在超时时 ``flag.set()``，worker 线程能读到同一只 Event。

        ``cancel_scope`` 每个 attempt 重新进一次（否则第 2 次尝试拿到已 set 的同一个
        Event，工具一进循环就自杀，重试全部秒失败）。
        """
        observed: list[bool] = []

        @tool
        def cooperative(tag: str = "t") -> str:
            """Wait for the cancel flag, then report whether it was set."""
            from liteagent.tools.base import current_cancel_flag

            flag = current_cancel_flag()
            if flag is None:
                return "no-flag"
            flag.wait(5.0)
            observed.append(flag.is_set())
            return "cancelled"

        registry = ToolRegistry([cooperative])
        async with _make_executor(registry, default_timeout_s=0.05) as executor:
            result = await executor.execute(ToolCall.create("cooperative", {"tag": "c"}))
            # 等 worker 线程观察到 flag（它是同一只 threading.Event）
            deadline = time.monotonic() + 5.0
            while not observed and time.monotonic() < deadline:
                await asyncio.sleep(0.01)

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolTimeoutError")
        self.assertEqual(observed, [True])


# ======================================================================================
# 重试与退避
# ======================================================================================


class RetryTests(_AsyncExecutorTestCase):
    """§7.4.1 步骤 5.f 的重试判定与退避序列。"""

    async def test_backoff_sequence_with_recording_sleep(self) -> None:
        """``jitter=0`` + ``RecordingSleep``：把"墙钟耗时"换成**延迟序列**（D-26）。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep,
            retry_policy=RetryPolicy(max_retries=3, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(
                ToolCall.create("flaky", {"fail_times": 2})
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "recovered")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(sleep.delays, [0.1, 0.2])
        self.assertAlmostEqual(sleep.total, 0.3)

    async def test_retry_exhaustion_wraps_last_error(self) -> None:
        """用尽且重试过 -> ``ToolRetryExhaustedError`` + ``metadata["last_error_type"]``。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep,
            retry_policy=RetryPolicy(max_retries=2, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(
                ToolCall.create("flaky", {"fail_times": 99})
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolRetryExhaustedError")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.metadata["last_error_type"], "_RetryableTestError")
        self.assertEqual(result.metadata["feedback_kind"], "infrastructure")
        self.assertEqual(sleep.delays, [0.1, 0.2])
        self.assertTrue(result.content.startswith("ERROR("))

    async def test_non_retryable_error_is_not_retried(self) -> None:
        """``ToolExecutionError.retryable is False`` -> 只尝试 1 次（§3.4 白名单）。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3)
        ) as executor:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "execution"})
            )
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.error_type, "ToolExecutionError")
        self.assertEqual(sleep.delays, [])

    async def test_non_idempotent_tool_is_not_retried_by_default(self) -> None:
        """``idempotent=False`` + 默认 ``allow_retry_on_non_idempotent=False`` -> 强制 1 次。

        非幂等工具重试 = 可能重复产生副作用（下单、发信）。默认不许。
        """
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3, jitter=0.0)
        ) as executor:
            result = await executor.execute(ToolCall.create("non_idempotent_tool", {}))
        self.assertEqual(result.attempts, 1)
        self.assertEqual(sleep.delays, [])

    async def test_non_idempotent_retried_when_explicitly_allowed(self) -> None:
        """显式打开 ``allow_retry_on_non_idempotent`` 后，重试预算恢复。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep,
            allow_retry_on_non_idempotent=True,
            retry_policy=RetryPolicy(max_retries=3, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(ToolCall.create("non_idempotent_tool", {}))
        self.assertEqual(result.attempts, 4)
        self.assertEqual(sleep.delays, [0.1, 0.2, 0.4])

    async def test_retry_after_s_extends_the_delay(self) -> None:
        """退避取 ``max(backoff, exc.retry_after_s)``（服务端说"多久之后再来"）。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep,
            retry_policy=RetryPolicy(max_retries=1, jitter=0.0, backoff_base_s=0.1),
        ) as executor:
            result = await executor.execute(
                ToolCall.create("flaky", {"fail_times": 1, "retry_after_s": 5.0})
            )
        self.assertTrue(result.ok)
        self.assertEqual(sleep.delays, [5.0])

    async def test_custom_is_retryable_hook(self) -> None:
        """注入的 ``is_retryable`` 完全接管判定（连 retryable=False 的异常也重试）。"""
        sleep = RecordingSleep()
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(
                sleep_fn=sleep,
                retry_policy=RetryPolicy(max_retries=1, jitter=0.0, backoff_base_s=0.1),
            ),
            is_retryable=lambda exc: True,
        )
        try:
            result = await executor.execute(
                ToolCall.create("raise_typed_error", {"kind": "execution"})
            )
        finally:
            await executor.aclose()
        self.assertEqual(result.attempts, 2)
        self.assertEqual(sleep.delays, [0.1])

    async def test_spec_max_retries_is_used(self) -> None:
        """重试解析：参数 > call.metadata > ``tool.spec.max_retries`` > 配置。"""
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=0, jitter=0.0, backoff_base_s=0.1)
        ) as executor:
            result = await executor.execute(ToolCall.create("spec_retries_tool", {}))
        self.assertEqual(result.attempts, 3)  # spec.max_retries == 2
        self.assertEqual(sleep.delays, [0.1, 0.2])

    async def test_call_metadata_max_retries_overrides_spec(self) -> None:
        call = ToolCall(
            id="call_meta_retries",
            name="spec_retries_tool",
            arguments={},
            metadata={"max_retries": 1},
        )
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=0, jitter=0.0, backoff_base_s=0.1)
        ) as executor:
            result = await executor.execute(call)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(sleep.delays, [0.1])

    async def test_execute_argument_overrides_everything(self) -> None:
        sleep = RecordingSleep()
        async with _make_executor(
            sleep_fn=sleep, retry_policy=RetryPolicy(max_retries=3, jitter=0.0, backoff_base_s=0.1)
        ) as executor:
            result = await executor.execute(
                ToolCall.create("spec_retries_tool", {}), max_retries=0
            )
        self.assertEqual(result.attempts, 1)
        self.assertEqual(sleep.delays, [])

    async def test_retry_event_is_emitted(self) -> None:
        """每次重试前后发 ``TOOL_RETRY``（含 delay_s 与 error_type）。"""
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(
                sleep_fn=RecordingSleep(),
                retry_policy=RetryPolicy(max_retries=1, jitter=0.0, backoff_base_s=0.1),
            ),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            await executor.execute(
                ToolCall.create("flaky", {"fail_times": 1})
            )
        finally:
            await executor.aclose()

        names = [name for name, _ in events]
        self.assertEqual(names, ["tool_started", "tool_retry", "tool_started", "tool_finished"])
        retry = events[1][1]
        self.assertEqual(retry["delay_s"], 0.1)
        self.assertEqual(retry["error_type"], "_RetryableTestError")
        self.assertEqual(retry["attempt"], 0)

    async def test_success_event_carries_truncation_flag(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(), ExecutorConfig(max_result_chars=50),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            await executor.execute(ToolCall.create("long_result", {"size": 200}))
        finally:
            await executor.aclose()
        finished = [data for name, data in events if name == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertIs(finished[0]["truncated"], True)

    async def test_error_event_is_emitted_for_missing_tool(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(), ExecutorConfig(),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            await executor.execute(ToolCall.create("nope", {}))
        finally:
            await executor.aclose()
        self.assertEqual([name for name, _ in events], ["tool_error"])
        self.assertEqual(events[0][1]["error_type"], "ToolNotFoundError")

    async def test_on_event_callback_failure_does_not_break_execution(self) -> None:
        """回调是用户代码：它抛异常不能打死执行器（红线 10：记日志而不是吞）。"""

        def broken(name: str, data: dict[str, Any]) -> None:
            raise RuntimeError("callback blew up")

        executor = ToolExecutor(_registry(), ExecutorConfig(), on_event=broken)
        try:
            result = await executor.execute(ToolCall.create("echo_args", {"a": 3}))
        finally:
            await executor.aclose()
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "3")


# ======================================================================================
# 结果截断与 _stringify
# ======================================================================================


class ResultRenderingTests(unittest.TestCase):
    """§7.4.3 的 ``_stringify`` 与 §7.4.1 步骤 5.d 的截断。"""

    def test_stringify_none_and_str(self) -> None:
        self.assertEqual(_stringify(None), "")
        self.assertEqual(_stringify("already text"), "already text")

    def test_stringify_bytes_is_single_line_base64(self) -> None:
        encoded = _stringify(b"hello bytes")
        self.assertEqual(encoded, base64.b64encode(b"hello bytes").decode("ascii"))
        self.assertNotIn("\n", encoded)

    def test_stringify_json_scalars_and_containers(self) -> None:
        self.assertEqual(json.loads(_stringify({"a": 1})), {"a": 1})
        self.assertEqual(json.loads(_stringify([1, 2])), [1, 2])
        self.assertEqual(_stringify(5), "5")
        self.assertEqual(_stringify(1.5), "1.5")
        self.assertEqual(_stringify(True), "true")
        self.assertEqual(_stringify(False), "false")

    def test_stringify_dataclass_uses_to_jsonable(self) -> None:
        @dataclasses.dataclass
        class _Payload:
            name: str = "x"
            size: int = 1

        self.assertEqual(json.loads(_stringify(_Payload())), {"name": "x", "size": 1})

    def test_stringify_other_marks_the_type_name(self) -> None:
        """其它类型 -> ``str(obj)`` 且 ``metadata["stringified"]`` 记下原始类型（红线 12）。"""
        value = object()
        self.assertEqual(_stringify(value), str(value))
        self.assertEqual(_stringify_extras(value), {"stringified": "object"})

    def test_stringify_tool_result_merges_metadata(self) -> None:
        inner = ToolResult.success(
            ToolCall(id="inner", name="inner"), "inner content", metadata={"k": "v"}
        )
        self.assertEqual(_stringify(inner), "inner content")
        self.assertEqual(_stringify_extras(inner), {"k": "v"})

    def test_stringify_returns_empty_extras_for_plain_values(self) -> None:
        for value in (None, "s", b"b", {"a": 1}, 5):
            self.assertEqual(_stringify_extras(value), {}, msg=repr(value))

    def test_no_stringified_marker_for_known_types(self) -> None:
        self.assertNotIn("stringified", _stringify_extras({"a": 1}))


class TruncationTests(_AsyncExecutorTestCase):
    """结果截断**只做一次**，且 metadata 里的口径必须自洽。"""

    async def test_long_result_is_truncated(self) -> None:
        async with _make_executor(max_result_chars=50) as executor:
            result = await executor.execute(ToolCall.create("long_result", {"size": 200}))

        self.assertTrue(result.ok)
        self.assertIs(result.metadata["truncated"], True)
        self.assertEqual(result.metadata["original_chars"], 200)
        self.assertLess(len(result.content), 200)
        self.assertIn("[truncated", result.content)

    async def test_short_result_is_not_truncated(self) -> None:
        async with _make_executor(max_result_chars=8000) as executor:
            result = await executor.execute(ToolCall.create("long_result", {"size": 20}))

        self.assertTrue(result.ok)
        self.assertIs(result.metadata["truncated"], False)
        self.assertEqual(result.metadata["original_chars"], 20)
        self.assertEqual(result.content, "abcdefghij" * 2)

    async def test_zero_max_chars_disables_truncation(self) -> None:
        """``max_result_chars <= 0`` 视为不截断。"""
        async with _make_executor(max_result_chars=0) as executor:
            result = await executor.execute(ToolCall.create("long_result", {"size": 300}))
        self.assertIs(result.metadata["truncated"], False)
        self.assertEqual(len(result.content), 300)


# ======================================================================================
# 并发：顺序保持、峰值上限、fail_fast、sequential_tools
# ======================================================================================


class ConcurrencyTests(_AsyncExecutorTestCase):
    """``execute_many`` 的 gather 语义（§7.4.2）。"""

    async def test_execute_many_preserves_order(self) -> None:
        """返回顺序与 ``calls`` 严格一致（**不看完成顺序**）。"""
        delays = [0.03, 0.0, 0.02, 0.01]
        calls = [
            ToolCall.create("record_async", {"tag": f"t{i}", "delay": delay})
            for i, delay in enumerate(delays)
        ]
        async with _make_executor(max_concurrency=4) as executor:
            results = await executor.execute_many(calls)

        self.assertEqual([r.content for r in results], ["t0", "t1", "t2", "t3"])
        for index, result in enumerate(results):
            self.assertEqual(result.call_id, calls[index].id)
            self.assertEqual(result.name, calls[index].name)

    async def test_execute_many_empty_returns_empty(self) -> None:
        async with _make_executor() as executor:
            self.assertEqual(await executor.execute_many([]), [])

    async def test_execute_many_respects_max_concurrency_async(self) -> None:
        """计数工具断言峰值 <= N（同时 >= N，证明真的并发过）。"""
        calls = [
            ToolCall.create("counted_async", {"tag": f"t{i}", "delay": 0.02})
            for i in range(6)
        ]
        async with _make_executor(max_concurrency=2) as executor:
            results = await executor.execute_many(calls)

        self.assertEqual([r.content for r in results], [f"t{i}" for i in range(6)])
        self.assertLessEqual(_PEAK["peak"], 2)
        self.assertEqual(_PEAK["peak"], 2)

    async def test_execute_many_respects_max_concurrency_sync(self) -> None:
        """同步工具同样受信号量约束（它们跑在线程池里，但限流在 await 层）。"""
        calls = [
            ToolCall.create("counted_sync", {"tag": f"t{i}", "delay": 0.03})
            for i in range(6)
        ]
        async with _make_executor(max_concurrency=2, thread_pool_size=6) as executor:
            results = await executor.execute_many(calls)

        self.assertEqual(len(results), 6)
        self.assertTrue(all(r.ok for r in results))
        self.assertLessEqual(_PEAK["peak"], 2)

    async def test_concurrency_argument_lowers_the_peak(self) -> None:
        calls = [
            ToolCall.create("counted_async", {"tag": f"t{i}", "delay": 0.02})
            for i in range(4)
        ]
        async with _make_executor(max_concurrency=4) as executor:
            await executor.execute_many(calls, concurrency=1)
        self.assertEqual(_PEAK["peak"], 1)

    async def test_execute_many_never_raises_for_tool_failures(self) -> None:
        """**永不向外抛业务异常**：全失败也要返回一条与 calls 等长的列表。"""
        calls = [
            ToolCall.create("raise_typed_error", {"kind": "value"}),
            ToolCall.create("does_not_exist", {}),
            ToolCall.create("echo_args", {"a": "bad"}),
            ToolCall.create("record_async", {"tag": "ok"}),
        ]
        async with _make_executor(fail_fast=False) as executor:
            results = await executor.execute_many(calls)

        self.assertEqual(len(results), len(calls))
        self.assertEqual(
            [r.error_type for r in results],
            ["ToolExecutionError", "ToolNotFoundError", "ToolValidationError", None],
        )
        for index, result in enumerate(results):
            self.assertEqual(result.call_id, calls[index].id)
            self.assertEqual(result.name, calls[index].name)

    async def test_fail_fast_cancels_siblings_and_keeps_alignment(self) -> None:
        """**重点**：fail_fast 取消兄弟任务后，结果列表仍与 calls 对齐且**不抛异常**。"""
        calls = [
            ToolCall.create("raise_typed_error", {"kind": "execution"}),
            ToolCall.create("record_async", {"tag": "s1", "delay": 0.05}),
            ToolCall.create("record_async", {"tag": "s2", "delay": 0.05}),
            ToolCall.create("record_async", {"tag": "s3", "delay": 0.05}),
        ]
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(fail_fast=True, max_concurrency=4),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            results = await executor.execute_many(calls, concurrency=1)
        finally:
            await executor.aclose()

        self.assertEqual(len(results), len(calls))
        for index, result in enumerate(results):
            self.assertEqual(result.call_id, calls[index].id)
            self.assertEqual(result.name, calls[index].name)
            self.assertFalse(result.ok)
            self.assertTrue(result.content.startswith("ERROR("), msg=result.content)
        self.assertEqual(results[0].error_type, "ToolExecutionError")
        for result in results[1:]:
            self.assertEqual(result.error_type, "ToolSkippedError")
            self.assertIs(result.metadata["skipped_by_fail_fast"], True)

        signals = [data for name, data in events if "fail_fast_first_index" in data]
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["fail_fast_first_index"], 0)
        self.assertEqual(signals[0]["skipped"], 3)

    async def test_concurrent_fail_fast_batches_do_not_share_batch_state(self) -> None:
        """[v3 回归] fail_fast 是**批次级**状态：并发的两个 ``execute_many`` 不得互相污染。

        v2 把"本批次是否触发过 fail_fast"存在实例属性上，每个 ``execute_many`` 开头重置。
        于是批次 B 一开始就把 A 的标志抹掉，A 收尾时把自己 fail_fast 亲手取消的兄弟任务
        误判成"调用方取消了 execute_many"而 ``raise CancelledError`` —— 违反 §7.4.2
        规则 1（fail_fast 必须仍返回与 calls 等长的列表），在 Agent 路径上还会让整次
        run 变成 ABORTED。
        """
        fired = asyncio.Event()

        def on_event(name: str, data: dict[str, Any]) -> None:
            if "fail_fast_first_index" in data:
                fired.set()

        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(fail_fast=True, max_concurrency=4),
            on_event=on_event,
        )
        a_calls = [
            ToolCall.create("raise_typed_error", {"kind": "execution"}),
            ToolCall.create("cancel_lingering", {"tag": "stubborn", "linger": 0.3}),
        ]
        b_calls = [
            ToolCall.create("record_async", {"tag": "b1"}),
            ToolCall.create("record_async", {"tag": "b2"}),
        ]
        try:
            # A 并发起两个：一个立刻失败（触发 fail_fast），一个被取消后在清理路径里
            # 逗留 0.3s —— 这段时间正好让 B 落进"A 已置位、A 还没收尾"的窗口。
            a_task = asyncio.ensure_future(executor.execute_many(a_calls))
            await asyncio.wait_for(fired.wait(), timeout=5.0)
            b_results = await executor.execute_many(b_calls)
            a_results = await asyncio.wait_for(a_task, timeout=5.0)
        finally:
            await executor.aclose()

        # B 批次正常返回，且不该被 A 的 fail_fast 影响
        self.assertEqual(len(b_results), len(b_calls))
        self.assertTrue(all(r.ok for r in b_results), msg=[r.content for r in b_results])
        # A 批次必须仍返回**与 calls 等长的列表**（不能抛 CancelledError）
        self.assertEqual(len(a_results), len(a_calls))
        self.assertEqual(a_results[0].error_type, "ToolExecutionError")
        self.assertEqual(a_results[1].error_type, "ToolSkippedError")
        self.assertIs(a_results[1].metadata["skipped_by_fail_fast"], True)

    async def test_fail_fast_disabled_runs_every_sibling(self) -> None:
        calls = [
            ToolCall.create("raise_typed_error", {"kind": "execution"}),
            ToolCall.create("record_async", {"tag": "s1"}),
            ToolCall.create("record_async", {"tag": "s2"}),
        ]
        async with _make_executor(fail_fast=False) as executor:
            results = await executor.execute_many(calls)
        self.assertEqual([r.ok for r in results], [False, True, True])
        self.assertNotIn("skipped_by_fail_fast", results[1].metadata)

    async def test_execute_many_all_fail_fast_all_failed(self) -> None:
        """全部失败时 fail_fast 不该漏填任何位置。"""
        calls = [
            ToolCall.create("raise_typed_error", {"kind": "value"}) for _ in range(3)
        ]
        async with _make_executor(fail_fast=True) as executor:
            results = await executor.execute_many(calls)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(not r.ok for r in results))

    async def test_batch_guard_is_not_used_when_concurrency_matches_config(self) -> None:
        """``concurrency`` 与配置相同时不多造一层信号量（两层叠加只在本函数内生效）。"""
        calls = [ToolCall.create("echo_args", {"a": i}) for i in range(3)]
        async with _make_executor(max_concurrency=4) as executor:
            results = await executor.execute_many(calls, concurrency=4)
        self.assertEqual([r.content for r in results], ["0", "1", "2"])


class SequentialToolsTests(_ExecutorTestCase):
    """``sequential_tools`` 的跨调用互斥（§7.4.1 步骤 5.b）。

    必须用 ``threading.Lock`` 而不是 loop-bound ``asyncio.Lock``：同步工具会被
    ``execute_many`` 丢进不同 worker 线程，每个线程里的 ``asyncio.run`` 都是**新 loop**。
    本用例的 ``nested_loop_tool`` 就是"delegate 式"的工具 —— 它在自己的线程里
    再起一个 ``asyncio.run``，正是 loop-bound 锁会失效的场景。
    """

    def test_sequential_tools_serialise_across_two_nested_loops(self) -> None:
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(
                sequential_tools=frozenset({"nested_loop_tool"}),
                max_concurrency=4,
                thread_pool_size=4,
                retry_policy=RetryPolicy(max_retries=0),
            ),
        )
        try:
            for round_index in range(2):
                calls = [
                    ToolCall.create("nested_loop_tool", {"tag": f"r{round_index}-t{i}"})
                    for i in range(3)
                ]
                results = asyncio.run(executor.execute_many(calls))
                self.assertEqual(len(results), 3)
                self.assertTrue(all(r.ok for r in results), msg=[r.content for r in results])
                # 峰值恒为 1 -> 真的串行（若用 asyncio.Lock，这里会是 3）
                self.assertEqual(
                    _PEAK["peak"],
                    1,
                    msg=f"round {round_index}: sequential_tools did not serialise",
                )
                _PEAK["peak"] = 0
        finally:
            asyncio.run(executor.aclose())

    def test_non_sequential_tools_may_overlap(self) -> None:
        """对照组：同样的工具不放进 ``sequential_tools`` 时峰值 > 1（证明上一条有效）。"""
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(max_concurrency=4, thread_pool_size=4, retry_policy=RetryPolicy(max_retries=0)),
        )
        try:
            calls = [
                ToolCall.create("nested_loop_tool", {"tag": f"t{i}"}) for i in range(3)
            ]
            results = asyncio.run(executor.execute_many(calls))
        finally:
            asyncio.run(executor.aclose())
        self.assertTrue(all(r.ok for r in results))
        self.assertGreaterEqual(_PEAK["peak"], 2)


# ======================================================================================
# sequential_tools 的取消安全（异步：必须继承 IsolatedAsyncioTestCase）
# ======================================================================================


class SequentialToolsCancellationTests(_AsyncExecutorTestCase):
    """取消等锁者不得让 ``threading.Lock`` 变成孤儿（§7.4.1 步骤 5.b 的取消语义）。"""

    async def test_cancelled_lock_waiter_does_not_orphan_the_sequential_lock(self) -> None:
        """[v3 回归] 取消一个还在等锁的调用后，锁必须仍能归还（不得变成孤儿）。

        v2 用 ``await asyncio.to_thread(lock.acquire)``：取消只取消了 await 层，
        worker 线程仍会拿到锁，而释放锁的 ``finally`` 属于**已被取消的协程**，永不执行
        —— 该工具此后每次调用都卡在 acquire 上；卡死的线程还会让 ``asyncio.run`` /
        ``_run_and_cleanup`` 在 ``shutdown_default_executor`` 处永久挂起（无异常、无超时）。
        """
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(
                sequential_tools=frozenset({"block_on_gate_seq"}),
                max_concurrency=4,
                thread_pool_size=4,
                retry_policy=RetryPolicy(max_retries=0),
            ),
        )
        try:
            holder = asyncio.ensure_future(
                executor.execute(ToolCall.create("block_on_gate_seq", {"tag": "holder"}))
            )
            await asyncio.sleep(0.1)  # 让 holder 真的拿到锁并阻塞在闸门上
            waiter = asyncio.ensure_future(
                executor.execute(ToolCall.create("block_on_gate_seq", {"tag": "waiter"}))
            )
            await asyncio.sleep(0.1)  # 让 waiter 的 worker 线程停在 lock.acquire() 上
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

            _GATE.set()  # 放 holder 走完；锁应当被"取消方"归还，而不是被孤儿线程吞掉
            self.assertTrue((await asyncio.wait_for(holder, timeout=5.0)).ok)

            third = await asyncio.wait_for(
                executor.execute(ToolCall.create("block_on_gate_seq", {"tag": "third"})),
                timeout=5.0,
            )
            self.assertTrue(third.ok)
            self.assertEqual(third.content, "third")
        finally:
            _GATE.set()
            await executor.aclose()



# ======================================================================================
# 审批（HITL）
# ======================================================================================


class ApprovalTests(_AsyncExecutorTestCase):
    """§7.4.1 步骤 4.5 的冻结伪代码：**fail-closed**，不是 fail-open。"""

    async def test_no_policy_denies(self) -> None:
        """``policy is None`` 且 ``requires_approval=True`` -> 一律拒绝。"""
        async with _make_executor() as executor:
            result = await executor.execute(
                ToolCall.create("needs_approval", {"x": 1})
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolApprovalDeniedError")
        self.assertIs(result.metadata["approved"], False)
        # 回灌文案是**冻结字面量**（不追加任何后缀）
        self.assertEqual(
            result.content,
            "ERROR(ToolApprovalDeniedError): this tool requires human approval",
        )
        # 工具**没有**被执行
        self.assertNotIn("needs_approval", _SYNC_CALLS)

    async def test_policy_returning_false_denies(self) -> None:
        async with _make_executor(approval_policy=lambda call, tool: False) as executor:
            result = await executor.execute(
                ToolCall.create("needs_approval", {"x": 1})
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolApprovalDeniedError")
        self.assertIs(result.metadata["approved"], False)
        self.assertEqual(result.attempts, 1)
        self.assertNotIn("needs_approval", _SYNC_CALLS)

    async def test_policy_returning_true_executes(self) -> None:
        seen: list[str] = []

        def policy(call: ToolCall, tool: Tool) -> bool:
            seen.append(tool.name)
            return True

        async with _make_executor(approval_policy=policy) as executor:
            result = await executor.execute(
                ToolCall.create("needs_approval", {"x": 7})
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "7")
        self.assertIs(result.metadata["approved"], True)
        self.assertEqual(seen, ["needs_approval"])
        self.assertIn("needs_approval", _SYNC_CALLS)

    async def test_policy_error_fails_closed(self) -> None:
        """审批策略自身故障 -> **拒绝**（fail-closed）并把原因回灌。"""

        def policy(call: ToolCall, tool: Tool) -> bool:
            raise ConfigError("approval backend is down")

        async with _make_executor(approval_policy=policy) as executor:
            result = await executor.execute(ToolCall.create("needs_approval", {}))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ConfigError")
        self.assertIs(result.metadata["approved"], False)
        self.assertNotIn("needs_approval", _SYNC_CALLS)

    async def test_policy_abort_raises_cancelled(self) -> None:
        """用户在审批回调里主动中止 -> 走**取消**路径（不是失败结果）。"""

        def policy(call: ToolCall, tool: Tool) -> bool:
            raise AgentAbortedError("user pressed stop")

        async with _make_executor(approval_policy=policy) as executor:
            with self.assertRaises(asyncio.CancelledError):
                await executor.execute(ToolCall.create("needs_approval", {}))

    async def test_approval_event_is_emitted(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(approval_policy=lambda call, tool: True),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            await executor.execute(ToolCall.create("needs_approval", {"x": 1}))
        finally:
            await executor.aclose()
        approvals = [data for name, data in events if name == "tool_approval"]
        self.assertEqual(len(approvals), 1)
        self.assertIs(approvals[0]["approved"], True)
        self.assertIs(approvals[0]["policy_present"], True)

    async def test_tools_without_requires_approval_skip_the_step(self) -> None:
        """没标 ``requires_approval`` 的工具不该走审批（哪怕配了 policy）。"""
        calls: list[str] = []

        def policy(call: ToolCall, tool: Tool) -> bool:
            calls.append(tool.name)
            return False

        async with _make_executor(approval_policy=policy) as executor:
            result = await executor.execute(ToolCall.create("echo_args", {"a": 2}))
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "2")
        self.assertEqual(calls, [])
        self.assertNotIn("approved", result.metadata)


# ======================================================================================
# 熔断
# ======================================================================================


class CircuitBreakerTests(_AsyncExecutorTestCase):
    """§7.4.1 步骤 4.6：连续 N 次 infrastructure 失败后熔断。"""

    async def test_opens_after_three_consecutive_infrastructure_failures(self) -> None:
        _SCRIPT["breaker"] = ["fail"] * 5
        async with _make_executor(disable_tool_after_failures=3) as executor:
            results = [
                await executor.execute(
                    ToolCall.create("scripted_tool", {"script_key": "breaker"})
                )
                for _ in range(4)
            ]

        self.assertEqual([r.ok for r in results], [False, False, False, False])
        self.assertNotIn("disabled", results[0].metadata)
        self.assertNotIn("disabled", results[1].metadata)
        self.assertNotIn("disabled", results[2].metadata)
        self.assertIs(results[3].metadata["disabled"], True)
        self.assertEqual(results[3].metadata["feedback_kind"], "infrastructure")
        # **第 4 次没有真正执行**：脚本里只消耗了 3 条 "fail"
        self.assertEqual(len(_SCRIPT["breaker"]), 2)
        self.assertEqual(len(_FAIL_CALLS), 3)

    async def test_disabled_event_is_emitted(self) -> None:
        _SCRIPT["events"] = ["fail"] * 4
        events: list[tuple[str, dict[str, Any]]] = []
        executor = ToolExecutor(
            _registry(),
            ExecutorConfig(disable_tool_after_failures=1),
            on_event=lambda name, data: events.append((name, data)),
        )
        try:
            await executor.execute(ToolCall.create("scripted_tool", {"script_key": "events"}))
            events.clear()
            result = await executor.execute(
                ToolCall.create("scripted_tool", {"script_key": "events"})
            )
        finally:
            await executor.aclose()
        self.assertIs(result.metadata["disabled"], True)
        disabled = [data for name, data in events if data.get("disabled") is True]
        self.assertEqual(len(disabled), 1)

    async def test_recoverable_failures_do_not_trip_the_breaker(self) -> None:
        """校验错误是**模型的输入问题**，把它算进熔断会让"打错一次名字"禁掉健康工具。"""
        async with _make_executor(disable_tool_after_failures=3) as executor:
            results = [
                await executor.execute(ToolCall.create("echo_args", {"a": "bad"}))
                for _ in range(5)
            ]
        self.assertTrue(all(r.error_type == "ToolValidationError" for r in results))
        self.assertTrue(all("disabled" not in r.metadata for r in results))

    async def test_success_resets_the_consecutive_counter(self) -> None:
        """熔断计的是**连续**失败；一次成功就清零。"""
        _SCRIPT["reset"] = ["fail", "fail", "ok", "fail", "fail"]
        async with _make_executor(disable_tool_after_failures=3) as executor:
            results = [
                await executor.execute(
                    ToolCall.create("scripted_tool", {"script_key": "reset"})
                )
                for _ in range(5)
            ]
        self.assertEqual([r.ok for r in results], [False, False, True, False, False])
        self.assertTrue(all("disabled" not in r.metadata for r in results))

    async def test_breaker_disabled_by_default_zero(self) -> None:
        _SCRIPT["nolimit"] = ["fail"] * 6
        async with _make_executor(disable_tool_after_failures=0) as executor:
            results = [
                await executor.execute(
                    ToolCall.create("scripted_tool", {"script_key": "nolimit"})
                )
                for _ in range(5)
            ]
        self.assertTrue(all("disabled" not in r.metadata for r in results))
        self.assertEqual(len(_SCRIPT["nolimit"]), 1)

    async def test_two_tools_have_independent_counters(self) -> None:
        """熔断计数**按工具名**（§7.4.1 步骤 4.6）—— 一个工具熔断不牵连另一个。"""
        _SCRIPT["tool_a"] = ["fail"] * 3
        _SCRIPT["tool_b"] = ["fail"] * 1
        async with _make_executor(disable_tool_after_failures=2) as executor:
            for _ in range(2):
                await executor.execute(
                    ToolCall.create("scripted_tool", {"script_key": "tool_a"})
                )
            result_b = await executor.execute(
                ToolCall.create("scripted_tool_b", {"script_key": "tool_b"})
            )
        # tool_a 已经熔断（2 次），不应影响 tool_b 的计数器
        self.assertNotIn("disabled", result_b.metadata)
        self.assertEqual(result_b.error_type, "ToolExecutionError")

    async def test_breakers_are_per_tool_name_not_per_arguments(self) -> None:
        """同一个工具的**不同参数**共享同一个计数器（计数按名字，不按调用）。"""
        _SCRIPT["pername"] = ["fail"] * 3
        async with _make_executor(disable_tool_after_failures=2) as executor:
            first = await executor.execute(
                ToolCall.create("scripted_tool", {"script_key": "pername"})
            )
            second = await executor.execute(
                ToolCall.create("scripted_tool", {"script_key": "pername"})
            )
            third = await executor.execute(
                ToolCall.create("scripted_tool", {"script_key": "pername"})
            )
        self.assertNotIn("disabled", first.metadata)
        self.assertNotIn("disabled", second.metadata)
        self.assertIs(third.metadata["disabled"], True)


# ======================================================================================
# 跨 loop 复用与生命周期（守 §0.4 R-LOOP）
# ======================================================================================


class CrossLoopLifecycleTests(_ExecutorTestCase):
    """R-LOOP：``asyncio`` 原语**禁止**在 ``__init__`` 创建，必须按 loop 懒建。"""

    def test_no_asyncio_primitives_are_created_in_init(self) -> None:
        """构造后 ``LoopBoundPool`` 里必须是空的（否则第二次 ``asyncio.run`` 必炸）。"""
        executor = _make_executor(max_concurrency=1)
        self.assertEqual(len(executor._pool), 0)
        self.assertEqual(executor._pool._by_loop, {})  # type: ignore[attr-defined]

    def test_reuse_across_two_asyncio_run(self) -> None:
        """跨两次 ``asyncio.run`` 复用同一 executor **不炸**（守 §0.4 R-LOOP）。

        `[v3 修正]` 上一版 docstring 写"且池不泄漏"，但本用例的唯一池断言是
        ``assertGreaterEqual(len(executor._pool), 1)`` —— 那个方向**不可能**因为泄漏而
        失败（池涨到 100 也会 PASS），对"不泄漏"这个属性是一条空洞断言。
        真实契约是：手写 raw ``asyncio.run`` 时清理责任在调用方（每个 loop 会留下
        注册项；同步工具还会多留一个私有线程池），必须显式 ``aclose()`` /
        ``release_loop()`` 才归零；只有 ``execute_sync`` / ``Agent.run`` 走
        ``config._run_and_cleanup`` 自动清理。本用例因此显式 aclose 之后断言归零，
        而"不泄漏"由 ``ExecuteSyncTests`` 那条"每次同步调用后池内条数为 0"负责。
        """
        executor = _make_executor(max_concurrency=1)

        async def call(tag: str) -> ToolResult:
            return await executor.execute(
                ToolCall.create("record_async", {"tag": tag})
            )

        first = asyncio.run(call("first"))
        second = asyncio.run(call("second"))

        self.assertTrue(first.ok, msg=first.content)
        self.assertTrue(second.ok, msg=second.content)
        self.assertEqual(first.content, "first")
        self.assertEqual(second.content, "second")
        # 两个 loop 各自登记了一份 per-loop 原语（还没释放）
        self.assertGreaterEqual(len(executor._pool), 1)

        asyncio.run(executor.aclose())
        self.assertEqual(len(executor._pool), 0)

    def test_reuse_across_run_with_contended_semaphore(self) -> None:
        """R-LOOP 的精确复现：**争用**才会让 ``asyncio.Semaphore`` 绑定 loop。"""
        executor = _make_executor(max_concurrency=1)

        async def batch(prefix: str) -> list[ToolResult]:
            calls = [
                ToolCall.create("record_async", {"tag": f"{prefix}-{i}", "delay": 0.01})
                for i in range(2)
            ]
            return await executor.execute_many(calls)

        first = asyncio.run(batch("a"))
        second = asyncio.run(batch("b"))

        self.assertEqual([r.content for r in first], ["a-0", "a-1"])
        self.assertEqual([r.content for r in second], ["b-0", "b-1"])

        asyncio.run(executor.aclose())
        self.assertEqual(len(executor._pool), 0)

    def test_execute_sync_releases_the_pool_each_call(self) -> None:
        """``execute_sync`` 走 ``config.run_sync``，它的 ``finally`` 负责 release。"""
        executor = _make_executor()
        try:
            first = executor.execute_sync(ToolCall.create("record_sync", {"tag": "s1"}))
            self.assertTrue(first.ok)
            self.assertEqual(len(executor._pool), 0)

            second = executor.execute_sync(ToolCall.create("record_sync", {"tag": "s2"}))
            self.assertTrue(second.ok)
            self.assertEqual(len(executor._pool), 0)
        finally:
            executor.close()
        self.assertEqual(len(executor._pool), 0)

    def test_external_thread_pool_is_not_closed_by_aclose(self) -> None:
        """外部池的所有权属于调用方：``aclose`` 不关它。"""
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="external-pool")
        try:
            executor = ToolExecutor(
                _registry(), ExecutorConfig(), thread_pool=pool
            )

            async def go() -> ToolResult:
                result = await executor.execute(
                    ToolCall.create("record_sync", {"tag": "external"})
                )
                await executor.aclose()
                return result

            result = asyncio.run(go())
            self.assertTrue(result.ok)
            # 池仍可用（没被 shutdown）
            self.assertEqual(pool.submit(lambda: "alive").result(timeout=5), "alive")
        finally:
            pool.shutdown(wait=True)

    def test_private_pool_is_shut_down_by_aclose(self) -> None:
        executor = _make_executor()
        asyncio.run(executor.execute(ToolCall.create("record_sync", {"tag": "x"})))
        self.assertGreaterEqual(len(executor._pool), 1)
        asyncio.run(executor.aclose())
        self.assertEqual(len(executor._pool), 0)

    def test_close_is_the_sync_mirror_of_aclose(self) -> None:
        executor = _make_executor()
        asyncio.run(executor.execute(ToolCall.create("record_sync", {"tag": "x"})))
        executor.close()
        self.assertEqual(len(executor._pool), 0)

    def test_context_manager_protocols(self) -> None:
        with _make_executor() as executor:
            self.assertIsInstance(executor, ToolExecutor)
        self.assertEqual(len(executor._pool), 0)

    def test_repr_mentions_configuration(self) -> None:
        executor = _make_executor(max_concurrency=3, fail_fast=True)
        text = repr(executor)
        self.assertIn("max_concurrency=3", text)
        self.assertIn("fail_fast=True", text)


class ExecuteSyncTests(_AsyncExecutorTestCase):
    """``execute_sync`` 的 loop 检测（绝不 ``run_until_complete`` 套娃）。"""

    async def test_execute_sync_inside_running_loop_raises_config_error(self) -> None:
        async with _make_executor() as executor:
            with self.assertRaises(ConfigError):
                executor.execute_sync(ToolCall.create("echo_args", {"a": 1}))

    async def test_sync_api_does_not_build_a_coroutine_inside_loop(self) -> None:
        """失败必须发生在**构造协程之前**，否则 3.10 会打 "coroutine was never awaited"。"""
        async with _make_executor() as executor:
            with self.assertRaises(ConfigError):
                executor.execute_sync(ToolCall.create("echo_args", {"a": 1}))
        # 没炸出 RuntimeWarning-as-error 就说明没漏协程（-W error 下会直接失败）


class ExecuteSyncOutsideLoopTests(_ExecutorTestCase):
    """同步入口在**没有**运行 loop 时的正常路径。"""

    def test_execute_sync_returns_a_result(self) -> None:
        executor = _make_executor()
        try:
            result = executor.execute_sync(ToolCall.create("echo_args", {"a": 42}))
        finally:
            executor.close()
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "42")

    def test_execute_sync_encodes_failures(self) -> None:
        executor = _make_executor()
        try:
            result = executor.execute_sync(ToolCall.create("does_not_exist", {}))
        finally:
            executor.close()
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ToolNotFoundError")

    def test_execute_sync_is_repeatable(self) -> None:
        """第二次同步调用正是 R-LOOP 的爆点（'bound to a different event loop'）。"""
        executor = _make_executor(max_concurrency=1)
        try:
            for index in range(3):
                result = executor.execute_sync(
                    ToolCall.create("record_sync", {"tag": f"n{index}"})
                )
                self.assertTrue(result.ok, msg=result.content)
            self.assertEqual(_SYNC_CALLS, ["n0", "n1", "n2"])
        finally:
            executor.close()

    def test_execute_sync_never_raises_for_tool_errors(self) -> None:
        executor = _make_executor()
        try:
            result = executor.execute_sync(
                ToolCall.create("raise_typed_error", {"kind": "value"})
            )
        finally:
            executor.close()
        self.assertFalse(result.ok)
        self.assertTrue(result.content.startswith("ERROR("))


# ======================================================================================
# 取消传播
# ======================================================================================


class CancellationTests(_AsyncExecutorTestCase):
    """§7.4.0：``CancelledError`` 绝不被吞成一次工具失败。"""

    async def test_cancelling_execute_propagates(self) -> None:
        async with _make_executor(default_timeout_s=None) as executor:
            task = asyncio.ensure_future(
                executor.execute(
                    ToolCall.create("record_async", {"tag": "cancel", "delay": 5.0})
                )
            )
            await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(task.cancelled())

    async def test_cancelling_execute_many_propagates(self) -> None:
        async with _make_executor(default_timeout_s=None) as executor:
            calls = [
                ToolCall.create("record_async", {"tag": f"c{i}", "delay": 5.0})
                for i in range(3)
            ]
            task = asyncio.ensure_future(executor.execute_many(calls))
            await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_tool_raising_cancelled_error_is_treated_as_cancellation(self) -> None:
        """工具自己抛的 ``CancelledError`` 视为调用方取消，原样上抛（不转 ToolResult）。"""

        @tool
        async def self_cancel() -> str:
            """Raise CancelledError as a tool author would on shutdown."""
            raise asyncio.CancelledError()

        registry = ToolRegistry([self_cancel])
        async with _make_executor(registry) as executor:
            with self.assertRaises(asyncio.CancelledError):
                await executor.execute(ToolCall.create("self_cancel", {}))

    async def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        """``KeyboardInterrupt`` 是控制流而不是"工具失败"（§13 红线 6 的例外之一）。

        用**同步**工具触发：若写成异步工具，``Task.__step`` 会在 3.10 把
        ``KeyboardInterrupt`` 直接重抛到事件循环外（在 assertRaises 之前），
        那是解释器行为而不是本执行器的行为，无法在这里断言。
        """

        @tool
        def interrupted() -> str:
            """Raise KeyboardInterrupt as a tool author would."""
            raise KeyboardInterrupt()

        registry = ToolRegistry([interrupted])
        async with _make_executor(registry) as executor:
            with self.assertRaises(KeyboardInterrupt):
                await executor.execute(ToolCall.create("interrupted", {}))


# ======================================================================================
# 调用入口（§7.2 的 ``Tool.run`` / ``Tool.arun`` / ``make_function_tool``）
# ======================================================================================


class ToolInvocationEntryPointTests(_ExecutorTestCase):
    """§7.2 的两个调用入口。

    §12 没有为 ``tools/base.py`` 单列测试文件，而执行器的同步/异步分派完全建立在
    ``run`` / ``arun`` 的契约上，所以这几条放在这里 —— 它们是 M-2 的第一道防线：
    ``asyncio.to_thread(tool.arun, args)`` 只会返回一个**没人 await 的协程对象**，
    必须靠 ``Tool.run`` 的 ``TypeError`` 把它变成响亮的失败。
    """

    def test_run_on_async_tool_raises_type_error(self) -> None:
        registry = _registry()
        async_tool = registry.get("record_async")
        with self.assertRaises(TypeError) as ctx:
            async_tool.run({"tag": "x"})
        self.assertIn("use arun()", str(ctx.exception))
        self.assertIn("record_async", str(ctx.exception))

    def test_run_returns_the_raw_value_not_a_coroutine(self) -> None:
        result = record_sync.run({"tag": "direct"})
        self.assertEqual(result, "direct")
        self.assertFalse(asyncio.iscoroutine(result))

    def test_run_respects_mapping_pass_style(self) -> None:
        """``pass_style="mapping"`` -> ``func(args_dict)``；默认 ``kwargs`` -> ``func(**args)``。"""
        captured: list[Any] = []

        def sink(payload: dict[str, Any]) -> str:
            captured.append(payload)
            return "mapped"

        from liteagent.tools.base import make_function_tool

        mapped = make_function_tool(
            name="sink",
            description="d",
            parameters={"type": "object", "properties": {}},
            func=sink,
        )
        self.assertEqual(mapped.spec.pass_style, "mapping")
        self.assertEqual(mapped.run({"a": 1}), "mapped")
        self.assertEqual(captured, [{"a": 1}])

    def test_arun_on_sync_tool_delegates_to_run(self) -> None:
        result = asyncio.run(record_sync.arun({"tag": "via-arun"}))
        self.assertEqual(result, "via-arun")
        self.assertIn("via-arun", _SYNC_CALLS)

    def test_make_function_tool_warns_when_parameters_missing(self) -> None:
        """缺 ``parameters=`` 时必须留 warning（宽 permissive schema 不是错，静默才是）。"""
        from liteagent.tools.base import make_function_tool

        built = make_function_tool(name="loose", description="d", func=lambda args: "x")
        self.assertEqual(
            built.parameters,
            {"type": "object", "properties": {}, "additionalProperties": True},
        )
        self.assertTrue(
            any("no parameters=" in w for w in built.spec.warnings),
            msg=built.spec.warnings,
        )

    def test_make_function_tool_with_parameters_has_no_warning(self) -> None:
        from liteagent.tools.base import make_function_tool

        built = make_function_tool(
            name="tight",
            description="d",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
            func=lambda args: args.get("a", 0),
        )
        self.assertEqual(built.spec.warnings, ())

    def test_unknown_tool_option_raises_tool_definition_error(self) -> None:
        """拼错 kwarg（`require_approval=` 少了 s）必须**当场**炸，不能被 `**kwargs` 吞掉。"""
        from liteagent.errors import ToolDefinitionError

        def plain(a: int = 0) -> int:
            """Plain function."""
            return a

        with self.assertRaises(ToolDefinitionError):
            Tool.from_function(plain, require_approval=True)
        # 拼对的名字当然可以通过
        self.assertTrue(
            Tool.from_function(plain, requires_approval=True).spec.requires_approval
        )

    def test_from_function_is_idempotent_for_existing_tools(self) -> None:
        """`@tool` 套两次不该产生"包装的包装"（否则 `tool.raw` 指向另一个 Tool）。"""
        self.assertIs(Tool.from_function(record_sync), record_sync)
        self.assertIs(record_sync.raw, record_sync.spec.func)

    def test_cancel_scope_is_nestable(self) -> None:
        """``cancel_scope`` 用 ``reset(token)`` 精确还原，嵌套时不吃掉外层信号。"""
        from liteagent.tools.base import cancel_scope, current_cancel_flag

        self.assertIsNone(current_cancel_flag())
        with cancel_scope() as outer:
            self.assertIs(current_cancel_flag(), outer)
            with cancel_scope() as inner:
                self.assertIsNot(inner, outer)
                self.assertIs(current_cancel_flag(), inner)
            self.assertIs(current_cancel_flag(), outer)
        self.assertIsNone(current_cancel_flag())


# ======================================================================================
# conftest 级别的辅助（防止误用其它测试的注册表）
# ======================================================================================


class RegistryIsolationTests(_ExecutorTestCase):
    """本文件自己造注册表，绝不碰 ``get_default_registry()``（并行安全）。"""

    def test_make_registry_helper_is_independent(self) -> None:
        first = make_registry(add_tool)
        second = make_registry(add_tool)
        self.assertIsNot(first, second)
        self.assertEqual(first.names(), ["add"])

    def test_registry_used_by_executor_is_the_one_passed(self) -> None:
        registry = _registry()
        executor = _make_executor(registry)
        self.assertIs(executor.registry, registry)
        self.assertIn("record_sync", registry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
