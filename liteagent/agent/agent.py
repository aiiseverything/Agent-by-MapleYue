from __future__ import annotations

"""`Agent` —— ReAct 状态机（§9.4，全项目最关键的一个文件）。

设计要点（面试时值得展开的三条）：

1. **两条路径归约到同一个状态机**（D-02）。原生 function calling 与文本 ReAct 的差异只有
   5 个点（工具如何暴露 / 模型输出如何解析 / 观察如何回灌 / 能否一轮多调用 / 何时终结），
   全部被 `mode` 变量隔离；循环本身、事件、重复检测、自纠正、截断续写、budget、usage
   统计是**完全共用**的。新增第三种模式（JSON-mode）只需改两处分支。
2. **一切"降级"都必须留痕**（红线 12）。解析失败、重复动作、截断、budget 超限、工具熔断
   都会在 trace 里留下事件或消息，绝不静默continue。
3. **`arun` 永不抛业务异常**（红线 6）。所有失败编码进 `AgentResult(status=FAILED, error=...)`；
   唯一例外是 `asyncio.CancelledError`（必须继续抛出，取消传播不能被打断，§9.4.7）与
   `KeyboardInterrupt`，以及调用方显式要求 `raise_on_error=True` 时的 re-raise。

并发：`arun` **不可重入**（§9.4 冻结）。`Agent.state` 是实例级累积状态，同一个 Agent 被
manager 在一个 step 里并行委派两次时两个协程会互相覆盖，守卫把它变成一次**可见的失败
工具调用**而不是静默的数据竞争。
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from liteagent.agent.callbacks import (
    CallbackLike,
    CallbackManager,
    EventType,
    FunctionCallback,
    TraceEvent,
    as_llm_callback,
)
from liteagent.agent.parser import ParsedAction, ReActParser
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import (
    DEFAULT_MAX_TOOLS_IN_PROMPT,
    DEFAULT_TOOL_FAILURE_LIMIT,
    AgentConfig,
    ExecutorConfig,
    estimate_cost_usd,
    render_template,
    run_sync,
    utc_now,
)
from liteagent.errors import (
    AgentError,
    BudgetExceededError,
    ConfigError,
    LiteAgentError,
    MaxStepsExceededError,
    ReActParseError,
    RepeatedActionError,
    RunTimeoutError,
    ToolExecutionError,
)
from liteagent.llm.base import LLMClient
from liteagent.llm.message import Message, messages_tokens
from liteagent.memory.base import MemoryConfig
from liteagent.memory.manager import MemoryManager
from liteagent.tools.base import Tool
from liteagent.tools.executor import ToolExecutor
from liteagent.tools.registry import ToolRegistry
from liteagent.types import LLMResponse, ToolCall, ToolResult

__all__ = ["Agent"]

_LOGGER = logging.getLogger("liteagent.agent")

#: `arun(**overrides)` 的白名单（§9.4 冻结）。原先未冻结，各实现者会支持不同的键集 ——
#: 那会让"我传了 temperature 却没生效"变成一个只在运行期才暴露的悬案。
_RUN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "max_steps",
        "temperature",
        "max_tokens",
        "tool_choice",
        "mode",
        "max_total_tokens",
        "max_wall_clock_s",
    }
)

#: 合法的 `mode` 取值（冻结的第三态是 `"auto"`，由 `LLMClient.resolve_mode` 决定）。
_VALID_MODES: frozenset[str] = frozenset({"auto", "native", "text"})

#: astream 队列上限（§9.4 冻结为 1000）。满时**丢最新并记 WARNING**：热点路径上不做
#: O(n) 的 `get_nowait` 淘汰，宁可丢一个观测事件也不拖慢主循环。
_ASTREAM_QUEUE_MAXSIZE = 1000

#: §9.4.6 分支 (a) 的续写提示（冻结字面量）。
_TRUNCATION_NUDGE = (
    "Your previous reply was truncated before it finished. Continue from where you stopped."
)

#: §9.4.4 的"无进展"追加句（冻结字面量，`{n}` 是同一份观察重复出现的次数）。
_NO_PROGRESS_SUFFIX = (
    "Your last {n} calls returned identical results — the approach is not working."
)


def _coerce_int(value: Any, *, name: str) -> int | None:
    """`overrides` 的类型闸。

    §9.4 只冻结了"哪些键合法"，没有冻结"值的类型"。这里选择**早失败**：把 `"3"` 之类
    能隐式转换的放过、把 `object()` 之类拒绝掉，比让它在循环条件里抛 TypeError
    （最终变成一个含混的 FAILED）更好定位。
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes)):
        raise ConfigError(
            f"Agent override {name!r} must be an integer, got {type(value).__name__}"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Agent override {name!r} must be an integer, got {value!r}"
        ) from exc


def _coerce_float(value: Any, *, name: str) -> float | None:
    """`_coerce_int` 的浮点版本（用于 `temperature` / `max_wall_clock_s`）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes)):
        raise ConfigError(
            f"Agent override {name!r} must be a number, got {type(value).__name__}"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Agent override {name!r} must be a number, got {value!r}") from exc


class Agent:
    """把 LLM、工具、记忆粘成一个能跑的 ReAct 循环（§9.4）。

    生命周期：`arun` 里创建局部状态 -> 跑到终结/失败 -> `finally` 把局部状态
    **一次性**赋给 `self._state`。运行期的状态从不写实例属性，否则并发/嵌套运行会互相覆盖
    （v1 的 bug，v2 用"只在结束时赋值 + 运行守卫"两件事一起修掉）。
    """

    # ---------------------------------------------------------------- 构造
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        memory: MemoryManager | None = None,
        config: AgentConfig | None = None,
        executor: ToolExecutor | None = None,
        callbacks: Sequence[CallbackLike] | None = None,
        name: str | None = None,
        description: str = "",
    ) -> None:
        """装配四件套（llm/tools/memory/executor），缺省的全部按规范造默认值。

        **红线 15**：`memory is None` 时构造的默认记忆**必须复用自身的 llm**
        （`MemoryManager.from_config(MemoryConfig(), llm=self.llm)`）。否则 `SummaryMemory`
        永远走抽取式兜底，"LLM 摘要成功路径"在 Agent 级路径上永远测不到。
        memory 由外部传入时**不得**替换其 summarizer 的 llm（那是调用方的对象）。

        **R-LOOP（§0.4）**：这里只创建 `threading.Lock`（`_run_guard`），不创建任何
        asyncio 原语 —— `Agent.run()` 每次都会 `asyncio.run()`，把 Semaphore 放在
        `__init__` 里必然在第二次同步调用时炸。
        """
        if llm is None:  # pragma: no cover - 关键字参数一定存在，纯防御
            raise ConfigError("Agent requires an llm instance")
        self.llm: LLMClient = llm
        self.config: AgentConfig = config if config is not None else AgentConfig()
        self.tools: ToolRegistry = tools if tools is not None else ToolRegistry()
        self.callbacks: CallbackManager = CallbackManager(callbacks or ())
        # §9.4：`name or config.name`（**name 参数优先**）。存成 `_name` 再由 property 暴露，
        # 既满足"name 是只读视图"的冻结形态，又允许 multiagent 在构造后改名（setter）。
        self._name: str = name or self.config.name
        self.description: str = description or self.config.description

        # 低层事件（llm/tools/memory 的字符串事件）转成 TraceEvent 的唯一适配器（§2.7）。
        # 它捕获的是 CallbackManager 本身，所以运行中后加的回调（如 astream 的灌队列回调）
        # 也能收到低层事件。
        self._low_level_event = as_llm_callback(self.callbacks)

        if memory is None:
            self.memory: MemoryManager = MemoryManager.from_config(
                MemoryConfig(), llm=self.llm
            )  # 红线 15：复用自身 llm
            # from_config 没有 on_event 形参（§8.6 冻结签名），构造后补挂一次。
            # 只对自己造的记忆这么做：外部传入的 manager 的所有权是调用方的，
            # 改它的 on_event 会把调用方自己的订阅顶掉。
            self.memory.on_event = self._low_level_event
        else:
            self.memory = memory

        if executor is None:
            self.executor: ToolExecutor = ToolExecutor(
                self.tools, ExecutorConfig(), on_event=self._low_level_event
            )
            self._owns_executor = True
        else:
            self.executor = executor
            self._owns_executor = False  # 外部传入的 executor 由调用方负责关闭（§9.4）

        #: arun 的运行守卫。**必须是 threading.Lock**：它要跨线程/跨 loop 生效（红线 13），
        #: 而 asyncio.Lock 在 `Agent.run()` 的两次 `asyncio.run()` 之间会绑定不同的 loop（R-LOOP）。
        self._run_guard = threading.Lock()
        #: 最近一次完整运行的状态；未运行过时是 IDLE 空状态（不抛异常，便于 CLI 探测）。
        self._state: AgentState = AgentState.create("", agent_name=self.name)

    # ---------------------------------------------------------------- 只读视图
    @property
    def name(self) -> str:
        """Agent 的名字（`name` 参数优先于 `config.name`）。"""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """允许改名（multiagent 会用 `agent_{i}` 之类的名字装配 worker）。"""
        self._name = str(value)

    @property
    def state(self) -> AgentState:
        """**最近一次完整运行**的状态；未运行过时返回 `status=IDLE` 的空状态。

        [v2 变更] `self._state` 只在 `arun` 结束时一次性赋值，运行期状态存在局部变量里
        （否则并发/嵌套运行会互相覆盖）。
        """
        return self._state

    # ---------------------------------------------------------------- 同步入口
    def run(self, input: str, **kwargs: Any) -> AgentResult:
        """`run_sync(lambda: self.arun(input, **kwargs))`（§2.3 的 sync 包装冻结形态）。

        收的是**工厂函数**而不是协程对象：协程对象在"已有运行中的 loop"时会永远不被 await，
        3.10 会打 `RuntimeWarning: coroutine ... was never awaited` 污染测试输出。
        """
        return run_sync(lambda: self.arun(input, **kwargs))

    # ---------------------------------------------------------------- 主循环
    async def arun(
        self,
        input: str,
        *,
        state: AgentState | None = None,
        callbacks: Sequence[CallbackLike] | None = None,
        **overrides: Any,
    ) -> AgentResult:
        """跑一次完整的 ReAct 循环。

        **永不抛异常**（除 `asyncio.CancelledError` / `KeyboardInterrupt`）：
        所有失败编码进 `AgentResult(status=FAILED, error=...)`。
        仅当 `config.raise_on_error=True` 时才 re-raise。

        `**overrides` 只允许 §9.4 白名单里的键，非法键 -> `ConfigError`（这是**参数错误**，
        不是运行失败，所以它真的抛，而不是编码成 FAILED —— 与 §9.4 的字面规定一致）。

        `state` 注入时**必须保留注入 state 的 `run_id`**（Agent 不得重新生成）：
        `test_agent_*` 的 trace 断言要按 run_id 把事件串起来。
        """
        # ---- 0. 参数校验放在取守卫之前：参数错误不该受并发守卫的影响，也不需要释放锁 ----
        unknown = set(overrides) - _RUN_OVERRIDE_KEYS
        if unknown:
            raise ConfigError(
                f"unsupported arun override(s): {sorted(unknown)}; "
                f"allowed keys are {sorted(_RUN_OVERRIDE_KEYS)}"
            )

        cfg = self.config
        max_steps = _coerce_int(overrides.get("max_steps", cfg.max_steps), name="max_steps")
        if max_steps is None:  # pragma: no cover - cfg.max_steps 是 int，overrides 传 None 才有意义
            max_steps = cfg.max_steps
        temperature = _coerce_float(
            overrides["temperature"] if "temperature" in overrides else cfg.temperature,
            name="temperature",
        )
        max_tokens = _coerce_int(
            overrides["max_tokens"] if "max_tokens" in overrides else cfg.max_tokens,
            name="max_tokens",
        )
        max_total_tokens = _coerce_int(
            overrides["max_total_tokens"]
            if "max_total_tokens" in overrides
            else cfg.max_total_tokens,
            name="max_total_tokens",
        )
        max_wall_clock_s = _coerce_float(
            overrides["max_wall_clock_s"]
            if "max_wall_clock_s" in overrides
            else cfg.max_wall_clock_s,
            name="max_wall_clock_s",
        )
        tool_choice = (
            overrides["tool_choice"] if "tool_choice" in overrides else cfg.tool_choice
        )
        mode_cfg = overrides.get("mode", cfg.mode)
        if not isinstance(mode_cfg, str) or mode_cfg not in _VALID_MODES:
            raise ConfigError(
                f"unknown mode {mode_cfg!r}; expected one of {sorted(_VALID_MODES)}"
            )

        # ---- 模式解析（唯一出处是 LLMClient.resolve_mode，§9.4.5）----
        has_tools = len(self.tools) > 0
        mode = (
            self.llm.resolve_mode(has_tools=has_tools) if mode_cfg == "auto" else mode_cfg
        )
        if mode == "native" and not self.llm.supports_tool_calling:
            # §9.4.8：显式失败好过静默降级 —— 静默走文本路径会让"我明明配了 native"
            # 变成只能靠读 trace 才能发现的实现细节。
            raise ConfigError(
                f"mode='native' requires a tool-calling LLM, but {type(self.llm).__name__} "
                "has supports_tool_calling=False"
            )
        if mode == "native" and not has_tools:
            _LOGGER.warning(
                "agent %s: mode='native' with an empty tool registry; the model gets no tools",
                self.name,
            )

        # ---- 1. 不可重入守卫（§9.4 冻结）----
        if not self._run_guard.acquire(blocking=False):
            # 编码成 FAILED 而不是抛异常：并发委派时调用方（manager）把它当成一次
            # 失败的工具调用，可以自纠正；抛异常会让委派路径直接崩。
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                agent_name=self.name,
                error=AgentError(f"agent {self.name} already has a run in flight"),
            )

        run_state: AgentState = (
            state if state is not None else AgentState.create(input, agent_name=self.name)
        )
        extra_callbacks = list(callbacks or ())
        for callback in extra_callbacks:
            self.callbacks.add(callback)

        # ---- 2. 冻结的局部簿记（§9.4.9）。刻意**不**塞进 AgentState：它们是纯运行时
        #         变量，出现在 trace 里只会让 to_dict 的断言变脆。----
        nudged_keys: set[str] = set()
        last_assistant_text: str = ""
        parsed: ParsedAction | None = None
        executor_ref: ToolExecutor = self.executor
        last_finish_reason: str = ""

        parser = self._build_parser()
        disable_after = self._executor_failure_limit(executor_ref)

        def _fail(exc: LiteAgentError, *, output: str | None = None) -> AgentResult:
            """终结为 FAILED 的唯一捷径：mark_finished + RUN_FAILED + 构造结果。

            所有失败出口都走这里，保证"状态已终结 / 事件已发 / metadata 里有 finish_reason"
            这三件事不会漏掉任何一处。
            """
            run_state.mark_finished(AgentStatus.FAILED, error=exc)
            self._emit(
                run_state,
                EventType.RUN_FAILED,
                error_type=type(exc).__name__,
                message=str(exc),
                aborted=False,
            )
            return self._result(
                run_state,
                output=last_assistant_text if output is None else output,
                status=AgentStatus.FAILED,
                error=exc,
                metadata={"finish_reason": last_finish_reason},
            )

        async def _self_correct(exc: ReActParseError) -> AgentResult | None:
            """§9.4.2 的 parse-error 自纠正分支（唯一实现，两个触发点共用）。

            触发点：文本模式 `parser.parse` 抛 `ReActParseError`；以及原生模式
            `finish_reason == "tool_calls"` 但一个 call 都没给（§9.4.6 分支 (c)）。

            **可断言不变式**：一次 parse error 恰好新增 **2** 条消息
            （1 条 assistant + 1 条 NUDGE）。assistant 原文已由步骤 2 写入（唯一写入点），
            v1 在这里又 add 了一次，导致同一条内容在 messages 里出现两遍。
            """
            run_state.parse_errors += 1
            self._emit(
                run_state,
                EventType.PARSE_ERROR,
                step=run_state.step,
                reason=exc.reason,
                offset=exc.offset,
                raw_len=len(exc.raw),
                attempt=run_state.parse_errors,
            )
            # 顺序照 §9.4.2 字面：**先注入反馈，再判上限**。这样"每次 parse error 都新增
            # 2 条消息"是无条件成立的不变式（若先判上限，最后一次就只新增 1 条）。
            feedback = parser.build_parse_error_feedback(exc)
            message = Message.user(feedback, kind="nudge")
            run_state.add_message(message)
            await self.memory.aadd(message)
            run_state.nudges.append(feedback)
            self._emit(run_state, EventType.NUDGE, step=run_state.step, text=feedback)
            if run_state.parse_errors > cfg.max_parse_retries:
                # 消耗 step（D-10），且**不重试 LLM 调用本身**（LLM 层已重试过）——
                # 否则重试会放大成 max_steps × max_parse_retries 次调用。
                return _fail(
                    AgentError(
                        f"model output could not be parsed after {run_state.parse_errors} "
                        f"attempt(s): {exc.reason}",
                        cause=exc,
                    )
                )
            return None

        async def _finalize(resp: LLMResponse, this_parsed: ParsedAction | None) -> AgentResult:
            """§9.4.3 的终结分支。

            **不再 add 一次 assistant 消息**：原始响应已在步骤 2 写入 transcript
            （v1 在这里二次写入，导致最终答案在 messages 里出现两遍）。
            写进记忆的 `Message.assistant(answer)` **不传 `auto_write=True`**：
            §8.5 的 `should_auto_write` 对 `role != "user"` 恒 False，写 True 会让模型自己的
            答案被当成"用户事实"落进长期库、下一轮又被当事实召回。
            """
            nonlocal last_assistant_text
            if mode == "text" and this_parsed is not None:
                raw_answer = this_parsed.final_answer
            else:
                raw_answer = resp.content
            answer = parser.strip_markers(raw_answer or "").strip()
            if not answer:
                # 兜底：绝不返回空字符串（空 output 的 FINISHED 与 FAILED 无法区分）。
                answer = (resp.content or "").strip()
            last_assistant_text = answer
            # [v3 修正] 同一条响应在短期窗口里只留一个版本：步骤 2 已经把**原始**响应
            # 写进 buffer，这里把它原地规范化成最终答案，而不是**再追加一条**
            # （v2 追加 -> text 模式窗口里是 'Final Answer: X' 与 'X' 两条，native 模式
            # 是逐字相同的两条；多轮对话的窗口以 1.5 倍速度膨胀、模型也看到自己的答案两遍）。
            # 末尾不是本轮 assistant 消息时（窗口被裁空等）退化回 aadd，绝不丢答案。
            if not self.memory.replace_last_in_buffer(Message.assistant(answer)):
                await self.memory.aadd(Message.assistant(answer))
            run_state.mark_finished(AgentStatus.FINISHED)
            self._emit(
                run_state,
                EventType.RUN_FINISHED,
                step=run_state.step,
                output_len=len(answer),
                steps=run_state.step,
                usage=run_state.usage.to_dict(),
            )
            return self._result(
                run_state,
                output=answer,
                status=AgentStatus.FINISHED,
                metadata={"finish_reason": last_finish_reason},
            )

        try:
            self._emit(
                run_state,
                EventType.RUN_STARTED,
                input=input,
                mode=mode,
                tools=self.tools.names(),
            )
            run_state.status = AgentStatus.THINKING

            # ---- 0. 用户输入入 buffer（恰好一次，唯一写入点）----
            # MEMORY_WRITE 由 MemoryManager.aadd 自己发（§2.7 的唯一发射者矩阵 + 红线 18），
            # 这里**不得**重复发一条。
            await self.memory.aadd(Message.user(input))

            while run_state.step < max_steps:
                # ---- 0.5 预算与墙钟检查（每轮开头）----
                if max_wall_clock_s is not None:
                    elapsed = utc_now() - run_state.started_at
                    if elapsed > max_wall_clock_s:
                        _LOGGER.warning(
                            "agent %s: wall clock budget exceeded (%.3fs > %.3fs)",
                            self.name,
                            elapsed,
                            max_wall_clock_s,
                        )
                        return _fail(
                            RunTimeoutError(
                                timeout_s=float(max_wall_clock_s), elapsed_s=float(elapsed)
                            )
                        )
                run_state.step += 1
                self._emit(run_state, EventType.STEP_STARTED, step=run_state.step)

                # ---- 1. 取回长期记忆 + 组装 prompt ----
                messages = await self.memory.abuild_prompt(
                    system=self._system_prompt_for(mode),
                    user_input=input if run_state.step == 1 else "",
                    append_user_input=(run_state.step == 1),
                )
                # MEMORY_RETRIEVE 同样由 MemoryManager.abuild_prompt 自己发。
                if cfg.max_prompt_tokens is not None:
                    used_tokens = messages_tokens(messages, self.llm.count_tokens)
                    if used_tokens > cfg.max_prompt_tokens:
                        # SPEC-AMBIGUITY: §5.3 定义了 `max_prompt_tokens`（"单次 prompt 上限"），
                        # 但 §9.4.1 的伪代码没用它。裁决：尊重配置项的存在，用与 token 预算
                        # 完全相同的形状（BUDGET_EXCEEDED + BudgetExceededError）实现，
                        # kind="prompt_tokens" 便于与 "total_tokens" 区分。
                        err = BudgetExceededError(
                            limit=int(cfg.max_prompt_tokens),
                            used=int(used_tokens),
                            kind="prompt_tokens",
                        )
                        self._emit(
                            run_state,
                            EventType.BUDGET_EXCEEDED,
                            kind=err.kind,
                            limit=err.limit,
                            used=err.used,
                        )
                        return _fail(err)

                # ---- 2. 调 LLM ----
                tools_arg = self.tools.schemas(fmt="openai") if mode == "native" else None
                try:
                    resp = await self.llm.achat(
                        messages,
                        tools=tools_arg,
                        tool_choice=tool_choice if mode == "native" else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except asyncio.CancelledError:
                    raise  # §9.4.7：取消原样上抛，交给外层统一处理
                except LiteAgentError as exc:
                    # LLM 层已按 retryable 重试过；到这里说明重试已用尽 -> 终结。
                    run_state.llm_errors += 1
                    _LOGGER.warning("agent %s: LLM call failed: %s", self.name, exc)
                    return _fail(exc)

                last_finish_reason = resp.finish_reason or ""
                run_state.usage += resp.usage
                # assistant 消息入 transcript（**唯一写入点**）并同步进短期窗口。
                # 同一个 Message 对象两处共享：Message 按不可变使用，共享比复制更省
                # （也避免"两处内容一样但对象不同"在调试时看着像 bug）。
                assistant_message = resp.to_message()
                run_state.add_message(assistant_message)
                await self.memory.aadd(assistant_message)
                if resp.content and resp.content.strip():
                    last_assistant_text = resp.content.strip()

                # ---- 2.5 token 预算检查 ----
                if (
                    max_total_tokens is not None
                    and run_state.usage.total_tokens > max_total_tokens
                ):
                    err = BudgetExceededError(
                        limit=int(max_total_tokens),
                        used=run_state.usage.total_tokens,
                        kind="total_tokens",
                    )
                    self._emit(
                        run_state,
                        EventType.BUDGET_EXCEEDED,
                        kind=err.kind,
                        limit=err.limit,
                        used=err.used,
                    )
                    _LOGGER.warning(
                        "agent %s: token budget exceeded (%d > %d)",
                        self.name,
                        err.used,
                        err.limit,
                    )
                    return _fail(err)

                # ---- 3. 决定本轮是"行动"还是"终结" ----
                calls: list[ToolCall] = list(resp.tool_calls) if mode == "native" else []
                parsed = None

                if resp.finish_reason == "content_filter":
                    # (b) 内容过滤：两种模式都**立即失败**（§9.4.6 的表格没有 mode 条件）。
                    # finish_reason 由 _fail 写进 AgentResult.metadata。
                    _LOGGER.warning("agent %s: response blocked by content filter", self.name)
                    return _fail(AgentError("response blocked by content filter"))

                if resp.finish_reason == "length" and not calls:
                    # (a) 截断续写。`not calls` 这个条件是两种读法的并集：
                    # §9.4.1 把本分支放在"原生模式且无 call"下，而 §9.4.6 的表格说触发条件
                    # 是 `mode != "native"`。取并集后：原生模式带 tool_calls 的截断响应仍会
                    # 执行（模型至少给全了一部分调用），其余截断一律续写。
                    run_state.truncation_errors += 1
                    if run_state.truncation_errors > cfg.max_truncation_retries:
                        _LOGGER.warning(
                            "agent %s: output truncated %d time(s); giving up",
                            self.name,
                            run_state.truncation_errors,
                        )
                        return _fail(AgentError("output truncated by max_tokens"))
                    # SPEC-AMBIGUITY: §9.4.6 (a) 要求 `emit LLM_ERROR {reason: "length", ...}`，
                    # 但 §2.7 的归属矩阵把 LLM_ERROR 的唯一发射者写死为
                    # `BaseLLMClient._emit` 并明确"Agent 不得重复发"（红线 18 同义）。
                    # 裁决：**不发 LLM_ERROR**，改发 Agent 自己拥有的 NUDGE 事件；
                    # 截断次数在 `state.truncation_errors` 里可观测（红线 12 不破）。
                    truncation_message = Message.user(_TRUNCATION_NUDGE, kind="nudge")
                    run_state.add_message(truncation_message)
                    await self.memory.aadd(truncation_message)
                    run_state.nudges.append(_TRUNCATION_NUDGE)
                    self._emit(
                        run_state,
                        EventType.NUDGE,
                        step=run_state.step,
                        text=_TRUNCATION_NUDGE,
                    )
                    continue  # 消耗 step；不重试 LLM 调用本身

                if mode == "native" and not calls:
                    if resp.finish_reason == "tool_calls":
                        # (c) 声明了工具却没给出合法 tool_call -> 走自纠正，**不**当成最终答案
                        # （把它当答案是"模型说要用工具，框架却直接结束"的静默错误）。
                        err = ReActParseError(
                            raw=resp.content,
                            offset=0,
                            reason=(
                                "finish_reason='tool_calls' but the response contained "
                                "no tool_call"
                            ),
                        )
                        outcome = await _self_correct(err)
                        if outcome is not None:
                            return outcome
                        continue
                    # (d) 正常终结：finish_reason == "stop" 且无 tool_calls
                    return await _finalize(resp, None)

                if not calls:
                    # 文本模式：解析 Thought/Action/Action Input/Final Answer
                    try:
                        parsed = parser.parse(resp.content)
                    except ReActParseError as exc:
                        outcome = await _self_correct(exc)
                        if outcome is not None:
                            return outcome
                        continue
                    if parsed.thought:
                        self._emit(
                            run_state,
                            EventType.THOUGHT,
                            step=run_state.step,
                            text=parsed.thought,
                        )
                    if parsed.is_action():
                        # Action **优先于** Final Answer（§9.3 步骤 4）：模型"调用工具后
                        # 又补一句 Final Answer"是常见行为，"丢掉工具调用直接终结"是明显
                        # 错误的行为。
                        calls = [
                            ToolCall.create(
                                parsed.action or "",
                                parsed.action_input,
                                call_id=f"call_text_{run_state.step}",
                            )
                        ]
                        self._emit(
                            run_state,
                            EventType.ACTION_PARSED,
                            step=run_state.step,
                            action=parsed.action,
                            arguments=parsed.action_input,
                        )
                    else:
                        # is_final() 或"既无 action 也无 final"（保守：终结而不是空转）
                        return await _finalize(resp, parsed)

                # ---- 4. 重复/无进展动作检测（真正执行之前，三层判定）----
                if cfg.repeat_action_policy != "off" and calls:
                    for call in calls:
                        run_state.record_tool_call(call)
                    over: list[tuple[ToolCall, bool]] = []
                    for call in calls:
                        key = call.canonical_key()
                        hit_key = (
                            run_state.action_counts.get(key, 0) >= cfg.repeat_action_threshold
                        )
                        # 无进展：同一份观察反复出现（同一个错误/同一段结果刷屏）
                        hit_digest = any(
                            value >= cfg.repeat_action_threshold
                            for value in run_state.observation_digests.values()
                        )
                        hit_name = (
                            disable_after > 0
                            and run_state.tool_failure_counts.get(call.name, 0)
                            >= disable_after
                        )
                        if hit_key or hit_digest or hit_name:
                            over.append((call, hit_digest))
                    if over:
                        first_call, first_no_progress = over[0]
                        first_key = first_call.canonical_key()
                        self._emit(
                            run_state,
                            EventType.REPEAT_DETECTED,
                            step=run_state.step,
                            action_key=first_key,
                            count=run_state.repeat_count(first_call),
                        )
                        if cfg.repeat_action_policy in ("nudge", "nudge_then_fail"):
                            if first_key in nudged_keys:
                                # 已经提醒过一次还这么干 -> 结束，别再烧 token 空转
                                return _fail(
                                    RepeatedActionError(
                                        action_key=first_key,
                                        count=run_state.repeat_count(first_call),
                                    )
                                )
                            nudge_text = self._build_nudge_text(
                                first_call,
                                run_state.repeat_count(first_call),
                                run_state,
                                no_progress=first_no_progress,
                            )
                            nudge_message = Message.user(nudge_text, kind="nudge")
                            run_state.add_message(nudge_message)
                            await self.memory.aadd(nudge_message)
                            run_state.nudges.append(nudge_text)
                            nudged_keys.add(first_key)
                            self._emit(
                                run_state,
                                EventType.NUDGE,
                                step=run_state.step,
                                text=nudge_text,
                            )
                            continue  # 消耗 step
                        # policy == "fail"
                        return _fail(
                            RepeatedActionError(
                                action_key=first_key,
                                count=run_state.repeat_count(first_call),
                            )
                        )

                # ---- 5. 执行工具 ----
                # 本处**不**发 TOOL_STARTED/TOOL_FINISHED：它们的唯一发射者是 ToolExecutor
                # （§2.7 + 红线 18），Agent 重复发会让每个工具调用出现两条事件。
                run_state.status = AgentStatus.ACTING
                results = await self._execute_calls(executor_ref, calls)
                run_state.tool_calls.extend(calls)
                run_state.tool_results.extend(results)
                for result in results:
                    if not result.ok:
                        run_state.record_tool_failure(result.name)
                    run_state.record_observation(result)

                # ---- 6. 把观察结果写回 ----
                run_state.status = AgentStatus.OBSERVING
                await self._write_back(
                    mode=mode,
                    results=results,
                    parser=parser,
                    state=run_state,
                    max_observation_chars=cfg.max_observation_chars,
                )

                # ---- 6.5 可选：把 thought 记进历史（文本模式）----
                if (
                    mode == "text"
                    and cfg.include_thought_in_history
                    and parsed is not None
                    and parsed.thought
                ):
                    run_state.add_message(
                        Message.assistant(parser.strip_markers(parsed.thought), kind="react")
                    )

                # ---- 6.7 transcript 上限 ----
                if len(run_state.messages) > cfg.max_transcript_messages:
                    run_state.trim_transcript(cfg.max_transcript_messages)

                # ---- 7. 压缩检查（每轮**一次**）----
                # 放在轮次末尾而不是每条 observation 之后：否则一轮 N 个工具会触发
                # N 次 LLM 摘要判定（§8.6 冻结调用契约）。
                await self.memory.acompress_if_needed()
                self._emit(
                    run_state,
                    EventType.STEP_FINISHED,
                    step=run_state.step,
                    status=run_state.status.value,
                )
                run_state.status = AgentStatus.THINKING

            # ---- while 结束：步数用尽 ----
            _LOGGER.warning("agent %s: max_steps=%s exhausted", self.name, max_steps)
            return _fail(MaxStepsExceededError(max_steps=int(max_steps)))

        except asyncio.CancelledError:
            # §9.4.7：取消必须**继续抛出**（取消传播不能被打断），因此这里
            # **不构造 AgentResult**，只把状态标成 ABORTED 并留下 RUN_FAILED 事件。
            run_state.mark_finished(AgentStatus.ABORTED)
            self._emit(
                run_state,
                EventType.RUN_FAILED,
                error_type="CancelledError",
                message="run cancelled",
                aborted=True,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - 红线 6：arun 永不向外抛业务异常
            if isinstance(exc, LiteAgentError) and self.config.raise_on_error:
                # `_result` 里 raise_on_error 的**有意**上抛：它必然是 LiteAgentError，
                # 而且一定会路过这里（`_fail` 大多在某个 except 子句里被调用，
                # 而那个子句位于本 try 的 body 内，所以外层的 handler 仍然会先接住它）。
                # 这条分支必须放在日志之前，否则每次 raise_on_error 都会打一份"意外错误"
                # 的 traceback —— 把有意的行为报成 bug 比不报更糟。
                raise
            # 走到这里说明是**框架自身的意外**（自定义 executor/memory 抛出的非 LiteAgentError、
            # 或者本文件的 bug）。仍然按契约编码成 FAILED，但必须留痕（红线 10）。
            _LOGGER.exception("agent %s: unexpected error during arun", self.name)
            if self.config.raise_on_error:
                # 同上：错误在日志里已经留痕，这里按 §9.4 的语义 re-raise
                # （"仅当 config.raise_on_error=True 时才 re-raise"）。
                raise
            return _fail(AgentError(f"unexpected error during run: {exc}", cause=exc))
        finally:
            if run_state.finished_at is None:
                # 保证 trace 不被截断（v1 的取消路径没有这一步）。
                run_state.finished_at = utc_now()
            for callback in extra_callbacks:
                self.callbacks.remove(callback)
            # [v2 变更] `self._state` 只在 arun 结束时一次性赋值（运行期状态是局部变量，
            # 否则并发/嵌套运行会互相覆盖）。
            self._state = run_state
            self._run_guard.release()

    # ---------------------------------------------------------------- 事件流
    async def astream(self, input: str, **kwargs: Any) -> AsyncIterator[TraceEvent]:
        """边跑边 yield 事件（§9.4 冻结伪代码）。

        v1 的"在同一个任务里 await self.arun"在字面上不可实现：generator 只有在 `arun`
        返回后才有机会 yield，那时事件早就发完了。因此这里把 `arun` 放进独立 task，
        用一个回调把事件灌进队列，消费者边拿边 yield。

        三个容易写错的点：
        1. **线程判定**用"创建时的 `threading.get_ident()`"而不是 main_thread ——
           `astream` 可能跑在非主线程的 loop 里（subagent 场景），main_thread 判定会
           把同线程的事件错误地走 `call_soon_threadsafe`（多一跳延迟，且顺序可能乱）。
        2. 队列**有界**（1000），满时**丢最新并记 WARNING**，不做 O(n) 淘汰。
        3. `[v3 修正]` 取消**不是**"break 时同步发生"的：Python 不保证 `async for` + `break`
           立刻终结 async generator —— 只有 generator 被回收（或调用方显式
           `await stream.aclose()`）时才会收到 `GeneratorExit`，`finally` 才会
           `task.cancel()` + `gather(return_exceptions=True)`。也就是说：
           * 写 `async for ev in agent.astream(...): ... break`（引用立刻掉 0）时，取消会
             延迟至少一个 loop tick 才发生；
           * 若调用方**持有 stream 引用**再 break，generator 永不回收、finalizer 永不触发、
             `finally` 永不执行 —— 运行**不会**被取消，arun 会跑完整个 ReAct 循环并写记忆
             （这正是 v2 注释声称已被修掉的 v1 行为）。
           需要立刻停止请显式 `await stream.aclose()`（或用 `contextlib.aclosing`），
           且不要在 break 后立刻对同一个 Agent 再发 run（run guard 仍被占用，
           会按 §9.4 的不可重入契约返回 FAILED "already has a run in flight"）。

           另外：**取消失效还有一个更隐蔽的来源**。v2 的 `ToolExecutor._invoke` 用
           `asyncio.wait_for` 包超时，而 3.10 的实现在"内层 future 与取消落在同一个 tick"
           时会 `return fut.result()` 把取消**静默吞掉**（GH-86296 / bpo-42130）。
           于是带工具调用的 agent 里，`task.cancel()` 发出去了也返回 True，run 却继续跑。
           v3 已把 `_invoke` 的超时换成取消安全的 `_await_with_deadline`，回归测试见
           `tests/test_agent_features.py::AstreamTests::test_early_break_stops_a_tool_using_run`。
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=_ASTREAM_QUEUE_MAXSIZE)
        sentinel = object()
        loop = asyncio.get_running_loop()
        owner_thread = threading.get_ident()

        def _safe_put(item: Any) -> None:
            """队列的**唯一**写入点：既被本线程直接调用，也被 `call_soon_threadsafe` 调用。"""
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                _LOGGER.warning(
                    "astream queue full; dropping event %s",
                    getattr(getattr(item, "type", None), "value", item),
                )

        def _put(event: TraceEvent) -> None:
            if threading.get_ident() != owner_thread:
                try:
                    loop.call_soon_threadsafe(_safe_put, event)
                except RuntimeError as exc:
                    # M-4：对已关闭的 loop 调 call_soon_threadsafe 会抛
                    # RuntimeError: Event loop is closed。丢事件但留痕，绝不往外冒。
                    _LOGGER.warning(
                        "astream: cannot deliver event %s (loop closed): %s",
                        event.type.value,
                        exc,
                    )
            else:
                _safe_put(event)

        def _push_sentinel(_task: "asyncio.Task[Any]") -> None:
            """`arun` 结束时投递终止哨兵（done_callback 可能在线程池里被调用）。"""
            try:
                if loop.is_closed():  # M-4：先检查，再投递
                    _LOGGER.warning(
                        "astream: loop closed before the terminating sentinel was delivered"
                    )
                    return
                loop.call_soon_threadsafe(_safe_put, sentinel)
            except RuntimeError as exc:  # pragma: no cover - M-4 的竞态窗口
                _LOGGER.warning("astream: failed to deliver the terminating sentinel: %s", exc)

        callback = FunctionCallback(_put)
        self.callbacks.add(callback)
        task = asyncio.create_task(self.arun(input, **kwargs))
        task.add_done_callback(_push_sentinel)
        try:
            while True:
                event = await queue.get()
                if event is sentinel:
                    break
                yield event
        finally:
            self.callbacks.remove(callback)
            if not task.done():
                task.cancel()
            # 不吞 GeneratorExit / CancelledError：gather 只把子 task 的异常收成返回值。
            await asyncio.gather(task, return_exceptions=True)

    # ---------------------------------------------------------------- 维护
    def reset(self, *, clear_memory: bool = False) -> None:
        """把 Agent 复位成"未运行过"的样子（`_state` 换成 IDLE 空状态）。

        `clear_memory=True` 时顺带清空短期记忆。同步方法里不能 `await memory.aclear()`，
        所以走 `run_sync`；**在运行中的 loop 里** `run_sync` 会抛 `ConfigError`，
        此时退化为同步清空短期窗口并记 WARNING（降级必须留痕，红线 12）。
        """
        self._state = AgentState.create("", agent_name=self.name)
        if not clear_memory:
            return
        try:
            run_sync(lambda: self.memory.aclear(long_term=False, summary=True))
        except (ConfigError, RuntimeError) as exc:
            _LOGGER.warning(
                "Agent.reset(clear_memory=True) degraded to a short-term buffer clear "
                "(cannot run_sync inside a running loop): %s",
                exc,
            )
            self.memory.buffer.clear()

    def add_tool(self, tool: Tool) -> None:
        """注册一个工具（重名 -> `ToolDefinitionError`，不静默覆盖）。"""
        self.tools.register(tool)

    def remove_tool(self, name: str) -> None:
        """注销一个工具（不存在 -> `ToolNotFoundError`）。"""
        self.tools.unregister(name)

    def describe(self) -> dict[str, Any]:
        """给 CLI / 调试用的自述：`{"name","description","model","tools","mode","max_steps"}`。"""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.llm.model,
            "tools": self.tools.names(),
            "mode": self._effective_mode(),
            "max_steps": self.config.max_steps,
        }

    async def aclose(self) -> None:
        """关闭**自建**的 executor（外部传入的不关 —— 所有权是调用方的，§9.4）。"""
        if self._owns_executor:
            await self.executor.aclose()

    # ---------------------------------------------------------------- 私有 helper
    def _effective_mode(self) -> str:
        """不加锁、不抛异常的 mode 解析（`describe()` 与 `_system_prompt()` 用）。

        `config.mode == "native"` 但 llm 不支持时会抛 `ConfigError`（§9.4.8 的冻结行为），
        可这两个探测型 API 不该因为一个配置错误就炸掉 CLI。裁决：**降级为 text 并记
        WARNING**（红线 12：降级必须留痕），真正的运行路径（`arun`）仍然显式抛错。
        """
        mode_cfg = self.config.mode
        if mode_cfg not in _VALID_MODES:
            _LOGGER.warning(
                "agent %s: unknown mode %r; falling back to 'auto'", self.name, mode_cfg
            )
            mode_cfg = "auto"
        if mode_cfg == "auto":
            return self.llm.resolve_mode(has_tools=len(self.tools) > 0)
        if mode_cfg == "native" and not self.llm.supports_tool_calling:
            _LOGGER.warning(
                "agent %s: mode='native' but %s does not support tool calling; "
                "describing/rendering as 'text'",
                self.name,
                type(self.llm).__name__,
            )
            return "text"
        return mode_cfg

    def _system_prompt(self) -> str:
        """system prompt 的**模式相关**渲染（§9.4 冻结规则）。

        - `config.system_prompt is not None` -> 直接返回它（不做渲染，调用方要完全控制）；
        - 否则用 `config.system_prompt_template` 渲染，**native 模式不把工具清单塞进
          prompt**（工具已经通过 `tools=` 参数结构化地给出，再塞一遍是纯烧 token）；
          但**仍使用同一个模板** —— 模板里的 "Never invent tool names" 之类的规则对
          native 同样有效。
        """
        return self._system_prompt_for(self._effective_mode())

    def _system_prompt_for(self, mode: str) -> str:
        """`_system_prompt` 的实现体：由 `arun` 传入本轮**已解析**的 mode。

        §9.4 写的是"先按 text 渲染，native 时再渲染一次"—— 两次渲染的结果只有
        `tools` 那一个占位符不同，先渲染再丢弃纯属浪费，这里等价地只渲染一次。
        """
        cfg = self.config
        if cfg.system_prompt is not None:
            return cfg.system_prompt
        tools_text = self.tools.to_prompt(fmt="text", max_tools=DEFAULT_MAX_TOOLS_IN_PROMPT)
        tool_names = ", ".join(self.tools.names())[:2000]
        if mode == "native":
            tools_text = "(provided via the tools parameter)"
        return render_template(
            cfg.system_prompt_template,
            {"name": self.name, "tools": tools_text, "tool_names": tool_names},
        )

    def _build_parser(self) -> ReActParser:
        """每轮 run 造一个解析器：工具集可能在本轮被改（`add_tool`），解析器必须跟上。"""
        tool_param_names: dict[str, tuple[str, ...]] = {}
        for tool in self.tools.list():
            properties = (tool.spec.parameters or {}).get("properties") or {}
            tool_param_names[tool.name] = tuple(properties.keys())
        return ReActParser(
            tool_names=self.tools.names(), tool_param_names=tool_param_names
        )

    def _executor_failure_limit(self, executor: ToolExecutor) -> int:
        """熔断阈值。

        SPEC-AMBIGUITY: §9.4.1 的伪代码写 `config.disable_tool_after_failures`，但该字段
        属于 `ExecutorConfig` 而不是 `AgentConfig`（§5.3 逐字段可查）。裁决：读
        `self.executor.config.disable_tool_after_failures` —— 语义上就该与 executor 自己的
        熔断阈值同源（两者判定的是同一件事），自定义 executor 没有 `config` 时回退到
        `DEFAULT_TOOL_FAILURE_LIMIT` 并记 debug 日志。
        """
        executor_config = getattr(executor, "config", None)
        limit = getattr(executor_config, "disable_tool_after_failures", None)
        if isinstance(limit, int) and not isinstance(limit, bool):
            return limit
        _LOGGER.debug(
            "agent %s: executor %r exposes no disable_tool_after_failures; "
            "using DEFAULT_TOOL_FAILURE_LIMIT",
            self.name,
            type(executor).__name__,
        )
        return DEFAULT_TOOL_FAILURE_LIMIT

    def _last_result_text(self, state: AgentState, call: ToolCall) -> str:
        """nudge 里回放的"上一次结果"：优先同名工具的最近一条，退到任意最近一条。"""
        for result in reversed(state.tool_results):
            if result.name == call.name:
                return result.to_observation(max_chars=1000)
        return "(no previous result)"

    def _build_nudge_text(
        self,
        call: ToolCall,
        count: int,
        state: AgentState,
        *,
        no_progress: bool,
    ) -> str:
        """§9.4.4 的 nudge 文本（冻结字面量）。

        `{n}` 是同一 canonical_key 的累计次数；命中"无进展"时在末尾追加一句 ——
        `"Thought: ...\\nFinal Answer: ..."` 里的 `\\n` 是**字面两字符**（原文写在一行里，
        这里逐字保留）。
        """
        text = (
            f"You already called {call.name} with these exact arguments {count} times.\n"
            f"The previous result was:\n{self._last_result_text(state, call)}\n"
            "Do not repeat it. Either use a different tool/arguments, or give your final answer now\n"
            'as "Thought: ...\\nFinal Answer: ...".'
        )
        if no_progress:
            # "你的最近 n 次调用返回了完全相同的结果" —— n 取出现次数最多的那份摘要的计数，
            # 它才是"同一份观察反复出现"的实际规模。
            identical = max(state.observation_digests.values(), default=count)
            text += "\n" + _NO_PROGRESS_SUFFIX.format(n=identical)
        return text

    async def _execute_calls(
        self, executor: ToolExecutor, calls: Sequence[ToolCall]
    ) -> list[ToolResult]:
        """执行本轮的全部工具调用（§9.4.1 步骤 5）。

        `parallel_tool_calls=True` 且一轮多个调用时走 `execute_many`（返回顺序与 calls
        严格一致，§7.4.2）；否则逐个 `execute`（文本模式天然一轮一个 action）。

        **防御性包装**：`execute`/`execute_many` 的契约是"永不抛业务异常"，但 `Agent` 的
        executor 可以由用户传入自定义实现（红线 6 明确要求防御性捕获）。任何异常都转成
        `ToolResult.failure` 回灌给模型 —— 崩掉整轮 run 对模型没有任何信息量。
        """
        call_list = list(calls)
        if not call_list:
            return []
        if self.config.parallel_tool_calls and len(call_list) > 1:
            try:
                results = list(await executor.execute_many(call_list))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 自定义 executor 可能抛任何东西
                _LOGGER.warning(
                    "agent %s: custom executor raised from execute_many (%s); "
                    "encoding it as per-call failures",
                    self.name,
                    type(exc).__name__,
                )
                return [self._executor_failure(call, exc) for call in call_list]
            if len(results) != len(call_list):
                # 契约破坏：长度不齐会让"结果与调用一一对应"的所有下游断言失效。
                _LOGGER.warning(
                    "agent %s: execute_many returned %d result(s) for %d call(s); "
                    "padding the tail with failures",
                    self.name,
                    len(results),
                    len(call_list),
                )
                padded = results[: len(call_list)]
                for call in call_list[len(padded):]:
                    padded.append(
                        self._executor_failure(
                            call,
                            ToolExecutionError(
                                tool_name=call.name,
                                call_id=call.id,
                                message="executor returned fewer results than calls",
                            ),
                        )
                    )
                return padded
            return results

        results = []
        for call in call_list:
            try:
                results.append(await executor.execute(call))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 同上
                _LOGGER.warning(
                    "agent %s: custom executor raised for tool %s (%s); encoding it as a failure",
                    self.name,
                    call.name,
                    type(exc).__name__,
                )
                results.append(self._executor_failure(call, exc))
        return results

    @staticmethod
    def _executor_failure(call: ToolCall, exc: BaseException) -> ToolResult:
        """把 executor 抛出的异常转成一条"看起来像工具自己失败"的结果。"""
        if not isinstance(exc, LiteAgentError):
            exc = ToolExecutionError(
                tool_name=call.name, call_id=call.id, message=str(exc), cause=exc
            )
        return ToolResult.failure(call, exc)

    async def _write_back(
        self,
        *,
        mode: str,
        results: Sequence[ToolResult],
        parser: ReActParser,
        state: AgentState,
        max_observation_chars: int,
    ) -> None:
        """把工具结果写回模型可见的上下文（§9.4.5 的两种回灌形态）。

        - native：每条结果一条 `role="tool"` 消息，带 `tool_call_id`（OpenAI 协议要求
          tool 消息必须能配对到 assistant 的 tool_calls，否则 API 400）；
        - text：把**全部**结果渲染成一条 `Observation:` 的 user 消息 —— 文本 ReAct 没有
          原生 tool 角色，Observation 只能以"用户侧输入"的形式回灌。
        """
        if mode == "native":
            for result in results:
                message = result.to_message()
                state.add_message(message)
                await self.memory.aadd(message)
            return
        observation = parser.build_observation(
            results,
            # token 预算与字符上限联动（§8.6 的 tokenizer_chars_budget）：否则会出现
            # "单条观察占满半个上下文"。
            max_chars=min(max_observation_chars, self.memory.tokenizer_chars_budget()),
            step=state.step,
        )
        message = Message.observation(observation, step=state.step)
        state.add_message(message)
        await self.memory.aadd(message)

    def _emit(
        self,
        state: AgentState,
        event_type: EventType | str,
        *,
        step: int | None = None,
        **data: Any,
    ) -> TraceEvent:
        """Agent 侧事件的唯一出口：自动补 run_id / agent_name / step（§2.7）。

        `step` 默认取当前轮次；显式传 `step=` 的场景是"事件属于某个特定轮"（例如
        PARSE_ERROR 属于本轮而不是下一轮）。`data` 里**不得**出现保留键
        （type/run_id/agent_name/step/ts）——`emit_type` 会在构造前就抛 `ConfigError`。
        """
        return self.callbacks.emit_type(
            event_type,
            run_id=state.run_id,
            agent_name=self.name,
            step=state.step if step is None else step,
            **data,
        )

    def _result(
        self,
        state: AgentState,
        *,
        output: str,
        status: AgentStatus,
        error: LiteAgentError | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        """所有 `AgentResult` 构造的唯一出口（关键字参数，禁止位置参数）。

        为什么必须收敛到一处：`AgentResult` 的第一个字段是 `output`，位置参数构造会把
        `AgentStatus.FAILED` 塞进 `output` 而 `status` 保持默认的 `FINISHED`
        （"失败却 ok=True"）。这里还统一补 `cost_usd` 与 `finish_reason`，
        并作为 `raise_on_error=True` 的**唯一**触发点 —— 放在 `_result` 里保证不会漏。
        """
        merged: dict[str, Any] = dict(state.scratchpad.get("agent_metadata") or {})
        merged.update(metadata or {})
        merged.setdefault("cost_usd", estimate_cost_usd(state.usage, model=self.llm.model))
        if self.config.raise_on_error and status != AgentStatus.FINISHED:
            raise error or AgentError("agent failed without an error object")
        return AgentResult(
            output=output,
            status=status,
            steps=state.step,
            tool_calls=list(state.tool_calls),
            tool_results=list(state.tool_results),
            usage=state.usage,
            error=error,
            state=state,
            duration_ms=state.duration_ms,
            agent_name=self.name,
            metadata=merged,
        )
