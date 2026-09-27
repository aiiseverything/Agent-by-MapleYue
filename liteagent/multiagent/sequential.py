from __future__ import annotations

# liteagent/multiagent/sequential.py —— 顺序编排：A -> B -> C（冻结规范 §10.3）
#
# 顺序编排是最"无聊"但也最常用的多 Agent 形态：每一步的输入由上一步的输出构造。
# 本文件的四个设计要点（面试可以直接讲）：
#
#   1. **输入靠模板而不是靠约定**：`input_template` 里可写 `{input}`（原始用户输入）、
#      `{prev}`（上一步输出）、`{steps[x]}`（任意历史阶段的输出）。缺省规则冻结为
#      "第 0 步用 {input}，其余用 {prev}"，所以最简单的三步流水线**不需要写任何模板**。
#      模板用 `config.render_template` 渲染 —— 它用 `_SafeDict`，缺 key 时保留 `{key}`
#      字面量而不是抛 KeyError，因此"引用一个还不存在的阶段"是可见的（输出里留着
#      `{steps[x]}`）而不是崩掉。
#
#   2. **每个阶段一个全新的 `AgentState`**：复用父 state 会让阶段之间的 transcript 互相
#      污染（第 3 步会"看见"第 1 步的 Thought），也让 trace 无法按阶段切分。父上下文
#      只通过 `state.scratchpad["delegation"]` 传下去（§2.4 的预留 key）。
#
#   3. **失败有三种走法（propagate_failure）**：raise（抛 DelegationError）/ return
#      （立即停，把失败编码进 AgentResult）/ continue（把失败当作观察继续）。`optional=True`
#      的阶段无视 propagate_failure，一律"继续"。注意 v2 修掉了 v1 的一个 bug：失败时的
#      `output` 只有**一个**公式（见下面 `_FAILURE_OUTPUT` 注释处），v1 写了两个互斥的值。
#
#   4. **黑板是团队共享的**：每个阶段的输出都写黑板（`tags=("stage",)`），key 默认是阶段名，
#      可用 `output_key` 覆盖（两个阶段想写同一个 key 时用得上）。`metadata["blackboard"]`
#      里带一份快照，CLI/trace 不必再去读黑板。
#
# 依赖方向（§1.1 / E3）：只依赖 multiagent/base 与更低层；`Blackboard` 只在注解里出现，
# 因此走 TYPE_CHECKING（运行期由 base 持有实例，本模块不 new 黑板）。

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from liteagent.agent.callbacks import CallbackLike, EventType
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import TeamConfig, render_template, truncate_head_tail, utc_now
from liteagent.errors import ConfigError, DelegationError
from liteagent.multiagent.base import AgentLike, DelegationContext, MultiAgent

if TYPE_CHECKING:  # 带 `from __future__ import annotations`，注解不执行 -> 无运行期依赖
    from liteagent.multiagent.blackboard import Blackboard

__all__ = ["SequentialAgent", "SequentialStep"]

logger = logging.getLogger("liteagent.multiagent")

# `propagate_failure` 的合法取值（与 §5.3 的 TeamConfig 注释一致）。写成常量是为了
# 让"取值写错了"变成一条清晰的 ConfigError，而不是悄悄走成默认分支。
PROPAGATE_MODES: tuple[str, ...] = ("return", "raise", "continue")


def _status_value(status: Any) -> str:
    """把 `AgentStatus` / 字符串统一成 `"FINISHED"` 这样的字符串。

    事件 data 里的值必须已过 `to_jsonable`（§2.7）；这里直接给字符串，
    比依赖下游对 Enum 的宽容处理更省事。假 Agent（测试里的 stub）可能塞字符串，
    因此对两种情况都做兼容。
    """
    return str(getattr(status, "value", status))


def _is_finished(status: Any) -> bool:
    """`status == FINISHED` 的宽容判定（成员、字符串值、字符串名都认）。"""
    return _status_value(status).upper() == AgentStatus.FINISHED.value


@dataclass
class SequentialStep:
    """一个流水线阶段：一个 Agent + 它怎么拿输入 + 它的输出写到哪里。"""

    agent: AgentLike
    # 阶段名：默认取 agent.name。`Optional` 只是为了让用户可以不写；__post_init__ 之后
    # 一定是非空字符串（下游代码因此不必到处 `or "agent"`）。
    name: str | None = None
    input_template: str | None = None  # None -> 第 0 步 "{input}"，其余 "{prev}"
    output_key: str | None = None      # 非 None 时写入 blackboard（默认 name）
    optional: bool = False             # True 时失败不中止
    max_chars: int = 0                 # >0 时对输出截断后再传给下一步

    def __post_init__(self) -> None:
        """补默认阶段名。

        名字为空会一路传到黑板 key 与事件里的 `to` 字段（`write("")` 会抛 ConfigError），
        所以在这里就把它定死，避免"构造时看着没事、跑起来才炸"。
        """
        if not self.name:
            self.name = str(getattr(self.agent, "name", "") or "agent")

    def to_dict(self) -> dict[str, Any]:
        """阶段配置的自描述（CLI `multi` 与 trace 用）。**不含 agent 实例**（不可序列化）。"""
        return {
            "name": self.name,
            "agent": getattr(self.agent, "name", type(self.agent).__name__),
            "input_template": self.input_template,
            "output_key": self.output_key,
            "optional": self.optional,
            "max_chars": self.max_chars,
        }


class SequentialAgent(MultiAgent):
    """顺序编排：A -> B -> C，每一步的输入由模板从上一步输出构造。

    执行语义逐条冻结在 §10.3；本类的 `arun` 里每个步骤都有对应的字母注释（a~i），
    便于与规范对读。
    """

    steps: list[SequentialStep]

    def __init__(self, steps: Sequence[SequentialStep | AgentLike], *,
                 name: str = "sequential", config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: Blackboard | None = None) -> None:
        """收"混合序列"：`SequentialStep` 与裸 `AgentLike` 都可以写。

        允许裸 Agent 是刻意的：最常见的用法（三步流水线、不需要自定义模板）应当
        写成 `SequentialAgent([a, b, c])`，而不是三行 dataclass 构造。
        """
        normalized: list[SequentialStep] = []
        for item in steps:
            if isinstance(item, SequentialStep):
                normalized.append(item)
                continue
            if hasattr(item, "arun") and hasattr(item, "name"):
                normalized.append(SequentialStep(agent=item))
                continue
            # 早失败：把一个"不是 Agent 的东西"放进流水线，晚一点会在 arun 里
            # 报一个完全看不懂的 AttributeError。
            raise ConfigError(
                "SequentialAgent steps must be SequentialStep or AgentLike (with .name/.arun); "
                f"got {type(item).__name__}"
            )
        # `agents` 传给基类：`describe()` / `aclose()` 都按 agents 遍历。
        super().__init__(
            [step.agent for step in normalized],
            name=name, config=config, callbacks=callbacks, blackboard=blackboard,
        )
        self.steps = normalized

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult:
        """跑完整条流水线。

        `state` 参数**只用于取 run_id**（规范步骤 1 的 `root_run_id`）：每个阶段都会新建
        自己的 `AgentState`，父 state 不被复用（见文件头第 2 点）。`**kwargs` 刻意不透传
        给子 Agent —— 规范的冻结语义是"子 Agent 只收 stage_input 与 state（MultiAgent
        再收 context）"，把未知 kwargs 透传下去会让 `Agent.arun` 的 overrides 白名单
        抛 ConfigError。
        """
        ctx = context if context is not None else DelegationContext(
            stack=[self.name],
            depth=0,
            root_run_id=getattr(state, "run_id", "") or "",
            # budget 的默认值来源是 config.max_rounds（§10.4 的落点表），
            # 否则 max_rounds 只是个没人读的装饰品。
            budget=self.config.max_rounds,
        )
        run_id = ctx.root_run_id or getattr(state, "run_id", "") or ""

        previous_output = ""
        stage_outputs: dict[str, str] = {}     # 模板变量 {steps[name]}
        summaries: list[dict[str, Any]] = []   # metadata["steps"]
        results: list[AgentResult] = []        # usage / steps 聚合
        last_child_state: AgentState | None = None

        for index, step in enumerate(self.steps):
            step_name = step.name or f"step{index}"

            # (a) 三道闸：深度 / 预算 / 环。判决点在这里而不是在子 Agent 里，
            #     因为"能不能再往下委派"只有编排器知道（它持有 stack 与 budget）。
            self._check_depth(ctx, to_agent=step_name)

            # (b) 渲染阶段输入。缺省模板按位置选择：第一步吃原始输入，其余吃上一步输出。
            template = step.input_template or ("{input}" if index == 0 else "{prev}")
            values = {
                "input": input,
                "prev": previous_output,
                "steps": dict(stage_outputs),  # 传副本：模板不该看到本步尚未写入的东西
            }
            try:
                stage_input = render_template(template, values)
            except Exception as exc:  # noqa: BLE001 - 模板里的任何毛病都归为配置错误
                # `render_template` 用 format_map，格式串非法（如 "{prev!q}"）会抛
                # ValueError/KeyError。这是**配置级**错误，不是运行期业务错误，
                # 所以包成 ConfigError 并保留原异常链。
                raise ConfigError(
                    f"step {step_name!r} input_template {template!r} failed to render: {exc}",
                    context={"step": step_name, "template": template},
                    cause=exc,
                ) from exc

            # (c) 委派事件。"from"/"to" 只能用 dict 展开传 —— `from` 是 Python 关键字，
            #     写成关键字参数会直接语法错误。`step`（序号）是 emit_type 的保留键，
            #     它进 TraceEvent.step 而不是事件的 data（§2.7 的公共键）。
            self.callbacks.emit_type(
                EventType.AGENT_DELEGATE,
                run_id=run_id, agent_name=self.name, step=index,
                **{"from": self.name, "to": step_name,
                   "depth": ctx.depth + 1, "refused": False},
            )

            # (d) 子状态：每次新建，父 state 不复用；委派上下文注入 scratchpad。
            child_state = AgentState.create(stage_input, agent_name=step_name)
            child_context = ctx.child(step_name)
            child_state.scratchpad["delegation"] = child_context
            last_child_state = child_state

            # (e) share_memory 注入 + 真正执行。
            self._apply_shared_memory(step.agent)
            started = utc_now()
            if isinstance(step.agent, MultiAgent):
                # 只有 MultiAgent 认识 `context=`；普通 Agent.arun 会把它当非法 override
                # 抛 ConfigError（规范步骤 e 的冻结分支）。
                result = await step.agent.arun(
                    stage_input, state=child_state, context=child_context
                )
            else:
                result = await step.agent.arun(stage_input, state=child_state)
            duration_ms = max(0.0, (utc_now() - started) * 1000.0)

            status_value = _status_value(result.status)
            failed = not _is_finished(result.status)
            raw_output = result.output or ""

            # (f) 返回事件：在失败处理**之前**发，保证"每次委派都有配对的两条事件"，
            #     即使随后要 raise，trace 里也能看到子 Agent 到底返回了什么。
            self.callbacks.emit_type(
                EventType.AGENT_RETURN,
                run_id=run_id, agent_name=self.name, step=index,
                **{"from": self.name, "to": step_name, "status": status_value,
                   "steps": int(getattr(result, "steps", 0) or 0),
                   "output_len": len(raw_output), "duration_ms": duration_ms,
                   "failed": failed},
            )

            # (g) 截断：只截传给下一步的版本，原始输出仍在 `result` 里（不丢信息）。
            output = (
                truncate_head_tail(raw_output, step.max_chars)
                if step.max_chars > 0 else raw_output
            )

            # (h) 写黑板 + 记入 {steps[name]}。key 默认是阶段名，可用 output_key 覆盖。
            self.blackboard.write(
                step.output_key or step_name, output,
                author=step_name, tags=("stage",),
            )
            stage_outputs[step_name] = output
            results.append(result)

            # (i) 失败处理。
            if failed:
                output = self._handle_failure(
                    step=step, step_name=step_name, index=index, result=result,
                    output=output, previous_output=previous_output,
                    summaries=summaries, results=results, child_state=child_state,
                )
                if isinstance(output, AgentResult):
                    # propagate_failure="return"：立即停止，返回"部分成功"的结果。
                    return output

            summaries.append({
                "name": step_name,
                "status": status_value,
                "steps": int(getattr(result, "steps", 0) or 0),
                "output": output,
                "duration_ms": duration_ms,
            })
            previous_output = output

        # 全部成功（或全部被 optional/continue 吞掉）-> 返回最后一步的结果，
        # 但 usage/steps 换成全阶段之和，metadata 换成流水线视角。
        if not results:
            # 空流水线：比抛异常更友好的选择是给一个"什么都没发生"的 FINISHED 结果，
            # 调用方仍能拿到黑板快照与空 steps。
            logger.warning("SequentialAgent %r has no steps; returning an empty result", self.name)
            return AgentResult(
                output="", status=AgentStatus.FINISHED, agent_name=self.name,
                metadata={"steps": [], "blackboard": self.blackboard.snapshot()},
            )

        last = results[-1]
        return AgentResult(
            output=previous_output,
            status=last.status,
            steps=sum(int(getattr(item, "steps", 0) or 0) for item in results),
            tool_calls=list(getattr(last, "tool_calls", []) or []),
            tool_results=list(getattr(last, "tool_results", []) or []),
            usage=self._aggregate_usage(results),
            error=last.error,
            # 规范步骤 4：result.state 指向最后一个阶段的 child state（调试用）。
            state=last_child_state,
            duration_ms=float(getattr(last, "duration_ms", 0.0) or 0.0),
            agent_name=self.name,
            metadata={"steps": summaries, "blackboard": self.blackboard.snapshot()},
        )

    # ------------------------------------------------------------------
    # 失败分支（单独成方法，避免 arun 的主干被三路分支淹没）
    # ------------------------------------------------------------------

    def _handle_failure(self, *, step: SequentialStep, step_name: str, index: int,
                        result: AgentResult, output: str, previous_output: str,
                        summaries: list[dict[str, Any]], results: list[AgentResult],
                        child_state: AgentState) -> str | AgentResult:
        """某个阶段失败时的三种走向；返回 `str` 表示"继续"，返回 `AgentResult` 表示"停下"。

        冻结的 output 公式（v2 只保留**一个**，v1 在同一条里写了两个互斥的值）：
        `output = result.output or previous_output or ""`
        —— 优先用"失败阶段已有的部分输出"；它是空串才回退"上一步的输出"；最后兜底空串。
        为什么这个顺序：模型常常在失败前已经打完半截答案，那半截往往比上一步的输出
        更贴近用户想要的东西；只有它真的什么都没有时才回退。
        """
        mode = (self.config.propagate_failure or "return").strip().lower()
        if mode not in PROPAGATE_MODES:
            raise ConfigError(
                f"TeamConfig.propagate_failure must be one of {list(PROPAGATE_MODES)}; got "
                f"{self.config.propagate_failure!r}"
            )

        if step.optional:
            # `optional=True` 时**无视** propagate_failure，一律 continue 语义。
            logger.info(
                "sequential: stage %r failed (status=%s) but is optional; continuing",
                step_name, _status_value(result.status),
            )
            return f"[stage {step_name} failed: {result.error}]"

        if mode == "raise":
            raise DelegationError(
                from_agent=self.name, to_agent=step_name,
                message=(
                    f"stage {step_name!r} failed with status "
                    f"{_status_value(result.status)}: {result.error}"
                ),
                context={"step": step_name, "index": index,
                         "failed": True, "error": str(result.error)},
                cause=result.error,
            )

        if mode == "return":
            # 立即停止。`metadata["steps"]` 只含**已完成**（成功跑完）的阶段摘要 ——
            # 失败阶段自己的状态由 `failed_stage` 表达，避免同一件事写两遍而互相矛盾。
            return AgentResult(
                output=result.output or previous_output or "",
                status=result.status,
                steps=sum(int(getattr(item, "steps", 0) or 0) for item in results),
                usage=self._aggregate_usage(results),
                error=result.error,
                state=child_state,
                agent_name=self.name,
                metadata={"failed_stage": step_name, "steps": list(summaries)},
            )

        # mode == "continue"：把失败变成给下一步看的观察文本。
        return f"[stage {step_name} failed: {result.error}]"
