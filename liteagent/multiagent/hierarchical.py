from __future__ import annotations

# liteagent/multiagent/hierarchical.py —— 层级编排：manager 分解 + workers 协同（冻结规范 §10.4）
#
# 这是整个框架里并发与安全最密集的一处。四条设计主线（面试可讲）：
#
#   1. **委派即工具（delegation as a tool）**：每个 worker 被包装成
#      `delegate_to_<name>` 工具塞进 manager 的工具集。manager 是**普通 Agent**，
#      它不需要知道"多 Agent"这回事 —— ReAct 循环照样跑，只是工具箱里多了几个
#      "找同事帮忙"的工具。这条路线的好处是复用（不需要给编排层写新的状态机），
#      代价是委派结果必须被压缩成**字符串**（模型侧只认文本）。
#
#   2. **同步工具 + 新建 loop**：delegate 工具是**同步函数**（`pass_style="mapping"`），
#      会被 manager 的 executor 丢到工作线程里跑；闭包内部再用 `run_sync(worker.arun)`
#      启动一个**全新的 event loop**。因此限流与互斥**必须**是 threading 原语
#      （§13 红线 13）：loop-bound 的 asyncio.Semaphore 在每个 delegate 里都是全新的、
#      计数永远是满的 —— 那等于没有限流（v1 的真实缺陷）。
#
#   3. **上下文只能走 contextvar**：`DelegationContext`（委派栈/深度/预算）放在
#      `_CURRENT_CTX` 里，而不是实例属性 —— manager 可能被并发复用，实例属性会被
#      多个委派互相覆盖。**并且它绝不从 `args` 里取**：工具参数叫 `extra_context`
#      （v1 叫 `context`，与 contextvar 同名，实现者几乎必然写成
#      `ctx = args.get("context") or _CURRENT_CTX.get()`，于是环检测/深度判定被模型
#      传进来的任意文本顶掉 -> 无限委派。这是本文件最值得讲的一个真实安全修复）。
#
#   4. **失败是观察，不是异常**：worker 失败（返回 FAILED、甚至抛异常）都被压缩成
#      字符串回灌给 manager，manager 可以重试/改派/放弃 —— 单个 worker 的失败
#      **从不**中止 manager。只有 manager 自己整体失败时才轮到 `propagate_failure`。
#
# 另外 `arun_plan` 提供了"先规划、再按依赖分层执行"的第二条路径（Plan/depends_on
# 在 v1 里是死代码）：它跑在**父 loop** 内，所以这里**可以**用 asyncio 原语
# （`LoopBoundPool` 懒创建的 per-loop 信号量）。

import asyncio
import contextvars
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from liteagent.agent.callbacks import CallbackLike, EventType
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import (
    NO_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT_S,
    SUBAGENT_HEAD_RATIO,
    LoopBoundPool,
    TeamConfig,
    render_template,
    run_sync,
    truncate_head_tail,
    utc_now,
)
from liteagent.errors import (
    ConfigError,
    CycleDetectedError,
    DelegationError,
    LiteAgentError,
    ReActParseError,
    ToolDefinitionError,
)
from liteagent.llm.message import Message
from liteagent.multiagent.base import (
    AgentLike,
    DelegationContext,
    MultiAgent,
    compress_subagent_output,
)
from liteagent.tools.base import Tool, make_function_tool
from liteagent.tools.registry import ToolRegistry

if TYPE_CHECKING:  # 注解专用（带 `from __future__ import annotations`，运行期不求值）
    from liteagent.agent.agent import Agent
    from liteagent.multiagent.blackboard import Blackboard

__all__ = [
    "HierarchicalAgent",
    "Plan",
    "SubTask",
    "DECOMPOSE_PROMPT_TEMPLATE",
    "PLAN_JSON_SCHEMA",
]

logger = logging.getLogger("liteagent.multiagent")

# 委派上下文的**唯一**来源（§10.4 的 [v2 冻结]）：
# `args['extra_context']` 只是给 worker 看的文本，绝不参与环检测/深度判定。
_CURRENT_CTX: contextvars.ContextVar[DelegationContext | None] = contextvars.ContextVar(
    "liteagent_delegation_ctx", default=None
)

# 工具名里不允许出现 `-`（注册表的合法字符集里有它，但冻结用例要求
# 'a-b' 与 'a_b' 冲突 —— 说明 sanitize 的目标是"下划线化的标识符"）。
_TOOL_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.]")

# 委派工具的 tags：`register_all` / 工具面板可以据此把委派工具单独列出来。
DELEGATE_TOOL_TAGS: tuple[str, ...] = ("delegation",)

# 委派工具参数的冻结 schema（§10.4）：
# `required=["task"]` —— 光说"帮我看看"是不够的；`extra_context` 是可选的补充信息。
DELEGATE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The subtask to delegate, written as a self-contained instruction.",
        },
        "extra_context": {
            "type": "string",
            "description": (
                "Optional extra context for the delegate (background, constraints, "
                "already-known facts). It is appended to the task text and is NOT "
                "interpreted by the framework."
            ),
        },
    },
    "required": ["task"],
    "additionalProperties": False,
}


def _status_value(status: Any) -> str:
    """`AgentStatus` / 字符串 -> `"FINISHED"` 这样的字符串（事件 data 必须 JSON 友好）。"""
    return str(getattr(status, "value", status))


def _is_finished(status: Any) -> bool:
    """宽容的"是否成功"判定（成员、字符串值、字符串名都认）。"""
    return _status_value(status).upper() == AgentStatus.FINISHED.value


def _error_text(error: Any) -> str:
    """异常 -> 单行文本（事件与回灌字符串都不该出现换行）。"""
    if error is None:
        return ""
    return f"{type(error).__name__}: {error}".replace("\n", " ")


def _sanitize_tool_suffix(name: str) -> str:
    """把 worker 名字变成工具名后缀（非法字符 -> `_`；`-` 也算非法，见文件头的说明）。"""
    return _TOOL_NAME_UNSAFE.sub("_", name)


# --------------------------------------------------------------------------------------
# Plan / SubTask
# --------------------------------------------------------------------------------------


@dataclass
class SubTask:
    """计划中的一步：做什么、谁来做、依赖谁。"""

    id: str
    description: str
    assignee: str = ""                 # worker 名字；空表示由 manager 自行决定
    depends_on: tuple[str, ...] = ()
    status: str = "pending"            # "pending" | "running" | "done" | "failed"
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        """全字段输出（§2.2）：`depends_on` 转 list 以保持 JSON 可序列化。"""
        return {
            "id": self.id,
            "description": self.description,
            "assignee": self.assignee,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubTask":
        """宽容反序列化：缺字段用默认值，绝不为"模型少写了一个键"抛异常。

        `depends_on` 允许写成字符串（模型常把单依赖写成 `"depends_on": "t1"`），
        这里统一包成 tuple —— 比后续在拓扑排序里到处判类型便宜得多。
        """
        raw_id = data.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            # from_json 会补 `task_{index}`；直接调 from_dict 的调用方自己负责 id。
            logger.debug("SubTask.from_dict: missing or non-string id (%r)", raw_id)
            raw_id = str(raw_id) if raw_id is not None else ""
        depends = data.get("depends_on") or ()
        if isinstance(depends, (str, bytes)):
            depends = (depends,)
        return cls(
            id=raw_id,
            description=str(data.get("description") or ""),
            assignee=str(data.get("assignee") or ""),
            depends_on=tuple(str(dep) for dep in depends),
            status=str(data.get("status") or "pending"),
            result=str(data.get("result") or ""),
        )


@dataclass
class Plan:
    """一份显式计划：目标 + 若干带依赖的子任务。"""

    goal: str = ""
    subtasks: list[SubTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "subtasks": [st.to_dict() for st in self.subtasks]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Plan":
        raw_items = data.get("subtasks") or []
        subtasks: list[SubTask] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                logger.warning("Plan.from_dict: skipping non-object subtask %r", item)
                continue
            subtasks.append(SubTask.from_dict(item))
        return cls(goal=str(data.get("goal") or ""), subtasks=subtasks)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        """容错解析模型输出（冻结步骤见 §10.4）：

        1. 剥围栏与首尾空白；2. 取首 `{` 到末 `}` 的子串（容忍解释性文字）；
        3. `json.loads` 失败 -> `ReActParseError(raw=text, reason="plan is not valid JSON")`；
        4. 接受 `{"goal","subtasks":[...]}` 或**直接一个 list**；
        5. 缺 `id` -> `"task_{index}"`；缺 `description` -> 跳过该条；
        6. 不做 schema 校验以外的事（`assignee` 合法性由调用方处理）。

        SPEC-AMBIGUITY: 步骤 2（取 `{...}` 子串）与步骤 4（接受裸 list）在
        `[{"id": "t1", ...}]` 这种输入上互相矛盾 —— 取子串会得到 `{...}` 单个对象，
        丢掉其余子任务。裁决：**先按文本首字符分派** —— 以 `[` 开头就整体解析成
        list，否则才取 `{...}` 子串；两者都失败才抛 `ReActParseError`。
        这样既满足"容忍解释文字"，也满足"接受裸 list"。
        """
        cleaned = _strip_code_fences(text)
        data = _loads_plan_json(cleaned, text)

        if isinstance(data, list):
            raw_items: Sequence[Any] = data
            goal = ""
        elif isinstance(data, Mapping):
            raw_items = data.get("subtasks") or []
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
                logger.warning(
                    "Plan.from_json: 'subtasks' is not a list (%r); treating as empty",
                    type(raw_items).__name__,
                )
                raw_items = []
            goal = str(data.get("goal") or "")
        else:
            # 合法 JSON，但不是我们认识的形状（例如裸字符串/数字）
            raise ReActParseError(
                raw=text, reason="plan is not valid JSON",
                cause=TypeError(f"expected object or list, got {type(data).__name__}"),
            )

        subtasks: list[SubTask] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                logger.warning("Plan.from_json: skipping non-object subtask #%d", index)
                continue
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                # 冻结步骤 5：缺 description 的条目直接跳过（一条没有内容的子任务
                # 派给谁都是浪费一次 LLM 调用）。
                logger.warning("Plan.from_json: skipping subtask #%d without description", index)
                continue
            raw_id = item.get("id")
            sid = str(raw_id) if isinstance(raw_id, (str, int)) and str(raw_id) else f"task_{index}"
            depends = item.get("depends_on") or ()
            if isinstance(depends, (str, bytes)):
                depends = (depends,)
            deps = tuple(str(dep) for dep in depends) if isinstance(depends, Sequence) else ()
            subtasks.append(SubTask(
                id=sid,
                description=description,
                assignee=str(item.get("assignee") or ""),
                depends_on=deps,
            ))
        return cls(goal=goal, subtasks=subtasks)

    def render(self) -> str:
        """`- [id] description (assignee) <- depends_on` 多行文本。

        `assignee` / `depends_on` 为空时对应片段整体省略 —— 冻结格式给出的是
        "全字段都在"时的样子，而省略空片段比渲染出 `() <- ` 这样的噪声更可读。
        """
        lines: list[str] = []
        for st in self.subtasks:
            line = f"- [{st.id}] {st.description}"
            if st.assignee:
                line += f" ({st.assignee})"
            if st.depends_on:
                line += " <- " + ", ".join(st.depends_on)
            lines.append(line)
        return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    """剥掉 ```json / ``` 围栏与首尾空白（冻结步骤 1）。"""
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    newline = cleaned.find("\n")
    cleaned = cleaned.lstrip("`") if newline == -1 else cleaned[newline + 1:]
    cleaned = cleaned.rstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _loads_plan_json(cleaned: str, raw_text: str) -> Any:
    """按"裸 list 优先、否则取 `{...}` 子串"的顺序尝试解析（见 `Plan.from_json` 的裁决）。

    全部候选都失败时抛 `ReActParseError`，并把 `JSONDecodeError.pos` 带进 `offset` ——
    回灌给模型时能指出"大概从哪一列开始坏了"。
    """
    candidates: list[str] = []
    stripped = cleaned.strip()
    if stripped.startswith("["):
        candidates.append(stripped)                       # 裸 list（冻结步骤 4）
    first, last = stripped.find("{"), stripped.rfind("}")
    if first != -1 and last > first:
        candidates.append(stripped[first:last + 1])       # 容忍前后解释文字（冻结步骤 2）
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    last_exc: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
    raise ReActParseError(
        raw=raw_text,
        offset=getattr(last_exc, "pos", 0) or 0,
        reason="plan is not valid JSON",
        cause=last_exc,
    )


# 冻结字面量（§10.4）：`{{` / `}}` 由 `str.format_map` 还原成 `{` / `}`，
# 所以渲染后给出的是一个**可复制的 JSON 骨架**。
DECOMPOSE_PROMPT_TEMPLATE: str = """\
Break the following task into at most {max_subtasks} concrete subtasks.

Available workers:
{workers}

Rules:
- Each subtask must be independently executable by exactly one worker.
- Use the worker names exactly as listed.
- depends_on lists ids of subtasks that must finish first; use [] when independent.
- Reply with ONLY a JSON object, no prose, no markdown fences:
{{"goal": "...", "subtasks": [{{"id": "t1", "description": "...", "assignee": "<worker name>", "depends_on": []}}]}}

Task:
{task}
"""

# `Plan` 的 JSON Schema（自检/文档用；`from_json` 走的是更宽容的手写解析）。
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "assignee": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["subtasks"],
}


# --------------------------------------------------------------------------------------
# HierarchicalAgent
# --------------------------------------------------------------------------------------


class HierarchicalAgent(MultiAgent):
    """层级编排：manager（普通 Agent）+ workers（被包装成 `delegate_to_<name>` 工具）。"""

    manager: "Agent"
    workers: list[AgentLike]

    def __init__(self, manager: "Agent", workers: Sequence[AgentLike], *,
                 name: str = "hierarchy", config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: Blackboard | None = None,
                 worker_descriptions: Mapping[str, str] | None = None) -> None:
        """冻结创建的实例状态（全部是 **threading** 原语，理由见文件头第 2 点）。

        `worker_descriptions` 让调用方给委派工具写更"有说服力"的 description ——
        模型选工具几乎只看 description，`delegate_to_researcher` 这种名字本身信息量很低。
        """
        self.manager = manager
        self.workers = list(workers)
        self._worker_descriptions: dict[str, str] = dict(worker_descriptions or {})
        super().__init__(
            self.workers,
            name=name, config=config, callbacks=callbacks, blackboard=blackboard,
        )
        concurrency = int(self.config.subagent_concurrency)
        if concurrency <= 0:
            # `threading.Semaphore(0)` 会让第一次委派**永久阻塞**（没有超时参数时），
            # 这是"配置错误"而不是"运行期故障"，必须在构造点大声失败。
            raise ConfigError(
                f"TeamConfig.subagent_concurrency must be >= 1; got {self.config.subagent_concurrency!r}"
            )
        self._subagent_concurrency = concurrency

        # ---- 冻结的实例状态（§10.4，[v2] 全部 threading） ----
        self._delegation_seq: int = 0
        self._delegation_lock = threading.Lock()
        self._subagent_sem = threading.Semaphore(concurrency)
        self._serial_lock = threading.Lock()     # parallel_subagents=False 时用
        # `arun_plan` 里的 asyncio.Semaphore 必须"本 loop 懒创建"（§5.4）。
        # 实例级持有一个 pool（而不是每次调用 new 一个）：同一 loop + 同一 key
        # 拿到的是同一个信号量，语义不变，同时不会让 `_ALL_POOLS` 无界增长。
        self._pools = LoopBoundPool()
        # 记住"上一次 merge 出来的注册表对象"，让同一实例的第二次 arun 幂等
        # （否则 `manager.tools.merge(...)` 会在第二次调用时因重名抛 ToolDefinitionError）。
        self._merged_tools: ToolRegistry | None = None

    # ------------------------------------------------------------------
    # workers / delegate 工具
    # ------------------------------------------------------------------

    def worker_names(self) -> list[str]:
        """worker 名字列表（顺序与 `self.workers` 一致）。"""
        return [str(getattr(worker, "name", "") or "") for worker in self.workers]

    def delegate_tools(self) -> list[Tool]:
        """为每个 worker 生成一个 `delegate_to_<name>` 工具（同步、`pass_style="mapping"`）。

        名字里的非法字符（含 `-`）替换为 `_`；两个 worker 归一化后同名 -> **抛
        `ToolDefinitionError`**（例如 `'a-b'` 与 `'a_b'`：静默让其中一个不可达是更坏的结局）。
        `timeout_s=NO_TIMEOUT`：委派可能跑很久（子 Agent 自己也在做多轮 ReAct），
        30s 的默认工具超时对它是错的。
        """
        tools: list[Tool] = []
        owner_of: dict[str, str] = {}
        for worker in self.workers:
            worker_name = str(getattr(worker, "name", "") or "")
            if not worker_name:
                raise ConfigError("every worker must have a non-empty .name to be delegated to")
            tool_name = f"delegate_to_{_sanitize_tool_suffix(worker_name)}"
            previous = owner_of.get(tool_name)
            if previous is not None:
                raise ToolDefinitionError(
                    tool_name,
                    message=(
                        f"delegate tool name {tool_name!r} would be generated by two workers: "
                        f"{previous!r} and {worker_name!r}; rename one of them"
                    ),
                    context={"workers": [previous, worker_name]},
                )
            owner_of[tool_name] = worker_name

            description = (
                self._worker_descriptions.get(worker_name)
                or self._worker_descriptions.get(tool_name)
                or getattr(worker, "description", "")
                or f"Delegate a subtask to the '{worker_name}' agent."
            )
            tools.append(make_function_tool(
                name=tool_name,
                description=str(description),
                parameters=dict(DELEGATE_PARAMETERS),
                func=self._make_delegate_func(worker, worker_name),
                is_async=False,
                tags=DELEGATE_TOOL_TAGS,
                dangerous=False,
                requires_approval=False,
                idempotent=False,          # 委派有副作用（会真的跑一遍子 Agent）
                timeout_s=NO_TIMEOUT,      # [v2] 委派不受 30s 默认工具超时约束
                max_retries=None,
            ))
        return tools

    def _make_delegate_func(self, worker: AgentLike,
                            worker_name: str) -> Callable[[dict[str, Any]], Any]:
        """构造委托闭包（冻结步骤 1~8，全部在**工作线程**里执行）。

        顺序很重要：**预算 -> 环/深度 -> 限流 -> 执行**。把限流放在最前面会让
        "预算已耗尽"变成"排队等信号量"，白等 30 秒才拿到一句 refused。
        """

        def _delegate(args: dict[str, Any]) -> str:
            payload = dict(args or {})
            task = str(payload.get("task") or "")
            # `extra_context` 只是文本，**绝不**参与环检测/深度判定（[v2 冻结]）。
            extra_context = str(payload.get("extra_context") or "")

            ctx = _CURRENT_CTX.get()
            if ctx is None:
                # 工具被脱离 Hierarchical 使用（例如被 register_all 后单独调用）：
                # 造一个只含自己的上下文，让它仍然受深度/预算约束而不是无限制委派。
                logger.warning(
                    "delegate_to_%s called outside HierarchicalAgent.arun; "
                    "falling back to a self-only DelegationContext", worker_name,
                )
                ctx = DelegationContext(
                    stack=[self.name], depth=0, root_run_id="",
                    budget=self.config.max_rounds,
                )

            # 步骤 2.a：预算（max_rounds 的真正落点）
            if ctx.exhausted():
                text = "[delegation refused: budget exhausted]"
                self._emit_refused(worker_name, ctx, "budget", text)
                return text

            # 步骤 2.b：环检测 / 深度。**返回字符串，不抛异常** —— 抛异常会让
            # 一次"模型选错了同事"升级成整轮 run 失败。
            cycled = ctx.would_cycle(worker_name)
            if cycled or ctx.depth >= self.config.max_depth:
                reason = "cycle" if cycled else "depth"
                text = (
                    f"[delegation refused: cycle detected: "
                    f"stack=[{','.join(ctx.stack)}] -> {worker_name}]"
                )
                if reason == "depth":
                    # 复用同一句冻结文案（§10.4 只冻结了一句），但把真实原因写进
                    # 事件与日志，避免"看到 cycle 字样其实是深度不够"的误导。
                    logger.warning(
                        "hierarchical: delegation to %r refused by max_depth "
                        "(depth=%d, max_depth=%d, stack=%s)",
                        worker_name, ctx.depth, self.config.max_depth, ctx.stack,
                    )
                self._emit_refused(worker_name, ctx, reason, text)
                return text

            # 步骤 2.c/2.d：限流（真串行 或 信号量）
            if not self.config.parallel_subagents:
                with self._serial_lock:
                    return self._run_delegate(worker, worker_name, ctx, task, extra_context)

            acquired = self._subagent_sem.acquire(timeout=DEFAULT_TOOL_TIMEOUT_S)
            if not acquired:
                text = (
                    f"[delegation refused: all {self._subagent_concurrency} "
                    f"subagent slots are busy]"
                )
                self._emit_refused(worker_name, ctx, "busy", text)
                return text
            try:
                # 冻结的红线 14：信号量先拿、seq 锁后拿（`_run_delegate` 里的
                # `_archive` 才碰 seq 锁）。反序即死锁。
                return self._run_delegate(worker, worker_name, ctx, task, extra_context)
            finally:
                self._subagent_sem.release()

        return _delegate

    def _run_delegate(self, worker: AgentLike, worker_name: str,
                      ctx: DelegationContext, task: str, extra_context: str) -> str:
        """步骤 3~8：建状态 -> 跑 worker -> 压缩 -> 归档 -> 发事件 -> 返回字符串。

        **任何失败都返回字符串**（worker 抛异常也一样）：manager 需要的是"观察"，
        不是"整轮崩掉"。唯一的例外是 `asyncio.CancelledError`（BaseException，
        不在这里的捕获范围内），取消信号必须继续传播。
        """
        started = utc_now()
        child_context = ctx.child(worker_name)
        # `extra_context` 拼进 worker 看到的输入文本（而不是只在 state.input 里），
        # 否则对"只读 input 参数"的 Agent 实现来说这段信息就丢了。
        prompt = task if not extra_context else f"{task}\n\n{extra_context}"
        child_state = AgentState.create(prompt, agent_name=worker_name)
        child_state.scratchpad["delegation"] = child_context

        self._apply_shared_memory(worker)

        try:
            result = run_sync(lambda: worker.arun(prompt, state=child_state))
        except Exception as exc:  # noqa: BLE001 - 红线 6 的精神：委派失败 = 一次失败的观察
            logger.exception(
                "hierarchical: worker %r raised %s; returning it as an observation",
                worker_name, type(exc).__name__,
            )
            text = self._format_exception(worker_name, exc)
            self._archive(worker_name, child_state, text, output="")
            self._emit_return(worker_name, ctx, status="FAILED", steps=0,
                              text=text, started=started, failed=True)
            self._emit_failure_marker(worker_name, ctx, _error_text(exc))
            return text

        failed = not _is_finished(result.status)
        status_value = _status_value(result.status)
        steps = int(getattr(result, "steps", 0) or 0)
        text = self._format_result(worker_name, result, failed=failed)
        self._archive(worker_name, child_state, text, output=result.output or "")
        self._emit_return(worker_name, ctx, status=status_value, steps=steps,
                          text=text, started=started, failed=failed)
        if failed:
            # §10.4 步骤 8 的 [v2] 可观测性要求：失败**再**发一条最小的
            # `AGENT_RETURN {failed: True}`，让"只看失败"的消费者不必解析 status。
            self._emit_failure_marker(worker_name, ctx, _error_text(result.error))
        return text

    # ---- 结果压缩 ----

    def _format_result(self, worker_name: str, result: AgentResult, *, failed: bool) -> str:
        """把子 Agent 结果压成回灌字符串（冻结格式见 §10.4 的"结果如何回传"一节）。

        - 成功：`compress_subagent_output`（header + head 70%/tail 30%）；
        - 失败：header 追加 `| error=类型: 消息`，正文仍按同一规则截断 ——
          "为什么失败"经常就在正文末尾，不能因为是失败路径就把正文丢掉。
        - `config.compress_subagent_output=False` 时**不截断正文**（header 保留：
          它是状态标注，不是压缩，失败路径同样需要它）。
        """
        output = result.output or ""
        steps = int(getattr(result, "steps", 0) or 0)
        status_value = _status_value(result.status)
        body = self._compress_body(output)
        if not failed:
            max_chars = self.config.subagent_output_max_chars if self.config.compress_subagent_output else 0
            return compress_subagent_output(
                worker_name, output, status=status_value, steps=steps, max_chars=max_chars,
            )
        header = f"[worker {worker_name} | status={status_value} | steps={steps}"
        error_text = _error_text(result.error)
        if error_text:
            header += f" | error={error_text}"
        header += "]"
        return f"{header}\n{body}" if body else header

    def _format_exception(self, worker_name: str, exc: BaseException) -> str:
        """worker 抛异常时的回灌字符串（与失败结果同构，manager 无需区分两种失败）。"""
        return (
            f"[worker {worker_name} | status=FAILED | steps=0 | "
            f"error={_error_text(exc)}]"
        )

    def _compress_body(self, output: str) -> str:
        """正文截断（head 70% + tail 30%），受 `compress_subagent_output` 开关控制。"""
        if not self.config.compress_subagent_output:
            return output
        max_chars = self.config.subagent_output_max_chars
        if max_chars <= 0:
            return output
        return truncate_head_tail(output, max_chars, head_ratio=SUBAGENT_HEAD_RATIO)

    # ---- 归档 / 事件 ----

    def _archive(self, worker_name: str, child_state: AgentState, text: str,
                 *, output: str) -> None:
        """冻结键与计数器（§10.4 步骤 6）：`subagent:{name}:{seq}`，seq 从 1 递增。

        计数器必须受 `_delegation_lock` 保护：闭包会被 manager 的线程池**并发**触发。
        原始输出（未压缩）同时进 `child_state.scratchpad["subagent_results"]`，
        黑板里放压缩版 —— 于是"不丢信息"与"不炸上下文"两件事同时成立。
        """
        with self._delegation_lock:
            self._delegation_seq += 1
            seq = self._delegation_seq
        self.blackboard.write(
            f"subagent:{worker_name}:{seq}", text,
            author=worker_name, tags=("subagent",),
        )
        results = child_state.scratchpad.setdefault("subagent_results", {})
        if isinstance(results, dict):
            results[f"{worker_name}:{seq}"] = output
        else:  # pragma: no cover - 只有外部往 scratchpad 塞了怪东西才会走到
            logger.warning(
                "scratchpad['subagent_results'] is %r, not a dict; skipping archive for %s:%d",
                type(results).__name__, worker_name, seq,
            )

    def _emit_refused(self, worker_name: str, ctx: DelegationContext,
                      reason: str, text: str) -> None:
        """拒绝委派：发 `AGENT_DELEGATE {refused: True, refused_reason: ...}` 并记日志。

        "拒绝"是**正常控制流**（预算/环/并发满都是预期内的），所以用 WARNING 而不是
        exception；但绝不能静默 —— 否则 trace 里只会看到一个莫名其妙的字符串返回。
        """
        logger.warning("hierarchical: delegation to %r refused (%s): %s",
                       worker_name, reason, text)
        self.callbacks.emit_type(
            EventType.AGENT_DELEGATE,
            run_id=ctx.root_run_id, agent_name=self.name, step=ctx.depth,
            **{"from": self.name, "to": worker_name, "depth": ctx.depth + 1,
               "refused": True, "refused_reason": reason},
        )

    def _emit_return(self, worker_name: str, ctx: DelegationContext, *,
                     status: str, steps: int, text: str, started: float,
                     failed: bool) -> None:
        """`AGENT_RETURN`：每次委派（无论成败）都以一条返回事件收尾，trace 才能配对。"""
        self.callbacks.emit_type(
            EventType.AGENT_RETURN,
            run_id=ctx.root_run_id, agent_name=self.name, step=ctx.depth,
            **{"from": self.name, "to": worker_name, "status": status,
               "steps": steps, "output_len": len(text),
               "duration_ms": max(0.0, (utc_now() - started) * 1000.0),
               "failed": failed},
        )

    def _emit_failure_marker(self, worker_name: str, ctx: DelegationContext,
                             error_text: str) -> None:
        """失败时的额外一条 `AGENT_RETURN {failed: True}`（§10.4 步骤 8 的 [v2] 要求）。"""
        self.callbacks.emit_type(
            EventType.AGENT_RETURN,
            run_id=ctx.root_run_id, agent_name=self.name, step=ctx.depth,
            **{"from": self.name, "to": worker_name, "failed": True,
               "error": error_text, "status": AgentStatus.FAILED.value, "steps": 0,
               "output_len": 0, "duration_ms": 0.0},
        )

    # ------------------------------------------------------------------
    # 分解
    # ------------------------------------------------------------------

    def _workers_text(self) -> str:
        """给分解 prompt 用的 worker 清单（名字 + description）。"""
        lines: list[str] = []
        for worker in self.workers:
            name = str(getattr(worker, "name", "") or "")
            description = (
                self._worker_descriptions.get(name)
                or getattr(worker, "description", "")
                or "no description"
            )
            lines.append(f"- {name}: {description}")
        return "\n".join(lines) or "- (no workers registered)"

    async def adecompose(self, task: str, *, max_subtasks: int = 8) -> Plan:
        """用一次 LLM 调用做显式任务分解。

        失败（LLM 错误 / JSON 解析失败）-> 抛 `ReActParseError` 或原 `LLMError`，
        **不静默**：一个"分解不出来"的计划比一层清晰的异常危险得多。
        这里直接调 `manager.llm`（而不是 `manager.arun`）：分解要的是**一次原始
        LLM 调用**，走 ReAct 循环反而会把 JSON 塞进 Thought/Action 的解析器里。
        """
        llm = getattr(self.manager, "llm", None)
        if llm is None or not hasattr(llm, "achat"):
            raise ConfigError(
                "HierarchicalAgent.adecompose requires manager.llm (an LLMClient); "
                f"got {type(llm).__name__}"
            )
        limit = int(max_subtasks)
        if limit <= 0:
            raise ConfigError(f"max_subtasks must be >= 1; got {max_subtasks!r}")

        prompt = render_template(DECOMPOSE_PROMPT_TEMPLATE, {
            "max_subtasks": limit,
            "workers": self._workers_text(),
            "task": task,
        })
        response = await llm.achat([Message.user(prompt)])
        return Plan.from_json(response.content)

    # ------------------------------------------------------------------
    # 执行（manager 主路径）
    # ------------------------------------------------------------------

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None,
                   plan: Plan | None = None, **kwargs: Any) -> AgentResult:
        """冻结语义见 §10.4。`plan` 非 None 时直接走 `arun_plan`（把分解接到执行上）。"""
        ctx = context if context is not None else DelegationContext(
            stack=[self.name], depth=0,
            root_run_id=getattr(state, "run_id", "") or "",
            budget=self.config.max_rounds,
        )

        # 入口也判一次闸门（§10.4 的"判定点"）：manager 可能拿到一个已经很深的
        # 上下文。to_agent 用 manager 的名字而**不是** self.name —— self.name 已经在
        # `ctx.stack` 里（新建上下文时就是这么建的），拿它判环必然自撞。
        # SPEC-AMBIGUITY: §10.4 只说"arun 自身在入口也调用一次 _check_depth"，
        # 没说 to_agent 传谁；裁决见上。
        self._check_depth(ctx, to_agent=str(getattr(self.manager, "name", "") or "manager"))

        # 步骤 2：把 delegate 工具并进 manager 的工具集
        self._install_delegate_tools()

        # 步骤 3：contextvar 保存 ctx（不用实例属性：manager 可能被并发复用）
        token = _CURRENT_CTX.set(ctx)
        try:
            # 步骤 4：显式计划
            if plan is not None:
                return await self.arun_plan(plan, context=ctx)

            # 步骤 5：状态（复用传入的 state，或用 input 新建）
            run_state = state if state is not None else AgentState.create(input, agent_name=self.name)
            run_state.scratchpad["delegation"] = ctx

            # 步骤 6：manager 跑（worker 的失败在工具层被吸收，不会走到这里）
            result = await self.manager.arun(input, state=run_state)
        finally:
            _CURRENT_CTX.reset(token)

        # 步骤 7：汇总委派记录（从黑板收集，键格式 `subagent:{name}:{seq}`）
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, dict):
            logger.warning("manager result.metadata is %r; replacing with a dict",
                           type(metadata).__name__)
            result.metadata = {}
            metadata = result.metadata
        metadata["delegations"] = [
            entry.to_dict() for entry in self.blackboard.list(tags=("subagent",))
        ]
        result.agent_name = self.name

        # 步骤 8：manager 整体失败才轮到 propagate_failure（单个 worker 的失败从不中止）
        if not _is_finished(result.status) and (self.config.propagate_failure or "").strip().lower() == "raise":
            raise DelegationError(
                from_agent=self.name,
                to_agent=str(getattr(self.manager, "name", "") or "manager"),
                message=(
                    f"manager failed with status {_status_value(result.status)}: "
                    f"{_error_text(result.error)}"
                ),
                context={"status": _status_value(result.status)},
                cause=result.error,
            )
        return result

    def _install_delegate_tools(self) -> ToolRegistry:
        """把 `delegate_tools()` 合并进 manager 的工具集（冻结步骤 2）。

        两条刻意的偏离（都记在这里，避免被当成"漏读规范"）：

        1. **幂等**：同一实例第二次 `arun` 时 `manager.tools` 正是我们上次 merge 出来的
           那个注册表，再 merge 一次必然因重名抛 `ToolDefinitionError`。用 `_merged_tools`
           的身份判定跳过；用户自己预先放进去的 `delegate_to_*` 仍然照抛不误。
        2. **manager 没有 tools 时**（测试里的鸭子类型 stub）直接挂上委派工具并记 WARNING；
           静默跳过会让"委派永远不生效"变成最难查的一类 bug。
        3. **执行器也要换到新注册表**：真 `Agent` 在 `__init__` 里就把 `self.tools` 交给
           `ToolExecutor(registry=...)`（§9.4 的装配顺序），而 `ToolRegistry.merge` 返回的是
           **新对象** —— 只改 `manager.tools` 的话，执行器手里还是那个旧注册表，
           委派工具对模型"看得到、调不到"（实测报 `ToolNotFoundError`）。
           这里额外把新注册表同步进 `manager.executor.registry`（执行器在每次调用时读它），
           让冻结语义"manager 一次委派后出最终答案"真的成立。
        """
        registry = ToolRegistry(self.delegate_tools())
        current = getattr(self.manager, "tools", None)
        if self._merged_tools is not None and current is self._merged_tools:
            return self._merged_tools

        merge = getattr(current, "merge", None)
        if callable(merge):
            merged = merge(registry)          # 重名（含用户自带的 delegate_to_*）-> ToolDefinitionError
        elif current is None:
            logger.warning(
                "manager %r has no ToolRegistry; installing delegate tools only",
                getattr(self.manager, "name", self.manager),
            )
            merged = registry
        else:
            raise ConfigError(
                f"manager.tools must be a ToolRegistry (or None); got {type(current).__name__}"
            )
        self.manager.tools = merged
        self._sync_executor_registry(merged)
        self._merged_tools = merged
        return merged

    def _sync_executor_registry(self, merged: ToolRegistry) -> None:
        """把合并后的注册表同步给 manager 的执行器（见 `_install_delegate_tools` 第 3 点）。

        只动 `registry` 这一个属性：执行器在 `execute` 里按名字现查（§7.4.1），
        换掉引用即可生效。鸭子类型判定，不认识的对象只是记一条 INFO ——
        自定义执行器可能有自己的工具来源，硬改只会把它弄坏。
        """
        executor = getattr(self.manager, "executor", None)
        if executor is None:
            return
        current = getattr(executor, "registry", None)
        if current is merged:
            return
        if isinstance(current, ToolRegistry):
            executor.registry = merged
            logger.debug(
                "hierarchical: re-pointed %r's executor registry to the merged registry",
                getattr(self.manager, "name", self.manager),
            )
            return
        logger.info(
            "hierarchical: manager executor %r exposes no ToolRegistry (registry=%r); "
            "delegate tools may not be reachable by the model",
            type(executor).__name__, type(current).__name__,
        )

    # ------------------------------------------------------------------
    # 执行（显式 Plan 路径）
    # ------------------------------------------------------------------

    def _layers(self, plan: Plan) -> list[list[SubTask]]:
        """按 `depends_on` 做拓扑分层（Kahn 的分层版）：同层之间的任务互不依赖。

        - `depends_on` 指向不存在的 id -> `DelegationError`（早失败，别让它变成
          "某个任务永远等不到");
        - 存在环（含自依赖）-> `CycleDetectedError(stack=[剩余 ids])`；
        - 重复 id -> `DelegationError`（重复 id 会让 depends_on 的语义有歧义）。
        """
        subtasks = list(plan.subtasks)
        if not subtasks:
            return []
        by_id: dict[str, SubTask] = {}
        for st in subtasks:
            if st.id in by_id:
                raise DelegationError(
                    from_agent=self.name, to_agent=st.id,
                    message=f"duplicate subtask id {st.id!r} in plan",
                    context={"goal": plan.goal},
                )
            by_id[st.id] = st
        for st in subtasks:
            for dep in st.depends_on:
                if dep not in by_id:
                    raise DelegationError(
                        from_agent=self.name, to_agent=st.id,
                        message=(
                            f"subtask {st.id!r} depends on unknown id {dep!r}; "
                            f"known ids: {sorted(by_id)}"
                        ),
                        context={"depends_on": dep},
                    )

        remaining = list(subtasks)
        done: set[str] = set()
        layers: list[list[SubTask]] = []
        while remaining:
            layer = [st for st in remaining if all(dep in done for dep in st.depends_on)]
            if not layer:
                # 每个任务都还在等别人 -> 剩下的必然成环（依赖图有限）
                cycle = [st.id for st in remaining]
                raise CycleDetectedError(
                    cycle, message=f"plan has a dependency cycle among {cycle}",
                    context={"goal": plan.goal},
                )
            layers.append(layer)
            for st in layer:
                done.add(st.id)
            layer_ids = {st.id for st in layer}
            remaining = [st for st in remaining if st.id not in layer_ids]
        return layers

    async def arun_plan(self, plan: Plan, *, context: DelegationContext) -> AgentResult:
        """按依赖分层执行一个显式 Plan（[v2 新增]，见 §10.4）。

        - 层内并发（`parallel_subagents=True`）用**本 loop** 的 asyncio.Semaphore：
          `arun_plan` 跑在父 loop 里，没有跨线程问题，所以这里**可以**用 asyncio 原语；
        - 层内严格按 `depends_on` 由分层保证，**同层内的完成顺序不做保证**
          （测试不要断言同层完成顺序）；
        - 单个 subtask 失败（抛异常或返回 FAILED）**不中断**整体：压成字符串写进
          `st.result`、置 `st.status="failed"`，后续层照跑。
        """
        started = utc_now()
        layers = self._layers(plan)
        if not layers:
            # [v3 可观测性] 空计划（模型返回 subtasks: []，或所有条目都因缺 description
            # 被 `Plan.from_json` 跳过）会一路走到 FINISHED + output='results: (none)'：
            # 没有 RUN_FAILED、没有 error、没有 BUDGET/NUDGE 事件 —— 对"失败必须编码在
            # AgentResult 里"的契约来说这是最坏的一类静默（一个字段名写错就能让
            # "主 Agent 做了任务分解"退化成"什么都没干但报告成功"）。
            # §10.4 冻结了"计划执行跑完了就是 FINISHED，失败在 subtask 层"，而空计划
            # 的 subtask 层没有任何失败 —— 改返回形状需要规范修订，所以这里不动
            # AgentResult，但**至少必须留痕**（§13 红线 12：降级可以是设计，静默不行）。
            logger.warning(
                "arun_plan: plan for goal=%r has no executable subtasks "
                "(subtasks=%d); returning FINISHED with empty results",
                plan.goal,
                len(plan.subtasks),
            )
        run_id = context.root_run_id
        worker_by_name = {
            str(getattr(worker, "name", "") or ""): worker for worker in self.workers
        }
        index_of = {st.id: idx for idx, st in enumerate(plan.subtasks)}

        summaries: list[dict[str, Any]] = []
        total_steps = 0

        for layer in layers:
            if self.config.parallel_subagents:
                sem = self._pools.semaphore("plan", self._subagent_concurrency)

                async def _one(st: SubTask, _sem: "asyncio.Semaphore" = sem) -> dict[str, Any]:
                    # 信号量包住整段 subtask 执行：并发峰值因此 <= subagent_concurrency。
                    async with _sem:
                        return await self._run_subtask(
                            st, context=context, worker_by_name=worker_by_name,
                            run_id=run_id, index=index_of.get(st.id, 0),
                        )

                # return_exceptions=True：即使某个 subtask 逃逸出异常，也不让 gather
                # 直接失败（§0.1 明确用 gather 而不是 TaskGroup）。
                settled = await asyncio.gather(*(_one(st) for st in layer), return_exceptions=True)
                for st, item in zip(layer, settled):
                    if isinstance(item, BaseException):
                        if isinstance(item, asyncio.CancelledError):
                            raise item       # 取消必须传播，不能当成"子任务失败"
                        # 这里在 except 块之外，用 logger.error 而不是 logger.exception
                        # （后者在没有活动异常时只会打出 "NoneType: None"）。
                        logger.error(
                            "plan subtask %r crashed before returning: %s: %s",
                            st.id, type(item).__name__, item,
                        )
                        summaries.append(self._failure_summary(st, item))
                        st.status = "failed"
                        st.result = f"[subtask {st.id} failed: {_error_text(item)}]"
                        continue
                    summaries.append(item)
            else:
                for st in layer:
                    summaries.append(await self._run_subtask(
                        st, context=context, worker_by_name=worker_by_name,
                        run_id=run_id, index=index_of.get(st.id, 0),
                    ))

        for summary in summaries:
            total_steps += int(summary.get("steps", 0) or 0)

        state = AgentState.create(f"plan: {plan.goal}", agent_name=self.name)
        state.scratchpad["delegation"] = context
        state.mark_finished(AgentStatus.FINISHED)
        return AgentResult(
            output=self._render_plan_summary(plan),
            status=AgentStatus.FINISHED,      # 计划执行"跑完了"就是 FINISHED，失败在 subtask 层
            steps=total_steps,
            state=state,
            agent_name=self.name,
            duration_ms=max(0.0, (utc_now() - started) * 1000.0),
            # [v3 可观测性] 除了 §10.4 冻结的 "plan" / "steps"，额外写两个**顶层**信号，
            # 让"这次到底分了几个子任务"不用去钻 metadata["plan"]["subtasks"] 的长度：
            #   - subtasks: 可执行子任务的条数（0 = 空计划，见上面的 WARNING）；
            #   - empty_plan: 布尔快捷方式（空计划时 output 是 'results: (none)'，
            #     调用方靠它做告警/降级，不必解析字符串）。
            # 只**新增** key，不改 "plan" / "steps" 的名字、语义与写入者归属（§2.4）。
            metadata={
                "plan": plan.to_dict(),
                "steps": summaries,
                "subtasks": len(plan.subtasks),
                "empty_plan": not plan.subtasks,
            },
        )

    async def _run_subtask(self, st: SubTask, *, context: DelegationContext,
                           worker_by_name: Mapping[str, AgentLike], run_id: str,
                           index: int) -> dict[str, Any]:
        """跑一个 subtask 并把它落进 `st` / 黑板 / 返回的摘要（**不抛**业务异常）。"""
        started = utc_now()
        st.status = "running"
        worker = worker_by_name.get(st.assignee) if st.assignee else None

        if worker is None:
            # assignee 为空或不在 workers 里 -> 回退给 manager（§10.4 冻结）
            manager_name = str(getattr(self.manager, "name", "") or "manager")
            logger.warning(
                "plan subtask %r assignee %r is not a known worker (known: %s); "
                "falling back to the manager %r",
                st.id, st.assignee, sorted(worker_by_name), manager_name,
            )
            author = manager_name
            result = await self._call_subagent(st, target=self.manager, author=author,
                                               task_text=f"[subtask {st.id}] {st.description}",
                                               context=context, index=index, run_id=run_id)
        else:
            author = str(getattr(worker, "name", "") or "")
            result = await self._call_subagent(st, target=worker, author=author,
                                               task_text=st.description,
                                               context=context, index=index, run_id=run_id)

        failed = not _is_finished(result.status)
        status_value = _status_value(result.status)
        steps = int(getattr(result, "steps", 0) or 0)
        st.status = "failed" if failed else "done"
        text = self._format_result(author, result, failed=failed)
        st.result = text
        # 黑板键冻结为 `subtask:{id}`（与 delegate 路径的 `subagent:{name}:{seq}` 区分）
        self.blackboard.write(f"subtask:{st.id}", text, author=author, tags=("subtask",))
        self._emit_return(author, context, status=status_value, steps=steps,
                          text=text, started=started, failed=failed)
        return {
            "id": st.id,
            "assignee": author,
            "status": st.status,
            "steps": steps,
            "output": text,
            "duration_ms": max(0.0, (utc_now() - started) * 1000.0),
        }

    async def _call_subagent(self, st: SubTask, *, target: Any, author: str,
                             task_text: str, context: DelegationContext,
                             index: int, run_id: str) -> AgentResult:
        """真正调一次子 Agent，并把"抛异常"归一化成 `AgentResult(FAILED)`。

        归一化的价值：`_run_subtask` 只有一条尘埃落定后的处理路径，
        "失败不中断整体"就变成结构性保证，而不是散落各处的 try/except。
        """
        child_state = AgentState.create(task_text, agent_name=author)
        child_state.scratchpad["delegation"] = context.child(author)
        self._apply_shared_memory(target)
        self.callbacks.emit_type(
            EventType.AGENT_DELEGATE,
            run_id=run_id or context.root_run_id, agent_name=self.name, step=index,
            **{"from": self.name, "to": author, "depth": context.depth + 1,
               "refused": False},
        )
        try:
            result = await target.arun(task_text, state=child_state)
        except Exception as exc:  # noqa: BLE001 - 子任务失败不中断整体（见 docstring）
            logger.exception("plan subtask %r via %r raised %s", st.id, author, type(exc).__name__)
            return AgentResult(
                output="", status=AgentStatus.FAILED, state=child_state, agent_name=author,
                # `AgentResult.error` 的类型是 `LiteAgentError | None`：非框架异常包一层，
                # 保证下游（trace / raise_for_status）永远拿到可序列化的框架异常。
                error=exc if isinstance(exc, LiteAgentError) else DelegationError(
                    from_agent=self.name, to_agent=author,
                    message=f"{type(exc).__name__}: {exc}", cause=exc,
                ),
            )
        if not isinstance(result, AgentResult):  # pragma: no cover - 防御性归一化
            logger.warning("subagent %r returned %r, not an AgentResult", author, type(result))
            return AgentResult(
                output=str(result), status=AgentStatus.FAILED, state=child_state,
                agent_name=author,
                error=DelegationError(
                    from_agent=self.name, to_agent=author,
                    message=f"subagent returned {type(result).__name__}, expected AgentResult",
                ),
            )
        return result

    def _failure_summary(self, st: SubTask, exc: BaseException) -> dict[str, Any]:
        """`gather` 真的把异常送回调用方时的摘要（极端情况，保持一致的结构）。"""
        return {
            "id": st.id, "assignee": st.assignee, "status": "failed", "steps": 0,
            "output": f"[subtask {st.id} failed: {_error_text(exc)}]", "duration_ms": 0.0,
        }

    def _render_plan_summary(self, plan: Plan) -> str:
        """`arun_plan` 的 output = `Plan.render()` + 每条子任务的完成态一行。"""
        sections: list[str] = []
        rendered = plan.render()
        if rendered:
            sections.append(rendered)
        result_lines = [
            f"- [{st.id}] {st.status}: {truncate_head_tail(st.result or '', 200)}"
            for st in plan.subtasks
        ]
        sections.append("results:\n" + "\n".join(result_lines) if result_lines else "results: (none)")
        return "\n\n".join(sections)
