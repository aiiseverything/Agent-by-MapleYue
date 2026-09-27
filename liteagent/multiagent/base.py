from __future__ import annotations

# liteagent/multiagent/base.py —— 多 Agent 编排的公共基类（冻结规范 §10.1）
#
# 本文件是 L5 的"地基"，它只做四件事，但每件都有明确理由：
#
#   1. **`AgentLike` 协议**：编排对象不需要继承任何基类，只要有 `name` + `arun` 就能被
#      组合。于是 `Agent`、`MultiAgent`（递归嵌套）、以及用户自己写的假 Agent 都能进团队，
#      测试里可以塞一个 20 行的 stub 而不必拉起 LLM/工具栈。
#
#   2. **`DelegationContext`**：委派栈 / 深度 / 预算。`budget` 在 v1 里"递减后没人读"，
#      v2 把它接到 `exhausted()` 上，由 `_check_depth` 与 hierarchical 的 delegate 闭包
#      **真正判定** —— 否则 `max_rounds` 只是个装饰品。
#
#   3. **结果压缩 `compress_subagent_output`**：子 Agent 的原始输出（可能几万字符）不能
#      原样回灌给父 Agent，否则上下文瞬间爆掉。压缩**头尾都保留**（head 70% + tail 30%）：
#      头部通常是结论/结构，尾部是最近的进展，中间是最可牺牲的细节。原始输出不丢 ——
#      它同时写进黑板与 `child_state.scratchpad["subagent_results"]`。
#
#   4. **`_apply_shared_memory`（share_memory）**：让多个子 Agent 共用同一个
#      `MemoryManager`（`worker.memory is manager.memory`）。共享的记忆是可选的，
#      因为它把"每个 Agent 各自的学习"变成"全队一条记忆"，有利有弊。
#
# 依赖方向（§1.1 / E3）：本文件只向下依赖 agent/* 与 config/errors/types/blackboard；
# `build_team` 对 sequential/hierarchical 的 import **必须在函数体内** —— 那两个模块要
# 继承本文件的 `MultiAgent`，顶层 import 会直接成环。

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

from liteagent.agent.callbacks import CallbackLike, CallbackManager
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import (
    DEFAULT_TEAM_MAX_ROUNDS,
    SUBAGENT_HEAD_RATIO,
    DEFAULT_SUBAGENT_MAX_CHARS,
    TeamConfig,
    run_sync,
    truncate_head_tail,
)
from liteagent.errors import (
    ConfigError,
    CycleDetectedError,
    LiteAgentError,
    MaxDepthExceededError,
)
from liteagent.multiagent.blackboard import Blackboard
from liteagent.types import TokenUsage

if TYPE_CHECKING:  # 仅供类型检查（本文件带 `from __future__ import annotations`，注解不执行）
    from liteagent.agent.agent import Agent
    from liteagent.memory.manager import MemoryManager

__all__ = [
    "AgentLike",
    "DelegationContext",
    "MultiAgent",
    "TeamConfig",  # 唯一归属地是 config.py（§5.3），这里只 re-export
    "build_team",
    "compress_subagent_output",
]

logger = logging.getLogger("liteagent.multiagent")

# `build_team` 的合法 mode 取值（§10.1）。写成常量而不是散落的字面量，防止
# `build_team` 与 CLI 的 `multi --mode` 两边漂移。
TEAM_MODES: tuple[str, ...] = ("sequential", "hierarchical")


@runtime_checkable
class AgentLike(Protocol):
    """任何可被编排的东西（`Agent` 或另一个 `MultiAgent`）。

    `runtime_checkable` 是刻意加的：`SequentialAgent` 需要 `isinstance(step.agent,
    MultiAgent)` 来判定"要不要额外传 `context=`"（普通 `Agent.arun` 不认识这个参数），
    而测试里也常需要 `isinstance(x, AgentLike)` 来断言"这东西能被编排"。
    结构协议（只查属性是否存在）正好表达"我不在乎你是什么类，只在乎你会不会干活"。
    """

    @property
    def name(self) -> str: ...

    async def arun(self, input: str, **kwargs: Any) -> AgentResult: ...


@dataclass
class DelegationContext:
    """随 `AgentState.scratchpad["delegation"]` 传递的委派上下文（§2.4 的预留 key）。"""

    stack: list[str] = field(default_factory=list)  # 祖先 Agent 名字（含当前）
    depth: int = 0
    root_run_id: str = ""
    parent_run_id: str | None = None
    budget: int = DEFAULT_TEAM_MAX_ROUNDS  # 剩余可委派轮次

    def child(self, name: str) -> "DelegationContext":
        """返回新的上下文：stack 追加 name、depth+1、budget-1。**不修改自身**。

        只改这三个字段是规范冻结的；`root_run_id` / `parent_run_id` 原样带下去
        （`child()` 拿不到父 run 的 id，"父是谁"在 `from_state` 里由 state 决定）。
        """
        return DelegationContext(
            stack=[*self.stack, name],
            depth=self.depth + 1,
            root_run_id=self.root_run_id,
            parent_run_id=self.parent_run_id,
            budget=self.budget - 1,
        )

    def would_cycle(self, name: str) -> bool:
        """委派栈里已经有同名 Agent -> 再委派就成环（A -> B -> A）。"""
        return name in self.stack

    def exhausted(self) -> bool:
        """[v2] `budget <= 0`：委派预算耗尽。

        **必须被真正判定**（v1 递减后无人读）。判定点：`MultiAgent._check_depth`
        与 hierarchical 的 `delegate_to_*` 闭包（§10.4）。
        """
        return self.budget <= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": list(self.stack),
            "depth": self.depth,
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "budget": self.budget,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DelegationContext":
        """宽容反序列化（缺字段用默认值）。`stack` 里非字符串的条目丢掉但记日志。"""
        raw_stack = data.get("stack") or []
        stack = [item for item in raw_stack if isinstance(item, str)]
        if len(stack) != len(list(raw_stack)):
            logger.debug("DelegationContext.from_dict: dropped non-str stack items")
        depth = data.get("depth")
        budget = data.get("budget")
        parent = data.get("parent_run_id")
        return cls(
            stack=stack,
            depth=int(depth) if depth is not None else 0,
            root_run_id=str(data.get("root_run_id") or ""),
            parent_run_id=str(parent) if parent is not None else None,
            budget=int(budget) if budget is not None else DEFAULT_TEAM_MAX_ROUNDS,
        )

    @classmethod
    def from_state(cls, state: AgentState, *, name: str) -> "DelegationContext":
        """从 `state.scratchpad["delegation"]` 取；不存在则新建（`stack=[name], depth=0`）。

        三种形态都要吃得下：`DelegationContext`（正常路径）、`dict`（state 被
        `to_dict`/`from_dict` 往返过）、缺失（顶层入口）。取到已有对象时**返回原对象**
        （不复制）：调用方通常紧接着 `context.child(...)`，而 `child()` 本身就是不可变的。
        """
        scratchpad = getattr(state, "scratchpad", None) or {}
        existing = scratchpad.get("delegation") if isinstance(scratchpad, Mapping) else None
        if isinstance(existing, DelegationContext):
            return existing
        if isinstance(existing, Mapping):
            return cls.from_dict(existing)
        return cls(stack=[name], depth=0,
                   root_run_id=getattr(state, "run_id", "") or "",
                   budget=DEFAULT_TEAM_MAX_ROUNDS)


def compress_subagent_output(name: str, output: str, *, status: str = "FINISHED",
                            steps: int = 0,
                            max_chars: int = DEFAULT_SUBAGENT_MAX_CHARS) -> str:
    """把子 Agent 的输出压缩成回传给父 Agent 的字符串（**冻结格式**）：

        '[worker {name} | status={status} | steps={steps}]\\n{body}'

    - `body` 在 `len(output) <= max_chars` 时就是 `output` 原样；
    - 否则 = head 70% + `'\\n...[truncated {k} chars]...\\n'` + tail 30%
      （`truncate_head_tail(output, max_chars, head_ratio=SUBAGENT_HEAD_RATIO)`）；
    - `max_chars <= 0` 时**不做 body 压缩**（仍加 header）—— 这是显式的"别截断"开关；
    - **header 本身不计入 `max_chars`**：预算是对子 Agent 正文的，不是对元信息的。

    头尾都保留而不是只留头部：子 Agent 的结论常出现在开头（"结论先说"），
    而最后的观察/错误常出现在结尾；只留头会把"为什么失败"整段切掉。
    """
    header = f"[worker {name} | status={status} | steps={steps}]"
    body = output if max_chars <= 0 else truncate_head_tail(
        output, max_chars, head_ratio=SUBAGENT_HEAD_RATIO
    )
    return f"{header}\n{body}"


class MultiAgent(ABC):
    """多 Agent 编排器的公共基类（`SequentialAgent` / `HierarchicalAgent` 继承它）。"""

    def __init__(self, agents: Sequence[AgentLike], *, name: str = "team",
                 config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: Blackboard | None = None) -> None:
        self.agents: list[AgentLike] = list(agents)
        self.config: TeamConfig = config if config is not None else TeamConfig()
        # 默认各建一个：共享黑板是刻意的默认（多 Agent 协作的常见诉求就是共享状态），
        # 而"不共享黑板"的团队没有任何理由存在（传 None 只会得到第二个空黑板）。
        self.blackboard: Blackboard = blackboard if blackboard is not None else Blackboard()
        self.callbacks: CallbackManager = CallbackManager(callbacks or ())
        self._name = name or "team"
        # share_memory 的载体，见 `_shared_memory` 属性
        self._shared_memory = None

    # ---- 身份 ----

    @property
    def name(self) -> str:
        return self._name

    # ---- 共享记忆（[v2] share_memory 的落点）----

    @property
    def _shared_memory(self) -> "MemoryManager | None":
        """`share_memory=True` 时子 Agent 共用的 `MemoryManager`（**懒解析**）。

        三条解析顺序（越靠前优先级越高）：

        1. 显式赋值（子类在自己的 `__init__` 里 `self._shared_memory = ...`）；
        2. `self.manager.memory` —— `HierarchicalAgent` 天生有 `manager` 这个 Agent，
           而 §10.4 冻结的断言是 `worker.memory is manager.memory`：**manager 的记忆
           才是那个"共享点"**，于是这里用鸭子类型把 manager 的记忆接出来，
           子类什么都不用做就能满足该断言；
        3. 都没有（`SequentialAgent` 这类没有 manager 的编排器）-> 惰性 `from_config()`
           自建一个：整队共用同一个实例，即"共享记忆"。

        只有 `share_memory=True` 时才解析，避免不需要共享的团队白白造一个 MemoryManager。
        """
        if self.__shared_memory_override is not None:
            return self.__shared_memory_override
        if not self.config.share_memory:
            return None
        manager = getattr(self, "manager", None)  # 只有 HierarchicalAgent 有
        manager_memory = getattr(manager, "memory", None) if manager is not None else None
        if manager_memory is not None:
            self.__shared_memory_override = manager_memory
            return manager_memory
        # 函数内延迟 import：避免"import multiagent 就拉起整个 memory 层"
        from liteagent.memory.manager import MemoryManager

        self.__shared_memory_override = MemoryManager.from_config()
        return self.__shared_memory_override

    # setter 必须存在：子类（或测试）极可能直接写 `self._shared_memory = X`，
    # 只读属性会让那行代码 AttributeError。
    @_shared_memory.setter
    def _shared_memory(self, value: Any) -> None:
        self.__shared_memory_override = value

    def _apply_shared_memory(self, agent: Any) -> Any:
        """`config.share_memory=True` 时把共享记忆注入子 Agent（`agent.memory = ...`）。

        False 时**不动**（规范逐字要求）：子 Agent 保留自己 `__init__` 里造的那份。
        注入前用 `hasattr` 判定：嵌套的 `MultiAgent` 没有 `memory` 属性，
        硬 setattr 只会在它内部再被忽略一次，不如在这里留下痕迹（§13 红线 12）。
        返回 agent 本身，方便调用方写成 `await self._apply_shared_memory(x).arun(...)`。
        """
        if not self.config.share_memory:
            return agent
        memory = self._shared_memory
        if memory is None:  # 理论上不可达：share_memory=True 时上面的属性一定会给实例
            logger.warning("share_memory=True but no shared memory could be resolved; skip")
            return agent
        if not hasattr(agent, "memory"):
            logger.info(
                "share_memory=True but %r has no 'memory' attribute; skip injection",
                getattr(agent, "name", agent),
            )
            return agent
        setattr(agent, "memory", memory)
        return agent

    # ---- 执行 ----

    @abstractmethod
    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult: ...

    def run(self, input: str, **kwargs: Any) -> AgentResult:
        """同步入口 = `run_sync(lambda: self.arun(...))`（§2.3 冻结的包装形态）。

        必须传**工厂函数**而不是协程对象：`run_sync` 在"当前已在运行 loop 内"时会抛
        `ConfigError`，若先构造了协程再抛，那个协程永远不会被 await，
        3.10 会打 `RuntimeWarning: coroutine ... was never awaited`（§5.2）。
        """
        return run_sync(lambda: self.arun(input, **kwargs))

    # ---- 子类共用的判定/聚合工具 ----

    def _check_depth(self, context: DelegationContext, *, to_agent: str) -> None:
        """委派前的三道闸（顺序冻结，§10.1）：

        1. `depth > max_depth` -> `MaxDepthExceededError`；
        2. [v2] `context.exhausted()` -> `MaxDepthExceededError`（让 `max_rounds` 真正生效）；
        3. `would_cycle(to_agent)` 且 `enable_cycle_detection` -> `CycleDetectedError`。

        顺序为什么重要：先报"太深"再报"预算耗尽"，最后才是环 —— 环的判定依赖 stack，
        而 stack 在预算耗尽时往往已经很长；按这个顺序报错，消息最贴近根因。
        """
        if context is None:  # 防御：被脱离编排器直接调用时不做判定（调用方自担）
            return
        if context.depth > self.config.max_depth:
            raise MaxDepthExceededError(
                context.depth, self.config.max_depth,
                context={"to_agent": to_agent, "stack": list(context.stack)},
            )
        if context.exhausted():
            # 复用同一个异常类型：语义都是"不允许再往下委派了"，区别写在 message 里
            raise MaxDepthExceededError(
                context.depth, self.config.max_depth,
                message=(
                    f"delegation budget exhausted (budget={context.budget}) "
                    f"while delegating to {to_agent!r}"
                ),
                context={"to_agent": to_agent, "budget": context.budget,
                         "stack": list(context.stack)},
            )
        if self.config.enable_cycle_detection and context.would_cycle(to_agent):
            raise CycleDetectedError(
                list(context.stack),
                context={"to_agent": to_agent},
            )

    def _aggregate_usage(self, results: Sequence[AgentResult]) -> TokenUsage:
        """把多个阶段的 usage 加总（顺序编排的 `usage` 是全阶段之和）。

        用 `+`（`TokenUsage.__add__`）而不是 `+=`：前者每次生成新对象，
        不会把某一个子 Agent 的 usage 对象改掉（子 Agent 的 result 可能被外部持有）。
        """
        total = TokenUsage()
        for result in results:
            usage = getattr(result, "usage", None)
            if isinstance(usage, TokenUsage):
                total = total + usage
        return total

    def _failed(self, error: LiteAgentError, *, agent_name: str,
                metadata: Mapping[str, Any] | None = None) -> AgentResult:
        """把"整个编排失败"收敛成 `AgentResult`（而不是到处 raise）。

        `output` 用 `str(error)`：编排层的失败信息就是最有用的输出，留空串会让 CLI
        与 trace 只剩一个 status=FAILED。**禁止位置参数构造**（§9.4：字段顺序陷阱）。
        """
        return AgentResult(
            output=str(error),
            status=AgentStatus.FAILED,
            error=error,
            agent_name=agent_name,
            metadata=dict(metadata or {}),
        )

    def describe(self) -> dict[str, Any]:
        """给 CLI `multi` / trace 用的自描述。不是 `Agent.describe()` 的替代品。"""
        return {
            "type": type(self).__name__,
            "name": self.name,
            "agents": [getattr(agent, "name", str(agent)) for agent in self.agents],
            "blackboard_entries": len(self.blackboard),
            "config": self.config.to_dict(),
        }

    async def aclose(self) -> None:
        """逐层关闭子 Agent（释放线程池 / 文件句柄），以及 hierarchical 的 manager。

        单个子 Agent 关闭失败只记日志、不中断其它 —— 收尾路径抛异常会把真正的
        业务错误盖掉。`CancelledError` 不在捕获范围内（`except Exception`）。
        """
        targets: list[Any] = list(self.agents)
        manager = getattr(self, "manager", None)  # HierarchicalAgent 的 manager 不在 agents 里
        if manager is not None and all(manager is not item for item in targets):
            targets.append(manager)
        for agent in targets:
            closer = getattr(agent, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 - 收尾失败只留痕，不掩盖主错误
                logger.exception("aclose() failed for agent %r", getattr(agent, "name", agent))


def build_team(agents: Sequence[AgentLike], *, mode: str = "sequential",
               config: TeamConfig | None = None, blackboard: Blackboard | None = None,
               manager: "Agent | None" = None, name: str = "team") -> MultiAgent:
    """按 `mode` 造一个编排器（工厂函数，CLI 与 examples 的统一入口）。

    **两个子模块必须函数内 import**（§1.1 的 E3）：它们要继承 `MultiAgent`，
    顶层 import 会让 `base -> sequential -> base` 成环，import 顺序一变就炸。

    `mode` 大小写/首尾空白不敏感（"Sequential" 也认），未知值抛 `ConfigError`
    并把合法取值列出来（错误消息要能自纠正，别让调用方去翻文档）。
    `hierarchical` 缺 `manager` 同样抛 `ConfigError` —— 层级编排的 manager 是必需品，
    没有它就只能猜一个 worker 当 manager，那是错的。
    """
    normalized = (mode or "").strip().lower()
    if normalized == "sequential":
        from liteagent.multiagent.sequential import SequentialAgent

        return SequentialAgent(
            agents, name=name, config=config, blackboard=blackboard,
        )
    if normalized == "hierarchical":
        if manager is None:
            raise ConfigError(
                "build_team(mode='hierarchical') requires a manager Agent; got manager=None"
            )
        from liteagent.multiagent.hierarchical import HierarchicalAgent

        return HierarchicalAgent(
            manager, agents, name=name, config=config, blackboard=blackboard,
        )
    raise ConfigError(
        f"unknown team mode {mode!r}; expected one of {list(TEAM_MODES)}"
    )
