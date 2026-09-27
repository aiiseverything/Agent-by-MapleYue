from __future__ import annotations

# examples/06_multiagent_hierarchical.py —— Hierarchical 层级编排（manager + workers）
#
# 这个示例讲清楚另一件事：**一个主 Agent 怎么把任务拆开、交给若干个下属去做**。
# 和 05 的 Sequential 不同，这里没有"固定顺序的流水线"，而是：
#
#     manager（普通 ReAct Agent） + workers（被包装成 delegate_to_<name> 工具）
#
# 读完这个文件，你应该能回答下面六个问题（也是面试里最值得讲的六处）：
#
#   1. **委派即工具（delegation as a tool）**：每个 worker 被包装成一个
#      `delegate_to_<worker>` 工具塞进 manager 的工具集。manager 是**普通 Agent**，
#      它压根不知道"多 Agent"这回事 —— ReAct 循环照跑，只是工具箱里多了几个
#      "找同事帮忙"的工具。好处是复用（不用给编排层另写一个状态机），
#      代价是委派结果必须被压成**字符串**（模型侧只认文本）。
#
#   2. **extra_context 只是文本**：`task` 是"要做什么"，`extra_context` 是补充背景，
#      两者会被拼成 worker 的输入。它**绝不**参与环检测/深度判定 ——
#      如果委派上下文从工具参数里取，模型随便写一句 "depth=0 stack=[]" 就能绕过所有闸门。
#
#   3. **结果压缩**：worker 的原始输出可能几万字符，不能原样回灌给 manager。
#      压缩保留**头 70% + 尾 30%**（`compress_subagent_output`）：头是结论/结构，
#      尾是最近的进展/错误。原始全文不丢 —— 它同时进黑板与 `scratchpad`。
#
#   4. **失败是观察，不是异常**：worker 失败（返回 FAILED、甚至抛异常）都被压成
#      字符串回灌给 manager，manager 可以重试/改派/放弃 —— **单个 worker 的失败
#      从不中止 manager**。只有 manager 自己整体失败时才轮到 `propagate_failure`。
#
#   5. **环检测返回字符串而不是崩**：A 委派 B、B 又委派 A 时，闭包直接返回
#      `[delegation refused: cycle detected: ...]`，manager 把它当成一条普通观察 ——
#      一次"模型选错了同事"不应该升级成整轮 run 失败。
#
#   6. **任务分解还有第二条路径**：`adecompose()` 用一次 LLM 调用产出结构化 `Plan`
#      （`{"goal", "subtasks": [{id, description, assignee, depends_on}]}`），
#      `arun(plan=plan)` 按 `depends_on` 分层执行、层内并发。
#
# 跑法：`python3 examples/06_multiagent_hierarchical.py --offline`
# `--offline` 用 `ScriptedLLM`（脚本化假模型）驱动 manager 的分解与综合，
# **不联网、不需要 API key、输出确定**。

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# 允许直接 `python3 examples/06_multiagent_hierarchical.py` 运行（仓库没有 pip install 过）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liteagent import (
    Agent,
    Blackboard,
    DelegationContext,
    EventType,
    HierarchicalAgent,
    LLMConfig,
    LLMError,
    NO_TIMEOUT,
    Plan,
    ScriptedLLM,
    ScriptedResponse,
    SequentialAgent,
    SequentialStep,
    TeamConfig,
    ToolRegistry,
    build_llm,
    compress_subagent_output,
    tool,
)
from liteagent.multiagent.hierarchical import DECOMPOSE_PROMPT_TEMPLATE

# --------------------------------------------------------------------------------------
# 输出排版
# --------------------------------------------------------------------------------------

WIDTH = 88


def title(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def subtitle(text: str) -> None:
    print()
    print(f"--- {text} " + "-" * max(0, WIDTH - len(text) - 5))


def note(text: str) -> None:
    for line in text.strip("\n").split("\n"):
        print(f"  {line}".rstrip())


def field(label: str, value: Any) -> None:
    print(f"  {label:<22} {value}")


def clip(text: Any, limit: int = 70) -> str:
    """把一段文本压成单行短串（打印用）。"""
    flat = str(text).replace("\n", "\\n")
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="liteagent 示例 06：Hierarchical 多 Agent（manager 分解 + workers 协同）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--offline", action="store_true",
                        help="离线运行：用 ScriptedLLM 脚本驱动，不联网、不需要 API key（推荐）")
    parser.add_argument("--provider", default="echo",
                        help="非离线时使用的 provider（echo 也视为离线）")
    parser.add_argument("--model", default="", help="模型名，留空按 provider 取缺省值")
    parser.add_argument("--section", default="all",
                        choices=("all", "roster", "delegate", "run", "compression", "cycle", "plan"),
                        help="只跑某一节（默认 all）")
    return parser.parse_args(argv)


def build_llm_for(script: Sequence[ScriptedResponse], *, offline: bool,
                  provider: str, model: str) -> Any:
    if offline:
        return ScriptedLLM(list(script), model=model or "scripted-1")
    return build_llm(LLMConfig(provider=provider, model=model or "gpt-4o-mini"))


# --------------------------------------------------------------------------------------
# 玩具工具：让每个 worker 有工具可调（于是走 native/function-calling 模式）
# --------------------------------------------------------------------------------------


@tool
def read_spec(topic: str) -> str:
    """读取 topic 的需求说明。"""
    return f"[spec] {topic} 的需求：支持离线、零依赖、可扩展"


@tool
def write_patch(module: str) -> str:
    """为 module 生成一个（假的）代码补丁。"""
    return f"[patch] {module}: +def handle():\n    return 'ok'"


@tool
def run_lint(module: str) -> str:
    """对 module 做（假的）静态检查。"""
    return f"[lint] {module}: 0 errors, 1 warning (line too long)"


# --------------------------------------------------------------------------------------
# worker 包装：记录 delegations 里拿到的上下文，便于在示例里打印
# --------------------------------------------------------------------------------------


class InputRecorder:
    """订阅某个 Agent 的 `RUN_STARTED`，记录它**真正收到**的输入。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, event: Any) -> None:
        if getattr(event, "type", None) == EventType.RUN_STARTED:
            self.seen.append(str(event.data.get("input", "")))

    def last(self) -> str:
        return self.seen[-1] if self.seen else ""


# --------------------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------------------


def build_workers(*, offline: bool, provider: str, model: str,
                  recorders: dict[str, InputRecorder]) -> list[Agent]:
    """造三个 worker（下属）。

    worker 的"剧本"（`ScriptedLLM` 的响应队列）就是它在真实场景里会做的事：
    查需求 -> 写补丁 -> 做检查。它们各自有工具，因此 ReAct 走 function-calling。
    """

    def worker(name: str, description: str, tool_obj: Any, script: list[Any]) -> Agent:
        recorder = InputRecorder()
        recorders[name] = recorder
        return Agent(
            llm=build_llm_for(script, offline=offline, provider=provider, model=model),
            tools=ToolRegistry([tool_obj]),
            name=name,
            description=description,
            callbacks=[recorder],
        )

    return [
        worker("researcher", "查需求、给事实", read_spec,
               [ScriptedResponse.tool("read_spec", {"topic": "liteagent"}),
                ScriptedResponse.text("需求要点：离线可跑、零依赖、工具可扩展。")]),
        worker("coder", "按需求写代码补丁", write_patch,
               [ScriptedResponse.tool("write_patch", {"module": "liteagent.cli"}),
                ScriptedResponse.text("补丁：给 cli 加了 handle() 入口。")]),
        worker("reviewer", "审补丁、跑静态检查", run_lint,
               [ScriptedResponse.tool("run_lint", {"module": "liteagent.cli"}),
                ScriptedResponse.text("审校结论：补丁可用，1 条 warning 不阻塞。")]),
    ]


def build_manager(*, offline: bool, provider: str, model: str,
                  script: list[Any], recorder: InputRecorder | None = None,
                  name: str = "manager") -> Agent:
    """造 manager。它是**普通 Agent**：分解、挑下属、综合，全靠 ReAct + delegate 工具。"""
    return Agent(
        llm=build_llm_for(script, offline=offline, provider=provider, model=model),
        tools=ToolRegistry([]),           # 一开始没有工具：delegate 工具在 arun 时才装上
        name=name,
        description="把任务拆开、分给下属、再综合结果",
        callbacks=[recorder] if recorder is not None else None,
    )


WORKER_DESCRIPTIONS: dict[str, str] = {
    "researcher": "负责查需求与事实，输入是一个主题",
    "coder": "负责按需求写代码补丁，输入是模块名",
    "reviewer": "负责审补丁与静态检查，输入是模块名",
}


# --------------------------------------------------------------------------------------
# 各节
# --------------------------------------------------------------------------------------


def section_roster(args: argparse.Namespace, offline: bool, workers: list[Agent],
                   manager: Agent, team: HierarchicalAgent) -> None:
    """[1/6] 演员表 + delegate 工具面板。"""
    title("[1/6] 演员表：一个 manager + 三个 worker")
    note("""
`HierarchicalAgent(manager, workers)` 里的 manager 是**普通 Agent**（和 01/02 里那个
一模一样），workers 是实现 `name` + `arun` 的任意对象（甚至不必是 `Agent`）。
框架在 `arun` 的时候才把每个 worker 包装成一个 `delegate_to_<name>` 工具塞进 manager。

`worker_descriptions` 值得认真写：模型选工具几乎只看 name + description，
`delegate_to_researcher` 这个名字本身的信息量很低，"负责查需求与事实"才有用。
    """)
    field("team.name", team.name)
    field("manager", f"{manager.name}（tools 初始={manager.tools.names()}）")
    for worker in workers:
        field(f"worker [{worker.name}]",
              f"{worker.description}；工具={worker.tools.names()}")
    print()
    for line in str(team.describe()).split("\n"):
        print(f"    {line}")


def section_delegate(args: argparse.Namespace, offline: bool,
                     team: HierarchicalAgent) -> None:
    """[2/6] delegate 工具长什么样。"""
    title("[2/6] 委派工具面板：delegate_to_<worker>")
    note("""
每个 worker 会生成一个同步工具，参数只有两个：`task`（必填）与 `extra_context`（选填）。
注意下面三处刻意的设计：

* `required=["task"]` —— 光说"帮我看看"是不够的，必须写清楚要做什么；
* `idempotent=False` —— 委派有副作用（会真的跑一遍子 Agent），所以不会被自动重试；
* `timeout_s=NO_TIMEOUT` —— 委派可能跑很久（子 Agent 自己也在做多轮 ReAct），
  30 秒的默认工具超时对它来说是错的。
    """)
    tools = team.delegate_tools()
    for tool_obj in tools:
        field(tool_obj.name, f"timeout_s={'NO_TIMEOUT' if tool_obj.spec.timeout_s == NO_TIMEOUT else tool_obj.spec.timeout_s} "
                             f"idempotent={tool_obj.spec.idempotent} tags={tool_obj.spec.tags}")
        print(f"      description: {tool_obj.description}")
    print()
    note("""
参数的 schema 三个工具是**同一份**（框架冻结的形状），所以只打一遍：
    """)
    print(f"    {tools[0].parameters}")
    print()
    note("""
`NO_TIMEOUT` 是一个**负数哨兵**（不是 0 秒），执行器看到它就完全不设超时：
    """)
    field("NO_TIMEOUT", NO_TIMEOUT)


def section_run(args: argparse.Namespace, offline: bool, team: HierarchicalAgent,
                recorders: dict[str, InputRecorder]) -> None:
    """[3/6] 真跑一次：分解 -> 委派 -> 压缩回灌 -> 综合。"""
    title("[3/6] 真跑一次：manager 分解 + 委派 + 综合")
    note("""
manager 的脚本是三步：
    1. 调 `delegate_to_researcher`，task 写清楚要什么，extra_context 带上已知背景；
    2. 调 `delegate_to_coder`；
    3. 拿到两个压缩字符串后，给最终答案。

**为什么 manager 不用"先规划再执行"**：ReAct 本身就是"边想边做"，
delegate 工具让"找人帮忙"变成循环里的一个普通动作。想要显式计划见 [6/6]。
    """)
    result = team.run("给 liteagent 的 CLI 加一个 handle 入口")

    field("status", result.status)
    field("agent_name", result.agent_name)
    field("steps", result.steps)
    field("final answer", clip(result.output, 100))
    print()
    print("  worker 真正收到的输入（= task + extra_context 拼接）：")
    for name, recorder in recorders.items():
        if recorder.seen:
            field(f"worker [{name}]", clip(recorder.last(), 80))
        else:
            field(f"worker [{name}]", "(本次没有被委派)")
    print()
    note("""
researcher 那一条注意看：`extra_context` 被拼在 task 后面，中间空了一行 ——
对 worker 来说这只是一段普通输入文本，它并不知道"哪半句是背景"。
    """)
    print()
    subtitle("黑板上的归档记录：键是 subagent:<worker>:<seq>")
    note("""
每发生一次委派，序号就 +1（同一个 worker 被委派两次 -> `:1` 和 `:2`）。
序号计数器用 `threading.Lock` 保护 —— 因为委派是**同步工具**，会被 manager 的执行器
丢到线程池里并发跑（见 [4/6] 的说明）。
    """)
    for entry in team.blackboard.list(tags=("subagent",)):
        field(f"{entry.key}", f"v{entry.version} by {entry.author!r} -> {clip(entry.value, 60)}")
    subtitle("result.metadata['delegations']（编排器从黑板汇总出来的）")
    for item in result.metadata.get("delegations", []):
        field(item.get("key"), clip(item.get("value"), 60))


def section_compression(args: argparse.Namespace, offline: bool) -> None:
    """[4/6] 结果压缩：头 70% + 尾 30%。"""
    title("[4/6] 子 Agent 结果怎么被压缩回灌")
    long_output = ("【结论】这段输出故意写得很长，用来演示压缩。" * 8) + \
                  "【最后的进展】已经跑完 8 轮 ReAct，剩余风险是 warning。"
    field("原始长度", f"{len(long_output)} 字符")

    compressed = compress_subagent_output("worker_x", long_output, status="FINISHED",
                                          steps=8, max_chars=120)
    field("压缩后长度", f"{len(compressed)} 字符（含 header 那一行）")
    print()
    for line in compressed.split("\n"):
        print(f"      {clip(line, 100)}")
    print()
    note("""
`max_chars` 是**正文**的预算：header `[worker x | status=... | steps=...]` 不计入。
正文超预算时压成"头 70% + 尾 30%"，中间那一段被换成 `...[truncated N chars]...`。
为什么头尾都留：子 Agent 的结论常出现在开头，而"为什么失败"常出现在结尾 ——
只留头会把失败原因整段切掉。

**真实路径上这件事由 `TeamConfig.subagent_output_max_chars` 控制**，
`compress_subagent_output=False` 则完全不截断（只保留 header）。
    """)

    subtitle("对照：把一个 worker 的输出设成 40 字符预算")
    recorder = InputRecorder()
    worker = Agent(
        llm=build_llm_for([ScriptedResponse.text(long_output)], offline=offline,
                          provider=args.provider, model=args.model),
        tools=ToolRegistry([read_spec]), name="chatty", description="话很多的 worker",
        callbacks=[recorder],
    )
    manager = build_manager(offline=offline, provider=args.provider, model=args.model, script=[
        ScriptedResponse.tool("delegate_to_chatty", {"task": "随便说点什么"}),
        ScriptedResponse.text("manager 已经拿到压缩后的摘要。"),
    ])
    team = HierarchicalAgent(manager, [worker], name="compress_demo",
                             config=TeamConfig(subagent_output_max_chars=40))
    result = team.run("演示压缩")
    archived = team.blackboard.read("subagent:chatty:1")
    field("worker 原始输出长度", f"{len(long_output)} 字符")
    field("回灌给 manager 的长度", f"{len(archived)} 字符")
    field("黑板里的样子", clip(archived, 90))
    field("结果（manager 最终答案）", clip(result.output, 60))

    subtitle("worker 失败同样只是「一段字符串」（manager 不会因此中止）")
    note("""
下面的 worker 模型坏了（返回一条 LLMError）。注意三件事：
  1. 委派**没有抛异常**，manager 照常跑完并给出最终答案（status=FINISHED）；
  2. 回灌字符串的 header 里多了 `error=...`，格式仍是 `header + 正文`；
     正文一旦存在就**照样回灌** —— "为什么失败"经常就写在正文末尾，
     不能因为是失败路径就把正文丢掉（本例里模型第一次调用就炸了，所以正文是空的）；
  3. 失败会额外多发一条 `AGENT_RETURN {failed: True}`，方便只订阅失败的消费者。
    """)
    flaky = Agent(
        llm=build_llm_for([ScriptedResponse.error(LLMError("子 Agent 的模型宕了"))],
                          offline=offline, provider=args.provider, model=args.model),
        tools=ToolRegistry([read_spec]), name="flaky", description="模型坏掉的 worker",
    )
    recovering_manager = build_manager(
        offline=offline, provider=args.provider, model=args.model, name="recovering_manager",
        script=[ScriptedResponse.tool("delegate_to_flaky", {"task": "把这件事办一下"}),
                ScriptedResponse.text("manager 看到 flaky 失败了，改由自己给出结论。")],
    )
    recovering = HierarchicalAgent(recovering_manager, [flaky], name="recovery_demo")
    recovered = recovering.run("跑一个会失败的子任务")
    field("status", recovered.status)
    field("output", clip(recovered.output, 60))
    field("黑板里的 subagent:flaky:1", clip(recovering.blackboard.read("subagent:flaky:1"), 90))


def section_cycle(args: argparse.Namespace, offline: bool) -> None:
    """[5/6] 环检测：A -> B -> A，返回 refused 字符串。"""
    title("[5/6] 环检测：A 委派 B、B 又要委派回 A")
    note("""
委派栈（`DelegationContext.stack`）会随委派一层层变长：
外层是 `pipeline`，它把活交给 `subteam`，于是 subteam 看到的栈是
`[pipeline, subteam]`。此时 subteam 再想委派给一个叫 `pipeline` 的下属 ——
**栈里已经有这个名字**，闭包直接返回：

    [delegation refused: cycle detected: stack=[pipeline,subteam] -> pipeline]

注意它是**返回一个字符串**而不是抛异常：manager 会把它当成一条普通观察，
于是"模型选错了同事"不会升级成整轮 run 失败。

**这里为什么要用 Sequential 包一层**：`SequentialAgent` 在把步骤交给一个
`MultiAgent` 时会显式传 `context=`，所以委派栈能跨编排器传下去。
（`HierarchicalAgent` 的 delegate 路径是 `run_sync(worker.arun(task, state=...))`，
不传 `context`，因此**嵌套的层级编排会看不到祖先的栈** —— 这是本框架的一个已知边界，
见文件末尾"已知边界"一节。这也是为什么"环"用 Sequential 外层来演示最自然。）
    """)

    class BackEdge:
        """一条"指回上游编排器"的边：名字与祖先同名，用来触发环检测。

        它**故意**在 `arun` 里抛错：万一环检测失效、这条边真的被执行了，
        我们会立刻在委派结果里看到一条 FAILED 观察，而不是陷入无界递归。
        `invocations` 计数器则用来在示例末尾证明"它一次都没跑过"。
        """

        def __init__(self, target_name: str) -> None:
            self.name = target_name
            self.description = "把任务打回给上游编排器（用于演示环检测）"
            self.invocations = 0

        async def arun(self, input: str, **kwargs: Any) -> Any:
            self.invocations += 1
            raise RuntimeError("back-edge executed: cycle detection should have refused it")

    back = BackEdge("pipeline")
    sub_manager = build_manager(
        offline=offline, provider=args.provider, model=args.model, name="subteam_manager",
        script=[
            # 第一步就想把活打回上游 -> 会被拒
            ScriptedResponse.tool("delegate_to_pipeline",
                                  {"task": "上游的同学，麻烦你接手",
                                   "extra_context": "本组资源不够"}),
            # 拿到 refused 字符串后，自己给出结论
            ScriptedResponse.text("子团队结论：上游被拒（环），我们本地处理完了。"),
        ],
    )
    subteam = HierarchicalAgent(sub_manager, [back], name="subteam")
    outer = SequentialAgent(
        [SequentialStep(agent=subteam, name="subteam")], name="pipeline",
    )
    result = outer.run("请研究一下 liteagent 的架构")

    field("外层 status", result.status)
    field("外层 output", clip(result.output, 80))
    field("back-edge 被执行次数", back.invocations)
    print()
    subtitle("manager 真正看到的那条观察（拒绝**不会**写黑板，它只是工具返回值）")
    note("""
被拒绝的委派在 `_run_delegate` 之前就返回了，所以黑板上不会留下 subagent:* 记录 ——
想证明"模型真的看到了这句话"，只能去 manager 自己的 transcript 里找。
下面是从 manager 的消息历史里翻出来的、包含 refused 的那几条：
    """)
    hits = [m for m in sub_manager.state.messages if "delegation refused" in str(m.content)]
    if not hits:
        print("    (没有找到：manager 的 transcript 里没有 refused 文本)")
    for message in hits:
        field(str(getattr(message, "role", "?")), clip(message.content, 95))
    print()
    subtitle("subteam 的黑板（只有真正跑过的委派才会归档）")
    field("subagent:* 条数", len(subteam.blackboard.list(tags=("subagent",))))
    note("""
顺带一提：拒绝委派时编排器还会发一条 `AGENT_DELEGATE {refused: True,
refused_reason: "cycle"}` 事件（日志里也能看到 WARNING）。
"拒绝"是**正常控制流**（环/预算/并发满都是预期内的），所以不抛异常，
但绝不静默 —— 否则 trace 里只会看到一个莫名其妙的字符串返回。
    """)

    subtitle("另外两种拒绝：预算耗尽 与 深度超限")
    note("""
预算（`TeamConfig.max_rounds`）与深度（`max_depth`）走的是同一套"返回字符串"的逻辑。
下面直接构造一个已耗尽的 `DelegationContext` 来演示 —— 委派工具会**先判预算再排队**，
不会让你白等信号量。
    """)
    # 这里刻意**新造**一个 worker：共享的那批 worker 的脚本是给 [3/6] 用的，
    # 借来跑探针会把它们的响应队列提前消费掉（示例自己踩过的坑）。
    probe_recorders: dict[str, InputRecorder] = {}
    probe_worker = build_workers(offline=offline, provider=args.provider, model=args.model,
                                 recorders=probe_recorders)[0]
    probe = HierarchicalAgent(
        build_manager(offline=offline, provider=args.provider, model=args.model,
                      script=[ScriptedResponse.text("(不会跑到这里)")]),
        [probe_worker], name="probe", config=TeamConfig(max_rounds=1, max_depth=1),
    )
    delegate_tool = probe.delegate_tools()[0]
    import contextvars  # noqa: PLC0415 - 示例：只在这一节用一次
    from liteagent.multiagent.hierarchical import _CURRENT_CTX  # noqa: PLC0415

    for label, ctx in (
        ("budget 耗尽", DelegationContext(stack=["probe"], depth=0, budget=0)),
        ("depth 已达上限", DelegationContext(stack=["probe"], depth=1, budget=5)),
        ("正常", DelegationContext(stack=["probe"], depth=0, budget=5)),
    ):
        token = _CURRENT_CTX.set(ctx)
        try:
            text = str(delegate_tool.run({"task": "试一下"}))
        finally:
            _CURRENT_CTX.reset(token)
        field(label, clip(text, 80))
    note("""
注意"depth 已达上限"那一行：返回的文案与"环"**一模一样**（框架只冻结了那一句），
真实原因写在事件与日志里（`refused_reason="depth"` 和一条 WARNING）——
"看到 cycle 字样其实是深度不够"是个很容易踩的坑，示例把它如实标出来。
    """)


def section_plan(args: argparse.Namespace, offline: bool) -> None:
    """[6/6] 显式计划：adecompose + arun_plan。"""
    title("[6/6] 显式计划：adecompose() + arun(plan=...)")
    note("""
前面走的是"边想边做"的 ReAct 路线。第二条路径是**先规划、再执行**：

    plan = await ha.adecompose(task)        # 一次 LLM 调用，产出结构化 Plan
    result = await ha.arun(task, plan=plan) # 按 depends_on 分层执行，层内可并发

`adecompose` 用的是**一次原始 LLM 调用**（直接调 `manager.llm.achat`），
不走 ReAct 循环 —— 否则那份 JSON 会被塞进 Thought/Action 解析器里。
它的输出是容错解析的：剥 ```json 围栏、容忍前后的解释文字、`depends_on` 写成
单个字符串也认、缺 `description` 的条目直接跳过。
    """)
    plan_json = """```json
{"goal": "给 CLI 加 handle 入口",
 "subtasks": [
   {"id": "t1", "description": "查需求", "assignee": "researcher", "depends_on": []},
   {"id": "t2", "description": "写补丁", "assignee": "coder", "depends_on": ["t1"]},
   {"id": "t3", "description": "审补丁", "assignee": "reviewer", "depends_on": ["t2"]}
 ]}
```"""
    field("分解 prompt 的模板", "DECOMPOSE_PROMPT_TEMPLATE（渲染后是一段 JSON-only 的指令）")
    for line in DECOMPOSE_PROMPT_TEMPLATE.strip().split("\n")[:6]:
        print(f"    {clip(line, 90)}")
    print("    ...")

    plan = Plan.from_json(plan_json)
    subtitle("Plan.render()（人读的计划）")
    for line in plan.render().split("\n"):
        print(f"    {line}")
    field("plan.goal", plan.goal)
    field("plan.to_dict()", clip(plan.to_dict(), 90))

    subtitle("真的执行一遍（三分层、逐层依赖）")
    note("""
每个 subtask 会跑对应的 worker，结果写进黑板 `subtask:<id>`。
`arun_plan` 的返回值里 `metadata['plan']` 是计划的最终形态（含每条的 status/result）。
层内的完成顺序**不做保证**（`subagent_concurrency` 控制并发峰值）。
    """)
    manager_for_plan = build_manager(
        offline=offline, provider=args.provider, model=args.model, name="planner",
        script=[ScriptedResponse.text(plan_json)],
    )
    # 同样新造一批 worker：它们的响应队列要留给这一次计划执行。
    plan_recorders: dict[str, InputRecorder] = {}
    plan_workers = build_workers(offline=offline, provider=args.provider, model=args.model,
                                 recorders=plan_recorders)
    team = HierarchicalAgent(manager_for_plan, plan_workers, name="planned_team")
    import asyncio  # noqa: PLC0415 - 示例：只在这一节用一次

    async def _run() -> Any:
        decomposed = await team.adecompose("给 liteagent 的 CLI 加一个 handle 入口")
        return await team.arun("给 liteagent 的 CLI 加一个 handle 入口", plan=decomposed)

    result = asyncio.run(_run())
    field("status", result.status)
    field("output", clip(result.output, 90))
    print()
    for item in result.metadata.get("steps", []):
        field(f"subtask [{item.get('id')}] {item.get('assignee')}",
              f"status={item.get('status')} -> {clip(item.get('output'), 55)}")
    subtitle("黑板：subtask:<id>")
    for entry in team.blackboard.list(tags=("subtask",)):
        field(entry.key, clip(entry.value, 70))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


SECTION_ORDER = ("roster", "delegate", "run", "compression", "cycle", "plan")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    offline = bool(args.offline) or args.provider == "echo"
    if not offline:
        has_key = any(os.environ.get(name) for name in
                      ("LITEAGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                       "ANTHROPIC_API_KEY"))
        if not has_key:
            print(f"[!] provider={args.provider!r} 需要 API key；本机没有检测到。")
            print("    离线演示请用：--offline（或 --provider echo）")
            return 2

    title("liteagent 示例 06：Hierarchical 多 Agent")
    field("模式", "offline（ScriptedLLM 脚本驱动）" if offline
          else f"live（provider={args.provider}）")
    field("节", args.section)

    # 一次装配，各节复用（[6/6] 会另外造一套，因为它要在 `arun_plan` 里跑）。
    recorders: dict[str, InputRecorder] = {}
    workers = build_workers(offline=offline, provider=args.provider, model=args.model,
                            recorders=recorders)
    manager = build_manager(offline=offline, provider=args.provider, model=args.model, script=[
        ScriptedResponse.tool("delegate_to_researcher",
                              {"task": "查一下 liteagent CLI 的现状",
                               "extra_context": "已知：CLI 是 argparse 子命令风格"}),
        ScriptedResponse.tool("delegate_to_coder", {"task": "给 liteagent.cli 写 handle 入口"}),
        ScriptedResponse.text("综合结论：researcher 给了需求要点，coder 给了补丁，"
                              "可以进 review。"),
    ])
    blackboard = Blackboard()
    team = HierarchicalAgent(
        manager, workers, name="dev_team", blackboard=blackboard,
        worker_descriptions=WORKER_DESCRIPTIONS,
    )

    runners = {
        "roster": lambda: section_roster(args, offline, workers, manager, team),
        "delegate": lambda: section_delegate(args, offline, team),
        "run": lambda: section_run(args, offline, team, recorders),
        "compression": lambda: section_compression(args, offline),
        "cycle": lambda: section_cycle(args, offline),
        "plan": lambda: section_plan(args, offline),
    }
    sections = SECTION_ORDER if args.section == "all" else (args.section,)
    for name in sections:
        runners[name]()

    title("小结（面试可以这么讲）")
    note("""
1. 委派即工具：worker 变成 `delegate_to_<name>` 工具，manager 仍是普通 ReAct Agent ——
   零新增状态机，代价是结果必须压成字符串。
2. `extra_context` 只是文本，绝不参与环/深度/预算判定（否则模型一句话就能绕过闸门）。
3. 压缩保留头 70% + 尾 30%，原始全文进黑板与 scratchpad，不丢信息。
4. worker 的失败是"观察"不是"异常"：单点失败从不中止 manager。
5. 环检测返回字符串而不是抛异常；拒绝也会发 AGENT_DELEGATE{refused: True} 事件。
6. 想要显式计划就用 `adecompose()` + `arun(plan=...)`，按 depends_on 分层并发。

已知边界（如实记录，不粉饰）：`HierarchicalAgent` 的 delegate 路径不向子编排器
传 `context=`，因此**嵌套的层级编排拿不到祖先的委派栈**（深度/预算/环检测
在跨实例时会重新计数）。本示例的环检测因此用 Sequential 外层来演示。
    """)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
