from __future__ import annotations

# examples/05_multiagent_sequential.py —— Sequential 流水线（A -> B -> C）
#
# 这个示例讲清楚一件事：**多个 Agent 怎么像流水线一样串起来**。
# 三个各司其职的 Agent（研究员 -> 写作者 -> 审校者）共享一块黑板，每一步的输入
# 由 `SequentialStep.input_template` 决定，失败时走 `propagate_failure`。
#
# 读完这个文件，你应该能回答下面五个问题（也是面试里最常被追问的五处）：
#
#   1. 步骤之间的输入是怎么传的？
#      —— 模板，不是约定。`{input}` 是原始用户输入，`{prev}` 是上一步的输出，
#         `{steps[阶段名]}` 是任意历史阶段的输出。缺省规则：第 0 步 `{input}`，
#         其余 `{prev}`。所以最常见的"三步流水线"里**一个字都不用写**。
#      模板由 `config.render_template` 渲染，它用 `_SafeDict`：**顶层**缺 key 时保留
#      `{key}` 字面量而不是抛 KeyError；但 `{steps[未知阶段]}` 是 dict 元素查找，
#      会抛 KeyError -> 被编排器包成 ConfigError（配置错误，当场失败，见 [3/6]）。
#
#   2. 阶段之间还传了什么？
#      —— 只有黑板和 `scratchpad["delegation"]`（委派上下文：栈/深度/预算）。
#        每个阶段都会新建自己的 `AgentState`，**不复用父 state**：否则第 3 步会
#        在 transcript 里"看见"第 1 步的 Thought，trace 也没法按阶段切分。
#
#   3. 每一步的输出去哪了？
#      —— 写进共享黑板（`tags=("stage",)`，key 默认是阶段名，可用 `output_key` 覆盖），
#        同时在 `result.metadata["steps"]` 里留一份摘要。
#        黑板是**带版本号**的：同一个 key 写两次 -> version 1 -> 2。
#
#   4. 某个阶段失败了会怎样？
#      —— 三种走法（`propagate_failure`）：
#          return   立即停，把失败编码进 `AgentResult`（status=FAILED）
#          raise    抛 `DelegationError`
#          continue 把失败当成一条观察文本交给下一步
#        `optional=True` 的阶段**无视** propagate_failure，一律"继续"。
#
#   5. 怎么跑？
#      `python3 examples/05_multiagent_sequential.py --offline`
#      `--offline` 用 `ScriptedLLM`（脚本化的假模型）驱动，**不联网、不需要 API key、
#      输出内容完全确定**（只有墙钟耗时数字会变）。不带 `--offline` 时用 `--provider` 指定真实模型（默认 echo，
#      离线占位；要真效果请给 `--provider deepseek` 之类并配好 API key）。

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# 允许直接 `python3 examples/05_multiagent_sequential.py` 运行（仓库没有 pip install 过，
# `liteagent` 包不在 site-packages 里）。这一行必须出现在 `import liteagent` 之前。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liteagent import (
    Agent,
    Blackboard,
    DelegationError,
    EventType,
    LLMConfig,
    LLMError,
    ScriptedLLM,
    ScriptedResponse,
    SequentialAgent,
    SequentialStep,
    TeamConfig,
    ToolRegistry,
    build_llm,
    tool,
)
from liteagent.config import render_template
from liteagent.errors import ConfigError

# --------------------------------------------------------------------------------------
# 小工具：把输出排得好看一点（示例是给人看的，输出可读性算交付质量的一部分）
# --------------------------------------------------------------------------------------

WIDTH = 84


def title(text: str) -> None:
    """打印一级标题。"""
    print()
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def subtitle(text: str) -> None:
    """打印二级标题。"""
    print()
    print(f"--- {text} " + "-" * max(0, WIDTH - len(text) - 5))


def note(text: str) -> None:
    """打印一段说明（缩进两格；空行不留尾随空格）。"""
    for line in text.strip("\n").split("\n"):
        print(f"  {line}".rstrip())


def field(label: str, value: Any) -> None:
    """打印一个 '标签 + 值' 行（标签左对齐到 20 列，超长则用一个空格分隔）。"""
    print(f"  {label:<20} {value}")


# --------------------------------------------------------------------------------------
# 一组玩具工具：它们让每个 Agent 有"可以调的工具"，于是 ReAct 走 function-calling(native) 模式。
# 注意工具是纯函数的**离线假实现** —— 示例不联网、不读写真实文件。
# --------------------------------------------------------------------------------------


@tool
def search_notes(topic: str) -> str:
    """检索内部笔记库里与 topic 相关的要点。"""
    return (
        f"[notes] {topic} 的三条要点: "
        f"(1) {topic} 的背景定义; (2) {topic} 的常见误区; (3) {topic} 的最佳实践"
    )


@tool
def count_words(text: str) -> int:
    """统计一段文本的词数（按空白切分）。"""
    return len(text.split())


@tool
def check_style(text: str) -> str:
    """对一段文本做风格检查，返回整改建议。"""
    return f"[style] 共 {len(text)} 字符；建议：结论前置、每段不超过 5 行"


#: 每个阶段给哪些工具（让 `.tools.names()` 的输出看得见）
RESEARCHER_TOOLS: tuple[Any, ...] = (search_notes,)
WRITER_TOOLS: tuple[Any, ...] = (count_words,)
REVIEWER_TOOLS: tuple[Any, ...] = (check_style,)

#: 真实 provider 的缺省模型（不带 --model 时用）
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-3-5-sonnet-latest",
    "echo": "echo-1",
}


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行。

    `--offline`（或 `--provider echo`）是**离线**开关：用 `ScriptedLLM` 脚本驱动，
    零网络、零 API key、输出确定。示例的课程内容全部在离线路径上演示。
    """
    parser = argparse.ArgumentParser(
        description="liteagent 示例 05：Sequential 多 Agent 流水线（研究员 -> 写作者 -> 审校者）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="离线运行：用 ScriptedLLM 脚本驱动，不联网、不需要 API key（推荐）",
    )
    parser.add_argument(
        "--provider", default="echo",
        help="非离线时使用的 provider（echo/openai/deepseek/anthropic...）；echo 也视为离线",
    )
    parser.add_argument("--model", default="", help="模型名，留空按 provider 取缺省值")
    parser.add_argument(
        "--section", default="all",
        choices=("all", "basic", "template", "blackboard", "failure", "extras"),
        help="只跑某一节（默认 all）",
    )
    return parser.parse_args(argv)


def build_llm_for(script: Sequence[ScriptedResponse], *, offline: bool,
                  provider: str, model: str) -> Any:
    """按模式造一个 LLM 客户端。

    离线 -> `ScriptedLLM`（按脚本一条条吐响应）。
    在线 -> `build_llm(LLMConfig(...))`，走真实 provider（本机无 key 时不会走到这条路）。
    """
    if offline:
        # model= 只是给 LLMResponse.model 一个名字；ScriptedLLM 永远不会发请求。
        return ScriptedLLM(list(script), model=model or "scripted-1")
    return build_llm(LLMConfig(provider=provider, model=model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")))


# --------------------------------------------------------------------------------------
# 阶段输入记录器：证明"模板渲染出来的东西"和"Agent 真正收到的输入"是同一个
# --------------------------------------------------------------------------------------


class StageInputRecorder:
    """订阅每个阶段 Agent 的 `RUN_STARTED` 事件，记录它**真正**收到的输入。

    为什么需要它：`SequentialStep.input_template` 是我们自己写的字符串，
    如果我们只是"再调用一次 render_template 把结果打印出来"，那打印的只是我们的
    期望值 —— 一旦框架的渲染规则变了，打印仍然好看，但用户会被误导。
    订阅事件拿到的则是**事实**（`RUN_STARTED.data["input"]` 是 Agent 收到的那串）。
    """

    def __init__(self) -> None:
        self.seen: dict[str, list[str]] = {}

    def __call__(self, event: Any) -> None:
        if getattr(event, "type", None) != EventType.RUN_STARTED:
            return
        name = str(getattr(event, "agent_name", "") or "?")
        value = str(event.data.get("input", ""))
        self.seen.setdefault(name, []).append(value)

    def last(self, agent_name: str, index: int = 0) -> str:
        """取某次运行的输入（index 支持同一个 Agent 被跑多次的情况）。"""
        values = self.seen.get(agent_name) or [""]
        return values[index] if index < len(values) else values[-1]


class TeamEventRecorder:
    """订阅团队级事件，打印 `AGENT_DELEGATE` / `AGENT_RETURN` 的配对。

    编排器每委派一个阶段会发两条事件：一条 `agent_delegate`（from -> to, depth, refused）、
    一条 `agent_return`（status, steps, output_len, duration_ms, failed）。
    配对是**保证**的：`agent_return` 在失败处理之前发，所以即使随后要 raise，
    trace 里也能看到子 Agent 到底返回了什么。
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def __call__(self, event: Any) -> None:
        if getattr(event, "type", None) in (EventType.AGENT_DELEGATE, EventType.AGENT_RETURN):
            self.events.append(event)

    def render(self) -> str:
        lines = []
        for event in self.events:
            data = event.data
            arrow = f"{data.get('from', '?')} -> {data.get('to', '?')}"
            if event.type == EventType.AGENT_DELEGATE:
                lines.append(f"    [delegate] step={event.step} {arrow} "
                             f"depth={data.get('depth')} refused={data.get('refused')}")
            else:
                lines.append(f"    [return  ] step={event.step} {arrow} "
                             f"status={data.get('status')} steps={data.get('steps')} "
                             f"output_len={data.get('output_len')} "
                             f"failed={data.get('failed')}")
        return "\n".join(lines)


def print_entries(blackboard: Blackboard, *, label: str = "黑板快照") -> None:
    """打印黑板上每条记录的 版本号 / 作者 / 标签 / 值（这是"共享状态"的可见证据）。"""
    subtitle(label)
    entries = blackboard.list()
    if not entries:
        print("  (空)")
        return
    for entry in entries:
        value = str(entry.value).replace("\n", "\\n")
        if len(value) > 60:
            value = value[:57] + "..."
        field(f"{entry.key}", f"v{entry.version} by {entry.author!r} tags={entry.tags} -> {value!r}")


def print_stage_summaries(result: Any) -> None:
    """打印 `result.metadata["steps"]`（每个阶段的摘要）。"""
    subtitle("metadata['steps']（流水线视角的阶段摘要）")
    steps = result.metadata.get("steps") or []
    if not steps:
        print("  (空)")
        return
    for item in steps:
        out = str(item.get("output", "")).replace("\n", "\\n")
        if len(out) > 50:
            out = out[:47] + "..."
        field(f"[{item.get('name')}]",
              f"status={item.get('status')} steps={item.get('steps')} "
              f"duration={item.get('duration_ms', 0):.2f}ms output={out!r}")


# --------------------------------------------------------------------------------------
# 装配：三个各司其职的 Agent
# --------------------------------------------------------------------------------------


def build_stage_agents(*, offline: bool, provider: str, model: str,
                       recorder: StageInputRecorder,
                       failing_writer: bool = False) -> dict[str, Agent]:
    """造"研究员 / 写作者 / 审校者"三个 Agent。

    离线模式下每个 Agent 的脚本（`ScriptedLLM` 的响应队列）就是它"脑子里的剧本"：

    * 研究员：先调 `search_notes` 工具拿资料，再给出结论（一次完整的 Thought-Action-Observation）；
    * 写作者：直接给一版草稿；
    * 审校者：给出审校意见。

    `failing_writer=True` 时写作者的模型"坏掉"（返回一条错误响应），用来演示失败分支。
    """

    def agent(name: str, description: str, tools: tuple[Any, ...], script: list[Any]) -> Agent:
        return Agent(
            llm=build_llm_for(script, offline=offline, provider=provider, model=model),
            tools=ToolRegistry(list(tools)),
            name=name,
            description=description,
            callbacks=[recorder],  # 记录"我真正收到的输入"
        )

    researcher = agent(
        "researcher", "查资料并给出事实要点", RESEARCHER_TOOLS,
        [
            ScriptedResponse.tool("search_notes", {"topic": "liteagent"}),
            ScriptedResponse.text("研究结论：liteagent 是零依赖的 Agent Harness，"
                                  "核心是 LLM 抽象层 / 工具系统 / 记忆层三层。"),
        ],
    )

    writer_script = (
        [ScriptedResponse.error(LLMError("writer model is down (示例里故意制造的故障)"))]
        if failing_writer
        else [ScriptedResponse.text("草稿 v1：liteagent 把 LLM、工具、记忆和 ReAct 循环粘在一起，"
                                    "多 Agent 编排是它的上层能力。")]
    )
    writer = agent("writer", "把要点写成一段草稿", WRITER_TOOLS, writer_script)

    reviewer = agent(
        "reviewer", "对草稿做风格与事实审校", REVIEWER_TOOLS,
        [ScriptedResponse.text("审校通过：结论前置、无事实错误，建议补一句 latency 数据。")],
    )
    return {"researcher": researcher, "writer": writer, "reviewer": reviewer}


def build_pipeline(*, steps: Sequence[SequentialStep],
                   config: TeamConfig | None = None,
                   blackboard: Blackboard | None = None,
                   recorder: TeamEventRecorder | None = None,
                   name: str = "content_pipeline") -> SequentialAgent:
    """把阶段装配成流水线。`steps` 由调用方给（这样每一节能演示不同的写法）。"""
    return SequentialAgent(
        list(steps),
        name=name,
        config=config,
        blackboard=blackboard if blackboard is not None else Blackboard(),
        callbacks=[recorder] if recorder is not None else None,
    )


# --------------------------------------------------------------------------------------
# 各节
# --------------------------------------------------------------------------------------


def section_basic(args: argparse.Namespace, offline: bool) -> None:
    """[2/6] 最简流水线：一个字都不写。"""
    title("[2/6] 最简流水线：不写任何 input_template")
    note("""
框架的缺省规则是：第 0 步吃原始输入 `{input}`，其余每一步吃上一步的输出 `{prev}`。
所以"研究员 -> 写作者 -> 审校者"这条链**不需要写任何模板**。
    """)

    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    team_recorder = TeamEventRecorder()
    blackboard = Blackboard()
    team = build_pipeline(
        steps=[SequentialStep(agent=agents[name])
               for name in ("researcher", "writer", "reviewer")],
        blackboard=blackboard, recorder=team_recorder,
    )

    result = team.run("介绍一下 liteagent 这个项目")

    field("status", result.status)
    field("agent_name", result.agent_name)
    field("steps (总步数)", result.steps)
    field("output", result.output)
    print()
    print("  每个阶段**真正收到**的输入（订阅 RUN_STARTED 事件拿到的）：")
    for name in ("researcher", "writer", "reviewer"):
        value = recorder.last(name).replace("\n", "\\n")
        print(f"    {name:<12}<- {value}")

    print()
    print("  委派 / 返回事件配对：")
    print(team_recorder.render())

    print_stage_summaries(result)
    print_entries(blackboard)


def section_template(args: argparse.Namespace, offline: bool) -> None:
    """[3/6] 模板：把 {input} / {prev} / {steps[x]} 串起来。"""
    title("[3/6] input_template：{input} / {prev} / {steps[阶段名]}")
    note("""
模板里能引用三个变量：`{input}` 原始用户输入、`{prev}` 上一步输出、
`{steps[阶段名]}` 任意**已经跑完**的阶段输出。

下面让写作者同时拿到"原始需求 + 研究结论"，审校者拿到"原始需求 + 草稿"。
`{steps[researcher]}` 里的名字就是 `SequentialStep.name`：框架在跑每一步之前，
会把**已经跑完**的阶段输出塞进渲染环境（还没跑的阶段不在里面）。
    """)

    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    blackboard = Blackboard()
    team = build_pipeline(blackboard=blackboard, steps=[
        SequentialStep(
            agent=agents["researcher"], name="researcher",
            input_template="请围绕这个主题查资料：{input}",
        ),
        SequentialStep(
            agent=agents["writer"], name="writer",
            input_template=(
                "原始需求：{input}\n"
                "研究员给的要点：{steps[researcher]}\n"
                "请基于以上要点写一段草稿。"
            ),
        ),
        SequentialStep(
            agent=agents["reviewer"], name="reviewer",
            input_template="原始需求：{input}\n待审草稿：{prev}\n请审校。",
        ),
    ])

    result = team.run("介绍一下 liteagent 这个项目")

    field("status", result.status)
    field("output", result.output)
    print()
    print("  每个阶段**真正收到**的输入：")
    for name in ("researcher", "writer", "reviewer"):
        print(f"    --- {name} ---")
        for line in recorder.last(name).split("\n"):
            print(f"      {line}")
    print()
    note("""
写作者收到的输入里，"研究员给的要点"已经被替换成了研究员的真实输出 ——
这就是 `{steps[researcher]}` 的效果：**跨阶段引用**而不是只能看上一步。
    """)
    print_entries(blackboard)

    subtitle("两种「缺失」的行为不一样（这是最容易踩的一个坑）")
    note("""
* 顶层变量缺失：`{nosuchvar}` -> `render_template` 用的 `_SafeDict` 把它**原样保留**。
  这是刻意的：用户自定义的 system prompt 里常有不认识的占位符，缺 key 直接炸会很难用。
* `{steps[未知阶段]}` 缺失：这是**对 `steps` 这个 dict 的一次元素查找**，
  `_SafeDict.__missing__` 管不到它 -> 抛 KeyError -> 被编排器包成 `ConfigError`
  （"模板渲染其它异常 -> ConfigError"是框架的冻结语义）。
  也就是说：**引用一个不存在的阶段是配置错误，会当场失败**，不会静默变成空串。
    """)
    field("{nosuchvar}", repr(render_template("值={nosuchvar}", {"input": "x"})))
    try:
        render_template("值={steps[ghost]}", {"input": "x", "steps": {"researcher": "r"}})
    except KeyError as exc:
        field("{steps[ghost]}", f"KeyError: {exc}（编排器会把它包成 ConfigError）")

    subtitle("真的跑一次：引用一个还没跑的阶段 -> ConfigError（早失败）")
    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    team = build_pipeline(steps=[
        SequentialStep(agent=agents["researcher"], name="researcher"),
        SequentialStep(agent=agents["writer"], name="writer",
                       input_template="参照还不存在的阶段：{steps[reviewer]}"),
    ])
    try:
        team.run("介绍一下 liteagent 这个项目")
    except ConfigError as exc:
        field("ConfigError", str(exc)[:100])


def section_blackboard(args: argparse.Namespace, offline: bool) -> None:
    """[4/6] 共享黑板：版本号与 output_key。"""
    title("[4/6] 共享黑板：版本号 + output_key")
    note("""
黑板是**团队共享**的（默认所有阶段共用一个 `Blackboard` 实例）。
每一次写入都会把该 key 的 version +1，并且留下 author / tags / 时间戳。

下面让"写作者"和"审校者"都写同一个 key `draft`（用 `output_key="draft"` 覆盖
默认的"以阶段名作 key"）：于是黑板上的 `draft` 会有 v1 和 v2 两个版本 ——
审校后的版本把草稿覆盖掉了，而历史仍在 `blackboard.history()` 里。
    """)

    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    blackboard = Blackboard()
    team = build_pipeline(blackboard=blackboard, steps=[
        SequentialStep(agent=agents["researcher"], name="researcher"),
        # 两个阶段都写 key="draft"：第二次写入 -> version 2
        SequentialStep(agent=agents["writer"], name="writer", output_key="draft"),
        SequentialStep(agent=agents["reviewer"], name="reviewer", output_key="draft"),
    ])

    result = team.run("介绍一下 liteagent 这个项目")
    field("status", result.status)

    print_entries(blackboard, label="黑板快照（注意 draft 的版本号）")
    subtitle("blackboard.keys() / 按标签过滤")
    field("keys()", blackboard.keys())
    field("keys(tags=('stage',))", blackboard.keys(tags=("stage",)))
    field("len(blackboard)", len(blackboard))

    subtitle("blackboard.history()（变更日志，最新在最后）")
    for entry in blackboard.history(limit=10):
        value = str(entry.value).replace("\n", "\\n")
        print(f"    v{entry.version} {entry.key:<12} by {entry.author!r} "
              f"-> {value[:48]!r}")

    subtitle("黑板读回来的是什么")
    note("""`blackboard.read('draft')` 拿到的是**最新版本**（审校后的那版）。
`read_entry('draft')` 则能拿到带版本号的完整记录。""")
    draft = blackboard.read("draft")
    field("read('draft')", str(draft)[:70] + ("..." if len(str(draft)) > 70 else ""))
    entry = blackboard.read_entry("draft")
    field("read_entry('draft')", f"version={entry.version} author={entry.author!r} tags={entry.tags}")
    field("metadata['blackboard']", "（result.metadata 里也有同样一份快照，CLI/trace 不必再读黑板）")


def section_failure(args: argparse.Namespace, offline: bool) -> None:
    """[5/6] propagate_failure：三种失败走法 + optional。"""
    title("[5/6] 失败处理：propagate_failure = return / raise / continue")
    note("""
让"写作者"这一步的模型坏掉（返回一条 LLM 错误响应），看三种策略的差别。

失败阶段的 `output` 只有一个公式（框架里冻结的写法）：
    output = result.output or previous_output or ""
也就是"失败阶段已有的半截输出"优先，它是空串才回退上一步的输出。
下面三种策略都用同一个失败点，所以你能直接对照。
    """)

    for mode in ("return", "raise", "continue"):
        subtitle(f"propagate_failure = {mode!r}")
        recorder = StageInputRecorder()
        agents = build_stage_agents(offline=offline, provider=args.provider,
                                    model=args.model, recorder=recorder,
                                    failing_writer=True)
        blackboard = Blackboard()
        team = build_pipeline(
            blackboard=blackboard,
            config=TeamConfig(propagate_failure=mode),
            steps=[SequentialStep(agent=agents["researcher"], name="researcher"),
                   SequentialStep(agent=agents["writer"], name="writer"),
                   SequentialStep(agent=agents["reviewer"], name="reviewer")],
        )
        try:
            result = team.run("介绍一下 liteagent 这个项目")
        except DelegationError as exc:  # 只有 "raise" 会走到这里
            field("抛出的异常", f"{type(exc).__name__}: {str(exc)[:80]}")
            field("exc.from_agent -> to_agent", f"{exc.from_agent!r} -> {exc.to_agent!r}")
            field("exc.context", exc.context if hasattr(exc, "context") else None)
            note("""
抛出的是 `DelegationError`（编排层的异常），`cause` 才是子 Agent 的原始错误
（这里是一条 `LLMError`）。编排层的失败信息比"某个 LLM 报错了"更有用：它告诉你
**是哪一步**、在**第几个**位置失败的。""")
            print_entries(blackboard, label="黑板（研究员已经写进去了）")
            continue

        field("status", result.status)
        field("output", str(result.output)[:80])
        field("error", f"{type(result.error).__name__}: {result.error}" if result.error else None)
        if "failed_stage" in result.metadata:
            field("metadata['failed_stage']", result.metadata["failed_stage"])
            note("""
`return` 模式下 `metadata` 里只有**已完成**阶段的摘要，失败阶段由
`metadata['failed_stage']` 单独表达 —— 同一件事写两遍只会互相矛盾。""")
        print_stage_summaries(result)
        print_entries(blackboard, label="黑板")

    subtitle("optional=True：无视 propagate_failure，一律继续")
    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder, failing_writer=True)
    blackboard = Blackboard()
    team = build_pipeline(
        blackboard=blackboard,
        # 故意把全局策略设成最容易炸的 "raise"：optional 阶段照样不会中止流水线。
        config=TeamConfig(propagate_failure="raise"),
        steps=[SequentialStep(agent=agents["researcher"], name="researcher"),
               # optional=True：这一步失败只是被记下来，流水线继续往下走
               SequentialStep(agent=agents["writer"], name="writer", optional=True),
               SequentialStep(agent=agents["reviewer"], name="reviewer")],
    )
    result = team.run("介绍一下 liteagent 这个项目")
    field("status", result.status)
    field("output", str(result.output)[:80])
    print_stage_summaries(result)

    subtitle("失败阶段的 output 公式：两条分支各跑一次")
    note("""
公式是 `output = result.output or previous_output or ""`。上面三条演示走的都是
"失败阶段没有半截输出 -> 回退上一步"（所以 return 模式的 output 是研究员的结论）。

第一条分支（**失败阶段有半截输出**）需要一个"跑了一半才失败"的阶段。下面的做法是
把一整条**子流水线**当成一个阶段：子流水线自己的 `propagate_failure="return"` 会把
"草稿已产出、润色失败"编码成 `output=<那半截草稿>` + `status=FAILED`，
外层于是拿到了一个"有部分输出的失败阶段"。
    """)
    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    broken = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder, failing_writer=True)
    # 子流水线：第一步正常产出半截草稿，第二步（润色）的模型坏掉。
    # 子流水线自己的 propagate_failure 是默认的 "return" -> 它返回
    # status=FAILED 但 output="半截草稿" —— 这就是"有部分输出的失败阶段"。
    inner = build_pipeline(name="draft_team", steps=[
        SequentialStep(agent=agents["writer"], name="drafter"),
        SequentialStep(agent=broken["writer"], name="polisher"),
    ])
    outer = build_pipeline(name="outer_pipeline", steps=[
        SequentialStep(agent=agents["researcher"], name="researcher"),
        SequentialStep(agent=inner, name="draft_team"),
    ])
    result = outer.run("介绍一下 liteagent 这个项目")
    field("status", result.status)
    field("output", str(result.output)[:80])
    note("""
output 是子流水线留下的那半截草稿，**不是**上一步研究员的结论 ——
这就是公式里 `result.output` 优先于 `previous_output` 的效果：
模型常常在失败前已经打完半截答案，那半截往往比上一步的输出更贴近用户想要的东西。
    """)


def section_extras(args: argparse.Namespace, offline: bool) -> None:
    """[6/6] 输出截断 + 模板渲染失败的报错。"""
    title("[6/6] 两个边角：max_chars 截断 与 模板写错的报错")

    subtitle("max_chars：给下游上下文的硬预算（头 70% + 尾 30%）")
    note("""
`SequentialStep.max_chars > 0` 时，这一步的输出会先被截断成"头 70% + 尾 30%"，
再传给下一步、写进黑板（截断标记里带着被丢掉多少字符）。
为什么头尾都留：子 Agent 的结论常出现在开头，最近的进展/错误常出现在结尾，
只留头会把"为什么失败"整段切掉。

下面把研究员这一步的 max_chars 设成 40，你可以直接看到写作者收到的是一段
带 `...[truncated N chars]...` 的短文本。

注意一个容易误判的细节：截断后的版本会同时进**黑板**和 `metadata['steps']`，
而未截断的原文只存在于那一步**局部的** `AgentResult` 里 —— 它不会被流水线返回。
所以 `max_chars` 的语义是"给下游的上下文预算"，不是"某种无损压缩"。
    """)
    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    blackboard = Blackboard()
    team = build_pipeline(blackboard=blackboard, steps=[
        SequentialStep(agent=agents["researcher"], name="researcher", max_chars=40),
        SequentialStep(agent=agents["writer"], name="writer"),
        SequentialStep(agent=agents["reviewer"], name="reviewer"),
    ])
    result = team.run("介绍一下 liteagent 这个项目")
    field("writer 收到的输入", f"{recorder.last('writer')!r}")
    field("黑板里的 researcher", f"{blackboard.read('researcher')!r}")
    field("status", result.status)

    subtitle("模板写错 = 配置级错误（ConfigError，不是运行期失败）")
    note("""
模板里的非法格式串（例如 `{prev!q}` 这种没有意义的转换）会被包成 `ConfigError`，
并在消息里带上阶段名与模板原文。**早失败**比"跑了一半才炸"便宜得多。
    """)
    recorder = StageInputRecorder()
    agents = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    team = build_pipeline(steps=[
        SequentialStep(agent=agents["researcher"], name="researcher", input_template="{prev!q}"),
    ])
    try:
        team.run("随便什么输入")
    except ConfigError as exc:
        field("ConfigError", str(exc))
        field("context", exc.context if hasattr(exc, "context") else None)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


SECTION_RUNNERS = {
    "basic": section_basic,
    "template": section_template,
    "blackboard": section_blackboard,
    "failure": section_failure,
    "extras": section_extras,
}


def main(argv: Sequence[str] | None = None) -> int:
    """入口。返回退出码（0 = 成功），不调用 sys.exit，便于被别的脚本 import。"""
    args = parse_args(argv)
    # `--offline` 或 `--provider echo` 都走离线路径：echo 是离线占位 provider，
    # 用真实模型才有意义的示例在离线路径上本来就是"演示用脚本"。
    offline = bool(args.offline) or args.provider == "echo"
    if not offline and args.provider != "echo":
        # 真实 provider 需要 API key（这里不 print 任何 key 内容）。本机没有 key，
        # 所以这条路径**没有被实测过** —— 示例里如实说明，而不是假装它能跑。
        has_key = any(os.environ.get(name) for name in
                      ("LITEAGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                       "ANTHROPIC_API_KEY"))
        if not has_key:
            print(f"[!] provider={args.provider!r} 需要 API key；本机没有检测到。")
            print("    离线演示请用：--offline（或 --provider echo）")
            return 2

    title("liteagent 示例 05：Sequential 多 Agent 流水线")
    field("模式", "offline（ScriptedLLM 脚本驱动）" if offline
          else f"live（provider={args.provider}, model={args.model or DEFAULT_MODELS.get(args.provider, '?')}）")
    field("节", args.section)

    # [1/6] 先把"演员表"亮出来：名字、职责、工具（工具描述会自动进 JSON Schema，
    # 而模型选工具几乎只看 name + description）。
    title("[1/6] 演员表：三个各司其职的 Agent")
    recorder = StageInputRecorder()
    actors = build_stage_agents(offline=offline, provider=args.provider, model=args.model,
                                recorder=recorder)
    for name, agent in actors.items():
        field(name, f"{agent.description}；工具={agent.tools.names()}")
    print()
    print("  工具的 JSON Schema 是**从函数签名和 docstring 自动生成**的（装饰器 + 反射）：")
    schema = RESEARCHER_TOOLS[0].spec.parameters  # type: ignore[attr-defined]
    for line in str(schema).split("\n"):
        print(f"    {line}")

    sections = ("basic", "template", "blackboard", "failure", "extras") \
        if args.section == "all" else (args.section,)
    for name in sections:
        SECTION_RUNNERS[name](args, offline)

    title("小结（面试可以这么讲）")
    note("""
1. 输入靠模板而不是约定：`{input}` / `{prev}` / `{steps[x]}`，缺省规则让最简单的
   三步流水线零配置；模板用 _SafeDict 渲染，**顶层**缺 key 保留字面量，
   而 `{steps[未知阶段]}` 会当场抛 ConfigError（配置错误早失败）。
2. 每个阶段新建自己的 AgentState，只通过黑板 + scratchpad['delegation'] 与外界交互。
3. 黑板是共享的、带版本号的；`output_key` 允许两个阶段写同一个 key（v1 -> v2）。
4. 失败有三种走法，`optional=True` 无视全局策略；失败时的 output 有唯一公式
   （半截输出 > 上一步输出 > 空串）。
5. 每次委派都会发配对的 agent_delegate / agent_return 事件，即使随后要 raise。
    """)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
