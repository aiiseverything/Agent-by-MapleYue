from __future__ import annotations

# =============================================================================
# 03_react_text_mode.py —— 文本 ReAct 模式：完整 Thought/Action/Observation 轨迹
# =============================================================================
#
# 一个 Agent 循环有两种"模型怎么表达要调工具"的方式，本框架两种都支持：
#
#   ┌──────────────┬─────────────────────────────┬─────────────────────────────┐
#   │              │ native（原生 function call） │ text（文本 ReAct）           │
#   ├──────────────┼─────────────────────────────┼─────────────────────────────┤
#   │ 工具怎么暴露  │ 请求里的 `tools=[schema...]` │ 渲染进 system prompt 的 {tools}│
#   │ 模型怎么输出  │ 结构化字段 tool_calls        │ 一段文本 Thought/Action/...  │
#   │ 怎么解析      │ 直接取字段                   │ ReActParser.parse()          │
#   │ 观察怎么回灌  │ role="tool" + tool_call_id   │ role="user" 的 "Observation:"│
#   │ 一轮几个工具  │ 天然支持并发多个             │ **一轮一个 Action**          │
#   │ 何时结束      │ 无 tool_calls 且 stop        │ 出现 "Final Answer:"         │
#   └──────────────┴─────────────────────────────┴─────────────────────────────┘
#
# 01_quickstart.py 走的是 native。本文件专讲 **text 模式**，两个原因：
#
#   1. 它是"没有 function calling 能力的模型"的唯一出路（本地小模型、老模型、
#      自建推理服务、纯文本 completion 接口）。框架用 `LLMClient.supports_tool_calling`
#      这一个布尔量把两条路统一起来：mode="auto" 时自动选，选不到 native 就退 text。
#   2. 它的中间过程是**纯文本**的 —— 每一步的思考、动作、观察都看得见。
#      排查"模型为什么绕圈子"时，这比一堆结构化字段直观得多。
#
# 本文件演两件事：
#   §A 完整多步轨迹：3 步推理（5 行代码的剧本），把每一步的
#      Thought / Action / Action Input / Observation / Final Answer 打成轨迹。
#   §B parse-error 自纠正：模型输出格式错了 -> 解析失败 -> 框架把"格式说明书"
#      连同错误原因回灌给模型 -> 模型改对 -> 循环继续。这是面试很好讲的鲁棒性设计。
#
# 运行方式（离线，不联网、不需要 API key）：
#     python3 examples/03_react_text_mode.py --offline

import argparse
import asyncio
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# ---- 让 examples/ 下的脚本"直接 python3 就能跑"（详见 01_quickstart.py 的说明）----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent import (  # noqa: E402 - 必须在 sys.path 引导之后
    Agent,
    AgentConfig,
    EventType,
    ScriptedLLM,
    ScriptedResponse,
    ToolRegistry,
    get_llm,
    tool,
)
from liteagent.agent.callbacks import as_llm_callback  # noqa: E402
from liteagent.agent.parser import ReActParser  # noqa: E402
from liteagent.errors import ReActParseError  # noqa: E402

BAR = "=" * 74
LINE = "-" * 74


def section(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR)


# =============================================================================
# 工具：故意做成"一步算不完"的样子，才有多步推理可看
# =============================================================================


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product.

    Args:
        a: The first factor.
        b: The second factor.
    """
    return a * b


@tool
def power(base: int, exponent: int) -> int:
    """Raise base to exponent and return the result.

    Args:
        base: The base number.
        exponent: A non-negative exponent.
    """
    if exponent < 0:
        # 返回值就是 Observation：工具"业务上失败"时把错误**当成字符串返回**，
        # 让模型看到并自行纠正；只有基础设施故障才走 ok=False 的失败路径。
        return f"ERROR: negative exponent {exponent} is not supported"
    return base ** exponent


#: 第三个工具（power）在本示例里**不会被调用** —— 它存在的意义是让 system prompt
#: 里的工具清单有三个，更接近真实场景；同时也说明"模型能选的工具越多，
#: 文本模式下它对 Action 名字的选择就越容易出错"，这正是 §B 要处理的问题。
REGISTRY = ToolRegistry([add, multiply, power])

#: 问题本身。注意它**不能一步算完**：必须先算 12+30，再把结果乘 2。
QUESTION = "What is (12 + 30) * 2 ? Please compute it step by step."


def build_text_agent(llm: Any, *, max_steps: int = 6, name: str = "react-text") -> Agent:
    """造一个**强制文本模式**的 Agent。

    `AgentConfig(mode="text")` 是这里唯一的开关。它在内部做两件事：
      * system prompt 渲染时，把 `ToolRegistry.to_prompt()` 的结果填进 {tools} 占位符
        —— 文本模式没有 `tools=` 参数，模型只能从 prompt 里知道有哪些工具；
      * 拿到模型输出后走 `ReActParser.parse()` 而不是读 `resp.tool_calls`。

    除此之外的状态机、事件、重复检测、自纠正、usage 统计**两条路完全共用** ——
    这也是为什么本示例不用重讲"循环怎么转"，只需要讲"文本长什么样"。
    """
    return Agent(
        llm=llm,
        tools=REGISTRY,
        config=AgentConfig(mode="text", max_steps=max_steps),
        name=name,
    )


# =============================================================================
# §A 一段完整的多步轨迹
# =============================================================================
# 离线剧本用 `ScriptedResponse.react(...)` 写。它不是"随便一段字符串"：
# 它渲染出的文本**与解析器的语法严格对齐**，所以离线跑通的东西，换成真模型
# 只要能吐出同样格式的文本，行为就完全一致。
#
#   ScriptedResponse.react("思考", action="add", action_input={"a": 12, "b": 30})
#     -> 'Thought: 思考\nAction: add\nAction Input: {"a": 12, "b": 30}'
#
#   ScriptedResponse.react("思考", final="84")
#     -> 'Thought: 思考\nFinal Answer: 84'
#
# 一个细节：`action` 与 `final` 同时给时 **action 优先**。这不是任意选择，
# 而是与解析器的规则保持一致（§9.3 步骤 4：两者同时出现时执行 Action 且不终结）。
# 夹具与解析器对同一段文本的理解必须一样，否则测试会自相矛盾。


def build_reasoning_script() -> list[ScriptedResponse]:
    """3 步推理的剧本：算加法 -> 算乘法 -> 给最终答案。"""
    return [
        # 第 1 步：拆解问题，先算括号。
        ScriptedResponse.react(
            "I need to compute the parentheses first: 12 + 30.",
            action="add",
            action_input={"a": 12, "b": 30},
        ),
        # 第 2 步：**模型看到的 Observation 是上一步的真实结果 42**，
        # 所以它这里能直接引用 42 —— 这就是"观察回灌"的意义。
        ScriptedResponse.react(
            "12 + 30 = 42. Now multiply by 2.",
            action="multiply",
            action_input={"a": 42, "b": 2},
        ),
        # 第 3 步：给最终答案。文本模式的终结条件就是出现 "Final Answer:"。
        ScriptedResponse.react(
            "42 * 2 = 84. I have the final answer.",
            final="84",
        ),
    ]


def render_trajectory(agent: Agent) -> None:
    """把 `agent.state.messages` 渲染成"人能读的 ReAct 轨迹"。

    state.messages 是**这一轮真正在模型上下文里的对话**（不含 memory 拼出的 system 段）。
    文本模式的轨迹长这样，四种角色交替出现：

        assistant: Thought/Action/Action Input   <- 模型说的话
        user:      Observation: ...              <- 框架回灌的工具结果
        assistant: Thought/Final Answer          <- 模型收尾

    这里用 `ReActParser` 去解析 assistant 消息，而不是把原始字符串直接打出来 ——
    因为"模型写的文本"和"框架理解到的结构"**可能不一致**，把两者并排展示，
    读者才能看出解析器到底认了什么、忽略了什么。

    （还有一条消息 kind="react"：框架会把 Thought 单独存一条，方便把
      "推理过程"与"动作"分开检索/裁剪。它不参与循环控制。）
    """
    parser = ReActParser(tool_names=tuple(REGISTRY.names()))
    step = 0
    for message in agent.state.messages:
        role = message.role.value
        kind = message.metadata.get("kind") or "-"

        if role == "assistant" and kind == "react":
            # 框架把 Thought 单独存了一条（kind="react"）：它**不参与循环控制**，
            # 只是为了让"推理过程"与"工具动作"能分开检索/裁剪（长对话里最该先丢的就是它）。
            # 因此它不会出现在下一轮发给模型的上下文里，这里也只做展示。
            print(f"  [存档] Thought (kind=react, 不参与循环): {message.content!r}")
        elif role == "assistant" and message.content:
            try:
                parsed = parser.parse(message.content)
            except ReActParseError:
                print(f"  assistant  (raw)      : {message.content!r}")
                continue
            step += 1
            print(f"  --- step {step} ---")
            if parsed.thought:
                print(f"  Thought               : {parsed.thought}")
            if parsed.is_action():
                print(f"  Action                : {parsed.action}")
                print(f"  Action Input          : {_dumps(parsed.action_input)}")
            elif parsed.is_final():
                print(f"  Final Answer          : {parsed.final_answer}")
        elif role == "user" and kind == "observation":
            # 观察是**框架**写进去的，不是模型写的。区分这一点很重要：
            # 模型无法伪造 Observation —— 它写什么，下一轮由框架盖过去。
            print(f"  Observation (由框架回灌): {message.content}")
        elif role == "user" and kind == "nudge":
            print("  Nudge (框架注入的纠正提示):")
            for raw_line in message.content.splitlines():
                print(f"      | {raw_line}")
        elif message.tool_calls:
            for call in message.tool_calls:
                print(f"  assistant -> tool_call : {call.name}({_dumps(call.arguments)})")
        else:
            print(f"  {role:<21} : {message.content!r}   (kind={kind})")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def demo_full_trajectory() -> ScriptedLLM:
    """§A 主流程：造剧本 -> 跑 -> 打印轨迹与结果。"""
    section("§A  完整多步轨迹：模型的每一步都看得见")
    print("问题：", QUESTION)
    print()
    print("离线剧本（ScriptedResponse.react 渲染出的模型输出）：")
    script = build_reasoning_script()
    for index, response in enumerate(script, start=1):
        print(f"  [{index}] {response.content!r}")

    llm = ScriptedLLM(script, model="scripted-1")
    agent = build_text_agent(llm)

    # 把 LLM 层的低层事件也接进 agent 的回调管理器：
    # 这样我们能用同一份 trace 看到 tool_started / tool_finished / thought ...
    events: list[Any] = []
    agent.callbacks.subscribe(events.append)
    llm.on_event = as_llm_callback(agent.callbacks)

    print()
    print("system prompt 里的工具清单（文本模式靠它告诉模型有哪些工具）：")
    for raw_line in REGISTRY.to_prompt(fmt="text").splitlines():
        print(f"  | {raw_line}")

    result = asyncio.run(agent.arun(QUESTION))

    print()
    print(LINE)
    print("完整轨迹（agent.state.messages 渲染）")
    print(LINE)
    render_trajectory(agent)

    print()
    print(LINE)
    print("结果")
    print(LINE)
    print(f"  status          : {result.status.value}")
    print(f"  output          : {result.output!r}")
    print(f"  steps           : {result.steps}   (LLM 被调用 {llm.call_count} 次)")
    print(f"  usage           : {result.usage.to_dict()}")
    print(f"  parse_errors    : {agent.state.parse_errors}   (本轮解析失败 0 次)")
    print(f"  tool_calls 轨迹 : "
          f"{[c.name + '(' + _dumps(c.arguments) + ')' for c in result.tool_calls]}")

    # 离线剧本自检：3 条预设响应必须被恰好消费完（多一次会抛 ScriptedExhaustedError，
    # 少一次这里会失败）。这是"3 步推理"这个结论的机器证明，而不是肉眼数出来的。
    llm.assert_exhausted()
    print()
    print(f"剧本自检：{llm.call_count} 次 LLM 调用恰好消费完 {llm.call_count} 条预设响应")

    print()
    print("事件流（框架在每一步发出来的 trace 事件，节选）：")
    print("  说明：tool_started / tool_finished 的**唯一发射者**是 ToolExecutor（执行器），")
    print("  它并不知道 Agent 的循环步号，所以这两个事件的 step=0；")
    print("  循环步号只出现在 Agent 层事件（step_started / thought / action_parsed）上。")
    print("  这条归属是规范冻结的：一件事只有一个出处，避免同一条事件被发两遍。")
    interesting = {
        EventType.RUN_STARTED, EventType.STEP_STARTED, EventType.THOUGHT,
        EventType.ACTION_PARSED, EventType.TOOL_FINISHED, EventType.STEP_FINISHED,
        EventType.RUN_FINISHED,
    }
    for event in events:
        if event.type in interesting:
            print(f"  step={event.step:<2} {event.type.value:<16} {_dumps(event.data)}")
    return llm


# =============================================================================
# §B parse-error 自纠正：模型格式错了怎么办
# =============================================================================
# 这是文本模式**独有**的失败模式：模型说的话得能被解析出结构，否则循环就卡住了。
# 真实模型（尤其是小模型）经常这样翻车：
#
#     "I think 12+30 is 42, so the answer should be 84."     <- 一句人话，没有 Action
#
# 框架的应对**不是抛异常终止**，而是把它当成一次可恢复的对话：
#
#   1. 解析失败 -> 记一次 parse_errors，发一条 PARSE_ERROR 事件（带 reason/offset/raw_len）；
#   2. 把**错误的原文**（已由正常流程写入 messages）保留在上下文里，
#      再注入一条 kind="nudge" 的 user 消息，内容 = "你的输出没法解析：<原因>。
#      请严格用这个格式：Thought / Action(从 [add, multiply] 里选) / Action Input ..."；
#   3. **消耗一个 step**，但不重试 LLM 调用本身 —— 否则最坏情况是
#      max_steps × max_parse_retries 次真实请求，成本会炸；
#   4. 模型下一轮看到这条反馈，按格式重写 -> 循环继续。
#
# 面试可以这么讲："格式错误不是异常，是可观测、可恢复的对话状态。"
# 可断言的不变量：一次 parse error 恰好新增 2 条消息（1 条 assistant 原文 + 1 条 nudge），
# 因为 assistant 原文只有一个写入点 —— 早期版本在这里又写了一遍，导致同一条内容
# 在上下文里出现两次（既费 token，又让模型以为自己在复读）。


def demo_parse_error_recovery(args: argparse.Namespace) -> None:
    section("§B  parse-error 自纠正：模型格式错了，循环照样转回来")
    print("第 1 步的模型输出（故意写错格式 —— 一句人话，没有 Action/Final Answer）：")
    bad_output = "Hmm, 12 + 30 is 42, so the answer should be 84. That seems right."
    print(f"  {bad_output!r}")
    print("这既不是 Action 也不是 Final Answer，`ReActParser.parse` 会抛 ReActParseError。")

    if args.provider is not None:
        # 联网模式下没法保证真实模型一定先给出"错误格式"，所以只跑离线剧本。
        print()
        print("（--provider 模式下跳过：真实模型不保证按剧本先输出错误格式，")
        print("  这一段的自纠正演示只在离线剧本下确定可复现。）")
        return

    llm = ScriptedLLM(
        [
            # 第 1 次调用：格式错误。
            ScriptedResponse.text(bad_output),
            # 第 2 次调用：模型看懂了 nudge 里的格式说明，改对格式直接给答案。
            ScriptedResponse.react(
                "Sorry, let me use the proper format.", final="84"
            ),
        ],
        model="scripted-1",
    )
    agent = build_text_agent(llm, max_steps=4, name="react-parse-recovery")

    events: list[Any] = []
    agent.callbacks.subscribe(events.append)
    llm.on_event = as_llm_callback(agent.callbacks)

    # nudge 的 role 是 user（文本模式的 Observation 也是），但它**不是用户说的话**。
    # 记忆层会拒绝对它做长期写入，并用一条 RuntimeWarning 留痕（降级必须可观测）。
    # 这里把 warning 抓下来放到输出里展示，而不是让它悄悄打到 stderr 去。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(agent.arun(QUESTION))
    warnings_seen = [f"{w.category.__name__}: {w.message}" for w in caught]

    print()
    print(LINE)
    print("恢复过程（agent.state.messages 渲染）")
    print(LINE)
    render_trajectory(agent)

    print()
    print(LINE)
    print("结果与不变量")
    print(LINE)
    print(f"  status        : {result.status.value}    （没有失败，循环恢复了）")
    print(f"  output        : {result.output!r}")
    print(f"  steps         : {result.steps}  (LLM 调用 {llm.call_count} 次)")
    print(f"  parse_errors  : {agent.state.parse_errors}")

    parse_events = [e for e in events if e.type is EventType.PARSE_ERROR]
    print(f"  PARSE_ERROR 事件 : {len(parse_events)} 条")
    for event in parse_events:
        print(f"      data = {_dumps(event.data)}")

    bad_echoes = sum(1 for m in agent.state.messages if m.content == bad_output)
    nudges = [m for m in agent.state.messages if m.metadata.get("kind") == "nudge"]
    print()
    print("  可断言的不变量：")
    print(f"    1) 错误原文在上下文里恰好出现 {bad_echoes} 次  "
          "（不是 2 次 —— assistant 原文只有一个写入点）")
    print(f"    2) 注入了 {len(nudges)} 条 nudge 反馈，"
          f"其中含 'could not be parsed' = "
          f"{bool(nudges) and 'could not be parsed' in nudges[0].content}")
    print("    3) 一次 parse error 的消息增量恰好是 2（1 条 assistant + 1 条 nudge）")
    print(f"    4) parse error 消耗 step 但不重试 LLM："
          f"steps={result.steps} == LLM 调用次数 {llm.call_count}")

    if nudges:
        print()
        print("  回灌给模型的完整反馈原文（这就是模型下一轮看到的东西）：")
        for raw_line in nudges[0].content.splitlines():
            print(f"      | {raw_line}")

    if warnings_seen:
        print()
        print("  运行期框架留下的 warning（不是错误，是「我做了降级」的留痕）：")
        for text in warnings_seen:
            print(f"      ! {text}")
        print("      含义：nudge 虽然 role=user，但它是框架生成的，不能被当成"
              "\"用户事实\"写进长期记忆，")
        print("      否则下一轮它会被当成既定事实召回，把模型的错误格式固化下来。")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="liteagent 示例 03：文本 ReAct 模式与 parse-error 自纠正（默认离线）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线运行（默认；ScriptedLLM 驱动，不联网、不需要 API key）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="改用真实 provider 跑 §A（如 openai）；§B 的自纠正演示只在离线剧本下复现",
    )
    parser.add_argument("--model", default=None, help="配合 --provider 指定模型名")
    args = parser.parse_args(argv)
    if args.provider is None:
        args.offline = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    print(BAR)
    print("liteagent 示例 03 —— 文本 ReAct 模式（Thought / Action / Observation / Final Answer）")
    print(f"运行模式：{'离线（--offline）' if args.offline else '联网 ' + str(args.provider)}")
    print(BAR)

    if args.provider is not None:
        # 联网路径：真模型 + 真 ReAct 提示词。**未验证** —— 本环境无外网，
        # 这段代码只保证"装配得上"，不保证模型一定会按格式输出（那正是 §B 要处理的情况）。
        try:
            llm: Any = get_llm(args.provider if args.model is None
                               else f"{args.provider}:{args.model}")
        except Exception as exc:
            print(f"装配失败：{type(exc).__name__}: {exc}")
            return 2
        agent = build_text_agent(llm, name="react-live")
        result = asyncio.run(agent.arun(QUESTION))
        print(LINE)
        print("轨迹")
        print(LINE)
        render_trajectory(agent)
        print()
        print(f"status = {result.status.value}  output = {result.output!r}  "
              f"steps = {result.steps}")
        return 0 if result.ok else 1

    demo_full_trajectory()
    demo_parse_error_recovery(args)

    print()
    print(BAR)
    print("小结：文本模式与原生模式共用同一套 ReAct 状态机，差异只在两处 ——")
    print("  * `ToolRegistry.to_prompt()` 把工具清单渲染进 system prompt（而不是 tools= 参数）")
    print("  * `ReActParser.parse()` 把模型的一句话解析成 list[ToolCall]")
    print("  排查线上问题时，`agent.state.messages` 就是完整的现场。")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
