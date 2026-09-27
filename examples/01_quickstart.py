from __future__ import annotations

# =============================================================================
# 01_quickstart.py —— 最小可运行的 liteagent Agent
# =============================================================================
#
# 这个文件只回答一个问题：**"用 liteagent 跑一个 Agent，最少要写几行？"**
#
# 答案是 4 行（见文件末尾打印出来的"最小用法"）：
#
#     llm    = ScriptedLLM([...])                  # 1. 一个模型（离线用脚本模型）
#     tools  = ToolRegistry([add, word_count])     # 2. 一组工具
#     agent  = Agent(llm=llm, tools=tools)         # 3. 粘起来
#     result = await agent.arun("2 + 40 = ?")      # 4. 跑一轮 ReAct 循环
#
# 剩下的全是"为了让这段代码能跑、能看懂、能排查"的脚手架：
#   * sys.path 引导（让 examples/ 里的文件不装包也能 import liteagent）；
#   * argparse 的 --offline 开关（离线跑，不需要任何 API key）；
#   * 把 result 的每个字段打印成人能读的样子。
#
# 阅读顺序建议：先跳到 main()，再回头看 _build_offline_llm()。
#
# 运行方式（任选其一，都不联网、不需要 key）：
#     python3 examples/01_quickstart.py            # 默认即离线
#     python3 examples/01_quickstart.py --offline  # 显式声明离线
# 想打真实模型（需要 OPENAI_API_KEY）：
#     python3 examples/01_quickstart.py --provider openai

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# 第 0 步：让这个脚本"从任意目录直接 python3 也能跑"
# -----------------------------------------------------------------------------
# 为什么需要这 3 行：`python3 examples/01_quickstart.py` 时，Python 把
# sys.path[0] 设成**脚本所在目录**（examples/），而不是你敲命令时所在的目录。
# 于是仓库根目录不在搜索路径里，`import liteagent` 会直接 ModuleNotFoundError。
#
# 三种常见替代方案，以及本项目为什么不选：
#   1) `pip install -e .` —— 最正统，但要求本机可安装（本环境不允许 pip install）。
#   2) 让用户自己设 PYTHONPATH —— 把"能跑"的责任推给读者，examples 应当开箱即跑。
#   3) 做成 test 里才用的相对 import —— 脚本直接执行时 __package__ 是 None，会崩。
# 所以：显式把仓库根插到 sys.path 最前面。只在"确实不在路径里"时才插，
# 避免重复执行（run_all_examples.py 会 import 本模块）时把 sys.path 撑成一长串。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent import (  # noqa: E402 - 必须在上面的 sys.path 引导之后
    Agent,
    AgentConfig,
    ScriptedLLM,
    ScriptedResponse,
    ToolRegistry,
    get_llm,
    tool,
)


# -----------------------------------------------------------------------------
# 第 1 步：写工具 —— 一个 @tool 装饰器就够了
# -----------------------------------------------------------------------------
# 关键点：**没有 JSON Schema 要手写**。@tool 会反射函数的
#   * 类型注解  -> parameters.properties 的 type / items / enum ...
#   * docstring -> 函数的 description，以及 Args 段里每个参数的 description
#   * 无默认值的参数 -> required 列表
# 这份 schema 就是"模型看到的工具说明书"，也是执行前校验入参的依据。
#
# 注意 docstring 的写法：用 Google 风格的 `Args:` 段，框架会把它拆成逐参描述。
# 参数形态的完整展示（Optional / Literal / list[str] / 嵌套 dataclass）
# 在 02_tools_custom.py，这里刻意只留最简单的两种。


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


@tool
def word_count(text: str) -> int:
    """Count how many whitespace-separated words a text contains.

    Args:
        text: The text to count words in.
    """
    return len(text.split())


#: 工具集合。ToolRegistry 是"给模型看的工具清单"的唯一载体：
#: 它负责注册、重名检查、别名、按名取用，以及导出成各家 provider 的 schema 格式。
#: **这里刻意不用全局默认注册表**：显式持有自己的 registry，两个 Agent 之间就不会
#: 通过进程级全局状态互相污染（这也是测试里 K 个用例能并行跑的前提）。
TOOLS = ToolRegistry([add, word_count])


# -----------------------------------------------------------------------------
# 第 2 步：造 LLM —— 离线用 ScriptedLLM，联网用 get_llm()
# -----------------------------------------------------------------------------
def _build_offline_llm() -> ScriptedLLM:
    """离线模型：**按预设脚本**逐条吐响应，零网络、零随机、可断言。

    这是本框架可测试性的地基：ReAct 循环的正确性不该依赖"模型这次心情好不好"。
    ScriptedLLM 每次 achat 消费队列里的下一条；队列格式与真实 provider 的
    LLMResponse 完全一致，所以"离线跑通"与"联网跑通"走的是同一套循环代码。

    本示例给的两条剧本：
      第 1 次调用 -> 要求调用工具 add(2, 40)   （finish_reason 自动推导为 "tool_calls"）
      第 2 次调用 -> 一句纯文本作为最终答案     （finish_reason 自动推导为 "stop"）
    """
    return ScriptedLLM(
        [
            # ScriptedResponse.tool(...) 等价于"模型回了 content 为空 + 一个 tool_call"。
            # call_id 留空即可：真正的 id 由 ScriptedLLM 在消费时分配（call_0 / call_1 ...），
            # 因为纯数据类拿不到递增序号。
            ScriptedResponse.tool("add", {"a": 2, "b": 40}),
            ScriptedResponse.text("The answer is 42."),
        ],
        model="scripted-1",
    )


def _build_live_llm(provider: str, model: str | None) -> Any:
    """联网模型：`get_llm("openai:gpt-4o-mini")` 这种一行式装配。

    spec 的语法是 `provider[:model][@base_url]`，例如：
        openai
        openai:gpt-4o-mini
        openai-compatible:qwen@http://localhost:8000/v1

    **未验证声明**：本示例的开发环境没有外网，这条分支**没有真实跑过**。
    代码只走到构造 client 为止，真实请求需要 OPENAI_API_KEY 与出网能力。
    缺 key 时 build_llm 会抛 LLMAuthError（不是裸的 KeyError），下面会转成
    一句人话提示再退出。
    """
    spec = f"{provider}:{model}" if model else provider
    return get_llm(spec)


# -----------------------------------------------------------------------------
# 第 3 步：跑一轮 ReAct 循环
# -----------------------------------------------------------------------------
def _render_result(result: Any) -> None:
    """把 AgentResult 打成人能读的样子。

    AgentResult 是 liteagent 对外的**唯一**结果契约（不抛异常，失败也编码在对象里）：
        status / output / error / steps / tool_calls / tool_results / usage / duration_ms
    注意 `ok` 是 **property 不是方法**（`result.ok`，不是 `result.ok()`）。
    """
    line = "-" * 74
    print(line)
    print("运行结果")
    print(line)
    print(f"  状态 status      : {result.status.value}   (result.ok = {result.ok})")
    print(f"  最终输出 output  : {result.output!r}")
    print(f"  步数 steps       : {result.steps}")
    print(f"  运行耗时         : {result.duration_ms:.1f} ms")
    print(f"  Token 用量 usage : prompt={result.usage.prompt_tokens}, "
          f"completion={result.usage.completion_tokens}, "
          f"total={result.usage.total_tokens}")

    if result.tool_calls:
        print("  工具调用轨迹     :")
        for index, (call, res) in enumerate(
            zip(result.tool_calls, result.tool_results), start=1
        ):
            print(f"    [{index}] {call.name}({call.arguments}) -> {res.content!r}"
                  f"   ok={res.ok}")
    else:
        print("  工具调用轨迹     : (无)")

    if result.error is not None:
        # arun 永不抛业务异常（红线 6）：失败被编码进 status=FAILED + error。
        # 想看 traceback 的话把 AgentConfig(raise_on_error=True) 打开，它会改成 re-raise。
        print(f"  错误 error       : {type(result.error).__name__}: {result.error}")


def _render_trace(agent: Agent) -> None:
    """把 agent.state.messages 打出来 —— 这就是 ReAct 循环留在纸面上的全部痕迹。

    state.messages 是**这一轮实际发给模型的对话**（不含 memory 拼出来的 system 段）。
    原生 function-calling 模式下你会看到：
        assistant(tool_calls)  ->  tool(tool_result)  ->  assistant(final)
    文本模式下同一份轨迹长成：
        assistant(Thought/Action/Action Input) -> user(Observation:) -> assistant(Final Answer:)
    两种模式的差异被隔离在 agent 内部，messages 的形状就是最好的说明书。
    """
    line = "-" * 74
    print(line)
    print("对话轨迹（agent.state.messages）")
    print(line)
    for index, message in enumerate(agent.state.messages):
        body = message.content.replace("\n", "\\n")
        if len(body) > 84:
            body = body[:81] + "..."
        # metadata["kind"] 是框架给"非普通对话"打的标签：observation / nudge / summary ...
        kind = message.metadata.get("kind") or "-"
        print(f"  [{index}] {str(message.role):9} kind={kind:12} {body!r}")
        # 原生模式的 assistant 消息 content 往往是空的 —— 信息全在 tool_calls 里。
        # 不打印出来，读者会误以为"模型什么都没说"。
        for call in message.tool_calls:
            print(f"        -> tool_call id={call.id} {call.name}({call.arguments})")
        if message.tool_call_id:
            print(f"        <- tool_result for id={message.tool_call_id} name={message.name}")


async def main_async(args: argparse.Namespace) -> int:
    """示例主体。注意它是 async 的 —— Agent 的核心入口是 `await agent.arun(...)`。

    为什么核心是异步：一轮 ReAct 里"并发调用多个工具"是常态（03 有演示），
    用同步阻塞写会让慢工具串行拖垮整个循环。同步场景用 `agent.run(...)` 即可，
    它内部是 `asyncio.run(...)` 的薄包装。
    """
    # ---- 3.1 装配：llm + tools + config ----
    if args.offline:
        print("[1/4] LLM   : ScriptedLLM（离线脚本模型，预设 2 条响应，不联网）")
        llm: Any = _build_offline_llm()
    else:
        print(f"[1/4] LLM   : 真实 provider {args.provider!r}（需要对应的 API key）")
        try:
            llm = _build_live_llm(args.provider, args.model)
        except Exception as exc:  # 缺 key / 未知 provider 都会在这里变成一句人话
            print(f"      装配失败：{type(exc).__name__}: {exc}")
            print("      提示：去掉 --provider 即可跑离线版（不需要任何 key）。")
            return 2

    print(f"      工具 : {TOOLS.names()}")

    # AgentConfig 是所有可调旋钮的集中地：max_steps / mode / temperature /
    # max_parse_retries / raise_on_error ...。这里全部用默认值，只为演示"可以不配置"。
    agent = Agent(llm=llm, tools=TOOLS, config=AgentConfig(max_steps=5), name="quickstart")
    print(f"[2/4] Agent : {agent.describe()}")

    # ---- 3.2 跑 ----
    question = "What is 2 + 40 ? Use the add tool, then answer in one sentence."
    print(f"[3/4] 输入  : {question!r}")
    result = await agent.arun(question)
    print(f"[4/4] 跑完  : status={result.status.value}，共 {result.steps} 步")

    # 离线剧本的收尾自检：预设的响应**恰好**被消费完。
    # 这条断言是"离线测试"之所以可靠的根据 —— 它同时排除两种漂移：
    #   * 多调了一次 LLM（队列空了会抛 ScriptedExhaustedError）；
    #   * 少调了一次（队列还有剩，assert_exhausted 会失败）。
    # 换成真模型时没有这个钩子，所以"循环次数的正确性"必须靠离线测试来守。
    if isinstance(llm, ScriptedLLM):
        try:
            llm.assert_exhausted()
        except Exception as exc:  # ScriptedExhaustedError / AssertionError
            print(f"      剧本自检失败：{type(exc).__name__}: {exc}")
            return 1
        print(f"      剧本自检：{llm.call_count} 次调用恰好消费完 {llm.call_count} 条预设响应")

    print()
    _render_result(result)
    print()
    _render_trace(agent)

    # ---- 3.3 把"最小用法"再打印一遍 ----
    line = "=" * 74
    print()
    print(line)
    print("最小用法：去掉上面所有脚手架后，真正不能省的只有这 4 行")
    print(line)
    print('    llm    = ScriptedLLM([ScriptedResponse.tool("add", {"a": 2, "b": 40}),')
    print('                          ScriptedResponse.text("The answer is 42.")])')
    print("    agent  = Agent(llm=llm, tools=ToolRegistry([add]))")
    print('    result = await agent.arun("What is 2 + 40 ?")')
    print("    print(result.status, result.output, result.usage)")
    print(line)

    # 失败也返回非 0：让 shell / CI 能直接用退出码判断这次示例演示成功没有。
    return 0 if result.ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """只提供两条路：离线（默认）与真实 provider。

    --offline 存在的意义不只是"方便"：它让这个示例**可以在 CI 里跑**。
    一个需要 API key 才能执行的示例，在绝大多数环境里等于"从没被运行过"。
    """
    parser = argparse.ArgumentParser(
        description="liteagent 快速上手：一个最小的 ReAct Agent（默认离线，不需要 API key）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="用 ScriptedLLM 离线驱动（默认；不联网、不需要 API key）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="改用真实 provider，如 openai / anthropic / deepseek（需要 API key）",
    )
    parser.add_argument("--model", default=None, help="配合 --provider 指定模型名")
    args = parser.parse_args(argv)
    # 没给 --provider 就默认离线：这样 `python3 examples/01_quickstart.py` 裸跑也能成功。
    if args.provider is None:
        args.offline = True
    return args


def main(argv: list[str] | None = None) -> int:
    """同步入口 —— examples 用 argparse 保持"像命令行程序"的直觉。

    这里做一次 asyncio.run()，而不是让整个脚本裸跑协程：
    同步的 main() 便于 run_all_examples.py import 后直接调用、也便于断言退出码。
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    print("=" * 74)
    print("liteagent 快速上手 —— 最小可运行的 Agent")
    print("=" * 74)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
