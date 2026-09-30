# 第 5 章 · ReAct 循环：Agent 的心脏

> **本章目标**：彻底理解简历里「完整的 Thought-Action-Observation 循环」。
> 这是全项目**最重要**的一章。读完你能：
> ① 画出完整的状态机；② 说清两种模式的区别与统一方式；
> ③ 讲明白五层循环防护；④ 知道出错时框架会怎么处理。

---

## 5.1 先看一个真实的循环

跑这个例子：

```bash
python3 examples/03_react_text_mode.py --offline
```

它演示的是**文本 ReAct 模式**（模型用纯文本表达思考与行动）。真实输出：

```
完整轨迹（agent.state.messages 渲染）
--------------------------------------------------------------------------
  --- step 1 ---
  Thought               : I need to compute the parentheses first: 12 + 30.
  Action                : add
  Action Input          : {"a": 12, "b": 30}
  Observation (由框架回灌): Observation: 42
  [存档] Thought (kind=react, 不参与循环): 'I need to compute the parentheses first: 12 + 30.'
  --- step 2 ---
  Thought               : 12 + 30 = 42. Now multiply by 2.
  Action                : multiply
  Action Input          : {"a": 42, "b": 2}
  Observation (由框架回灌): Observation: 84
  --- step 3 ---
  Thought               : 42 * 2 = 84. I have the final answer.
  Final Answer          : 84

status          : FINISHED
output          : '84'
steps           : 3   (LLM 被调用 3 次)
tool_calls 轨迹 : ['add({"a": 12, "b": 30})', 'multiply({"a": 42, "b": 2})']
```

**这就是 Thought-Action-Observation 循环。** 注意每一轮的三段结构：

```
Thought（想） → Action（做） → Observation（看结果） → 回到 Thought
```

最后一轮用 `Final Answer` 代替 `Action`，循环结束。

> **最值得体会的一点**：第 2 轮的模型输出里引用了 `42`——
> 但那不是模型算出来的，是**第 1 轮工具的真实返回值被回灌进了上下文**。
> 这就是"Observation"存在的意义：**让模型看到真实世界的结果，而不是它自己的想象。**

---

## 5.2 状态机：一轮里到底发生了什么

下面按代码执行顺序，把一轮循环完整拆开。**这张表是本章的核心，建议对照代码读。**

### 循环开始之前（只做一次）

| # | 做什么 | 关键点 |
| --- | --- | --- |
| 0a | 校验参数、检查 `**overrides` 白名单 | 传了非法键 → **抛 `ConfigError`**（不是失败结果） |
| 0b | **决定模式**（native 还是 text） | 见 5.3 |
| 0c | **获取运行锁**（不可重入） | 已在运行 → 返回 FAILED，error 含 `already has a run in flight` |
| 0d | 发 `RUN_STARTED` 事件 | |
| 0e | **把用户输入写进记忆 buffer** | `state.input` 也记一份，但**不写进 transcript**（见 5.6） |

### 每一轮（`while step < max_steps`）

| # | 做什么 | 关键点 |
| --- | --- | --- |
| 1 | **检查墙钟预算** | 超时 → `RunTimeoutError` → **FAILED** |
| 2 | `step += 1`，发 `STEP_STARTED` | |
| 3 | **组装 prompt** | 系统提示 + 摘要 + 长期记忆 + 窗口 + 本轮输入（第 6 章详述） |
| 4 | **检查 prompt token 预算** | 超了 → `BudgetExceededError(kind="prompt_tokens")` → FAILED |
| 5 | **调 LLM** | native 模式传 `tools=`，text 模式传 `None` |
| 6 | 累加 token 用量，记录 `finish_reason` | |
| 7 | **把 assistant 消息写进 transcript** | ⭐ **这是 transcript 上唯一的 assistant 写入点** |
| 8 | **检查总 token 预算** | 超了 → `BudgetExceededError(kind="total_tokens")` → FAILED |
| 9 | **决定：行动还是终结？** | 见 5.4 的四条分支 |
| 10 | **重复/无进展检测** | 见 5.5，命中可能 nudge 或直接失败 |
| 11 | **执行工具** | 并发执行（`parallel_tool_calls` 且多于 1 个时用 `execute_many`） |
| 12 | **观察回灌** | native：每条结果一条 `role="tool"` 消息；text：渲染成**一条** Observation |
| 13 | （文本模式）把 Thought 存档进历史 | `include_thought_in_history` 控制 |
| 14 | **transcript 上限裁剪** | 超过 `max_transcript_messages`（默认 1000）就裁 |
| 15 | **压缩检查**（每轮一次） | 触发就调 LLM 做摘要压缩 |
| 16 | 发 `STEP_FINISHED`，回到第 1 步 | |
| — | **步数用尽** | `MaxStepsExceededError` → FAILED |

### 一个刻意的不变式

第 7 步那句注释很重要：**assistant 消息只有一个写入点**。

早期版本里，"parse error 自纠正"时框架又添加了一次 assistant 原文，
导致同一条消息在上下文里出现两遍——模型会看到自己说了两遍同一句话。
修复后冻结为一条可断言的不变式：

> **一次 parse error 恰好新增 2 条消息**（1 条 assistant 原文 + 1 条 nudge 反馈）。

这条不变式**被写进了测试**，所以这个 bug 不会复活。

> 面试可以这么讲：状态机里"单一写入点"这种约束，**光靠代码审查守不住**，
> 必须写成可断言的测试。我们把它变成了一个具体的数字断言（增量恰好 2）。

---

## 5.3 两种模式：统一在一份数据结构上

第 1 章讲过两种"模型表达要调工具"的方式。本项目**两种都支持**，而且关键设计是：

```
原生 function calling（结构化字段） ──┐
                                     ├──▶ list[ToolCall] ──▶ 同一个 ReAct 状态机
文本 ReAct 解析（正则解析文本）   ────┘
```

**控制流只有一份。** 两种模式只是它的两个入口。

### 5.3.1 模式是怎么决定的

```python
has_tools = len(self.tools) > 0
mode = llm.resolve_mode(has_tools=has_tools)     # 只有一行：
#     return "native" if (self.supports_tool_calling and has_tools) else "text"
```

`mode="auto"`（默认）时按上面的规则自动选。也可以强制：

- `AgentConfig(mode="text")` → 永远走文本路径；
- `AgentConfig(mode="native")` → 强制原生，但如果模型不支持会**抛 `ConfigError`**。

### 5.3.2 两种模式在一轮里的差异

| | native 模式 | text 模式 |
| --- | --- | --- |
| 工具怎么给模型 | `tools=` 参数传结构化 schema | 渲染成文字清单塞进 system prompt |
| 模型怎么表达调工具 | `tool_calls` 字段 | `Action:` / `Action Input:` 文本 |
| 观察怎么回灌 | 每条结果一条 `role="tool"` 消息 | 渲染成**一条** `Observation:` 文本 |
| 推理过程可见吗 | 不可见（除非模型自己写了 content） | **完全可见**（Thought 是明文） |
| 解析会失败吗 | 基本不会（结构化） | **会**，需要容错与自纠正 |

**怎么选？**

- 能用 native 就用 native（更可靠、更省 token）；
- 需要**调试"模型为什么绕圈子"**、或模型不支持原生调用时，用 text 模式——
  因为它的中间过程全是明文。

---

## 5.4 第 9 步的决策树：行动还是终结

这一轮模型给了响应之后，框架怎么判断该干什么？按顺序检查：

```
拿到的响应
   │
   ├─ finish_reason == "content_filter"?
   │     → 立即 FAILED（被内容审核拦了，没什么可做的）
   │
   ├─ finish_reason == "length" 且没有 tool_calls?
   │     → 输出被 max_tokens 截断了
   │     → 注入一条"请从中断处继续"的提示，消耗一步继续（最多 max_truncation_retries=1 次）
   │     → 超过次数 → FAILED("output truncated by max_tokens")
   │
   ├─ native 模式 + finish_reason == "tool_calls" 但一个 call 都没有?
   │     → 模型说"我要用工具"却没说用哪个 = 格式错误
   │     → 走自纠正（和 parse error 同一条路径）
   │
   ├─ native 模式 + 没有 tool_calls?
   │     → 这是最终答案 → 终结
   │
   └─ text 模式
         → 解析文本：
             解析失败      → 自纠正
             有 Action     → 构造 ToolCall，去执行
             有 Final      → 终结
             只有 Thought  → 保守终结
             两者都有      → **执行 Action**（Action 优先）
```

### 5.4.1 `finish_reason` 参与控制流

这一点很值得讲：很多框架把 `finish_reason` 当"参考信息"记个日志就完了。
本项目让它**参与控制流**，因为三种取值意味着三种完全不同的处境：

| 值 | 含义 | 框架该做什么 |
| --- | --- | --- |
| `stop` | 正常结束 | 接受这个答案 |
| `tool_calls` | 想调工具 | 执行工具 |
| `length` | 被 max_tokens 截断了 | **答案是不完整的**，应该续写而不是接受 |
| `content_filter` | 被内容审核拦截 | 继续下去没有意义，立即失败 |

**如果不处理 `length`**，你会拿到一个半截的答案，还以为是完整的——这是很隐蔽的 bug。

### 5.4.2 自纠正：解析失败不是终点

文本模式最大的风险是"模型不按格式输出"。框架的处理不是抛异常，而是**把格式说明书回灌**：

真实的恢复过程（来自示例输出）：

```
第 1 步的模型输出（故意写错格式）：
  'Hmm, 12 + 30 is 42, so the answer should be 84. That seems right.'

框架注入的 nudge：
  | Your previous output could not be parsed: no ReAct structure found.
  | Reply using exactly this format:
  | Thought: <your reasoning>
  | Action: <one of [add, multiply, power]>
  | Action Input: <JSON object>
  | or
  | Thought: <...>
  | Final Answer: <...>

下一轮模型输出：
  Thought: Sorry, let me use the proper format.
  Final Answer: 84

结果：status = FINISHED（没有失败，循环恢复了）
```

三个可断言的不变量（示例里直接打印出来验证）：

```
1) 错误原文在上下文里恰好出现 1 次（不是 2 次 —— assistant 原文只有一个写入点）
2) 注入了 1 条 nudge 反馈，其中含 'could not be parsed'
3) 一次 parse error 的消息增量恰好是 2（1 条 assistant + 1 条 nudge）
4) parse error 消耗 step 但不重试 LLM：steps == LLM 调用次数
```

**第 4 条尤其重要**：自纠正消耗一步但**不重试 LLM 调用本身**。
否则重试次数会放大成 `max_steps × max_parse_retries` 次调用，成本失控。

### 5.4.3 一个额外的安全设计

nudge 消息虽然角色是 `user`，但它是**框架生成的**，不是用户说的。
所以它**不会被写进长期记忆**——否则下一轮它会被当成"用户说过的事实"召回，
把模型的错误格式固化下来。运行时会看到这样一条 warning：

```
MemoryManager.aadd: skipping long-term auto-write for framework message
kind='nudge' (only real user messages become long-term facts)
```

**这是一个很细但很重要的设计**：区分"谁说的"而不只是"什么角色"。

---

## 5.5 五层循环防护（面试必问）

> **面试官最常问的问题之一：「怎么防止 Agent 无限循环？」**

本项目有**五层**独立防护，任何一层生效都能终止循环：

| # | 机制 | 默认值 | 触发后的行为 |
| --- | --- | --- | --- |
| 1 | **最大步数** `max_steps` | 10 | 用尽 → `MaxStepsExceededError` → FAILED |
| 2 | **重复动作检测**（`canonical_key` 去重） | 阈值 2 次 | 第一次命中 → **nudge**（可继续）；再命中 → FAILED |
| 3 | **无进展检测**（观察结果摘要重复） | 阈值 2 次 | 同上 |
| 4 | **token 预算** `max_total_tokens` | 200000 | 超出 → `BudgetExceededError` → FAILED |
| 5 | **墙钟超时** `max_wall_clock_s` | None（不限） | 超出 → `RunTimeoutError` → FAILED |

### 5.5.1 第 2 层：重复动作检测

```python
call.canonical_key()
# = f'{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)}'
```

比如 `add({"a": 2, "b": 3})` 和 `add({"b": 3, "a": 2})` 会得到**同一个 key**
（因为参数被排序了）——这是刻意的，参数顺序不同不算不同调用。

`repeat_action_policy` 有四种取值：

| 取值 | 行为 |
| --- | --- |
| `off` | 不检测 |
| `nudge` | 只提醒，永不失败 |
| `fail` | 命中即失败 |
| `nudge_then_fail`（默认） | 第一次提醒，第二次失败 |

实测行为：第 1 次调用不触发，第 2 次触发 nudge，第 3 次 FAILED。

### 5.5.2 第 3 层：无进展检测（更隐蔽的情况）

**重复动作检测有个漏洞**：如果模型每次都稍微改一下参数呢？

```python
read_file("a.py", start_line=1)      # 第 1 次
read_file("a.py", start_line=2)      # 第 2 次 —— canonical_key 不同！
read_file("a.py", start_line=3)      # 第 3 次 —— 又不同
```

参数每次略变，去重就失效了，但它**显然是卡住了**。

本项目的第二道防线是**看结果而不是看请求**：对每次工具返回的内容做
`blake2b` 摘要并计数。如果**同一个结果反复出现**，说明无论怎么调参数，世界没有变化。

```python
observation_digests: dict[str, int]   # 摘要 -> 出现次数
```

判定是**全局的**：只要任意一份观察重复到阈值，就认为"没有进展"。
nudge 文本会追加一句很直白的提示：

```
Your last N calls returned identical results — the approach is not working.
```

> **这一层是很能体现思考深度的设计。** 面试时被问"怎么防止死循环"，
> 答"max_steps"只能得一半分；答出"重复动作检测 + 看结果的进展检测"才说明
> 你想过**模型如何绕过你的检测**。

### 5.5.3 第 4 层：双层 token 预算

| 参数 | 检查时机 | 查什么 |
| --- | --- | --- |
| `max_prompt_tokens` | 组装完 prompt 之后、调模型**之前** | 这一轮送进去的 prompt 有多大 |
| `max_total_tokens` | 每轮 LLM 调用**之后** | 整个 run 累计花了多少 |

**两道都要有**：前者防"单次请求就撑爆上下文"，后者防"每次都不大但累积起来很贵"。

### 5.5.4 第 5 层：墙钟超时

`max_wall_clock_s` 在**每轮开头**检查。它不是"精确的定时器"，
而是"至少每轮检查一次"——因为工具执行本身可能很慢（`NO_TIMEOUT` 的工具尤其）。
真正需要硬性中断，要靠 `asyncio` 的取消机制。

### 5.5.5 补充：工具失败熔断

除了上面五层，执行器还有熔断（第 4 章 4.7.6）：同一个工具连续失败 3 次后
不再执行。这也间接防止了"模型反复调一个坏工具"的循环。

### 5.5.6 小结：五种失败对应的异常

```
max_steps 用尽           → MaxStepsExceededError
重复动作（第二次命中）    → RepeatedActionError
token 预算超限            → BudgetExceededError
墙钟超时                  → RunTimeoutError
解析失败用尽              → AgentError("model output could not be parsed ...")
输出一直被截断            → AgentError("output truncated by max_tokens")
```

**全部返回 `status=FAILED`，并把异常放在 `result.error` 里**（不抛出去）。

---

## 5.6 一个反直觉的设计：用户输入不在 transcript 里

这是很多人在读代码时会困惑的地方，值得单独讲。

```python
result = await agent.arun("USER_ORIGINAL_INPUT")
result.state.input          # 'USER_ORIGINAL_INPUT'  ✓ 在这里
result.state.messages       # 里面**没有**这句用户输入！
```

实测的 `state.messages`（一轮工具 + 终结）：

```
[0] role=assistant  'Thought: thinking\nAction: echo\nAction Input: {"text": "hi"}'
[1] role=user       'Observation: hi'          ← 框架注入的观察，不是用户说的
[2] role=assistant  'thinking'                 ← 存档的 Thought
[3] role=assistant  'Thought: done\nFinal Answer: 42'
```

**为什么这样设计？**

用户输入只写一次到**记忆的 buffer** 里，每轮组装 prompt 时才拼进去（第 6 段）。
好处是：

1. **避免重复**：如果既写 transcript 又写 buffer，组装 prompt 时会拼出两份。
2. **职责清晰**：`state.messages` 是"循环过程中产生的消息"，buffer 是"要送给模型的上下文"。

**代价是**：`state.messages` **不是**一个完整可回放的对话记录。
如果你要做"把用户输入也算进去"的完整轨迹，得自己补。

> **这也是一个面试可讲的点**：`AgentState.messages` 的语义是
> "模型可见 transcript 的一个子集"，而不是"完整对话"。
> 术语在 `docs/INTERFACES.md` 的术语表里被严格区分了
> （transcript / window / buffer / evicted 四个词各有精确定义）。

---

## 5.7 事件：可观测性的骨架

循环的每一步都会发事件，共 27 种。常用的：

| 事件 | 什么时候发 | 谁发（唯一发射者） |
| --- | --- | --- |
| `run_started` / `run_finished` / `run_failed` | run 开始/成功/失败 | Agent |
| `step_started` / `step_finished` | 每轮开始/结束 | Agent |
| `llm_request` / `llm_response` / `llm_error` | 每次 LLM 调用 | **LLM 层** |
| `thought` / `action_parsed` | 文本模式解析出思考/行动 | Agent |
| `parse_error` / `nudge` | 解析失败 / 注入纠正提示 | Agent |
| `repeat_detected` | 重复或无进展命中 | Agent |
| `tool_started` / `tool_retry` / `tool_finished` / `tool_error` | 工具执行各阶段 | **执行器** |
| `tool_approval` | 审批结果 | 执行器 |
| `memory_write` / `memory_retrieve` / `memory_compress` | 记忆读写压缩 | **记忆层** |
| `budget_exceeded` / `context_truncated` | 预算/上下文超限 | Agent / 记忆层 |
| `agent_delegate` / `agent_return` | 多 Agent 委派 | multiagent |
| `blackboard_write` / `blackboard_read` | 黑板读写 | Blackboard |

**注意"唯一发射者"这一列**——这是本项目的一条红线：

> **同一条事件只能由一处发出。** 如果 Agent 层和 LLM 层都发 `llm_error`，
> 事件序列里就会出现两条，任何"数一数发生了几次"的断言都会失效。

这个约束在实际开发中**真的抓到过 bug**（见 `docs/BUILD_LOG.md` 踩坑故事 7）。

### 怎么订阅

```python
from liteagent import Agent, JsonlTraceCallback, TokenCounterCallback
from liteagent.agent import as_llm_callback

agent = Agent(llm=llm, tools=registry,
              callbacks=[TokenCounterCallback(), JsonlTraceCallback("run.jsonl")])

# 或者运行期订阅
unsubscribe = agent.callbacks.subscribe(lambda event: print(event.summary()))
```

**注意一个坑**：`Agent` 只给自己创建的 memory/executor 挂事件转发，
**不会**改你传进来的 `llm` 实例。所以要让 trace 里出现 `llm_request` 之类的事件，
必须显式接上：

```python
llm.on_event = as_llm_callback(agent.callbacks)
```

不接的话，`trace_stats()` 里的 `llm_calls` / `llm_latency_ms` 会一直是 0。

---

## 5.8 动手：当一次"导演"

理解循环最快的方式，是**自己写剧本**，看框架怎么反应。

```python
import asyncio
from liteagent import Agent, AgentConfig, EventType
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def get_weather(city: str) -> str:
    """查询城市天气。

    Args:
        city: 城市名。
    """
    return f"{city}: 晴，25℃"


async def main():
    events = []
    llm = ScriptedLLM([
        ScriptedResponse.tool("get_weather", {"city": "北京"}),
        ScriptedResponse.text("北京今天晴，25 度。"),
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([get_weather]),
                  config=AgentConfig(max_steps=5))
    agent.callbacks.subscribe(lambda e: events.append(e.type.value))

    result = await agent.arun("北京天气怎么样？")
    print("状态:", result.status.value)
    print("回答:", result.output)
    print("事件流:", " → ".join(events))
    llm.assert_exhausted()


asyncio.run(main())
```

**实验清单**（改一处跑一次，观察差异）：

| 实验 | 怎么改 | 你会看到 |
| --- | --- | --- |
| ① 让模型卡住 | 剧本写成反复调用同一个工具 | 第 2 次调用触发 nudge，第 3 次 FAILED |
| ② 参数每次略变 | 剧本里每次换个参数值但工具返回一样 | **无进展检测**命中（第 3 层防护） |
| ③ 步数不够 | `max_steps=1` 但剧本要 2 步 | `MaxStepsExceededError` |
| ④ 格式错误 | 用 `.react()` 输出一段不带 Action 的废话 | parse error 自纠正 |
| ⑤ 剧本不够 | 只给 1 条响应但循环要 2 条 | `ScriptedExhaustedError`（这其实是好事，说明次数符合预期） |
| ⑥ 切文本模式 | `AgentConfig(mode="text")` | 事件流里多了 `thought` / `action_parsed` |

---

## 5.9 本章小结

1. **循环的四拍**：Thought（模型）→ Action（框架）→ Observation（框架）→ 回到 Thought。
2. **两种模式归约到同一个 `list[ToolCall]`**，所以控制流只有一份。
3. **`finish_reason` 参与控制流**：`length` 要续写，`content_filter` 立即失败。
4. **五层循环防护**：步数 / 重复动作 / 无进展 / token 预算 / 墙钟超时。
5. **解析失败不抛异常，而是把格式说明书回灌**（自纠正），且不重试 LLM 调用。
6. **用户输入不在 `state.messages` 里**——它在 `state.input` 和记忆 buffer 里。
7. **事件有唯一发射者**，这是让"数次数"的断言能成立的前提。

### 自测题

1. 说出循环一轮里的四个阶段，以及每个阶段由谁负责。
2. `finish_reason="length"` 时框架做什么？为什么不直接接受这个答案？
3. "重复动作检测"能被模型绕过吗？怎么绕？第二层防护是什么？
4. 一次 parse error 后，消息数量增加几条？为什么是这个数字？
5. 为什么用户输入不在 `state.messages` 里？
6. 为什么"同一条事件只能由一个地方发出"很重要？

<details>
<summary>参考答案</summary>

1. Thought（模型输出）、Action（框架解析并执行）、Observation（框架回灌结果）、循环判定（框架）。
2. 注入"请继续"提示，消耗一步重试一次，超过上限则 FAILED；因为截断的答案是不完整的，直接接受会返回半截结果。
3. 能——每次把参数改一点点，`canonical_key` 就不同了；第二层是"无进展检测"，对工具**返回结果**做摘要计数，看世界有没有变化。
4. 2 条（1 条 assistant 原文 + 1 条 nudge 反馈）；因为 assistant 只有一个写入点，这是被测试断言锁住的不变式。
5. 它写在记忆 buffer 里，每轮组装 prompt 时才拼进去；避免同一句话被记录两遍。
6. 否则"这条事件发生了几次"无法断言，观测数据自相矛盾，排障时会误导。

</details>

---

**下一章**：[第 6 章 · 记忆管理](06-memory.md) —— 三层记忆怎么协作。
