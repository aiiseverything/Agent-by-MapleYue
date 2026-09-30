# 第 2 章 · 跑通你的第一个 Agent

> **本章目标**：亲手跑通一个能调用工具的 Agent，并看懂它内部发生了什么。
> 读完你应该能：① 自己写一个工具；② 用 4 行代码组装一个 Agent；③ 读懂运行轨迹。

---

## 2.1 先跑起来

打开终端，进入项目根目录：

```bash
cd /home/ml-user/workdir/project-3
python3 examples/01_quickstart.py --offline
```

你会看到这样的输出（这是真实输出，只有耗时数字每次不同）：

```
==========================================================================
liteagent 快速上手 —— 最小可运行的 Agent
==========================================================================
[1/4] LLM   : ScriptedLLM（离线脚本模型，预设 2 条响应，不联网）
      工具 : ['add', 'word_count']
[2/4] Agent : {'name': 'quickstart', 'description': '', 'model': 'scripted-1',
               'tools': ['add', 'word_count'], 'mode': 'native', 'max_steps': 5}
[3/4] 输入  : 'What is 2 + 40 ? Use the add tool, then answer in one sentence.'
[4/4] 跑完  : status=FINISHED，共 2 步
      剧本自检：2 次调用恰好消费完 2 条预设响应

--------------------------------------------------------------------------
运行结果
--------------------------------------------------------------------------
  状态 status      : FINISHED   (result.ok = True)
  最终输出 output  : 'The answer is 42.'
  步数 steps       : 2
  Token 用量 usage : prompt=20, completion=10, total=30
  工具调用轨迹     :
    [1] add({'a': 2, 'b': 40}) -> '42'   ok=True

--------------------------------------------------------------------------
对话轨迹（agent.state.messages）
--------------------------------------------------------------------------
  [0] Role.ASSISTANT kind=-            ''
        -> tool_call id=call_0 add({'a': 2, 'b': 40})
  [1] Role.TOOL kind=-            '42'
        <- tool_result for id=call_0 name=add
  [2] Role.ASSISTANT kind=-            'The answer is 42.'
```

**先别管细节，感受一下发生了什么**：你给了一个问题，Agent 自己决定去调 `add(2, 40)`，
拿到结果 `42`，然后生成了最终答案。**这个"自己决定"就是 Agent 和普通程序的区别。**

---

## 2.2 这段代码到底做了什么

### 2.2.1 真正不能省的只有 4 行

示例文件末尾自己给出了答案：

```python
llm    = ScriptedLLM([ScriptedResponse.tool("add", {"a": 2, "b": 40}),
                      ScriptedResponse.text("The answer is 42.")])
agent  = Agent(llm=llm, tools=ToolRegistry([add]))
result = await agent.arun("What is 2 + 40 ?")
print(result.status, result.output, result.usage)
```

就这么简单。剩下的 300 行都是脚手架（打印、格式化、自检）。

### 2.2.2 逐行拆解

```python
llm = ScriptedLLM([...])
```

**这是一个"假模型"。** 它不联网、不需要 API key，而是**按你给的剧本一条条吐响应**。
第一条剧本说"我要调用 `add` 工具，参数是 `{"a": 2, "b": 40}`"，
第二条剧本说"我的最终答案是 `The answer is 42.`"。

> 为什么要用假模型？因为它让**整个 Agent 循环变得确定、可复现、可离线运行**。
> 真模型每次回答都可能不一样，你没法写测试，也没法确定性地学习。
> 这个 `ScriptedLLM` 是本项目最重要的教学工具，第 3 章会专门讲。

```python
agent = Agent(llm=llm, tools=ToolRegistry([add]))
```

**组装一个 Agent**：给它一个模型客户端 + 一个工具注册表。就这两样。

`ToolRegistry([add])` 是"工具清单"的载体。为什么需要它而不是直接传个列表？
因为它还要负责：重名检查、别名、按名字查找、**把工具导出成各家模型厂商要的 JSON Schema 格式**。

```python
result = await agent.arun("What is 2 + 40 ?")
```

**跑起来**。`arun` 是"async run"。它内部就是我们第 1 章讲的 ReAct 循环。
注意它**不抛异常**——所有失败都编码在返回值里（下面讲）。

```python
print(result.status, result.output, result.usage)
```

**读结果**。`result` 是 `AgentResult`，把所有信息都装在一个对象里。

---

## 2.3 结果对象 `AgentResult`

`arun` 永远返回一个 `AgentResult`，字段如下（逐字来自代码）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `output` | `str` | 最终答案文本（失败时可能是空串或部分输出） |
| `status` | `AgentStatus` | 见下表 |
| `steps` | `int` | 循环跑了几轮 |
| `tool_calls` | `list[ToolCall]` | 这次 run 里模型要求的所有工具调用 |
| `tool_results` | `list[ToolResult]` | 对应的执行结果 |
| `usage` | `TokenUsage` | token 用量（prompt / completion / total） |
| `error` | `LiteAgentError \| None` | 失败时的异常对象 |
| `state` | `AgentState \| None` | 完整状态快照（含完整对话轨迹） |
| `duration_ms` | `float` | 耗时毫秒 |
| `agent_name` | `str` | Agent 名字 |
| `metadata` | `dict` | 附加信息（如 `cost_usd`、`finish_reason`） |

`status` 有 7 个取值：`IDLE`（还没跑）、`THINKING`、`ACTING`、`OBSERVING`（运行中的三个中间态）、
`FINISHED`（成功）、`FAILED`（失败）、`ABORTED`（被取消）。

还有一个方便的属性：

```python
result.ok      # 等价于 result.status == AgentStatus.FINISHED，只有 FINISHED 算 True
```

### 为什么不抛异常？

这是这个框架一个刻意的设计决策：

> **`arun` 永不向外抛业务异常。** 失败也返回 `AgentResult`，错误信息放在 `result.error` 和
> `result.status` 里。想按异常风格用，就调 `result.raise_for_status()`。

理由：Agent 的失败是**常态**（模型抽风、工具报错、超步数），不是"意外"。
如果每次失败都抛异常，调用方就得写一堆 try/except，而且容易把"部分成功"的现场丢掉。
把它编码成数据，反而更容易处理：

```python
result = await agent.arun("...")
if result.ok:
    print(result.output)
else:
    print(f"失败了：{type(result.error).__name__}: {result.error}")
    print(f"但已经跑到第 {result.steps} 步，最后的输出是：{result.output!r}")
```

---

## 2.4 看懂对话轨迹

再看一眼输出里的"对话轨迹"部分，它揭示了循环内部发生了什么：

```
  [0] Role.ASSISTANT  ''          -> tool_call id=call_0 add({'a': 2, 'b': 40})
  [1] Role.TOOL       '42'        <- tool_result for id=call_0 name=add
  [2] Role.ASSISTANT  'The answer is 42.'
```

翻译成第 1 章的循环四拍：

| 轨迹 | 对应循环的哪一拍 | 谁产生的 |
| --- | --- | --- |
| `[0] assistant: 我想调用 add` | **① 思考** | 模型（这里是假模型按剧本吐的） |
| `[1] tool: 42` | **③ 观察** | 框架（真的执行了 `add` 并转成文本） |
| `[2] assistant: 答案是 42` | **① 思考**（第二次）| 模型 |

注意 `[0]` 那行的 `content` 是空的 `''`——因为这一轮模型没说人话，
它只输出了一个结构化的"工具调用请求"（`tool_calls`）。这是 native 模式的特征。

> **要点**：`state.messages` 是"模型看到的对话历史"的骨架。
> 但要注意——**用户最初的那句 `What is 2 + 40 ?` 并不在里面**！
> 它存在 `state.input` 和记忆的 buffer 里，在每轮组装 prompt 时才被拼进去。
> 这个反直觉的设计是有意的（避免同一条用户消息被重复记录），第 5 章会详细解释。

---

## 2.5 自己动手：写一个你自己的工具

现在轮到你了。创建一个新文件 `my_agent.py`：

```python
from __future__ import annotations

import asyncio
from typing import Literal

from liteagent import Agent, AgentConfig
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def bmi(weight_kg: float, height_m: float) -> str:
    """计算身体质量指数（BMI）。

    Args:
        weight_kg: 体重，单位千克。
        height_m: 身高，单位米。
    """
    value = weight_kg / (height_m ** 2)
    if value < 18.5:
        level = "偏瘦"
    elif value < 24:
        level = "正常"
    else:
        level = "偏胖"
    return f"BMI={value:.1f}（{level}）"


async def main() -> None:
    # 先看看框架从这个函数里"读"出了什么
    import json
    print("自动生成的 JSON Schema：")
    print(json.dumps(bmi.parameters, indent=2, ensure_ascii=False))

    # 导演一部两幕剧：第 1 幕模型要求调工具，第 2 幕模型给出答案
    llm = ScriptedLLM([
        ScriptedResponse.tool("bmi", {"weight_kg": 70, "height_m": 1.75}),
        ScriptedResponse.text("你的 BMI 是 22.9，属于正常范围。"),
    ])

    agent = Agent(llm=llm, tools=ToolRegistry([bmi]), config=AgentConfig(max_steps=5))
    result = await agent.arun("我 70 公斤、1.75 米，BMI 是多少？")

    print("\n状态：", result.status.value)
    print("答案：", result.output)
    print("步数：", result.steps)
    for r in result.tool_results:
        print(f"工具 {r.name} 返回：{r.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

跑它：

```bash
python3 my_agent.py
```

你应该看到：

```json
{
  "type": "object",
  "properties": {
    "weight_kg": {"type": "number", "description": "体重，单位千克。"},
    "height_m": {"type": "number", "description": "身高，单位米。"}
  },
  "required": ["weight_kg", "height_m"],
  "additionalProperties": false
}
```

**你一行 schema 都没写，它自己生成出来了。** 这就是第 4 章的主题：
`@tool` 装饰器怎么把 type hints 和 docstring 反射成 JSON Schema。

### 试试改坏它

做几个小实验，加深理解：

**实验 1：给参数加默认值**

```python
@tool
def bmi(weight_kg: float, height_m: float = 1.70) -> str:
    ...                       # 函数体不变，只改了签名
```

再跑一次，看 `required` 数组变成了什么。（答案：只剩 `weight_kg` —— 有默认值的参数不是必填。）

**实验 2：用 `Literal` 限定取值**

```python
@tool
def bmi(weight_kg: float, height_m: float, unit: Literal["metric", "imperial"] = "metric") -> str:
    ...                       # 函数体不变
```

看 schema 里 `unit` 变成了 `{"type": "string", "enum": ["metric", "imperial"]}`。
**模型看到 `enum` 就知道只能从这几个值里选**，这比在描述里写"请传 metric 或 imperial"可靠得多。

**实验 3：模型传错参数会怎样**

把剧本改成 `ScriptedResponse.tool("bmi", {"weight_kg": "七十", "height_m": 1.75})`，
跑一下，看返回的 `ToolResult`。你会发现：

```
ERROR(ToolValidationError): invalid arguments for tool: ...
  $.weight_kg: expected number, got string
fix the arguments to match the tool's JSON schema and call this tool again, ...
```

**关键点**：框架在**执行工具之前**就用同一份 schema 校验了参数，**函数根本不会被调用**。
然后这个错误信息会作为"观察结果"回灌给模型，模型有机会自己修正参数重试。
这是"同一份 schema 双向使用"的价值——既告诉模型怎么用，又用于拦截错误。

---

## 2.6 三个必须知道的 API

学完本章，你应该记住这三个：

| API | 作用 | 什么时候用 |
| --- | --- | --- |
| `await agent.arun(input)` | 异步跑一次 | 你的代码已经在 asyncio 里 |
| `agent.run(input)` | 同步跑一次 | 写脚本、没有事件循环时 |
| `agent.describe()` | 返回 Agent 的自我描述 dict | 调试"它到底有哪些工具" |

`run` 和 `arun` 的关系（这是一个值得学的工程细节）：

```python
# agent.py 里 run 的实现
def run(self, input: str, **kwargs) -> AgentResult:
    return run_sync(lambda: self.arun(input, **kwargs))
```

注意它传的是 **`lambda`（一个工厂函数）**，不是直接传 `self.arun(...)` 的调用结果。
为什么？因为如果写成 `run_sync(self.arun(input))`，协程对象在进入 `run_sync` 之前就被创建了；
一旦 `run_sync` 出于任何原因没 await 它，Python 会打印
`RuntimeWarning: coroutine ... was never awaited`，而且**这个协程永远不会执行**——
一个静默的 bug。传工厂函数就杜绝了这种可能。

> 这个小细节在很多代码库里都是 bug 来源。本项目把它作为一条红线写进了规范。

---

## 2.7 常见错误对照表

新手最容易踩的几个坑：

| 报错 | 原因 | 怎么修 |
| --- | --- | --- |
| `ScriptedExhaustedError: consumed: 1` | 剧本条数不够，循环还想再要一条响应 | 加剧本，或给 `ScriptedLLM(..., loop=True)` |
| `ConfigError: mode='native' requires ...` | 显式指定了 native 模式，但用的模型不支持工具调用 | 改成 `mode="text"` 或用支持 tool calling 的模型 |
| `ToolDefinitionError: ... declares **kwargs` | 工具函数用了 `**kwargs` | JSON Schema 无法表达"任意键"，改成显式参数 |
| `ConfigError: no prompt provided` | （CLI）没给 prompt | 加 `-p "你的问题"` |
| `ToolNotFoundError: name='xxx'` | 模型要求调用一个没注册的工具 | 检查工具名拼写；错误信息里会列出**所有可用工具名** |

---

## 2.8 本章小结

1. **组装一个 Agent 只需要两样东西**：一个模型客户端 + 一组工具。
2. **`arun` 永不抛异常**，结果全在 `AgentResult` 里；用 `result.ok` 判断成功。
3. **`@tool` + type hints + docstring 就自动生成 JSON Schema**，不用手写。
4. **同一份 schema 双向使用**：给模型看 + 执行前校验参数。
5. **`ScriptedLLM` 让你能"导演"Agent 的每一步**，这是学习和测试的利器。

### 自测题

1. `Agent(llm=..., tools=...)` 里两个参数分别是什么？`tools` 为什么不是简单的 list？
2. `result.ok` 为 False 时，你怎么知道失败原因？
3. 为什么 `run()` 里传的是 `lambda` 而不是直接调用 `arun()`？
4. 模型传了错误类型的参数，工具函数会被执行吗？为什么这很重要？
5. 为什么 `state.messages` 里没有用户最初的问题？

<details>
<summary>参考答案</summary>

1. 模型客户端和工具注册表；因为 `ToolRegistry` 还负责重名检查、别名、按名查找，以及**导出成各家厂商要求的 schema 格式**。
2. 看 `result.error`（异常对象）和 `result.status`；也可以用 `result.raise_for_status()`。
3. 避免协程对象在没被 await 的情况下被创建，导致"never awaited"警告和静默不执行。
4. 不会。schema 校验在**执行之前**，所以错误参数根本进不到你的函数里，不会造成副作用；错误信息会回灌给模型让它自己修。
5. 用户输入存在 `state.input` 和记忆 buffer 里，组装 prompt 时才拼进去；这样避免了同一条用户消息被重复记录。

</details>

---

**下一章**：[第 3 章 · LLM 抽象层](03-llm-layer.md) —— 为什么要有这一层，以及 `ScriptedLLM` 的完整玩法。
