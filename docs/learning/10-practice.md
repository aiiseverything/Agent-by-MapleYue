# 第 10 章 · 练习与自测

> **本章目标**：把"读过"变成"会做"。
> 练习按难度分四级，每道都有**可运行的验收标准**——做没做对，跑一下就知道。
> 所有练习都离线可做，不需要 API key。

---

## 0. 怎么用这一章

**建议做法**：先自己写，卡住了再看提示，最后对答案。
每道题的"验收"部分给出了**具体的断言**，跑通即通过。

准备工作：新建一个练习目录，避免污染仓库。

```bash
mkdir -p ~/liteagent-practice && cd ~/liteagent-practice
# 让 python 能找到 liteagent（或者直接把练习文件放在仓库根目录下跑）
export PYTHONPATH=/home/ml-user/workdir/project-3:$PYTHONPATH
```

---

## 1. 入门级（熟悉 API）

### 练习 1.1：写一个带约束的工具

**任务**：写一个 `convert_temperature` 工具，把摄氏度转成华氏度。
要求：
- 参数 `celsius: float`；
- 参数 `unit: Literal["F", "K"] = "F"`（转华氏度或开尔文）；
- 用 `Param` 给 `celsius` 加一个下界约束 `ge=-273.15`（绝对零度）。

**验收**：

```python
import json
schema = convert_temperature.parameters
assert schema["properties"]["unit"]["enum"] == ["F", "K"]
assert schema["properties"]["celsius"]["minimum"] == -273.15
assert schema["required"] == ["celsius"]           # unit 有默认值，不是必填
print("✓ 通过")
```

<details>
<summary>参考答案</summary>

```python
from typing import Annotated, Literal
from liteagent.tools import tool
from liteagent.tools.schema import Param


@tool
def convert_temperature(
    celsius: Annotated[float, Param(description="摄氏温度", ge=-273.15)],
    unit: Literal["F", "K"] = "F",
) -> float:
    """把摄氏度转换成其他温度单位。

    Args:
        celsius: 摄氏温度。
        unit: 目标单位，F 表示华氏度，K 表示开尔文。
    """
    if unit == "F":
        return celsius * 9 / 5 + 32
    return celsius + 273.15
```

> 注意：`Param(ge=-273.15)` 会生成 `"minimum": -273.15`。
> 而 `unit` 因为有默认值 `"F"`，不会进 `required`。

</details>

### 练习 1.2：导演一部三步剧

**任务**：用 `ScriptedLLM` 驱动一个 Agent，让它完成"先算 3×4，再把结果加 10"，
最后输出答案。工具自己写（`multiply` 和 `add`）。

**验收**：

```python
assert result.status.value == "FINISHED"
assert result.output == "22"           # 或你让模型说的那句话
assert [c.name for c in result.tool_calls] == ["multiply", "add"]
assert result.steps == 3               # 2 次工具 + 1 次终结
llm.assert_exhausted()                 # 剧本被精确消费完
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from liteagent import Agent, AgentConfig
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def multiply(a: int, b: int) -> int:
    """两个整数相乘。"""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


async def main():
    llm = ScriptedLLM([
        ScriptedResponse.tool("multiply", {"a": 3, "b": 4}),   # 第 1 步：12
        ScriptedResponse.tool("add", {"a": 12, "b": 10}),      # 第 2 步：22
        ScriptedResponse.text("22"),
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([multiply, add]),
                  config=AgentConfig(max_steps=5))
    result = await agent.arun("先算 3×4，再加 10")
    assert result.status.value == "FINISHED"
    assert result.steps == 3
    llm.assert_exhausted()
    print(result.output)


asyncio.run(main())
```

**注意**：第 2 步的剧本里写的是 `{"a": 12, ...}`——
**这个 12 是"模型"（剧本作者，也就是你）写死的**。
真实场景里，模型是看到第 1 步的 Observation（`12`）才知道该填 12 的。

</details>

---

## 2. 进阶级（理解机制）

### 练习 2.1：证明"无进展检测"生效

**任务**：构造一个场景，让 Agent **参数每次略变**但工具返回完全一样，
验证第 3 层防护（无进展检测）会触发。

**提示**：用 `.react()` 或 `.tool()` 让工具被调用多次，每次参数不同，
但你的工具函数**忽略参数、总是返回同一个字符串**。

**验收**：

```
最终 status 应该是 FAILED
error 类型应该是 RepeatedActionError（或包含 "not working" 的 nudge 后失败）
state.observation_digests 里有一个摘要的计数 >= 2
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from liteagent import Agent, AgentConfig
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def read_file(path: str, start_line: int = 1) -> str:
    """总是返回同样的内容（模拟"文件没变过"）。"""
    return "def add(a, b): return a - b       # 恒定内容"


async def main():
    # 参数每次都不一样（start_line 递增），canonical_key 因此不同
    llm = ScriptedLLM([
        ScriptedResponse.tool("read_file", {"path": "a.py", "start_line": 1}),
        ScriptedResponse.tool("read_file", {"path": "a.py", "start_line": 2}),
        ScriptedResponse.tool("read_file", {"path": "a.py", "start_line": 3}),
        ScriptedResponse.tool("read_file", {"path": "a.py", "start_line": 4}),
        ScriptedResponse.text("算了，我看不出来"),
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([read_file]),
                  config=AgentConfig(max_steps=8, repeat_action_policy="fail"))
    result = await agent.arun("a.py 有什么问题？")

    print("status:", result.status.value)
    print("error :", type(result.error).__name__ if result.error else None)
    print("观察摘要计数:", result.state.observation_digests)

    # 关键断言：虽然每次参数不同（canonical_key 不同），
    # 但因为**返回结果一样**，无进展检测仍然抓到了它
    assert result.status.value == "FAILED"
    assert any(v >= 2 for v in result.state.observation_digests.values())


asyncio.run(main())
```

**这道题的意义**：它证明了"只看请求去重是不够的"。
`canonical_key` 对这四次调用**各不相同**，但 `observation_digests` 发现"结果没变过"。

</details>

### 练习 2.2：验证记忆确实是靠"重新发送"实现的

**任务**：证明大模型没有记忆——"记忆"是框架每轮把历史重新发过去实现的。

**做法**：跑一个两轮的 Agent，然后检查**第 2 轮模型收到的消息里，
是否包含第 1 轮的内容**。

**验收**：

```python
# 第 2 轮的消息列表里应该能看到第 1 轮说过的话
round2 = llm.calls[1].messages
text = "\n".join(str(m.content) for m in round2)
assert "第一轮的暗号是 XYZ" in text          # 第 1 轮的用户输入
assert "42" in text                          # 第 1 轮的工具结果
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from liteagent import Agent, AgentConfig
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def compute(x: int) -> int:
    """返回 x 的平方。"""
    return x * x


async def main():
    llm = ScriptedLLM([
        ScriptedResponse.tool("compute", {"x": 42}),   # 第 1 轮
        ScriptedResponse.text("好的，我记住了。"),       # 第 2 轮
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([compute]),
                  config=AgentConfig(max_steps=5))
    await agent.arun("第一轮的暗号是 XYZ，请计算 42 的平方")

    print("=== 第 1 轮模型看到的消息 ===")
    for m in llm.calls[0].messages:
        print(f"  [{m.role.value}] {str(m.content)[:60]}")

    print("\n=== 第 2 轮模型看到的消息 ===")
    for m in llm.calls[1].messages:
        print(f"  [{m.role.value}] {str(m.content)[:60]}")

    text = "\n".join(str(m.content) for m in llm.calls[1].messages)
    assert "XYZ" in text, "第 2 轮应该能看到第 1 轮的用户输入"
    assert "1764" in text, "第 2 轮应该能看到第 1 轮的工具结果"


asyncio.run(main())
```

**你会看到什么**：第 2 轮的消息比第 1 轮多了几条
（assistant 的工具调用、工具结果）。**这就是"记忆"的真相**——
模型本身没记住任何东西，是框架把历史重新拼进去的。

**顺带一个发现**：第 1 轮的消息里**没有**用户的原始输入吗？
有——因为 `abuild_prompt(append_user_input=True)` 在第 1 步把它拼进去了。
但 `agent.state.messages` 里没有（第 5 章 5.6 讲过这个区别）。

</details>

### 练习 2.3：观察摘要压缩

**任务**：用很小的 buffer 预算跑几轮对话，触发摘要压缩，观察 `compression_count`
和 `<conversation_summary>` 段。

**验收**：

```python
manager.stats()["compressions"] >= 1
# 组装出的 prompt 里能看到摘要段
assert "<conversation_summary>" in "\n".join(str(m.content) for m in prompt)
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.memory import MemoryConfig, MemoryManager, HashingEmbedder
from liteagent.llm.message import Message


async def main():
    summarizer = ScriptedLLM(
        [ScriptedResponse.text("- 用户想写一个 CSV 处理脚本\n- 要求零依赖")],
        loop=True,          # 多次压缩时复用
    )
    manager = MemoryManager.from_config(
        MemoryConfig(
            buffer_max_tokens=80,        # 故意设得很小
            buffer_max_messages=6,
            summary_trigger_ratio=0.5,   # 一半就压缩
            summary_min_evict=2,
            max_summary_chars=200,
        ),
        llm=summarizer,                 # 不传就退化成抽取式兜底
        embedder=HashingEmbedder(dim=64),
    )

    # 灌几轮对话，把窗口撑爆
    for i in range(6):
        await manager.aadd(Message.user(f"第 {i} 轮：我在写一个 CSV 处理脚本，要求零依赖。"))
        await manager.aadd(Message.assistant(f"第 {i} 轮回复：建议用 csv 模块。"))

    # ⚠️ 关键顺序：必须先"组装一次 prompt"，再压缩。
    #    因为 `BufferMemory._evicted` 是 `window()` 计算时的**副产物**
    #    （window() 每次重算窗口，同时把被裁掉的消息记进 _evicted）。
    #    真实 Agent 循环里每轮都会先 abuild_prompt() 再 acompress_if_needed()，
    #    所以永远不会踩到这个坑；但如果你跳过组装直接压缩，drain 会返回空，静默不压缩。
    await manager.abuild_prompt(system="你是一个助手", user_input="继续说")

    await manager.acompress_if_needed()
    stats = manager.stats()
    print("压缩次数:", stats["compressions"])
    print("待压缩:", stats["evicted_pending"])
    print("摘要内容:", repr(manager.summarizer.summary)[:100])

    prompt = await manager.abuild_prompt(system="你是一个助手", user_input="继续")
    text = "\n".join(str(m.content) for m in prompt)
    print("\n组装出的 prompt 段落：")
    for m in prompt:
        print(f"  [{m.role.value}] {str(m.content)[:70]}")

    assert stats["compressions"] >= 1
    assert "<conversation_summary>" in text


asyncio.run(main())
```

**试试改坏它**：把 `llm=summarizer` 去掉，重跑。
你会看到 warning 和抽取式兜底摘要——**摘要失败了，但流程没崩**。

</details>

---

## 3. 挑战级（理解设计）

### 练习 3.1：实现一个"必须串行"的工具

**任务**：两个工具都会写同一个文件。证明如果不配 `sequential_tools`，
并发调用会导致竞态；配上之后就安全了。

**验收**：

```
不配 sequential_tools：最终文件内容长度 < 预期（后写的覆盖了先写的）
配了 sequential_tools：最终文件长度 == 两段内容之和
```

<details>
<summary>参考答案</summary>

```python
import asyncio, tempfile, pathlib
from liteagent import ToolCall                      # 注意：ToolCall 在顶层/types，不在 liteagent.tools
from liteagent.tools import ToolExecutor, ExecutorConfig, ToolRegistry, tool


@tool
def slow_append(path: str, text: str) -> str:
    """（模拟）慢速追加写入。"""
    import time
    p = pathlib.Path(path)
    content = p.read_text() if p.exists() else ""
    time.sleep(0.05)                       # 模拟慢 I/O —— 竞态窗口
    p.write_text(content + text)
    return f"appended {len(text)}"


async def run_case(sequential: bool):
    with tempfile.TemporaryDirectory() as d:
        target = str(pathlib.Path(d) / "out.txt")
        reg = ToolRegistry([slow_append])
        cfg = ExecutorConfig(
            max_concurrency=4,
            sequential_tools=frozenset({"slow_append"}) if sequential else frozenset(),
        )
        ex = ToolExecutor(reg, cfg)
        calls = [ToolCall.create("slow_append", {"path": target, "text": f"{i},"})
                 for i in range(4)]
        await ex.execute_many(calls)
        result = pathlib.Path(target).read_text()
        await ex.aclose()
        return result


async def main():
    unsafe = await run_case(False)
    safe = await run_case(True)
    print("不配 sequential_tools：", repr(unsafe))
    print("配了 sequential_tools：", repr(safe))
    assert len(safe) > len(unsafe), "串行版本应该不丢内容"


asyncio.run(main())
```

**这道题说明了什么**：`sequential_tools` 不是可选的装饰，
它是**当工具不是并发安全时唯一正确的选择**。而且这个配置**默认是空的**——
框架不会替你猜"哪些工具不并发安全"，必须你自己配。

</details>

### 练习 3.2：给 Agent 加一个自定义回调做监控

**任务**：写一个回调，统计每个工具的调用次数和失败率，在 run 结束时打印报表。

**验收**：

```
=== 工具报表 ===
read_file    调用 2 次, 失败 0 次
write_file   调用 1 次, 失败 0 次
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from collections import defaultdict
from liteagent import Agent, AgentConfig, EventType, ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, tool


@tool
def read_file(path: str) -> str:
    """读文件。"""
    return "内容"


@tool
def write_file(path: str, content: str) -> str:
    """写文件。"""
    raise RuntimeError("磁盘满了")          # 故意失败，看报表能不能抓到


class ToolReporter:
    """统计每个工具的成功/失败次数。

    注意两个容易踩的点（都是实测确认的）：
      · 事件 data 里的工具名字段叫 `tool_name`，不是 `name`；
      · 失败时发的是 `tool_error` 事件，而不是 `tool_finished` + ok=False。
    """

    def __init__(self):
        self.calls = defaultdict(int)
        self.failures = defaultdict(int)

    def __call__(self, event):
        if event.type == EventType.TOOL_FINISHED:
            self.calls[event.data.get("tool_name", "?")] += 1
        elif event.type == EventType.TOOL_ERROR:
            name = event.data.get("tool_name", "?")
            self.calls[name] += 1
            self.failures[name] += 1

    def report(self) -> str:
        lines = ["=== 工具报表 ==="]
        for name in sorted(self.calls):
            lines.append(f"{name:12} 调用 {self.calls[name]} 次, 失败 {self.failures[name]} 次")
        return "\n".join(lines)


async def main():
    reporter = ToolReporter()
    llm = ScriptedLLM([
        ScriptedResponse.tool("read_file", {"path": "a.py"}),
        ScriptedResponse.tool("read_file", {"path": "b.py"}),
        ScriptedResponse.tool("write_file", {"path": "c.py", "content": "x"}),
        ScriptedResponse.text("完成了"),
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([read_file, write_file]),
                  config=AgentConfig(max_steps=6, repeat_action_policy="off"),
                  callbacks=[reporter])
    await agent.arun("读两个文件，然后写一个")
    print(reporter.report())


asyncio.run(main())
```

**要点**：`ToolReporter` 只需要是一个可调用对象（`__call__(event)`），
不需要继承任何基类——框架同时支持"有 `on_event` 方法的对象"和"单参可调用对象"两种形态。

</details>

---

## 4. 终极挑战

### 练习 4.1：给框架加一个新工具并端到端验证

**任务**：实现一个 `word_frequency` 工具（统计文本里词频最高的 N 个词），
然后写一个测试文件 `tests/test_my_tool.py`，用 `unittest` 风格覆盖：
① schema 生成正确；② 正常调用；③ 参数校验失败。

**验收**：

```bash
python3 -m unittest tests.test_my_tool -v
# 应该全绿，且至少 3 个用例
```

**提示**：参考 `tests/test_tools_schema.py` 的写法。注意：
- 文件第一行必须是 `from __future__ import annotations`；
- 异步用例继承 `unittest.IsolatedAsyncioTestCase`；
- 用 `self.assertEqual` 而不是裸 `assert`。

### 练习 4.2：故意破坏框架，看测试能不能抓住

**任务**：这是本项目的"变异测试"实践。挑一处实现，**故意改坏它**，
跑测试，看有没有用例变红。

建议试这几个（难度递增）：

| 变异 | 改哪里 |
| --- | --- |
| ① 让 `required` 判定永远返回 True | `tools/schema.py` 的 required 判定 |
| ② 去掉重试的 `idempotent` 检查 | `tools/executor.py` |
| ③ 让近因权重变成 0 | `memory/vector.py` 的默认权重 |
| ④ 去掉环检测的栈判断 | `multiagent/base.py` 的 `would_cycle` |

**做之前先备份，做完必须改回来！**

```bash
cp liteagent/tools/schema.py /tmp/schema.py.bak
# ...改坏...
python3 -m unittest discover -s tests -t . 2>&1 | tail -3
cp /tmp/schema.py.bak liteagent/tools/schema.py     # 恢复
```

**验收与思考**：

- 如果测试**变红**了 → 恭喜，这处分支被测到了；
- 如果测试**全绿** → **你发现了一个测试盲区**，这才是最有价值的收获。
  参考本项目已经发现的三处盲区（见 `docs/BUILD_LOG.md` 阶段 5）。

> 这道题的目的不是写代码，而是**建立"测试质量"的判断力**。
> 能问出"我的测试真的测到了吗"的人，和只会看覆盖率数字的人，是两个层次。

---

## 5. 综合自测（面试模拟）

合上所有文档，试着完整回答下面这些。答不上来的，回对应章节复习。

### 概念层

1. 用一句话解释 Agent 和普通程序的区别。
2. Agent 循环的四个阶段分别由谁负责？为什么边界必须这样划？
3. Function Calling 和文本 ReAct 的本质区别是什么？本项目怎么统一的？

### 机制层

4. `@tool` 装饰器怎么把 type hints 变成 JSON Schema？举三个边界情况。
5. 一份 JSON Schema 在框架里被用在**哪两个地方**？
6. ReAct 循环怎么防止无限循环？说出至少四层防护。
7. 长期记忆的写入为什么要限制 `role == "user"`？防的是什么？
8. 检索为什么要混合打分而不是纯向量相似度？MMR 解决什么？

### 工程层

9. `asyncio.Semaphore` 在 `__init__` 里创建会有什么问题？怎么解决的？
10. 同步工具超时后为什么不能重试？框架怎么"诚实"地报告这件事？
11. 为什么要零第三方依赖？降级时为什么必须留痕？
12. "变异测试"是什么？它和"覆盖率"有什么区别？

### 判断题（对/错，并说明理由）

13. 模型说它修改了文件，文件就一定改了吗？
14. `state.messages` 里包含用户的原始输入吗？
15. 多 Agent 一定比单 Agent 效果好？
16. 默认 embedder 能判断"你好"和"hi"语义相近？
17. `execute_many` 的返回结果顺序是完成顺序吗？

<details>
<summary>判断题答案</summary>

13. **错**。模型只能输出文本，文件是否被改由框架的工具执行决定；唯一可信的事实来源是工具的返回值。
14. **不含**。用户输入在 `state.input` 和记忆 buffer 里，组装 prompt 时才拼进去。
15. **错**。多 Agent 有上下文隔离和职责分离两个好处，但代价是通信开销、错误传播和调试难度；能用单 Agent + 好工具解决就别上多 Agent。
16. **错**。默认是哈希（词面）embedding，跨语言/同义词相似度为 0；要语义得换 `RemoteEmbedder`。
17. **不是**。返回列表严格按输入顺序对齐，与完成先后无关。

</details>

---

## 6. 学完之后能做什么

如果你完成了上面大部分练习，你应该能：

| 能力 | 具体表现 |
| --- | --- |
| **用** | 用 20 行代码搭一个能调工具的 Agent，并给它写工具 |
| **懂** | 对着任意一个模块，说清它解决什么问题、有哪些设计取舍 |
| **调** | Agent 行为异常时，通过事件流、`state`、`score_breakdown` 定位问题 |
| **改** | 新增工具/provider/协作模式，不会破坏既有架构（有守门测试保护） |
| **讲** | 面试时把"我做了个 Agent 框架"讲成 20 分钟的技术讨论，而不是 2 分钟的功能罗列 |

**下一步可以试试**：

- 给 `liteagent` 加一个新的 LLM provider（比如本地 Ollama）；
- 实现一种新的多 Agent 模式（比如"辩论式"：两个 Agent 互相质疑）；
- 把默认 embedder 换成真实的语义 embedding，对比检索效果；
- 给 CLI 加一个 `--replay` 子命令，从 trace 文件重放一次 run。

做完这些，你就不只是"读过这个项目"，而是**能改这个项目**了。

---

**回到**：[学习手册首页](README.md) ｜ **术语速查**：[附录 · 术语表](appendix-glossary.md)
