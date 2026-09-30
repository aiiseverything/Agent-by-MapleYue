# 第 8 章 · 工程化：异步、并发、可观测性、测试

> **本章目标**：这部分是"能跑的 demo"和"能用的框架"之间的差距。
> 学完你能：① 避开 asyncio 的三个经典陷阱；② 理解为什么要零依赖；
> ③ 知道怎么给 Agent 做可观测性和确定性测试。

---

## 8.1 零依赖：一个被约束倒逼出来的决策

### 8.1.1 事实

本项目**运行时零第三方依赖**——只用 Python 标准库。

`numpy` / `rich` / `requests` 都是**可选加速**，缺了会自动降级：

| 库 | 用途 | 缺了会怎样 |
| --- | --- | --- |
| `numpy` | 向量运算加速 | 退化成纯 Python 的余弦相似度 |
| `rich` | 终端彩色输出 | 退化成普通文本 |
| `requests` / `httpx` | HTTP 传输 | 退化成 `urllib.request`（stdlib） |
| `PyYAML` | 读 YAML 配置 | 只有 JSON 配置可用 |

**这不是"为了炫技"。** 起因是开发环境里没有外网、装不上包。
但约束一旦接受，收益就出现了：

1. **启动零成本**：不用 import 一堆库，CLI 秒开。
2. **不会被上游 breaking change 打断**：没有依赖就没有依赖升级。
3. **在任何 Python 环境里都能跑**：客户的内网机器、CI 容器、别人的笔记本。
4. **可读性**：你读到的每一行都是这个项目自己写的，没有"这是 LangChain 内部的魔法"。

### 8.1.2 怎么做到"可选依赖"

标准写法（**三方 import 必须在 `try` 里**）：

```python
try:                      # pragma: no cover - 环境相关
    import numpy as _np
    NUMPY_AVAILABLE = True
except ImportError:       # pragma: no cover
    _np = None
    NUMPY_AVAILABLE = False
```

然后在使用处分支：

```python
if NUMPY_AVAILABLE:
    return _np.dot(a, b)
return sum(x * y for x, y in zip(a, b))     # 纯 Python 兜底
```

**为什么暴露 `XXX_AVAILABLE` 常量？** 为了让测试能断言"降级路径也被覆盖到了"，
也为了让用户能查"我现在到底跑在哪条路径上"。

> **一条工程原则**：**降级必须留痕**。本项目把它写成红线：
> 任何降级（schema 退化成 `{}`、embedding 用哈希、摘要用抽取式、搜索用空后端、
> 上下文被裁剪、工具被熔断）**都必须有可观测的痕迹**（warning / 事件 / 日志），
> **不许静默**。
>
> 因为静默降级会让系统"看起来在工作"，直到某天你发现检索结果全是垃圾。

---

## 8.2 asyncio 的三个经典陷阱（本项目的血泪教训）

这三个坑都是**实测复现过**的，写进了项目规范。

### 8.2.1 陷阱一：asyncio 原语绑定事件循环

**症状**：同样的代码，第一次调用正常，**第二次调用崩溃**：

```
RuntimeError: <asyncio.locks.Semaphore object> is bound to a different event loop
```

**原因**：`asyncio.Semaphore` 在**第一次发生争用**时，会把当前的事件循环
存进 `self._loop`。而本框架的同步 API（`agent.run()`）每次调用都会
`asyncio.run(...)` 创建一个**新的**事件循环。于是：

```python
executor = ToolExecutor(...)          # __init__ 里创建了 asyncio.Semaphore
asyncio.run(use(executor))            # 第一次：OK（并绑定了 loop #1）
asyncio.run(use(executor))            # 第二次：RuntimeError！（sem 还绑在 loop #1）
```

**最坑的地方**：它**不是每次都崩**——只有在"发生争用"（即真的有两个任务同时抢信号量）时
才会绑定 loop。所以单元测试可能全绿，线上偶发崩溃。

**解法：`LoopBoundPool`**

所有 asyncio 原语**按事件循环懒创建**：

```python
pool = LoopBoundPool()
sem = pool.semaphore("exec", 4)      # 在运行中的 loop 内调用，每个 loop 一套
```

两条不变式：
1. **必须在运行中的 loop 内调用**，否则抛 `ConfigError`；
2. 所有同步入口的 `finally` 里必须 `release(loop)` 清理。

**还有一个更隐蔽的后续问题**：第一版用 `WeakKeyDictionary[loop]` 来缓存，
以为 loop 用完会被回收。实测发现**它反而泄漏**——因为"原语强引用 loop"，
value 强引用 key，**弱引用的 key 永远不会失效**。每 3 次 `asyncio.run` 泄漏 3 个 loop。

> **面试可以这么讲**：这个 bug 的价值在于它**不是每次都复现**——
> 只有发生争用时才绑定 loop，所以单测全绿、线上偶发。
> 我最后用"按 loop 懒创建 + 显式清理 + 测试断言不泄漏"三件事解决的。
> 顺便说，我第一版用 `WeakKeyDictionary` 缓存，实测发现它反而泄漏。

### 8.2.2 陷阱二：`to_thread` 静默吞掉协程

**症状**：工具"看起来执行了"，但什么都没发生，**而且没有任何异常**。

**原因**：

```python
asyncio.to_thread(async_fn, args)     # ← 不会执行 async_fn！
```

`to_thread` 会把 `async_fn` 当普通可调用对象在线程里调用——而调用一个 async 函数
只会**创建一个协程对象**，然后这个对象被直接丢弃。没有 `await`，没有任何警告。

**解法**：同步工具走同步入口，异步工具直接 await：

```python
if tool.spec.is_async:
    coro = tool.arun(args)                  # 直接 await
else:
    coro = run_in_executor(tp, tool.run, args)   # 同步入口，丢线程池
```

并且 `tool.run()` 对异步工具会**主动抛 TypeError**，把静默失效变成大声报错。

> 这类 bug 的教训：**"看起来在工作"比"崩溃"更危险**。

### 8.2.3 陷阱三：`contextvars` 的读能穿透、写不能回传

**实测结论**：

| 操作 | 能否穿透 `to_thread` / 线程池 |
| --- | --- |
| **读** contextvar | ✅ 能（worker 线程看得到调用方设置的值） |
| **写** contextvar | ❌ 不能（子上下文里的 `set` 父上下文看不到） |

**影响**：

1. executor 把同步工具丢线程池时，必须**显式复制上下文**：
   ```python
   ctx = copy_context()
   run_in_executor(tp, ctx.run, tool.run, args)
   ```
   不复制的话，worker 线程里读到的取消标志**恒为 None**，协作式取消完全失效。
2. 子 Agent 的委派上下文必须**显式携带**（通过 `scratchpad`），不能指望自动回传。

> **实用建议**：`contextvars` 的传播规则在 Python 里很反直觉，
> **不要凭记忆写，实测一下**。

### 8.2.4 还有一个：`wait_for` 会吞掉取消

本项目没有用 `asyncio.wait_for` 做超时，而是**手写了一个 `_await_with_deadline`**。
原因（代码注释）：

> Python 3.10 的 `asyncio.wait_for` 会在"内层 future 在同一 tick 已完成"时
> **吞掉调用方的取消**（CPython issue GH-86296），
> 让 `Agent.astream` 的 `task.cancel()` 变成空操作。

**这是很典型的"标准库有坑"的经历**——它逼着你理解底层机制，而不是无脑调用 API。

---

## 8.3 可观测性

一个 Agent 跑出问题了，你怎么知道它在哪一步、为什么出错？

### 8.3.1 事件总线

第 5 章列过 27 种事件。机制上：

```python
agent = Agent(llm=..., callbacks=[cb1, fn2])         # 构造时挂
agent.callbacks.subscribe(fn)                        # 运行期挂，返回取消函数
await agent.arun(input, callbacks=[...])             # 只对这一次 run 生效
```

**一条重要的容错契约**：**回调抛异常绝不影响主流程**。
单个回调炸了，会被记进 `manager.errors` 并记日志，其余回调继续。

> 这个设计是对的：**观测代码不应该能搞挂业务代码**。

### 8.3.2 内置的几个回调

| 回调 | 用途 |
| --- | --- |
| `LoggingCallback` | 打到标准日志（错误类事件用 WARNING，其余用 INFO） |
| `JsonlTraceCallback` | 逐行写 JSONL 文件（**多线程写不丢行**） |
| `RichCallback` | 终端彩色输出（有 rich 时好看，没有则降级） |
| `TokenCounterCallback` | 累计 token 和费用 |
| `MemoryTraceCallback` | 只收集记忆相关事件 |

### 8.3.3 `trace_stats`：一次 run 的全景

```python
from liteagent import TraceRecorder, trace_stats
from liteagent.agent import as_llm_callback

recorder = TraceRecorder("trace.jsonl")
llm.on_event = as_llm_callback(recorder.manager)      # ← 别忘了这一行！见下
agent = Agent(llm=llm, tools=registry, callbacks=[recorder.manager.emit])
...
result = await agent.arun("...")
stats = trace_stats(recorder.events)
```

真实输出（旗舰示例）：

```
事件总数          : 53
steps             : 6
llm_calls         : 6
tool_calls        : 5   (按 call_id 去重)
tool_failures     : 0
usage             : {'prompt_tokens': 60, 'completion_tokens': 30, 'total_tokens': 90}
llm_latency_ms    : total=120.0, mean=20.0, p50=20.0, p95=20.0
tool_latency_ms   : {'read_file': {'count': 2, 'mean': 1.58, 'p95': 1.18},
                     'write_file': {'count': 1, ...},
                     'run_tests': {'count': 1, 'mean': 105.35, ...}}
errors            : {}
```

**注意 `run_tests` 的平均耗时是 105ms，而 `read_file` 只有 1.58ms**——
这种数据能直接告诉你"哪里是性能瓶颈"。

### 8.3.4 一个必须知道的坑

`Agent` **只给自己创建的 memory/executor 挂事件转发，不会改你传进来的 `llm` 实例**。

所以如果你不写 `llm.on_event = as_llm_callback(agent.callbacks)`，
trace 里**永远不会出现 `llm_request` / `llm_response`**，
`trace_stats()` 的 `llm_calls` 和 `llm_latency_ms` 会一直是 0。

> 为什么这样设计？因为**不该擅自修改别人传进来的对象**——
> 那个 client 可能被多个 Agent 共享，改了会互相干扰。宁可让你显式接线。

---

## 8.4 离线确定性测试

### 8.4.1 核心工具是 `ScriptedLLM`（第 3 章已详述）

一句话总结它的价值：

> **因为框架对 LLM 的依赖被收敛在 `LLMClient` 这一个接口上，
> 假模型就是一个"合法的一等实现"，而不是测试补丁。**

于是整个 Agent 循环（含多 Agent 协作）都能离线做端到端测试。

### 8.4.2 为什么用 `unittest` 而不是 `pytest`

**直接原因**：环境里没有 pytest，且装不上。

**但结果是好的**：

| 影响 | 说明 |
| --- | --- |
| 零依赖测试 | 任何人 clone 下来直接能跑，不用先建虚拟环境装东西 |
| 没有魔法 | 没有 fixture 的隐式注入、没有插件行为，读测试就是读普通 Python |
| 强制显式 | 异步测试必须显式用 `IsolatedAsyncioTestCase`，一目了然 |

### 8.4.3 测试规模

```
$ python3 -m unittest discover -s tests -t .
Ran 1652 tests in 24s
OK
```

- **1652 个用例**，37 个测试文件；
- **全程离线**，不需要 API key；
- **24 秒跑完**（因为不需要真网络调用）。

### 8.4.4 测试里几条重要的纪律

| 纪律 | 原因 |
| --- | --- |
| **不真睡** | 退避等待用 `RecordingSleep`（只记录不睡），否则测试要跑几分钟 |
| **不用真实时钟** | 时间戳断言用 `frozen_time` 打桩，否则输出不可复现 |
| **不真联网** | HTTP 一律用 `FakeTransport` |
| **不污染全局状态** | 用了 `auto_register` 的测试必须在 `tearDown` 调 `reset_default_registry()` |
| **不断言不稳定字段** | 规范里有一份"禁止断言的字段清单"（如耗时、时间戳） |

### 8.4.5 "全绿不等于被测到"——变异测试

这是本项目最有价值的测试实践。**手动改坏一行实现代码，看有没有测试变红。**

项目在文档里如实记录了三处**曾经没有被测试抓住**的分支：

| 变异 | 结果 |
| --- | --- |
| 反转 `parser.py` 的整段 JSON 解析路径 | 1652 个测试**全绿** ← 说明没覆盖 |
| 删掉 `schema.py` 里嵌套 dataclass 的 `default_factory` 判定 | 全绿 ← 说明没覆盖 |
| 删掉 `registry.py` 里 `unregister` 的别名清理 | 全绿 ← 说明没覆盖 |

发现之后**补了回归测试**，再变异一次确认变红：

```
M1: parser.py   `if self.allow_json_object:` → `if False:`          → FAILED (errors=7)  ✓
M2: schema.py   删掉 default_factory 判定                             → FAILED (failures=2) ✓
M3: registry.py 删掉 unregister 的别名清理循环                        → FAILED (failures=3) ✓
```

> **面试可以这么讲**：测试"全绿"只说明现有断言都成立，不说明实现对。
> 我用手动变异测试去检验测试本身的有效性，结果确实发现三处关键分支
> **删掉也不会让任何测试变红**——这直接证明了"覆盖率"这个指标是会骗人的。
> 补上回归测试之后，同样的变异会立刻失败。
>
> 这段比"我写了 1652 个测试"有说服力得多。

---

## 8.5 项目里的其他工程细节

### 8.5.1 时间只有唯一来源

所有"取当前时间"都必须走 `config.utc_now`，**禁止直接 `datetime.now()`**。

理由：**可测试性**。只有唯一入口，测试里才能一处打桩就控制全局时间。
（实测有对应的检查：`liteagent/` 里没有 `datetime.now`、没有 `time.time`。）

### 8.5.2 随机数只用实例级 `random.Random`

**禁止模块级 `random.random()`**——那样退避序列不可复现，测试没法断言。
每个组件在 `__init__` 里建一个 `random.Random(seed)` 并复用。

### 8.5.3 异常的结构化

每个异常都带 `context` 字典，`to_dict()` 能序列化：

```python
ToolValidationError("...", errors=["$.a: expected integer"], tool_name="add").to_dict()
# {'type': 'ToolValidationError', 'message': '...', 'context': {...}, 'retryable': False, ...}
```

**为什么？** 因为异常要进 trace、要写日志、要能回灌给模型。
结构化之后，"按异常类型统计"这种事才能做（`trace_stats` 的 `errors` 字段就是这么来的）。

### 8.5.4 结果截断保留头尾

第 4 章讲过，这里强调设计思想：**保留尾部**。
因为错误信息、汇总数字、总结通常在末尾。**只截头会丢最关键的诊断信息。**

### 8.5.5 `run_sync` 收的是工厂函数而不是协程

```python
def run(input: str, **kwargs) -> AgentResult:
    return run_sync(lambda: self.arun(input, **kwargs))    # ← lambda，不是调用结果
```

传协程对象的话，一旦没被 await 就会有"never awaited"警告，
而且**协程永远不会执行**——一个静默的 bug。

---

## 8.6 本章小结

1. **零依赖是被约束倒逼的决策**，但收益是真实的：启动快、不会被打断、可读。
2. **降级必须留痕**——静默降级会让系统"看起来在工作"。
3. **asyncio 三个坑**：原语绑定 loop（第二次调用才崩）、`to_thread` 吞协程（静默失效）、
   `contextvars` 写不回传。
4. **可观测性靠事件总线**，且回调异常绝不影响主流程。
5. **`ScriptedLLM` 让离线确定性测试成为可能**，1652 个用例 24 秒跑完。
6. **"全绿不等于被测到"**——用变异测试检验测试本身。

### 自测题

1. 为什么 `asyncio.Semaphore` 在 `__init__` 里创建会在第二次 `asyncio.run` 时崩溃？
2. `asyncio.to_thread(async_fn)` 会发生什么？为什么这个 bug 特别危险？
3. `contextvars` 的读和写在跨线程时行为有什么不同？这影响了哪两处实现？
4. 为什么测试用 `unittest` 而不是 `pytest`？这带来了什么好处？
5. "变异测试"是什么？本项目用它发现了什么问题？
6. 为什么所有时间必须走 `config.utc_now`？

<details>
<summary>参考答案</summary>

1. 原语在第一次发生争用时把当前 loop 存进自身；新的 `asyncio.run` 是另一个 loop，于是抛 "bound to a different event loop"。
2. 它只创建协程对象就丢掉了，函数不会执行，而且没有任何异常——"看起来在工作"比崩溃更难排查。
3. 读能穿透，写不能回传；影响了 executor 丢线程池时必须显式复制上下文、以及委派上下文必须显式携带。
4. 因为环境里装不上 pytest；好处是零依赖、没有魔法（无 fixture 隐式注入）、异步测试必须显式声明。
5. 手动改坏实现代码看测试是否变红；发现了三处关键分支（整段 JSON 解析、嵌套 dataclass 的 `default_factory` 判定、`unregister` 的别名清理）删掉后 1652 个测试仍然全绿。
6. 为了可测试性——唯一时间入口才能在测试里一处打桩控制全局。

</details>

---

**下一章**：[第 9 章 · 项目导览与阅读顺序](09-code-map.md)
