# liteagent —— 零依赖的轻量级 Agent Harness

> **一句话定位**：一个**零第三方依赖**（stdlib only）、**离线可测、确定性可复现**的 LLM Agent 开发框架，
> 用「LLM 抽象层 / 工具系统 / 记忆管理」三层架构 + 一个显式 ReAct 状态机，跑通单 Agent 推理与多 Agent 协作。

```
Python 3.10  |  Zero Third-Party Dependency  |  41 modules / 24,765 lines  |  ReAct + Function Calling
3-Layer Memory (window / summary / vector)  |  Sequential + Hierarchical Multi-Agent  |  Offline & Deterministic
```

> 上面那行是纯文本 badges（刻意不引外链图片：本项目在无外网环境下开发，README 里不放会 404 的图）。

---

## 1. 核心特性

按简历三条 bullet 的三个能力域组织。每条给出**真实模块路径**与一句实现要点。

### 1.1 LLM 抽象层 —— 统一多模型 API

| 模块 | 实现要点 |
|---|---|
| `liteagent/llm/base.py` | `LLMClient` ABC + `BaseLLMClient`：把「重试 / 事件发射 / token 计数 / 流式」收敛到基类，子类只写「怎么发一个请求、怎么解析一个响应」 |
| `liteagent/llm/providers.py` | 四个 provider：`OpenAIChatClient` / `OpenAICompatibleClient` / `DeepSeekChatClient` / `AnthropicChatClient`；OpenAI 系共用 `HTTPChatClient` 基类，Anthropic 单独处理 system 提取与 tool_result 合并 |
| `liteagent/llm/transport.py` | `Transport` ABC 隔离 HTTP：urllib（永远可用）/ requests / httpx 三种适配器 + 统一错误映射（401/403/429/4xx/5xx → 异常树） |
| `liteagent/llm/registry.py` | `get_llm("openai:gpt-4o-mini")` 一行式装配，spec 语法 `provider[:model][@base_url]` |
| `liteagent/llm/scripted.py` | `ScriptedLLM`：**离线确定性地基**。按预设脚本逐条吐 `LLMResponse`，零网络零随机，还能 `assert_exhausted()` 自检循环次数 |
| `liteagent/llm/message.py` | `Role` / `Message` 及其序列化、`drop_orphan_tool_messages` 等协议规整函数 |

**要点**：两种「模型怎么表达要调工具」的形态 —— 原生 function calling 与文本 ReAct —— 被**归约到同一份 `list[ToolCall]`**，
所以下面的 ReAct 状态机是**唯一**的控制流，两种模式只是它的两个入口（见 D-02）。

### 1.2 工具系统 —— 装饰器自动注册 + JSON Schema 生成

| 模块 | 实现要点 |
|---|---|
| `liteagent/tools/schema.py` | **反射生成 JSON Schema**：从 type hints 生成 `parameters`，从 docstring（Google / Sphinx 风格）生成 `description`；另有纯 stdlib 的 `validate_instance` 校验器 |
| `liteagent/tools/base.py` | `@tool` 装饰器 / `ToolSpec` / `Tool`；装饰器默认**不**自动注册（见 D-03），需要全局可见时显式 `@tool(auto_register=True)` |
| `liteagent/tools/registry.py` | 注册 / 重名检查 / 别名 / 子集 / 导出为 OpenAI 或 Anthropic 格式的 schema / 导出为给模型看的文本清单 |
| `liteagent/tools/executor.py` | `ToolExecutor`：审批（fail-closed）、入参校验、**并发**（顺序严格对齐）、超时、**重试（default-deny）**、结果截断、协作式取消、熔断 |
| `liteagent/tools/builtin/` | 内置工具：`files.py`（沙箱文件读写）、`shell.py`（带 denylist 的受控 shell）、`code.py`（AST 白名单 `python_eval` + 子进程 `python_exec` + `run_tests`）、`web.py`（`web_search` / `fetch_url`）、`memory_tools.py`（`remember` / `recall`） |

**要点**：写一个工具不需要手写 JSON Schema —— 注解 + docstring 就是说明书，**同一份 schema 既给模型看、又用来在执行前校验入参**。
边界（一句话）：无法精确表达的注解（`Union[A, B]`、`Mapping[...]`、`Callable[...]` 等）会**降级成 `{}`（any）** 并记进 `ToolSpec.warnings`（可用 `python3 -m liteagent tools show <name>` 看到）—— 此时模型被告知的是 `any`，执行前校验对那个参数**恒真**（即不做类型校验）。详见 `docs/TOOLS.md` 的降级表。
`[v3]` 这条降级**不再静默**：执行器会把"哪些参数根本没被校验覆盖"写进 `ToolResult.metadata["validation_gaps"]` 并发一条 WARNING —— 同一类留痕也覆盖另一条路径（provider 侧 JSON 解析失败把整份 arguments 丢成 `{}`：required 字段仍会拦，但全 optional 的工具会以空参数执行，此前没有任何信号）。

### 1.3 记忆管理 —— 短期对话历史 + 长期向量存储

| 模块 | 实现要点 |
|---|---|
| `liteagent/memory/buffer.py` | `BufferMemory`：滑动窗口 + 双约束（条数 + token 预算）裁剪，含 `_repair_tool_pairs` 修复被裁断的 tool_call/tool 配对 |
| `liteagent/memory/summary.py` | `SummaryMemory`：触发判定 → LLM 摘要 → **抽取式兜底**（LLM 失败不抛异常，降级为抽取） |
| `liteagent/memory/vector.py` | `VectorMemory`：混合打分（相似度 + 近因衰减 + 重要度）+ MMR 去冗；`save` / `load` 支持**跨进程持久化**（落盘 JSONL 含 embedding，另一进程可直接 load 并复现检索顺序；**测试覆盖只到同进程往返**，见 `docs/VERIFICATION.md` §3 U-9 / §4 G-2） |
| `liteagent/memory/embeddings.py` | `Embedder` ABC：`HashingEmbedder`（纯 stdlib，默认）/ `NumpyHashingEmbedder`（仅 `NUMPY_AVAILABLE` 时定义）/ `RandomProjectionEmbedder` / `CallableEmbedder` / `RemoteEmbedder` |
| `liteagent/memory/manager.py` | `MemoryManager`：三层协作的**唯一入口**，负责把 6 个段落组装成 prompt 并强制上下文预算 |

**要点**：默认 embedder 是纯 stdlib 的 hashing trick，能力边界（词面相似 ≠ 语义相似）**写进了文档并有实测数据**（见 D-04 与 §7）。

### 1.4 ReAct 循环 —— 完整的 Thought-Action-Observation

| 模块 | 实现要点 |
|---|---|
| `liteagent/agent/agent.py` | `Agent.arun`：显式状态机 `THINK → ACT → OBSERVE → …`，含自纠正、重复/无进展检测、预算与墙钟截断 |
| `liteagent/agent/parser.py` | `ReActParser`：文本 ReAct 语法的容错解析（跨行 JSON、```json 围栏、全角冒号、中文 marker、Action 优先于 Final Answer） |
| `liteagent/agent/state.py` | `AgentState` / `AgentStatus` / `AgentResult`；`AgentResult` 是**唯一**对外结果契约（不抛异常，失败编码在对象里） |
| `liteagent/agent/callbacks.py` | `EventType` / `TraceEvent` / `CallbackManager` / `TraceRecorder` / `trace_stats`；可写 JSONL trace 并用 CLI 回放 |

**要点**：`finish_reason` **参与控制流** —— `length` 触发截断续写、`content_filter` 立即失败、
`tool_calls` 但列表为空触发自纠正（见 D-14）。循环防护分三层：`canonical_key` 去重 / observation 摘要无进展检测 / 熔断（见 D-16）。

### 1.5 多 Agent 协作 —— 两种模式 + 任务分解

| 模块 | 实现要点 |
|---|---|
| `liteagent/multiagent/sequential.py` | `SequentialAgent`：A → B → C 流水线，`input_template` 支持 `{input}` / `{prev}` / `{steps[x]}`，`propagate_failure` 三种失败策略 |
| `liteagent/multiagent/hierarchical.py` | `HierarchicalAgent`：主 Agent 规划 → 拆出 `Plan`/`SubTask` → 通过自动生成的 `delegate_to_<worker>` 工具委派给子 Agent；含环检测、`max_depth`、预算；`arun_plan` 按 `depends_on` 并发执行 |
| `liteagent/multiagent/blackboard.py` | `Blackboard`：`threading.RLock` 保护的共享黑板，支持版本冲突检测（`if_version`）、TTL、订阅 |
| `liteagent/multiagent/base.py` | `MultiAgent` ABC / `TeamConfig` / `DelegationContext`（用 `contextvar` 做隔离）/ 子 Agent 输出压缩 |

**要点**：子 Agent 就是普通 `Agent`；委派上下文通过 `DelegationContext` 传递，
子 Agent 的输出会被**头尾保留地压缩**后回灌给主 Agent，避免上下文被长输出挤爆。

---

## 2. 快速开始

### 2.1 零依赖，无需 pip install

```bash
# 克隆后直接在仓库根目录运行即可，不需要 pip install，也不联网
cd project-3
python3 -m liteagent version      # -> 0.1.0
python3 -m liteagent tools list   # 列出内置工具及其 tags / dangerous / approval 标记
```

唯一的运行要求是 **Python >= 3.10**（本机实测 3.10.12）。
`liteagent/` 下所有模块的**顶层 import 只允许 stdlib**，这条红线由守门测试 `tests/test_zero_dependency.py`
用 AST 静态断言（不是靠自觉）。`pyproject.toml` 里的 `dependencies = []` 是空列表；
numpy / rich / requests / PyYAML 全是 `optional-dependencies`，框架在 import 时探测，
缺席就静默回退到纯 stdlib 实现（如纯 Python 余弦相似度、纯文本 ANSI 渲染）。

### 2.2 最小可运行示例（约 15 行，离线，不需要 API key）

下面这段从 `examples/01_quickstart.py` 摘出，是我**本机实跑通过**的最小骨架（返回 `FINISHED The answer is 42. 30`）。

```python
# demo.py —— 放在仓库根目录，然后 `python3 demo.py`
from __future__ import annotations

import asyncio

from liteagent import Agent, AgentConfig, ScriptedLLM, ScriptedResponse, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


async def main() -> None:
    llm = ScriptedLLM([
        ScriptedResponse.tool("add", {"a": 2, "b": 40}),   # 第 1 轮：模型要求调用 add
        ScriptedResponse.text("The answer is 42."),        # 第 2 轮：给出最终答案
    ])
    agent = Agent(llm=llm, tools=ToolRegistry([add]), config=AgentConfig(max_steps=5))
    result = await agent.arun("What is 2 + 40 ? Use the add tool.")
    print(result.status.value, result.output, result.usage.total_tokens)


asyncio.run(main())
```

三点值得注意：

- **`ScriptedLLM` 不是玩具**。它消费的队列格式与真实 provider 返回的 `LLMResponse` 完全一致，
  所以「离线跑通」和「联网跑通」走的是**同一套循环代码**，切换只需换掉 `llm=` 这一个参数。
- **JSON Schema 不用手写**。上面 `@tool` 会反射 `a: int` / `b: int` 生成 `parameters`，
  把 Google 风格 docstring 的 `Args:` 段拆成逐参 `description`，并把无默认值的参数列进 `required`。
- **想看真实模型**：把 `ScriptedLLM(...)` 换成 `get_llm("openai:gpt-4o-mini")` 并设置 `OPENAI_API_KEY` 即可。
  但请注意：**本环境没有外网，这条路径没有在本机验证过**（详见 §8）。

---

## 3. 架构图

分层与依赖方向（依赖**只允许向下**，包内自由、包间按冻结表白名单）：

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  L6  cli.py        liteagent run / chat / tools / trace / schema / multi     │
│                    main(argv) -> int （不 sys.exit，便于单测）                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  L5  multiagent/   MultiAgent(ABC) / Blackboard / SequentialAgent /           │
│                    HierarchicalAgent（manager + delegate_to_<worker>）        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  子 Agent 就是普通 Agent；上下文走 DelegationContext
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  L4  agent/        ★ ReAct 状态机（全框架唯一的控制流）                        │
│                    state.py / callbacks.py / parser.py / agent.py            │
│                    Agent.arun = THINK → ACT → OBSERVE → THINK ...            │
└───────┬───────────────────────┬────────────────────────┬─────────────────────┘
        │                       │                        │
┌───────▼─────────┐   ┌─────────▼──────────┐   ┌─────────▼──────────────────┐
│ L2 llm/         │   │ L3 tools/          │   │ L3 memory/                 │
│ LLMClient(ABC)  │   │ ToolSpec / Tool    │   │ MemoryStore(ABC, sync)     │
│  ├ OpenAI       │   │ ToolRegistry       │   │  ├─ BufferMemory  (窗口)   │
│  ├ Anthropic    │   │ ToolExecutor       │   │  ├─ SummaryMemory (摘要)   │
│  ├ DeepSeek     │   │  ├ 校验/审批        │   │  └─ VectorMemory  (长期)   │
│  ├ Echo         │   │  ├ 并发/超时/重试   │   │ Embedder(Hashing/Numpy/    │
│  └ Scripted ★   │   │  └ 取消/熔断        │   │          Callable/Remote)  │
│ Transport(ABC)  │   │ builtin/*.py       │   │ MemoryManager（唯一入口）  │
│  urllib/req/htx │   │  files/shell/code  │   │ Tokenizer(中英分离启发式)  │
└───────┬─────────┘   └─────────┬──────────┘   └─────────┬──────────────────┘
        │                       │                        │
┌───────▼───────────────────────▼────────────────────────▼─────────────────────┐
│  L1  config.py   AppConfig/AgentConfig/ExecutorConfig/MemoryConfig/TeamConfig │
│                  RetryPolicy / LoopBoundPool / run_sync / utc_now /          │
│                  to_jsonable / render_template / estimate_cost_usd           │
│      types.py    TokenUsage / ToolCall / ToolResult / LLMResponse            │
│      llm/message.py  Role / Message                                          │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  L0  errors.py   LiteAgentError 树（retryable 是类属性 = 重试策略表）          │
└──────────────────────────────────────────────────────────────────────────────┘
```

三条贯穿全框架的运行时约束（都是**实测踩过的坑**，不是理论洁癖）：

| 约束 | 内容 | 违反的后果（实测） |
|---|---|---|
| **R-LOOP** | 禁止在 `__init__` / 模块级创建 `asyncio.Semaphore/Lock` 等；必须经 `LoopBoundPool` 在运行中的 loop 内懒创建 | `Agent.run()` 第二次调用直接 `RuntimeError: bound to a different event loop` |
| **mutex = threading** | 跨线程/跨 loop 的限流与互斥一律用 `threading` 原语 | delegate 工具跑在 worker 线程的**新 loop** 里，loop-bound 原语完全不受约束，并发失控 |
| **取消优先** | 任何宽 `except` 的首行保证 `CancelledError` 不被吞；超时只认 `except asyncio.TimeoutError` | `except TimeoutError` 在 3.10 抓不到 `asyncio.TimeoutError`（实测二者不同） |

---

## 4. 目录结构（简版）

```text
project-3/
├── liteagent/                  # 框架本体，41 个 .py（封闭清单，见 INTERFACES.md §1.2）
│   ├── errors.py               # L0  异常树，retryable 是类属性
│   ├── types.py  config.py     # L1  核心数据类 + 跨层纯函数
│   ├── llm/                    # L2  LLM 抽象层（base/providers/transport/registry/scripted/message）
│   ├── tools/                  # L3  工具系统（schema/base/registry/executor + builtin/）
│   ├── memory/                 # L3  记忆层（base/buffer/summary/vector/manager/embeddings）
│   ├── agent/                  # L4  ReAct 状态机（agent/state/parser/callbacks）
│   ├── multiagent/             # L5  协作编排（base/blackboard/sequential/hierarchical）
│   ├── cli.py                  # L6  argparse 子命令
│   └── __main__.py             #     `python3 -m liteagent` 入口
├── tests/                      # 37 个 test_*.py + helpers.py + __init__.py（stdlib unittest）
├── examples/                   # 7 个示例 + run_all_examples.py（共 8 个文件）
├── benchmarks/                 # 3 个微基准脚本（结果自动写进 docs/VERIFICATION.md）
├── scripts/                    # check_spec_consistency.py（规范一致性检查，不参与测试）
├── docs/                       # INTERFACES / ARCHITECTURE / DESIGN_DECISIONS / VERIFICATION / BUILD_LOG
├── Makefile                    # make test / demo / lint / tree
└── pyproject.toml              # dependencies = []（核心零依赖）
```

---

## 5. 测试

测试**全部**用标准库 `unittest` 写成（没有 pytest），**离线、确定性、不需要 API key**。

```bash
# 方式一：完整（Makefile 的 test 目标串联了两条命令）
make test

# 方式二：直跑（推荐，输出更干净）
python3 -m unittest discover -s tests -t .

# 只看汇总
python3 -m unittest discover -s tests -t . 2>&1 | tail -3
```

**本机实测结果（[v3 刷新]，Python 3.10.12）**：

```text
Ran 1652 tests     ->  OK
```

| 指标 | 实测值 |
|---|---|
| 测试方法总数（`ast` 统计 `test_*`） | **1652** |
| 测试文件数 | **37** 个 `test_*.py` |
| 全套耗时 | **约 30 秒**（本机两次重跑实测 **29.7s / 31.2s**；共享机器，负载不同会波动） |
| 结果 | **1652 通过 / 0 失败** |

**关于「那 1 个失败」（`[v3]` 已经不存在了，但过程值得留）**：
上一版这里写着「守门测试 `FrozenLayoutTests::test_all_frozen_test_files_exist` 故意保持红灯」，
并列出了当时还没落地的 4 个测试文件（`test_e2e_code_assistant` / `test_examples_offline` /
`test_examples_import` / `test_docs_coverage`）。那 4 份文件随后都写出来了，守门测试**自动转绿**
—— 这正是这条红线该有的闭环：**先写下缺口、再补上文件、由断言验收**，
而不是靠删断言换一个好看的绿色。

保留这段历史有两个原因：一是它展示了「诚实台账 → 补齐 → 转绿」的过程；
二是它是一个教训 —— 台账写完之后**必须重跑刷新**，否则「当前有 1 个失败」会变成一句
一敲命令就被推翻的假陈述（这一批数字在 v3 里已经全部重跑过）。

为什么这套测试值得信：

- **离线**：真实网络被 `FakeTransport`（可编程假传输层）替代，请求构造与响应解析是**真的被执行**的，
  只有 socket 是假的；`web_search` 走 `FakeSearchBackend`。
- **确定性**：时间通过 `frozen_time` / `now=` 注入、随机性通过 `rng_seed` 固定、
  退避的 `sleep_fn` 用 `RecordingSleep`（只记录不真睡），所以「重试 2 次、间隔 0.25s/0.5s」是可断言的。
- **循环次数的正确性**靠 `ScriptedLLM.assert_exhausted()` 守：多调一次 LLM 会耗尽队列抛错，
  少调一次会断言失败 —— 换成真模型就没有这个钩子了，所以这个自检只存在于离线测试里。

---

## 6. 示例

`examples/` 下共 **8 个文件** = 7 个编号示例 + 1 个批量运行器。全部支持 `--offline`，默认即离线。

| 示例 | 一句话说明 | 离线运行命令 |
|---|---|---|
| `01_quickstart.py` | 最小可运行的 ReAct Agent；结尾把「不可省的 4 行」单独打出来 | `python3 examples/01_quickstart.py --offline` |
| `02_tools_custom.py` | 自定义工具：展示 type hints → JSON Schema 的完整映射（Optional / Literal / list[str] / 嵌套 dataclass） | `python3 examples/02_tools_custom.py --offline` |
| `03_react_text_mode.py` | 文本 ReAct 模式：打印完整 Thought / Action / Observation 轨迹，与 01 的原生模式对照 | `python3 examples/03_react_text_mode.py --offline` |
| `04_memory.py` | 三层记忆的最小教案：窗口 / 摘要 / 向量各解决什么问题 | `python3 examples/04_memory.py --offline` |
| `05_multiagent_sequential.py` | Sequential 流水线：研究员 → 写作者 → 审校者，共享黑板 | `python3 examples/05_multiagent_sequential.py --offline` |
| `06_multiagent_hierarchical.py` | Hierarchical 编排：主 Agent 拆任务、委派给下属、汇总 | `python3 examples/06_multiagent_hierarchical.py --offline` |
| `07_code_assistant.py` | **旗舰示例**：端到端代码助手，真的读文件 → 改文件 → 在子进程里跑测试 | `python3 examples/07_code_assistant.py --offline` |
| `run_all_examples.py` | 批量运行器：以**子进程**方式依次跑 **01、03–07 共 6 个**示例（每个加硬超时），汇总退出码并打印 PASS/FAIL 表 | `python3 examples/run_all_examples.py` |

也可以直接：

```bash
make demo        # 等价于 python3 examples/07_code_assistant.py --offline
```

`run_all_examples.py` 的 `TARGETS` 列表**不含 `02`**，代码里的注释写的是「02 需要真实网络」。
我把这个记在这里而不是照抄：**实测 `python3 examples/02_tools_custom.py`（不带 `--offline`）退出码是 0**，
输出自己声明「离线，ScriptedLLM 驱动」；02 的源码注释也写着「不写 `--offline` 也是离线」。
也就是说**那行注释与实测行为不一致**（02 只有显式传 `--provider` 才会联网）。
本轮任务限定只改 `README.md`，所以我**没有动** `run_all_examples.py`，只把差异如实记在这里。
我实跑 `run_all_examples.py` 的结果是：`共 6 个示例：PASS 6，FAIL 0，TIMEOUT 0，MISSING 0；总耗时 6.94s`
（两次重跑 6.94s / 7.58s，随机器负载波动）。

`examples/07_code_assistant.py --offline` 我**本机实跑通过**（退出码 0），它自己会做 4 条断言：
文件**真的**被改写了、测试**真的**在子进程里跑过。它跑完打印的 trace 统计是
`steps=6, llm_calls=6, tool_calls=5, tool_failures=0`，沙箱用的是 `tempfile.TemporaryDirectory()`，
所以**永远碰不到仓库或 home 下的真实文件**（`PathSandbox` 是第二重保险）。

---

## 7. 设计取舍（挑 5 条最硬的）

每条都指向 `docs/DESIGN_DECISIONS.md` 里对应的决策，「面试怎么讲」原文也在那儿。

**① 为什么零第三方依赖（D-01 / D-13）**
核心数据层用 stdlib `dataclass` 而不是 pydantic，CLI 用 `argparse` 而不是 typer，
embedding 用哈希特征而不是 faiss/tiktoken。理由有三条，**按重要性排序**：零依赖红线、
可变累积语义（`AgentState.messages` 在 ReAct 循环里每轮 append，pydantic 的 field 浅拷贝是纯开销）、
trace 字段全量可控。**性能只排第四，而且只有一格站得住**：`docs/VERIFICATION.md` 的实测补充里，
交错测量（A/B 逐轮交替、best-of-20、5 轮，看比值分布）显示**构造** dataclass 快约 3.6x，
但**序列化** pydantic 反而略快（中位数 0.79x）。我没有把后一格藏起来 ——
面试时能说出「这一格我测不出来」，比硬报一个单次数字更可信。

**② 为什么不用 pytest（BUILD_LOG §4.1）**
这是一条**环境约束，不是技术偏好**，我不打算把它包装成设计。本机无外网、装不上 pytest，
所以测试全部用标准库 `unittest`。它带来的**真实代价**是：没有 fixture 魔法、没有参数化、
异步必须继承 `IsolatedAsyncioTestCase`，所有夹具都得手写。好处是测试零安装依赖、随处可跑。
（顺带一提，`test_zero_dependency.py` 里有一条断言专门查「没有 `import pytest`」，防止谁无意中破线。）

**③ 为什么需要 `LoopBoundPool`（D-20 / D-12）**
最初用 `loop.set_default_executor` 来设线程池容量，后来发现三个问题：
(a) 「用户是否已设过」在 3.10 **无法可靠探测**（只有私有属性 `_default_executor`，没有 getter）；
(b) 「只设一次」的标志如果存在 executor 实例上，第二次 `asyncio.run()` 的新 loop 就会**跳过设置**，
`thread_pool_size` **静默失效** —— 而这正是我专门写测试守的场景；
(c) 一个库去改宿主 loop 的全局状态是不礼貌的。所以改成 executor 自持的 **per-loop 私有池**。
顺带实测到一个内存泄漏：原方案用 `WeakKeyDictionary` 缓存 asyncio 原语，
但 `asyncio.Semaphore` 一旦争用就会把 loop 存进 `self._loop`，**value 强引用 key，弱引用永不失效**，
3 次 `asyncio.run` 就泄漏 3 个 loop。现在改用 `dict[int, …]` + 显式 `release_loop`，并有测试断言清理后池内条数为 0。

**④ 重试是 default-deny，不是 default-retry（D-07）**
`LiteAgentError.retryable` 是**类属性**，默认 `False`；只有 429（尊重 `Retry-After`）、
超时、连接错误四类标 `True`。因为 agent 工具里全是**有副作用**的操作 ——
盲目重试一个超时的 `write_file` 会导致重复写入，而用户在 trace 里根本看不出来。
**在 agent 系统里，一个被重复执行的副作用比一次失败严重得多。** 还有一条更硬的规则：
工具声明 `idempotent=False` 时默认**强制只尝试一次**；且同步工具的超时**不重试**
（超时杀不掉线程，重试会再起一个线程去写同一份资源），异步工具才重试。

**⑤ 承认自己测不出来的东西（D-04 / D-08）**
默认 Embedder 是纯 stdlib 的 hashing trick。`benchmarks/bench_embedding_similarity.py` 的实测结果
**与直觉相反**：同义句平均余弦 0.6239，**反义句平均 0.7153 —— 反义得分更高**；
跨语言翻译对（`Good morning` / `早上好`）余弦是 **0.0000**。
机制是清楚的：token 级特征哈希度量的是「共享了多少 token」，反义句几乎总共享实词（enable/disable），
同义改写反而整套换词。**这不是实现 bug，是选型的固有限制**，所以我把它写进文档，
而不是把它说成「语义检索」。同理，`bench_retrieval_ranking.py` 证明了混合打分 + MMR 把 top-5 里的冗余
从 3 条压到 1 条、主题覆盖从 3 提到 5；但我也明说这张表只能证明**打分公式与去冗算法的行为**，
不能证明检索质量（因为 embedder 本身是词面的）。

---

## 8. 诚实的边界

这一节必须存在。**面试官最欣赏知道自己边界在哪的人**，而我也确实有边界没跨过去。

**本环境的硬约束**：无外网（`pip install` 超时）、无任何 API key、无 `openai`/`anthropic` SDK、
无 `tiktoken`/`faiss`。

因此：

> **真实 provider 的端到端调用未在本机验证过。**
> 我只验证到**请求构造 / 响应解析 / 错误映射**这一层，
> 用的是手写的 `FakeTransport`（在 `tests/helpers.py` 里冻结签名，由 `test_transport.py` 与
> `test_llm_providers.py` 共用），
> 断言的粒度是「发出去的 body 长什么样」「给回来的 JSON 被解析成什么」「HTTP 429 映射成哪个异常」。

具体到「没验证过」的清单：

| 能力 | 状态 | 说明 |
|---|---|---|
| OpenAI / Anthropic / DeepSeek 的真实 HTTP 往返 | **未验证** | 只测到请求体构造与响应解析；socket 层被 `FakeTransport` 替换 |
| `UrllibTransport` 的真实出网 | **未验证** | 测试用 `unittest.mock.patch` 替换 `urlopen`，只测请求构造与错误映射 |
| `requests` / `httpx` 适配器 | **未验证** | 本机两库**都已安装**（requests 2.34.2 / httpx 0.28.1），`default_transport()` 按冻结优先级 httpx→requests→urllib 实际选中 **HttpxTransport** —— 即本机真实 HTTP 路径走的就是它。未验证的是这两个适配器的 `send()`（`tests/` 对它们的引用数为 0）与**真实出网**；「降级到 urllib」的分支是 `code-only-not-run` |
| `web_search` 的真实搜索后端 | **未验证** | 成功路径用 `FakeSearchBackend`；`NullSearchBackend` 返回可读的失败串而不是抛异常 |
| `numpy` / `rich` / `PyYAML` 加速路径 | **部分未验证** | 本机有 numpy 1.26.4，`NumpyHashingEmbedder` 的一致性用例在 `NUMPY_AVAILABLE` 时才会跑 |
| `ScriptedLLM` 驱动的完整 ReAct 循环 | **已验证（offline）** | 这是框架逻辑正确性的主证据链 |
| 工具系统 / 记忆层 / 多 Agent 编排 | **已验证（offline）** | 1652 个通过用例覆盖 |

三个我不想含糊的点：

1. **`ScriptedLLM` 能覆盖循环逻辑，但覆盖不了「模型不听话」。** 真实的模型会输出畸形 tool_call、
   在文本模式里漏写 `Action Input`、把两件事塞进一轮。这些路径我用手写脚本模拟了（并有对应用例），
   但**模拟的畸形分布和真实分布不一样** —— 这一点我无法在本环境证伪。
2. **`HashingEmbedder` 不是语义 embedder。** 如上节所述，它测的是词面重合。
   生产要语义必须注入 `RemoteEmbedder` 或自研向量服务（`VectorMemory(embedder=...)` 已冻结
   「注入的 embedder 维度是权威」）。
3. **性能数字都是共享机器上的微基准**，不是内存占用、不是吞吐、不含 GC 强制回收。
   脚本会把「这不是严格等价的比较」逐条打印出来，宁可输出「测不出差异」也不报单次结果。

`[v3 修正]` 上一版这里写「`docs/` 冻结的 7 个文件目前只有 5 个，`docs/TOOLS.md` 与
`docs/INTERVIEW.md` 尚未落地；§12 冻结的 4 个测试文件也尚未落地（守门测试因此保持 1 个红灯）」。
这三条现在都不成立：`docs/` 7 个文件齐全，4 份测试文件全部落地，
`python3 scripts/check_spec_consistency.py` 报 `error=0 warn=0 info=0`。
仍然成立的缺口只剩一条 —— **持久化的「跨进程往返」只做到同进程**
（JSONL 格式是进程无关的，但没有 subprocess 级往返测试，见 `docs/VERIFICATION.md` §4 G-2 / §3 U-9）。

---

## 9. 参考文档索引

| 文档 | 内容 | 状态 |
|---|---|---|
| `docs/INTERFACES.md` | **冻结接口规范**（唯一契约）：文件清单、依赖 DAG、测试清单、Verification 格式 | 存在（5518 行，FROZEN v2.0） |
| `docs/ARCHITECTURE.md` | 架构说明：分层规则、ReAct 轮次时序图、多 Agent 数据流 | 存在（424 行） |
| `docs/DESIGN_DECISIONS.md` | **27 条设计决策**，每条含背景/备选/选择/理由/代价 + **「面试怎么讲」** | 存在（1548 行，FROZEN v2.0） |
| `docs/VERIFICATION.md` | `claim \| evidence \| status` 能力对账表 + 3 个 benchmark 的**实测补充** | 存在（466 行，含自动生成块） |
| `docs/BUILD_LOG.md` | 搭建过程记录：环境侦察、踩坑、每一步「为什么」 | 存在（行数会随编辑漂移，本轮实测 571 行） |
| `docs/TOOLS.md` | 工具作者指南（`retryable` / `idempotent` 清单等） | 存在（966 行） |
| `docs/INTERVIEW.md` | 面试能力清单与话术 | 存在（1345 行） |

> 上表的**行数是快照**（实测于 2026-09-27 04:2x UTC），会随文档编辑漂移 ——
> 需要准确值时请自己 `wc -l docs/*.md`，别引用这个数字。

> **文档的冻结与所有权**：`INTERFACES.md` 头部是 `FROZEN v2.0`（唯一契约，由独立的接口 writer 维护）；
> `BUILD_LOG.md` 由搭建过程的记录者维护 —— 这两份**不在本文档的刷新范围内**。
> `ARCHITECTURE.md` / `DESIGN_DECISIONS.md` 头部同样写着 `FROZEN v2.0`，但那冻结的是**接口与决策的语义**；
> 本轮审计只**修正了其中的数字与不实主张**（例如把"跨进程往返有测试"改成"只到同进程"），
> 未改动任何接口、类名或决策结论。`VERIFICATION.md` 里的三个 benchmark 小节由
> `benchmarks/*.py --json` **自动覆盖**，请勿手改标记块之间的内容。

复现 benchmark 与规范检查：

```bash
python3 benchmarks/bench_dataclass_vs_pydantic.py --json   # 填 D-01 的实测补充
python3 benchmarks/bench_embedding_similarity.py --json    # 填 D-04
python3 benchmarks/bench_retrieval_ranking.py --json       # 填 D-08
python3 scripts/check_spec_consistency.py                  # 检查 §1.2 文件清单与实际仓库是否一致（不参与测试）
```
