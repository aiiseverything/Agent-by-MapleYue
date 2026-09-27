# liteagent 面试作战手册（INTERVIEW.md）

> **这份文档的用途**：把 `liteagent` 这个项目**在面试里讲清楚**。
> 每一条主张都对应代码里的一个落点、或一份实测记录。
>
> **真实性声明（先读这一节）**
>
> * 标注「**实测**」的是我在**本机真的跑过**的（命令与输出都在文里）；
> * 标注「**代码**」的是我读了真实代码后确认的实现事实（给出行号或文件名）；
> * 标注「**未验证**」的是**我没跑过**的（例如真实 provider 联网、真实检索质量），
>   面试时必须主动说明边界，**不许把它说成跑过了**。
> * 本文件不复制 `docs/INTERFACES.md` 的规范细节；规范才是权威，
>   本文只负责「怎么讲」与「往哪指」。

---

## 0. 数字速记卡（面试前 30 秒过一遍）

全部为**本机实测**（命令附在 §5）：

| 项 | 数值 |
|---|---|
| 语言/环境 | Python 3.10.12，**无外网**、**禁止 pip install** |
| 内核规模 | `liteagent/` **41 个 `.py`、24,765 行**，零第三方依赖 |
| 测试规模 | `tests/` 39 个 `.py`（其中 37 个 `test_*.py`）、22,409 行；**`Ran 1652 tests`** → **1652 通过 / 0 失败**（本机两次重跑实测 29.7s / 31.2s，随机器负载波动） |
| 唯一失败例 | **无** —— 上一版这里写的是 `FrozenLayoutTests.test_all_frozen_test_files_exist`（§12 冻结的 4 份测试文件未落地），那 4 份文件现已落地、守门测试转绿（§6 第 8 条已改） |
| 最重的两个测试文件 | `test_tools_schema.py`（91 例）、`test_tools_executor.py`（119 例） |
| 文档规模 | `INTERFACES.md` 5518 行 + `DESIGN_DECISIONS.md` 1548 行 + `ARCHITECTURE.md` 424 行（**行数是快照，会随编辑漂移**） |
| 示例 | `examples/01`~`07` 共 7 个，可 `--offline` 无 key 跑 |
| 内置工具 | **13 个**：`read_file write_file list_dir search_files delete_file run_shell python_exec python_eval run_tests web_search fetch_url remember recall` |
| 版本 | `python3 -m liteagent version` → `0.1.0` |
| 联网次数 / API key | 跑测试需要 **0 次联网、0 个 key** |

**一句话卖点**：一个**零依赖、离线可复现、控制流完全显式**的 Agent Harness ——
不是「我写了很多测试」，而是「**1600 多个测试不需要网络和 key，半分钟内跑完，而且能断言模型看到了什么**」。

---

## 1. 项目介绍

### 1.1 一分钟版（口述稿，约 140 字）

> liteagent 是一个**零第三方依赖**的 Agent Harness，2.4 万行纯标准库 Python。
> 它把 ReAct 循环、工具系统、三层记忆和多 Agent 协作收敛成**一份显式状态机**：
> 文本模式和原生 Function Calling 归约到同一个 `list[ToolCall]`，
> 工具用装饰器反射 type hints 生成 JSON Schema。
> 实测 1652 个单元测试**全部离线、不需要 API key、半分钟内跑完**，
> 而且能反向断言「模型这一轮到底看到了哪几条消息、哪几个工具」。
>
> （时间不够就砍到最后一句：「离线 + 可断言模型输入」是它跟同类项目最大的差别。）

### 1.2 三分钟版（为什么做 → 做了什么 → 结果数字）

**① 为什么做（30 秒）**

我开始写第一行代码之前先做了环境侦察，结论是：**没有外网、装不上任何包**。
这个约束直接变成了设计原则：

```text
无外网 ──┬─> 内核零第三方依赖（纯 stdlib）
         ├─> 测试只能用 stdlib unittest，于是必须自造「脚本化假 LLM」
         └─> 一切第三方能力做成「可选适配器 + 纯 stdlib 兜底」
             （numpy 缺失 → 纯 Python 余弦；rich 缺失 → 纯文本；requests 缺失 → urllib）
```

所以我**不是**「为了炫技才不用 LangChain」，而是**约束倒逼**：
零依赖同时买到了「秒级启动、不被上游 breaking change 打断、任何 Python 环境可跑」。

**② 做了什么（90 秒）** —— 三层架构 + 一个循环 + 两种协作模式：

* **LLM 抽象层**：统一 `LLMClient` 接口，4 个 provider（OpenAI / Anthropic / DeepSeek / Echo），
  所有 HTTP 走一个可替换的 `Transport` 接口，于是**离线测试能注入假传输层**。
* **工具系统**：`@tool` 装饰器反射 `inspect.signature` + type hints + docstring 生成 JSON Schema，
  边界情况（`bool` vs `int`、`Literal`、`Optional`、嵌套 dataclass、递归、`Annotated`）**逐条有测试**；
  执行器管审批、并发、超时、重试、截断、取消、熔断。
* **记忆层**：短期滑动窗口 + 摘要压缩 + 长期向量记忆（纯 stdlib hashing trick embedding，
  混合打分 + MMR 去冗 + JSONL 持久化）。
* **ReAct 循环**：Thought-Action-Observation 状态机，**文本模式与原生 Function Calling 共用同一份**。
* **多 Agent**：Sequential 流水线 + Hierarchical（任务分解 / 委派 / 依赖图执行），复用同一个 Agent。

**③ 结果数字（30 秒）**

* **1652 个单元测试**，`Ran 1652 tests` / `OK`，**0 次联网、0 个 API key**（实测）；
* 内核 **24,765 行 / 41 个模块**，顶层 import **只有 stdlib**（有 AST 守门测试）；
* 三份基准（`benchmarks/*.py`）产出的**真实数字**，都写进了 `docs/VERIFICATION.md`：
  例如「近因衰减 + MMR 把 top-5 里的冗余从 3 条压到 1 条、组内最大两两余弦从 0.985 降到 0.405」。

### 1.3 面试官说「再展开讲一个技术点」

**不要**从架构图讲起。直接讲一条**有实测证据的坑**（§4 里挑一个），
例如「`asyncio.Semaphore` 跨事件循环」——我现场复现过，第二次调用必崩，而且**只在争用时崩**。
先给现象，再给根因，最后给修法和代价。**面试官要的是你的判断力，不是你的模块清单。**

---

## 2. 能力清单 → 代码落点表

> 这张表是**简历主张的可验证化**。左边是简历上的原子能力，右边是「哪个模块实现、哪个测试文件证明、
> 能不能现场演示」。它同时也是 `docs/VERIFICATION.md` §12.2 主表的行来源（§12.3 的 9 行对账我全部覆盖）。
>
> 「现场演示」列的含义：`✅` = 无网络无 key 可以直接跑；`⚠️` = 只能演示**不联网的那一半**
> （报文编解码 / 错误映射），真实端点未跑；`—` = 不适合现场跑。

### 2.1 三层架构（简历 bullet 1）

| # | 简历原子能力 | 实现落点 | 测试证据 | 现场演示 |
|---|---|---|---|---|
| 1 | LLM 抽象层统一多模型 API | `llm/base.py`（`LLMClient`/`BaseLLMClient`）、`llm/providers.py`（4 个 provider）、`llm/transport.py`（`Transport` 抽象 + `UrllibTransport`） | `tests/test_llm_providers.py`（66 例）、`tests/test_transport.py`、`tests/test_llm_registry.py` | ⚠️ 离线可演示请求体构造与响应解析（`FakeTransport`）；**真实端点未跑** |
| 2 | provider 注册与解析 `provider:model@base_url` | `llm/registry.py`（`build_llm`/`get_llm`/`reset_default_registry`） | `tests/test_llm_registry.py`（未知 provider 报错且列出可用项） | ✅ |
| 3 | 工具层**装饰器自动注册** | `tools/base.py` 的 `@tool(auto_register=True)` → `get_default_registry().register(t, override=False)`；**默认关**（D-03） | `tests/test_tools_registry.py` 的 4 条 `auto_register` 用例 | ✅ |
| 4 | type hints → **JSON Schema 生成** | `tools/schema.py`（`annotation_to_schema` / `build_tool_schema` / `validate_instance`），64.9 KB | `tests/test_tools_schema.py`（**91 例**，逐条映射断言） | ✅（§5 演示 3） |
| 5 | 记忆层**短期对话历史** | `memory/buffer.py`（滑动窗口，双约束：条数 + token）、`memory/summary.py`（摘要压缩） | `tests/test_memory_buffer.py`、`tests/test_memory_summary.py` | ✅ |
| 6 | 记忆层**长期向量存储** | `memory/vector.py`（混合打分 + MMR）、`memory/embeddings.py`（`HashingEmbedder`）、`save`/`load` JSONL | `tests/test_memory_vector.py`、`tests/test_memory_persistence.py` | ✅ |
| 7 | 三层记忆编排 | `memory/manager.py`（`abuild_prompt` 六段组装 + `_enforce_context_budget`） | `tests/test_memory_manager.py` | ✅ |

### 2.2 ReAct 循环（简历 bullet 2）

| # | 简历原子能力 | 实现落点 | 测试证据 | 现场演示 |
|---|---|---|---|---|
| 8 | 完整 **Thought-Action-Observation** 循环 | `agent/agent.py`（`arun` 状态机）、`agent/parser.py`（文本语法）、`agent/state.py` | `tests/test_agent_react_text.py`、`tests/test_agent_react_native.py` | ✅ |
| 9 | **文本模式**（`Thought:/Action:/Observation:`） | `agent/parser.py`（8 步解析 + 容错）、`tools/registry.py::to_prompt` | `tests/test_agent_react_text.py`（全角冒号/中文 marker/Markdown 加粗/跨行 JSON/围栏 JSON） | ✅ |
| 10 | **原生 Function Calling** | `llm/providers.py` 的 `tools=` 报文 + `role="tool"` 回灌 | `tests/test_agent_react_native.py` | ⚠️ 报文离线验证；真实端点未跑 |
| 11 | 两种模式**归约到同一份状态机** | `agent/agent.py::_next_calls` / `_write_back`（差异只在这 5 个点，D-02） | 两个 react 测试文件共用断言骨架 | ✅ |
| 12 | 执行器：**错误处理与自动重试** | `tools/executor.py`、`config.py::RetryPolicy.delay_for`、`errors.py` 的 `retryable` 类属性 | `tests/test_tools_executor.py`（119 例）、`tests/test_llm_retry.py` | ✅ |
| 13 | **并发工具调用** | `execute_many` + `LoopBoundPool.semaphore("exec", max_concurrency)` | `test_tools_executor.py`（顺序保持 / 峰值上限 / `fail_fast`） | ✅ |
| 14 | 循环防护 / 止损 | `agent/state.py`（`action_counts` / `observation_digests` / `tool_failure_counts`）、`executor` 熔断、`max_wall_clock_s` / `max_total_tokens` | `test_agent_features.py`、`test_tools_executor.py` 的熔断用例 | ✅ |
| 15 | HITL 审批（fail-closed） | `tools/executor.py::_approve` + `ExecutorConfig.approval_policy` | `test_tools_executor.py` 审批三例 | ✅ |
| 16 | 可观测性（事件总线 / JSONL trace / 计费 / 耗时） | `agent/callbacks.py`（`EventType` / `TraceEvent` / `JsonlTraceCallback` / `TokenCounterCallback` / `trace_stats`）、`config.py::estimate_cost_usd` | `tests/test_callbacks.py` | ✅ |

### 2.3 多 Agent 与扩展（简历 bullet 3）

| # | 简历原子能力 | 实现落点 | 测试证据 | 现场演示 |
|---|---|---|---|---|
| 17 | **Sequential** 协作模式 | `multiagent/sequential.py`（`SequentialAgent` / `SequentialStep` / `input_template` / `propagate_failure`） | `tests/test_multiagent_sequential.py` | ✅ |
| 18 | **Hierarchical** 协作 + 主 Agent 任务分解 | `multiagent/hierarchical.py`（`delegate_to_*` 工具、`adecompose`、`arun_plan` 依赖图） | `tests/test_multiagent_hierarchical.py`（79 例） | ✅ |
| 19 | 共享状态（黑板） | `multiagent/blackboard.py`（`Blackboard` + `RLock` + 版本号 + TTL + `awatch`） | `tests/test_blackboard.py`（64 例，含多线程并发写） | ✅ |
| 20 | 防止无限委派 | `multiagent/base.py::DelegationContext`（`would_cycle` / `depth` / `budget`）+ `threading.Semaphore(subagent_concurrency)` | `test_multiagent_hierarchical.py`（环检测 / `max_depth` / budget / busy） | ✅ |
| 21 | 内置工具：网页搜索 / 代码执行 / 文件操作 | `tools/builtin/{files,shell,code,web,memory_tools}.py`（**13 个**） | `test_builtin_{files,shell,code,web}.py` | ✅（沙箱 / 复杂度闸 / 注入白名单现场可跑） |
| 22 | 文档与示例 + 代码助手场景可用性验证 | `examples/01`~`07`、`examples/07_code_assistant.py` | `test_builtin_*.py` 四份 + `test_e2e_code_assistant.py`（8 例）+ `test_examples_offline.py`（7 例），**全部落地且真绿**（`[v3]` 上一版这里把后两份写成「尚未落地」） | ✅ `examples/07 --offline` 手工实跑通过（退出码 0），且现在有 `subprocess` 级回归测试兜着 |

---

## 3. 高频追问 Q&A

> 每条给「**标准答案**」（照着背）+「**追问再深一层**」（面试官顺着问时用）。
> 标 ⭐ 的是必考。

---

### Q1 ⭐ 为什么不用 LangChain？你们的取舍是什么？

**标准答案**：

> 「我参考了 LangChain 的设计模式，但没有用它。不是"我要造轮子"，
> 而是我有两条硬约束：**内核零第三方依赖**、和**离线可测**。
> LangChain 帮我省掉的每一行代码，都要用**依赖体积、黑盒控制流、不可断言的事件**来换，
> 在这个项目里我选择把这三样换回来。
>
> 我把它整理成了 8 条对比轴（写在 `DESIGN_DECISIONS.md` 的 D-13），挑三条讲：
> ① **依赖**：LangChain 是几十个传递依赖，我是 0 个，三方库全是可选加速器；
> ② **事件模型**：LangChain 的 callback 是 `on_llm_start(**kwargs)` 一族，**kwargs 魔法、
> 字段随版本漂移，我用强类型 `EventType` 枚举 + 冻结字段表，trace 可以做精确断言；
> ③ **可测性**：这是我最在意的 —— LangChain 的 `FakeListLLM` 只能返回文本，
> 我**没法断言"模型到底看到了哪几个工具"**，而工具暴露正是 agent 框架最容易出错的地方。
> 所以我自己写了 `ScriptedLLM`，它记录每次调用的 messages、tools、kwargs 和消费到的响应。」

**追问再深一层**：「那你从零写，丢了什么？」

> 「丢了三样，我说实话：① **没有 provider 生态**——每接一个新厂商要自己写报文编解码
> （DeepSeek 只花了 5 行因为它 OpenAI 兼容，Anthropic 报文差异大花了约 120 行）；
> ② **没有 LCEL 那套声明式组合**，复杂链要手写 Python——不过这个项目不需要；
> ③ **没有社区和文档**，出问题只能自己看代码，所以我把 `INTERFACES.md` 写到 5518 行。
> 我的结论是：liteagent 不是 LangChain 的替代品，
> 是**把 LangChain 里我真正需要的那 20% 显式实现一遍**。」

*（对比表明细：`docs/DESIGN_DECISIONS.md` D-13。第 1 条轴的量化证据——`pip install langchain-core`
的传递依赖数量——**未验证**，本环境无外网，不许声称实测过。）*

---

### Q2 ⭐ `@tool` 装饰器怎么把 type hints 变成 JSON Schema？边界情况有哪些？

**标准答案**（读完 `liteagent/tools/schema.py` 后）：

> 「入口是 `Tool.from_function()` → `build_tool_schema()`，
> 核心是一个递归函数 `annotation_to_schema(annotation, depth, seen)`，
> 返回 `(schema 片段, warnings)` 二元组。
> 参数有两个来源：`inspect.signature` 拿注解与默认值，
> `parse_docstring()` 解析 Google / Sphinx 两种风格的 docstring 拿逐参描述。
> `required` 的判定只有**一条**公式（§7.1.5，顶层参数与嵌套 dataclass 字段共用）：
> 某个参数进 `required`，当且仅当它**没有默认值**、**不是 `Optional[...]`**、
> 并且 `Annotated` 里**没有** `Param(default=...)` —— 三个条件同时成立。少看任何一条都会写错。
>
> 边界情况我记得比较清楚，因为**每一条都有测试**：
>
> | 边界 | 我们的行为 |
> |---|---|
> | `bool` vs `int` | **`bool` 必须先判**：`isinstance(True, int) is True`，顺序写反 `flag: bool` 就会变成 `integer` |
> | `Optional[X]` | **不生成** `"type": ["X","null"]`，只用来决定「不放进 `required`」（D-05） |
> | `Literal['a','b']` | 生成 `{"type":"string","enum":[...]}`；**混型**（`Literal['a',1]`）时**省略 `type`** 只留 `enum` |
> | `Enum` | 同 Literal，按**定义顺序**取 `member.value` |
> | 嵌套 `dataclass` | 递归展开成内联 `object` schema，带 `additionalProperties: false` |
> | **递归 dataclass** | `seen` 记录当前递归路径上的类型，命中就截断 + warning |
> | 深度 | `MAX_SCHEMA_DEPTH` 之上直接降级为 `{}`（any）+ warning |
> | `Annotated[...]` | **先剥壳、metadata 最后统一应用**，逐参描述优先于 docstring |
> | `Union[A,B]` | 降级为 `{}`（any）+ warning —— 多类型 schema 会打崩 provider 的 grammar 转换器 |
> | `*args` / `**kwargs` | 抛 `ToolDefinitionError`（不是静默忽略） |
> | 定长 `tuple[X,Y]` | 降级成 `array` + `minItems/maxItems` + warning |
>
> **关键设计是「降级必须留痕」**：凡是猜不准的，一律降级为 `{}` + 把 warning 记进
> `ToolSpec.warnings`，而不是硬猜一个错的类型。这是我自己定的红线 12。」

**追问再深一层**：「`Annotated[int, Param(default=5)]` 怎么探测？」

> 「**这是我在 3.10 上实测出来的坑**：直觉写法 `isinstance(annotation, Annotated)` 返回
> **`False`，而且不报错**。因为 `typing.Annotated` 在 3.10 里是个普通类，
> `Annotated[int,'x']` 的实例是 `typing._AnnotatedAlias`，两者**没有继承关系**。
> 所以那是**死代码**，最可怕的是它不崩，只是功能静默失效——
> `Annotated[int, Param(default=5)]` 会被判成必填，schema 里同时出现
> `"required":["x"]` 和 `"default":5`，**自相矛盾，正确的调用反而被我的校验器拒掉**。
> 正确写法是 `getattr(annotation, "__metadata__", ())` 或
> `get_origin(x) is Annotated`，我把它写进了规范并加了三个针对性测试用例。」

*（实现：`tools/schema.py` 的 `annotated_metadata()` / `_meta_default()`；决策：D-19。）*

---

### Q3 ⭐ 文本 ReAct 模式的解析怎么做鲁棒？模型不按格式输出怎么办？

**标准答案**：

> 「解析器是 `agent/parser.py` 的 `ReActParser.parse()`，是一个**冻结的 8 步流水线**：
> ① 规范化 → ② 整段 JSON 对象 → ③ 逐行 marker 扫描 → ④ **Action 优先**
> （同时出现 `Action:` 和 `Final Answer:` 时执行 Action、不终结）→ ⑤ Action Input 取值
> → ⑥ Final Answer → ⑦ 失败则抛 `ReActParseError` → ⑧ 自纠正。
>
> 容错的地方（**每条都有测试**）：
> * marker 大小写不敏感、支持**全角冒号**`：`、支持**中文 marker**（`动作:` / `动作输入:` / `思考:`）；
> * Markdown 加粗 `**Action:**` —— 这里有个细节：正则会把 `**` 当成列表符号吃掉，
>   之后还要单独剥一次前导 `**`（`_leading_emphasis_run` + `_strip_wrapping`）；
> * Action Input 支持 ```json 围栏、跨行 JSON、裸值；
> * **裸值只敢绑到"恰好一个已知参数"的工具上**——参数多于一个时无法判断裸值属于谁，
>   宁可回落成 `{"input": 值}`，让模型看到自己的原文，比我们猜错参数名更容易自纠正；
> * `Action:` 后面没有 `Action Input:` 行时，**只认同一行内的 `{...}` 映射**，
>   不放宽到裸值 —— 否则 `Action: search` 后面的散文会被当成参数，
>   产生「看似成功、参数全错」的假阳性，比空参数更难排查。
>
> **模型完全不按格式输出**：抛 `ReActParseError`，框架把**错误 + 正确格式示例 +
> 可用工具名清单**作为一条 NUDGE 消息回灌给模型，让它自己改（自纠正）。
> 这里有两条我特意定的规则：
> ① **一次 parse error 恰好新增 2 条消息**（1 条 assistant 原文 + 1 条 NUDGE），
> 这个不变式**写进了测试断言**；
> ② 自纠正**消耗 `max_steps` 预算**（D-10）—— 因为每次自纠正都是一次**真实的 LLM 调用**，
> 不计入的话，一个持续输出乱格式的模型就能无限循环，变成成本放大器。
> 我另外单设了 `max_parse_retries`（默认 2）：两个计数器职责不同，
> `max_steps` 管总成本，`max_parse_retries` 管格式病态程度，触发的是不同的终止原因，
> trace 里能直接区分『任务太难』和『模型不会用这个格式』。」

**追问再深一层**：「会不会把 `Final Answer` 里的内容误当成 Action？」

> 「不会——判定的顺序是**Action 优先**：只要文本里有一个合法的 `Action:` 行，
> 就执行 Action，`Final Answer` 被丢弃并记一条 warning。
> 反过来只有 `Final Answer:` 时才终结。这是 §9.3 步骤 4 冻结的语义，
> 有一条专门的测试用例（『Action 与 Final Answer 同时出现时执行 Action 且不终结』）。」

*（实测：我用 `ReActParser` 现场跑过 plain / fenced / 全角冒号 / Markdown 加粗 / 中文 marker /
裸值 / Final Answer 七种输入，全部按预期解析，命令见 §5。）*

---

### Q4 ⭐ 怎么防止 Agent 无限循环？

**标准答案**（**五层防护**，全部落地成真实字段，不是嘴上说）：

> 「单靠"完全相同的动作算重复"是抓不住真实情况的——模型每次把参数改一个字符
> （`read_file('a.py')` → `read_file('./a.py')`）就绕过去了。所以我做了五层：
>
> | 层 | 载体（真实字段） | 命中条件 |
> |---|---|---|
> | **L1 相同动作** | `AgentState.action_counts[canonical_key]` | `>= repeat_action_threshold`（默认 2） |
> | **L2 无进展** | `AgentState.observation_digests[blake2b(content)[:8]]` | 任一摘要计数超阈值 |
> | **L3 工具熔断** | `AgentState.tool_failure_counts[name]` + `disable_tool_after_failures=3` | 同一工具连续 3 次 infrastructure 失败后，executor **直接返回失败、不再真正执行** |
> | **L4 墙钟** | `AgentConfig.max_wall_clock_s` → `RunTimeoutError` | 每轮开头检查 `utc_now() - state.started_at` |
> | **L5 预算** | `AgentConfig.max_total_tokens` → `BudgetExceededError` | 每次 LLM 响应后检查 |
>
> 其中 **L2 是我最满意的一层**：它用**工具结果的 `blake2b` 内容摘要**来判，
> 抓的是「**参数变了但结果没变**」——模型没拿到新信息却还在动，这是无进展的强信号，
> 正好补上 L1 的盲区。L3 是**止损**而不是检测：连续三次 infrastructure 失败，
> 模型再调它也是浪费一次 LLM 轮次。
> L4 是唯一能兜住『每轮都在进展但总量失控』的机制——`max_steps` 只管次数、不管耗时。」

**追问再深一层**：「`canonical_key` 怎么算？参数顺序不一样算不算同一个？」

> 「`canonical_key = name + ":" + json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)`
> —— `sort_keys=True` 让参数顺序不影响结果，`default=str` 是最后一道保险：
> 就算有人往 arguments 里塞了非 JSON 类型，键也仍然可比较、不会抛异常。
> 这几条在 `types.py::ToolCall.canonical_key` 里，有专门的稳定性测试。」

*（实现：`agent/state.py` 的三个计数字段 + `agent/agent.py` 的每轮检查 + `tools/executor.py` 的熔断分支；
决策：D-16。**"我做了防护但字段不存在 = 面试官随手一点就露馅"**——所以我把它们都做成了 state 字段。）*

---

### Q5 ⭐ 并发工具调用怎么实现？**同步阻塞工具**怎么办？超时了怎么办？

**标准答案**：

> 「`ToolExecutor.execute_many(calls)` 用 `asyncio.ensure_future` 起 N 个任务，
> 两层限流：**批次级**（`concurrency` 参数，可选）+ **全局**（`max_concurrency`，默认 4，
> 由 `LoopBoundPool.semaphore("exec", ...)` 提供）。
> 返回时用 `asyncio.gather(..., return_exceptions=True)` 收尾，
> **返回列表顺序与 `calls` 严格一致，不看完成顺序**——顺序错乱会让模型把结果对错工具。
>
> **同步阻塞工具**走线程池：`loop.run_in_executor(tp, ...)`，池是 per-loop 私有的
> （`LoopBoundPool.thread_pool("exec", thread_pool_size)`，默认 8），
> **不用 `loop.set_default_executor`**，因为那是改宿主 loop 的全局状态（见 Q6）。
>
> **超时**这块我要说清楚，因为它是最容易骗自己的地方：
> `asyncio.wait_for` **只能取消 await 层**——跑在 worker 线程里的同步函数**不会停**，
> 它会继续跑到自己结束。我**没有假装解决了**：
> 超时后我会把 `ToolResult.metadata["orphan_thread"] = True` 并记 WARNING，
> **让"线程还在跑"这件事可见**。同时提供**协作式取消**：
> 用 `contextvars` 把一个 `threading.Event` 传进工具线程，
> 循环型的长耗时工具自己检查 `current_cancel_flag()` 主动退出
> （内置的 `subprocess.run` 类工具本来就该传 `timeout=` 给它自己，我正是这么做的）。
>
> **为什么不用进程隔离**：我的内置工具大量是**闭包**
> （`make_file_tools(sandbox)`、`make_memory_tools(memory)`、delegate 工具捕获子 Agent），
> **闭包不可 pickle**，进程方案会直接废掉一半内置工具。
>
> **为什么同步工具超时后不重试**：超时杀不掉线程，重试会**再起一个线程**去写同一份资源——
> 对 `write_file`/`run_shell` 就是"两个线程同时写"。所以我的规则是
> **同步工具的 `ToolTimeoutError` 不重试，异步工具的才重试**。」

**追问再深一层**：「用 contextvar 传取消信号，有什么前提？」

> 「**有，而且我实测过**：contextvar 的**读**能穿透线程边界，但**三条提交路径行为不一致**——
>
> ```text
> asyncio.to_thread(fn)            -> worker 能看到 flag   ✅
> loop.run_in_executor(tp, fn)     -> worker 看到 None     ❌
> 裸 ThreadPoolExecutor.submit(fn) -> worker 看到 None     ❌
> ```
>
> 我在这台机器上跑过这三行，**裸跑的实测输出是 `FLAG / None / None`** ——
> 第三条只有自己再包一层 `contextvars.copy_context().run` 才会变成 `FLAG`。所以走 `run_in_executor` 的同步分支
> **必须自己包一层** `ctx.run(...)`，否则 `current_cancel_flag()` 在同步工具里**恒为 None**、
> "协作式取消"会**静默失效**——而恰恰是同步工具（不可中断）最需要它。
> 另外一个方向也是单向的：**读能穿透、写不回传**，所以 `flag.set()` 只可能发生在
> executor 所在的线程。还有一条硬约束：`cancel_scope()` **必须每个 attempt 重新进一次**，
> 提到重试循环外面的话，第二次尝试拿到的是**已 set 的同一个 Event**，工具一进循环就自杀，
> 重试全部秒失败。」

*（实现：`tools/executor.py` 的 `execute_many` / `_invoke` / 超时分支；`tools/base.py` 的
`current_cancel_flag` / `cancel_scope`（那里的注释就是上面这张实测表）；
决策：D-09。）*

---

### Q6 ⭐ `asyncio.Semaphore` 跨事件循环崩溃是怎么回事？`LoopBoundPool` 解决什么问题？

**标准答案**（**这是我最想讲的一条，因为我现场复现过**）：

> 「我的 Agent 每次 `run()` 都是一次 `asyncio.run()`，也就是**每次都新建一个 event loop**。
> 如果我在 `__init__` 里创建 `asyncio.Semaphore`，第二次同步调用就会崩：
>
> ```text
> run #1 -> ['a', 'b']
> run #2 -> RuntimeError: <asyncio.locks.Semaphore object ...> is bound to a different event loop
> ```
>
> **注意它只在"争用"的时候崩** —— 我第一次试的时候 value=1、只有一个 acquirer，第二次调用
> 是**正常的**；我把它改成 `gather` 两个任务去抢同一把信号量，第二次调用立刻就崩了。
> 根因是：`asyncio.Semaphore` 只有在**必须等待**（有人持锁且有 waiter）时才把当时的 loop
> 记进 `self._loop`，于是这个原语就"绑死"在第一个 loop 上了。
> **这就是最阴险的地方**：单线程、不争用的时候测试全绿，一到真实并发就炸。
>
> 修法是 `LoopBoundPool`：所有 asyncio 原语**按运行中的 loop 懒创建**，
> 没有运行 loop 时入口直接抛 `ConfigError`（第一道防线），
> 键是 `id(loop)`，并且 `_loops` 里**强引用**着 loop（防止 id 被回收后复用）。
>
> 顺带我还实测到第二个坑：我原来用 `WeakKeyDictionary[loop]` 缓存原语，
> 但**value 强引用 key，弱引用永远不会失效**——
> 我跑了 3 次 `asyncio.run`，缓存里就剩 3 个 key、3 个泄漏的 loop。
> 现在改成 `dict[int, ...]` + 显式 `release_loop` / `aclose`，
> 并且有一条测试断言『连续 3 次同步调用后池内条数为 0』。」

**追问再深一层**：「`threading` 原语为什么就没事？那你为什么不干脆全用 threading？」

> 「`threading.Lock/Semaphore/RLock` 跟事件循环**完全无关**，跨线程跨 loop 天然正确，
> 所以凡是**跨线程/跨 loop 必须生效**的限流与互斥，我一律用 threading 原语（这是红线 13）。
> 但**不能全用**：asyncio 原语在 await 语义上有它的价值（例如同一 loop 内的公平排队），
> 而且我在 `LoopBoundPool` 里同时管着**线程池**——`thread_pool_size` 的语义因此变得明确：
> 『每个 loop 私有池多大』，而不是『进程默认线程池多大』这种含混说法。
> 关键约束是**临界区里不能有 await、不能有 I/O**，只操作内存 dict，这时 RLock 在语义上完全正确。」

*（实现：`config.py::LoopBoundPool`；决策：D-12 / D-20 / D-21。上面两个实测输出都是本机复现的。）*

---

### Q7 ⭐ 为什么必须「先并发信号量、后顺序锁」？反序为什么会死锁？

**标准答案**：

> 「因为我们有两把锁：**全局并发信号量**（`semaphore("exec", max_concurrency)`）和
> **按工具名的顺序锁**（`sequential_tools`，比如 `run_shell` 必须串行）。
> 获取顺序**必须**是『先信号量、后 seq 锁』，这是红线 14 冻结的，**反序即死锁**。
>
> 讲清楚为什么：假设反序，A 拿到 `seq:run_shell` 之后去等信号量，B 也拿到
> `seq:shell`（不同工具的锁）之后去等信号量——**此时信号量已经被 N 个调用者占满**，
> 而占满信号量的那些调用者正卡在等 seq 锁上。于是：**每个都握着 seq 锁等信号量，
> 谁都不放**，形成真正的循环等待。正序就不会——信号量是**最外层**的门，
> 只有拿到并发额度的人才会去竞争顺序锁，锁的持有时间被压到最短。」

**追问再深一层**：「`sequential_tools` 为什么必须用 `threading.Lock` 而不是 `asyncio.Lock`？」

> 「因为 `sequential_tools` **存在的唯一理由就是跨调用互斥**，而同步工具会被
> `execute_many` 丢进**不同的 worker 线程**，每个线程里还可能各自 `asyncio.run`。
> loop-bound 的 `asyncio.Lock` 在这种情况下**完全不串行**——不是"不太准"，是"完全无效"。
> 更麻烦的是，我原来的测试只测**单 loop**，所以**测试是通过的**，缺陷被掩盖了。
> 我现在加了一个用例：`sequential_tools` 在**两个嵌套 loop**（一个普通 agent +
> 一个 delegate）下仍真正串行——只有这种测法才能守住它。」

*（实现：`tools/executor.py` 步骤 5.b 的两次 `async with`；决策：D-21；红线 14。）*

---

### Q8 ⭐ 长期记忆怎么写入、怎么检索？为什么用「相似度 + 近因 + 频次」混合打分？MMR 解决什么？

**标准答案**：

> 「**写入**：`VectorMemory.add/upsert`。有两条路径——
> ① 显式写入：`remember` 工具（模型自己决定要记什么）；
> ② 自动写入：`should_auto_write(message)` 只在**真正的用户消息**上为真
> （`role != "user"` **恒 False**），并且要命中「记住/我的名字是/I prefer…」这类 marker。
> `upsert` 先按 `dedup_threshold`（0.95）去重，避免同一事实写五遍。
>
> **检索**：`search()` 的冻结公式是
>
> ```text
> score      = 1.0 * sim + 0.15 * recency + 0.10 * importance
> sim        = cosine(query_vec, item.embedding)      # 负值直接丢弃
> recency    = 2 ** (-clamp(age_days, 0, +inf) / 7.0) # 半衰期 7 天
> importance = clamp(item.importance, 0, 1)           # 用户可通过 remember(importance=) 控制
> ```
>
> **为什么不是纯向量相似度**——纯相似度有两个典型失效，我用一个 20 条记忆的基准**量过**：
> ① **冗余挤占**：纯相似度 top-5 里 **3/5** 是同一事实（三条写法彼此余弦 0.92~0.98），
> **超过一半的上下文预算被浪费**；
> ② **过时冒充**：一条 200 天前的 MySQL 版本事实在纯相似度下排 **#3**、进了 top-5，
> 加了近因衰减（半衰期 7 天 → recency≈0）之后被踢出 top-5。
>
> **MMR** 解决的是 top-k 内部的**重复**：`mmr_i = λ*score_i - (1-λ)*max_{j∈selected} sim(i,j)`，λ=0.7。
> 实测效果：top-5 里的冗余从 **3 → 1 条**，主题覆盖从 3 → **5**，
> 组内最大两两余弦从 **0.985 → 0.405**。
>
> **权重我故意让相似度占绝对主导**（1.0 对 0.15 和 0.1）——因为默认的 `HashingEmbedder`
> 得分区间只有 0~0.5，如果 recency 权重大，排序会被时间主导，那是错的：
> **时间只应该用于打破接近分数的平局**（基准表里 hybrid 和 pure 的前两名完全一致，就是证据）。
> 用 `2^(-age/half_life)` 连续衰减而不是硬时间窗，是因为硬窗户会在边界产生
> 『昨天还在、今天消失』的跳变，半衰期更好解释、也更好调。
>
> 还有一个容易忽略的点：**排序键是三级稳定排序 `(-score, -created_at, id)`**，
> 否则同分条目的顺序依赖 dict 迭代序，测试会**随机失败**。」

**追问再深一层**：「你说的是『词面相似』还是『语义相似』？」

> 「**是词面相似，不是语义相似**，这一点我写在文档里，不含糊。
> 默认 embedder 是纯 stdlib 的 feature hashing（blake2b → 256 维、带符号累加、
> `1+log1p(tf)` 加权、L2 归一化），它没有语义。
> 我的基准里**反义句平均余弦 0.7153，比同义句的 0.6239 还高**
> （`fast`/`slow`、`enable`/`disable` 词面几乎一样），而 `Good morning` / `早上好`
> 的余弦是 **0.0000**。这两行数字就是"它不是语义检索"的铁证。
> 另外它是 **O(n) 线性扫描、没有 ANN 索引**（faiss 不可用），
> `max_items=10000` 时每次 search 是近万次点积——**我不会说它是"向量数据库"**。
> 生产环境注入 `RemoteEmbedder` 即可，接口完全一样。
> 把一个假的能力说成真的，才是这类项目最大的技术债。」

**追问更狠一层**：「这个基准能证明『检索质量提升』吗？」

> 「**不能**，我在 `VERIFICATION.md` 里专门写了『诚实边界』：
> 它只证明**打分公式与去冗算法的行为**，不证明检索质量——
> 因为默认 embedder 是词面的，sim 列整体换一个语义 embedder 就会变。
> 而且『哪些是冗余、哪些是过时』是我**人工标注**的，标签是这张表的真值来源，
> 不是我测出来的结论。所以脚本把 dup 两两余弦和各模式下的名次一并打印出来，
> 让读者自己核对。样本也只有 20 条记忆、1 条查询、top-5，是演示性的，不是检索评测集。」

*（实现：`memory/vector.py` 的 `_score_parts` / `search` / `_mmr_select`；决策：D-08；
数据：`docs/VERIFICATION.md` 的 `bench-retrieval-ranking` 与 `bench-embedding-similarity` 两段。）*

---

### Q9 ⭐ 上下文窗口怎么管？

**标准答案**（**三层**）：

> 「三层，从内到外：
>
> ① **滑动窗口**（`BufferMemory`）：**双约束**——同时受 `max_messages`（默认 50）
> 和 `max_tokens`（预算）约束，先到先裁；`keep_system` 保住 pinned system 提示，
> `keep_last_n` 保证最近 N 条一定在；被裁掉的消息进 `evicted` 队列，
> 并且有一层 `_repair_tool_pairs()` **修复被裁断的 tool_call / tool_result 配对**——
> 孤儿 `role="tool"` 消息会让某些 provider 直接 400。
>
> ② **摘要压缩**（`SummaryMemory`）：触发是 **OR 关系**——
> `current_tokens >= ceil(max_tokens * 0.8)` **或** `pending_evicted >= 4`。
> 为什么是 OR：这是两条独立的成本曲线，(a) 是"再不放摘要下次就超上下文"，
> (b) 是"积压的 evicted 已经够多，再攒只能丢"。压缩调 LLM 生成摘要，
> **LLM 失败就走抽取式兜底且不抛异常**，`previous_summary` 会并进下一次的 prompt。
>
> ③ **token 预算反推 + 最终闸门**（`MemoryManager`）：
> 这是我最想讲的一层。我一开始把窗口预算**写死 3000 token**，
> 但单条工具结果的字符上限是 **8000 字符**——按我自己的中文换算（1 字 ≈ 1 token），
> **一条 observation 就能把窗口打爆 2.6 倍**。
> 面试官问"你怎么保证不炸 context length"，我原来只能答"我写了 3000"。
> 所以改成**从 `context_window_tokens` 反推**：
> 减去给回复预留的 `reserve_completion_tokens`（1024）、减去工具 schema 预留、
> 再减去 system prompt 的实占，下限 512。
> 而且 `max_observation_chars` 的生效值取
> `min(config.max_observation_chars, tokenizer_chars_budget())`，
> 不再让三处各自写死 8000。
> 最后在 `abuild_prompt` 拼装完之后还有一道**总量校验** `_enforce_context_budget()`：
> 超了就按『**长期记忆块 → 窗口最旧消息**』的顺序裁，**不动 pinned system、不动最后一条 user**
> （最后一条 user 被裁掉就等于"这一轮没有输入"，模型会开始自由发挥），
> 并且**发一个 `CONTEXT_TRUNCATED` 事件**——降级必须可见，这是我给自己定的红线 12。」

**追问再深一层**：「token 数你怎么数的？」

> 「**自己估的，而且是中英分离的**：CJK 字符算 1 token，其余按 4 字符 1 token，
> 用 `functools.lru_cache` 缓存。因为最常见的那句 `len/4` **对中文低估 4 倍**——
> 40 个汉字它算 10 个 token，实际接近 40，结果就是窗口超预算、撞 context length 错误。
> 判 CJK 用**硬编码码点区间**（`0x4E00-0x9FFF` 等），不用 `unicodedata` 名字匹配，
> 因为那在不同 Unicode 版本下结果会变。
> 边界统一：空串 → 0，非空但估算 < 1 → 1。
> 而且它是个 `Tokenizer` 抽象，`tiktoken` 可用时自动切换，用户也能注入自己的实现——
> **三处用到 token 数的地方共用同一个函数**（窗口裁剪、摘要触发、记忆截断），
> 阈值才不会互相打架。我明确接受 ±20% 误差，偏保守（宁可高估）。」

*（实现：`memory/buffer.py` / `memory/summary.py` / `memory/manager.py`；决策：D-06、D-15。）*

---

### Q10 ⭐ 多 Agent 怎么防止无限委派？

**标准答案**：

> 「`HierarchicalAgent` 把每个子 Agent 包装成一个**普通工具**（`delegate_to_<name>`），
> 所以委派本身就是一次工具调用，复用同一套 executor。防无限委派有四道闸，
> 全部落在 `DelegationContext` 上：
>
> | 闸 | 字段 | 行为 |
> |---|---|---|
> | **环检测** | `would_cycle(name)` = `name in stack` | 返回一个 refused **字符串**（**不抛异常**），模型能看到"为什么被拒" |
> | **深度上限** | `depth >= max_depth` | 同上，refused + 事件 |
> | **预算** | `budget <= 0`（`max_rounds`，每层 `child()` 减 1） | 同上 |
> | **并发上限** | `threading.Semaphore(subagent_concurrency)` | `acquire(timeout=...)`，拿不到就返回 `busy` |
>
> **两个设计细节**：① refused 一律**编码成字符串回灌**、不抛异常——
> 因为对 manager 来说那只是一次"工具调用失败"，它可以自纠正（换一条路或直接给答案），
> 抛异常会让整条委派链崩掉；
> ② **环检测与深度判定要分开报**——我在事件和日志里把 `refused_reason` 区分成
> `cycle` / `depth` / `budget` / `busy`，否则会出现『看到 cycle 字样其实只是深度不够』的误导。
>
> 还有一个我自己抓出来的**完全无效的机制**：`subagent_concurrency` 原来是用
> loop-bound 的 asyncio 信号量实现的，但 delegate 工具是**同步工具**、
> 跑在 worker 线程里、闭包里再 `run_sync` **新建一个 loop**——
> 于是每个 delegate 拿到的都是**自己 loop 的全新信号量，计数永远是满的，并发完全不受限**。
> 后来一律改成 `threading.Semaphore`（红线 13）。
> 例外是 `arun_plan`：它跑在**父 loop** 里、`gather` 是 async 的，那里可以用 asyncio 原语——
> 规范里明确写出这个区别，避免实现者"为了统一"把两处写成一样。」

**追问再深一层**：「`arun_plan` 的依赖图是怎么跑的？」

> 「`Plan`/`SubTask` 带 `depends_on`。执行时先做**环检测**（依赖图里有环就直接报错，
> 而不是跑到一半卡死），然后一个 subtask 的依赖全部完成后才调度它，
> 用 `asyncio.gather` 并发跑可并发的部分，**并发峰值受 `subagent_concurrency` 约束**，
> 并且**单个 subtask 失败不中断整张图**（其他分支继续跑，失败信息进结果）。
> 这三条都有测试：断言执行顺序满足 `depends_on`、并发峰值 ≤ 上限、某个 subtask 失败不中断。」

*（实现：`multiagent/base.py::DelegationContext`、`multiagent/hierarchical.py` 的 delegate 闭包与 `arun_plan`；决策：D-21。）*

---

### Q11 为什么工具结果要截断？为什么 `ToolResult.content` **必须**是字符串？

**标准答案**：

> 「**截断**有两个原因：① 一条 8000 字符的中文 observation 按我的换算就是 8000 token，
> 能把窗口打爆（见 Q9）；② 工具输出不可信——`run_tests` 或 `read_file` 可能吐几十万字符，
> 不截断就是一次隐性的上下文炸弹。
> 我的截断是 `truncate_head_tail`（**头尾都保留**），不是简单 `[:N]`——
> 因为工具输出最有用的是**开头**（是什么）和**结尾**（成功/失败、错误码），
> 中间往往是大段重复内容。这个函数全项目只有一份实现，有独立测试。
>
> **`content` 必须是字符串**，这条是红线 9。理由：模型侧只认文本——
> 所有 chat API 的 tool result 都是文本载荷；一旦允许 `dict`/`list` 混进去，
> ① 序列化进 trace 会有类型不确定，② 文本 ReAct 模式的 `Observation:` 拼接会崩，
> ③ 每个下游都要写一遍"这个 content 到底是什么类型"的判断。
> 所以我们在**进 `ToolResult` 之前**就做归一化：`_stringify()` 把 `None`、`str`、
> `bytes`、`dict`、`list`、dataclass、异常……统一转成文本，
> 非字符串的原始返回放进 `metadata["extras"]`（结构化信息不丢，但走另一条通道）。
> 归一化规则本身有测试（`test_tools_executor.py` 的 `_stringify` 各类型用例）。」

---

### Q12 可观测性怎么做？

**标准答案**：

> 「三个层面：
>
> ① **强类型事件总线**：`EventType` 是枚举，`TraceEvent` 的 data 字段表是**冻结**的，
> 而且 `TraceEvent.__post_init__` 里**构造即校验**保留键（保留键写进 data 会直接抛 `ConfigError`，
> 而不是等到 `to_dict` 时静默覆盖）。回调是 `Callback` 接口，`on_event` 返回 `None`
> ——**回调不该能改控制流**（审批走独立通道，见 Q13）。
> `CallbackManager` 里任何一个回调抛异常都**不影响主流程**，只记日志。
>
> ② **唯一发射者矩阵**（这是我最想讲的一条）：我原来 `TOOL_STARTED` 在 Agent 步骤里发一次、
> 在 executor 里又发一次，`LLM_REQUEST` 也是。平时看 trace 只会觉得"事件好像多了一点"，
> 但一旦要**断言事件序列**（我确实有这种测试），就完全没有确定答案。
> 所以我冻结了一张**排他**的归属矩阵：每条事件**有且只有一个发射者**，
> LLM 的归 LLM 客户端、工具的归 executor、状态机的归 Agent，并明确规定"Agent 不得重复发"。
> 顺带修掉两个小 bug：一个事件的 data 里引用了一个**不存在的变量**；
> 另一个为了拿计数会**再调一次检索**——而检索是会改 `access_count` 的，重复调用就污染了数据。
>
> ③ **JSONL trace + 统计**：`JsonlTraceCallback` 写 JSONL（一行一事件，
> 多线程写不丢行，读的时候跳过坏行），`trace_stats(events)` 对固定事件序列
> 返回**整个 dict 精确定义**的统计：`runs / steps / llm_calls / tool_calls / tool_failures /
> retries / parse_errors / nudges / usage / cost_usd / llm_latency_ms(total,mean,p50,p95) /
> tool_latency_ms(按工具名) / errors`。
> 计费是显式纯函数 `estimate_cost_usd(usage, model=...)` + 一张**只有三条模型**的
> `MODEL_PRICES` 示例表，**未命中一律返回 `None`**——我宁可没有数字，也不要编一个数字。
> `cost_usd` 会写进 `AgentResult.metadata`，所以"这一次 run 花了多少钱"在 trace 里直接能看到。」

**追问再深一层**：「`--json` 输出能被 `jq` 消费吗？」

> 「能。我做了一条硬保证：**`--json` 模式下 stdout 只有 JSON，日志全部走 stderr**，
> 而且**绝不把 traceback 打到 stdout**——否则 `| jq` 直接崩。
> 退出码也是可控的：`0` 成功 / `1` agent 失败 / `2` 用法错误 / `3` provider 错误。
> 另外 `main(argv) -> int` **不调 `sys.exit`**，所以 CLI 能被单元测试直接断言返回值，
> 不用起子进程——`test_cli.py` 有 94 个用例就是这么写的。」

---

### Q13 ⭐ 重试策略怎么定？哪些异常重试、哪些不重试？

**标准答案**：

> 「我的重试是**默认拒绝（default-deny）**，不是默认重试。
> `LiteAgentError.retryable` 是一个**类属性**，默认为 `False`；只有四类标了 `True`：
> `LLMRateLimitError`（429，会尊重 `Retry-After`）、`LLMTimeoutError`、
> `LLMConnectionError`、`ToolTimeoutError`；`ToolExecutionError` **仅当工具作者显式标了
> `retryable=True`** 才重试。
> **把 `retryable` 做成异常类属性**的好处是：异常层次表**同时就是重试策略表**，
> 不用把 `isinstance` 判断散落在各处。
>
> **为什么默认拒绝**：agent 工具里有大量**有副作用**的操作。盲目重试一个超时的 `write_file`
> 会导致重复写入；用户在 trace 里看到"重试 3 次"却不知道为什么文件被追加了三次。
> **在 agent 系统里，一个被重复执行的副作用比一次失败严重得多。**
> 所以我加了一条更硬的规则：工具声明 `idempotent=False` 时，
> 在默认配置下**强制只尝试一次**。
>
> 还有一条更细的：**同步工具的 `ToolTimeoutError` 不重试**——超时杀不掉线程，
> 重试会再起一个线程去写同一份资源（见 Q5）。异步工具的超时才允许重试。
>
> 退避参数是**可复现**的：`rng_seed` 非空时用实例级 `random.Random(seed)`，
> 测试里配 `jitter=0.0` + 一个 `RecordingSleep`（只记录 delay、立即返回），
> 就能**精确断言**『重试了 2 次，间隔是 0.25s / 0.5s』。
> **这不是为了好看**：我有一个 429 的夹具带 `Retry-After: 60`，
> 如果不注入 sleep，那个测试就会**真的睡 60 秒**。」

**追问再深一层**：「重试之外，你怎么区分错误的性质？」

> 「我分了**两层**。第一层是常规的"可重试/不可重试"。第二层对 agent 更关键：
> **这个错模型自己能改吗**？我叫 `feedback_kind`：
> * `recoverable`（参数类型错、工具名拼错、arguments 非法 JSON）→
>   回灌文案**必须告诉模型怎么改**：可用工具名列表、schema 期望的类型、正确格式示例；
> * `infrastructure`（工具内部 bug、沙箱拒绝、超时、重试耗尽）→
>   文案**必须劝模型换策略**：`do not retry this tool with the same arguments;
>   try a different approach or give your final answer`。
>
> 这张映射表和 D-16 的熔断是**同源**的：`infrastructure` 失败计数进 `tool_failure_counts`，
> 连续 3 次熔断——**文案说别重试，机制也真的不让它重试**，
> 不会出现"嘴上说别重试、实际上放它一直重试"的分裂。
> 而且 `metadata["feedback_kind"]` 让 trace 里能统计『这次 run 里有多少失败是模型自己的锅、
> 多少是环境的锅』——这是排查 agent 表现最有用的一条切分。」

*（实现：`errors.py` 的 `retryable` 类属性、`config.py::RetryPolicy`、
`tools/executor.py::feedback_kind_for`；决策：D-07、D-24。）*

---

### Q14 ⭐ 怎么做离线确定性测试？`ScriptedLLM` 的设计为什么比 mock 好？

**标准答案**：

> 「因为环境里没有 pytest、也没有 mock 框架的魔法可用，我必须给 LLM 层造一个
> **脚本化假模型**：`ScriptedLLM([ScriptedResponse.tool('read_file', {...}),
> ScriptedResponse.text('我把 bug 修好了')])`，跑完 `llm.assert_exhausted()`
> ——**少调一次、多调一次都会失败**。
>
> **为什么它比 mock 好**（这是关键）：
> ① 我**没有 mock 内部函数**，而是把"模型"抽象成一个可编程的响应队列。
> 因为框架对 LLM 的依赖被收敛在 `LLMClient` **一个接口**上，
> 假模型就成了一个**合法的一等实现**，而不是测试补丁——它可以进 `examples/`，
> 可以驱动端到端演示。
> ② **它可以反向断言调用契约**：`llm.calls[i]` 记录 `ScriptedCall(messages, tools, kwargs, response)`
> ——第 i 轮模型**实际看到的** prompt、**实际拿到的**工具 schema、温度、`call_id` 全都在。
> 于是「记忆有没有被正确注入」「工具 schema 有没有真的传给模型」这类断言
> 可以直接写成**数据断言**。`FakeListLLM` 那种只能返回文本的替身做不到这一点，
> 而**工具暴露正是 agent 框架最容易出错的地方**。
> ③ **能精确构造罕见分支**：模型返回非法 JSON、返回空 `tool_calls`、
> 返回 `finish_reason="length"`、连续重试后失败——这些用真实 API 几乎没法稳定复现。
> ④ 结果是：**1652 个测试离线、确定性、不需要 API key，半分钟内跑完**。
> 里面还包含 `astream_chat` 的流式分支、scripted 的耗尽/循环、
> `calls[i].messages` 不被后续轮次污染的断言。」

**追问再深一层**：「确定性只靠 ScriptedLLM 够吗？」

> 「不够，时间、随机性、sleep 三样都得能注入，这是我专门做的一层：
> * **sleep**：`RetryPolicy.sleep_fn` / `ExecutorConfig.sleep_fn` / `LLMConfig.sleep_fn`，
>   **一切退避等待都必须经由它**（红线 16：禁止直接 `await asyncio.sleep`）；
> * **随机数**：`ToolExecutor.__init__` 与 `BaseLLMClient.__init__` **各创建一次**
>   `random.Random(seed)` 并复用——每次重建 rng 会让重试序列不可复现，断言就写不出来；
> * **时钟**：`config.utc_now` 是**唯一**时钟，测试用 `frozen_time` 一个 contextmanager
>   **只 patch 它**，不碰全局 `time.time`，避免干扰 logging 和 asyncio 自己的计时。
>
> 另外有两条测试卫生规则：`tearDown` 里必须 `reset_default_registry()`（否则
> `auto_register` 会污染别的测试）；断言前先查"禁止断言的字段清单"（防止 flaky）。」

*（实现：`llm/scripted.py`、`tests/helpers.py`；决策：D-26、D-03。）*

---

### Q15 ⭐ 这个项目最大的技术难点是什么？

**标准答案**（选**一个**讲透，不要罗列）：

> 「不是 ReAct 循环，也不是 schema 反射——那些是体力活。
> **最大的难点是并发语义在不同执行上下文里不一致**，具体说就是：
> **我的同步 API 每次调用都 `asyncio.run()`（新建 loop），而我的工具跑在线程池的
> worker 线程里，worker 线程里根本没有事件循环。**
>
> 这一个事实同时炸出四个问题：
> ① `asyncio.Semaphore` 在 `__init__` 里创建 → 第二次调用必崩（跨 loop）；
> ② `subagent_concurrency` 用 loop-bound 信号量实现 → **语义上完全无效**，因为每个 delegate
>   拿到的是自己新 loop 的全新信号量；
> ③ `sequential_tools` 的 asyncio 锁跨线程完全不串行 → 而它存在的**唯一理由**就是跨调用互斥；
> ④ `contextvars` 传递取消信号 → **读能穿透线程、写不能回传**，
>   而且 `to_thread` 能穿透、`run_in_executor` **不能**。
>
> 我最后的结论是一条规则：**跨线程/跨 loop 必须生效的限流与互斥，一律用 `threading` 原语；
> loop-bound 原语只允许用在"绝不离开当前 loop"的场景**（这条我升级成了红线 13）。
> 而 `threading` 原语没问题的前提是**临界区里不能有 await、不能有 I/O**——
> 我的记忆存储和黑板都满足这个前提（纯内存 dict 操作），
> 所以 `MemoryStore` 用**同步 API + RLock** 反而是对的：
> **同步 API 是唯一能同时服务"主 loop"和"worker 线程"两侧的形态。**
>
> 最让我警惕的不是这些 bug 本身，而是**它们的测试会通过**——
> `subagent_concurrency` 和 `sequential_tools` 的单 loop 测试**全绿**，缺陷被完全掩盖。
> 所以修完之后我补的是**测试**：『两个嵌套 loop 下仍真正串行』
> 『跨两次 `asyncio.run` 复用同一 executor 且池内条数为 0』。」

---

### Q16 ⭐ 如果让你重做，你会改什么？

**标准答案**（要有具体的东西，不要说"我会写得更好"）：

> 「三件事，按优先级：
>
> ① **先把"执行上下文"这件事在规范里写死**。我上面说的四个坑（Q15）本质上是同一个根因：
> 我一开始没在文档里明确『哪些代码在哪个 loop、哪个线程里跑』。
> 重做的话我会在规范第二章就画一张**执行上下文表**：
> 每类调用（Agent 主循环 / executor / worker 线程 / delegate / CLI 同步入口）
> 分别运行在哪个 loop、哪个线程、能用哪种原语。
> 这些 bug 如果等到写完 24000 行再发现，修复成本是现在的几十倍。
>
> ② **`MemoryStore` 的接口应该一开始就为"将来换成真 I/O"留形**。
> 现在是同步 API + RLock，前提是"临界区只有内存操作"。
> 如果以后后端换成数据库或 Redis，同步 API 会阻塞事件循环。
> 接口不用变（实现改成 `await asyncio.to_thread(...)`），
> 但**这个前提必须写在读者第一眼能看到的地方**。
>
> ③ **`MODEL_PRICES` 那张价格表不该硬编码在库里**。
> 它是我唯一一处"疑似硬编码业务常量"，虽然我明确标了"示例价格、会变、未命中返回 None"，
> 但更干净的做法是把它做成必须显式传入的参数，库里一个默认值都不给。
> 我宁可调用方写两行，也不想让用户以为那个数字是准的。
>
> 另外还有一件我**已经做对了、希望面试官注意到**的事：
> 我在设计阶段就用 5 个对抗性视角去攻击自己的规范（异步并发 / 可测性 / 面试价值 /
> 一致性 / 简历覆盖度），**真的抓出了 12 个 blocker 级问题**——
> 包括那个 asyncio 原语绑定 loop 的"第二次调用必崩"。
> 所以"重做"这件事里我最想保留的就是这一步。」

---

### Q17 HITL 审批为什么是 fail-closed？

**标准答案**：

> 「`requires_approval=True` 的工具，如果调用方**没有配置审批回调**，
> 我的行为是**拒绝执行**，而不是放行。
> 理由很简单：一个安全机制在**最坏情况下**（用户忘了配 policy）必须往安全侧倒。
> 如果"没配就放行"，那忘配一次就等于这个工具从来没被保护过。
>
> 两个设计细节：① 审批的位置是『**参数校验之后、真正执行之前**』——
> 这样模型不会因为参数错就弹一次审批；
> ② 我**刻意没有改回调协议**——`Callback.on_event` 返回 `None` 是个已经冻结的好设计
> （回调不该能改控制流），审批走**独立**的 `approval_policy` 回调，返回 `bool`。
>
> 另外 `dangerous` 和 `requires_approval` 是**两个不同的东西**：
> `dangerous` 只管**展示过滤**（`registry.list(include_dangerous=)`），
> `requires_approval` 管**控制流**。把它们合成一个 bool 会同时失去这两种语义。
>
> 这套东西还顺手让 `AgentAbortedError` 有了**真实的触发路径**：
> 用户在审批回调里抛它 → executor 转成取消 → `Agent.arun` 走取消路径 →
> `state.mark_finished(ABORTED)` + `RUN_FAILED {aborted: True}`。
> 在这个设计之前，那个异常类是**全文没有任何触发路径**的死类——面试官一追问就穿帮。」

---

### Q18 `finish_reason` 为什么要进控制流？

**标准答案**：

> 「因为我发现了一个**最坏的结果**：模型被 `max_tokens` 截断时，
> `finish_reason` 是 `"length"` 而且**没有 tool_calls**，
> 而我原来的终结条件只是"没有 tool_calls"——于是**半截答案被判定为成功的最终答案**，
> `status=FINISHED` 返回。调用方以为成功了，下游拿到的是不完整数据。
>
> 修法是按 `finish_reason` 分支：
> * `length` → 注入一句『你的输出被截断了，请从断点继续』并**限次续写**（默认 1 次，
>   否则又是一个无限循环面）；
> * `content_filter` → **立即失败**并记录原因（重试同一个 prompt 只会得到同一个结果，
>   这是不可恢复的）；
> * `finish_reason == "tool_calls"` 但 `calls` 为空 → 这是模型"说要调工具但没给出合法调用"，
>   属于**可自纠正**的错误，走自纠正分支，**不能当成最终答案**。
>
> 终态判定也从"无 tool_calls"收紧成"无 tool_calls **且** `finish_reason == 'stop'`"。
> 这条改动让原生模式特有的一种失败，文本模式也能通过同一条分支被正确处理。」

---

### Q19 `ToolResult` / `Message` 为什么用 `dataclass` 而不是 pydantic？

**标准答案**：

> 「环境里 pydantic 2.13.5 是**可用**的，但我依然选了 dataclass，三个理由：
> ① **零依赖红线优先**——核心结构用 pydantic，"零依赖"就是假的，
> `test_zero_dependency.py` 也守不住；
> ② **语义更贴合**——agent 状态是**可变累积**的（每轮 append 消息、累加 usage、改 status），
> pydantic v2 默认 `validate_assignment=False` 且鼓励不可变风格，用它反而要频繁 `model_copy`；
> ③ **trace 需要字段全量输出**——pydantic 的 `model_dump(exclude_none=True)` 是常见默认，
> 会让 trace 结构随内容变化，测试没法做精确字典比对。
>
> 代价我也清楚：**失去运行期类型校验**，所以要手写 `to_dict()`/`from_dict()`
> （`from_dict` 缺字段抛 `SerializationError`），工具参数校验交给独立一层
> `validate_instance`（那层本来就必须手写，要校验的是 JSON 不是 Python 对象）。
>
> pydantic 我只当成**可选的鸭子类型适配器**用在 schema 生成里——
> 识别 `pydantic.Field` 元数据和 `BaseModel` 注解，但**框架不 import 它**，
> 环境里没装 pydantic 也能跑全部功能。」

**追问再深一层**：「性能上 dataclass 真的更快吗？」

> 「**只有一格站得住，我说实话**。我写了一个交错测量的微基准
> （`benchmarks/bench_dataclass_vs_pydantic.py`，A/B 逐轮交替、每侧 best-of-20、跑 5 轮，
> 看**比值的分布**而不是单次结果，并且预先定了噪声带）：
> * **构造 1000 条消息**：dataclass **3.258 ms** vs pydantic **11.965 ms**，
>   比值中位数 **3.68x**，区间 `[3.62x, 3.74x]` → dataclass 更快，跨轮稳定；
> * **序列化**：比值中位数 **0.79x** → **pydantic 更快**，我不藏这一点
>   （`model_dump()` 是 pydantic-core 的 Rust 递归序列化器，输给它不丢人，
>   而且这一格根本不在瓶颈上）；
> * **序列化 + `json.dumps`**：**测不出差异**（比值跨过 1.0），
>   因为时间被 stdlib 的 JSON 编码器吃掉了。
>
> 所以我的选型理由是**零依赖和语义匹配，不是性能**。
> 而且这个基准教了我一件事：**不做交错测量时，同一份代码在同一台机器上的比值
> 在 0.71~1.71 之间乱跳**——单次跑出来的方向根本不可复现。
> 所以我报的是比值的 min/median/max + 噪声带，跨过 1.0 就明说"测不出差异"。
> **能说出"这一格我测不出来"，比硬报一个单次数字更可信。**」

---

### Q20 工具结果为什么要按工具名截断 / 结果太长怎么办？

见 Q11（截断与字符串化）。**补充追问**：「截断会不会让模型误判？」

> 「会，所以我做的是**头尾保留式截断**（`truncate_head_tail`）而不是砍尾巴，
> 并且在中间插入省略标记。另外**摘要压缩**走的是抽取式兜底（LLM 失败时），
> 也是保证不静默丢信息。更根本的一条是红线 12：
> **任何降级（schema 降级、embedding 用哈希、摘要用抽取式、上下文被裁、工具被熔断）
> 必须留下可观测痕迹**，不许静默。」

---

### Q21 为什么要给内置工具做「沙箱 + 复杂度闸 + 注入白名单」？

**标准答案**：

> 「因为内置工具是**唯一真正碰宿主系统**的地方，而它们的参数是**模型生成的**。
> 三层：
> ① **文件工具**：所有路径相对 `PathSandbox` 根解析，`..`、绝对路径、**symlink 逃逸**
> 一律抛 `SandboxViolationError`；`metadata["sandbox"]` 存 realpath；
> `delete_file` 必须显式 `confirm=True`（删除不可逆，让"删"多写一个参数
> 等于给模型一次自检，也让审批层有一个稳定的判定点）。
> ② **shell 工具**：一张 `SHELL_DENY_PATTERNS` 正则表（`sudo`、`rm -rf /`、`mkfs`、
> `dd if=`、`curl | sh`、fork bomb……）先拦一道；而且**默认禁用**——
> 要 `LITEAGENT_ALLOW_SHELL=1` 才开，没开时返回
> `shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)` 而不是抛异常。
> ③ **代码执行**：`python_eval` 不是 `eval()`，而是**AST 白名单**——
> 只放行 `BinOp/UnaryOp/BoolOp/Compare/Call(仅白名单函数)/Subscript/IfExp…`，
> `Attribute` 只允许 `str/list/dict` 的只读方法；`__import__`/`getattr`/`eval`/`type` 全禁；
> 还加了一道**复杂度闸**（`10**10**10`、`range(10**7)` 这类直接拒），
> 防的是"语法合法但 CPU 爆炸"的输入。
> `python_exec` 和 `run_tests` 走子进程 + `timeout=`，超时抛 `ToolTimeoutError`。
>
> 我明确**不支持** f-string、Lambda、推导式、`*` 解包——因为这些会让 AST 白名单
> 迅速膨胀，而收益很小。这是**有意的能力边界**，写进了文档。」

**追问再深一层（安全题，必须主动降调）**：「这是安全审计吗？」

> 「**不是，这是"已知模式的拦截"，我在 `VERIFICATION.md` 的 U-6 里标成未验证。**
> 我只测了 `SHELL_DENY_PATTERNS` 逐条命中、AST 白名单与复杂度闸的行为，
> **没有做过任何真实的逃逸尝试**（编码绕过、`ctypes`、资源耗尽）。
> 正则是拦不住有心人的，它的价值是挡住模型**无意的**破坏性命令——
> 对模型来说已经够了，对攻击者来说完全不够。这一点我不会含糊。」

---

### Q22 双 API（同步 + 异步）是怎么设计的？

**标准答案**：

> 「默认 **async-first**：先写 `a*`，同步版是薄包装。
> 但有**四个 sync-first 例外**：`Blackboard` / `MemoryStore` / `Tokenizer` / `BufferMemory`
> ——因为 worker 线程里没有事件循环（见 Q15），纯 async API 它们用不了。
> 方向**不许反**，这条写在规范里。
>
> 同步包装的签名我踩过一个坑：原来是 `run_sync(coro)` 收**协程对象**，
> 但如果当前已经在一个运行中的 loop 里，我是**先构造协程再检查**的——
> 抛 `ConfigError` 之后那个协程永远不会被 await，Python 会打
> `RuntimeWarning: coroutine ... was never awaited`。我有 **5 个**同步包装 API，
> 全都会触发；开 `-W error` 之后测试**直接失败**。
> 修法是把签名改成收**工厂函数**：`run_sync(lambda: self.achat(...))`
> ——在**"不该执行"的路径上根本不产生协程对象**，
> 比"记得在异常前 close"可靠得多。而且 `lambda:` 这个写法本身就逼调用方意识到
> "这段代码是在另一个线程/另一个 loop 里执行的"。」

---

### Q23 长期记忆真的"长期"吗？

**标准答案**：

> 「一开始**不是**。我的 `VectorMemory` 最初是纯内存 dict，
> 进程一退就全没了——但简历上写的是"长期**存储**"，那这个词就是虚的。
> 所以我自己把它补成了真的：`VectorMemory.save(path)` / `load(path)`，
> JSONL 格式，一行一个记忆条目（带 embedding），人可读、可 `grep`、可增量写、**坏行跳过**；
> `MemoryManager` 上还有 `persist` / `restore`。
> 加载时做**维度校验**：embedding 维度与 `self.dim` 不符就抛 `MemoryStoreError`。
> 测试断言『往返之后条数与检索顺序一致、embedding 逐元素相等、坏行跳过、维度不符报错』。
>
> 我的原则是：**如果只做检索不做持久化，我就把简历改成"向量检索 + 可注入持久化后端"——
> 宁可把话说明白，也不要让简历上的一个动词落空。**
> 代价也说清楚：这是简单 JSONL，**没有 schema 版本迁移**，格式兼容性由用户负责。」

---

### Q24 你用了什么设计方法来保证这么多模块能并行写出来还不打架？

**标准答案**：

> 「**先冻结接口，再并行实现**。流程是：
> 架构师起草 `INTERFACES.md` → 5 个对抗性审查视角并行开火
> （异步并发 / 可测性 / 面试价值 / 一致性 / 简历覆盖度）→ 规范负责人逐条评判
> （采纳 or 驳回并说明理由）→ 冻结 → 分 4 个波次并行实现。
>
> 关键手段是 **§1.2 的「封闭文件清单」**：41 个文件写成封闭清单，
> 规定"不得新增、改名、删除"。好处是并行实现者**不可能撞车**
> （每个文件恰好一个 owner），也不会出现"两个人都造了一个 helper 模块"。
> 配合 §1.1 的**单向依赖 DAG + 12 条同层白名单**，
> `test_zero_dependency.py` 还能**用 AST 自动验证依赖方向没被破坏**。
>
> 这一步**真的抓到了 bug**：12 个 blocker 级问题，举三个最有代表性的——
> ① ReAct 循环的**重复写入**（parse error 时 assistant 原文被写两遍）；
> ② `asyncio.Semaphore` 跨事件循环崩溃；
> ③ `asyncio.to_thread` 的**静默陷阱**：把 async 函数交给 `to_thread`，
> 它**不会被执行**，只是返回一个 coroutine 对象然后被丢掉——工具"看起来跑了"但什么也没干。
> 这三个如果等到写完 24000 行再发现，修复成本是现在的几十倍。」

---

### Q25 你的测试里有没有"为了通过而写"的测试？

**标准答案**（诚实题，要有）：

> 「有，我拆掉过几个，也修过一个。举两个真事：
>
> ① `test_zero_dependency.py` 本来是**文本子串匹配**去查 3.11 API 的，
> 但那会误报（字符串里出现 `asyncio.TaskGroup` 也算）。我改成了 **AST 判定**。
> ② 更重要的是：**我的 `sequential_tools` 串行测试原来是单 loop 的，
> 它是通过的，但它掩盖了一个真实缺陷**——loop-bound 锁跨线程完全不串行。
> 我是后来重新推演并发路径时发现的，修完之后补的是**两个嵌套 loop 的用例**。
> 这件事教我的不是"要多写测试"，而是
> **测试必须覆盖真实的执行上下文，否则绿得越整齐越危险**。
>
> 另外我有几处**主动标注的诚实边界**：真实 provider 报文只用手写的 `FakeTransport`
> 验证了请求构造与响应解析，**没有对真实端点跑过**；
> embedding 只证明了词面相似；基准只证明了算法行为、没证明检索质量。
> 这些我都写进 `docs/VERIFICATION.md` 的 `status` 列（`offline-verified` /
> `code-only-not-run` / `not-implemented`）。
> 我觉得**一个项目最值钱的部分不是它证明了什么，而是它诚实地标出了它没证明什么**。」

---

### Q26 流式（streaming）做了吗？

**标准答案**（**这题必须诚实，因为答案有一半是"没做"**）：

> 「一半做了，一半**明确没做**，我在 `VERIFICATION.md` 的 U-4 里标成未验证。
>
> **做了的**：`ScriptedLLM.astream_chat` 的流式路径——
> 单 chunk 全量、多 chunk 的顺序与 index、`finish_reason`、`stream_error` 中断，
> 以及 `Agent.astream` 的事件顺序，还有一条我比较在意的测试：
> **消费者提前 `break` 不泄漏 task**（把迭代器丢掉时后台任务必须被收掉）。
>
> **没做的**：真实 provider 的流式。**不是"没测"，是"没实现"**——
> `LLMClient.astream_chat` 的基类实现直接抛 `NotImplementedError`，
> `HTTPChatClient` **不覆写它**，调用方必须捕获并回退到 `achat`。
>
> **为什么不假装有**：我的 `Transport` 接口只提供**整段响应**。
> 在它之上做"伪流式"（先拿完整响应、再切成 chunk 发出来）只会**假装在流**——
> 首 token 延迟一点没降，反而多了一层事件，用户更难排查。
> 要做真流式，得先给 `Transport` 加一个 `stream()` 接口（返回行迭代器），
> 再让两个 provider 各自处理 SSE / chunked 编码。**这是我知道该怎么做、但这一版没做的事。**」

---

## 4. 可讲的踩坑故事

> 每个故事按「**现象 → 排查 → 根因 → 修法 → 学到什么**」讲。
> 挑 1~2 个讲透即可，不要背完。标 🎯 的是**我本机现场复现过**的，最值得讲。

### 故事 1 🎯 `asyncio.Semaphore` 跨事件循环：第二次调用必崩，**而且只在争用时崩**

* **现象**：`Agent.run()` 第二次调用抛
  `RuntimeError: <asyncio.locks.Semaphore object ...> is bound to a different event loop`。
* **排查**：我第一次复现时 value=1、只有**一个** acquirer，第二次调用是**正常的**——
  差点得出"没问题"的结论。改成 `gather` 两个任务去抢同一把信号量，第二次调用**立刻崩**。
  这条"只有争用才崩"的性质，就是它能潜伏到生产的原因。
* **根因**：`asyncio.Semaphore.acquire()` 只有在**必须等待**时（有人持锁且有 waiter）
  才把当时的 loop 记进 `self._loop`。而我的 `Agent.run()` **每次都是一次 `asyncio.run()`**，
  每次新建 loop → 第二次复用同一个 executor 就撞上绑定在旧 loop 上的原语。
* **修法**：`LoopBoundPool`——所有 asyncio 原语**按运行中的 loop 懒创建**，
  键是 `id(loop)`、`_loops` 强引用保活，没有运行 loop 时入口直接抛 `ConfigError`；
  退出时显式 `release_loop` / `aclose`。同时升级成红线：**跨线程/跨 loop 必须生效的
  限流与互斥一律用 `threading` 原语**。
* **学到什么**：**"单线程不争用"的通过毫无意义**。
  一个并发缺陷如果在串行路径上不复现，那它的测试大概率是绿的——
  而绿色的测试会让你彻底放心。我后来给这类东西统一加"跨两次 `asyncio.run`"的用例。

### 故事 2 🎯 用 `WeakKeyDictionary` 缓存 asyncio 原语 → **内存泄漏**

* **现象**：我原以为 `WeakKeyDictionary[loop] → primitive` 是标准做法（"loop 死了就自动清理"）。
  实测：**跑 3 次 `asyncio.run`，缓存里剩 3 个 key**，3 个 loop 全泄漏。
* **排查**：把缓存里的 value 拿出来看 `sem._loop` —— **不是 None**。
* **根因**：**value 强引用 key**。`asyncio.Semaphore` 争用后把 loop 存进 `self._loop`，
  于是 `weakref` 指向的 loop 永远有一个强引用者，**弱引用永不失效**。
  弱引用的前提是"value 不反向引用 key"，我违反了它。
* **修法**：改成普通 `dict[int, ...]` + 显式 `release_loop` / `aclose`，
  并加断言『连续 3 次同步调用后池内条数为 0』。
* **学到什么**：**弱引用的正确性依赖"value 不引用 key"这个不成文的约定**，
  而第三方对象（`asyncio.Semaphore`）的内部实现你控制不了。
  用弱引用缓存别人的对象之前，先问一句"它的 value 里会不会存 key？"

### 故事 3 🎯 `contextvars` 的三条提交路径行为不一致（**读能穿透，写不能回传**）

* **现象**：我按"contextvar 能穿透线程"的直觉写了协作式取消，
  结果 `current_cancel_flag()` 在**同步工具里恒为 None**，取消机制**静默失效**。
* **排查**：我写了一个最小复现——同一个 contextvar，三条路径各跑一次：

  ```text
  asyncio.to_thread(fn)            -> FLAG
  loop.run_in_executor(tp, fn)     -> None
  裸 ThreadPoolExecutor.submit(fn) -> None
  ```

  和我预期完全相反：**只有 `to_thread` 会复制 context**。
* **根因**：`asyncio.to_thread` 内部走 `copy_context().run`；
  `run_in_executor` **不复制 context**。而我自己的 executor 走的正是 `run_in_executor`
  ——**恰恰是同步工具（不可中断）最需要取消信号，而它正好拿不到。**
* **修法**：同步分支自己包一层
  `ctx = contextvars.copy_context(); loop.run_in_executor(tp, functools.partial(ctx.run, tool.run, args))`。
  并且发现第二个方向性问题：**写不回传**，所以 `flag.set()` 只能发生在 executor 所在线程。
  还有一条：`cancel_scope()` **必须每个 attempt 重新进一次**——
  提到重试循环外，第二次尝试拿到的是**已 set 的同一个 Event**，重试全部秒失败。
* **学到什么**：**"读能穿透线程"这句话是有前提的，而前提写在别人的实现里。**
  凡是依赖"隐式上下文传播"的机制，一定要**三条路径各测一遍**再写进规范。

### 故事 4 近因分 `2 ** (-age_days / half_life)` 在负 age 下**数值爆炸**

* **现象**：`VectorMemory.search()` 抛 `OverflowError: Numerical result out of range`，
  整条记忆检索链路炸掉。
* **排查**：触发场景是**时钟回拨**、或**从磁盘恢复出未来时间戳的数据**——
  此时 `now < created_at`，`age_days` 为负，`-age_days/half_life` 变成**正数**，
  `2.0 ** 大正数` 直接溢出。是测试作者在写"混合打分公式**手算对照**"用例时踩出来的。
* **根因**：**规范本身的空缺**——§8.5.1 只定义了 `age >= 0` 时的公式，
  没写负 age 怎么办。所以这不是"实现有 bug"，是"规范有个洞"。
* **修法**：按规范精神把 `age_days` **钳制下界为 0**（语义是"未来的时间戳 = 现在"，
  单调安全：越老的记忆 recency 越小），**并且把这个裁决记进文档**。
* **学到什么**：**测出 bug 不稀奇，测出"规范本身的空缺"才值钱。**
  而且这个 bug 只有写了**手算对照**的精确断言才会暴露——
  如果只写"大差不差"的模糊断言，它会一直潜伏到线上遇到时钟回拨。
  我也是在那之后才把 `now` 一路透传到 `MemoryManager.retrieve/abuild_prompt`，
  否则断言会随运行时刻漂移。

### 故事 5 假传输层把 **HTTP 429 当成 200** 解析

* **现象**：限流（429）的测试**全绿**，但它是**假绿**——真实世界里必然失败。
* **排查**：看 `FakeTransport` 的实现：它只**回放响应**、不映射状态码。
  而 §6.2 对 `Transport.send` 的契约是"**失败必须以 `LiteAgentError` 子类抛出**"。
* **根因**：假实现**没有遵守真实现的契约**。这类 bug 最危险的地方是：
  它让"照着文档写限流测试的人"得到一个**永远通过**的错误保证。
* **修法**：给假传输层补上 `map_http_error` 全表（401/403/429 带 `Retry-After`/400/404/500/未知），
  并为"错误映射"单独写测试。
* **学到什么**：**测试替身必须和真身遵守同一份契约，否则它保护的是错误的东西。**

### 故事 6 `run_sync(coro)` 收到协程对象 → `never awaited` 警告把测试打挂

见 Q22。**现象**是 5 个同步包装 API 全打 `RuntimeWarning: coroutine ... was never awaited`，
`-W error` 下**测试直接失败**；**根因**是"先构造协程再检查 loop"，
异常路径上协程永远不会被 await；**修法**是签名改成收**工厂函数**
（`run_sync(lambda: self.achat(...))`），**在"不该执行"的路径上根本不产生协程对象**；
**学到什么**：修警告的正确姿势是**改结构让它不可能发生**，而不是"记得在异常前 `close()` "。

### 故事 7 事件被两层各发一遍 → **事件序列无法断言**

见 Q12 ②。**现象**：trace 里"事件好像多了一点"，但要断言事件序列时**完全没有确定答案**；
**根因**：没有规定"谁该发这条事件"，`TOOL_STARTED` 在 Agent 与 executor 各发一次；
**修法**：冻结**排他**的归属矩阵（每条事件有且只有一个发射者，Agent 不得重复发），
顺带修掉"data 里引用不存在的变量"和"为了拿计数又调一次 search 而污染 `access_count`"；
**学到什么**：**"看起来多一点"的冗余，代价是你失去了整条 trace 的可断言性。**

---

## 5. 现场演示脚本

> 三条命令，**全部不需要网络、不需要 API key**。每条先说"该让面试官看什么"。

### 演示 1：跑测试 —— 证明"离线、确定性、有规模"

```bash
cd /home/ml-user/workdir/project-3

# (a) 最相关的两个模块：schema 反射 + 工具执行器，210 个用例
python3 -m unittest tests.test_tools_schema tests.test_tools_executor -v 2>&1 | tail -5

# (b) 全量
python3 -m unittest discover -s tests -t . 2>&1 | tail -3
```

**让面试官看什么**：

* **(a)** —— `Ran 210 tests in 2.419s ... OK`。这两个文件刚好覆盖我最有话说的两块：
  `test_tools_schema.py` 91 例（type hints → JSON Schema 的每条边界）、
  `test_tools_executor.py` 119 例（并发顺序 / 峰值上限 / 同步工具孤儿线程 / 熔断 / 审批三例 / 校验覆盖率缺口）。
  **2.4 秒**跑完 210 例，这就是"零依赖 + 无网络"换来的东西。
* **(b)** —— `Ran 1652 tests` → `OK`。**没有网络、没有 key、半分钟**。
  可以顺手指出：整套测试是靠 `ScriptedLLM` 驱动的，**不是靠 mock 桩内部函数**。
* `[v3 修正]` **诚实提醒**：上一版这里写着「全量跑会有 1 个失败（§12 冻结的 4 个测试文件
  尚未落地）」，并教候选人**主动**说出这个失败。那段话已经**失效** ——
  4 份文件都已落地、守门测试通过、套件全绿（`Ran 1652 tests` / `OK`）。
  照旧话术去说，面试官当场敲一条命令就会看到 `OK`，反而显得「我从没真跑过」。
  现在的诚实边界请改用 §6 第 8 条的版本（真实 provider 端点未验证）。

### 演示 2：跑代码助手示例 —— 证明"框架真的能用"

```bash
python3 examples/07_code_assistant.py --offline
```

**让面试官看什么**（我实跑的关键输出）：

* 结尾的 `离线验收：4 条断言全部通过 ✓（文件真被改写、测试真在子进程里跑过）`
  —— 这是"端到端可用"而不是"单元测试通过"；
* `trace_stats` 那一段：`事件总数 54`、`steps 6`、`llm_calls 6`、`tool_calls 5`、
  `tool_latency_ms` **按工具名**给出的 count/mean/p95、`usage` 与 `cost_usd`；
* 最后一行说明：**沙箱是 `tempfile.TemporaryDirectory()`，main 返回后自动清理**——
  这个示例**永远碰不到仓库或用户 home 下的真实文件**（`PathSandbox` 是第二重保险）。
  这一行同时展示了 Q21 的安全设计。
* 顺带可以说：这条路径是 `Makefile` 的 `make demo`，另外还有 `make demo-live`
  （需要真 key，跑同一个场景的联网版）。

### 演示 3：展示 JSON Schema 自动生成 —— 证明"装饰器真的在反射"

```bash
python3 - <<'PY'
from typing import Annotated, Literal, Optional
from dataclasses import dataclass
from liteagent.tools import tool
import json

@dataclass
class Filter:
    field: str
    op: Literal["eq", "gt"]

@tool
def search(query: str, limit: int = 5, exact: bool = False,
           tags: Optional[list[str]] = None,
           filters: Annotated[list[Filter], "nested dataclass"] = None) -> str:
    """Search the index.

    Args:
        query: the search phrase
        limit: max hits
    """
    return query

print(json.dumps(search.spec.parameters, indent=2, ensure_ascii=False))
print("WARNINGS:", search.spec.warnings)
PY
```

**让面试官看什么**（这是我实跑的输出要点）：

* `query` → `{"type":"string","description":"the search phrase"}` ——
  描述来自 **docstring 的 Google 风格 Args**，不是手写的；
* `limit: int = 5` → `"integer"`；`exact: bool = False` → **`"boolean"`**，
  **没有变成 `"integer"`** —— 指出来：`isinstance(True, int) is True`，
  所以我的映射表里 **`bool` 必须先于 `int` 判定**，顺序写反就会静默出错；
* `tags: Optional[list[str]]` → `{"type":"array","items":{"type":"string"}}`，
  **没有 `"type": ["array","null"]`** —— 这就是 D-05：
  `Optional` 只用来决定"不放进 `required`"，因为 function calling 生态对联合类型支持很差
  （OpenAI strict mode 不接受 null 联合，很多兼容端的 schema-to-grammar 转换遇到
  `type` 数组会直接崩）；
* `filters: Annotated[list[Filter], ...]` → **嵌套 dataclass 被递归展开**成内联 object
  （`required: ["field","op"]` + `additionalProperties: false`），
  而 `Annotated` 的字符串元数据变成了这一层的 `description`；
* `required` 里**只有 `query`** —— 有默认值的都不进 required；
* `WARNINGS: ()` —— **没有降级**。可以补一句：
  「如果我写一个 `Union[int, str]` 或者一个递归 dataclass，这里会出一条 warning，
   schema 降级成 `{}`——**降级必须留痕**，这是我定的红线 12，因为我宁可让模型看到
  『这个参数是任意类型』，也不要让它看到一个我猜错的类型。」

---

## 6. 可能的减分项与应对

> 原则：**主动说边界，不要等被问穿**。面试官几乎一定会去点这几个地方。
>
> **权威台账在 `docs/VERIFICATION.md`**（`claim | evidence | status` 三列，
> `status` 只有 `offline-verified` / `code-only-not-run` / `not-implemented` 三个取值）。
> 它有两节是面试时的护身符，**建议面试前把这两节的编号背下来**：
> * **§3「未验证 / 只到代码层」U-1 ~ U-10** —— 逐条列出我没在真实条件下跑过的能力
>   （真实 provider 端点、真实搜索引擎后端、真实 embedding 服务、流式、
>   性能压测、沙箱对抗、真实时钟、畸形 HTML、跨进程持久化、多机）；
> * **§4「已知缺口」G-1 ~ G-5** —— 但**当前真正成立的只有两条**：G-2（持久化的跨进程往返
>   只做到同进程）、G-4（校验覆盖率缺口的递归深度上限 4）。G-1/G-3 已关闭（冻结清单的测试文件
>   与文档都已落地），G-5 是「设计如此、已留痕」（跑完套件后 `LoopBoundPool._ALL_POOLS`
>   仍积着 371 个池对象，但**线程数实测为 0**）。
>
> **这张表的作用不是说"我很差"，而是说"我知道我的边界在哪"。**
> 面试官问「这里面你最没底的是哪一条」，就直接答 **U-1**：
> 「三个真实 LLM 端点我一次都没调过，无网无 key。我只验证了请求构造、响应解析、错误映射，
> 走的是手写的 `FakeTransport`。**"换成真 key 就能跑通"是我的推断，不是实测结论。**」
> —— 这句话我在 `VERIFICATION.md` 里就是这么写的。

| # | 减分项 | 怎么答才不掉分 |
|---|---|---|
| 1 | **没有真实 API key 验证过 provider** | 「对，这是最大的诚实边界。OpenAI / Anthropic / DeepSeek 三个适配器我**只用手写的 `FakeTransport` 验证了请求体构造与响应解析**，没有对真实端点跑过——本环境无外网。我在 `docs/VERIFICATION.md` 里把这三行标成 `code-only-not-run` 而不是 `offline-verified`。**但设计上我把这个风险隔离了**：所有 HTTP 走一个 `Transport` 接口，真实现是 `UrllibTransport`，所以"报文编解码错"和"网络层错"是两个可独立排查的问题。」 |
| 2 | **单人项目 / 没有协作与评审** | 「是单人项目，但我用了一个**替代机制**：设计阶段用 5 个对抗性审查视角并行攻击自己的规范（异步并发 / 可测性 / 面试价值 / 一致性 / 简历覆盖度），**抓出了 12 个 blocker 级问题**。另外每个实现者都被要求填 `not_done` 字段，如实列出没做到的部分——这直接变成了 `VERIFICATION.md` 里 `code-only-not-run` 那一列的素材。**没有 code review 的时候，我把"对抗"做成了流程。**」 |
| 3 | **没做生产级压测 / 没有性能数据** | 「对，我只做了三个**微基准**，而且每个都说明了方法上的局限。我刻意**没有**报"QPS / 吞吐"这类数字——因为在一个共享机器上跑单进程 CPython 的墙钟时间，报吞吐就是骗人。我报的是**比值的分布 + 噪声带**，跨过 1.0 就写"测不出差异"。**能说出"这一格我测不出来"比硬报一个数字更可信。**」 |
| 4 | **没有人用 / 没有 star / 不是开源项目** | 「这是个人项目，目标是**把三层架构与 ReAct 循环真的做出来**，不是做一个产品。我用「可验证性」代替「用户量」：1652 个离线测试、三份基准、一份逐条对账的验证表。如果您想，我现在就能跑给您看。」 |
| 5 | **检索是词面相似、不是语义** | 「对，我在文档里明确写了。默认 embedder 是纯 stdlib 的 feature hashing，**反义句的平均余弦（0.7153）比同义句（0.6239）还高**，而中英翻译对的余弦是 **0.0000**——这两行数字就是"它不是语义检索"的证据。而且它是 **O(n) 线性扫描、没有 ANN 索引**。**我不会说它是"向量数据库"。** 生产环境注入 `RemoteEmbedder` 即可，接口完全一样。」 |
| 6 | **基准只能证明算法行为，不能证明检索质量** | 「对，`VERIFICATION.md` 里有专门的『诚实边界』节：样本只有 20 条记忆、1 条查询、top-5；而且『哪些是冗余、哪些是过时』是我**人工标注**的，标签是这张表的真值来源。所以我让脚本把 dup 两两余弦和各模式下的名次一并打印，供读者自行核对。」 |
| 7 | **价格表 `MODEL_PRICES` 是硬编码的** | 「是，而且这是我在这个项目里唯一一处"疑似硬编码业务常量"。我明确标了『示例价格、会变、未命中一律返回 `None`』——**我宁可没有数字，也不要编一个数字**。如果要我重做，我会把它改成必须由调用方显式传入，库里一个默认值都不给。」 |
| 8 | **（已关闭）测试里曾经有一个失败（§12 清单未完成）** | `[v3 已关闭]` 这条**曾经**成立：守门测试 `FrozenLayoutTests.test_all_frozen_test_files_exist` 故意保持红灯，因为 §12 冻结的 4 份测试文件（`test_e2e_code_assistant` / `test_examples_offline` / `test_examples_import` / `test_docs_coverage`）还没落地。那 4 份现已全部落地，守门测试转绿，**现在全量是 `Ran 1652 tests` / `OK`（0 失败）**。如果面试官说"我在你文档里看到过有失败"——照实说这是**已关闭的历史缺口**：先记下缺口、补齐文件、由断言验收，而不是删断言换个好看的绿色。**当前仍然成立的边界**见 `VERIFICATION.md` §3 U-1~U-10 与 §4 G-2 / G-4。 |
| 9 | **`AnthropicChatClient` 比 OpenAI 那个大好几倍** | 「因为 Anthropic 的报文差异大：system 要抽出来单独放、`tool_result` 要**合并进同一条 user 消息**、block 结构要展开、`stop_reason` 的 `max_tokens` 要映射成 `length`。DeepSeek 只花了 5 行，因为它 OpenAI 兼容。**这个对比本身就是"为什么 provider 适配层不能想当然"的例子。**」 |
| 10 | **「你这不就是重写 LangChain 吗？」** | 「不是。**我没有追它的抽象，而是把控制流显式写出来。** 一个具体后果是：`finish_reason` 这种东西会自然进入我的控制流——模型被 `max_tokens` 截断时我会注入"从断点继续"并限次续写；而在一个黑盒 `AgentExecutor` 里，它很容易就只被当成一个日志字段。**这就是我说的『把 LangChain 里真正需要的 20% 显式实现一遍』的具体含义。**」 |

---

## 7. 附录

### 7.1 关键配置默认值（被追问"参数怎么定的"时用）

| 配置 | 默认值 | 一句话理由 |
|---|---|---|
| `AgentConfig.max_steps` | 10 | 语义是"最多调用几次 LLM"，是用户能理解的成本上限 |
| `max_total_tokens` | 200000 | 步数管"几次"，token 管"多贵"，两个上限都要有 |
| `max_wall_clock_s` | `None` | 唯一能兜住"每轮都在进展但总量失控"的机制 |
| `repeat_action_threshold` | 2 | 相同 `canonical_key` 出现 2 次就 nudge |
| `max_concurrency` | 4 | 全局并发信号量 |
| `thread_pool_size` | 8 | **每个 loop 私有**的线程池大小 |
| `default_timeout_s` | 30.0 | 工具默认超时 |
| `max_retries` | 2 | 配合 `backoff_base_s=0.25` → 间隔 0.25 / 0.5 |
| `disable_tool_after_failures` | 3 | 连续 3 次 infrastructure 失败后熔断 |
| `max_result_chars` / `max_observation_chars` | 8000 | 生效值取 `min(config, tokenizer_chars_budget())` |
| `context_window_tokens` | `None` | `None` = 不反推（向后兼容）；非 None 才反推 buffer 预算 |
| `retrieve_limit` | 5 | 长期记忆 top-k |
| `recency_half_life_days` | 7.0 | 近因半衰期 |
| `w_sim / w_recency / w_importance` | 1.0 / 0.15 / 0.1 | **相似度占绝对主导**，时间只打破平局 |
| `mmr_lambda` | 0.7 | 0.7 偏相关性、0.3 惩罚冗余；1.0 退化为纯 top-k |
| `HashingEmbedder.dim` | 256 | 纯 stdlib 哈希空间与碰撞的权衡值 |
| `buffer_max_tokens` / `buffer_max_messages` | 3000 / 50 | 滑动窗口**双约束** |
| `summary trigger_ratio / min_evict_batch` | 0.8 / 4 | 压缩触发是 **OR** 关系 |
| `subagent_concurrency` / `max_depth` / `max_rounds` | TeamConfig | 委派并发 / 深度 / 预算三道闸 |

### 7.2 数字出处（面试官问"这个数你怎么来的"）

| 数字 | 出处 |
|---|---|
| 24,765 行 / 41 个模块 | `find liteagent -name '*.py' \| xargs wc -l \| tail -1`（实测） |
| `Ran 1652 tests` / `OK` | `python3 -m unittest discover -s tests -t .`（实测；耗时随机器负载波动 —— 本机两次重跑 29.7s / 31.2s，别写死具体秒数） |
| 3.68x / 0.79x / 测不出差异 | `benchmarks/bench_dataclass_vs_pydantic.py`，结果在 `VERIFICATION.md` |
| 反义 0.7153 vs 同义 0.6239 | `benchmarks/bench_embedding_similarity.py` |
| 冗余 3→1、覆盖 3→5、0.985→0.405 | `benchmarks/bench_retrieval_ranking.py` |
| semaphore 跨 loop 崩溃 / WeakKeyDictionary 泄漏 / contextvar 三条路径 | **本机现场复现**（最小复现代码见故事 1~3） |

### 7.3 「一句话版本」备选（面试官赶时间时用）

* **要一句话讲清价值**：「**内核零第三方依赖 + 1652 个离线测试**，
  不是"我写了很多测试"，而是"**没人能反驳我的测试真的跑过**"。」
* **要一句话讲清技术含量**：「我把 agent 里所有**隐式的东西显式化**了——
  执行上下文、事件发射归属、重试白名单、降级痕迹，全都有一张冻结的表。」
* **要一句话讲清成长**：「这个项目教我的最大一件事是
  **测试通过不等于代码正确**——我的 `sequential_tools` 串行测试全绿，
  但它掩盖了一个"跨线程完全不串行"的真实缺陷。」

### 7.4 开场 30 秒自检

* [ ] 我能说出「**1652** / 半分钟 / 0 联网 / 0 key」这四个数？
* [ ] 我能**现场解释**「`asyncio.Semaphore` 为什么第二次调用才崩」？
* [ ] 我能说出「先信号量后锁」的**死锁机制**（不是只背结论）？
* [ ] 我能说出**至少一个**我诚实标了"没验证"的地方？
* [ ] 我准备好了**主动**说出一个**真实存在**的未验证边界（§6 第 8 条 / VERIFICATION §3 的 U-1..U-10），而不是背一个已经消失的失败？

---

*本文档只负责「怎么讲」。接口细节以 `docs/INTERFACES.md` 为准，
决策理由以 `docs/DESIGN_DECISIONS.md` 为准，
实测数据以 `docs/VERIFICATION.md` 与 `docs/BUILD_LOG.md` 为准。*
