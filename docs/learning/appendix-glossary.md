# 附录 · 术语表与速查

> 读文档时遇到不认识的词，回这里查。术语按主题分组，每条给出**本项目里的精确定义**
> （不是泛泛的行业解释——同一个词在不同框架里含义可能不同）。

---

## A. Agent 基础概念

| 术语 | 本项目里的定义 |
| --- | --- |
| **LLM（大语言模型）** | 一个"输入文本、输出文本"的函数。没有记忆、不能执行动作、知识冻结。 |
| **Agent** | 让 LLM 能"感知 → 思考 → 行动 → 再感知"的一整套工程结构。 |
| **Harness（挽具）** | Agent 框架的另一种叫法。强调"它本身没有智能，只是把模型的文本输出转换成现实动作"。 |
| **工具（Tool）** | 模型可以要求调用的外部能力。模型只表达"我想调什么"，真正执行的是框架。 |
| **ReAct** | Reason + Act。Thought（想）→ Action（做）→ Observation（看结果）的循环。 |
| **Thought / Action / Observation** | 文本 ReAct 模式里三段式输出的名字。Observation 是**框架**回灌的真实结果，不是模型想的。 |
| **Function Calling** | 模型厂商提供的原生能力：模型直接返回结构化的 `tool_calls` 字段，而不是自然语言。 |
| **JSON Schema** | 描述"参数长什么样"的标准格式。本项目里它**双向使用**：给模型看 + 执行前校验。 |
| **上下文窗口** | 一次请求能带上的最大 token 数。超出会被截断或报错。 |
| **token** | 模型处理文本的最小单位。中文大约 1 个字 ≈ 1 token，英文约 4 字符 ≈ 1 token。 |
| **幻觉** | 模型自信地编造不存在的事实。本项目通过"工具返回值是唯一事实来源"来抑制它。 |

---

## B. 本项目的数据结构

| 名称 | 位置 | 含义 |
| --- | --- | --- |
| **`Message`** | `llm/message.py` | 一条消息。字段：`role` / `content` / `name` / `tool_calls` / `tool_call_id` / `metadata`。**注意：没有 `kind` 字段**，`kind` 只是 `metadata` 里的一个键。 |
| **`Role`** | `llm/message.py` | 枚举：`SYSTEM` / `USER` / `ASSISTANT` / `TOOL`。 |
| **`ToolCall`** | `types.py` | 模型要求调用一个工具：`id` / `name` / `arguments` / `raw_arguments` / `metadata`。`canonical_key()` 用于去重。 |
| **`ToolResult`** | `types.py` | 工具执行结果。**永不抛异常**，失败也返回对象（`ok=False`）。 |
| **`LLMResponse`** | `types.py` | 模型的一次响应：`content` / `tool_calls` / `finish_reason` / `usage` / `latency_ms`。 |
| **`TokenUsage`** | `types.py` | token 用量：`prompt_tokens` / `completion_tokens` / `total_tokens`。 |
| **`AgentState`** | `agent/state.py` | 一次 run 的完整状态：`messages` / `step` / `usage` / `action_counts` / `observation_digests` / `status` / `error` 等。 |
| **`AgentStatus`** | `agent/state.py` | 7 个取值：`IDLE` / `THINKING` / `ACTING` / `OBSERVING` / `FINISHED` / `FAILED` / `ABORTED`。 |
| **`AgentResult`** | `agent/state.py` | `arun` 的唯一产物。字段顺序冻结：`output` 第一、`status` 第二。 |
| **`TraceEvent`** | `agent/callbacks.py` | 一条事件：`type` / `run_id` / `agent_name` / `step` / `timestamp` / `data`。 |
| **`MemoryItem`** | `memory/base.py` | 一条记忆：`id` / `content` / `role` / `importance` / `embedding` / `created_at` / `access_count` / `score` / `score_breakdown` 等。 |
| **`ToolSpec`** | `tools/base.py` | 工具的元数据。**`frozen=True`**（不可变）。 |

---

## C. 记忆层术语（这四个词最容易混）

| 术语 | 精确定义 |
| --- | --- |
| **buffer** | 跨轮累积的**完整**历史（只增不减）。`BufferMemory._messages`。 |
| **window** | 裁剪后**实际送进模型**的那批消息。`BufferMemory.window()`。 |
| **evicted** | 在 buffer 里但不在 window 里、**等待摘要**的消息。 |
| **transcript** | 一次 run 内的全部消息。`AgentState.messages`。**注意：它不含用户原始输入**。 |
| **observation** | 工具结果回灌给模型的文本载荷。`ToolResult.to_observation()`。 |
| **summary** | 被裁掉的历史压缩成的一段话。`SummaryMemory`。 |
| **长期记忆** | 跨 run / 跨会话的事实库。`VectorMemory`。 |
| **`should_auto_write`** | 自动写入长期记忆的判定。三种策略：`manual` / `turn` / `selective`。 |
| **混合打分** | `score = w_sim×sim + w_recency×recency + w_importance×importance`。 |
| **MMR** | Maximal Marginal Relevance。去冗余：既相关又彼此不同。 |
| **哈希 embedding** | 默认 embedder。**词面**相似，不是语义相似。"你好"和"hi"相似度为 0。 |

---

## D. 工具与执行

| 术语 | 含义 |
| --- | --- |
| **`@tool`** | 装饰器。把带类型注解和 docstring 的函数变成 `Tool` 对象，自动生成 JSON Schema。 |
| **`auto_register`** | `@tool(auto_register=True)` 时才把工具注册进**全局**注册表。默认 `False`。 |
| **`ToolRegistry`** | 工具清单的载体。负责注册、重名检查、别名、导出 schema。 |
| **`ToolExecutor`** | 真正执行工具的人。负责校验、审批、并发、超时、重试、熔断、截断。 |
| **`required`** | JSON Schema 里的必填参数数组。"没有默认值 且 不是 Optional 且 没有 Param(default=)"。 |
| **`retryable`** | 异常的一个类属性。**只有 4 个异常为 True**（限流/超时/连接/工具超时）。 |
| **幂等（idempotent）** | 重复执行不产生额外副作用。`idempotent=False` 的工具默认不重试。 |
| **孤儿线程（orphan thread）** | 同步工具超时后，worker 线程还在跑（Python 无法中断同步代码）。`metadata["orphan_thread"]=True`。 |
| **协作式取消** | 长耗时工具主动检查取消标志并自己退出。见 `current_cancel_flag()`。 |
| **fail-closed** | 审批策略没配置时**拒绝**执行。"没有审批人"不等于"审批通过"。 |
| **熔断** | 同一工具连续失败 N 次后不再执行。默认 N=3。 |
| **`NO_TIMEOUT`** | 哨兵值 `-1.0`，表示显式禁用超时。 |
| **`sequential_tools`** | 必须串行执行的工具名集合。**默认为空**，需要用户显式配置。 |
| **`PathSandbox`** | 路径沙箱。防 `..` / 绝对路径 / symlink 逃逸。`root=None` 直接报错。 |

---

## E. 并发与异步

| 术语 | 含义 |
| --- | --- |
| **R-LOOP 陷阱** | `asyncio.Semaphore` 等在争用时绑定事件循环，第二次 `asyncio.run` 会崩。 |
| **`LoopBoundPool`** | 本项目按"事件循环"懒创建 asyncio 原语的容器。解决 R-LOOP。 |
| **`run_sync`** | 同步包装异步的工具函数。**收工厂函数**（lambda），不收协程对象。 |
| **`contextvars`** | Python 的上下文变量。**读能穿透线程，写不能回传**（本项目实测结论）。 |
| **`copy_context()`** | 丢线程池前必须显式复制上下文，否则 worker 里读不到取消标志。 |
| **overrides 白名单** | `arun(**overrides)` 只接受 7 个键；传非法键**抛 `ConfigError`**。 |

---

## F. 多 Agent

| 术语 | 含义 |
| --- | --- |
| **Sequential** | 顺序流水线 A → B → C。`input_template` 支持 `{input}` / `{prev}` / `{steps[x]}`。 |
| **Hierarchical** | 主 Agent 分解任务 + 委派给子 Agent。 |
| **委派即工具** | 关键设计：子 Agent 被包装成 `delegate_to_<name>` **工具**，编排复用 ReAct 循环。 |
| **`extra_context`** | 委派时附带的背景文本。**绝不参与环检测/深度判定**（这是安全设计）。 |
| **环检测** | 委派栈里已有该 Agent 就拒绝。**返回字符串，不抛异常**。 |
| **`Blackboard`** | 共享黑板。支持版本冲突检测（`if_version`）、TTL、订阅。 |
| **`awatch`** | 黑板的异步监听。**合并语义**（多次写入可能只 yield 最新一条）。 |
| **`DelegationContext`** | 委派上下文：`stack` / `depth` / `budget` / `run_id`。 |
| **`arun_plan`** | 按 `depends_on` 拓扑分层的计划执行。层内可并发，层间串行。 |

---

## G. 可观测性与测试

| 术语 | 含义 |
| --- | --- |
| **事件（Event）** | 循环每一步发出的结构化消息，共 27 种。**每条事件有唯一发射者**。 |
| **`Callback` / `CallbackManager`** | 事件订阅机制。**回调抛异常不影响主流程**。 |
| **`TraceRecorder`** | 收集事件 + 写 JSONL 文件的上下文管理器。 |
| **`trace_stats`** | 从事件列表算出报表：步数、工具调用、延迟分位数、token、费用。 |
| **`as_llm_callback`** | 把 LLM 层的低层事件适配成 `TraceEvent` 的唯一适配器。**忘了接它 trace 里就没有 LLM 事件**。 |
| **`ScriptedLLM`** | 脚本化假模型。离线确定性测试的核心。 |
| **`ScriptedResponse`** | 一条剧本。构造器：`.text()` / `.tool()` / `.tool_raw()` / `.tools()` / `.react()` / `.error()`。 |
| **`FakeTransport`** | 假 HTTP 传输层。用来测 429 重试、错误映射。 |
| **`RecordingSleep`** | 记录但不真睡的 sleep 替身。用于断言退避序列。 |
| **`frozen_time`** | 冻结 `utc_now` 的上下文管理器。让时间戳断言可复现。 |
| **变异测试** | 手动改坏实现代码，看测试是否变红。用来检验"测试本身有效吗"。 |

---

## H. 设计原则（本项目的"宪法"）

这几条贯穿全项目，理解它们就能预测代码的行为：

| 原则 | 具体表现 |
| --- | --- |
| **默认拒绝** | 重试白名单制、审批 fail-closed、未知 `@tool` 参数直接报错。 |
| **静默失败最危险** | 降级必须留痕；`to_thread` 吞协程被写成红线；异步工具走同步入口主动抛错。 |
| **严格区分"谁说的"** | `role != "user"` 不写长期记忆；框架消息（nudge）即使 role=user 也拒绝写入。 |
| **单一写入点** | assistant 消息只在一处写入 transcript，并被测试断言锁住。 |
| **单一发射者** | 每条事件只能由一处发出，否则"数次数"的断言失效。 |
| **唯一时钟/随机源** | 时间走 `config.utc_now`，随机走实例级 `random.Random`。 |
| **把昂贵计算推迟** | `BufferMemory.add()` 是 O(1)，裁剪推迟到 `window()`。 |
| **大声失败优于静默错误** | `OpenAICompatibleClient` 缺 base_url 直接报错；`PathSandbox` 不接受 `root=None`。 |
| **观测代码不能搞挂业务代码** | 回调抛异常只记录，不影响主流程。 |
| **降级不丢信息** | 截断保留头尾（结论常在尾部）；子 Agent 压缩但原文写黑板。 |
| **不擅自改别人的对象** | `Agent` 不给外部传入的 `llm` 挂事件转发，逼你显式接线。 |

---

## I. 一句话速查卡

面试前 30 秒过一遍：

```
41 个模块 / 2.48 万行 / 零第三方依赖 / 1652 个测试约 22 秒跑完
LLM 层：4 个 provider，DeepSeek 适配器只有 2 行（协议兼容）
工具层：@tool 反射生成 JSON Schema，同一份 schema 兼做执行前校验
记忆层：窗口 + 摘要 + 向量；检索是 相似度 + 近因 + 重要度，再用 MMR 去冗
循环：  Thought(模型) → Action(框架) → Observation(框架)，五层防护
多 Agent：委派即工具；环检测返回字符串不抛异常
安全：默认拒绝 / fail-closed 审批 / 路径沙箱 / 闭包注入
测试：ScriptedLLM 让全链路离线确定性可测；用变异测试检验测试本身
```

---

## J. 中英对照

读英文资料时会遇到的说法：

| 中文 | 英文 |
| --- | --- |
| 工具调用 | tool call / function calling |
| 可观测性 | observability |
| 追踪 | tracing |
| 幂等 | idempotent |
| 退避 | backoff |
| 抖动 | jitter |
| 熔断 | circuit breaker |
| 沙箱 | sandbox |
| 人机协同审批 | human-in-the-loop (HITL) |
| 上下文窗口 | context window |
| 摘要压缩 | summarization |
| 向量检索 | vector retrieval / semantic search |
| 冗余去除 | deduplication / MMR |
| 委派 | delegation |
| 编排 | orchestration |
| 系统提示词 | system prompt |
| 温度 | temperature |
| 流式 | streaming |

---

**回到**：[学习手册首页](README.md) ｜ [练习与自测](10-practice.md)
