# liteagent 架构说明（ARCHITECTURE.md）

> 配套文档：`INTERFACES.md`（冻结接口，唯一契约）、`DESIGN_DECISIONS.md`（为什么这么设计）、
> `TOOLS.md`（工具作者指南）、`INTERVIEW.md`（面试话术）、`VERIFICATION.md`（能力对账）。
>
> **本文档的定位**：读完之后你应该能在白板上画出数据流、说清每一层的职责边界、
> 以及**哪些能力是真的、哪些是词面的**。最后一点比前两点更重要。

---

## 1. 一句话与三十秒版本

**一句话**：liteagent 是一个**零第三方依赖**的轻量 Agent Harness，
把「LLM 抽象 / 工具系统 / 记忆管理」三层与一个 ReAct 状态机组合起来，离线可测、可解释、可扩展。

**三十秒版本**：
- **三层架构**：LLM 层统一多 provider（原生 function calling 与文本 ReAct 二选一，归约到同一份
  状态机）；工具层用装饰器反射生成 JSON Schema、带重试/超时/并发/审批；记忆层是
  「滑动窗口 + 摘要压缩 + 向量检索」的三层协作。
- **一次 run**：`Agent.arun` 跑一个显式的状态机，每轮「组装 prompt → 调 LLM → 决定行动/终结 →
  重复检测 → 执行工具（可并发）→ 回灌观察 → 压缩检查」。
- **关键取舍**：内核零三方依赖（`dataclass` 而非 pydantic、`argparse` 而非 typer、
  哈希特征而非 faiss）；同步工具承认线程不可中断（`orphan_thread` 标记 + 协作式取消）；
  重试**默认拒绝**。

---

## 2. 整体架构图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  L6  cli.py            liteagent run / chat / tools / trace / schema / multi  │
│                        main(argv) -> int   （不 sys.exit，便于单测）           │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  L5  multiagent/                                                             │
│      ├── base.py          MultiAgent(ABC) / TeamConfig / DelegationContext   │
│      ├── blackboard.py    Blackboard（threading.RLock + awatch）             │
│      ├── sequential.py    SequentialAgent    A → B → C                       │
│      └── hierarchical.py  HierarchicalAgent  manager + delegate_to_<worker>  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  子 Agent 就是普通 Agent；上下文经 DelegationContext
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  L4  agent/             ★ ReAct 状态机（全框架唯一的控制流）                   │
│      ├── state.py      AgentState / AgentStatus / AgentResult                │
│      ├── callbacks.py  EventType / TraceEvent / CallbackManager / trace_stats │
│      ├── parser.py     ReActParser（文本语法）                                │
│      └── agent.py      Agent.arun  = THINK → ACT → OBSERVE → THINK ...        │
└───────┬───────────────────────┬────────────────────────┬─────────────────────┘
        │                       │                        │
┌───────▼─────────┐   ┌─────────▼──────────┐   ┌─────────▼──────────────────┐
│ L2 llm/         │   │ L3 tools/          │   │ L3 memory/                 │
│ LLMClient(ABC)  │   │ ToolSpec / Tool    │   │ MemoryStore(ABC, sync)     │
│ BaseLLMClient   │   │ ToolRegistry       │   │  ├─ BufferMemory  (窗口)   │
│  ├ OpenAI       │   │ ToolExecutor       │   │  ├─ SummaryMemory (摘要)   │
│  ├ Anthropic    │   │  ├ 校验/审批        │   │  └─ VectorMemory  (长期)   │
│  ├ DeepSeek     │   │  ├ 并发/超时/重试   │   │ Embedder(Hashing/Numpy/    │
│  ├ Echo         │   │  └ 取消/熔断        │   │          Callable/Remote)  │
│  └ Scripted ★   │   │ builtin/*.py       │   │ MemoryManager（唯一入口）  │
│ Transport(ABC)  │   │  files/shell/code  │   │ Tokenizer(中英分离启发式)  │
│  urllib/req/htx │   │  web/memory_tools  │   │                            │
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

### 2.1 分层规则

- **依赖只允许向下**（L 编号小的可被 L 编号大的 import）。**包内自由**，包间按表。
- 同层边只有 §1.1 的 `E1..E12` 十二条白名单（每一条都在 `test_zero_dependency.py` 里被断言）。
- **`errors.py` 是 L0**：任何层都能 import 而不引入依赖。
- **`config.py` 是"跨层纯函数"的家**：`RetryPolicy`、`LoopBoundPool`、`run_sync`、`utc_now`、
  `to_jsonable`、`render_template`。之所以不新开文件，是为了不让文件清单漂移
  （41 个文件是封闭清单，红线 1 守着它）。

### 2.2 三条贯穿全框架的运行时约束

| 约束 | 内容 | 违反的后果 |
|---|---|---|
| **R-LOOP** | 禁止在 `__init__`/模块级创建 `asyncio.Semaphore/Lock/Condition/Event`；必须经 `LoopBoundPool` 在运行中的 loop 内懒创建 | `Agent.run()` 第二次调用直接 `RuntimeError: bound to a different event loop`（实测） |
| **`mutex = threading`** | 跨线程/跨 loop 生效的限流与互斥一律用 `threading` 原语 | delegate 工具跑在 worker 线程的**新 loop** 里，loop-bound 原语完全不受约束（并发失控） |
| **取消优先** | 任何宽 `except` 的首行保证 `CancelledError` 不被吞；超时只认 `except asyncio.TimeoutError` | 取消被吞成一次工具失败；`except TimeoutError` 抓不到 `asyncio.TimeoutError`（3.10 实测二者不同） |

---

## 3. 一次完整的 ReAct 轮次（时序图）

下面的时序是 `Agent.arun` 在一个 step 内的真实调用顺序。
**注意事件是由谁发的**：`LLM_*` 归 LLM 层、`TOOL_*` 归 executor、其余归 Agent
（冻结矩阵见 `INTERFACES.md` §2.7）。

```text
 Agent            MemoryManager      LLMClient        ToolExecutor     Callbacks(观察者)
   │                   │                │                 │                  │
   │ run 开始                                                               │
   │──── aadd(Message.user(input)) ───▶│                 │                  │
   │    (用户输入只入 buffer 一次)       │                 │                  │
   │                   │                │                 │     RUN_STARTED ─┤
   │                   │                │                 │                  │
   │ ╔═════════════════ 每轮 (while state.step < max_steps) ══════════════╗  │
   │ ║ 预算/墙钟检查（max_wall_clock_s → RunTimeoutError）                 ║  │
   │ ║                 │                │                 │  STEP_STARTED ─┤  │
   │ ║ 1. abuild_prompt(system, user_input, now)          │                  │
   │ ║──── abuild_prompt ──────────────▶│                 │                  │
   │ ║    ├ 段1 system                 │                 │                  │
   │ ║    ├ 段2 <conversation_summary> │                 │                  │
   │ ║    ├ 段3 <relevant_memories>  ◀── VectorMemory.search(now=...)        │
   │ ║    ├ 段4 buffer.window()        │                 │                  │
   │ ║    ├ 段5 extra                  │                 │                  │
   │ ║    └ 段6 user_input（仅首轮）    │                 │                  │
   │ ║       → _enforce_context_budget() → 超限则裁剪 + CONTEXT_TRUNCATED ──┤  │
   │ ║                 │       MEMORY_RETRIEVE{count=len(last_retrieved)} ──┤  │
   │ ║                 │                │                 │                  │
   │ ║ 2. achat(messages, tools=…)      │                 │                  │
   │ ║─────────────────────────────────▶│                 │  LLM_REQUEST ───┤  │
   │ ║                 │                ├ _with_retry（429/超时/连接 → 退避）  │
   │ ║                 │                │  sleep_fn(delay)│                  │
   │ ║◀──── LLMResponse ────────────────┤                 │  LLM_RESPONSE ──┤  │
   │ ║    state.usage += resp.usage ; state.add_message(resp.to_message())   │  │
   │ ║──── aadd(resp.to_message()) ────▶│                 │                  │
   │ ║    预算检查（max_total_tokens → BUDGET_EXCEEDED）                     │  │
   │ ║                 │                │                 │                  │
   │ ║ 3. 决定行动 or 终结                                                  │  │
   │ ║    native: calls = resp.tool_calls                                   │  │
   │ ║            ├ finish_reason=length  → 截断续写（注入 continue 提示）    │  │
   │ ║            ├ finish_reason=content_filter → 立即 FAILED               │  │
   │ ║            ├ finish_reason=tool_calls 但 calls 空 → 自纠正            │  │
   │ ║            └ 否则 → 终结分支（§9.4.3）                                │  │
   │ ║    text  : parser.parse(resp.content)                                │  │
   │ ║            ├ ReActParseError → 自纠正（NUDGE，消耗 step，D-10）        │  │
   │ ║            ├ is_action() → 构造 ToolCall（Action 优先！）  ACTION_PARSED
   │ ║            └ is_final()  → 终结分支                                  │  │
   │ ║                 │                │                 │                  │  │
   │ ║ 4. 重复/无进展检测（三层：canonical_key / observation digest / 熔断） │  │
   │ ║    → 命中 → NUDGE 或 FAILED(RepeatedActionError) →  REPEAT_DETECTED ──┤  │
   │ ║                 │                │                 │                  │  │
   │ ║ 5. 执行工具（并发开关）                                              │  │
   │ ║   ├ execute_many(calls)  ───────▶│                 │                  │  │
   │ ║   │   gather(return_exceptions=True)                │                  │  │
   │ ║   │   每项：semaphore("exec") → seq_lock →          │                  │  │
   │ ║   │        ┌ 审批（requires_approval）    TOOL_APPROVAL ───────────────┤  │
   │ ║   │        ├ 熔断检查（连续失败 ≥ N）                 │                  │  │
   │ ║   │        ├ 校验（validate_instance）                │                  │  │
   │ ║   │        ├ cancel_scope() ──▶ run_in_executor(tp, tool.run)          │  │
   │ ║   │        │   （同步工具在 worker 线程；异步直接 await）                 │  │
   │ ║   │        │   超时 → ToolTimeoutError + orphan_thread=True             │  │
   │ ║   │        └ 失败且 retryable → TOOL_RETRY + sleep_fn(delay)          │  │
   │ ║   │   归一化：非 ToolResult 一律包成 failure（永不外抛）                 │  │
   │ ║   └ [ToolResult...] （长度与顺序严格对齐 calls）    TOOL_FINISHED/ERROR ┤  │
   │ ║                 │                │                 │                  │  │
   │ ║ 6. 回灌观察                                                          │  │
   │ ║    native → Message(role=tool, tool_call_id)                         │  │
   │ ║    text   → Message(role=USER, kind="observation", "Observation: …")  │  │
   │ ║    state.add_message + aadd；trim_transcript(max_transcript_messages) │  │
   │ ║    include_thought_in_history → Message(assistant, kind="react")      │  │
   │ ║                 │                │                 │                  │  │
   │ ║ 7. acompress_if_needed()（**每轮一次**，避免 N 个工具触发 N 次摘要）  │  │
   │ ║                 │       MEMORY_COMPRESS ──────────────────────────────┤  │
   │ ║                 │                │                 │  STEP_FINISHED ─┤  │
   │ ╚═══════════════════════════════════════════════════════════════════════╝  │
   │                                                                          │
   │ 终结分支：strip_markers → aadd(assistant) → mark_finished(FINISHED)         │
   │                                   RUN_FINISHED ──────────────────────────┤
   │ step 用尽：MaxStepsExceededError  RUN_FAILED ────────────────────────────┤
   │ 被取消   ：CancelledError 原样上抛 + state.mark_finished(ABORTED)          │
```

### 3.1 两条模式如何共用同一个状态机

```text
                     ┌──────────────────────────────────────┐
   native 模式 ──────▶│  resp.tool_calls                     │
                     │  Message(role=tool)  回灌             │──┐
                     │  一轮 N 个动作（可并发）               │  │
                     └──────────────────────────────────────┘  │
                                                               ├─▶ list[ToolCall]
                     ┌──────────────────────────────────────┐  │       │
   text 模式  ──────▶│  ReActParser.parse()（Action 优先）    │──┘       ▼
                     │  Message(role=user, Observation:)     │   ToolExecutor.execute_many
                     │  一轮 1 个动作                        │          │
                     └──────────────────────────────────────┘          ▼
                                                             list[ToolResult]
      差异被隔离在 _next_calls() 与 _write_back() 两个私有方法里       │
      ────────────────────────────────────────────────────────────┘
      共用：max_steps / 重复检测 / 自纠正 / 截断 / usage / 事件 / trace
```

---

## 4. 多 Agent 两种模式的数据流

### 4.1 Sequential：流水线

```text
        input
          │
   ┌──────▼───────┐   stage_input = render_template(template, {input, prev, steps})
   │  step[0]     │   template 默认 "input"（首步）/ "prev"（其余）
   │  planner     │   child_state = AgentState.create(...)  （每步新建，不复用父 state）
   └──────┬───────┘   state.scratchpad["delegation"] = context.child(name)
          │ output  │   blackboard.write(name, output, tags=("stage",))
          │         │   share_memory=True 时注入同一个 MemoryManager
   ┌──────▼───────┐
   │  step[1]     │   propagate_failure ∈ {return, raise, continue}
   │  coder       │   · return   → 立即停止：output = result.output or prev or ""
   └──────┬───────┘   · raise    → DelegationError(from_agent, to_agent)
          │         │   · continue → "[stage X failed: ...]" 继续
   ┌──────▼───────┐   · optional=True → 无视 propagate_failure，一律 continue
   │  step[2]     │
   │  reviewer    │
   └──────┬───────┘
          ▼
    AgentResult(output=末步输出, usage=Σ各阶段, metadata={"steps":[...], "blackboard":{...}})
```

### 4.2 Hierarchical：manager + 委派工具

```text
                        ┌────────────────────────────────────────────┐
   用户任务 ────────────▶│  HierarchicalAgent.arun(input, plan=None)   │
                        │  ctx = DelegationContext(stack=[self])      │
                        │  _CURRENT_CTX.set(ctx)   ← contextvar        │
                        │  manager.tools.merge(delegate_tools())      │
                        └───────────────┬────────────────────────────┘
                                        ▼
                        ┌────────────────────────────────────────────┐
                        │  manager（一个普通 Agent，跑 ReAct）         │
                        │  它"看到"的工具里多了 delegate_to_coder 等    │
                        └───────────────┬────────────────────────────┘
                                        │ 模型发出 N 个 delegate_to_*（可并发）
                                        ▼
                        ┌────────────────────────────────────────────┐
                        │  execute_many → 每个 delegate 在**线程**里   │
                        │  1) ctx = _CURRENT_CTX.get()（唯一来源）     │
                        │  2) ctx.exhausted()? → refused               │
                        │     would_cycle?/depth? → refused 字符串      │
                        │     threading.Semaphore(subagent_concurrency) │
                        │     或 _serial_lock（parallel_subagents=False）│
                        │  3) run_sync(worker.arun(task))  ← 新 loop    │
                        │  4) compress_subagent_output（head70+tail30）│
                        │  5) blackboard.write(subagent:<name>:<seq>)   │
                        │  6) 失败也返回字符串（前缀 status=FAILED）     │
                        └───────────────┬────────────────────────────┘
                                        ▼
                        manager 继续 ReAct（把压缩后的字符串当观察结果）
                                        │
                                        ▼
                              AgentResult(metadata={"delegations": [...]})
```

### 4.3 `arun_plan`：把显式分解接到执行上

```text
   Plan(goal, subtasks=[t1, t2, t3(depends_on t1,t2)])
          │
          ▼  _layers(plan)  拓扑分层（环 → CycleDetectedError；未知 id → DelegationError）
   ┌──────────────┐
   │  层 1: t1,t2 │  parallel_subagents=True → asyncio.gather + per-loop Semaphore
   │              │  False                   → for 循环串行
   └──────┬───────┘  每个完成后 blackboard.write("subtask:t1")
          │          回填 SubTask.status / SubTask.result
   ┌──────▼───────┐  assignee 为空或不在 workers → 回退给 manager（记 WARNING）
   │  层 2: t3    │  单个 subtask 失败**不中断**整体（压缩成字符串写进 result）
   └──────┬───────┘
          ▼
   AgentResult(metadata={"plan": plan.to_dict(), "steps": [...]})
```

**为什么两种模式的并发机制不同**（这是最容易踩的坑，也是面试的好素材）：

| | 并发原语 | 理由 |
|---|---|---|
| delegate 工具路径 | `threading.Semaphore` | delegate 是**同步工具**，被 `execute_many` 丢进**不同线程**，每个线程里 `run_sync` 都**新建一个 loop**。loop-bound 的 asyncio 信号量在每个线程里都是全新的、计数永远是满的 —— 并发完全不受限 |
| `arun_plan` 路径 | per-loop `asyncio.Semaphore` | `arun_plan` 本身跑在父 loop 里，worker 的 `arun` 是 async 的，直接 `gather` 即可，没有跨线程问题 |

**同理**：`sequential_tools` 的互斥必须用 `threading.Lock`（跨线程/跨 loop 都要串行），
而不是 `asyncio.Lock`。

---

## 5. 记忆层的数据流

```text
  写侧                                                读侧
  ────                                                ────
  Message ──▶ MemoryManager.aadd
                  │
       ┌──────────┴───────────┐
       ▼                      ▼
  BufferMemory.add       should_auto_write?
  （全量累积，不裁剪）      ├ role != "user" → False（恒）
       │                  ├ manual   → False
       │                  ├ turn     → True
       │                  └ selective→ 长度 ≥ 40 或命中 marker
       │                        │
       │                        ▼
       │                  VectorMemory.upsert
       │                   ├ 最大余弦 ≥ dedup_threshold → 更新已有（created_at 刷新、
       │                   │                          importance 取 max、access_count 保留）
       │                   └ 否则新建；超 max_items → FIFO 淘汰
       │
       │  window()（裁剪）              abuild_prompt
       │   ├ 取前导 SYSTEM 为 pinned      ├ 段1 system
       │   ├ 从尾部累加至 max_messages    ├ 段2 <conversation_summary>
       │   │   或 token 预算（keep_last_n）├ 段3 <relevant_memories>
       │   ├ drop_orphan_tool_messages    │      ▲
       │   └ _repair_tool_pairs           │      └ VectorMemory.search(query, now=)
       │                                  │          score = w_sim*sim
       ▼                                  │                + w_recency*2^(-age/HL)
  drain_evicted() ──▶ should_compress?    │                + w_importance*importance
       │   ├ tokens ≥ 0.8*max             │          排序 (-score,-created_at,id)
       │   └ pending ≥ min_evict_batch    │          取 3*limit 候选 → MMR 重排
       ▼                                  │          → access_count += 1（**有副作用**）
  SummaryMemory.acompress                 ├ 段4 buffer.window()
   ├ LLM 摘要（失败 → 抽取式兜底）         ├ 段5 extra
   └ truncate_head_tail(max_summary_chars) └ 段6 user_input（仅首轮）
                                              │
                                              ▼
                                    _enforce_context_budget()
                                    （超限 → 裁长期记忆块 → 裁最旧消息
                                      + CONTEXT_TRUNCATED 事件）
```

---

## 6. 与 LangChain / Nanobot 的对比

> 这一节是**面试第一问**的答案。完整对比表（8 条轴 + 结论）在 `DESIGN_DECISIONS.md` 的 D-13，
> 这里给架构层面的结论。

### 6.1 为什么不用 LangChain

| 轴 | LangChain | liteagent | 差异的真实代价/收益 |
|---|---|---|---|
| **依赖体积** | 数十个传递依赖（`langchain-core` + provider 包 + `pydantic` + `aiohttp`…） | **0 个**（三方库全部是可选加速器，探测式降级） | 收益：**零安装**（`pyproject.toml` 的 `dependencies` 是空列表，克隆即可跑）、离线可测；代价：要自己写 HTTP、embedding、token 估算、CLI。**注意**：LangChain 侧的"传递依赖数量"是描述而非实测——本环境无外网，`pip install langchain-core` 拉多少包**未验证** |
| **事件模型** | `CallbackHandler` 的 `on_llm_start(**kwargs)` 一族，**kwargs 魔法、字段随版本漂移 | `EventType` 强类型枚举 + `TraceEvent` 全量字段 + `trace_stats()` 冻结字段表 | 收益：trace 可精确定义断言、可 JSONL、可 `jq`；代价：新增事件要改枚举（我们认为是好事） |
| **工具 schema** | pydantic-only（`args_schema: BaseModel`） | 反射 `inspect.signature` + `Annotated` + **docstring 解析**，无 pydantic 也完整可用 | 收益：写工具不用学 pydantic；代价：schema 反射有降级路径（我们把它**显式记账**在 `ToolSpec.warnings` 里） |
| **prompt 组装** | LCEL `Runnable` 表达式树 | 一个**显式的六段函数** `abuild_prompt` | 收益：顺序与内容可读、可测；代价：没有声明式组合（不需要） |
| **控制流可见性** | `AgentExecutor` 内部黑盒 + 若干 `<...>` 提示词魔术 | `Agent.arun` 是**一个读得完的显式状态机**（`arun` 本体 583 行，`ast` 实测） | 收益：能回答"第 N 轮发生了什么"；这是 agent 框架最该透明的地方 |
| **可测性** | `FakeListLLM` 只能回文本，**无法断言 tools 是否真的暴露给模型** | `ScriptedLLM` 记录 `ScriptedCall(messages, tools, kwargs, response)`，可断言消息构成、工具暴露、call_id、重试次数 | 收益：能测"模型看到的到底是哪几条消息"；代价：多写 659 行确定性替身（`liteagent/llm/scripted.py`，`wc -l` 实测，值） |
| **离线可跑** | 需要真 key 才能跑绝大多数路径 | 无 key 无网跑**全部**测试；`--provider echo` 演示 | 收益：CI 秒级；面试能当场跑 |
| **多 Agent** | LangGraph 图 DSL（另一套心智模型与状态机） | 两种内置模式（Sequential / Hierarchical）+ 黑板，**复用同一个 Agent 状态机** | 收益：没有第二套执行引擎；代价：没有任意图拓扑（当前不需要） |

**结论**：liteagent 不是"LangChain 的替代品"，而是**把 LangChain 里我真正需要的 20% 显式地实现一遍**。
LangChain 帮你省掉的每一行代码，都要用"依赖体积 + 黑盒控制流 + 不可断言的事件"来换；
在这个项目里我选择把这三样换回来。**更重要的收益是**：
这些取舍每一条都能写进文档并附上实测数据（见 `VERIFICATION.md`），
而不是背一段"最佳实践"。

### 6.2 借鉴 Nanobot / 其他轻量 Agent 的三个设计模式

| 模式 | 来源 | 我们的落地 |
|---|---|---|
| **工具即 schema**：工具的自描述能力决定了模型能不能用对它 | Nanobot 的 tool registry | `@tool` 从 type hints + docstring 反射出 JSON Schema，并用 `summary_line()` 给文本 ReAct 生成紧凑的工具清单 |
| **显式循环而非隐式链**：ReAct 的 Thought/Action/Observation 是一个**可中断、可观测**的循环 | ReAct 论文 / Nanobot | `Agent.arun` 的 while + 每轮 5 个可观测事件；`max_steps` 是硬上限 |
| **分层记忆**（短期窗口 + 长期检索） | 多数轻量实现 | 三层：`BufferMemory`（token 预算裁剪）+ `SummaryMemory`（滚动摘要）+ `VectorMemory`（混合打分 + MMR） |

### 6.3 我们**没有**抄的东西（也是诚实的边界）

- **没有** LCEL / 表达式树 / 声明式链。"组合"用普通 Python 函数与 `MultiAgent` 子类完成。
- **没有**自动的工具选择 / plan-and-execute 论文实现。分解是**显式**的：
  要么让 manager 在 ReAct 里自己选 `delegate_to_*`，要么用 `adecompose()` + `arun_plan()`。
- **没有** ANN 索引 / 语义 embedding / rerank 模型（见下节）。
- **没有**任意图拓扑的多 Agent 编排（Sequential 与 Hierarchical 覆盖了绝大多数教学与原型场景）。

---

## 7. 能力 vs 非能力（诚实清单）

> **把假的能力说成真的，才是这类项目最大的技术债。** 这张表是 `INTERVIEW.md` 里
> "已知局限"那一节的来源。

| 能力 | 状态 | 精确边界 |
|---|---|---|
| LLM 多 provider 统一 | ✅ 真实 | 4 个适配器：OpenAI / Anthropic / **DeepSeek**（5 行子类）/ Echo；**未联网实测过**（环境无外网），HTTP 报文形态靠单元测试与 fixture 保证 |
| 原生 function calling 与文本 ReAct 统一 | ✅ 真实 | 归约到 `list[ToolCall]`，共用一份状态机（D-02） |
| 工具 JSON Schema 反射 | ✅ 真实 | 支持 `Literal`/`Enum`/`Optional`/嵌套 `dataclass`/`Annotated(Param)`/pydantic 鸭子类型；**不支持** `Union[A,B]`（降级 `{}` + warning）、`prefixItems`、`**kwargs`（抛 `ToolDefinitionError`） |
| 工具并发调用 | ✅ 真实 | `gather(return_exceptions=True)` + 全局信号量 + 顺序严格对齐 |
| 超时 | ✅ 真实（但**有边界**） | 异步工具可被真正取消；**同步工具的线程不可中断**，超时后线程仍在跑，用 `metadata["orphan_thread"]=True` **如实标记**，并提供 `current_cancel_flag()` 协作式取消 |
| 重试 | ✅ 真实 | **default-deny**：只有 4 类异常重试；同步工具超时**不**重试（避免叠加孤儿线程）；非幂等工具强制单次 |
| 人工审批（HITL） | ✅ 真实 | `requires_approval` + `approval_policy`，**fail-closed**（无 policy = 拒绝） |
| 短期对话历史 | ✅ 真实 | token 预算 + 条数双约束裁剪，工具对完整性修复 |
| 摘要压缩 | ✅ 真实 | LLM 摘要 + 抽取式兜底（永不抛） |
| 长期「向量存储」 | ⚠️ **词面相似，不是语义相似** | `HashingEmbedder` 是 **feature hashing**（词面），**不是** embedding 模型。生产替换路径：注入 `RemoteEmbedder`，接口不变 |
| 检索规模 | ⚠️ **O(n) 线性扫描** | `DEFAULT_MAX_MEMORY_ITEMS=10000` 时每次 search 是近万次点积。**不能**说"向量数据库 / 可扩展检索"；没有 ANN 索引（faiss 不可用） |
| 持久化 | ⚠️ 真实，但**测试覆盖只到同进程** | `VectorMemory.save/load`（JSONL）+ `MemoryManager.persist/restore`。JSONL 文件格式本身与进程无关，但 `tests/test_memory_persistence.py` 里**没有 `subprocess` 级往返测试**——"另一进程能 load 并复现检索顺序"是**推断**，不是实测（见 `VERIFICATION.md` §3 U-9 / §4 G-2） |
| 多 Agent 协作 | ✅ 真实 | Sequential 流水线、Hierarchical 委派、显式 Plan 分层执行（`arun_plan`） |
| Python 代码执行 | ⚠️ **不是安全沙箱** | `python_eval` 是"受限表达式求值"（AST 白名单 + 静态复杂度闸），**不是**安全边界；`python_exec` 是任意代码执行，`dangerous=True` + `requires_approval=True`，只在受控环境使用 |
| 真实网络验证 | ❌ 未做 | 环境**无外网**。所有联网路径用 `FakeTransport` + 真实响应形态的 fixture 覆盖，`VERIFICATION.md` 里标 `code-only-not-run` |

---

## 8. 代码量分布（`[v3 变更]` 已按实测值更新，用于判断"哪里是重点"）

> `[v3 变更]` v2 这一节是**实现初期的预估**，与落地后的仓库严重不符（且自身不自洽：
> 各行"文件数"相加 = 39，**合计**却写 41；总行数写 ~14400，实际 24765）。
> 现按 `find liteagent -name '*.py'` + `wc -l` 的实测值重写。逐行数字取自
> `liteagent/**.py`（含各包 `__init__.py`），行数四舍五入到十位。

| 层 | 文件数 | 实测行数 | 说明 |
|---|---|---|---|
| `errors.py` + `types.py` + `config.py` | 3 | ~3170 | 地基。`config.py` 最大（跨层纯函数都在这） |
| `llm/` | 7 | ~3070 | 三个真实 provider 的报文编解码是主要体量 |
| `tools/` | 11 | ~7050 | `schema.py`（反射）与 `executor.py`（并发/超时/重试/审批）是两座大山 |
| `memory/` | 7 | ~3540 | `vector.py`（打分/MMR/持久化）与 `embeddings.py`（哈希实现）是重点 |
| `agent/` | 5 | ~3890 | `agent.py` 是唯一的控制流，必须逐行可读 |
| `multiagent/` | 5 | ~2670 | `hierarchical.py` 最大（delegate + plan） |
| `cli.py` + `__main__.py` + `__init__.py` | 3 | ~1390 | argparse 样板 |
| **合计** | **41** | **24,765** | 外加 `tests/`（37 个 `test_*.py`，共 39 个 `.py`；1652 个用例）与 `docs/`（7 个文件） |

---

*本文档描述的是**设计意图与数据流**；任何与 `INTERFACES.md` 冲突的表述以 `INTERFACES.md` 为准。*
