# 第 9 章 · 项目导览与阅读顺序

> **本章目标**：给你一张地图。41 个文件、2.4 万行代码，从哪读起、每个文件干什么、
> 调试时该看哪里。

---

## 9.1 全局结构

```
liteagent/
├── __init__.py          184 行   公共 API 出口（附录 B 的 __all__ 在这里）
├── __main__.py            6 行   python -m liteagent 的入口
├── errors.py           1031 行   全部异常类
├── types.py             705 行   核心数据结构（Message / ToolCall / ToolResult / LLMResponse）
├── config.py           1429 行   配置 + 跨层纯函数 + LoopBoundPool + 所有常量默认值
├── cli.py              1203 行   命令行接口
│
├── llm/                3065 行   【第 1 层】LLM 抽象层
│   ├── message.py       348      Role / Message 及序列化
│   ├── base.py          359      LLMClient 接口 + BaseLLMClient
│   ├── providers.py     889      四个 provider 适配器 + EchoLLM
│   ├── transport.py     582      HTTP 抽象（urllib/requests/httpx）+ 错误映射
│   ├── registry.py      198      get_llm("provider:model@base_url")
│   └── scripted.py      659      ScriptedLLM（离线测试核心）
│
├── tools/              6878 行   【第 2 层】工具系统
│   ├── schema.py       1534      ★ type hints → JSON Schema 反射（最技术密集）
│   ├── base.py          584      @tool 装饰器 / ToolSpec / cancel_scope
│   ├── registry.py      462      ToolRegistry
│   ├── executor.py     1349      ★ 并发/超时/重试/审批/熔断（最难的模块）
│   └── builtin/        3033      13 个内置工具
│       ├── files.py     588      路径沙箱 + 文件工具
│       ├── shell.py     295      受控 shell
│       ├── code.py      963      python_eval(AST 白名单) / python_exec / run_tests
│       ├── web.py       765      网页搜索 + 抓取 + HTML 转文本
│       ├── memory_tools.py 161   remember / recall
│       └── __init__.py  261      register_all
│
├── memory/             3536 行   【第 3 层】记忆管理
│   ├── base.py          499      MemoryItem / MemoryStore / Tokenizer
│   ├── embeddings.py    548      5 种 embedder + 余弦相似度
│   ├── buffer.py        389      短期滑动窗口 + 工具对修复
│   ├── summary.py       303      摘要压缩 + 抽取式兜底
│   ├── vector.py        795      ★ 长期向量库（混合打分 + MMR）
│   └── manager.py       946      三层编排 + prompt 组装
│
├── agent/              3886 行   【核心】ReAct 循环
│   ├── state.py         577      AgentState / AgentStatus / AgentResult
│   ├── callbacks.py    1200      事件定义 + 回调 + trace 分析
│   ├── parser.py        815      ★ 文本 ReAct 容错解析
│   └── agent.py        1249      ★★ ReAct 状态机（全项目心脏）
│
└── multiagent/         2671 行   【上层】多 Agent 协作
    ├── base.py          419      MultiAgent ABC / TeamConfig / 输出压缩
    ├── blackboard.py    708      共享黑板（版本/TTL/订阅/异步监听）
    ├── sequential.py    360      Sequential 流水线
    └── hierarchical.py 1155      Hierarchical：委派即工具 + 环检测 + 计划执行
```

标 ★ 的是"技术密度最高、最值得读"的文件。

---

## 9.2 依赖方向（只能向下）

```
          cli.py                         ← 什么都用
             │
        multiagent/                      ← 用 agent
             │
          agent/                         ← 用下面全部
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
   memory/  tools/  llm/               ← 三层互相独立
      └──────┼──────┘
             ▼
    config.py  types.py  errors.py      ← 谁都能用，它们不依赖上面任何东西
```

**这条规则是强制的**：有一个守门测试用 AST 解析所有模块的顶层 import，
断言依赖方向没被破坏。**反向 import 会让构建直接失败。**

> **为什么要这么严格？** 因为 Agent 框架最容易演变成"什么都互相 import"的泥球。
> 用自动检查守住依赖方向，是让 41 个模块还能被单独理解的**唯一办法**。

---

## 9.3 按目标的阅读路线

### 路线 A：只想学会用（约 1 小时）

```
examples/01_quickstart.py        跑一遍
examples/02_tools_custom.py      跑一遍，看 schema 怎么生成的
examples/07_code_assistant.py    跑一遍，看完整链路
docs/learning/02-first-agent.md  本章前文
```

**不用读源码。** 先把 API 用熟。

### 路线 B：想搞懂原理（约 1 天）

按这个顺序读，每个文件都**跑一遍它的测试**：

| 顺序 | 文件 | 带着什么问题读 |
| --- | --- | --- |
| 1 | `agent/agent.py` | 循环每一轮做了什么？（对照 5.2 的表） |
| 2 | `agent/parser.py` | 文本格式的容错有哪些？ |
| 3 | `tools/schema.py` | type hints 怎么变成 JSON Schema？ |
| 4 | `tools/executor.py` | 并发/超时/重试怎么实现的？ |
| 5 | `memory/vector.py` | 混合打分和 MMR 的公式在哪？ |
| 6 | `multiagent/hierarchical.py` | 委派工具怎么生成的？ |

**读代码的诀窍**：先读类/函数的 docstring 和签名，再看测试怎么用它，
最后才读实现细节。本项目的 docstring 写得比较详细（很多"为什么"都在注释里）。

### 路线 C：想改代码/面试（约 3 天）

在前面的基础上加：

```
docs/INTERFACES.md      冻结接口规范（5518 行，查签名用）
docs/DESIGN_DECISIONS.md 27 条设计决策（含"面试怎么讲"）
docs/INTERVIEW.md       26 个 Q&A + 7 个踩坑故事
docs/VERIFICATION.md    哪些结论真跑过、哪些只是代码写好了
docs/BUILD_LOG.md       项目是怎么搭出来的
```

### 路线 D：只想快速验证"它真的能跑"

```bash
cd /home/ml-user/workdir/project-3
python3 -m unittest discover -s tests -t .     # 1652 个测试，24 秒
python3 examples/run_all_examples.py           # 6 个示例，11 秒
```

---

## 9.4 调试指南：出问题看哪里

| 症状 | 先看这里 |
| --- | --- |
| Agent 一直循环不停 | `state.action_counts`（重复）、`state.observation_digests`（无进展）、`state.step` |
| 模型说调了工具但没效果 | `result.tool_results` 里每条的 `ok` / `error_type` / `metadata` |
| 参数校验失败 | `ToolResult.content`（错误信息里有 JSON Path）；再去看工具的 `parameters` schema |
| 记忆好像没生效 | `manager.stats()`、`manager.last_retrieved`；检查 `role` 是不是 `user` |
| 检索结果不相关 | `item.score_breakdown`（看是 sim / recency / importance 哪一项在起作用） |
| 多 Agent 委派被拒 | trace 里的 `agent_delegate` 事件（`refused_reason` 会说明是环/深度/预算/并发槽满） |
| trace 里没有 LLM 事件 | 忘了 `llm.on_event = as_llm_callback(agent.callbacks)` |
| 工具超时后状态奇怪 | `metadata["orphan_thread"]`（同步工具的线程还在跑） |
| 上下文被裁掉了 | `context_truncated` 事件（含 before/after/dropped） |

### 几个万能命令

```bash
# 看这个工具的 schema、warnings、参数表
python3 -m liteagent tools show read_file

# 导出成模型看到的格式
python3 -m liteagent tools schema read_file --format openai

# 用 trace 文件做统计
python3 -m liteagent trace run.jsonl --stats

# 离线交互式玩一玩
python3 -m liteagent chat --provider echo
```

---

## 9.5 关键数据结构速查

读代码时最常遇到的几个类型：

```python
# 一条消息
Message(
    role=Role.USER,                      # system / user / assistant / tool
    content="...",
    tool_calls=[...],                    # assistant 要求调用的工具
    tool_call_id="call_0",               # tool 消息对应哪个调用
    metadata={},                         # 含 kind="observation" / "nudge" 等标记
)

# 模型要求调用一个工具
ToolCall(id="call_0", name="read_file", arguments={"path": "a.py"}, raw_arguments='{"path": "a.py"}')

# 工具执行结果（永不抛异常，失败也返回对象）
ToolResult(
    call_id="call_0", name="read_file",
    content="文件内容...",                # ok=False 时是 "ERROR(类型): 消息"
    ok=True,
    error=None, error_type=None,
    duration_ms=1.5, attempts=1,
    metadata={},                          # truncated / orphan_thread / approved / ...
)

# 模型的一次响应
LLMResponse(
    content="...", tool_calls=[...],
    finish_reason="stop",                 # stop / tool_calls / length / content_filter / error
    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    model="...", raw=None, latency_ms=20.0,
)

# 一次 run 的完整结果
AgentResult(
    output="最终答案", status=AgentStatus.FINISHED, steps=2,
    tool_calls=[...], tool_results=[...], usage=TokenUsage(...),
    error=None, state=AgentState(...), duration_ms=9.3,
    agent_name="agent", metadata={"cost_usd": None, "finish_reason": "stop"},
)
```

---

## 9.6 三个"读代码时才发现的坑"

这些不属于知识体系，但能帮你少走弯路：

**① `Message` 没有 `kind` 字段**

`kind` 只是 `metadata["kind"]` 这个约定（取值 `"observation"` / `"nudge"` / `"react"` / `"summary"` / `"memories"`）。
想按 kind 过滤消息，得 `msg.metadata.get("kind")`。

**② `AgentConfig` 定义在 `config.py`，不在 `agent/` 里**

`agent/state.py` 只是 re-export 它。原因是配置类要跨层共享，放在最底层避免循环依赖。

**③ `ExecutorConfig` 也定义在 `config.py`**

同理。`tools/executor.py` 只是 re-export。

> 这体现了一个组织原则：**跨层共享的东西放在依赖图的最底层**，
> 哪怕它"逻辑上属于"某个上层模块。

---

## 9.7 本章小结

1. **41 个文件分成 6 层**：config/types/errors（底座）→ llm/tools/memory（三层）→ agent（核心）→ multiagent → cli。
2. **依赖方向只能向下，有 AST 守门测试强制**。
3. **最值得读的四个文件**：`agent/agent.py`、`tools/schema.py`、`tools/executor.py`、`memory/vector.py`。
4. **读代码的诀窍**：docstring → 测试 → 实现。
5. **调试先看状态和事件**，它们记录了"到底发生了什么"。

---

**下一章**：[第 10 章 · 练习与自测](10-practice.md) —— 检验你的掌握程度。
