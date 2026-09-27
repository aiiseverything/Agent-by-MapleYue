# liteagent 关键设计决策（DESIGN_DECISIONS.md）

> **状态：FROZEN v2.0** —— 与 `INTERFACES.md`（FROZEN v2.0）配套。
> `INTERFACES.md` 冻结的是"怎么写的接口"，本文件记录的是"为什么这么设计"，
> 以及**每条决策在面试里怎么讲**。
>
> 每条决策统一按同一个模板写：**背景 / 备选方案 / 选择 / 理由 / 代价 / 面试怎么讲 / 实测补充**。
> 标注 `★面试亮点` 的条目是面试时最值得主动讲、且**都有实测证据**的决策。
> 「实测补充」留空待填的，由对应模块的实现者在完成该模块后填入，
> 并在 `docs/VERIFICATION.md` 里链接（漏掉它们，BUILD_LOG 的故事就不闭环）。
>
> 阅读顺序建议：**D-01 → D-03 → D-12 → D-13 → D-02 → D-07 → D-09 → D-04 → D-08**。

---

## 决策索引

| # | 决策 | 一句话 | 面试亮点 |
|---|---|---|---|
| D-01 | 核心结构用 stdlib `dataclass` 而非 pydantic | 零依赖 + 可变累积语义 + trace 字段全量输出 | ★ |
| D-02 | 文本 ReAct 与原生功能调用归约到 `list[ToolCall]` | 一份状态机，差异只有 5 个点 | ★ |
| D-03 | `@tool` 默认**不**自动注册 | import 副作用会毁掉测试确定性与多 Agent 隔离 | ★ |
| D-04 | Embedding 用纯 stdlib 的 hashing trick，并写清能力边界 | 词面相似 ≠ 语义相似，**写在文档里** | ★ |
| D-05 | `Optional[X]` 只表达"非必填"，不生成 null 联合 | 牺牲 schema 理论完整性换跨 provider 兼容 | |
| D-06 | Token 估算用中英分离的启发式 | `len/4` 对中文低估 4 倍 | ★ |
| D-07 | 重试是 default-deny | 一个被重复执行的副作用比一次失败严重得多 | ★ |
| D-08 | 混合打分 + MMR 去冗 | 三联稳定排序，测试才不 flaky | ★ |
| D-09 | 同步工具超时：承认线程不可中断 | `orphan_thread` 标记 + 协作式取消 | ★ |
| D-10 | Parse error 自纠正**消耗** `max_steps` | 否则是成本放大器 | |
| D-11 | CLI 用 stdlib `argparse` 不用 typer | `main(argv) -> int` 让 CLI 可被单测 | |
| D-12 | MemoryStore 用同步 API，Blackboard 用 `threading.RLock` | 实测 `asyncio.Semaphore` 跨 loop 会崩 | ★ |
| **D-13** | **为什么不用 LangChain** | 8 条对比轴，每条都有取舍理由 | ★ |
| **D-14** | **`finish_reason` 参与控制流** | 截断/内容过滤/空 tool_calls 各有分支 | ★ |
| **D-15** | **上下文窗口反推 token 预算** | 3000 是我们算出来的，不是拍出来的 | ★ |
| **D-16** | **循环防护三层** | 参数微变/无进展/熔断/墙钟 | ★ |
| **D-17** | **HITL 审批 fail-closed** | 无 policy = 拒绝，不是放行 | ★ |
| **D-18** | **token 与成本预算** | `estimated_cost_usd` 不再是一个永远返回 None 的 API | ★ |
| **D-19** | **`Annotated` 的探测方式（3.10 实测坑）** | `isinstance(x, Annotated)` 恒 False | ★ |
| **D-20** | **每 loop 私有线程池，放弃 `set_default_executor`** | 探测不到 + 状态放错位置 = 静默失效 | ★ |
| **D-21** | **跨 loop 限流一律用 threading 原语** | delegate 路径每个线程都是新 loop | ★ |
| **D-22** | **`execute_many` 永不抛（`fail_fast` 的修正）** | 与 `execute` 的契约必须一致 | |
| **D-23** | **把"长期存储"做成真的（save/load）** | 简历说"存储"，就必须能跨进程 | ★ |
| **D-24** | **错误分类的第二层** | recoverable（模型能改）vs infrastructure（模型改不了） | ★ |
| **D-25** | **双 API 与 `run_sync` 的工厂签名** | 收协程对象会打 `never awaited` 警告 | |
| **D-26** | **时间与随机性的注入点** | 没有它们，退避与检索的断言不可写 | |
| **D-27** | **事件发射归属矩阵** | "谁发这条事件"必须有唯一出处 | ★ |

---

## D-01 核心数据层用 stdlib `dataclass`，而不是 pydantic ★面试亮点

**背景**
环境中 `pydantic 2.13.5` 是可用的。项目要求"LLM 抽象层、工具系统、记忆管理"三层架构，
其中 `Message`、`ToolCall`、`ToolResult`、`AgentState` 这些结构体是**热路径**：
一次 ReAct 循环里会被构造/拷贝几十次，还会被序列化进 trace。

**备选方案**
1. 全部用 pydantic `BaseModel`：免费的类型校验、嵌套校验、`model_json_schema()`。
2. 全部用 `dataclass`：零依赖、快、`dataclasses.replace/asdict` 够用。
3. 混合：对外 API 用 pydantic（用户手写参数的那一侧），内核用 dataclass。

**选择**
方案 2。核心数据结构全部 `@dataclass`（叶子结构加 `slots=True`）。
pydantic **只**作为一个可选的、运行时探测的适配器出现在 `tools/schema.py` 里，
用于识别用户传入的 `pydantic.Field` 元数据以及 `BaseModel` 类型的参数注解。

**理由**
1. **零依赖红线优先**。整个 `liteagent/` 的顶层 import 只能是 stdlib；如果核心结构是 pydantic，
   那"零依赖"就是假的，`test_zero_dependency.py` 也守不住。
2. **语义更贴合**。agent 状态是**可变累积**的（每轮 append 消息、累加 usage、改 status），
   pydantic v2 默认 `validate_assignment=False` 且鼓励不可变风格，用它反而要频繁 `model_copy(update=...)`。
3. **trace 需要字段全量输出**。pydantic 的 `model_dump(exclude_none=True)` 是常见默认，
   会让 trace 结构随内容变化，测试无法做精确字典比对。dataclass + 手写 `to_dict()` 更可控。
   （v2 里这条被强化为 §2.2 的"字段全量输出，除三个显式 opt-out"规则。）
4. **性能**。pydantic v2 的校验开销在单对象上虽小，但 `AgentState.messages` 每次组装 prompt
   都要触碰，实测在纯 python 侧 dataclass 明显更轻（见「实测补充」）。

**代价**
- 失去运行期类型校验。缓解：`from_dict()` 手工校验并抛 `SerializationError`；
  工具参数校验由**独立的一层**（`tools/schema.validate_instance`）承担，
  这一层本来就必须手写（要校验的是 JSON，不是 python 对象）。
- 需要自己写 `to_dict`/`from_dict`（约 200 行样板）。这是有意的：显式 > 隐式。
- `slots=True` 与动态属性挂载互斥（`INTERFACES.md` §4.2 已冻结"不要在 `ToolCall`/`ToolResult`
  上临时挂属性"）。

**面试怎么讲**
「我用 dataclass 而不是 pydantic 做核心结构，因为项目硬约束是内核零第三方依赖，
而且 agent 状态是可变累积型对象，pydantic 的不可变风格反而增加拷贝。
pydantic 我只当作**可选适配器**用：在工具 schema 生成时做鸭子类型识别，
用户用 `Annotated[int, Field(description=...)]` 时能拿到描述和约束，
但框架不 import 它——环境里没装 pydantic 也能跑全部功能。」

**实测补充**
- （待填）`AgentState.to_dict()` 在 1000 条消息下的耗时；pydantic 对照组的耗时。
  落点：`benchmarks/bench_dataclass_vs_pydantic.py`。

---

## D-02 文本 ReAct 与原生 Function Calling 的统一：归约到 `list[ToolCall]` ★面试亮点

**背景**
模型侧有两条完全不同的交互形态：
- 支持 function calling 的模型返回结构化 `tool_calls` 数组，一轮可以有多个，
  工具结果以 `role="tool"` + `tool_call_id` 回灌。
- 不支持（或用户用纯 prompt 驱动）的模型只能靠文本约定：
  `Thought: ... / Action: ... / Action Input: {...}`，一轮一个动作，
  结果以 `Observation:` 文本回灌。

如果为这两种形态各写一套循环，会有两个 `max_steps` 实现、两套事件、两套错误处理、
两套重复检测——这是同类项目最常见的腐化点。

**备选方案**
1. 两套独立循环，各自演进（简单，但代码翻倍、行为漂移）。
2. 只支持原生 function calling，文本模式用"把工具描述塞进 prompt 再解析"的**策略对象**替换，
   但循环仍写两遍。
3. 定义一个中间表示：两条路径最后都归约到 `list[ToolCall]`，之后的执行/回灌/统计**完全共用**。

**选择**
方案 3。差异被压缩到 5 个点，且全部隔离在 `_next_calls()`（产出 ToolCall）与
`_write_back()`（写回观察）两个私有方法里：

| 差异点 | native | text |
|---|---|---|
| 工具如何暴露 | `tools=[...]` 请求参数 | `{tools}` 渲染进 system prompt |
| 模型输出如何变成 ToolCall | 直接读 `resp.tool_calls` | `ReActParser.parse()` 解析文本 |
| 一轮几个动作 | N 个（天然并发） | 1 个 |
| 结果如何回灌 | `role="tool"` 消息 | `role="user"` 的 `Observation:` 消息 |
| 何时终结 | `finish_reason=="stop"` 且无 tool_calls | 出现 `Final Answer:` |

`mode="auto"` 的判定（唯一出处是 `LLMClient.resolve_mode`）：
`llm.supports_tool_calling and len(tools) > 0` → native，否则 text。

**理由**
1. **状态机只有一份**：`max_steps`、重复动作检测、parse error 自纠正、结果截断、
   usage 累加、事件序列全部复用。测试只需要一套断言骨架。
2. **`Observation` 用 `role="user"`** 不是偷懒：文本 ReAct 没有 tool 角色的协议位，
   而所有 chat API 都接受 user 消息。这是一个必须写在文档里的兼容性取舍。
3. **可扩展**：加 JSON-mode（模型输出 `{"action": ...}`）只需在 `_next_calls` 里加一个分支。

**代价**
- `Agent` 里多了一层"模式"的心智负担，`AgentConfig.mode` 有三个取值。
- 文本模式一轮只能一个动作，无法利用并发。文档必须明确说明。
- 两种模式的 trace 事件序列略有差异（text 模式额外有 `action_parsed`），测试要分别写。

**面试怎么讲**
「我没有为文本 ReAct 和原生 function calling 写两套循环。我把两条路径都归约成一个中间表示
——`list[ToolCall]`，之后的执行、并发、重试、结果回灌、token 统计、事件发射全部共用同一份状态机。
差异只有 5 个点，我把它隔离在 `_next_calls` 和 `_write_back` 两个方法里。
这样加第三种模式（比如 JSON-mode）只需要改这两个方法。我最满意的细节是：
文本模式下 Observation 必须以 `role="user"` 回灌，因为没有 tool 角色的协议位——
这是一个实测出来的兼容性约束。
**v2 里我把这个统一又推进了一步**：`finish_reason` 也进了控制流（D-14），
所以"模型被 max_tokens 截断"这种原生模式特有的失败，文本模式也能通过同一条分支被正确处理。」

**实测补充**
- （待填）两种模式在同一个任务上的 token 消耗对比。

---

## D-03 `@tool` 装饰器默认**不**自动注册（但保留 `auto_register=True`）★面试亮点

**背景**
简历上写的是"工具层基于装饰器实现自动注册与 JSON Schema 生成"。最直觉的实现是
`@tool` 一执行就把工具塞进一个模块级全局注册表——import 即注册。

**备选方案**
1. `@tool` 直接写全局注册表（最符合"自动注册"的字面意思）。
2. `@tool` 只生成 `Tool` 对象，注册必须显式 `registry.register(tool)`。
3. 折中：默认不注册，但提供 `@tool(auto_register=True)` 与 `get_default_registry()`。

**选择**
方案 3。默认 `auto_register=False`；`liteagent.tools.builtin` 里的内置工具一律显式注册，
由 `register_all(registry, ...)` 统一完成。
`[v2 变更]` 语义写死：`auto_register=True` 等价于
`get_default_registry().register(t, override=False)`，**重名抛 `ToolDefinitionError`**
（v1 没说重名怎么办，两个实现者会写出相反行为）；
`reset_default_registry()` 冻结为"替换为一个全新的空注册表"。

**理由**
1. **import 副作用会毁掉测试确定性**。测试之间会互相污染，而工具名冲突要抛
   `ToolDefinitionError`——结果是"测试顺序决定成败"。
2. **注册表应该是被注入的**，不是被 import 的。`Agent(tools=registry)` 让"这个 Agent 能看见哪些工具"
   成为显式参数，这对多 Agent 场景是刚需。
3. **自动注册的能力仍然保留**（简历里那句话成立），只是它变成**显式 opt-in**。
4. 全局注册表在并发场景是隐患（多线程 import 竞争），显式注册顺带规避了。

**代价**
- 与"import 即注册"的直觉不符，文档要显眼地说明。
- 用户写单文件 demo 时多一行 `registry = ToolRegistry([my_tool])`。缓解：`examples/01` 第一屏就展示。
- 全局注册表仍需存在，所以"全局可变状态"没有完全消灭，只是从默认路径移开了。

**面试怎么讲**
「装饰器自动注册我做了，但**默认关掉**。因为 import 副作用会让注册表变成隐式全局状态：
测试之间互相污染、工具名冲突、多 Agent 场景下没法给不同子 Agent 不同工具集。
装饰器只负责**反射 + 生成 JSON Schema**（这才是它真正的价值），注册交给显式的
`registry.register()`，而 `Agent(tools=registry)` 让工具集变成显式参数。
如果用户想要 import 即注册，`@tool(auto_register=True)` 加默认注册表就行——
能力都在，只是不默认触发；重名我会直接报错而不是静默覆盖。」

**实测补充**
- （待填）`ToolSpec` 生成耗时（微秒级）、注册 100 个工具的内存占用。

---

## D-04 Embedding 用纯 stdlib 的 Hashing Trick，并把能力边界写进文档 ★面试亮点

**背景**
长期记忆需要向量检索。环境里 `faiss`、`sentence-transformers`、`openai` 都没装，
`numpy` 装了但不能作为核心依赖；而且**没有外网**，任何真实 embedding API 都调不通。

**备选方案**
1. 依赖 `numpy` + 随机投影，假装是语义 embedding。
2. 只提供接口，不给默认实现，让用户必须注入 embedder（那离线测试就跑不起来）。
3. 纯 stdlib 的**特征哈希（hashing trick）**：token 哈希到固定维度向量，
   带符号累加、亚线性 TF 加权、L2 归一化。同时提供 numpy 加速版和远程 API 版。

**选择**
方案 3。`HashingEmbedder(dim=256)` 是默认实现，纯 stdlib、确定性、跨进程可复现。

关键实现细节（这些细节才是这个决策的价值所在）：
- 用 `hashlib.blake2b(token, digest_size=8)`，**绝不用内置 `hash()`**：
  Python 的 str hash 受 `PYTHONHASHSEED` 随机化影响，同一份数据在不同进程会得到不同向量。
- 前 4 字节决定落到哪一维，第 5 字节的最低位决定符号（避免所有 token 同号导致方向偏置）。
- 权重 `1 + log1p(tf)`：抑制高频 token（廉价的 IDF 替代）。
- 中文按**单字 + 相邻双字 bigram** 切，因为中文没有空格，按字切会丢"北京"这种双字词的区分度。
- 提供 `NumpyHashingEmbedder`，并在测试里断言它与纯 python 版**数学等价**（1e-9 内）。

`[v2 新增]` **必须同时写进文档的两条限制**（`ARCHITECTURE.md` §7 的"能力 vs 非能力"表）：
1. 检索是 **O(n) 线性扫描**，`max_items=10000` 时每次 search 是近万次点积
   —— **不能**声称"向量数据库 / 可扩展检索"。
2. **没有 ANN 索引**（faiss 不可用），也**没有 rerank 模型**。

**理由**
1. **离线可测是硬需求**。没有默认 embedder，"记忆检索"这条链路就只能靠 mock，
   测不出真问题（排序、去重、MMR 都要真实向量才有意义）。
2. **确定性**。同一段文本在任何机器、任何进程、任何 `PYTHONHASHSEED` 下得到同一向量。
3. **诚实**。它提供的是**词面相似**，不是语义相似。这一点必须写进文档，
   不能说成"语义检索"。生产环境注入 `RemoteEmbedder` 即可，接口完全一致。

**代价**
- 检索质量明显弱于真实 embedding：同义不同词（"报错" vs "exception"）检索不到。
  缓解：混合打分里带 `importance` 与近因项；提供 `remember` 工具显式写入；文档给"生产替换路径"。
- 256 维的哈希空间会有碰撞。默认 256 是权衡值。
- 新增约 150 行需要仔细测试的数学代码（符号、归一化、零向量边界）。

**面试怎么讲**
「长期记忆我没有依赖 faiss 或 tiktoken，而是用纯标准库实现了 feature hashing：
token 哈希到 256 维、带符号累加、`1+log1p(tf)` 加权、L2 归一化。
关键细节是哈希必须用 `hashlib.blake2b` 而不是内置 `hash()`——
内置 hash 受 `PYTHONHASHSEED` 随机化，向量在不同进程不一致，确定性测试就废了。
中文我按单字加相邻双字 bigram 切。我在文档里明确写了能力边界：
这是**词面相似**不是语义相似，而且是 **O(n) 线性扫描、没有 ANN 索引**，
生产环境应该注入远程 embedding，接口是一样的——
把一个假的能力说成真的，才是这类项目最大的技术债。」

**实测补充**
- （待填）同义句/反义句的相似度实测值。落点：`benchmarks/bench_embedding_similarity.py`。

---

## D-05 `Optional[X]` 只表达"非必填"，不生成 `"type": ["X", "null"]`

**背景**
`def search(query: str, limit: Optional[int] = None)` 这种签名太常见了。
从 JSON Schema 语义看，`Optional[int]` 对应 `{"type": ["integer", "null"]}`。

**备选方案**
1. 严格按 JSON Schema：`{"type": ["integer","null"]}`。
2. 用 `anyOf: [{"type":"integer"},{"type":"null"}]`。
3. 只生成 `{"type":"integer"}`，把 `Optional` 的信息**只**用在 `required` 判定上。

**选择**
方案 3。

**理由**
1. **联合类型在 function calling 生态里是雷区**。OpenAI 的 strict mode 不接受 null 联合；
   很多兼容端（vLLM、Ollama 的 JSON-schema-to-grammar 转换、部分代理网关）遇到
   `"type": [...]` 数组会直接报错或静默降级。我们的目标是"一套 schema 能发给所有 provider"。
2. **模型不需要这个信息**。模型看的是 `required: ["query"]` ——「limit 可以不传」这个语义
   已经由"不在 required 里"完整表达。`null` 是 Python 内部概念。
3. **校验器行为要一致**：非必填字段收到显式 `null` 时直接放行（写进 `validate_instance`）。
   这样即便模型真传了 `null`，也不会被拒。

**代价**
- 生成的 schema 在"严格 JSON Schema 语义"上不完整：一个必须显式传 null 的参数无法表达。
  这是明确的**已知限制**，写进 `docs/TOOLS.md`。
- 用户如果非要 `null` 联合，可以传 `parameters=` 完全接管 schema（逃生口）。

**面试怎么讲**
「`Optional[X]` 我没有生成 `"type": ["X","null"]`。因为 function calling 生态对联合类型支持很差：
OpenAI 的 strict mode 不接受 null，很多兼容端的 schema-to-grammar 转换遇到 `type` 数组会直接崩。
我改成只用 `Optional` 决定"是否进 required"，模型看到 `required` 里没有它就是可以不传。
这是**牺牲 schema 的理论完整性，换取跨 provider 的兼容性**，而且我给了逃生口。」

**实测补充**
- （待填）把带 `type` 数组的 schema 发给几个主流 provider 的实际报错记录（作为决策证据）；
  本环境无外网，此项标 `code-only-not-run`。

---

## D-06 Token 估算用"中英分离的启发式"，而不是 `len/4` 或 tiktoken ★面试亮点

**背景**
短期记忆的滑动窗口按 **token 预算**裁剪，摘要压缩也有触发阈值，都需要一个"这段文本大概多少 token"。

**备选方案**
1. `ceil(len(text) / 4)`：最常见的经验公式。
2. 引入 tiktoken（违反零依赖，且模型换 tokenizer 就不准）。
3. 中英分离启发式：CJK 字符算 1 token，其余字符按 4 字符 1 token。

**选择**
方案 3。区间判定用硬编码的码点范围（不用 `unicodedata` 名字匹配，那在不同 Unicode 版本下不稳定）：
`0x4E00-0x9FFF`、`0x3400-0x4DBF`、`0x3000-0x303F`、`0xFF00-0xFFEF`、
`0xAC00-0xD7AF`、`0x3040-0x30FF`。CJK 之外按 `n_other / 4.0`。用 `functools.lru_cache(4096)` 缓存。

`[v2 变更]` 边界行为统一：**空文本 -> 0**；非空但估算结果 < 1 -> 返回 1。
`DEFAULT_TOKEN_CHAR_RATIO` **只被 `HeuristicTokenizer` 使用**
（`messages_tokens` 的 ASCII 近似是独立兜底，两者都遵守"空串 -> 0"）。

**理由**
1. **中文是被 `len/4` 严重低估的重灾区**。一句 40 字的中文，`len/4` 算 10 token，
   实际接近 40 —— **低估 4 倍**，直接导致窗口超预算、API 报 context length 错误。
2. **保守 > 精确**。启发式整体偏保守（宁可高估）。预算判定留 20% 余量。
3. **可注入**。`Tokenizer` 是抽象类，`CallableTokenizer` 让用户接任意实现，
   `TiktokenTokenizer` 在库可用时自动启用。
4. **估算函数必须唯一**。窗口裁剪、摘要触发、记忆截断三处共用同一个函数，否则阈值互相对不上。

**代价**
- ±20% 误差。极端情况（大量代码/JSON/URL）会高估，导致窗口比理论值小。
- 码点范围是硬编码的，Unicode 新增区块需要手工维护（注释说明维护点）。
- 比一个 `len/4` 多几十行代码和一次 `lru_cache` 查找。

**面试怎么讲**
「滑动窗口按 token 预算裁剪，token 数我是自己估的，而且是**中英分离**的：
CJK 字符算 1 token，其余按 4 字符 1 token。因为最常见的那句 `len/4` 对中文会低估 4 倍——
40 个汉字它算 10 个 token，实际接近 40，结果就是窗口超预算、直接撞 context length 错误。
计算用硬编码码点区间判定 CJK，不用 `unicodedata` 的名字匹配，因为那在不同 Unicode 版本
下结果会变。同时我把它做成了 `Tokenizer` 接口，tiktoken 可用时自动切换，
用户也能注入自己的实现——三处用到 token 数的地方共用同一个函数，阈值才不会互相打架。」

**实测补充**
- （待填）中文/英文/代码三类文本的误差表（对照 `len/4`）。

---

## D-07 重试策略是"默认拒绝"（default-deny），而不是"默认重试" ★面试亮点

**背景**
执行器和 LLM 客户端都需要重试。直觉做法是"捕获异常就重试 N 次"。

**备选方案**
1. 默认重试一切异常（除少数白名单）。
2. 默认不重试，只有显式标记 `retryable=True` 的异常才重试。

**选择**
方案 2。`LiteAgentError.retryable` 是**类属性**，默认为 `False`。可重试的只有四类：
`LLMRateLimitError`（429，尊重 `Retry-After`）、`LLMTimeoutError`、`LLMConnectionError`、
`ToolTimeoutError`；`ToolExecutionError` 仅当工具作者显式标 `retryable=True`。

另一条更硬的规则：`ToolSpec.idempotent=False` 的工具，在
`ExecutorConfig.allow_retry_on_non_idempotent=False`（默认）下**强制只尝试一次**。

`[v2 变更]` **同步工具的 `ToolTimeoutError` 不重试**（见 D-21 / `INTERFACES.md` §3.4）：
同步工具超时后 worker 线程仍在跑，重试会**再起一个线程**，
对 `write_file`/`run_shell` 意味着"两个线程同时写同一份资源"。异步工具仍允许重试。

**理由**
1. **盲目重试会重复产生副作用**。`write_file`、`run_shell`、`delete_file` 超时后重试，
   可能造成重复写入/重复执行；用户在 trace 里看到"重试 3 次"却不知道为什么文件被追加了三次。
   **在 agent 系统里，一个被重复执行的副作用比一次失败严重得多。**
2. **很多异常重试没有意义**：`ToolValidationError`、`ToolNotFoundError`、`LLMAuthError`、`ConfigError`。
3. **`retryable` 作为类属性**，让"哪些错可以重试"成为类型系统的一部分，
   而不是散落在各处的 `isinstance` 判断里。异常层次表本身就是重试策略表。
4. **退避参数可复现**：`rng_seed` 非空时用 `random.Random(seed)`，测试里再配 `jitter=0.0`，
   就能断言"重试了 2 次，每次间隔 0.25s / 0.5s"（v2 还加了 `sleep_fn` 注入，见 D-26）。

**代价**
- 工具作者必须显式声明 `retryable=True`（如网络类工具）。
  缓解：内置的 `web_search`/`fetch_url` 已声明；`TOOLS.md` 有清单。
- 用户可能在业务工具里抛 `ToolExecutionError` 却期待自动重试。缓解：错误消息里带上
  "this error is not retryable by default; set retryable=True if the operation is idempotent"。
- `idempotent` 要求作者思考语义，增加一点心智负担。

**面试怎么讲**
「重试我是**默认拒绝**的：`LiteAgentError.retryable` 默认 False，只有明确可重试的才标 True
——429、超时、连接错误。因为 agent 工具里有大量有副作用的操作，盲目重试一个超时的 `write_file`
会导致重复写入，而且用户在 trace 里根本看不出来。我加了一条更硬的规则：
工具声明 `idempotent=False` 时默认强制只尝试一次。
**还有一个更细的坑**：同步工具的"超时"其实杀不掉线程，重试会再起一个线程去写同一份资源，
所以我让同步工具的超时**不重试**，异步工具的才重试。
另外我把 `retryable` 做成异常的类属性，这样异常层次表同时就是重试策略表。」

**实测补充**
- （待填）429 场景下的重试行为回放（用 `FakeTransport` 造响应序列 + `RecordingSleep` 断言间隔）。

---

## D-08 长期记忆的检索打分 = 相似度 + 近因衰减 + 重要度，再上 MMR 去冗 ★面试亮点

**背景**
长期记忆存的是跨会话的事实。只按向量相似度取 top-k 有两个问题：
**过时信息**（3 个月前的偏好和昨天的决定分数一样高）与**冗余**（同一事实写入 5 次占满 top-5）。

**备选方案**
1. 纯余弦相似度 top-k。
2. 相似度 + 固定的时间过滤（只取最近 N 天）。
3. 加权混合打分 + MMR 重排。
4. Cross-Encoder 重排（需要模型，本环境不可行）。

**选择**
方案 3。冻结公式：

```text
score     = w_sim * sim + w_recency * recency + w_importance * importance
sim       = cosine(query_vec, item.embedding)            # 负值直接丢弃
recency   = 2 ** (-age_days / half_life_days)            # 半衰期 7 天
importance= clamp(item.importance, 0, 1)
默认权重   = w_sim 1.0, w_recency 0.15, w_importance 0.1
```

排序键是**三级稳定排序** `(-score, -created_at, id)`；排序后取 `max(limit*3, limit)` 作为候选池，
再做 MMR：`mmr_i = λ * score_i - (1-λ) * max_{j∈selected} sim(i,j)`，λ=0.7。

`[v2 变更]` `VectorMemory.__init__` 里**注入的 embedder 的维度是权威**
（`self.dim = embedder.dim` 并写回 config）—— 没有这条，测试注入 dim=4 的 `CallableEmbedder`
后每次 `add` 都会被 `MemoryStoreError` 打死，"混合打分公式手算对照"这条测试根本写不出来。

**理由**
1. **三个信号互补**：`sim` 解决"相关"但会被过时信息和冗余击败；`recency` 用连续衰减
   （不是硬时间窗——硬窗户会在边界产生"昨天还在、今天消失"的跳变，半衰期更好解释）；
   `importance` 提供**用户显式控制**的入口（`remember(content, importance=0.9)`）。
2. **权重故意让 sim 占绝对主导**（1.0 vs 0.15/0.1）。因为 `HashingEmbedder` 只得 0~0.5 的分值区间，
   如果 recency 权重大，排序会被时间主导——那是错的，时间只应该**打破接近分数的平局**。
3. **MMR 的 λ=0.7**：0.7 偏相关性、0.3 惩罚冗余；`λ=1.0` 退化成纯 top-k。
4. **排序必须完全确定**：加了 `created_at` 和 `id` 作为二三级键，否则同分条目顺序依赖 dict 迭代序，
   测试会随机失败。

**代价**
- 3 个超参 + 1 个 λ，调参空间变大。缓解：全部给默认值、可配、文档给出推导。
- `recency` 让检索结果随时间变化 → **给测试带来不确定性**。
  缓解：`search(now=...)` 注入；`[v2 变更]` **`MemoryManager.retrieve/aretrieve/abuild_prompt`
  也全部透传 `now`**（v1 只在最底层有这条缝，Manager 层把它切断了，导致断言 `<relevant_memories>`
  文本的测试会随运行时刻漂移）。
- `search()` 有副作用（更新 `access_count`/`last_access_at`），调用方必须知道它不是只读的；
  `[v2 变更]` 副作用与读操作在**同一把锁**内（D-12 的线程模型）。
- MMR 需要候选两两相似度，O(k²)；k ≤ 15 可忽略。

**面试怎么讲**
「长期记忆的检索我用了混合打分：`w_sim * 余弦 + w_recency * 半衰期衰减 + w_importance * 用户标注`。
纯相似度有两个典型失效：三个月的旧偏好和昨天的决定分数可能一样高；
同一个事实被写入五次时 top-5 全是它。所以加了近因衰减（半衰期 7 天，
用 `2^(-age/half_life)` 而不是硬时间窗，避免边界跳变），加了 importance 让用户能显式加权，
最后用 MMR 重排去冗余。权重上我刻意让相似度占绝对主导（1.0 对 0.15 和 0.1），
因为时间只应该用于**打破接近分数的平局**。
还有一个容易忽略的点：排序我加了 `created_at` 和 `id` 做二三级键，保证同分时顺序确定——
否则测试会因为 dict 迭代顺序随机而 flaky。
为了让这条链路真的可测，我把 `now` 一路透传到最上层，并且让注入的 embedder 的维度成为权威。」

**实测补充**
- （待填）20 条记忆（含 3 条冗余 + 2 条过时）的检索结果对照表：
  纯相似度 vs 混合打分 vs 混合+MMR。落点：`benchmarks/bench_retrieval_ranking.py`。

---

## D-09 同步工具的超时：承认线程不可中断，用 contextvar 做协作式取消 ★面试亮点

**背景**
工具可能是同步阻塞函数（`requests.get`、`subprocess.run`、读写大文件）。
在 asyncio 里跑它们必须用线程池。但 `asyncio.wait_for` **只能取消 await 层**——
worker 线程里的函数**继续跑到自己结束**。

**备选方案**
1. 用 `asyncio.wait_for` 包住，假装解决了（最常见，也是最危险的）。
2. 用进程替代线程（`multiprocessing`）：真能杀，但需要 pickle 函数与参数，
   闭包工具（如沙箱注入路径的文件工具）根本没法 pickle。
3. 承认不可中断，但提供**协作式取消**通道 + 明确的可观测痕迹。

**选择**
方案 3：
- `asyncio.wait_for` 仍然用（它保证**调用方**不再等待，语义上是"超时返回"）。
- 超时后把结果标记 `metadata["orphan_thread"] = True` 并记 WARNING：**绝不假装线程已经停了**。
- 通过 `contextvars.ContextVar` 把 `threading.Event` 传进工具线程；
  长耗时工具可以在循环里检查 `current_cancel_flag()` 主动退出。
- executor 用 `cancel_scope()` 上下文管理器负责 Event 的创建与 set。

`[v2 变更]` **`cancel_scope()` 必须每个 attempt 进一次**（详见 D-21）。
v1 允许把它放在重试循环外层，结果是：第一次尝试超时时 set 了这个 Event，
第二次尝试复制到的仍是**同一个已 set 的 Event**，工具一进循环就自杀 —— 重试全部秒失败。

**理由**
1. **诚实性是设计原则**：一个"看起来超时成功、实际线程还在跑"的框架，
   会在生产里造成难以复现的诡异状态（文件被写了一半、连接泄漏）。
2. **协作式取消对真实场景够用**：需要被取消的工具通常是**循环型**的（轮询、分块下载、逐行处理）。
   而对 `subprocess.run` 这类，正确做法本来就是传 `timeout=` 给它自己（内置工具正是这么做的）。
3. **为什么不用进程**：工具函数是**闭包**（`make_file_tools(sandbox)` 捕获沙箱路径、
   `make_memory_tools(memory)`、delegate 工具捕获子 Agent）。闭包不可 pickle，
   进程方案会直接废掉一半内置工具。
4. **为什么 contextvar 而不是传参**：工具函数签名是**给模型看的 JSON Schema**，
   加一个 `_cancel_flag` 参数会污染 schema、浪费 token、还会让模型试图去填它。
   `[v2 实测]` contextvar 的**读**能穿透 `to_thread`（worker 看到调用方的值），
   **写**不能回传 —— 所以 `flag.set()` 必须在调用方线程（即 `with cancel_scope()` 所在的 task）完成。

**代价**
- 真正卡死的同步工具**仍然无法被杀死**，最坏情况是线程泄漏 + 进程无法退出。
  缓解：executor 关闭时线程池用 `shutdown(wait=False, cancel_futures=True)`；文档明确写出边界。
- 用户**必须知道**要主动检查取消标志，框架无法强制。这是"约定"而非"机制"。
- contextvar 的传播链有一个坑（写错位置静默失效），已写进规范。

**面试怎么讲**
「同步工具在 asyncio 里的超时是个陷阱：`asyncio.wait_for` 只能取消 await 层，
worker 线程里的函数会继续跑到结束。我没有假装解决了——超时后我会在 `ToolResult.metadata` 里
标 `orphan_thread=True` 并记警告，**让"线程还在跑"这件事可见**。
同时我提供了协作式取消：用 `contextvars` 把一个 `threading.Event` 传进工具线程，
循环型的工具自己检查并退出。用 contextvar 而不是加函数参数，是因为工具签名会变成 JSON Schema
暴露给模型——加参数既污染 schema 又浪费 token。
我实测确认了 contextvar 的读能穿透 `to_thread`、写不能回传，所以 `set()` 的位置是硬约束。
至于为什么不用进程隔离：我的内置工具大量是**闭包**，闭包不可 pickle，进程方案会废掉一半工具。
**更进一步**，因为超时杀不掉线程，我还让同步工具的超时**不重试**——
否则重试会再起一个线程去写同一份资源。」

**实测补充**
- （待填）超时后线程仍存活的证据（`threading.enumerate()` 快照）。
  这条**已有确定的测试落点**：`tests/test_tools_executor.py` 的
  「同步工具超时后 `attempts==1` 且孤儿线程确实还活着」。

---

## D-10 Parse error 的自纠正**消耗** `max_steps` 预算

**背景**
文本 ReAct 模式下模型可能输出不符合格式的文本。框架把解析错误**回灌**给模型，
附上正确格式的提醒，让它自己改。问题是：这次"重试"要不要计入 `max_steps`？

**备选方案**
1. 不计入（`max_steps` 只数"成功的推理轮次"）。
2. 计入。

**选择**
方案 2。另外单独用 `AgentState.parse_errors` 计数，并有独立的 `max_parse_retries`（默认 2）做上限。

`[v2 新增]` **不重试 LLM 调用本身**（LLM 层已按 `retryable` 重试过）——
避免重试放大成 `max_steps × max_parse_retries` 次真实调用。

**理由**
1. **每次自纠正都是一次真实的 LLM 调用**，消耗真实的 token 和钱。若不计入，
   一个持续输出乱格式的模型可以无限循环——**成本放大器**，也是潜在的 DoS 面。
2. **`max_steps` 的语义应该是"最多调用几次 LLM"**，这是用户能理解的成本上限。
3. 两个计数器分工明确：`max_steps` 管**总成本**，`max_parse_retries` 管**格式病态程度**，
   两者触发不同的终止原因，trace 里能直接区分"任务太难"与"模型不会用这个格式"。

**代价**
- 一个只会输出乱格式的模型会更快耗尽 `max_steps`。
  缓解：终止结果里带上 `parse_errors` 计数与最后一次的原始输出。
- 心智负担：文档里用一句话说明两个上限的关系。

**面试怎么讲**
「文本 ReAct 下解析失败我会把错误**回灌**给模型让它自己改格式，但这次重试我**计入** `max_steps`
——因为每次自纠正都是一次真实的 LLM 调用，不计入的话，一个持续输出乱格式的模型就能无限循环，
变成成本放大器。我另外单设了 `max_parse_retries`，两个计数器职责不同：
`max_steps` 管总成本，`max_parse_retries` 管格式病态程度，
触发的是不同的终止原因，在 trace 里能直接区分『任务太难』和『模型不会用这个格式』。
另外我明确**不重试 LLM 调用本身**，因为 LLM 层已经重试过了，再叠一层就是乘法放大。」

**实测补充**
- （待填）构造一个"永远输出乱格式"的 `ScriptedLLM`，验证在第 3 次 parse 失败时终止。

---

## D-11 CLI 用 stdlib `argparse`，不用 `typer`

**背景**
环境里装了 `typer` 和 `rich`。`typer` 能让 CLI 代码少一半、自动生成帮助、带类型提示。

**备选方案**
1. 用 `typer`（开发体验最好）。
2. 用 `argparse`（零依赖）。
3. 双轨：有 typer 用 typer，没有就退化到 argparse。

**选择**
方案 2。`rich` 只在**渲染**处可选使用（`RichCallback` 有 print 兜底）。

**理由**
1. **CLI 是"框架可用性"的门面，必须零依赖可跑**。如果 `liteagent run` 需要装 typer，
   那"内核零第三方依赖"这句话在用户第一次运行时就破产了。
2. **双轨方案是负资产**：两套解析器 = 两套帮助文本、两套错误信息、两套退出码行为。
3. **退出码可控**。CLI 需要区分 4 种退出码，并且**不能把 traceback 打到 stdout**
   （否则 `--json` 的输出无法被 `jq` 消费）。
4. `main(argv) -> int` 的形态（不调 `sys.exit`）让 CLI 能直接被单元测试断言返回值——
   这是测试策略的一部分，而 `[v2 变更]` 补上了 `test_cli.py` 对 **chat / multi** 两个子命令的覆盖。

**代价**
- 每个子命令多约 15 行 argparse 配置代码，共约 120 行样板。
- 没有 typer 那种自动补全 + 彩色帮助。缓解：`rich` 用于结果渲染。
- 参数校验要手写（如 `--tools` 与 `--no-tools` 冲突检测）。`[v2]` 这些冲突检测都有测试用例。

**面试怎么讲**
「CLI 我坚持用 `argparse` 而不是 typer，虽然环境里 typer 是可用的。
理由是 CLI 是框架的门面，如果 `liteagent run` 需要装 typer，
那『内核零第三方依赖』在用户第一次运行时就破产了。
我也没做双轨，因为两套解析器就是两套帮助文本、两套错误信息、两套退出码行为，是坏交易。
另外 `main(argv) -> int` 不调 `sys.exit`，这样 CLI 能被单元测试直接断言退出码，不用起子进程；
`--json` 模式下我保证 stdout 只有 JSON、日志全走 stderr，可以被 `jq` 直接消费。」

**实测补充**
- （待填）`python -m liteagent run --provider echo --json -p "hi" | jq .` 的实测输出。

---

## D-12 并发模型：MemoryStore 用**同步** API，Blackboard 用 `threading.RLock` ★面试亮点

**背景**
框架里有两个共享可变结构：记忆存储（`MemoryStore`）和跨 Agent 黑板（`Blackboard`）。
它们都会被多个 Agent / 多个工具线程访问。asyncio 生态的直觉是"全都用 `asyncio.Lock`"。

**备选方案**
1. 全部异步：`async def add/get/search`，用 `asyncio.Lock` 保护。
2. 全部同步：纯内存操作用普通方法，用 `threading.RLock` 保护。
3. 双份 API（async + sync 都提供），各自一套锁。

**选择**
方案 2，并配一个关键约束：**临界区内不做任何 await、不做 I/O**。
- `MemoryStore`：全同步（`BufferMemory` / `VectorMemory` 各持有 `self._lock = threading.RLock()`）。
- `Blackboard`：同步方法为规范实现，用 `threading.RLock`；`a*` 方法只是薄包装，
  `awatch` 额外用 loop-bound `asyncio.Condition` 做变更通知。
- `MemoryManager` 保留 async 镜像，但实现内部**没有真 await**（除了调 LLM 的摘要路径）。
- `[v2 变更]` **同一个 RLock 也保护语义**：`search` 的副作用（`access_count`/`last_access_at`/
  `score`）必须在同一把锁内更新；向量矩阵的读也走同一把锁。`TeamConfig.share_memory=True`
  时多个子 Agent 共享一个 `MemoryManager`，worker 在各自线程 + 各自 loop 里跑，
  主 loop 同时可能在检索 —— 无锁会得到 `RuntimeError: list changed size during iteration`
  或读到一半的淘汰状态。

**理由**
1. **`threading.RLock` 不受 R-LOOP 陷阱影响**。这是决定性的：实测
   `asyncio.Semaphore` 在 `__init__` 里创建、跨两次 `asyncio.run()` 争用时抛
   `RuntimeError: ... is bound to a different event loop`。而 `Agent.run()` 每次调用都
   `asyncio.run()`——所以任何在 `__init__` 里创建的 asyncio 原语都是定时炸弹。
2. **工具的同步本质**。内置工具大量是同步函数，跑在 worker 线程里。
   它们访问 Blackboard 时**根本没有事件循环**（工作线程里 `get_running_loop()` 抛异常）。
   **同步 API 是唯一能同时服务"主 loop"与"worker 线程"两侧的形态。**
3. **临界区足够小**：只操作内存 dict，没有 await 点，因此 `RLock` 提供的互斥在语义上完全正确。
4. **不用 `asyncio.Lock` 反而避免了死锁**：如果写入方在 worker 线程，就必须
   `loop.call_soon_threadsafe` 转发，一旦忙等待/重入就会死锁。`threading.RLock` 可重入。

**代价**
- **不能有真 I/O**。如果将来记忆后端换成数据库/Redis，同步 API 会阻塞事件循环。
  缓解：接口不变，实现改成 `await asyncio.to_thread(...)`。
- **两套 API 的心智负担**。规范明确"sync 为规范实现、async 为薄包装、方向不许反"。
- `awatch` 的跨线程唤醒必须走 `loop.call_soon_threadsafe`，且在 worker 线程直接
  `condition.notify()` 是未定义行为。`[v2 变更]` 还补了两条实测约束：
  **写入方不做任何 asyncio 调用**（它只拿 watcher 快照，出锁后逐个转发），
  且对**已关闭的 loop** 必须 `if loop.is_closed(): continue` + try/except
  （实测 `call_soon_threadsafe` 对已关闭 loop 抛 `RuntimeError`）。
  `[v3 修正]` 但**载荷**才是真正踩过的雷：规范冻结的 `call_soon_threadsafe(cond.notify_all)`
  在本机 3.10.12 实测**唤醒不了任何 watcher** —— `asyncio.Condition.notify_all()` 要求调用方
  **持锁**，而 `cond.wait()` 期间锁是**释放**的，于是抛
  `RuntimeError: cannot notify on un-acquired lock`（等待方最后等到 timeout）。
  现在的载荷是 `_schedule_wake`：在目标 loop 的线程内先 `async with cond` 再 `notify_all`，
  满足持锁要求（见 `blackboard.py` 顶部 SPEC-AMBIGUITY 与 `_forward_notify`）。
  `[v3 变更]` 同轮还修了两个更深的洞：判脏依据从**会回退**的 `entry.version` 换成单调的
  `_write_seq_of(key)`（否则 TTL 过期 / delete / clear 后重写会「版本号撞回旧值、写入被静默吞掉」），
  并把判谓词移进 `async with cond` 临界区、放在 `wait()` **之前**（否则消费者挂在 yield 上
  处理上一条 entry 期间发生的写入会被永久丢弃）。回归测试见 `test_blackboard.py::WatchTests`。
- `MemoryStore.search()` 会更新 `access_count`（有副作用），调用方需知悉。

**面试怎么讲**
「并发模型上我做了个反直觉的选择：记忆和黑板**主要用同步 API + `threading.RLock`**。
两个原因。第一，我的内置工具大多是同步函数、跑在线程池里，那些 worker 线程里**根本没有事件循环**，
纯 async API 它们就用不了——同步 API 是唯一能同时服务主 loop 和 worker 线程的形态。
第二，我实测过一个坑：`asyncio.Semaphore` 如果在 `__init__` 里创建、
然后跨两次 `asyncio.run()` 争用，会抛 `RuntimeError: bound to a different event loop`。
而我的 `Agent.run()` 每次调用都是一次 `asyncio.run()`——所以那是个定时炸弹。
`threading.RLock` 跟事件循环无关，天然安全；前提是临界区里只有内存操作。
跨线程唤醒 watcher 时我走 `loop.call_soon_threadsafe`，但**载荷**不能直接是 `cond.notify_all`
—— 那是规范里冻结的写法，我在 3.10 上实测它**唤醒不了任何人**：`Condition.notify_all()` 要求
调用方持锁，而 `wait()` 期间锁是释放的，它会抛 `cannot notify on un-acquired lock`。
我改成在目标 loop 的线程里创建一个唤醒任务，先 `async with cond` 再 `notify_all`。
另外还要挡掉"loop 已关闭"的情况——对已关闭的 loop 调 `call_soon_threadsafe` 会直接抛
RuntimeError，这个也是实测出来的。」

**实测补充**
- （待填）`asyncio.Semaphore` 跨 loop 复用的报错复现记录（本条决策的证据）。
- （待填）Blackboard 多线程并发写 1000 次的正确性测试结果。

---

## D-13 为什么不用 LangChain（以及我们的取舍）★面试亮点

**背景**
简历第一句就是"参考 LangChain 和 Nanobot 设计模式"。这是面试的**第一问**。
但"参考"不等于"照抄"，也不等于"重写"——必须能说清**沿着哪条轴做了不同的选择、代价是什么**。

**备选方案**
1. 直接用 LangChain：几十个依赖、开箱即用的 provider 适配与 LCEL。
2. 用 LangChain 的核心抽象，替换掉不需要的部分。
3. 从零实现，把 LangChain 里我真正需要的 20% 显式写一遍。

**选择**
方案 3。并在本节给出 8 条对比轴，**每条都标注它对应的 `DESIGN_DECISIONS` 编号**
（无对应的要补决策——这就是本节存在的意义）。

| # | 轴 | LangChain 的做法 | liteagent 的做法 | 差异的理由 | 对应决策 |
|---|---|---|---|---|---|
| 1 | **依赖体积与安装** | 数十个传递依赖（`langchain-core` + provider 包 + `pydantic` + `aiohttp` …） | **0 个**（三方库全部是可选加速器，`try/except ImportError` 探测式降级） | 离线可测、CI 秒级、`pip install` 不拉一堆东西；代价是 HTTP/embedding/token 估算/CLI 全部自己写 | D-01 / D-04 / D-06 / D-11 |
| 2 | **事件模型** | `CallbackHandler` 的 `on_llm_start(**kwargs)` 一族，**kwargs 魔法、字段随版本漂移 | `EventType` 强类型枚举 + `TraceEvent` 全量字段 + `trace_stats()` **冻结字段表** | trace 可精确定义断言、可 JSONL、可 `jq`；代价是新增事件要改枚举（这是好事） | D-27 |
| 3 | **工具 schema** | **pydantic-only**（`args_schema: BaseModel`） | 反射 `inspect.signature` + `Annotated` + docstring 解析；pydantic 只是**可选**鸭子类型适配器 | 写工具不用学 pydantic；代价是 schema 反射有降级路径——所以我们把它**显式记账**在 `ToolSpec.warnings` 里（红线 12） | D-19 / D-05 |
| 4 | **prompt 组装** | LCEL `Runnable` 表达式树 | 一个**显式的六段函数** `abuild_prompt` | 顺序与内容可读、可测；代价是没有声明式组合（这个项目不需要） | — |
| 5 | **控制流可见性** | `AgentExecutor` 内部黑盒 + 若干提示词魔术 | `Agent.arun` 是**一个读得完的显式状态机**（每轮 7 步，事件矩阵冻结） | 能回答"第 N 轮到底发生了什么"——这是 agent 框架最该透明的地方；代价是没有并行分支/图的通用性 | D-02 / D-14 / D-27 |
| 6 | **可测性** | `FakeListLLM` 只能回文本，**无法断言 tools 是否真的暴露给模型** | `ScriptedLLM` 记录 `ScriptedCall(messages, tools, kwargs, response)` | 能测"模型看到的到底是哪几条消息、哪几个工具、call_id 是什么、重试了几次"；代价是多写 659 行确定性替身（`liteagent/llm/scripted.py`，`wc -l` 实测） | D-26 |
| 7 | **离线可跑** | 需要真 key 才能跑绝大多数路径 | 无 key 无网跑**全部**测试；`--provider echo` 可演示 | 面试能当场跑；代价是 provider 报文只能用 fixture 覆盖（诚实标 `code-only-not-run`） | — |
| 8 | **多 Agent** | LangGraph 图 DSL（另一套执行引擎与状态模型） | 两种内置模式（Sequential / Hierarchical）+ 黑板，**复用同一个 Agent 状态机** | 没有第二套执行引擎与第二套调试心智；代价是没有任意图拓扑 | D-21 |

**结论**：liteagent 不是"LangChain 的替代品"，而是**把 LangChain 里真正需要的 20% 显式地实现一遍**。
在面试里这条要讲成**取舍**而不是**否定**：
「LangChain 帮我省掉的每一行代码，都要用依赖体积、黑盒控制流、不可断言的事件来换；
在这个项目里我选择把这三样换回来。而且每条取舍我都能附上实测数据。」
`docs/INTERVIEW.md` 里的 8 行对比表就是本节第 1/2/3/6 行的浓缩版。

**代价**
- 没有 provider 生态：每接一个新厂商要自己写报文编解码（DeepSeek 只用了 5 行，因为它是
  OpenAI 兼容；Anthropic 的报文差异较大，花了 ~120 行）。
- 没有 LCEL，写复杂链要手写 Python（本项目的场景不需要）。
- 没有社区与文档，出问题只能自己看代码——所以本文档与 `INTERFACES.md` 必须写得足够细。

**面试怎么讲**
「我参考了 LangChain 的设计模式，但没有用它。不是因为"我要造轮子"，
而是因为我有两条硬约束：内核零第三方依赖、以及**离线可测**。
LangChain 帮我省掉的每一行代码，都要用依赖体积、黑盒控制流、不可断言的事件来换。
比如说它的 `FakeListLLM` 只能返回文本，我没法断言"模型到底看到了哪几个工具"——
而工具暴露正是 agent 框架最容易出错的地方。所以我自己写了 `ScriptedLLM`，
它记录每次调用的 messages、tools、kwargs 和消费到的响应。
再比如事件模型：LangChain 的 callback 是 `**kwargs` 魔法，版本之间字段会漂移，
我用强类型枚举 + 冻结的 data 字段表，trace 可以直接做精确断言。
**我把这个对比整理成了 8 条轴写在 `DESIGN_DECISIONS.md` 的 D-13 里**，
每一条都对应一个有理由、有代价的选择。」

**实测补充**
- （待填）`pip install langchain-core` 的传递依赖数量与耗时（作为第 1 条轴的量化证据）。
  本环境无外网，此项标记为 `code-only-not-run`（列出的是 `pip download --no-deps` 的解析结果或
  官方元数据），不得写成"实测安装成功"。

---

## D-14 `finish_reason` 参与控制流 ★面试亮点

**背景**
v1 的规范里，`LLMResponse.finish_reason` 只被塞进 `LLM_RESPONSE` 事件的 data，
**从不参与控制流**。于是：模型被 `max_tokens` 截断（`finish_reason=="length"` 且无 tool_calls）
会被判定为"模型给出最终答案"，`status=FINISHED`，返回半截 output。
面试必问"模型输出被截断了怎么办"，v1 答不上来。

**备选方案**
1. 忽略 `finish_reason`：模型说什么就信什么（最常见，也是最危险的）。
2. 截断就直接失败。
3. 按 `finish_reason` 分支：截断 -> 续写一次；内容过滤 -> 立即失败；
   声明了 tool_calls 但没给出 -> 走自纠正。

**选择**
方案 3（`INTERFACES.md` §9.4.6 的四个分支，`DEFAULT_MAX_TRUNCATION_RETRIES=1`）。

**理由**
1. **"半截答案 + FINISHED"是最坏的结果**：调用方以为成功了，下游拿到的是不完整数据。
2. **截断是可恢复的**：注入一句 "Your previous reply was truncated... Continue from where you
   stopped" 通常能让模型接着写完；但要**限次**（默认 1 次），否则又是一个无限循环面。
3. **`content_filter` 不可恢复**：重试同一个 prompt 会得到同一个结果，立即失败并记录
   `metadata["finish_reason"]` 才是诚实的。
4. **`finish_reason=="tool_calls"` 但 `calls` 为空**：这是模型"说要调用工具但没给出合法
   tool_call"，属于**可自纠正**的错误，应该走 D-24 的 `recoverable` 分支而不是当成最终答案。

**代价**
- 终止条件从"无 tool_calls"变成了"无 tool_calls **且** `finish_reason=="stop"`"，
  实现者要记住这条（写进了 §9.4.5 表格与 §9.4.6）。
- 多了一个 `max_truncation_retries` 配置项与 `truncation_errors` 计数器。
- Anthropic 没有 `content_filter`，映射表里刻意不复现该值（§6.4 第 8 条）。

**面试怎么讲**
「我一开始把 `finish_reason` 只放在 trace 事件里，后来发现那是个坑：
模型被 `max_tokens` 截断时 `finish_reason` 是 `length` 而且没有 tool_calls，
我的"无 tool_calls 就终结"逻辑会把它当成**成功的最终答案**返回，而内容其实是半截的。
修法是把 `finish_reason` 提到控制流里：`length` 就注入一句"你的输出被截断了，请从断点继续"并
**限次**续写；`content_filter` 直接失败并记录原因；`tool_calls` 但没给出合法调用则走自纠正分支。
终态的判定也从"无 tool_calls"收紧成"无 tool_calls **且** `finish_reason == 'stop'`"。」

**实测补充**
- （待填）三条分支的测试输出（`test_agent_react_native.py` 的 length / content_filter /
  tool_calls-empty 三例）。

---

## D-15 用模型的上下文窗口反推 token 预算 ★面试亮点

**背景**
v1 里窗口预算写死 3000 token，而单条工具结果上限是 8000 **字符**。
按 D-06 的换算，8000 个中文字符 ≈ 8000 token —— **一条 observation 就能把窗口打爆 2.6 倍**。
全规范没有任何地方用模型的 context window 反推预算，也没有对拼装后的最终 prompt 做总量校验。
面试官问"你怎么保证不炸 context length"，答案只有"我写了 3000"。

**备选方案**
1. 写死一个保守常量（3000），靠经验。
2. 要求用户自己算好所有参数。
3. 从 `context_window_tokens` 反推：减去回复预留、工具 schema 预留、system prompt 实占，
   并用同一个 tokenizer 估算。

**选择**
方案 3（`MemoryConfig.context_window_tokens` / `reserve_completion_tokens` /
`tools_schema_tokens_reserve`，`MemoryManager.from_config` 里反推，下限 512）。

**理由**
1. **预算必须由"对手"决定**：模型的窗口是外部约束，工具结果与历史是内部消耗，
   只有把外部约束显式建模，才能回答"为什么是 3000"。
2. **同一个 tokenizer 估算三处**（D-06 的"估算函数唯一"在这里兑现）：
   否则反推出来的预算和实际裁剪用的预算会对不上。
3. **最终校验兜底**：即便反推正确，`abuild_prompt` 拼装完仍可能超（工具 schema 估算不准、
   extra 注入很大），所以末尾必须有 `_enforce_context_budget()`：按
   「长期记忆块 → 窗口最旧消息（不动 pinned system 与最后一条 user）」的顺序裁剪，
   并 **emit `CONTEXT_TRUNCATED`**（兑现红线 12：任何降级必须可观测）。
4. **token 与字符的联动**：`max_observation_chars` 的生效值取
   `min(config.max_observation_chars, memory.tokenizer_chars_budget())`，
   不再让三处各自写死 8000。

**代价**
- 用户要知道自己模型的窗口大小（`context_window_tokens` 默认 `None` = 不反推，
  保持 v1 的行为，向后兼容）。
- 多了一个 `CONTEXT_TRUNCATED` 事件与四个 `stats()` 字段。
- 反推依赖 `estimate` 的准确性（±20% 误差），所以 reserve 值要给得保守。

**面试怎么讲**
「窗口预算我没有写死。我是从模型的上下文窗口**反推**的：
用 `context_window_tokens` 减去给回复预留的 token、减去工具 schema 的预留、
再减去 system prompt 的实占长度，下限 512。
**为什么必须这么做**：我一开始写死 3000 token，但单条工具结果的字符上限是 8000——
按我自己的中文换算（1 字 ≈ 1 token），一条 observation 就能把窗口打爆 2.6 倍。
而且我在组装 prompt 的最后还有一道总量校验，超了就从长期记忆块和窗口最旧的消息开始裁，
并**发一个 `CONTEXT_TRUNCATED` 事件**——降级必须可见，这是我给自己定的红线。」

**实测补充**
- （待填）三条不同窗口（4k / 8k / 128k）下的反推预算数值与最终 prompt 的估算 token 数对照表。

---

## D-16 循环防护三层 ★面试亮点

**背景**
v1 的重复检测**只有一个键**：`canonical_key = name + 完整 arguments 的 JSON`。
于是模型把参数改一个字符（`read_file('a.py')` → `read_file('./a.py')`）就永远不触发，
只能把 `max_steps` 烧完；A→B→A→B 的振荡检测不到；"观察结果没变"的无进展检测不到；
也没有整轮 run 的墙钟超时。面试问"无限循环怎么防"，v1 只能答"完全相同的动作算重复"。

**备选方案**
1. 只查完全相同动作（v1）。
2. 只用步数上限。
3. 多层防护：相同动作 + 无进展（观察摘要）+ 工具熔断 + 墙钟超时。

**选择**
方案 3，四个信号统一进「重复检测」那一步（任一命中就 nudge/fail）：

| 层 | 载体 | 命中条件 |
|---|---|---|
| L1 相同动作 | `AgentState.action_counts[canonical_key]` | `>= repeat_action_threshold`（默认 2） |
| L2 无进展 | `AgentState.observation_digests[blake2b(content)[:8]]` | 任一摘要计数 `>= repeat_action_threshold` |
| L3 工具熔断 | `AgentState.tool_failure_counts[name]` + `ExecutorConfig.disable_tool_after_failures` | 同一工具连续 infrastructure 失败 `>= 3` 后，executor 直接返回失败结果并 emit `TOOL_ERROR {disabled: True}`，**不再真正执行** |
| L4 墙钟 | `AgentConfig.max_wall_clock_s` + `RunTimeoutError` | 每轮开头检查 `utc_now() - state.started_at` |
| L5 预算 | `AgentConfig.max_total_tokens` + `BudgetExceededError` | 每次 LLM 响应后检查（见 D-18） |

**理由**
1. **参数微变是最常见的绕圈方式**，L1 抓不到；而"观察结果完全一样"是**无进展**的强信号
   （模型没拿到新信息却还在动），用内容摘要（而不是参数）来判，正好补上 L1 的盲区。
2. **熔断是"止损"而不是"检测"**：一个工具连续三次 infrastructure 失败，
   模型再调它也是浪费一次 LLM 轮次 —— 直接在 executor 层拒掉并给出
   `do not retry this tool with the same arguments; try a different approach` 的提示（D-24）。
3. **墙钟是唯一能兜住"每轮都在进展但总量失控"的机制**（`max_steps` 只管次数，不管耗时）。
4. **全部落地成 state 字段**（`tool_name_counts` / `observation_digests` / `tool_failure_counts`），
   因为 claim "我做了防护" 而字段不存在 = 面试官随手一点就露馅。

**代价**
- 多三个 `AgentState` 字段与两个配置项；`to_dict()` 会变长。
- L2 有轻微误伤：连续两次读同一个文件确实会被 nudge（但那是"相同动作"，本来就该 nudge）。
- 墙钟测试要用可控时钟（`tests.helpers.frozen_time`），否则会 flaky。

**面试怎么讲**
「无限循环我做了**三层**防护，因为单靠"完全相同的动作算重复"是抓不住真实情况的：
模型每次把参数改一个字符（`a.py` 换成 `./a.py`）就绕过去了。
第一层是一样的 canonical_key 计数；第二层是**无进展检测**——
我把每次工具结果的 `blake2b` 摘要计数，同一份观察反复出现就说明模型没拿到新信息，
这一层抓的是"参数变了但结果没变"；第三层是**工具熔断**：
某个工具连续三次 infrastructure 失败之后，我直接在 executor 层拒掉它，
并告诉模型"别用相同参数重试这个工具了"。
另外我加了整轮 run 的**墙钟上限**和 **token 总量上限**——
步数上限只管"调用几次"，管不了"每次调用多贵、总共跑多久"。」

**实测补充**
- （待填）两个用例的实际 trace：① 参数每次略变的重复调用被 nudge；② `max_wall_clock_s` 触发。

---

## D-17 HITL 审批：fail-closed ★面试亮点

**背景**
v1 里 `ToolSpec.dangerous` 只是个 bool 标记，`Callback.on_event` 返回 `None` 无法否决，
`ExecutorConfig` 没有 approval 字段，CLI 没有 `--yes`。
结果是 §3.3 声明"用户取消 / 回调要求中止"时抛的 `AgentAbortedError` **在全文没有任何触发路径**，
面试官问"危险工具怎么人审"，只能答"我留了一个异常类"。

**备选方案**
1. 只保留 `dangerous` 标记，让用户在 prompt 里自己处理。
2. 让 `Callback.on_event` 返回 bool 来否决（改掉回调协议）。
3. 显式的审批步：`requires_approval` + `approval_policy` 回调 + `ToolApprovalDeniedError`。

**选择**
方案 3。`dangerous` 管**展示过滤**（`registry.list(include_dangerous=)`），
`requires_approval` 管**控制流**，两者语义分开。
审批步位于「参数校验之后、执行之前」（§7.4.1 步骤 4.5）。

**理由**
1. **fail-closed 是唯一正确的默认**：`requires_approval=True` 且**没有** policy 时**拒绝**，
   而不是放行。一个安全机制在最坏情况下（用户忘了配 policy）必须往安全侧倒。
2. **不改回调协议**：`Callback.on_event` 返回 `None` 是个已经冻结的、很好的设计
   （回调不该能改控制流）。审批走独立的 `policy` 回调，返回 `bool`，语义清晰。
3. **`AgentAbortedError` 终于有了真实触发路径**：用户在 `approval_policy` 里抛它 →
   executor 转成 `CancelledError` → `Agent.arun` 走 §9.4.7 的取消路径 →
   `state.mark_finished(ABORTED)` + `RUN_FAILED {aborted: True}`。
4. **拒绝也回灌给模型**：`ERROR(ToolApprovalDeniedError): this tool requires human approval`
   让模型知道"不是工具坏了，是没被批准"，从而换一个不需要审批的路径。

**代价**
- 多一个异常类、一个配置字段、一个 CLI 参数（`--approve {never,write,all}`）
  与一个交互确认提示。
- `requires_approval` 需要工具作者思考（内置的 `delete_file`/`run_shell`/`python_exec` 已标）。
- 交互式审批在无人值守场景要显式配 `--approve all`，否则工具会被拒绝（这是有意的）。

**面试怎么讲**
「我的危险工具是有**人工审批**的，而且是 fail-closed 的：
如果某个工具标了 `requires_approval` 但调用方没有配置审批回调，我会**拒绝执行**而不是放行——
安全机制在最坏情况下必须往安全侧倒。审批的粒度是"参数校验之后、真正执行之前"，
这样模型不会因为参数错就弹一次审批。
我刻意没有改回调协议（`Callback.on_event` 返回 `None` 是很好的设计，回调不该改控制流），
而是加了独立的 `approval_policy` 回调，返回 bool。
用户在审批回调里可以抛 `AgentAbortedError` 主动中止整个 run，
它会一路传到 `Agent.arun` 变成 `ABORTED` 状态，trace 里也能看到 `aborted: True`。」

**实测补充**
- （待填）三个用例：无 policy 时拒绝并返回失败结果 / policy 返回 False / policy 返回 True 正常执行。

---

## D-18 token 与成本预算 ★面试亮点

**背景**
v1 的止损手段只有 `max_steps`（LLM 调用次数），而单次调用的 token 量可以差两个数量级
（一条 8000 字符的中文 observation ≈ 8000 token）。
`TokenUsage.estimated_cost_usd` 是一个"永远返回 None 的公开 API"，
被追问"它存在的意义"时无法回答；`AgentResult` 里也没有成本字段。

**备选方案**
1. 只保留 `max_steps`。
2. `max_steps` + 一个硬编码的 token 上限。
3. 三层预算：调用次数（`max_steps`）+ 总量 token（`max_total_tokens`）+ 单次 prompt token
   （`max_prompt_tokens`），并把成本做成可解释的函数而不是属性。

**选择**
方案 3：
- `AgentConfig.max_total_tokens`（建议默认 200000）、`max_prompt_tokens`
  （发请求前若估算 prompt 超限则先强制压缩一次，仍超限才失败）。
- `BudgetExceededError(limit, used, kind)` + `EventType.BUDGET_EXCEEDED`。
- 成本做成**显式纯函数** `config.estimate_cost_usd(usage, model=...)` + 可选的
  `MODEL_PRICES`（只列三条模型，**明确标注是示例价格、会变、未命中返回 None**）。
  `TokenUsage.estimated_cost_usd` 委托它；`AgentResult.metadata["cost_usd"]` 成为冻结字段；
  `TokenCounterCallback.cost_usd` 累加。

**理由**
1. **"步数上限"与"成本上限"是两个不同的东西**：一次 `read_file` 返回 8000 字符的中文
   和返回 20 字符的英文，消耗的 token 差三个数量级。两个上限都要有。
2. **价格表必须是可选且诚实的**：硬编码价格会过期。做法是"给一个小表让成本字段有值可展示，
   未命中一律返回 None"——**宁可没有数字，也不要编一个数字**。
3. **成本进 `AgentResult.metadata` 和 `TokenCounterCallback`**，
   这样"一次 run 花了多少钱"是 trace 里能直接看到的，而不是只有你知道怎么算。

**代价**
- 价格表的维护责任在用户（文档明确说明）。
- `MODEL_PRICES` 是唯一一处"疑似硬编码业务常量"的地方，必须在注释里说明它的性质。
- 多两个配置项、一个异常、一个事件。

**面试怎么讲**
「我的止损不止 `max_steps`。因为步数只管"调用几次"，管不了"每次多贵"——
一条 8000 字的中文工具结果按我自己的换算就是 8000 token，
所以我又加了**总量 token 上限**和**单次 prompt token 上限**。
成本这块我做了个显式的函数 `estimate_cost_usd(usage, model=...)` 加一张很小的价格表，
**未命中的模型一律返回 None**——我宁可没有数字，也不要编一个数字。
成本会写进 `AgentResult.metadata["cost_usd"]`，所以"这一次 run 花了多少钱"在 trace 里直接能看到。
超预算时抛 `BudgetExceededError` 并发一个 `BUDGET_EXCEEDED` 事件，不是静默失败。」

**实测补充**
- （待填）价格表命中/未命中两例的输出，以及一次真实 ScriptedLLM run 的 `cost_usd` 数值。

---

## D-19 `Annotated` 的探测方式（Python 3.10 实测坑）★面试亮点

**背景**
工具 schema 反射要识别 `Annotated[int, Param(default=5)]` 才能知道"这个参数其实有默认值"。
直觉写法是 `isinstance(annotation, Annotated)`。

**备选方案**
1. `isinstance(annotation, Annotated)`。
2. `typing.get_origin(annotation) is typing.Annotated`。
3. `getattr(annotation, "__metadata__", ())`。

**选择**
方案 3（也接受方案 2 作为等价写法；**明令禁止方案 1**）。

**理由（实测，非推测）**
在本机 Python 3.10.12 上：

```python
>>> isinstance(Annotated[int, 'x'], Annotated)
False                       # 不报错，静默 False
>>> get_origin(Annotated[int, 'x']) is Annotated
True
>>> hasattr(Annotated[int, 'x'], "__metadata__")
True
```

`typing.Annotated` 在 3.10 里是个普通类，`Annotated[int, 'x']` 的类型是 `typing._AnnotatedAlias`，
两者没有子类关系。所以 `isinstance` 写法**恒为 False 且不报错**——
死代码，而且是最难发现的那种（不崩、只是功能静默失效）。
后果：`Annotated[int, Param(default=5)]` 永远被判为 required，schema 里同时出现
`"required": ["x"]` 和 `"default": 5` —— **自相矛盾，正确的调用会被拒**。

同时 v1 的公式 `any Param.default is not _UNSET in metadata` 根本不是合法 Python，
实现者只能猜。

**代价**
- 冻结公式变长了一点（多一个 `_meta_default` 辅助函数），但它是**可直接编译**的。
- 三条边界用例必须进测试（非必需的，但能防止回归）：
  `Annotated[int, Param(default=5)]` 不进 required 且 properties 里有 default；
  `Annotated[int, 'desc']` 只加 description 不影响 required；
  `Annotated[str, Param(description='d')]` 的逐参描述优先于 docstring。

**面试怎么讲**
「这里有个我在 3.10 上实测出来的坑：`isinstance(Annotated[int, 'x'], Annotated)` 返回 **False**，
而且不报错。因为 `typing.Annotated` 在 3.10 里是个普通类，`Annotated[...]` 的实例是
`_AnnotatedAlias`，两者没有继承关系。所以任何写成 `isinstance(x, Annotated)` 的判断都是**死代码**，
最可怕的是它不崩，只是功能静默失效——`Annotated[int, Param(default=5)]` 会被当成必填参数，
模型不传就被我的校验器拒掉。
正确的写法是 `getattr(annotation, "__metadata__", ())` 或 `get_origin(x) is Annotated`，
我把这条写进了规范，并且加了三个针对性的测试用例。」

**实测补充**
- （待填）三条用例的实测输出（`test_tools_schema.py`）。

---

## D-20 每 loop 私有线程池，放弃 `set_default_executor` ★面试亮点

**背景**
v1 的规范里四处不自洽：
- §7.4.1 说"用 `asyncio.to_thread` 而**不自建** ThreadPoolExecutor"，
  同时又说"首次使用时 `loop.set_default_executor(ThreadPoolExecutor(...))` 设定容量"——
  到底建不建？
- "每个 loop 只设一次；若用户已设过则不覆盖"在 3.10 **无法可靠探测**（只有私有属性
  `loop._default_executor`，没有 getter）。
- 如果"已设置"的标志存在 executor 实例上，第二次 `asyncio.run()` 的新 loop 就会跳过设置，
  `thread_pool_size` 静默失效 —— 而这正是 §12 专门测试的场景（同一 executor 跨两次 `asyncio.run`）。
- `aclose` 说"关闭自建的线程池"，但如果根本没自建池，close 就是 no-op，`thread_pool` 参数从未被使用。

**备选方案**
1. 继续用 `set_default_executor`，补齐 per-loop 状态与探测写法。
2. 放弃它，executor 自持 per-loop 私有线程池（`LoopBoundPool.thread_pool(key, size)`），
   同步工具走 `loop.run_in_executor(tp, tool.run, args)`。

**选择**
方案 2。

**理由**
1. **语义清晰**：`thread_pool_size` 明确是"**每个 loop 的私有池**大小"，不再是
   "进程默认线程池大小"这种含混说法。
2. **状态位置正确**：per-loop 状态放在 `LoopBoundPool` 的 per-loop dict 里，
   第二次 `asyncio.run` 必然拿到新 loop 的新池 —— `thread_pool_size` 不会静默失效。
3. **关闭责任明确**：`aclose` 遍历池中所有 `ThreadPoolExecutor` 调
   `shutdown(wait=False, cancel_futures=True)`；外部传入的 `thread_pool` 不关。
   这也是 D-09 那条"真卡死的工具最坏情况是线程泄漏"的缓解措施真正落地的地方。
4. **不碰 loop 的全局状态**：`set_default_executor` 会改掉整个 loop 的默认执行器，
   一个库去改宿主 loop 的全局状态是不礼貌的（还可能砸掉宿主自己的池）。

**代价**
- 每个 loop 会多一个线程池（嵌套 loop 场景下 delegate 线程里各一个），
  需要靠 `aclose` / `LoopBoundPool.release_loop` 清理 —— 这条清理路径**有测试**。
- 比 `to_thread` 多两行代码（要显式取池、显式 `run_in_executor`）。
- `[v2 实测]` 顺带发现了 `WeakKeyDictionary` 方案的泄漏：`asyncio.Semaphore` 争用后把 loop
  存进 `self._loop`，value 强引用 key，**弱引用永不失效**（实测 3 次 `asyncio.run` 泄漏 3 个 loop
  与 3 个原语）。所以池改用 `dict[int, ...]` + 显式 `release_loop`。

**面试怎么讲**
「线程池我改过一次设计。原来我用 `loop.set_default_executor` 来设容量，
后来发现三个问题：一是"用户是否已经设过"在 3.10 探测不了（只有私有属性）；
二是我把"只设一次"的标志放在 executor 实例上，结果第二次 `asyncio.run` 的新 loop 就会跳过设置，
`thread_pool_size` **静默失效**——而这正是我专门写测试守的场景；
三是会改掉宿主 loop 的全局状态，一个库不该干这个。
所以我把线程池改成 executor 自持的 **per-loop 私有池**，
`thread_pool_size` 的语义也就明确了：每个 loop 一个多大的池。
顺带我还实测到一个内存泄漏：我原来用 `WeakKeyDictionary` 缓存 asyncio 原语，
但 `asyncio.Semaphore` 一旦争用就会把 loop 存进 `self._loop`——
**value 强引用 key，弱引用永远不会失效**，3 次 `asyncio.run` 就泄漏 3 个 loop。
现在改成显式的 `release_loop`，并且有测试断言"连续 3 次同步调用后池内条数为 0"。」

**实测补充**
- （待填）`len(pool)` 在连续 3 次 `execute_sync` 后的值为 0 的断言输出。
- （已实测）`WeakKeyDictionary` 方案的泄漏复现：3 次 `asyncio.run` 后 `pool` 仍持有 3 个 key。

---

## D-21 跨 loop 限流一律用 threading 原语 ★面试亮点

**背景**
`HierarchicalAgent` 的 delegate 工具是**同步工具**，被 manager 的 executor 用线程池丢进
worker 线程，闭包里再 `run_sync(worker.arun(...))` —— 每次都 `asyncio.run()` **新建一个 loop**。
后果：
- `[v1]` `subagent_concurrency=3` 在全规范里**没有任何执行点**；用 loop-bound 信号量实现也没意义
  （每个 delegate 拿到的是自己新 loop 的全新信号量，计数永远是满的，并发完全不受限）。
- 同理 `sequential_tools` 的 `pool.lock("seq:"+name)` 只在单个 loop 内互斥，跨 loop/跨线程完全不串行
  —— 而它存在的唯一理由就是跨调用互斥（如 `run_shell`）。§12 的"sequential_tools 真正串行"
  测试因为只测单 loop 会**通过并掩盖真实缺陷**。
- 每次 delegate 还各自设一个线程池，3 个并发 delegate = 24+ 线程常驻。

**备选方案**
1. 继续用 loop-bound 原语，并在文档里说明"多线程委派时并发不受限"。
2. 跨线程/跨 loop 必须生效的限流与互斥**一律用 threading 原语**；
   loop-bound 原语只允许用于"绝不离开当前 loop"的场景。

**选择**
方案 2，并把它升级为**红线 13**。

**理由**
1. **正确性优先**：`threading.Semaphore`/`Lock` 与事件循环无关，跨线程跨 loop 天然正确；
   loop-bound 原语在"每次调用都新建 loop"的路径上**语义上就不成立**，不是"不太准"而是"完全无效"。
2. **测试必须能抓住它**：`test_tools_executor.py` 里加了
   「`sequential_tools` 在**两个嵌套 loop**（一个普通 agent + 一个 delegate）下仍真正串行」
   这个用例——单 loop 的测试会通过并掩盖缺陷。
3. **`subagent_concurrency` 终于有落点**：`HierarchicalAgent.__init__` 里的
   `threading.Semaphore(config.subagent_concurrency)`，delegate 闭包里
   `acquire(timeout=...)`，拿不到就返回 refused 字符串（与环检测同形）。
4. **`arun_plan` 是例外**：它跑在父 loop 里、worker 的 `arun` 是 async 的，
   所以那里**可以**用 per-loop 的 `asyncio.Semaphore`（甚至更自然）。规范明确写出这个区别，
   避免实现者"为了统一"把两处都写成一样。

**代价**
- 同一份"限流"在两条路径上有两种实现（threading vs asyncio），需要在文档里解释清楚
  （已在 `ARCHITECTURE.md` §4.3 做成对照表）。
- `sequential_tools` 的 `threading.Lock` 在 async 调用方一侧要用
  `await asyncio.to_thread(guard.acquire)` 获取，避免阻塞事件循环；
  代价是等待期间会占一个线程（数量很小，因为它们是串行的）。
  `[v3 修正]` 但**朴素写法有洞**，这条代价比看上去重：`await asyncio.to_thread(lock.acquire)`
  在调用方被取消时，asyncio 只取消 `await` 层，worker 线程仍会把 `lock.acquire()` 跑完；
  而归还锁的 `finally` 属于**已被取消的协程**、永不执行 —— 锁被一个没有逻辑归属者的线程
  永久持有，该工具此后每次调用都卡在 `lock.acquire()` 上。更重的是卡死的是**默认执行器**
  里的非 daemon 线程，`asyncio.run` 收尾 `shutdown_default_executor()` 去 join 它时，
  同步 API（`execute_sync` / `Agent.run`）会**无异常、无超时地永久挂起**。
  现在用 `_CancelSafeAcquire`（取到锁后若发现已取消就**原地归还**）关掉这个窗口，
  代价是多了一层 `_aborted` / `_release_lock` 状态机来保证锁恰好归还一次。
- docstring 必须写明已知限制：同步工具超时后 orphan 线程可能仍持有底层资源，
  而锁已在调用方释放。

**面试怎么讲**
「这里有个我自己抓出来的**完全无效的机制**：我的子 Agent 并发上限原来是用 loop-bound 的
asyncio 信号量实现的。但 delegate 工具是**同步工具**，跑在 worker 线程里，
而它在闭包里会 `run_sync` 新建一个 event loop——所以每个 delegate 拿到的都是**自己 loop 的
全新信号量**，计数永远是满的，并发完全不受限。同理我的 `sequential_tools` 互斥也只在单 loop 内有效。
更麻烦的是，我原来的测试只测了单 loop，所以**测试是通过的**，缺陷被掩盖了。
修法是：**跨线程/跨 loop 必须生效的限流与互斥一律用 threading 原语**，
loop-bound 原语只允许用在"绝不离开当前 loop"的场景（比如显式的 plan 执行，那是在父 loop 里 gather）。
我还补了一个"两个嵌套 loop 下仍然真正串行"的测试用例来守这条。」

**实测补充**
- 嵌套 loop 下的串行性：`tests/test_tools_executor.py::ConcurrencyTests::test_sequential_tools_serialise_across_two_nested_loops`
  把 `nested_loop_tool` 放进 `sequential_tools`，连续两次 `asyncio.run(execute_many(3 个并发调用))`，
  断言**峰值并发恒为 1**（对照组 `test_non_sequential_tools_may_overlap` 把同一个工具**不**放进
  `sequential_tools`、其余配置相同，断言峰值 ≥ 2 —— 证明上面那条断言不是恒真）。
  跑法：`python3 -m unittest tests.test_tools_executor -k sequential` → `Ran 3 tests ... OK`。
- 取锁被取消的路径：`test_cancelled_lock_waiter_does_not_orphan_the_sequential_lock`（[v3] 新增，
  守 `_CancelSafeAcquire`）——取消等锁者之后，该工具的下一次调用仍能在毫秒级拿到锁。

---

## D-22 `execute_many` 永不抛（`fail_fast` 的语义修正）

**背景**
v1 冻结为"`config.fail_fast=True` 时，首个非 ok 结果会取消其余任务并抛 `ToolRetryExhaustedError`"。
这同时违反三处：
(a) 同一节的 `execute()` 明确"永不向外抛工具异常"；
(b) §13 红线 6 禁止 `Agent.arun`/`execute` 向外抛业务异常；
(c) §9.4.1 里 `results = await executor.execute_many(calls)` **没有 try/except**。
异常类型也错：首个失败的根因不一定不是重试耗尽（可能是校验失败、工具不存在）。

**备选方案**
1. 保留抛错语义，在 §9.4.1 补 try/except，并把 `execute_many` 加进红线的例外清单。
2. 让 `execute_many` 与 `execute` 的契约一致：**失败一律编码为 `ToolResult`，永不抛**。

**选择**
方案 2。`fail_fast=True` 时：发现首个 `ok=False` 后对其余未完成 future 调 `cancel()`，
再 `await asyncio.gather(*futures, return_exceptions=True)` 收尾，
被取消的位置填 `ToolSkippedError(reason="cancelled_by_fail_fast")`，
返回列表长度与 `calls` 一致，并 emit `TOOL_ERROR {fail_fast_first_index: i, skipped: N}`。

**理由**
1. **同一个类里的两个方法不该有相反的异常契约**。v1 的写法让调用方必须知道
   "`execute` 不会抛但 `execute_many` 会抛"，这是纯粹的心智负担。
2. **违反红线的措辞比违反红线本身更危险**：v1 的红线 6 只列了 `execute`，
   于是 `Agent` 实现者会照抄伪代码写出"异常直接穿透 `arun`"的版本。
3. **`ToolRetryExhaustedError` 是错的类型**：它要求带 `.attempts`/`.last_error`，
   而校验失败、工具不存在根本没有 `attempts`。

**代价**
- 调用方要自己遍历结果判断是否有失败（本来也该这么做）。
- `fail_fast` 的"快速失败"价值从"抛异常"变成"提前取消兄弟任务省时间"，
  语义变弱了一点——但换来契约统一，值得。
- 多了一个异常类 `ToolSkippedError` 与一个 `metadata` 键（`skipped_by_fail_fast`）。
- `[v3 修正]` 还有一个**容易写错**的代价：「本批次是否触发过 fail_fast」必须是**批次级局部状态**，
  绝不能挂在 executor 实例上。同一个 executor 会被并发的两个 `execute_many` 批次共享，若 B 批次
  开头重置了实例标志，A 批次收尾时就会把自己亲手取消的兄弟任务误判成「调用方取消」而重新抛
  `CancelledError` —— 直接破坏「结果列表与 `calls` 等长」的契约。现在它由
  `_watch_fail_fast` 返回一个局部布尔（见 `execute_many`）。

**面试怎么讲**
「`execute_many` 我一开始设计成 `fail_fast=True` 时抛异常，后来发现这是自相矛盾的：
同一个类里的 `execute` 承诺"永不抛工具异常、失败一律编码成 `ToolResult`"，
但 `execute_many` 会抛——调用方得记住这个不对称。
更糟的是我自己写的 Agent 主循环里直接 `await execute_many(calls)`，没有 try，
所以只要用户开一次 `fail_fast`，异常就会穿透 `Agent.arun`、违反我自己定的红线。
而且抛的异常类型也是错的：首个失败往往不是"重试耗尽"。
现在改成：**永远返回与 calls 等长的结果列表**，被 `fail_fast` 取消的位置填
`ToolSkippedError`，并额外发一个事件记录"首个失败的下标与跳过了几个"。」

**实测补充**
- `python3 -m unittest tests.test_tools_executor -k fail_fast` → `Ran 4 tests ... OK`。其中
  `test_fail_fast_cancels_siblings_and_keeps_alignment` 的实测形态：4 个调用、`concurrency=1`、
  `fail_fast=True` 时 —— 结果长度 **== len(calls) == 4**，`results[0].error_type == "ToolExecutionError"`，
  其余 3 项 `error_type == "ToolSkippedError"` 且 `metadata["skipped_by_fail_fast"] is True`；
  事件流里**恰好 1 条**带 `fail_fast_first_index=0, skipped=3`。
- 并发批次不共享状态：[v3] 新增 `test_concurrent_fail_fast_batches_do_not_share_batch_state`
  在同一次运行里通过（守上面「代价」最后一条）。

---

## D-23 把「长期存储」做成真的（`save`/`load`）★面试亮点

**背景**
简历 bullet 1 写的是"记忆层集成短期对话历史与长期**向量存储**"。
但 v1 的 `VectorMemory` 是纯内存 dict，`VectorConfig` 没有 `path` 字段，
`MemoryManager` 只有 `to_dict()` 没有 `from_dict()`，进程退出即全部丢失——
"长期"仅在单次进程内成立。面试官问"存哪、怎么跨进程"时没有任何规范依据可答。

**备选方案**
1. 加持久化：`VectorMemory.save(path)` / `load(path)`（JSONL），`MemoryManager.persist` / `restore`。
2. 明确降级：在文档里写"长期 = 跨会话生命周期（进程内），不跨进程持久化"，并把简历话术改掉。

**选择**
方案 1（方案 2 作为兜底话术的一部分也写进 `INTERVIEW.md`）。

**理由**
1. **"存储"和"检索"是两件事**。只做检索却在简历里写"存储"，是典型的"能力膨胀"。
   加 60 行 JSONL 读写就能让这句话变成真的，收益远大于成本。
2. **零依赖友好的格式**：JSONL 每行一个 `MemoryItem.to_dict(include_embedding=True)`，
   人可读、可 `grep`、可增量写、坏行可跳过。
3. **维度校验仍然是硬约束**：`load` 时 embedding 维度与 `self.dim` 不符 -> `MemoryStoreError`
   （与 D-08 的"注入 embedder 的维度是权威"一致）。
4. **两类测试都值钱**：往返后 `len` 与 `search` 顺序一致、embedding 逐元素相等、
   坏行跳过、维度不符报错——这四条测试本身就是"我做了持久化"的证据。

**代价**
- 格式与版本兼容性由用户负责（文档写明"这是简单 JSONL，没有 schema 版本迁移"）。
- 大 embedding（1536 维 × 10000 条）会产生几十 MB 文件；
  缓解：`include_embedding=False` 可只存文本（检索时重算），并在文档说明这个取舍。

**面试怎么讲**
「我的"长期记忆"一开始只是内存里的一个 dict。后来我自己意识到一个问题：
简历上写的是"长期**存储**"，但进程一退就全没了，那这个词就是虚的。
所以我加了 `save`/`load`：JSONL 每行一个记忆条目（带 embedding），
加载时做维度校验、坏行跳过；`MemoryManager` 上还有 `persist`/`restore`。
我有测试专门断言"往返之后条数和检索顺序一致、embedding 逐元素相等"。
如果只做检索不做持久化，我会把简历改成"向量检索 + 可注入持久化后端"——
**宁可把话说明白，也不要让简历上的一个动词落空**。」

**实测补充**
- （待填）往返测试的输出（条数、顺序、维度），以及 10000 条 × 256 维的落盘体积。

---

## D-24 错误分类的第二层：recoverable vs infrastructure ★面试亮点

**背景**
v1 的重试分类只到"瞬时（可重试）vs 永久（不可重试）"一层。
缺了对 agent 更重要的第二层：**模型能自纠正**（参数错、工具名错、arguments 非法 JSON）
vs **环境或实现故障**（工具内部 bug、沙箱拒绝、依赖缺失）。
两者当前都被塞成 `ERROR(type): message` 回灌给模型，
模型对第二类毫无办法，只能反复调用同一个坏工具直到烧完 `max_steps`。

**备选方案**
1. 只保留"可重试/不可重试"一层（v1）。
2. 加第二层分类，并让**回灌文案**随类别不同。

**选择**
方案 2：`feedback_kind ∈ {"recoverable", "infrastructure"}`，写进 `ToolResult.metadata`。

| `feedback_kind` | 触发类型 | 回灌文案要求 |
|---|---|---|
| `recoverable` | `ToolValidationError` / `ToolNotFoundError` / `ToolDefinitionError` | **必须告诉模型怎么改**：可用工具名列表、schema 期望的类型、正确格式示例 |
| `infrastructure` | `ToolExecutionError` / `SandboxViolationError` / `ToolTimeoutError` / `ToolRetryExhaustedError` / `ToolSkippedError` / `MemoryStoreError` / `ToolApprovalDeniedError` | **必须劝模型换策略**：`do not retry this tool with the same arguments; try a different approach or give your final answer` |

**理由**
1. **回灌文案是提示工程的一部分**，而提示工程最忌讳"对所有错误说同一句话"。
   `recoverable` 的失败要给出**具体的修法**（工具名清单、类型期望），
   模型才有机会一次改对；`infrastructure` 的失败要明确**劝阻重试**，
   否则模型会礼貌地把同一个坏调用再发一遍。
2. **它和 D-16 的熔断是配套的**：`infrastructure` 失败计数进
   `tool_failure_counts`，连续 3 次就熔断——文案与机制同源，不会出现
   "文案说别重试，机制却放它一直重试"的分裂。
3. **它是可观测的**：`metadata["feedback_kind"]` 让 trace 里能统计
   "这次 run 里有多少失败是模型自己的锅、多少是环境的锅"——
   这是排查 agent 表现最有用的一条切分。

**代价**
- 多一个 `metadata` 键与一张映射表（必须逐条实现，不能"看着办"）。
- 分类本身有边界情况（例如 `ToolExecutionError` 里如果是"参数语义错"，
  我们仍归为 infrastructure —— 因为工具作者可以通过 `retryable` 与
  返回自定义 `ToolResult` 来精细控制，这是逃生口）。

**面试怎么讲**
「错误我分了**两层**。第一层是常规的"可重试/不可重试"——我的 `retryable` 是异常的类属性，
所以异常层次表本身就是重试策略表。但 agent 场景里第二层更关键：
这个错**模型自己能改吗**？参数类型错、工具名拼错、arguments 不是合法 JSON，
这些是模型能自纠正的，我回灌的文案里会带上"可用工具名列表"和"schema 期望的类型"，
让它一次改对。而工具内部 bug、沙箱拒绝、超时这些是环境故障，模型改不了，
我的文案会明确说"别用相同参数重试这个工具，换个方法或者直接给最终答案"。
这两类我还分别接了不同的机制：可恢复的只计数，基础设施类的连续三次会**熔断**，
不再真正执行——文案和机制同源，不会出现"嘴上说别重试、实际上放它一直重试"的分裂。」

**实测补充**
- （待填）两类回灌文案的实际字符串，以及一次 run 里两类失败的计数统计。

---

## D-25 双 API 与 `run_sync` 的工厂签名

**背景**
v1 的 `run_sync(coro)` 收**协程对象**，同时文档要求"已有运行中的 loop 时抛 `ConfigError`"。
调用方按文档写 `run_sync(self.achat(...))` 时，**协程已经在检查之前构造好了**，
抛异常后这个协程永远不会被 await —— 在 3.10 上会打
`RuntimeWarning: coroutine ... was never awaited`。
5 个同步包装 API（`chat`/`execute_sync`/`build_prompt`/`run`/`MultiAgent.run`）全会触发，
`-W error` 下测试直接失败。

**备选方案**
1. 保持收协程对象，并在抛 `ConfigError` **之前** `coro.close()`。
2. 改成收**工厂函数**：`run_sync(factory: Callable[[], Coroutine])`，
   所有调用点写 `run_sync(lambda: self.achat(...))`。

**选择**
方案 2（唯一形态；方案 1 作为注释里的备选说明，不作为契约）。

**理由**
1. **工厂签名在"不该执行"的场景下天然不产生协程对象**，从根上消除 `never awaited` 警告，
   而不是靠"记得 close"。
2. **它让"是否真的会跑"变成一个显式的决定**：`lambda:` 的写法逼调用方意识到
   "这段代码是在另一个线程/另一个 loop 里执行的"。
3. **同步 API 的方向也被冻结**：默认 async-first（先写 `a*`，同步版是薄包装），
   但 `Blackboard`/`MemoryStore`/`Tokenizer`/`BufferMemory` 是 **sync-first 例外**
   （sync 是规范实现，async 是薄包装），因为 worker 线程里没有 loop（D-12）。

**代价**
- 所有调用点多一层 `lambda:`，可读性略降（换来的是正确性）。
- 需要一条测试守 `run_sync` 在 loop 内抛 `ConfigError`（已列入 `test_tools_executor.py`）。
- `sync-first 例外`是"方向不许反"的一条额外规则，实现者要多读一行。

**面试怎么讲**
「同步包装异步的 API 我踩过一个测试输出的坑：`run_sync(coro)` 收的是协程对象，
但如果当前已经在一个运行中的 loop 里，我是**先构造协程再检查**的——
抛异常后那个协程永远不会被 await，Python 会打 `coroutine was never awaited` 警告。
我有 5 个同步包装 API，全都会触发；开了 `-W error` 之后测试直接失败。
修法是把签名改成收**工厂函数**，调用点统一写 `run_sync(lambda: self.achat(...))`——
**在"不该执行"的路径上根本不产生协程对象**，比"记得在异常前 close"可靠得多。
另外我把双 API 的方向也写死了：默认 async-first，
但黑板和记忆存储是 sync-first 例外，因为 worker 线程里没有事件循环。」

**实测补充**
- （待填）修改前后 `python3 -W error -m unittest` 的输出对比。

---

## D-26 时间与随机性的注入点

**背景**
v1 给了退避的可复现手段（`jitter=0` / `rng_seed`），但：
- **没有任何可注入的 sleep 缝**：默认 `backoff_base_s=0.25` + `max_retries=2` 就是每个重试测试
  真睡 0.75s；而 `delay = max(delay, retry_after_s)` 遇到一个 `Retry-After: 60` 的 429 夹具
  会让测试睡 60 秒。
- `delay_for(attempt, rng=None)` 的 `rng is None` 行为**完全未定义**，
  三种实现都"符合文档"：崩溃 / 用模块级全局 random（不可复现 → flaky）/ 用 `rng_seed` 现造。
- 规范定义了唯一时钟 `utc_now()` 却**没有任何模块被要求使用它**，
  各处都是 `default_factory=time.time`；测试要固定时间只能全局 patch `time.time`（进程级，
  可能干扰 logging/asyncio 计时）。

**备选方案**
1. 让每个测试自己 patch `asyncio.sleep` / `time.time`。
2. 在配置里冻结注入点：`SleepFn`、实例级 `random.Random`、`utc_now` 作为唯一时钟。

**选择**
方案 2：

| 注入点 | 冻结内容 |
|---|---|
| `SleepFn = Callable[[float], Awaitable[None]]` | `RetryPolicy.sleep_fn` / `ExecutorConfig.sleep_fn` / `LLMConfig.sleep_fn`（优先级写死）；**一切退避等待、`ScriptedResponse.delay_s`、`ScriptedLLM.latency_s` 都必须经由它，禁止直接 `await asyncio.sleep`** |
| rng | `ToolExecutor.__init__` 与 `BaseLLMClient.__init__` 各创建一次 `self._rng = random.Random(policy.rng_seed)` 并在实例内复用；`rng is None` 时用 `random.Random(self.rng_seed)`；**禁止模块级 `random.*` 全局函数** |
| 时钟 | `config.utc_now` 是唯一时钟，所有时间戳 `default_factory` 用它；测试用 `tests.helpers.frozen_time(ts)` |

**理由**
1. **测试的墙钟时间必须可控**：一个 `Retry-After: 60` 的夹具不应该让 CI 卡 60 秒。
   `RecordingSleep` 只记录 delays、立即返回 —— 断言从"墙钟耗时"变成"延迟序列"，既准确又快。
2. **rng 的实例化时机决定可复现性**：每次调用重建 rng，重试序列就不可复现，
   与 D-07 的"断言每次间隔 0.25/0.5"直接冲突。
3. **唯一时钟让"固定时间"变成一行 patch**：`frozen_time` 只 patch `liteagent.config.utc_now`，
   不碰 `time.time`（避免干扰 logging/asyncio 的计时）。

**代价**
- 配置类多一个 `sleep_fn` 字段，实现者要在 `__init__` 里解析一次并保存为 `self._sleep`。
- `frozen_time` 的有效性依赖"各模块从 config 取时钟"这条纪律（已写进红线 17 与 §2.2）。
- 测试必须显式注入才能得到确定性；不注入时行为与生产一致（这是对的）。

**面试怎么讲**
「可测性我专门做了**注入点**。退避等待全部走一个可注入的 `sleep_fn`，
测试里塞一个只记录延迟、立即返回的 `RecordingSleep`——
所以我可以精确断言"重试了两次，间隔是 0.25 和 0.5 秒"，而且 CI 一秒都不会真睡。
**这个不是为了好看**：我有一个 429 的夹具带 `Retry-After: 60`，
如果不注入，那个测试就会真的睡 60 秒。
随机数我冻结成"每个实例持有一个 `random.Random(seed)` 并复用"，
因为每次重建 rng 会让重试序列不可复现——断言就写不出来了。
时间也一样：我定义了唯一的时钟函数 `utc_now`，
测试用一个 contextmanager 只 patch 它，不去 patch 全局的 `time.time`，
避免干扰 logging 和 asyncio 自己的计时。」

**实测补充**
- （待填）`RecordingSleep` 记录的 delays 序列（`[0.25, 0.5]`）。

---

## D-27 事件发射归属矩阵 ★面试亮点

**背景**
v1 里同一个事件被两层重复发射，且没有任何一节规定归属：
- `TOOL_STARTED` 既在 `Agent` 的 `§9.4.1` 第 5 步发，又在 `ToolExecutor` 的 `§7.4.1` 5.a 发；
- `LLM_REQUEST` 既在 `BaseLLMClient._emit` 发，又在 `Agent` 第 2 步发；`LLM_RESPONSE`/`LLM_ERROR` 同理；
- `MEMORY_RETRIEVE` 的 data 里有一个**从未定义过**的变量 `retrieved`。
结果是 §12 明确要断言的"事件序列"会有 2^n 种实现。

**备选方案**
1. 每个实现者自己判断"这条事件该谁发"。
2. 冻结一张**排他**的归属矩阵：每条事件有且只有一个发射者，Agent 不得重复发。

**选择**
方案 2（`INTERFACES.md` §2.7）。

**理由**
1. **事件序列是 trace 的公共契约**：它是 `--json` 输出、`trace_stats` 统计、
   测试断言、以及用户排障的共同基础。只要有两处可能发同一条事件，
   这个契约就不成立。
2. **归属与"谁拥有那段状态"一致**：LLM 的请求/响应只有 LLM 层知道细节
   （`retry` 次数、`latency_ms`、`usage`）；工具的开始/重试/结束只有 executor 知道
   （`attempt`、`orphan_thread`、`approved`）；Agent 只发它自己状态机里的东西。
3. **顺手修掉了两个具体 bug**：`MEMORY_RETRIEVE` 的 count 改成
   `len(memory.last_retrieved)`（`abuild_prompt` 里记录、**不再触发第二次 search**，
   因为 `search` 有写 `access_count` 的副作用）；`TraceEvent.__post_init__` 里
   **构造即校验**保留键（而不是 `to_dict` 时静默覆盖）。

**代价**
- 一张必须逐条实现的表（写进 §2.7），实现者不能"就近顺手发一下"。
- 新增事件类型时要想清归属（这是好事）。
- `Agent` 的伪代码里少了那两行 emit，读起来"不像完整流程"——
  但 `ARCHITECTURE.md` 的时序图标注了每条事件的发射者，弥补了这一点。

**面试怎么讲**
「有一类 bug 特别隐蔽：**同一个事件在两层被发了两遍**。
我原来 `TOOL_STARTED` 在 Agent 的步骤里发一次、在 executor 里又发一次，
`LLM_REQUEST` 也是。平时看 trace 只会觉得"事件好像多了一点"，
但一旦你要**断言事件序列**（我确实有这种测试），就完全没有确定答案。
所以我冻结了一张**排他**的归属矩阵：每条事件有且只有一个发射者，
LLM 的归 LLM 客户端、工具的归 executor、状态机的归 Agent，
并且明确规定"Agent 不得重复发"。
顺带还发现两个小 bug：一个事件的 data 里引用了一个不存在的变量，
还有一个事件为了拿到计数会**再调一次检索**——而检索是会改 `access_count` 的，
重复调用就污染了数据。现在改成从上一次检索的缓存里取。」

**实测补充**
- （待填）`test_callbacks.py` 的 `EventType` 集合快照与 `trace_stats` 的完整 dict。

---

## 附：决策之间的依赖关系

```text
零依赖红线（INTERFACES §1.3，umbrella，不单独编号）
   ├──> D-01 核心结构用 dataclass 而非 pydantic
   ├──> D-03 @tool 默认不自动注册（注册表显式注入）
   ├──> D-04 HashingEmbedder（没有 faiss/numpy 也要能跑）
   ├──> D-06 混合 token 估算（没有 tiktoken）
   ├──> D-11 CLI 用 argparse（没有 typer）
   ├──> D-13 与 LangChain 的取舍（零依赖是第一条对比轴）
   └──> D-12 同步 MemoryStore + threading.RLock

D-02 双模式统一 ──┬─> D-10 parse 重试计入 step（只在 text 模式存在）
                  └─> D-14 finish_reason 进控制流（native 的截断/空调用分支）

D-07 default-deny 重试 ──┬─> D-09 协作式取消（超时是重试的触发条件之一）
                         └─> D-22 execute_many 永不抛（契约一致）
D-09 协作式取消 ──> D-21 同步工具超时不重试 + cancel_scope 每 attempt 一次 + threading 限流
D-20 per-loop 线程池 ──> D-09（池的关闭责任终于落地）

D-04 确定性向量 ──> D-08 混合检索（需要可复现向量才能测出排序正确性）
D-06 唯一 token 估算 ──┬─> D-08（记忆截断）
                       ├─> D-15 上下文预算反推
                       └─> D-16（无进展检测的摘要长度）

D-08 检索 ──> D-23 持久化（save/load 的往返测试要断言 search 顺序一致）

D-12 threading 并发模型 ──┬─> D-21 跨 loop 限流一律用 threading
                          └─> D-17 审批回调可能在工作线程里被调用

D-16 循环防护 ──┬─> D-24 错误分类第二层（infrastructure 计数进熔断）
                └─> D-18 token/成本预算（同属"止损"）

D-27 事件归属矩阵 ──> D-14 / D-15（新事件 BUDGET_EXCEEDED / CONTEXT_TRUNCATED / TOOL_APPROVAL）
D-25 run_sync 工厂签名 ──> 所有同步包装 API
D-26 注入点 ──> D-07（退避可断言）/ D-08（now 可注入）/ D-16（墙钟可测）

D-05 Optional 不生成 null ─> （独立，无下游依赖）
D-19 Annotated 探测 ──────> D-05（required 判定的第二个输入）
```

**如果面试官只问一个问题**，就讲 **D-01 + D-03 + D-12 + D-21** 这条线：
「零依赖」不是一个口号，它具体地决定了数据结构选型、注册表形态、并发模型，
**以及当"跨线程/跨 loop"遇上"loop-bound 原语"时你会踩到什么坑**。
这四条串起来能讲 8 分钟，而且每一条都有我实测过的证据（不是背下来的最佳实践）。

**如果面试官问"你做过的取舍里最不显然的一个"**，讲 **D-13 + D-14**：
「我不追 LangChain 的抽象，而是把控制流显式写出来 —— 直接的结果是
`finish_reason` 这种东西会自然进入我的控制流，而在一个黑盒 AgentExecutor 里，
它很容易就只被当成一个日志字段。」

---

*本文档描述的是**决策与理由**；接口细节以 `INTERFACES.md` 为准。*
