# 第 6 章 · 记忆管理：短期、摘要、长期

> **本章目标**：理解简历里「记忆层集成短期对话历史与长期向量存储」。
> 学完你能：① 说清三层记忆各解决什么问题；② 解释混合打分公式；
> ③ 知道什么时候该写长期记忆、什么时候不该。

---

## 6.1 问题：为什么"记忆"不是一件事

回顾第 1 章：**大模型没有记忆，每次调用都是独立的。**
你看到的"它记得我刚才说的话"，全是框架**每轮把历史重新发过去**实现的。

但"把历史全发过去"立刻撞上两个硬限制：

1. **上下文窗口有上限**——发不了无限长的历史。
2. **token 是要花钱的**——每轮重复发一遍历史，成本随对话长度线性上涨。

所以"记忆"实际上是**三个不同的问题**，需要三种不同的机制：

| 问题 | 机制 | 类比 |
| --- | --- | --- |
| 最近说了什么？ | **短期滑动窗口** | 你桌面上摊开的文件 |
| 更早的说了什么？ | **摘要压缩** | 你把旧文件归档成一张便签 |
| 跨会话的长期事实？ | **长期向量库** | 你的笔记本 |

本项目对应三个类：

```
MemoryManager（编排者，唯一入口）
    ├── BufferMemory    短期窗口：有界、可裁剪
    ├── SummaryMemory   摘要压缩：把裁掉的历史压成一段话
    └── VectorMemory    长期向量库：跨 run / 跨进程的事实
```

> **先建立一个关键直觉**：这三层**不是"三级缓存"那种替代关系**，
> 而是**同时生效、各管一段**。每一轮组装 prompt 时，三层的内容会被拼在一起送进去。

---

## 6.2 跑一遍看效果

```bash
python3 examples/04_memory.py --offline
```

真实输出（节选）：

```
第 1 层 / 短期滑动窗口：预算一满就从最旧的那条开始挤出去
==============================================================================
预算            : max_tokens=140, max_messages=8, keep_last_n=2
buffer（全量）  : 17 条消息, 约 276 tokens
  -> 全量是**完整历史**，跨轮累积、只增不减；它不是要送进模型的东西。

window（裁剪后）: 8 条消息, 约 138 tokens
被挤出去(evicted): 9 条消息

窗口内容（这就是模型这一轮真正看到的历史）：
  [system   ] 你是一个严谨的 Python 助手，回答要给出可执行的代码。
  [assistant] 那就用 pathlib.Path('data').glob('*.csv') 遍历。
  ...
```

**请特别注意四个术语的区别**（本项目在规范里严格区分了它们）：

| 术语 | 含义 |
| --- | --- |
| **buffer** | 跨轮累积的完整历史（只增不减） |
| **window** | 实际送进模型的那批（裁剪后） |
| **evicted** | 在 buffer 里但不在 window 里、等待摘要的消息 |
| **transcript** | 一次 run 内的全部消息（Agent 层概念） |

新手最容易把 buffer 和 window 混为一谈。**buffer 是"我们记得的"，window 是"这次告诉模型的"。**

---

## 6.3 第一层：短期滑动窗口

### 6.3.1 双约束裁剪

```python
BufferConfig(
    max_tokens=3000,      # token 预算
    max_messages=50,      # 条数上限
    keep_last_n=2,        # 无论如何至少保留最近 2 条
    keep_system=True,     # 开头的 SYSTEM 消息钉住不动
)
```

裁剪算法（从后往前累加）：

```
1. 先把开头连续的 SYSTEM 消息取出来「钉住」（不参与裁剪）
2. 剩下的从最新往旧累加：
     - 如果已达 max_messages 条 → 停（条数上限是无条件的）
     - 如果保留条数 >= keep_last_n 且再加一条就超 token 预算 → 停
     - 否则继续加
3. 拼回去：钉住的 SYSTEM + 保留下来的
4. 修工具对（见下）
```

测量一下效果（示例输出）：

```
buffer 全量：17 条 / 276 tokens   →    window：8 条 / 138 tokens
```

### 6.3.2 `_repair_tool_pairs`：为什么必须修

这是一个**非常容易踩的坑**，值得单独讲。

裁剪是按"条数/token"从旧到新砍的，它**不知道消息之间的语义配对**。
于是可能出现这种情况：一条 assistant 消息带着 `tool_calls`，但对应的工具结果已经被裁掉了。

**这时候如果直接发给 API，OpenAI / Anthropic 会直接返回 400 错误**——
因为协议要求：assistant 说"我要调用工具 X"，就必须有对应的工具结果。

本项目的修法是**两道闸**：

| 闸 | 粒度 | 做什么 |
| --- | --- | --- |
| `drop_orphan_tool_messages` | 粗 | 删掉孤立的 tool 消息；剥掉 assistant 上没有结果的那部分 `tool_calls`（但**保留 assistant 消息本体**） |
| `_repair_tool_pairs` | 细 | 专治"assistant 的半截工具对"——把整条 assistant 消息**整个丢弃** |

**为什么宁可丢掉整条 assistant 消息？** 代码注释写得很直白：

> 宁可整条 assistant 丢掉，也不要把残缺的工具对送进 API。

丢一条消息影响不大（模型少看到一句话），但发一个非法请求会让整个 run 直接失败。

### 6.3.3 为什么裁剪推迟到 `window()` 才做

一个性能设计：`add()` 每轮都会被调用（每次观察回灌后），所以它必须是 **O(1)**——
只 append，不裁剪。真正的裁剪发生在 `window()` 被调用时（每轮组装 prompt 时一次）。

> 这是一个通用的性能原则：**把昂贵的计算推迟到真正需要结果的时候。**

---

## 6.4 第二层：摘要压缩

### 6.4.1 什么时候压缩

```python
SummaryConfig(
    enabled=True,
    trigger_ratio=0.8,      # 窗口用量 >= 预算的 80%
    min_evict_batch=4,      # 或者积压了 4 条待压缩消息
    max_summary_chars=1200,
)
```

**两个条件是 OR 关系**，任一成立就压缩：

- 条件 1（`trigger_ratio`）：**"再不压下一次就超上下文了"**
- 条件 2（`min_evict_batch`）：**"积压够多了，再攒只能丢掉"**

真实效果（示例输出）：

```
当前 buffer=25 条, window=9 条, 待压缩 evicted=16 条
压缩产物（摘要文本）：
  - 目标：写一个零依赖的 CSV 处理脚本
  - 约束：不引三方库；文件可能很大，必须流式读取；编码可能不是 utf-8
  - 已定方案：csv 模块 + 显式 encoding + 手写进度输出

compression_count = 1
再调一次 acompress_if_needed() -> None；compression_count 仍为 1
  -> 没有待压缩批次时直接返回 None：计数不会被空转调用推高。
```

### 6.4.2 最重要的设计：**摘要失败不能让整个 Agent 挂掉**

摘要是"锦上添花"的功能——它失败时，最坏情况只是上下文短一点。
所以本项目规定：**`acompress` 永不抛异常**。

三条降级路径，每条都会记 warning 并走**抽取式兜底**：

| 情况 | 行为 |
| --- | --- |
| 没有配 LLM | warning + 抽取式摘要 |
| LLM 调用抛异常 | warning + 抽取式摘要 |
| LLM 返回空内容 | warning + 抽取式摘要 |

**抽取式兜底**是什么？就是完全不用 LLM，直接把每条消息截断拼接：

```
[user] 我想写一个 CSV 脚本...
[assistant] 那就用 csv 模块...
(16 messages summarized)
```

它很粗糙，但**零依赖、零成本、永不失败**——保住了"至少有个摘要"这个底线。

> **面试可以这么讲**：我给摘要设计了降级路径——LLM 不可用时退化成零 LLM 的
> 抽取式摘要。理由是"摘要失败"绝不该让整个 Agent 挂掉，而"没有任何摘要"和
> "粗糙的摘要"之间，后者明显更好。**这类"降级设计"是可靠性工程的常态。**

### 6.4.3 滚动更新

`update_existing=True`（默认）时，压缩会**把上一次的摘要也一起喂给 LLM**，
prompt 里写着：

```
Existing summary so far (merge, do not repeat):
{previous}
```

这样摘要是"滚雪球"式的——新信息被合并进来，而不是每次重写（重写会丢掉更早的要点）。

---

## 6.5 第三层：长期向量记忆

前面两层管的都是"这次对话"，长期记忆管的是**跨会话的事实**：

> "我偏好用 uv 管理 Python 项目的依赖，不喜欢 pip。"

这种事实应该记住**一周后**的对话里还能用上。

### 6.5.1 写入策略：不是所有消息都值得记

```python
MemoryConfig(write_policy="selective")     # 默认
```

| 策略 | 行为 |
| --- | --- |
| `manual` | 只有显式调用 `remember()` 才写 |
| `turn` | 每一轮用户消息都写（激进，噪声大） |
| `selective`（默认） | 满足条件才写 |

`selective` 的判定条件：

```
role == "user"                     ← 第一道硬性约束
  且（内容长度 >= 40 字符  或  命中关键词）
  且 内容"有点东西"（不是 "。。。" 这种）
```

关键词表是冻结的（中英双语）：

```python
("记住", "请记", "记下", "我是", "我的名字", "我偏好", "我喜欢", "以后", "下次",
 "remember", "my name is", "i prefer", "i like", "note that", "keep in mind",
 "always use", "never use", "don't forget")
```

注意英文用的是**短语**而不是单词：用 `my name is` 而不是 `name`，
否则 "the name of the file" 也会误命中。

> **为什么"误报比漏报更贵"？** 误报会把一句无关的话写进长期库，
> 之后**每一轮**都会被检索出来塞进 prompt，持续污染上下文。而漏报的代价只是
> "这次没记住"。所以宁可保守。

### 6.5.2 最重要的一条约束：`role != "user"` 永不写入

**无论哪种策略，AI 自己说的话永远不会被写进长期记忆。** 代码注释解释了原因：

> assistant 的答案**不该**被当成"用户事实"写进长期库。否则模型自己编的一句话
> 会被下一轮当成用户说过的事实召回，形成**"自我确认循环"**——
> 幻觉被写进长期记忆后，连纠正的机会都没有了。

**这是一个非常深刻的设计**：它防的不是"记错"，而是**"错误被固化"**。
一旦模型的一句幻觉变成了"长期事实"，后面所有对话都会基于它推理，错误会滚雪球。

同理，第 5 章提到的 **nudge 消息虽然是 `user` 角色，但因为是框架生成的，也被拒绝写入**。

> 面试可以这么讲：长期记忆的写入路径我做了 role 约束 + 框架消息豁免两层过滤，
> 目的是切断"模型幻觉 → 写进长期库 → 后续每轮都被当成事实召回"这个自我确认循环。
> 记忆系统最大的风险不是记不住，而是**记住错的**。

### 6.5.3 检索：混合打分（面试核心）

检索不是"找最相似的"，而是**相似度 + 近因 + 重要度**三者加权：

```
score = w_sim × sim  +  w_recency × recency  +  w_importance × importance

其中：
  sim        = 余弦相似度(query向量, 记忆向量)          范围 [-1, 1]
  recency    = 2 ** (-age_days / half_life_days)        范围 (0, 1]
  importance = clamp(记忆的 importance 字段, 0, 1)

默认权重：w_sim=1.0, w_recency=0.15, w_importance=0.1
        half_life_days=7.0（半衰期 7 天）
```

看真实输出（示例）：

```
检索 query（注意：**换了措辞**）：'Python 项目的依赖该用什么工具管理比较好？'

命中 3 条（已按 score 降序、并经 MMR 去冗）：
  #1 score=0.6528  breakdown={'sim': +0.4128, 'recency': 1.0000, 'importance': 0.90}
      我偏好用 uv 管理 Python 项目的依赖，不喜欢 pip。
  #2 score=0.3841  breakdown={'sim': +0.1641, 'recency': 1.0000, 'importance': 0.70}
      我的项目统一用 ruff 做 lint 和格式化。
```

**为什么要加近因和重要度，不能只看相似度？**

| 只看相似度的问题 | 加权解决的 |
| --- | --- |
| 三个月前的一条高相似记忆，和昨天的一条中等相似记忆，应该选谁？ | 近因权重让"新的"占优 |
| "我叫小林"这种身份信息和一句闲聊，相似度可能差不多 | 重要度让你能手动抬高某些记忆的优先级 |

而且 `score_breakdown` 把三个分量都暴露出来，**调试检索质量时可以直接看是哪一项在起作用**。

### 6.5.4 两个必须知道的实现细节

**细节 1：负 age 会被钳制（一个真实 bug 的遗迹）**

近因公式在 `now < created_at`（时钟回拨、或从磁盘恢复出未来时间戳的数据）时，
指数会变成正数：轻则 `recency > 1` 违反值域，重则 `2 ** 大正数` **抛 OverflowError**。

本项目的修法是在打分侧把 `age_days` **钳到 >= 0**（"来自未来"等价于"刚写入"）。

> 这个 bug 是被测试抓出来的，而且根因**不在代码里，在规范里**——
> 规范只定义了 `age >= 0` 时的公式。详见 `docs/BUILD_LOG.md` 阶段 4。

**细节 2：`search` 不是只读操作**

检索命中后会把 `access_count += 1`、刷新 `last_access_at`。
所以它在锁内完成（不能和写操作并发）。**这个副作用是有意的**——
访问频次本身就是"这条记忆有用"的信号。

### 6.5.5 MMR：去冗余

假设你检索"Python 依赖管理"，返回了 5 条**几乎一模一样**的记忆：

```
我偏好用 uv 管理依赖
我用 uv 管理依赖
uv 是我的依赖管理工具
...
```

**这等于只检索到 1 条，还白白花掉 5 条的上下文预算。**

MMR（Maximal Marginal Relevance）的思路是"既要相关，又要彼此不同"：

```
mmr_i = λ × score_i  −  (1−λ) × max{ 与已选中的任意一条的相似度 }
         ↑ 相关性            ↑ 冗余惩罚

默认 λ = 0.7
```

每次挑 `mmr` 最大的那条，直到选满。λ=1 就退化成"只按分数取前 k"，
λ=0 就退化成"只挑最不像的"。

---

## 6.6 embedding：相似度是怎么算出来的

### 6.6.1 默认用的是"哈希 embedding"，不是语义 embedding

这是一个**必须讲清楚的边界**。本项目的默认 embedder 是纯 stdlib 的 `HashingEmbedder`：

```
文本 → 切 token → 每个 token 哈希到固定维度的一个下标 → 加权 → L2 归一化
```

它的**能力边界**（代码注释原文）：

> 这是词面哈希，不是语义 embedding。**'你好' 与 'hi' 的相似度是 0。**

也就是说：它只能捕捉**字面重合**，同义词、跨语言完全无效。

**为什么还用它的？** 因为：
1. 它是**确定性的**（同样的文本永远得到同样的向量）；
2. **零依赖、离线可用**——这让整套记忆系统可以在没有网络、没有 API key 的环境里测试；
3. 生产环境可以直接换成 `RemoteEmbedder`（调真实的 embedding 服务），接口完全一样。

**一个必须避免的坑**：不能用 Python 内置的 `hash()`——
它受 `PYTHONHASHSEED` 影响，**同一个文本在不同进程里会得到不同的向量**，
持久化后读回来就检索不到了。必须用 `hashlib`。

### 6.6.2 可选的实现

| 类 | 说明 |
| --- | --- |
| `HashingEmbedder` | 纯 stdlib，默认 |
| `NumpyHashingEmbedder` | 数学等价但用 numpy 加速（环境有 numpy 时可用） |
| `RandomProjectionEmbedder` | 可复现的伪随机（测试用） |
| `CallableEmbedder` | 包装任意函数（测试注入确定向量最方便） |
| `RemoteEmbedder` | 调 OpenAI 兼容的 `/embeddings` 接口 |

---

## 6.7 `MemoryManager`：三层怎么拼在一起

### 6.7.1 组装出的 prompt 有 6 段（按顺序）

```
1. system 消息（纯系统提示）
2. <conversation_summary>  ... </conversation_summary>      ← 摘要段（有才拼）
3. <relevant_memories>     ... </relevant_memories>         ← 长期记忆段（有才拼）
4. buffer 的 window（裁剪后的历史对话）
5. extra（额外注入的消息，多 Agent 场景用）
6. 本轮的 user 输入                                          ← 只在第 1 步追加
```

真实效果（`<relevant_memories>` 段的原文）：

```
<relevant_memories>
- (score=0.65, 2026-09-30) 我偏好用 uv 管理 Python 项目的依赖，不喜欢 pip。
- (score=0.38, 2026-09-30) 我的项目统一用 ruff 做 lint 和格式化。
- (score=0.27, 2026-09-30) 我叫小林，是一名后端工程师。
</relevant_memories>
```

**为什么用 XML 风格的标签？** 因为模型对结构化标签的边界感知很强——
`<relevant_memories>` 明确告诉它"这一块是背景事实，不是用户当前说的话"。
这比在正文里混着写要可靠得多。

### 6.7.2 上下文预算的强制兜底

即使三层都做了裁剪，组装完还是可能超预算。所以最后有一道 `_enforce_context_budget()`：

```
超预算时按「长期记忆块 → 窗口里最旧的消息」的顺序裁剪
但绝不动：钉住的 system 消息、最后一条 user 消息
并发出 context_truncated 事件（含 before/after/dropped 数）
```

**"不动最后一条 user 消息"** 很关键——那是用户当前的问题，丢了它整个回答就跑偏了。

### 6.7.3 怎么用

```python
from liteagent.memory import MemoryManager, MemoryConfig, HashingEmbedder

manager = MemoryManager.from_config(
    MemoryConfig(buffer_max_tokens=3000, long_term_enabled=True),
    llm=my_llm,                       # 可选：用于摘要；不传就走抽取式兜底
    embedder=HashingEmbedder(dim=256) # 可选：不传用默认
)

await manager.aadd(message)                 # 写入（自动决定是否进长期库）
await manager.aremember("我叫小林", importance=0.8)   # 显式写入长期库
hits = manager.retrieve("我的名字")          # 检索长期库
prompt = await manager.abuild_prompt(system="...", user_input="...")   # 组装
print(manager.stats())                      # 12 个统计字段
```

`stats()` 返回的 12 个键很适合做监控：

```
short_term_messages, short_term_tokens, window_messages, window_tokens,
evicted_pending, long_term_items, has_summary, summary_tokens,
compressions, embedder, buffer_budget_tokens, context_window_tokens
```

### 6.7.4 持久化

```python
manager.persist("memory.jsonl")     # 存到磁盘（JSONL 格式）
manager.restore("memory.jsonl")     # 读回来
```

实测保证：往返后 `len` 与 `search` 顺序一致、embedding 逐元素相等。
坏行会被跳过并记 warning（不是整个文件失败）。

**一个诚实的边界**：本项目验证的是**同进程往返**，
没有验证"进程 A 写、进程 B 读"的跨进程场景。详见 `docs/VERIFICATION.md`。

---

## 6.8 本章小结

1. **记忆是三个问题**：最近（窗口）、更早（摘要）、长期（向量库），**同时生效**。
2. **双约束裁剪**（条数 + token），且必须**修工具对**，否则 API 会 400。
3. **摘要永不抛异常**，LLM 失败时退化成零依赖的抽取式摘要。
4. **`role != "user"` 永不写长期记忆**——切断"幻觉 → 记忆 → 自我确认"的循环。
5. **检索是混合打分**：相似度 + 近因 + 重要度，再用 MMR 去冗余。
6. **默认 embedder 是词面哈希，不是语义**——这个边界必须知道并如实说明。

### 自测题

1. buffer 和 window 的区别是什么？为什么要分开？
2. 裁剪后为什么必须修"工具对"？不修会怎样？
3. 摘要压缩失败时会发生什么？为什么这样设计？
4. 为什么 AI 自己的回答永远不写进长期记忆？
5. 检索为什么不能只看余弦相似度？近因权重解决什么问题？
6. MMR 解决什么问题？用一个具体例子说明。

<details>
<summary>参考答案</summary>

1. buffer 是跨轮累积的完整历史（只增不减），window 是裁剪后实际送进模型的那批；分开是为了让写入 O(1)、把昂贵裁剪推迟到真正需要时。
2. 因为 assistant 的 `tool_calls` 必须有对应的结果，否则 OpenAI/Anthropic 直接返回 400；宁可丢掉整条 assistant 消息也不发非法请求。
3. 记 warning 并退化成抽取式摘要（直接截断拼接历史），永不抛异常；因为摘要是锦上添花，失败不该让整个 Agent 挂掉。
4. 否则模型的一句幻觉会被下一轮当成"用户事实"召回，形成自我确认循环，错误被固化后无法纠正。
5. 因为"三个月前的高相似记忆"和"昨天的中等相似记忆"该选谁，只看相似度答不了；近因权重让新记忆占优。
6. 检索结果冗余——5 条几乎一样的记忆等于只检索到 1 条却花掉 5 份上下文预算；MMR 在"相关"和"彼此不同"之间做权衡。

</details>

---

**下一章**：[第 7 章 · 多 Agent 协作](07-multiagent.md)
