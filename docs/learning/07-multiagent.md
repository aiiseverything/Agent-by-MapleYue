# 第 7 章 · 多 Agent 协作

> **本章目标**：理解简历里「Sequential、Hierarchical 两种多 Agent 协作模式」。
> 学完你能：① 说清两种模式各自的适用场景；② 看懂"委派即工具"这个关键设计；
> ③ 知道多 Agent 的防护机制（环检测、深度、预算）。

---

## 7.1 先说结论：多 Agent 不是银弹

在学具体实现之前，先建立一个判断力：

**多 Agent 的两个真实好处**：

| 好处 | 说明 |
| --- | --- |
| **上下文隔离** | 子 Agent 有自己独立的对话历史，不会把主 Agent 的上下文撑爆 |
| **职责分离** | 每个 Agent 有专属的 system prompt 和工具集，行为更可控 |

**三个真实代价**：

| 代价 | 说明 |
| --- | --- |
| **通信开销** | Agent 之间传递的信息要序列化成文本，有损 |
| **错误传播** | 上游 Agent 理解错了，下游全错 |
| **调试难度指数上升** | 单 Agent 的 trace 你已经要读半天了，多 Agent 是树状的 |

> **实践原则**：能用"单 Agent + 几个好工具"解决的，不要上多 Agent。
> 多 Agent 适合的是**任务边界需要动态判断**、或**子任务确实需要不同角色设定**的场景。

---

## 7.2 Sequential：顺序流水线

### 7.2.1 结构

```
用户输入 ──▶ [Agent A] ──▶ [Agent B] ──▶ [Agent C] ──▶ 最终输出
                │             │             │
                └─────────────┴─────────────┘
                      共享黑板（Blackboard）
```

适合**步骤固定**的流程，比如"研究员 → 写作者 → 审校者"。

### 7.2.2 跑一遍

```bash
python3 examples/05_multiagent_sequential.py --offline
```

真实输出（节选）：

```
[2/6] 最简流水线：不写任何 input_template
  status               FINISHED
  steps (总步数)          4
  output               审校通过：结论前置、无事实错误，建议补一句 latency 数据。

  每个阶段**真正收到**的输入（订阅 RUN_STARTED 事件拿到的）：
    researcher  <- 介绍一下 liteagent 这个项目
    writer      <- 研究结论：liteagent 是零依赖的 Agent Harness，核心是...
    reviewer    <- 草稿 v1：liteagent 把 LLM、工具、记忆和 ReAct 循环粘在一起...

  委派 / 返回事件配对：
    [delegate] step=0 content_pipeline -> researcher depth=1 refused=False
    [return  ] step=0 content_pipeline -> researcher status=FINISHED steps=2
    [delegate] step=1 content_pipeline -> writer depth=1 refused=False
    ...
```

**注意"每个阶段真正收到的输入"这一栏**——这是理解 Sequential 的关键：
**阶段之间不共享对话历史，只通过"输入文本"传递信息**。

### 7.2.3 核心 API：`input_template`

```python
from liteagent import SequentialAgent, SequentialStep

team = SequentialAgent([
    SequentialStep(agent=researcher, name="researcher",
                   input_template="请围绕这个主题查资料：{input}"),
    SequentialStep(agent=writer, name="writer",
                   input_template="原始需求：{input}\n研究员给的要点：{steps[researcher]}\n请写草稿。"),
    SequentialStep(agent=reviewer, name="reviewer",
                   input_template="原始需求：{input}\n待审草稿：{prev}\n请审校。"),
])
```

三个占位符：

| 占位符 | 含义 |
| --- | --- |
| `{input}` | 最初传给整个团队的输入 |
| `{prev}` | 上一个阶段的输出 |
| `{steps[阶段名]}` | **任意**已完成阶段的输出 |

**不写模板时的默认规则**：第 0 步用 `{input}`，其余用 `{prev}`。
所以最简单的三步流水线**一个字模板都不用写**（上面的输出里就是这么跑的）。

一个贴心的容错：**引用一个还不存在的阶段名不会崩**——
`render_template` 遇到缺失的 key 会**保留 `{key}` 字面量**。
于是你能在输出里**看到**"这里没填上"，而不是让整个流程抛 KeyError。

### 7.2.4 失败处理：`propagate_failure` 三种策略

某个阶段失败了怎么办？

| 取值 | 行为 | 适用 |
| --- | --- | --- |
| `return`（默认） | **立即停止**，返回失败结果（带 `metadata["failed_stage"]`） | 后续依赖前一步，失败就没必要继续 |
| `raise` | 抛 `DelegationError`（带 from_agent / to_agent / context） | 交给调用方处理 |
| `continue` | 把失败信息当文本继续下一步 | 后面的步骤能容忍缺失 |

真实输出（示例故意让 writer 挂掉）：

```
--- propagate_failure = 'return' ---
  status               FAILED
  output               研究结论：liteagent 是零依赖的 Agent Harness，核心是 LLM 抽象层...
  error                LLMError: writer model is down (示例里故意制造的故障)
  metadata['failed_stage'] writer

--- propagate_failure = 'raise' ---
  抛出的异常            DelegationError: stage 'writer' failed with status FAILED: ...
  exc.from_agent -> to_agent  'content_pipeline' -> 'writer'
```

注意 `return` 策略下**仍然返回了部分输出**——
公式是"失败阶段的部分输出 > 上一步输出 > 空串"，尽量不丢已经做出来的工作。

还有一个 `optional=True` 的开关：标了它的步骤**无视** `propagate_failure`，
一律走 continue 语义。

---

## 7.3 Hierarchical：层级编排（主 Agent + 子 Agent）

### 7.3.1 关键设计：委派即工具

这一节是本章最重要的内容。

**朴素做法**：写一个"编排器"类，里面有 `if 需要委派: 调用子 agent` 的逻辑。

**本项目的做法**：**把每个子 Agent 包装成一个工具**。

```
manager 是**一个普通的 ReAct Agent**
它的工具箱里除了普通工具，还有：
    delegate_to_researcher(task, extra_context)   ← 自动生成
    delegate_to_coder(task, extra_context)        ← 自动生成
    delegate_to_reviewer(task, extra_context)     ← 自动生成
```

于是"什么时候委派、委派给谁、任务怎么描述"**全部由 manager 通过正常的 ReAct 循环决定**——
**编排层不需要自己的状态机**。

> **这是本章最值得讲的设计**。它把"多 Agent 编排"这个看起来需要新范式的需求，
> **归约成了已有的"工具调用"**。带来的好处：
> - 委派决策直接受益于 ReAct 循环的所有能力（多轮、自纠错、循环防护）；
> - 开发者不需要学第二套 API；
> - 委派的失败可以像工具失败一样回灌，而不是让整个流程崩掉。
>
> 面试可以这么讲：我没有为多 Agent 写新的控制流，而是把子 Agent **降维成工具**，
> 复用了单 Agent 的全部机制。这是"用已有抽象解决问题"的一个例子。

### 7.3.2 delegate 工具的 schema（逐字来自代码）

```json
{
  "type": "object",
  "properties": {
    "task": {"type": "string", "description": "The subtask to delegate, written as a self-contained instruction."},
    "extra_context": {"type": "string", "description": "Optional extra context ... It is appended to the task text and is NOT interpreted by the framework."}
  },
  "required": ["task"],
  "additionalProperties": false
}
```

工具名是 `delegate_to_<worker名字>`（非法字符会被规整；两个 worker 规整后同名会**报错**，
而不是让其中一个静默不可达）。

几个刻意的属性设置：

| 属性 | 值 | 为什么 |
| --- | --- | --- |
| `timeout_s` | `NO_TIMEOUT`（-1.0） | 子 Agent 可能要跑很久，30 秒的默认超时对它是错的 |
| `idempotent` | `False` | 委派有副作用，默认不重试 |
| `tags` | `("delegation",)` | 便于区分和过滤 |

### 7.3.3 跑一遍

```bash
python3 examples/06_multiagent_hierarchical.py --offline
```

真实输出（节选）：

```
[3/6] 真跑一次：manager 分解 + 委派 + 综合
  status                 FINISHED
  steps                  3
  final answer           综合结论：researcher 给了需求要点，coder 给了补丁，可以进 review。

  worker 真正收到的输入（= task + extra_context 拼接）：
  worker [researcher]    查一下 liteagent CLI 的现状\n\n已知：CLI 是 argparse 子命令风格
  worker [coder]         给 liteagent.cli 写 handle 入口

  subagent:researcher:1  v1 by 'researcher' -> [worker researcher | status=FINISHED | steps=2]
                                                  需求要点：离线可...
  subagent:coder:2       v1 by 'coder' -> [worker coder | status=FINISHED | steps=2]
                                                  补丁：给 cli 加了 h...
```

注意 worker 收到的输入是 **`task` + `extra_context` 拼接**——
`extra_context` 只是**给 worker 看的文本**。

### 7.3.4 `extra_context` 的一个安全设计（很值得讲）

代码注释里记录了一个**真实的设计修正**：

> 这个参数以前叫 `context`，与内部的 contextvar 同名。
> 实现者几乎必然会写成 `ctx = args.get("context") or _CURRENT_CTX.get()`，
> 于是**环检测和深度判定被模型传进来的任意文本顶掉了** → 无限委派。

所以现在它叫 `extra_context`，并且明确规定：

> **它绝不参与环检测/深度判定**，只是拼进 worker 的任务描述里。

**这是一个很好的教训**：**安全相关的状态绝不能和模型可控的输入共用一个名字或通道**。
命名在这里不是风格问题，是安全问题。

### 7.3.5 子 Agent 输出怎么回传：压缩后回灌

子 Agent 可能输出几千字，全塞回 manager 的上下文会把它撑爆。所以有一个**冻结的压缩格式**：

```
[worker researcher | status=FINISHED | steps=2]
<正文，超长时头尾保留地截断>
```

细节：

- 默认**保留头 70% + 尾 30%**（结论常在末尾）；
- header 本身**不计入**字符上限；
- 失败时 header 追加 `| error=类型: 消息`，**正文仍然截断**（"为什么失败"常在末尾）；
- **原始输出不丢**：完整原文写进黑板（`subagent:名字:序号`）和子 Agent 的 `scratchpad`。

实测压缩效果：`原始 212 字符 → 回灌 113 字符`。

### 7.3.6 三重防护

多 Agent 最大的风险是**无限委派**。本项目有三重防护：

**① 环检测**

```python
def would_cycle(name: str) -> bool:
    """委派栈里已经有这个 Agent 了吗？（实现只有这一行）"""
    return name in self.stack
```

比如 A 委派 B、B 又委派回 A。触发时的行为是一个**刻意的设计选择**：

```
返回字符串："[delegation refused: cycle detected: stack=[pipeline,subteam] -> pipeline]"
```

**返回字符串，而不是抛异常。** 代码注释解释了原因：

> 抛异常会让一次"模型选错了同事"升级成整轮 run 失败。

**这个区别很重要**：模型选错同事是一个**正常的、可恢复的**错误——
它应该像"工具调用失败"一样被回灌，让模型自己换个策略。
把它升级成整个 run 失败，是把小问题放大成大问题。

**② 深度上限**

`TeamConfig(max_depth=3)`：委派链的最大深度。

**③ 预算**

`TeamConfig(max_rounds=10)`：还能委派几轮。

三者触发时的返回文案（实测）：

```
budget 耗尽   → [delegation refused: budget exhausted]
depth/环      → [delegation refused: cycle detected: stack=[probe] -> researcher]
并发槽满      → [delegation refused: all N subagent slots are busy]
```

真实输出里的环检测演示：

```
[5/6] 环检测：A 委派 B、B 又要委派回 A
      [delegation refused: cycle detected: stack=[pipeline,subteam] -> pipeline]
  外层 status              FINISHED          ← 注意：外层没有失败！
  外层 output              子团队结论：上游被拒（环），我们本地处理完了。
  back-edge 被执行次数        0
```

**注意"外层 status 是 FINISHED"**——环被挡住了，业务逻辑自己绕过去了。
这就是"拒绝但不崩溃"的价值。

### 7.3.7 显式计划：`arun_plan`

除了"让 manager 自己决定怎么委派"，你也可以**显式给一个计划**：

```python
plan = Plan(goal="给 CLI 加功能", subtasks=[
    SubTask(id="t1", description="查需求", assignee="researcher"),
    SubTask(id="t2", description="写补丁", assignee="coder", depends_on=("t1",)),
    SubTask(id="t3", description="审补丁", assignee="reviewer", depends_on=("t2",)),
])
result = await team.arun_plan(plan)
```

它按 `depends_on` **拓扑分层**执行：

```
t1  ──▶  t2  ──▶  t3        层与层之间串行
                            同一层内可以并发（parallel_subagents=True）
```

真实输出：

```
subtask [t1] researcher status=done -> [worker researcher | status=FINISHED | steps=2]...
subtask [t2] coder     status=done -> [worker coder | status=FINISHED | steps=2]...
subtask [t3] reviewer  status=done -> [worker reviewer | status=FINISHED | steps=2]...
```

三条保证：

- **单个 subtask 失败不中断整体**（压成字符串、标 `status="failed"`，后续层照跑）；
- 计划里成环（互相依赖）→ 抛 `CycleDetectedError`；
- 依赖指向不存在的 id → 抛 `DelegationError`。

> **注意**：`arun_plan` 跑完就是 `FINISHED`（失败在 subtask 层体现），
> 这个语义要记住，否则会误判。

---

## 7.4 共享黑板 `Blackboard`

### 7.4.1 是什么

Agent 之间传递信息的**共享存储**：

```python
from liteagent import Blackboard

bb = Blackboard(max_entries=1000)
bb.write("draft", "草稿内容...", author="writer", tags=("stage",))
bb.read("draft")                    # '草稿内容...'
bb.read("nope", default="兜底")      # '兜底'
```

### 7.4.2 为什么不用一个普通 dict

因为它要解决**并发**问题：

| 能力 | API | 解决什么 |
| --- | --- | --- |
| **版本冲突检测** | `write(k, v, if_version=N)` | 乐观锁：仅当版本还是 N 才写，否则抛 `VersionConflictError` |
| **原子自增** | `increment(k, amount=1)` | 计数器不需要"读-改-写"，避免竞态 |
| **局部更新** | `update(k, patch)` | 浅合并，不用整个读出来改再写回 |
| **过期** | `write(k, v, ttl_s=60)` | 惰性过期（到期即视为不存在） |
| **历史** | `history(limit=100)` | 看最近 N 次写入，调试用 |
| **订阅** | `subscribe(fn)` | 同步回调，每次写入都通知 |
| **异步监听** | `awatch(key)` | async generator，**合并语义** |
| **并发安全** | 内部 `threading.RLock` | 多线程写不丢更新 |

一个容易忽略的点：`write(k, v, if_version=0)` 表示"**仅当 key 不存在时创建**"
（不存在的 key 版本按 0 计）——这是一个很实用的"创建或失败"语义。

### 7.4.3 `subscribe` 和 `awatch` 的区别

| | `subscribe(fn)` | `awatch(key)` |
| --- | --- | --- |
| 同步/异步 | 同步回调 | async generator |
| 过滤 | 所有 key | 可按 key 过滤 |
| 触发 | **每次写入都回调** | **合并语义**（多次写入可能只 yield 最新一条） |
| 在哪个线程跑 | 写入线程里 | 必须在运行中的 loop 内注册 |

**为什么 `awatch` 是合并语义而不是逐条队列？** 代码注释说明了：
为了**避免慢消费者把内存吃到无界**。如果每次写入都排队，一个卡住的消费者会让队列无限增长。

**想看中间版本怎么办？** 去 `history()` 取。

### 7.4.4 一个特别值得讲的 bug

`awatch` 的实现里有一个坑，是**对抗性审计抓出来的 critical bug**：

> **问题**：早期实现用 `entry.version` 作为"有没有新写入"的判据。
> 但 `version` 在 TTL 过期 / delete / clear / 淘汰之后会**回退到 1**，
> 于是 `version != last` 可能**恒为假** → 写入发生了却一次都不 yield，**watcher 静默失联**。
>
> **修法**：改用**单调递增的写入序号**作为判脏量，而不是版本号。

还有第二个相关的修复：

> `asyncio.Condition.notify_all()` 在 Python 3.10 起**要求调用方持有锁**，
> 实测对未持锁调用会抛 `RuntimeError: cannot notify on un-acquired lock`
> **且唤醒不了任何 watcher**。所以唤醒必须在目标 loop 内、拿到锁之后再调。

**这类 bug 的共同特征**：它们不会让程序崩溃，只会让功能**静默失效**——
而静默失效是最难排查的。详见 `docs/BUILD_LOG.md` 阶段 5。

---

## 7.5 `DelegationContext`：委派上下文

```python
@dataclass
class DelegationContext:
    stack: list[str] = []       # 祖先 Agent 名字（含当前）
    depth: int = 0              # 当前深度
    root_run_id: str = ""
    parent_run_id: str | None = None
    budget: int = 10            # 剩余可委派轮次
```

它承载"我是谁派来的、现在多深、还能派几次"。

### 一个关于 `contextvars` 的重要事实

本项目用 `contextvars` 传递委派上下文（因为 delegate 工具跑在**工作线程**里，
普通的实例属性在那边访问不到）。但有一条**实测结论**必须记住：

> **`contextvars` 的读能穿透线程，写不能回传。**

具体说：子线程/子上下文里 `set()` 的值，**父上下文看不到**。
所以：

- 委派上下文必须**显式携带**（通过 `state.scratchpad["delegation"]` 传给子 Agent），
  而不是指望它"自动回传"；
- executor 把同步工具丢线程池时，必须**显式复制上下文**
  （`copy_context().run(...)`），否则 worker 线程里读到的取消标志恒为 `None`。

> 这是本项目踩过的真实坑之一。**`contextvars` 的传播规则在 Python 里是反直觉的**，
> 写并发代码时务必实测确认。

---

## 7.6 本章小结

1. **多 Agent 不是银弹**——两个好处（上下文隔离、职责分离），三个代价（通信、错误传播、调试难度）。
2. **Hierarchical 的关键设计是"委派即工具"**：子 Agent 被包装成工具，
   编排复用 ReAct 循环，不需要新的状态机。
3. **子 Agent 输出压缩后回灌**（头 70% + 尾 30%），原文不丢（写黑板）。
4. **三重防护**：环检测、深度、预算。触发时**返回字符串而不是抛异常**——
   把"模型选错同事"当正常错误处理。
5. **`extra_context` 绝不参与安全判定**——安全状态不能和模型可控输入共用一个通道。
6. **`awatch` 用单调序号判脏**，因为 `version` 会回退，会导致 watcher 静默失联。

### 自测题

1. 说出多 Agent 的两个好处和两个代价。
2. "委派即工具"是什么意思？这样设计有什么好处？
3. 环检测触发时为什么不抛异常？抛异常会怎样？
4. 子 Agent 的输出为什么要压缩？压缩时保留头还是尾？为什么？
5. `extra_context` 为什么不能参与环检测？它以前叫什么名字，出过什么问题？
6. `awatch` 为什么不能用 `entry.version` 判断有没有新写入？

<details>
<summary>参考答案</summary>

1. 好处：上下文隔离、职责分离；代价：通信开销、错误传播、调试难度上升（任答两个）。
2. 把每个子 Agent 包装成一个 `delegate_to_xxx` 工具，manager 就是普通 ReAct Agent；好处是复用整个 ReAct 循环的机制，不需要为编排再写一套状态机。
3. 因为"模型选错同事"是正常可恢复的错误，应该像工具失败一样回灌让模型换策略；抛异常会把小问题升级成整轮 run 失败。
4. 防止子 Agent 的长输出撑爆主 Agent 上下文；头尾都保留（默认头 70% + 尾 30%），因为结论和错误信息常在末尾。
5. 因为它曾经叫 `context`，与内部 contextvar 同名，实现者会写成 `args.get("context") or _CURRENT_CTX.get()`，导致安全判定被模型传入的文本顶掉，造成无限委派。
6. 因为 `version` 在 TTL 过期 / delete / clear / 淘汰后会回退，`version != last` 可能恒为假，写入发生了却永不 yield，watcher 静默失联；应该用单调递增的写入序号。

</details>

---

**下一章**：[第 8 章 · 工程化](08-engineering.md) —— 异步、并发、可观测性与离线测试。
