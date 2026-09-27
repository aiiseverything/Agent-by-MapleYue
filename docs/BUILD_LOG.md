# 搭建过程记录（Build Log）

> 这份文档记录 liteagent 从零搭建的**真实过程**：每一步做了什么、为什么这么做、踩了什么坑、
> 以及在面试里可以怎么讲。按时间顺序写成一份连贯的工程叙事。
>
> **关于本文件的数字**：`docs/BUILD_LOG.md` 是分阶段增量写成的，早期快照出现过 1574 / 1614
> 这类数字，与后来的仓库对不上。**本轮已把全文数字重跑了一遍并统一刷新**（2026-09-27），
> 现在文中的每一个数字都是本轮实测的产物，复现命令见 §7.3。
> 凡属于「历史时点」的数字，一律显式标注为历史，不与当前值混用 —— 这条纪律本身就是
> 这个项目学到的东西之一（§5.5 与 §8）。

---

## 阶段 0 · 项目缘起

简历条目：

> **Agent：轻量级 Agent Harness 框架设计与实现**（个人项目）
> 从零构建可扩展的 LLM Agent 开发框架，实现 ReAct 循环与工具系统
> - 核心架构设计：参考 LangChain 和 Nanobot 设计模式，实现 LLM 抽象层、工具系统、记忆管理三层架构……
> - ReAct 循环实现：完整的 Thought-Action-Observation 循环……并发工具调用。
> - 多 Agent 协作与扩展：Sequential、Hierarchical 两种协作模式……验证框架在代码助手场景的可用性。

本项目就是把这三点**真的做出来**：可运行的代码 + 可复现的测试 + 可讲的文档。
整个搭建过程分六个阶段，本文件按顺序讲下来。

---

## 阶段 1 · 环境侦察（第一步，也是最关键的一步）

在写第一行代码之前先摸清环境，因为它直接决定了技术选型。本轮我把当时的侦察重跑了一遍，
结果如下（**下表是本轮实测**，不是抄的）：

| 能力 | 状态 | 实测值 / 说明 |
| --- | --- | --- |
| 网络（pip / HTTPS 出网） | ❌ 不通 | `pip install pytest` 超时 |
| `pytest` / `pytest-asyncio` | ❌ 未安装 | `import pytest` -> ModuleNotFoundError |
| `openai` / `anthropic` SDK | ❌ 未安装 | 两个都 import 不到 |
| `tiktoken` / `faiss` / `python-dotenv` | ❌ 未安装 | 三个都 import 不到 |
| `pydantic`、`numpy` | ✅ 有 | 2.13.5 / 1.26.4 |
| `requests` / `httpx` | ✅ **有** | 2.34.2 / 0.28.1（这一点早期文档写错过，见 §5.5 第 ⑥ 条） |
| `PyYAML` / `jinja2` / `rich` / `typer` | ✅ 有 | 全部可 import |
| 标准库（`asyncio` / `sqlite3` / `urllib` / `unittest`） | ✅ 有 | Python 3.10.12 |

**这个约束反而成了项目最好的设计输入**，直接推出三条硬性架构决策：

1. **框架内核零第三方依赖（pure stdlib）**。不是为了炫技，而是因为没网络就装不上包；
   而一旦内核零依赖，它同时获得「秒级启动」「不会被上游 breaking change 打断」
   「可以在任何 Python 环境里跑」三个真实收益。落成红线后由
   `tests/test_zero_dependency.py` 用 AST 自动看守。
2. **测试用标准库 `unittest`，不用 `pytest`**。于是测试必须靠**脚本化假 LLM（ScriptedLLM）**
   驱动，而不是靠 mock 框架的魔法。结果：整套测试是**离线、确定性、无 API key 可复现**的。
3. **所有第三方能力都做成可选适配器 + 纯 stdlib 兜底**：`numpy` 缺失 -> 纯 Python 余弦相似度；
   `rich` 缺失 -> 纯文本渲染；`requests` 缺失 -> `urllib.request`；真实 embedding 缺失 ->
   确定性哈希 embedding。这是「依赖倒置 + 优雅降级」的具体落地。

> **面试可以这么讲**：拿到环境先做 feasibility recon，发现装不了依赖，于是把约束翻译成设计原则：
> 内核零依赖 + 能力插件化 + 离线确定性测试。约束倒逼出了比「直接 pip install langchain」更好的架构。
> ——注意我不是「被迫降级」，是把约束当成了需求：一个 5MB、零依赖、clone 下来就能跑测试的框架，
> 本身就是轻量级 Agent 框架该有的样子。

---

## 阶段 2 · 设计冻结（先冻结接口，再并行实现）

单人项目也值得先写接口规范。原因：实现阶段由多个 Agent 并行写不同模块，
**任何接口歧义都会在集成时变成 bug**。所以流程是：

```
架构师起草 INTERFACES.md
        ↓
5 个对抗性审查视角并行开火（异步并发 / 可测性 / 面试价值 / 一致性 / 简历覆盖度）
        ↓
规范负责人逐条评判（采纳 or 驳回并说明理由）-> 冻结 INTERFACES.md
        ↓
产出 ARCHITECTURE.md + DESIGN_DECISIONS.md + 实现波次划分
```

### 2.1 设计阶段真实产出的规模（本轮实测行数）

| 文档 | 行数 | 内容 |
| --- | --- | --- |
| `docs/INTERFACES.md` | 5518 | 41 个文件的**封闭清单** + 依赖 DAG + 每个公开签名 + 全部常量/事件/异常 |
| `docs/DESIGN_DECISIONS.md` | 1509 | 27 条「背景/备选/选择/理由/代价/面试怎么讲」 |
| `docs/ARCHITECTURE.md` | 424 | 分层图 + ReAct 时序 + 多 Agent 数据流 + 与 LangChain 对比 |

### 2.2 对抗性审查**真的抓到了 bug**（本阶段最有价值的部分）

规范 v1 被 5 个视角并行攻击后，暴露出 12 个 blocker 级问题。举三个最有代表性的：

1. **ReAct 循环里的重复写入 bug**。v1 规定「parse error 时把 assistant 原文回灌」，
   但同一份原文在步骤 2 已经入过 transcript，于是同一条消息出现两遍，模型会看到
   自己说了两遍同一句话。v2 冻结为：**一次 parse error 恰好新增 2 条消息**
   （1 条 assistant + 1 条 NUDGE），并把这个**不变式写进了测试断言**
   （实现见 `liteagent/agent/agent.py::_self_correct` 的 docstring）。
2. **`asyncio.Semaphore` 跨事件循环崩溃**。实测复现：同步 API 每次调用都 `asyncio.run()`，
   而 `asyncio.Semaphore` 一旦发生**争用**就会把当时的事件循环存进 `self._loop`，
   第二次 `asyncio.run()` 复用同一个 executor 就会
   `RuntimeError: ... is bound to a different event loop`。
   v2 的解法是 `LoopBoundPool`：所有 asyncio 原语**按 loop 懒创建**，退出时显式清理；
   并且实测发现用 `WeakKeyDictionary[loop]` 反而会泄漏（原语强引用 loop -> key 永不失效），
   必须显式 `clear`（对应面试题 `docs/INTERVIEW.md` Q6 / 故事 1、2）。
3. **`asyncio.to_thread` 的静默陷阱**。同步/异步工具走同一个执行器时，
   如果把 async 函数交给 `to_thread`，它**不会被执行**，只是返回一个 coroutine 对象然后被丢掉 ——
   工具「看起来跑了」但其实什么也没干。v2 冻结为：同步分支必须调用同步入口
   （`loop.run_in_executor(tp, tool.run, args)`，`tool.run` 是同步入口）。

> **面试可以这么讲**：我在写代码前先冻结接口，然后故意用 5 个对抗视角去攻击自己的设计。
> 抓出来的不是格式问题，而是 12 个会导致线上偶发崩溃的并发/状态机缺陷 —— 比如 asyncio
> 原语绑定事件循环导致的「第二次调用必崩」。这些 bug 如果等到写完两万多行代码再发现，
> 修复成本是现在的几十倍。
> **可追问点**：为什么是「对抗视角」而不是「评审」？因为评审会客气，对抗不会 ——
> 我给每个审查者的 KPI 是「找出让这份规范无法实现的理由」，找不出来才算过。

---

## 阶段 3 · 并行实现（4 个波次、18 个实现者）

按依赖 DAG 分成 4 个波次并行实现，**每个波次结束才进入下一波**（因为下一波要真的 import
上一波的代码做冒烟验证）：

```
Wave A (L0-L3 基础)   errors+types | config | llm/message+transport | tools/schema | memory/base+embeddings
Wave B (L3 服务)      llm layer    | scripted | tools/base+registry | tools/executor | memory/buffer+summary
Wave C (L3-L5 上层)   memory/vector+manager | agent/state+callbacks | agent/parser | multiagent/base+blackboard
Wave D (L5-L6 顶层)   agent/agent.py | tools/builtin(6) | multiagent/sequential+hierarchical | cli + 包入口
```

结果：18 个实现者、0 个失败、编译通过、`import liteagent` 成功。

### 3.1 「封闭文件清单」是并行实现能成功的关键

规范里 §1.2 把 41 个文件写成**封闭清单**，并规定「不得新增、改名、删除」
（唯一例外是 `liteagent/__main__.py`，其 6 行逐字内容被冻结在规范里）。
好处是并行的 18 个实现者不可能撞车（每个文件恰好一个 owner），
也不可能出现「两个人都造了一个 helper 模块」这种典型并行事故。
配合 §1.1 的**单向依赖 DAG + 12 条同层白名单**，
`tests/test_zero_dependency.py` 还能用 AST 自动验证依赖方向没被破坏。
封闭清单还带来一个附加收益：`scripts/check_spec_consistency.py` 可以拿它当**唯一真值源**
去核对仓库，本轮实测 `error=0 warn=0 info=0`。

### 3.2 实现者**主动报告了自己没验证的东西**

每个实现者被要求填 `not_done` 字段，结果是 18 个 agent 全都如实列出了自己没做到的部分
（例如「没有真实 API key，Anthropic 适配器只用手写 `FakeTransport` 验证了请求构造与响应解析，
没有对真实端点跑过」）。**这种诚实是有价值的**：它直接变成了 `docs/VERIFICATION.md`
里 `code-only-not-run` 那一列的素材（U-1 ~ U-10），而不是让面试官在追问时把你问穿。

> **面试可以这么讲**：并行写两万多行代码不撞车，靠的不是「大家小心点」，而是**结构性约束** ——
> 文件清单封闭 + 依赖方向单向 + owner 唯一。把「不要撞车」从人的自觉变成了机器能检查的规则。
> 另外我要求每个实现者主动申报 `not_done`，因为**一个不知道自己边界在哪的实现者，
> 比一个承认自己没验证过的人危险得多**。

---

## 阶段 4 · 测试阶段

### 4.1 为什么不用 pytest

环境里没有 pytest（装不上），所以测试全部用标准库 `unittest`。
**这不是妥协，反而逼出了更好的设计**：因为没有 `mock` 框架的魔法可用，
我们必须给 LLM 层造一个**脚本化假模型** `ScriptedLLM`：

```python
llm = ScriptedLLM([
    ScriptedResponse.tool("read_file", {"path": "app.py"}),        # 第 1 轮：模型要求读文件
    ScriptedResponse.react(action="run_tests", action_input={}),   # 第 2 轮：文本模式要求跑测试
    ScriptedResponse.text("我把 bug 修好了"),                       # 第 3 轮：给最终答案
])
agent = Agent(llm=llm, tools=registry, config=AgentConfig(max_steps=8))
result = asyncio.run(agent.arun("修一下 app.py 里的 bug"))
llm.assert_exhausted()      # 断言脚本被精确消费完 —— 少调一次/多调一次都会失败
```

这个设计带来三个真实收益：

1. **整套 1652 个测试离线、确定性、不需要 API key**，任何人 clone 下来约 30 秒跑完。
2. 可以精确构造**罕见分支**：模型返回非法 JSON、返回空的 tool_calls、返回
   `finish_reason="length"`、返回超长文本、连续重试后失败 —— 这些用真实 API 几乎没法稳定复现。
3. 可以**反向断言调用契约**：`llm.calls[i].messages` 记录了第 i 轮模型实际看到的 prompt，
   于是「记忆是否被正确注入」「工具 schema 是否被传给模型」这类断言可以直接写成数据断言。
   （第 5 阶段那条最严重的 bug，正是靠这个能力抓出来的 —— 见 §5.2 第 ④ 条。）

> **面试可以这么讲**：我没有用 mock 去桩掉内部函数，而是把「模型」抽象成一个可编程的响应队列。
> 因为框架对 LLM 的依赖被收敛在 `LLMClient` 这一个接口上，
> 假模型就成了一个**合法的一等实现**，而不是测试补丁。结果是整个 Agent 循环（含多 Agent 协作）
> 都能在离线环境里做端到端测试。
> **一个反直觉的点**：pytest 装不上反而逼我做出了一个更干净的设计 —— 用 mock 容易越桩越深，
> 最后测的是「mock 之间的对话」；而 ScriptedLLM 测的是真实的循环。

### 4.2 测试规模（本轮实测）

```
$ python3 -m unittest discover -s tests -t .
Ran 1652 tests in 30.171s
OK                                              # EXIT=0（连跑两次结果一致）
```

| 指标 | 本轮实测值 |
| --- | --- |
| 测试用例数 | **1652**（`Ran 1652 tests`，AST 同口径计数也是 1652） |
| 测试文件数 | **37** 个 `tests/test_*.py` |
| 全量耗时 | 约 22 ~ 34 秒（共享机器，随负载浮动） |
| 用例最多的文件 | `test_config.py` 169、`test_tools_executor.py` 119、`test_cli.py` 94、`test_tools_schema.py` 91 |
| 体积最大的测试文件 | `test_tools_executor.py` 2105 行、`test_config.py` 1286 行、`test_zero_dependency.py` 1217 行 |
| 联网 | 0 次 |
| API key | 不需要 |

> 说明：早前快照里出现过 1574 这类数字，那是 §12 冻结的 4 份测试文件落地**之前**的时点。
> **本文件不再引用历史数字作为当前值**；当前值的唯一来源是上面这条命令。

### 4.3 测试阶段**抓到的真实 bug**（这批 bug 是这个阶段最大的产出）

| # | 现象 | 根因 | 影响 |
| --- | --- | --- | --- |
| 1 | `VectorMemory.search()` 抛 `OverflowError: Numerical result out of range` | 近因分 `2.0 ** (-age_days / half_life)` 在 `now < created_at`（时钟回拨、或从磁盘恢复出未来时间戳的数据）时指数变正，数值爆炸 | 记忆检索整条链路炸掉；规范只写了公式没写负 age 的钳制 |
| 2 | 假传输层把 HTTP 429 当成 200 解析 | `FakeTransport` 只回放响应不映射状态码，违背 §6.2 对 `Transport.send` 的契约（「失败必须以 `LiteAgentError` 子类抛出」） | 任何照着文档写限流测试的人都会写出假绿的测试 |
| 3 | 4 个文件首行不是 `from __future__ import annotations` | 模块 docstring 抢占了第一行 | 违反 §2.1 冻结约定，被守门测试抓住 |

第 1 条尤其值得讲：它是**测试作者在写「混合打分公式手算对照」用例时踩出来的**，
如果只写「大差不差」的模糊断言，这个 bug 会一直潜伏到线上遇到时钟回拨。

> **面试可以这么讲**：测出 bug 不稀奇，稀奇的是**测出「规范本身的空缺」**。
> 第 1 条 bug 的根因不在代码里，而在规范只定义了 `age >= 0` 时的公式。
> 我的处理是：按规范精神钳制 age 下界，同时把这个裁决记进文档，
> 而不是默默改一行代码让它过去。

---

## 阶段 5 · 对抗性审计（本轮新增 —— 也是最有面试价值的一段）

测试全绿之后，我做了一件在个人项目里不太常见的事：**不接受「全绿」作为终点，
而是组织了一轮针对自己作品的对抗性审计**。

### 5.1 审计是怎么组织的

- **规模**：71 个 agent、5 个审计视角并行开火。
  五个视角是：**并发正确性 / 规范一致性 / 测试有效性 / 简历主张证伪 / 文档诚实性**。
  注意最后一个和前四个不同 —— 它审的不是代码，是**我说过的话是不是真的**。
- **产出**：**37 条发现**。
- **关键机制：对抗性验证（adversarial verification）**。
  每条发现**不直接交给修复者**，而是先发给**2 个独立复核者去「证伪」**：
  复核者的任务是「证明这条发现是错的 / 是我看错了 / 是环境噪声」。
  **只有 2 个复核者都没能证伪的发现，才存活下来进入修复队列** —— 最终 **30 条存活**。
- **为什么值得这么做**：审计 agent 和实现 agent 一样会犯错。如果带着「审计说的都对」的心态去改，
  会改出「为了修一个不存在的问题而引入一个新问题」。复核层把信噪比提上来的同时，
  也把「这条为什么是真问题」的证据链固定了下来。

> 下面的每条发现，都对应仓库里真实存在的**回归用例**（我逐条跑过，见 §5.2 的用例名）。
> 而 71 / 5 / 37 / 30 这几个是**审计过程的记录数字**，过程本身不可回放，所以我不把它当作
> 「我复跑过的事实」，只当作审计台账来引用；**我能复跑的是它修出来的结果** —— 这才是真正该被相信的部分。

### 5.2 抓到了什么（挑六条最有代表性的）

#### ① `Blackboard.awatch` 丢唤醒 + 版本回退时静默失联

- **现象 A（丢唤醒）**：消费者正在 `yield` 上处理上一条 entry 期间发生的写入，**永远收不到**。
- **现象 B（静默失联）**：TTL 过期 / `delete` / `clear` 之后同 key 重写，版本号**回退**到 1，
  判脏条件 `v != last` 恒假 —— 写入明明发生了，却一次都不 yield。
- **根因**：v2 冻结的 `cond.notify_all()` 是**提示式**唤醒（没有等待者时是空操作），
  而从「处理上一条」到「重新进入 wait()」之间有个窗口，落在窗口里的唤醒被直接丢弃；
  加上判脏量用了**会回退的 `entry.version`**。
- **修复**：改为标准 monitor 写法 —— **先登记 watcher、后读基线**（顺序不可颠倒）+
  判脏量换成**单调递增的写入序号 `_write_seq_of(key)`** + 在 cond 锁的临界区内判谓词。
  另外实测发现 3.10 起 `asyncio.Condition.notify_all()` **要求调用方持锁**，
  未持锁会抛 `RuntimeError: cannot notify on un-acquired lock`，
  所以唤醒载荷改成「在目标 loop 内创建一次唤醒任务」。
- **回归用例**（`tests/test_blackboard.py::WatchTests`）：
  `test_awatch_registers_watcher_before_reading_baseline`、
  `test_awatch_delivers_write_that_lands_while_consumer_handles_previous`、
  `test_awatch_delivers_rewrite_after_ttl_expiry`、
  `test_awatch_delivers_rewrite_after_delete_and_clear`。
- **面试价值**：这类 bug **只在时序窗口里出现**，单线程顺序测试永远测不到；
  而且它不会崩，只是「安静地不工作」—— 比崩溃难查一个数量级。

#### ② `ToolExecutor` 的 `fail_fast` 标志是实例级共享 -> 并发的两个批次互相污染

- **现象**：同一个 executor 被并发共享时，批次 B 一开头就把批次 A 的 `fail_fast` 标志抹掉。
- **后果**：批次 A 收尾时，把自己 `fail_fast` 亲手取消的兄弟任务**误判成「调用方取消了
  `execute_many`」**，于是 `raise CancelledError` —— 既违反「必须返回与 `calls` 等长的列表」
  的契约，在 Agent 路径上还会让**整次 run 被误当作外部取消**。
- **根因**：v2 把「本批次是否触发过 fail_fast」挂在实例上（`self._fail_fast_triggered`）。
- **修复**：改成本批次**局部变量**，由 `_watch_fail_fast(...) -> bool` 返回；
  监视器只负责取消未完成的兄弟，收尾的 `gather` 由调用方做。
- **回归用例**：`tests/test_tools_executor.py::ConcurrencyTests::test_concurrent_fail_fast_batches_do_not_share_batch_state`。

#### ③ 取消一个正在等待顺序锁的任务 -> **永久泄漏该锁**，后续调用全部挂死

- **现象**：朴素写法 `await asyncio.to_thread(lock.acquire)` 有个洞 —— 调用方被取消时，
  asyncio 只能取消 **await 那一层**，worker 线程仍会把 `lock.acquire()` 跑完；
  而「释放锁」的 `finally` 属于**已经被取消的协程**，永不执行。
  锁于是被一个**没有逻辑归属者的线程**永久持有，该工具此后每次调用都卡在 `acquire` 上。
- **最坏的部分**：卡死的是**默认执行器里的非 daemon 线程**，
  `asyncio.run` / `_run_and_cleanup` 收尾时 `shutdown_default_executor()` 会去 join 它 ——
  于是**同步 API（`execute_sync` / `Agent.run`）无异常、无超时地永久挂起**。
- **触发面比想象中宽**：不止 `fail_fast`，外层 `asyncio.wait_for`、`Agent.astream` 提前 `break`
  的 `task.cancel()`、Ctrl-C、父任务取消，都会命中。
- **修复**：引入 `_CancelSafeAcquire` —— 把「取消已发生」传回线程，线程拿到锁后若发现已取消，
  **立刻原地归还**，绝不让锁变成孤儿（`abort()` 与 `acquire()` 用一把小锁串行化，保证恰好归还一次）。
- **回归用例**：`tests/test_tools_executor.py::SequentialToolsCancellationTests::test_cancelled_lock_waiter_does_not_orphan_the_sequential_lock`。

#### ④ 每次 `Agent.arun` 把当前用户消息发了两遍（**本轮最严重的功能性 bug**）

- **现象**：prompt 里当前用户消息出现两次 —— 首轮就是 `[system, user, user]`。
- **后果**：白付一份 token；多轮对话里模型看到用户同一句话说了两遍；
  在「重复检测」逻辑上还可能造成误判。
- **根因**：**两个写入点各写了一次**。`Agent` 在 run 开始时把用户输入写进 buffer
  （步骤 0，唯一写入点）；而 `MemoryManager.abuild_prompt` 的第 4 段会把**整个 window 倒出来**，
  第 6 段又**无条件追加**一次当轮用户输入 —— 于是同一句话进了两次。
- **修复**：在第 6 段加**去重守卫** —— 窗口末尾已经是同一条用户消息时跳过追加。
  退化配置（`buffer_max_messages=0`，窗口为空）下守卫不成立，仍会追加，语义与 v2 一致，不丢输入。
- **回归用例**：`tests/test_agent_features.py::test_current_user_message_appears_once_in_every_prompt`
  （配套还有 `test_one_response_leaves_exactly_one_assistant_message`，管住 assistant 侧的同类问题）。
- **为什么说它最严重**：前三条是并发时序 bug，需要特定交织才触发；
  **这一条是「每次运行、每个用户、每一轮都发生」的确定性 bug** —— 只是因为测试全绿而没人发现。
  它恰好说明：**测试覆盖率不等于正确性**，缺的是「换个角度问一句『这个数对不对』」。

#### ⑤ 变异测试：**全绿不等于被测到**

审计对测试套件本身做了一次**变异测试（mutation testing）**：故意把若干条真实存在的分支
「删掉 / 短路掉」，看测试会不会红。审计报告的原话是：**有三处分支删掉后，整套测试依然全绿**：

1. **整个-JSON 解析路径**（文本 ReAct 的 §9.3 步骤 2，`ReActParser.allow_json_object`）——
   把它短路成 `if False:`，没有任何一条用例会失败。更糟的是，它产出的 `ParsedAction.json_mode`
   字段此前**全项目无人断言过**。
2. **嵌套 dataclass 的 `default_factory` 判定**——
   `field.default_factory is dataclasses.MISSING` 这一支删掉后无人发现，
   意味着 `field(default_factory=list)` 的字段会被**错误地标进 `required`**。
3. **`unregister` 的别名清理** —— 删掉后，悬空别名会静默留在表里；
   更坏的是「**静默改绑**」：`unregister` 后注册一个**同名但不同实现**的新工具，
   旧别名会悄悄指向新工具。

**本轮我重放了这三刀，得到的是修复后的结果**（在 `/tmp` 的独立副本上做，不动仓库；
命令见 §7.3）：

| 变异 | 改动 | 结果（本轮实测） |
| --- | --- | --- |
| M1 | `parser.py`: `if self.allow_json_object:` -> `if False:` | `Ran 1652 tests` -> **FAILED (errors=7)** |
| M2 | `schema.py`: 删掉 `and field.default_factory is dataclasses.MISSING` | `Ran 1652 tests` -> **FAILED (failures=2)** |
| M3 | `registry.py`: 删掉 `unregister` 的别名清理循环 | `Ran 1652 tests` -> **FAILED (failures=3)** |

也就是说，**修复阶段补的 7 / 2 / 3 条回归用例，现在是真正承重的** ——
删掉对应分支，套件立刻变红。

> **为什么这张表比「1652 全绿」更有说服力**：全绿只证明「我没测出问题」，
> 变异实验才证明「**我的测试真的在测东西**」。
> 顺带一个诚实的对账：审计是在**更早一个 revision** 上做的，那次快照的用例数与本轮实测的
> 1652 不同 —— 差的正是修复阶段补进去的这批回归用例。
> 那个历史数字我**无法在本轮复现**（也正因如此才对不上），所以本文件不把它写成当前值；
> 凡属历史时点的数字一律显式标注为历史，当前值只认本轮跑出来的那一个。

#### ⑥ 文档的过期数字与不实声明

- **抓到的典型**：README 里写着「`requests` / `httpx` 没装」，而**本机其实两个都装了**
  （实测 `requests 2.34.2` / `httpx 0.28.1`）——
  更微妙的是，`default_transport()` 按冻结优先级 httpx -> requests -> urllib 实际选中的就是
  `HttpxTransport`，所以那句声明不只是「数字旧了」，而是**把「本机真实走的 HTTP 路径」
  说成了不存在的东西**。
- 同类问题还有：把已经关闭的缺口（G-1 / G-3）继续写成「当前的红灯」；
  文档里 `docs/` 各文件的行数与实际不符；`[v3 修正]` 之前的多处「现在时」假陈述。
- **修复**：README §8 改成如实陈述（「本机两库都已安装，未验证的是这两个适配器的 `send()`
  与真实出网」）；`docs/VERIFICATION.md` 增加了「与旧数字对账」一节，
  并把数字的**唯一来源**钉死在 §1；另加了一条守门断言
  （`tests/test_docs_coverage.py::test_declared_missing_test_files_really_are_missing`）
  防止同类漂移再次发生。

### 5.3 复核者的作用：**2 条发现被两位复核者一致否决**

对抗性验证机制不只是「提高信噪比」的口号，它**真的拦下了东西**：

- 有 **2 条发现**被两位独立复核者**一致否决**，没有进入修复队列。
- 其中最有意思的一条是「**测试套件不确定**」（同样的代码、同样的命令，跑出来结果不一致）。
  这个指控如果成立，整个测试体系的可信度就没了。两位复核者分别去查，**查明那是假象**：
  审计期间**另一个 agent 正在并发修改代码**，被测的代码在两次运行之间被换了。
  复核者随后用 **md5 把复验的 revision 锁定**，在同一个 revision 上重复跑，结果稳定一致 ——
  指控被证伪。

**这个细节特别能体现工程严谨性**：面对「测试不确定」这种最吓人的指控，
既没有「宁可信其有」地大改一通，也没有「我本地是绿的」地直接否认，
而是**先锁定被测对象（md5 固定 revision），再谈可复现性**。
「先固定被测量的东西」这件事，是任何性能/正确性结论的前提 ——
这和 benchmarks 里「交错测量 + 噪声带」是同一个思想（见 `docs/VERIFICATION.md` 文末）。

### 5.4 审计之后做了什么

37 条发现 -> 30 条存活 -> 逐条修复 + **每条都补一个能证伪它的回归用例**。
最终回到 `1652 / 1652 / 0`（`Ran 1652 tests` / `OK` / `EXIT=0`），
并且 §5.2 第 ⑤ 条的变异实验证明这批回归用例是承重的。

> **面试可以这么讲（这一段建议完整背下来）**：
> 「测试全绿之后我又做了一轮**对抗性审计**：71 个 agent、5 个视角，其中一个是专门审
> **『我说过的话是不是真的』**的文档诚实性视角。产出了 37 条发现，但我没有直接改 ——
> **每条发现先发给 2 个独立复核者去『证伪』，两个人都没能证伪才进入修复队列**，最后存活 30 条。
> 这一步救了我两次：有 2 条发现被两位复核者一致否决，其中一条是『测试套件不确定』，
> 复核者查明那是审计期间**另一个 agent 在并发改代码**造成的假象，并用 **md5 锁定了复验的
> revision** 才把它证伪。
> 修复阶段抓到最严重的一条是：**每次 `arun` 都把当前用户消息发了两遍** ——
> 首轮 prompt 就是 `[system, user, user]`。这是个 100% 复现的确定性 bug，
> 而它一直没被发现，因为**测试是全绿的**。
> 于是我又做了变异测试：把三处真实分支删掉，套件依然全绿。修复补了回归用例后我重放这三刀，
> 现在分别会红 7 / 2 / 3 条。
> 所以我现在对『全绿』的态度是：全绿只说明我没测出问题，**它不等于被测到**。」

---

## 阶段 6 · 规范与实现重新对齐

修复阶段不可避免地要**偏离几处冻结规范**：有些是规范写错了（如 `awatch` 的判脏量），
有些是规范留了空缺（如负 age 的近因分），有些是实现做了更合理的裁决（如 `execute_many`
的 `fail_fast` 语义修正）。**规范是冻结的，但冻结不等于不能改 —— 等于「改必须留痕」。**
于是回头做了一件事：**给规范打 `[v3 变更]` 标注，把规范与实现重新对齐**。

做法与原则：

1. **每一处偏离，都在规范原文旁标 `[v3 变更]` / `[v3 修正]` / `[v3 新增]`**，
   写明「v2 原文是什么 -> 现在是什么 -> **为什么改** -> **哪条测试看守它**」。
   本轮实测 `docs/INTERFACES.md` 里有 **27 处 `[v3 变更]` + 3 处 `[v3 新增]`**。
2. **分清三类改动**，标注的措辞不同：
   - 「v2 写错了，按实现修正」—— 这是**纠错**；
   - 「v2 没写，补上」—— 这是**补空缺**（如嵌套 dataclass 的 `default_factory` 分支，
     v2 只写了顶层函数参数那一支）；
   - 「v2 是这么冻结的，但实测发现不可行」—— 这是**被现实推翻**（如 `awatch` 的
     `entry.version` 判脏量）。
3. **规范里出现「规格单方面的声明」时，也要标出来**。例如 `Message.metadata["memory_id"]`
   这一行：v2 把它列成「memory 写入」，而全仓库（`liteagent/` + `tests/` + `examples/`）
   **没有任何地方写入它**。处理方式是**保留 key、保留语义钉死**（若将来要关联回
   `MemoryItem.id`，必须用这个 key，禁止自定义同义 key），但在标注里如实写明
   「**当前无写入者**，v2 那句是规范的声明而非事实」。
4. **推翻一条冻结约定，必须留下「推翻的理由」**。例如 `execute_many` 的
   `fail_fast` 标志从实例属性改成局部变量，规范里写清了「为什么实例级共享在并发下是错的」
   以及那条回归用例的名字。**规范从「我想让它怎样」变成了「它现在是这样，理由是……」**。

> **面试可以这么讲**：代码和规范一定会分叉，关键是你**怎么处理分叉**。
> 我的原则是：**规范是活文档，但每次修改都要留痕与理由** ——
> 不允许「代码悄悄改了、文档没跟上」，也不允许「文档写着但代码没实现」。
> 具体到操作：每处偏离都打 `[v3 变更]` 标注，写清「原文 / 改动 / 理由 / 看守它的测试」。
> 本轮一共 30 处。有意思的是，**改动本身没有价值，改动的理由才有** ——
> 面试官不会问你 `fail_fast` 标志存在哪，他会问「你为什么知道原来那样是错的」。

---

## 7. 最终交付清单

### 7.1 文件树（顶层）

```
project-3/
├── README.md                    434 行   项目门面：特性 / 快速开始 / 架构 / 诚实边界
├── Makefile                      28 行   test / demo / lint / tree
├── pyproject.toml                42 行   包元数据（零运行时依赖）
├── liteagent/                  41 个 .py，24765 行  ← 框架本体（零第三方依赖）
├── tests/                      37 个 test_*.py + helpers.py，22409 行
├── examples/                    8 个 .py（含 run_all_examples.py）
├── benchmarks/                  3 个微基准脚本
├── scripts/                     check_spec_consistency.py
└── docs/                        7 份文档
```

### 7.2 真实统计表（**本轮实测，全部可复现**）

| 项 | 数值 | 命令 |
| --- | --- | --- |
| 框架模块数 | **41** 个 `.py` | `find liteagent -name '*.py' \| wc -l` |
| 框架代码行数 | **24765** 行 | `find liteagent -name '*.py' \| xargs wc -l \| tail -1` |
| 测试用例数 | **1652**（全绿，EXIT=0） | `python3 -m unittest discover -s tests -t .` |
| 测试文件数 | **37** 个 `tests/test_*.py` | `ls tests/test_*.py \| wc -l` |
| 测试代码行数 | **22409** 行（含 `helpers.py` 486 行） | `find tests -name '*.py' \| xargs wc -l \| tail -1` |
| 示例 / 基准 / 脚本行数 | **6184** 行 | `find examples benchmarks scripts -name '*.py' \| xargs wc -l \| tail -1` |
| 仓库体积 | **5.0M** | `du -sh .` |
| CLI 版本 | `0.1.0` | `python3 -m liteagent version` |
| 运行时第三方依赖 | **0** | `tests/test_zero_dependency.py` |
| 规范一致性 | `error=0 warn=0 info=0` | `python3 scripts/check_spec_consistency.py` |

**框架代码量分布**（`24765` = 下面各行之总和，可对账）：

| 层 | 行数 | 说明 |
| --- | ---: | --- |
| 顶层（`config` / `errors` / `types` / `cli` / `__init__` / `__main__`） | 4558 | 跨层基础设施 |
| `agent/` | 3886 | ReAct 状态机、解析器、事件回调 |
| `tools/`（不含 builtin） | 4016 | schema 反射、装饰器、注册表、执行器 |
| `tools/builtin/` | 3033 | 网页搜索 / 代码执行 / 文件操作 / shell / memory 工具 |
| `memory/` | 3536 | 短期窗口、摘要、向量库、编排 |
| `multiagent/` | 2671 | 黑板、Sequential、Hierarchical |
| `llm/` | 3065 | 抽象层、传输层、四家 provider、ScriptedLLM |
| **合计** | **24765** | |

**文档规模**（本轮实测行数）：

| 文档 | 行数 | 内容 |
| --- | ---: | --- |
| `docs/INTERFACES.md` | 5518 | 冻结接口规范（唯一契约，含 30 处 `[v3 变更/新增]`） |
| `docs/DESIGN_DECISIONS.md` | 1509 | 27 条设计决策 + 面试话术 |
| `docs/INTERVIEW.md` | 1341 | 面试作战手册（26 个 Q&A + 7 个踩坑故事） |
| `docs/TOOLS.md` | 966 | 工具作者指南 |
| `docs/VERIFICATION.md` | 462 | claim / evidence / status 对账表 + 3 个实测补充 |
| `docs/ARCHITECTURE.md` | 424 | 架构图 / 时序图 / LangChain 对比 |
| `docs/BUILD_LOG.md` | 本文件 | 搭建过程记录 |
| `README.md` | 434 | 项目门面 |

### 7.3 复现命令（一条条照抄即可）

```bash
cd /home/ml-user/workdir/project-3

# --- 基线数字 ---
python3 -m unittest discover -s tests -t . > /tmp/b1.txt 2>&1; echo EXIT=$?; tail -3 /tmp/b1.txt  # Ran 1652 tests / OK
ls tests/test_*.py | wc -l                                     # 37
find liteagent -name '*.py' | wc -l                            # 41
find liteagent -name '*.py' | xargs wc -l | tail -1            # 24765 total
find examples benchmarks scripts -name '*.py' | xargs wc -l | tail -1   # 6184 total
du -sh . ; python3 -m liteagent version                        # 5.0M / 0.1.0

# --- 规范一致性 ---
python3 scripts/check_spec_consistency.py                      # error=0 warn=0 info=0

# --- 变异实验（§5.2 第 ⑤ 条；在独立副本上做，不动仓库）---
cp -r . /tmp/mut && cd /tmp/mut
# M1: parser.py 里 `if self.allow_json_object:` -> `if False:`       -> FAILED (errors=7)
# M2: schema.py 里删掉 `and field.default_factory is dataclasses.MISSING` -> FAILED (failures=2)
# M3: registry.py 里删掉 unregister 的别名清理循环                   -> FAILED (failures=3)
python3 -m unittest discover -s tests -t . > /tmp/mut.txt 2>&1; echo EXIT=$?; tail -3 /tmp/mut.txt
```

---

## 8. 这个项目教会我什么

1. **约束倒逼架构。**
   没有外网、没有 pytest，看起来是纯粹的坏消息。但它把我推向了「内核零依赖 +
   能力插件化 + 离线确定性测试」—— 于是有了一个 5MB、clone 下来 30 秒跑完测试、
   在任何 Python 环境都能启动的框架。**先问「在这个约束下什么设计才成立」，
   比先问「业界都用什么」有用。**

2. **先冻结接口，再并行。**
   并行写 41 个文件、两万多行代码而没有一次「两个人都造了同一个 helper」的事故，
   靠的不是小心，是**结构性约束**：文件清单封闭、依赖方向单向、owner 唯一。
   **把协调问题变成可机器检查的规则，是规模化的唯一办法。**
   而且这套规则还顺手变成了守门测试的输入（`test_zero_dependency.py` + `check_spec_consistency.py`）。

3. **全绿不等于被测到。**
   这是本项目最大的认知升级。测试全绿的状态下，审计抓到了一个
   **100% 复现的功能性 bug**（用户消息发两遍），变异测试又证明**三处分支删掉后套件依然全绿**。
   **覆盖率衡量的是「代码被跑到」，不是「行为被断言」。**
   修复之后我重放了那三刀，套件分别红 7 / 2 / 3 条 —— 这才叫「被测到」。
   以后我评价一套测试，第一个问题会变成：「删掉哪一行代码，它会红？」

4. **对抗性验证比自我检查有效。**
   自己审自己，会下意识地绕开自己拿不准的地方。我的做法是：**每条发现先交给
   2 个独立复核者去「证伪」，两个人都证伪不掉才修**。
   结果是有 2 条发现被一致否决，其中「测试套件不确定」那条的真相是
   **审计期间另一个 agent 在并发改代码** —— 复核者用 md5 锁定 revision 才把它证伪。
   **先固定被测量的对象，再谈结论** —— 这条纪律同时适用于正确性和性能。

5. **先固定被测量的东西，再谈「快」「慢」「对不对」。**
   三份 benchmarks 都是「交错测量 + 预定义噪声带 + 报比值分布」，
   结论里明确写着「这一格测不出差异」；候选设计里说「dataclass 在纯 python 侧更轻」的那半句，
   实测在序列化上**不成立**（pydantic 反而略快），于是被改掉。
   **宁可输出「不知道」，也不报一个单次跑出来的方向。**

6. **诚实标注边界，比夸大更能赢得信任。**
   `docs/VERIFICATION.md` 里专门有一节 U-1 ~ U-10，逐条写「这条我没跑过」；
   文档里凡没亲自跑出绿字的一律标黄（宁可标黄，不许标绿）。
   18 个实现者如实申报 `not_done`；审计专门设了一个「文档诚实性」视角来审我说过的话。
   **面试官真正在意的不是你做了多少，而是「你说的每一句，我能不能信」。**
   把边界主动划清楚，反而让边界之内的部分更有分量。

7. **规范是活文档，但每次修改都要留痕与理由。**
   代码和规范一定会分叉。允许分叉、但**每一次偏离都打 `[v3 变更]` 标注**：
   原文是什么、现在是什么、为什么改、哪条测试看守它（本轮 30 处）。
   这样规范就从「我想让它怎样」变成了「它现在是这样，以及为什么」——
   **改动的理由，比改动本身更值钱。**
