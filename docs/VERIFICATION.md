# liteagent 验证记录（VERIFICATION.md）

> 本文件由多个部分拼装而成，各部分由不同的 owner 维护：
>
> * **§1 基线数字 / §2 主表（§12.2 + §12.3 对账）/ §3 未验证边界 / §4 已知缺口 / §5 复现命令** —— 主实现者维护；
> * **`## 实测补充（benchmarks）`** 下的各小节 —— 由 `benchmarks/*.py` 脚本**自动追加/更新**；
>   每个小节被一对 HTML 注释标记包住，脚本只重写标记之间的内容，**不会碰其它小节**。
>
> 重跑命令（只重写各自标记块，不动别的内容）：
>
> ```bash
> python3 benchmarks/bench_dataclass_vs_pydantic.py --json
> python3 benchmarks/bench_embedding_similarity.py --json
> python3 benchmarks/bench_retrieval_ranking.py --json
> ```

## 0. 怎么读这张表（状态词表的含义，§12.2 冻结）

`status` 只有三个取值，**含义是冻结的**：

| status | 含义 | 面试时怎么表述 |
|---|---|---|
| `offline-verified` | 在**本机离线**跑过，有可复现的命令与落点，跑出来是绿的 | 「这条我真跑过」 |
| `code-only-not-run` | 代码写了、接口在，但**没有在真实条件下跑过**（缺网络/缺 key/只测了替身） | 「这条只到代码层，边界我清楚」 |
| `not-implemented` | 冻结清单点名了，但**东西还不存在** | 「这条还没做」 |

**本文件的写作纪律（§12.2 与任务书共同要求）**：宁可标黄，不许标绿。
凡没有亲自跑出绿字的，一律不进 `offline-verified`。

## 1. 本次实测的基线与环境（数字都是跑出来的，不是抄的）

| 项 | 实测值 | 怎么得到的 |
|---|---|---|
| 生成时间 | 2026-09-27 04:27 UTC（[v4 收尾对齐]，上一版是 2026-09-27 03:48 UTC） | `date -u` |
| Python | 3.10.12 (main, Mar 3 2026) [GCC 11.4.0] | `python3 -VV` |
| 平台 | Linux 5.15.0-78-generic（共享机器） | `uname` / `platform` |
| `liteagent/` 模块数 | **41** 个 `.py` | `find liteagent -name '*.py' \| wc -l` |
| `liteagent/` 代码行数 | **24765** 行 | 见 §5「环境与规模」第 4 条命令 |
| 已落地测试文件 | **37** 个 `tests/test_*.py`（冻结清单 37 个，**缺 0 个**，G-1 已关闭） | `ls tests/test_*.py \| wc -l` |
| 测试方法总数（AST 计数，与守门测试同口径） | **1652** | 见 §5 复现命令 |
| 主套件实跑结果 | `Ran 1652 tests` → **OK**，即 **1652 通过 / 0 失败**（EXIT=0） | `python3 -m unittest discover -s tests -t .` |
| 唯一失败例 | **无**（`FrozenLayoutTests.test_all_frozen_test_files_exist` 现在通过：§12 冻结的 37 份测试文件全部落地） | 见 §4 G-1（已关闭） |
| CLI 版本 | `0.1.0` | `python3 -m liteagent version` |

> **与旧数字对账（[v3 重写]）**：仓库其它地方若出现「1574」「1586」这类数字，那是更早时点的
> 快照。上一版台账把 1586 / 1585 / 1 写成"当前实测"，但那份快照是在 §12 冻结的 4 份测试文件
> 落地**之前**拍的 —— 文件落地后套件就转绿了，而数字没跟着刷新，于是台账自己成了一份
> "现在时"的假陈述（§4 的 G-1 整节、§5 的逐条复现命令、面试手册里的开场数字都受影响）。
> **本次已重跑全部命令并刷新**：1652 / 1652 / 0，`check_spec_consistency` 为
> `error=0 warn=0 info=0`。数字性声明一律以本节为唯一来源，其余文件引用本节。
> 下一节起凡带具体用例数的括号（`（66 例）`）仍是**写入时快照**，会随重构漂移 ——
> §12 明令 `test_docs_coverage.py` 不做方法级断言，所以那些数字没有人看守，请以总数与实跑为准。
> `[v4]` 本次已用同一份 `ast` 脚本逐文件重数了 §2.1/§2.2 的每一个括号：只有
> `test_cli`（92→94）需要改，其余全部对齐；§5 复现命令里 `test_zero_dependency` 的
> 注释 `Ran 33 tests` 也已按实跑改成 `Ran 34 tests`。但**下一轮重构仍会让这些数字漂移**
> —— 这条纪律不变。

## 2. §12.2 主表（claim | evidence | status）

### 2.1 §12.3 简历原子能力对账（**逐行覆盖，9 行一一对应**）

`evidence` 列的写法冻结为两种形式：「一份 `tests/` 下的测试文件路径」或
「`command: python3 -m ...`」；括号里的用例数是 `ast` 数出来的 `test_*` 方法数。

| claim | evidence | status |
|---|---|---|
| **[§12.3 #1]** LLM 抽象层统一多模型 API（OpenAI / Anthropic / DeepSeek / Echo 四家 + `openai-compatible`） | `tests/test_llm_providers.py`（66 例，走 `tests/helpers.FakeTransport` 回放）+ `tests/test_llm_registry.py`（30 例，解析 `provider:model@base_url`） | offline-verified |
| **[§12.3 #2]** 工具层装饰器自动注册 + JSON Schema 生成 | `tests/test_tools_schema.py`（91 例）+ `tests/test_tools_registry.py`（41 例，含 `auto_register` 四例；[v3] 别名表状态与 `unregister` 悬空别名的回归用例） | offline-verified |
| **[§12.3 #3]** 记忆层短期对话历史（窗口裁剪 + token 预算） | `tests/test_memory_buffer.py`（30 例）+ `tests/test_memory_manager.py`（48 例） | offline-verified |
| **[§12.3 #4]** 记忆层长期向量存储（upsert / 混合打分 / MMR / 持久化） | `tests/test_memory_vector.py`（48 例）+ `tests/test_memory_persistence.py`（15 例） | offline-verified |
| **[§12.3 #5]** 完整 Thought-Action-Observation 循环（native 与 text 两种模式） | `tests/test_agent_react_text.py`（30 例，含 [v3] §9.3 步骤 2「整段 JSON 对象」的 7 条补测）+ `tests/test_agent_react_native.py`（23 例） | offline-verified |
| **[§12.3 #6]** Function Calling 与执行器（错误处理与自动重试） | `tests/test_tools_executor.py`（119 例，含 [v3] 校验覆盖率缺口 6 例：注解降级成 `{}` 的参数 / provider 把 arguments 丢成 `{}` 都必须留痕）+ `tests/test_llm_retry.py`（21 例） | offline-verified |
| **[§12.3 #7]** 支持并发工具调用（顺序保持 / 峰值上限 / `fail_fast`） | `tests/test_tools_executor.py`（`test_execute_many_preserves_order`、`test_execute_many_respects_max_concurrency_async/sync`、`test_concurrency_argument_lowers_the_peak`、`test_fail_fast_cancels_siblings_and_keeps_alignment`） | offline-verified |
| **[§12.3 #8]** Sequential / Hierarchical 两种协作模式 + 主 Agent 任务分解 | `tests/test_multiagent_sequential.py`（48 例）+ `tests/test_multiagent_hierarchical.py`（79 例，含 `arun_plan` 的 `depends_on` 顺序与并发峰值断言，以及 [v3] 空计划必须留痕（WARNING + `metadata["subtasks"]` / `empty_plan`）的回归用例） | offline-verified |
| **[§12.3 #9]** 内置工具（网页搜索 / 代码执行 / 文件操作）+ 文档与示例 | `tests/test_builtin_code.py`（52）+ `tests/test_builtin_files.py`（45）+ `tests/test_builtin_shell.py`（30）+ `tests/test_builtin_web.py`（38）+ `tests/test_e2e_code_assistant.py`（8）+ `tests/test_examples_offline.py`（7）+ `command: python3 examples/07_code_assistant.py --offline` | offline-verified（**[v3] 由 code-only-not-run 升级**：G-1 关闭后，原先缺失的两份自动化验收测试已落地并转绿） |

**关于第 9 行（[v3] 已由黄转绿，写明依据）**：
四份 `test_builtin_*.py` 是真绿（我逐个跑过，见 §5）。
上一版这一行标黄，理由是 §12.3 点名的另一半证据 —— `test_examples_offline` 与
`test_e2e_code_assistant` 两份冻结测试 —— **文件还没落地**（G-1）。G-1 已关闭：
这两份文件都已落地并转绿（`python3 -m unittest tests.test_e2e_code_assistant` -> `Ran 8 tests` / `OK`；
`tests.test_examples_offline` 用 `subprocess` 真的把示例当命令行程序跑、断言退出码 0）。
因此这一行升级为 `offline-verified` —— 依据是可复现的命令，不是"我觉得应该没问题"。

### 2.2 支撑性 claim（§12.3 九行之外，我额外列出来）

| claim | evidence | status |
|---|---|---|
| **[支撑 #10]** 零第三方依赖红线（41 模块顶层无三方 import、每个模块首行 `from __future__ import annotations`） | `tests/test_zero_dependency.py`（34 例；零依赖 / 3.11 API / import 边 / `__all__` / 套件线程卫生（[v3] 新增：跑完套件后不得残留 `liteagent-*` 工作线程）各例全过） | offline-verified |
| **[支撑 #11]** CLI（`version` / `tools list` / `tools show` / `tools schema` / `run` / `trace` / `chat`） | `tests/test_cli.py`（94 例）+ `command: python3 -m liteagent version` + `command: python3 -m liteagent tools schema read_file --format openai` | offline-verified |
| **[支撑 #12]** `ScriptedLLM` 确定性夹具（队列消费 / `loop` / `strict` / `tools=None`） | `tests/test_scripted.py`（54 例） | offline-verified |
| **[支撑 #13]** 流式：`Agent.astream` 事件顺序 + 消费者提前 `break` 不泄漏 task | `tests/test_agent_features.py`（25 例，含 [v3] `test_early_break_stops_a_tool_using_run`：带工具调用的 run 在 break 后必须真的停下）+ `tests/test_llm_streaming.py`（20 例，只覆盖 `ScriptedLLM`） | code-only-not-run（**真实 provider 未实现流式，见 §3 U-4**） |
| **[支撑 #14]** 可观测性：事件类型 / JSONL trace / `trace_stats` | `tests/test_callbacks.py`（33 例）+ `command: python3 -m liteagent trace <file> --stats --json` | offline-verified |
| **[支撑 #15]** 多 Agent 共享黑板（版本冲突 / TTL / 多线程写 / `awatch`） | `tests/test_blackboard.py`（64 例，含 [v3] 4 条丢唤醒 / 版本回退 / 登记顺序的回归用例） | offline-verified（[v3] v2 时 `awatch` 的"每次写入都能收到"是**不成立**的：消费者处理上一条 entry 期间发生的写入会永久丢失，TTL 过期/delete/clear 后重写会被静默吞掉；现已修复并补测） |
| **[支撑 #16]** 测试总量阈值（>= 220 个 test method；`test_tools_schema` / `test_tools_executor` 各 >= 30） | `command: python3 -m unittest discover -s tests -t .`（`Ran 1652 tests` / `OK`，阈值大幅满足） | offline-verified |
| **[支撑 #17]** 三份 benchmarks 脚本可离线复现 | `command: python3 benchmarks/bench_retrieval_ranking.py --json --no-write`（退出码 0，产物见文末「实测补充」） | offline-verified |
| **[支撑 #18]** 冻结接口一致性守门（§1.2 文件清单 / §1.4 与附录 B 的 `__all__` 可解析） | `command: python3 scripts/check_spec_consistency.py`（`error=0 warn=0 info=0`，EXIT=0；[v3] 上一版记的 `warn=2` 是交付物尚未落地时的快照，见 §4 G-3 已关闭） | offline-verified |

## 3. 未验证 / 只到代码层（**这一节是这份台账存在的理由**）

以下每一条我都**没有**在真实条件下跑过。写成黄/红，不写成绿。

* **U-1 三个真实 LLM 端点的端到端调用（OpenAI / Anthropic / DeepSeek）**：
  本机**无外网、无 API key**（`pip install` 都超时）。因此只验证了：
  请求体构造（tools 形态、消息转换、system 提取）、响应解析（含 arguments 非法 JSON -> `__raw__` 且不抛）、
  HTTP 错误映射（401/403/429 带 `Retry-After`/400/404/500/未知）。
  全部走 `tests/helpers.FakeTransport` 回放或 `mock.patch("urlopen")`。
  **「换成真 key 就能跑通」是我的推断，不是实测结论。**
* **U-2 `web_search` 的真实后端（DuckDuckGo HTML / Tavily / Serper）**：
  只验证了 `NullSearchBackend`（返回 `[]` 的降级路径）与注入的 `FakeTransport` 假后端。
  `DuckDuckGoHTMLBackend` 的跳转链还原、`TavilyBackend`/`SerperBackend` 的 JSON 解析
  都**只到代码层**；真实搜索结果页的 HTML 结构会变，这里没有任何对抗真实站点的证据。
* **U-3 `RemoteEmbedder` 的真实 HTTP embedding 服务**：
  只验证了它没有 key 时抛错、维度过 `_check_dim` 检查、以及缺 `data` 字段时的错误文案。
  **没有一次真实 embedding 请求。** 默认 `HashingEmbedder` 是词面哈希，不是语义（见文末 D-04 实测补充）。
* **U-4 流式（`Agent.astream` / `astream_chat`）**：
  只在 `ScriptedLLM` 与事件顺序上验证过。真实 provider **根本没有实现流式** ——
  `LLMClient.astream_chat` 的基类实现直接抛 `NotImplementedError`，`HTTPChatClient` 不覆写它
  （冻结理由：`Transport` 只有整段响应接口，伪流式只会假装在流）。
  所以真实 provider 上的流式**不是「没测」，是「没做」** —— 调用方必须捕获并回退到 `achat`。
* **U-5 任何性能 / 并发规模的压测结论**：
  文末三份 benchmarks 都是**共享机器上的微基准**（脚本自己在 `caveats` 里逐条声明），
  只证明「哪边更快/更慢」的**方向与量级**，不是吞吐或容量结论。
  并发只验证到「峰值 <= `max_concurrency`」这种**正确性**断言，
  **没有**做过 1000 并发的吞吐、延迟分布或资源占用测试。
* **U-6 shell / 代码执行的沙箱对抗**：
  只测了 `SHELL_DENY_PATTERNS` 的逐条命中与 `python_eval` 的 AST 白名单/复杂度闸。
  这是**已知模式的拦截**，不是安全审计；没有做过真实逃逸尝试（如编码绕过、`ctypes`、资源耗尽）。
* **U-7 真实时钟与真实 sleep**：按 §12 测试卫生规则，测试**禁止真实 sleep**，
  重试退避用 `tests.helpers.RecordingSleep` 断言 `delays` 列表；
  记忆的「过时/近因」用硬编码 `now=1700000000`（2023-11-14 UTC）注入。
  真实的墙钟漂移、跨时区、时钟回拨**未验证**。
* **U-8 真实网站 HTML 的鲁棒性**：`HTMLTextExtractor` 只用固定样本验证了跳过 `script/style`、
  `html.unescape`、`href` 白名单。真实页面的畸形嵌套 HTML **未验证**。
* **U-9 跨进程持久化**：`test_memory_persistence.py` 是**同进程** `save` -> `load` 往返
  （JSONL 文件格式本身与进程无关，但没有 subprocess 级别的往返测试）。见 §4 G-2。
* **U-10 真实多机 / 分布式**：`LoopBoundPool`、黑板、`share_memory` 都只在**单进程单机**验证。

## 4. 已知缺口（红色的地方，主动列出来）

| # | 缺口 | 实测证据 | 影响 |
|---|---|---|---|
| G-1（**已关闭**） | §12 冻结的 4 份测试文件（`test_e2e_code_assistant` / `test_examples_offline` / `test_examples_import` / `test_docs_coverage`）**已全部落地**（冻结清单 37 份 vs 实际 37 份） | `python3 -m unittest tests.test_e2e_code_assistant` -> `Ran 8 tests` / `OK`；主套件 `Ran 1652 tests` / `OK` | 关闭前的影响：第 9 行的 e2e/示例自动化验收为空、`FrozenLayoutTests` 是红的。**注意**：这条曾经被误写成"当前的红"，后来文件落地而台账没刷新 —— 现在有一条守门断言（`tests/test_docs_coverage.py::test_declared_missing_test_files_really_are_missing`）防止同类漂移 |
| G-2 | 持久化的「跨进程往返」只做到**同进程**文件往返 | `tests/test_memory_persistence.py` 内无 `subprocess` 调用（`grep subprocess` 无命中） | U-9；JSONL 格式是进程无关的，但**没有测试证明** |
| G-3（**已关闭**） | 顶层交付物 `README.md`、`docs/INTERVIEW.md`、`docs/TOOLS.md` **都已落地**（`docs/` 7 份、根交付物 4 份齐全） | `command: python3 scripts/check_spec_consistency.py` -> `error=0 warn=0 info=0`（EXIT=0） | 关闭前的影响：WARN 不影响退出码，但文档交付不完整 |
| G-4（**收窄**） | 「校验覆盖率缺口」的**递归深度上限 4**：`unvalidatable_parameters()` 会下钻嵌套 object 片段（`payload.inner`），但超过 4 层的降级字段不会被列进 `ToolResult.metadata["validation_gaps"]`。另外**恰好**超过 4 层时才静默 —— 1~4 层都有信号 | `tests/test_tools_executor.py::ValidationCoverageGapTests` 覆盖 0~1 层；`python3 -c "from liteagent.tools.schema import unvalidatable_parameters"` 打印深度常量（`_MAX_COVERAGE_DEPTH = 4`） | 影响面：手写 4 层以上嵌套 dataclass 且**内层**注解无法反射。降级本身仍写在 `ToolSpec.warnings` 里（`tools show` 可见），只是执行期不再重复提示；要彻底关闭就得给 `validate_instance` 也加上"深度超限即视为不可校验"的语义，属于规范修订 |
| G-5（**设计如此，已留痕**） | `LoopBoundPool._ALL_POOLS` 是模块级强引用列表：跑完整套件后仍会积着 ~371 个**池对象**（线程已在 `tearDownModule` 里全部关闭，实测线程数 0） | `python3 -c "import unittest,threading; ...; print(len(LoopBoundPool._ALL_POOLS))"` -> `371`（线程数 0） | 池对象本身是同步 dict，成本可忽略；线程（非 daemon）才是真代价，已归零。要消掉这 371 个对象需要让 `_ALL_POOLS` 持弱引用 —— 那是 §5.4 冻结的 `ClassVar[list[...]]`，改动面在规范侧，本轮**不做** |

> `[v3 重写]` 上一版这里写的是"G-1 是**当前唯一让主套件变红**的原因"。那句在当时是真的，
> 但那 4 份测试文件随后就落地了，台账没跟着刷新 —— 于是一条"现在时"的红灯陈述在
> `Ran 1652 tests / OK` 的现实面前成了假陈述（§5 的逐条复现命令也因此复现不出注释里的结果）。
> 现在 G-1 / G-3 都已关闭，唯一仍然成立的是 G-2（持久化的跨进程往返只做到同进程）。

## 5. 如何复现本台账（一条条照抄即可）

```bash
cd /home/ml-user/workdir/project-3

# --- 0. 环境与规模 ---
python3 -VV                                                   # 3.10.12
find liteagent -name '*.py' | wc -l                           # 41
find liteagent -name '*.py' | xargs wc -l | tail -1           # 24765 total
ls tests/test_*.py | wc -l                                    # 37（冻结清单 37，缺 0）

# --- 1. 主套件（本台账 §1 的 1652 / 1652 / 0 就是这两条命令） ---
python3 -m unittest discover -s tests -t . 2>&1 | tail -5        # Ran 1652 tests, OK（EXIT=0）
python3 -m unittest discover 2>&1 | tail -5                      # §0.3 的另一条，实测同为 Ran 1652 / OK（EXIT=0）

# --- 2. 测试方法数（与守门测试同口径，纯 ast） ---
python3 - <<'PY'
import ast, pathlib
tot = 0
for p in sorted(pathlib.Path('tests').glob('test_*.py')):
    t = ast.parse(p.read_text())
    tot += sum(1 for n in ast.walk(t)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_'))
print('AST test-method total:', tot)                          # 1652
PY

# --- 3. §12.3 每一行的落点，逐个跑（除了最后一条都是 OK） ---
python3 -m unittest tests.test_llm_providers tests.test_llm_registry            # 第 1 行
python3 -m unittest tests.test_tools_schema tests.test_tools_registry          # 第 2 行
python3 -m unittest tests.test_memory_buffer tests.test_memory_manager         # 第 3 行
python3 -m unittest tests.test_memory_vector tests.test_memory_persistence     # 第 4 行
python3 -m unittest tests.test_agent_react_text tests.test_agent_react_native  # 第 5 行
python3 -m unittest tests.test_tools_executor tests.test_llm_retry             # 第 6/7 行
python3 -m unittest tests.test_multiagent_sequential tests.test_multiagent_hierarchical  # 第 8 行
python3 -m unittest tests.test_builtin_code tests.test_builtin_files \
                   tests.test_builtin_shell tests.test_builtin_web             # 第 9 行（工具那一半）
python3 -m unittest tests.test_zero_dependency                                 # 第 10 行：Ran 34 tests / OK（G-1 已关闭）

# --- 4. CLI 与示例（手工烟测，§2.1 第 9 行的另一半） ---
python3 -m liteagent version
python3 -m liteagent tools list
python3 -m liteagent tools show read_file
python3 -m liteagent tools schema read_file --format openai
python3 examples/01_quickstart.py --offline             # EXIT=0
python3 examples/03_react_text_mode.py --offline        # EXIT=0
python3 examples/04_memory.py --offline                 # EXIT=0
python3 examples/07_code_assistant.py --offline         # EXIT=0，末尾打印 4 条断言全通过

# --- 5. 冻结接口一致性 ---
python3 scripts/check_spec_consistency.py               # error=0 warn=0 info=0（G-3 已关闭，EXIT=0）

# --- 6. 三份 benchmarks（--no-write 只打印，不改本文件） ---
python3 benchmarks/bench_dataclass_vs_pydantic.py --json --no-write
python3 benchmarks/bench_embedding_similarity.py --json --no-write
python3 benchmarks/bench_retrieval_ranking.py --json --no-write
# 去掉 --no-write 即把结果写回本文件文末的对应标记块
```

> **关于文末「实测补充」的时点**：我用 `--no-write` 跑过这三份脚本（三份都退出码 0，
> 即脚本本身可复现），但**没有**重新生成标记块 —— 文末三节的 `生成于 ... UTC` 时间戳
> 来自更早一次写入（`01:24` 那批），本节的基线数字则来自 `01:28` 那次主套件run。
> 两者是同一台机器、同一份代码，但我**不把「我重新生成过标记块」写成事实**。
> 需要刷新时，去掉 `--no-write` 重跑即可。

**注**：§0.3 要求 `python3 -m unittest discover` 与 `python3 -m unittest discover -s tests -t .`
**两条都通过**。两条我都跑了，结论一致：`Ran 1652 tests` / `OK`，**两条都是绿的（EXIT=0）**。
`[v3 重写]` 上一版这里写的是"眼下两条都是红的（EXIT=1）……G-1 的四份测试文件还没落地" ——
那份快照已过期（文件已落地、套件已转绿）。

<!--
  下面是 benchmarks 脚本自动维护的区域。上面的内容由主实现者维护。
  下面的每个小节被一对 BEGIN / END 标记包住，脚本只重写标记之间的字节，
  不会碰上面任何一行。
  （本注释里刻意不写标记原文，避免提前闭合注释。）
-->

## 实测补充（benchmarks）

<!-- BEGIN bench-dataclass-vs-pydantic (auto-generated; 由脚本覆盖，请勿手改本块) -->
### 实测补充 · D-01 核心结构用 dataclass 而非 pydantic

**落点**：`benchmarks/bench_dataclass_vs_pydantic.py`（`python3 benchmarks/bench_dataclass_vs_pydantic.py --json`）

**环境**：Python 3.10.12 ｜ pydantic 2.13.5 ｜ Linux-5.15.0-78-generic-x86_64-with-glibc2.35 ｜ 夹具 1000 条消息 ｜ repeat=20 ｜ trials=5 ｜ 生成于 2026-09-27 01:24:17 UTC

**方法**：A=dataclass、B=pydantic，**交错**跑 5 轮（每轮每侧 `repeat=20` 次取 best-of），报的是**比值的分布**而不是单次结果。判定阈值 `noise_band=1.10`：比值的整条 [min,max] 区间必须全部越过它，否则结论记为「测不出差异」。

| 阶段 | dataclass best (ms) | pydantic best (ms) | 比值中位数 | 比值区间 | 判定 |
|---|---:|---:|---:|---|---|
| 构造 1000 条消息 | 3.258 | 11.965 | 3.68x | [3.62x, 3.74x] | dataclass 更快（跨轮稳定） |
| 序列化 `to_dict()` / `model_dump()` | 1.213 | 0.970 | 0.79x | [0.76x, 0.84x] | pydantic 更快（跨轮稳定） |
| 序列化 + `json.dumps` | 3.589 | 3.344 | 0.94x | [0.90x, 1.05x] | 测不出差异（噪声带跨过 1.0） |

逐轮比值（构造）：`3.74, 3.73, 3.62, 3.64, 3.68`

逐轮比值（序列化）：`0.79, 0.76, 0.84, 0.84, 0.76`

**实测结论**

1. **构造**：dataclass 更快（跨轮稳定）（比值中位数 3.68x，区间 [3.62x, 3.74x]）——pydantic 每次构造都要做校验与 `dict` 字段浅拷贝，而 `AgentState` 的 `messages` 在 ReAct 循环里每轮都在 append。**这一格才是「可变累积语义」真正的成本所在，也是 D-01 唯一站得住的性能论据。**
2. **序列化**：pydantic 更快（跨轮稳定）（比值中位数 0.79x，区间 [0.76x, 0.84x]）。注意这一格的结论**只有靠交错测量才拿得到**：不做交错（A 跑完再跑 B）时，同一份代码在同一台机器上的比值实测在 0.71~1.71 之间乱跳，单次跑出来的方向不可复现。**所以 D-01 里原先那句「dataclass 在纯 python 侧明显更轻」在序列化这一格上不成立**——要么删掉，要么改成本节这张表。
3. **序列化 + `json.dumps`**：测不出差异（噪声带跨过 1.0）（比值中位数 0.94x）——把 `json.dumps` 也算进来后两边的差距被摊平：两侧都从 ~1.4 ms 涨到 ~3.7 ms，说明这一段的时间被 **stdlib 的 JSON 编码器**（两侧共用）吃掉了，序列化器的差异在这里不再显著。
4. **数量级**：两者都是**毫秒级**（`to_dict` 每条约 1.25 us、`model_dump` 每条约 0.98 us），而一次 LLM 往返是**百毫秒级**；把 `json.dumps` 加上之后两侧都涨到 ~3.5 ms（stdlib 的 JSON 编码器成了共同瓶颈）。

**因此 D-01 的论证要修正为**：选 dataclass 的理由是 **(a) 零第三方依赖（红线）、(b) 可变累积语义、(c) trace 字段全量可控**；性能上**只有「构造」这一格稳定支持它**（这一格恰好是热路径：ReAct 循环每轮 append 一条消息）。

序列化那一格实测是 **pydantic 略快（中位数 0.79x）**，这一点不藏：`model_dump()` 是 pydantic-core 的 Rust 递归序列化器，而 `to_dict()` 是手写 Python——**序列化本来就是 pydantic 的强项**，输给它不丢人，而且这一格根本不在瓶颈上。

面试时被追问性能，就讲这套方法：「我跑的是**交错测量**的微基准——A/B 逐轮交替、每侧取 best-of-20、跑 5 轮，看**比值的分布**而不是单次结果，并预先定了一个噪声带，比值跨过 1.0 就判『测不出差异』。结论是构造上 dataclass 稳定快约 3.6 倍，序列化上 pydantic 略快、且不在瓶颈上。所以我的选型理由是零依赖和语义匹配，不是性能。」**能说出「这一格我测不出来」，比硬报一个单次数字更可信。**

**这不是严格等价的比较 / 这是共享机器上的微基准**（脚本输出里逐条列出）：

- 不是严格等价的比较：dataclass 侧是手写 Python 的嵌套 to_dict()，pydantic 侧是 pydantic-core 的 Rust 递归序列化器（且序列化 schema 有缓存）。
- 两侧夹具的装载方式不同（Message vs PMessage），构造阶段各自单独计时。
- pydantic 的 dict 字段在构造时做浅拷贝与校验，dataclass 不做 —— 这让「构造」一格对 pydantic 不利，但对「序列化」一格无影响。
- 只测单进程 CPython 3.10 的墙钟时间，不含 GC 强制回收，也不是内存占用。
- 这是**共享机器上的微基准**。实测教训：不做交错、A 跑完再跑 B 时，序列化那一格的比值在 0.71~1.71 之间乱跳（同机同码）；改成逐轮交错后收敛到约 0.80~0.88，但在别的进程抢 CPU 时仍会退化回 'inconclusive'。所以本脚本报的是比值的 min/median/max + 一个噪声带，跨过 1.0 就明说「测不出差异」——宁可输出'不知道'，也不报一个单次跑出来的方向。
- 另一个容易骗过自己的点：`model_dump()` 的序列化 schema 只在首次调用时构建，所以 warmup 是必须的（`measure` 里固定做 2 次预热）；不预热的话 pydantic 会被冤枉成慢好几倍。

**结构指纹**（两侧输出的键集是否一致）：`shape_match = True`

- dataclass: `{"top_level_keys": ["action_counts", "agent_name", "error", "finished_at", "input", "llm_errors", "messages", "nudges", "observation_digests", "parse_errors", "run_id", "scratchpad", "started_at", "status", "step", "tool_calls", "tool_failure_counts", "tool_name_counts", "tool_results", "truncation_errors", "usage"], "message_keys": ["content", "metadata", "name", "role", "tool_call_id", "tool_calls"], "message_count": 1000}`
- pydantic : `{"top_level_keys": ["action_counts", "agent_name", "error", "finished_at", "input", "llm_errors", "messages", "nudges", "observation_digests", "parse_errors", "run_id", "scratchpad", "started_at", "status", "step", "tool_calls", "tool_failure_counts", "tool_name_counts", "tool_results", "truncation_errors", "usage"], "message_keys": ["content", "metadata", "name", "role", "tool_call_id", "tool_calls"], "message_count": 1000}`
<!-- END bench-dataclass-vs-pydantic -->
<!-- BEGIN bench-embedding-similarity (auto-generated; 由脚本覆盖，请勿手改本块) -->
### 实测补充 · D-04 Embedding 用纯 stdlib 的 Hashing Trick

**落点**：`benchmarks/bench_embedding_similarity.py`（`python3 benchmarks/bench_embedding_similarity.py --json`）

**环境**：Python 3.10.12 ｜ HashingEmbedder(dim=256) ｜ 26 句 / 13 对照对 ｜ 生成于 2026-09-27 01:24:23 UTC

**相似度矩阵**（对称，保留 2 位小数）

| | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 | S14 | S15 | S16 | S17 | S18 | S19 | S20 | S21 | S22 | S23 | S24 | S25 | S26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S01** | 1.00 | 0.68 | 0.00 | 0.00 | 0.24 | 0.22 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.57 | 0.57 | 0.24 | 0.24 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.22 | 0.00 | -0.23 | 0.20 | 0.12 |
| **S02** | 0.68 | 1.00 | 0.14 | 0.14 | 0.20 | 0.18 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.37 | 0.37 | 0.20 | 0.20 | 0.00 | 0.00 | 0.15 | 0.00 | 0.00 | 0.18 | 0.00 | -0.24 | 0.20 | 0.12 |
| **S03** | 0.00 | 0.14 | 1.00 | 0.63 | 0.00 | 0.00 | 0.07 | 0.15 | 0.00 | -0.08 | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S04** | 0.00 | 0.14 | 0.63 | 1.00 | 0.18 | 0.00 | 0.14 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S05** | 0.24 | 0.20 | 0.00 | 0.18 | 1.00 | 0.67 | 0.20 | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.22 | 0.50 | 0.50 | 0.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S06** | 0.22 | 0.18 | 0.00 | 0.00 | 0.67 | 1.00 | 0.09 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.20 | 0.22 | 0.22 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.22 | 0.00 |
| **S07** | 0.08 | 0.08 | 0.07 | 0.14 | 0.20 | 0.09 | 1.00 | 0.51 | 0.27 | 0.26 | -0.12 | -0.12 | 0.00 | 0.00 | 0.10 | 0.10 | 0.21 | 0.17 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.06 |
| **S08** | 0.00 | 0.00 | 0.15 | 0.08 | 0.11 | 0.20 | 0.51 | 1.00 | 0.24 | 0.28 | 0.00 | -0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 |
| **S09** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.27 | 0.24 | 1.00 | 0.38 | 0.00 | 0.00 | 0.00 | -0.12 | 0.00 | 0.00 | 0.20 | 0.21 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.08 |
| **S10** | 0.00 | 0.00 | -0.08 | 0.00 | 0.00 | 0.00 | 0.26 | 0.28 | 0.38 | 1.00 | -0.14 | -0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.31 | 0.20 | 0.00 | 0.00 | -0.18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 |
| **S11** | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | -0.12 | 0.00 | 0.00 | -0.14 | 1.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.16 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S12** | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | -0.12 | -0.13 | 0.00 | -0.14 | 0.67 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S13** | 0.57 | 0.37 | 0.00 | 0.00 | 0.22 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.80 | 0.22 | 0.22 | 0.00 | 0.00 | 0.34 | 0.00 | 0.00 | 0.20 | -0.32 | -0.26 | 0.22 | 0.00 |
| **S14** | 0.57 | 0.37 | 0.00 | 0.00 | 0.22 | 0.20 | 0.00 | 0.00 | -0.12 | 0.00 | 0.00 | 0.00 | 0.80 | 1.00 | 0.22 | 0.22 | 0.00 | 0.00 | 0.34 | 0.00 | 0.00 | 0.20 | 0.00 | -0.26 | 0.22 | 0.00 |
| **S15** | 0.24 | 0.20 | 0.00 | 0.18 | 0.50 | 0.22 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.22 | 1.00 | 0.75 | 0.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | 0.15 |
| **S16** | 0.24 | 0.20 | 0.00 | 0.18 | 0.50 | 0.22 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.22 | 0.75 | 1.00 | 0.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | 0.15 |
| **S17** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 | 0.40 | 0.20 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 |
| **S18** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.18 | 0.21 | 0.20 | -0.16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.64 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 |
| **S19** | 0.33 | 0.15 | 0.00 | 0.00 | 0.19 | 0.17 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.34 | 0.34 | 0.19 | 0.19 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S20** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S21** | 0.00 | 0.00 | 0.29 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.18 | 0.22 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 |
| **S22** | 0.22 | 0.18 | 0.00 | 0.00 | 0.22 | 0.20 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.20 | 0.20 | 0.22 | 0.22 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **S23** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **S24** | -0.23 | -0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.26 | -0.26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 1.00 | -0.29 | 0.00 |
| **S25** | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.22 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.29 | 1.00 | 0.00 |
| **S26** | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.07 | 0.08 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | 0.15 | 0.08 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

**人工标注的对照对与实测余弦**

| 关系 | 句 a | 句 b | 余弦 |
|---|---|---|---:|
| synonym | S01 `The cat is sitting on the mat.` | S02 `A cat sits on the mat.` | 0.6770 |
| synonym | S03 `I would like to book a flight to Paris.` | S04 `Please book me a plane ticket to Paris.` | 0.6340 |
| synonym | S05 `Please restart the server.` | S06 `Could you restart the server?` | 0.6708 |
| synonym | S07 `用户偏好使用 Python 编写后端服务。` | S08 `用户喜欢用 Python 做后端开发。` | 0.5139 |
| hard-synonym | S09 `用户需要重置密码。` | S10 `用户想把密码改掉。` | 0.3757 |
| antonym | S11 `I love programming.` | S12 `I hate programming.` | 0.6667 |
| antonym | S13 `The service is very fast.` | S14 `The service is very slow.` | 0.8000 |
| antonym | S15 `Please enable the cache.` | S16 `Please disable the cache.` | 0.7500 |
| antonym | S17 `用户喜欢这个方案。` | S18 `用户讨厌这个方案。` | 0.6445 |
| unrelated | S19 `The weather in Tokyo is rainy today.` | S20 `Quantum computing uses qubits.` | 0.0000 |
| unrelated | S21 `I need to buy milk and eggs.` | S22 `The stock market fell sharply.` | 0.0000 |
| translation | S23 `Good morning` | S24 `早上好` | 0.0000 |
| translation | S25 `Thank you very much.` | S26 `非常感谢。` | 0.0000 |

**分组统计**

| 关系 | n | mean | min | max |
|---|---:|---:|---:|---:|
| antonym | 4 | 0.7153 | 0.6445 | 0.8000 |
| hard-synonym | 1 | 0.3757 | 0.3757 | 0.3757 |
| synonym | 4 | 0.6239 | 0.5139 | 0.6770 |
| translation | 2 | 0.0000 | 0.0000 | 0.0000 |
| unrelated | 2 | 0.0000 | 0.0000 | 0.0000 |

**实测结论（⚠️ 与直觉相反）**

- 同义对平均 **0.6239**，反义对平均 **0.7153** —— **反义句得分更高**。
- 判别力 `margin = mean(synonym) - mean(antonym) = -0.0914` -> **none**。
- 同义但**词面重合极少**的改写（hard-synonym）平均只有 **0.3757**，比反义对还低。
- 跨语言翻译对平均 **0.0000**：'Good morning' 与 '早上好' 的余弦是 0.0。

机制解释：`HashingEmbedder` 是 **token 级的特征哈希**——把 token 用 `blake2b` 映射到 `dim` 维、带符号累加、`1 + log1p(count)` 加权、再 L2 归一化。它度量的是「两句共享了多少 token」。反义句几乎总是共享大部分实词（`enable`/`disable`、喜欢/讨厌），而同义改写常常整套换词（`book a flight` / `book a plane ticket`），所以前者分更高。**这不是实现 bug，是选型的固有限制。**

**能力边界（必须如实说明，不许说成「语义检索」）**

- 捕捉的是**词面（token）重叠**，不是语义：同义改写换词后分数会掉，反义句因为共享大部分实词反而得分更高。
- 跨语言零重合：'Good morning' 与 '早上好' 的余弦是 0.0；任何中英混排的语料里，翻译等价的记忆互相检索不到。
- 中文按**单字 + 相邻双字**分词，字符级重合会虚高：意思无关但用字相近的两句（如都含'用户'/'方案'）也会得到非零分。
- 对停用词敏感：'the'/'is'/'a' 也进哈希，短句之间靠虚词就能拿到分数。
- 维度 dim 只影响哈希碰撞概率与内存，不改变上述性质；调大 dim 不会让它在语义上变准。
- 结论只在**本机 CPython 3.10.12 + 该句子集**上成立；样本 26 句、对照 13 对，是演示性的，不是评测集。

**因此 D-04 的论证要修正为**：默认 `HashingEmbedder` 的价值是「零依赖 + 确定性 + 离线可跑通检索链路」，**不是**语义质量；生产要语义必须把 `RemoteEmbedder`/自研向量服务注入 `VectorMemory(embedder=...)`（v2 已冻结「注入的 embedder 的维度是权威」）。面试时把这张表拿出来讲，比说「我实现了向量检索」有说服力得多——因为它同时证明了我知道**自己方案的能力边界在哪**。
<!-- END bench-embedding-similarity -->
<!-- BEGIN bench-retrieval-ranking (auto-generated; 由脚本覆盖，请勿手改本块) -->
### 实测补充 · D-08 混合打分 + MMR 去冗

**落点**：`benchmarks/bench_retrieval_ranking.py`（`python3 benchmarks/bench_retrieval_ranking.py --json`）

**环境**：Python 3.10.12 ｜ HashingEmbedder(dim=256) ｜ 生成于 2026-09-27 01:24:24 UTC

**夹具（冻结）**：20 条记忆 = 17 条普通事实 + **3 条冗余** + **2 条过时**；查询 `用户的编程语言和数据库偏好是什么？`；`now=2023-11-14 22:13:20 UTC`（硬编码，不是真实时钟）；半衰期 7 天；MMR λ=0.7；top-k=5。

**冗余标签的证据**（三条 dup 的实测两两余弦，说明它们确实是同一事实）：

| a | b | 余弦 |
|---|---|---:|
| m00 | m01 | 0.9845 |
| m00 | m02 | 0.9386 |
| m01 | m02 | 0.9221 |

**三种策略的冻结定义**

| 策略 | w_sim | w_recency | w_importance | use_mmr |
|---|---:|---:|---:|---|
| 纯相似度 | 1.00 | 0.00 | 0.00 | False |
| 混合打分 | 1.00 | 0.15 | 0.10 | False |
| 混合 + MMR | 1.00 | 0.15 | 0.10 | True |

**逐条对照表**（按 id 逐行对齐；`sim` 三模式相同故只列一次；`#n` 是该条在**该模式全量排序**里的名次，`*` 表示落在 top-5 内）

| id | 标签 | 主题 | sim | 纯相似度 名次/分数 | 混合打分 名次/分数 | 混合 + MMR 名次/分数 |
|---|---|---|---:|---|---|---|
| m00 | dup | lang | 0.322 | #2 / 0.322 \* | #2 / 0.438 \* | #16 / 0.438 |
| m01 | dup | lang | 0.327 | #1 / 0.327 \* | #1 / 0.449 \* | #1 / 0.449 \* |
| m02 | dup | lang | 0.259 | #4 / 0.259 \* | #7 / 0.387 | #18 / 0.387 |
| m03 | stale | db | 0.299 | #3 / 0.299 \* | #6 / 0.389 | #6 / 0.389 |
| m04 | stale | infra | 0.067 | #20 / 0.067 | #18 / 0.157 | #15 / 0.157 |
| m05 | fact | db | 0.208 | #11 / 0.208 | #5 / 0.389 \* | #2 / 0.389 \* |
| m06 | fact | cache | 0.218 | #9 / 0.218 | #9 / 0.359 | #8 / 0.359 |
| m07 | fact | api | 0.231 | #7 / 0.231 | #4 / 0.392 \* | #4 / 0.392 \* |
| m08 | fact | process | 0.095 | #18 / 0.095 | #19 / 0.143 | #19 / 0.143 |
| m09 | fact | process | 0.149 | #15 / 0.149 | #16 / 0.210 | #13 / 0.210 |
| m10 | fact | style | 0.225 | #8 / 0.225 | #13 / 0.255 | #14 / 0.255 |
| m11 | fact | style | 0.078 | #19 / 0.078 | #20 / 0.111 | #20 / 0.111 |
| m12 | fact | identity | 0.249 | #5 / 0.249 \* | #3 / 0.414 \* | #3 / 0.414 \* |
| m13 | fact | team | 0.232 | #6 / 0.232 | #12 / 0.283 | #11 / 0.283 |
| m14 | fact | process | 0.150 | #14 / 0.150 | #15 / 0.224 | #10 / 0.224 |
| m15 | fact | lang | 0.208 | #12 / 0.208 | #10 / 0.350 | #9 / 0.350 |
| m16 | fact | db | 0.216 | #10 / 0.216 | #11 / 0.341 | #7 / 0.341 |
| m17 | fact | ops | 0.128 | #16 / 0.128 | #17 / 0.181 | #17 / 0.181 |
| m18 | fact | db | 0.124 | #17 / 0.124 | #14 / 0.234 | #12 / 0.234 |
| m19 | fact | perf | 0.187 | #13 / 0.187 | #8 / 0.380 | #5 / 0.380 \* |

**top-5 明细**

- **纯相似度**（use_mmr=False）：`m01 → m00 → m03 → m02 → m12`
  - #1 `m01`（dup, sim=0.327, rec=0.410, score=0.327）用户偏好用 Python 写后端服务，主力语言是 Python
  - #2 `m00`（dup, sim=0.322, rec=0.371, score=0.322）用户偏好用 Python 写后端服务，主力语言是 Python。
  - #3 `m03`（stale, sim=0.299, rec=0.000, score=0.299）用户的数据库是 MySQL 5.7，一直在用。
  - #4 `m02`（dup, sim=0.259, rec=0.453, score=0.259）用户偏好用 Python 写后端服务，主力语言为 Python。
  - #5 `m12`（fact, sim=0.249, rec=0.500, score=0.249）用户的名字是 Alice。
- **混合打分**（use_mmr=False）：`m01 → m00 → m12 → m07 → m05`
  - #1 `m01`（dup, sim=0.327, rec=0.410, score=0.449）用户偏好用 Python 写后端服务，主力语言是 Python
  - #2 `m00`（dup, sim=0.322, rec=0.371, score=0.438）用户偏好用 Python 写后端服务，主力语言是 Python。
  - #3 `m12`（fact, sim=0.249, rec=0.500, score=0.414）用户的名字是 Alice。
  - #4 `m07`（fact, sim=0.231, rec=0.673, score=0.392）用户偏好用 FastAPI 写 HTTP 接口。
  - #5 `m05`（fact, sim=0.208, rec=0.743, score=0.389）用户现在用 PostgreSQL 做生产数据库。
- **混合 + MMR**（use_mmr=True）：`m01 → m05 → m12 → m07 → m19`
  - #1 `m01`（dup, sim=0.327, rec=0.410, score=0.449）用户偏好用 Python 写后端服务，主力语言是 Python
  - #2 `m05`（fact, sim=0.208, rec=0.743, score=0.389）用户现在用 PostgreSQL 做生产数据库。
  - #3 `m12`（fact, sim=0.249, rec=0.500, score=0.414）用户的名字是 Alice。
  - #4 `m07`（fact, sim=0.231, rec=0.673, score=0.392）用户偏好用 FastAPI 写 HTTP 接口。
  - #5 `m19`（fact, sim=0.187, rec=0.820, score=0.380）用户的目标是把延迟降到 50ms 以下。

**量化指标**

| 策略 | top-5 里的冗余条数 | top-5 里的过时条数 | top-5 主题覆盖 | top-5 内最大两两余弦 |
|---|---:|---:|---:|---:|
| 纯相似度 | 3 | 1 | 3 | 0.985 |
| 混合打分 | 2 | 0 | 4 | 0.985 |
| 混合 + MMR | 1 | 0 | 5 | 0.405 |

**两个过时条目的名次变化**（名次越小越靠前；`-` 表示该模式下它排在最后之外）

| 策略 | m03（MySQL 5.7，200 天前） | m04（Ubuntu 16.04，220 天前） |
|---|---:|---:|
| 纯相似度 | #3 | #20 |
| 混合打分 | #6 | #18 |
| 混合 + MMR | #6 | #15 |

**实测结论**

1. **冗余挤占**：纯相似度的 top-5 里有 **3/5** 条是同一事实（三条写法彼此余弦 0.92~0.98）——**超过一半的上下文预算被浪费**，而这正是 D-08 说的「同一事实写入 5 次占满 top-5」。
2. **过时冒充**：200 天前那条 MySQL 事实在纯相似度下排 **#3**（进了 top-5），混合打分的近因衰减（半衰期 7 天 -> recency=0.000）把它踢出 top-5。
3. **MMR 去冗**：`hybrid_mmr` 把 top-5 里的冗余从 2 条压到 **1 条**，主题覆盖从 4 提到 **5**，组内最大两两余弦从 0.985 降到 **0.405**。
4. **权重仍然是相似度主导**：`w_sim=1.0` vs `w_recency=0.15` / `w_importance=0.1`，所以时间只用来**打破接近分数的平局**，而不是主导排序——表里 hybrid 与 pure 的前两名完全一致就是证据。

**诚实边界（不许把这张表说成「检索质量提升」）**

- 默认 embedder 是 HashingEmbedder（词面哈希）：它对'数据库偏好'的召回靠的是token 重合，所以本表只能证明**打分公式与去冗算法的行为**，不能证明检索质量。换成真实的语义 embedder，sim 列会整体变化，但 recency/MMR 的机制不变。
- '过时'与'冗余'是我**人工标注**的（stale = 200/220 天前且已被新事实取代；dup = 同一事实的三种写法）。标签是这张表的真值来源，不是我测出来的结论；脚本把 dup 两两余弦与 stale 在各模式下的名次一并打印，供读者自行核对。
- `now` 是硬编码的 1700000000（2023-11-14 UTC），不是真实时钟：recency 是连续衰减，用真实时钟会让这张表不可复现。
- 只测 20 条记忆、1 条查询、top-5。样本是演示性的，不是检索评测集。

**面试怎么讲**：把这三行 `top_ids` 拿出来，先说失效（3/5 是同一件事、200 天前的旧数据库排第 3），再说修法（半衰期衰减 + MMR），最后说**代价**（recency 让结果随时间漂移，所以要 `search(now=...)` 注入；MMR 是 O(k²)、k ≤ 15 可忽略）。面试官问的从来不是「你用了什么算法」，而是「你怎么知道它坏了、你怎么知道它修好了」——这张表就是答案。
<!-- END bench-retrieval-ranking -->
