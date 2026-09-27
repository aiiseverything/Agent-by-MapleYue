# liteagent 冻结接口规范 (INTERFACES.md)

> **状态：FROZEN v2.0** —— 本文件是多个并行实现者之间的唯一契约。
> 实现者只读本文件，不得修改任何已冻结的 import 路径、类名、字段名、方法签名、常量名与常量值。
> **本文件已并入 5 个视角的对抗性审查（async-concurrency / testability-offline / interview-value /
> impl-consistency / resume-coverage）的全部 blocker 与 major 结论。v2.0 与 v1.0 不兼容的地方在
> 文中以 `[v2 变更]` 标注，实现者必须先读完 §0 与 §2。**
> 若发现规范内部矛盾，**不要自行发明**：在实现文件中写 `# SPEC-AMBIGUITY: <描述>`，
> 并采用本文件的字面规定。
> 每个实现者开始前先读 §1（依赖 DAG）、§2（全局约定）、§0（环境与实测事实），再读自己负责那一节。

---

## 0. 环境事实（实测，非推测）

| 项 | 实测值 |
|---|---|
| 工作目录 | `/home/ml-user/workdir/project-3`（**所有命令必须在仓库根运行**） |
| Python | 3.10.12（`/opt/venv/main/bin/python3`） |
| 可用三方库 | `pydantic 2.13.5`, `numpy 1.26.4`, `requests`, `httpx`, `PyYAML`, `jinja2`, `rich`, `typer` |
| **不可用** | `pytest`, `pytest-asyncio`, `openai`, `anthropic`, `tiktoken`, `faiss`, `python-dotenv` |
| 网络 | **无外网**，`pip install` 超时 |

### 0.1 Python 3.10 缺失 API（实测 MISSING，禁止使用）

```
asyncio.TaskGroup        X   -> 用 asyncio.gather(..., return_exceptions=True)
asyncio.timeout()        X   -> 用 asyncio.wait_for(coro, timeout)
asyncio.Runner           X   -> 用 asyncio.run()
typing.Self              X   -> 用字符串字面量 "ClassName"
datetime.UTC             X   -> 用 datetime.timezone.utc
enum.StrEnum             X   -> 用 class X(str, Enum)
tomllib                  X   -> 用 PyYAML 适配器或 json
ExceptionGroup           X   -> 用 list[BaseException] 手工聚合
```

### 0.2 已实测存在的 API（可用）

`asyncio.to_thread`、`asyncio.Semaphore`、`asyncio.Lock`、`asyncio.Event`、`asyncio.Condition`、
`typing.Literal`、`typing.get_origin`、`typing.get_args`、`typing.get_type_hints(include_extras=True)`、
`dataclasses.dataclass(slots=True)`、`functools.cache`、`hashlib.blake2b`、`enum` str-mixin、
`unittest.IsolatedAsyncioTestCase`、`subprocess.run(timeout=...)` 抛 `subprocess.TimeoutExpired`。

### 0.2.1 五条必须记住的实测结论（本版新增，全部在本机 3.10.12 复现过）

| # | 实测结论 | 对实现的硬约束 |
|---|---|---|
| M-1 | `isinstance(Annotated[int, 'x'], Annotated)` == **False**（不报错，静默 False）；`get_origin(...) is Annotated` == True；`hasattr(ann, "__metadata__")` == True | schema 反射**禁止** `isinstance(ann, Annotated)`，见 §7.1.5 |
| M-2 | `asyncio.to_thread(async_fn, args)` **不会执行** async_fn，它立即返回一个 coroutine 对象 | 同步分支**必须**调用同步入口 `Tool.run`，见 §7.4.1 步骤 5.b |
| M-3 | `asyncio.Semaphore` 一旦发生争用就把 loop 存进 `self._loop`，即**原语强引用 loop**；`WeakKeyDictionary[loop]` 的 key 因此永不失效（实测 3 次 `asyncio.run` 泄漏 3 个 loop 与 3 个原语） | `LoopBoundPool` 必须显式 `clear(loop)`，见 §5.4 |
| M-4 | `loop.call_soon_threadsafe(...)` 对**已关闭**的 loop 抛 `RuntimeError: Event loop is closed` | 唤醒 watcher 前必须 `if loop.is_closed(): continue` + try/except，见 §10.2.1 |
| M-5 | `asyncio.CancelledError` 继承 `BaseException`（不是 `Exception`）；`asyncio.TimeoutError is not builtins.TimeoutError`；`contextvars` **读**能穿透 `to_thread`（看到调用方的值），**写**不能回传 | 取消规则见 §7.4.1，contextvar 写入位置见 §7.4.1 步骤 5.c |

### 0.3 测试运行方式（两条命令都必须通过）

```bash
cd /home/ml-user/workdir/project-3
python3 -m unittest discover            # 默认发现（需要 tests/__init__.py）
python3 -m unittest discover -s tests -t . -v
```

**禁止 pytest 风格**：不得用裸 `assert` 作为断言载体（必须 `self.assertEqual` 等）、
不得用 `@pytest.fixture` / `@pytest.mark.asyncio`。异步测试一律继承 `unittest.IsolatedAsyncioTestCase`。

### 0.4 最重要的运行时陷阱 R-LOOP（实测复现，务必遵守）

`asyncio.Semaphore` / `Lock` / `Condition` 在 **3.10 的争用路径**上会绑定事件循环：

```python
sem = asyncio.Semaphore(1)          # 在 __init__ 里创建，此时没有 loop
async def use():
    async with sem:                 # 一旦发生争用 -> _get_loop() -> 绑定 loop
        await asyncio.sleep(0.01)
asyncio.run(gather(use(), use()))   # 第一次 OK
asyncio.run(gather(use(), use()))   # RuntimeError: ... is bound to a different event loop
```

**冻结规则 R-LOOP**：任何 `asyncio` 同步原语**禁止**在 `__init__` / 模块级创建。必须通过
§5.4 的 `LoopBoundPool`（懒创建）在**运行中的 loop 内**取得。`Agent.run()` /
`ToolExecutor.execute_sync()` 每次调用都会 `asyncio.run()`，违反 R-LOOP 必然在第二次同步调用时崩溃。

**R-LOOP 的适用边界（冻结，回答"哪些状态可以放 `__init__`"）**：
- **禁止**在 `__init__` 创建：`asyncio.Semaphore/Lock/Condition/Event/Queue`。
- **允许**在 `__init__` 创建：`threading.Lock/RLock/Semaphore/Event`、`functools` 缓存、
  普通 dict/list、`ThreadPoolExecutor`（但必须由持有者负责 `shutdown`，见 §5.4）。
  `threading` 原语与事件循环无关，这是它们被大量使用的理由（D-12）。

### 0.5 术语表（冻结，避免"窗口/历史/trace"混指）

| 术语 | 精确定义 | 载体 |
|---|---|---|
| **transcript** | 一次 run 内**全部**消息，含被裁剪掉的 | `AgentState.messages` |
| **window** | 实际送进模型的那批消息（已裁剪） | `BufferMemory.window()` |
| **buffer** | 跨 run 累积的短期记忆（未裁剪） | `BufferMemory._messages` |
| **evicted** | 被 window 裁掉、等待摘要的消息 | `BufferMemory.evicted()` |
| **state** | 单次 run 的全部状态 | `AgentState` |
| **trace** | 事件流（JSONL 或内存列表） | `TraceEvent` |
| **observation** | 工具结果回灌给模型的文本载荷 | `ToolResult.to_observation()` |
| **pool** | 按 loop 懒创建的 asyncio 原语/线程池容器 | `LoopBoundPool`（§5.4） |

---

## 1. 文件清单、职责边界与依赖方向

### 1.1 依赖 DAG（唯一允许的 import 方向）

**规则（冻结）**：**包内自由（同包任意模块可互相 import）；包间按下表，只允许向下。**
`liteagent/__init__.py` 允许 import 任意 `liteagent.*`；`X/__init__.py` 允许 import 本包任意模块。

```
L0  errors.py            <- 无 liteagent 内部依赖
L1  types.py             <- errors, (函数体内延迟) llm/message   [见 例外 E1]
L1  config.py            <- errors, types
L2  llm/message.py       <- errors, types
L2  llm/transport.py     <- errors, config
L2  llm/base.py          <- errors, types, config, llm/message
L2  llm/providers.py     <- errors, config, llm/{message,base,transport}
L2  llm/registry.py      <- errors, config, llm/base            (providers 用函数内延迟导入)
L2  llm/scripted.py      <- errors, types, config, llm/{message,base}
L3  memory/{base,embeddings,buffer,summary,vector,manager}.py <- errors, types, config, llm/*
L3  tools/{schema,base,registry,executor}.py                  <- errors, types, config
L3  tools/builtin/*.py                                        <- errors, types, config, tools/*,
                                                                  memory/base(仅 memory_tools)
L4  agent/{state,callbacks,parser,agent}.py                   <- 上述全部
L5  multiagent/{base,blackboard,sequential,hierarchical}.py   <- agent/*, 上述全部
L6  cli.py                                                    <- 全部
```

**依赖只允许向下，禁止任何反向 import。**

**同层边白名单（`ALLOWED_SAME_LEVEL_EDGES`，冻结清单，逐条写进 `test_zero_dependency.py`）**：

| # | 边 | 理由 |
|---|---|---|
| E1 | `types.py` -> `llm/message.py`（**仅允许函数体内延迟 import**） | `ToolResult.to_message` / `LLMResponse.to_message` 的返回类型是 `Message`。写法冻结为：`def to_message(self) -> "Message":\n    from liteagent.llm.message import Message, Role\n    return Message(...)`。`test_zero_dependency` 只看顶层 import，函数体内 import 天然放行 |
| E2 | `llm/registry.py` -> `llm/providers.py`（函数体内延迟） | 避免 import 注册表就拉起所有 provider |
| E3 | `multiagent/{sequential,hierarchical}.py` -> `multiagent/base.py` | ABC 继承；`base.py` 不得反向 import（`build_team` 用函数内延迟 import） |
| E4 | `tools/base.py` <-> `tools/schema.py` | `base` 调 `build_tool_schema`，`schema` 的 `to_openai_tool(spec: "ToolSpec")` 要 `ToolSpec`（用字符串注解 + `TYPE_CHECKING`） |
| E5 | `tools/executor.py` -> `tools/{base,registry,schema}` | 执行器需要三者 |
| E6 | `memory/manager.py` -> `memory/{base,buffer,vector,summary}` | 编排层 |
| E7 | `memory/{buffer,summary,vector}.py` -> `memory/{base,embeddings}` | 依赖基类与 embedder |
| E8 | `tools/builtin/{shell,code}.py` -> `tools/builtin/files.py` | `PathSandbox` 定义在 `files.py`，被另两个 factory 引用 |
| E9 | `tools/builtin/__init__.py` -> `tools/builtin/*` | 注册函数需要各 factory 的函数对象（§7.5） |
| E10 | `tools/builtin/memory_tools.py` -> `memory/manager.py`（注解用 `TYPE_CHECKING`，运行期 duck-typing） | `make_memory_tools(memory)` |
| E11 | `multiagent/base.py` -> `multiagent/blackboard.py` | `blackboard.py` 的真实层级是 **L1**（只依赖 errors/types/config），因此**不是同层边**；但为消除歧义，仍显式列入本表 |
| E12 | `memory/manager.py` -> `memory/embeddings.py` | `from_config` 造默认 embedder |

**`[v2 变更]`** `LoopBoundPool` 的归属从 `tools/executor.py` 挪到 **`config.py`**（它是零依赖纯工具，
L1 已就位），因此 `multiagent/blackboard.py` 与 `tools/executor.py` 都写
`from liteagent.config import LoopBoundPool`，`blackboard.py` 的依赖层级保持 L1。

**验证方式**：`test_zero_dependency.py` 里用 `ast` 解析所有模块的**顶层** import
并断言跨包边只出现在本表 + 上面的 L 编号表里；同层边只允许 E1..E12。
白名单以**本表为唯一真值源**，逐条转抄进测试文件（不新增数据模块 —— 新增 `.py` 会违反 §1.2 封闭清单）。

### 1.2 完整文件清单与职责边界（封闭清单，共 41 个 `.py`）

> `[v3 变更]` 最后一列从 v2 的"**预估**行数"改为"**实测**行数"：v2 的预估与落地后的
> 仓库普遍差 2 倍以上（例如 `errors.py` 预估 280 / 实测 1031，`agent.py` 预估 700 / 实测
> 1249），对照着读会显得规范与代码对不上。行数为 `wc -l`（含注释与空行）。

| # | 文件 | 职责（写什么、不写什么） | 实测行数 |
|---|---|---|---|
| 1 | `liteagent/__init__.py` | 公共 API re-export + `__version__`。**不写任何逻辑** | 184 |
| 2 | `liteagent/errors.py` | 全部异常类 + `retryable` 语义。**不 import 任何 liteagent 模块** | 1031 |
| 3 | `liteagent/types.py` | `TokenUsage`/`ToolCall`/`ToolResult`/`LLMResponse`/`ScriptedCall`。**纯 stdlib + E1 延迟边** | 705 |
| 4 | `liteagent/config.py` | **跨层**配置 dataclass + 跨层共享纯函数（`RetryPolicy`/`compute_backoff`/`run_sync`/`LoopBoundPool`/`utc_now`/`dotenv`/`to_jsonable`/`estimate_cost_usd`）。**不放业务逻辑** | 1429 |
| 5 | `liteagent/llm/__init__.py` | re-export（清单见 §1.4） | 30 |
| 6 | `liteagent/llm/message.py` | `Role`、`Message` 及其序列化/构造便捷方法 | 348 |
| 7 | `liteagent/llm/base.py` | `LLMClient` ABC、`LLMStreamChunk`、`BaseLLMClient`（通用重试/事件/token 计数） | 359 |
| 8 | `liteagent/llm/transport.py` | `HTTPRequest`/`HTTPResponse`/`Transport` ABC + urllib/requests/httpx 适配器 + 错误映射 | 582 |
| 9 | `liteagent/llm/providers.py` | `HTTPChatClient` 基类、`OpenAIChatClient`、`OpenAICompatibleClient`、`DeepSeekChatClient`、`AnthropicChatClient`、`EchoLLM`、编解码 | 889 |
| 10 | `liteagent/llm/registry.py` | `LLMRegistry`、`build_llm(LLMConfig)`、`get_llm(str)` | 198 |
| 11 | `liteagent/llm/scripted.py` | `ScriptedLLM`、`ScriptedResponse`、`ScriptedExhaustedError` 用法 | 659 |
| 12 | `liteagent/tools/__init__.py` | re-export（清单见 §1.4） | 87 |
| 13 | `liteagent/tools/schema.py` | type-hints -> JSON Schema 反射、docstring 解析、`validate_instance` stdlib 校验器 | 1455 |
| 14 | `liteagent/tools/base.py` | `ToolSpec`、`Tool`、`@tool` 装饰器、`make_function_tool`、取消通道 | 584 |
| 15 | `liteagent/tools/registry.py` | `ToolRegistry`（注册/查询/别名/子集/schema 导出/文本描述） | 462 |
| 16 | `liteagent/tools/executor.py` | `ExecutorConfig`、`ToolExecutor`（审批/校验/并发/超时/重试/截断/取消） | 1282 |
| 17 | `liteagent/tools/builtin/__init__.py` | `register_all(...)`、`BUILTIN_TOOL_GROUPS`、`BUILTIN_TOOL_NAMES` | 261 |
| 18 | `liteagent/tools/builtin/files.py` | `PathSandbox` + 文件类工具（闭包注入沙箱，`save/load` 无关） | 588 |
| 19 | `liteagent/tools/builtin/shell.py` | `SHELL_DENY_PATTERNS`、`check_command_allowed`、`run_shell`、`make_shell_tools`。**不含 `python_exec`**（在 code.py） | 295 |
| 20 | `liteagent/tools/builtin/code.py` | `python_eval`(AST 白名单)、`python_exec`(子进程)、`run_tests` | 963 |
| 21 | `liteagent/tools/builtin/web.py` | `SearchBackend` 抽象 + `web_search` + `fetch_url` + 纯 stdlib HTML 转文本 | 765 |
| 22 | `liteagent/tools/builtin/memory_tools.py` | `make_memory_tools(memory)` -> `remember`/`recall` | 161 |
| 23 | `liteagent/memory/__init__.py` | re-export（清单见 §1.4） | 56 |
| 24 | `liteagent/memory/base.py` | `MemoryItem`、`MemoryStore` ABC、`MemoryConfig`、`Tokenizer` 家族 | 499 |
| 25 | `liteagent/memory/embeddings.py` | `Embedder` ABC + Hashing/Numpy/RandomProjection/Callable/Remote | 548 |
| 26 | `liteagent/memory/buffer.py` | `BufferMemory` 滑动窗口 + 裁剪算法 + `_repair_tool_pairs` | 389 |
| 27 | `liteagent/memory/summary.py` | `SummaryMemory` 触发判定 + LLM 摘要 + 抽取式兜底 | 303 |
| 28 | `liteagent/memory/vector.py` | `VectorMemory` 写入策略/混合打分/MMR/余弦/`save`/`load` | 795 |
| 29 | `liteagent/memory/manager.py` | `MemoryManager` 编排 prompt 组装、三层协作、`persist`/`restore` | 946 |
| 30 | `liteagent/agent/__init__.py` | re-export（清单见 §1.4） | 45 |
| 31 | `liteagent/agent/state.py` | `AgentState`、`AgentStatus`、`AgentResult`；re-export `AgentConfig`（**类定义在 config.py**） | 577 |
| 32 | `liteagent/agent/callbacks.py` | `EventType`、`TraceEvent`、`Callback`、`CallbackManager`、内置回调、`TraceRecorder`、`trace_stats` | 1200 |
| 33 | `liteagent/agent/parser.py` | `ReActParser`、`ParsedAction`、文本 ReAct 语法与容错 | 815 |
| 34 | `liteagent/agent/agent.py` | `Agent`、ReAct 状态机、两种模式统一、自纠正、重复/无进展检测、预算与截断 | 1249 |
| 35 | `liteagent/multiagent/__init__.py` | re-export（§10.5） | 29 |
| 36 | `liteagent/multiagent/base.py` | `TeamConfig`、`MultiAgent` ABC、`DelegationContext`、结果压缩 | 419 |
| 37 | `liteagent/multiagent/blackboard.py` | `Blackboard`、`BlackboardEntry`、并发安全、watch/版本冲突 | 708 |
| 38 | `liteagent/multiagent/sequential.py` | `SequentialAgent`、`SequentialStep` | 360 |
| 39 | `liteagent/multiagent/hierarchical.py` | `HierarchicalAgent`、`Plan`/`SubTask`、delegate 工具、环检测、`arun_plan` | 1144 |
| 40 | `liteagent/cli.py` | argparse 子命令（run/tools/chat/trace/schema/multi/version） | 1203 |
| 41 | `liteagent/__main__.py` | **允许新增的唯一文件**：逐字内容见下 | 6 |

**`[v2 变更]` `liteagent/__main__.py` 的冻结逐字内容（6 行）**：

```python
from __future__ import annotations

from liteagent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

**根目录交付物（封闭清单）**：`pyproject.toml`、`README.md`、`Makefile`、`.gitignore`。
**`docs/`（封闭清单，7 个文件）**：`INTERFACES.md`（本文件）、`DESIGN_DECISIONS.md`、
`ARCHITECTURE.md`、`TOOLS.md`、`INTERVIEW.md`、`VERIFICATION.md`、`BUILD_LOG.md`。
**`tests/`**：见 §12（含 §12.1 的 `helpers.py` 冻结签名）。
**`examples/`（8 个）**：`01_quickstart.py`、`02_tools_custom.py`、`03_react_text_mode.py`、
`04_memory.py`、`05_multiagent_sequential.py`、`06_multiagent_hierarchical.py`、
`07_code_assistant.py`、`run_all_examples.py`。
**`benchmarks/`（3 个，封闭清单）**：`bench_dataclass_vs_pydantic.py`、`bench_embedding_similarity.py`、
`bench_retrieval_ranking.py`（职责见 §12.4；每个脚本把结果写入 `docs/VERIFICATION.md`）。
**`scripts/`**：允许存在 `scripts/`，但只允许放 `check_spec_consistency.py`（§12.4），
不得放与规范无关的文件。

**`pyproject.toml` / `Makefile` 的冻结值**：
- `version = "0.1.0"`；`requires-python = ">=3.10"`。
- `__version__` 的唯一来源是 `importlib.metadata.version("liteagent")`，失败时回退 `"0.1.0"`。
- `Makefile` 的 `demo` / `demo-live` 目标统一引用 `examples/07_code_assistant.py`；
  `test` 目标串联 §0.3 的两条命令。

### 1.3 零依赖红线

- `liteagent/` 下**所有**模块的顶层 import 只能是 stdlib 或 `liteagent.*`。
- 三方库只能出现在 `llm/transport.py`（requests/httpx）、`memory/embeddings.py`（numpy）、
  `config.py`（yaml）、`cli.py`（rich，可选）。

**"顶层 import"的精确定义（`test_zero_dependency.py` 按此实现，用 `ast` 而非正则）**：
遍历模块的 `ast.Module.body`，一个 `ast.Import`/`ast.ImportFrom` 语句算"顶层"，
**除非**它（含任意深度）位于 `ast.Try` 节点内 —— `try: import numpy / except ImportError:`
在 AST 语义上等价于"可选依赖"，必须被放行。
此外，位于 `ast.FunctionDef`/`ast.AsyncFunctionDef`/`ast.ClassDef`/`ast.If` 体内的 import 也不算顶层。
`ast.If` 被放行是因为 `if TYPE_CHECKING:` 与 `if REQUESTS_AVAILABLE:` 这两种惯用写法。
**未被放行的形态**：裸的 `import numpy`、`from pydantic import BaseModel` 直接写在模块顶部。

- 三方 import 一律写成如下形式（**顶层不得裸 import**）：

```python
try:                      # pragma: no cover - 环境相关
    import numpy as _np
    NUMPY_AVAILABLE = True
except ImportError:       # pragma: no cover
    _np = None
    NUMPY_AVAILABLE = False
```

并提供同名可用性常量：`NUMPY_AVAILABLE`、`REQUESTS_AVAILABLE`、`HTTPX_AVAILABLE`、`YAML_AVAILABLE`、
`RICH_AVAILABLE`、`TIKTOKEN_AVAILABLE`、`PYDANTIC_AVAILABLE`。**这些常量必须在模块顶层定义，供测试断言。**

**`[v2 变更]` 3.11 API 禁用检查用 AST，不用文本子串**（否则把 §0.1 抄进注释就会误报）。
冻结判定方式：遍历 AST，命中断言失败的条件是
`(ast.Attribute, attr in {"TaskGroup","Runner","timeout"}) | (ast.Name, id in {"TaskGroup","Runner"}) |
(ast.ImportFrom, module in {"tomllib"} or name in {"StrEnum","Self","UTC","ExceptionGroup"}) |
(ast.Attribute, attr in {"Self","UTC","StrEnum","ExceptionGroup"})`。
**明确冻结：docstring 与注释里的 API 名不触发失败；禁止文本子串匹配。**

### 1.4 各包 `__all__` 冻结清单（`[v2 变更]` —— 逐字实现，附录 B 的守门测试按此断言）

**公开 API 白名单由 `__init__.py` 的 `__all__` 显式列出，未列出的视为内部。**
`__all__` 里每个名字必须能从该包 import 到（守门用例见 §12）。

```python
# liteagent/llm/__init__.py
__all__ = [
    "LLMClient", "BaseLLMClient", "LLMStreamChunk", "Message", "Role", "LLMResponse",
    "LLMConfig", "ScriptedLLM", "ScriptedResponse", "ScriptedCall", "LLMRegistry",
    "build_llm", "get_llm", "messages_tokens", "render_transcript",
    "drop_orphan_tool_messages",
]

# liteagent/tools/__init__.py
__all__ = [
    "tool", "Tool", "ToolSpec", "ToolRegistry", "ExecutorConfig", "ToolExecutor",
    "make_function_tool", "is_tool", "current_cancel_flag", "cancel_scope",
    "get_default_registry", "reset_default_registry",
    # builtin 的入口在此 re-export，便于 `from liteagent.tools import register_all`
    "register_all", "BUILTIN_TOOL_NAMES", "BUILTIN_TOOL_GROUPS",
]

# liteagent/memory/__init__.py
__all__ = [
    "MemoryItem", "MemoryStore", "MemoryConfig", "Tokenizer", "HeuristicTokenizer",
    "CallableTokenizer", "get_default_tokenizer",
    "Embedder", "HashingEmbedder", "NumpyHashingEmbedder", "RandomProjectionEmbedder",
    "CallableEmbedder", "RemoteEmbedder", "default_embedder", "cosine_similarity",
    "cosine_similarity_matrix",
    "BufferMemory", "BufferConfig", "SummaryMemory", "SummaryConfig",
    "VectorMemory", "VectorConfig", "MemoryManager",
]

# liteagent/agent/__init__.py
__all__ = [
    "Agent", "AgentConfig", "AgentResult", "AgentState", "AgentStatus",
    "EventType", "TraceEvent", "Callback", "CallbackManager", "CallbackLike",
    "FunctionCallback", "LoggingCallback", "JsonlTraceCallback", "RichCallback",
    "TokenCounterCallback", "MemoryTraceCallback", "TraceRecorder",
    "load_trace", "total_usage", "events_of_type", "render_trace", "as_llm_callback",
    "trace_stats",
    "ReActParser", "ParsedAction",
]

# liteagent/multiagent/__init__.py  ->  见 §10.5
```

**`liteagent/__init__.py` 的 `__all__`** 见附录 B。**守门用例**（`[v3 变更]` 按实现修正）：
对附录 B 里每个名字 `getattr(liteagent, name)` 必须成功
（`tests/test_zero_dependency.py::PublicApiSurfaceTests::test_appendix_b_names_are_all_importable_from_top_level`），
且 `liteagent.__all__` 与附录 B **双向相等**
（`::test_top_level_all_matches_appendix_b`，同时断言 `len(APPENDIX_B_NAMES) == 120`）。

`[v3 变更]` v2 这里还写了一条件
`liteagent.llm.__all__ ⊂ (能从 liteagent top-level 拿到的名字)` —— **该条件实际不成立**：
`liteagent.llm.__all__` 里的 `messages_tokens` / `render_transcript` / `drop_orphan_tool_messages`
**不在**顶层 `liteagent.__all__`（顶层是精选导出，附录 B 的 120 个名字）。要求
"子包 `__all__ ⊆ 顶层"既与附录 B 冲突，也会让顶层命名空间被内部 helper 污染。
实现侧（`test_subpackage_all_names_are_resolvable`）改为断言**每个子包的 `__all__` 里的名字
能从该子包取到**，顶层可解析性由上面那条附录 B 用例单独负责。

---

## 2. 全局约定（所有实现者必须一致）

### 2.1 文件头

每个 `.py` 文件第一行必须是：

```python
from __future__ import annotations
```

（统一使用，使 `list[str]` / `X | None` 在注解位置无条件可用。）

### 2.2 序列化约定

- 所有对外数据结构都提供 `to_dict(self) -> dict[str, Any]`。
- 提供 `from_dict(cls, data: Mapping[str, Any]) -> Self`（返回类型注解写 `"ClassName"` 字符串）。
- `to_dict()` 的输出必须是 **JSON 可序列化**的：`Enum -> .value`、`set/tuple -> list`、
  `datetime/float 时间 -> float`（Unix 秒）、`dataclass -> dict`、`Path -> str`、`bytes -> base64 str`。
  由 `config.to_jsonable(obj)` 统一处理。
- **字段全量输出**：`to_dict()` **必须**输出全部字段（包括 `None`），
  以保证 trace 结构稳定、测试可用 `assertEqual(expected_dict, got.to_dict())` 精确比对。
- **`[v2 变更]` 唯一的显式 opt-out 例外**（写进 docstring）：
  `MemoryItem.to_dict(include_embedding: bool = False)` **默认不输出 `embedding`**
  （256 维 float 列表会让 trace/CLI/JSONL 膨胀数十倍）；
  需要 embedding 时显式 `include_embedding=True`（持久化往返用 True）。
  `LLMResponse.to_dict(include_raw=False)` 与 `AgentResult.to_dict(include_state=False)`
  是另外两个 opt-out。**除这三个显式开关外，字段一律全量输出。**
- `from_dict` 必须**容忍**缺失的 opt-out 字段（如缺 `embedding` 时置 `None`，**不重算**）。
- **所有时间统一为 `float`（Unix 秒）**，不使用 `datetime` 对象作为字段类型。
  需要可读时间时才用 `config.format_ts(ts)`。
- **`[v2 变更]` 唯一时钟**：`config.utc_now()`。L1 及以上所有模块（config/llm/memory/tools/agent/
  multiagent）里凡是"取当前时间"的地方（含所有 `default_factory`）**必须**用
  `field(default_factory=utc_now)`（或调用 `utc_now()`），**禁止** `time.time` / `datetime.now`。
  `types.py` 无时间字段，不受此约束。测试用 `tests.helpers.frozen_time(ts)`（§12.1）固定时钟。

### 2.3 命名约定

| 场景 | 约定 |
|---|---|
| 异步方法 | 前缀 `a`：`achat`、`arun`、`aadd`、`aexecute` |
| 同步镜像 | 同名去掉 `a`；**默认**先定义 async，同步版是薄包装（规则见下 "sync-first 例外"） |
| 可能返回 None 的查询 | `try_get` / `try_*` 前缀；不加前缀的 `get` 失败即抛异常 |
| 配置 dataclass | 后缀 `Config`，字段全部有默认值（保证 `XxxConfig()` 可无参构造） |
| 事件/枚举 | `class Xxx(str, Enum)`，成员用大写下划线 |
| 私有 | 单下划线前缀；模块级常量全大写下划线 |
| 公开 API 白名单 | 由 `__init__.py` 的 `__all__` 显式列出，未列出的视为内部 |
| 单下划线的含义 | 只表示"非公开 API"，**不表示不可测**。测试可以直接 import 单下划线成员 |

**`[v2 变更]` 双 API 的冻结规则（sync 包装 async）**：

1. **默认方向（async-first）**：先定义 `a*`，同步版是薄包装。
   `LLMClient.chat`、`ToolExecutor.execute_sync`、`MemoryManager.build_prompt`、
   `Agent.run`、`MultiAgent.run` 的**唯一**实现形态是：

   ```python
   def chat(self, messages, **kwargs) -> LLMResponse:
       return run_sync(lambda: self.achat(messages, **kwargs))
   ```

2. **`sync-first 例外（冻结）`**：以下模块以**同步实现为规范实现**，async 版是薄包装，
   **方向不许反**（理由见 D-12：worker 线程里没有事件循环）：
   `Blackboard`、`MemoryStore`/`Tokenizer`、`BufferMemory`、`LoopBoundPool` 本身的同步方法。
   `Blackboard.awrite` 等 async 版本的实现是 `await asyncio.to_thread(self.write, ...)` 或直接调用同步方法。
3. `run_sync` 的签名见 §5.2 —— 它收的是**工厂函数**，不是协程对象（避免"协程从未被 await"的
   `RuntimeWarning` 污染测试输出）。

### 2.4 预留 metadata key（跨模块协议，禁止自定义同义 key）

| 载体 | key | 类型 | 写入者 | 含义 |
|---|---|---|---|---|
| `Message.metadata` | `kind` | `str` | memory/agent | `"summary"` / `"observation"` / `"nudge"` / `"memories"` / `"react"` |
| `Message.metadata` | `step` | `int` | agent | 产生该消息的 ReAct 轮次 |
| `Message.metadata` | `tool_name` | `str` | agent | tool 消息对应的工具名（冗余，便于 trace） |
| `Message.metadata` | `memory_id` | `str` | （保留，当前无写入者） | **`[v3 变更]` 保留 key，实现尚未写入**：全仓库（`liteagent/` + `tests/` + `examples/`）没有任何地方写入它 —— v2 把它列成"memory 写入"是**规范单方面的声明**。保留该行是为了把语义钉死：**若将来要把一条消息关联回 `MemoryItem.id`，必须用这个 key**（禁止自定义同义 key）。当前可观测的关联信号是 `kind="memories"` + `memory_count`。 |
| `Message.metadata` | `memory_count` | `int` | memory | 长期记忆块里包含的条目数（`kind="memories"` 时必有）—— 实写点：`liteagent/memory/manager.py` 的 `Message.system(block, kind="memories", memory_count=len(included))` |
| `ToolCall.metadata` | `timeout_s` | `float` | caller | 覆盖本次调用的超时（`NO_TIMEOUT` 表示禁用） |
| `ToolCall.metadata` | `max_retries` | `int` | caller | 覆盖本次调用的重试次数 |
| `ToolResult.metadata` | `truncated` | `bool` | executor | 结果是否被截断（**只在 executor 截断，只写一次**） |
| `ToolResult.metadata` | `original_chars` | `int` | executor | 截断前的字符数 |
| `ToolResult.metadata` | `attempts` | `int` | executor | 实际尝试次数（= `ToolResult.attempts` 冗余副本） |
| `ToolResult.metadata` | `orphan_thread` | `bool` | executor | **同步**工具超时后线程仍在跑（D-09） |
| `ToolResult.metadata` | `approved` | `bool` | executor | 审批结果（`requires_approval` 工具必有） |
| `ToolResult.metadata` | `disabled` | `bool` | executor | 该工具因连续失败被熔断（§7.4.1 步骤 4.5） |
| `ToolResult.metadata` | `skipped_by_fail_fast` | `bool` | executor | 因 `fail_fast` 被取消（§7.4.2） |
| `ToolResult.metadata` | `feedback_kind` | `str` | executor | `"recoverable"` / `"infrastructure"`（§7.4.1 步骤 5.g） |
| `ToolResult.metadata` | `validation_errors` | `list[str]` | executor | 参数校验错误列表 |
| `ToolResult.metadata` | `error_context` | `dict` | executor | `LiteAgentError.context` 的副本 |
| `ToolResult.metadata` | `last_error_type` | `str` | executor | 重试耗尽时的根因类型 |
| `ToolResult.metadata` | `exit_code` | `int` | builtin/shell | 子进程退出码 |
| `ToolResult.metadata` | `sandbox` | `str` | builtin/files | 生效的沙箱根 realpath |
| `ToolResult.metadata` | `stringified` | `str` | executor | 被 `str()` 兜底转换的原始类型名 |
| `AgentState.scratchpad` | `delegation` | `DelegationContext` | multiagent | 委派栈/深度 |
| `AgentState.scratchpad` | `subagent_results` | `dict[str, str]` | multiagent | 子 Agent 输出归档 |
| `AgentResult.metadata` | `steps` | `list[dict]` | sequential | 各阶段摘要 |
| `AgentResult.metadata` | `failed_stage` | `str` | sequential | 失败的阶段名 |
| `AgentResult.metadata` | `delegations` | `list[dict]` | hierarchical | 委派记录 |
| `AgentResult.metadata` | `cost_usd` | `float \| None` | agent | 本次 run 的估算成本（无价格表时 `None`） |
| `AgentResult.metadata` | `finish_reason` | `str` | agent | 最后一次 LLM 响应的 `finish_reason` |

**`[v2 变更]` 写入者唯一性**：上表的"写入者"列是**排他的**。同一个 key 只能由该列指定的
那一层写；其它层要传信息必须走 `context` / 事件 / 新 key。

### 2.5 全局常量（冻结数值，实现者不得改动）

```python
# liteagent/config.py
DEFAULT_MAX_STEPS            = 10
DEFAULT_MAX_CONCURRENCY      = 4
DEFAULT_THREAD_POOL_SIZE     = 8
DEFAULT_TOOL_TIMEOUT_S       = 30.0
DEFAULT_LLM_TIMEOUT_S        = 60.0
DEFAULT_MAX_RETRIES          = 2
DEFAULT_BACKOFF_BASE_S       = 0.25
DEFAULT_BACKOFF_MAX_S        = 8.0
DEFAULT_BACKOFF_JITTER       = 0.5
DEFAULT_MAX_RESULT_CHARS     = 8000
DEFAULT_MAX_OBSERVATION_CHARS= 8000
DEFAULT_MAX_TRANSCRIPT       = 1000
NO_TIMEOUT                   = -1.0     # 超时哨兵：显式禁用超时（见 §5.3 / §7.4.1 步骤 2）

# [v2 新增]
DEFAULT_MAX_TOTAL_TOKENS     = 200000   # AgentConfig.max_total_tokens 的建议默认（None=不限）
DEFAULT_RESERVE_COMPLETION_TOKENS = 1024 # 上下文预算里给回复预留的 token
DEFAULT_MAX_TRUNCATION_RETRIES = 1      # finish_reason=="length" 的续写重试上限
DEFAULT_TOOL_FAILURE_LIMIT   = 3        # 同一工具连续 infrastructure 失败多少次后熔断
DEFAULT_MAX_TOOLS_IN_PROMPT  = 20       # 文本模式 system prompt 里最多渲染几个工具
DEFAULT_PERSIST_BATCH        = 500      # save() 每批写多少条（内存友好）

# memory
DEFAULT_BUFFER_MAX_TOKENS    = 3000
DEFAULT_BUFFER_MAX_MESSAGES  = 50
DEFAULT_SUMMARY_TRIGGER_RATIO= 0.8
DEFAULT_SUMMARY_MIN_EVICT    = 4
DEFAULT_MAX_SUMMARY_CHARS    = 1200
DEFAULT_HASHING_EMBED_DIM    = 256
DEFAULT_RECENCY_HALF_LIFE_D  = 7.0
DEFAULT_W_SIM                = 1.0
DEFAULT_W_RECENCY            = 0.15
DEFAULT_W_IMPORTANCE         = 0.1
DEFAULT_DEDUP_THRESHOLD      = 0.95
DEFAULT_MAX_MEMORY_ITEMS     = 10000
DEFAULT_MMR_LAMBDA           = 0.7
DEFAULT_RETRIEVE_LIMIT       = 5
DEFAULT_TOKEN_CHAR_RATIO     = 4.0      # ASCII
DEFAULT_CJK_CHAR_COST        = 1.0      # 每 CJK 字符算 1 token
DEFAULT_MEMORY_QUERY_CHARS   = 500      # <relevant_memories> 单条 content 截断
DEFAULT_MEMORY_BLOCK_CHARS   = 4000     # <relevant_memories> 整体截断

# agent
DEFAULT_PARSE_MAX_RETRIES    = 2
DEFAULT_REPEAT_THRESHOLD     = 2
DEFAULT_TEMPERATURE          = 0.0

# multiagent
DEFAULT_TEAM_MAX_DEPTH       = 3
DEFAULT_TEAM_MAX_ROUNDS      = 10
DEFAULT_SUBAGENT_CONCURRENCY = 3
DEFAULT_SUBAGENT_MAX_CHARS   = 2000
SUBAGENT_HEAD_RATIO          = 0.7      # head+tail 压缩：头 70%
```

### 2.6 环境变量（冻结）

| 变量 | 用途 | 读取者 |
|---|---|---|
| `LITEAGENT_API_KEY` | 通用 API key（优先级最高） | `LLMConfig.resolve_api_key`；另有 `RemoteEmbedder.__init__`（`memory/embeddings.py`） |
| `LITEAGENT_BASE_URL` | 覆盖 base_url | `LLMConfig.resolve_base_url`；另有 `RemoteEmbedder.__init__`（`memory/embeddings.py`） |
| `LITEAGENT_MODEL` | 覆盖 model | `LLMConfig.from_env` |
| `LITEAGENT_PROVIDER` | 覆盖 provider | `LLMConfig.from_env` |
| `LITEAGENT_MAX_STEPS` | 覆盖 `AgentConfig.max_steps` | `AppConfig.from_env` |
| `LITEAGENT_TRACE` | trace 输出文件（JSONL） | `cli` |
| `LITEAGENT_SANDBOX_ROOT` | 文件工具沙箱根 | `tools/builtin/__init__.py` 的 `register_all` / `_resolve_sandbox_root` |
| `LITEAGENT_ALLOW_SHELL` | `"1"` 才允许 `run_shell` 真实执行 | `builtin/shell.py` |
| `LITEAGENT_ALLOW_NETWORK` | `"0"` 关闭联网（等价 `allow_network=False`） | `builtin/web.py` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` | 按 provider 名的 fallback | `LLMConfig.resolve_api_key` |
| `TAVILY_API_KEY` / `SERPER_API_KEY` | 网页搜索后端 | `builtin/web.py` |

**`[v2 变更]`** `DEEPSEEK_API_KEY` 现在真的会被读到：§6.4 新增 `DeepSeekChatClient`，
`provider="deepseek"` 是合法取值（"LLM 层统一多模型 API" 的第三个真实适配器，
且它只有 5 行代码 —— 这是"统一抽象层"最有说服力的证据）。

**`[v3 变更]` 读取者归属按实现修正**（上表）：
- `LITEAGENT_SANDBOX_ROOT` 的读取者是 **`tools/builtin/__init__.py`**，**不是** `builtin/files.py`：
  `files.py` 对它**零引用**，而且它**不该**读 —— §7.5 冻结了 `make_file_tools(sandbox)` 的
  `sandbox` 必填、`None` 抛 `ConfigError`；若 `files.py` 自己回退到 cwd，在仓库根跑测试时
  `delete_file` 会真的删项目文件。`config.py` 的 `from_env` **不越权**读取它（同一个环境变量
  有两个读取者时，行为会随调用顺序变化），详见 `liteagent/config.py::AppConfig.from_env`
  的 `[v3 修正]` 段。
- `LITEAGENT_API_KEY` / `LITEAGENT_BASE_URL` 各有**两个**读取者：`LLMConfig` 的
  `resolve_api_key` / `resolve_base_url`，以及 `memory/embeddings.py::RemoteEmbedder.__init__`
  （凭据解析优先级刻意与 §6.3 对齐，避免"LLM 配好了 key、embedder 却没读到"）。
- `LITEAGENT_MODEL` / `LITEAGENT_PROVIDER` 的读取者是 `LLMConfig.from_env`；
  `LITEAGENT_MAX_STEPS` 的读取者是 `AppConfig.from_env`（v2 把三者笼统写成 `Config.from_env`）。

### 2.7 事件发射归属矩阵（`[v2 变更]` 冻结，谁发哪条事件有唯一出处）

| 事件 | **唯一**发射者 | 必需 data 键（除公共键外） |
|---|---|---|
| RUN_STARTED / RUN_FINISHED / RUN_FAILED | `Agent` | `{input, mode, tools}` / `{output_len, steps, usage}` / `{error_type, message, aborted}` |
| STEP_STARTED / STEP_FINISHED | `Agent` | `{step}` / `{step, status}` |
| THOUGHT / ACTION_PARSED / PARSE_ERROR / REPEAT_DETECTED / NUDGE | `Agent` | `{text}` / `{action, arguments}` / `{reason, offset, raw_len, attempt}` / `{action_key, count}` / `{text}` |
| BUDGET_EXCEEDED | `Agent` | `{kind, limit, used}` |
| CONTEXT_TRUNCATED | `MemoryManager` | `{before, after, dropped_messages}` |
| LLM_REQUEST / LLM_RESPONSE / LLM_ERROR | `BaseLLMClient._emit` | 见 §6.3。**Agent 不得重复发** |
| TOOL_STARTED / TOOL_RETRY / TOOL_FINISHED / TOOL_ERROR / TOOL_APPROVAL | `ToolExecutor` | 见 §7.4。**Agent 不得重复发**（v1 的 §9.4.1 里那两行 `emit TOOL_STARTED/TOOL_FINISHED` 已删除） |
| MEMORY_WRITE / MEMORY_RETRIEVE / MEMORY_COMPRESS | `MemoryManager` | `{kind, count}` / `{count, query_len}` / `{before_tokens, after_tokens, compressed}` |
| AGENT_DELEGATE / AGENT_RETURN | `MultiAgent` | `{from, to, step, depth, refused, refused_reason}` / `{from, to, status, steps, output_len, duration_ms, failed}` |
| BLACKBOARD_WRITE / BLACKBOARD_READ | `Blackboard` | `{key, version, author}` / `{key, hit}` |

**低层事件的统一签名（`[v2 变更]` 冻结，替代 v1 里三套互不兼容的注解）**：

```python
LowLevelEvent = Callable[[str, dict[str, Any]], None]
```

`BaseLLMClient.__init__` / `ToolExecutor.__init__` / `MemoryManager.__init__` /
`Blackboard.__init__` / `Embedder` 的 `on_event` **一律**是 `LowLevelEvent | None`，
**一律**只接收 `(event_type_str, data)` 二元组。
`event_type_str` 的合法取值就是 §9.2 `EventType` 的 `.value`（LLM/tools/memory 层**不得** import agent 层）。
`agent/callbacks.as_llm_callback(manager)` 是唯一的适配器，把字符串事件转成 `TraceEvent`；
它对未知字符串**忽略**（不抛异常）。
`v1` 里 `Callable[[Any], None]` 与 `Callable[[TraceEvent], None]` 两处写法**全部作废**。

**事件 data 的类型约束（冻结）**：所有进入 `data` 的值**必须已过 `config.to_jsonable`**。
因此 `TraceEvent.to_json()` 里的 `json.dumps` 永不因类型而抛 `TypeError`。
`LLM_RESPONSE` 的 `usage` 必须是 `resp.usage.to_dict()`（**不是** `TokenUsage` 实例）。

### 2.8 禁止断言的字段清单（`[v2 变更]`，防止 flaky 测试）

测试**不得**直接断言以下字段（它们本身是不确定的）：
`AgentState.run_id`（uuid4）、`MemoryItem.id`（uuid4）、`AgentResult.duration_ms`、
`ToolResult.duration_ms`、`ToolCall.id`（原生 provider 生成时）、`TraceEvent.timestamp`、
`LLMResponse.latency_ms`（**唯一例外**：`ScriptedLLM` 的 `latency_ms == delay_s * 1000` 可断言）、
`MemoryItem.access_count` / `last_access_at`（除非测试自己控制 `now`）、`BlackboardEntry.created_at`。

**对应的断言手法（冻结）**：
- 断言 `TraceEvent.to_dict()` 或 `AgentState.to_dict()` 前必须 `d.pop("ts")` / 忽略 `started_at`/`finished_at`；
- 断言 id 时只断言前缀（`startswith("call_")`）或从被测对象里取出来再比对；
- 断言耗时用 `assertGreaterEqual(x, 0.0)`，不断言具体值。

---

## 3. `liteagent/errors.py`

**选择理由**：异常必须是最底层模块，任何层都能 import 而不引入依赖。

### 3.1 基类语义

```python
class LiteAgentError(Exception):
    """所有框架异常的基类。"""

    # 类属性：该类异常默认是否可重试。子类按需覆盖。
    retryable: bool = False
    # 类属性：建议的重试间隔秒数，None 表示由调用方的退避策略决定。
    retry_after_s: float | None = None

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None: ...

    message: str          # 实例属性
    context: dict[str, Any]
    cause: BaseException | None

    def to_dict(self) -> dict[str, Any]:
        """-> {"type": <类名>, "message": str, "context": dict, "retryable": bool}"""
        ...

    def __str__(self) -> str:
        """有 context 时输出 'message (k=v, k2=v2)'，保证日志可读。"""
        ...
```

**构造规则（冻结）**：
1. `message` 为位置参数且默认 `""`，所以 `raise ToolNotFoundError("read_file")` 合法。
2. `context` 中的值必须是 JSON 可序列化类型；实现者若不确定，先过 `config.to_jsonable`。
3. `LiteAgentError` **不可以**被框架在成功路径上捕获后静默吞掉；只有明确文档写了"降级"的地方才可以。
4. **`cause` 只用于链式诊断**，`to_dict()` 输出里含 `"cause": str(self.cause) | None`。

### 3.2 完整异常树与抛出时机

```
Exception
└── LiteAgentError                         retryable=False
    ├── ConfigError                        retryable=False
    ├── SerializationError                 retryable=False
    ├── ScriptedExhaustedError             retryable=False
    ├── SandboxViolationError              retryable=False
    ├── LLMError                           retryable=False
    │   ├── LLMAuthError                   retryable=False
    │   ├── LLMBadRequestError             retryable=False
    │   ├── LLMResponseFormatError         retryable=False
    │   ├── LLMRateLimitError              retryable=True   (retry_after_s 可设)
    │   ├── LLMTimeoutError                retryable=True
    │   └── LLMConnectionError             retryable=True   (5xx / 网络中断)
    ├── ToolError                          retryable=False
    │   ├── ToolNotFoundError              retryable=False
    │   ├── ToolValidationError            retryable=False   (+ .errors: list[str])
    │   ├── ToolDefinitionError            retryable=False
    │   ├── ToolExecutionError             retryable=False   (可被工具作者置 True)
    │   ├── ToolTimeoutError               retryable=True    (但同步工具不重试，见 §3.4)
    │   ├── ToolSkippedError               retryable=False   [v2 新增] fail_fast 取消的兄弟调用
    │   ├── ToolApprovalDeniedError        retryable=False   [v2 新增] HITL 拒绝
    │   └── ToolRetryExhaustedError        retryable=False   (+ .attempts, .last_error)
    ├── MemoryStoreError                   retryable=False
    ├── ReActParseError                    retryable=False   (+ .raw, .offset, .reason)
    ├── AgentError                         retryable=False
    │   ├── MaxStepsExceededError          retryable=False   (+ .max_steps)
    │   ├── RepeatedActionError            retryable=False   (+ .action_key, .count)
    │   ├── BudgetExceededError            retryable=False   [v2 新增] (+ .limit, .used, .kind)
    │   ├── RunTimeoutError                retryable=False   [v2 新增] (+ .timeout_s, .elapsed_s)
    │   └── AgentAbortedError              retryable=False
    └── MultiAgentError                    retryable=False
        ├── DelegationError                retryable=False   (+ .from_agent, .to_agent)
        ├── MaxDepthExceededError          retryable=False   (+ .depth, .max_depth)
        ├── CycleDetectedError             retryable=False   (+ .stack)
        └── VersionConflictError           retryable=False   (+ .key, .expected, .actual)
```

### 3.3 各异常的字段与抛出时机（逐条冻结）

| 类 | 额外字段 | 何时抛出 |
|---|---|---|
| `ConfigError` | — | 配置缺失/非法（如 `provider` 未知、YAML 需要但不可用、`semaphore` 同 key 不同 value） |
| `SerializationError` | `target: str` | `from_dict` 收到缺字段/错类型 |
| `ScriptedExhaustedError` | `consumed: int` | `ScriptedLLM` 队列耗尽且 `strict=True` 且 `loop=False` |
| `SandboxViolationError` | `path: str`, `root: str` | 文件/命令工具越出沙箱根；**denylist 命中时 `path=command`、`root="denylist:<reason>"`** |
| `LLMAuthError` | `status_code: int` | HTTP 401/403、缺 api_key |
| `LLMBadRequestError` | `status_code: int`, `body: str` | HTTP 400/404/422 |
| `LLMResponseFormatError` | `body: str` | 响应 200 但缺 `choices`/`content` 等必需字段 |
| `LLMRateLimitError` | `status_code: int` | HTTP 429；`retry_after_s` 取 `Retry-After` 头 |
| `LLMTimeoutError` | `timeout_s: float` | socket/read 超时、`asyncio.TimeoutError` |
| `LLMConnectionError` | `status_code: int \| None` | 5xx、DNS/连接失败 |
| `ToolNotFoundError` | `name: str`, `available: list[str]` | `registry.get()` 未命中 |
| `ToolValidationError` | `errors: list[str]`, `tool_name: str` | 参数 JSON Schema 校验失败 |
| `ToolDefinitionError` | `tool_name: str` | 装饰器阶段无法生成 schema（如 `**kwargs`、重名） |
| `ToolExecutionError` | `tool_name: str`, `call_id: str` | 工具函数内部抛异常，被 executor 包装 |
| `ToolTimeoutError` | `tool_name: str`, `timeout_s: float` | `asyncio.wait_for` 超时 |
| `ToolSkippedError` | `tool_name: str`, `reason: str` | `fail_fast=True` 时被取消的兄弟调用（`reason="cancelled_by_fail_fast"`） |
| `ToolApprovalDeniedError` | `tool_name: str`, `reason: str` | `requires_approval=True` 且无 policy 或 policy 返回 False |
| `ToolRetryExhaustedError` | `attempts: int`, `last_error: LiteAgentError`, `tool_name: str` | 重试次数用尽 |
| `MemoryStoreError` | `store: str` | 存储层内部错误（向量维度不一致等） |
| `ReActParseError` | `raw: str`, `offset: int`, `reason: str` | 文本模式解析不出 Action/Final Answer；Plan JSON 非法 |
| `MaxStepsExceededError` | `max_steps: int` | 达到 `AgentConfig.max_steps` 仍在行动 |
| `RepeatedActionError` | `action_key: str`, `count: int` | 重复/无进展动作策略判定失败 |
| `BudgetExceededError` | `limit: int`, `used: int`, `kind: str` | 超过 `max_total_tokens`（`kind="total_tokens"`） |
| `RunTimeoutError` | `timeout_s: float`, `elapsed_s: float` | 超过 `AgentConfig.max_wall_clock_s` |
| `AgentAbortedError` | `reason: str` | 用户在 `approval_policy` 里抛它 → 转成 `ABORTED`；或外部要求中止 |
| `DelegationError` | `from_agent: str`, `to_agent: str` | 委派失败且 `propagate_failure="raise"` |
| `MaxDepthExceededError` | `depth: int`, `max_depth: int` | 委派深度超限或委派预算耗尽 |
| `CycleDetectedError` | `stack: list[str]` | 委派栈中出现同名 Agent |
| `VersionConflictError` | `key: str`, `expected: int`, `actual: int` | Blackboard 乐观并发写失败 |

### 3.4 重试白名单（冻结，唯一出处）

`ToolExecutor` 与 `BaseLLMClient` 只对 `exc.retryable is True` 的异常重试：

| 异常 | `retryable` | 默认是否重试 | 备注 |
|---|---|---|---|
| `LLMRateLimitError` | True | 是 | 退避取 `max(backoff, retry_after_s)` |
| `LLMTimeoutError` | True | 是 | |
| `LLMConnectionError` | True | 是 | 5xx / 网络中断 |
| `ToolTimeoutError` | True | **异步工具：是；同步工具：[v2 例外] 否** | 见下 |
| `ToolExecutionError` | False | 否 | 工具作者对**幂等**操作可显式 `retryable=True` |
| 其余全部 | False | 否 | |

**`[v2 变更]` 同步工具超时不重试（冻结）**：`asyncio.wait_for` 无法中断 worker 线程里的同步代码，
超时后线程仍在跑（`orphan_thread=True`）。此时重试会**再起一个线程**，
对 `write_file` / `run_shell` 这类工具意味着"两个线程同时写同一份资源" → 数据破坏。
**规则**：`if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async: 不重试`。
理由必须写进代码注释（引用本节）。异步工具的 `ToolTimeoutError` 允许重试。

**错误分类的第二层（`[v2 变更]`，面试必答）**：除了"瞬时 vs 永久"，还要区分
**模型能自纠正** vs **环境/实现故障**，它决定回灌给模型的文案（见 §7.4.1 步骤 5.g）：

| `feedback_kind` | 触发的 `error_type` | 回灌文案要求 |
|---|---|---|
| `"recoverable"` | `ToolValidationError` / `ToolNotFoundError` / `ToolDefinitionError` | **必须告诉模型怎么改**：可用工具名列表、schema 期望的类型、正确格式示例 |
| `"infrastructure"` | `ToolExecutionError` / `SandboxViolationError` / `ToolTimeoutError` / `ToolRetryExhaustedError` / `ToolSkippedError` / `MemoryStoreError` / `ToolApprovalDeniedError` | **必须劝模型换策略**：`do not retry this tool with the same arguments; try a different approach or give your final answer` |

---

## 4. `liteagent/types.py`

**dataclass vs pydantic 的选择（冻结决策 D-01）**：核心数据结构一律用 **stdlib `dataclass`**，不用 pydantic。
- 理由：(a) 零依赖红线；(b) agent 状态是高频可变对象，pydantic 的校验/拷贝开销不划算；
  (c) `dataclasses.replace`/`asdict` 已够用；(d) trace 结构需要**字段全量输出**，pydantic 的
  `exclude_none` 默认行为反而别扭。
- 代价：没有运行期类型校验。缓解：`from_dict` 手工校验并抛 `SerializationError`；
  **唯一**使用 pydantic 的地方是 `tools/schema.py` 的可选校验适配器（运行时探测）。
- `slots=True` 只用于**叶子**数据类（`TokenUsage`/`ToolCall`/`ToolResult`/`ScriptedCall`）；
  `LLMResponse` 与所有含 `dict` 默认值的数据类**不用 slots**（避免与 `default_factory` 组合的坑）。

### 4.1 `TokenUsage`

```python
@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        """若 total_tokens == 0 且 prompt/completion > 0，则自动置为两者之和。"""
        ...

    def __add__(self, other: "TokenUsage") -> "TokenUsage": ...
    def __iadd__(self, other: "TokenUsage") -> "TokenUsage": ...
    def is_empty(self) -> bool: ...
    @property
    def estimated_cost_usd(self) -> float | None:
        """[v2 变更] 委托 config.estimate_cost_usd(self, model=self.model_hint)；
        model_hint 是实例属性（默认 ""），由 LLM 层在累加时写入。无价格表条目 -> None。"""
        ...
    model_hint: str = ""          # [v2 新增] 供成本估算使用的模型名（不参与 to_dict 的必需键）
    def to_dict(self) -> dict[str, int]:
        """只输出 {"prompt_tokens","completion_tokens","total_tokens"}（3 个 int）。"""
        ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenUsage": ...
```

### 4.2 `ToolCall`

```python
@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, arguments: Mapping[str, Any] | None = None,
               *, call_id: str | None = None) -> "ToolCall":
        """call_id 为 None 时生成 'call_' + uuid4().hex[:12]。
        [v2 变更] arguments 里的每个值必须是 JSON 可序列化类型，否则 SerializationError。"""
        ...

    @classmethod
    def from_arguments_json(cls, name: str, raw: str,
                            *, call_id: str | None = None) -> "ToolCall":
        """解析模型返回的 arguments 字符串；失败时抛 ReActParseError（带 raw/offset）。"""
        ...

    @classmethod
    def try_from_arguments_json(cls, name: str, raw: str,
                                *, call_id: str | None = None) -> tuple["ToolCall", str | None]:
        """[v2 新增] **不抛异常**的变体：失败时返回 (arguments={"__raw__": raw} 的 ToolCall, 错误消息)。
        供 provider 层使用（§6.4 冻结：provider 绝不在这里抛异常，否则模型失去自纠正机会）。
        两个方法是唯一入口，provider 只用 try_ 版本。"""
        ...

    def canonical_key(self) -> str:
        """重复动作检测用的稳定键：f'{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)}'"""
        ...

    def to_dict(self) -> dict[str, Any]:
        """[v2 变更] **必须**输出 {"id","name","arguments","raw_arguments","metadata"} 五个键。"""
        ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCall": ...
```

**注意**：`ToolCall` 使用 `slots=True` 但 `metadata` 有 `default_factory` —— 3.10 支持，已实测。
`slots=True` 与 `metadata` 动态属性赋值互斥：**冻结规则：不要在 `ToolCall`/`ToolResult` 上临时挂属性。**

### 4.3 `ToolResult`

```python
@dataclass(slots=True)
class ToolResult:
    call_id: str
    name: str
    content: str = ""
    ok: bool = True
    error: str | None = None
    error_type: str | None = None        # LiteAgentError 子类名，如 "ToolValidationError"
    duration_ms: float = 0.0
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, call: ToolCall, content: str, *, duration_ms: float = 0.0,
                attempts: int = 1, metadata: Mapping[str, Any] | None = None) -> "ToolResult": ...

    @classmethod
    def failure(cls, call: ToolCall, error: BaseException, *,
                duration_ms: float = 0.0, attempts: int = 1,
                metadata: Mapping[str, Any] | None = None) -> "ToolResult":
        """error_type = type(error).__name__；error = str(error)；
        若 error 是 LiteAgentError 则把 error.context 合并进 metadata['error_context']。
        [v2 变更] **冻结**：content = self.error_text()（见下）。失败结果**必须**带非空 content，
        否则模型收到的 tool 消息是空字符串、无法自纠正。"""
        ...

    def error_text(self) -> str:
        """[v2 新增] ok=True -> self.content；
        否则 -> f"ERROR({self.error_type}): {self.error}" + ("\\n" + self.content if self.content else "")。
        这是**唯一**的错误文本生成处；executor / to_message / to_observation 全部复用它。"""
        ...

    def to_message(self) -> Message:
        """-> Message(role=Role.TOOL, content=self.error_text(), name=self.name,
                     tool_call_id=self.call_id, metadata={'tool_name': self.name})"""
        ...

    def to_observation(self, *, max_chars: int = 8000) -> str:
        """文本 ReAct 模式的 Observation 载荷。
        [v2 变更] 冻结：ok=False 时**直接返回 self.content**（即 error_text() 的结果，
        **不再二次拼 'ERROR(...):' 前缀** —— 否则错误文本会出现两遍）。
        ok=True 时返回 self.content（超长则 truncate_head_tail，
        尾部追加 '\\n...[truncated N chars]...'）。
        """
        ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolResult": ...
```

### 4.4 `LLMResponse`

```python
@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"          # "stop" | "tool_calls" | "length" | "error" | "content_filter"
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw: dict[str, Any] | None = None
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool: ...
    def to_message(self) -> Message:
        """-> Message(role=Role.ASSISTANT, content=self.content, tool_calls=self.tool_calls)"""
        ...
    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMResponse": ...
```

### 4.5 `ScriptedCall`（测试可观测性）

```python
@dataclass(slots=True)
class ScriptedCall:
    index: int
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    kwargs: dict[str, Any]
    response: "ScriptedResponse | None" = None   # [v2 新增] 本次调用消费到的脚本响应（便于断言）
    def to_dict(self) -> dict[str, Any]: ...
```

**`types.py` 的 import 白名单**：`json`、`time`、`uuid`、`dataclasses`、`typing`、`collections.abc`、
`liteagent.errors`。**不得** import `config`（避免 L1 变成 L2 循环）——
`ToolResult.to_observation` 的默认值 `max_chars` 用**字面量 `8000`** 并在注释里标注同步位置。
**唯一例外**：`to_message` 函数体内延迟 import `liteagent.llm.message`（§1.1 的 E1）。

---

## 5. `liteagent/config.py`

**职责**：**跨层**配置 dataclass + 跨层共享的纯函数。**不放业务逻辑。**
各层内部配置（`BufferConfig` / `SummaryConfig` / `VectorConfig`）定义在各自模块（§8.3–8.5），
**不属于** config.py。

### 5.1 `RetryPolicy` 与退避计算（被 executor 与 llm 复用）

```python
@dataclass
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES         # 额外尝试次数（总尝试 = 1 + max_retries）
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    backoff_max_s: float = DEFAULT_BACKOFF_MAX_S
    jitter: float = DEFAULT_BACKOFF_JITTER         # 0.0 <= jitter <= 1.0
    rng_seed: int | None = None                    # 非 None 时退避可复现（测试用）
    sleep_fn: "SleepFn | None" = None              # [v2 新增] 见 §5.5

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """等价于 compute_backoff(attempt, base_s=self.backoff_base_s,
                                max_s=self.backoff_max_s, jitter=self.jitter, rng=rng)。
        **两个方法必须调用同一个函数（禁止各自手写）**，且 `RetryPolicy.delay_for` 的
        逐字实现就是这一行委托 —— 冻结，测试会断言两者对同一组参数返回完全相同的值。"""
        ...

def compute_backoff(attempt: int, *, base_s: float = DEFAULT_BACKOFF_BASE_S,
                    max_s: float = DEFAULT_BACKOFF_MAX_S,
                    jitter: float = DEFAULT_BACKOFF_JITTER,
                    rng: random.Random | None = None) -> float:
    """attempt 从 0 开始（第 1 次失败后调 compute_backoff(0)）。
    full = min(max_s, base_s * 2 ** attempt)
    jitter=0.0      -> full
    jitter=1.0      -> rng.uniform(0, full)                  （full jitter）
    0<jitter<1      -> rng.uniform(1 - jitter, 1.0) * full    （equal jitter，默认 0.5）
    返回值 clamp 到 [0.0, max_s]（full jitter 下也不超过 max_s）。

    [v2 变更] **rng 为 None 时的行为冻结**（v1 完全未定义，三个实现都"符合文档"）：
      rng is None -> rng = random.Random(rng_seed_arg) 由调用方传入；本函数**不**自己造 rng。
      若调用方确实传了 None：等价于 random.Random(None)（即不可复现），
      **禁止**使用模块级 `random.random()` / `random.uniform()` 全局函数。
    """
    ...

rng_seed_arg: int | None   # 仅供上一条注释引用；实现里不需要这个变量，见 §5.5 的冻结规则
```

**为什么放在 `config.py`**：为了不新增文件、不让文件清单漂移，跨层共享的纯函数集中在 config.py。
`config.py` 只依赖 `errors`/`types`，被 llm 与 tools 同时依赖不会形成环。

### 5.2 辅助纯函数

```python
def run_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """在同步上下文跑协程。
    - 若当前线程无运行中的 loop: _run_and_cleanup(factory)
    - 若已有运行中的 loop: 抛 ConfigError("cannot call sync API from a running event loop; use the 'a*' variant")
      （绝不使用 run_until_complete 套娃 / nest_asyncio）

    [v2 变更] **签名冻结为收工厂函数**，不收协程对象。
    理由：调用方若写 run_sync(self.achat(...))，协程对象在检查之前就已构造，
    抛 ConfigError 后该协程永远不会被 await，3.10 会打
    RuntimeWarning: coroutine ... was never awaited —— 污染测试输出（-W error 下直接失败）。
    **所有调用点必须写成 `run_sync(lambda: self.achat(...))`。**
    （若实现者坚持同时接受协程对象，则必须在抛 ConfigError **之前** coro.close()；
     但规范冻结的唯一形态是工厂函数。）
    """
    ...

def _run_and_cleanup(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """内部：新建 loop -> asyncio.run(factory()) -> 在 finally 里
    `LoopBoundPool.release_loop(loop)`（§5.4：丢弃该 loop 的全部原语、shutdown 它的私有线程池）。
    所有同步入口（run_sync / ToolExecutor.execute_sync / Agent.run）共用它，
    这是 §12 的 `test_cross_run_no_leak`（连续 3 次同步调用后池内条数为 0）成立的前提。"""
    ...

def utc_now() -> float:
    """**唯一时钟**（[v2 变更]）。实现就是 `return time.time()`（Unix 秒）。
    命名保留 UTC 语义便于阅读。所有模块的时间戳 default_factory 必须用它。"""
    ...

def frozen_now() -> float:  # 仅供测试覆盖，不在 __all__
    ...

def format_ts(ts: float, *, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """使用 datetime.fromtimestamp(ts, tz=timezone.utc) 格式化（3.10 无 datetime.UTC）。"""
    ...

def to_jsonable(obj: Any) -> Any:
    """递归把 dataclass/Enum/set/tuple/Path/bytes/Exception 转成 JSON 可序列化结构。
    dataclass -> {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    Enum -> obj.value; set/tuple/frozenset -> list; Path -> str; bytes -> base64 str
    Exception -> {"type": type(obj).__name__, "message": str(obj)}
    **重要**：to_jsonable(MemoryItem) 输出包含 embedding（它不感知 opt-out；
      要省略请显式调 obj.to_dict(include_embedding=False) 再过 to_jsonable）。
    未知类型 -> repr(obj) 并附 'unserializable': True 标记
    """
    ...

def truncate_head_tail(text: str, max_chars: int, *, head_ratio: float = 0.7,
                       marker: str = "\n...[truncated {n} chars]...\n") -> str:
    """max_chars<=0 或 len(text)<=max_chars 时原样返回。
    否则保留前 int(max_chars*head_ratio) 与后 max_chars-int(max_chars*head_ratio) 个字符，
    中间插入 marker（marker 里的 {n} 替换为被丢弃的字符数）。"""
    ...

def parse_dotenv(text: str) -> dict[str, str]:
    """手写 .env 解析（无 python-dotenv）。
    规则：跳过空行与 '#' 注释；'KEY=VALUE'；支持 export 前缀；VALUE 两侧单/双引号剥离；
    双引号内支持 \\n \\t 转义；不支持变量插值。"""
    ...

def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """文件不存在 -> 返回 {}（不抛异常）；成功则写入 os.environ 并返回解析结果。"""
    ...

def parse_bool(value: str | bool | None, *, default: bool = False) -> bool:
    """'1'/'true'/'yes'/'on'（大小写不敏感）-> True；'0'/'false'/'no'/'off'/'' -> False。
    其它值 -> default。"""
    ...

def estimate_cost_usd(usage: "TokenUsage", *, model: str) -> float | None:
    """[v2 新增] 按 MODEL_PRICES 估算成本（美元）。model 未命中价格表 -> None。
    公式：prompt_tokens/1000 * price_in + completion_tokens/1000 * price_out。"""
    ...

MODEL_PRICES: dict[str, tuple[float, float]] = {
    # 每 1k token 的 (输入价, 输出价)，美元。[v2 新增]
    # 冻结说明：这是**示例价格表**，用于让 `AgentResult.metadata["cost_usd"]` 有值可展示；
    # 价格会变，生产使用前请自行更新。未列出的模型一律返回 None（诚实 > 猜）。
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-5-haiku": (0.0008, 0.004),
    "deepseek-chat": (0.00027, 0.0011),
}
```

### 5.3 配置 dataclass（全部字段有默认值）

```python
@dataclass
class LLMConfig:
    provider: str = "openai"              # "openai" | "anthropic" | "openai-compatible"
                                          # | "deepseek" [v2 新增] | "echo"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    temperature: float | None = None
    max_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    sleep_fn: "SleepFn | None" = None    # [v2 新增] 覆盖 retry_policy.sleep_fn

    def resolve_api_key(self) -> str | None:
        """优先级：self.api_key -> LITEAGENT_API_KEY -> <PROVIDER>_API_KEY（provider 大写、'-'->'_'）"""
        ...
    def resolved_base_url(self) -> str | None:
        """self.base_url or LITEAGENT_BASE_URL or 各 provider 的内置默认"""
        ...
    def to_dict(self) -> dict[str, Any]: ...      # 不含 api_key（脱敏为 "***" 或 None）
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMConfig": ...
    @classmethod
    def from_env(cls) -> "LLMConfig": ...

@dataclass
class MemoryConfig:
    buffer_max_tokens: int = DEFAULT_BUFFER_MAX_TOKENS
    buffer_max_messages: int = DEFAULT_BUFFER_MAX_MESSAGES
    buffer_keep_last_n: int = 2
    summary_enabled: bool = True
    summary_trigger_ratio: float = DEFAULT_SUMMARY_TRIGGER_RATIO
    summary_min_evict: int = DEFAULT_SUMMARY_MIN_EVICT
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    long_term_enabled: bool = True
    embedder_dim: int = DEFAULT_HASHING_EMBED_DIM
    write_policy: str = "selective"        # "selective" | "turn" | "manual"
    auto_write_min_chars: int = 40
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    max_items: int = DEFAULT_MAX_MEMORY_ITEMS
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    retrieve_min_score: float = 0.0
    w_sim: float = DEFAULT_W_SIM
    w_recency: float = DEFAULT_W_RECENCY
    w_importance: float = DEFAULT_W_IMPORTANCE
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_D
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    token_char_ratio: float = DEFAULT_TOKEN_CHAR_RATIO
    cjk_char_cost: float = DEFAULT_CJK_CHAR_COST
    # [v2 新增] 上下文窗口反推预算（D-15）
    context_window_tokens: int | None = None          # 非 None 时反推 buffer_max_tokens
    reserve_completion_tokens: int = DEFAULT_RESERVE_COMPLETION_TOKENS
    tools_schema_tokens_reserve: int = 0
    persist_path: str | None = None                   # 非 None 时 MemoryManager 启动即 restore

@dataclass
class ExecutorConfig:
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    # [v2 变更] default_timeout_s 的语义写死为：None = **对所有工具禁用超时**；
    # 想给单个工具禁用超时请用 NO_TIMEOUT（§7.4.1 步骤 2）。
    default_timeout_s: float | None = DEFAULT_TOOL_TIMEOUT_S
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    thread_pool_size: int = DEFAULT_THREAD_POOL_SIZE   # 每个 loop 的**私有**线程池大小
    sequential_tools: frozenset[str] = frozenset()     # 这些工具永不并发执行（跨线程生效）
    fail_fast: bool = False
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    allow_retry_on_non_idempotent: bool = False
    sleep_fn: "SleepFn | None" = None                  # [v2 新增] 优先于 retry_policy.sleep_fn
    approval_policy: "ApprovalPolicy | None" = None    # [v2 新增] HITL，见 §7.4.1 步骤 4.5
    disable_tool_after_failures: int = DEFAULT_TOOL_FAILURE_LIMIT  # [v2 新增] 0 = 关闭熔断

@dataclass
class AgentConfig:
    max_steps: int = DEFAULT_MAX_STEPS
    mode: str = "auto"                     # "auto" | "native" | "text"
    tool_choice: str | dict[str, Any] = "auto"
    temperature: float | None = DEFAULT_TEMPERATURE
    max_tokens: int | None = None
    parallel_tool_calls: bool = True
    max_parse_retries: int = DEFAULT_PARSE_MAX_RETRIES
    repeat_action_policy: str = "nudge_then_fail"   # "off"|"nudge"|"fail"|"nudge_then_fail"
    repeat_action_threshold: int = DEFAULT_REPEAT_THRESHOLD
    raise_on_error: bool = False
    system_prompt: str | None = None                # 非 None 时完全替换默认模板
    system_prompt_template: str = DEFAULT_REACT_SYSTEM_PROMPT
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    max_transcript_messages: int = DEFAULT_MAX_TRANSCRIPT
    include_thought_in_history: bool = True
    name: str = "agent"
    description: str = ""
    # [v2 新增] 成本与循环防护（面试必问的三件事）
    max_total_tokens: int | None = DEFAULT_MAX_TOTAL_TOKENS      # None = 不限
    max_prompt_tokens: int | None = None                         # 单次 prompt 上限
    max_wall_clock_s: float | None = None                        # 整轮墙钟上限
    max_truncation_retries: int = DEFAULT_MAX_TRUNCATION_RETRIES

@dataclass
class TeamConfig:
    max_depth: int = DEFAULT_TEAM_MAX_DEPTH
    max_rounds: int = DEFAULT_TEAM_MAX_ROUNDS
    parallel_subagents: bool = True
    subagent_concurrency: int = DEFAULT_SUBAGENT_CONCURRENCY
    share_memory: bool = False
    compress_subagent_output: bool = True
    subagent_output_max_chars: int = DEFAULT_SUBAGENT_MAX_CHARS
    propagate_failure: str = "return"      # "return" | "raise" | "continue"
    enable_cycle_detection: bool = True

@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    tools: list[str] = field(default_factory=list)      # 空 = 注册全部内置工具，见 §7.5
    trace_file: str | None = None
    sandbox_root: str | None = None
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig": ...
    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "AppConfig": ...
    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """需要 PyYAML；不可用时抛 ConfigError('PyYAML not installed; use JSON config')。"""
        ...
    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """按扩展名分派 .json/.yaml/.yml/.toml(->ConfigError，3.10 无 tomllib)。"""
        ...
    @classmethod
    def from_env(cls) -> "AppConfig": ...
    def to_dict(self) -> dict[str, Any]: ...
```

**`DEFAULT_REACT_SYSTEM_PROMPT`（冻结字面量）** —— 放在 `config.py`，因为 agent 与 multiagent 都要用：

```text
You are {name}, a helpful AI agent that solves tasks step by step.

You have access to the following tools:
{tools}

Use the following format:

Thought: your reasoning about what to do next
Action: the name of the tool to use, one of [{tool_names}]
Action Input: the arguments, as a JSON object
Observation: the result of the tool (this is filled in for you, do not write it yourself)
... (Thought/Action/Action Input/Observation may repeat)
Thought: I now know the final answer
Final Answer: the final answer to the user

Rules:
- Emit exactly ONE Action per step. Wait for the Observation before the next Thought.
- Action Input must be a single-line JSON object matching the tool's parameters.
- Never invent tool names. Available tools: [{tool_names}]
- If your output is truncated, continue from where you stopped.
- When you have enough information, stop calling tools and emit "Final Answer: ...".
```

模板渲染规则（冻结）：用 `template.format_map(_SafeDict(...))` 渲染，`_SafeDict.__missing__`
返回字面量 `"{" + key + "}"`（缺 key 不报错）。传入的 `tools` 值已经是渲染好的**多行文本**，
因此工具 description 里的 `{}` 不会被二次解析。`config.py` 需导出：

```python
class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str: ...
def render_template(template: str, values: Mapping[str, Any]) -> str: ...
```

### 5.4 `LoopBoundPool`（`[v2 变更]` 从 §7.4 挪来；同时修掉实测 M-3 的 loop 泄漏）

```python
class LoopBoundPool:
    """按运行中的 event loop 懒创建 asyncio 原语与线程池的容器（解决 §0.4 的 R-LOOP）。

    [v2 变更] 实现**不用** WeakKeyDictionary：
    实测（M-3）asyncio.Semaphore 争用后会把 loop 存进 self._loop，
    value 强引用 key，弱引用永不失效 —— 3 次 asyncio.run 就泄漏 3 个 loop 与 3 个原语。
    冻结实现：
        self._by_loop: dict[int, dict[str, Any]]      # key = id(loop)
        self._loops:   dict[int, asyncio.AbstractEventLoop]   # 保活，供 clear 使用
    必须在**运行中的 loop 内**调用以下方法；无运行 loop 时抛 ConfigError。
    所有入口的 finally 必须调用 release(loop)（§5.2 的 _run_and_cleanup 已代办）。

    用法（冻结）：
        pool = LoopBoundPool()
        sem = pool.semaphore("exec", config.max_concurrency)
    """

    def semaphore(self, key: str, value: int) -> asyncio.Semaphore:
        """同 key 且 value 不同 -> ConfigError（**不静默返回旧对象**，避免两个调用方
        以为自己在用不同的并发上限）。value <= 0 -> ConfigError。
        调用方必须用 owner 前缀构造 key（如 "exec" / "subagent:worker_name"）。"""
        ...
    def lock(self, key: str) -> asyncio.Lock: ...
    def condition(self, key: str) -> asyncio.Condition: ...
    def event(self, key: str) -> asyncio.Event: ...
    def thread_pool(self, key: str, max_workers: int) -> "ThreadPoolExecutor":
        """[v2 新增] 懒建的本 loop 私有线程池（异步工具 / 同步工具走 run_in_executor）。
        同 key 不同 max_workers -> ConfigError。"""
        ...
    def loop_of(self, primitive: Any) -> "asyncio.AbstractEventLoop | None":
        """[v2 新增] 返回该原语绑定的 loop（未绑定时 None）。Blackboard 用它做唤醒转发。"""
        ...
    def release(self, loop: "asyncio.AbstractEventLoop | None" = None) -> None:
        """[v2 新增，取代 v1 的 clear] 丢弃该 loop 的全部原语，并对它的全部线程池
        调 shutdown(wait=False, cancel_futures=True)。loop 为 None 时对所有 loop 执行。
        必须由 run_sync / execute_sync / Agent.run 在 asyncio.run 返回后的 finally 里调用。"""
        ...
    def current(self) -> "asyncio.AbstractEventLoop | None":
        """无运行 loop 时返回 None（不抛）。"""
        ...
    def __len__(self) -> int:
        """当前持有的 loop 个数。**测试用它断言不泄漏**（§12）。"""
        ...

    # ---- 全局登记（冻结，让 config.run_sync 能清理任意实例持有的 loop 资源）----
    _ALL_POOLS: ClassVar[list["LoopBoundPool"]] = []      # 模块级，__init__ 里 append(self)

    @classmethod
    def release_loop(cls, loop: "asyncio.AbstractEventLoop | None") -> None:
        """遍历 `_ALL_POOLS` 对每个实例调 `release(loop)`。
        由 `config._run_and_cleanup` 在 `asyncio.run` 返回后的 finally 里调用
        —— 这是"谁负责清理"这个问题的唯一答案（v1 用 WeakKeyDictionary 但没有清理点，
        实测每 3 次 asyncio.run 就泄漏 3 个 loop 与 3 个原语）。
        `pytest`/`unittest` 之外的长驻进程可调 `LoopBoundPool.aclose_all()` 注销全部实例。"""
        ...
    @classmethod
    def aclose_all(cls) -> None:
        """release(None) 所有实例并从 `_ALL_POOLS` 注销它们（长驻进程用）。"""
        ...
```

### 5.5 时间与随机性的注入点（`[v2 变更]`，为可测性冻结）

```python
SleepFn = Callable[[float], Awaitable[None]]

async def default_sleep(seconds: float) -> None:
    """内部 await asyncio.sleep(seconds)。"""
    ...

ApprovalPolicy = Callable[["ToolCall", "Tool"], bool]
```

**冻结规则（全部实现者必须遵守）**：
1. **一切退避等待**（`ToolExecutor` 的重试、`BaseLLMClient._with_retry` 的重试）、
   `ScriptedResponse.delay_s`、`ScriptedLLM.latency_s` **必须**经由 `sleep_fn` 调用，
   **禁止**直接 `await asyncio.sleep(...)`。
2. **解析优先级（冻结）**：`ExecutorConfig.sleep_fn` > `RetryPolicy.sleep_fn` > `default_sleep`；
   `LLMConfig.sleep_fn` > `LLMConfig.retry_policy.sleep_fn` > `default_sleep`。
   在 `ToolExecutor.__init__` / `BaseLLMClient.__init__` 里**解析一次**并存为
   `self._sleep: SleepFn`，**不要**在每次重试时重新解析。
3. **rng 冻结**：`ToolExecutor.__init__` 与 `BaseLLMClient.__init__` 各创建一次
   `self._rng: random.Random = random.Random(config.retry_policy.rng_seed)` 并**在该实例的
   全部退避里复用**（每次重建会让重试序列不可复现，与 D-07 的断言直接冲突）。
   `compute_backoff` / `delay_for` 的调用点一律显式传 `rng=self._rng`。
   `rng_seed is None` 时退避不可复现 —— 所有退避断言测试必须设 `jitter=0.0` **或** `rng_seed`。
4. 禁止在任何地方使用模块级 `random.*` 全局函数（`random.random()`/`random.uniform()`）；
   需要随机性时**必须**有显式的 `random.Random(...)` 实例（`RandomProjectionEmbedder`、
   `RetryPolicy` 都遵守此规则）。

---

## 6. `liteagent/llm/` —— LLM 抽象层

### 6.1 `llm/message.py`

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def coerce(cls, value: "Role | str") -> "Role":
        """接受 'system'/'System'/Role.SYSTEM；未知值抛 SerializationError。"""
        ...

@dataclass
class Message:
    role: Role = Role.USER
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- 构造便捷方法（全部为 classmethod，返回 Message）----
    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message": ...
    @classmethod
    def user(cls, content: str, **metadata: Any) -> "Message": ...
    @classmethod
    def assistant(cls, content: str = "",
                  tool_calls: Sequence[ToolCall] | None = None, **metadata: Any) -> "Message": ...
    @classmethod
    def tool(cls, result: "ToolResult") -> "Message":
        """等价 result.to_message()。"""
        ...
    @classmethod
    def observation(cls, content: str, *, step: int | None = None,
                    tool_name: str | None = None) -> "Message":
        """文本 ReAct 模式的注释消息：role=USER，metadata['kind']='observation'。
        为什么要 role=USER：文本 ReAct 没有原生 tool 角色，Observation 必须以"用户侧输入"
        的形式回灌，这是与多数 chat API 兼容的唯一做法。"""
        ...

    # ---- 变换 ----
    def copy(self, **changes: Any) -> "Message": ...     # dataclasses.replace
    def text(self) -> str:
        """用于 token 估算与摘要：content + 每个 tool_call 的 'name(canonical_args)'"""
        ...
    def is_tool_pair_start(self) -> bool: ...            # role==ASSISTANT and tool_calls
    def to_dict(self) -> dict[str, Any]:
        """{"role","content","name","tool_calls","tool_call_id","metadata"}
        role 输出 .value 字符串；tool_calls 为 list[dict]；不用 None 省略字段。"""
        ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message": ...

# 模块级工具函数（冻结）
def messages_tokens(messages: Sequence[Message],
                    counter: Callable[[str], int] | None = None) -> int:
    """**注意**：这里收的是 counter 函数而不是 Tokenizer 对象 —— llm 层不能 import memory 层
    （反向依赖）。memory 侧调用时传 `tokenizer.estimate` 即可。
    counter 为 None 时用内置的 ASCII 近似 `max(1, ceil(len(text)/4))`（仅用于展示，不用于预算判定）。
    [v2 变更] 空文本（`text() == ""` 的全部消息）返回 **0**，与 `HeuristicTokenizer("") == 0` 对齐。"""
    ...
def render_transcript(messages: Sequence[Message], *, include_tool_calls: bool = True) -> str:
    """把消息渲染成纯文本（给摘要器/LLM 摘要 prompt 用）。
    格式：'[role] content'，tool 消息带 '[tool:<name>]'，assistant 的 tool_calls 渲染为
    '  -> <name>(<canonical json>)'。"""
    ...
def drop_orphan_tool_messages(messages: Sequence[Message]) -> list[Message]:
    """删除 tool_call_id 找不到对应 assistant.tool_calls 的 TOOL 消息，
    以及没有匹配 TOOL 结果的 assistant.tool_calls（后者仅从 content 中剥离 tool_calls）。
    作用：裁剪窗口后保持 OpenAI/Anthropic 的消息合法性（否则 API 会 400）。"""
    ...
```

### 6.2 `llm/transport.py`

```python
@dataclass
class HTTPRequest:
    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    json_body: Any = None            # 与 data 互斥；非 None 时序列化并设 Content-Type
    data: bytes | None = None
    params: dict[str, str] = field(default_factory=dict)
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S

@dataclass
class HTTPResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    url: str = ""

    @property
    def ok(self) -> bool: ...                    # 200 <= status < 300
    def json(self) -> Any:                       # 解析失败抛 LLMResponseFormatError
        ...

class Transport(ABC):
    name: str = "abstract"

    @abstractmethod
    def send(self, request: HTTPRequest) -> HTTPResponse: ...
    async def asend(self, request: HTTPRequest) -> HTTPResponse:
        """默认实现：await asyncio.to_thread(self.send, request)。
        所有 Transport 只实现同步 send，异步版本由此默认实现提供。
        [v2 变更] 取消语义：调用方取消时 asend 会被取消，但 HTTP 线程同样成为孤儿。
        冻结：**不重试已发送的请求**（`LLMTimeoutError` 的 retryable 在传输层保持 True，
        但 `ToolSpec` 之外的 LLM 调用在 timeout 后是否重发由 `_with_retry` 决定，
        而 `HTTPChatClient` 对 `LLMTimeoutError` **不重试**（幂等性未知）——
        见 §6.4 的 `retry_on_timeout = False` 类属性）。"""
        ...
    def close(self) -> None: ...                 # 默认 no-op

class UrllibTransport(Transport):   name = "urllib"     # 纯 stdlib，永远可用
class RequestsTransport(Transport): name = "requests"   # 需要 requests，session 复用
class HttpxTransport(Transport):    name = "httpx"

def default_transport() -> Transport:
    """优先级：httpx -> requests -> urllib。结果缓存（functools.cache）。"""
    ...

def map_http_error(status: int, body: str, *, url: str = "",
                   headers: Mapping[str, str] | None = None,
                   timeout_s: float | None = None) -> LLMError:
    """统一错误映射（冻结表）：
      401/403        -> LLMAuthError
      429            -> LLMRateLimitError（retry_after_s 解析 Retry-After 头，支持纯数字秒）
      408/409/425    -> LLMConnectionError（可按 retryable 处理）
      400/404/413/422-> LLMBadRequestError
      5xx            -> LLMConnectionError
      其它 4xx       -> LLMBadRequestError
    """
    ...

def wrap_transport_exception(exc: BaseException, *, timeout_s: float | None) -> LLMError:
    """socket.timeout/TimeoutError/asyncio.TimeoutError -> LLMTimeoutError
       urllib.error.URLError / ConnectionError / OSError / httpx.TransportError -> LLMConnectionError
       其它 -> LLMError
    [v2 变更] **asyncio.CancelledError 必须原样向上抛**（它是 BaseException，M-5），
    绝不能被本函数包装（否则取消传播被破坏）。"""
    ...
```

**红线**：`UrllibTransport` 是**唯一**必须实现的传输层（因为它零依赖）。`RequestsTransport`
与 `HttpxTransport` 只在库可用时定义（`if REQUESTS_AVAILABLE:` 包裹类定义），否则模块仍可 import。

**urllib 使用要点（易错，冻结）**：
- 用 `urllib.request.Request(url, data=body, headers=..., method=...)`，`body = json.dumps(json_body).encode("utf-8")`。
- 必须捕获 `urllib.error.HTTPError`（它是 `URLError` 子类，**同时是一个可读的响应对象**），
  读 `e.code` / `e.read().decode()` 后才能交给 `map_http_error`。
- 必须用 `contextlib.closing` 或 `with urlopen(...) as resp` 关闭连接。
- `urlopen(timeout=...)` 的超时同时覆盖连接与读取。

### 6.3 `llm/base.py`

```python
@dataclass
class LLMStreamChunk:
    delta: str = ""
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    index: int = 0

class LLMClient(ABC):
    """所有 LLM 客户端的最小契约。实现者应继承 BaseLLMClient 而非直接继承本类。"""

    #: 该 provider 是否支持原生 function calling / tool use
    supports_tool_calling: bool = False
    #: 是否需要 api_key（echo/scripted 不需要）
    requires_api_key: bool = True

    @property
    def model(self) -> str: ...
    def count_tokens(self, text: str) -> int: ...     # 默认走 heuristic（空串 -> 0）
    def resolve_mode(self, *, has_tools: bool) -> str:
        """[v2 新增] `'native' if (self.supports_tool_calling and has_tools) else 'text'`。
        Agent 的 mode='auto' 就用这一个函数，保证只有一处判定。"""
        ...

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """config.run_sync(lambda: self.achat(messages, **kwargs))；在运行中的 loop 内抛 ConfigError。"""
        ...

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any
                           ) -> AsyncIterator[LLMStreamChunk]:
        """默认实现抛 NotImplementedError；调用方必须捕获并回退到 achat。
        [v2 变更] ScriptedLLM **必须**实现它（§6.6 冻结），否则流式路径零覆盖。"""
        ...

    async def aclose(self) -> None: ...

class BaseLLMClient(LLMClient):
    """提供：重试（RetryPolicy）、trace 事件发射、token 计数缓存、温度/最大 token 默认值合并。"""

    def __init__(self, config: LLMConfig | None = None, *,
                 transport: Transport | None = None,
                 on_event: LowLevelEvent | None = None) -> None:
        """[v2 变更] on_event 类型统一为 LowLevelEvent = Callable[[str, dict[str, Any]], None]。
        __init__ 里额外冻结创建：
            self._rng: random.Random = random.Random(config.retry_policy.rng_seed)
            self._sleep: SleepFn = config.sleep_fn or config.retry_policy.sleep_fn or default_sleep
        """
        ...

    config: LLMConfig
    on_event: LowLevelEvent | None

    def _resolve_params(self, temperature: float | None, max_tokens: int | None
                        ) -> tuple[float | None, int | None]: ...
    async def _with_retry(self, fn: Callable[[], Awaitable[T]], *, what: str) -> T:
        """对 retryable 的 LiteAgentError 重试；退避用 config.compute_backoff(rng=self._rng)，
        等待用 self._sleep(delay)（**禁止**直接 asyncio.sleep）。
        每次重试发 `LLM_REQUEST` 事件带 `retry` 字段（**Agent 不得重复发**，见 §2.7）。
        [v2 变更] 取消规则：宽 except 的首行必须是
        `except asyncio.CancelledError: raise`（M-5：它继承 BaseException）。"""
        ...
    def _emit(self, event_type: str, **data: Any) -> None:
        """[v2 变更] 签名冻结为 `_emit(self, event_type: str, **data: Any) -> None`。
        v1 的 `"EventType"` 注解作废（LLM 层不得 import agent 层）。
        event_type 的合法取值即 §9.2 EventType 的 .value。
        实现：`if self.on_event is not None: self.on_event(event_type, to_jsonable(data))`。
        data 里的值必须已过 to_jsonable（§2.7）。
        **严禁**在 data 里放入 TokenUsage 实例 —— 必须 to_dict()。
        """
        ...

    # ---- 冻结的 LLM_REQUEST / LLM_RESPONSE / LLM_ERROR data 键 ----
    # LLM_REQUEST  : {"messages_count": int, "tools_count": int, "retry": int, "model": str}
    # LLM_RESPONSE : {"content_len": int, "tool_calls": int, "finish_reason": str,
    #                 "latency_ms": float, "usage": dict}     <- usage 是 TokenUsage.to_dict()
    # LLM_ERROR    : {"error_type": str, "message": str, "retry": int}
```

**环依赖规避（冻结）**：`llm/base.py` **不得** import `agent/callbacks.py`。因此 LLM 层的事件回调
签名是轻量的 `LowLevelEvent`（`str` 是事件类型名，如 `"llm_request"`）。
`agent/callbacks.as_llm_callback(manager)` 是唯一的适配器。
这是本规范里唯一一处"字符串事件名"，其它地方一律用 `EventType` 枚举。

### 6.4 `llm/providers.py`

```python
class HTTPChatClient(BaseLLMClient):
    """所有真实 provider 的公共基类：请求体构造交给子类，
    发送/错误映射/重试/响应解码由基类统一完成。"""

    default_base_url: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    supports_tool_calling: bool = True
    #: [v2 新增] 超时后是否重发请求。默认 False（请求可能已到达服务端并产生副作用/计费）
    retry_on_timeout: bool = False

    def __init__(self, config: LLMConfig | None = None, *,
                 transport: Transport | None = None,
                 on_event: LowLevelEvent | None = None) -> None: ...

    # ---- 子类必须实现 ----
    def _endpoint(self) -> str: ...
    def _headers(self, api_key: str | None) -> dict[str, str]: ...
    def _build_body(self, messages: Sequence[Message], *, tools, tool_choice,
                    temperature, max_tokens, stop, stream: bool) -> dict[str, Any]: ...
    def _parse_response(self, payload: dict[str, Any], *, latency_ms: float) -> LLMResponse: ...

    # ---- 基类实现 ----
    async def achat(self, messages, *, tools=None, tool_choice=None, temperature=None,
                    max_tokens=None, stop=None, **kwargs) -> LLMResponse: ...
    def _require_api_key(self) -> str:
        """缺失时抛 LLMAuthError('missing API key for provider X ...')"""
        ...

class OpenAIChatClient(HTTPChatClient):
    """POST {base_url}/chat/completions。默认 base_url = https://api.openai.com/v1。
    请求：{"model","messages","tools":[{"type":"function","function":{...}}],"tool_choice",...}
    消息转换：role=tool 的消息 -> {"role":"tool","tool_call_id":...,"content":...}
             assistant.tool_calls -> [{"id","type":"function","function":{"name","arguments":<json str>}}]
    响应：choices[0].message.{content, tool_calls}; usage.{prompt_tokens,completion_tokens,total_tokens}
          tool_calls[i].function.arguments 是**字符串**，用
          [v2 变更] `ToolCall.try_from_arguments_json` 解析（**不抛异常**版本）：
          解析失败 -> 该 ToolCall.arguments = {"__raw__": <str>}，并在 metadata["parse_error"]
          记错误消息。**绝不在这里抛异常**（否则模型失去自纠正机会）。"""
    name = "openai"

class OpenAICompatibleClient(OpenAIChatClient):
    """vLLM/Ollama/DeepSeek/OpenRouter 等。必须显式传 base_url，否则 ConfigError。"""
    name = "openai-compatible"

class DeepSeekChatClient(OpenAICompatibleClient):
    """[v2 新增] 只需 5 行就多接一个真实 provider —— 这是"统一多模型 API"最直接的证据。
    完整实现（逐字）：
        name = "deepseek"
        default_base_url = "https://api.deepseek.com/v1"
    """
    name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"

class AnthropicChatClient(HTTPChatClient):
    """POST {base_url}/messages，默认 base_url = https://api.anthropic.com。
    头：x-api-key + anthropic-version: 2023-06-01 + content-type: application/json
    **关键差异（易错，冻结）**：
      1. system 消息**不能**放进 messages 数组，必须提取为顶层 "system": <拼接文本>。
      2. 工具 schema 形态是 {"name","description","input_schema"}，不是 OpenAI 的嵌套 function。
      3. assistant 的 tool_use 是 content block：{"type":"tool_use","id","name","input":{...}}
      4. tool 结果必须作为 **user** 消息的 content block：
         {"role":"user","content":[{"type":"tool_result","tool_use_id":...,"content":<str>}]}
         且连续的多个 tool_result 必须合并进同一个 user 消息。
      5. 响应 content 是 block 列表：取 type=="text" 的文本拼接；type=="tool_use" 转 ToolCall。
      6. stop_reason 映射："end_turn"->"stop"、"tool_use"->"tool_calls"、"max_tokens"->"length"。
      7. usage 字段名是 input_tokens/output_tokens，需映射到 prompt/completion。
      8. [v2 新增] 无 tool_use 时 finish_reason 一律映射为 "stop"；`content` 为空但
         `stop_reason=="max_tokens"` 时必须给出 `finish_reason="length"`（供 §9.4.6 的截断分支使用）。
         注意：**不复现** v1 的 "content_filter"（Anthropic 无此 stop_reason）。
    """
    name = "anthropic"

class EchoLLM(BaseLLMClient):
    """离线占位 provider：不需要 key、不联网。
    行为：返回一段包含收到的消息条数与最后一条用户消息前 200 字符的文本；
    若 tools 非空且最后一条用户消息包含 'use:<tool_name>'，则返回一个对应的 tool_call。
    [v2 变更] 触发规则冻结：`use:<name>` 后可跟一个**可选** JSON 对象作为参数，
    如 `use:add {"a":1,"b":2}`；缺失或非法 JSON 时用 `{}`。
    用途：CLI 无 key 演示、文档示例、冒烟测试（不是单元测试的确定性替身，ScriptedLLM 才是）。"""
    name = "echo"
    supports_tool_calling = True
    requires_api_key = False

# 模块级注册（冻结，registry.py 依赖这些名字）
PROVIDER_CLASSES: dict[str, type[LLMClient]] = {
    "openai": OpenAIChatClient,
    "openai-compatible": OpenAICompatibleClient,
    "deepseek": DeepSeekChatClient,          # [v2 新增]
    "anthropic": AnthropicChatClient,
    "echo": EchoLLM,
}
```

### 6.5 `llm/registry.py`

```python
class LLMRegistry:
    def __init__(self, factories: Mapping[str, Callable[..., LLMClient]] | None = None) -> None: ...
    def register(self, name: str, factory: Callable[..., LLMClient], *,
                 override: bool = False) -> None:
        """重名且 override=False -> ConfigError"""
        ...
    def create(self, name: str, **kwargs: Any) -> LLMClient:
        """未知 name -> ConfigError（消息里**必须**列出 available()，[v2 变更] 原先漏了这句）"""
        ...
    def available(self) -> list[str]: ...          # 已排序
    def __contains__(self, name: object) -> bool: ...

def get_default_registry() -> LLMRegistry:
    """首次调用时从 providers.PROVIDER_CLASSES 懒加载（函数内 import providers）。"""
    ...

def reset_default_registry() -> None:
    """[v2 新增] 清空默认 LLM 注册表（测试隔离用，与 tools 侧同名函数对称）。"""
    ...

def build_llm(config: LLMConfig | None = None, **overrides: Any) -> LLMClient:
    """由 LLMConfig 造 client；config 为 None 时用 LLMConfig.from_env()。"""
    ...

def get_llm(spec: str | LLMClient, **kwargs: Any) -> LLMClient:
    """spec 支持 'openai'、'openai:gpt-4o-mini'、'openai-compatible:qwen@http://host/v1'。
    ':' 后为 model；'@' 后为 base_url。已是 LLMClient 实例则原样返回。"""
    ...
```

### 6.6 `llm/scripted.py`（离线确定性测试的核心）

```python
@dataclass
class ScriptedResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None          # None -> 自动：有 tool_calls 则 "tool_calls" 否则 "stop"
    usage: TokenUsage | None = None           # None -> 用 ScriptedLLM.default_usage
    error: BaseException | None = None        # 非 None -> 抛该异常（模拟 provider 故障）
    delay_s: float = 0.0                      # 模拟网络延迟（语义见下）
    stream_chunks: list[str] | None = None    # [v2 新增] astream_chat 用；None -> [content]
    stream_error: BaseException | None = None # [v2 新增] 在第 N 个 chunk 之后抛（模拟流中断）

    # ---- 构造器（测试的主要入口）----
    @classmethod
    def text(cls, content: str, **kwargs: Any) -> "ScriptedResponse": ...

    @classmethod
    def tool(cls, name: str, arguments: Mapping[str, Any] | None = None, *,
             call_id: str | None = None, content: str = "",
             **kwargs: Any) -> "ScriptedResponse":
        """单个 tool call。
        [v2 变更] call_id=None 表示**占位**（ToolCall.id = ""），真正的 id 在
        ScriptedLLM.achat 消费时分配（见下方规则 1）。这里拿不到实例，无法递增。"""
        ...

    @classmethod
    def tool_raw(cls, name: str, raw_arguments: str, *, call_id: str | None = None,
                 content: str = "", **kwargs: Any) -> "ScriptedResponse":
        """[v2 新增] 造一个"模型给了非法 JSON 的 arguments"的响应。
        内部：ToolCall(name=name, id=call_id or "", raw_arguments=raw_arguments,
                      arguments={"__raw__": raw_arguments})。
        没有它，测试只能手搓 ToolCall 且极易漏掉 raw_arguments。"""
        ...

    @classmethod
    def tools(cls, *calls: ToolCall | tuple[str, Mapping[str, Any]],
              content: str = "", **kwargs: Any) -> "ScriptedResponse":
        """多个并发 tool call。元素可以是 ToolCall 或 (name, args) 元组。"""
        ...

    @classmethod
    def react(cls, thought: str = "", *, action: str | None = None,
              action_input: Mapping[str, Any] | None = None,
              final: str | None = None, **kwargs: Any) -> "ScriptedResponse":
        """渲染文本 ReAct 载荷（与 parser 的语法严格对齐）：
           action 非 None:
               'Thought: {thought}\\nAction: {action}\\nAction Input: {json}'
               （action_input 为 None 时渲染 '{}'）
           final 非 None:
               'Thought: {thought}\\nFinal Answer: {final}'
           两者都 None -> 'Thought: {thought}'
        """
        ...

    @classmethod
    def error(cls, exc: BaseException, **kwargs: Any) -> "ScriptedResponse": ...

class ScriptedLLM(BaseLLMClient):
    """脚本化假 LLM：按预设队列返回响应。零依赖、离线、确定性、可断言调用历史。"""

    supports_tool_calling: bool = True
    requires_api_key: bool = False
    name: str = "scripted"

    def __init__(
        self,
        responses: Sequence[ScriptedResponse | str | Callable[[Sequence[Message]], ScriptedResponse | str]] = (),
        *,
        model: str = "scripted-1",
        default_usage: TokenUsage | None = None,
        loop: bool = False,               # True -> 队列耗尽后从头循环
        strict: bool = True,              # True -> 耗尽时抛 ScriptedExhaustedError
        latency_s: float = 0.0,           # 每次调用额外 sleep（经由 sleep_fn）
        on_event: LowLevelEvent | None = None,
        config: LLMConfig | None = None,
    ) -> None: ...

    # ---- 运行时状态（测试断言用）----
    calls: list[ScriptedCall]             # 每次 achat 追加一条
    exhausted_count: int                  # 队列耗尽被触发的次数
    events: list[tuple[str, dict[str, Any]]]   # [v2 新增] 由 _emit 记录，便于断言事件

    @property
    def remaining(self) -> int: ...       # loop=True 时返回 0
    @property
    def call_count(self) -> int: ...

    def push(self, response: ScriptedResponse | str) -> None:
        """追加到队尾（可在运行中动态喂响应，用于测试中途改变行为）。"""
        ...

    def extend(self, responses: Sequence[ScriptedResponse | str]) -> None: ...

    def assert_exhausted(self) -> None:
        """断言队列已耗尽且所有响应都被消费（用 assertEqual 实现，便于 unittest 报错定位）。"""
        ...

    def last_call(self) -> ScriptedCall:
        """无调用时抛 AssertionError。"""
        ...

    def last_messages(self) -> list[Message]: ...

    async def achat(self, messages, *, tools=None, tool_choice=None, temperature=None,
                    max_tokens=None, stop=None, **kwargs) -> LLMResponse: ...

    async def astream_chat(self, messages, **kwargs) -> AsyncIterator[LLMStreamChunk]:
        """[v2 新增，必须实现] 逐个 yield LLMStreamChunk(delta=chunk, index=i)；
        最后一个 chunk 带 finish_reason=resp.finish_reason or ("tool_calls" if resp.tool_calls else "stop")；
        resp.stream_error 非 None 时在最后一个 chunk 之后抛出。
        同时把这次调用记入 self.calls（与 achat 一致）。"""
        ...

    # 工具：从调用历史里抽取"渲染给模型的工具名列表"（断言工具确实被暴露给模型）
    def tool_names_seen(self, index: int = -1) -> list[str]:
        """[v2 变更] 语义冻结：**只**从 `calls[index].tools` 提取 `name`
        （同时兼容 openai 的 tools[i].function.name 与 anthropic 的 tools[i].name 两种形态）；
        `tools is None`（文本模式）时返回 **[]**（不去解析 system prompt）。"""
        ...
```

**`ScriptedLLM` 的确定性要求（冻结，`[v2 变更]` 已修掉 v1 的两处不可实现规则）**：

1. **`call_id` 的分配时机**：`ScriptedResponse.tool/tools/tool_raw` 的 `call_id=None` 是**占位**
   （`ToolCall.id == ""`）。真正的 id 在 `ScriptedLLM.achat` 里**消费该响应时**分配：
   ```python
   for tc in resp.tool_calls:
       if tc.id == "":
           tc.id = f"call_{self._seq}"
           self._seq += 1
   ```
   `self._seq: int = 0` 是**实例属性**、从 0 递增、**一次 achat 内的多个 call 依次递增**。
   显式传入的 `call_id` **原样保留且不消耗 seq**。（v1 的"由 `ScriptedResponse.tool` 生成"在
   Python 里不可能实现——`ScriptedResponse.tool` 是 classmethod，构造时 ScriptedLLM 实例还不存在。）
2. `LLMResponse.latency_ms = delay_s * 1000`（真实耗时不计入，保证可断言）。
3. `usage` 未给时用 `default_usage`；`default_usage` 为 None 时用
   `TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)`（**固定常量**）。
4. `error` 响应：抛异常前也要把这次调用**记入 `calls`**（否则"重试了几次"无法断言）。
5. **`delay_s` 的语义（冻结）**：若 `self.config.timeout_s` 非 None 且
   `delay_s > self.config.timeout_s` → `await asyncio.wait_for(self._sleep(delay_s), config.timeout_s)`
   并抛 `LLMTimeoutError(timeout_s=config.timeout_s)`；否则 `await self._sleep(delay_s)`。
   `latency_s`（构造参数）**无条件** `await self._sleep(latency_s)`，不受 timeout 约束。
6. **`strict` 与 `loop` 同时为真时的优先级**：`loop=True` 优先 —— 永不抛
   `ScriptedExhaustedError`，`remaining` 恒为 `0`；`exhausted_count` **只在 `strict and not loop`
   时递增**（loop 模式下重复消费不算"耗尽"）。
7. **`ScriptedCall` 的记录（冻结）**：
   ```python
   self.calls.append(ScriptedCall(
       index=len(self.calls),
       messages=list(messages),                                  # 浅拷贝外层 list
       tools=None if tools is None else [dict(t) for t in tools], # 逐项浅拷贝
       kwargs={k: v for k, v in kwargs.items()},                 # 不含 messages/tools
       response=resp,
   ))
   ```
   - `messages` **必须是浅拷贝**：若 Agent 把 `state.messages` 同一个 list 传进来，
     后续轮次的追加会污染已记录的 `calls[i].messages`，测试的数量断言会静默失效。
   - `kwargs` **含** Agent 显式传入的 `temperature` / `max_tokens` / `tool_choice` / `stop`
     （**为 None 也保留**，便于断言"未传"），**不含** `messages`/`tools`。
   - **Message 对象不深拷贝**（契约：Agent 追加消息后不得再原地修改已有的 Message）。
   - 若 `responses` 元素是 `Callable`，调用时同样传 `list(messages)` 副本；返回值也计入 `calls`。
8. **未知 kwargs 的行为**：`ScriptedResponse.text/tool/react` 等构造器的 `**kwargs` 直接
   `cls(**{...})` 构造，**未知键抛 `TypeError`**（不要静默丢弃 —— 静默会让测试断言写错却不报错）。

#### 6.6.1 真实测试代码示例（测试实现者照抄，`[v2 变更]` 已修正 v1 的两处笔误）

```python
import asyncio
import unittest

from liteagent import Agent, AgentConfig
from liteagent.llm import ScriptedLLM, ScriptedResponse
from liteagent.tools import ToolRegistry, ToolExecutor, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestReActNativeMode(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry([add])

    def _agent(self, responses) -> Agent:
        llm = ScriptedLLM(responses)
        agent = Agent(llm=llm, tools=self.registry, config=AgentConfig(max_steps=5))
        self.llm = llm                      # 局部变量保留，便于断言调用历史
        return agent

    async def test_native_tool_call_then_answer(self) -> None:
        agent = self._agent([
            ScriptedResponse.tool("add", {"a": 2, "b": 3}),
            ScriptedResponse.text("Final Answer: 5"),
        ])

        result = await agent.arun("what is 2+3?")

        self.assertEqual("FINISHED", result.status.value)
        self.assertEqual("5", result.output)
        self.assertEqual(2, result.steps)
        self.assertEqual(1, len(result.tool_results))
        self.assertTrue(result.tool_results[0].ok)
        self.assertEqual("5", result.tool_results[0].content)
        self.llm.assert_exhausted()                     # 断言式方法无返回值契约

    async def test_parallel_tool_calls_preserve_order(self) -> None:
        agent = self._agent([
            ScriptedResponse.tools(("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})),
            ScriptedResponse.text("done"),
        ])
        result = await agent.arun("add twice")
        self.assertEqual(["2", "4"], [r.content for r in result.tool_results])

    async def test_validation_error_is_fed_back_to_model(self) -> None:
        agent = self._agent([
            ScriptedResponse.tool("add", {"a": "not-an-int", "b": 1}),   # 故意类型错
            ScriptedResponse.text("Final Answer: sorry"),
        ])
        result = await agent.arun("bad args")
        first = result.tool_results[0]
        self.assertFalse(first.ok)
        self.assertEqual("ToolValidationError", first.error_type)
        self.assertTrue(first.content.startswith("ERROR(ToolValidationError):"))  # §4.3 冻结
        # 关键断言：错误被回灌进第二次 LLM 调用的消息里
        second_call_messages = self.llm.calls[1].messages
        self.assertTrue(any("ToolValidationError" in m.content for m in second_call_messages))
```

---

## 7. `liteagent/tools/` —— 工具系统

### 7.1 `tools/schema.py` —— type hints 到 JSON Schema 的反射

这是全项目最容易写歪的模块，逐条冻结。

```python
SCHEMA_DRAFT: str = "https://json-schema.org/draft/2020-12/schema"
MAX_SCHEMA_DEPTH: int = 8
_UNSET: Any = object()          # Param.default 的哨兵（模块级，必须先于 Param 定义）

@dataclass
class SchemaResult:
    schema: dict[str, Any]                 # {"type":"object","properties":{...},"required":[...], "additionalProperties": False}
    descriptions: dict[str, str]           # 从 docstring 抽出的每参描述
    warnings: list[str]                    # 降级到 {} 的字段、不支持的注解等
    name: str
    description: str

def build_tool_schema(func: Callable[..., Any], *,
                      name: str | None = None,
                      description: str | None = None,
                      parameters: dict[str, Any] | None = None,
                      docstring_style: str = "auto",     # "auto"|"google"|"sphinx"|"none"
                      ) -> SchemaResult:
    """parameters 非 None 时直接短路返回（用户完全接管 schema，仍会跑一遍 validate_schema 做自检）。"""
    ...

# ---- 内部但公开可测的子函数（测试可以直接调用，签名冻结）----
def annotation_to_schema(annotation: Any, *, depth: int = 0,
                         seen: frozenset[type] = frozenset()) -> tuple[dict[str, Any], list[str]]:
    """返回 (schema_fragment, warnings)。深度/环超限时返回 ({}, [warning])。"""
    ...

def is_optional(annotation: Any) -> bool:
    """get_origin 是 Union 且 type(None) 在 get_args 中；或注解 is type(None)。"""
    ...

def annotated_metadata(annotation: Any) -> tuple[Any, ...]:
    """[v2 新增] 返回 Annotated 的 metadata 元组，非 Annotated 时返回 ()。
    冻结实现（M-1）：`return getattr(annotation, "__metadata__", ())`。
    **禁止** `isinstance(annotation, Annotated)`（3.10 实测恒 False，静默失效）。"""
    ...

def _meta_default(meta: Any) -> Any:
    """[v2 新增] 从一条 metadata 里取 default：Param -> .default；
    pydantic FieldInfo 鸭子类型 -> .default；其它 -> _UNSET。"""
    ...

def parse_docstring(doc: str | None, *, style: str = "auto") -> tuple[str, dict[str, str]]:
    """返回 (summary, {param_name: description})。summary = 第一段落，压缩内部换行与多余空格。
    style="auto" 时先尝试 Google（Args:/Arguments:/参数:），失败再尝试 Sphinx（:param x:）。"""
    ...

def validate_instance(instance: Any, schema: dict[str, Any], *,
                      path: str = "$", max_errors: int = 32) -> list[str]:
    """纯 stdlib 的 JSON Schema 子集校验器。返回错误消息列表，空列表 = 通过。
    支持的关键字：type, enum, const, properties, required, additionalProperties(bool 或 schema),
                 items, minItems, maxItems, uniqueItems, minimum, maximum,
                 exclusiveMinimum, exclusiveMaximum, minLength, maxLength, pattern,
                 anyOf, oneOf(取第一个通过的分支), nullable。
    不支持的 key 一律**忽略**（宽容），并在返回值里不产生错误 —— 因为 schema 可能来自
    外部 provider 或 pydantic，严格失败会造成假阳性。
    路径表示：'$.a.b[0]'。
    类型判定：bool 必须在 int 之前判定（isinstance(True, int) 为 True）。
    数字：int 满足 "number"；float 不满足 "integer"
      （**冻结：不满足**，保持实现简单可预测，并在文档中说明）。
    [v2 新增] 非必填字段（不在 required 且 schema 无 default）收到显式 None 时**直接放行**（D-05 的配套）。
    """
    ...

def validate_schema(schema: dict[str, Any]) -> list[str]:
    """校验我们**自己生成**的 schema 是否合法（内部自检用，如 required 里的名字必须出现在
    properties 中）。返回问题列表。仅开发/测试期调用。"""
    ...

def to_openai_tool(spec: "ToolSpec") -> dict[str, Any]:
    """{"type":"function","function":{"name","description","parameters"}}"""
    ...

def to_anthropic_tool(spec: "ToolSpec") -> dict[str, Any]:
    """{"name","description","input_schema": parameters}"""
    ...
```

#### 7.1.1 注解到 JSON Schema 的映射表（冻结，逐行实现）

| Python 注解 | JSON Schema | 备注 |
|---|---|---|
| `str` | `{"type": "string"}` | |
| `int` | `{"type": "integer"}` | |
| `float` | `{"type": "number"}` | |
| `bool` | `{"type": "boolean"}` | **必须先于 `int` 判断** |
| `None` / `type(None)` | `{}` | 任意类型 |
| `Any` / 无注解 | `{}` | |
| `Literal["a","b"]` | `{"enum": ["a","b"], "type": "string"}` | type 由所有字面量值的类型推断；混型则省略 type |
| `Literal[1,2]` | `{"enum": [1,2], "type": "integer"}` | |
| `Literal[True,False]` | `{"enum": [true,false], "type": "boolean"}` | |
| `Literal["a", 1]` | `{"enum": ["a", 1]}` | 无 type |
| `MyEnum`（`Enum` 子类） | `{"enum": [m.value...]}` | 值的类型推断 type；成员顺序按定义顺序 |
| `IntEnum` 子类 | `{"enum": [1,2], "type": "integer"}` | |
| `list[X]` / `List[X]` | `{"type": "array", "items": <X>}` | |
| `list` / `List`（裸） | `{"type": "array"}` | |
| `list[dict]` | `{"type": "array", "items": {"type": "object"}}` | |
| `set[X]` / `frozenset[X]` | `{"type": "array", "items": <X>, "uniqueItems": true}` | |
| `tuple[X, ...]` | `{"type": "array", "items": <X>}` | 变长元组 |
| `tuple[X, Y]` | `{"type": "array", "items": {}, "minItems": 2, "maxItems": 2}` | **降级**：不支持 `prefixItems`，warnings 里记一条 |
| `dict[str, X]` | `{"type": "object", "additionalProperties": <X>}` | |
| `dict`（裸） | `{"type": "object"}` | |
| `Optional[X]` | `<X>`，且**不进入 required** | **不生成 `"type": ["X","null"]`**，理由见 D-05 |
| `Union[A, B]`（非 Optional） | `{}` + warning | 多态不支持，降级为任意 |
| 嵌套 `@dataclass` | 递归对象 schema | 见 7.1.2 |
| pydantic `BaseModel` 子类 | `cls.model_json_schema()` 后**清洗**（删 `title`/内联 `$defs`） | 仅当 `PYDANTIC_AVAILABLE` |
| `Annotated[X, meta...]` | `<X>` + meta 增强 | 见 7.1.3 |
| 其它（自定义类、Callable 等） | `{}` + warning | |

**为什么 `Optional[X]` 不生成 `"type": ["X","null"]`（冻结决策 D-05）**：
OpenAI 与 Anthropic 的 function calling 对联合类型支持不一致，很多兼容端（vLLM/Ollama 的
JSON-schema-to-grammar 转换）会直接报错。做法是「Optional 只表达**非必填**，不表达**可为 null**」，
校验器对非必填字段收到 `None` 时直接放行。这是一个明确的、写在文档里的取舍。

#### 7.1.2 嵌套 dataclass 的映射

1. 若是 `dataclasses.is_dataclass(annotation)`（且是类型而非实例）：
   递归生成 `{"type":"object","properties":{...},"required":[...],"additionalProperties":false}`。
2. 嵌套层级同样走 `annotation_to_schema`，因此嵌套 dataclass 里的 `Annotated`/`Literal`/嵌套 dataclass 都生效。
3. 嵌套字段的默认值规则与顶层完全一致（`default is empty` **且 `default_factory is empty`**
   且非 Optional 才 required）。`[v3 变更]` v2 只写了 `default is empty`，**漏掉了
   `field(default_factory=...)` 这一支**：`field(default_factory=list)` / `field(default_factory=utc_now)`
   是"有默认值"，绝不能进 required —— 否则生成的 JSON Schema 会告诉模型"这个字段必须提供"，
   是**对外可见的 schema 错误**（模型被要求提供本可省略的字段）。
   回归测试：`tests/test_tools_schema.py::NestedDataclassTests::test_default_factory_fields_are_not_required`、
   `::test_default_factory_branch_is_load_bearing_for_validation`。
4. 嵌套 dataclass 的字段描述：优先取该字段的 `Annotated[..., Param(description=...)]`；
   否则取字段名（**不去解析嵌套 dataclass 的 docstring 的 Args 段**——理由：Google 风格里
   嵌套类型的 docstring 结构不可靠，且会引入解析歧义。此限制需在 `TOOLS.md` 说明）。
5. **环检测**：`seen` 集合记录当前递归路径上的类型；若目标类型已在 `seen` 中，返回 `{}` 并
   warning `"recursive dataclass <name> truncated at depth N"`。
6. **深度上限**：`depth > MAX_SCHEMA_DEPTH` 时返回 `{}` + warning。
7. 若 `PYDANTIC_AVAILABLE` 且注解是 pydantic 模型：调用 `model_json_schema()`，然后
   **内联 `$defs`**（把 `{"$ref": "#/$defs/X"}` 替换为 `$defs["X"]`），并删除所有 `title` 键。
   内联实现为 `_inline_refs(schema) -> dict`。该函数必须在无 pydantic 时也不可用（只在有引用时调用）。

#### 7.1.3 `Annotated` 元数据识别（冻结）

```python
@dataclass(frozen=True)
class Param:
    """框架自带的结构化参数元数据（零依赖，不 import pydantic）。"""
    description: str | None = None
    default: Any = _UNSET                 # 哨兵，区别于 None
    ge: float | None = None               # -> minimum
    le: float | None = None               # -> maximum
    gt: float | None = None               # -> exclusiveMinimum
    lt: float | None = None               # -> exclusiveMaximum
    min_length: int | None = None         # str -> minLength, list -> minItems
    max_length: int | None = None         # str -> maxLength, list -> maxItems
    pattern: str | None = None            # -> pattern
    examples: tuple[Any, ...] = ()        # -> examples
    enum: tuple[Any, ...] | None = None   # 覆盖 -> enum
    title: str | None = None              # **会被丢弃**，仅为 API 完整性保留
```

**`[v2 变更]` 取 metadata 的唯一方式（M-1 实测：`isinstance(ann, Annotated)` 在 3.10 恒 False）**：

```python
metas = getattr(annotation, "__metadata__", ())     # 非 Annotated -> ()
```

识别顺序（对 `metas` 里每个 meta）：
1. `isinstance(meta, Param)` → 直接使用。
2. **鸭子类型**识别 pydantic `FieldInfo`：`hasattr(meta, "description") and hasattr(meta, "default")`
   且 `type(meta).__module__.startswith("pydantic")`。读取
   `description / default / ge / le / gt / lt / min_length / max_length / pattern / examples`。
   **绝不 import pydantic**。
3. `isinstance(meta, str)` → 当作 description（`Annotated[int, "count of items"]` 这种写法很常见）。
4. 其它对象：忽略并记 warning。

**`Param.default` 的语义**：**若设置了，则该参数视为可选**（即使函数签名里没有默认值），
schema 的 `default` 取该值，且不进入 `required`。
`Param.default` 的字段默认值是模块级哨兵 `_UNSET: Any = object()`（定义在 `Param` 之前）。
**禁止**用 `None` 当"未设置"（会造成"显式 None"与"未设置"不可区分）。

#### 7.1.4 docstring -> description（冻结规则）

`description` 来源链：函数参数 `description=` > **docstring summary** > `name` > `""`。

summary 提取（`parse_docstring`）：
1. `inspect.getdoc(func)`；None → `("", {})`。
2. 用 `textwrap.dedent` + 去首尾空行。
3. summary = 到**第一个空行**为止的所有行，行间用单个空格拼接，压缩连续空白。
4. 若首个空行不存在（整段就是 summary），summary 就是全部内容。

逐参描述（Google 风格，`style` 为 `"auto"`/`"google"`）：
1. 找到标题行，正则（大小写不敏感）：`^\s*(Args|Arguments|Parameters|参数)\s*:\s*$`
2. 从下一行开始，直到遇到**同级或更低缩进**的下一个 section 标题
   （`^\s*(Returns|Raises|Yields|Examples|Notes|Example|Note|Attributes|返回|异常|示例)\s*:\s*$`）
   或文档结束。
3. 段内每行匹配条目正则：`^\s*(\*{0,2}\w+)\s*(\(([^)]*)\))?\s*:\s*(.*)$`
   - group(1) = 参数名（剥离 `*`）
   - group(3) = 类型（**忽略**，我们以 type hints 为准；若类型里含 `optional` 字样且签名无默认值，
     记录 warning 但**不改变 required 判定**）
   - group(4) = 描述首行
4. 后续**缩进更深**的行作为描述续行，用空格拼接。
5. 参数名不在签名里的条目 → 丢弃 + warning。
6. 只保留签名中实际存在的参数名。

Sphinx 风格（`style` 为 `"sphinx"` 或 auto 的兜底）：正则
`^\s*:param\s+(\w+)\s*:\s*(.*)$`，续行为更深缩进；`^\s*:type\s+` 忽略。

`style="none"` → 只取 summary，不做逐参解析。

**逐参描述的优先级（冻结）**：`Annotated[..., Param(description=...)]` >
docstring 的 Args 段 > 无（不写 `description` 键）。

#### 7.1.5 required 判定（冻结，唯一公式，`[v2 变更]` v1 的公式在 3.10 上恒失效）

```python
metas = getattr(annotation, "__metadata__", ())          # M-1：禁止 isinstance(x, Annotated)
required = (
    <该参/字段没有默认值>                                  # 见下：来源随上下文不同
    and not is_optional(annotation)
    and not any(_meta_default(m) is not _UNSET for m in metas)
)
```

`<该参/字段没有默认值>`（`[v3 变更]` 补齐嵌套 dataclass 分支 —— v2 只写了顶层的那一支）：

- **顶层函数参数**：`param.default is inspect.Parameter.empty`（`param` 是
  `inspect.Parameter`；函数参数没有 `default_factory` 这一概念）。
- **嵌套 dataclass 字段**（§7.1.2 规则 3）：`field.default is dataclasses.MISSING`
  **且** `field.default_factory is dataclasses.MISSING`（`field` 是 `dataclasses.Field`）。
  `field(default_factory=list)` / `field(default_factory=utc_now)` 是"有默认值"，
  **绝不能**进 required。回归测试：
  `tests/test_tools_schema.py::NestedDataclassTests::test_default_factory_fields_are_not_required`、
  `::test_default_factory_branch_is_load_bearing_for_validation`。

（两条分支共用"唯一公式"的其余两项：`is_optional` 与 `_meta_default`。）

`_meta_default(m)` 见 §7.1：`Param` 取 `.default`、pydantic `FieldInfo` 鸭子类型取 `.default`、
其它返回 `_UNSET`。

`*args`（VAR_POSITIONAL）**永不 required**（映射为 array，缺省即空数组）。
`**kwargs`（VAR_KEYWORD）→ **抛 `ToolDefinitionError`**（我们无法为任意键生成 schema；
若确需，用户必须传 `parameters=` 显式接管）。

**必须有的三条测试用例（§12 已列入 `test_tools_schema.py`）**：
`Annotated[int, Param(default=5)]` 不进 required 且 `properties['x']['default'] == 5`；
`Annotated[int, 'desc']` 只加 description 不影响 required；
`Annotated[str, Param(description='d')]` 的逐参描述优先于 docstring。

### 7.2 `tools/base.py`

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]              # 参数 JSON Schema
    func: Callable[..., Any]
    is_async: bool = False
    pass_style: str = "kwargs"              # "kwargs" -> func(**args) ; "mapping" -> func(args)
    tags: tuple[str, ...] = ()
    dangerous: bool = False                 # 只影响**展示过滤**（list(include_dangerous=)）
    requires_approval: bool = False         # [v2 新增] 影响**控制流**（§7.4.1 步骤 4.5）
    idempotent: bool = True
    timeout_s: float | None = None          # None -> 继承 ExecutorConfig.default_timeout_s
                                            # NO_TIMEOUT -> 显式禁用超时
    max_retries: int | None = None          # None -> 用 ExecutorConfig.retry_policy.max_retries
    warnings: tuple[str, ...] = ()
    version: str = "1"

    def to_openai_schema(self) -> dict[str, Any]: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...        # 不含 func；含 "signature": str
    def summary_line(self) -> str:
        """'name(a: integer, b: string) - description 首行'，给文本 ReAct 的 {tools} 用。"""
        ...

class Tool:
    """装饰器产物。可调用、带 spec、可被注册。"""

    def __init__(self, spec: ToolSpec) -> None: ...

    spec: ToolSpec

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> dict[str, Any]: ...
    @property
    def is_async(self) -> bool: ...
    @property
    def raw(self) -> Callable[..., Any]: ...        # 原始函数（测试常用）

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """直接调用原始函数（不做校验）。"""
        ...

    def run(self, args: Mapping[str, Any]) -> Any:
        """[v2 新增，**必须实现**] **同步**调用原始函数（按 pass_style）。
        - is_async 为 True 时抛 `TypeError("tool <name> is async; use arun()")`
          （防止 M-2 的静默失效：`asyncio.to_thread(tool.arun, args)` 只会返回一个没人 await 的协程）。
        - pass_style=="mapping" -> self.spec.func(dict(args))
        - pass_style=="kwargs"  -> self.spec.func(**dict(args))
        同步工具在 executor 里**只走这一条路径**。"""
        ...

    async def arun(self, args: Mapping[str, Any]) -> Any:
        """按 pass_style 调用原始函数；is_async 时 await。
        is_async 为 False 时**同步调用**原始函数（等价 self.run(args)）。
        **不做 schema 校验**——校验只在 ToolExecutor 里做一次，避免两处语义漂移。"""
        ...

    def to_openai_schema(self) -> dict[str, Any]: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...                  # '<Tool name(parameters) at 0x..>'

    @classmethod
    def from_function(cls, func: Callable[..., Any], **kwargs: Any) -> "Tool": ...

@overload
def tool(func: Callable[..., Any]) -> Tool: ...
@overload
def tool(*, name: str | None = ..., description: str | None = ...,
         parameters: dict[str, Any] | None = ..., tags: Sequence[str] = ...,
         dangerous: bool = ..., requires_approval: bool = ..., idempotent: bool = ...,
         timeout_s: float | None = ..., max_retries: int | None = ...,
         auto_register: bool = ..., docstring_style: str = ...) -> Callable[[Callable[..., Any]], Tool]: ...

def tool(func: Callable[..., Any] | None = None, **kwargs: Any) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """同时支持 @tool 与 @tool(...) 两种用法。
    返回的 Tool 对象：不清除原函数（func 属性仍在 Tool.spec.func / Tool.raw）。

    [v2 变更] `auto_register` 的语义**写死**：
      - `auto_register=True` 等价于 `get_default_registry().register(t, override=False)`；
        重名 -> `ToolDefinitionError`（**不静默覆盖**）。
      - `auto_register=False`（默认）**不得触碰任何全局状态**（受 test_zero_dependency 守门）。
    """
    ...

def make_function_tool(*, name: str, description: str,
                       parameters: dict[str, Any] | None = None,
                       func: Callable[[dict[str, Any]], Any],
                       is_async: bool = False,
                       tags: Sequence[str] = (), dangerous: bool = False,
                       requires_approval: bool = False,
                       idempotent: bool = True, timeout_s: float | None = None,
                       max_retries: int | None = None) -> Tool:
    """动态构建工具（用于 Hierarchical 的 delegate 工具、用户运行时造工具）。
    pass_style 固定为 "mapping"：func 接收一个已校验的 args dict。"""
    ...

def is_tool(obj: Any) -> bool: ...                  # isinstance(obj, Tool)

# ---- 协作式取消（配合 §7.4.1 步骤 5.c 的超时语义，D-09）----
_CANCEL_FLAG: contextvars.ContextVar["threading.Event | None"] = contextvars.ContextVar(
    "liteagent_tool_cancel", default=None)

def current_cancel_flag() -> "threading.Event | None":
    """返回当前工具调用的取消信号（`threading.Event`），无调用上下文时返回 None。
    长耗时的同步工具应在循环里检查 `flag is not None and flag.is_set()` 并主动返回。
    为什么需要它：asyncio.wait_for 无法中断已经跑在 worker 线程里的同步代码（D-09）。"""
    ...

@contextmanager
def cancel_scope() -> Iterator["threading.Event"]:
    """executor 内部使用：进入时用 `threading.Event()` 造一个新 Event 并 set 到 contextvar，
    退出时 reset。**必须在 `asyncio.run_in_executor` / `asyncio.to_thread` 的调用方所在线程
    设置**（M-5 实测：contextvar 的**读**能穿透到 worker 线程，**写**不能回传）。
    [v2 变更] **每个 attempt 必须重新进入一次**（v1 允许放在重试循环外，
    会导致第 2 次尝试拿到已 set 的同一个 Event，工具一进循环就自杀，重试全部秒失败）。"""
    ...
```

**`@tool` 的完整语义小结（面试要能讲）**：
1. 反射 `inspect.signature` 拿参数与默认值。
2. 反射 `typing.get_type_hints(func, include_extras=True)` 拿注解；解析失败（NameError，
   常见于 `from __future__ import annotations` + 局部导入）则回退 `func.__annotations__`
   并把字符串注解按 `typing.get_type_hints` 失败处理 → 记 warning，该参数降级为 `{}`。
3. 按 §7.1.1 表映射成 JSON Schema；`Annotated` 元数据增强约束与描述（§7.1.3）。
4. `inspect.getdoc` → summary 进 `description`，`Args:` 段进各属性 `description`。
5. `bool` 必须在 `int` 前判定。
6. `**kwargs` → `ToolDefinitionError`。
7. 所有降级都记进 `ToolSpec.warnings`（可观测、可测试），**不静默**。
8. **不注册**（除非 `auto_register=True`）：import 无副作用，测试可重复。

### 7.3 `tools/registry.py`

```python
class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None, *,
                 aliases: Mapping[str, str] | None = None) -> None: ...

    def register(self, tool: Tool, *, override: bool = False) -> Tool:
        """重名且 override=False -> ToolDefinitionError；
        名字不匹配 ToolSpec.name 的合法模式 ^[A-Za-z_][A-Za-z0-9_.-]{0,63}$ -> ToolDefinitionError。"""
        ...
    def register_function(self, func: Callable[..., Any], **kwargs: Any) -> Tool:
        """便捷：等价 register(Tool.from_function(func, **kwargs))。"""
        ...
    def unregister(self, name: str) -> None: ...        # 不存在 -> ToolNotFoundError
    def get(self, name: str) -> Tool: ...               # 未命中 -> ToolNotFoundError(available=...)
    def try_get(self, name: str) -> Tool | None: ...
    def alias(self, alias: str, name: str) -> None:
        """别名解析在 get/try_get 里做，别名不进 names()。别名指向不存在 -> ToolNotFoundError。"""
        ...
    def names(self) -> list[str]: ...                   # 排序；不含别名
    def list(self, *, tags: Sequence[str] | None = None,
             include_dangerous: bool = True) -> list[Tool]: ...
    def subset(self, names: Sequence[str]) -> "ToolRegistry":
        """未命中任一名字 -> ToolNotFoundError（早失败）。"""
        ...
    def schemas(self, *, fmt: str = "openai") -> list[dict[str, Any]]:
        """fmt: "openai" | "anthropic"。未知 fmt -> ConfigError。"""
        ...
    def to_prompt(self, *, fmt: str = "text",
                  max_tools: int = DEFAULT_MAX_TOOLS_IN_PROMPT) -> str:
        """[v2 变更] `max_tools` 默认从 64 降到 20（DEFAULT_MAX_TOOLS_IN_PROMPT），
        因为文本模式下这些摘要行会直接进 system prompt 的 token 预算。
        fmt="text" -> 每行 spec.summary_line()；"json" -> 缩进 JSON 数组。
        超过 max_tools 时截断并追加 '... (N more tools omitted)'。"""
        ...
    def describe(self, name: str, *, fmt: str = "markdown") -> str:
        """单工具详情（CLI `tools show` 用），含完整参数表。"""
        ...
    def merge(self, other: "ToolRegistry", *, override: bool = False) -> "ToolRegistry":
        """返回**新** registry（不改自身）。重名且 override=False -> ToolDefinitionError。"""
        ...
    def __contains__(self, name: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tool]: ...           # 遍历 Tool，按 names() 顺序
    def to_dict(self) -> dict[str, Any]: ...

def get_default_registry() -> ToolRegistry:
    """模块级全局注册表（仅 auto_register=True 使用；测试用 reset_default_registry() 清理）。"""
    ...

def reset_default_registry() -> None:
    """[v2 变更] 冻结语义：把全局注册表**替换为一个全新的空 ToolRegistry**（而不是清空原对象），
    保证任何持有旧引用的代码不会看到"被清空"的状态。测试的 tearDown 必须调用它。"""
    ...
```

### 7.4 `tools/executor.py` —— 审批、并发、超时、重试、取消

```python
class ToolExecutor:
    def __init__(self, registry: ToolRegistry, config: ExecutorConfig | None = None,
                 *, thread_pool: "ThreadPoolExecutor | None" = None,
                 is_retryable: Callable[[BaseException], bool] | None = None,
                 on_event: LowLevelEvent | None = None) -> None:
        """[v2 变更] __init__ 里冻结创建（全部同步原语，不受 R-LOOP 约束）：
            self._pool = LoopBoundPool()                     # per-loop 的 semaphore/lock/thread_pool
            self._rng = random.Random(self.config.retry_policy.rng_seed)
            self._sleep = config.sleep_fn or config.retry_policy.sleep_fn or default_sleep
            self._seq_locks: dict[str, threading.Lock] = {n: threading.Lock()
                                                          for n in config.sequential_tools}
            self._failure_counts: dict[str, int] = {}        # 熔断计数（按工具名）
            self._external_pool = thread_pool                # 非 None 时 aclose 不关它
        `thread_pool` 的语义（冻结）：非 None 时**所有**同步工具都在它里面跑，`aclose` 不关它；
        为 None 时按 loop 懒建私有池（`_pool.thread_pool("exec", config.thread_pool_size)`），
        `aclose` 全部 shutdown。**不使用 `loop.set_default_executor`**
        （v1 的写法在 3.10 无法可靠探测"用户是否已设过"，且状态放在实例上会让第二次
        `asyncio.run` 的新 loop 静默跳过设置 —— 而"跨两次 asyncio.run 复用同一 executor"
        正是 §12 要守的场景）。"""
        ...

    registry: ToolRegistry
    config: ExecutorConfig

    async def execute(self, call: ToolCall, *, timeout_s: float | None = None,
                      max_retries: int | None = None) -> ToolResult:
        """单次调用。**永不向外抛工具异常**（唯一例外：`asyncio.CancelledError` /
        `KeyboardInterrupt`，见 §7.4.1 步骤 5 之前的冻结规则）。"""
        ...

    async def execute_many(self, calls: Sequence[ToolCall], *,
                           concurrency: int | None = None,
                           timeout_s: float | None = None) -> list[ToolResult]:
        """并发执行；**返回顺序与 calls 严格一致**（不看完成顺序）。
        `concurrency=None` -> 用 `config.max_concurrency`；每个调用**同时**受全局信号量约束。
        `timeout_s` 的语义（[v2 冻结]）：**逐调用**超时，与 `execute` 的同名参数语义一致；
        "整批总预算"不在本参数范围内（未来可加 `total_budget_s`）。
        `config.fail_fast=True` 时：首个 `ok=False` 之后取消其余任务，
        **仍然返回与 calls 等长的列表**，被取消的位置填 `ToolSkippedError`；
        **绝不抛异常**（见 §7.4.2）。"""
        ...

    def execute_sync(self, call: ToolCall, **kwargs: Any) -> ToolResult:
        """在 `finally` 里调用 `self._pool.release(loop)` 后返回；
        在运行中的 loop 内调用抛 ConfigError。"""
        ...

    async def aclose(self) -> None:
        """`self._pool.release(None)`：shutdown 全部私有线程池（wait=False, cancel_futures=True）。
        外部传入的 `thread_pool` 不关闭。**不要** shutdown loop 的默认 executor
        （v2 已不再使用它）。"""
        ...
    def close(self) -> None: ...
    async def __aenter__(self) -> "ToolExecutor": ...
    async def __aexit__(self, *exc: Any) -> None: ...
    def __enter__(self) -> "ToolExecutor": ...
    def __exit__(self, *exc: Any) -> None: ...
```

#### 7.4.0 包裹 await 的冻结取消规则（`[v2 变更]`，写在 §7.4.1 步骤 5 之前，适用全文）

1. **任何包裹 `await` 的 `except` 必须保证 `asyncio.CancelledError` 不被捕获**。
   若必须用宽 `except`，**首行**必须是 `except asyncio.CancelledError: raise`
   （M-5：`CancelledError` 继承 `BaseException`，但**禁止**写 `except BaseException` 然后
   在分支里判断类型 —— 那样很容易把取消吞成一次工具失败）。
2. **超时判定只认 `except asyncio.TimeoutError`**，绝不写成 `except Exception` 再判断类型
   （M-5：`asyncio.TimeoutError is not builtins.TimeoutError`，写 `except TimeoutError` 抓不到）。
3. **工具自身抛出的 `CancelledError` 一律视为调用方取消**，原样向上抛，**不**转成 `ToolResult`。
4. `Agent.arun` 的取消处理见 §9.4.6。

#### 7.4.1 单次调用的精确步骤（冻结顺序）

1. `tool = registry.get(call.name)`；`ToolNotFoundError` → 立即返回失败结果，
   `content` 里**列出可用工具名**（`available[:20]`）供模型自纠正。
   `metadata["feedback_kind"]="recoverable"`。不重试。
2. 取超时（`[v2 变更]` 引入 `NO_TIMEOUT` 哨兵，因为 v1 的链里**没有任何值能表达"不超时"**）：

   ```text
   if timeout_s == NO_TIMEOUT or config.default_timeout_s is None:
       effective_timeout = None                     # 不设超时
   else:
       候选 = [timeout_s, call.metadata.get("timeout_s"), tool.spec.timeout_s,
               config.default_timeout_s]
       候选 = [v for v in 候选 if v is not None and v != NO_TIMEOUT]
       effective_timeout = min(候选) if 候选 else None
   ```

   即：参数 > `call.metadata["timeout_s"]` > `tool.spec.timeout_s` > `config.default_timeout_s`，
   取**最小有效值**；`NO_TIMEOUT` 或全局 `None` 表示禁用超时。
3. 取重试：`max_retries` 参数 > `call.metadata["max_retries"]` > `tool.spec.max_retries` >
   `config.retry_policy.max_retries`。
4. **参数校验**（仅此处）：`errors = schema.validate_instance(call.arguments, tool.parameters)`；
   非空 → 返回 `ToolValidationError` 失败结果，`metadata["validation_errors"] = errors`，
   `attempts=1`，**不消耗重试预算**，`metadata["feedback_kind"]="recoverable"`。
   若 `call.arguments` 含 `"__raw__"`（模型给的 arguments 不是合法 JSON）→ 同样走这条路径，
   错误消息**必须包含**子串 `"must be a valid JSON object"` **与**原始字符串前 200 字符。
4.5 **审批步（`[v2 变更]` HITL，冻结；位于校验之后、执行之前）**：

   ```text
   if tool.spec.requires_approval:
       policy = config.approval_policy
       approved = False
       if policy is not None:
           try:
               approved = bool(policy(call, tool))
           except AgentAbortedError as e:      # 用户在审批回调里主动中止
               emit TOOL_APPROVAL {tool_name, call_id, approved: False, aborted: True}
               raise asyncio.CancelledError() from None   # 走取消路径，由 Agent 转 ABORTED
           except LiteAgentError as e:
               return ToolResult.failure(call, e, metadata={"approved": False})
       emit TOOL_APPROVAL {tool_name, call_id, approved, policy_present: policy is not None}
       if not approved:
           err = ToolApprovalDeniedError(tool_name=tool.name,
                                         reason="no approval policy configured"
                                                if policy is None else "denied by policy")
           return ToolResult.failure(call, err, attempts=1,
                                     metadata={"approved": False, "feedback_kind": "infrastructure"})
       # 通过 -> metadata["approved"] = True 后继续
   ```

   回灌文案固定为 `ERROR(ToolApprovalDeniedError): this tool requires human approval`。
   **`policy is None` 且 `requires_approval=True` 时一律拒绝**（fail-closed，不是 fail-open）。
4.6 **熔断检查（`[v2 变更]`，`config.disable_tool_after_failures > 0` 时生效）**：

   ```text
   若 self._failure_counts.get(tool.name, 0) >= config.disable_tool_after_failures:
       返回 ToolResult.failure(call, ToolExecutionError(...), attempts=1,
              metadata={"disabled": True, "feedback_kind": "infrastructure"})
       emit TOOL_ERROR {tool_name, disabled: True}
       **不真正执行**。计数只在"连续 infrastructure 类失败"时累加、成功后清零。
   ```
5. 校验通过后进入尝试循环（总尝试 = `1 + max_retries`，但 `idempotent=False` 且
   `config.allow_retry_on_non_idempotent=False` 时强制为 1）：

   ```python
   total = 1 + max_retries
   if (not tool.spec.idempotent) and (not config.allow_retry_on_non_idempotent):
       total = 1
   delay = 0.0
   last_exc = None
   for attempt in range(total):                       # attempt 从 0 开始
       async with self._pool.semaphore("exec", config.max_concurrency):   # 顺序见 5.b
           with cancel_scope() as flag:               # 【每个 attempt 新建一次 Event】
               try:
                   raw = await self._invoke(tool, call, effective_timeout, flag)
               except asyncio.CancelledError:
                   raise                              # 规则 1：取消原样上抛
               except asyncio.TimeoutError:
                   flag.set()                         # 必须在 with 块**内部**完成
                   exc = ToolTimeoutError(tool_name=tool.name,
                                          timeout_s=effective_timeout)
                   exc.retryable = True
                   if not tool.spec.is_async:
                       _orphan = True                 # 见 5.c：同步工具超时 -> 标记 + 不重试
               except LiteAgentError as e:
                   exc = e
               except subprocess.TimeoutExpired as e:
                   exc = ToolTimeoutError(tool_name=tool.name, timeout_s=effective_timeout)
               except BaseException as e:             # CancelledError 已在上面拦掉
                   exc = ToolExecutionError(tool_name=tool.name, call_id=call.id, cause=e)
               else:
                   -> 成功：见 5.d
       # 信号量已释放，退避在**信号量之外**执行（见 5.b 的范围冻结）
       if <决定重试>:
           emit TOOL_RETRY {tool_name, call_id, attempt, delay_s, error_type}
           await self._sleep(delay)
   ```

   **5.a 事件**：每次尝试开始发 `TOOL_STARTED`
   `{tool_name, call_id, attempt, arguments}`。
   **5.b 并发与顺序（`[v2 变更]` 三条冻结）**：
   - **顺序写死：先并发信号量、后 seq 锁**，同一个 `with` 链：
     ```python
     async with self._pool.semaphore("exec", config.max_concurrency):
         async with self._seq_guard(tool.name, tool):     # 见下
             ...
     ```
     **禁止反序**（先拿 seq 锁再等信号量会在 N 个占满信号量的调用者之间形成死锁）——写进 §13 红线。
   - **范围：信号量只包单次 attempt**。退避 `await self._sleep(delay)` **必须**在释放信号量之后
     （包住整个重试循环时，一个正在退避的任务会占着并发槽，`max_concurrency=4` 时
     4 个重试中的工具就能把整个执行器饿死）。
   - **`sequential_tools` 的互斥用 `threading.Lock`（不是 loop-bound asyncio.Lock）**，
     理由见 [v2 变更] 说明：`delegate_to_*` 是**同步工具**，会被 `execute_many` 丢进不同线程，
     每个线程里的 `asyncio.run` 都是**新 loop**；loop-bound 锁跨 loop/跨线程完全不串行，
     而 `sequential_tools` 存在的唯一理由就是跨调用互斥（如 `run_shell`）。
     实现：

     ```python
     @asynccontextmanager
     async def _seq_guard(self, name: str, tool: Tool) -> AsyncIterator[None]:
         lock = self._seq_locks.get(name)          # __init__ 里为 config.sequential_tools 建好
         if lock is None:
             yield                                  # 不在 sequential_tools 里 -> 无锁
             return
         # [v3 变更] 在另一个线程里阻塞等待，避免阻塞事件循环；且**取消安全**。
         #   朴素写法 `await asyncio.to_thread(lock.acquire)` 有洞：调用方被取消时 asyncio
         #   只能取消 await 层，worker 线程仍会把 lock.acquire() 跑完，而释放锁的 finally
         #   属于**已被取消的协程**、永不执行 —— 锁被一个没有逻辑归属者的线程永久持有，
         #   该工具此后每次调用都卡在 acquire 上；卡死的还是默认执行器里的非 daemon 线程，
         #   `asyncio.run` / `_run_and_cleanup` 收尾时 shutdown_default_executor() 会去 join
         #   它，于是同步 API（execute_sync / Agent.run）**无异常、无超时地永久挂起**。
         #   `_CancelSafeAcquire` 把"取消已发生"传回线程：线程拿到锁后若发现已取消，
         #   立刻原地归还，绝不让锁变成孤儿（abort() 与 acquire() 用一把小锁串行化，
         #   保证恰好归还一次）。
         guard = _CancelSafeAcquire(lock)
         try:
             acquired = await asyncio.to_thread(guard.acquire)
         except asyncio.CancelledError:
             guard.abort()      # 取消只取消了 await 层：线程仍可能拿到锁，交给 abort() 归还
             raise
         if not acquired:
             raise asyncio.CancelledError()
         try:
             yield
         finally:
             lock.release()
     ```
     docstring 必须写明：**锁的临界区是"调用方可见的整段调用"；
     同步工具超时后 orphan 线程可能仍持有底层资源，此时锁已在调用方释放
     —— 这是已知限制，用 `orphan_thread=True` 标记可见**。
     `[v3 变更]` 触发取消的路径不止 fail_fast：外层 `asyncio.wait_for`、`Agent.astream`
     提前 break 的 `task.cancel()`、Ctrl-C、父任务取消都会命中。
     回归测试：`tests/test_tools_executor.py::SequentialToolsCancellationTests::test_cancelled_lock_waiter_does_not_orphan_the_sequential_lock`。
   **5.c 调用 `_invoke`（`[v2 变更]` 修掉 M-2 的 `to_thread(tool.arun)` 死代码与线程池归属）**：

   ```python
   async def _invoke(self, tool, call, timeout, flag):
       args = dict(call.arguments)
       if tool.spec.is_async:
           coro = tool.arun(args)
       else:
           tp = self._external_pool or self._pool.thread_pool("exec", config.thread_pool_size)
           loop = asyncio.get_running_loop()
           # 调用方线程提交；contextvars 在此复制（含 cancel_scope 的 Event）
           coro = loop.run_in_executor(tp, tool.run, args)     # tool.run 是**同步**入口
       if timeout is None:
           return await coro
       return await asyncio.wait_for(coro, timeout)
   ```

   - **取消语义（冻结）**：`asyncio.wait_for` 只能取消 await 层；**同步工具的实际线程不可中断**，
     超时后线程仍在跑。处理方式：把该结果标记为 `metadata["orphan_thread"]=True` 并记 WARNING。
     **绝不假装线程已经停了**。需要真取消的同步工具应实现协作式检查
     `liteagent.tools.base.current_cancel_flag()`：executor 在超时时 `flag.set()`
     （见上面的循环；`flag.set()` 必须在 `with cancel_scope()` 块**内部**完成）。
   - **`cancel_scope()` 每个 attempt 进一次**（禁止提到重试循环外，
     否则第 2 次尝试拿到的是已 set 的同一个 Event，工具一进循环就自杀、重试全部秒失败）。
   - **异步工具同样走同一个 `with cancel_scope()` 与同一段 try/except**（异步工具无孤儿线程，
     但同样应当收到取消信号）——**禁止**为两条分支写两份语义漂移的超时处理。
   **5.d 成功** → `ToolResult.success`，`content = _stringify(raw_return)`，
   截断到 `config.max_result_chars`（`truncate_head_tail`，并置 `metadata["truncated"]`
   与 `metadata["original_chars"]`）。`attempts = attempt + 1`。
   **截断只做一次**：`to_observation` 的 `max_chars` 由调用方传，
   **不得**再写 `metadata["truncated"]`（否则两处互相覆盖）。
   **5.e 异常包装**：`LiteAgentError` 原样保留类型；`subprocess.TimeoutExpired` → `ToolTimeoutError`；
   `SandboxViolationError` 原样；其它异常包成 `ToolExecutionError(tool_name, call_id, cause=exc)`。
   **5.f 判定重试**：

   ```python
   retryable = (self._is_retryable or _default_is_retryable)(exc)
   if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async:
       retryable = False          # [v2 例外] 同步工具超时 = 线程仍存活，重试会叠加孤儿线程
   if (not retryable) or (attempt + 1 >= total):
       last_exc = exc
       break
   delay = compute_backoff(attempt, base_s=..., max_s=..., jitter=..., rng=self._rng)
   if exc.retry_after_s: delay = max(delay, exc.retry_after_s)
   ```
   `_default_is_retryable(e) = isinstance(e, LiteAgentError) and e.retryable`。
   **5.g 结束**：若用尽且重试过 → 包成 `ToolRetryExhaustedError(attempts=..., last_error=exc)`，
   `metadata["last_error_type"] = type(exc).__name__`；否则直接返回 `exc` 对应的失败结果。
   两种情况都写 `metadata["feedback_kind"]`（映射表见 §3.4）。
   失败结果同时更新 `self._failure_counts`（`feedback_kind=="infrastructure"` 时 `+= 1`，
   成功时清零）。
6. `duration_ms` 累加**所有尝试**的总耗时；`attempts` 为实际尝试次数。
7. **`execute` 的返回不变式（冻结，`test_tools_executor` 断言）**：
   `r.call_id == call.id`、`r.name == call.name`、`r.attempts >= 1`、
   `r.ok is False ⇒ r.content 非空且以 "ERROR(" 开头`。

#### 7.4.2 `execute_many` 的 gather 语义（`[v2 变更]`，v1 只写了省略号）

```python
# [v3 变更] 批次级限流：`concurrency` 显式给定时，在**本函数存活期内**再叠一层信号量
#   （在运行中的 loop 里创建，不触 R-LOOP）；它与 `execute` 内部的全局信号量是**叠加**
#   而非替代。`concurrency`（或 `concurrency=None`）解析出的额度与 `config.max_concurrency`
#   相同时不造这一层。`limit <= 0` 记 WARNING 后回退到 `config.max_concurrency`（不抛）。
guard = None if limit == self.config.max_concurrency else asyncio.Semaphore(limit)

async def _run_one(call: ToolCall) -> ToolResult:
    if guard is None:
        return await self.execute(call, timeout_s=timeout_s)
    async with guard:
        return await self.execute(call, timeout_s=timeout_s)

# 批次级状态一律用局部变量（不要挂 self）：同一 executor 可能被并发共享。
fail_fast_triggered = False
futures = [asyncio.ensure_future(_run_one(c)) for c in calls]
```

**冻结规则**：

1. **必须** `await asyncio.gather(*futures, return_exceptions=True)`。
   用 `return_exceptions=False` 时任一子任务异常会让其余结果**全部丢失**、无法按下标对齐。
2. **逐项归一化**，结果列表**长度与 `calls` 严格一致、顺序严格一致**：

   ```python
   out: list[ToolResult] = []
   for i, r in enumerate(results):
       if isinstance(r, ToolResult):
           out.append(r)
       elif isinstance(r, asyncio.CancelledError):
           out.append(ToolResult.failure(calls[i], ToolSkippedError(
               tool_name=calls[i].name, reason="cancelled_by_fail_fast"),
               metadata={"skipped_by_fail_fast": True}))
       else:   # 只应在实现有 bug 时命中：execute 内部已把除 CancelledError 外的一切编码为 ToolResult
           out.append(ToolResult.failure(calls[i], r))
   ```

   断言式说明必须写进代码注释：**`execute` 已保证"除 `CancelledError` 外一切异常都编码为
   `ToolResult`"，因此第 3 个分支是防御性的**。
3. 若 `isinstance(r, asyncio.CancelledError)` 而**不是** `fail_fast` 引起的（调用方取消了
   `execute_many` 本身），则必须**重新抛出**该 `CancelledError`（取消传播优先于结果收集）：
   实现为在 gather 之前 `try/finally` 检查 `asyncio.current_task().cancelling()`? **不用** ——
   冻结为：**gather 之后先检查调用方自身是否被取消**：遍历 `results`，若存在
   `CancelledError` 且 `not fail_fast_triggered`，则 `raise` 第一条 `CancelledError`。
   `[v3 变更]` 这里的 `fail_fast_triggered` 必须是 `execute_many` 的**局部变量**
   （由 `_watch_fail_fast(...) -> bool` 返回），**不是** v2 冻结的
   `self._fail_fast_triggered` 实例属性 —— 理由见规则 4。
4. `fail_fast=True` 的语义：

   ```text
   监视结果：第一个 ok=False 出现后，
       fail_fast_triggered = True      # [v3 变更] 局部变量（由 _watch_fail_fast 返回）
       对其余**未完成**的 future 调 fut.cancel()
       再 await asyncio.gather(*futures, return_exceptions=True) 收尾（必须收尾，否则任务泄漏）
       被取消的位置按第 2 条填 ToolSkippedError
   metadata 里记录首个失败的下标：execute_many 不返回 metadata，
   因此改为 emit 一条 `TOOL_ERROR {tool_name, fail_fast_first_index: i, skipped: N}` 事件。
   完成顺序用 `asyncio.wait(pending, return_when=FIRST_COMPLETED)` 观察；
   同一批唤醒里多个任务都完成时，按 **calls 的下标**取胜者（确定性 tie-break，
   否则"首个失败"在 trace 里会抖）。
   ```

   **`[v3 变更]` 为什么"本批次是否触发过 fail_fast"必须是局部变量**：v2 把它挂在实例上
   （`self._fail_fast_triggered`），每个 `execute_many` 开头重置一次。同一个 executor 被
   **并发共享**时，批次 B 一开头就把批次 A 的标志抹掉，于是 A 收尾时把自己 fail_fast 亲手
   取消的兄弟任务误判成"调用方取消了 `execute_many`"而 `raise CancelledError` —— 既违反
   本条"必须仍返回与 `calls` 等长的列表"，在 Agent 路径上还会让整次 run 被误当作外部取消。
   实现：`_watch_fail_fast` 返回**本批次**的布尔，由 `execute_many` 的局部变量接收；
   监视器只负责"取消未完成的兄弟"，收尾的 `gather` 由调用方做（避免 await 两次）。
   回归测试：`tests/test_tools_executor.py::ConcurrencyTests::test_concurrent_fail_fast_batches_do_not_share_batch_state`。

   **`execute_many` 永不向外抛业务异常**（唯一例外仍是 `asyncio.CancelledError` /
   `KeyboardInterrupt`）。v1 的"抛 `ToolRetryExhaustedError`"**作废**：它同时违反
   `execute` 的契约、§13 红线 6、以及 §9.4.1 里没有 try/except 的事实；
   异常类型本身也错（首个失败的根因不一定是重试耗尽）。
5. `execute_many` 返回后，`calls` 与 `out` 的下标必须能一一对应 —— 这就是"顺序保持"的全部含义。

#### 7.4.3 `_stringify(raw_return)` 规则（冻结）

- `None` → `""`
- `str` → 原样
- `bytes` → `base64` 单行
- `ToolResult` → 取 `raw.content` 并合并其 metadata 与 ok 状态
- `dict`/`list`/`int`/`float`/`bool` → `json.dumps(..., ensure_ascii=False, indent=2, default=str)`
- `@dataclass` 实例 → `json.dumps(to_jsonable(obj), ...)`
- 其它 → `str(obj)`，并在 `metadata["stringified"]=type(obj).__name__`

### 7.5 `tools/builtin/`

**`[v2 变更]` 注入通道的冻结总规则（四组 factory 全部适用）**：

> **模块级函数是纯实现，但一律以 `_` 私有名定义；`make_*_tools(...)` 用闭包
> （或 `functools.partial`）生成包装函数，再用 `make_function_tool(name=..., parameters=<手写 schema，
> 与模块级函数签名逐字一致>, func=wrapped, pass_style="mapping")` 注册。
> 隐藏参数名必须以 `_` 开头且不得出现在 `parameters` 里。**
> 测试调用路径只有两条：`make_*_tools(...)[i].raw(...)`（绕过校验）或经 executor 调用。

**`register_all` 的沙箱解析优先级（冻结）**：`register_all(sandbox_root=)` 参数 >
`LITEAGENT_SANDBOX_ROOT` > `os.getcwd()`，最终都构造出一个 `PathSandbox` 传给 `make_file_tools`。
**`make_file_tools(sandbox)` 的 `sandbox` 是必填参数**（`None` → `ConfigError`），
不允许隐式落到 `os.getcwd()` —— 否则在仓库根跑测试时 `delete_file` 会真的删项目文件。

```python
# ---- files.py ----
class PathSandbox:
    """把相对路径解析到 root 下，并拒绝逃逸。"""
    def __init__(self, root: str | os.PathLike[str] | None = None,
                 *, allow_read_outside: bool = False) -> None: ...
    root: Path                              # .resolve() 后的绝对路径
    def resolve(self, path: str, *, write: bool = False) -> Path:
        """- 绝对路径：直接 resolve，若不在 root 下且未允许 -> SandboxViolationError
           - 相对路径：root / path 后 resolve；同样做前缀检查
           - 符号链接：用 Path.resolve() 解到最终真实路径后再检查（防 symlink 逃逸）
           - 检查用 Path.is_relative_to(3.9+，可用)
           - root 为 None 时 **抛 ConfigError**（不允许隐式 cwd）"""
        ...

# 私有实现（模块级，`_` 前缀，**不导出**）
def _read_file(_sandbox: PathSandbox, path: str, start_line: int | None = None,
               end_line: int | None = None, max_chars: int = 20000) -> str:
    """Read a UTF-8 text file, optionally a line range."""
def _write_file(_sandbox: PathSandbox, path: str, content: str, create_dirs: bool = True,
                overwrite: bool = True) -> str:
    """Write text to a file. Returns a one-line summary."""
def _list_dir(_sandbox: PathSandbox, path: str = ".", pattern: str | None = None,
              recursive: bool = False, max_entries: int = 500) -> str:
    """List files under a directory."""
def _search_files(_sandbox: PathSandbox, pattern: str, path: str = ".", glob: str = "**/*",
                  max_results: int = 50, case_sensitive: bool = False) -> str:
    """Regex-search file contents; returns 'path:line: text' lines."""
def _delete_file(_sandbox: PathSandbox, path: str, confirm: bool = False) -> str:
    """Delete a file. Requires confirm=True."""

def make_file_tools(sandbox: PathSandbox) -> list[Tool]:
    """返回 [read_file, write_file, list_dir, search_files, delete_file]（**同名新 Tool**，
    通过闭包注入 sandbox）。模块级**不得**定义同名的公开函数。"""
    ...
```

**模型可见的参数签名（`parameters` 里逐字一致，`_sandbox` 不出现）**：
`read_file(path, start_line=None, end_line=None, max_chars=20000)`；
`write_file(path, content, create_dirs=True, overwrite=True)`；
`list_dir(path=".", pattern=None, recursive=False, max_entries=500)`；
`search_files(pattern, path=".", glob="**/*", max_results=50, case_sensitive=False)`；
`delete_file(path, confirm=False)`。

装饰器元数据冻结：`read_file/list_dir/search_files` → `idempotent=True, dangerous=False`；
`write_file` → `idempotent=True, dangerous=True`；`delete_file` → `idempotent=False, dangerous=True`。
所有文件工具 `tags=("fs",)`。`search_files` 的 `glob` 只支持 `**/*` 与 `*` 两种形态
（不做完整 glob 引擎）。**处理顺序冻结**：先按 `glob` 过滤文件名，再逐文件跑 `pattern` 正则；
跳过二进制文件（前 8KB 含 `\x00`）与超过 1MB 的文件（记入返回文本）。

```python
# ---- shell.py ----（**不包含 python_exec**）
SHELL_DENY_PATTERNS: tuple[str, ...]     # 正则，冻结列表（大小写不敏感，匹配即拒绝）：
#   r"\brm\s+-[a-z]*r[a-z]*f?\s+/", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{", r"\bsudo\b",
#   r"\bchmod\s+777\s+/", r">\s*/dev/sd", r"\bcurl\b[^|]*\|\s*(ba)?sh", r"\bwget\b[^|]*\|\s*(ba)?sh",
#   r"\bshutdown\b", r"\breboot\b", r"\bkill\s+-9\s+1\b"

def check_command_allowed(command: str) -> str | None:
    """纯函数：返回拒绝原因，None 表示允许。逐条用 SHELL_DENY_PATTERNS 匹配
    （re.search，re.IGNORECASE）。空/纯空白命令 -> 返回 "empty command"。"""
    ...

def _run_shell(_sandbox: "PathSandbox | None", command: str, cwd: str | None = None,
               timeout_s: float = 30.0, env: dict[str, str] | None = None,
               max_output_chars: int = 10000) -> str:
    """Run a shell command. Returns exit code, stdout and stderr.
    [v2 冻结] 行为：
      1. 第一行 `reason = check_command_allowed(command)`；非 None 时
         `raise SandboxViolationError(command, "denylist:" + reason)`
         （即 path=command、root="denylist:<reason>"，保持字段类型不破坏）。
      2. subprocess.run(shell=True, cwd=<沙箱根>, timeout=timeout_s, capture_output=True)；
         `subprocess.TimeoutExpired` -> `ToolTimeoutError(tool_name="run_shell", timeout_s=timeout_s)`。
      3. cwd 的相对路径必须先过 `sandbox.resolve(cwd)`；越界抛 SandboxViolationError。
         命令的实际执行 cwd 固定为沙箱根。
      4. 输出超 max_output_chars 时 truncate_head_tail。
    元数据：idempotent=False, dangerous=True, requires_approval=True,
            timeout_s=NO_TIMEOUT（由参数控制）, tags=("shell",)"""
    ...

def make_shell_tools(allow_shell: bool | None = None, sandbox: "PathSandbox | None" = None) -> list[Tool]:
    """allow_shell 默认取 parse_bool(os.environ.get("LITEAGENT_ALLOW_SHELL"))。
    allow_shell=False 时仍返回工具，但调用即返回失败结果
    'shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)'
    —— **保持工具可见**，让模型知道能力存在但被禁用。
    [v2 冻结] allow_shell=False 时**不进入** check_command_allowed（直接返回失败字符串）。"""
    ...
```

```python
# ---- code.py ----
ALLOWED_AST_NODES: frozenset[type]      # 冻结：Expression, Constant, Name(见下),
#   BinOp/UnaryOp/BoolOp/Compare + 对应 operator 节点, Call(仅白名单函数),
#   List/Tuple/Dict/Set, Subscript, Slice, IfExp,
#   Attribute(仅白名单属性: str/list/dict 的只读方法)
ALLOWED_BUILTINS: frozenset[str]        # abs,all,any,bool,chr,dict,divmod,enumerate,float,
#   format,hash,hex,int,isinstance,len,list,max,min,oct,ord,pow,range,repr,reversed,round,
#   set,slice,sorted,str,sum,tuple,zip
FORBIDDEN_NAMES: frozenset[str]         # __import__,eval,exec,compile,open,globals,locals,
#   getattr,setattr,delattr,vars,input,breakpoint,memoryview,object,type,super
#   [v2] 明确**不支持**：JoinedStr(f-string)、Lambda、ListComp/SetComp/DictComp/Starred

def safe_eval_ast(expression: str, variables: Mapping[str, Any] | None = None) -> Any:
    """AST 白名单求值，**绝不调用 builtin eval**。违规 -> ToolExecutionError(retryable=False)。
    [v2 冻结] 三条规则：
      1. `Name` 允许 ALLOWED_BUILTINS 里的名字，**或** `variables` 里存在的键
         （v1 的注释写"仅白名单内置"，与 `variables` 参数直接冲突）；其余抛 ToolExecutionError。
      2. **静态复杂度闸**（因为 timeout_s 无法中断求值，见 D-09 的诚实原则）：
         拒绝 Pow 且指数绝对值 > 1000；拒绝 range 字面参数 > 1e6；
         拒绝表达式长度 > 2000；拒绝递归深度 > 32。
         docstring 必须写明："timeout_s 只保证调用方不再等待，不保证中断求值，
         因此复杂度必须在静态检查阶段拦截。"
      3. 本函数不是安全边界（资源耗尽、未来节点遗漏都会破）。
         文档话术统一为"受限表达式求值（非安全沙箱）"。"""
    ...

def _python_eval(_sandbox: "PathSandbox | None", expression: str, variables_json: str = "{}") -> str:
    """Restricted expression evaluation (NOT a security sandbox)."""
    # 元数据：idempotent=True, dangerous=False, timeout_s=5.0, tags=("code",)

def _python_exec(_sandbox: "PathSandbox | None", code: str, timeout_s: float = 10.0,
                 cwd: str | None = None) -> str:
    """Execute Python code in an isolated subprocess. Returns stdout/stderr/exit code.
    实现：[sys.executable, "-I", "-c", code]，subprocess.run(timeout=...)；
      TimeoutExpired -> ToolTimeoutError；超长输出截断。
    元数据：idempotent=False, dangerous=True, requires_approval=True, tags=("code",)"""

def _run_tests(_sandbox: "PathSandbox | None", path: str = "tests", pattern: str = "test_*.py",
               timeout_s: float = 300.0, extra_args: str = "") -> str:
    """Run unittest discovery and return a summary. 用 [sys.executable, "-m", "unittest",
    "discover", "-s", path, "-p", pattern] + '-v'，解析 'Ran N tests' 与 'OK'/'FAILED'。
    [v2 冻结] path 经 sandbox.resolve 解析；**若解析结果落在本仓库的 tests/ 目录内，
    在返回文本里追加一行 WARNING**（防止测试里 run_tests() 递归跑整套用例）。
    元数据：idempotent=False, dangerous=True, tags=("code",)"""

def make_code_tools(sandbox: "PathSandbox | None" = None) -> list[Tool]:
    """返回 [python_eval, python_exec, run_tests]。sandbox 仅用于把 run_tests 的
    path 解析到沙箱内（python_exec 的代码不受沙箱约束 —— 它本就是任意代码执行工具，
    因此标记 dangerous=True + requires_approval=True 并在文档里明确警告）。"""
    ...
```

```python
# ---- web.py ----
class SearchBackend(ABC):
    name: str = "abstract"
    @abstractmethod
    def search(self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
               ) -> list[SearchHit]: ...

@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    def to_dict(self) -> dict[str, Any]: ...

class NullSearchBackend(SearchBackend):
    """零依赖兜底：始终返回 []，并让 web_search 返回可读的失败说明。"""
    name = "null"

class DuckDuckGoHTMLBackend(SearchBackend):
    """用 Transport 请求 https://html.duckduckgo.com/html/?q=...，用纯 stdlib 解析结果。
    [v2 冻结] `__init__(self, transport: Transport | None = None, *,
                          endpoint: str = "https://html.duckduckgo.com/html/")`
    —— transport 可注入是离线测试成功路径的**唯一**手段。"""
    name = "duckduckgo"

class TavilyBackend(SearchBackend):     # 需要 TAVILY_API_KEY
    """[v2 冻结] __init__(self, api_key: str | None = None, *,
                            transport: Transport | None = None)"""
    name = "tavily"
class SerperBackend(SearchBackend):     # 需要 SERPER_API_KEY
    """[v2 冻结] 同 TavilyBackend 的签名"""
    name = "serper"

def default_search_backend() -> SearchBackend:
    """优先级：TAVILY_API_KEY -> SERPER_API_KEY -> duckduckgo -> NullSearchBackend"""
    ...

class HTMLTextExtractor(html.parser.HTMLParser):
    """纯 stdlib 的 HTML -> 文本：跳过 script/style/head，块级标签补换行，
    压缩连续空行，解码实体（用 html.unescape）。"""
    def __init__(self) -> None: ...
    def get_text(self) -> str: ...

def html_to_text(html: str, *, max_chars: int = 20000) -> str: ...

def _web_search(_backend: SearchBackend, _allow_network: bool, query: str,
                max_results: int = 5) -> str:
    """Search the web. Returns title/url/snippet lines."""
def _fetch_url(_backend: SearchBackend, _transport: "Transport | None", _allow_network: bool,
               url: str, max_chars: int = 20000, timeout_s: float = 15.0) -> str:
    """Fetch a URL and return its main text content."""

def make_web_tools(backend: SearchBackend | None = None,
                   *, allow_network: bool = True,
                   transport: "Transport | None" = None) -> list[Tool]:
    """[v2 变更] 新增 `transport` 参数（**离线测试的唯一注入点**）。
    返回 [web_search, fetch_url]。allow_network=False 时两者都返回
    'network access is disabled' 失败结果（仍可见）。"""
    ...
```

冻结元数据：两者 `idempotent=True, dangerous=False, tags=("web",)`；
`web_search`/`fetch_url` 都标 `retryable=True`（网络类工具）；`href` 过滤掉非 http/https。

```python
# ---- memory_tools.py ----
def make_memory_tools(memory: "MemoryManager") -> list[Tool]:
    """返回 [remember, recall]；闭包捕获 memory 实例。
    remember(content: str, importance: float = 0.5) -> str
        'Stored a long-term memory (id=<id>).'  （importance 0..1，越界由 schema 的
        minimum/maximum 约束拦截）
    recall(query: str, limit: int = 5) -> str
        渲染为 'score=0.83 | 2026-09-27 | <content>' 多行
    两者 idempotent=True, tags=("memory",)。
    注意：remember 需要 event loop —— 工具是同步函数，跑在 worker 线程里（没有 loop），
    因此内部用 `config.run_sync(lambda: memory.aremember(...))` 新建一个 loop 执行。
    这是安全且期望的行为（线程里没有别人的 loop，不会嵌套）。"""
    ...

# ---- builtin/__init__.py ----
BUILTIN_TOOL_GROUPS: dict[str, Callable[..., list[Tool]]] = {   # [v2 变更] 函数对象，不是字符串名
    "files": make_file_tools, "shell": make_shell_tools,
    "code": make_code_tools, "web": make_web_tools, "memory": make_memory_tools,
}
BUILTIN_TOOL_NAMES: tuple[str, ...]      # 全部内置工具名，冻结：
# ("read_file","write_file","list_dir","search_files","delete_file",
#  "run_shell","python_exec","python_eval","run_tests",
#  "web_search","fetch_url","remember","recall")

def register_all(registry: ToolRegistry, *,
                 include: Sequence[str] | None = None,
                 exclude: Sequence[str] | None = None,
                 sandbox_root: str | None = None,
                 allow_shell: bool | None = None,
                 allow_network: bool = True,
                 memory: "MemoryManager | None" = None,
                 search_backend: SearchBackend | None = None,
                 web_transport: "Transport | None" = None) -> ToolRegistry:
    """把内置工具注册进 registry 并返回它（便于链式调用）。

    [v2 冻结] include/exclude 的语义（v1 在"组名还是工具名"上完全悬空）：
      - 每个元素**先按组名**查 `BUILTIN_TOOL_GROUPS`，命中即展开该组的全部工具；
        **未命中再按工具名**查 `BUILTIN_TOOL_NAMES`；两者都不中 -> `ConfigError`。
      - `include=None` = 全部组（memory 组仅当 `memory is not None` 时注册）。
      - `include=[]` = **注册 0 个工具**，与 `None` 语义**不同**，必须区分。
      - `exclude` 在 include 展开**之后**应用，优先级高于 include（同名时以 exclude 为准）。
      - `exclude` 接受组名或工具名，未命中 -> `ConfigError`。
      - `AppConfig.tools` 非空时等价于 `register_all(include=config.tools)`；
        CLI 的 `--tools ""` 等价于 `include=[]`。
    - `sandbox_root` 默认取 `LITEAGENT_SANDBOX_ROOT`，再退到 `os.getcwd()`；
      files/code 组必须能确定一个非 None 的 sandbox_root，否则 `ConfigError`。
    **本函数是唯一有"批量副作用"的地方，且必须显式调用**（不依赖 import 副作用）。"""
    ...
```

---

## 8. `liteagent/memory/` —— 记忆层（三层）

### 8.1 `memory/base.py`

```python
@dataclass
class MemoryItem:
    id: str
    content: str
    role: str = "user"                       # "user" | "assistant" | "system" | "tool"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=utc_now)       # [v2] 唯一时钟
    last_access_at: float = field(default_factory=utc_now)   # [v2]
    importance: float = 0.5                  # 0..1
    access_count: int = 0
    embedding: list[float] | None = None
    score: float | None = None               # 检索时填充（混合分）
    score_breakdown: dict[str, float] | None = None   # {"sim":..,"recency":..,"importance":..}
    source: str = ""                         # 来源标记，如 "agent:main" / "manual"

    @classmethod
    def create(cls, content: str, *, role: str = "user", importance: float = 0.5,
               metadata: Mapping[str, Any] | None = None, source: str = "",
               item_id: str | None = None) -> "MemoryItem":
        """item_id 默认 'mem_' + uuid4().hex[:12]（存储层 ID 允许随机，测试断言请用内容而非 ID）。"""
        ...

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """[v2 变更] **本方法只有一个定义**（v1 在同一处写了两行互斥签名）。
        冻结签名与语义：
          - include_embedding=False（**默认**）-> 省略 "embedding" 键
          - include_embedding=True -> 输出 embedding（list[float]，可能很长）
        其余字段一律全量输出（含 None）。理由见 §2.2：embedding 会让 trace/CLI 膨胀。
        持久化往返（§8.5 的 save/load）与 to_dict 的精确比对测试用 True。"""
        ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryItem":
        """未提供 embedding 时置 None，**不重算**。"""
        ...
    def text(self) -> str: ...                    # 用于 token 估算
    def age_days(self, *, now: float | None = None) -> float: ...

class MemoryStore(ABC):
    """同步内存存储契约。**同步**的理由：纯内存操作是微秒级，async 只会引入无谓的
    事件循环耦合；并且工具大多跑在 worker 线程里，那些线程没有运行中的事件循环，
    纯 async API 会在那里直接抛异常。需要 LLM 的摘要走 MemoryManager 的 async API。
    完整论证见 `DESIGN_DECISIONS.md` D-12。

    [v2 变更] **线程模型（冻结小节，与 §10.2.1 同构）**：
      1. 每个具体实现（BufferMemory / VectorMemory）各持有
         `self._lock = threading.RLock()`，在 **`__init__`** 创建
         （threading 原语不受 §0.4 的 R-LOOP 约束 —— 这正是选它的理由）。
      2. `add/add_many/get/search/all/delete/clear/upsert/window/drain_evicted`
         **全部**在锁内完成，且临界区内**没有 await、没有 I/O**。
      3. `search` 的副作用（`access_count` / `last_access_at` / `score`）**必须在同一把锁内**更新。
      4. 向量矩阵（numpy 数组）的读也走同一把锁；**不允许裸露的 `_matrix`**。
      背景：`TeamConfig.share_memory=True` 时多个子 Agent 共享一个 MemoryManager；
      Hierarchical 的 worker 在各自线程 + 各自新建的 loop 里跑，主 loop 同时可能在检索 ——
      无锁下会得到 `RuntimeError: list changed size during iteration` 或读到一半的淘汰状态。"""

    name: str = "abstract"

    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...
    def add_many(self, items: Sequence[MemoryItem]) -> None: ...
    @abstractmethod
    def get(self, item_id: str) -> MemoryItem | None: ...
    @abstractmethod
    def search(self, query: str, *, limit: int = DEFAULT_RETRIEVE_LIMIT,
               **kwargs: Any) -> list[MemoryItem]: ...
    @abstractmethod
    def all(self) -> list[MemoryItem]:
        """按 created_at 升序。**冻结**：只复制外层 list，**元素是同一批 MemoryItem 实例**
        （共享可变），测试可以用 `is` 断言身份。文档写明这一点。"""
        ...
    @abstractmethod
    def delete(self, item_id: str) -> bool: ...
    @abstractmethod
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

# ---- Tokenizer 家族（冻结）----
class Tokenizer(ABC):
    name: str = "abstract"
    @abstractmethod
    def estimate(self, text: str) -> int: ...
    def estimate_many(self, texts: Sequence[str]) -> int: ...

class HeuristicTokenizer(Tokenizer):
    """默认实现（零依赖）。est = ceil(n_cjk * cjk_char_cost + n_other / char_ratio)
    - n_cjk：码点落在 CJK 区间的字符数。冻结的区间判定（逐条实现，不要用 unicodedata 名字匹配，
      那在不同 Unicode 版本下不稳定）：
        0x4E00-0x9FFF   中日韩统一表意
        0x3400-0x4DBF   扩展 A
        0x3000-0x303F   CJK 标点
        0xFF00-0xFFEF   全角字符
        0xAC00-0xD7AF   韩文音节
        0x3040-0x30FF   平假名/片假名
    - n_other：其余码点数（含空格与 ASCII 标点）
    - 参数默认 char_ratio=DEFAULT_TOKEN_CHAR_RATIO(4.0), cjk_char_cost=1.0
    - 结果用 functools.lru_cache(maxsize=4096) 缓存（按 (text, ratio, cost) 元组）
    - **空文本 -> 0**；非空且估算结果 < 1 -> 返回 1（最小 1 token 的保守假设）
    """
    name = "heuristic"

class CallableTokenizer(Tokenizer):
    def __init__(self, fn: Callable[[str], int], *, name: str = "callable") -> None: ...
    name: str

class TiktokenTokenizer(Tokenizer):
    """仅当 tiktoken 可 import 时定义；用 cl100k_base 编码。
    encode 失败时回退到 HeuristicTokenizer（并记 warning）。"""
    name = "tiktoken"

def get_default_tokenizer() -> Tokenizer:
    """优先级：TIKTOKEN_AVAILABLE -> TiktokenTokenizer("cl100k_base") -> HeuristicTokenizer()
    结果缓存（functools.cache）。"""
    ...
```

**token 估算的选择（冻结决策 D-06）**：**混合估算**——CJK 按 1 字 1 token，其余按 4 字符 1 token。
理由：纯 `len/4` 对中文严重低估，会让窗口超预算；引入 tiktoken 会违反零依赖。
代价：±20% 误差。缓解：预算留 20% 余量，并允许注入 `CallableTokenizer` 或 tiktoken。
**这个估算函数还必须被用于摘要触发判定与向量记忆的截断，必须唯一。**
**`DEFAULT_TOKEN_CHAR_RATIO` 只被 `HeuristicTokenizer` 使用**
（`llm.message.messages_tokens` 在 counter=None 时的 ASCII 近似是独立的兜底，两者都遵守"空串 -> 0"）。

### 8.2 `memory/embeddings.py`

```python
class Embedder(ABC):
    name: str = "abstract"
    dim: int = 0
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """返回 L2 归一化后的向量（**归一化是契约的一部分**，检索时直接用点积当余弦）。"""
        ...
    def embed_one(self, text: str) -> list[float]: ...
    def is_available(self) -> bool: ...          # 默认 True

class HashingEmbedder(Embedder):
    """纯 stdlib 的 hashing trick + 特征哈希（冻结算法，测试会断言确定性）：

    1. 分词 tokenize(text) -> list[str]（冻结）：
       - 转小写
       - 英文/数字：正则 r"[a-z0-9_]+" 提取，保留长度 >= 1 的 token
       - CJK：对每个 CJK 连续片段取 **单字** 与 **相邻双字 bigram** 两种 token
       - 丢弃纯空白与单字符 ASCII 标点
    2. 每个 token t:
          h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
          idx = int.from_bytes(h[:4], "big") % dim
          sign = 1.0 if (h[4] & 1) == 0 else -1.0
          w = 1.0 + math.log1p(count_of_t)
          vec[idx] += sign * w
       **禁止使用内置 hash()**：Python 的 str hash 受 PYTHONHASHSEED 随机化影响，
       会导致同一进程外不可复现 —— 这是必须写进文档的坑。
    3. L2 归一化；零向量（空文本）返回全 0（不要 NaN）。
    """
    def __init__(self, dim: int = DEFAULT_HASHING_EMBED_DIM) -> None: ...

class NumpyHashingEmbedder(Embedder):
    """与 HashingEmbedder 数学等价但用 numpy 加速。仅当 NUMPY_AVAILABLE。
    **测试必须断言两者对同一输入的输出在 1e-9 内一致**（双实现一致性）。"""
    def __init__(self, dim: int = DEFAULT_HASHING_EMBED_DIM) -> None: ...

class RandomProjectionEmbedder(Embedder):
    """测试用：由 seed 决定的可复现伪随机向量。
    每个 token 的向量由 random.Random(seed ^ crc32(token)) 生成后累加，再归一化。
    用途：断言"给定构造好的相似度顺序时，VectorMemory 的排序正确"。
    """
    def __init__(self, dim: int = 32, *, seed: int = 0) -> None: ...

class CallableEmbedder(Embedder):
    def __init__(self, fn: Callable[[Sequence[str]], list[list[float]]],
                 *, dim: int, name: str = "callable") -> None: ...

class RemoteEmbedder(Embedder):
    """用 Transport 调 OpenAI 兼容 /embeddings 端点（无需 openai SDK）。
    需要 api_key；无网络/无 key 时 embed 抛 LLMError，由 VectorMemory 决定是否降级。"""
    def __init__(self, *, model: str, api_key: str | None = None,
                 base_url: str | None = None, dim: int = 1536,
                 transport: Transport | None = None,
                 timeout_s: float = DEFAULT_LLM_TIMEOUT_S) -> None: ...

def default_embedder(dim: int = DEFAULT_HASHING_EMBED_DIM) -> Embedder:
    """永远返回 **HashingEmbedder**（不依赖 numpy，保证跨实现者行为一致）。
    想用 numpy 加速需显式传 NumpyHashingEmbedder()。"""
    ...

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """两向量已归一化时 = 点积；否则除以模长乘积。零向量 -> 0.0（不抛异常）。
    维度不等 -> MemoryStoreError。"""
    ...

def cosine_similarity_matrix(query: Sequence[float],
                             matrix: Sequence[Sequence[float]]) -> list[float]: ...
```

**embedding 抽象的离线兜底（冻结决策 D-04）**：默认 `HashingEmbedder` 是**纯 stdlib、确定性、
无需网络**的。它不是语义 embedding（只是词面哈希），因此文档**必须**诚实说明：
`HashingEmbedder` 提供**词面相似**检索；生产环境应注入 `RemoteEmbedder`。
另外两条必须写进 `ARCHITECTURE.md` **能力 vs 非能力**表的限制（`[v2 新增]`）：
**(1)** 检索是 **O(n) 线性扫描**，`DEFAULT_MAX_MEMORY_ITEMS=10000` 时每次 search 都是近万次点积
—— **不能**声称"向量数据库 / 可扩展检索"；
**(2)** 没有 ANN 索引（faiss 不可用），也没有 rerank 模型。

### 8.3 `memory/buffer.py` —— 短期滑动窗口

```python
@dataclass
class BufferConfig:
    max_tokens: int = DEFAULT_BUFFER_MAX_TOKENS
    max_messages: int = DEFAULT_BUFFER_MAX_MESSAGES
    keep_last_n: int = 2                # 至少保留最近 N 条消息（即使超预算）
    keep_system: bool = True
    tokenizer: Tokenizer | None = None  # None -> get_default_tokenizer()

# [v2 变更] 模块级公开可测函数（v1 只在散文里提到它，代码块里没有）
def _repair_tool_pairs(messages: Sequence[Message]) -> list[Message]:
    """工具对完整性修复：若保留的 assistant 消息带 tool_calls，则它的**全部** tool 结果
    必须也在列表里，否则把该 assistant 消息整体丢弃（连同它的工具结果）。
    在 window() 的第 5 步里被调用。**测试直接 `from liteagent.memory.buffer import
    _repair_tool_pairs`**（§2.3：单下划线只表示非公开 API，不表示不可测）。"""
    ...

class BufferMemory:
    """短期记忆：有界消息窗口，按 token 预算 + 条数双约束裁剪。
    [v2] **线程安全**：所有公开方法在 `self._lock = threading.RLock()` 内完成（§8.1）。"""

    def __init__(self, config: BufferConfig | None = None,
                 *, on_evict: Callable[[list[Message]], None] | None = None) -> None: ...

    config: BufferConfig

    def add(self, message: Message) -> None:
        """追加；不在此处裁剪（裁剪在 window/trim 时做），保证 add 是 O(1)。"""
    def extend(self, messages: Sequence[Message]) -> None: ...
    def messages(self) -> list[Message]: ...            # 全量（未裁剪）副本
    def window(self) -> list[Message]: ...              # 裁剪后的窗口（每次调用重算）
    def estimated_tokens(self) -> int: ...              # 全量的估算
    def window_tokens(self) -> int: ...
    def evicted(self) -> list[Message]: ...             # 被裁掉的消息（供摘要使用）
    def drain_evicted(self) -> list[Message]:
        """取出并清空 evicted（摘要器调用；取走后不会重复摘要）。"""
    def last(self) -> Message | None: ...
    def clear(self, *, keep_system: bool = True) -> None: ...
    def __len__(self) -> int: ...
```

**裁剪算法 `window()`（冻结，逐步实现）**：

1. 取 `msgs = self._messages`（全量，按插入序）。
2. 若 `keep_system`，把前导的 `role==SYSTEM` 消息全部单独取出为 `pinned`（连续前缀）。
3. 从**尾部**向前累加消息，直到任意一条约束先触发：
   - `len(kept) == max_messages` → 停
   - `tokens(kept) + tokens(pinned) > max_tokens` → **不保留该条，停止**（即保证不超预算），
     但若 `len(kept) < keep_last_n`，则继续保留直到满足 `keep_last_n`（允许超预算，
     这是最小可用上下文）。
4. 输出 = `pinned + list(reversed(kept))`。
5. **工具对完整性修复**：先 `message.drop_orphan_tool_messages(result)`，
   再 `_repair_tool_pairs(result)`（§8.3 的模块级函数）。
6. `evicted = [m for m in msgs if m not in window]`，按原序返回；`self._evicted` 累积
   （`drain_evicted` 清空）。

### 8.4 `memory/summary.py` —— 摘要压缩

```python
SUMMARY_PROMPT_TEMPLATE: str     # 冻结字面量，见下
FALLBACK_SUMMARY_MAX_CHARS: int = DEFAULT_MAX_SUMMARY_CHARS

@dataclass
class SummaryConfig:
    enabled: bool = True
    trigger_ratio: float = DEFAULT_SUMMARY_TRIGGER_RATIO
    min_evict_batch: int = DEFAULT_SUMMARY_MIN_EVICT
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    max_input_chars: int = 16000        # 喂给摘要模型的对话文本上限
    update_existing: bool = True        # True -> 滚动更新（把旧摘要一起喂进去）

class SummaryMemory:
    def __init__(self, llm: "LLMClient | None" = None,
                 config: SummaryConfig | None = None,
                 *, tokenizer: Tokenizer | None = None) -> None: ...

    @property
    def summary(self) -> str: ...
    @property
    def summary_tokens(self) -> int: ...
    @property
    def compression_count(self) -> int: ...

    def should_compress(self, *, current_tokens: int, max_tokens: int,
                        pending_evicted: int) -> bool:
        """冻结判定（OR 关系，任一成立即压缩）：
          1. enabled 为 False -> 永不压缩
          2. current_tokens >= ceil(max_tokens * trigger_ratio)
          3. pending_evicted >= min_evict_batch
        """
        ...

    async def acompress(self, messages: Sequence[Message], *,
                        previous_summary: str | None = None) -> str:
        """生成新摘要。**永不抛异常**：
        - llm 为 None 或调用失败 -> 走 fallback_extractive_summary
        - 结果超过 max_summary_chars -> truncate_head_tail
        - 每次都递增 compression_count
        返回新的摘要文本（空字符串表示无可摘要内容，调用方不应写入空摘要）。"""
        ...

    def fallback_extractive_summary(self, messages: Sequence[Message]) -> str:
        """抽取式兜底（零 LLM）：逐条取 '[role] ' + content 前 80 字符，按行拼接，
        最后由 stats 行收尾：'(N messages summarized)'。截断到 max_summary_chars。"""
        ...
```

`SUMMARY_PROMPT_TEMPLATE`（冻结字面量，`{variables}` 用 `config.render_template` 渲染）：

```text
You are compressing a conversation history for an AI agent.
Write a concise summary in the SAME LANGUAGE as the conversation.

It MUST preserve, verbatim where possible:
- the user's original goal and any explicit constraints
- file paths, function/class names, commands, URLs, numbers
- decisions already made and the reasons
- what has been tried and FAILED (so it is not retried)
- the current state of the task and what remains

Do NOT add commentary, do NOT invent facts. Use bullet points. Max {max_chars} characters.

{previous_block}
Conversation to compress:
{transcript}

Summary:
```

`previous_block` 渲染为 `"Existing summary so far (merge, do not repeat):\n<prev>\n\n"` 或 `""`。

**摘要内容如何进入 prompt（冻结）**：摘要以**单条 system 消息**插入，位置在系统提示之后、
检索到的长期记忆之前，内容包裹为：

```text
<conversation_summary>
{summary}
</conversation_summary>
```

`Message.metadata["kind"] == "summary"`，且**不进入短期窗口的计数预算**
（它由摘要器独立管理），即 `MemoryManager.abuild_prompt` 里把它作为独立段拼装。

### 8.5 `memory/vector.py` —— 长期向量记忆

```python
@dataclass
class VectorConfig:
    dim: int = DEFAULT_HASHING_EMBED_DIM
    write_policy: str = "selective"          # "selective" | "turn" | "manual"
    auto_write_min_chars: int = 40
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    max_items: int = DEFAULT_MAX_MEMORY_ITEMS
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    retrieve_min_score: float = 0.0
    w_sim: float = DEFAULT_W_SIM
    w_recency: float = DEFAULT_W_RECENCY
    w_importance: float = DEFAULT_W_IMPORTANCE
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_D
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    use_numpy: bool = True                   # NUMPY_AVAILABLE 时是否用矩阵加速

AUTO_WRITE_MARKERS: tuple[str, ...]          # 冻结关键字（小写匹配，命中即自动写入）：
# ("记住","请记","记下","我是","我的名字","我偏好","我喜欢","以后","下次",
#  "remember","my name is","i prefer","i like","note that","keep in mind",
#  "always use","never use","don't forget")

class VectorMemory(MemoryStore):
    """长期记忆：向量检索 + 近因/重要度混合打分 + MMR 去冗。
    [v2] **线程安全**：所有公开方法在 `self._lock = threading.RLock()` 内完成（§8.1）。"""

    def __init__(self, embedder: Embedder | None = None,
                 config: VectorConfig | None = None,
                 *, on_event: LowLevelEvent | None = None) -> ...
        """[v2 变更] **维度权威性冻结**：
        - `embedder is not None` -> `self.dim = embedder.dim` 并写回 `self.config.dim = self.dim`
          （注入的 embedder 的维度**是权威**）。没有这条，测试注入 dim=4 的
          `CallableEmbedder` 后每次 add 都会被 `MemoryStoreError` 打死。
        - `embedder is None` -> 用 `config.dim` 造 `HashingEmbedder(dim=config.dim)`。
        - 所有维度校验一律对 `self.dim`。"""

    name = "vector"
    dim: int                     # [v2 变更] 公开属性
    embedder: Embedder

    # MemoryStore 实现
    def add(self, item: MemoryItem) -> None:
        """未算 embedding 时补算；维度不符 -> MemoryStoreError；
        超过 max_items -> FIFO 淘汰最旧（记 WARNING 并发事件）。"""
        ...
    def get(self, item_id: str) -> MemoryItem | None: ...
    def search(self, query: str, *, limit: int = DEFAULT_RETRIEVE_LIMIT,
               min_score: float | None = None,
               now: float | None = None,
               use_mmr: bool | None = None,
               metadata_filter: Mapping[str, Any] | None = None) -> list[MemoryItem]: ...
    def all(self) -> list[MemoryItem]: ...
    def delete(self, item_id: str) -> bool: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

    # 向量层专有
    def upsert(self, item: MemoryItem) -> tuple[MemoryItem, bool]:
        """去重写入。返回 (最终条目, 是否新建)。
        若与已有条目最大余弦相似度 >= dedup_threshold：**更新**已有条目
        （content 替换、created_at 刷新、importance 取 max、access_count 保留），
        否则新建。dict 里始终保留 item.id -> 已存在则视为更新而非新建。"""
        ...
    def score_item(self, item: MemoryItem, query_vec: Sequence[float] | None, *,
                   now: float | None = None) -> float:
        """见 §8.5.1 的冻结公式。query_vec=None 时 sim 项为 0。"""
        ...
    def should_auto_write(self, message: Message) -> bool:
        """按 write_policy 判定：
        - "manual"    -> 恒 False（只有显式 remember 才写）
        - "turn"      -> role=="user" 时恒 True
        - "selective" -> role=="user"
                         且 (len(content) >= auto_write_min_chars
                             或 任一 AUTO_WRITE_MARKERS 出现在 content.lower() 中)
                         且 content 不为空、不是纯符号
        [v2 冻结] **role 约束不可豁免**：`role != "user"` 时恒 False（任何策略下）。
        `MemoryManager.aadd` 传 `auto_write=True` 只豁免长度/marker 启发式，
        **不得**豁免 role 约束（对非 user 消息传 auto_write=True 记 WARNING 并拒绝写入）。
        理由：assistant 的答案不该被当成"用户事实"写进长期库、下一轮又被召回。"""
        ...
    def stats(self) -> dict[str, Any]: ...

    # [v2 新增] 持久化（让"长期存储"真的跨进程）
    def save(self, path: str | os.PathLike[str], *, include_embedding: bool = True) -> int:
        """JSONL 追加写（'w' 覆盖），每行 `MemoryItem.to_dict(include_embedding=...)`。
        每 DEFAULT_PERSIST_BATCH 条 flush 一次。返回写入条数。
        父目录不存在则创建；写失败 -> MemoryStoreError。"""
        ...
    def load(self, path: str | os.PathLike[str]) -> int:
        """读 JSONL 并 add 进内存。返回加载条数。
        坏行跳过并记 WARNING（不抛）；embedding 维度与 self.dim 不符 -> MemoryStoreError。"""
        ...
```

#### 8.5.1 混合打分公式（冻结，面试核心）

```text
score = w_sim * sim + w_recency * recency + w_importance * importance

  sim        = cosine_similarity(query_vec, item.embedding)        范围 [-1, 1]（HashingEmbedder 常见 0~0.5）
  recency    = 2 ** ( - age_days / half_life_days )                范围 (0, 1]；age_days = (now - created_at)/86400
  importance = clamp(item.importance, 0.0, 1.0)
```

默认权重 `w_sim=1.0, w_recency=0.15, w_importance=0.1`，`half_life_days=7.0`
（即 7 天前的记忆近因项衰减到 0.5）。`now` 为 `None` 时取 `utc_now()`。

**排序与截断（冻结，保证确定性）**：
1. 过滤：`sim < 0` 的条目丢弃（负相关明显不相关）；`metadata_filter` 不满足的丢弃。
2. 计算 `score`，把 `sim/recency/importance/score` 写进 `item.score_breakdown` 与 `item.score`。
3. 按 `(-score, -created_at, id)` 排序（**三级稳定排序**）。
4. 取候选池 `candidates = sorted[: max(limit * 3, limit)]`。
5. 若 `use_mmr`（默认：`limit > 1` 时为 True）：对候选池做 MMR 重排
   ```text
   mmr_i = mmr_lambda * score_i - (1 - mmr_lambda) * max_{j in selected} sim(i, j)
   每轮选 argmax(mmr)，直到选满 limit 或无候选；并列时取 id 字典序最小者（确定性）
   ```
6. 命中条目 `access_count += 1`、`last_access_at = now`（**持久化副作用**，
   调用方需知道 search 不是只读的）。**整段在同一把锁内完成**。

### 8.6 `memory/manager.py` —— 编排

```python
class MemoryManager:
    """三层协作的唯一入口。Agent 只与本类交互，不直接碰 Buffer/Vector/Summary。"""

    def __init__(self, *, buffer: BufferMemory | None = None,
                 long_term: VectorMemory | None = None,
                 summarizer: SummaryMemory | None = None,
                 config: MemoryConfig | None = None,
                 tokenizer: Tokenizer | None = None,
                 on_event: LowLevelEvent | None = None) -> ...

    config: MemoryConfig
    buffer: BufferMemory
    long_term: VectorMemory | None            # config.long_term_enabled=False 时为 None
    summarizer: SummaryMemory | None
    last_retrieved: list[MemoryItem]          # [v2 新增] 最近一次 abuild_prompt 检索到的条目
                                              # （**只读属性**，见下）

    # ---- 写 ----
    async def aadd(self, message: Message, *, auto_write: bool | None = None) -> MemoryItem | None:
        """1) buffer.add(message)
           2) 若 long_term 启用且 should_auto_write(message)（auto_write 参数可覆盖启发式判定）
              -> await awrite_long_term(...)（走 upsert 去重）
           3) emit MEMORY_WRITE {kind: message.metadata.get("kind"), count}
           返回新写入的长期条目或 None。**不做**压缩检查（压缩在轮次末尾统一做，见下）。"""
        ...
    async def aadd_turn(self, user_message: Message, assistant_message: Message | None = None
                        ) -> list[MemoryItem]:
        """保留 API（当前没有调用点）。文档明确标注"保留 API"，
        避免实现者猜它是否有副作用。"""
        ...
    async def aremember(self, content: str, *, importance: float = 0.5,
                        metadata: Mapping[str, Any] | None = None,
                        source: str = "") -> MemoryItem:
        """显式长期写入（memory_tools.remember / Agent 调用）。走 upsert 去重。
        long_term 为 None 时抛 MemoryStoreError。"""
        ...
    async def awrite_long_term(self, item: MemoryItem) -> MemoryItem: ...

    # ---- 读（[v2 变更] 全部支持 now 注入）----
    def retrieve(self, query: str, *, limit: int | None = None,
                 now: float | None = None, min_score: float | None = None,
                 use_mmr: bool | None = None) -> list[MemoryItem]:
        """同步检索（纯内存，无 LLM）。**逐级透传 now 给 VectorMemory.search**。"""
        ...
    async def aretrieve(self, query: str, *, limit: int | None = None,
                        now: float | None = None, min_score: float | None = None,
                        use_mmr: bool | None = None) -> list[MemoryItem]:
        """异步镜像（当前实现只是 await 一个 sync 调用；保留异步形态是为了未来
        支持 RemoteEmbedder 的网络调用而不破坏 API）。"""
        ...
    async def abuild_prompt(self, *, system: str, user_input: str,
                            extra: Sequence[Message] = (),
                            retrieve: bool = True,
                            memory_query: str | None = None,
                            now: float | None = None,
                            append_user_input: bool = True) -> list[Message]:
        """**组装顺序（冻结）**：
           1. Message(SYSTEM, system)
           2. [摘要消息]  若 summarizer.summary 非空 -> Message(SYSTEM, '<conversation_summary>...')
              metadata={"kind": "summary"}
           3. [长期记忆块] 若 retrieve 且检索有结果 -> Message(SYSTEM, '<relevant_memories>...')
              metadata={"kind": "memories", "memory_count": n}   # [v2 变更] 补上写 kind
              memory_query 默认用 user_input；渲染格式（冻结）：
                  <relevant_memories>
                  - (score=0.83, 2026-09-27) <content>
                  ...
                  </relevant_memories>
              日期用 `config.format_ts(item.created_at)`（**UTC**）渲染；
              单条 content 超过 500 字符时截断；整体超过 4000 字符时按分数截条数。
           4. buffer.window() 的全部消息
           5. extra 里的消息（如 multiagent 的上下文注入）
           6. [v2 变更] `append_user_input=True` 时追加 `Message(USER, user_input)`。
              语义冻结：`user_input` = **当前轮的增量用户指令**；
              Agent 只在 `state.step == 1` 传 run input，后续轮传 `""` 且 `append_user_input=False`
              （`user_input == ""` 时**不追加**，避免造出空消息）。
              历史用户消息已在 window 里（Agent 在 run 开始时 `aadd` 过一次，见 §9.4.1）。

        **[v2 新增] 检索结果的可观测性**：本方法把检索到的条目存进 `self.last_retrieved`
        （**不再触发第二次 search**；`VectorMemory.search` 有写 `access_count` 的副作用，
        重复调用会污染数据）。§9.4.1 的 `MEMORY_RETRIEVE` 事件用
        `len(memory.last_retrieved)` 而不是伪代码里的未定义变量 `retrieved`。
        [v2 新增] 末尾的最终预算校验见 `_enforce_context_budget()`。
        注意：第 2/3 段是**每轮重新生成**的，不写回 buffer，避免重复累积。"""
        ...
    def build_prompt(self, **kwargs: Any) -> list[Message]:
        """run_sync(lambda: self.abuild_prompt(**kwargs))；在运行中的 loop 内抛 ConfigError
        （Agent 内部一律用 abuild_prompt）。"""
        ...

    # ---- 压缩 ----
    async def acompress_if_needed(self, *, force: bool = False) -> str | None:
        """判定 + 执行 + 写回：取 buffer.drain_evicted()，若满足 should_compress 则
        await summarizer.acompress(evicted, previous_summary=self.summarizer.summary)，
        结果存入 summarizer，emit MEMORY_COMPRESS {before_tokens, after_tokens, compressed}。
        返回新摘要或 None。
        summarizer 为 None 时：若 evicted 非空则直接丢弃（记 WARNING）。
        **调用契约（冻结）**：Agent 只在每轮末尾调用一次（§9.4.1 步骤 6 之后），
        **禁止**在单轮内为每条 observation 都调一次 —— 否则一轮 N 个工具会触发 N 次 LLM 摘要判定。"""
        ...
    async def acompress(self, *, force: bool = False) -> str | None: ...   # 同上的显式别名

    # ---- 维护 ----
    async def aclear(self, *, long_term: bool = False, summary: bool = True) -> None: ...
    def stats(self) -> dict[str, Any]:
        """{"short_term_messages","short_term_tokens","window_messages","window_tokens",
            "evicted_pending","long_term_items","has_summary","summary_tokens",
            "compressions","embedder","buffer_budget_tokens","context_window_tokens"}
        [v2 变更] 末尾两个键是 §8.6 的预算反推结果（`context_window_tokens` 为 None 时
        `buffer_budget_tokens` 就等于 `config.buffer_max_tokens`）。"""
        ...
    def to_dict(self) -> dict[str, Any]: ...     # 用于 trace / CLI /memory
    @classmethod
    def from_config(cls, config: MemoryConfig | None = None, *, llm: "LLMClient | None" = None,
                    embedder: Embedder | None = None) -> "MemoryManager":
        """按配置装配三层；llm 为 None 时摘要器用抽取式兜底。
        [v2 冻结] **必须把 embedder 透传给 VectorMemory**；
        `MemoryConfig.embedder_dim` 仅在 `embedder is None` 时生效。
        `tokenizer` 参数覆盖三层内部各自的 tokenizer（构造后统一 set）。
        `persist_path` 非 None 时在构造末尾执行 `restore`（失败记 WARNING 不抛）。"""
        ...
    # [v2 新增]
    def persist(self, path: str | os.PathLike[str] | None = None) -> int:
        """把长期记忆写到磁盘（默认用 config.persist_path，为 None 时 ConfigError）。
        返回写入条数。long_term 为 None 时返回 0。"""
        ...
    def restore(self, path: str | os.PathLike[str] | None = None) -> int:
        """从磁盘恢复长期记忆。文件不存在 -> 返回 0（不抛）。"""
        ...
    def tokenizer_chars_budget(self) -> int:
        """[v2 新增] 给 executor 用的字符预算：`buffer_budget_tokens * DEFAULT_TOKEN_CHAR_RATIO`。
        §7.4.1 步骤 5.d 的 `max_observation_chars` 用它取 min，把 token 预算与字符上限联动。"""
        ...
```

**`[v2 变更]` `MemoryConfig` -> 各层 `Config` 的冻结映射表（`from_config` 逐字段实现）**：

| `MemoryConfig` 字段 | 目标 | 目标字段 |
|---|---|---|
| `buffer_max_tokens` | `BufferConfig` | `max_tokens`（**改名**） |
| `buffer_max_messages` | `BufferConfig` | `max_messages`（改名） |
| `buffer_keep_last_n` | `BufferConfig` | `keep_last_n`（改名） |
| —（无对应） | `BufferConfig` | `keep_system=True`（固定） |
| `summary_enabled` | `SummaryConfig` | `enabled`（改名） |
| `summary_trigger_ratio` | `SummaryConfig` | `trigger_ratio`（改名） |
| `summary_min_evict` | `SummaryConfig` | `min_evict_batch`（改名） |
| `max_summary_chars` | `SummaryConfig` | `max_summary_chars`（同名直传） |
| —（无对应） | `SummaryConfig` | `max_input_chars=16000`、`update_existing=True`（固定） |
| `embedder_dim` | `VectorConfig` | `dim`（**改名**，且仅在 embedder is None 时生效） |
| `write_policy` / `auto_write_min_chars` / `dedup_threshold` / `max_items` / `retrieve_limit` / `w_sim` / `w_recency` / `w_importance` / `mmr_lambda` | `VectorConfig` | 同名直传 |
| `retrieve_min_score` | `VectorConfig` | `retrieve_min_score`（同名；而 `VectorMemory.search` 的参数叫 `min_score` —— 这是**唯一**一处刻意改名，实现者注意） |
| `recency_half_life_days` | `VectorConfig` | `half_life_days`（改名） |
| —（无对应） | `VectorConfig` | `use_numpy=True`（固定） |

**`[v2 新增]` 上下文窗口反推预算（D-15）**：`from_config` 里若
`config.context_window_tokens is not None`，则

```python
budget = (config.context_window_tokens
          - config.reserve_completion_tokens
          - config.tools_schema_tokens_reserve
          - tokenizer.estimate(system_prompt))          # system_prompt 由调用方提供，
                                                        # 无则用 DEFAULT_REACT_SYSTEM_PROMPT 渲染后的长度
buffer.config.max_tokens = max(512, budget)             # 下限 512
```

四个数放进 `MemoryManager.stats()`。**最终校验 `_enforce_context_budget()`**：
`abuild_prompt` 末尾若整段估算超过 `context_window_tokens`，按
「长期记忆块 -> 窗口最旧消息（不动 pinned system 与最后一条 user）」的顺序裁剪，
并 `emit CONTEXT_TRUNCATED {before, after, dropped_messages}`（兑现 §13 红线 12）。

**同步/异步双 API 的冻结规则**：`MemoryManager` 里所有需要 LLM 的方法**只有 async 版本**
（`acompress_if_needed`、`aadd`）。`retrieve` / `stats` / `build_prompt` 有同步版本，
但 `build_prompt` 的同步版本在运行中的 loop 内调用会抛 `ConfigError`。
`Agent` 内部一律用 `a*` 版本。

---

## 9. `liteagent/agent/` —— ReAct 循环

### 9.1 `agent/state.py`

```python
class AgentStatus(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

@dataclass
class AgentState:
    run_id: str
    input: str = ""
    agent_name: str = "agent"
    messages: list[Message] = field(default_factory=list)      # 未压缩的完整 transcript
    step: int = 0
    status: AgentStatus = AgentStatus.IDLE
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    action_counts: dict[str, int] = field(default_factory=dict)   # canonical_key -> 次数
    # [v2 新增] 循环防护的第二/第三层（D-16）
    tool_name_counts: dict[str, int] = field(default_factory=dict)        # 只按 name 计数
    observation_digests: dict[str, int] = field(default_factory=dict)     # blake2b 摘要 -> 次数
    tool_failure_counts: dict[str, int] = field(default_factory=dict)     # infrastructure 失败次数
    parse_errors: int = 0
    llm_errors: int = 0
    truncation_errors: int = 0                                            # [v2] length 截断续写次数
    nudges: list[str] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    error: LiteAgentError | None = None
    started_at: float = field(default_factory=utc_now)          # [v2] utc_now
    finished_at: float | None = None

    @classmethod
    def create(cls, input: str, *, agent_name: str = "agent",
               run_id: str | None = None) -> "AgentState":
        """run_id 默认 'run_' + uuid4().hex[:12]。"""
        ...

    def add_message(self, message: Message) -> Message:
        """追加到 messages 并返回它。**（[v2 变更]）**不再写"由调用方控制上限"这句话：
        上限由 `Agent` 在每步末尾用 `trim_transcript(config.max_transcript_messages)` 统一执行。"""
        ...
    def trim_transcript(self, max_messages: int) -> int:
        """[v2 新增] 保留前导 SYSTEM 消息 + 最近 max_messages 条（用
        `drop_orphan_tool_messages` 先修工具对，否则会切断 assistant/tool 配对）。
        返回被丢弃的条数。被丢弃的消息**仍然保留在 trace 事件里**（事件已经在 emit 时发出）。"""
        ...
    def record_tool_call(self, call: ToolCall) -> int:
        """冻结顺序：1) `action_counts[call.canonical_key()] += 1`；
        2) `tool_name_counts[call.name] += 1`；返回 canonical_key 的累计次数。"""
        ...
    def repeat_count(self, call: ToolCall) -> int: ...
    def record_observation(self, result: ToolResult) -> int:
        """[v2 新增] 计算 `blake2b(result.content.encode(), digest_size=8).hexdigest()`，
        `observation_digests[digest] += 1` 并返回该计数。"""
        ...
    def record_tool_failure(self, name: str) -> int: ...
    def last_message(self) -> Message | None: ...
    def last_tool_results(self, n: int = 1) -> list[ToolResult]: ...
    def mark_finished(self, status: AgentStatus, *, error: LiteAgentError | None = None) -> None:
        """设置 status/error/**finished_at = utc_now()**。**幂等**：已 FINISHED/FAILED/ABORTED 时
        不覆盖 status（只补 finished_at），保证 finally 里的补救调用不会改写结果。"""
        ...
    def clone(self) -> "AgentState":
        """浅拷贝容器、深拷贝 messages 列表（**用于 multiagent 里给子 Agent 传状态**）。
        明确不做深拷贝 Message 内部（Message 视为不可变使用）。"""
        ...
    @property
    def duration_ms(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentState": ...

@dataclass
class AgentResult:
    output: str = ""
    status: AgentStatus = AgentStatus.FINISHED
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: LiteAgentError | None = None
    state: AgentState | None = None
    duration_ms: float = 0.0
    agent_name: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool: ...                          # status == FINISHED
    def to_dict(self, *, include_state: bool = False) -> dict[str, Any]: ...
    @classmethod
    def from_state(cls, state: AgentState, *, output: str = "",
                   metadata: Mapping[str, Any] | None = None) -> "AgentResult": ...
    def raise_for_status(self) -> None:
        """status != FINISHED 时 raise self.error 或 AgentError(...)。"""
        ...
```

**字段顺序的冻结说明（`[v2 变更]`，v1 的伪代码照抄会崩）**：`AgentResult` 的**第一个**字段是
`output`、第二个是 `status`。因此**禁止位置参数构造** —— `AgentResult(FAILED, error=e)` 会把
`AgentStatus.FAILED` 塞进 `output`、`status` 保持默认的 `FINISHED`（"失败却 ok=True"）。
**所有构造点必须写关键字参数**，统一使用一个局部构造：

```python
AgentResult(output=last_assistant_text, status=AgentStatus.FAILED, error=e, state=state,
            agent_name=self.name, duration_ms=state.duration_ms,
            steps=state.step, tool_calls=list(state.tool_calls),
            tool_results=list(state.tool_results), usage=state.usage)
```

（`Agent` 内部**必须**有一个私有 helper `_result(state, output, status, error=None, metadata=None)`
把所有构造收敛到一处；测试用 `test_zero_dependency` 之外的 `ast` 守门用例断言
`AgentResult(` 的出现形式只能是关键字形式 —— 见 §12。）

**`AgentConfig` 的唯一归属地是 `config.py`**；`agent/state.py` 只
`from liteagent.config import AgentConfig` 再 re-export。
`state.py` 里**不允许**出现 `AgentConfig` 的类定义（v1 残留的那句"定义在 state.py"已删除）。

### 9.2 `agent/callbacks.py`

```python
class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    THOUGHT = "thought"
    ACTION_PARSED = "action_parsed"
    PARSE_ERROR = "parse_error"
    REPEAT_DETECTED = "repeat_detected"
    NUDGE = "nudge"
    TOOL_STARTED = "tool_started"
    TOOL_RETRY = "tool_retry"
    TOOL_FINISHED = "tool_finished"
    TOOL_ERROR = "tool_error"
    TOOL_APPROVAL = "tool_approval"          # [v2 新增] HITL 审批结果
    MEMORY_WRITE = "memory_write"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_COMPRESS = "memory_compress"
    BUDGET_EXCEEDED = "budget_exceeded"      # [v2 新增] token/成本预算超限
    CONTEXT_TRUNCATED = "context_truncated"  # [v2 新增] 上下文裁剪（降级可观测）
    AGENT_DELEGATE = "agent_delegate"
    AGENT_RETURN = "agent_return"
    BLACKBOARD_WRITE = "blackboard_write"
    BLACKBOARD_READ = "blackboard_read"

    @classmethod
    def coerce(cls, value: "EventType | str") -> "EventType":
        """未知字符串 -> ConfigError"""
        ...

@dataclass
class TraceEvent:
    type: EventType
    run_id: str = ""
    agent_name: str = ""
    step: int = 0
    timestamp: float = field(default_factory=utc_now)      # [v2] utc_now
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """[v2 新增] **构造即校验**：`data` 里出现保留键
        {"type","run_id","agent_name","step","ts"} -> `ConfigError`。
        冻结在这里抛（而不是 to_dict 时静默覆盖），见 §12 的 `test_callbacks` 用例。"""
        ...

    def to_dict(self) -> dict[str, Any]:
        """{"type": <value>, "run_id", "agent_name", "step", "ts", **data}
        **恒输出键名 `ts`**（不是 `timestamp`）。data 的键**平铺**到顶层
        （便于 JSONL 直接 grep），因此 data 里禁止出现那五个保留键。
        data 里的值必须已过 to_jsonable（§2.7）。"""
        ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceEvent":
        """[v2 变更] 补齐 v1 漏掉的映射规则：
        - 平铺形态：`ts` -> `timestamp`（**同时接受 `timestamp` 键，二者都存在时以 `ts` 为准**）；
          `type` 用 `EventType.coerce`；其余非保留键整体进 `.data`。
        - 嵌套形态：`data` 字段内的键整体进 `.data`。
        - **保留键集合固定为 {type, run_id, agent_name, step, ts}**。
        这条规则是 `to_json`/`from_json` 往返测试的前提。"""
        ...
    def to_json(self) -> str: ...                     # 单行，ensure_ascii=False
    @classmethod
    def from_json(cls, line: str) -> "TraceEvent": ...
    def summary(self) -> str:
        """单行人类可读摘要，CLI `trace` 与 LoggingCallback 用，
        如 '[3] tool_finished read_file ok=True 12.3ms'。"""
        ...

class Callback(Protocol):
    def on_event(self, event: TraceEvent) -> None: ...

# 允许的订阅形态（冻结）
CallbackLike = Union[Callback, Callable[[TraceEvent], None]]

class CallbackManager:
    def __init__(self, callbacks: Sequence[CallbackLike] = ()) -> None: ...
    def add(self, callback: CallbackLike) -> None: ...
    def remove(self, callback: CallbackLike) -> None: ...
    def subscribe(self, fn: Callable[[TraceEvent], None]) -> Callable[[], None]:
        """返回一个 unsubscribe 闭包。"""
        ...
    def emit(self, event: TraceEvent) -> None:
        """**永不抛异常**：单个回调抛错时记入 self.errors 并继续调用其余回调，
        同时向 logging.getLogger("liteagent.callbacks") 记 exception。
        **[v2 变更] 线程安全**：`self._lock = threading.Lock()` 保护回调列表的迭代与
        `errors` 的追加（内置回调会被 worker 线程调用，见 §2.7）。"""
        ...
    def emit_type(self, event_type: EventType | str, *, run_id: str = "",
                  agent_name: str = "", step: int = 0, **data: Any) -> TraceEvent:
        """构造 + emit，返回构造出的 event。
        [v2 变更] **提前校验保留键**（在构造 TraceEvent 之前），避免 to_dict 静默覆盖。"""
        ...
    errors: list[tuple[CallbackLike, BaseException]]     # 实例属性
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

# 内置回调（全部定义在本模块，**[v2 冻结] 全部必须线程安全**）
class FunctionCallback:
    """把任意函数包成 Callback；函数签名可以是 (event) 或 (event_type: str, data: dict)，
    用 `inspect.signature` 的参数个数判定。"""
    def __init__(self, fn: Callable[..., None], *, event_types: Sequence[EventType] | None = None,
                 name: str = "") -> None: ...

class LoggingCallback:
    """logging.getLogger(logger_name)，按 level 输出 event.summary()。
    level 默认 INFO；ERROR 级事件（run_failed/llm_error/tool_error）用 WARNING。"""
    def __init__(self, logger: Any = None, *, level: int = logging.INFO,
                 event_types: Sequence[EventType] | None = None) -> None: ...

class JsonlTraceCallback:
    """把事件按行追加写入文件（'a' 模式）；close() 关闭。
    [v2 变更] **必须**用 `self._lock = threading.Lock()` 包住写入
    （行缓冲写入不是原子的，delegate 路径会在 worker 线程里 emit）。"""
    def __init__(self, path: str | os.PathLike[str]) -> None: ...
    def close(self) -> None: ...
    def load(self) -> list[TraceEvent]: ...            # 读回（含之前已存在的事件）

class RichCallback:
    """有 rich 时彩色输出；无 rich 时退化为 print。必须两种环境都能 import 与运行。"""
    def __init__(self, *, show_tokens: bool = True, console: Any = None) -> None: ...

class TokenCounterCallback:
    """累计 usage 并暴露聚合结果。usage 从 llm_response 事件的 data['usage'] 读取
    （[v2 变更] 用 `TokenUsage.from_dict(data['usage'])`，因为 data 里的 usage 已经是 dict）。
    [v2 新增] `cost_usd: float | None` 属性：按 `config.estimate_cost_usd` 与事件里的 model 累加；
    全部模型都未命中价格表时返回 None。"""
    def __init__(self) -> None: ...
    usage: TokenUsage
    calls: int
    cost_usd: float | None
    def reset(self) -> None: ...

class MemoryTraceCallback:
    """把 memory_* 事件收集成列表，测试用。"""
    def __init__(self) -> None: ...
    events: list[TraceEvent]

class TraceRecorder:
    """上下文管理器：创建 run_id、挂载 JSONL 回调、结束时关闭。
        with TraceRecorder("trace.jsonl") as rec:
            rec.manager.emit_type(EventType.RUN_STARTED)
            rec.events   # 内存中的全部事件
    """
    def __init__(self, path: str | os.PathLike[str] | None = None, *,
                 callbacks: Sequence[CallbackLike] = (), run_id: str | None = None) -> None: ...
    manager: CallbackManager
    run_id: str
    events: list[TraceEvent]        # list.append 是原子的，无需加锁（文档注明即可）
    def __enter__(self) -> "TraceRecorder": ...
    def __exit__(self, *exc: Any) -> None: ...

def load_trace(path: str | os.PathLike[str]) -> list[TraceEvent]:
    """读 JSONL；跳过空行与无法解析的行（记 WARNING），不抛异常。"""
    ...

def total_usage(events: Sequence[TraceEvent]) -> TokenUsage: ...
def events_of_type(events: Sequence[TraceEvent], event_type: EventType | str) -> list[TraceEvent]: ...
def render_trace(events: Sequence[TraceEvent], *, indent: bool = True) -> str:
    """给 CLI `trace` 的树状渲染：按 step 分组，缩进显示 thought/action/tool。"""
    ...

def trace_stats(events: Sequence[TraceEvent]) -> dict[str, Any]:
    """[v2 新增] **冻结的字段表**（CLI `trace --stats --json` 的输出就是它）：
    {
      "runs": int, "steps": int, "llm_calls": int,
      "llm_latency_ms": {"total": float, "mean": float, "p50": float, "p95": float},
      "tool_calls": int, "tool_failures": int,
      "tool_latency_ms": {"<name>": {"count": int, "mean": float, "p95": float}, ...},
      "retries": int, "parse_errors": int, "nudges": int,
      "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
      "cost_usd": float | None,
      "errors": {"<error_type>": int, ...}
    }
    **p50/p95 的算法冻结**：对已排序的长度为 n 的序列，取下标
    `int(0.5 * (n - 1))` 与 `int(0.95 * (n - 1))`（nearest-lower，无插值）。
    n == 0 时 total=mean=p50=p95=0.0。`cost_usd` 在所有 usage 事件的 model 都未命中价格表时为 None。
    """
    ...

def as_llm_callback(manager: CallbackManager) -> LowLevelEvent:
    """把 LLM/tools/memory 层的轻量回调 (event_type_str, data) 转成 TraceEvent 并 emit。
    对未知字符串**忽略**（不抛异常），保证低层不会因为高层新增事件类型而崩。
    实现：`EventType.coerce` 失败时 `logging.getLogger("liteagent.callbacks").debug(...)` 后 return。"""
    ...
```

### 9.3 `agent/parser.py` —— 文本 ReAct 语法

```python
ACTION_MARKERS: tuple[str, ...] = ("action", "行动", "动作")
THOUGHT_MARKERS: tuple[str, ...] = ("thought", "思考", "想法")
OBSERVATION_MARKERS: tuple[str, ...] = ("observation", "观察", "结果")
FINAL_MARKERS: tuple[str, ...] = ("final answer", "最终答案", "最终回答", "答案")

@dataclass
class ParsedAction:
    thought: str | None = None
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    raw: str = ""
    json_mode: bool = False          # True 表示来自整段 JSON 解析
    def is_final(self) -> bool: ...
    def is_action(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...

class ReActParser:
    def __init__(self, *, tool_names: Sequence[str] | None = None,
                 tool_param_names: Mapping[str, Sequence[str]] | None = None,
                 strict: bool = True,
                 allow_json_block: bool = True,
                 allow_json_object: bool = True,
                 allow_final_answer: bool = True,
                 case_sensitive: bool = False,
                 allow_cjk_markers: bool = True) -> None: ...

    tool_names: tuple[str, ...] | None

    def parse(self, text: str) -> ParsedAction:
        """解析顺序（冻结）：
        1. 规范化：\r\n -> \n、剥离首尾空白。全角冒号 U+FF1A -> ':' **只在 marker 匹配时用**，
           不修改原文，以保证 offset 可定位。
        2. 若 allow_json_object 且整段能被 json.loads 成 dict：
           - 含 "final_answer" -> ParsedAction(final_answer=...)
           - 含 "action" -> ParsedAction(action=..., action_input=data.get("action_input") or {},
                                          thought=data.get("thought"))
           - 含 "tool"/"tool_name" 亦视为 action（容错别名）
           - 否则继续第 3 步
        3. 逐行扫描找 marker（大小写不敏感；若 allow_cjk_markers 也匹配中文 marker；
           允许 Markdown 强调符包裹与任意前导空白/列表符号 - * >）：
           marker 正则构造：r"^[\\s>*\\-]*(?:\\*\\*|__)?\\s*" + marker_alt + r"\\s*(?:\\*\\*|__)?\\s*[:：]\\s*"
        4. **Action 优先于 Final Answer（[v2 变更]，v1 这里自相矛盾）**：
           **先扫描 Action marker**，只要存在合法的 `Action:` 行就解析为 action，
           同段落里的 Final Answer 文本**丢弃并记 warning**；
           仅当**没有任何 Action marker**时才取 Final Answer。
           不变量：`ParsedAction` 里 `action` 与 `final_answer` **至多一个非 None**，
           `is_action()` 与 `is_final()` **互斥**。
           理由：模型输出 Thought/Action/Action Input 后再补一句 Final Answer 是常见行为，
           "丢掉工具调用直接终结"是明显的错误行为。
        5. Action Input 的取值（冻结，按序尝试）：
           a. 紧随其后的 ```json / ``` 围栏块内容
           b. 从 marker 之后开始的**花括号配对**扫描（处理跨行 JSON）
           c. 单行剩余文本：先 json.loads；失败再 ast.literal_eval；再失败：
              若 tool_names 已知且该工具恰有 1 个必填参数 -> {"<该参数名>": 文本}
              （需 tool_param_names 提供映射）
              否则 -> 视为 {"input": 文本}
           d. 空 -> {}
        6. Final Answer 的取值：从 marker 之后到行尾/文档尾；若其中出现下一个
           Thought/Action marker 则截断到那里。做 strip 与首尾引号剥离。
        7. 严格模式：既无 action 也无 final_answer 也无 thought -> 抛 ReActParseError(
              raw=text, offset=0, reason="no ReAct structure found")。
           非严格模式：整个 text 当作 final_answer。
        8. 若 parse 出 action 但 action 名称不在 tool_names 中：
           **不抛异常**（返回 action 原样）——由 Agent 决定如何自纠正（更可测）。
        """
        ...

    def parse_tool_calls(self, response: "LLMResponse") -> list[ToolCall]:
        """**只做原生提取**：返回 `response.tool_calls` 的副本（arguments 为空时按 schema
        补默认值，不做校验）。
        [v2 变更] `response.tool_calls` 为空时**返回 []** ——
        **永不回落到文本解析**（文本模式由 Agent 直接调 parse()）。
        这样 mode='native' 的语义不会漂移。"""
        ...

    def extract_thought(self, text: str) -> str | None:
        """只取 Thought 段（不需要完整结构）。"""
        ...

    def strip_markers(self, text: str) -> str:
        """删除 Thought/Action/Action Input/Observation 行，返回剩余文本（用于最终答案清洗）。"""
        ...

    def build_observation(self, results: Sequence["ToolResult"], *,
                          max_chars: int = 8000, step: int | None = None) -> str:
        """多条结果渲染（冻结格式）：
           单个：'Observation: <obs>'
           多个：'Observation:\\n[1] <name> -> <obs1>\\n[2] <name> -> <obs2>'
           obs 用 ToolResult.to_observation(max_chars=max_chars/max(1,len(results)))"""
        ...

    def build_parse_error_feedback(self, error: "ReActParseError") -> str:
        """把解析失败回灌给模型的提示文本（冻结格式）：
           'Your previous output could not be parsed: <reason>.\\n'
           'Reply using exactly this format:\\n'
           'Thought: <your reasoning>\\nAction: <one of [names]>\\nAction Input: <JSON object>\\n'
           'or\\nThought: <...>\\nFinal Answer: <...>'
        """
        ...
```

### 9.4 `agent/agent.py` —— 状态机

```python
class Agent:
    def __init__(self, *, llm: LLMClient,
                 tools: ToolRegistry | None = None,
                 memory: MemoryManager | None = None,
                 config: AgentConfig | None = None,
                 executor: ToolExecutor | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 name: str | None = None,
                 description: str = "") -> None:
        """[v2 冻结]
        - `self._run_guard = threading.Lock()`（**arun 不可重入**，见下）
        - `self._state = AgentState.create("", agent_name=self.name)`（IDLE 空状态）
        - `memory is None` 时**必须**构造 `MemoryManager.from_config(MemoryConfig(), llm=self.llm)`
          —— 默认记忆**复用自身的 llm**，否则 SummaryMemory 永远走抽取式兜底，
          `test_memory_summary` 的"LLM 摘要成功路径"在 Agent 级路径上永远测不到。
          memory 由外部传入时**不得**替换其 summarizer 的 llm。写进 §13 红线。
        - `name or config.name`（`name` 参数优先）。"""
        ...

    llm: LLMClient
    tools: ToolRegistry
    memory: MemoryManager
    config: AgentConfig
    callbacks: CallbackManager
    executor: ToolExecutor

    @property
    def name(self) -> str: ...
    @property
    def state(self) -> AgentState:
        """**最近一次完整运行**的状态；未运行过时返回 status=IDLE 的空 AgentState
        （不抛异常，便于 CLI 探测）。
        [v2 变更] `self._state` **只在 arun 结束时一次性赋值**，运行期状态存在局部变量里
        （否则并发/嵌套运行会互相覆盖）。"""
        ...

    async def arun(self, input: str, *, state: AgentState | None = None,
                   callbacks: Sequence[CallbackLike] | None = None,
                   **overrides: Any) -> AgentResult:
        """**永不抛异常**（除 asyncio.CancelledError/KeyboardInterrupt）：
        所有失败编码进 AgentResult(status=FAILED, error=...)。
        仅当 config.raise_on_error=True 时才 re-raise。

        [v2 变更] **arun 不可重入（冻结）**：
        ```python
        if not self._run_guard.acquire(blocking=False):
            return AgentResult(output="", status=AgentStatus.FAILED,
                               agent_name=self.name,
                               error=AgentError(f"agent {self.name} already has a run in flight"))
        try:
            ...
        finally:
            self._run_guard.release()          # 必须用 finally 保证 CancelledError 路径也释放
        ```
        理由：`Agent.state` / `state.usage` / `state.messages` 是实例级累积状态；
        同一 worker 被 manager 在一个 step 里并行委派两次时，两个线程会互相覆盖、
        `ScriptedLLM._seq` 自竞态导致 call_id 重复。守卫把它变成一次**可见的失败工具调用**。

        [v2 变更] `**overrides` 的白名单（冻结）：
        只允许 `max_steps` / `temperature` / `max_tokens` / `tool_choice` / `mode` /
        `max_total_tokens` / `max_wall_clock_s`；非法键 -> `ConfigError`。
        原先未冻结，实现者会各自支持不同的键集。

        [v2 变更] `state` 注入：**必须保留注入 state 的 `run_id`**（Agent 不得重新生成），
        否则 test_agent_* 的 trace 断言只能绕着 run_id 走。"""
        ...

    def run(self, input: str, **kwargs: Any) -> AgentResult:
        """run_sync(lambda: self.arun(input, **kwargs))。"""
        ...

    async def astream(self, input: str, **kwargs: Any) -> AsyncIterator[TraceEvent]:
        """边跑边 yield 事件。**[v2 变更] v1 的"在同一个任务里 await self.arun"
        在字面上不可实现**（generator 只有在 arun 返回后才有机会 yield）。
        冻结伪代码：
        ```python
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        _SENTINEL = object()
        loop = asyncio.get_running_loop()

        def _put(ev: TraceEvent) -> None:
            if threading.current_thread() is not threading.main_thread() and \\
               loop is not asyncio.get_running_loop_safe():      # 见下方"线程判定"说明
                loop.call_soon_threadsafe(_safe_put, ev)
            else:
                _safe_put(ev)

        def _safe_put(ev) -> None:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                logging.getLogger("liteagent.agent").warning("astream queue full; dropping event %s",
                                                             ev.type.value)

        cb = FunctionCallback(_put)
        self.callbacks.add(cb)
        task = asyncio.create_task(self.arun(input, **kwargs))
        task.add_done_callback(lambda _t: loop.call_soon_threadsafe(_safe_put, _SENTINEL))
        try:
            while True:
                ev = await q.get()
                if ev is _SENTINEL:
                    break
                yield ev
        finally:
            self.callbacks.remove(cb)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)     # 不吞 GeneratorExit/CancelledError
        ```
        - **线程判定**：用 `threading.current_thread() is not loop 所属线程`；
          实现时冻结为记录创建时的 `threading.get_ident()`，
          在 `_put` 里与 `threading.get_ident()` 比较（不要用 main_thread 判定，
          因为 `astream` 可能跑在非主线程的 loop 里）。
        - **队列有界**（maxsize=1000），满时丢最旧？**不** —— 冻结为**丢最新并记 WARNING**
          （`put_nowait` 失败即丢），避免在热点路径上做 O(n) 的 `get_nowait`。
        - **`[v3 变更]` 消费者提前 `break` 的终结语义（按实测重写，v2 的"收到
          `GeneratorExit` 就取消、不会泄漏"不成立）**：Python **不保证** `async for` +
          `break` 立刻终结 async generator —— 取消是**延迟一拍**发生的，且**只在 generator
          被回收时**才发生。具体：
          * 写 `async for ev in agent.astream(...): ... break`（循环变量引用立刻掉 0）时，
            generator 不会在 `break` 的同一 tick 收到 `GeneratorExit`；要等到下一轮 GC
            回收它，`finally` 才 `task.cancel()` + `gather(return_exceptions=True)`。
            **在这段延迟窗口里 run guard（`_run_guard`）仍被占用**，`arun` 可能已经跑远。
          * 若调用方**持有 stream 引用**再 break（例如 `stream = agent.astream(...)` 后
            只消费一部分），generator 永不回收、finalizer 永不触发、`finally` 永不执行
            —— 运行**不会**被取消，`arun` 会跑完整个 ReAct 循环并写记忆（这正是 v2 声称
            已被修掉的 v1 行为）。
          * 想立刻停止：显式 `await stream.aclose()`（或用 `contextlib.aclosing`），
            且不要在 break 后立刻对同一个 `Agent` 再发 run（run guard 仍被占用，
            会按 §9.4 的不可重入契约返回 FAILED `"already has a run in flight"`）。
          实现侧的确定部分：`finally` 里 `self.callbacks.remove(callback)` + 未完成就
          `task.cancel()` + `await asyncio.gather(task, return_exceptions=True)`
          （**不吞** `GeneratorExit`/`CancelledError`）。
          回归测试：`tests/test_agent_features.py::AstreamTests::test_early_break_does_not_leak_the_run_task`、
          `::test_early_break_stops_a_tool_using_run`（后者还覆盖 3.10 `wait_for` 吞取消的
          GH-86296 陷阱 —— v3 已把 executor 的超时换成取消安全的 `_await_with_deadline`）。
        """
        ...

    def reset(self, *, clear_memory: bool = False) -> None: ...
    def add_tool(self, tool: Tool) -> None: ...
    def remove_tool(self, name: str) -> None: ...
    def describe(self) -> dict[str, Any]:
        """{"name","description","model","tools":[names],"mode","max_steps"}"""
        ...
    def _system_prompt(self) -> str:
        """[v2 新增，必须实现] 冻结规则（**模式无关**）：
        - `config.system_prompt is not None` -> **直接返回它**（不做渲染）
        - 否则取 `config.system_prompt_template`，用
          `render_template(t, {"name": self.name,
                               "tools": self.tools.to_prompt(fmt="text",
                                            max_tools=DEFAULT_MAX_TOOLS_IN_PROMPT),
                               "tool_names": ", ".join(self.tools.names())[:2000]})`
          **先按 text 渲染**；当解析出的模式是 `"native"` 时再渲染一次：
          `render_template(t, {..., "tools": "(provided via the tools parameter)"})`
          （native 模式不把工具清单塞进 prompt，省 token；但**仍使用同一个模板**，
           模板里的 "Never invent tool names" 规则对 native 同样有效）。
        """
        ...
    async def aclose(self) -> None:
        """关闭自建的 executor（外部传入的不关）。"""
        ...
```

#### 9.4.1 一轮（step）的精确步骤（冻结伪代码，`[v2 变更]` 已修掉 v1 的 6 处碰撞）

> **伪代码的记法约定**（实现者必读）：
> `FAILED` / `FINISHED` / `THINKING` 等裸名字都指 `AgentStatus.X`；
> **所有 `AgentResult` 构造一律写成 `self._result(state, output=..., status=..., error=...)`
> 或全关键字形式**（§9.1 的冻结说明）；`emit X {a, b}` 表示
> `callbacks.emit_type(EventType.X, a=..., b=...)`。
> 伪代码里的 `mode` 是 `llm.resolve_mode(has_tools=len(self.tools) > 0)` 的结果（§9.4.5）。
> `parser` 是 `ReActParser(tool_names=self.tools.names(), ...)` 的实例（文本模式才需要）。

```text
emit RUN_STARTED {input, mode, tools: [names]}
state.status = THINKING

# --- 0. 用户输入入 buffer（恰好一次）---
await memory.aadd(Message.user(input))          # [v2 新增] 用户输入只在此处入 buffer 一次
emit MEMORY_WRITE {kind: "user", count: 1}

while state.step < config.max_steps:
    # --- 0.5 预算与墙钟检查（每轮开头）---
    if config.max_wall_clock_s is not None and (utc_now() - state.started_at) > config.max_wall_clock_s:
        -> state.mark_finished(FAILED, RunTimeoutError(...)); emit RUN_FAILED; return self._result(...)
    state.step += 1
    emit STEP_STARTED {step}

    # --- 1. 取回长期记忆 + 组装 prompt ---
    messages = await memory.abuild_prompt(
        system=self._system_prompt(),
        user_input=input if state.step == 1 else "",
        append_user_input=(state.step == 1))
    emit MEMORY_RETRIEVE {count: len(memory.last_retrieved), query_len: len(input)}

    # --- 2. 调 LLM ---
    try:
        resp = await llm.achat(messages, tools=schemas_or_None, tool_choice=...,
                               temperature=..., max_tokens=...)
    except asyncio.CancelledError:
        raise                                        # [v2] 取消原样上抛（见 §9.4.6）
    except LiteAgentError as e:
        state.llm_errors += 1
        # LLM 层已按 retryable 重试过；到这里说明用尽 -> 终结
        state.mark_finished(FAILED, error=e); emit RUN_FAILED {aborted: False}; return self._result(...)
    state.usage += resp.usage
    state.add_message(resp.to_message())             # assistant 消息入 transcript（**唯一写入点**）
    await memory.aadd(resp.to_message())             # 也入短期窗口（assistant 侧不触发长期写入）

    # --- 2.5 token 预算检查（[v2 新增]）---
    if config.max_total_tokens is not None and state.usage.total_tokens > config.max_total_tokens:
        err = BudgetExceededError(limit=config.max_total_tokens, used=state.usage.total_tokens,
                                  kind="total_tokens")
        emit BUDGET_EXCEEDED {kind: "total_tokens", limit, used}
        state.mark_finished(FAILED, error=err); emit RUN_FAILED; return self._result(...)

    # --- 3. 决定本轮是"行动"还是"终结" ---
    calls = resp.tool_calls if mode == "native" else []
    native_no_calls = (mode == "native" and not calls)
    if native_no_calls:
        if resp.finish_reason == "length":
            -> §9.4.6 分支 (a) 截断续写
        elif resp.finish_reason == "content_filter":
            -> §9.4.6 分支 (b) 立即失败
        elif resp.finish_reason == "tool_calls":
            -> §9.4.6 分支 (c) 声明了工具但没给出 -> 自纠正
        else:                                        # finish_reason == "stop"
            -> 走 §9.4.3 的终结分支
    elif not calls:
        # 文本模式：解析
        try:
            parsed = parser.parse(resp.content)
        except ReActParseError as e:
            -> 走 §9.4.2 的 parse-error 自纠正分支，continue
        if parsed.is_action():                       # [v2] Action **优先**
            calls = [ToolCall.create(parsed.action, parsed.action_input,
                                     call_id=f"call_text_{state.step}")]
            emit ACTION_PARSED {action, arguments}
        elif parsed.is_final():
            -> 走 §9.4.3 的终结分支
        else:
            -> 走终结分支（保守）

    # --- 4. 重复/无进展动作检测（在真正执行之前，[v2 变更] 三层判定）---
    if config.repeat_action_policy != "off":
        for c in calls: state.record_tool_call(c)     # 同步累加 action_counts 与 tool_name_counts
        over = []
        for c in calls:
            key = c.canonical_key()
            hit_key    = state.action_counts[key] >= config.repeat_action_threshold
            hit_digest = any(v >= config.repeat_action_threshold
                             for v in state.observation_digests.values())   # 无进展：同一份观察反复出现
            hit_name   = (config.disable_tool_after_failures > 0
                          and state.tool_failure_counts.get(c.name, 0)
                              >= config.disable_tool_after_failures)
            if hit_key or hit_digest or hit_name: over.append(c)
        if over:
            emit REPEAT_DETECTED {action_key: over[0].canonical_key(),
                                  count: state.repeat_count(over[0])}
            if policy in ("nudge", "nudge_then_fail"):
                if 该 key 已经 nudge 过:
                    -> state.mark_finished(FAILED, RepeatedActionError(...)); emit RUN_FAILED; return self._result(...)
                else:
                    注入 nudge 消息（§9.4.4 文本 + 无进展时追加一句），
                    emit NUDGE {text}; state.nudges.append(text); continue
            if policy == "fail":
                -> 终结为 FAILED(RepeatedActionError)

    # --- 5. 执行工具（[v2] 不在这里 emit TOOL_STARTED/TOOL_FINISHED —— 归 executor）---
    state.status = ACTING
    if config.parallel_tool_calls and len(calls) > 1:
        results = await executor.execute_many(calls)          # 永不抛业务异常（§7.4.2）
    else:
        results = [await executor.execute(c) for c in calls]
    state.tool_calls.extend(calls); state.tool_results.extend(results)
    for r in results:
        if not r.ok: state.record_tool_failure(r.name)
        state.record_observation(r)                  # 记录 observation 摘要（无进展检测用）

    # --- 6. 把观察结果写回 ---
    state.status = OBSERVING
    if mode == "native":
        for r in results:
            m = r.to_message()                       # role=tool，content = r.error_text()
            state.add_message(m); await memory.aadd(m)
    else:
        obs = parser.build_observation(results,
                                       max_chars=min(config.max_observation_chars,
                                                     memory.tokenizer_chars_budget()),
                                       step=state.step)
        m = Message.observation(obs, step=state.step)   # role=USER, kind="observation"
        state.add_message(m); await memory.aadd(m)

    # --- 6.5 可选：把 thought 记进历史（文本模式）---
    if mode == "text" and config.include_thought_in_history and parsed.thought:
        state.add_message(Message.assistant(parser.strip_markers(parsed.thought), kind="react"))

    # --- 6.7 transcript 上限 ---
    if len(state.messages) > config.max_transcript_messages:
        state.trim_transcript(config.max_transcript_messages)

    # --- 7. 压缩检查（每轮**一次**，[v2 变更]）---
    await memory.acompress_if_needed()
    emit STEP_FINISHED {step, status}; state.status = THINKING

# while 结束：步数用尽
state.mark_finished(FAILED, error=MaxStepsExceededError(max_steps=config.max_steps))
emit RUN_FAILED
return self._result(state, output=last_assistant_text, status=FAILED, error=...)
```

**`[v2 变更]` 事件归属**：本伪代码里**没有** `TOOL_STARTED` / `TOOL_FINISHED` 的 emit
（v1 有，会与 executor 重复发送）。`LLM_REQUEST` / `LLM_RESPONSE` 也不在这里发
（由 `BaseLLMClient._emit` 发）。矩阵见 §2.7。

#### 9.4.2 parse-error 自纠正（冻结，`[v2 变更]` 修掉重复写 assistant 消息）

触发：文本模式下 `parser.parse` 抛 `ReActParseError`。

```text
state.parse_errors += 1；emit PARSE_ERROR {reason, offset, raw_len, attempt: state.parse_errors}
# ★ [v2 冻结] **不得**再追加 assistant 消息：
#   本条 assistant 原文**已由 §9.4.1 第 2 步写入**（唯一写入点）。
#   v1 在这里又 add 了一次，导致同一条内容在 messages 里出现两遍。
#   **可断言不变式**：一次 parse error 恰好新增 2 条消息
#   （1 条 assistant + 1 条 NUDGE），len(state.messages) 增量 == 2。
注入反馈消息：Message.user(parser.build_parse_error_feedback(err), kind="nudge")
             state.add_message(...) + await memory.aadd(...)
emit NUDGE
若 state.parse_errors > config.max_parse_retries → 终结为 FAILED(AgentError(...))，
  error=ReActParseError 作为 cause；否则 continue。
**消耗 step**（D-10）。**不重试 LLM 调用本身**（LLM 层已重试过）——
避免重试放大成 max_steps × max_parse_retries 次调用。
```

#### 9.4.3 终结分支（冻结，`[v2 变更]` 修掉 assistant 二次写入与 assistant 自动写长期）

```text
answer = parsed.final_answer if 文本模式 else resp.content
answer = parser.strip_markers(answer).strip()      # 去掉可能残留的 'Final Answer:' 前缀
if not answer: answer = resp.content.strip()        # 兜底：不返回空字符串
# ★ 不 add_message：assistant 的原始响应已在第 2 步加过（唯一写入点）
await memory.aadd(Message.assistant(answer))        # [v2] **不传 auto_write=True**
#   v1 写 auto_write=True 会让 assistant 的答案被当成"用户事实"写进长期库、
#   下一轮又被当事实召回。§8.5 的 should_auto_write 对 role != "user" 恒 False。
state.mark_finished(FINISHED)
emit RUN_FINISHED {output_len, steps, usage}
return self._result(state, output=answer, status=FINISHED)
```

#### 9.4.4 nudge 文本（冻结字面量）

```text
You already called {name} with these exact arguments {n} times.
The previous result was:
{last_result}
Do not repeat it. Either use a different tool/arguments, or give your final answer now
as "Thought: ...\nFinal Answer: ...".
```

`[v2 新增]` 无进展命中时，在末尾追加一句：
`Your last {n} calls returned identical results — the approach is not working.`

#### 9.4.5 两种模式如何统一（冻结决策 D-02）

| | native（原生 function calling） | text（文本 ReAct） |
|---|---|---|
| 工具暴露 | `tools=[schema...]` 参数 | `{tools}` 渲染进 system prompt |
| 模型输出 | `resp.tool_calls` | `Thought/Action/Action Input` 文本 |
| 解析 | 直接取结构化字段 | `ReActParser.parse` |
| 观察回灌 | `role="tool"` 消息 + `tool_call_id` | `role="user"` 的 `Observation:` 消息 |
| 并发 | 天然支持一轮多 tool_call | **一轮一个 action** |
| 结束 | `finish_reason=="stop"` 且无 tool_calls | `Final Answer:` |

**统一层**：两条路径都归约到同一个 `list[ToolCall]` → `executor.execute_many` →
`list[ToolResult]` → 写回消息。**状态机、事件、重复检测、自纠正、截断、usage 统计完全共用**，
差异只在这张表的 5 个点上，全部由 `mode` 分支隔离在 `_next_calls()` 与 `_write_back()` 两个
私有方法里。新增第三种模式（如 JSON-mode）只需再实现这两处。

`mode="auto"` 的解析规则（冻结，唯一出处是 `LLMClient.resolve_mode`）：
```python
mode = llm.resolve_mode(has_tools=len(self.tools) > 0)
# == "native" if (llm.supports_tool_calling and has_tools) else "text"
```

#### 9.4.6 finish_reason 参与控制流（`[v2 变更]`，D-14；v1 只把它塞进事件）

| 分支 | 触发条件 | 冻结行为 |
|---|---|---|
| (a) 截断续写 | `finish_reason == "length"` 且 `mode != "native"` | `state.truncation_errors += 1`；emit `LLM_ERROR {reason: "length", retries: n}`；注入 `Message.user("Your previous reply was truncated before it finished. Continue from where you stopped.", kind="nudge")`；`continue`（**消耗 step**）。连续超过 `config.max_truncation_retries` 次 -> `mark_finished(FAILED, AgentError("output truncated by max_tokens"))` |
| (b) 内容过滤 | `finish_reason == "content_filter"` | 立即 `mark_finished(FAILED, AgentError("response blocked by content filter"))`，并在 `AgentResult.metadata["finish_reason"]` 记录 |
| (c) 空 tool_calls | `finish_reason == "tool_calls"` 但 `len(calls) == 0` | 走 §9.4.2 的自纠正分支（反馈"你声明要调用工具但没有给出合法 tool_call"），**不**当成最终答案 |
| (d) 正常终结 | `finish_reason == "stop"` 且无 tool_calls | 走 §9.4.3 |

**原生模式的终结条件写死为：无 tool_calls 且 `finish_reason == "stop"`。**

#### 9.4.7 `arun` 的取消处理（`[v2 变更]`，冻结）

```python
try:
    <上面的 ReAct 循环>
except asyncio.CancelledError:
    state.mark_finished(AgentStatus.ABORTED)
    emit RUN_FAILED {aborted: True, error_type: "CancelledError"}
    raise                      # **必须继续抛出**（取消传播不能被打断）
finally:
    if state.finished_at is None:
        state.finished_at = utc_now()        # 保证 trace 不被截断（v1 的取消路径没有这一步）
```

`ABORTED` 路径**不构造 AgentResult**（因为必须继续抛出）。

#### 9.4.8 非原生模式的 `mode` 强制

`config.mode` 为 `"native"` 但 `llm.supports_tool_calling is False` 时 -> `ConfigError`
（显式失败好过静默降级）；`config.mode == "text"` 时永远走文本路径，即使 llm 支持 function calling。

#### 9.4.9 `arun` 内部的冻结局部变量与私有 helper（`[v2 新增]`，消除伪代码里的未定义名字）

`arun` 在进入 `while` 之前**必须**声明这几个局部变量（不要塞进 `AgentState`，
它们是纯运行时簿记，不该出现在 trace 里）：

```python
nudged_keys: set[str] = set()          # 已经被 nudge 过的 canonical_key（§9.4.1 步骤 4）
last_assistant_text: str = ""          # 最近一次 assistant 的可见文本（终局输出的兜底）
parsed: ParsedAction | None = None     # 本轮解析结果（仅文本模式；步骤 6.5 要用）
executor_ref: ToolExecutor = self.executor
```

以及一个私有 helper（**所有 `AgentResult` 构造都必须经过它**）：

```python
def _result(self, state: AgentState, *, output: str, status: AgentStatus,
            error: LiteAgentError | None = None,
            metadata: Mapping[str, Any] | None = None) -> AgentResult:
    """冻结实现（关键字参数，禁止位置参数）：
        md = dict(state.scratchpad.get("agent_metadata") or {})
        md.update(metadata or {})
        md.setdefault("cost_usd", estimate_cost_usd(state.usage, model=self.llm.model))
        return AgentResult(output=output, status=status, steps=state.step,
                           tool_calls=list(state.tool_calls),
                           tool_results=list(state.tool_results),
                           usage=state.usage, error=error, state=state,
                           duration_ms=state.duration_ms, agent_name=self.name,
                           metadata=md)
    """
```

**`raise_on_error=True` 的唯一触发点**：`_result` 返回前，若
`config.raise_on_error and status != AgentStatus.FINISHED`，则 `raise error or AgentError(...)`。
（`error is None` 时抛 `AgentError("agent failed without an error object")` —— v1 没写。）
放在 `_result` 里保证**只有一个出口**，不会漏。

---

## 10. `liteagent/multiagent/` —— 多 Agent 协作

### 10.1 `multiagent/base.py`

```python
class AgentLike(Protocol):
    """任何可被编排的东西（Agent 或另一个 MultiAgent）。"""
    @property
    def name(self) -> str: ...
    async def arun(self, input: str, **kwargs: Any) -> AgentResult: ...

@dataclass
class DelegationContext:
    """随 AgentState.scratchpad["delegation"] 传递的委派上下文。"""
    stack: list[str] = field(default_factory=list)      # 祖先 Agent 名字（含当前）
    depth: int = 0
    root_run_id: str = ""
    parent_run_id: str | None = None
    budget: int = DEFAULT_TEAM_MAX_ROUNDS                # 剩余可委派轮次

    def child(self, name: str) -> "DelegationContext":
        """返回新的上下文，stack 追加 name、depth+1、budget-1。**不修改自身**。"""
        ...
    def would_cycle(self, name: str) -> bool: ...        # name in self.stack
    def exhausted(self) -> bool:
        """[v2 新增] `self.budget <= 0`。**必须被真正判定**（v1 递减后无人读，
        budget 是装饰品）。判定点见 §10.4。"""
        ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_state(cls, state: AgentState, *, name: str) -> "DelegationContext":
        """从 state.scratchpad 取；不存在则新建（stack=[name], depth=0）。"""
        ...

def compress_subagent_output(name: str, output: str, *, status: str = "FINISHED",
                            steps: int = 0,
                            max_chars: int = DEFAULT_SUBAGENT_MAX_CHARS) -> str:
    """把子 Agent 的输出压缩成回传给父 Agent 的字符串（冻结格式）：
       '[worker {name} | status={status} | steps={steps}]\\n{body}'
       body = output 原样（len <= max_chars 时）
       body = head 70% + '\\n...[truncated {k} chars]...\\n' + tail 30%
       max_chars <= 0 时不做 body 压缩（仍加 header）。header 本身不计入 max_chars。
       用 truncate_head_tail(output, max_chars, head_ratio=SUBAGENT_HEAD_RATIO)。"""
    ...

class MultiAgent(ABC):
    """多 Agent 编排器的公共基类。"""
    def __init__(self, agents: Sequence[AgentLike], *, name: str = "team",
                 config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: "Blackboard | None" = None) -> ...

    agents: list[AgentLike]
    config: TeamConfig
    blackboard: Blackboard
    callbacks: CallbackManager

    @property
    def name(self) -> str: ...

    @abstractmethod
    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult: ...

    def run(self, input: str, **kwargs: Any) -> AgentResult: ...      # run_sync 包装

    def _check_depth(self, context: DelegationContext, *, to_agent: str) -> None:
        """depth > max_depth -> MaxDepthExceededError；
        [v2 变更] `context.exhausted()` -> MaxDepthExceededError（**让 max_rounds 真正生效**）；
        context.would_cycle(to_agent) 且 enable_cycle_detection -> CycleDetectedError。"""
        ...
    def _aggregate_usage(self, results: Sequence[AgentResult]) -> TokenUsage: ...
    def _failed(self, error: LiteAgentError, *, agent_name: str,
                metadata: Mapping[str, Any] | None = None) -> AgentResult: ...
    def _apply_shared_memory(self, agent: Any) -> Any:
        """[v2 新增] `config.share_memory=True` 时把 `self` 持有的 shared memory
        注入子 Agent（`agent.memory = self._shared_memory`），False 时**不动**。
        子类在每次 arun 子 Agent 之前调用它。判定用例见 §12。"""
        ...
    def describe(self) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...                              # 逐层 aclose 子 Agent

def build_team(agents: Sequence[AgentLike], *, mode: str = "sequential",
               config: TeamConfig | None = None, blackboard: "Blackboard | None" = None,
               manager: Agent | None = None, name: str = "team") -> MultiAgent:
    """mode: "sequential" | "hierarchical"（函数内 import 两个子模块，避免顶层循环依赖）。
    hierarchical 且 manager is None -> ConfigError。"""
    ...
```

### 10.2 `multiagent/blackboard.py` —— 共享状态

```python
@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str = ""
    version: int = 1
    created_at: float = field(default_factory=utc_now)     # [v2] utc_now
    updated_at: float = field(default_factory=utc_now)     # [v2]
    tags: tuple[str, ...] = ()
    expires_at: float | None = None
    def is_expired(self, *, now: float | None = None) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlackboardEntry": ...

class Blackboard:
    """跨 Agent 共享的键值黑板。**并发安全**（见 10.2.1）。"""

    def __init__(self, *, max_entries: int = 1000,
                 on_event: LowLevelEvent | None = None) -> ...
        """[v2 新增] 冻结创建的实例状态：
            self._lock = threading.RLock()
            self._pools = LoopBoundPool()                       # watch 用
            self._watch_loops: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Condition]]] = {}
            self._watch_locks: dict[str, threading.Lock] = {}   # 保护 _watch_loops 的注册/注销
            self._write_seq: dict[str, int] = {}                # [v3 新增] key -> 单调写入序号
              # 只增不减，**不受**条目生命周期影响（delete/过期/clear/max_entries 淘汰都不重置）。
              # `awatch` 的唯一判脏依据（`_write_seq_of(key)`），见 §10.2.1。
        """

    # ---- 同步 API（规范实现，全部在 threading.RLock 内完成）----
    def write(self, key: str, value: Any, *, author: str = "",
              tags: Sequence[str] = (), ttl_s: float | None = None,
              if_version: int | None = None) -> BlackboardEntry:
        """- key 非空字符串，否则 ConfigError
           - 已存在 -> version+1、updated_at 刷新、created_at 保留、author 覆盖
           - 不存在 -> version=1
           - if_version 非 None 且与当前版本不符 -> VersionConflictError(key, expected, actual)
           - 触发 subscribe 回调与 on_event("blackboard_write", {...})（**在锁外**调用回调）
           - 超过 max_entries -> 淘汰最旧的（按 updated_at），记 WARNING
           - **[v2 变更] 唤醒 watcher 的顺序（冻结，见 §10.2.1）**：
             在 RLock 内取 `targets = self._snapshot_watchers(key)`（list 快照），
             **出锁之后**对每个 (loop, cond) 做 `if loop.is_closed(): continue` +
             try/except RuntimeError 的 `loop.call_soon_threadsafe(<在目标 loop 内创建唤醒任务>)`。
             `[v3 变更]` 转发载荷**不是** `cond.notify_all`（`asyncio.Condition.notify_all()`
             要求持锁，3.10 实测会抛 RuntimeError 且唤醒不了，详见 §10.2.1 第 2 步）。
             绝不直接从 worker 线程调 `cond.notify()`（未定义行为）。"""
        ...
    def read(self, key: str, *, default: Any = None) -> Any: ...
    def read_entry(self, key: str) -> BlackboardEntry | None:
        """过期条目按不存在处理（惰性过期）。"""
        ...
    def delete(self, key: str, *, author: str = "") -> bool: ...
    def keys(self, *, prefix: str = "", tags: Sequence[str] = ()) -> list[str]: ...
    def list(self, *, prefix: str = "", tags: Sequence[str] = (),
             include_expired: bool = False) -> list[BlackboardEntry]: ...   # 按 updated_at 升序
    def snapshot(self) -> dict[str, Any]: ...
    def update(self, key: str, patch: Mapping[str, Any], *, author: str = "") -> BlackboardEntry:
        """value 必须是 dict；浅合并。否则 -> **ConfigError**。"""
        ...
    def increment(self, key: str, *, amount: float = 1, author: str = "") -> float:
        """原子自增（不存在的 key 从 0 开始）；value 非数值 -> ConfigError。"""
        ...
    def clear(self) -> None: ...
    def subscribe(self, fn: Callable[[str, BlackboardEntry], None]) -> Callable[[], None]: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...

    # ---- 异步 API（薄包装 + watch）----
    async def awrite(self, *args: Any, **kwargs: Any) -> BlackboardEntry: ...
    async def aread(self, key: str, *, default: Any = None) -> Any: ...
    async def aread_entry(self, key: str) -> BlackboardEntry | None: ...
    async def adelete(self, key: str, *, author: str = "") -> bool: ...
    async def alist(self, **kwargs: Any) -> list[BlackboardEntry]: ...
    async def asnapshot(self) -> dict[str, Any]: ...
    async def aclear(self) -> None: ...
    async def awatch(self, key: str, *, timeout_s: float | None = None) -> AsyncIterator[BlackboardEntry]:
        """每次该 key 被写入时 yield **最新的** entry。
        **[v2 变更] 语义冻结为"合并语义"**：多次写入快于消费时只 yield 最新一条，
        中间版本可通过 `history()` 查询（避免实现者做成逐版本队列导致内存无界）。
        `timeout_s` 的语义（冻结）：从进入 async generator 起的**总超时**，
        到点抛 `asyncio.TimeoutError`（消费者在 `async for` 处收到），yield 后**不重新计时**。
        `timeout_s=None` -> 永不超时。
        实现见 §10.2.1（loop-bound Condition + **单调写入序号**判脏，**不用轮询**；
        `[v3 变更]` 判脏量是 `_write_seq_of(key)` 而非会回退的 `entry.version`，
        且谓词在持 cond 锁的临界区内、`wait()` 之前判定 —— 理由与回归测试见 §10.2.1）。"""
        ...

    # ---- 变更日志（给 trace / 调试用）----
    def history(self, *, limit: int = 100) -> list[BlackboardEntry]: ...
```

#### 10.2.1 并发安全策略（冻结，理由见 `DESIGN_DECISIONS.md` D-12；`[v2 变更]` 补齐三处漏洞）

- **唯一锁**：`self._lock = threading.RLock()`，在 `__init__` 里创建（`threading` 锁与事件循环无关，
  因此**不受 §0.4 的 R-LOOP 陷阱影响**）。
- 所有同步方法在临界区内**不做任何 await、不做 I/O**，只操作内存 dict。
- 迭代安全：`list()` / `keys()` / `snapshot()` 一律返回**快照副本**，不返回内部 dict 引用。
- `value` 不做深拷贝（性能）；文档写明**写入方不应在写入后再改可变对象**。

**[v2 变更] 三处细节**（`[v3 变更]` 按实现修正第 1/2/3 步，理由与测试见各步末尾）：

1. **watcher 的注册（谁拿得到 loop）**：写入方可能在工作线程（同步工具里写黑板），
   而工作线程**没有运行中的 loop**，按 §5.4 的约束它调用 `pool.condition(key)` 会直接 `ConfigError`。
   因此冻结为：**由 `awatch` 自己在运行中的 loop 内注册**（`[v3 变更]` 全部按实现重写）：

   ```python
   # awatch 内部
   cond = self._pools.condition("bbwatch")          # 必须在运行中的 loop 内调用
   loop = asyncio.get_running_loop()
   # [v3 变更] **先登记 watcher、后读基线**（顺序不可颠倒）：登记之后的写入一定在 watcher
   #   快照里（会被转发唤醒），登记之前的写入会被紧随其后的基线读取吸收 —— 两种落在
   #   "读基线 / 登记"之间的写入都不会漏。顺序反了会制造一段"既不转发通知、又不在基线里"
   #   的空洞，写入永久丢失。
   guard = self._watch_locks.setdefault(key, threading.Lock())   # setdefault 原子，同 key 只一个锁对象
   with guard:
       self._watch_loops.setdefault(key, []).append((loop, cond))
   deadline = None if timeout_s is None else (loop.time() + timeout_s)
   try:
       # [v3 变更] 判脏基线是**单调写入序号**，不是 `entry.version`：
       #   后者在 TTL 过期 / delete / clear / max_entries 淘汰后会回退（同 key 重写时
       #   version 从 1 重新开始），`version != last` 可能恒假 -> 写入发生了却一次都不 yield。
       last = self._write_seq_of(key)
       while True:
           async with cond:
               remaining = None if deadline is None else max(0.0, deadline - loop.time())
               if deadline is not None and remaining == 0.0:
                   raise asyncio.TimeoutError()
               # [v3 变更] **持 cond 锁判定谓词**（标准 monitor 写法），且放在 wait() **之前**：
               #   写侧的唤醒（`_wake_watchers` 的 `async with cond: notify_all()`）必须先拿到
               #   同一把 cond 锁，因此"判脏"与"登记为等待者"互斥 —— 要么在 wait() 之前就看到
               #   新序号（本次不停车，出锁直接 yield），要么本次 wait() 一定在 notify_all 之前
               #   进入、必然被唤醒。只在仍干净时才停车，关掉"消费者处理上一条 entry 期间发生的
               #   写入，其 notify_all 无人可唤醒"的丢唤醒窗口。
               if self._write_seq_of(key) == last:
                   try:
                       await asyncio.wait_for(cond.wait(), remaining)
                   except asyncio.TimeoutError:
                       raise           # 显式重抛（不吞）：总超时是契约的一部分
           # 出锁后再读一次（yield 的用户代码绝不能跑在持锁状态下）
           seq = self._write_seq_of(key)
           if seq != last:
               entry = self.read_entry(key)
               if entry is not None:
                   yield entry                       # 合并语义：只 yield 最新
               last = seq
   finally:
       with self._watch_locks.get(key, self._lock):   # 注销必须在 finally：break/取消/超时都要摘
           watchers = self._watch_loops.get(key)
           if watchers is not None:
               try:
                   watchers.remove((loop, cond))
               except ValueError:
                   logger.debug("blackboard watcher already deregistered key=%s", key)
   ```

   `[v3 变更]` 理由：v2 冻结的"`last = 进入时的版本号`（即 `entry.version`）、`cond.wait()`
   返回之后再比对 `v = self._version_of(key)`"有两个洞（正是对抗性审计的 critical finding）：
   (a) 唤醒是**提示式**的（`notify_all` 在没有等待者时是空操作），一次写入若落在"消费者正挂在
   yield 上处理上一条 entry"到"重新进入 `wait()`"之间，那次唤醒被丢弃、版本比对再也不会执行，
   消费者永久停在旧值；(b) `entry.version` 会回退（过期/删除/淘汰后同 key 重写回到 1），
   `v != last` 恒假 —— 写入发生了却一次都不 yield（**静默失联**）。
   实现改成「持锁判谓词 + 单调写入序号 `_write_seq_of(key)`」。
   回归测试：`tests/test_blackboard.py::WatchTests::test_awatch_registers_watcher_before_reading_baseline`、
   `::test_awatch_delivers_write_that_lands_while_consumer_handles_previous`、
   `::test_awatch_delivers_rewrite_after_ttl_expiry`、
   `::test_awatch_delivers_rewrite_after_delete_and_clear`。

2. **唤醒（写入方不做任何 asyncio 调用）**：写入方在 RLock 内取
   `targets = list(self._watch_loops.get(key, ()))` 快照，**出锁后**：

   ```python
   for loop, cond in targets:
       if loop.is_closed():                       # 实测 M-4
           continue
       try:
           # [v3 变更] 载荷是"在目标 loop 内创建一次唤醒任务"，**不是** `cond.notify_all`：
           #   `asyncio.Condition.notify_all()` 要求调用方**持锁**（3.10 起是硬要求），
           #   本机 3.10.12 实测对未持锁的调用抛 `RuntimeError: cannot notify on
           #   un-acquired lock`，且唤醒不了任何 watcher（`wait_for(cond.wait())` 期间锁是
           #   释放的）。唤醒任务在 loop 线程内先 `async with cond` 再 `notify_all`，满足
           #   持锁要求且实测能唤醒。可观测形态（`is_closed()` 判定 / `try-except
           #   RuntimeError` / 日志文案）与冻结原文逐字一致。
           loop.call_soon_threadsafe(_schedule_wake, loop, cond, key)
       except RuntimeError:                       # 已关闭的 loop
           logging.getLogger("liteagent.multiagent").info(
               "blackboard watcher loop is closed; skip notify key=%s", key)
   ```

   必须**吞掉已关闭 loop 的 RuntimeError 但记日志**（否则一次同步 write 就能把异常抛给调用方，
   直接违反 §13 红线 6/10）。
   回归测试：`tests/test_blackboard.py::ClosedLoopTests::test_write_from_worker_thread_after_watcher_loop_closed`、
   `::test_write_swallows_runtime_error_and_logs`、`::test_dead_loop_does_not_block_other_watchers`。

3. **判脏量是单调写入序号，且在持锁临界区内比对**（`[v3 变更]`，取代 v2 的
   "唤醒后比对 `entry.version`"）：`awatch` 的循环写成
   `while: async with cond: if self._write_seq_of(key) == last: await cond.wait();
   seq = self._write_seq_of(key); if seq != last: yield entry; last = seq`。
   两点缺一不可：(a) 谓词判定必须在 `async with cond` **内**、`wait()` **之前**，否则唤醒丢失；
   (b) 判脏量必须是**永不回退**的写入序号，否则条目过期/删除后版本号撞回旧值、写入被静默吞掉。
   用 `entry.version` 时，`每次写入 yield 一次` 的语义无法保证（N 次写可能只 yield 1 次，
   或被旧版本唤醒而 yield 同一条 entry）。

### 10.3 `multiagent/sequential.py`

```python
@dataclass
class SequentialStep:
    agent: AgentLike
    name: str | None = None                       # 默认 agent.name
    input_template: str | None = None             # None -> 第 0 步用 "{input}"，其余 "{prev}"
    output_key: str | None = None                 # 非 None 时写入 blackboard（默认 name）
    optional: bool = False                        # True 时失败不中止
    max_chars: int = 0                            # >0 时对输出截断后再传给下一步

class SequentialAgent(MultiAgent):
    """顺序编排：A -> B -> C，每一步的输入由模板从上一步输出构造。"""

    def __init__(self, steps: Sequence[SequentialStep | AgentLike], *,
                 name: str = "sequential", config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: Blackboard | None = None) -> ...

    steps: list[SequentialStep]

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult: ...
```

**执行语义（冻结）**：

1. `context = context or DelegationContext(stack=[self.name], root_run_id=...)`。
2. 对每个 step（索引 i）：
   a. `self._check_depth(context, to_agent=step.name)` → 可能抛 `MaxDepthExceededError`/`CycleDetectedError`。
   b. 构造输入：
      - `template = step.input_template or ("{input}" if i == 0 else "{prev}")`
      - `values = {"input": <原始用户输入>, "prev": <上一步输出>, "steps": <dict[name -> output]>}`
      - `stage_input = render_template(template, values)`
      - 模板渲染其它异常 → `ConfigError`。
   c. emit `AGENT_DELEGATE {from, to, step: i, depth: context.depth+1, refused: False}`。
   d. 子 Agent 的 `state`：**每次新建**（`AgentState.create(stage_input, agent_name=step.name)`），
      并把 `state.scratchpad["delegation"] = context.child(step.name)` 注入；
      **不复用父 state**；`blackboard` 共享。
   e. `[v2 冻结] self._apply_shared_memory(step.agent)`（`share_memory=True` 时注入同一 MemoryManager）。
      `result = await step.agent.arun(stage_input, state=child_state)`；**仅当
      `isinstance(step.agent, MultiAgent)` 时额外传 `context=`**（普通 `Agent.arun` 不认识
      `context` 参数；它通过 `state.scratchpad` 拿到上下文）。
   f. emit `AGENT_RETURN {from, to, status, steps, output_len, duration_ms, failed}`。
   g. `output = truncate_head_tail(result.output, step.max_chars)`（`max_chars>0` 时）。
   h. 写黑板：`blackboard.write(step.output_key or step.name, output, author=step.name,
      tags=("stage",))`；`stage_outputs[step.name] = output`。
   i. 失败处理（`result.status != FINISHED` 且 `not step.optional`）：
      - `propagate_failure == "raise"` → 抛 `DelegationError(from_agent=self.name, to_agent=step.name)`
        （`cause=result.error`）。
      - `"return"` → **立即停止**，返回
        `AgentResult(output=<见下冻结公式>, status=result.status, error=result.error,
         metadata={"failed_stage": step.name, "steps": [...已完成的阶段摘要...]})`。
        **[v2 变更] output 的唯一公式（v1 在同一条里写了两个互斥的值）**：
        ```python
        output = result.output or previous_output or ""
        # 优先"失败阶段的部分输出"；为空才回退"上一步输出"；最后兜底空串
        ```
      - `"continue"` → `output = f"[stage {step.name} failed: {result.error}]"`，继续下一步。
      - `step.optional=True` 时**无视** `propagate_failure`，一律 continue 语义。
3. 全部成功 → 返回最后一步的 `AgentResult`，但 **output 用最后一步的 output**，
   `usage` 为**所有阶段之和**（`_aggregate_usage`），`steps` 为所有阶段 `steps` 之和，
   `metadata["steps"] = [{"name","status","steps","output","duration_ms"}...]`，
   `metadata["blackboard"] = blackboard.snapshot()`，`agent_name = self.name`。
4. `result.state` 指向最后一个阶段的 child state（便于调试）。

### 10.4 `multiagent/hierarchical.py`

```python
@dataclass
class SubTask:
    id: str
    description: str
    assignee: str = ""                 # worker 名字；空表示由 manager 自行决定
    depends_on: tuple[str, ...] = ()
    status: str = "pending"            # "pending" | "running" | "done" | "failed"
    result: str = ""
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubTask": ...

@dataclass
class Plan:
    goal: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Plan": ...
    @classmethod
    def from_json(cls, text: str) -> "Plan":
        """容错解析（冻结步骤）：
        1. 剥掉 ```json / ``` 围栏与首尾空白
        2. 取第一个 '{' 到最后一个 '}' 的子串（容忍模型加了解释文字）
        3. json.loads；失败 -> 抛 ReActParseError(raw=text, reason="plan is not valid JSON")
        4. 接受 {"goal","subtasks":[...]} 或直接是一个 list（当作 subtasks）
        5. subtask 缺 id -> 用 "task_{index}"；缺 description -> 跳过该条
        6. 不做 schema 校验以外的事；assignee 不在 workers 里由调用方处理。
        """
        ...
    def render(self) -> str:
        """'- [id] description (assignee) <- depends_on' 多行"""
        ...

DECOMPOSE_PROMPT_TEMPLATE: str        # 冻结字面量，见下
PLAN_JSON_SCHEMA: dict[str, Any]      # 用于校验 Plan 的 JSON Schema（自检用）

class HierarchicalAgent(MultiAgent):
    """层级编排：manager（普通 Agent）+ workers（被包装成 delegate_to_<name> 工具）。"""

    def __init__(self, manager: Agent, workers: Sequence[AgentLike], *,
                 name: str = "hierarchy", config: TeamConfig | None = None,
                 callbacks: Sequence[CallbackLike] | None = None,
                 blackboard: Blackboard | None = None,
                 worker_descriptions: Mapping[str, str] | None = None) -> ...
        """[v2 新增] 冻结创建的实例状态（**全部是 threading 原语**，理由见下）：
            self._delegation_seq: int = 0
            self._delegation_lock = threading.Lock()
            self._subagent_sem = threading.Semaphore(config.subagent_concurrency)
            self._serial_lock   = threading.Lock()     # parallel_subagents=False 时用
        """

    manager: Agent
    workers: list[AgentLike]
    def worker_names(self) -> list[str]: ...
    def delegate_tools(self) -> list[Tool]:
        """为每个 worker 生成一个 Tool（`make_function_tool`，pass_style="mapping"）：
           name        = f"delegate_to_{worker.name}"（名字里的非法字符替换为 '_'；
                         **冲突时报 ToolDefinitionError**，例如 'a-b' 与 'a_b'）
           description = worker_descriptions.get(name) or getattr(worker, 'description', '')
                         or f"Delegate a subtask to the '{name}' agent."
           parameters  = {"type":"object",
                          "properties":{"task":{"type":"string","description":...},
                                        "extra_context":{"type":"string","description":...}},
                          "required":["task"], "additionalProperties": False}
           func        = 闭包，签名 func(args: dict) -> str（同步包装 async，用 run_sync）
           idempotent  = False（委派有副作用）
           timeout_s   = NO_TIMEOUT          # [v2 变更] 委派不受 30s 默认工具超时约束
        **[v2 变更] 参数名从 `context` 改为 `extra_context`**：
        v1 里工具参数 `context`（一个字符串）与闭包从 contextvar 取的 `DelegationContext`
        同名，实现者极可能写成 `ctx = args.get("context") or _CURRENT_CTX.get()`，
        把模型传的文本当委派上下文用 —— 环检测与深度判定直接失效（安全后果：无限委派）。
        **[v2 冻结] `DelegationContext` 的唯一来源是 contextvar `_CURRENT_CTX`；
        `args['extra_context']` 只作为文本拼进 worker 的 task 描述，绝不参与环检测/深度判定。**
        闭包内部（冻结）：
           1. `ctx = _CURRENT_CTX.get()`；为 None 时（工具被脱离 Hierarchical 使用）
              构造一个只含自己的 context。
           2. **预算与并发**（[v2 变更] 全部用 threading 原语）：
              - `if ctx.exhausted(): return '[delegation refused: budget exhausted]'`
              - 环检测：`ctx.would_cycle(worker.name)` 或 `ctx.depth >= config.max_depth`
                -> **返回字符串**（不抛异常）：
                   '[delegation refused: cycle detected: stack=[a,b] -> c]'
                并 emit AGENT_DELEGATE {refused: True, refused_reason: "cycle"|"budget"|"depth"}
              - `parallel_subagents=False` 时 `with self._serial_lock:` 包住第 3..7 步（**真串行**）。
              - `parallel_subagents=True` 时先 `acquired = self._subagent_sem.acquire(timeout=<等待秒数>)`；
                超时（默认 `DEFAULT_TOOL_TIMEOUT_S`）-> 返回
                `'[delegation refused: all {n} subagent slots are busy]'`
                并 emit AGENT_DELEGATE {refused: True, refused_reason: "busy"}；成功则 finally release。
              **为什么必须是 threading.Semaphore**：delegate 工具是**同步工具**，
              会被 manager 的 executor 用线程池跑在 worker 线程里，闭包里再
              `run_sync(worker.arun(...))` -> 每次都新建一个 loop。
              loop-bound 的 asyncio.Semaphore 在每个 delegate 里都是全新的、计数永远是满的，
              并发完全不受限（这是 v1 的缺陷）。
           3. child_state = AgentState.create(task + (extra_context 拼接), agent_name=worker.name)
              child_state.scratchpad['delegation'] = ctx.child(worker.name)
              `self._apply_shared_memory(worker)`（share_memory=True 时注入 manager 的 memory）
           4. result = run_sync(lambda: worker.arun(task, state=child_state))
           5. 结果字符串 = compress_subagent_output(worker.name, result.output,
                                 status=result.status.value, steps=result.steps,
                                 max_chars=config.subagent_output_max_chars)
              压缩开关 config.compress_subagent_output=False 时不压缩
           6. 归档（[v2 变更] 键与计数器冻结）：
              with self._delegation_lock: self._delegation_seq += 1; seq = self._delegation_seq
              blackboard.write(f"subagent:{worker.name}:{seq}", 压缩后的字符串,
                               author=worker.name, tags=("subagent",))
              child_state.scratchpad['subagent_results'][f"{worker.name}:{seq}"] = result.output
              （**删掉 v1 里不存在的 `subtask_id`**）
           7. emit AGENT_RETURN {from: self.name, to: worker.name, status, steps, output_len,
                                duration_ms, failed: result.status != FINISHED}
           8. **失败也返回字符串**（前缀 '[worker ... | status=FAILED | error=...]'），
              不抛 -> manager 能看到失败并重试/改派。
              额外 emit `AGENT_RETURN {failed: True}`（[v2 可观测性要求]）。
        """
        ...

    async def adecompose(self, task: str, *, max_subtasks: int = 8) -> Plan:
        """用一次 LLM 调用做显式任务分解。
        失败（LLM 错误 / JSON 解析失败）-> 抛 ReActParseError 或原 LLMError，**不静默**。"""
        ...

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None,
                   plan: Plan | None = None, **kwargs: Any) -> AgentResult:
        """冻结语义（[v2 变更] 新增 `plan` 参数，把分解接到执行上）：
        1. ctx = context or DelegationContext(stack=[self.name], root_run_id=...)
        2. 把 delegate_tools() 合并进 manager 的工具集：
           - 若 manager 已有同名的 delegate_to_* 工具，**报 ToolDefinitionError**
           - 合并方式：`manager.tools = manager.tools.merge(ToolRegistry(self.delegate_tools()))`
             （merge 返回**新** registry；重名报错）
        3. 用 contextvar 保存 ctx：`_CURRENT_CTX.set(ctx)`，结束 reset（try/finally）。
           用 contextvar 而不是实例属性：manager 可能被并发复用。
        4. `plan is not None` -> 委托 `await self.arun_plan(plan, context=ctx)` 并返回其结果。
        5. state = state or AgentState.create(input, agent_name=self.name)
           state.scratchpad['delegation'] = ctx
        6. result = await self.manager.arun(input, state=state)
        7. 汇总：result.metadata['delegations'] = <从黑板收集的委派记录>
                  result.agent_name = self.name
        8. manager 失败 -> 直接上抛该 result（不回退为别的策略）。
        """
        ...

    async def arun_plan(self, plan: Plan, *, context: DelegationContext) -> AgentResult:
        """[v2 新增] **按依赖分层执行一个显式 Plan**（否则 `Plan`/`depends_on` 是死代码）。

        冻结算法：
        1. `layers = self._layers(plan)`：拓扑分层，`depends_on` 全部在更早层的任务进入本层；
           `depends_on` 里出现不存在的 id -> `DelegationError`；
           存在环 -> `CycleDetectedError(stack=[ids])`。
        2. 逐层执行；层内：
           - `config.parallel_subagents=True` 时用
             `await asyncio.gather(*[_one(st) for st in layer])`，
             并发上限用**本 loop 的** asyncio.Semaphore：
             `sem = LoopBoundPool().semaphore("plan", config.subagent_concurrency)`
             （此处**可以**用 asyncio 原语：`arun_plan` 本身就在父 loop 里，没有跨线程问题）。
           - `parallel_subagents=False` 时严格串行（for 循环 await）。
        3. 每个 subtask：
           - `assignee` 为空或不在 `self.workers` 里 -> 回退给 manager：构造一个最小输入
             `f"[subtask {id}] {description}"` 调 `self.manager.arun(...)`，并记 WARNING。
           - 否则调对应 worker 的 `arun`（`_apply_shared_memory` 后）。
           - 完成后写黑板 `blackboard.write(f"subtask:{st.id}", output, author=worker_name,
             tags=("subtask",))` 并回填 `st.status`/`st.result`。
           - **单个 subtask 失败不中断整体**：把失败压缩成字符串写进 `st.result`，
             `st.status = "failed"`，继续后续层。
        4. 返回：`AgentResult(status=FINISHED, output=<Plan.render() 的完成态摘要>,
            metadata={"plan": plan.to_dict(), "steps": [...]}, state=...)`。
           **注意**：`arun_plan` 的执行确定性由分层的先后决定；
           同层内的完成顺序不做保证（**测试不要断言同层内的完成顺序**）。
        """
        ...

    def _layers(self, plan: Plan) -> list[list[SubTask]]: ...
```

`DECOMPOSE_PROMPT_TEMPLATE`（冻结字面量）：

```text
Break the following task into at most {max_subtasks} concrete subtasks.

Available workers:
{workers}

Rules:
- Each subtask must be independently executable by exactly one worker.
- Use the worker names exactly as listed.
- depends_on lists ids of subtasks that must finish first; use [] when independent.
- Reply with ONLY a JSON object, no prose, no markdown fences:
{{"goal": "...", "subtasks": [{{"id": "t1", "description": "...", "assignee": "<worker name>", "depends_on": []}}]}}

Task:
{task}
```

**子 Agent 结果如何回传与压缩（冻结）**：
1. 原始输出 + status + steps 全部写入 `child_state` 与黑板（**不丢信息**）。
2. 回传给 manager 的是**压缩版字符串**（head 70% + tail 30%，带 worker 状态头）。
3. manager 后续如需细节可主动读黑板。**冻结：不默认注册 `read_blackboard` 工具**
   （避免工具集污染）；需要时由用户 `register_all` 后手动加。
4. 归档键格式 `subagent:{worker_name}:{seq}`，`seq` 从 1 递增（**计数器受
   `self._delegation_lock` 保护**，因为闭包可能被并发触发）。

**`TeamConfig` 四个"看起来有、实际没接上"的字段（`[v2 变更]` 逐条写死落点）**：

| 字段 | 落点（写进伪代码，不能只留在 dataclass） |
|---|---|
| `subagent_concurrency` | `HierarchicalAgent.__init__` 的 `threading.Semaphore`（delegate 路径）+ `arun_plan` 的 per-loop `asyncio.Semaphore`（plan 路径） |
| `parallel_subagents` | `False` 时 delegate 闭包用 `self._serial_lock` 串行；`arun_plan` 层内也串行 |
| `share_memory` | `True` 时 `MultiAgent._apply_shared_memory` 把同一个 `MemoryManager` 注入每个 worker（`worker.memory is manager.memory` 为测试断言）；`False` 时每个子 Agent 用 `MemoryManager.from_config()` 新建 |
| `max_rounds` | `DelegationContext.budget` 的默认值来源；`DelegationContext.exhausted()` 在 delegate 闭包第 2 步与 `_check_depth` 里被**判定**，触发 `MaxDepthExceededError` 或 refused 字符串 |

**最大深度/环检测的判定点（冻结）**：只在 `delegate_to_*` 工具闭包**内部**判定，
`HierarchicalAgent.arun` 自身在入口也调用一次 `_check_depth`（manager 可能直接调 worker）。

**失败传播（冻结）**：`propagate_failure` 对 Hierarchical 的语义是
——manager 的**整体**失败（`status != FINISHED`）时：`"raise"` → 抛 `DelegationError`；
`"return"`/`"continue"` → 返回 manager 的 result。
单个 worker 的失败**从不**中止 manager（它是工具失败，走观察回灌）。

### 10.5 `multiagent/__init__.py` 导出

```python
__all__ = [
    "AgentLike", "MultiAgent", "TeamConfig", "DelegationContext",
    "Blackboard", "BlackboardEntry", "SequentialAgent", "SequentialStep",
    "HierarchicalAgent", "Plan", "SubTask", "build_team", "compress_subagent_output",
]
```

---

## 11. `liteagent/cli.py`

**技术选型（冻结决策 D-11）**：用 **stdlib `argparse`**，不用 `typer`。`rich` 只在**渲染**处可选使用。

```python
PROG: str = "liteagent"
EXIT_OK: int = 0
EXIT_AGENT_FAILED: int = 1
EXIT_USAGE: int = 2
EXIT_PROVIDER_ERROR: int = 3

def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int:
    """返回退出码（不调用 sys.exit），便于测试直接断言返回值。
    捕获：ConfigError -> EXIT_USAGE；LLMError -> EXIT_PROVIDER_ERROR；
    AgentResult not ok -> EXIT_AGENT_FAILED；KeyboardInterrupt -> EXIT_AGENT_FAILED。
    **保证不向 stdout 打印 traceback**（-v 时打印到 stderr）。"""
    ...

# ---- 子命令实现（全部签名冻结，便于单测直接调用）----
def cmd_run(args: argparse.Namespace) -> int: ...
def cmd_tools(args: argparse.Namespace) -> int: ...
def cmd_chat(args: argparse.Namespace) -> int: ...
def cmd_trace(args: argparse.Namespace) -> int: ...
def cmd_schema(args: argparse.Namespace) -> int: ...
def cmd_multi(args: argparse.Namespace) -> int: ...
def cmd_version(args: argparse.Namespace) -> int: ...

# ---- 可复用的装配函数（CLI 与 examples 共用，也必须被测试覆盖）----
def build_agent_from_args(args: argparse.Namespace) -> Agent: ...
def build_registry_from_args(args: argparse.Namespace) -> ToolRegistry: ...
def build_memory_from_args(args: argparse.Namespace) -> MemoryManager: ...
def render_result(result: AgentResult, *, as_json: bool = False) -> str: ...
```

**子命令接口（冻结）**：

```text
liteagent [-h] [--version] <command> ...

liteagent run [-p PROMPT | -f FILE | --stdin]
              [--provider PROVIDER] [--model MODEL] [--base-url URL] [--api-key KEY]
              [--tools NAME[,NAME...]] [--no-tools] [--no-builtin]
              [--sandbox-root DIR] [--allow-shell] [--no-network]
              [--max-steps N] [--mode {auto,native,text}] [--temperature T] [--max-tokens N]
              [--max-total-tokens N] [--max-wall-clock S]
              [--approve {never,write,all}]
              [--no-memory] [--no-long-term] [--memory-max-tokens N]
              [--trace FILE] [--json] [--config FILE] [--dotenv FILE]
              [-v | -q]

liteagent tools list [--format {table,json}] [--tags T[,T...]] [--include-dangerous]
liteagent tools show NAME [--format {markdown,json}] [--show-schema]
liteagent tools schema [--format {openai,anthropic}] [-o FILE]

liteagent chat [--provider ...] [--tools ...] [--trace FILE] [--config FILE]
   # 交互式 REPL；内建命令（行首以 / 开头）：
   #   /help /reset /tools /memory /trace /stats /mode {auto,native,text} /quit

liteagent trace FILE [--type TYPE] [--agent NAME] [--step N] [--json] [--stats] [--limit N]

liteagent schema [-o FILE] [--format {openai,anthropic}] [--builtin]

liteagent multi --mode {sequential,hierarchical} --agents FILE [--prompt TEXT] [--json]
   # agents FILE 是 JSON：{"mode":"sequential","steps":[{"agent":"planner"},...]}
   # 或 {"mode":"hierarchical","manager":"planner","workers":["coder","reviewer"]}

liteagent version
```

**`--tools` 解析规则（冻结，`[v2 变更]` 补齐三处歧义）**：逗号分隔，每个元素**先按组名、
再按工具名**解析（与 `register_all(include=)` 同一套规则，§7.5）；
`--tools` 出现但值为空字符串 → 注册 0 个工具（等价 `include=[]`）；
`--no-tools` 等价 `--tools ""`；`--no-builtin` 只注册用户通过 `--config` 声明的工具；
两者同时出现 → `EXIT_USAGE` + 明确错误信息。
**必须有三条区分用例**：`--tools read_file`（1 个工具）、`--tools files`（5 个工具）、
`--tools ""`（0 个工具）。

**`--approve` 与 HITL（`[v2 新增]`）**：`never`（默认）= `approval_policy=None`（所有
`requires_approval` 工具一律拒绝并返回失败结果）；`write` = 只放行 `write_file`/`delete_file`；
`all` = 全部放行。`chat` 子命令下用 `input("approve <tool>(<args>)? [y/N] ")` 交互确认。

**`render_result` 输出格式（冻结）**：
- `as_json=False`：
  ```
  status: FINISHED
  steps: 3
  tokens: prompt=120 completion=45 total=165
  cost: $0.000123
  duration: 1234.5ms
  ---
  <output>
  ```
- `as_json=True`：`json.dumps(result.to_dict(), ensure_ascii=False, indent=2)`，
  **stdout 只输出这个 JSON**（日志一律走 stderr），保证可被 `jq` 消费。
- `liteagent trace FILE --stats --json` 的输出是
  `json.dumps(trace_stats(load_trace(FILE)), ensure_ascii=False, indent=2)`。

---

## 12. 测试文件清单（每个文件至少 4 个用例；异步用例继承 `IsolatedAsyncioTestCase`）

**`[v2 变更]` 实现顺序冻结**（避免互相阻塞）：
`tests/__init__.py` -> `tests/helpers.py`（按 §12.1 的签名）-> `test_zero_dependency.py` ->
`test_types.py` -> `test_config.py` -> `test_tools_schema.py` -> 其余按依赖顺序。
`tests/__init__.py` 是**空文件**，由第一个提交者在冻结 PR 里创建。

| 文件 | 覆盖重点 |
|---|---|
| `tests/__init__.py` | 空文件（`unittest discover` 需要） |
| `tests/helpers.py` | 冻结签名见 **§12.1** |
| `tests/test_types.py` | `TokenUsage.__add__`/`total_tokens` 自动求和、`ToolCall.canonical_key` 稳定性、`try_from_arguments_json` 不抛且带 `__raw__`、`ToolResult.success/failure`（**断言 `failure().content.startswith("ERROR(")`**）、`to_observation` 在 `ok=False` 时**不带重复前缀**、`LLMResponse.to_message`、`from_dict` 缺字段抛 `SerializationError` |
| `tests/test_errors.py` | 继承关系、`retryable` 类属性表（**逐条对照 §3.4**）、`to_dict`、`__str__` 含 context、`ToolSkippedError`/`ToolApprovalDeniedError`/`BudgetExceededError`/`RunTimeoutError` 的字段 |
| `tests/test_config.py` | 默认值、`RetryPolicy.delay_for` 与 `compute_backoff` **逐字等价**、`rng_seed` 可复现、`rng=None` 不崩、`truncate_head_tail` 头尾都保留、`parse_dotenv` 引号/注释/export、`to_jsonable` 各类型、`render_template` 缺 key 不报错、`estimate_cost_usd` 命中/未命中、**`LoopBoundPool` 的 3 条**（同 key 不同 value 抛 `ConfigError`、`release()` 后 `len(pool)==0`、无运行 loop 时抛 `ConfigError`） |
| `tests/test_message.py` | `Role.coerce`、`to_dict/from_dict` 往返、`drop_orphan_tool_messages`（两种）、`messages_tokens("")==0` |
| `tests/test_transport.py` | `FakeTransport` 记录请求/回放响应、`map_http_error` 全表（401/403/429 带 Retry-After/400/404/500/未知）、`wrap_transport_exception`（**含 `CancelledError` 原样上抛**）、`UrllibTransport` 只测请求构造与错误映射（用 `unittest.mock.patch` 替换 `urlopen`） |
| `tests/test_llm_providers.py` | `OpenAIChatClient` 请求体（tools 形态、消息转换）、响应解析（含 arguments 非法 JSON -> `__raw__` **且不抛异常**）；`AnthropicChatClient` 断言 system 被提取、tool_result 合并进同一 user 消息、block 解析、stop_reason 映射（含 `max_tokens`->`length`）；`DeepSeekChatClient` 的 `default_base_url`；`EchoLLM` 的 `use:add {"a":1,"b":2}` |
| `tests/test_llm_retry.py` | 429 重试后成功；`retry_after_s` 被尊重（**用 `tests.helpers.RecordingSleep` 断言 delays，不真睡**）；重试耗尽抛 `LLMRateLimitError`；不可重试错误只尝试 1 次；`LLMTimeoutError` **不重试**（`retry_on_timeout=False`） |
| `tests/test_llm_registry.py` | `build_llm`/`get_llm` 解析 `provider:model@base_url`、未知 provider 抛 `ConfigError` **且消息里列出 available**、缺 key 抛 `LLMAuthError`、`reset_default_registry` |
| `tests/test_llm_streaming.py` | `[v2 新增]` `ScriptedLLM.astream_chat`：单 chunk 全量、多 chunk 顺序与 index、`finish_reason`、`stream_error` 中断、`astream_chat` 也计入 `calls` |
| `tests/test_scripted.py` | 队列消费、`loop=True` 优先于 `strict`、`loop=False+strict=True` 耗尽抛 `ScriptedExhaustedError`、`error` 响应被记为一次调用、`assert_exhausted` 在未耗尽时失败、**call_id 占位分配**（`ScriptedResponse.tool("t")` 的 `id == ""`，消费后变 `call_0`）、`tool_raw`、`tool_names_seen`（tools=None 时 `[]`）、`calls[i].messages` **不被后续轮次污染**、`kwargs` 含 `temperature=None` |
| `tests/test_tools_schema.py` | **最重（>= 30 例）**：全部映射表逐行断言（bool 先于 int、Literal、Enum、list[dict]、嵌套 dataclass、递归截断、深度上限、set/tuple、dict[str,X]、Union 降级 warning、Optional 不进 required、`*args`、`**kwargs` 抛 `ToolDefinitionError`、`Param` 元数据、pydantic FieldInfo 鸭子类型、`Annotated[str,"desc"]`、docstring Google/Sphinx/无 Args、summary 提取、`validate_instance` 各关键字）+ **三条 Annotated required 用例**（§7.1.5） |
| `tests/test_tools_registry.py` | 注册/重名/别名/`subset` 未命中抛错/`schemas` 两种格式/`to_prompt` 截断（默认 20）/`merge` 不改自身/非法工具名 + **`[v2 新增]` 四条 auto_register 用例**：`@tool(auto_register=True)` 后 `get_default_registry().names()` 含该工具；同名再注册抛 `ToolDefinitionError`；`reset_default_registry()` 后 `names()==[]`；裸 `@tool` 装饰后 `names()` 不变 |
| `tests/test_tools_executor.py` | **最重（>= 30 例）**：并发顺序保持、`max_concurrency` 上限（计数工具断言峰值 <= N）、同步工具**真的执行了**（`tool.run` 被调用，而非返回 coroutine）、异步工具、**同步工具超时：`attempts==1` 且 `metadata["orphan_thread"] is True` 且用 `threading.enumerate()` 快照证明线程还活着**、异步工具超时**仍可重试**（与同步工具区别对待，§3.4）、重试次数与退避（`jitter=0` + `RecordingSleep` 断言 delays）、`retryable=False` 不重试、`idempotent=False` 不重试、验证失败不消耗重试次数、`ToolNotFoundError` 返回可用工具名、结果截断、`_stringify` 各类型、**`sequential_tools` 在两个嵌套 loop（一个普通 agent + 一个 delegate）下仍真正串行**、`execute_many` 顺序、**`fail_fast` 取消兄弟任务后结果列表仍与 calls 对齐且不抛异常**、`execute_sync` 在 loop 内抛 `ConfigError`、**跨两次 `asyncio.run` 复用同一 executor（守 §0.4）：两个 loop 各自登记一份 per-loop 原语后 `assertGreaterEqual(len(executor._pool), 1)`，显式 `aclose()` 后归零**（`[v3 变更]`：v2 这一行写的是"且 `len(executor._pool) == 0` 不泄漏"，与实现相反 —— 手写 raw `asyncio.run` 时清理责任在**调用方**（每个 loop 会留下注册项、同步工具还多留一个私有线程池），只有 `execute_sync` / `Agent.run` 走 `config._run_and_cleanup` 才自动清理；"每次同步调用后池内条数为 0（不泄漏）"由 `CrossLoopLifecycleTests::test_execute_sync_releases_the_pool_each_call` 负责，构造后为空由 `CrossLoopLifecycleTests::test_no_asyncio_primitives_are_created_in_init` 负责）、**`[v3 新增]` 取消等锁者不使 seq 锁变成孤儿**（`SequentialToolsCancellationTests::test_cancelled_lock_waiter_does_not_orphan_the_sequential_lock`）、**`[v3 新增]` 并发的两个 `fail_fast` 批次不共享批次状态**（`ConcurrencyTests::test_concurrent_fail_fast_batches_do_not_share_batch_state`）、审批三例（无 policy 拒绝 / policy 返回 False / policy 返回 True 正常执行）、**熔断**（连续 3 次 infrastructure 失败后第 4 次不执行且 `disabled=True`）、两类回灌文案断言 |
| `tests/test_memory_base.py` | `MemoryItem` 序列化（`include_embedding` 两种）、`from_dict` 缺 embedding 不重算、`HeuristicTokenizer`（中英混排、空串 -> 0）、`cosine_similarity`（零向量、维度不符抛 `MemoryStoreError`） |
| `tests/test_memory_embeddings.py` | `HashingEmbedder` 确定性（跨实例一致、**用 hashlib 手算 idx 做数学断言**）、归一化、相似文本相似度 > 不相似、空文本零向量、`NumpyHashingEmbedder` 一致（`NUMPY_AVAILABLE` 时）、`RandomProjectionEmbedder` 可复现 |
| `tests/test_memory_buffer.py` | 双约束裁剪、`keep_last_n` 保证、`keep_system`、`_repair_tool_pairs`（直接 import 模块级函数）、`evicted`/`drain_evicted` 幂等、token 预算、**多线程 add 与 window 并发不抛 `RuntimeError`** |
| `tests/test_memory_summary.py` | `should_compress` 三条判定、LLM 摘要成功路径、LLM 失败走抽取式兜底且不抛、超长截断、`compression_count`、`previous_summary` 被并入 prompt |
| `tests/test_memory_vector.py` | `upsert` 去重、`should_auto_write` 三种策略与 marker（**含 `role="assistant"` 恒 False**）、**注入 dim=4 的 `CallableEmbedder` 后 `vm.dim == 4` 且 add 成功**、混合打分公式手算对照（**传 `now`**）、排序三级稳定、MMR 去冗、`min_score` 过滤、`metadata_filter`、`max_items` FIFO 淘汰、`access_count` 递增、`score_breakdown` |
| `tests/test_memory_persistence.py` | `[v2 新增]` `save`/`load` 往返后 `len` 与 `search` 顺序一致、embedding 逐元素相等、坏行跳过、维度不符抛 `MemoryStoreError`、`MemoryManager.persist`/`restore` |
| `tests/test_memory_manager.py` | `abuild_prompt` 段落顺序与内容（**传 `now` 后才断言 `<relevant_memories>` 文本**）、`kind="memories"` 与 `memory_count`、`<conversation_summary>` 格式、`last_retrieved` 不触发第二次 search（用 `access_count` 断言）、`stats` 字段、`aclear` 两种范围、`build_prompt` 在 loop 内抛 `ConfigError`、`append_user_input=False` 时不追加第 6 段、`context_window_tokens` 反推 `buffer_budget_tokens` |
| `tests/test_agent_react_native.py` | 单工具一轮、多工具并发顺序、多轮、`finish_reason=stop` 即终结、`max_steps` 用尽 -> `FAILED` + `MaxStepsExceededError`、usage 累加、事件序列断言（**按 §2.7 的归属矩阵**）、`raise_on_error=True` 时抛、**`finish_reason=length` 续写一次**、**`content_filter` 立即失败**、**`tool_calls` 但空 -> 自纠正**、`max_total_tokens` 用尽 -> `BudgetExceededError` + `BUDGET_EXCEEDED` 事件、`max_wall_clock_s` -> `RunTimeoutError`、非法 `**overrides` 键 -> `ConfigError` |
| `tests/test_agent_react_text.py` | `ScriptedResponse.react` 的 action/final 两种、跨行 JSON、```json 围栏、单参数容错、全角冒号、中文 marker、Markdown 加粗 marker、`Observation:` 回灌内容、parse error 自纠正（**断言消息增量恰好 2**）、`max_parse_retries` 用尽、`mode="text"` 强制、**Action 与 Final Answer 同时出现时执行 Action 且不终结**（§9.3 步骤 4） |
| `tests/test_agent_features.py` | 重复动作四种策略、nudge 文本内容、验证错误回灌、未知工具名回灌含可用工具列表、**参数每次略变的重复调用被 nudge 或终止**、`astream` 产出事件顺序 + **消费者提前 break 后 run task 最终被取消、不泄漏**（`[v3 变更]`：v2 写的"不泄漏 task"过于绝对 —— Python 的 async generator 终结是**延迟一拍**的，见 §9.4 该条的 `[v3 变更]`；用例名 `AstreamTests::test_early_break_does_not_leak_the_run_task`、`AstreamTests::test_early_break_stops_a_tool_using_run`）、`describe`、`reset`、`callbacks` 抛错不影响主流程、**arun 不可重入**（并发两次 -> 一次 FAILED 且 error 提到 "already has a run in flight"） |
| `tests/test_callbacks.py` | `EventType` 成员集合快照（含 v2 新增三个）、`TraceEvent.to_dict` 平铺与保留键、`to_json/from_json` 往返（**含 `ts` -> `timestamp` 映射**）、`data` 含保留键时构造即抛 `ConfigError`、`JsonlTraceCallback` 写读往返 + **多线程写不丢行**、`load_trace` 跳过坏行、`TokenCounterCallback` 聚合 + `cost_usd`、`TraceRecorder` 上下文、`render_trace` 非空、**`trace_stats` 对固定事件序列的整个 dict 精确定义** |
| `tests/test_blackboard.py` | 读写/版本递增/`if_version` 冲突/`increment`/`update`/`ttl` 惰性过期/`snapshot` 是副本/`history`/`max_entries` 淘汰/subscribe 取消/**多线程并发写 100 次后计数正确**/**`awatch` 收到写入事件**（异步）/**写入方在 worker 线程写完、loop 已关闭时不抛异常（实测 M-4）** |
| `tests/test_multiagent_sequential.py` | 三步流水线、`input_template` 渲染 `{input}`/`{prev}`/`{steps[x]}`、黑板写入、`propagate_failure` 三种 + **两条 output 公式用例**（失败阶段有部分输出 / 输出为空串回退上一步）、`optional`、usage 聚合、`metadata["steps"]`、输出截断、`share_memory=True` 时 `worker.memory is manager.memory` |
| `tests/test_multiagent_hierarchical.py` | delegate 工具生成（名字/schema/`extra_context`/`timeout_s is NO_TIMEOUT`）、manager 一次委派后出最终答案、worker 失败被压缩为字符串回灌、环检测返回 refused 字符串（不抛）、`max_depth`、`budget` 耗尽 refused、`compress_subagent_output` 头尾都保留、`adecompose` 正常与非法 JSON、contextvar 隔离、**`extra_context` 不影响环检测/深度判定**、**`share_memory` 注入**、**`arun_plan`：3 个 subtask 两层依赖，断言执行顺序满足 `depends_on`、并发峰值 <= `subagent_concurrency`、某个 subtask 失败不中断** |
| `tests/test_builtin_files.py` | `[v2 新增]` `..`/绝对路径/symlink 逃逸抛 `SandboxViolationError`、`allow_read_outside`、`delete_file` 需 `confirm=True`、`metadata['sandbox']` 是 realpath、`make_file_tools(None)` 抛 `ConfigError` |
| `tests/test_builtin_shell.py` | `[v2 新增]` `SHELL_DENY_PATTERNS` 逐条命中（纯函数）、`run_shell("sudo rm -rf /")` 抛 `SandboxViolationError`（用 `patch.dict(os.environ, {"LITEAGENT_ALLOW_SHELL": "1"})`）、未开启时返回 'shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)' 且工具仍可见、`subprocess.TimeoutExpired` -> `ToolTimeoutError`、`metadata['exit_code']` |
| `tests/test_builtin_code.py` | `[v2 新增]` `python_eval` 合法表达式、**`Name` 允许 `variables` 里的键**、拒绝 `__import__`/`eval`/属性白名单外访问、**复杂度闸（`10**10**10`、`range(10**7)`）**、`python_exec` 的 `TimeoutExpired` -> `ToolTimeoutError`、`run_tests` 解析 'Ran N tests'、**`run_tests` 指向本仓库 tests/ 时返回文本含 WARNING** |
| `tests/test_builtin_web.py` | `[v2 新增]` `HTMLTextExtractor` 跳过 script/style、`html.unescape`、`NullSearchBackend` 返回 [] 且 `web_search` 返回可读失败串、`href` 只留 http/https、`allow_network=False`、**`make_web_tools(transport=FakeTransport(...))` 的成功路径** |
| `tests/test_cli.py` | `main(["version"]) == 0`、`tools list`、`tools show`、`tools schema --format openai` 是合法 JSON、`run -p "..." --provider echo --json` 可 `json.loads`、`run` 缺 prompt -> 2、未知子命令 -> 2、`trace FILE` 与 `trace FILE --stats --json`、**`--tools read_file` / `--tools files` / `--tools ""` 三种结果区分**、`--no-tools` 与 `--no-builtin` 冲突 -> 2、**chat 子命令**、**stdout 无 traceback**、`render_result` 两种格式 |
| `tests/test_e2e_code_assistant.py` | 端到端：`ScriptedLLM` 驱动"读文件 -> 搜索 -> 写文件"（`tempfile.TemporaryDirectory` 做沙箱），断言文件真的被写出、trace 事件齐全、usage 累计、`examples/07` 的核心流程可跑 |
| `tests/test_examples_offline.py` | `[v2 新增]` 用 `subprocess` **真跑** `examples/01|03|04|07`（`--offline` / `--provider echo`，超时 60s），断言退出码 0、stdout 含预期关键串、`test_examples_import` 只对需要网络的 02/05/06 做 `compile()` |
| `tests/test_examples_import.py` | 用 `ast` 静态检查每个 `examples/*.py` 可编译（`compile(src, path, "exec")`），**不执行**；断言每个示例文件有 `if __name__ == "__main__":` 与 `--offline` 开关 |
| `tests/test_zero_dependency.py` | **守门测试（必须最先实现）**：三方 import 只允许在 `ast.Try` 内或函数/类体内；每个模块顶层有 `from __future__ import annotations`；无 `import pytest`；**3.11 API 用 AST 判定（禁止文本子串匹配）**；模块间 import 边符合 §1.1 的表（含 E1..E12）；`__main__.py` 逐字内容；**附录 B 每个名字都能 `getattr(liteagent, name)`**；`liteagent.llm.__all__ ⊂ 可取到的名字`；**`AgentResult(` 的所有构造点都是关键字形式**；**用 `ast` 统计 `test_*` 方法数 >= 220 且 `test_tools_schema`/`test_tools_executor` 各 >= 30** |
| `tests/test_docs_coverage.py` | `[v2 新增，轻量版]` 读 `docs/VERIFICATION.md`，断言每行 `evidence` 列里提到的 `tests/test_*.py` 文件**真实存在**（不做方法级断言，避免改文档就挂测试） |

**测试总量目标**：>= 220 个 test method，其中 `test_tools_schema.py` 与 `test_tools_executor.py`
各 >= 30 个。计数方式冻结为**用 `ast` 统计 `FunctionDef` 名字以 `test_` 开头的节点数**
（放在 `test_zero_dependency.py` 里），不依赖覆盖率工具。

### 12.1 `tests/helpers.py` 冻结签名（`[v2 变更]`，10 个实现者共享的唯一夹具文件）

**没有这一节，`test_transport.py` 与 `test_llm_providers.py` 会各写一套不兼容的 `FakeTransport`
并在合并时冲突。以下签名逐字实现。**

```python
class FakeTransport(Transport):
    """可编程的假传输层。requests 记录每一次 send 的请求。"""
    name = "fake"
    def __init__(self, responses: Sequence[HTTPResponse | BaseException] = (),
                 *, status: int = 200, body: str = "{}") -> None: ...
    requests: list[HTTPRequest]          # 每次 send 追加
    def queue(self, response: HTTPResponse | BaseException) -> None: ...  # 追加一个响应
    def send(self, request: HTTPRequest) -> HTTPResponse: ...
    # 队列为空且未给默认响应时 -> 抛 AssertionError("FakeTransport ran out of queued responses")

class FakeSearchBackend(SearchBackend):
    """固定返回 3 条 SearchHit（成功路径用）。"""
    name = "fake"
    def __init__(self, hits: Sequence[SearchHit] = ()) -> None: ...
    def search(self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
               ) -> list[SearchHit]: ...

class RecordingSleep:
    """替代真实 sleep 的 SleepFn 记录器：**默认立即返回、不真睡**。"""
    delays: list[float]
    def __init__(self) -> None: ...
    async def __call__(self, seconds: float) -> None: ...      # 只 append，不 await sleep

@contextmanager
def frozen_time(ts: float) -> Iterator[None]:
    """在 with 块内 patch `liteagent.config.utc_now` 返回传入的 ts。
    实现：unittest.mock.patch("liteagent.config.utc_now", return_value=ts)。
    **注意**：各模块必须写成 `config.utc_now()` 或
    `field(default_factory=utc_now)`（模块级 import 的裸名不会被 patch 到），
    因此冻结的写法是**统一从 config 模块取**：`from liteagent import config` +
    `field(default_factory=lambda: config.utc_now())`。
    为降低实现者心智负担，允许直接 `default_factory=utc_now`（同一个函数对象），
    但**断言时间戳的测试必须走 frozen_time + 上述 lambda 形态**。"""
    ...

def make_registry(*tools: Tool) -> ToolRegistry: ...
def collect_events() -> tuple[CallbackManager, list[TraceEvent]]: ...
@contextmanager
def make_temp_sandbox() -> Iterator[PathSandbox]:
    """在 tempfile.TemporaryDirectory 里造一个 PathSandbox，退出时清理。"""
    ...
def det_embedder(table: Mapping[str, Sequence[float]]) -> CallableEmbedder:
    """按"文本 -> 向量"的查找表构造确定性 embedder（dim 由表里的向量长度决定）。"""
    ...
def assert_no_error_events(testcase: unittest.TestCase, events: Sequence[TraceEvent]) -> None:
    """断言 events 里没有 run_failed / llm_error / tool_error。"""
    ...

# 常用工具
@tool
def add(a: int, b: int) -> int: ...          # EchoTool 家族
@tool
def echo(text: str) -> str: ...
@tool
def boom(msg: str = "boom") -> str: ...      # 总是抛 ValueError（测异常路径）
```

**冻结的测试卫生规则（`[v2 变更]`）**：
1. 任何 `setUp`/`tearDown` 里用到 `auto_register` 的测试，**必须**在 `tearDown` 调
   `reset_default_registry()`。
2. 不得依赖真实网络、真实时钟、真实 sleep。
3. 断言不确定字段前先查 §2.8 的禁止清单。

### 12.2 `docs/VERIFICATION.md` 的冻结表结构（`[v2 新增]`）

三列表格 `claim | evidence | status`，`status` 只能取
`offline-verified` / `code-only-not-run` / `not-implemented`。
每个简历原子能力一行（至少 9 行，见 `docs/INTERVIEW.md` 的能力清单）。
`evidence` 必须写成 `tests/test_x.py` 或 `command: python3 -m ...` 的形式。

### 12.3 与简历三条 bullet 的对账（`[v2 新增]`，`docs/VERIFICATION.md` 必须逐行覆盖）

| # | 简历主张 | 证据（offline-verified 的落点） |
|---|---|---|
| 1 | LLM 抽象层统一多模型 API | `test_llm_providers.py`（4 个 provider）+ `test_llm_registry.py` |
| 2 | 工具层装饰器自动注册 + JSON Schema 生成 | `test_tools_schema.py` + `test_tools_registry.py`（auto_register 四例） |
| 3 | 记忆层短期对话历史 | `test_memory_buffer.py` + `test_memory_manager.py` |
| 4 | 记忆层长期向量存储 | `test_memory_vector.py` + `test_memory_persistence.py`（跨进程往返） |
| 5 | 完整 Thought-Action-Observation 循环 | `test_agent_react_text.py` + `test_agent_react_native.py` |
| 6 | Function Calling 与执行器（错误处理与自动重试） | `test_tools_executor.py` + `test_llm_retry.py` |
| 7 | 支持并发工具调用 | `test_tools_executor.py`（顺序保持 + 峰值上限 + fail_fast） |
| 8 | Sequential / Hierarchical 两种协作模式 + 任务分解 | `test_multiagent_sequential.py` + `test_multiagent_hierarchical.py`（含 `arun_plan`） |
| 9 | 内置工具（网页搜索/代码执行/文件操作）+ 文档与示例 | `test_builtin_*.py`（4 个）+ `test_examples_offline.py` + `test_e2e_code_assistant.py` |

### 12.4 `benchmarks/` 与 `scripts/` 的冻结职责（`[v2 新增]`）

- `benchmarks/bench_dataclass_vs_pydantic.py`：`AgentState.to_dict()` 在 1000 条消息下
  dataclass vs pydantic 的耗时对照（填 D-01 的「实测补充」）。
- `benchmarks/bench_embedding_similarity.py`：同义句/反义句的相似度矩阵（填 D-04）。
- `benchmarks/bench_retrieval_ranking.py`：20 条记忆（含 3 条冗余 + 2 条过时）的
  纯相似度 vs 混合打分 vs 混合+MMR 的 top-k 对照表（填 D-08）。
- 每个脚本必须支持 `--json` 并把结果同时写进 `docs/VERIFICATION.md`。
- `scripts/check_spec_consistency.py`：读本文件，检查 §1.2 的文件清单与实际仓库一致、
  §1.4/附录 B 的 `__all__` 名字都能解析。可选工具，**不参与测试**。

---

## 13. 冻结红线（实现者不得违反的清单）

1. **不得新增、改名、删除 `liteagent/` 下的 41 个文件**（`liteagent/__main__.py` 是唯一例外，
   内容仍受 §1.2 约束）。`docs/`、`tests/`、`examples/` 由各自章节的**封闭清单**约束。
2. 不得新增第三方依赖；不得在任何文档或代码里建议 `pip install`（本环境装不了）。
3. 不得在 `liteagent/` 顶层 import 三方库（`test_zero_dependency.py` 会抓）。
4. 不得使用 §0.1 列出的 3.11 API。
5. 不得在 `__init__`/模块级创建 `asyncio` 同步原语（R-LOOP，见 §0.4）；`threading` 原语不受此限。
6. 不得让 `Agent.arun` / `ToolExecutor.execute` / `ToolExecutor.execute_many` /
   `MemoryManager.acompress` 向外抛业务异常
   （唯一例外：`asyncio.CancelledError`、`KeyboardInterrupt`、且仅当 `raise_on_error=True`）。
   **`execute_many` 与 `execute` 的契约必须一致**（v1 里 `execute_many` 会抛
   `ToolRetryExhaustedError`，与本节冲突；v2 已改为永不抛，见 §7.4.2）。
   **`Agent` 仍必须能捕获 executor 抛出的 `LiteAgentError`** —— 这不是多余要求：
   `Agent` 的 `executor` 可以由用户传入自定义实现，防御性捕获是红线 10 的一部分。
7. 不得改变 §2.5 的常量名与默认值。
8. 不得改变 §2.4 预留 metadata key 的名字、语义与**写入者归属**（§2.7）。
9. 不得让 `ToolResult.content` 变成非字符串类型（模型侧只认文本）。
10. 不得吞掉异常而不记录：任何 `except` 块必须要么重新抛出、要么写日志/事件、要么
    写入 `ToolResult.metadata`。`except: pass` 一律视为 bug。
11. 不得在 `search`/`retrieve` 之外的地方泄漏内部可变状态（一律返回副本）。
12. 任何降级（schema 降级为 `{}`、embedding 用哈希、摘要用抽取式、搜索用 Null 后端、
    上下文被裁剪、工具被熔断）**必须**留下可观测痕迹（`warnings` / 事件 / 日志），不得静默。
13. **`[v2 新增]` 并发原语的选型**：跨线程/跨 loop 必须生效的限流与互斥**一律用 `threading` 原语**；
    `LoopBoundPool` 只允许用于"绝不离开当前 loop"的场景。
14. **`[v2 新增]`** 获取顺序必须是**先并发信号量、后 seq 锁**（§7.4.1 步骤 5.b）；反序即死锁。
15. **`[v2 新增]`** `Agent.__init__` 默认构造的记忆**必须复用自身的 llm**（§9.4）。
16. **`[v2 新增]`** 一切退避等待必须经由 `sleep_fn`（§5.5），禁止直接 `await asyncio.sleep`。
17. **`[v2 新增]`** 所有"取当前时间"必须走 `config.utc_now`（§2.2）。
18. **`[v2 新增]`** 同一事件只能由 §2.7 指定的唯一发射者发出。

---

## 附录 A：实现者开工前自检清单

- [ ] 我负责的文件在 §1.2 表格里，行数（实测）与我的实现量级相符
- [ ] 我 import 的模块编号都比我小（或属于 §1.1 的 L 编号表 / E1..E12 白名单）
- [ ] 我用到的常量都从 `config.py` 引入，没有自己写魔法数字
- [ ] 我的公开方法签名与本文档逐字一致（参数名、默认值、返回类型）
- [ ] 我的返回结构 `to_dict()` 字段齐全（含 `None`），除 §2.2 的三个显式 opt-out
- [ ] 我的异常都是 `errors.py` 里已定义的类（没有自己造新异常）
- [ ] 我发的每条事件都在 §2.7 里、且我是它的唯一发射者
- [ ] 我的退避走 `sleep_fn`、随机数走实例级 `random.Random`
- [ ] 我的时间戳走 `config.utc_now`
- [ ] 我的同步原语要么是 `threading`，要么经 `LoopBoundPool` 懒创建
- [ ] 我的 `except` 块首行保证了 `CancelledError` 不被吞
- [ ] 我写了 `tests/test_<我的模块>.py` 且 `python3 -m unittest discover` 通过
- [ ] 我没有在顶层 import 三方库
- [ ] 我没有在 `__init__` 里创建 `asyncio.Semaphore/Lock/Condition/Event`

## 附录 B：`liteagent/__init__.py` 的 `__all__`（冻结，`[v2 变更]`）

```python
__version__ = "0.1.0"        # 运行时优先从 importlib.metadata 读取，失败回退此值

__all__ = [
    # core
    "Agent", "AgentConfig", "AgentResult", "AgentState", "AgentStatus",
    # llm
    "LLMClient", "BaseLLMClient", "LLMConfig", "LLMResponse", "LLMStreamChunk",
    "Message", "Role", "TokenUsage", "ToolCall", "ToolResult",
    "ScriptedLLM", "ScriptedResponse", "ScriptedCall", "LLMRegistry",
    "build_llm", "get_llm",
    # tools
    "tool", "Tool", "ToolSpec", "ToolRegistry", "ExecutorConfig", "ToolExecutor",
    "make_function_tool", "is_tool", "current_cancel_flag", "cancel_scope",
    "register_all", "BUILTIN_TOOL_NAMES", "BUILTIN_TOOL_GROUPS",
    "get_default_registry", "reset_default_registry",
    # memory
    "MemoryManager", "MemoryConfig", "MemoryItem", "MemoryStore", "MemoryStoreError",
    "BufferMemory", "BufferConfig", "VectorMemory", "VectorConfig",
    "SummaryMemory", "SummaryConfig", "HashingEmbedder", "Embedder", "Tokenizer",
    "get_default_tokenizer",
    # multiagent
    "SequentialAgent", "SequentialStep", "HierarchicalAgent", "Plan", "SubTask",
    "Blackboard", "BlackboardEntry", "TeamConfig", "MultiAgent", "DelegationContext",
    "build_team", "compress_subagent_output",
    # callbacks
    "EventType", "TraceEvent", "CallbackManager", "TraceRecorder",
    "LoggingCallback", "JsonlTraceCallback", "RichCallback", "TokenCounterCallback",
    "MemoryTraceCallback", "load_trace", "render_trace", "trace_stats",
    # config
    "AppConfig", "RetryPolicy", "LoopBoundPool", "run_sync", "utc_now", "format_ts",
    "to_jsonable", "parse_dotenv", "load_dotenv", "parse_bool", "render_template",
    "estimate_cost_usd", "MODEL_PRICES", "NO_TIMEOUT",
    # errors
    "LiteAgentError", "ConfigError", "ToolError", "ToolNotFoundError",
    "ToolValidationError", "ToolExecutionError", "ToolTimeoutError", "ToolSkippedError",
    "ToolRetryExhaustedError", "ToolApprovalDeniedError", "ToolDefinitionError",
    "LLMError", "LLMRateLimitError", "LLMTimeoutError",
    "LLMConnectionError", "LLMAuthError", "AgentError", "MaxStepsExceededError",
    "RepeatedActionError", "ReActParseError", "MultiAgentError", "DelegationError",
    "MaxDepthExceededError", "CycleDetectedError", "VersionConflictError",
    "SandboxViolationError", "ScriptedExhaustedError", "SerializationError",
    "BudgetExceededError", "RunTimeoutError", "AgentAbortedError",
]
```

`__init__.py` **不得**在 import 时做重活（不扫描环境、不发请求、不注册工具）。
`liteagent.llm`、`liteagent.tools`、`liteagent.memory`、`liteagent.agent`、`liteagent.multiagent`
的 `__all__` 见 §1.4 / §10.5。

---

*规范结束。任何与本文档不一致的实现都应在 code review 中被拒绝。*
