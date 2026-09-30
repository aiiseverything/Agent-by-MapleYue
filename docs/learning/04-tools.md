# 第 4 章 · 工具系统：装饰器与 Schema 自动生成

> **本章目标**：理解简历里「工具层基于装饰器实现自动注册与 JSON Schema 生成」到底是怎么做的。
> 这是整个项目**技术密度最高**的一章，也是面试最容易被追问的地方。

---

## 4.1 问题：工具怎么"告诉模型"自己能干什么

回顾第 1 章：模型只会输出文本。要让它能调工具，你必须先**告诉它有哪些工具、每个工具要什么参数**。

对 `add(a, b)` 这个函数，你要给模型这样一段 JSON：

```json
{
  "name": "add",
  "description": "把两个整数相加",
  "parameters": {
    "type": "object",
    "properties": {
      "a": {"type": "integer", "description": "第一个加数"},
      "b": {"type": "integer", "description": "第二个加数"}
    },
    "required": ["a", "b"]
  }
}
```

这个东西叫 **JSON Schema**，它规定了参数的名称、类型、是否必填、取值范围。

**手工维护它有四个问题**：

1. **重复**：`def add(a: int, b: int)` 里的类型信息，你又要用 JSON 写一遍。
2. **会不同步**：改了函数签名忘了改 schema，模型就会按错的说明传参。
3. **易错**：JSON 少个逗号、`required` 写错，模型的行为就莫名其妙。
4. **描述容易写歪**：`description` 是模型理解工具用途的**唯一**依据，写不好工具就废了。

### 本项目的解法

```python
@tool
def add(a: int, b: int) -> int:
    """把两个整数相加。

    Args:
        a: 第一个加数。
        b: 第二个加数。
    """
    return a + b
```

**类型注解就是类型，docstring 就是描述。** 框架在装饰时就把它反射成了上面那段 JSON。

这就是简历里「基于装饰器实现自动注册与 JSON Schema 生成」的含义。

---

## 4.2 `@tool` 装饰器的三种用法

### 用法一：裸装饰（最常用）

```python
from liteagent.tools import tool

@tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b
```

装饰完 `add` **不再是一个函数，而是一个 `Tool` 对象**。它身上带着：

```python
add.name          # 'add'                （函数名）
add.description   # '把两个整数相加。'   （docstring 摘要）
add.parameters    # JSON Schema
add.spec          # ToolSpec：完整的元数据 + 原始函数
add.raw           # 原始的那个函数（万一你要直接调）
```

### 用法二：带参数装饰

```python
@tool(name="sum", description="把两个整数相加（自定义名字和描述）", tags=["math"], dangerous=False)
def add(a: int, b: int) -> int:
    return a + b
```

可以传的键（白名单，传错会**当场报错**而不是静默忽略）：

```
name, description, parameters, tags, dangerous, requires_approval,
idempotent, timeout_s, max_retries, version, is_async, pass_style, docstring_style
```

```python
@tool(require_approval=True)     # 少写了个 s
# → ToolDefinitionError: unknown tool option(s) ['require_approval']; supported: [...]
```

**为什么这里要"大声失败"？** 因为如果你把 `requires_approval=True` 拼错了，
而框架静默忽略，你就得到了一个**你以为需要审批、实际不需要**的危险工具。
这类静默失效是安全事故的温床。

### 用法三：自动注册到全局

```python
@tool(auto_register=True)
def add(a: int, b: int) -> int:
    """..."""
    return a + b

from liteagent.tools import get_default_registry
get_default_registry().names()      # ['add']
```

**默认是 `False`——装饰一个工具不会碰任何全局状态。** 这是一个刻意的设计决策：

> 全局注册表是**进程级共享状态**。如果 `@tool` 默认往里塞，你的测试就会互相污染
> （A 测试注册的工具出现在 B 测试里），而且两个 Agent 会莫名其妙看到彼此的工具。
>
> 所以本项目的默认姿势是**显式持有自己的 `ToolRegistry`**：
> ```python
TOOLS = ToolRegistry([add, word_count])
agent = Agent(llm=llm, tools=TOOLS)```
> 只有你明确知道自己在做什么时，才用 `auto_register=True`。

重名不会被静默覆盖：

```python
# → ToolDefinitionError: tool 'add' is already registered; pass override=True to replace it
```

---

## 4.3 核心：type hints → JSON Schema 的完整映射

**这是本章最重要的部分。** 下面每一行都是在本机实测过的真实输出。

### 4.3.1 基础类型

| 你写的注解 | 生成的结果 |
| --- | --- |
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `None` / 无注解 / `Any` | `{}`（任意类型） |
| `Literal["a","b"]` | `{"type": "string", "enum": ["a","b"]}` |
| `Literal[1,2]` | `{"type": "integer", "enum": [1,2]}` |
| `Literal[True,False]` | `{"type": "boolean", "enum": [true,false]}` |
| `Literal[1,"b"]`（混型） | `{"enum": [1,"b"]}`（**没有 type**） |

### 4.3.2 容器类型

| 注解 | 结果 |
| --- | --- |
| `list[int]` | `{"type":"array","items":{"type":"integer"}}` |
| 裸 `list` | `{"type":"array"}`（没有 items） |
| `set[str]` | `{"type":"array","items":{...},"uniqueItems":true}` |
| `dict[str,int]` | `{"type":"object","additionalProperties":{"type":"integer"}}` |
| `tuple[int,str]`（固定长度） | `{"type":"array","items":{},"minItems":2,"maxItems":2}` + **一条 warning** |

### 4.3.3 三个反直觉的坑（面试高频）

**坑 1：`bool` 必须先于 `int` 判定**

```python
isinstance(True, int)   # True ！
```

Python 里布尔值是整数的子类。如果先判 `int`，`True` 会被判成 `integer`。
本项目在**两处**处理这个问题，而且**性质不同**：

- 注解分支用 `annotation is bool` 的**身份比较**（不是 `isinstance`），顺序在这里其实不影响结果，
  但代码注释明确写着"bool 必须先于 int 判定"——**这是一个冻结契约，防止后人重排**。
- **值推断**分支用 `isinstance`，顺序**真的会**改变结果：
  ```python
  if isinstance(value, bool):  return "boolean"   # 必须在前
  if isinstance(value, int):   return "integer"
  ```

**坑 2：`Annotated` 不能用 `isinstance` 判断**

这是 Python 的一个真实陷阱，实测：

```python
>>> isinstance(Annotated[int, "x"], Annotated)
False          # ← 不报错，静默返回 False！
>>> getattr(Annotated[int, "x"], "__metadata__", ())
('x',)         # ← 正确的判断方式
```

所以本项目里**没有任何一处** `isinstance(..., Annotated)`，一律用 `__metadata__` 属性探测。
**如果实现者不知道这一点，`Annotated` 支持会静默失效**——不报错，只是描述和约束全丢了。

**坑 3：不支持的类型要"降级 + 留痕"，不能静默**

```python
@tool
def degraded(a: Union[int, str], b: bytes, c: "Undefined") -> None:
    """..."""
```

生成的结果里，这些参数变成 `{}`（任意类型），同时 `warnings` 里有 5 条明确记录：

```
union type typing.Union[int, str] is unsupported (multi-type schemas break
  OpenAI/Anthropic-compatible grammar converters); degraded to {} (any)
unsupported annotation <class 'bytes'>; degraded to {} (any)
...
```

**为什么法则是"留痕"而不是"报错"？**

因为一个参数类型复杂就让整个工具无法注册，太严格了；但静默降级更糟——
模型会以为这个参数随便传（schema 是 `{}`），而执行时也不会校验，
错误要到运行时才暴露。**降级 + 可查询的痕迹**是平衡点的选择。

这个痕迹还是**三层可观测**的：`ToolSpec.warnings`（定义期）→
`metadata["validation_gaps"]`（执行期）→ CLI 的 `tools show` 会渲染出来。

### 4.3.4 嵌套 dataclass

```python
@dataclass
class GeoPoint:
    lat: float
    lon: float

@tool
def find(near: GeoPoint) -> str:
    """..."""
```

`GeoPoint` 会被递归展开成：

```json
{"type":"object",
 "properties":{"lat":{"type":"number"},"lon":{"type":"number"}},
 "required":["lat","lon"],
 "additionalProperties":false}
```

两条边界规则：

- **深度上限 8 层**（`MAX_SCHEMA_DEPTH = 8`）。超过就降级为 `{}` 并记 warning，
  防止恶意或意外的超深嵌套把递归撑爆。
- **环检测**：自引用类型（`Optional["Node"]`）在第 1 层就被截断，
  记一条 `recursive dataclass Node truncated at depth 1`。

### 4.3.5 `Annotated` + `Param`：给参数加约束和描述

`Param` 是本项目提供的元数据助手：

```python
from typing import Annotated
from liteagent.tools.schema import Param

@tool
def search(
    keyword: str,
    min_price: Annotated[float, Param(description="最低价（含税）", ge=0)] = 0.0,
    limit: Annotated[int, Param(default=5, ge=1, le=100)] = 5,
) -> str:
    """..."""
```

生成：

```json
"min_price": {"type":"number", "description":"最低价（含税）", "minimum":0},
"limit":     {"type":"integer", "description":"...", "minimum":1, "maximum":100, "default":5}
```

支持的约束关键字包括 `ge` / `le` / `gt` / `lt` / `minLength` / `maxLength` / `pattern` / `default` 等。

---

## 4.4 `required` 的判定公式（唯一一条）

**一个参数什么时候是必填？** 顶层函数参数的规则（逐字来自代码）：

```python
if (param.default is inspect.Parameter.empty        # 没有默认值
    and not is_optional(annotation)                 # 且不是 Optional
    and not any(有 Param(default=...) 元数据)):      # 且没有用 Param 给默认值
    required.append(param_name)
```

嵌套 dataclass 字段的规则多一条 `default_factory` 分支：

```python
if (field.default is MISSING
    and field.default_factory is MISSING            # ← 这一条曾经漏掉过
    and not is_optional(annotation)
    and not any(有 Param(default=...))):
```

实测对照：

| 函数签名 | 是否 required | 说明 |
| --- | --- | --- |
| `def f(a)` | ✅ | 无注解无默认值也算必填 |
| `def f(b: int = 3)` | ❌ | |
| `def f(c: int \| None = None)` | ❌ | `Optional` 不参与必填 |
| `def f(d: Annotated[int, Param(default=7)])` | ❌ | **签名没有默认值，但 `Param(default=)` 也算** |

**两个容易踩的边界**：

- **`*args`** → 映射成 `{"type":"array","items":...}`，**永远不进 required**。
- **`**kwargs`** → **直接抛 `ToolDefinitionError`**：
  ```
  'with_kwargs' declares **kwargs; arbitrary keys cannot be reflected into a
  JSON Schema. Pass parameters= explicitly to take over the schema.
  ```
  因为"任意键"在 JSON Schema 里无法表达（除非放弃类型约束），
  与其生成一个骗人的 schema，不如明确拒绝并提供逃生舱（手写 `parameters=`）。

> **一个值得记住的行为**：**签名默认值不会生成 schema 里的 `"default"` 键**。
> 只有 `Param(default=...)` 才会。这是实测行为——`required` 的判定用了签名默认值，
> 但 schema 里不暴露它。知道这个差异可以在面试时体现你对代码的熟悉程度。

---

## 4.5 docstring → description

### 4.5.1 description 的来源链

```
参数 description=  >  docstring 摘要  >  函数名  >  ""
```

"摘要"的定义：**到第一个空行为止的内容**，多行会拼成一行。
所以写 docstring 时，**第一行必须是一句话概括**，详细说明放在空行之后——它不会进 schema。

### 4.5.2 支持两种风格 + 真实对照

**Google 风格**（最常用）：

```python
def google_style(path: str, count: int = 3, *, verbose: bool = False) -> str:
    """Read a file and count lines.

    This longer explanation should NOT appear in description.

    Args:
        path: Path to the file to read.
        count (int): How many lines to take.
        verbose: Print extra diagnostics.

    Returns:
        The file content.
    """
```

实测生成的 schema：

```json
{
  "type": "object",
  "properties": {
    "path":    {"type": "string",  "description": "Path to the file to read."},
    "count":   {"type": "integer", "description": "How many lines to take."},
    "verbose": {"type": "boolean", "description": "Print extra diagnostics."}
  },
  "required": ["path"],
  "additionalProperties": false
}
```

注意三件事：
1. 第二段解释**没有**进 description（摘要只到第一个空行）；
2. `count (int)` 括号里的类型串被吃掉、描述正常保留；
3. **没有 `"default": 3`**。

**Sphinx 风格**（`:param x:`）：

```python
def sphinx_style(path, pattern, limit=10):
    """Search a file for a pattern.

    :param path: Path to search in.
    :param pattern: Regex pattern.
    :type pattern: str
    :param limit: Max results.
    """
```

也能正确解析出三个参数的描述（`:type` 行被忽略）。

风格可以通过 `@tool(docstring_style="google"|"sphinx"|"none")` 指定，默认 `"auto"`
（先试 Google，没有条目再试 Sphinx）。

### 4.5.3 一个忠告

> **description 是模型理解工具用途的唯一依据。**
>
> 反面例子：`"""处理数据。"""` ——模型完全不知道这个工具处理什么数据、什么时候该用。
> 正面例子：`"""从 CSV 文件里读出指定列，返回去重后的值列表。适合需要枚举某列的取值时使用。"""`
>
> 写工具描述时，想象你在给一个**从没看过你代码的同事**写使用说明。

---

## 4.6 `ToolRegistry`：工具清单的管理者

```python
registry = ToolRegistry([add, word_count])       # 构造时注册
registry.register(other_tool)                     # 注册（重名默认报错）
registry.register(t, override=True)               # 强制覆盖
registry.unregister("name")                       # 注销（连带清理指向它的别名）
registry.alias("sum", "add")                      # 起别名：sum 也能调到 add
registry.get("add")                               # 按名取（未命中 → ToolNotFoundError，且列出所有可用名）
registry.names()                                  # 全部工具名（排序后，不含别名）
registry.list(tags=["math"])                      # 按标签过滤（任一命中）
registry.subset(["add", "mul"])                   # 取子集（未命中任一 → 报错）
registry.merge(other)                             # 合并成新表（不改动自身）
```

### 4.6.1 导出给模型的两种格式

```python
registry.schemas(fmt="openai")      # [{"type":"function","function":{"name":...,"parameters":...}}]
registry.schemas(fmt="anthropic")   # [{"name":...,"description":...,"input_schema":...}]
```

**只有这两种格式**（传别的会抛 `ConfigError` 并列出可用值）。这正是第 3 章说的
"厂商差异"在工具层的体现——**同一份内部 schema，导出成两家要的形状**。

### 4.6.2 `to_prompt()`：给文本模式用

```python
registry.to_prompt()
# add(a: integer, b: integer) - Add two integers.
# word_count(text: string) - Count words.
```

native 模式走 `schemas()`（结构化字段），**文本 ReAct 模式走 `to_prompt()`**
（渲染进 system prompt 的文字清单）。同一个工具集，两种投喂方式。

超过 20 个工具会截断并追加 `... (N more tools omitted)`——**不静默截断**。

---

## 4.7 `ToolExecutor`：真正执行工具的人

工具被模型"点名"之后，是谁真的去调用的？是 `ToolExecutor`。
它承担了所有"脏活"：

```
收到 ToolCall
   │
   ├─ 1. 工具存在吗？         不存在 → ToolNotFoundError（并列出可用工具）
   ├─ 2. 参数合法吗？         schema 校验失败 → ToolValidationError（不消耗重试次数）
   ├─ 3. 需要审批吗？         需要但没有审批策略 → 拒绝（fail-closed）
   ├─ 4. 熔断了吗？           连续失败太多次 → 直接拒绝，不再执行
   ├─ 5. 并发（信号量）        控制同时在跑的工具数量
   ├─ 6. 顺序锁（可选）        某些工具必须串行
   ├─ 7. 执行 + 超时           同步工具丢线程池，异步工具直接 await
   ├─ 8. 失败重试？           按 retryable 白名单 + 幂等性决定
   └─ 9. 结果截断             太长就头尾保留地截断
   │
   ▼
ToolResult（永不抛异常，失败也返回对象）
```

### 4.7.1 并发：三个不同的机制，别混淆

这是全项目最容易讲错的地方。本项目的并发控制其实是**三层**：

| 机制 | 用什么实现 | 控制什么 |
| --- | --- | --- |
| **全局并发闸** | `asyncio.Semaphore`（按事件循环懒创建） | 同时执行的工具总数 ≤ `max_concurrency`（默认 4） |
| **同步工具的线程池** | 每个 loop 一个私有 `ThreadPoolExecutor` | 同步阻塞工具在哪跑（默认 8 个 worker） |
| **单工具串行锁** | `threading.Lock` | 同一工具（或 `sequential_tools` 里的工具）不能并发跑 |

实测效果（6 个各睡 0.15 秒的同步工具）：

```
max_concurrency=2:  峰值并发=2   总耗时 0.55s
max_concurrency=4:  峰值并发=4   总耗时 0.40s
max_concurrency=6:  峰值并发=6   总耗时 0.25s
```

**为什么同步工具必须丢线程池？** 因为同步函数会**阻塞整个事件循环**——
一个 `time.sleep(10)` 会让所有其他工具、所有其他 Agent 全部卡住。丢到线程池里，
事件循环就能继续调度其他任务。

**为什么获取顺序写死为"先并发信号量、后顺序锁"？**

假设反序：先拿顺序锁、再等并发信号量。那么 N 个调用者可能各自拿着不同的顺序锁，
同时等着信号量空位——而信号量里正在跑的调用者，可能正需要其中某把顺序锁。
**互相等待 = 死锁**。本项目的规范里把这条写成红线：
"反序即死锁"。

### 4.7.2 超时

超时时间的解析是**四级取最小有效值**：

```
参数 timeout_s  >  call.metadata["timeout_s"]  >  tool.spec.timeout_s  >  config.default_timeout_s(30s)
                                                                    ↓
                                                              取其中最小的那个
```

特殊哨兵 `NO_TIMEOUT = -1.0`：表示**显式禁用超时**。
内置的 `run_shell` / `python_exec` / `run_tests` 都用它——因为编译、跑测试可能要好几分钟，
30 秒的默认超时对它们是错的。

### 4.7.3 重试：白名单制

**默认只有 4 个异常会重试**（`retryable=True`）：

```
LLMRateLimitError     （429，服务端限流）
LLMTimeoutError       （超时）
LLMConnectionError    （网络中断、5xx）
ToolTimeoutError      （工具超时）—— 但有个重要例外，见下
```

其余一律不重试。**这个设计是"默认不重试，显式声明可重试"**（default-deny），
而不是"默认重试，黑名单排除"。

> **为什么默认不重试更安全？** 因为"重试"意味着**再执行一次副作用**。
> 如果工具是"下单"，重试就是重复下单。默认放行重试 = 默认允许重复副作用。

**两个额外的保护**：

1. **非幂等工具不重试**：`@tool(idempotent=False)` 的工具总尝试次数被强制为 1。
   要放开得显式配 `allow_retry_on_non_idempotent=True`。
2. **同步工具超时后不重试**（本项目的一条特殊规则）：
   ```python
   if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async:
       retryable = False
   ```
   **为什么？** 因为超时只取消了 `await` 那一层，**跑在 worker 线程里的同步代码无法被中断**——
   线程还在跑！如果重试，就会有**两个线程同时写同一份资源**，导致数据损坏。
   框架的做法很诚实：不假装它停了，而是打上标记并记日志：
   ```
   executor: tool slow_sync timed out after 0.2s but its worker thread is still
   running (orphan_thread=True); the result is marked and this attempt will NOT be retried
   ```
   于是 `ToolResult.metadata["orphan_thread"] = True`，调用方可以据此判断。

   实测对照：
   ```
   同步工具超时 → attempts=1   orphan_thread=true
   异步工具超时 → attempts=3   （异步没有孤儿线程问题，可以安全重试）
   ```

**那长耗时的同步工具怎么办？** 项目提供了**协作式取消**通道：
工具自己在循环里检查取消标志。

```python
from liteagent.tools.base import current_cancel_flag

@tool
def long_task(n: int) -> str:
    """一个可以被打断的长任务。"""
    flag = current_cancel_flag()
    for i in range(n):
        if flag is not None and flag.is_set():
            return f"aborted at {i}"      # 主动退出，线程干净结束
        do_work(i)
    return "done"
```

> **这个"孤儿线程"的处理是很好的面试素材**：它体现的不是"我知道怎么重试"，
> 而是"我知道什么时候**不能**重试，以及为什么"。

### 4.7.4 退避公式

```
基础延迟 = min(最大延迟, 基础值 × 2^尝试次数)
jitter = 0     → 直接用基础延迟         （测试时用这个，结果可复现）
jitter = 1     → 在 [0, 基础延迟] 里随机（full jitter）
jitter = 0.5   → 在 [0.5×基础, 基础] 里随机（equal jitter，默认值）
```

默认参数：`max_retries=2`（额外重试 2 次，总尝试 3 次）、`backoff_base_s=0.25`、
`backoff_max_s=8.0`、`jitter=0.5`。

**为什么要加抖动（jitter）？** 假设 100 个客户端同时被限流，如果都按 `2^n` 秒精确重试，
它们会**在同一时刻再次一起冲过来**（惊群效应），服务端继续限流。
加随机抖动把重试时间打散。

**服务端如果给了 `Retry-After`，会取 `max(退避延迟, Retry-After)`**——尊重服务端的指示。

### 4.7.5 审批：Human-in-the-loop，且 fail-closed

危险操作可以要求人工确认：

```python
@tool(requires_approval=True)
def delete_file(path: str) -> str:
    """删除文件。"""
    ...
```

然后配置审批策略：

```python
ExecutorConfig(approval_policy=lambda call, tool: input(f"允许调用 {tool.name} 吗？(y/n)") == "y")
```

**关键设计是 fail-closed（失败关闭）**：

| 情况 | 结果 |
| --- | --- |
| `requires_approval=True` 但**没配** `approval_policy` | **拒绝**执行 |
| policy 返回 `False` | 拒绝 |
| policy 抛异常 | **拒绝**（不是放行） |
| policy 返回 `True` | 执行，`metadata["approved"] = True` |

**"没有审批人"绝不等于"审批通过"。** 这个默认值的方向（默认拒绝 vs 默认放行）
决定了系统在配置出错时的行为——前者安全，后者是事故。

### 4.7.6 熔断

某个工具连续失败 3 次（`disable_tool_after_failures`，0 = 关闭）之后，
后续调用**不再执行**，直接返回失败：

```
call 1: error_type=ToolExecutionError  disabled=None
call 2: error_type=ToolExecutionError  disabled=None
call 3: error_type=ToolExecutionError  disabled=None
call 4: error_type=ToolExecutionError  disabled=True    ← 第 4 次直接被拦
```

**只统计 infrastructure 类失败**（工具自身的问题），不统计 recoverable 类（模型参数错误）——
因为后者是模型的输入问题，改一次就好了；把它算进熔断会让"模型打错一次工具名"
直接禁掉一个完全健康的工具。

### 4.7.7 结果截断

工具返回太长会撑爆上下文：

```python
ExecutorConfig(max_result_chars=8000)     # 默认 8000
```

截断方式是**头尾保留**（默认头 70% + 尾 30%，中间插 `... [truncated N chars] ...`）。

**为什么保留尾部而不是只留头部？** 因为**结论常常在末尾**——
错误信息、汇总数字、最后几行日志。只截头会丢掉最关键的诊断信息。

并且如实记录：`metadata["truncated"]=True`、`metadata["original_chars"]=原始长度`。

### 4.7.8 三个入口方法

| 方法 | 用途 | 返回值 |
| --- | --- | --- |
| `await executor.execute(call)` | 执行单个工具调用 | `ToolResult` |
| `await executor.execute_many(calls)` | **并发**执行多个 | `list[ToolResult]`，**顺序与输入严格一致** |
| `executor.execute_sync(call)` | 同步版本 | `ToolResult`（在已有事件循环内调用会抛 `ConfigError`） |

`execute_many` 的两个保证值得记：

- **顺序对齐**：返回列表的第 i 项永远是 `calls[i]` 的结果，**与完成先后无关**。
  这让上层代码可以按下标索引，不用去查 id。
- **`fail_fast=True` 时仍返回等长列表**：第一个失败后取消其余，被取消的位置填
  `ToolSkippedError` + `metadata["skipped_by_fail_fast"]=True`，而不是把列表变短。

---

## 4.8 异步工具怎么写

```python
@tool
async def fetch_data(url: str, retries: int = 1) -> str:
    """异步获取数据。

    Args:
        url: 目标地址。
        retries: 重试次数。
    """
    await asyncio.sleep(0)
    return url
```

装饰器会自动检测 `async def` 并设置 `is_async=True`。执行器会直接 `await` 它
（**不丢线程池**），因此：

- 异步工具**没有孤儿线程问题**，**超时后可以安全重试**；
- 同步入口 `tool.run(args)` 会抛 `TypeError: tool xxx is async; use arun()`。

**为什么要故意抛错而不是自动兼容？** 因为曾经有个真实的坑：
如果实现者用 `asyncio.to_thread(async_fn, args)` 去跑异步函数，
它**不会执行**这个函数，只会返回一个协程对象然后被丢掉——
调用方拿到一个 coroutine 而不是结果，**而且没有任何异常**。
这种静默失效极难排查，所以宁可大声报错。

---

## 4.9 13 个内置工具

跑一下看真实清单（默认只显示非危险工具，加 `--include-dangerous` 显示全部）：

```bash
python3 -m liteagent tools list --include-dangerous --allow-shell
```

| 工具 | 分组 | 危险 | 需审批 | 幂等 | 用途 |
| --- | --- | :-: | :-: | :-: | --- |
| `read_file` | fs | | | ✅ | 读文本文件，可指定行范围 |
| `write_file` | fs | 🔴 | | ✅ | 写文件 |
| `list_dir` | fs | | | ✅ | 列目录 |
| `search_files` | fs | | | ✅ | 正则搜索文件内容 |
| `delete_file` | fs | 🔴 | | ❌ | 删文件（需 `confirm=True`） |
| `run_shell` | shell | 🔴 | 🔴 | ❌ | 跑 shell 命令（默认禁用） |
| `python_exec` | code | 🔴 | 🔴 | ❌ | 子进程执行任意 Python |
| `python_eval` | code | | | ✅ | 受限表达式求值（AST 白名单） |
| `run_tests` | code | 🔴 | | ❌ | 跑 unittest 并返回摘要 |
| `web_search` | web | | | ✅ | 网页搜索 |
| `fetch_url` | web | | | ✅ | 抓网页正文 |
| `remember` | memory | | | ✅ | 写入长期记忆 |
| `recall` | memory | | | ✅ | 检索长期记忆 |

### 4.9.1 安全设计（面试重点）

内置工具会在**真实文件系统**上操作，所以安全是硬需求。本项目有四层防护：

**① 路径沙箱 `PathSandbox`**

```python
from liteagent.tools.builtin.files import PathSandbox, make_file_tools

sandbox = PathSandbox("/tmp/work")      # root 必填！
registry = ToolRegistry(make_file_tools(sandbox))
```

- `root=None` 直接抛 `ConfigError`——**不允许隐式使用当前目录**。
  理由很直接：如果默认是 cwd，`delete_file` 就可能删掉调用方的真实项目文件。
- 绝对路径、`..`、指向外部的 symlink 一律拒绝，抛 `SandboxViolationError`。
- `~` **不做 expanduser**（否则 `~` 就成了一条逃逸通道）。
- `allow_read_outside=True` **只放宽读**，写路径永远被限制在 root 内。

**② 沙箱通过闭包注入，不出现在 schema 里**

```python
def make_file_tools(sandbox: PathSandbox) -> list[Tool]:
    def _read_file(path: str) -> str: ...   # sandbox 从闭包捕获
    return [make_function_tool(_read_file, ...)]
```

**模型看到的参数里没有 `sandbox`**——它无法从参数侧把路径指到别处。
这比"给模型一个 sandbox 参数然后校验"强得多：**攻击面直接不存在**。

**③ `run_shell` 的双重门控**

- 默认**禁用**：调用会返回 `shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)`
  ——注意工具**依然可见**（模型知道有这个能力），只是调用被拒。
  > 为什么"可见但被拒"比"隐藏"好？因为隐藏会让模型反复猜"为什么没有 shell 工具"，
  > 而"可见但给出可读的拒绝理由"是一句它能理解的反馈。
- 即使开启，也有 `SHELL_DENY_PATTERNS` 拒绝列表（`rm -rf /`、`mkfs`、`dd if=`、
  fork 炸弹、`sudo`、`curl ... | sh` 等 12 条正则）命中即抛 `SandboxViolationError`。

**④ `python_eval` 的 AST 白名单 + 复杂度闸**

不是"黑名单禁用 `eval`"，而是**只允许白名单里的语法节点和函数**：
- 允许：算术/比较/布尔运算、`Call`（仅白名单函数如 `len`/`sum`/`sorted`）、
  `List`/`Dict`/`Subscript`/`Slice`/`IfExp`、只读属性访问。
- 拒绝：`__import__`、`eval`、`getattr`、`setattr`、`type`、`vars` 等。
- **复杂度闸**：`10**10**10`、`range(10**7)` 这类会让进程卡死的表达式被直接拒绝。

> ⚠️ **诚实的边界**：`python_eval` 的 docstring 里明确写着
> "**NOT a security sandbox**"。AST 白名单能挡住常见的表达式注入，
> 但 Python 的对象模型很复杂，绕过手法可能超出白名单的覆盖。
> **真正的隔离要靠子进程 + 容器/权限**，这也是 `python_exec` 走子进程的原因。

**⑤ 危险工具需要审批**：`run_shell` 和 `python_exec` 都标了 `requires_approval=True`，
配合 4.7.5 的 fail-closed 审批策略。

---

## 4.10 本章小结

1. **`@tool` 把 type hints + docstring 反射成 JSON Schema**，一份 schema 双向使用
   （给模型看 + 执行前校验）。
2. **三个反直觉的坑**：bool 与 int 的判定顺序、`Annotated` 不能用 `isinstance`、
   不支持的类型必须"降级 + 留痕"。
3. **`required` 有明确公式**，`Optional` 和 `default` 都会让它变可选，
   `**kwargs` 直接拒绝。
4. **执行器是安全边界**：默认不重试、fail-closed 审批、路径沙箱、闭包注入、AST 白名单。
5. **并发有三层**：并发信号量 / 线程池 / 单工具串行锁，获取顺序不能颠倒。

### 自测题

1. 为什么 `bool` 必须先于 `int` 判定？项目里哪两处处理了这个问题，性质有何不同？
2. 一个参数什么时候会被写进 `required`？`Param(default=...)` 和签名默认值有什么区别？
3. 同步工具超时后为什么不能重试？框架怎么"诚实"地处理这个情况？
4. `**kwargs` 为什么被拒绝？逃生舱是什么？
5. 审批策略没配置时，需要审批的工具会被执行吗？为什么这样设计？
6. 沙箱为什么用闭包注入而不是作为参数传给模型？

<details>
<summary>参考答案</summary>

1. `isinstance(True, int)` 为真，布尔是整数的子类；注解分支用 `is`（顺序不改变结果但作为冻结契约），值推断分支用 `isinstance`（顺序真的会改变结果）。
2. 没有默认值、不是 Optional、没有 `Param(default=)`；签名默认值影响 `required` 但**不会**生成 schema 的 `default` 键，只有 `Param(default=)` 会。
3. 线程仍在运行且不可中断，重试会导致两个线程同时写同一资源；框架打 `orphan_thread=True` 标记并记 warning，同时提供协作式取消通道让工具自己退出。
4. JSON Schema 无法表达"任意键"；可以显式传 `parameters=` 手写 schema 接管。
5. 不会，拒绝执行——fail-closed，"没有审批人"不等于"审批通过"。
6. 让模型**根本无法**从参数侧指定路径，攻击面直接不存在，比"传参后校验"更可靠。

</details>

---

**下一章**：[第 5 章 · ReAct 循环](05-react-loop.md) —— 整个项目的心脏。
