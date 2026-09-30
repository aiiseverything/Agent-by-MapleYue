# 第 3 章 · LLM 抽象层：统一多模型 API

> **本章目标**：理解为什么要给大模型套一层抽象，以及本项目怎么做到的
> （对应简历「LLM 层统一多模型 API」）。学完你能：
> ① 一行代码切换模型厂商；② 看懂 OpenAI 与 Anthropic 的协议差异；③ 用 `ScriptedLLM` 做确定性测试。

---

## 3.1 问题：为什么需要"抽象层"

假设你的代码直接调 OpenAI：

```python
import openai
resp = openai.ChatCompletion.create(model="gpt-4", messages=msgs, tools=tools)
text = resp.choices[0].message.content
```

现在产品说"换成 Claude，便宜些"。你去看 Anthropic 的文档，发现**长得完全不一样**：

| 你可能以为一样的东西 | OpenAI | Anthropic |
| --- | --- | --- |
| 系统提示词 | `messages` 里放一条 `role="system"` | **顶层 `system` 参数** |
| 工具 schema 字段名 | `parameters` | `input_schema` |
| 模型说要调工具 | `tool_calls[].function.arguments`（**JSON 字符串**） | content block `{"type":"tool_use", "input":{...}}`（**已解析对象**） |
| 工具结果怎么回传 | 一条 `role="tool"` 消息 | `role="user"` 里的 `tool_result` block，**连续多个必须合并进同一条 user 消息** |
| 结束原因 | `finish_reason` | `stop_reason`，取值也不一样（`end_turn`/`tool_use`/`max_tokens`） |
| token 计数字段 | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` |

**如果你把厂商细节写进了业务代码，每换一家就要改一遍，而且改错的地方往往很隐蔽**
（比如忘了合并 `tool_result`，API 直接 400）。

### 解法：把"变化的部分"关进一层

```
      你的业务代码（Agent 循环 / 工具 / 记忆）
                     │
                     │  只认这一个接口
                     ▼
        ┌────────────────────────────┐
        │  LLMClient（抽象接口）      │   ← 稳定
        └────────────┬───────────────┘
                     │
     ┌───────────────┼───────────────┬──────────────┐
     ▼               ▼               ▼              ▼
 OpenAI 适配器   Anthropic 适配器  DeepSeek 适配器  ScriptedLLM
     │               │               │              │
     ▼               ▼               ▼              ▼
  各家 SDK/HTTP 协议（这些差异被适配器吃掉了）
```

**这就是"依赖倒置"**：业务代码依赖抽象，而不是依赖具体的厂商实现。
换模型 = 换一个适配器，业务代码**一行都不用改**。

> 面试可以这么讲：我在 LLM 层用适配器模式把厂商协议差异收敛在一个类里。
> 上层的 ReAct 循环只认 `LLMClient` 这一个接口，所以新增一家厂商只需要写一个适配器，
> 不碰任何控制流代码——这也是为什么我能把"文本 ReAct 模式"和"原生 function calling"
> 归约到同一个数据结构上。

---

## 3.2 接口长什么样

### 3.2.1 `LLMClient`：只有一个抽象方法

这一点很多人会意外——一个"统一多模型"的接口，**抽象方法只有 1 个**：

```python
class LLMClient(ABC):
    supports_tool_calling: bool = False   # 能力标志：这家模型支持原生工具调用吗
    requires_api_key: bool = True         # 能力标志：需要 API key 吗

    @property
    def model(self) -> str: ...           # 模型名

    def count_tokens(self, text: str) -> int: ...        # 估算 token 数
    def resolve_mode(self, *, has_tools: bool) -> str:   # 决定用 native 还是 text 模式

    @abstractmethod
    async def achat(self, messages, *, tools=None, tool_choice=None,
                    temperature=None, max_tokens=None, stop=None, **kwargs) -> LLMResponse:
        """唯一的抽象方法：发一次请求，拿一个回应。"""

    def chat(self, messages, **kwargs) -> LLMResponse:    # 同步镜像
        return run_sync(lambda: self.achat(messages, **kwargs))

    async def astream_chat(self, messages, **kwargs): ... # 流式（默认抛 NotImplementedError）
    async def aclose(self) -> None: ...                   # 资源释放（默认 no-op）
```

**为什么只需要一个方法？** 因为"调模型"这件事的本质就是：
**给我一串消息，还我一个回应**。其他所有东西（重试、事件、token 计数、模式判定）
都可以在基类里统一实现。

这个 `resolve_mode` 就是第 1 章讲的那个判定，它只有一行：

```python
return "native" if (self.supports_tool_calling and has_tools) else "text"
```

**有工具、且模型支持原生工具调用 → 用 native；否则退回文本 ReAct。**

### 3.2.2 `BaseLLMClient`：把公共部分一次写完

真实 provider 都继承 `BaseLLMClient`，它把四件事统一实现了：

| 能力 | 说明 |
| --- | --- |
| **重试 + 退避** | 只对 `retryable=True` 的异常重试；退避公式统一；尊重服务端给的 `Retry-After` |
| **事件发射** | 每次请求/响应/失败都发事件（`llm_request` / `llm_response` / `llm_error`），供可观测性使用 |
| **token 计数缓存** | 同一段文本的 token 数缓存起来（容量 2048，超限整体清空） |
| **参数解析** | "显式传入的参数"优先于"配置里的默认值" |

**于是子类只需要写"怎么拼请求体、怎么解析响应"**——大约两个方法。

> 这是模板方法模式（Template Method）：基类定好骨架，子类填变化的部分。
> 面试可以讲：我把重试、事件、计数这些**横切关注点**上提到基类，
> 子类只留协议适配，这样新增一家 provider 的代码量大约是 50 行。

---

## 3.3 四个 provider 适配器

本项目的继承结构（`__mro__` 实测）：

```
HTTPChatClient(BaseLLMClient)                 ← 通用的 HTTP 客户端
├── OpenAIChatClient            name="openai"
│   └── OpenAICompatibleClient  name="openai-compatible"   ← 强制要求显式 base_url
│       └── DeepSeekChatClient  name="deepseek"
├── AnthropicChatClient         name="anthropic"
EchoLLM(BaseLLMClient)          name="echo"         ← 离线占位，不联网
ScriptedLLM(BaseLLMClient)      name="scripted"     ← 离线脚本模型（测试核心）
```

**最能说明"抽象有效"的证据是 `DeepSeekChatClient` 的全部实现：**

```python
class DeepSeekChatClient(OpenAICompatibleClient):
    name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
```

**两行。** 因为 DeepSeek 的 API 与 OpenAI 兼容，所以只要继承 + 改个默认地址。
这就是抽象层的价值：**协议相同 → 零成本接入**。

### 3.3.1 `OpenAICompatibleClient` 的一个安全设计

它把父类（OpenAI）的默认 base_url **清空成 `""`，并在构造时校验**：

```python
# 缺 base_url 直接抛 ConfigError
```

为什么？因为如果忘了配 base_url，请求会**静默地打到 OpenAI 官方端点**——
你以为在用本地 vLLM，实际上把数据发给了 OpenAI，还按 OpenAI 的价格计费。
这种"静默的错"比"启动时报错"危险得多。**大声失败优于静默错误**是一条重要工程原则。

### 3.3.2 Anthropic 适配器：差异是怎么被吃掉的

这是全项目最有教学价值的一段适配代码。看**工具结果合并**：

```python
pending_results = None   # 正在累积 tool_result 的那条 user 消息
for message in messages:
    role = message.role
    if role == "system":
        continue                                    # system 走顶层字段，这里丢弃
    if role == "tool":
        block = {"type": "tool_result",
                 "tool_use_id": message.tool_call_id or "",
                 "content": message.content}
        if pending_results is None:
            pending_results = {"role": "user", "content": [block]}
            payload.append(pending_results)         # 新建一条 user 消息
        else:
            pending_results["content"].append(block)  # 连续的合并进同一条 ← 关键
        continue
    pending_results = None                          # 非 tool 的消息打断合并
```

**如果少了最后那个 `pending_results = None`，或者每次都新建消息，Anthropic 会报错。**
这种细节就是"适配器必须存在"的理由——它不该污染上层。

`stop_reason` 的映射表也是冻结的：

```python
{"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
 "stop_sequence": "stop", "pause_turn": "stop"}
```

映射之后，上层的 ReAct 循环只看到统一的 `finish_reason`，
不需要知道它是 `end_turn` 还是 `stop_sequence`。

---

## 3.4 Transport：为什么 HTTP 还要再抽一层

如果你觉得"适配器已经够了"，那这一层会让你更吃惊——**HTTP 发送本身也被抽象了**：

```python
class Transport(ABC):
    @abstractmethod
    def send(self, request: HTTPRequest) -> HTTPResponse: ...
    async def asend(self, request) -> HTTPResponse:
        return await asyncio.to_thread(self.send, request)   # 默认实现
```

三个实现，优先级与降级：

```
httpx（如果有） → requests（如果有） → urllib（stdlib，永远可用）
```

本项目实测优先用 `httpx`；如果环境里没有 httpx/requests，自动落到 `urllib.request`，
功能完全一样。

### 3.4.1 为什么值得

**理由一：让"整条 LLM 链路"可以离线测试。**

这是最关键的理由。如果 provider 里直接写 `requests.post(...)`，你就**没法测试**
"收到 429 时会不会重试"——总不能真去把 OpenAI 打限流吧？

有了 Transport 抽象，测试里塞一个假传输层就行：

```python
transport = FakeTransport(status=429, body="slow down")
client = OpenAIChatClient(LLMConfig(provider="openai", api_key="k"), transport=transport)
# 现在可以精确断言：抛的是 LLMRateLimitError，且 retry_after_s 正确
```

**理由二：错误映射集中在一处。**

`map_http_error(status, body, ...)` 是一张冻结的表：

| HTTP 状态 | 映射成 | 可重试？ |
| --- | --- | --- |
| 401 / 403 | `LLMAuthError` | ❌ |
| 429 | `LLMRateLimitError`（解析 `Retry-After`） | ✅ |
| 408 / 409 / 425 | `LLMConnectionError` | ✅ |
| 400 / 404 / 413 / 422 | `LLMBadRequestError` | ❌ |
| 5xx | `LLMConnectionError` | ✅ |

**"错在哪一类"决定了要不要重试**——这是一个贯穿全项目的设计主线（第 5、8 章还会遇到）。

> 注意 `map_http_error` 是**返回**异常对象而不是直接抛出，调用方写 `raise map_http_error(...)`。
> 这样测试可以直接断言返回类型，不用 `assertRaises` 包一层。

---

## 3.5 一行代码切换模型：注册表

```python
from liteagent.llm import get_llm

llm = get_llm("echo")                                             # 离线占位
llm = get_llm("openai:gpt-4o-mini")                               # OpenAI
llm = get_llm("anthropic:claude-sonnet-4-5")                      # Anthropic
llm = get_llm("deepseek:deepseek-chat")                           # DeepSeek
llm = get_llm("openai-compatible:qwen@http://localhost:8000/v1")  # 本地 vLLM/Ollama
```

配置字符串的语法（冻结）：

```
provider[:model][@base_url]
```

注意它**先切 `@` 再切 `:`**——因为 base_url 里既可能有 `:`（端口）也可能有 `/`（路径）。
如果先切 `:`,`http://localhost:8000/v1` 就会被切坏。**这就是读代码比读文档更可靠的地方。**

### 环境变量（key 的解析顺序）

```
LLMConfig.api_key  →  LITEAGENT_API_KEY  →  <PROVIDER>_API_KEY
```

`<PROVIDER>` 是大写的 provider 名（`-` 换成 `_`），例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`。

安全细节：`LLMConfig.to_dict()` 会把 `api_key` 脱敏成 `"***"`，
所以打印配置不会泄露密钥。**这个习惯值得你在自己项目里养成。**

---

## 3.6 `ScriptedLLM`：把模型变成可编程的

这是本项目最有教学价值的组件。它是"离线确定性测试"的地基。

### 3.6.1 基本玩法：当一个"剧本播放器"

```python
from liteagent.llm import ScriptedLLM, ScriptedResponse

llm = ScriptedLLM([
    ScriptedResponse.tool("read_file", {"path": "a.py"}),   # 第 1 次调用返回这个
    ScriptedResponse.text("我看完了，文件里有个 bug。"),      # 第 2 次调用返回这个
])
```

**每次 `achat` 就按顺序吐一条**。循环问 3 次就给你 3 条。

### 3.6.2 构造器全集（逐字来自代码）

| 构造器 | 用途 |
| --- | --- |
| `.text(content)` | 返回一段纯文本（终结） |
| `.tool(name, arguments)` | 返回一个工具调用（native 模式） |
| `.tool_raw(name, raw_arguments)` | 返回一个**参数是非法 JSON** 的工具调用（测容错） |
| `.tools(*calls)` | 一次返回**多个**工具调用（测并发） |
| `.react(thought=..., action=..., action_input=..., final=...)` | 渲染成文本 ReAct 格式 |
| `.error(exc)` | 让这次调用**抛异常**（测重试） |

`.react()` 渲染出的原文（实测）：

```python
ScriptedResponse.react("need numbers", action="add", action_input={"a": 1, "b": 2}).content
# => 'Thought: need numbers\nAction: add\nAction Input: {"a": 1, "b": 2}'
```

注意 `action` **优先于** `final`——两者同时给，`action` 生效。这个行为与解析器
「Action 优先于 Final Answer」的规则一致。

### 3.6.3 队列耗尽了怎么办：三种策略

| 配置 | 行为 |
| --- | --- |
| `loop=True` | **从头再来**，永不报错。适合"这个响应会反复用到"（如摘要压缩） |
| `strict=True`（默认） | 抛 `ScriptedExhaustedError`。**这是好事**——它证明循环次数符合预期 |
| 两个都关 | 打一条 warning，返回空响应 |

`loop=True` **优先于** `strict=True`：两者都为真时不会抛异常。

### 3.6.4 反向断言：检查"模型收到了什么"

这是它比普通 mock 强的地方——**它记录了每一次调用的完整输入**：

```python
llm.calls               # list[ScriptedCall]，每次调用追加一条
llm.calls[0].messages   # 第 1 轮模型看到的完整消息列表（浅拷贝，不会被后续轮次污染）
llm.calls[0].kwargs     # 只装 temperature/max_tokens/tool_choice/stop（None 也保留）
llm.tool_names_seen(0)  # 第 1 轮传给模型的工具名列表；文本模式返回 []
llm.call_count          # 调用了几次
llm.assert_exhausted()  # 断言剧本被精确消费完
```

**`calls[i].messages` 是最有价值的**：它让你能直接断言
"记忆有没有被正确注入 prompt""工具 schema 有没有传给模型"这类问题，而不是间接猜测。

看一个真实例子：

```python
# 断言"用户的问题在送给模型的消息里恰好出现一次"
hits = [i for i, m in enumerate(llm.calls[0].messages) if "我的问题" in m.content]
assert len(hits) == 1
```

> 这个断言在项目开发过程中**真的抓出过一个 bug**：早期版本里用户消息被发了两遍。
> 见 `docs/BUILD_LOG.md` 的对抗性审计一节。

### 3.6.5 一个有意思的实现细节

`ScriptedResponse` 有一个字段叫 `error`，同时还要有一个构造器也叫 `error`。
但 `@dataclass` 会把类体里的 `error = classmethod(...)` 当成**字段默认值**抓走——
于是每次 `achat` 都会试图抛出一个"方法"作为异常。

本项目的解法是在 dataclass 处理完之后再挂回去：

```python
ScriptedResponse.error = classmethod(_scripted_error)   # 类体外面挂
```

**运行期两者互不干扰**：`ScriptedResponse.error` 拿到构造器，实例的 `resp.error` 拿到字段值。

> 这个坑值得记：**dataclass 会把所有带注解的类属性当成字段**。
> 面试时如果被问到"你实现 ScriptedLLM 时遇到什么坑"，这是一个很具体的答案。

---

## 3.7 流式

```python
async for chunk in llm.astream_chat(messages):
    print(chunk.delta, end="", flush=True)
```

`LLMStreamChunk` 有四个字段：`delta`（本次新增的文本）、`tool_call_delta`、
`finish_reason`、`index`。

**一个诚实的说明**：真实 provider 的 HTTP 客户端**没有实现流式**
（调用会抛 `NotImplementedError`，调用方应回退到 `achat`）。
原因写在代码里：`Transport` 抽象只有"整段响应"这一种形态，
硬要"伪流式"就只能假装在流，不如不做。

**只有 `ScriptedLLM` 真正实现了流式**，用来测试上层代码在流式下的行为。

> 这条要如实说。面试时被问"流式做了吗"，答"框架的流式接口和事件顺序有实现并用
> `ScriptedLLM` 完整测试过，真实 provider 的 SSE 解析没做，因为传输层抽象不支持"——
> 比含糊其辞好得多。详见 `docs/VERIFICATION.md`。

---

## 3.8 本章小结

1. **抽象层的价值**：把厂商协议差异关进适配器，上层只认一个接口。
   证据是 `DeepSeekChatClient` 只有两行。
2. **`LLMClient` 只有一个抽象方法 `achat`**，其余都在基类统一实现。
3. **Transport 抽一层是为了可测试**——用假传输层就能精确测试 429 重试、错误映射。
4. **错误分类决定重试策略**：`retryable` 是一个贯穿全项目的核心概念。
5. **`ScriptedLLM` 不只是 mock**：它记录每次调用的输入，支持反向断言。

### 自测题

1. 换一家新模型厂商，你需要改多少代码？为什么？
2. `LLMClient` 为什么只需要一个抽象方法？
3. Anthropic 的 `tool_result` 为什么要合并进同一条 user 消息？不合并会怎样？
4. `ScriptedLLM` 的 `loop=True` 和 `strict=True` 同时开，哪个生效？
5. 为什么 `OpenAICompatibleClient` 要强制要求显式 base_url？

<details>
<summary>参考答案</summary>

1. 只写一个适配器类；因为上层只依赖 `LLMClient` 抽象，协议差异被适配器吃掉了。
2. 因为"调模型"的本质就是"给一串消息、还一个回应"，其余都是可以统一实现的横切逻辑。
3. 这是 Anthropic 的 API 约定：连续的工具结果必须作为同一条 user 消息里的多个 block；不合并会被服务端拒绝。
4. `loop=True` 生效，永不抛 `ScriptedExhaustedError`。
5. 防止忘记配置时静默请求到 OpenAI 官方端点——那既泄露数据又产生费用；宁可启动时报错。

</details>

---

**下一章**：[第 4 章 · 工具系统](04-tools.md) —— `@tool` 装饰器怎么凭空生成 JSON Schema。
