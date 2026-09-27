from __future__ import annotations

"""框架异常体系（依赖图中的 **L0**，全项目最底层）。

为什么异常必须放在最底层：LLM / tools / memory / agent / multiagent / cli 每一层都要抛错与捕获，
把异常独立成既不 import 三方库、也不 import 任何 liteagent 模块的 L0，就保证了
"任何层都能 import 它，而它不引入任何依赖"（§1.1），这也是 `test_zero_dependency.py` 能成立的前提。

三个属性构成重试/诊断协议：

* ``retryable``（类属性，默认 False）——**唯一**决定工具层与 LLM 层是否重试的开关（§3.4 重试白名单）。
  默认 False 是刻意的保守选择：只有明确知道"重发一次不会破坏状态"的异常才置 True。
  重试一个已产生副作用的操作（写文件、跑 shell）比直接失败更糟。
* ``retry_after_s``（类属性，默认 None）——服务端建议的等待秒数（如 HTTP ``Retry-After``）。
  它是**重试间隔的下限**而不是重试开关；None 表示交给调用方的退避策略。
* ``cause``（实例属性）——原始异常，只用于链式诊断，通过 ``to_dict()["cause"]`` 暴露，不参与控制流。

**默认 message 的约定（本实现的关键裁决，见文件末尾 SPEC-AMBIGUITY）**：规范里所有抛出点都只传
字段、不传 message（例如 ``ToolExecutionError(tool_name=..., call_id=..., cause=e)``）。若 message
默认是空串，``ToolResult.failure()`` 回灌给模型的文本就只剩 ``"ERROR(ToolExecutionError): "``
—— 模型拿到一条没有任何信息的错误，只能靠猜，这与 §4.3"失败必须带非空 content，否则模型无法
自纠正"和 §13 红线 12 的意图相反。因此每个子类在 ``message`` 为空时用**自身字段**渲染一句默认文案；
唯一例外是 ``ToolApprovalDeniedError``，它的文案被 §7.4.1 步骤 4.5 逐字冻结。

注意：本模块**不得 import 任何 liteagent 内部模块**（§1.2 第 2 行的职责边界）。
"""

from typing import Any

__all__ = [
    "LiteAgentError",
    "ConfigError",
    "SerializationError",
    "ScriptedExhaustedError",
    "SandboxViolationError",
    "LLMError",
    "LLMAuthError",
    "LLMBadRequestError",
    "LLMResponseFormatError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolDefinitionError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolSkippedError",
    "ToolApprovalDeniedError",
    "ToolRetryExhaustedError",
    "MemoryStoreError",
    "ReActParseError",
    "AgentError",
    "MaxStepsExceededError",
    "RepeatedActionError",
    "BudgetExceededError",
    "RunTimeoutError",
    "AgentAbortedError",
    "MultiAgentError",
    "DelegationError",
    "MaxDepthExceededError",
    "CycleDetectedError",
    "VersionConflictError",
]


def _default_message(prefix: str, **fields: Any) -> str:
    """把抛出点已知的结构化字段渲染成一句可读的默认 message。

    只在调用方没有显式给 message 时使用。空值（None/""/0/[]/{}）不渲染 —— 它们的缺席本身就是
    信息（例如 ``LLMConnectionError(status_code=None)`` 表示"压根没拿到状态码"），
    渲染成 ``status_code=None`` 只是噪声。

    字段顺序即 kwargs 书写顺序，因此同参数必然生成同文本（可断言、可 diff）。
    """
    parts = []
    for key, value in fields.items():
        if value is None or value == "" or value == 0 or value == [] or value == {}:
            continue
        parts.append(f"{key}={value!r}")
    return f"{prefix}: {', '.join(parts)}" if parts else prefix


class LiteAgentError(Exception):
    """所有框架异常的基类。

    构造规则（§3.1 冻结）：
    1. ``message`` 是位置参数且默认 ``""``，因此 ``raise ToolNotFoundError("read_file")`` 合法
       —— 子类把"表里列出的额外字段"排在 ``message`` 之前（顺序照 §3.3 的字段表）。
    2. ``context`` 里的值必须是 JSON 可序列化类型（写入 trace / JSONL 前由 ``config.to_jsonable`` 兜底）。
       本模块是最底层，不能 import config，所以这里**不做**转换，只做浅拷贝。
    3. 框架不允许在成功路径上捕获它之后静默吞掉；只有文档明确写了"降级"的地方才可以（§13 红线 10）。
    """

    # 类属性：该类异常默认是否可重试。子类按需覆盖（§3.4 是唯一真值源）。
    retryable: bool = False
    # 类属性：建议的重试间隔秒数，None 表示由调用方的退避策略决定。
    retry_after_s: float | None = None

    message: str
    context: dict[str, Any]
    cause: BaseException | None

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # 传给 Exception 的是 message 而不是所有字段：这样 args 保持可读，
        # 子类的额外字段是普通属性，不污染 args（否则日志里会多出一长串元组）。
        super().__init__(message)
        self.message = message
        # 浅拷贝：异常对象常被长期持有（写进 metadata["error_context"]、trace），
        # 不能与调用方共享同一个 dict（§13 红线 11 的同一精神）。
        self.context = dict(context) if context else {}
        # 只存属性、不写 self.__cause__：§3.1 明确 cause 只用于"链式诊断"且只经 to_dict 暴露；
        # 设 __cause__ 会改变解释器的异常链打印行为，属于规范没要求的副作用。
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """-> {"type": <类名>, "message": str, "context": dict, "retryable": bool,
                "cause": str | None}

        §3.1 冻结的键（第 5 个键由 §3.1 规则 4 追加）。**不**展开子类的额外字段：
        额外字段是给代码看的，to_dict 是给 trace 看的，两者的读者不同。
        总是复制 context，防止调用方通过返回值改到异常内部状态。
        """
        return {
            "type": type(self).__name__,
            "message": self.message,
            "context": dict(self.context),
            "retryable": bool(self.retryable),
            "cause": None if self.cause is None else str(self.cause),
        }

    def __str__(self) -> str:
        """有 context 时输出 'message (k=v, k2=v2)'，保证日志可读（§3.1）。"""
        if not self.context:
            return self.message
        joined = ", ".join(f"{k}={v}" for k, v in self.context.items())
        # message 为空时不留一个前导空格（" (a=1)" -> "(a=1)"）。
        return f"{self.message} ({joined})" if self.message else f"({joined})"


# --------------------------------------------------------------------------------------
# 配置 / 序列化 / 脚本 / 沙箱
# --------------------------------------------------------------------------------------


class ConfigError(LiteAgentError):
    """配置缺失或非法：``provider`` 未知、YAML 需要但不可用、``semaphore`` 同 key 不同 value 等。

    没有"额外字段"可渲染，所以默认 message 保持空串 —— 调用方必须自带说明（例如
    "unknown provider 'x'; available: [...]"）。
    """

    retryable = False


class SerializationError(LiteAgentError):
    """``from_dict`` 收到缺字段/错类型。

    ``target`` 用 "类名.字段名" 的形式，便于定位是哪一次反序列化出的问题。
    """

    retryable = False

    target: str

    def __init__(
        self,
        target: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("serialization error", target=target),
            context=context,
            cause=cause,
        )
        self.target = target


class ScriptedExhaustedError(LiteAgentError):
    """``ScriptedLLM`` 队列耗尽且 ``strict=True`` 且 ``loop=False``（§6.6）。

    ``consumed`` 是已消费的响应数，用于断言"恰好消费了几条"。
    """

    retryable = False

    consumed: int

    def __init__(
        self,
        consumed: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("scripted responses exhausted", consumed=consumed),
            context=context,
            cause=cause,
        )
        self.consumed = consumed


class SandboxViolationError(LiteAgentError):
    """文件/命令工具越出沙箱根。

    **denylist 命中时**（§7.5）：``path=command``、``root="denylist:<reason>"``，
    这样才能复用同一个异常类而不新增类型。

    注意它按 §3.2 的异常树**直接继承 ``LiteAgentError``**（不在 ``ToolError`` 下），
    executor 对它是"原样保留类型"而不是包成 ``ToolExecutionError``（§7.4.1 步骤 5.e）。
    """

    retryable = False

    path: str
    root: str

    def __init__(
        self,
        path: str = "",
        root: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("path escapes sandbox root", path=path, root=root),
            context=context,
            cause=cause,
        )
        self.path = path
        self.root = root


# --------------------------------------------------------------------------------------
# LLM 层（§6）
# --------------------------------------------------------------------------------------


class LLMError(LiteAgentError):
    """LLM 层异常基类。默认不可重试 —— 子类按 §3.4 白名单逐条覆盖。"""

    retryable = False


class LLMAuthError(LLMError):
    """HTTP 401/403、缺 api_key。重试没有意义（凭据不会自己变好）。"""

    retryable = False

    status_code: int

    def __init__(
        self,
        status_code: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("LLM authentication failed", status_code=status_code),
            context=context,
            cause=cause,
        )
        self.status_code = status_code


class LLMBadRequestError(LLMError):
    """HTTP 400/404/413/422（以及其它 4xx）。请求体本身有问题，重发同样会被拒。"""

    retryable = False

    status_code: int
    body: str

    def __init__(
        self,
        status_code: int = 0,
        body: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("LLM rejected the request", status_code=status_code),
            context=context,
            cause=cause,
        )
        self.status_code = status_code
        self.body = body


class LLMResponseFormatError(LLMError):
    """响应 200 但缺 ``choices``/``content`` 等必需字段。保留 ``body`` 供排查。"""

    retryable = False

    body: str

    def __init__(
        self,
        body: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # body 可能是几 KB 的 HTML 错误页，不塞进 message（它会一路进 ToolResult/to_dict）；
        # 需要原文的调用方直接读 .body。
        super().__init__(message or "malformed LLM response", context=context, cause=cause)
        self.body = body


class LLMRateLimitError(LLMError):
    """HTTP 429。**可重试**：限流是典型的瞬时故障。

    ``retry_after_s`` 取 ``Retry-After`` 头（§6.3 的 ``map_http_error``）。
    这里额外接受一个同名的 keyword-only 参数，是为了让传输层"构造即带延迟"；
    直接赋值（``exc.retry_after_s = 1.5``）同样有效，两条路径都支持。
    """

    retryable = True

    status_code: int

    def __init__(
        self,
        status_code: int = 0,
        *,
        retry_after_s: float | None = None,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("LLM rate limited", status_code=status_code),
            context=context,
            cause=cause,
        )
        self.status_code = status_code
        # 只在显式给出时写实例属性：无条件写会把类属性的默认值覆盖成 None，
        # 让"是否设置了建议延迟"这件事变得不可判定。
        if retry_after_s is not None:
            self.retry_after_s = float(retry_after_s)


class LLMTimeoutError(LLMError):
    """socket/read 超时、``asyncio.TimeoutError``。

    注意 §6.3：``HTTPChatClient`` 对 ``LLMTimeoutError`` **不重试**（``retry_on_timeout=False``，
    请求幂等性未知），但 ``retryable`` 在这里仍是 True —— 那是"要不要重试"的判断，
    在不同层各自表达：本类属性描述异常性质，client 类属性描述策略。
    """

    retryable = True

    timeout_s: float

    def __init__(
        self,
        timeout_s: float = 0.0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("LLM request timed out", timeout_s=timeout_s),
            context=context,
            cause=cause,
        )
        self.timeout_s = timeout_s


class LLMConnectionError(LLMError):
    """5xx、DNS/连接失败。``status_code`` 为 None 表示压根没拿到状态码（网络层就断了）。"""

    retryable = True

    status_code: int | None

    def __init__(
        self,
        status_code: int | None = None,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("LLM connection failed", status_code=status_code),
            context=context,
            cause=cause,
        )
        self.status_code = status_code


# --------------------------------------------------------------------------------------
# 工具层（§7）
# --------------------------------------------------------------------------------------


class ToolError(LiteAgentError):
    """工具层异常基类。默认不可重试（§3.4：白名单之外一律不重试）。"""

    retryable = False


class ToolNotFoundError(ToolError):
    """``registry.get()`` 未命中。

    ``available`` 是可用工具名列表：executor 必须把它渲染给模型（§7.4.1 步骤 1），
    否则模型不知道"正确名字是什么"，失去自纠正机会。默认 message 里也带一份，
    这样即使调用方只做 ``str(err)`` 也不会丢掉这条关键信息。
    """

    retryable = False

    name: str
    available: list[str]

    def __init__(
        self,
        name: str = "",
        available: list[str] | None = None,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # 复制一份：异常对象会被长期持有，不能与注册表的内部列表共享同一个 list。
        names = list(available) if available else []
        super().__init__(
            message or _default_message("tool not found", name=name, available=names),
            context=context,
            cause=cause,
        )
        self.name = name
        self.available = names


class ToolValidationError(ToolError):
    """参数 JSON Schema 校验失败。``errors`` 是逐条错误文案，会被回灌给模型（recoverable）。"""

    retryable = False

    errors: list[str]
    tool_name: str

    def __init__(
        self,
        errors: list[str] | None = None,
        tool_name: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        items = list(errors) if errors else []
        super().__init__(
            message
            or _default_message("invalid arguments for tool", tool_name=tool_name, errors=items),
            context=context,
            cause=cause,
        )
        self.errors = items
        self.tool_name = tool_name


class ToolDefinitionError(ToolError):
    """装饰器阶段无法生成 schema：``**kwargs``、重名、非法工具名（§7.1/§7.2）。

    这是**开发期**错误，模型看不到装饰器，所以不可重试、也无法自纠正。
    """

    retryable = False

    tool_name: str

    def __init__(
        self,
        tool_name: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("cannot build tool schema", tool_name=tool_name),
            context=context,
            cause=cause,
        )
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """工具函数内部抛异常，被 executor 包装（§7.4.1 步骤 5.e）。

    默认 retryable=False，但**工具作者对幂等操作可以显式置 True**（例如只读查询），
    这就是它保留可变的实例属性、而不是写死成类最终态的原因（§3.4 表格备注）。
    """

    retryable = False

    tool_name: str
    call_id: str

    def __init__(
        self,
        tool_name: str = "",
        call_id: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # 把 cause 的类型与消息渲染进默认 message：工具的原始异常（如
        # FileNotFoundError: /tmp/x）是模型唯一能据以改参数的线索，只放在 cause 属性里
        # 就等于没回灌给模型。
        detail: dict[str, Any] = {"tool_name": tool_name, "call_id": call_id}
        if cause is not None:
            detail["cause"] = f"{type(cause).__name__}: {cause}"
        super().__init__(
            message or _default_message("tool execution failed", **detail),
            context=context,
            cause=cause,
        )
        self.tool_name = tool_name
        self.call_id = call_id


class ToolTimeoutError(ToolError):
    """``asyncio.wait_for`` 超时。

    ``retryable=True`` 只对**异步**工具成立。§3.4 的 [v2 变更] 冻结了同步工具的例外：
    ``asyncio.wait_for`` 无法中断 worker 线程里的同步代码，超时后线程仍在写它的资源
    （``orphan_thread=True``），再重试等于"两个线程同时写同一份资源" -> 数据破坏。
    判定写在 executor：``if isinstance(exc, ToolTimeoutError) and not tool.spec.is_async: 不重试``。
    """

    retryable = True

    tool_name: str
    timeout_s: float

    def __init__(
        self,
        tool_name: str = "",
        timeout_s: float = 0.0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("tool timed out", tool_name=tool_name, timeout_s=timeout_s),
            context=context,
            cause=cause,
        )
        self.tool_name = tool_name
        self.timeout_s = timeout_s


class ToolSkippedError(ToolError):
    """[v2 新增] ``fail_fast=True`` 时被取消的兄弟调用，``reason="cancelled_by_fail_fast"``（§7.4.2）。

    它只用于占位 —— 结果列表必须与 ``calls`` 严格对齐，被取消的位置不能凭空消失。
    """

    retryable = False

    tool_name: str
    reason: str

    def __init__(
        self,
        tool_name: str = "",
        reason: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("tool call skipped", tool_name=tool_name, reason=reason),
            context=context,
            cause=cause,
        )
        self.tool_name = tool_name
        self.reason = reason


class ToolApprovalDeniedError(ToolError):
    """[v2 新增] HITL 审批拒绝：``requires_approval=True`` 且无 policy 或 policy 返回 False（§7.4.1 步骤 4.5）。

    **默认 message 是固定的回灌文案**："this tool requires human approval"。
    §7.4.1 步骤 4.5 冻结了文本 ``ERROR(ToolApprovalDeniedError): this tool requires human approval``，
    而该文本由 ``ToolResult.failure().content == error_text()`` 生成、其正文就是本异常的 ``message``；
    所以这个默认值不是"编出来的文案"，而是把规范里那条冻结串落在唯一可能的位置上。
    具体原因（"no approval policy configured" / "denied by policy"）放在 ``reason`` 字段里。

    这里也**不把 tool_name/reason 塞进 context**：一旦塞进去，``__str__`` 会输出
    "message (reason=...)"，§7.4.1 的冻结文案就不再成立。
    """

    retryable = False

    tool_name: str
    reason: str

    def __init__(
        self,
        tool_name: str = "",
        reason: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or "this tool requires human approval", context=context, cause=cause
        )
        self.tool_name = tool_name
        self.reason = reason


class ToolRetryExhaustedError(ToolError):
    """重试次数用尽（§7.4.1 步骤 5.g）。

    ``tool_name`` 有默认值，因为规范里的构造点只给了 ``attempts`` 与 ``last_error``。
    """

    retryable = False

    attempts: int
    last_error: LiteAgentError | None
    tool_name: str

    def __init__(
        self,
        attempts: int = 0,
        last_error: LiteAgentError | None = None,
        tool_name: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message(
                "tool retries exhausted",
                tool_name=tool_name,
                attempts=attempts,
                # 根因文本是回灌文案里最有用的一段（"为什么重试没用"）。
                last_error=None
                if last_error is None
                else (str(last_error) or type(last_error).__name__),
            ),
            context=context,
            cause=cause,
        )
        self.attempts = attempts
        self.last_error = last_error
        self.tool_name = tool_name


# --------------------------------------------------------------------------------------
# 记忆层（§8）
# --------------------------------------------------------------------------------------


class MemoryStoreError(LiteAgentError):
    """存储层内部错误：向量维度不一致、持久化文件损坏等（§8）。"""

    retryable = False

    store: str

    def __init__(
        self,
        store: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("memory store failure", store=store),
            context=context,
            cause=cause,
        )
        self.store = store


# --------------------------------------------------------------------------------------
# 解析（文本 ReAct / Plan JSON）
# --------------------------------------------------------------------------------------


class ReActParseError(LiteAgentError):
    """文本模式解析不出 Action/Final Answer，或 Plan JSON 非法（§9.3）。

    ``offset`` 是出错位置（JSON 解析失败时为 ``JSONDecodeError.pos``，其它情况为 0），
    ``raw`` 保留原文以便回灌给模型让它自己修（§9.4.2 的 parse-error 自纠正分支）。
    """

    retryable = False

    raw: str
    offset: int
    reason: str

    def __init__(
        self,
        raw: str = "",
        offset: int = 0,
        reason: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # raw 可能很长，只在默认 message 里带一小段（回灌全文由调用方决定，见 §7.4.1 步骤 4
        # 对"前 200 字符"的要求 —— 那里由 types.ToolCall.from_arguments_json 负责拼）。
        super().__init__(
            message
            or _default_message(
                "cannot parse model output",
                reason=reason,
                offset=offset,
                raw=raw[:200],
            ),
            context=context,
            cause=cause,
        )
        self.raw = raw
        self.offset = offset
        self.reason = reason


# --------------------------------------------------------------------------------------
# Agent 层（§9）
# --------------------------------------------------------------------------------------


class AgentError(LiteAgentError):
    """Agent 层异常基类。"""

    retryable = False


class MaxStepsExceededError(AgentError):
    """达到 ``AgentConfig.max_steps`` 仍在行动（§9.4.6）。"""

    retryable = False

    max_steps: int

    def __init__(
        self,
        max_steps: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("max steps exceeded", max_steps=max_steps),
            context=context,
            cause=cause,
        )
        self.max_steps = max_steps


class RepeatedActionError(AgentError):
    """重复/无进展动作策略判定失败（§9.4.1）。

    ``action_key`` 是 ``ToolCall.canonical_key()``，``count`` 是累计次数。
    """

    retryable = False

    action_key: str
    count: int

    def __init__(
        self,
        action_key: str = "",
        count: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message("repeated action detected", action_key=action_key, count=count),
            context=context,
            cause=cause,
        )
        self.action_key = action_key
        self.count = count


class BudgetExceededError(AgentError):
    """[v2 新增] 超过 ``max_total_tokens``（``kind="total_tokens"``，§9.4.6）。"""

    retryable = False

    limit: int
    used: int
    kind: str

    def __init__(
        self,
        limit: int = 0,
        used: int = 0,
        kind: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message("budget exceeded", kind=kind, limit=limit, used=used),
            context=context,
            cause=cause,
        )
        self.limit = limit
        self.used = used
        self.kind = kind


class RunTimeoutError(AgentError):
    """[v2 新增] 超过 ``AgentConfig.max_wall_clock_s``（§9.4.6）。"""

    retryable = False

    timeout_s: float
    elapsed_s: float

    def __init__(
        self,
        timeout_s: float = 0.0,
        elapsed_s: float = 0.0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message("run timed out", timeout_s=timeout_s, elapsed_s=elapsed_s),
            context=context,
            cause=cause,
        )
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s


class AgentAbortedError(AgentError):
    """用户在 ``approval_policy`` 里主动抛它 -> 转成 ``ABORTED``；或外部要求中止（§7.4.1 步骤 4.5）。"""

    retryable = False

    reason: str

    def __init__(
        self,
        reason: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message or _default_message("agent aborted", reason=reason),
            context=context,
            cause=cause,
        )
        self.reason = reason


# --------------------------------------------------------------------------------------
# 多 Agent 层（§10）
# --------------------------------------------------------------------------------------


class MultiAgentError(LiteAgentError):
    """多 Agent 协作层异常基类。"""

    retryable = False


class DelegationError(MultiAgentError):
    """委派失败且 ``propagate_failure="raise"``（§10.3）。"""

    retryable = False

    from_agent: str
    to_agent: str

    def __init__(
        self,
        from_agent: str = "",
        to_agent: str = "",
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message("delegation failed", from_agent=from_agent, to_agent=to_agent),
            context=context,
            cause=cause,
        )
        self.from_agent = from_agent
        self.to_agent = to_agent


class MaxDepthExceededError(MultiAgentError):
    """委派深度超限或委派预算耗尽（§10.3 的 ``_check_depth``）。"""

    retryable = False

    depth: int
    max_depth: int

    def __init__(
        self,
        depth: int = 0,
        max_depth: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message("delegation depth exceeded", depth=depth, max_depth=max_depth),
            context=context,
            cause=cause,
        )
        self.depth = depth
        self.max_depth = max_depth


class CycleDetectedError(MultiAgentError):
    """委派栈中出现同名 Agent（A -> B -> A）。``stack`` 是当前委派栈快照。"""

    retryable = False

    stack: list[str]

    def __init__(
        self,
        stack: list[str] | None = None,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        items = list(stack) if stack else []
        super().__init__(
            message or _default_message("delegation cycle detected", stack=items),
            context=context,
            cause=cause,
        )
        self.stack = items


class VersionConflictError(MultiAgentError):
    """Blackboard 乐观并发写失败：``if_version`` 与当前版本不符（§10.2）。"""

    retryable = False

    key: str
    expected: int
    actual: int

    def __init__(
        self,
        key: str = "",
        expected: int = 0,
        actual: int = 0,
        *,
        message: str = "",
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message
            or _default_message(
                "blackboard version conflict", key=key, expected=expected, actual=actual
            ),
            context=context,
            cause=cause,
        )
        self.key = key
        self.expected = expected
        self.actual = actual


# --------------------------------------------------------------------------------------
# SPEC-AMBIGUITY 记录
# --------------------------------------------------------------------------------------
#
# SPEC-AMBIGUITY: §3.1 冻结了 message 默认 ""，而 §4.3 要求 ToolResult.failure() 的 content
# 必须"非空且可自纠正"、§13 红线 12 要求降级必须可观测；但规范里所有抛出点都只传字段不传 message
# （已核对全文：无任何 `raise XxxError(..., message=...)` 的示例）。若严格只保留空 message，
# 回灌给模型的文本会退化成 "ERROR(ToolExecutionError): "（实测），与"模型能自纠正"的整体精神相反。
# 裁决：子类在 message 为空时用自身字段渲染默认文案（_default_message），
# 唯一例外是 ToolApprovalDeniedError —— 它的文案被 §7.4.1 步骤 4.5 逐字冻结。
