from __future__ import annotations

"""§6.6 脚本化 LLM —— **离线确定性测试的基石**。

职责：提供一个零依赖、离线、确定性、可断言调用历史的 ``BaseLLMClient`` 实现，
让 Agent / 工具 / 记忆 / 多 Agent 的端到端测试在**没有任何 API key、没有网络**的环境下
也能驱动完整的 Thought-Action-Observation 循环。

三个设计要点（也是面试里最容易被追问的三处）：

1. **队列 + 消费语义**：每次 ``achat`` 从队首消费一条 ``ScriptedResponse``。
   队列耗尽时 ``loop=True`` 优先于 ``strict``（永不抛 ``ScriptedExhaustedError``，
   ``remaining`` 恒为 0），只有 ``strict=True and loop=False`` 才抛错 —— 这条优先级是**冻结**的
   （§6.6 规则 6），因为"从头循环"的意图必须压过"严格模式"的默认值。
2. **``call_id`` 在消费时才分配**：``ScriptedResponse.tool()`` 是 ``classmethod``，
   构造它时 ``ScriptedLLM`` 实例根本还不存在，所以那里只能留一个 ``id=""`` 的**占位**；
   真正的 ``call_{seq}`` 由 ``ScriptedLLM`` 在消费该响应时按出现顺序分配（§6.6 规则 1）。
   显式传入的 ``call_id`` 原样保留且**不消耗** seq。
3. **两条时间线都走 ``sleep_fn``**：``latency_s``（每次调用的固定开销）与
   ``delay_s``（该条响应的"模型耗时"）都必须经由 ``self._sleep``（§5.5 规则 1/16），
   且 ``latency_ms = delay_s * 1000`` 而不是真实墙钟 —— 这样"耗时"才是可断言的
   （§2.8 点名了这是 ``latency_ms`` 唯一可断言的例外）。

本模块在依赖图里是 **L2**，只允许 import ``errors`` / ``types`` / ``config`` / ``llm/{message,base}``。
它**不得** import ``agent`` 层：事件回调因此是轻量的 ``LowLevelEvent``（§6.3 的环依赖规避）。
"""

import asyncio
import json
import unittest
import warnings
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from liteagent.config import LLMConfig, to_jsonable
from liteagent.errors import ConfigError, LiteAgentError, LLMTimeoutError, ScriptedExhaustedError
from liteagent.llm.base import BaseLLMClient, LLMStreamChunk, LowLevelEvent
from liteagent.llm.message import Message
from liteagent.types import LLMResponse, ScriptedCall, TokenUsage, ToolCall

__all__ = ["ScriptedLLM", "ScriptedResponse"]

# §2.7 的冻结定义式由 ``llm/base.py`` 提供并列入其 ``__all__``（``LowLevelEvent``），这里直接复用，
# 保证"低层事件回调"全项目只有一份定义（memory/embeddings.py 因 L2 不能 import L2 才本地再声明一份）。


def _default_usage() -> TokenUsage:
    """§6.6 规则 3 的固定常量（``default_usage`` 与响应 ``usage`` 均为 None 时使用）。

    每次新建而不是共享一个模块级实例：``TokenUsage`` 是可变对象，且 Agent 会把它累加进
    ``AgentState.usage``。共享实例会让某一次调用对 usage 的原地改写泄漏到下一次调用
    （§13 红线 11 的同一精神：不要把内部可变状态泄漏出去）。
    """
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _new_tool_call(
    name: str,
    *,
    call_id: str | None,
    arguments: Mapping[str, Any] | None = None,
    raw_arguments: str | None = None,
) -> ToolCall:
    """构造一个工具调用。

    **刻意不用 ``ToolCall.create()``**：那个 classmethod 在 ``call_id is None`` 时会生成
    ``uuid4``，而 §6.6 规则 1 冻结要求占位态必须是 ``id == ""``（否则"消费时分配 call_{seq}"
    这条规则永远不会触发，测试也就无法断言确定性的 id）。
    """
    kwargs: dict[str, Any] = {"id": call_id or "", "name": name}
    if raw_arguments is not None:
        kwargs["raw_arguments"] = raw_arguments
        kwargs["arguments"] = {"__raw__": raw_arguments}
    else:
        kwargs["arguments"] = dict(arguments) if arguments else {}
    return ToolCall(**kwargs)


@dataclass
class ScriptedResponse:
    """一条预设的模型响应（§6.6）。

    字段语义（全部冻结）：

    * ``finish_reason``：``None`` -> 自动推导（有 tool_calls 则 ``"tool_calls"``，否则 ``"stop"``）。
    * ``usage``：``None`` -> 用 ``ScriptedLLM.default_usage``，再为 ``None`` 时用固定常量。
    * ``error``：非 None -> 在该次 ``achat`` 里抛出它（模拟 provider 故障）。**抛之前照样记账**。
    * ``delay_s``：模拟网络/推理延迟，语义见 §6.6 规则 5（可能被 ``config.timeout_s`` 截断）。
    * ``stream_chunks``：[v2 新增] 仅 ``astream_chat`` 用；``None`` -> ``[content]``。
    * ``stream_error``：[v2 新增] 流中断：在**最后一个 chunk 之后**抛出。
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    error: BaseException | None = None
    delay_s: float = 0.0
    stream_chunks: list[str] | None = None
    stream_error: BaseException | None = None

    # ------------------------------------------------------------------ 构造器
    @classmethod
    def text(cls, content: str, **kwargs: Any) -> "ScriptedResponse":
        """纯文本响应（无工具调用）。"""
        return cls(content=content, **kwargs)

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        call_id: str | None = None,
        content: str = "",
        **kwargs: Any,
    ) -> "ScriptedResponse":
        """单个工具调用。

        ``call_id=None`` 表示**占位**（``ToolCall.id == ""``），真正的 id 由 ``ScriptedLLM``
        在消费时分配（名字在这里拿不到实例，无法递增 seq）。
        """
        return cls(
            content=content,
            tool_calls=[_new_tool_call(name, call_id=call_id, arguments=arguments)],
            **kwargs,
        )

    @classmethod
    def tool_raw(
        cls,
        name: str,
        raw_arguments: str,
        *,
        call_id: str | None = None,
        content: str = "",
        **kwargs: Any,
    ) -> "ScriptedResponse":
        """[v2 新增] 造一个"模型给了非法 JSON 的 arguments"的响应。

        没有它，测试只能手搓 ``ToolCall`` 且极易漏掉 ``raw_arguments`` —— 而
        ``raw_arguments`` 正是"把原文回灌给模型让它自己修"的依据（§4.2 / §6.4）。
        """
        return cls(
            content=content,
            tool_calls=[
                _new_tool_call(name, call_id=call_id, raw_arguments=raw_arguments)
            ],
            **kwargs,
        )

    @classmethod
    def tools(
        cls,
        *calls: "ToolCall | tuple[str, Mapping[str, Any]]",
        content: str = "",
        **kwargs: Any,
    ) -> "ScriptedResponse":
        """多个（并发的）工具调用。元素可以是 ``ToolCall`` 或 ``(name, args)`` 二元组。"""
        built: list[ToolCall] = []
        for item in calls:
            if isinstance(item, ToolCall):
                built.append(item)
            else:
                # 只接受二元组：解包失败会自然抛 TypeError/ValueError（§6.6 规则 8 的精神：
                # 测试写错了要立刻炸，不要静默丢弃）。
                name, arguments = item
                built.append(_new_tool_call(name, call_id=None, arguments=arguments))
        return cls(content=content, tool_calls=built, **kwargs)

    @classmethod
    def react(
        cls,
        thought: str = "",
        *,
        action: str | None = None,
        action_input: Mapping[str, Any] | None = None,
        final: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedResponse":
        """渲染文本 ReAct 载荷（与 §9.3 parser 的语法严格对齐）。

        ``action`` 优先于 ``final``：这与 parser 的"Action 优先于 Final Answer"规则
        （§9.3 步骤 4）保持一致，避免"造了一个带 Action 的响应，却因为同时给了 final
        而被解析成终结"这种自相矛盾的测试夹具。
        """
        if action is not None:
            payload = json.dumps(
                dict(action_input) if action_input is not None else {},
                ensure_ascii=False,
            )
            body = f"Thought: {thought}\nAction: {action}\nAction Input: {payload}"
        elif final is not None:
            body = f"Thought: {thought}\nFinal Answer: {final}"
        else:
            body = f"Thought: {thought}"
        return cls(content=body, **kwargs)

    # 注意：`error` 构造器**不在类体内定义**，见本类下方的 `_scripted_error`。

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        """§2.2：字段全量输出（含 None）。

        为什么需要它：``types.ScriptedCall.to_dict`` 用 ``_jsonable`` 递归转换 ``response``
        字段，而 ``_jsonable`` 的第一条分支就是"有 ``to_dict`` 的 dataclass 就调它"；
        没有本方法，``calls[i].to_dict()["response"]`` 会退化成 ``repr(...)``（tests 里
        一旦有人断言 trace 结构就会静默比对到一段字符串）。

        **不提供 ``from_dict``**：``error`` / ``stream_error`` 是 ``BaseException`` 实例，
        无法从数据往返（§2.2 的 from_dict 约定针对可往返的对外数据结构，本类不是）。
        """
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "error": None if self.error is None else to_jsonable(self.error),
            "delay_s": self.delay_s,
            "stream_chunks": None if self.stream_chunks is None else list(self.stream_chunks),
            "stream_error": None if self.stream_error is None else to_jsonable(self.stream_error),
        }


def _scripted_error(cls: type[ScriptedResponse], exc: BaseException, **kwargs: Any) -> ScriptedResponse:
    """``ScriptedResponse.error(exc)`` 的实现体（模拟 provider 故障、限流、超时……）。"""
    return cls(error=exc, **kwargs)


# 让 `ScriptedResponse.error.__name__`（以及 help()/traceback 里的显示名）就是 "error"，
# 与 §6.6 声明的方法名逐字一致 —— 私有函数名只是实现细节，不该泄漏到自省结果里。
_scripted_error.__name__ = "error"


# SPEC-AMBIGUITY（规范内部冲突，必须这样解）：§6.6 的 `ScriptedResponse` 同时有
#   ①字段  `error: BaseException | None = None`
#   ②构造器 `@classmethod def error(cls, exc, **kwargs)`
# 两者同名。若把 classmethod 写在类体里，`@dataclass` 会在处理注解时用 `getattr(cls, "error")`
# 取默认值，取到的就是那个 classmethod 对象 —— 实测后果：`ScriptedResponse.text("hi").error`
# 是一个 bound method（非 None），于是 `if resp.error is not None` 恒为真，**每一次 achat 都会
# 试图抛一个 method 并炸出 `TypeError: exceptions must derive from BaseException`**。
# 裁决：类体里只留字段（默认值自然是 None），构造器在 dataclass 处理完之后用
# `classmethod(...)` 挂回类上。运行期两者互不干扰：类属性访问 `ScriptedResponse.error` 拿到构造器，
# 实例属性访问 `resp.error` 拿到字段值。两个名字都按 §6.6 逐字可用。
ScriptedResponse.error = classmethod(_scripted_error)  # type: ignore[assignment]


class ScriptedLLM(BaseLLMClient):
    """脚本化假 LLM：按预设队列返回响应。零依赖、离线、确定性、可断言调用历史。"""

    supports_tool_calling: bool = True
    requires_api_key: bool = False
    name: str = "scripted"

    def __init__(
        self,
        responses: Sequence[
            "ScriptedResponse | str | Callable[[Sequence[Message]], ScriptedResponse | str]"
        ] = (),
        *,
        model: str = "scripted-1",
        default_usage: TokenUsage | None = None,
        loop: bool = False,
        strict: bool = True,
        latency_s: float = 0.0,
        on_event: LowLevelEvent | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        # config 先建好再交给基类：基类用它解析 self._sleep / self._rng，而 model 属性读的也是它
        # （构造参数 model= 因此必须写进 config，否则 LLMResponse.model 会是 gpt-4o-mini）。
        # provider 选 "echo" 是因为它是 §5.3 里唯一"离线、不需要 key"的合法取值 —— 脚本客户端
        # 永远不会发请求，base_url/api_key 全程不被读到。
        # 显式传入 config 时以 config 为准（调用方给了更具体的信息，包括 timeout_s/sleep_fn）；
        # 此时 model= 参数被忽略，这一点写在 docstring 里而不是"猜"。
        if config is None:
            config = LLMConfig(provider="echo", model=model)
        # R-LOOP（§0.4）：这里只调基类，不创建任何 asyncio 原语 —— ScriptedLLM 不需要并发限流。
        super().__init__(config, on_event=on_event)

        self.default_usage: TokenUsage | None = default_usage
        self.loop: bool = loop
        self.strict: bool = strict
        self.latency_s: float = latency_s

        # _script 是"权威脚本"（构造 + push + extend 的累加），_queue 是待消费的那部分。
        # 分开存是为了让 reset() 能把队列恢复回去，而不用让调用方重新构造实例。
        self._script: list[Any] = list(responses)
        self._queue: list[Any] = list(self._script)

        # ---- 运行时状态（测试断言用，全部冻结）----
        self.calls: list[ScriptedCall] = []
        self.exhausted_count: int = 0
        self.events: list[tuple[str, dict[str, Any]]] = []
        #: `call_id` 占位分配的序号：一次 achat 内的多个 call 依次递增，显式 call_id 不消耗
        self._seq: int = 0
        #: 真正从脚本里消费掉的响应条数（loop 模式下会超过 len(_script)）
        self._consumed: int = 0

    # ------------------------------------------------------------------ 只读视图
    @property
    def remaining(self) -> int:
        """队列里还剩几条。``loop=True`` 时恒为 0（§6.6 规则 6：永不耗尽）。"""
        return 0 if self.loop else len(self._queue)

    @property
    def call_count(self) -> int:
        """``calls`` 的长度（= 目前发生过几次 ``achat``/``astream_chat``）。"""
        return len(self.calls)

    # ------------------------------------------------------------------ 动态喂料
    def push(self, response: "ScriptedResponse | str") -> None:
        """追加到队尾（可在运行中动态喂响应，用于测试中途改变行为）。"""
        self._script.append(response)
        self._queue.append(response)

    def extend(self, responses: Sequence["ScriptedResponse | str"]) -> None:
        """``push`` 的批量版本。"""
        for response in responses:
            self.push(response)

    def reset(self) -> None:
        """恢复到"刚构造完"的状态：队列按 _script 复原，调用历史/计数器/事件清空。

        用途：一个测试类里复用同一个实例跑多个用例。
        注意 ``_seq`` 与已消费过的 ``ScriptedResponse`` 上的 ``call_id`` 不会被清掉 ——
        那些 id 已经写进了**调用方持有**的 ``ToolCall`` 对象，按 §6.6 规则 1"只在 id == '' 时
        分配"，reset 之后不会重新编号。这是刻意的：id 只分配一次才可复现。
        """
        self._queue = list(self._script)
        self.calls.clear()  # 原地清空而不是重新赋值：外部可能持有 llm.calls 的引用
        self.events.clear()
        self.exhausted_count = 0
        self._consumed = 0
        self._seq = 0

    # ------------------------------------------------------------------ 断言辅助
    def assert_exhausted(self) -> None:
        """断言队列已耗尽且所有响应都被消费（用 ``assertEqual`` 实现，便于 unittest 报错定位）。

        为什么借道 ``unittest.TestCase().assertEqual`` 而不是裸 ``assert`` / ``AssertionError``：
        §0.3 禁止裸 ``assert`` 作为断言载体（``python -O`` 会把它们全部抹掉），而
        ``assertEqual`` 的失败信息带 diff，测试作者一眼能看出"还剩 2 条没消费"。
        """
        case = unittest.TestCase()
        case.assertEqual(0, self.remaining, "scripted queue is not exhausted")
        if not self.loop:
            # loop=True 时消费数会超过脚本长度，这条检查没有意义（且 remaining 恒为 0）。
            case.assertEqual(
                len(self._script),
                self._consumed,
                "not all scripted responses were consumed",
            )

    def last_call(self) -> ScriptedCall:
        """最近一次调用记录。无调用时抛 AssertionError（而不是 IndexError）。

        这里**故意**用 AssertionError 而非 ``unittest.TestCase``：它是"测试写错了"的用法错误，
        不是断言失败，抛一条带解释的消息比 ``list index out of range`` 可读得多。
        """
        if not self.calls:
            raise AssertionError("ScriptedLLM: no LLM call has been recorded yet")
        return self.calls[-1]

    def last_messages(self) -> list[Message]:
        """最近一次调用收到的消息（**浅拷贝**：与 ``calls[i].messages`` 同一份内容）。"""
        return list(self.last_call().messages)

    def tool_names_seen(self, index: int = -1) -> list[str]:
        """从调用历史里抽取"渲染给模型的工具名列表"（断言工具确实被暴露给模型）。

        语义冻结（§6.6）：**只**从 ``calls[index].tools`` 提取，同时兼容
        OpenAI 的 ``tools[i].function.name`` 与 Anthropic 的 ``tools[i].name`` 两种形态；
        ``tools is None``（文本模式）时返回 ``[]`` —— 绝不去解析 system prompt 猜工具名
        （那是"测试自己实现了一遍渲染逻辑"，一旦渲染变了测试会一起错）。
        """
        call = self.calls[index]
        if call.tools is None:
            return []
        names: list[str] = []
        for spec in call.tools:
            function = spec.get("function")
            raw = function.get("name") if isinstance(function, Mapping) else spec.get("name")
            if raw:
                names.append(str(raw))
        return names

    # ------------------------------------------------------------------ 事件
    def _emit(self, event_type: str, **data: Any) -> None:
        """记录事件到 ``self.events``，再照常转发给 ``on_event``。

        §6.6 只冻结了"由 _emit 记录"，没说"不要转发" —— 所以两个都做：测试既能直接断言
        ``llm.events``，又能把 on_event 接到真实 ``CallbackManager`` 上走完整 trace 链路
        （§2.7：LLM 层的三条事件唯一发射者就是 ``BaseLLMClient._emit``）。
        """
        self.events.append((event_type, to_jsonable(dict(data))))
        super()._emit(event_type, **data)

    def _emit_request(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None) -> None:
        """§6.3 冻结的 LLM_REQUEST data 键。

        先调基类 ``_prepare_request`` 记下"请求画像"，再从它读计数 —— 与
        ``BaseLLMClient._with_retry`` 的事件填法**逐字相同**，这样"真实 provider 的 trace
        长什么样，脚本客户端的 trace 就长什么样"，Agent 层的 trace 断言对两者都成立。
        LLM_RESPONSE 直接复用基类的 ``_emit_response(resp)``（``resp.latency_ms`` 已经等于
        ``delay_s * 1000``），因此本模块不重复定义那个方法、也不会覆盖基类的同名方法。
        """
        self._prepare_request(messages, tools)
        self._emit(
            "llm_request",
            messages_count=self._request_messages_count,
            tools_count=self._request_tools_count,
            retry=0,  # 脚本客户端不做重试（§3.4 的重试属于真实 provider 的传输层）
            model=self.model,
        )

    def _emit_error(self, exc: BaseException) -> None:
        """§6.3 冻结的 LLM_ERROR data 键。"""
        self._emit(
            "llm_error",
            error_type=type(exc).__name__,
            message=str(exc),
            retry=0,
        )

    # ------------------------------------------------------------------ 队列消费
    def _next_response(self, messages: Sequence[Message]) -> ScriptedResponse:
        """取队首响应并按 §6.6 规则 1 分配 ``call_id``。

        消费优先级（冻结）：正常出队 -> ``loop`` 回绕 -> ``strict`` 抛错 -> 兜底空响应。
        ``loop`` 必须排在 ``strict`` 之前，这是 §6.6 规则 6 的核心：两个都为真时 loop 赢。
        """
        if self._queue:
            item = self._queue.pop(0)
        elif self.loop and self._script:
            # loop=True：队列耗尽就从权威脚本从头再来。**永不抛 ScriptedExhaustedError**，
            # 也**不**递增 exhausted_count（重复消费不算"耗尽"）。
            self._queue = list(self._script)
            item = self._queue.pop(0)
        elif self.strict and not self.loop:
            # 唯一的抛错分支（§3.3：ScriptedExhaustedError 的唯一抛出时机）。
            self.exhausted_count += 1
            raise ScriptedExhaustedError(consumed=self._consumed)
        else:
            # strict=False（或 loop=True 但脚本为空）：不抛，但这是**降级**，必须留可观测痕迹
            # （§13 红线 12）。返回一条空文本响应：它没有任何 tool_call，Agent 会正常收尾，
            # 不会陷入"假 LLM 永远要求调工具"的死循环。
            warnings.warn(
                "ScriptedLLM: responses exhausted and strict=False; "
                "returning an empty response (no finish_reason/tool_calls)",
                RuntimeWarning,
                stacklevel=2,
            )
            return ScriptedResponse()

        # 元素可以是 Callable：调用时传 list(messages) 副本，保证被调方拿不到可被后续轮次
        # 污染的那个 list（§6.6 规则 7 的同一理由）。
        if callable(item):
            item = item(list(messages))
        if isinstance(item, str):
            item = ScriptedResponse.text(item)
        if not isinstance(item, ScriptedResponse):
            raise ConfigError(
                f"scripted response must be ScriptedResponse | str | Callable, "
                f"got {type(item).__name__}"
            )

        self._consumed += 1
        self._assign_call_ids(item)
        return item

    def _assign_call_ids(self, response: ScriptedResponse) -> None:
        """把占位 ``id == ""`` 换成 ``call_{seq}``。

        显式传入的 ``call_id`` 原样保留且**不消耗** seq —— 这条很重要：否则测试里
        "混着一个显式 id 和一个占位 id"的用例会得到取决于实现细节的编号。
        """
        for call in response.tool_calls:
            if call.id == "":
                call.id = f"call_{self._seq}"
                self._seq += 1

    def _record(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        prompt_kwargs: dict[str, Any],
        response: ScriptedResponse | None,
    ) -> None:
        """§6.6 规则 7 的冻结记账。

        * ``messages`` **必须是浅拷贝**：Agent 会把同一个 ``state.messages`` list 传进来，
          后续轮次的 ``append`` 若共享同一个 list，``calls[i].messages`` 会被静默污染，
          "第 2 轮看到了几条消息"这类断言就永远为真 —— 这类 bug 最难查。
          Message 对象本身**不深拷贝**（契约：Agent 追加消息后不得再原地修改已有的 Message）。
        * ``tools`` 逐项浅拷贝外层 dict；``kwargs`` 只装四个冻结键（为 None 也保留）。
        """
        self.calls.append(
            ScriptedCall(
                index=len(self.calls),
                messages=list(messages),
                tools=None if tools is None else [dict(t) for t in tools],
                kwargs=dict(prompt_kwargs),
                response=response,
            )
        )

    def _consume(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        prompt_kwargs: dict[str, Any],
    ) -> ScriptedResponse:
        try:
            response = self._next_response(messages)
        except LiteAgentError as exc:
            # 队列耗尽：这一次 achat 没有消费到任何响应，但属性契约是"每次 achat 追加一条"
            # （且 ScriptedCall.response 的默认值正是 None，就是为这种"没拿到响应"的记录准备的）。
            # 先记账再抛：否则"跑了多少轮才耗尽"无法断言。
            self._record(messages, tools, prompt_kwargs, None)
            self._emit_error(exc)
            raise
        self._record(messages, tools, prompt_kwargs, response)
        return response

    # ------------------------------------------------------------------ 延迟
    async def _sleep_latency(self) -> None:
        """``latency_s``：每次调用的固定开销，**无条件**走 sleep_fn、不受 timeout 约束。

        只走 ``self._sleep`` 而绝不 ``await asyncio.sleep``（§5.5 规则 1 / 红线 16），
        否则测试要么真睡、要么无法用 ``RecordingSleep`` 断言等待序列。
        """
        await self._sleep(self.latency_s)

    async def _sleep_delay(self, response: ScriptedResponse) -> None:
        """``delay_s`` 的冻结语义（§6.6 规则 5）。

        ``config.timeout_s`` 非 None 且 ``delay_s > timeout_s`` 时改成
        ``asyncio.wait_for(self._sleep(delay_s), timeout_s)`` 并抛
        ``LLMTimeoutError(timeout_s=...)``；否则直接 sleep。

        注意这是**规则 5 的字面实现**：这里不自己判断"会不会超时"，而是把判断交给 wait_for。
        当调用方注入了 ``RecordingSleep``（立即返回）时，wait_for 会直接完成、不抛错 ——
        只有真实 ``default_sleep`` 才会命中超时。这正是"超时路径可测"的写法：
        用 ``ScriptedLLM(..., config=LLMConfig(timeout_s=0.01))`` + ``delay_s=1.0`` 即可。
        """
        timeout_s = self.config.timeout_s
        if timeout_s is not None and response.delay_s > timeout_s:
            try:
                await asyncio.wait_for(self._sleep(response.delay_s), timeout_s)
            except asyncio.CancelledError:
                # M-5：CancelledError 继承 BaseException；无论哪条分支都不能把它吞掉/换掉类型。
                raise
            except (asyncio.TimeoutError, TimeoutError) as exc:
                # 3.10 的 asyncio.TimeoutError 与 builtins.TimeoutError 不是同一个类（M-5），
                # 两个都接住以覆盖不同实现细节；统一映射成 §3.2 里 retryable=True 的 LLMTimeoutError。
                raise LLMTimeoutError(timeout_s=timeout_s) from exc
        else:
            await self._sleep(response.delay_s)

    # ------------------------------------------------------------------ 响应构造
    def _to_llm_response(self, response: ScriptedResponse) -> LLMResponse:
        """把脚本响应翻译成 ``LLMResponse``（§6.6 规则 2/3）。"""
        tool_calls = list(response.tool_calls)
        finish_reason = response.finish_reason or ("tool_calls" if tool_calls else "stop")
        usage = response.usage or self.default_usage or _default_usage()
        return LLMResponse(
            content=response.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            raw=None,
            # 真实耗时**不计入**：否则 latency_ms 不可断言（§2.8 明确给它开了唯一例外）。
            latency_ms=response.delay_s * 1000.0,
        )

    # ------------------------------------------------------------------ 主入口
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
    ) -> LLMResponse:
        """消费一条脚本响应并返回 ``LLMResponse``。

        记账规则（冻结）：``calls[i].kwargs`` **只**装 ``temperature`` / ``max_tokens`` /
        ``tool_choice`` / ``stop``（为 None 也保留，便于断言"Agent 没传温度"），
        ``**kwargs`` 里的其它键既不记账也不报错 —— 真实 provider 会忽略未知键，脚本客户端
        必须同样宽容，否则"给 ScriptedLLM 和真实 client 传同一份 kwargs"的测试会炸。
        """
        prompt_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": tool_choice,
            "stop": stop,
        }
        self._emit_request(messages, tools)
        await self._sleep_latency()
        response = self._consume(messages, tools, prompt_kwargs)
        # delay_s 在抛 error 之前：模拟"慢失败"，也让 timeout 分支能被 error 响应的用例覆盖。
        await self._sleep_delay(response)
        if response.error is not None:
            self._emit_error(response.error)
            raise response.error
        out = self._to_llm_response(response)
        self._emit_response(out)
        return out

    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> AsyncIterator[LLMStreamChunk]:
        """[v2 新增，必须实现] 逐个 yield ``LLMStreamChunk``。

        冻结行为：第 i 个 chunk 的 ``index == i``；**最后一个** chunk 带
        ``finish_reason = resp.finish_reason or ("tool_calls" if resp.tool_calls else "stop")``；
        ``resp.stream_error`` 非 None 时在最后一个 chunk **之后**抛出。
        并把这次调用记入 ``self.calls``（与 ``achat`` 完全一致的记账口径）。
        """
        # 与 achat 相同地提取四个冻结键：流式路径的 calls[i].kwargs 不能少键，
        # 否则"同一段 Agent 代码在流式/非流式下记账一致"这个断言会不成立。
        tools = kwargs.get("tools")
        prompt_kwargs = {
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "tool_choice": kwargs.get("tool_choice"),
            "stop": kwargs.get("stop"),
        }
        self._emit_request(messages, tools)
        await self._sleep_latency()
        response = self._consume(messages, tools, prompt_kwargs)
        await self._sleep_delay(response)
        if response.error is not None:
            self._emit_error(response.error)
            raise response.error

        # stream_chunks is None -> [content]（**只有 None 走默认**：显式传 [] 表示"一个
        # chunk 都不发"，语义不同，不能按 falsy 合并）。
        chunks = [response.content] if response.stream_chunks is None else list(response.stream_chunks)
        finish_reason = response.finish_reason or (
            "tool_calls" if response.tool_calls else "stop"
        )
        last_index = len(chunks) - 1
        for index, delta in enumerate(chunks):
            yield LLMStreamChunk(
                delta=delta,
                index=index,
                finish_reason=finish_reason if index == last_index else None,
            )
        self._emit_response(self._to_llm_response(response))
        if response.stream_error is not None:
            # 流中断：chunk 已经吐完了，此时才炸（模拟服务端在最后一刻断开 SSE）。
            self._emit_error(response.stream_error)
            raise response.stream_error
