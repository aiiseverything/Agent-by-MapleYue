from __future__ import annotations

"""真实 provider 适配器 + 离线占位 provider（§6.4）。

**这是"LLM 层统一多模型 API"的落地文件**：所有 provider 的差异被压进四个钩子
（`_endpoint` / `_headers` / `_build_body` / `_parse_response`），发送、错误映射、重试、
事件、latency 统计由 `HTTPChatClient` 统一完成。新增一家 OpenAI 兼容的服务（DeepSeek）
因此只需要 5 行 —— 见 `DeepSeekChatClient`。

三家真实 provider 的**编解码差异**（也是最容易被面试追问的部分）：

| | OpenAI 系 | Anthropic |
|---|---|---|
| system | messages 里的 `role="system"` | 顶层 `system` 字段（**不能**留在 messages 里）|
| 工具 schema | `{"type":"function","function":{...}}` | `{"name","description","input_schema"}` |
| 助手工具调用 | `tool_calls[].function.arguments`（**JSON 字符串**）| content block `{"type":"tool_use","input":{...}}`（**已解析对象**）|
| 工具结果 | 独立 `role="tool"` 消息 | `role="user"` 的 `tool_result` block，**连续的必须合并进同一条 user 消息** |
| 结束原因 | `finish_reason` 直通 | `stop_reason` 需映射（`end_turn`->`stop`、`tool_use`->`tool_calls`、`max_tokens`->`length`）|
| usage | `prompt_tokens/completion_tokens` | `input_tokens/output_tokens` |
"""

import json
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from liteagent.config import LLMConfig
from liteagent.errors import ConfigError, LLMAuthError, LLMResponseFormatError
from liteagent.llm.base import BaseLLMClient, LLMClient, LowLevelEvent
from liteagent.llm.message import Message, Role
from liteagent.llm.transport import HTTPRequest, Transport, default_transport
from liteagent.types import LLMResponse, TokenUsage, ToolCall

__all__ = [
    "HTTPChatClient",
    "OpenAIChatClient",
    "OpenAICompatibleClient",
    "DeepSeekChatClient",
    "AnthropicChatClient",
    "EchoLLM",
    "PROVIDER_CLASSES",
]

_JSON_CONTENT_TYPE = "application/json"
#: Anthropic 要求 `max_tokens` **必填**，而 `LLMConfig.max_tokens` 默认是 None。
#: 这里给一个保守的兜底值，而不是让请求 400（调用方显式传 max_tokens 时以它为准）。
_ANTHROPIC_DEFAULT_MAX_TOKENS = 1024
_ANTHROPIC_VERSION = "2023-06-01"

#: §6.4 规则 6 冻结的 stop_reason 映射表。
_ANTHROPIC_STOP_REASONS: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "pause_turn": "stop",
}

#: `use:<tool_name>` 的触发规则（§6.4 EchoLLM）。工具名用宽松的标识符形态，
#: 后面可跟一个**可选**的 JSON 对象作为参数。
_ECHO_USE_RE = re.compile(r"use:([A-Za-z_][A-Za-z0-9_.\-]*)")


# --------------------------------------------------------------------------------------
# 角色辅助
# --------------------------------------------------------------------------------------


def _role_value(message: Message) -> str:
    """取消息的角色字符串。

    `Message.role` 的类型是 `Role`，但 dataclass 不做运行期校验，手工构造时可能是裸字符串；
    这里统一成 `.value`，避免 provider 在"用户手搓 Message"时抛 `AttributeError`。
    """
    role = message.role
    return role.value if isinstance(role, Role) else str(role)


def _provider_name(client: Any) -> str:
    """provider 名（用于错误信息）：子类都有 `name`，没有时退化到类名。"""
    return str(getattr(client, "name", type(client).__name__))


# --------------------------------------------------------------------------------------
# 工具 schema：两种形态之间的转换
# --------------------------------------------------------------------------------------


def _parameters_or_empty(tool: Mapping[str, Any]) -> Any:
    """从任意形态的工具描述里取出 JSON Schema（`parameters` 或 `input_schema`）。"""
    schema = tool.get("parameters")
    if schema is None:
        schema = tool.get("input_schema")
    if schema is None:
        # 没有参数的工具有两种常见写法：省略 schema 或写空对象。
        return {"type": "object", "properties": {}}
    return dict(schema) if isinstance(schema, Mapping) else schema


def _normalize_openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """把工具描述规范化成 OpenAI 的 `{"type":"function","function":{...}}` 形态。

    为什么要有这一步：`achat(tools=...)` 的契约是"provider 无关的工具描述列表"，
    调用方（Agent / CLI / 测试）可能给 `ToolRegistry.schemas(fmt="openai")`（已经是目标形态，
    这里原样放行）也可能给 `to_anthropic_tool` 的形态。少一次转换就少一类 400 错误，
    而这个转换对已经是目标形态的输入是**幂等**的。
    """
    if "function" in tool and isinstance(tool.get("function"), Mapping):
        return dict(tool)
    function: dict[str, Any] = {
        "name": str(tool.get("name") or ""),
        "parameters": _parameters_or_empty(tool),
    }
    description = tool.get("description")
    if description:
        function["description"] = str(description)
    return {"type": "function", "function": function}


def _normalize_anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """把工具描述规范化成 Anthropic 的 `{"name","description","input_schema"}` 形态。"""
    if "input_schema" in tool:
        return dict(tool)
    function = tool.get("function")
    source: Mapping[str, Any] = function if isinstance(function, Mapping) else tool
    normalized: dict[str, Any] = {
        "name": str(source.get("name") or ""),
        "input_schema": _parameters_or_empty(source),
    }
    description = source.get("description")
    if description:
        normalized["description"] = str(description)
    return normalized


def _anthropic_tool_choice(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """OpenAI 风格的 `tool_choice` -> Anthropic 的 `{"type": ...}`。

    `"none"` 在 Anthropic 没有对应值（它没有"禁止调用工具"的开关），退化为"不传"，
    即默认的 auto —— 比起编一个不存在的 `{"type":"none"}` 让请求 400，这是更保守的选择。
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if value == "required" or value == "any":
        return {"type": "any"}
    if value == "none":
        return None
    return {"type": "auto"}


# --------------------------------------------------------------------------------------
# 消息编解码
# --------------------------------------------------------------------------------------


def _arguments_json(call: ToolCall) -> str:
    """把 `ToolCall.arguments` 渲染成模型侧要求的 JSON **字符串**。

    刻意**不用** `raw_arguments`：它可能是非法 JSON（§6.4 的 `__raw__` 路径留下的原文），
    原样回灌历史会让下一次请求 400。这里统一重新序列化；`sort_keys=True` 让同一份
    arguments 永远得到同一个字符串（可断言、可 diff）。
    """
    return json.dumps(dict(call.arguments), ensure_ascii=False, sort_keys=True, default=str)


def _openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """`Message` -> OpenAI 的 messages 数组（§6.4 冻结的两种转换）。"""
    payload: list[dict[str, Any]] = []
    for message in messages:
        role = _role_value(message)
        if role == Role.TOOL.value:
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
            continue
        item: dict[str, Any] = {"role": role, "content": message.content}
        if role == Role.ASSISTANT.value and message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": _arguments_json(call)},
                }
                for call in message.tool_calls
            ]
        payload.append(item)
    return payload


def _anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """`Message` -> Anthropic 的 messages 数组（system 已被外层提走）。

    冻结的两条规则（§6.4 规则 3/4）：
    - assistant 的 tool_calls -> content block 列表里的 `tool_use`；
    - tool 结果必须变成 **user** 消息里的 `tool_result` block，且**连续的多个必须合并
      进同一条 user 消息** —— 否则 Anthropic 会把"tool_use 后面跟着两条 user 消息"
      判成非法（400）。
    """
    payload: list[dict[str, Any]] = []
    pending_results: dict[str, Any] | None = None  # 正在累积 tool_result 的那条 user 消息
    for message in messages:
        role = _role_value(message)
        if role == Role.SYSTEM.value:
            # system 走顶层字段（`_split_system` 已处理），这里必须丢弃。
            continue
        if role == Role.TOOL.value:
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            if pending_results is None:
                pending_results = {"role": "user", "content": [block]}
                payload.append(pending_results)
            else:
                pending_results["content"].append(block)
            continue
        pending_results = None  # 非 tool 的消息打断合并
        if role == Role.ASSISTANT.value and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                )
            payload.append({"role": "assistant", "content": blocks})
            continue
        payload.append({"role": role, "content": message.content})
    return payload


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """提取 system 文本 + 转换其余消息（§6.4 规则 1）。

    多条 system 消息用空行拼接：Anthropic 只接受一个顶层 `system` 字符串，
    丢弃后面的消息是不可接受的（会静默改变行为），拼接是唯一无损的做法。
    """
    system_parts = [m.content for m in messages if _role_value(m) == Role.SYSTEM.value and m.content]
    return "\n\n".join(system_parts), _anthropic_messages(messages)


# --------------------------------------------------------------------------------------
# usage / tool_call 解析
# --------------------------------------------------------------------------------------


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _openai_usage(payload: Mapping[str, Any]) -> TokenUsage:
    raw = payload.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    return TokenUsage(
        prompt_tokens=_int_or_zero(usage.get("prompt_tokens")),
        completion_tokens=_int_or_zero(usage.get("completion_tokens")),
        total_tokens=_int_or_zero(usage.get("total_tokens")),
    )


def _anthropic_usage(payload: Mapping[str, Any]) -> TokenUsage:
    """`input_tokens/output_tokens` -> `prompt_tokens/completion_tokens`（§6.4 规则 7）。"""
    raw = payload.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    prompt_tokens = _int_or_zero(usage.get("input_tokens"))
    completion_tokens = _int_or_zero(usage.get("output_tokens"))
    # Anthropic 不返回 total：由两段相加得出（`TokenUsage.__post_init__` 也会补，
    # 但显式写出来更清楚，且不会在两端都为 0 时留下歧义）。
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _call_from_json_arguments(name: str, raw: Any, call_id: Any) -> ToolCall:
    """从"字符串 arguments"造 `ToolCall`，**绝不抛异常**（§6.4 冻结）。

    解析失败 -> `arguments={"__raw__": <原文>}` 且 `metadata["parse_error"]` 记错误消息，
    然后照常返回。理由：模型拿到空响应就再也没有自纠正的机会，而收到
    `{"__raw__": "..."}` 时它能看见自己写错了什么。少数兼容实现直接回 dict 而不是字符串，
    这条路径也一并接受。
    """
    identifier = str(call_id) if call_id else None
    if isinstance(raw, Mapping):
        return ToolCall.create(name, raw, call_id=identifier)
    text = "" if raw is None else str(raw)
    call, error = ToolCall.try_from_arguments_json(name, text, call_id=identifier)
    if error is not None:
        # 兜底必须留可观测痕迹（§13 红线 10/12）：错误消息进 metadata 而不是被丢掉 ——
        # 这正是 §6.4 指定的观测点（"在 metadata["parse_error"] 记错误消息"）。
        # 这里**不**额外 warnings.warn：模型偶发写错 JSON 是常见且可自纠正的，
        # 每次刷一条 RuntimeWarning 会把真正的降级告警淹没，也会污染测试输出。
        call.metadata["parse_error"] = error
    return call


def _openai_tool_calls(message: Mapping[str, Any]) -> list[ToolCall]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return []
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = str(function.get("name") or "")
        calls.append(
            _call_from_json_arguments(name, function.get("arguments"), raw_call.get("id"))
        )
    return calls


def _anthropic_content_blocks(
    blocks: Sequence[Any],
) -> tuple[str, list[ToolCall]]:
    """Anthropic 的 content block 列表 -> (文本, tool_calls)（§6.4 规则 5）。"""
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text") or ""))
            continue
        if block_type == "tool_use":
            name = str(block.get("name") or "")
            raw_input = block.get("input")
            if isinstance(raw_input, Mapping):
                calls.append(
                    ToolCall.create(name, raw_input, call_id=block.get("id") or None)
                )
            else:
                # input 不是对象（极少见）：走不抛异常的解析路径并把原文留成 __raw__。
                calls.append(
                    _call_from_json_arguments(name, raw_input, block.get("id"))
                )
    return "".join(texts), calls


# --------------------------------------------------------------------------------------
# 公共基类
# --------------------------------------------------------------------------------------


class HTTPChatClient(BaseLLMClient):
    """所有真实 provider 的公共基类：请求体构造交给子类，

    发送/错误映射/重试/响应解码由基类统一完成。
    """

    default_base_url: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    supports_tool_calling: bool = True
    #: [v2 新增] 超时后是否重发请求。默认 False（请求可能已到达服务端并产生副作用/计费）
    retry_on_timeout: bool = False

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: Transport | None = None,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        super().__init__(config, transport=transport, on_event=on_event)
        # 注入的 transport 由调用方拥有吗？不：既然交给了本 client，就由本 client 负责关闭；
        # 而 `default_transport()` 是 functools.cache 的进程级单例，**不能**关它
        # （关掉会让同进程里其它 client 的后续请求全部失败）。
        self._owns_transport = transport is not None
        self._stream_warned = False

    # ---- 子类必须实现 ----

    def _endpoint(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement _endpoint()")

    def _headers(self, api_key: str | None) -> dict[str, str]:
        raise NotImplementedError(f"{type(self).__name__} must implement _headers()")

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement _build_body()")

    def _parse_response(self, payload: dict[str, Any], *, latency_ms: float) -> LLMResponse:
        raise NotImplementedError(f"{type(self).__name__} must implement _parse_response()")

    # ---- 基类实现 ----

    def _transport(self) -> Transport:
        """取得传输层：优先用注入的，否则懒取进程级默认（`urllib` 永远可用）。"""
        if self.transport is None:
            self.transport = default_transport()
        return self.transport

    def _base_url_or_none(self) -> str | None:
        url = self.config.resolved_base_url() or self.default_base_url or None
        return url.rstrip("/") if url else None

    def _base_url(self) -> str:
        """生效的 base_url；没有可用地址时抛 `ConfigError`（早失败，别等到 404/连接错误）。"""
        url = self._base_url_or_none()
        if not url:
            raise ConfigError(
                f"provider {_provider_name(self)!r} requires an explicit base_url; pass it via "
                f"LLMConfig.base_url / LITEAGENT_BASE_URL, or use "
                f"get_llm('{_provider_name(self)}:<model>@http://host:port/v1')"
            )
        return url

    def _require_api_key(self) -> str:
        """缺失时抛 `LLMAuthError('missing API key for provider X ...')`。

        不缓存解析结果：`LLMConfig.resolve_api_key` 每次都读环境变量，测试会
        `patch.dict(os.environ)` 临时换 key，缓存会让用例互相污染。
        """
        if not self.requires_api_key:
            return ""
        api_key = self.config.resolve_api_key()
        if not api_key:
            env_name = f"{self.config.provider.upper().replace('-', '_')}_API_KEY"
            raise LLMAuthError(
                0,
                message=(
                    f"missing API key for provider {_provider_name(self)!r}; set LLMConfig.api_key, "
                    f"LITEAGENT_API_KEY or {env_name}"
                ),
            )
        return api_key

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
        """发一次请求：构造请求体 -> （带重试地）发送 -> 解码 -> 发 LLM_RESPONSE 事件。

        非 2xx 不在这里判断：`Transport` 的契约就是"失败必须以 `LiteAgentError` 子类抛出"
        （`transport.map_http_error` 是唯一映射表），这样重试逻辑只面对一种异常形态。
        """
        api_key = self._require_api_key()
        temperature, max_tokens = self._resolve_params(temperature, max_tokens)

        if self.config.stream and not self._stream_warned:
            # 降级必须留可观测痕迹（§13 红线 12）：`Transport` 只返回"整段响应"，
            # 没有流式接口，所以 config.stream 在本类里被忽略。明确告诉调用方，
            # 而不是静默地按阻塞式返回（只警告一次，避免每轮刷屏）。
            warnings.warn(
                f"{type(self).__name__}.achat ignores config.stream: the transport layer has no "
                "streaming interface; use astream_chat/scripted paths for streaming",
                RuntimeWarning,
                stacklevel=2,
            )
            self._stream_warned = True

        passthrough = dict(kwargs)
        stream = bool(passthrough.pop("stream", False))
        body = self._build_body(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
        )
        if passthrough:
            # 未识别的 kwargs 透传进请求体（top_p / response_format / reasoning_effort 等
            # provider 特有能力）。写在 _build_body 之后，因此调用方可以覆盖任何默认字段。
            body.update(passthrough)

        request = HTTPRequest(
            method="POST",
            url=self._endpoint(),
            headers=self._headers(api_key),
            json_body=body,
            timeout_s=self.config.timeout_s,
        )
        self._prepare_request(messages, tools)
        started = time.perf_counter()
        response = await self._with_retry(
            lambda: self._transport().asend(request), what="achat"
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        payload = response.json()
        if not isinstance(payload, Mapping):
            raise LLMResponseFormatError(
                body=response.text[:2000],
                message=f"LLM response body is not a JSON object ({type(payload).__name__})",
            )
        resp = self._parse_response(dict(payload), latency_ms=latency_ms)
        self._emit_response(resp)
        return resp

    async def aclose(self) -> None:
        """关闭传输层。

        **只关自己注入的**：`default_transport()` 是 `functools.cache` 的进程级单例，
        关掉它会让同进程里其它 client 的后续请求全部失败（"我在别的对象里关了你的连接池"
        是最难查的一类 bug）。
        """
        if self._owns_transport and self.transport is not None:
            self.transport.close()
        return None


class OpenAIChatClient(HTTPChatClient):
    """POST `{base_url}/chat/completions`。默认 base_url = `https://api.openai.com/v1`。

    请求：`{"model","messages","tools":[{"type":"function","function":{...}}],"tool_choice",...}`
    消息转换：`role=tool` 的消息 -> `{"role":"tool","tool_call_id":...,"content":...}`；
              `assistant.tool_calls` -> `[{"id","type":"function","function":{"name","arguments":<json str>}}]`
    响应：`choices[0].message.{content, tool_calls}`；`usage.{prompt_tokens,completion_tokens,total_tokens}`；
          `tool_calls[i].function.arguments` 是**字符串**，用
          [v2 变更] `ToolCall.try_from_arguments_json` 解析（**不抛异常**版本）：
          解析失败 -> 该 `ToolCall.arguments = {"__raw__": <str>}`，并在 `metadata["parse_error"]`
          记错误消息。**绝不在这里抛异常**（否则模型失去自纠正机会）。
    """

    name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def _endpoint(self) -> str:
        return f"{self._base_url()}/chat/completions"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": _JSON_CONTENT_TYPE}
        if api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": _openai_messages(messages)}
        if tools:
            body["tools"] = [_normalize_openai_tool(tool) for tool in tools]
            if tool_choice is not None:
                # tools 为空时传 tool_choice 会被 API 拒绝，所以它跟着 tools 一起出现。
                body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = list(stop)
        if stream:
            body["stream"] = True
        body.update(self.config.extra_body)
        return body

    def _parse_response(self, payload: dict[str, Any], *, latency_ms: float) -> LLMResponse:
        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise LLMResponseFormatError(
                body=json.dumps(payload, ensure_ascii=False)[:2000],
                message="OpenAI response is missing a non-empty 'choices' array",
            )
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMResponseFormatError(
                body=json.dumps(payload, ensure_ascii=False)[:2000],
                message="OpenAI response choice is missing the 'message' object",
            )

        tool_calls = _openai_tool_calls(message)
        content = message.get("content")
        finish_raw = choice.get("finish_reason")
        finish_reason = (
            str(finish_raw)
            if finish_raw
            else ("tool_calls" if tool_calls else "stop")
        )
        usage = _openai_usage(payload)
        model = str(payload.get("model") or self.model)
        usage.model_hint = model  # 成本估算的输入（§4.1）
        return LLMResponse(
            content="" if content is None else str(content),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            raw=payload,
            latency_ms=latency_ms,
        )


class OpenAICompatibleClient(OpenAIChatClient):
    """vLLM/Ollama/DeepSeek/OpenRouter 等。必须显式传 base_url，否则 `ConfigError`。

    注意 DeepSeek 是这里的例外：它有公认的默认地址，因此子类 `DeepSeekChatClient`
    只要覆盖 `default_base_url` 就可用 —— 这正是"统一抽象层"的价值所在。
    """

    name = "openai-compatible"
    # 必须显式清空：父类 OpenAIChatClient 的默认地址是 api.openai.com，把它继承下来
    # 会让"忘了给 base_url"静默地打到 OpenAI 官方端点（既报 401 又可能计费）。
    default_base_url = ""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: Transport | None = None,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        super().__init__(config, transport=transport, on_event=on_event)
        # 构造期就校验（而不是等发请求）：vLLM/Ollama/OpenRouter 没有公认的默认地址，
        # 把"配置缺失"拖到运行期会伪装成"网络/服务故障"，是最耗时的一类排查。
        self._base_url()


class DeepSeekChatClient(OpenAICompatibleClient):
    """[v2 新增] 只需 5 行就多接一个真实 provider —— 这是"统一多模型 API"最直接的证据。

    完整实现（逐字，§6.4）：
        name = "deepseek"
        default_base_url = "https://api.deepseek.com/v1"

    其余全部继承：请求体、响应解码、重试、事件、usage 映射都与 OpenAI 一致
    （DeepSeek 的 HTTP 接口刻意与 OpenAI 兼容）。
    """

    name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"


class AnthropicChatClient(HTTPChatClient):
    """POST `{base_url}/messages`，默认 base_url = `https://api.anthropic.com`。

    头：x-api-key + anthropic-version: 2023-06-01 + content-type: application/json

    **关键差异（易错，冻结）**：
      1. system 消息**不能**放进 messages 数组，必须提取为顶层 `"system": <拼接文本>`。
      2. 工具 schema 形态是 `{"name","description","input_schema"}`，不是 OpenAI 的嵌套 function。
      3. assistant 的 tool_use 是 content block：`{"type":"tool_use","id","name","input":{...}}`
      4. tool 结果必须作为 **user** 消息的 content block：
         `{"role":"user","content":[{"type":"tool_result","tool_use_id":...,"content":<str>}]}`
         且连续的多个 tool_result 必须合并进同一个 user 消息。
      5. 响应 content 是 block 列表：取 `type=="text"` 的文本拼接；`type=="tool_use"` 转 ToolCall。
      6. stop_reason 映射：`"end_turn"->"stop"`、`"tool_use"->"tool_calls"`、`"max_tokens"->"length"`。
      7. usage 字段名是 input_tokens/output_tokens，需映射到 prompt/completion。
      8. [v2 新增] 无 tool_use 时 finish_reason 一律映射为 `"stop"`；`content` 为空但
         `stop_reason=="max_tokens"` 时必须给出 `finish_reason="length"`（供 §9.4.6 的截断分支使用）。
         注意：**不复现** v1 的 `"content_filter"`（Anthropic 无此 stop_reason）。
    """

    name = "anthropic"
    default_base_url = "https://api.anthropic.com"
    api_key_header = "x-api-key"
    api_key_prefix = ""

    def _endpoint(self) -> str:
        return f"{self._base_url()}/messages"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = {
            "Content-Type": _JSON_CONTENT_TYPE,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _build_body(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        system_text, converted = _split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            # Anthropic 的 max_tokens 必填，而 LLMConfig.max_tokens 默认为 None：
            # 不给就是 400，所以这里必须有兜底值。
            "max_tokens": max_tokens if max_tokens is not None else _ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = [_normalize_anthropic_tool(tool) for tool in tools]
            choice = _anthropic_tool_choice(tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        if temperature is not None:
            # 原样透传（不做 clamp）：Anthropic 的合法区间是 [0, 1]，越界让服务端明确报 400
            # 比本地静默改值更容易排查 —— 静默改值属于"降级"，必须留痕才允许。
            body["temperature"] = temperature
        if stop:
            body["stop_sequences"] = list(stop)
        if stream:
            body["stream"] = True
        body.update(self.config.extra_body)
        return body

    def _parse_response(self, payload: dict[str, Any], *, latency_ms: float) -> LLMResponse:
        blocks = payload.get("content")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise LLMResponseFormatError(
                body=json.dumps(payload, ensure_ascii=False)[:2000],
                message="Anthropic response is missing the 'content' block list",
            )
        content, tool_calls = _anthropic_content_blocks(blocks)

        stop_reason = payload.get("stop_reason")
        stop_reason_text = str(stop_reason) if stop_reason else ""
        if tool_calls:
            finish_reason = "tool_calls"
        elif stop_reason_text == "max_tokens":
            # §6.4 规则 8 的 MUST：**即使 content 为空**也必须给出 "length"，
            # 否则 §9.4.6 的"截断续写"分支永远不会被触发（模型被 max_tokens 截断后
            # 会静默地当作正常结束）。
            finish_reason = "length"
        else:
            finish_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason_text, "stop")

        usage = _anthropic_usage(payload)
        model = str(payload.get("model") or self.model)
        usage.model_hint = model
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            raw=payload,
            latency_ms=latency_ms,
        )


def _parse_inline_arguments(tail: str) -> tuple[dict[str, Any], str | None]:
    """解析 `use:<name>` 之后的**可选** JSON 对象（§6.4 EchoLLM 触发规则）。

    返回 `(参数, 错误消息 | None)`：缺失或非法时返回 `{}` + 错误消息，
    由调用方把错误消息写进 `ToolCall.metadata`（§13 红线 12：兜底必须留痕）。
    """
    text = tail.strip()
    if not text:
        return {}, None
    start = text.find("{")
    if start < 0:
        return {}, f"no JSON object found after 'use:': {text[:120]!r}"
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as exc:
        return {}, f"{exc}: {text[start : start + 120]!r}"
    if not isinstance(value, Mapping):
        return {}, f"expected a JSON object, got {type(value).__name__}"
    return dict(value), None


class EchoLLM(BaseLLMClient):
    """离线占位 provider：不需要 key、不联网。

    行为：返回一段包含收到的消息条数与最后一条用户消息前 200 字符的文本；
    若 tools 非空且最后一条用户消息包含 `'use:<tool_name>'`，则返回一个对应的 tool_call。

    [v2 变更] 触发规则冻结：`use:<name>` 后可跟一个**可选** JSON 对象作为参数，
    如 `use:add {"a":1,"b":2}`；缺失或非法 JSON 时用 `{}`。

    用途：CLI 无 key 演示、文档示例、冒烟测试（**不是**单元测试的确定性替身，
    `ScriptedLLM` 才是 —— 它才能断言完整的调用历史）。
    """

    name = "echo"
    supports_tool_calling = True
    requires_api_key = False

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
        last_user = ""
        for message in reversed(list(messages)):
            if _role_value(message) == Role.USER.value:
                last_user = message.content
                break

        self._prepare_request(messages, tools)

        async def _respond() -> LLMResponse:
            tool_calls: list[ToolCall] = []
            if tools and last_user:
                match = _ECHO_USE_RE.search(last_user)
                if match:
                    tool_calls = [self._build_call(match.group(1), last_user[match.end() :])]
            content = (
                f"echo: received {len(messages)} message(s); "
                f"last user message: {last_user[:200]}"
            )
            prompt_tokens = sum(self.count_tokens(message.text()) for message in messages)
            completion_tokens = self.count_tokens(content)
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            usage.model_hint = self.model
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                usage=usage,
                model=self.model,
            )

        # 走 _with_retry 是为了让 echo 与真实 provider 发同样的事件序列
        # （LLM_REQUEST / LLM_RESPONSE），CLI 与 trace 渲染因此不需要分支。
        resp = await self._with_retry(_respond, what="echo")
        self._emit_response(resp)
        return resp

    def _build_call(self, name: str, tail: str) -> ToolCall:
        arguments, error = _parse_inline_arguments(tail)
        call = ToolCall.create(name, arguments)
        if error is not None:
            # 兜底留痕（§13 红线 12）：非法 JSON 不抛异常，但原始片段与错误必须可查。
            call.metadata["parse_error"] = error
            call.raw_arguments = tail.strip()
        return call


#: 模块级注册（冻结，`registry.py` 依赖这些名字）
PROVIDER_CLASSES: dict[str, type[LLMClient]] = {
    "openai": OpenAIChatClient,
    "openai-compatible": OpenAICompatibleClient,
    "deepseek": DeepSeekChatClient,  # [v2 新增]
    "anthropic": AnthropicChatClient,
    "echo": EchoLLM,
}
