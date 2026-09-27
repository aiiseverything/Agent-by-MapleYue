from __future__ import annotations

"""核心数据结构（依赖图中的 **L1**）。

**决策 D-01（冻结）**：核心数据结构一律用 stdlib ``dataclass``，不用 pydantic。
理由：(a) 零依赖红线；(b) agent 状态是高频可变对象，校验/拷贝开销不划算；(c) ``dataclasses.replace``
已够用；(d) trace 需要**字段全量输出**，pydantic 的 ``exclude_none`` 默认行为反而别扭。
代价是没有运行期类型校验 —— 缓解手段是 ``from_dict`` 手工校验并抛 ``SerializationError``。

``slots=True`` 只用于**叶子**数据类（``TokenUsage``/``ToolCall``/``ToolResult``/``ScriptedCall``）；
``LLMResponse`` 不用 slots（§4）。注意冻结规则：**不要在 ``ToolCall``/``ToolResult`` 上临时挂属性**，
slots 会直接拒绝（这既是限制也是保护：手误写出的 ``call.foo = 1`` 会立刻暴露）。

本模块的 import 被 §4.5 白名单约束在 stdlib + ``liteagent.errors`` 之内：它不能被 config /
llm.message 反向依赖，否则 L1 会变成 L2 并产生循环。唯一允许的层间边是 §1.1 的 **E1**：
``to_message`` 在函数体内延迟 import ``liteagent.llm.message``。
"""

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from liteagent.errors import (
    LiteAgentError,
    ReActParseError,
    SerializationError,
)

__all__ = [
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    "ScriptedCall",
]

# 与 config.DEFAULT_MAX_OBSERVATION_CHARS / DEFAULT_MAX_RESULT_CHARS 同步的字面量。
# 为什么不用常量：§4.5 冻结了 types.py 不得 import config（L1 不能依赖 config 的实现细节，
# 否则 config 里任何一次改动都会牵动 L1 的 import 图）。改常量时必须同步改这里，注释即契约。
_DEFAULT_MAX_OBSERVATION_CHARS = 8000

# JSON 字段校验用的类型元组（见 _require/_optional）。
_STR = (str,)
_INT = (int,)
_NUM = (int, float)
_BOOL = (bool,)
_MAP = (dict, Mapping)


def _new_call_id() -> str:
    """生成工具调用 id：前缀 ``call_`` + 12 位十六进制。

    测试**不得**断言具体值（§2.8 禁止断言的字段清单），只能断言 ``startswith("call_")``。
    """
    return "call_" + uuid.uuid4().hex[:12]


def _as_mapping(data: Any, cls_name: str) -> Mapping[str, Any]:
    """from_dict 的入参守卫：不是 Mapping 就直接失败，避免后续 ``in`` 操作报出难懂的 TypeError。"""
    if not isinstance(data, Mapping):
        raise SerializationError(
            target=cls_name,
            message=f"{cls_name}.from_dict expects a mapping, got {type(data).__name__}",
        )
    return data


def _type_names(expected: tuple[type, ...]) -> str:
    return " | ".join(t.__name__ for t in expected)


def _require(data: Mapping[str, Any], key: str, cls_name: str, expected: tuple[type, ...]) -> Any:
    """取出必需字段并校验类型，失败抛 ``SerializationError``（§4：手工校验代替 pydantic）。"""
    if key not in data:
        raise SerializationError(
            target=f"{cls_name}.{key}",
            message=f"missing required field: {key!r}",
        )
    value = data[key]
    if not isinstance(value, expected):
        raise SerializationError(
            target=f"{cls_name}.{key}",
            message=f"field {key!r} must be {_type_names(expected)}, got {type(value).__name__}",
        )
    return value


def _optional(
    data: Mapping[str, Any], key: str, cls_name: str, expected: tuple[type, ...], default: Any
) -> Any:
    """取出可选字段（§2.2 的 opt-out 字段）：缺失或为 None 时返回 ``default``，**不重算**。

    注意与 ``_require`` 的区别：这里"字段不存在"是合法输入（例如 ``to_dict`` 默认省略了
    ``LLMResponse.raw``），但"字段存在却是错的类型"仍然要报错 —— 静默接受错类型会让
    反序列化错误推迟到更难排查的地方。
    """
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, expected):
        raise SerializationError(
            target=f"{cls_name}.{key}",
            message=f"field {key!r} must be {_type_names(expected)}, got {type(value).__name__}",
        )
    return value


def _required_nullable(
    data: Mapping[str, Any], key: str, cls_name: str, expected: tuple[type, ...]
) -> Any:
    """必需但**允许为 None** 的字段（``ToolResult.error`` / ``error_type`` 这类）。

    与 ``_optional`` 的区别：键必须存在（§2.2 的字段全量输出保证 to_dict 一定写了它），
    只是值可以是 None。这样"漏写字段"和"字段值为空"两种情况不会被混为一谈。
    """
    if key not in data:
        raise SerializationError(
            target=f"{cls_name}.{key}",
            message=f"missing required field: {key!r}",
        )
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, expected):
        raise SerializationError(
            target=f"{cls_name}.{key}",
            message=f"field {key!r} must be {_type_names(expected)} or None, "
            f"got {type(value).__name__}",
        )
    return value


def _jsonable(value: Any) -> Any:
    """把任意值转成 JSON 可序列化结构（供 ``ScriptedCall.to_dict`` 用）。

    types.py 是 L1，按 §4.5 白名单不能 import ``config.to_jsonable``，所以这里只实现本模块
    会遇到的窄集合。**规范的真值源仍然是 ``config.to_jsonable``**：完整的 Enum/Path/bytes/
    datetime 处理在那边；这里只覆盖 dataclass（有 ``to_dict``）、Mapping、序列、异常与标量，
    未知类型退化为 ``repr``（与 config 的 "unserializable" 标记策略同源，保证永不抛 TypeError）。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict_fn = getattr(value, "to_dict", None)
    if callable(to_dict_fn):
        return _jsonable(to_dict_fn())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return repr(value)


# ======================================================================================
# 4.1 TokenUsage
# ======================================================================================


@dataclass(slots=True)
class TokenUsage:
    """token 用量。累计语义是 ``+=``（agent 每轮把 LLM 响应的 usage 加进 state.usage）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # [v2 新增] 供成本估算使用的模型名。刻意放在最后：前三个字段的位置必须保持稳定
    # （ScriptedLLM 的固定常量 TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    #  依赖这个顺序可读）。它**不参与** to_dict 的必需键（§4.1）。
    model_hint: str = ""

    def __post_init__(self) -> None:
        """若 total_tokens == 0 且 prompt/completion > 0，则自动置为两者之和。

        为什么必须补算：部分 provider 只回 prompt/completion，不补算的话预算判定
        （§9.4.6 的 max_total_tokens）永远看不到消耗，上限形同虚设。
        """
        if self.total_tokens == 0 and (self.prompt_tokens + self.completion_tokens) > 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented  # type: ignore[return-value]
        # 逐项相加（而不是让 __post_init__ 重算 total）：provider 报的 total 可能包含
        # 缓存命中等额外口径，直接相加才是对账一致的。
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model_hint=self.model_hint or other.model_hint,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented  # type: ignore[return-value]
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        if not self.model_hint:
            self.model_hint = other.model_hint
        self.__post_init__()  # 两者 total 都是 0 而 prompt 非零时的兜底
        return self

    def is_empty(self) -> bool:
        """三个计数全为 0。用于"这次调用有没有真实消耗"的判定（如决定是否展示成本）。"""
        return (
            self.prompt_tokens == 0
            and self.completion_tokens == 0
            and self.total_tokens == 0
        )

    @property
    def estimated_cost_usd(self) -> float | None:
        """[v2 变更] 委托 ``config.estimate_cost_usd(self, model=self.model_hint)``。

        无价格表条目 -> None（诚实 > 猜）。延迟 import 的理由见文件末尾的 SPEC-AMBIGUITY 说明。
        """
        # SPEC-AMBIGUITY: §4.1 要求本属性委托 config.estimate_cost_usd，而 §4.5 的 import
        # 白名单禁止 types.py import config。裁决：遵循 §4.1 的行为要求，用函数体内延迟 import
        # （不产生模块级循环依赖，价格表也保持"唯一真值源"，不在本模块复制一份）。
        from liteagent.config import estimate_cost_usd

        return estimate_cost_usd(self, model=self.model_hint)

    def to_dict(self) -> dict[str, int]:
        """只输出 {"prompt_tokens","completion_tokens","total_tokens"}（3 个 int）。

        model_hint 刻意不输出：它是本进程的估算上下文，不是可回放的数据；写进 trace 只会
        在 diff 里制造噪声（§4.1 的显式例外）。
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenUsage":
        data = _as_mapping(data, "TokenUsage")
        return cls(
            prompt_tokens=_require(data, "prompt_tokens", "TokenUsage", _INT),
            completion_tokens=_require(data, "completion_tokens", "TokenUsage", _INT),
            total_tokens=_require(data, "total_tokens", "TokenUsage", _INT),
            # model_hint 不属于必需键：缺失时置 ""，**不重算**（§2.2）。
            model_hint=_optional(data, "model_hint", "TokenUsage", _STR, ""),
        )


# ======================================================================================
# 4.2 ToolCall
# ======================================================================================


@dataclass(slots=True)
class ToolCall:
    """模型发起的一次工具调用。

    ``arguments`` 是**已解析**的结构（native function calling 模式下由 provider 从
    ``function.arguments`` 字符串解析而来）；``raw_arguments`` 保留原始字符串，
    用于在解析失败时回灌给模型看它自己写了什么。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> "ToolCall":
        """[v2 变更] ``call_id`` 为 None 时生成 ``'call_' + uuid4().hex[:12]``。

        ``arguments`` 里的每个值必须是 JSON 可序列化类型，否则 ``SerializationError``：
        参数最终要进 trace / JSONL，也会被原样喂回模型，放任一个 set 或自定义对象通过
        只会在更远的地方炸掉（而且那时已经丢失了"是哪个参数"的信息）。
        """
        args: dict[str, Any] = dict(arguments) if arguments else {}
        for key, value in args.items():
            if not isinstance(key, str):
                raise SerializationError(
                    target=f"ToolCall.arguments[{key!r}]",
                    message="argument names must be str",
                )
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    target=f"ToolCall.arguments[{key!r}]",
                    message=f"argument {key!r} is not JSON-serializable: {exc}",
                    cause=exc,
                ) from exc
        # raw_arguments 刻意留空：这里是"代码构造"而不是"解析模型输出"，
        # 没有原始字符串可留（真要留就调 from_arguments_json）。
        return cls(id=call_id or _new_call_id(), name=name, arguments=args)

    @classmethod
    def from_arguments_json(
        cls, name: str, raw: str, *, call_id: str | None = None
    ) -> "ToolCall":
        """解析模型返回的 arguments 字符串；失败时抛 ``ReActParseError``（带 raw/offset）。

        错误 message 固定包含 "must be a valid JSON object" 与原始串前 200 字符 ——
        §7.4.1 步骤 4 要求这条文案出现在回灌给模型的校验错误里，写在这里可以保证
        provider -> executor 的整条链路都不需要再拼一次。
        """
        error_message = (
            f"tool call arguments for {name!r} must be a valid JSON object; "
            f"got: {raw[:200]}"
        )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReActParseError(
                raw=raw,
                offset=int(getattr(exc, "pos", 0) or 0),
                reason=f"arguments is not valid JSON: {exc}",
                message=error_message,
                cause=exc,
            ) from exc
        if not isinstance(parsed, dict):
            # "[1, 2]" / "42" / '"abc"' 都是合法 JSON 但不是对象，不能当成参数用。
            raise ReActParseError(
                raw=raw,
                offset=0,
                reason=f"arguments must be a JSON object, got {type(parsed).__name__}",
                message=error_message,
            )
        call = cls.create(name, parsed, call_id=call_id)
        # create() 刻意不填 raw_arguments（"代码构造"没有原始串可留），但解析路径**一定有**：
        # 原始串是回灌给模型看"你刚才到底写了什么"的唯一依据（模型写的是字符串，不是 dict），
        # 也是 trace 里排查 provider 编解码差异的抓手。
        call.raw_arguments = raw
        return call

    @classmethod
    def try_from_arguments_json(
        cls, name: str, raw: str, *, call_id: str | None = None
    ) -> tuple["ToolCall", str | None]:
        """[v2 新增] **不抛异常**的变体：失败时返回 (``arguments={"__raw__": raw}`` 的 ToolCall, 错误消息)。

        provider 层只用这个版本（§6.4 冻结）：解析失败若直接抛异常，模型就收不到任何回执，
        也就失去了自纠正的机会。这不是"吞异常"—— 错误消息作为第二个返回值交给调用方写入
        ``metadata["parse_error"]``（§13 红线 10/12：兜底必须留可观测痕迹）。
        """
        try:
            return cls.from_arguments_json(name, raw, call_id=call_id), None
        except LiteAgentError as exc:
            # 捕获范围覆盖 from_arguments_json 可能抛出的 ReActParseError，以及
            # 万一 create() 的手工校验失败时抛出的 SerializationError —— 两者都不该让
            # 一次模型响应把整个 agent 打死。
            call = cls(
                id=call_id or _new_call_id(),
                name=name,
                arguments={"__raw__": raw},
                raw_arguments=raw,
            )
            return call, str(exc)

    def canonical_key(self) -> str:
        """重复动作检测用的稳定键（§9.4.1）。

        ``sort_keys=True`` 让参数顺序不影响结果，``default=str`` 是最后一道保险：
        即使有人往 arguments 里塞了非 JSON 类型，键也仍然可比较而不会抛异常。
        """
        return (
            f"{self.name}:"
            f"{json.dumps(self.arguments, sort_keys=True, ensure_ascii=False, default=str)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """[v2 变更] **必须**输出 {"id","name","arguments","raw_arguments","metadata"} 五个键。"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
            "raw_arguments": self.raw_arguments,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCall":
        data = _as_mapping(data, "ToolCall")
        return cls(
            id=_require(data, "id", "ToolCall", _STR),
            name=_require(data, "name", "ToolCall", _STR),
            arguments=dict(_require(data, "arguments", "ToolCall", _MAP)),
            raw_arguments=_require(data, "raw_arguments", "ToolCall", _STR),
            metadata=dict(_require(data, "metadata", "ToolCall", _MAP)),
        )


# ======================================================================================
# 4.3 ToolResult
# ======================================================================================


@dataclass(slots=True)
class ToolResult:
    """一次工具调用的结果（成功或失败都编码成它，绝不向上抛业务异常）。

    ``ok=False`` 时 ``content`` 一定是非空且以 "ERROR(" 开头的错误文本（executor 的不变式，
    §7.4.1 步骤 7）：模型只认文本，空字符串会让它无从自纠正。
    """

    call_id: str
    name: str
    content: str = ""
    ok: bool = True
    error: str | None = None
    error_type: str | None = None  # LiteAgentError 子类名，如 "ToolValidationError"
    duration_ms: float = 0.0
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        call: ToolCall,
        content: str,
        *,
        duration_ms: float = 0.0,
        attempts: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            call_id=call.id,
            name=call.name,
            # 强制 str：§13 红线 9 不允许 content 变成非字符串类型（模型侧只认文本）。
            content=str(content),
            ok=True,
            error=None,
            error_type=None,
            duration_ms=duration_ms,
            attempts=attempts,
            metadata=dict(metadata) if metadata else {},
        )

    @classmethod
    def failure(
        cls,
        call: ToolCall,
        error: BaseException,
        *,
        duration_ms: float = 0.0,
        attempts: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        """把异常编码成失败结果。

        - ``error_type = type(error).__name__``；``error = str(error)``；
        - 若 error 是 ``LiteAgentError`` 则把 ``error.context`` 合并进 ``metadata['error_context']``；
        - **[v2 变更] 冻结**：``content = self.error_text()``。失败结果**必须**带非空 content。
        """
        md: dict[str, Any] = dict(metadata) if metadata else {}
        if isinstance(error, LiteAgentError) and error.context:
            # 复制一份：metadata 会被写进 trace，不能让 trace 与异常对象共享同一个 dict。
            md["error_context"] = dict(error.context)
        result = cls(
            call_id=call.id,
            name=call.name,
            content="",
            ok=False,
            error=str(error),
            error_type=type(error).__name__,
            duration_ms=duration_ms,
            attempts=attempts,
            metadata=md,
        )
        # 此时 content 为空，error_text() 只会产出 "ERROR(<type>): <error>" 这段前缀。
        result.content = result.error_text()
        return result

    def error_text(self) -> str:
        """[v2 新增] 唯一的错误文本生成处（executor / to_message / to_observation 全部复用它）。

        - ``ok=True`` -> ``self.content``；
        - 否则 -> ``f"ERROR({error_type}): {error}"`` + 非空 content 时追加 ``"\\n" + content``。

        SPEC-AMBIGUITY: §4.3 说 "content = self.error_text()"，而 error_text() 在 content 非空时
        会再拼一次；照字面实现会让**第二次**调用产出两遍错误文本（"ERROR(...): x\\nERROR(...): x"），
        与同一节的 "否则错误文本会出现两遍" 以及 §7.4.1 步骤 4.5 的固定回灌文案直接冲突。
        裁决：把 error_text() 实现成**幂等**的 —— 当 content 恰好等于这段前缀（即 failure() 写入的
        那个值）时不再追加。这样 `failure().content == failure().error_text()` 恒成立，
        任何调用序都不会出现两遍前缀。
        """
        if self.ok:
            return self.content
        text = f"ERROR({self.error_type}): {self.error}"
        if self.content and self.content != text:
            text = text + "\n" + self.content
        return text

    def to_message(self) -> "Message":
        """-> Message(role=Role.TOOL, content=self.error_text(), name=self.name,
        tool_call_id=self.call_id, metadata={'tool_name': self.name})

        E1（§1.1）：唯一允许的层间边，且**必须**写在函数体内，否则 test_zero_dependency
        的顶层 import 检查会失败。
        """
        from liteagent.llm.message import Message, Role

        # metadata['tool_name'] 是 §2.4 的预留 key（写入者：agent / 此处冗余副本），
        # 便于只看消息列表就能读出工具名，不必反查 tool_call_id。
        return Message(
            role=Role.TOOL,
            content=self.error_text(),
            name=self.name,
            tool_call_id=self.call_id,
            metadata={"tool_name": self.name},
        )

    def to_observation(self, *, max_chars: int = _DEFAULT_MAX_OBSERVATION_CHARS) -> str:
        """文本 ReAct 模式的 Observation 载荷（§0.5 的 observation）。

        [v2 变更] 冻结：``ok=False`` 时**直接返回 self.content**（它已经是 error_text() 的结果，
        **不再二次拼 'ERROR(...):' 前缀** —— 否则错误文本会出现两遍）。
        ``ok=True`` 时返回 content，超长则 truncate_head_tail 做头尾保留式截断。

        SPEC-AMBIGUITY: §4.3 点名要用 ``truncate_head_tail``，而 §4.5 的 import 白名单禁止
        import config。裁决：只在真的需要截断时才函数体内延迟 import config（不产生模块级
        循环依赖），保证全项目只有一份截断算法（§5.2 的 truncate_head_tail 有独立测试）。
        """
        if not self.ok:
            return self.content
        if max_chars <= 0 or len(self.content) <= max_chars:
            return self.content
        from liteagent.config import truncate_head_tail

        return truncate_head_tail(self.content, max_chars)

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出（含 None）；metadata 复制一份，避免 trace 与实例共享可变状态。"""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "content": self.content,
            "ok": self.ok,
            "error": self.error,
            "error_type": self.error_type,
            "duration_ms": self.duration_ms,
            "attempts": self.attempts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolResult":
        data = _as_mapping(data, "ToolResult")
        return cls(
            call_id=_require(data, "call_id", "ToolResult", _STR),
            name=_require(data, "name", "ToolResult", _STR),
            content=_require(data, "content", "ToolResult", _STR),
            ok=_require(data, "ok", "ToolResult", _BOOL),
            error=_required_nullable(data, "error", "ToolResult", _STR),
            error_type=_required_nullable(data, "error_type", "ToolResult", _STR),
            duration_ms=_require(data, "duration_ms", "ToolResult", _NUM),
            attempts=_require(data, "attempts", "ToolResult", _INT),
            metadata=dict(_require(data, "metadata", "ToolResult", _MAP)),
        )


# ======================================================================================
# 4.4 LLMResponse
# ======================================================================================


@dataclass
class LLMResponse:
    """一次 LLM 调用的完整结果。

    不使用 ``slots=True``（§4 决策）：它含 dict 默认值，且会被 provider 层填充额外信息，
    slots 带来的收益抵不过"不能挂临时属性/不能 replace"的不便。
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length" | "error" | "content_filter"
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw: dict[str, Any] | None = None
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        """ReAct 状态机用它在"继续行动"与"终结"之间分流（§9.4）。"""
        return bool(self.tool_calls)

    def to_message(self) -> "Message":
        """-> Message(role=Role.ASSISTANT, content=self.content, tool_calls=self.tool_calls)

        E1（§1.1）：函数体内延迟 import。
        """
        from liteagent.llm.message import Message, Role

        # 传浅拷贝（不是 self.tool_calls 本身）：Message 会被追加进 state.messages 并长期持有，
        # 与 response 共享同一个 list 会让"谁在改这个列表"变得不可追踪（§13 红线 11 同一精神）。
        return Message(
            role=Role.ASSISTANT,
            content=self.content,
            tool_calls=list(self.tool_calls),
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """字段全量输出（含 None）。

        ``include_raw`` 是 §2.2 允许的三个显式 opt-out 之一：``raw`` 是 provider 的原始响应体，
        体积大且常含样本无关的元数据，默认不写进 trace。**键仍在**（值为 None），
        这样 trace 结构稳定、可以 ``assertEqual`` 精确比对。
        """
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "raw": self.raw if include_raw else None,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMResponse":
        data = _as_mapping(data, "LLMResponse")
        raw_usage = _require(data, "usage", "LLMResponse", _MAP + (TokenUsage,))
        raw_calls = _require(data, "tool_calls", "LLMResponse", (list, tuple))
        return cls(
            content=_require(data, "content", "LLMResponse", _STR),
            tool_calls=[ToolCall.from_dict(tc) for tc in raw_calls],
            finish_reason=_require(data, "finish_reason", "LLMResponse", _STR),
            # 接受两种形态：dict（to_dict 的输出，也是 §9 里 data['usage'] 的形态）
            # 与已经是 TokenUsage 的实例（调用方直接透传时省一次转换）。
            usage=raw_usage
            if isinstance(raw_usage, TokenUsage)
            else TokenUsage.from_dict(raw_usage),
            model=_require(data, "model", "LLMResponse", _STR),
            # raw 是 opt-out 字段：缺失或 None 都置 None，**不重算**（§2.2）。
            raw=_optional(data, "raw", "LLMResponse", _MAP, None),
            latency_ms=_require(data, "latency_ms", "LLMResponse", _NUM),
        )


# ======================================================================================
# 4.5 ScriptedCall
# ======================================================================================


@dataclass(slots=True)
class ScriptedCall:
    """``ScriptedLLM`` 记录的一次调用（纯测试可观测性，§6.6 规则 7）。

    冻结约定：``messages`` 是**浅拷贝**（外层 list 是新的，Message 对象本身共享），
    否则 Agent 下一轮往 ``state.messages`` 里追加消息会污染已记录的调用历史，
    "第 2 轮看到了几条消息"这类断言会静默失效。
    """

    index: int
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    kwargs: dict[str, Any]
    # [v2 新增] 本次调用消费到的脚本响应（便于断言"到底喂进去了哪一条"）。
    # 类型是 llm/scripted.py 的 ScriptedResponse：L1 不能 import L2，只能靠前向引用。
    # 加引号是照 §2.2"返回/前向类型注解写字符串"的规定逐字抄（见文件末尾的 SPEC-AMBIGUITY 说明）。
    response: "ScriptedResponse | None" = None

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出。kwargs/tools/response 可能含任意对象，统一过 ``_jsonable`` 保证可序列化。"""
        return {
            "index": self.index,
            "messages": [m.to_dict() for m in self.messages],
            "tools": None if self.tools is None else _jsonable(self.tools),
            "kwargs": _jsonable(self.kwargs),
            "response": _jsonable(self.response),
        }


# --------------------------------------------------------------------------------------
# SPEC-AMBIGUITY 记录（实现时遇到的规范内部矛盾与裁决）
# --------------------------------------------------------------------------------------
#
# SPEC-AMBIGUITY 1: §4.1 要求 TokenUsage.estimated_cost_usd 委托 config.estimate_cost_usd、
#   §4.3 要求 ToolResult.to_observation 用 config.truncate_head_tail，但 §4.5 的 import 白名单
#   禁止 types.py import config（避免 L1 变成 L2）。裁决：服从行为要求，改用**函数体内延迟 import**
#   （不产生模块级循环依赖，也保住"截断算法/价格表全项目只有一份实现"），而不是在本模块复制逻辑。
#   实测已与真实的 liteagent/config.py 对接成功（见交付说明里的验证命令）。
#
# SPEC-AMBIGUITY 2: §4.3 写 "content = self.error_text()"，而 error_text() 的公式又会在 content
#   非空时把 content 追加到前缀之后 —— 照字面实现会让第二次调用产出**两遍**前缀，与同节
#   "否则错误文本会出现两遍" 以及 §7.4.1 步骤 4.5 的固定文案冲突。裁决：把 error_text() 实现成
#   幂等（content 恰好等于前缀时不追加），使 `failure().content == failure().error_text()` 恒成立。
#
# SPEC-AMBIGUITY 3: §2.2 要求 from_dict 的类型注解写成字符串（`-> "TokenUsage"`），§2.1 又要求
#   每个文件第一行 `from __future__ import annotations`（它使所有注解在运行期本就是字符串）。
#   两者叠加后 `inspect.signature` 会显示 `-> "'TokenUsage'"`（多一层引号）。
#   裁决：逐字服从 §2.2 的写法 —— 它不影响任何运行期行为（dataclass 不会求值注解，
#   项目里也没有任何模块对这几个方法调用 typing.get_type_hints）。
#
# SPEC-AMBIGUITY 4: §4.2 的 create 文档只规定 call_id 生成与 arguments 的 JSON 校验，没说
#   raw_arguments 填什么。裁决：create 留空（"代码构造"没有原始串），而 from_arguments_json 一定
#   填上模型给的原始字符串 —— 后者是失败时回灌给模型的唯一依据。
