from __future__ import annotations

# 文本 ReAct 语法解析（`agent/parser.py`，§9.3）。
#
# 为什么把这件事单独成一个模块：native（function calling）模式下模型返回的是结构化
# `tool_calls`，解析是零成本的；文本模式下模型返回的是一段自由文本，"容错空间"几乎无穷
# （全角冒号、Markdown 加粗、中文 marker、跨行 JSON、围栏、裸值……）。把这份脏活收进一个
# **纯函数式**（无 I/O、无 asyncio、无共享可变状态）的解析器，Agent 的状态机就只剩编排逻辑；
# 代价换来的是：这份容错规则可以被离线单测穷举所有畸形输入。
#
# 冻结的核心语义（§9.3 步骤 4，v1 在这里自相矛盾过）：**Action 优先于 Final Answer**。
# 模型输出 Thought/Action/Action Input 之后再补一句 Final Answer 是常见行为（它常常是在
# 脑补"如果这次调用成功会怎样"），丢掉工具调用直接终结是明显的错误行为。
#
# 不变量（全类共享，所有分支都必须维持）：`ParsedAction.action` 与
# `ParsedAction.final_answer` **至多一个非 None**，`is_action()` 与 `is_final()` **互斥**。
#
# 线程/异步安全：本模块没有可变共享状态，`__init__` 只编译正则（与 §0.4 的 R-LOOP 无关，
# 正则对象不绑定事件循环），同一个实例可被多线程 / 多个 loop 并发调用。
#
# 注意：本文件刻意不写模块级 docstring —— §2.1 冻结"第一行必须是
# `from __future__ import annotations`"，写成字符串字面量会让它退化成无意义的表达式语句。

import ast
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from liteagent.config import DEFAULT_MAX_OBSERVATION_CHARS
from liteagent.errors import ReActParseError
from liteagent.types import LLMResponse, ToolCall, ToolResult

__all__ = [
    "ACTION_MARKERS",
    "THOUGHT_MARKERS",
    "OBSERVATION_MARKERS",
    "FINAL_MARKERS",
    "ACTION_INPUT_MARKERS",
    "ParsedAction",
    "ReActParser",
]

logger = logging.getLogger("liteagent.agent.parser")

# ---- 冻结的 marker 表（§9.3，逐字）----
ACTION_MARKERS: tuple[str, ...] = ("action", "行动", "动作")
THOUGHT_MARKERS: tuple[str, ...] = ("thought", "思考", "想法")
OBSERVATION_MARKERS: tuple[str, ...] = ("observation", "观察", "结果")
FINAL_MARKERS: tuple[str, ...] = ("final answer", "最终答案", "最终回答", "答案")

# ---- 内部但公开可测的常量 ----
#
# "Action Input" 不在 §9.3 的四张 marker 表里，但步骤 5 与 `strip_markers` 的职责描述
# 都要求把它当作一种独立行首 marker 处理（它**不能**复用 ACTION_MARKERS：`Action Input:`
# 里 "Action" 之后紧跟的是 " Input"，ACTION 正则要求 marker 后立刻（允许空白）出现冒号，
# 因此天然不会误匹配 "Action Input"）。
ACTION_INPUT_MARKERS: tuple[str, ...] = (
    "action input",
    "action_input",
    "行动输入",
    "动作输入",
)

# `json.loads` / `ast.literal_eval` 都失败时的哨兵。用私有哨兵而不是 None，
# 是因为 None 本身是合法的 JSON 值（"Action Input: null"）。
_UNSET: Any = object()

# marker 行的形态（§9.3 步骤 3，冻结）：允许前导空白/列表符号 `- * >`，
# 允许 Markdown 强调符包裹，冒号全角/半角都认。
_MARKER_TEMPLATE = r"^[\s>*\-]*(?:\*\*|__)?\s*(?P<mk>{markers})\s*(?:\*\*|__)?\s*[:：]\s*"


def _normalize(text: str) -> str:
    """步骤 1：`\\r\\n` -> `\\n`、剥离首尾空白。

    全角冒号 **不在这里** 统一替换 —— §9.3 冻结"只在 marker 匹配时用"，因为把整段文本
    里的 `：` 换成 `:` 会让后续 `text[start:end]` 的 offset 与原文错位，
    而 offset 是 `ReActParseError.offset` 与 trace 定位的依据。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _line_end(text: str, pos: int) -> int:
    """返回 `pos` 所在行的结束下标（不含换行符）。"""
    nl = text.find("\n", pos)
    return len(text) if nl == -1 else nl


def _skip_ws(text: str, start: int, limit: int | None = None) -> int:
    """跳过 `start` 起的空白，可选地被 `limit` 截断（用于"只在本行内找"的场景）。"""
    end = len(text) if limit is None else min(limit, len(text))
    i = start
    while i < end and text[i].isspace():
        i += 1
    return i


def _try_json(text: str) -> Any:
    """`json.loads` 的哨兵版：失败返回 `_UNSET`（不抛异常，由调用方决定降级路径）。"""
    try:
        return json.loads(text)
    except ValueError:  # JSONDecodeError 是 ValueError 的子类
        return _UNSET


def _try_literal(text: str) -> Any:
    """`ast.literal_eval` 的哨兵版：容忍单引号的 Python 字面量字典。

    `ast.literal_eval` 对畸形输入可能抛 SyntaxError/ValueError/TypeError，
    极端输入（超长嵌套）还会抛 MemoryError/RecursionError —— 全部吞成 `_UNSET`
    并降级到"裸值"分支（这不是静默：调用方最终会把原文塞进 `{"input": ...}` 回灌给模型，
    模型能看到自己的原文，见 §9.3 步骤 5.c）。
    """
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return _UNSET


def _decode_value(text: str) -> Any:
    """先 `json.loads`、再 `ast.literal_eval`，都失败返回 `_UNSET`（§9.3 步骤 5.c 的前两档）。

    顺序不能反：JSON 是模型的"母语"，`{"a": 1}` 必须走 JSON 路径；`literal_eval` 只是
    为单引号写法（`{'a': 1}`）兜底。两者都不成立时返回值退化成 `_UNSET`，
    由调用方决定"裸值"还是"放弃" —— 这里不抛异常，因为解析器对畸形输入的唯一出口
    是步骤 7 的 `ReActParseError`。
    """
    value = _try_json(text)
    if value is _UNSET:
        value = _try_literal(text)
    return value


def _leading_emphasis_run(text: str) -> str:
    """返回 `text` 开头连续强调符/反引号构成的 run（没有则空串）。"""
    m = re.match(r"^[*_`]+", text)
    return m.group(0) if m is not None else ""


def _strip_wrapping(value: str) -> str:
    """剥离 marker 残留与成对包裹（§9.3 步骤 6 的"strip 与首尾引号剥离"）。

    两步，顺序不能反：

    1. **成对剥皮**：`**x**` / `__x__` / ``` `x` ``` / `"x"` / `'x'` 这类首尾同形的包裹
       直接脱掉。形如 `**Action:** {}` 的输入会让 marker 正则的 `[:：]\\s*` 之后残留一个
       前导 `**`（正则把 `**` 当成了 marker 前的列表符号吃掉了），第 2 步负责它。
    2. **落单的前导 run**：只有当该 run 在整段里**只出现一次**时才剥 —— 这一条判据把
       "`** 42`（marker 残留）"与 "`**Bold** answer`（模型真的在用 Markdown）"区分开，
       后者剥了就会破坏正文。
    """
    text = value.strip()
    for _ in range(3):  # 有限次：`**"x"**` 这类双层包裹需要两轮
        peeled = False
        for opener in ("```", "**", "__", '"', "'", "`"):
            if len(text) > 2 * len(opener) and text.startswith(opener) and text.endswith(opener):
                text = text[len(opener):len(text) - len(opener)].strip()
                peeled = True
                break
        if not peeled:
            break
    run = _leading_emphasis_run(text)
    if run and text.count(run) == 1:
        text = text[len(run):].strip()
    return text


def _coerce_text(value: Any, *, where: str) -> str:
    """把 JSON 里的值转成文本（`final_answer` / `thought` / `action` 共用）。

    非字符串（模型偶尔会返回 `{"final_answer": {"text": ...}}`）走 `json.dumps`，
    并留一条 warning —— §13 红线 12：任何降级都不能静默。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)) and not isinstance(value, (bytes, bytearray)):
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
    else:
        rendered = str(value)
    logger.warning("parser: expected a string for %s, coerced %s -> %r",
                   where, type(value).__name__, rendered[:120])
    return rendered


@dataclass
class ParsedAction:
    """一次模型输出的解析结果（§9.3，字段名冻结）。

    为什么 `action_input` 是 `dict` 而不是原始字符串：Agent 的下游是
    `ToolCall.create(name, arguments)` + executor 的参数校验，越早归一化成 dict，
    错误就越早以"参数校验失败（recoverable）"的形式回灌给模型（§3.4 的第二层分类）。
    """

    thought: str | None = None
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    raw: str = ""
    json_mode: bool = False          # True 表示来自整段 JSON 解析

    def is_final(self) -> bool:
        """是否是一次终结（有 Final Answer）。与 `is_action()` 互斥。"""
        return self.final_answer is not None

    def is_action(self) -> bool:
        """是否需要执行工具。**优先于** `is_final()` 判断（§9.3 步骤 4）。"""
        return self.action is not None

    def to_dict(self) -> dict[str, Any]:
        """§2.2：字段全量输出（含 None），保证 trace 结构稳定、可精确比对。"""
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": dict(self.action_input),
            "final_answer": self.final_answer,
            "raw": self.raw,
            "json_mode": self.json_mode,
        }


def _build_marker_regex(markers: Sequence[str], *,
                        allow_cjk_markers: bool,
                        case_sensitive: bool) -> "re.Pattern[str]":
    """构造 §9.3 步骤 3 的冻结正则。

    `allow_cjk_markers=False` 时只保留 ASCII marker（每个 marker 表的第一项是 ASCII 形态，
    这是四张表的固定约定，不是巧合）。
    """
    names = list(markers) if allow_cjk_markers else [markers[0]]
    # 长的排前面：否则 "最终答案:" 可能被更短的 "答案:" 在交替分支里抢先。
    # （正则锚定在行首，理论上不会误匹配，但让交替顺序与"最长优先"一致更不容易踩坑。）
    names = sorted(set(names), key=len, reverse=True)
    pattern = _MARKER_TEMPLATE.format(markers="|".join(re.escape(n) for n in names))
    flags = re.MULTILINE if case_sensitive else (re.MULTILINE | re.IGNORECASE)
    return re.compile(pattern, flags)


class ReActParser:
    """文本 ReAct 解析器（§9.3）。无状态、可复用、线程安全。"""

    def __init__(self, *, tool_names: Sequence[str] | None = None,
                 tool_param_names: Mapping[str, Sequence[str]] | None = None,
                 strict: bool = True,
                 allow_json_block: bool = True,
                 allow_json_object: bool = True,
                 allow_final_answer: bool = True,
                 case_sensitive: bool = False,
                 allow_cjk_markers: bool = True) -> None:
        # tool_names 归一化成 tuple：调用方常传 `registry.names()`（每次新 list），
        # 冻结成不可变副本既避免外部改动影响解析器，也让 `in` 判断语义明确。
        self.tool_names: tuple[str, ...] | None = (
            None if tool_names is None else tuple(str(n) for n in tool_names)
        )
        self.tool_param_names: dict[str, tuple[str, ...]] | None = (
            None if tool_param_names is None
            else {str(k): tuple(str(p) for p in v) for k, v in tool_param_names.items()}
        )
        self.strict = bool(strict)
        self.allow_json_block = bool(allow_json_block)
        self.allow_json_object = bool(allow_json_object)
        self.allow_final_answer = bool(allow_final_answer)
        self.case_sensitive = bool(case_sensitive)
        self.allow_cjk_markers = bool(allow_cjk_markers)

        kw = {"allow_cjk_markers": self.allow_cjk_markers,
              "case_sensitive": self.case_sensitive}
        self._action_re = _build_marker_regex(ACTION_MARKERS, **kw)
        self._thought_re = _build_marker_regex(THOUGHT_MARKERS, **kw)
        self._observation_re = _build_marker_regex(OBSERVATION_MARKERS, **kw)
        self._action_input_re = _build_marker_regex(ACTION_INPUT_MARKERS, **kw)
        # allow_final_answer=False 时**完全不认** Final Answer marker（它退化成普通文本），
        # 于是"只有一段自由文本"的输出在 strict 下会走步骤 7 的解析失败 → 自纠正，
        # 这正是调用方关掉它的目的。
        self._final_re: "re.Pattern[str] | None" = (
            _build_marker_regex(FINAL_MARKERS, **kw) if self.allow_final_answer else None
        )

    # ------------------------------------------------------------------ 公开 API

    def parse(self, text: str) -> ParsedAction:
        """解析顺序（§9.3 冻结的 8 步）。

        摘要：1) 规范化 → 2) 整段 JSON 对象 → 3) 逐行 marker 扫描 →
        4) **Action 优先**（有合法 Action 行就执行 Action，Final Answer 丢弃并 warning）→
        5) Action Input 取值（围栏 → 花括号配对 → 单行文本 → 空）→
        6) Final Answer 取值（截断到下一个 Thought/Action marker）→
        7) strict 下无任何结构 -> `ReActParseError(reason="no ReAct structure found")`，
        非 strict 下整段当作 final_answer → 8) action 名不在 tool_names 里**不抛异常**。

        注意步骤 8：把"未知工具名"的裁决权留给 Agent 是刻意的 —— Agent 要在那里发
        `TOOL_ERROR` / 注入 nudge，而解析器只知道语法、不知道运行时策略（更可测）。
        """
        # SPEC-AMBIGUITY: 步骤 2（整段 JSON 时 final_answer 判定在前）与步骤 4
        # （"Action 优先于 Final Answer"）在"同一个 dict 同时含 action 与 final_answer"时
        # 结论相反。按 §0 的裁决规则取字面规定：整段 JSON 路径按步骤 2 的书写顺序
        # （final_answer 优先）；步骤 4 只管辖文本 marker 路径。
        raw = text if isinstance(text, str) else ("" if text is None else str(text))
        norm = _normalize(raw)

        # --- 步骤 2：整段 JSON 对象 ---
        if self.allow_json_object:
            from_json = self._parse_json_object(norm, raw)
            if from_json is not None:
                return from_json

        # --- 步骤 3/4：marker 扫描，Action 优先 ---
        return self._parse_markers(norm, raw)

    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """**只做原生提取**（§9.3）：返回 `response.tool_calls` 的副本。

        `response.tool_calls` 为空时返回 `[]` —— 永不回落到文本解析。这样 `mode="native"`
        的语义不会漂移（文本模式由 Agent 直接调 `parse()`，两条路径的入口不同）。

        返回副本而不是原对象：`AgentState.tool_calls` 会长期持有这些 call，
        外部provider 的响应对象不得被状态机反向改写（§13 红线 11）。
        """
        calls: list[ToolCall] = []
        for call in list(getattr(response, "tool_calls", None) or ()):
            arguments = dict(call.arguments or {})
            if not arguments:
                defaults = self._default_arguments(call.name)
                if defaults:
                    logger.debug("parse_tool_calls: filled empty arguments for %s with %s",
                                 call.name, sorted(defaults))
                    arguments = defaults
            calls.append(ToolCall(
                id=call.id,
                name=call.name,
                arguments=arguments,
                raw_arguments=call.raw_arguments,
                metadata=dict(call.metadata or {}),
            ))
        return calls

    def extract_thought(self, text: str) -> str | None:
        """只取 Thought 段（不需要完整结构）。无 marker 或值为空时返回 None。"""
        norm = _normalize(text if isinstance(text, str) else str(text))
        match = self._thought_re.search(norm)
        if match is None:
            return None
        return self._thought_value(norm, match)

    def strip_markers(self, text: str) -> str:
        """删除 Thought/Action/Action Input/Observation 行，返回剩余文本（§9.3）。

        用于最终答案清洗（§9.4.3：`answer = parser.strip_markers(answer).strip()`）。
        四条 marker 行**连同该行正文一起删掉**（§9.3 docstring 的字面规定）；唯一例外是
        Final Answer 行 —— 它不在"删除"清单里，而 §9.4.3 的注释又明确要求
        "去掉可能残留的 'Final Answer:' 前缀"，所以对它**只剥前缀、保留同行正文**。
        """
        # SPEC-AMBIGUITY: "删除 Observation 行"（本方法 docstring）与"Observation: 前缀剥离"
        # （§9.3 步骤 6 的清洗要求）字面上冲突。裁决：本方法按 docstring 逐字删整行；
        # "前缀剥离"由步骤 6 的 _strip_leading_marker 负责，两处职责不重叠。
        norm = _normalize(text if isinstance(text, str) else str(text))
        kept: list[str] = []
        for line in norm.split("\n"):
            if (self._action_input_re.match(line) or self._action_re.match(line)
                    or self._thought_re.match(line) or self._observation_re.match(line)):
                continue
            if self._final_re is not None:
                match = self._final_re.match(line)
                if match is not None:
                    kept.append(line[match.end():].rstrip())
                    continue
            kept.append(line)
        return "\n".join(kept).strip()

    def build_observation(self, results: Sequence[ToolResult], *,
                          max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
                          step: int | None = None) -> str:
        """多条工具结果的渲染（格式冻结，§9.3）：

        单个：`Observation: <obs>`
        多个：`Observation:\\n[1] <name> -> <obs1>\\n[2] <name> -> <obs2>`

        单条结果不编号是刻意的：绝大多数 step 只有一条结果，编号只会白烧 token；
        多条时编号让模型能引用"[2] 的结果"（§9.4.5：文本模式一轮一个 action，
        多结果只可能来自 `execute_many` 的并行调用）。

        `step` 不参与冻结格式（§9.3 的格式里没有轮次信息），只用于 debug 日志留痕。
        """
        items = list(results)
        if not items:
            # 未定义行为：宁可返回空串也不编造文案，但必须留痕（§13 红线 12）。
            logger.warning("build_observation called with no results (step=%s)", step)
            return ""
        per_item = max_chars // max(1, len(items))
        if per_item <= 0:
            logger.warning("build_observation: per-item budget is %d (max_chars=%d, n=%d); "
                           "results will not be truncated", per_item, max_chars, len(items))
        observations = [r.to_observation(max_chars=per_item) for r in items]
        if len(observations) == 1:
            text = f"Observation: {observations[0]}"
        else:
            lines = ["Observation:"]
            for index, (result, obs) in enumerate(zip(items, observations), start=1):
                lines.append(f"[{index}] {result.name} -> {obs}")
            text = "\n".join(lines)
        logger.debug("build_observation: step=%s results=%d chars=%d",
                     step, len(items), len(text))
        return text

    def build_parse_error_feedback(self, error: ReActParseError) -> str:
        """把解析失败回灌给模型的提示文本（格式冻结，§9.3）。

        为什么反馈里要重复一遍完整格式：解析失败的模型下一轮往往"换个姿势"继续失败，
        把格式原文再贴一遍是最便宜的收敛手段（§9.4.2 的自纠正分支会把它作为 nudge 注入）。
        """
        names = ", ".join(self.tool_names or ())
        if not names:
            logger.warning("build_parse_error_feedback: no tool_names configured; "
                           "the model cannot be told which tools exist")
        reason = (getattr(error, "reason", "") or str(error) or "unknown error").strip()
        if not reason.endswith((".", "!", "?", "。")):
            reason += "."
        return (
            f"Your previous output could not be parsed: {reason}\n"
            "Reply using exactly this format:\n"
            f"Thought: <your reasoning>\nAction: <one of [{names}]>\nAction Input: <JSON object>\n"
            "or\n"
            "Thought: <...>\nFinal Answer: <...>"
        )

    # --------------------------------------------------------------- 步骤 2：JSON

    def _parse_json_object(self, norm: str, raw: str) -> ParsedAction | None:
        """整段就是一个 JSON 对象时的解析（§9.3 步骤 2）。

        键的判定顺序是 final_answer -> action -> tool/tool_name（`parse` 里有对应的
        SPEC-AMBIGUITY 说明）；解析不出任何结构时返回 None，表示"继续步骤 3"，
        **不是**错误 —— 整段 JSON 与行首 marker 可以在同一段文本里共存。
        """
        stripped = norm.strip()
        if not stripped.startswith("{"):
            return None  # 快速排除：整段 JSON 一定是对象形态
        data = _try_json(stripped)
        if data is _UNSET or not isinstance(data, dict):
            return None  # 不是合法 JSON 对象 -> 继续步骤 3（这不是错误，行内可能还有 marker）
        if data.get("final_answer") is not None:
            value = _coerce_text(data["final_answer"], where="final_answer").strip()
            # 值可能是 None（"final_answer": null）—— 那不算结构，继续往下看 action。
            return ParsedAction(final_answer=value, raw=raw, json_mode=True)
        name = ""
        for key in ("action", "tool", "tool_name"):  # tool/tool_name 是 §9.3 冻结的容错别名
            candidate = data.get(key)
            if candidate:
                name = _strip_wrapping(_coerce_text(candidate, where=key))
                break
        if not name:
            logger.debug("parser: JSON object has no usable action/final_answer key; "
                         "keys=%s", sorted(data))
            return None
        thought_raw = data.get("thought")
        thought = None
        if thought_raw is not None:
            cleaned = _strip_wrapping(_coerce_text(thought_raw, where="thought"))
            thought = cleaned or None
        return ParsedAction(
            thought=thought,
            action=name,
            action_input=self._coerce_input_map(data, where="action_input"),
            raw=raw,
            json_mode=True,
        )

    def _coerce_input_map(self, data: Mapping[str, Any], *, where: str) -> dict[str, Any]:
        """把 JSON 里的参数载荷统一成 dict。

        §9.3 只冻结了 `action_input`；`arguments` 是判定的容错别名（function calling 风格的
        载荷几乎总叫 arguments，模型把它和文本 ReAct 混用很常见）。字符串载荷会先尝试
        JSON/literal 解析，失败则原样包进 `{"input": ...}` —— 保留信息比丢弃更利于自纠正。
        """
        # SPEC-AMBIGUITY: 别名列表里的 arguments / tool_input 不在规范里（规范只冻结了
        # action_input），属于保守扩展；不认识它们只会让模型拿到空参数、白耗一轮自纠正。
        for key in (where, "action_input", "arguments", "tool_input"):
            if key not in data:
                continue
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items()}
            if isinstance(value, str):
                parsed = _try_json(value)
                if parsed is _UNSET:
                    parsed = _try_literal(value)
                if isinstance(parsed, Mapping):
                    logger.debug("parser: parsed string %s payload as a mapping", key)
                    return {str(k): v for k, v in parsed.items()}
            logger.debug("parser: %s is not a mapping (%s); wrapped as {'input': ...}",
                         key, type(value).__name__)
            return {"input": value}
        return {}

    # ------------------------------------------------------- 步骤 3/4：marker 扫描

    def _parse_markers(self, norm: str, raw: str) -> ParsedAction:
        # 先把 Final Answer marker 找出来：命中 Action 时要靠它决定是否记 "丢弃 final" 的 warning
        final_match = self._final_re.search(norm) if self._final_re is not None else None

        # --- 步骤 4：先扫描 Action marker，只要存在**合法**的 Action 行就解析为 action ---
        for match in self._action_re.finditer(norm):
            parsed = self._parse_action_at(norm, match, raw)
            if parsed is None:
                continue  # 空名字的 Action 行不算"合法"，继续找下一行
            if final_match is not None and final_match.start() > match.start():
                # §9.3 步骤 4：同段落里的 Final Answer 文本丢弃并记 warning。
                # 这条 warning 是"模型行为统计"的信号：频繁出现说明 prompt 需要收紧。
                logger.warning(
                    "parser: both Action and Final Answer found; executing Action %r and "
                    "discarding the Final Answer text (offset=%d)",
                    parsed.action, final_match.start())
            parsed.thought = self._thought_from_text(norm)
            return parsed

        # --- 步骤 6：仅当**没有任何** Action marker 时才取 Final Answer ---
        if final_match is not None:
            return ParsedAction(
                thought=self._thought_from_text(norm),
                final_answer=self._final_value(norm, final_match),
                raw=raw,
            )

        thought = self._thought_from_text(norm)
        if thought is not None:
            # 只有 Thought 不算解析失败（§9.3 步骤 7 要求三者皆无才抛），
            # Agent 会走"保守终结分支"（§9.4.1 步骤 3 的 else）。
            return ParsedAction(thought=thought, raw=raw)

        if self.strict:
            raise ReActParseError(raw=raw, offset=0, reason="no ReAct structure found")
        # 非严格模式（§9.3 步骤 7）：整段当作最终答案。留痕，否则调用方无法区分
        # "模型真的这么答"与"解析器放弃了"。
        logger.warning("parser: no ReAct structure found in %d chars; "
                       "falling back to final_answer in non-strict mode", len(norm))
        return ParsedAction(final_answer=norm, raw=raw)

    def _parse_action_at(self, text: str, match: "re.Match[str]",
                         raw: str) -> ParsedAction | None:
        """解析一个 `Action:` 行及其输入。名字为空 -> 返回 None（该行不合法）。"""
        line_value = text[match.end():_line_end(text, match.end())]
        name = self._resolve_action_name(line_value)
        if not name:
            logger.warning("parser: Action marker at offset %d has an empty tool name; "
                           "ignoring it", match.start())
            return None
        input_match = self._action_input_re.search(text, match.end())
        if input_match is not None:
            action_input = self._parse_action_input(text, input_match.end(), name)
        else:
            # 没有 Action Input 行：只在 action 行**本行内**找 `{...}`（"Action: f {..}"）。
            # 刻意不做单行裸值兜底 —— 否则 "Action: search" 会把工具名当成参数
            # 变成 {"input": "search"}，# 这是最容易踩的假阳性。
            action_input = self._parse_trailing_input(text, match.end())
        return ParsedAction(action=name, action_input=action_input, raw=raw)

    def _resolve_action_name(self, value: str) -> str:
        """从 `Action:` 行剩余文本里取工具名（§9.3 步骤 8：未知名字**不抛异常**）。

        容错：模型常写 "Action: search the web" / "Action: search(query=...)" /
        "Action: search."，这里按"合法名字前缀"逐个候选回退，**只在候选命中已知工具名时**
        才替换（否则保持原文，交由 Agent 判"未知工具"并回灌可用工具列表）。
        任何替换都记 debug —— 这是可观测的降级（§13 红线 12）。
        """
        name = _strip_wrapping(value)
        if not name:
            return ""
        if self.tool_names is None or name in self.tool_names:
            return name
        candidates = []
        prefix = re.match(r"^([A-Za-z_][A-Za-z0-9_.\-]*)", name)
        if prefix is not None:
            candidates.append(prefix.group(1))
        head = re.split(r"\s+", name, maxsplit=1)[0]
        candidates.append(head)
        for candidate in candidates:
            if candidate and candidate in self.tool_names:
                logger.debug("parser: cleaned action name %r -> %r", name, candidate)
                return candidate
        return name

    # ------------------------------------------------- 步骤 5：Action Input 的取值

    def _parse_action_input(self, text: str, start: int, name: str) -> dict[str, Any]:
        """§9.3 步骤 5 的冻结顺序：a 围栏 → b 花括号配对 → c 单行文本 → d 空。

        三个分支的**收口规则一致**：载荷解出映射（dict）就用它；解出非映射（标量/列表）
        或干脆解不出，就交给 `_bare_value` 做"裸值"处理（单参数工具 -> {参数名: 值}，
        否则 -> {"input": 值}）。统一收口是刻意的：同一种畸形输入不应该因为出现在
        围栏里还是行内而得到两种参数结构。
        """
        # a. 紧随其后的 ```json / ``` 围栏块（这个分支被 allow_json_block 开关控制）
        if self.allow_json_block:
            fence = _fence_block(text, start)
            if fence is not None:
                # 围栏是显式的载荷定界符：无论内容是否映射，都在这里收口 ——
                # 若继续往下走，单行文本分支只会读到游离的 "```"。
                return self._settle_payload(fence, name=name, source="fence")
        # b. 花括号配对扫描（处理跨行 JSON）
        braced = _brace_block(text, start)
        if braced is not None:
            decoded = _decode_value(braced)
            if isinstance(decoded, Mapping):
                return {str(k): v for k, v in decoded.items()}
            logger.debug("parser: brace block for %r is not a mapping (%s); "
                         "falling through to inline text", name, type(decoded).__name__)
        # c. 单行剩余文本
        inline = text[start:_line_end(text, start)].strip()
        if not inline:
            return {}  # d. 空 -> {}
        return self._settle_payload(inline, name=name, source="inline")

    def _settle_payload(self, payload: str, *, name: str, source: str) -> dict[str, Any]:
        """把一段载荷文本收口成参数 dict（映射直接用，非映射走裸值规则）。"""
        text = payload.strip()
        if not text:
            return {}
        decoded = _decode_value(text)
        if isinstance(decoded, Mapping):
            return {str(k): v for k, v in decoded.items()}
        if decoded is _UNSET:
            logger.debug("parser: %s payload for %r is neither JSON nor a literal (%r)",
                         source, name, text[:80])
            return self._bare_value(text, name)
        logger.debug("parser: %s payload for %r decoded to %s, not a mapping",
                     source, name, type(decoded).__name__)
        return self._bare_value(decoded, name)

    def _bare_value(self, value: Any, name: str) -> dict[str, Any]:
        """裸值容错（§9.3 步骤 5.c 的后两档）。

        仅当**恰好一个**已知参数时才敢把裸值绑到它上面；参数多于一个时无法判断裸值
        属于谁，宁可回落到 `{"input": 值}` —— 模型能从中看到自己的原文，
        比我们猜错参数名更容易自纠正。
        """
        if self.tool_names is not None and name in self.tool_names:
            params = self._known_params(name)
            if len(params) == 1:
                logger.debug("parser: bare value for single-parameter tool %r -> %r",
                             name, params[0])
                return {params[0]: value}
        return {"input": value}

    def _parse_trailing_input(self, text: str, start: int) -> dict[str, Any]:
        """没有 Action Input 行时的兜底：只认同一行内的**映射** `{...}`（其余一律 `{}`）。

        这里刻意不放宽到裸值：`Action: search` 后面的散文（或工具名本身）一旦被当成
        参数，就会产生"看似成功、参数全错"的假阳性，比空参数更难排查。
        """
        line_end = _line_end(text, start)
        segment = text[start:line_end]
        if segment.strip():
            braced = _brace_block(segment, 0)
            if braced is not None:
                decoded = _decode_value(braced)
                if isinstance(decoded, Mapping):
                    return {str(k): v for k, v in decoded.items()}
                logger.debug("parser: trailing braces are not a mapping (%s)",
                             type(decoded).__name__)
        return {}

    def _known_params(self, name: str) -> tuple[str, ...]:
        """取已知工具的参数名（无信息时返回空 tuple）。"""
        # SPEC-AMBIGUITY: §9.3 步骤 5.c 说"该工具恰有 1 个**必填**参数"，但
        # `tool_param_names` 的语义（只给必填，还是给全部参数）规范未定义。裁决：按
        # "调用方给的就是它愿意被裸值填充的参数名"处理 —— **恰好一个**才敢猜；多个时
        # 无法判断裸值属于谁，宁可回落到 `{"input": ...}` 让模型看到自己的原文。
        if self.tool_param_names is None:
            return ()
        return tuple(self.tool_param_names.get(name, ()))

    def _default_arguments(self, name: str) -> dict[str, Any]:
        """`parse_tool_calls` 的"arguments 为空时按 schema 补默认值"（§9.3）。"""
        # SPEC-AMBIGUITY: 解析器手里**没有** JSON Schema（构造函数只收 tool_names /
        # tool_param_names），拿不到真正的 default 值。裁决：用可得信息做最保守的填充 ——
        # 把 `tool_param_names[name]` 列出的参数补成 None；完全没有参数信息时返回 {}
        # （不编造参数名：编错的名字只会让模型更难自纠正）。
        params = self._known_params(name)
        if not params:
            return {}
        return {param: None for param in params}

    # ------------------------------------------------------- 步骤 6：Final Answer

    def _final_value(self, text: str, match: "re.Match[str]") -> str:
        """Final Answer 的取值（§9.3 步骤 6）：marker 之后到文档尾，
        遇到下一个 Thought/Action marker 则截断到那里；做 strip 与首尾引号剥离。"""
        start = match.end()
        end = len(text)
        for pattern in (self._action_re, self._thought_re):
            nxt = pattern.search(text, start)
            if nxt is not None and nxt.start() < end:
                end = nxt.start()
        value = _strip_wrapping(text[start:end])
        return self._strip_leading_marker(value)

    # ------------------------------------------------------------ Thought 的取值

    def _thought_from_text(self, text: str) -> str | None:
        match = self._thought_re.search(text)
        if match is None:
            return None
        return self._thought_value(text, match)

    def _thought_value(self, text: str, match: "re.Match[str]") -> str | None:
        """Thought 段的值：marker 之后到下一个任意 marker 行（支持跨行思考）。

        取到下一个 marker 而不是行尾，是因为模型经常写成
        `Thought: 先看 A\\n再看 B\\nAction: ...`；截断在行尾会丢掉一半推理，
        而推理恰恰是 trace / 面试演示里最有价值的部分。

        值为空（裸 `Thought:` 行）时返回 None：它不构成"有结构"，
        否则 strict 模式会把一条无意义的输出当成合法结果放过去。
        """
        start = match.end()
        end = len(text)
        patterns = [self._action_re, self._thought_re, self._observation_re]
        if self._final_re is not None:
            patterns.append(self._final_re)
        for pattern in patterns:
            nxt = pattern.search(text, start)
            if nxt is not None and nxt.start() < end:
                end = nxt.start()
        value = self._strip_leading_marker(_strip_wrapping(text[start:end]))
        if not value:
            logger.debug("parser: Thought marker at offset %d has no content", match.start())
            return None
        return value

    def _strip_leading_marker(self, value: str) -> str:
        """剥离值开头残留的 marker 前缀（"Observation: 前缀剥离"，§9.3 步骤 6 的清洗）。

        模型偶尔会把上一轮的 `Observation:` 一起复述出来（尤其在把它当上下文续写时），
        留在最终答案里会污染输出。最多剥 4 层，避免畸形输入把这里变成循环。
        """
        text = value
        patterns = [self._observation_re, self._action_re, self._thought_re,
                    self._action_input_re]
        if self._final_re is not None:
            patterns.append(self._final_re)
        for _ in range(4):
            for pattern in patterns:
                matched = pattern.match(text)
                if matched is not None:
                    text = text[matched.end():].lstrip()
                    break
            else:
                return text
        return text


# ------------------------------------------------------------------ 模块级工具函数


def _fence_block(text: str, start: int, *, limit: int | None = None) -> str | None:
    """读取 `start` 之后紧随的 ``` 围栏块内容（未闭合时返回 None，交给花括号扫描兜底）。"""
    pos = _skip_ws(text, start, limit)
    if limit is not None and pos >= limit:
        return None
    if not text.startswith("```", pos):
        return None
    newline = text.find("\n", pos)
    if newline == -1:
        return None
    close = text.find("```", newline + 1)
    if close == -1:
        logger.debug("parser: unterminated code fence at offset %d", pos)
        return None
    return text[newline + 1:close].strip()


def _brace_block(text: str, start: int, *, limit: int | None = None) -> str | None:
    """从 `start` 起做**花括号配对**扫描，返回第一个配平的 `{...}` 原文。

    为什么不能直接用正则：Action Input 常常是跨行 JSON（§9.3 步骤 5.b 明确要求处理），
    而正则表达不了配对。扫描同时跟踪字符串状态与转义，避免把 `{"a": "}"}` 提前截断。
    接受单引号作为字符串定界符是为了兼容 Python 字面量（`{'a': 1}`）。
    """
    end_limit = len(text) if limit is None else min(limit, len(text))
    open_at = text.find("{", start, end_limit)
    if open_at == -1:
        return None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(open_at, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in ('"', "'"):
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:index + 1]
    logger.debug("parser: unbalanced braces starting at offset %d", open_at)
    return None
