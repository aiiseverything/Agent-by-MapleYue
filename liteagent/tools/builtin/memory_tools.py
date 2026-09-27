from __future__ import annotations

"""长期记忆工具（规范 §7.5 的 ``memory_tools.py``）：``remember`` / ``recall``。

这两个工具的存在理由：**让模型自己决定"什么值得记"**。用户的偏好、项目约定、
踩过的坑，靠启发式（``should_auto_write``）只能捞到一部分；给模型一个显式的
``remember``，长期记忆就从"框架猜"变成"模型声明 + 框架执行"。

**同步函数怎么调异步 API**（面试会问的细节）：``MemoryManager.aremember`` 是 async 的
（未来换 ``RemoteEmbedder`` 时它真的会 await 一次网络调用），而工具是**同步函数**、
跑在 executor 的 worker 线程里 —— 线程里没有别人的事件循环，因此
``config.run_sync(lambda: memory.aremember(...))`` 新建一个 loop 执行是**安全且期望**
的行为（不会嵌套、不会与调用方的 loop 打架；若真在本线程发现有运行中的 loop，
``run_sync`` 会抛 ``ConfigError`` 而不是偷偷 nest_asyncio）。

``recall`` 则不需要 loop：``MemoryManager.retrieve`` 是同步的纯内存检索
（无 LLM、无网络），直接调用即可。
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from liteagent.config import format_ts, run_sync
from liteagent.errors import ConfigError, ToolValidationError
from liteagent.tools.base import Tool, make_function_tool

if TYPE_CHECKING:  # pragma: no cover - E10：注解用 TYPE_CHECKING，运行期 duck-typing
    from liteagent.memory.manager import MemoryManager

__all__ = ["make_memory_tools"]


def _remember(_memory: "MemoryManager", content: str, importance: float = 0.5) -> str:
    """Store a durable fact about the user or the task in long-term memory.

    ``importance`` 的取值范围是 0..1（越界由 schema 的 minimum/maximum 拦截）。
    """
    item = run_sync(
        lambda: _memory.aremember(content, importance=float(importance), source="tool:remember")
    )
    return f"Stored a long-term memory (id={item.id})."


def _recall(_memory: "MemoryManager", query: str, limit: int = 5) -> str:
    """Search long-term memory and return the most relevant entries."""
    items = _memory.retrieve(query, limit=int(limit))
    if not items:
        # 空结果也要说人话：模型据此才会换关键词，而不是以为工具坏了。
        return f"No long-term memory matched {query!r}."
    lines: list[str] = []
    for item in items:
        score = item.score if item.score is not None else 0.0
        # 日期用 config.format_ts（唯一时钟链路的展示端），模型靠它判断记忆是否过时。
        stamp = format_ts(item.created_at, fmt="%Y-%m-%d")
        lines.append(f"score={score:.2f} | {stamp} | {item.content}")
    return "\n".join(lines)


_REMEMBER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "The fact to remember, written as a standalone sentence.",
        },
        "importance": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How important this fact is (0..1).",
        },
    },
    "required": ["content"],
    "additionalProperties": False,
}

_RECALL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for in long-term memory."},
        "limit": {"type": "integer", "description": "Maximum number of memories to return."},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _call_args(
    args: dict[str, Any],
    *,
    tool_name: str,
    required: Sequence[str],
    optional: Sequence[str],
) -> dict[str, Any]:
    """整理成可 ``**`` 到纯实现的 kwargs（与 ``files._call_args`` 同一套约定）。

    每个模块各留一份而不是共享：``tools/builtin/`` 内部的同层 import 只允许 §1.1 的
    **E8**（``shell/code -> files``，因为 ``PathSandbox`` 在那里）。为一个 12 行的
    参数整理函数新增一条同层依赖边，会让守门测试的"允许边清单"变成一句空话。
    """
    missing = [key for key in required if args.get(key) is None]
    if missing:
        raise ToolValidationError(
            errors=[f"{key!r} is a required property" for key in missing],
            tool_name=tool_name,
        )
    cleaned = {key: args[key] for key in required}
    for key in optional:
        value = args.get(key)
        if value is not None:
            cleaned[key] = value
    return cleaned


def make_memory_tools(memory: "MemoryManager") -> list[Tool]:
    """返回 ``[remember, recall]``；闭包捕获 ``memory`` 实例。

    ``memory`` 为 ``None`` 时抛 ``ConfigError``：与 ``make_file_tools`` 同一个理由 ——
    "工具悄悄什么都不记"比"启动时报错"难排查得多（``register_all`` 也只在
    ``memory is not None`` 时注册本组）。
    """
    if memory is None:
        raise ConfigError(
            "make_memory_tools(memory) requires a MemoryManager instance; "
            "pass register_all(memory=...) or skip the 'memory' tool group"
        )

    def remember(args: dict[str, Any]) -> str:
        cleaned = _call_args(
            args,
            tool_name="remember",
            required=("content",),
            optional=("importance",),
        )
        return _remember(memory, **cleaned)

    def recall(args: dict[str, Any]) -> str:
        cleaned = _call_args(args, tool_name="recall", required=("query",), optional=("limit",))
        return _recall(memory, **cleaned)

    return [
        make_function_tool(
            name="remember",
            description=(
                "Store a durable fact in long-term memory (user preferences, project "
                "conventions, decisions). Use it when the user says 'remember ...'."
            ),
            parameters=_REMEMBER_PARAMETERS,
            func=remember,
            tags=("memory",),
            idempotent=True,
        ),
        make_function_tool(
            name="recall",
            description=_recall.__doc__ or "Search long-term memory.",
            parameters=_RECALL_PARAMETERS,
            func=recall,
            tags=("memory",),
            idempotent=True,
        ),
    ]
