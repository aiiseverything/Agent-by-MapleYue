from __future__ import annotations

"""liteagent 的公共 API 门面（规范 §1.2 / 附录 B）。

**取值方式：按需解析（PEP 562 的模块级 ``__getattr__``），而不是一排顶层 import。**
理由不是"好看"，而是三条硬约束：

1. 包内大量模块写着 ``from liteagent import errors as _errors`` —— 任何一次子模块 import
   都会**执行本文件**。若这里顶层 import 全部 40 个模块，任何一个兄弟模块有问题就会
   让"只想用 ``liteagent.types``"的人一起炸，import 错误与被导入的东西毫无关系。
2. 附录 B 的守门用例要求的是**行为**："``__all__`` 里每个名字都能 ``getattr`` 到"。
   按需解析满足它，而且未定义的名字依旧老老实实抛 ``ImportError``/``AttributeError``，
   不会变成静默的占位对象。
3. "不得在 import 时做重活"（附录 B 末句）：不扫描环境、不发请求、不注册工具。
   本文件顶层只有 stdlib，一次 import 只做一次 dict 构造。
"""

from typing import Any

#: 版本号回退值（附录 B 冻结的字面量）：运行时优先问 importlib.metadata，
#: 拿到发行版元数据时以它为准（打包安装后的真实版本）。
_FALLBACK_VERSION: str = "0.1.0"


def _resolve_version() -> str:
    """版本号的**唯一来源**是 ``importlib.metadata.version("liteagent")``。

    源码树里直接 ``python3 -m liteagent`` 时发行版并未安装（``PackageNotFoundError``），
    回退到与 ``pyproject.toml`` 对齐的字面量。任何元数据层的异常都不该让
    ``import liteagent`` 失败 —— 版本号再重要也不值得赔上整个包。
    """
    try:
        from importlib.metadata import version as _distribution_version

        return _distribution_version("liteagent")
    except Exception:  # PackageNotFoundError / 元数据目录损坏 / 权限问题
        return _FALLBACK_VERSION


__version__ = _resolve_version()

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

#: 名字 -> 定义它的模块。**每个名字只在这里出现一次**，`__all__` 与它的交集由下面
#: 的漂移检查守住 —— 两张手写清单迟早会分叉，机器检查比注释可靠。
#: 指向具体模块（而不是各包的 `__init__`）是刻意的：`__init__` 也是并行交付物，
#: 少一层间接就少一处"名字在包里、但包还没写出来"的中间态。
_EXPORT_SOURCES: dict[str, tuple[str, ...]] = {
    "liteagent.errors": (
        "LiteAgentError", "ConfigError", "ToolError", "ToolNotFoundError",
        "ToolValidationError", "ToolExecutionError", "ToolTimeoutError", "ToolSkippedError",
        "ToolRetryExhaustedError", "ToolApprovalDeniedError", "ToolDefinitionError",
        "LLMError", "LLMRateLimitError", "LLMTimeoutError",
        "LLMConnectionError", "LLMAuthError", "AgentError", "MaxStepsExceededError",
        "RepeatedActionError", "ReActParseError", "MultiAgentError", "DelegationError",
        "MaxDepthExceededError", "CycleDetectedError", "VersionConflictError",
        "SandboxViolationError", "ScriptedExhaustedError", "SerializationError",
        "BudgetExceededError", "RunTimeoutError", "AgentAbortedError", "MemoryStoreError",
    ),
    "liteagent.config": (
        "AppConfig", "AgentConfig", "LLMConfig", "MemoryConfig", "ExecutorConfig",
        "TeamConfig", "RetryPolicy", "LoopBoundPool", "run_sync", "utc_now", "format_ts",
        "to_jsonable", "parse_dotenv", "load_dotenv", "parse_bool", "render_template",
        "estimate_cost_usd", "MODEL_PRICES", "NO_TIMEOUT",
    ),
    "liteagent.types": (
        "TokenUsage", "ToolCall", "ToolResult", "LLMResponse", "ScriptedCall",
    ),
    "liteagent.llm.base": ("LLMClient", "BaseLLMClient", "LLMStreamChunk"),
    "liteagent.llm.message": ("Message", "Role"),
    "liteagent.llm.scripted": ("ScriptedLLM", "ScriptedResponse"),
    "liteagent.llm.registry": ("LLMRegistry", "build_llm", "get_llm"),
    "liteagent.tools.base": (
        "tool", "Tool", "ToolSpec", "make_function_tool", "is_tool",
        "current_cancel_flag", "cancel_scope",
    ),
    "liteagent.tools.registry": (
        "ToolRegistry", "get_default_registry", "reset_default_registry",
    ),
    "liteagent.tools.executor": ("ExecutorConfig", "ToolExecutor"),
    "liteagent.tools.builtin": ("register_all", "BUILTIN_TOOL_NAMES", "BUILTIN_TOOL_GROUPS"),
    "liteagent.memory.base": (
        "MemoryItem", "MemoryStore", "Tokenizer", "get_default_tokenizer",
    ),
    "liteagent.memory.embeddings": ("Embedder", "HashingEmbedder"),
    "liteagent.memory.buffer": ("BufferMemory", "BufferConfig"),
    "liteagent.memory.vector": ("VectorMemory", "VectorConfig"),
    "liteagent.memory.summary": ("SummaryMemory", "SummaryConfig"),
    "liteagent.memory.manager": ("MemoryManager",),
    "liteagent.multiagent.base": (
        "MultiAgent", "DelegationContext", "build_team", "compress_subagent_output",
    ),
    "liteagent.multiagent.sequential": ("SequentialAgent", "SequentialStep"),
    "liteagent.multiagent.hierarchical": ("HierarchicalAgent", "Plan", "SubTask"),
    "liteagent.multiagent.blackboard": ("Blackboard", "BlackboardEntry"),
    "liteagent.agent.state": ("AgentState", "AgentStatus", "AgentResult"),
    "liteagent.agent.agent": ("Agent",),
    "liteagent.agent.callbacks": (
        "EventType", "TraceEvent", "CallbackManager", "TraceRecorder",
        "LoggingCallback", "JsonlTraceCallback", "RichCallback", "TokenCounterCallback",
        "MemoryTraceCallback", "load_trace", "render_trace", "trace_stats",
    ),
}

_LAZY_SOURCES: dict[str, str] = {
    name: module for module, names in _EXPORT_SOURCES.items() for name in names
}

# 漂移检查：`__all__` 与 `_EXPORT_SOURCES` 是两张手写清单，分叉时**立刻**炸在 import 上，
# 而不是等到某个 `getattr(liteagent, "X")` 在测试里抛一个语焉不详的 AttributeError。
_UNRESOLVED = [name for name in __all__ if name not in _LAZY_SOURCES]
if _UNRESOLVED:  # pragma: no cover - 只有改了一张清单忘改另一张才会命中
    raise RuntimeError(
        f"liteagent.__all__ lists names with no source module: {', '.join(_UNRESOLVED)}"
    )
del _UNRESOLVED


def __getattr__(name: str) -> Any:
    """按 `_LAZY_SOURCES` 就地解析公开名字（PEP 562）。

    解析失败时 `ImportError` 原样冒泡：取不到某个名字一定是因为它所属的模块有问题，
    此时静默返回 None 会把"import 期就能报的错"推迟成"运行到一半才炸的悬案"。
    """
    module_name = _LAZY_SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise ImportError(
            f"cannot import name {name!r} from {__name__!r}: "
            f"{module_name!r} does not define it"
        ) from exc
    globals()[name] = value  # 缓存：后续访问退化成普通字典命中
    return value


def __dir__() -> list[str]:
    """让 `dir(liteagent)` 与自动补全看得到 `__all__` 里的名字（否则现实与 `__all__` 打架）。"""
    return sorted(set(__all__) | set(globals()))
