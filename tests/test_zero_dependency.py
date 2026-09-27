from __future__ import annotations

# 守门测试（gate test）：零依赖红线、公共 API 冻结面、文件封闭清单、测试总量。
#
# 真值源：``docs/INTERFACES.md``（FROZEN v2.0）
# - §1.1 依赖 DAG 与同层边白名单（L0..L6 + E1..E12）
# - §1.2 41 个 ``.py`` 的封闭清单、``__main__.py`` 的逐字 6 行、``pyproject.toml`` 冻结值
# - §1.3 零依赖红线的**精确定义**与 AST 判定方式（禁止正则/文本子串）
# - §1.4 各包 ``__all__``；§10.5 ``multiagent/__init__.py`` 导出
# - §0.1 Python 3.10 缺失 API 禁用清单
# - 附录 B ``liteagent/__init__.py`` 的 ``__all__``
# - §12 测试文件清单与总量目标（>= 220，其中 schema/executor 各 >= 30）
#
# 本文件是**最终验收闸门**：它只做静态判定（``ast`` / 文件系统 / ``getattr``），
# 不依赖任何其它测试文件的**夹具**、不联网、不睡时钟。
# 每一条断言都直接对应上面某个冻结条款，失败信息里带上文件与行号，便于定位。
#
# `[v3 修订]` 上面"只做静态"这条有两处**已声明的**例外，都收在文件末尾的
# `SuiteHygieneTests` 里（详见那个类的 docstring）：
#   1. 它 import ``tests.helpers`` 的 ``liteagent_worker_threads()``（线程谓词的唯一实现点，
#      复制一份到本文件才是真正的坏味道）；
#   2. 它用 ``threading.enumerate()`` + 有上限的轮询等 worker 退场（线程存活无法用 ``ast``
#      判定）。这两件事被限制在那一个类里，其余类仍然是纯静态的。
#
# 几个刻意的写法说明（都不是"绕过断言"，而是规范的字面要求）：
#
# * **一切形态判定都用 ``ast``，不用正则**（§1.3 冻结："3.11 API 用 AST 判定，禁止文本子串匹配"）。
#   下面 ``Python311ApiDetectionTests`` 里有一组自证用例：把禁用 API 的名字写进 docstring /
#   注释**不得**触发失败，写成真实代码**必须**触发失败。唯一的正则是解析 ``pyproject.toml``
#   的版本号（3.10 没有 ``tomllib``，而把 PyYAML 拉进守门测试是不必要的耦合）。
# * **顶层 import 的精确定义**（§1.3）：位于 ``ast.Try``（任意深度）或
#   ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``/``If`` 体内的 import **不算**顶层。
#   ``With``/``For``/``While`` 体内的 import 按字面仍算顶层（规范只放行了上面 5 种容器）。
# * ``AgentResult(...)`` 的关键字形式判定也只认 AST（文本匹配会命中 docstring 里的
#   ``AgentResult(FAILED, error=e)`` 这种说明文字，实现文件里真的存在这句话）。

import ast
import pathlib
import re
import sys
import time
import unittest

from tests.helpers import liteagent_worker_threads

# ---------------------------------------------------------------------------
# 基准路径
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LITEAGENT_DIR = REPO_ROOT / "liteagent"
TESTS_DIR = REPO_ROOT / "tests"

# ---------------------------------------------------------------------------
# §1.2 封闭清单：41 个 `.py`（逐条转抄，顺序与表格一致）
# ---------------------------------------------------------------------------

FROZEN_LITEAGENT_FILES = (
    "liteagent/__init__.py",
    "liteagent/errors.py",
    "liteagent/types.py",
    "liteagent/config.py",
    "liteagent/llm/__init__.py",
    "liteagent/llm/message.py",
    "liteagent/llm/base.py",
    "liteagent/llm/transport.py",
    "liteagent/llm/providers.py",
    "liteagent/llm/registry.py",
    "liteagent/llm/scripted.py",
    "liteagent/tools/__init__.py",
    "liteagent/tools/schema.py",
    "liteagent/tools/base.py",
    "liteagent/tools/registry.py",
    "liteagent/tools/executor.py",
    "liteagent/tools/builtin/__init__.py",
    "liteagent/tools/builtin/files.py",
    "liteagent/tools/builtin/shell.py",
    "liteagent/tools/builtin/code.py",
    "liteagent/tools/builtin/web.py",
    "liteagent/tools/builtin/memory_tools.py",
    "liteagent/memory/__init__.py",
    "liteagent/memory/base.py",
    "liteagent/memory/embeddings.py",
    "liteagent/memory/buffer.py",
    "liteagent/memory/summary.py",
    "liteagent/memory/vector.py",
    "liteagent/memory/manager.py",
    "liteagent/agent/__init__.py",
    "liteagent/agent/state.py",
    "liteagent/agent/callbacks.py",
    "liteagent/agent/parser.py",
    "liteagent/agent/agent.py",
    "liteagent/multiagent/__init__.py",
    "liteagent/multiagent/base.py",
    "liteagent/multiagent/blackboard.py",
    "liteagent/multiagent/sequential.py",
    "liteagent/multiagent/hierarchical.py",
    "liteagent/cli.py",
    "liteagent/__main__.py",
)

# §1.2 冻结的 `liteagent/__main__.py` 逐字内容（6 行 + 行尾换行）
FROZEN_MAIN_PY = (
    "from __future__ import annotations\n"
    "\n"
    "from liteagent.cli import main\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    raise SystemExit(main())\n"
)

# §12 冻结的测试文件清单（39 项，逐条转抄）
FROZEN_TEST_FILES = (
    "tests/__init__.py",
    "tests/helpers.py",
    "tests/test_types.py",
    "tests/test_errors.py",
    "tests/test_config.py",
    "tests/test_message.py",
    "tests/test_transport.py",
    "tests/test_llm_providers.py",
    "tests/test_llm_retry.py",
    "tests/test_llm_registry.py",
    "tests/test_llm_streaming.py",
    "tests/test_scripted.py",
    "tests/test_tools_schema.py",
    "tests/test_tools_registry.py",
    "tests/test_tools_executor.py",
    "tests/test_memory_base.py",
    "tests/test_memory_embeddings.py",
    "tests/test_memory_buffer.py",
    "tests/test_memory_summary.py",
    "tests/test_memory_vector.py",
    "tests/test_memory_persistence.py",
    "tests/test_memory_manager.py",
    "tests/test_agent_react_native.py",
    "tests/test_agent_react_text.py",
    "tests/test_agent_features.py",
    "tests/test_callbacks.py",
    "tests/test_blackboard.py",
    "tests/test_multiagent_sequential.py",
    "tests/test_multiagent_hierarchical.py",
    "tests/test_builtin_files.py",
    "tests/test_builtin_shell.py",
    "tests/test_builtin_code.py",
    "tests/test_builtin_web.py",
    "tests/test_cli.py",
    "tests/test_e2e_code_assistant.py",
    "tests/test_examples_offline.py",
    "tests/test_examples_import.py",
    "tests/test_zero_dependency.py",
    "tests/test_docs_coverage.py",
)

# §12 测试总量目标
MIN_TOTAL_TEST_METHODS = 220
MIN_HEAVY_FILE_TEST_METHODS = 30
HEAVY_TEST_FILES = ("tests/test_tools_schema.py", "tests/test_tools_executor.py")
MIN_TEST_METHODS_PER_FILE = 4

# ---------------------------------------------------------------------------
# 附录 B：`liteagent/__init__.py` 的 `__all__`（120 个名字，逐条转抄）
# ---------------------------------------------------------------------------

APPENDIX_B_NAMES = (
    # core
    "Agent", "AgentConfig", "AgentResult", "AgentState",
    "AgentStatus",
    # llm
    "LLMClient", "BaseLLMClient", "LLMConfig", "LLMResponse",
    "LLMStreamChunk", "Message", "Role", "TokenUsage",
    "ToolCall", "ToolResult", "ScriptedLLM", "ScriptedResponse",
    "ScriptedCall", "LLMRegistry", "build_llm", "get_llm",
    # tools
    "tool", "Tool", "ToolSpec", "ToolRegistry",
    "ExecutorConfig", "ToolExecutor", "make_function_tool", "is_tool",
    "current_cancel_flag", "cancel_scope", "register_all", "BUILTIN_TOOL_NAMES",
    "BUILTIN_TOOL_GROUPS", "get_default_registry", "reset_default_registry",
    # memory
    "MemoryManager", "MemoryConfig", "MemoryItem", "MemoryStore",
    "MemoryStoreError", "BufferMemory", "BufferConfig", "VectorMemory",
    "VectorConfig", "SummaryMemory", "SummaryConfig", "HashingEmbedder",
    "Embedder", "Tokenizer", "get_default_tokenizer",
    # multiagent
    "SequentialAgent", "SequentialStep", "HierarchicalAgent", "Plan",
    "SubTask", "Blackboard", "BlackboardEntry", "TeamConfig",
    "MultiAgent", "DelegationContext", "build_team", "compress_subagent_output",
    # callbacks
    "EventType", "TraceEvent", "CallbackManager", "TraceRecorder",
    "LoggingCallback", "JsonlTraceCallback", "RichCallback", "TokenCounterCallback",
    "MemoryTraceCallback", "load_trace", "render_trace", "trace_stats",
    # config
    "AppConfig", "RetryPolicy", "LoopBoundPool", "run_sync",
    "utc_now", "format_ts", "to_jsonable", "parse_dotenv",
    "load_dotenv", "parse_bool", "render_template", "estimate_cost_usd",
    "MODEL_PRICES", "NO_TIMEOUT",
    # errors
    "LiteAgentError", "ConfigError", "ToolError", "ToolNotFoundError",
    "ToolValidationError", "ToolExecutionError", "ToolTimeoutError", "ToolSkippedError",
    "ToolRetryExhaustedError", "ToolApprovalDeniedError", "ToolDefinitionError",
    "LLMError", "LLMRateLimitError", "LLMTimeoutError", "LLMConnectionError",
    "LLMAuthError", "AgentError", "MaxStepsExceededError", "RepeatedActionError",
    "ReActParseError", "MultiAgentError", "DelegationError", "MaxDepthExceededError",
    "CycleDetectedError", "VersionConflictError", "SandboxViolationError",
    "ScriptedExhaustedError", "SerializationError", "BudgetExceededError",
    "RunTimeoutError", "AgentAbortedError",
)

# §1.4 / §10.5：各包 `__all__` 冻结清单（逐条转抄）
FROZEN_SUBPACKAGE_ALL = {
    "liteagent.llm": (
        "LLMClient", "BaseLLMClient", "LLMStreamChunk", "Message", "Role", "LLMResponse",
        "LLMConfig", "ScriptedLLM", "ScriptedResponse", "ScriptedCall", "LLMRegistry",
        "build_llm", "get_llm", "messages_tokens", "render_transcript",
        "drop_orphan_tool_messages",
    ),
    "liteagent.tools": (
        "tool", "Tool", "ToolSpec", "ToolRegistry", "ExecutorConfig", "ToolExecutor",
        "make_function_tool", "is_tool", "current_cancel_flag", "cancel_scope",
        "get_default_registry", "reset_default_registry",
        "register_all", "BUILTIN_TOOL_NAMES", "BUILTIN_TOOL_GROUPS",
    ),
    "liteagent.memory": (
        "MemoryItem", "MemoryStore", "MemoryConfig", "Tokenizer", "HeuristicTokenizer",
        "CallableTokenizer", "get_default_tokenizer",
        "Embedder", "HashingEmbedder", "NumpyHashingEmbedder", "RandomProjectionEmbedder",
        "CallableEmbedder", "RemoteEmbedder", "default_embedder", "cosine_similarity",
        "cosine_similarity_matrix",
        "BufferMemory", "BufferConfig", "SummaryMemory", "SummaryConfig",
        "VectorMemory", "VectorConfig", "MemoryManager",
    ),
    "liteagent.agent": (
        "Agent", "AgentConfig", "AgentResult", "AgentState", "AgentStatus",
        "EventType", "TraceEvent", "Callback", "CallbackManager", "CallbackLike",
        "FunctionCallback", "LoggingCallback", "JsonlTraceCallback", "RichCallback",
        "TokenCounterCallback", "MemoryTraceCallback", "TraceRecorder",
        "load_trace", "total_usage", "events_of_type", "render_trace", "as_llm_callback",
        "trace_stats",
        "ReActParser", "ParsedAction",
    ),
    "liteagent.multiagent": (
        "AgentLike", "MultiAgent", "TeamConfig", "DelegationContext",
        "Blackboard", "BlackboardEntry", "SequentialAgent", "SequentialStep",
        "HierarchicalAgent", "Plan", "SubTask", "build_team", "compress_subagent_output",
    ),
}

# ---------------------------------------------------------------------------
# §1.1 依赖 DAG：L 编号表
# ---------------------------------------------------------------------------

ROOT_PACKAGE = "liteagent"

#: 包的 `__init__.py` 与其子模块同属一个包；根目录下的模块（errors/types/config/cli）
#: 归属 "liteagent"。这个粒度是 §1.1 的写法决定的：§1.1 把 `tools/*` 与 `tools/builtin/*`
#: 放在同一条 L3 里、把 `X/__init__.py` 明确写成"允许 import 本包任意模块"。
MODULE_LEVELS = {
    # L0
    "liteagent.errors": 0,
    # L1
    "liteagent.types": 1,
    "liteagent.config": 1,
    # L2
    "liteagent.llm": 2,
    "liteagent.llm.message": 2,
    "liteagent.llm.transport": 2,
    "liteagent.llm.base": 2,
    "liteagent.llm.providers": 2,
    "liteagent.llm.registry": 2,
    "liteagent.llm.scripted": 2,
    # L3
    "liteagent.memory": 3,
    "liteagent.memory.base": 3,
    "liteagent.memory.embeddings": 3,
    "liteagent.memory.buffer": 3,
    "liteagent.memory.summary": 3,
    "liteagent.memory.vector": 3,
    "liteagent.memory.manager": 3,
    "liteagent.tools": 3,
    "liteagent.tools.schema": 3,
    "liteagent.tools.base": 3,
    "liteagent.tools.registry": 3,
    "liteagent.tools.executor": 3,
    "liteagent.tools.builtin": 3,
    "liteagent.tools.builtin.files": 3,
    "liteagent.tools.builtin.shell": 3,
    "liteagent.tools.builtin.code": 3,
    "liteagent.tools.builtin.web": 3,
    "liteagent.tools.builtin.memory_tools": 3,
    # L4
    "liteagent.agent": 4,
    "liteagent.agent.state": 4,
    "liteagent.agent.callbacks": 4,
    "liteagent.agent.parser": 4,
    "liteagent.agent.agent": 4,
    # L5
    "liteagent.multiagent": 5,
    "liteagent.multiagent.base": 5,
    "liteagent.multiagent.blackboard": 5,
    "liteagent.multiagent.sequential": 5,
    "liteagent.multiagent.hierarchical": 5,
    # L6
    "liteagent.cli": 6,
    # `__main__.py` 不在 L 表里，但它是唯一的入口文件，等价于 L6 之上
    "liteagent.__main__": 7,
    # 根包 `liteagent/__init__.py`：§1.1 允许它 import 任意 `liteagent.*`，因此对所有
    # 子模块而言它总是"向下"的（无反向依赖风险）。
    "liteagent": -1,
}

#: §1.1「同层边白名单（冻结清单，逐条写进 test_zero_dependency.py）」E1..E12。
#: key 是规范里的编号，value 是该条允许的 (source_module, target_module) 边集合。
ALLOWED_SAME_LEVEL_EDGES = {
    # E1 types.py -> llm/message.py（仅允许函数体内延迟 import；顶层 import 会被本表放行，
    #    但 §1.1 同时说明"test_zero_dependency 只看顶层 import"）
    "E1": frozenset({("liteagent.types", "liteagent.llm.message")}),
    # E2 llm/registry.py -> llm/providers.py（函数体内延迟）
    "E2": frozenset({("liteagent.llm.registry", "liteagent.llm.providers")}),
    # E3 multiagent/{sequential,hierarchical}.py -> multiagent/base.py
    "E3": frozenset({
        ("liteagent.multiagent.sequential", "liteagent.multiagent.base"),
        ("liteagent.multiagent.hierarchical", "liteagent.multiagent.base"),
    }),
    # E4 tools/base.py <-> tools/schema.py
    "E4": frozenset({
        ("liteagent.tools.base", "liteagent.tools.schema"),
        ("liteagent.tools.schema", "liteagent.tools.base"),
    }),
    # E5 tools/executor.py -> tools/{base,registry,schema}
    "E5": frozenset({
        ("liteagent.tools.executor", "liteagent.tools.base"),
        ("liteagent.tools.executor", "liteagent.tools.registry"),
        ("liteagent.tools.executor", "liteagent.tools.schema"),
    }),
    # E6 memory/manager.py -> memory/{base,buffer,vector,summary}
    "E6": frozenset({
        ("liteagent.memory.manager", "liteagent.memory.base"),
        ("liteagent.memory.manager", "liteagent.memory.buffer"),
        ("liteagent.memory.manager", "liteagent.memory.vector"),
        ("liteagent.memory.manager", "liteagent.memory.summary"),
    }),
    # E7 memory/{buffer,summary,vector}.py -> memory/{base,embeddings}
    "E7": frozenset(
        (source, target)
        for source in ("liteagent.memory.buffer", "liteagent.memory.summary", "liteagent.memory.vector")
        for target in ("liteagent.memory.base", "liteagent.memory.embeddings")
    ),
    # E8 tools/builtin/{shell,code}.py -> tools/builtin/files.py
    "E8": frozenset({
        ("liteagent.tools.builtin.shell", "liteagent.tools.builtin.files"),
        ("liteagent.tools.builtin.code", "liteagent.tools.builtin.files"),
    }),
    # E9 tools/builtin/__init__.py -> tools/builtin/*（§7.5 各 factory 的函数对象）
    "E9": frozenset({
        ("liteagent.tools.builtin", "liteagent.tools.builtin.files"),
        ("liteagent.tools.builtin", "liteagent.tools.builtin.shell"),
        ("liteagent.tools.builtin", "liteagent.tools.builtin.code"),
        ("liteagent.tools.builtin", "liteagent.tools.builtin.web"),
        ("liteagent.tools.builtin", "liteagent.tools.builtin.memory_tools"),
    }),
    # E10 tools/builtin/memory_tools.py -> memory/manager.py
    "E10": frozenset({("liteagent.tools.builtin.memory_tools", "liteagent.memory.manager")}),
    # E11 multiagent/base.py -> multiagent/blackboard.py（blackboard 真实层级是 L1）
    "E11": frozenset({("liteagent.multiagent.base", "liteagent.multiagent.blackboard")}),
    # E12 memory/manager.py -> memory/embeddings.py（from_config 造默认 embedder）
    "E12": frozenset({("liteagent.memory.manager", "liteagent.memory.embeddings")}),
}

WHITELISTED_EDGES = frozenset(
    edge for edges in ALLOWED_SAME_LEVEL_EDGES.values() for edge in edges
)

# ---------------------------------------------------------------------------
# §1.3 零依赖红线：三方库允许出现的位置
# ---------------------------------------------------------------------------

#: §1.3 逐字：「三方库只能出现在 llm/transport.py（requests/httpx）、
#: memory/embeddings.py（numpy）、config.py（yaml）、cli.py（rich，可选）」
#: 括号里的库名也一起冻结：允许的文件与允许的库必须成对出现，否则"把 yaml 写进
#: transport.py"这种漂移会被漏掉。
FROZEN_THIRD_PARTY_FILES = {
    "liteagent/llm/transport.py": frozenset({"requests", "httpx"}),
    "liteagent/memory/embeddings.py": frozenset({"numpy"}),
    "liteagent/config.py": frozenset({"yaml"}),
    "liteagent/cli.py": frozenset({"rich"}),
}

#: 规范内部矛盾（已在本文件与实现文件里记录 SPEC-AMBIGUITY）：
#: §1.3 的四文件清单 vs §8.1「TiktokenTokenizer 仅当 tiktoken 可 import 时定义」与
#: §9.2「RichCallback 有 rich 时彩色输出、无 rich 时退化」。裁决：服从 §8.1/§9.2
#: （它们逐字要求这两个模块提供可选依赖能力），但把豁免**收紧到具体模块 + 具体包名**，
#: 任何新的第三方 import 都无法借这条豁免混进来。
DOC_MANDATED_THIRD_PARTY_FILES = {
    "liteagent/memory/base.py": frozenset({"tiktoken"}),   # §8.1 TiktokenTokenizer
    "liteagent/agent/callbacks.py": frozenset({"rich"}),   # §9.2 RichCallback
}

#: §1.3 要求提供的可用性常量（名字冻结）
#: value 为 None 表示"位置未在 §1.3 冻结"，只断言存在；否则断言文件与位置。
FROZEN_AVAILABILITY_CONSTANTS = {
    "NUMPY_AVAILABLE": "liteagent/memory/embeddings.py",
    "REQUESTS_AVAILABLE": "liteagent/llm/transport.py",
    "HTTPX_AVAILABLE": "liteagent/llm/transport.py",
    "YAML_AVAILABLE": "liteagent/config.py",
    "RICH_AVAILABLE": "liteagent/cli.py",
    "TIKTOKEN_AVAILABLE": None,
    "PYDANTIC_AVAILABLE": None,
}

# ---------------------------------------------------------------------------
# §0.1 / §1.3 3.11+ API 禁用清单（AST 判定）
# ---------------------------------------------------------------------------

#: §1.3 冻结的 AST 判定规则（逐条转抄）：
#:   (ast.Attribute, attr in {"TaskGroup","Runner","timeout"})
#: | (ast.Name, id in {"TaskGroup","Runner"})
#: | (ast.ImportFrom, module in {"tomllib"} or name in {"StrEnum","Self","UTC","ExceptionGroup"})
#: | (ast.Attribute, attr in {"Self","UTC","StrEnum","ExceptionGroup"})
BANNED_ATTRIBUTE_NAMES = frozenset({
    "TaskGroup", "Runner", "timeout", "Self", "UTC", "StrEnum", "ExceptionGroup",
})
BANNED_NAME_IDS = frozenset({"TaskGroup", "Runner"})
BANNED_FROM_MODULES = frozenset({"tomllib"})
BANNED_FROM_NAMES = frozenset({"StrEnum", "Self", "UTC", "ExceptionGroup"})

#: 顶层 import 精确定义里"放行"的容器节点（§1.3）
NON_TOPLEVEL_CONTAINERS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
)


# ---------------------------------------------------------------------------
# AST 工具函数（全部纯函数，无副作用，便于自证用例直接调用）
# ---------------------------------------------------------------------------


def _module_name(path: pathlib.Path) -> str:
    """`liteagent/memory/base.py` -> `liteagent.memory.base`；`__init__.py` -> 包名。"""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(module: str) -> str:
    """模块的"包"身份：`liteagent.tools.builtin.files` 与 `liteagent.tools.base` 同包。"""
    parts = module.split(".")
    if len(parts) == 1:
        return ROOT_PACKAGE
    return "liteagent." + parts[1]


def _liteagent_module_paths() -> list[pathlib.Path]:
    return sorted(
        path for path in LITEAGENT_DIR.rglob("*.py") if "__pycache__" not in path.parts
    )


def _module_index() -> dict[str, pathlib.Path]:
    return {_module_name(path): path for path in _liteagent_module_paths()}


def top_level_import_nodes(tree: ast.Module) -> list[ast.stmt]:
    """返回模块的"顶层 import"（§1.3 的精确定义）。

    一个 ``ast.Import``/``ast.ImportFrom`` 语句算顶层，**除非**它（含任意深度）位于
    ``ast.Try`` 节点内，或位于 ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``/``If``
    体内。``Try`` 被放行是因为 ``try: import numpy / except ImportError:`` 在 AST 语义上
    等价于"可选依赖"；``If`` 被放行是因为 ``if TYPE_CHECKING:`` 与
    ``if REQUESTS_AVAILABLE:`` 两种惯用写法。
    """
    found: list[ast.stmt] = []

    def walk(body: list[ast.stmt], inside: bool) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not inside:
                    found.append(node)
                continue
            if isinstance(node, ast.Try):
                # 「含任意深度」：Try 分支里的所有后代 import 都不再是顶层
                for statement in list(node.body) + list(node.orelse) + list(node.finalbody):
                    walk([statement], True)
                for handler in node.handlers:
                    walk(handler.body, True)
                continue
            child_inside = True if isinstance(node, NON_TOPLEVEL_CONTAINERS) else inside
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    walk(block, child_inside)
            for handler in getattr(node, "handlers", None) or ():
                walk(handler.body, child_inside)

    walk(list(tree.body), False)
    return found


def third_party_import_roots(node: ast.stmt) -> list[str]:
    """一条 import 语句里的三方库顶层包名（stdlib / liteagent / 相对 import 不算）。"""
    stdlib = set(sys.stdlib_module_names)
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:  # 相对 import 永远不是三方库
            return []
        names = [node.module or ""]
    roots = []
    for name in names:
        root = name.split(".")[0]
        if not root or root in stdlib or root in ("liteagent", "__future__"):
            continue
        roots.append(root)
    return roots


def python311_api_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """§1.3 冻结的 3.11 API AST 判定。返回 [(lineno, 命中的形态描述)]。

    **只认节点形态**：docstring 与注释里的 API 名字不会产生任何 AST 节点，
    因此天然不触发失败（这正是"禁止文本子串匹配"的原因）。
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTE_NAMES:
            hits.append((node.lineno, "Attribute." + node.attr))
        elif isinstance(node, ast.Name) and node.id in BANNED_NAME_IDS:
            hits.append((node.lineno, "Name." + node.id))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") in BANNED_FROM_MODULES:
                hits.append((node.lineno, "ImportFrom." + str(node.module)))
            for alias in node.names:
                if alias.name in BANNED_FROM_NAMES:
                    hits.append((node.lineno, "ImportFrom.name." + alias.name))
    return hits


def resolve_import_targets(current_module: str, node: ast.stmt, known: set[str]) -> list[str]:
    """把一条顶层 import 解析成它实际引入的 liteagent 模块名列表。

    ``from liteagent.memory import base`` 优先解析为子模块 ``liteagent.memory.base``
    （最长匹配），否则退化为 ``liteagent.memory``。相对 import 按 `current_module` 归一。
    """
    if isinstance(node, ast.Import):
        return [
            alias.name
            for alias in node.names
            if alias.name.startswith("liteagent") and alias.name in known
        ]
    if node.level:
        base = current_module
        if base in known:
            base = base.rsplit(".", 1)[0]
        for _ in range(node.level - 1):
            base = base.rsplit(".", 1)[0] if "." in base else base
        module = (base + "." + (node.module or "")).rstrip(".")
    else:
        module = node.module or ""
        if not module.startswith("liteagent"):
            return []
    submodules = [
        module + "." + alias.name for alias in node.names if module + "." + alias.name in known
    ]
    if submodules:
        return submodules
    return [module] if module in known else []


def agent_result_keyword_violations(root: ast.AST) -> list[tuple[int, str]]:
    """找出所有非关键字形式（位置参数 / ``*args`` 展开）的 ``AgentResult(...)`` 构造点。"""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name != "AgentResult":
            continue
        for argument in node.args:
            kind = "Starred" if isinstance(argument, ast.Starred) else "positional"
            violations.append((node.lineno, kind))
    return violations


def module_level_assignment_targets(tree: ast.Module) -> list[tuple[str, int]]:
    """模块顶层（import 期执行）的赋值目标名 + 行号。

    §1.3 要求可用性常量"必须在模块顶层定义"，而冻结写法是
    ``try: import numpy; NUMPY_AVAILABLE = True / except ImportError: NUMPY_AVAILABLE = False``
    —— 赋值位于 ``ast.Try`` 体内但仍是模块级语句（import 时执行），所以这里要穿过
    ``Try``/``If``/``With``/``For``/``While``，但不进 ``FunctionDef``/``ClassDef``。
    """
    found: list[tuple[str, int]] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.append((target.id, node.lineno))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    found.append((node.target.id, node.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    walk(block)
            for handler in getattr(node, "handlers", None) or ():
                walk(handler.body)

    walk(list(tree.body))
    return found


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*.py") if "__pycache__" not in path.parts
    )


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


# ---------------------------------------------------------------------------
# §1.2 文件封闭清单 / 入口文件 / 冻结的 pyproject 值
# ---------------------------------------------------------------------------


class FrozenLayoutTests(unittest.TestCase):
    """§1.2：`liteagent/` 下的 `.py` 集合、`__main__.py` 逐字内容、根交付物冻结值。"""

    def test_liteagent_file_set_matches_spec_exactly(self) -> None:
        actual = sorted(_relative(path) for path in _liteagent_module_paths())
        expected = sorted(FROZEN_LITEAGENT_FILES)
        self.assertEqual(
            [],
            sorted(set(expected) - set(actual)),
            "§1.2 冻结的文件缺失",
        )
        self.assertEqual(
            [],
            sorted(set(actual) - set(expected)),
            "§1.2 封闭清单之外新增了 liteagent/*.py（新增文件会违反红线 1）",
        )
        self.assertEqual(len(expected), len(actual))

    def test_liteagent_file_count_is_41(self) -> None:
        self.assertEqual(41, len(FROZEN_LITEAGENT_FILES))
        self.assertEqual(41, len(_liteagent_module_paths()))

    def test_main_module_is_verbatim_six_lines(self) -> None:
        actual = (LITEAGENT_DIR / "__main__.py").read_text(encoding="utf-8")
        self.assertEqual(FROZEN_MAIN_PY, actual, "§1.2 冻结的 __main__.py 必须逐字一致")
        self.assertEqual(6, len(actual.rstrip("\n").splitlines()))

    def test_pyproject_frozen_values(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version\s*=\s*"([^"]*)"', text, re.MULTILINE)
        self.assertIsNotNone(version, "pyproject.toml 缺少 version")
        self.assertEqual("0.1.0", version.group(1))
        requires_python = re.search(r'^requires-python\s*=\s*"([^"]*)"', text, re.MULTILINE)
        self.assertIsNotNone(requires_python, "pyproject.toml 缺少 requires-python")
        self.assertEqual(">=3.10", requires_python.group(1))
        self.assertRegex(
            text,
            r'(?m)^dependencies\s*=\s*\[\s*\]\s*$',
            "核心依赖必须为空（§13 红线 2：不得新增第三方依赖）",
        )

    def test_all_frozen_test_files_exist(self) -> None:
        missing = [
            relative
            for relative in FROZEN_TEST_FILES
            if not (REPO_ROOT / relative).is_file()
        ]
        self.assertEqual([], missing, "§12 冻结的测试文件尚未落地")

    def test_tests_init_is_empty(self) -> None:
        path = TESTS_DIR / "__init__.py"
        self.assertTrue(path.is_file(), "tests/__init__.py 必须存在（unittest discover 需要）")
        self.assertEqual("", path.read_text(encoding="utf-8").strip(), "§12：这是空文件")


# ---------------------------------------------------------------------------
# §1.3 零依赖红线
# ---------------------------------------------------------------------------


class ZeroDependencyTests(unittest.TestCase):
    """§1.3：顶层 import 只能是 stdlib 或 `liteagent.*`；三方 import 只允许可选依赖形态。"""

    def test_no_bare_third_party_top_level_imports(self) -> None:
        offenders = []
        for path in _liteagent_module_paths():
            tree = _parse(path)
            for node in top_level_import_nodes(tree):
                for root in third_party_import_roots(node):
                    offenders.append(f"{_relative(path)}:{node.lineno} import {root}")
        self.assertEqual(
            [],
            offenders,
            "顶层三方 import（§1.3）；应写成 try/except ImportError 的可选依赖形态",
        )

    def test_third_party_imports_are_inside_try_blocks(self) -> None:
        offenders = []
        for path in _liteagent_module_paths():
            tree = _parse(path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                roots = third_party_import_roots(node)
                if not roots:
                    continue
                if not _has_try_ancestor(tree, node):
                    offenders.append(f"{_relative(path)}:{node.lineno} import {roots}")
        self.assertEqual(
            [],
            offenders,
            "三方 import 必须写在 try/except ImportError 里（§1.3 冻结写法）",
        )

    def test_third_party_imports_confined_to_sanctioned_files(self) -> None:
        # 文件 -> 该文件允许出现的三方库顶层包名（版本冲突/漂移都能被抓到）
        sanctioned = dict(FROZEN_THIRD_PARTY_FILES)
        sanctioned.update(DOC_MANDATED_THIRD_PARTY_FILES)
        offenders = []
        for path in _liteagent_module_paths():
            tree = _parse(path)
            roots: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots.extend(third_party_import_roots(node))
            if not roots:
                continue
            relative = _relative(path)
            if relative not in sanctioned:
                offenders.append(f"{relative}: {sorted(set(roots))}（§1.3 未允许该文件）")
                continue
            unexpected = sorted(set(roots) - set(sanctioned[relative]))
            if unexpected:
                offenders.append(
                    f"{relative}: 不该出现的三方库 {unexpected}"
                    f"（该文件只允许 {sorted(sanctioned[relative])}）"
                )
        self.assertEqual([], offenders, "§1.3 三方库只能出现在冻结的四文件（+ 已记录的豁免）")

    def test_availability_constants_are_module_level_bools(self) -> None:
        index = _module_index()
        defined: dict[str, list[str]] = {name: [] for name in FROZEN_AVAILABILITY_CONSTANTS}
        for module, path in index.items():
            for target, _lineno in module_level_assignment_targets(_parse(path)):
                if target in defined and module not in defined[target]:
                    defined[target].append(module)
        missing = sorted(name for name, homes in defined.items() if not homes)
        self.assertEqual([], missing, "§1.3 的可用性常量必须在某个模块顶层定义")
        for name, home in FROZEN_AVAILABILITY_CONSTANTS.items():
            if home is None:
                continue
            expected_module = "liteagent." + home[len("liteagent/") : -len(".py")].replace(
                "/", "."
            )
            self.assertIn(
                expected_module,
                defined[name],
                f"§1.3：{name} 应定义在 {home}（实际出现在 {defined[name]}）",
            )
        # 运行期取值必须是 bool
        for name in FROZEN_AVAILABILITY_CONSTANTS:
            module_name = defined[name][0]
            module = __import__(module_name, fromlist=["__name__"])
            self.assertIsInstance(
                getattr(module, name), bool, f"{module_name}.{name} 必须是 bool"
            )

    def test_toplevel_import_classifier_matches_spec_definition(self) -> None:
        """自证用例：确认顶层判定与 §1.3 的字面定义一致（防止判定过松/过严）。"""
        cases = {
            # 源码 -> 期望的顶层 import 条数
            "import numpy\n": 1,
            "from pydantic import BaseModel\n": 1,
            "try:\n    import numpy\nexcept ImportError:\n    numpy = None\n": 0,
            "def f():\n    import numpy\n": 0,
            "async def f():\n    import numpy\n": 0,
            "class C:\n    import numpy\n": 0,
            "if TYPE_CHECKING:\n    from pydantic import BaseModel\n": 0,
            "try:\n    import numpy\nexcept ImportError:\n    try:\n        import numpy\n    except ImportError:\n        numpy = None\n": 0,
            "try:\n    import numpy\nexcept ImportError:\n    pass\nimport yaml\n": 1,
            # 规范只放行 Try/Func/AsyncFunc/Class/If：With/For 体内的 import 仍算顶层
            "for _ in range(1):\n    import numpy\n": 1,
            "with open('x'):\n    import numpy\n": 1,
        }
        for source, expected in cases.items():
            tree = ast.parse(source)
            self.assertEqual(
                expected,
                len(top_level_import_nodes(tree)),
                f"顶层判定与 §1.3 定义不符：{source!r}",
            )

    def test_third_party_root_detection_ignores_stdlib_and_liteagent(self) -> None:
        tree = ast.parse(
            "import os, json\n"
            "import liteagent\n"
            "from liteagent.config import utc_now\n"
            "from . import sibling\n"
            "import numpy\n"
        )
        detected = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                detected.extend(third_party_import_roots(node))
        self.assertEqual(["numpy"], detected)


def _has_try_ancestor(tree: ast.Module, target: ast.stmt) -> bool:
    """`target` 是否有 `ast.Try` 祖先（含自身为 Try 分支的情况）。"""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    current = parents.get(target)
    while current is not None:
        if isinstance(current, ast.Try):
            return True
        current = parents.get(current)
    return False


# ---------------------------------------------------------------------------
# §0.1 / §1.3 3.11+ API 禁用
# ---------------------------------------------------------------------------


class Python311ApiDetectionTests(unittest.TestCase):
    """§1.3：用 AST 判定 3.11 API，禁止文本子串匹配。"""

    def test_no_python311_apis_used_in_liteagent(self) -> None:
        offenders = []
        for path in _liteagent_module_paths():
            for lineno, shape in python311_api_hits(_parse(path)):
                offenders.append(f"{_relative(path)}:{lineno} {shape}")
        self.assertEqual(
            [], offenders, "使用了 §0.1 里 3.10 缺失的 API（红线 4）"
        )

    def test_detector_ignores_docstrings_and_comments(self) -> None:
        """禁用名字出现在 docstring / 注释里不得触发失败。"""
        source = (
            '"""Docstring: 用 asyncio.TaskGroup / asyncio.Runner / asyncio.timeout 都是 3.11，\n'
            "typing.Self、datetime.UTC、enum.StrEnum、tomllib、ExceptionGroup 同样不可用。\n"
            '"""\n'
            "# comment: asyncio.TaskGroup, tomllib.loads, StrEnum, Self, UTC, ExceptionGroup\n"
            "# 甚至带点的写法 asyncio.timeout(1) 也只是注释\n"
            "VALUE = 1\n"
        )
        self.assertEqual([], python311_api_hits(ast.parse(source)))

    def test_detector_flags_real_api_usage(self) -> None:
        source = (
            "import asyncio\n"
            "from datetime import UTC\n"
            "from enum import StrEnum\n"
            "from typing import Self\n"
            "from tomllib import loads\n"
            "async def f():\n"
            "    async with asyncio.TaskGroup() as group:\n"
            "        pass\n"
            "    await asyncio.timeout(1)\n"
            "runner = asyncio.Runner()\n"
            "bare = Runner()\n"
        )
        hits = python311_api_hits(ast.parse(source))
        shapes = {shape for _, shape in hits}
        self.assertIn("Attribute.TaskGroup", shapes)
        self.assertIn("Attribute.timeout", shapes)
        self.assertIn("Name.Runner", shapes)
        self.assertIn("ImportFrom.tomllib", shapes)
        self.assertIn("ImportFrom.name.StrEnum", shapes)
        self.assertIn("ImportFrom.name.Self", shapes)
        self.assertIn("ImportFrom.name.UTC", shapes)


# ---------------------------------------------------------------------------
# §1.1 依赖 DAG
# ---------------------------------------------------------------------------


class DependencyDagTests(unittest.TestCase):
    """§1.1：包内自由；包间只允许向下；同层边只允许 E1..E12。"""

    def test_whitelist_transcribes_all_twelve_entries(self) -> None:
        self.assertEqual(
            {f"E{i}" for i in range(1, 13)},
            set(ALLOWED_SAME_LEVEL_EDGES),
            "E1..E12 必须逐条转抄，编号不能缺",
        )

    def test_every_module_has_a_level(self) -> None:
        modules = {_module_name(path) for path in _liteagent_module_paths()}
        self.assertEqual(
            [],
            sorted(modules - set(MODULE_LEVELS)),
            "L 编号表缺少这些模块（§1.1 是唯一真值源）",
        )
        self.assertEqual(
            [],
            sorted(set(MODULE_LEVELS) - modules),
            "L 编号表里有仓库中不存在的模块",
        )

    def test_import_edges_follow_the_level_dag(self) -> None:
        known = set(MODULE_LEVELS)
        offenders = []
        for path in _liteagent_module_paths():
            current = _module_name(path)
            tree = _parse(path)
            for node in top_level_import_nodes(tree):
                for target in resolve_import_targets(current, node, known):
                    if current == ROOT_PACKAGE:
                        continue  # §1.1: `liteagent/__init__.py` 允许 import 任意 liteagent.*
                    if _package_of(current) == _package_of(target):
                        continue  # 包内自由
                    if MODULE_LEVELS[target] < MODULE_LEVELS[current]:
                        continue  # 只允许向下
                    if (current, target) in WHITELISTED_EDGES:
                        continue
                    offenders.append(
                        f"{current} -> {target} (line {node.lineno}): "
                        "跨包同层/向上边，不在 E1..E12 白名单里"
                    )
        self.assertEqual([], offenders, "§1.1 依赖 DAG 违规")

    def test_edge_resolution_prefers_the_longest_module_match(self) -> None:
        known = {"liteagent", "liteagent.errors", "liteagent.config", "liteagent.memory",
                 "liteagent.memory.base"}
        node = ast.parse("from liteagent import config\n").body[0]
        self.assertEqual(
            ["liteagent.config"], resolve_import_targets("liteagent.cli", node, known)
        )
        node = ast.parse("from liteagent.memory import base\n").body[0]
        self.assertEqual(
            ["liteagent.memory.base"], resolve_import_targets("liteagent.cli", node, known)
        )


# ---------------------------------------------------------------------------
# 附录 B / §1.4 公共 API 冻结面
# ---------------------------------------------------------------------------


class PublicApiSurfaceTests(unittest.TestCase):
    """附录 B：每个名字都能 `getattr(liteagent, name)`；§1.4/§10.5 各包 `__all__`。"""

    def test_appendix_b_names_are_all_importable_from_top_level(self) -> None:
        import liteagent

        missing = [name for name in APPENDIX_B_NAMES if not hasattr(liteagent, name)]
        self.assertEqual([], missing, "附录 B 的名字必须都能从 liteagent 取到")

    def test_top_level_all_matches_appendix_b(self) -> None:
        import liteagent

        declared = set(liteagent.__all__)
        frozen = set(APPENDIX_B_NAMES)
        self.assertEqual(
            sorted(frozen - declared), [], "附录 B 里的名字没有进 liteagent.__all__"
        )
        self.assertEqual(
            sorted(declared - frozen), [], "liteagent.__all__ 里有附录 B 未列出的名字"
        )
        self.assertEqual(120, len(APPENDIX_B_NAMES))

    def test_version_is_frozen_value(self) -> None:
        import liteagent

        self.assertEqual("0.1.0", liteagent.__version__)

    def test_subpackage_all_names_are_resolvable(self) -> None:
        """`[v3 澄清]` 断言的是「子包 `__all__` 里的名字能从**该子包**取到」。

        不是"能从 liteagent 顶层取到"：顶层是**精选导出**（附录 B 的 120 个名字，
        由 `test_top_level_all_matches_appendix_b` 双向锁死），
        像 `liteagent.llm.messages_tokens` 这类 helper 并不在顶层白名单里 ——
        要求子包 `__all__` ⊆ 顶层既与附录 B 冲突，也会让顶层命名空间被内部工具污染。
        顶层可解析性由 `test_appendix_b_names_are_all_importable_from_top_level` 负责。
        """
        for module_name in FROZEN_SUBPACKAGE_ALL:
            module = __import__(module_name, fromlist=["__name__"])
            declared = getattr(module, "__all__", None)
            self.assertIsNotNone(declared, f"{module_name} 必须有 __all__")
            missing = [name for name in declared if not hasattr(module, name)]
            self.assertEqual([], missing, f"{module_name}.__all__ 里的名字取不到")

    def test_subpackage_all_matches_frozen_lists(self) -> None:
        for module_name, expected in FROZEN_SUBPACKAGE_ALL.items():
            module = __import__(module_name, fromlist=["__name__"])
            declared = set(getattr(module, "__all__", ()))
            self.assertEqual(
                sorted(set(expected) - declared),
                [],
                f"{module_name}.__all__ 缺少规范里冻结的名字",
            )
            self.assertEqual(
                sorted(declared - set(expected)),
                [],
                f"{module_name}.__all__ 有规范未列出的名字",
            )


# ---------------------------------------------------------------------------
# `AgentResult(...)` 只能关键字构造
# ---------------------------------------------------------------------------


class AgentResultConstructionTests(unittest.TestCase):
    """§9：`AgentResult` 构造一律关键字形式（位置参数会把 status 塞进 output）。"""

    def test_agent_result_always_constructed_with_keywords(self) -> None:
        offenders = []
        scan_roots = [
            LITEAGENT_DIR,
            TESTS_DIR,
            REPO_ROOT / "examples",
            REPO_ROOT / "benchmarks",
            REPO_ROOT / "scripts",
        ]
        for directory in scan_roots:
            for path in _python_files(directory):
                for lineno, kind in agent_result_keyword_violations(_parse(path)):
                    offenders.append(f"{_relative(path)}:{lineno} AgentResult({kind})")
        self.assertEqual([], offenders, "AgentResult 必须全部用关键字参数构造")

    def test_detector_flags_positional_call_only(self) -> None:
        source = (
            '"""说明文字里的 AgentResult(FAILED, error=e) 不该被当成构造点。"""\n'
            "AgentResult(FAILED, error=e)\n"
            "AgentResult(status=AgentStatus.OK)\n"
            "AgentResult(*args)\n"
        )
        violations = agent_result_keyword_violations(ast.parse(source))
        self.assertEqual([(2, "positional"), (4, "Starred")], violations)


# ---------------------------------------------------------------------------
# §12 测试总量与测试卫生
# ---------------------------------------------------------------------------


class TestInventoryTests(unittest.TestCase):
    """§12：>= 220 个 test method，schema/executor 各 >= 30；每个文件 >= 4。"""

    def _test_files(self) -> list[pathlib.Path]:
        return sorted(TESTS_DIR.glob("test_*.py"))

    def _count_methods(self, path: pathlib.Path) -> int:
        tree = _parse(path)
        total = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                total += 1
        return total

    def test_total_test_method_count_is_at_least_220(self) -> None:
        counts = {_relative(path): self._count_methods(path) for path in self._test_files()}
        total = sum(counts.values())
        self.assertGreaterEqual(
            total,
            MIN_TOTAL_TEST_METHODS,
            f"§12 要求 >= {MIN_TOTAL_TEST_METHODS} 个 test_* 方法，当前 {total}：{counts}",
        )

    def test_heavy_test_files_have_at_least_30_methods(self) -> None:
        for relative in HEAVY_TEST_FILES:
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), f"{relative} 不存在")
            count = self._count_methods(path)
            self.assertGreaterEqual(
                count,
                MIN_HEAVY_FILE_TEST_METHODS,
                f"§12 要求 {relative} >= {MIN_HEAVY_FILE_TEST_METHODS} 个用例，当前 {count}",
            )

    def test_every_test_file_has_at_least_four_methods(self) -> None:
        offenders = [
            f"{_relative(path)}: {count}"
            for path in self._test_files()
            for count in (self._count_methods(path),)
            if count < MIN_TEST_METHODS_PER_FILE
        ]
        self.assertEqual([], offenders, f"§12：每个测试文件至少 {MIN_TEST_METHODS_PER_FILE} 个用例")

    def test_no_pytest_anywhere(self) -> None:
        offenders = []
        for directory in (LITEAGENT_DIR, TESTS_DIR, REPO_ROOT / "examples",
                          REPO_ROOT / "benchmarks", REPO_ROOT / "scripts"):
            for path in _python_files(directory):
                tree = _parse(path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "pytest" or alias.name.startswith("pytest."):
                                offenders.append(f"{_relative(path)}:{node.lineno} import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if (node.module or "").split(".")[0] == "pytest":
                            offenders.append(f"{_relative(path)}:{node.lineno} from {node.module}")
                    elif isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name) and node.value.id == "pytest":
                            offenders.append(f"{_relative(path)}:{node.lineno} pytest.{node.attr}")
        self.assertEqual([], offenders, "§0.3：禁止 pytest（环境里也装不了）")

    def test_no_bare_assert_in_test_files(self) -> None:
        offenders = []
        for path in self._test_files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    offenders.append(f"{_relative(path)}:{node.lineno}")
        self.assertEqual(
            [], offenders, "§0.3：不得用裸 assert 作为断言载体，必须 self.assertXxx"
        )

    def test_every_python_file_starts_with_future_annotations(self) -> None:
        offenders = []
        directories = [LITEAGENT_DIR, TESTS_DIR, REPO_ROOT / "examples",
                       REPO_ROOT / "benchmarks", REPO_ROOT / "scripts"]
        for directory in directories:
            for path in _python_files(directory):
                lines = path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    continue  # §12 冻结 tests/__init__.py 是空文件，没有"第一行"
                if lines[0] != "from __future__ import annotations":
                    offenders.append(f"{_relative(path)}: {lines[:1]!r}")
        self.assertEqual([], offenders, "§2.1：每个 .py 文件第一行必须是 from __future__ import annotations")

    def test_liteagent_modules_are_all_parseable(self) -> None:
        for path in _liteagent_module_paths():
            _parse(path)  # 语法错误会在这里炸出来


# ---------------------------------------------------------------------------
# [v3] 套件卫生：跑完之后不允许残留 liteagent 工作线程
# ---------------------------------------------------------------------------


class SuiteHygieneTests(unittest.TestCase):
    """[v3] 跑完整套件后进程里不得残留 `liteagent-*` 工作线程（`LoopBoundPool` 的池）。

    **这条用例为什么放在本文件**：`unittest` 的 `discover` 按模块名**排序**加载，
    `test_zero_dependency` 是字典序最后一个 `test_*.py`，所以它的用例在所有
    "会建 `Agent` / `ToolExecutor`"的模块（`test_agent_*` / `test_multiagent_*` /
    `test_tools_executor` / `test_e2e_code_assistant` …）之后才跑 —— 这里看到的
    就是**整套跑完**之后的进程状态。（单独 `python3 -m unittest tests.test_zero_dependency`
    时断言同样成立，只是覆盖面小。）

    **本文件的"只做静态判定"在这一个类上有唯一例外**（见文件头注释的修订说明）：
    线程存活状态无法用 `ast` 判定，只能真的去 `threading.enumerate()`；
    等待 worker 退场也只能真的等一小会儿。这两件事都被限制在本类里。

    **为什么不是 flaky**：`close_loop_bound_pools()` 走的是
    `ThreadPoolExecutor.shutdown(wait=False)` —— worker 的退出是**异步**的
    （要等它被唤醒、再读到停止标志），所以这里用**有上限的轮询**而不是
    "快照 + 固定 sleep"。worker 从被唤醒到退出是微秒级，`GRACE_S` 只是给 CI 的冗余；
    **它不是一个"多等等就绿了"的旋钮** —— 真的泄漏时它会等满上限然后失败。

    **这条断言拦什么（手动变异验证过载荷）**：把三处 `tearDownModule`
    （`test_agent_features` / `test_agent_react_native` / `test_agent_react_text`，
    它们调 `tests.helpers.close_loop_bound_pools()`）改成 `pass`，跑完整套件后残留
    **33** 个 `liteagent-exec_*` 线程，本用例随即变红。
    """

    #: 等 worker 退场的上限（秒）。理由见类 docstring。
    GRACE_S = 5.0
    #: 轮询间隔（秒）。10ms 量级足以让微秒级的退出瞬间被发现，又不空转 CPU。
    POLL_S = 0.02

    def test_suite_leaves_no_liteagent_worker_threads(self) -> None:
        deadline = time.monotonic() + self.GRACE_S
        leftover = liteagent_worker_threads()
        while leftover and time.monotonic() < deadline:
            time.sleep(self.POLL_S)
            leftover = liteagent_worker_threads()
        self.assertEqual(
            [],
            sorted(thread.name for thread in leftover),
            msg=(
                f"套件跑完后仍有 {len(leftover)} 个 liteagent 工作线程存活；"
                "对应模块应在 `tearDownModule` 里调用 "
                "`tests.helpers.close_loop_bound_pools()`"
                "（它 `aclose_all()` 之后还会自断言一次）"
            ),
        )


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
