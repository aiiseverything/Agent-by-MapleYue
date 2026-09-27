from __future__ import annotations

"""liteagent 的命令行入口（规范 §11；冻结决策 D-11：stdlib ``argparse``，不引入 ``typer``）。

这个模块只有三件事，分开写是为了让每一件都能被单独测：

1. **解析**：`build_parser()` 把 §11 冻结的用法字符串变成一棵 argparse 树。
2. **装配**：`build_agent_from_args` / `build_registry_from_args` /
   `build_memory_from_args` 把命令行参数变成"能跑的东西"。它们是 CLI 与
   ``examples/`` 的公共入口，因此**签名被冻结**，测试直接调用它们。
3. **渲染**：`render_result` 决定 stdout 上到底出现什么。

两条贯穿全文的硬约束：

* **退出码即契约**：``0`` 成功 / ``1`` agent 失败 / ``2`` 用法错误 / ``3`` provider 错误。
  `main()` 返回 int 而**不调用** ``sys.exit`` —— 单测可以直接 ``assertEqual(2, main([...]))``。
* **stdout 只放结果**：``--json`` 时 stdout 必须**只有**那一段 JSON（要能被 ``jq`` 吃），
  所以所有诊断、警告、traceback 一律走 stderr。这条约束解释了本文里
  ``_warn`` 全部写 stderr、``--json`` 时不挂 RichCallback 等一票决定。

模块级**不 import** `liteagent.tools.builtin`（`register_all`）：它的 import 代价最大，
而且缺了它不该让 ``import liteagent.cli`` 本身失败 —— 命令行工具必须能在"环境缺一块"时
给出人话。这条延迟加载是**真的生效**的（实测 `import liteagent.cli` 之后
``liteagent.tools.builtin`` 不在 `sys.modules` 里）。

`[v3 修正]` 上一版这里还写了"也**不 import** `liteagent.agent.agent`（Agent）"，
但那是**不成立的**：本模块必须 import `liteagent.agent.callbacks`（事件与回调契约），
而它会执行 `liteagent/agent/__init__.py`，后者的第 13 行就是
``from liteagent.agent.agent import Agent`` —— 也就是说 `Agent` 早就被拉进来了，
延迟加载那一半从来没有生效过（实测 `import liteagent.cli` 之后
``'liteagent.agent.agent' in sys.modules`` 为 True，且用 meta_path 屏蔽该模块会让
`import liteagent.cli` 直接失败）。因此 §11 冻结的 `build_agent_from_args -> Agent`
可以**按字面实现**：改成模块级 import 的边际代价约为 0，却能让
``inspect.signature`` 与 ``typing.get_type_hints`` 都拿到 `Agent`。
"""

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from liteagent.agent.callbacks import (
    CallbackLike,
    JsonlTraceCallback,
    LoggingCallback,
    RichCallback,
    TokenCounterCallback,
    EventType,
    load_trace,
    render_trace,
    trace_stats,
)
from liteagent.agent.agent import Agent
from liteagent.agent.state import AgentResult
from liteagent.config import AppConfig, load_dotenv, parse_bool
from liteagent.errors import ConfigError, LLMError, LiteAgentError
from liteagent.llm.registry import build_llm
from liteagent.memory.manager import MemoryManager
from liteagent.tools.executor import ExecutorConfig, ToolExecutor
from liteagent.tools.registry import ToolRegistry

# rich 是**可选加速器**：只在渲染处用（§1.3）。这里只探测可用性，真正的渲染交给
# `agent.callbacks.RichCallback`（它自己也有无 rich 的退化路径）。
try:  # pragma: no cover - 环境相关
    import rich as _rich  # noqa: F401

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rich = None
    RICH_AVAILABLE = False

_LOG = logging.getLogger("liteagent.cli")

# ---------------------------------------------------------------- 冻结常量（§11）
PROG: str = "liteagent"
EXIT_OK: int = 0
EXIT_AGENT_FAILED: int = 1
EXIT_USAGE: int = 2
EXIT_PROVIDER_ERROR: int = 3

#: `--approve write` 放行的工具（§11 冻结：只放行 write_file / delete_file）
_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "delete_file"})

#: 交互式 REPL 的内建命令（§11 冻结清单）
_CHAT_COMMANDS: tuple[str, ...] = (
    "/help", "/reset", "/tools", "/memory", "/trace", "/stats", "/mode", "/quit",
)

#: `--tools` 出现但值为空串时的语义 = 注册 0 个工具（§11 冻结）。
#: `_parse_tool_list("")` 必须返回 `[]`，而 `[e for e in x.split(",") if e]` 恰好做到。
_TOOL_SEP: str = ","


# ---------------------------------------------------------------- 输出通道


def _warn(message: str) -> None:
    """降级/兜底的可观测痕迹（§13 红线 12）。

    刻意**不走 logging**：CLI 的警告必须无条件出现在 stderr 上，
    而 logging 的输出目标取决于调用方怎么配置 root logger（测试里很可能被改过）。
    同时留一条 DEBUG 记录给"配置了 handler 的人"。
    """
    print(f"{PROG}: warning: {message}", file=sys.stderr)
    _LOG.debug("%s", message)


def _error(message: str, *, exc: BaseException | None = None, verbose: bool = False) -> None:
    """用户可读的错误行。stdout 永不出现 traceback 的兑现点之一。"""
    print(f"{PROG}: error: {message}", file=sys.stderr)
    if exc is not None:
        _LOG.debug("cli failure", exc_info=exc)
        if verbose:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _out(text: str = "") -> None:
    print(text)


def _configure_logging(verbose: bool) -> None:
    """把框架内部的日志引到 stderr。

    不写 `force=True`：调用方（测试、嵌入方）已经配好的 handler 不该被 CLI 抹掉。
    只保证"我们自己这条链"的级别正确，并**显式**把基础配置的 stream 指到 stderr
    —— 这是 `--json` 下 stdout 纯净性的前提（默认虽也是 stderr，但显式写出来才经得起 review）。
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(stream=sys.stderr, level=level)
    _LOG.setLevel(level)


def _version() -> str:
    """版本号唯一来源是包本身（importlib.metadata，失败回退常量），此处只做延迟读取。"""
    from liteagent import __version__

    return __version__


# ---------------------------------------------------------------- 参数默认值


#: 命令实现与装配函数都会被单测**直接**调用，可能拿到一个手工构造的 Namespace。
#: 因此每个字段在读取前都过一遍 `_normalize_args`：缺什么补什么（值就是 argparse
#: 里那个默认值）。集中在一处的收益是"字段名只有一份真值"。
_DEFAULT_ARG_VALUES: dict[str, Any] = {
    # run / chat / multi 共用
    "prompt": None,
    "prompt_file": None,
    "use_stdin": False,
    "provider": None,
    "model": None,
    "base_url": None,
    "api_key": None,
    "tools": None,
    "no_tools": False,
    "no_builtin": False,
    "sandbox_root": None,
    "allow_shell": False,
    "allow_network": True,
    "max_steps": None,
    "mode": None,
    "temperature": None,
    "max_tokens": None,
    "max_total_tokens": None,
    "max_wall_clock_s": None,
    "approve": None,  # None = 未显式指定（run 读作 never，chat 读作交互确认）
    "no_memory": False,
    "no_long_term": False,
    "memory_max_tokens": None,
    "trace": None,
    "json": False,
    "config": None,
    "dotenv": None,
    "verbose": False,
    "quiet": False,
    # tools / schema
    "format": None,
    "tags": None,
    "include_dangerous": False,
    "show_schema": False,
    "output": None,
    "builtin": False,
    "name": None,
    "tools_command": None,
    # trace
    "type": None,
    "agent": None,
    "step": None,
    "stats": False,
    "limit": None,
    # multi
    "agents": None,
    "manager": None,
    "workers": None,
    "max_depth": None,
    "subagent_concurrency": None,
    "team_name": None,
}


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """补齐缺失字段（原地修改并返回同一个对象）。"""
    for key, value in _DEFAULT_ARG_VALUES.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


# ---------------------------------------------------------------- 解析器


def _add_agent_options(
    parser: argparse.ArgumentParser, *, with_prompt: bool, with_mode: bool = True,
) -> None:
    """`run` / `chat` / `multi` 共用的选项集合（§11 冻结用法里的方括号部分）。

    `with_mode=False` 给 `multi`：它的 `--mode` 已经被冻结为**团队模式**
    （`{sequential,hierarchical}`），同一个解析器里注册两次 `--mode` 会直接
    `argparse.ArgumentError`。此时子 Agent 的工具调用模式由 `AppConfig.agent.mode` 决定。
    """
    if with_prompt:
        source = parser.add_mutually_exclusive_group()
        source.add_argument("-p", "--prompt", default=None, help="the user input to run")
        source.add_argument(
            "-f", "--file", dest="prompt_file", default=None, metavar="FILE",
            help="read the input from FILE",
        )
        source.add_argument(
            "--stdin", dest="use_stdin", action="store_true",
            help="read the input from stdin",
        )

    parser.add_argument("--provider", default=None, help="LLM provider name (openai/anthropic/echo/...)")
    parser.add_argument("--model", default=None, help="model name")
    parser.add_argument("--base-url", dest="base_url", default=None, help="override the provider base URL")
    parser.add_argument("--api-key", dest="api_key", default=None, help="override the API key")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--tools", default=None,
        help="comma separated tool/group names; '' registers zero tools",
    )
    selection.add_argument(
        "--no-tools", dest="no_tools", action="store_true",
        help="register zero tools (same as --tools '')",
    )
    selection.add_argument(
        "--no-builtin", dest="no_builtin", action="store_true",
        help="register only the tools declared in --config",
    )

    parser.add_argument("--sandbox-root", dest="sandbox_root", default=None, help="sandbox root for file tools")
    parser.add_argument("--allow-shell", dest="allow_shell", action="store_true", help="allow run_shell to really execute")
    parser.add_argument("--no-network", dest="allow_network", action="store_false", help="disable network access")

    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="ReAct step budget")
    if with_mode:
        parser.add_argument("--mode", choices=("auto", "native", "text"), default=None,
                            help="tool-calling mode")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None, help="max completion tokens")
    parser.add_argument("--max-total-tokens", dest="max_total_tokens", type=int, default=None,
                        help="total token budget for one run (0 = unlimited)")
    parser.add_argument("--max-wall-clock", dest="max_wall_clock_s", type=float, default=None,
                        help="wall clock budget in seconds for one run (0 = unlimited)")
    parser.add_argument(
        "--approve", choices=("never", "write", "all"), default=None,
        help="HITL policy for tools that require approval (default: never)",
    )

    parser.add_argument("--no-memory", dest="no_memory", action="store_true",
                        help="disable long-term memory and summarization")
    parser.add_argument("--no-long-term", dest="no_long_term", action="store_true",
                        help="disable long-term (vector) memory")
    parser.add_argument("--memory-max-tokens", dest="memory_max_tokens", type=int, default=None,
                        help="override the short-term window budget")

    parser.add_argument("--trace", default=None, metavar="FILE", help="append the event stream as JSONL to FILE")
    parser.add_argument("--config", default=None, metavar="FILE", help="AppConfig file (.json/.yaml/.yml)")
    parser.add_argument("--dotenv", default=None, metavar="FILE", help="load environment variables from FILE")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="stream events + tracebacks on stderr")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="suppress diagnostics")


def _add_registry_options(parser: argparse.ArgumentParser) -> None:
    """只造注册表、不跑 Agent 的命令（schema / tools）需要的选项。"""
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tools", default=None, help="comma separated tool/group names")
    selection.add_argument("--no-tools", dest="no_tools", action="store_true", help="register zero tools")
    selection.add_argument("--no-builtin", dest="no_builtin", action="store_true",
                           help="register only the tools declared in --config")
    parser.add_argument("--sandbox-root", dest="sandbox_root", default=None, help="sandbox root for file tools")
    parser.add_argument("--allow-shell", dest="allow_shell", action="store_true", help="allow run_shell to really execute")
    parser.add_argument("--no-network", dest="allow_network", action="store_false", help="disable network access")
    parser.add_argument("--config", default=None, metavar="FILE", help="AppConfig file (.json/.yaml/.yml)")
    parser.add_argument("--dotenv", default=None, metavar="FILE", help="load environment variables from FILE")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="verbose diagnostics")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="suppress diagnostics")


def build_parser() -> argparse.ArgumentParser:
    """§11 冻结用法字符串 -> argparse 树。每次调用都返回**新**解析器（无共享状态）。"""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="liteagent: a lightweight, dependency-free Agent Harness (ReAct + tools + memory).",
        epilog="run 'liteagent <command> -h' for command specific help",
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROG} {_version()}",
        help="show the version and exit",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- run ----
    p_run = sub.add_parser("run", help="run one ReAct turn and print the result")
    _add_agent_options(p_run, with_prompt=True)
    p_run.add_argument("--json", action="store_true", help="print the AgentResult as JSON only")
    p_run.set_defaults(func=cmd_run)

    # ---- chat ----
    p_chat = sub.add_parser("chat", help="interactive REPL")
    _add_agent_options(p_chat, with_prompt=False)
    p_chat.set_defaults(func=cmd_chat)

    # ---- tools ----
    p_tools = sub.add_parser("tools", help="inspect the tool registry")
    tools_sub = p_tools.add_subparsers(dest="tools_command", metavar="<action>")
    p_tl = tools_sub.add_parser("list", help="list registered tools")
    p_tl.add_argument("--format", choices=("table", "json"), default="table")
    p_tl.add_argument("--tags", default=None, help="comma separated tag filter")
    p_tl.add_argument("--include-dangerous", dest="include_dangerous", action="store_true",
                      help="include tools marked dangerous")
    _add_registry_options(p_tl)
    p_tl.set_defaults(func=cmd_tools)

    p_ts = tools_sub.add_parser("show", help="show one tool in detail")
    p_ts.add_argument("name", help="tool name")
    p_ts.add_argument("--format", choices=("markdown", "json"), default="markdown")
    p_ts.add_argument("--show-schema", dest="show_schema", action="store_true",
                      help="also print the provider schemas")
    _add_registry_options(p_ts)
    p_ts.set_defaults(func=cmd_tools)

    p_tc = tools_sub.add_parser("schema", help="export JSON schemas for the registry")
    # [SPEC-AMBIGUITY] §11 的用法串把 NAME 写成可选位置参数之外的形态：
    #   `liteagent tools schema [--format ...] [-o FILE]`
    # 但验收命令 `tools schema read_file --format openai` 需要一个 NAME。
    # 裁决：接受可选位置参数（超集），两者都成立。
    p_tc.add_argument("name", nargs="?", default=None, help="optional tool name filter")
    p_tc.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    p_tc.add_argument("-o", "--output", dest="output", default=None, metavar="FILE")
    _add_registry_options(p_tc)
    p_tc.set_defaults(func=cmd_tools)

    # ---- schema ----
    p_schema = sub.add_parser("schema", help="export the JSON schemas of the configured tool set")
    p_schema.add_argument("-o", "--output", dest="output", default=None, metavar="FILE")
    p_schema.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    p_schema.add_argument("--builtin", action="store_true", help="force the full builtin tool set")
    _add_registry_options(p_schema)
    p_schema.set_defaults(func=cmd_schema)

    # ---- trace ----
    p_trace = sub.add_parser("trace", help="inspect a JSONL trace file")
    p_trace.add_argument("file", help="the JSONL trace file")
    p_trace.add_argument("--type", default=None, help="only keep events of this type")
    p_trace.add_argument("--agent", default=None, help="only keep events of this agent")
    p_trace.add_argument("--step", type=int, default=None, help="only keep events of this step")
    p_trace.add_argument("--json", action="store_true", help="print JSON instead of the tree")
    p_trace.add_argument("--stats", action="store_true", help="print the aggregate statistics")
    p_trace.add_argument("--limit", type=int, default=None, help="keep the last N events")
    p_trace.set_defaults(func=cmd_trace)

    # ---- multi ----
    p_multi = sub.add_parser("multi", help="run a multi-agent team")
    p_multi.add_argument("--mode", choices=("sequential", "hierarchical"), required=True)
    p_multi.add_argument("--agents", required=True, metavar="FILE", help="team description (JSON)")
    p_multi.add_argument("--prompt", default=None, help="the task to run")
    p_multi.add_argument("--json", action="store_true", help="print the AgentResult as JSON only")
    p_multi.add_argument("--name", dest="team_name", default=None, help="team name")
    p_multi.add_argument("--max-depth", dest="max_depth", type=int, default=None)
    p_multi.add_argument("--subagent-concurrency", dest="subagent_concurrency", type=int, default=None)
    _add_agent_options(p_multi, with_prompt=False, with_mode=False)
    p_multi.set_defaults(func=cmd_multi)

    # ---- version ----
    p_version = sub.add_parser("version", help="print the version")
    p_version.set_defaults(func=cmd_version)

    return parser


# ---------------------------------------------------------------- 装配（公共入口）


def _parse_tool_list(raw: str) -> list[str]:
    """`--tools a,b` -> `["a", "b"]`；`""` -> `[]`（**注册 0 个工具**，与 None 语义不同）。"""
    return [item.strip() for item in raw.split(_TOOL_SEP) if item.strip()]


def _load_register_all() -> Callable[..., Any] | None:
    """加载内置工具注册入口（`liteagent/tools/builtin/`）。

    该包与其兄弟模块由不同实现者并行交付，缺失时**不能让整条 CLI 链失败**：
    降级为"注册 0 个内置工具"并**在 stderr 留一条 warning**（§13 红线 12）。
    """
    try:
        from liteagent.tools.builtin import register_all
    except ImportError as exc:
        _warn(f"builtin tools are unavailable ({exc}); no builtin tool will be registered")
        return None
    return register_all


def _load_agent_class() -> Any:
    """延迟加载 `Agent`（L4，import 代价最大的一层；也让本模块能独立被 import）。"""
    from liteagent.agent.agent import Agent

    return Agent


def _load_app_config(args: argparse.Namespace) -> AppConfig:
    """配置来源优先级：`--config` 文件 > 环境变量 > 内置默认值。

    `--dotenv` 显式给出的文件用 `override=True` 加载：用户都指名道姓了，
    让他被一个早已存在的空环境变量挡住是最难查的一类"配置不生效"。
    """
    if args.dotenv:
        load_dotenv(args.dotenv, override=True)
    if args.config:
        return AppConfig.from_file(args.config)
    return AppConfig.from_env()


def _build_llm(args: argparse.Namespace, app: AppConfig) -> Any:
    """命令行显式给的 LLM 字段覆盖配置；`None` 保留配置里的值（不是"改成 None"）。"""
    overrides: dict[str, Any] = {}
    for key, value in (
        ("provider", args.provider),
        ("model", args.model),
        ("base_url", args.base_url),
        ("api_key", args.api_key),
        ("temperature", args.temperature),
        ("max_tokens", args.max_tokens),
    ):
        if value is not None:
            overrides[key] = value
    config = replace(app.llm, **overrides) if overrides else app.llm
    return build_llm(config)


def _build_memory(args: argparse.Namespace, app: AppConfig, llm: Any = None) -> MemoryManager:
    """三层记忆的装配。

    `--no-memory` 的语义裁决（§11 只给了旗标名）：ReAct 循环**必须**有短期窗口
    （`Agent` 每轮都要 `abuild_prompt`），所以 `--no-memory` 关掉的是"长期向量 + 摘要压缩"，
    而不是把 MemoryManager 换成 None（那只会让 Agent 反过来造一个默认的、更重的记忆）。
    """
    config = app.memory
    overrides: dict[str, Any] = {}
    if args.no_memory:
        overrides["long_term_enabled"] = False
        overrides["summary_enabled"] = False
    elif args.no_long_term:
        overrides["long_term_enabled"] = False
    if args.memory_max_tokens is not None:
        overrides["buffer_max_tokens"] = args.memory_max_tokens
    if overrides:
        config = replace(config, **overrides)
    # llm 透传：摘要走 LLM 而不是抽取式兜底。外部传入 memory 时不替换其 summarizer。
    return MemoryManager.from_config(config, llm=llm)


def _resolve_includes(args: argparse.Namespace, app: AppConfig) -> list[str] | None:
    """`--tools` / `--no-tools` / `--no-builtin` / `AppConfig.tools` 的合并规则（§11/§7.5）。

    返回 `None` 表示"全部内置组"，`[]` 表示"一个都不注册" —— 这两者**必须**区分。
    """
    if args.no_tools:
        return []  # 等价 --tools ""
    if args.tools is not None:
        return _parse_tool_list(args.tools)  # "" -> []
    if args.no_builtin:
        return list(app.tools)  # 只注册 --config 声明的；没有声明就是 []
    return list(app.tools) or None


def _build_registry(
    args: argparse.Namespace, app: AppConfig, memory: MemoryManager | None, *,
    force_all: bool = False,
) -> ToolRegistry:
    """`build_registry_from_args` 的实现体：多一个 `memory` 注入点。

    为什么拆出来：`register_all(memory=...)` 需要的是**Agent 用的那一个** MemoryManager
    （remember/recall 要写进同一条长期记忆），而冻结的公开签名里没有这个参数。
    公开函数自建一个（独立可用），Agent 装配路径传共享实例（语义正确）。
    """
    registry = ToolRegistry()
    if args.no_tools and args.no_builtin:
        raise ConfigError(
            "--no-tools and --no-builtin are mutually exclusive: "
            "--no-tools registers zero tools, --no-builtin keeps the tools declared in --config"
        )
    include = None if force_all else _resolve_includes(args, app)
    register_all = _load_register_all()
    if register_all is None:
        return registry
    return register_all(
        registry,
        include=include,
        sandbox_root=args.sandbox_root or app.sandbox_root,
        # None = 交给 make_shell_tools 读 LITEAGENT_ALLOW_SHELL；True = 命令行显式放行。
        allow_shell=True if args.allow_shell else None,
        allow_network=bool(args.allow_network),
        memory=memory,
    )


def build_registry_from_args(args: argparse.Namespace) -> ToolRegistry:
    """从命令行参数造注册表（CLI 与 examples 共用；单测直接调用）。

    独立调用时自建 MemoryManager 是为了让 "memory" 工具组可用；
    Agent 装配路径（`build_agent_from_args`）走 `_build_registry` 复用同一实例。
    """
    args = _normalize_args(args)
    app = _load_app_config(args)
    memory = None
    if not (args.no_memory or args.no_long_term):
        memory = _build_memory(args, app)
    return _build_registry(args, app, memory)


def build_memory_from_args(args: argparse.Namespace) -> MemoryManager:
    """从命令行参数造记忆层（CLI 与 examples 共用；单测直接调用）。"""
    args = _normalize_args(args)
    return _build_memory(args, _load_app_config(args))


def _build_agent_config(args: argparse.Namespace, app: AppConfig, *, with_mode: bool = True) -> Any:
    """命令行显式给的字段覆盖 AgentConfig；`None` 保留配置里的值。

    `with_mode=False` 给 `multi` 的子 Agent 用：那里 `--mode` 已经被冻结为**团队模式**
    （`sequential`/`hierarchical`），把它写进 `AgentConfig.mode` 会产出非法取值。
    """
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if with_mode and args.mode is not None:
        overrides["mode"] = args.mode
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    if args.max_total_tokens is not None:
        # 0 读作"不限"：用户想表达的是"别管预算"，而不是"立刻超预算失败"。
        overrides["max_total_tokens"] = None if args.max_total_tokens == 0 else args.max_total_tokens
    if args.max_wall_clock_s is not None:
        overrides["max_wall_clock_s"] = None if args.max_wall_clock_s <= 0 else args.max_wall_clock_s
    return replace(app.agent, **overrides) if overrides else app.agent


def _approval_policy(kind: str | None, *, interactive: bool) -> Callable[..., bool] | None:
    """`--approve` -> `ExecutorConfig.approval_policy`（§11 + §7.4.1 步骤 4.5）。

    `never` -> None：执行器把"没有策略"判成拒绝，这正是规范冻结的默认行为。
    `chat` 里 policy 额外承担"交互确认"：规范把 `input(...)` 确认写在该子命令下，
    因此 chat 在自动放行之外**再问一次**用户（`never` 于是退化成"每次都问"）。
    """
    mode = kind or ("never" if not interactive else "interactive")

    def policy(call: Any, tool: Any) -> bool:
        if mode == "all":
            return True
        if mode == "write" and call.name in _WRITE_TOOLS:
            return True
        if not interactive:
            return False
        try:
            arguments = json.dumps(call.arguments, ensure_ascii=False)
        except (TypeError, ValueError):  # arguments 里混进了不可序列化的东西
            arguments = repr(call.arguments)
        try:
            answer = input(f"approve {tool.name}({arguments[:200]})? [y/N] ")
        except EOFError:  # 非交互 stdin（管道 / 测试）：按拒绝处理，绝不猜
            _warn("no interactive input available for approval; denying the tool call")
            return False
        return parse_bool(answer, default=False)

    if mode == "never" and not interactive:
        return None
    return policy


def _build_executor_config(args: argparse.Namespace, app: AppConfig, *, interactive: bool) -> ExecutorConfig:
    policy = _approval_policy(args.approve, interactive=interactive)
    if policy is None and app.executor.approval_policy is None:
        return app.executor
    return replace(app.executor, approval_policy=policy)


def _build_callbacks(
    args: argparse.Namespace, *, interactive: bool = False,
) -> list[CallbackLike]:
    """按 `--trace` / `-v` 造事件回调（**渲染归命令层**，装配函数不碰它）。

    `--json` 时不挂任何写 stdout 的回调：stdout 只能有那段 JSON。
    """
    callbacks: list[CallbackLike] = []
    trace_path = args.trace or os.environ.get("LITEAGENT_TRACE")
    if trace_path:
        callbacks.append(JsonlTraceCallback(trace_path))
    if args.quiet or args.json:
        return callbacks
    if args.verbose or interactive:
        if RICH_AVAILABLE and sys.stdout.isatty():
            callbacks.append(RichCallback())
        else:
            callbacks.append(LoggingCallback(_LOG))
    return callbacks


def _close_callbacks(callbacks: Sequence[CallbackLike]) -> None:
    """关闭持有文件句柄的回调（只有 JsonlTraceCallback 有 close）。"""
    for callback in callbacks:
        closer = getattr(callback, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # 关闭失败不该掩盖真正的结果
                _warn(f"failed to close callback {callback!r}: {exc}")


def _build_agent(args: argparse.Namespace, app: AppConfig, *, interactive: bool) -> Any:
    """装配 Agent（`build_agent_from_args` 的实现体，多一个"交互审批"开关）。

    刻意**不**在这里挂事件回调：同一个 Agent 对象在不同命令下的渲染要求不同
    （`run` 要 --json 纯净、`chat` 要一直流式），渲染属于命令层。
    """
    llm = _build_llm(args, app)
    memory = _build_memory(args, app, llm)
    registry = _build_registry(args, app, memory)
    agent_config = _build_agent_config(args, app)
    executor = ToolExecutor(registry, _build_executor_config(args, app, interactive=interactive))
    agent_cls = _load_agent_class()
    return agent_cls(
        llm=llm,
        tools=registry,
        memory=memory,
        config=agent_config,
        executor=executor,
        name=agent_config.name,
        description=agent_config.description,
    )


def build_agent_from_args(args: argparse.Namespace) -> Agent:
    """从命令行参数造一个装配完整的 Agent（CLI 与 examples 共用；单测直接调用）。"""
    args = _normalize_args(args)
    return _build_agent(args, _load_app_config(args), interactive=False)


# ---------------------------------------------------------------- 渲染


def render_result(result: AgentResult, *, as_json: bool = False) -> str:
    """把 AgentResult 变成 stdout 上的文本（§11 冻结格式）。

    `as_json=True` 时输出 `result.to_dict()` 的缩进 JSON —— 调用方必须保证
    它是 stdout 上**唯一**的东西（日志都在 stderr）。
    """
    if as_json:
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    usage = result.usage
    lines = [
        f"status: {result.status.value}",
        f"steps: {result.steps}",
        f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
        f"total={usage.total_tokens}",
        f"cost: {_format_cost(result)}",
        f"duration: {result.duration_ms:.1f}ms",
    ]
    # 失败信息只在真的失败时出现：`AgentResult.error` 为 None 时输出与冻结样例**逐字**一致。
    if result.error is not None:
        lines.append(f"error: {type(result.error).__name__}: {result.error}")
    lines.append("---")
    lines.append(result.output)
    return "\n".join(lines)


def _format_cost(result: AgentResult) -> str:
    """`metadata["cost_usd"]` 为 None 表示"模型不在价格表里"，如实写 n/a（诚实 > 猜）。"""
    cost = result.metadata.get("cost_usd") if isinstance(result.metadata, Mapping) else None
    if cost is None:
        return "n/a"
    return f"${float(cost):.6f}"


def _one_line(text: str, *, max_chars: int = 60) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


def _ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """零依赖的定宽表格（不用 rich：`tools list` 的输出要能被 grep/diff 稳定消费）。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def _schemas_payload(tools: Sequence[Any], fmt: str) -> str:
    if fmt == "anthropic":
        schemas = [tool.to_anthropic_schema() for tool in tools]
    else:
        schemas = [tool.to_openai_schema() for tool in tools]
    return json.dumps(schemas, ensure_ascii=False, indent=2)


def _write_output(text: str, path: str | None) -> None:
    if not path:
        _out(text)
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc}") from exc
    # 确认信息走 stderr：stdout 要么干净（重定向到文件时可能被继续管道化），要么就是结果本身。
    _LOG.info("wrote %d bytes to %s", len(text), path)


# ---------------------------------------------------------------- 子命令实现


def _resolve_prompt(args: argparse.Namespace) -> str:
    """`-p` / `-f` / `--stdin` 三选一；都没有 -> ConfigError（main 会映射成退出码 2）。"""
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            raise ConfigError(f"cannot read prompt file {args.prompt_file}: {exc}") from exc
    if args.use_stdin:
        return sys.stdin.read()
    raise ConfigError("no prompt provided; use -p/--prompt, -f/--file or --stdin")


def cmd_run(args: argparse.Namespace) -> int:
    args = _normalize_args(args)
    prompt = _resolve_prompt(args)  # 缺 prompt -> ConfigError -> EXIT_USAGE(2)
    agent = build_agent_from_args(args)
    callbacks = _build_callbacks(args)
    try:
        result = agent.run(prompt, callbacks=callbacks or None)
    finally:
        _close_callbacks(callbacks)
    _out(render_result(result, as_json=bool(args.json)))
    return EXIT_OK if result.ok else EXIT_AGENT_FAILED


def cmd_tools(args: argparse.Namespace) -> int:
    args = _normalize_args(args)
    action = args.tools_command
    if action is None:
        _error("missing tools action; expected one of: list, show, schema")
        return EXIT_USAGE

    registry = build_registry_from_args(args)
    if action == "list":
        return _tools_list(args, registry)
    if action == "show":
        return _tools_show(args, registry)
    if action == "schema":
        return _tools_schema(args, registry)
    _error(f"unknown tools action: {action}")  # argparse 的 choices 之外只剩手工构造的 Namespace
    return EXIT_USAGE


def _tools_list(args: argparse.Namespace, registry: ToolRegistry) -> int:
    tags = _parse_tool_list(args.tags) if args.tags is not None else None
    tools = registry.list(tags=tags or None, include_dangerous=bool(args.include_dangerous))
    if args.format == "json":
        _out(json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False, indent=2))
        return EXIT_OK
    rows = []
    for tool in tools:
        spec = tool.spec
        rows.append((
            spec.name,
            ",".join(spec.tags) or "-",
            "yes" if spec.dangerous else "no",
            "yes" if spec.requires_approval else "no",
            "yes" if spec.idempotent else "no",
            _one_line(spec.description),
        ))
    header = ("name", "tags", "dangerous", "approval", "idempotent", "description")
    _out(_ascii_table(header, rows))
    _LOG.info("%d tool(s) listed", len(rows))
    return EXIT_OK


def _tools_show(args: argparse.Namespace, registry: ToolRegistry) -> int:
    if not args.name:
        _error("missing tool name; usage: liteagent tools show NAME")
        return EXIT_USAGE
    if not args.show_schema:
        _out(registry.describe(args.name, fmt=args.format))
        return EXIT_OK

    tool = registry.get(args.name)  # 未命中 -> ToolNotFoundError（main 映射成退出码 1）
    if args.format == "json":
        payload = {
            "tool": tool.to_dict(),
            "openai": tool.to_openai_schema(),
            "anthropic": tool.to_anthropic_schema(),
        }
        _out(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _out(registry.describe(args.name, fmt="markdown"))
        _out("")
        _out("## schema (openai)")
        _out("```json")
        _out(json.dumps(tool.to_openai_schema(), ensure_ascii=False, indent=2))
        _out("```")
    return EXIT_OK


def _tools_schema(args: argparse.Namespace, registry: ToolRegistry) -> int:
    if args.name:
        tools = [registry.get(args.name)]
    else:
        tools = list(registry)
    _write_output(_schemas_payload(tools, args.format), args.output)
    return EXIT_OK


def cmd_schema(args: argparse.Namespace) -> int:
    """顶层 `schema`：导出"当前配置下的工具集"的 JSON Schema（`--builtin` 强制全部内置组）。"""
    args = _normalize_args(args)
    app = _load_app_config(args)
    memory = None
    if not (args.no_memory or args.no_long_term):
        memory = _build_memory(args, app)
    registry = _build_registry(args, app, memory, force_all=bool(args.builtin))
    _write_output(_schemas_payload(list(registry), args.format), args.output)
    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    args = _normalize_args(args)
    events = load_trace(args.file)
    if not events:
        _warn(f"no usable event found in {args.file}")

    if args.type is not None:
        wanted = EventType.coerce(args.type)  # 未知类型 -> ConfigError -> EXIT_USAGE
        events = [event for event in events if event.type is wanted]
    if args.agent is not None:
        events = [event for event in events if event.agent_name == args.agent]
    if args.step is not None:
        events = [event for event in events if event.step == args.step]
    if args.limit is not None and args.limit >= 0:
        events = events[-args.limit:] if args.limit else []

    if args.stats:
        stats = trace_stats(events)
        if args.json:
            _out(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            _out(_render_stats(stats))
        return EXIT_OK
    if args.json:
        _out(json.dumps([event.to_dict() for event in events], ensure_ascii=False, indent=2))
        return EXIT_OK
    _out(render_trace(events, indent=True))
    return EXIT_OK


def _render_stats(stats: Mapping[str, Any]) -> str:
    """人类可读的 stats 摘要（`--json` 时直接用 `trace_stats` 的冻结字段表，不经过这里）。"""
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in stats.items()]
    return "\n".join(lines)


def cmd_chat(args: argparse.Namespace) -> int:
    args = _normalize_args(args)
    app = _load_app_config(args)
    agent = _build_agent(args, app, interactive=True)

    token_counter = TokenCounterCallback()
    callbacks = _build_callbacks(args, interactive=True)
    callbacks.append(token_counter)
    # 挂在 Agent 自己的 CallbackManager 上（而不是每轮传参）：REPL 的每一轮都要流式渲染，
    # 挂一次就是全部轮次；`arun(callbacks=None)` 必然回落到 self.callbacks，否则这个
    # 公开属性就是死代码。
    for callback in callbacks:
        agent.callbacks.add(callback)

    _out(f"{PROG} chat ({agent.name}) — /help for commands, /quit to exit")
    try:
        while True:
            try:
                line = input(f"{PROG}> ")
            except EOFError:
                _out("")  # 非交互 stdin：干净退出，不把 EOF 当错误
                break
            except KeyboardInterrupt:
                _out("")
                _warn("interrupted; leaving chat")
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if _chat_command(line, agent, token_counter, args):
                    break
                continue
            try:
                result = agent.run(line)  # 事件回调已挂在 agent.callbacks 上
            except KeyboardInterrupt:
                _warn("interrupted; the run was aborted")
                continue
            if result.ok:
                _out(result.output)
            else:
                _error(f"{result.status.value}: {result.error or 'no output'}", verbose=args.verbose)
    finally:
        _close_callbacks(callbacks)
    _out("bye")
    return EXIT_OK


def _chat_command(
    line: str, agent: Any, token_counter: TokenCounterCallback, args: argparse.Namespace,
) -> bool:
    """处理一条 `/` 开头的内建命令。返回 True 表示"该退出 REPL 了"。"""
    head, _, rest = line.partition(" ")
    command, argument = head.lower(), rest.strip()

    if command in ("/quit", "/exit"):
        return True
    if command == "/help":
        _out("commands: " + "  ".join(_CHAT_COMMANDS))
        _out("  /mode {auto,native,text}   switch the tool-calling mode")
        return False
    if command == "/reset":
        agent.reset(clear_memory=True)
        _out("session reset")
        return False
    if command == "/tools":
        names = agent.tools.names()
        _out(f"{len(names)} tool(s): {', '.join(names)}" if names else "no tool registered")
        return False
    if command == "/memory":
        stats = agent.memory.stats()
        for key, value in stats.items():
            _out(f"{key}: {value}")
        return False
    if command == "/trace":
        path = args.trace or os.environ.get("LITEAGENT_TRACE")
        if path:
            _out(f"trace file: {path} ({len(load_trace(path))} event(s))")
        else:
            _out("no trace file; start chat with --trace FILE to record events")
        return False
    if command == "/stats":
        usage = token_counter.usage
        cost = "n/a" if token_counter.cost_usd is None else f"${token_counter.cost_usd:.6f}"
        _out(
            f"calls: {token_counter.calls}  prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens} cost={cost}"
        )
        return False
    if command == "/mode":
        if argument not in ("auto", "native", "text"):
            _error(f"/mode expects one of auto/native/text, got {argument!r}")
            return False
        agent.config.mode = argument
        _out(f"mode = {argument}")
        return False
    _error(f"unknown command {command!r}; try /help")
    return False


def cmd_multi(args: argparse.Namespace) -> int:
    args = _normalize_args(args)
    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()  # `echo "task" | liteagent multi ...`
    if not prompt:
        _error("no prompt provided; use --prompt TEXT (or pipe the task on stdin)")
        return EXIT_USAGE

    app = _load_app_config(args)
    spec = _load_agents_file(args.agents)
    file_mode = str(spec.get("mode") or "").strip().lower()
    if file_mode and file_mode != args.mode:
        _warn(f"--mode {args.mode} overrides the mode in {args.agents} ({file_mode})")

    team_config = app.team
    team_overrides: dict[str, Any] = {}
    if args.max_depth is not None:
        team_overrides["max_depth"] = args.max_depth
    if args.subagent_concurrency is not None:
        team_overrides["subagent_concurrency"] = args.subagent_concurrency
    if team_overrides:
        team_config = replace(team_config, **team_overrides)

    team = _build_team(args, app, spec, team_config)
    callbacks = _build_callbacks(args)
    try:
        result = team.run(prompt, callbacks=callbacks or None)
    finally:
        _close_callbacks(callbacks)
    _out(render_result(result, as_json=bool(args.json)))
    return EXIT_OK if result.ok else EXIT_AGENT_FAILED


def _load_agents_file(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"agents file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read agents file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path} must contain a JSON object, got {type(payload).__name__}")
    return dict(payload)


def _agents_from_spec(
    entries: Sequence[Any], args: argparse.Namespace, app: AppConfig,
) -> tuple[list[Any], list[Any]]:
    """把 JSON 里的名字/描述变成 Agent 列表。

    返回 `(agents, raw_entries)`：raw 里带上 `input_template` 等流水线参数，供
    sequential 直接构造 `SequentialStep`（`build_team` 只接受裸 Agent，丢模板）。
    """
    from liteagent.multiagent.sequential import SequentialStep

    agents: list[Any] = []
    steps: list[Any] = []
    seen: dict[str, Any] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            options: dict[str, Any] = {"name": entry}
        elif isinstance(entry, Mapping):
            options = dict(entry)
        else:
            raise ConfigError(f"agent entry #{index} must be a string or an object")
        name = str(options.get("name") or options.get("agent") or "").strip()
        if not name:
            raise ConfigError(f"agent entry #{index} has no 'agent'/'name'")
        if name not in seen:
            seen[name] = _make_named_agent(name, options, args, app)
        kwargs: dict[str, Any] = {"agent": seen[name]}
        for key in ("input_template", "output_key", "optional", "max_chars"):
            if options.get(key) is not None:
                kwargs[key] = options[key]
        steps.append(SequentialStep(**kwargs))
        if seen[name] not in agents:
            agents.append(seen[name])
    return agents, steps


def _make_named_agent(name: str, options: Mapping[str, Any], args: argparse.Namespace, app: AppConfig) -> Any:
    """按名字造一个 Agent：复用 CLI 的 provider/工具/记忆配置，只覆盖身份与步数。"""
    overrides: dict[str, Any] = {"name": name}
    if options.get("description"):
        overrides["description"] = str(options["description"])
    if options.get("system_prompt"):
        overrides["system_prompt"] = str(options["system_prompt"])
    if options.get("max_steps") is not None:
        overrides["max_steps"] = int(options["max_steps"])
    app_for_agent = replace(app, agent=replace(app.agent, **overrides))
    # 子 Agent 同样吃 --max-steps/--temperature/... （with_mode=False：--mode 归团队模式）
    agent_config = _build_agent_config(args, app_for_agent, with_mode=False)
    llm = _build_llm(args, app)
    memory = _build_memory(args, app, llm)
    registry = _build_registry(args, app, memory)
    executor = ToolExecutor(registry, _build_executor_config(args, app, interactive=False))
    return _load_agent_class()(
        llm=llm, tools=registry, memory=memory, config=agent_config,
        executor=executor, name=name, description=agent_config.description,
    )


def _build_team(args: argparse.Namespace, app: AppConfig, spec: Mapping[str, Any], team_config: Any) -> Any:
    """按 §10.3/§10.4 的两种 JSON 形态造团队。"""
    if args.mode == "sequential":
        raw_steps = spec.get("steps")
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)) or not raw_steps:
            raise ConfigError(f"--agents file must contain a non-empty 'steps' array for sequential mode")
        agents, steps = _agents_from_spec(list(raw_steps), args, app)
        from liteagent.multiagent.sequential import SequentialAgent

        return SequentialAgent(
            steps, name=args.team_name or "sequential", config=team_config, callbacks=None,
        )

    raw_manager = spec.get("manager")
    raw_workers = spec.get("workers")
    if not isinstance(raw_workers, Sequence) or isinstance(raw_workers, (str, bytes)) or not raw_workers:
        raise ConfigError("--agents file must contain a non-empty 'workers' array for hierarchical mode")
    manager_entries = [raw_manager] if raw_manager is not None else []
    managers, _ = _agents_from_spec(manager_entries, args, app)
    if not managers:
        raise ConfigError("hierarchical mode needs a 'manager' entry in the --agents file")
    workers, _ = _agents_from_spec(list(raw_workers), args, app)
    from liteagent.multiagent.base import build_team

    return build_team(
        workers, mode="hierarchical", config=team_config, manager=managers[0],
        name=args.team_name or "hierarchy",
    )


def cmd_version(args: argparse.Namespace) -> int:
    _out(_version())
    return EXIT_OK


# ---------------------------------------------------------------- 入口


def main(argv: Sequence[str] | None = None) -> int:
    """解析 + 分发 + 退出码映射（§11 冻结）。**返回**退出码，不调用 `sys.exit`。

    异常 -> 退出码的映射表（冻结 + 一处补充）：
    `ConfigError` -> 2；`LLMError` -> 3；`KeyboardInterrupt` -> 1；
    其余 `LiteAgentError` -> 1（它们全都是"这次跑不成"，与 agent 失败同属一类）；
    其余 `Exception` -> 1 并**保证不把 traceback 打到 stdout**（`-v` 时才打 stderr）。
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse 的 -h/--version 与用法错误都走这里
        return _exit_code_from(exc)

    _configure_logging(getattr(args, "verbose", False))

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    verbose = bool(getattr(args, "verbose", False))
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        _error("interrupted", verbose=verbose)
        return EXIT_AGENT_FAILED
    except SystemExit as exc:  # 子命令内部（如 chat）也可能触发解析器退出
        return _exit_code_from(exc)
    except ConfigError as exc:
        _error(str(exc), exc=exc, verbose=verbose)
        return EXIT_USAGE
    except LLMError as exc:
        _error(str(exc), exc=exc, verbose=verbose)
        return EXIT_PROVIDER_ERROR
    except LiteAgentError as exc:
        _error(str(exc), exc=exc, verbose=verbose)
        return EXIT_AGENT_FAILED
    except Exception as exc:  # 兜底：宁可给一个"1 + 一行人话"，也不给用户一段 traceback 到 stdout
        _error(f"unexpected {type(exc).__name__}: {exc}", exc=exc, verbose=verbose)
        return EXIT_AGENT_FAILED


def _exit_code_from(exc: SystemExit) -> int:
    """argparse 用 SystemExit 表达"退出码"，这里把它翻译成 int（0 表示正常退出）。

    `--help` / `--version` 走的是 `code=0`/`code=None` 这两条路径；用法错误是 2。
    """
    code = exc.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return EXIT_USAGE
