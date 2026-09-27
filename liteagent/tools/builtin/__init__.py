from __future__ import annotations

"""内置工具的注册入口（规范 §7.5 的 ``builtin/__init__.py``）。

**这里是整个框架唯一有"批量副作用"的地方，而且必须显式调用**（不依赖 import 副作用）：
`import liteagent.tools.builtin` 不会注册任何工具，只有 `register_all(registry, ...)`
才会往注册表里写。理由很实际 —— import 副作用会让"注册了什么"取决于 import 顺序，
在一个测试与 CLI 共享进程的仓库里，这是最难复现的一类 bug。

``include`` / ``exclude`` 的语义（[v2 冻结]，v1 在这上面完全悬空）：

- 每个元素**先按组名**查 ``BUILTIN_TOOL_GROUPS``（命中即展开该组全部工具），
  **未命中再按工具名**查 ``BUILTIN_TOOL_NAMES``，两者都不中 -> ``ConfigError``。
  这条"先组后名"的规则让 ``--tools files``（5 个）与 ``--tools read_file``（1 个）
  有确定且可断言的区别。
- ``include=None`` = 全部组（``memory`` 组仅当 ``memory is not None`` 时注册）；
  ``include=[]`` = **注册 0 个工具**，与 ``None`` 语义**不同**，必须区分。
  两者都写成"空/缺省"是最容易踩的一个坑：CLI 的 ``--tools ""`` 依赖这个区别。
- ``exclude`` 在 include 展开**之后**应用，优先级高于 include；同样接受组名或工具名，
  未命中 -> ``ConfigError``（拼错的排除项必须报错，否则"排除了但没生效"会被当成框架 bug）。

沙箱解析优先级（冻结）：``sandbox_root=`` 参数 > ``LITEAGENT_SANDBOX_ROOT`` > ``os.getcwd()``。
**只有实现 files/code 组时才真的去构造 ``PathSandbox``** —— 否则 ``include=["web"]``
会因为一个它根本用不到的沙箱参数而失败，这是纯粹的意外耦合。
"""

import os
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from liteagent.errors import ConfigError
from liteagent.tools.base import Tool
from liteagent.tools.builtin.code import make_code_tools
from liteagent.tools.builtin.files import PathSandbox, make_file_tools
from liteagent.tools.builtin.memory_tools import make_memory_tools
from liteagent.tools.builtin.shell import make_shell_tools
from liteagent.tools.builtin.web import (
    SearchBackend,
    make_web_tools,
)

if TYPE_CHECKING:  # pragma: no cover - 只为注解
    from liteagent.llm.transport import Transport
    from liteagent.memory.manager import MemoryManager
    from liteagent.tools.registry import ToolRegistry

__all__ = ["BUILTIN_TOOL_GROUPS", "BUILTIN_TOOL_NAMES", "register_all"]

# [v2 变更] 值是**函数对象**（不是字符串名）：注册表需要一个可直接调用的 factory，
# 字符串名会引入"再查一次表"的间接层，并且写错名字时只有运行时才发现。
BUILTIN_TOOL_GROUPS: dict[str, Callable[..., list[Tool]]] = {
    "files": make_file_tools,
    "shell": make_shell_tools,
    "code": make_code_tools,
    "web": make_web_tools,
    "memory": make_memory_tools,
}

# 全部内置工具名（冻结，顺序即注册顺序）。
BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "write_file",
    "list_dir",
    "search_files",
    "delete_file",
    "run_shell",
    "python_exec",
    "python_eval",
    "run_tests",
    "web_search",
    "fetch_url",
    "remember",
    "recall",
)

# 工具名 -> 组名。由 BUILTIN_TOOL_NAMES 与下面的分组表**交叉校验**，
# 而不是各写一份：两份手写清单迟早会漂移，而"某个工具不在任何组里"会让
# `include=["read_file"]` 这种用法静默失效。
_GROUP_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "files": ("read_file", "write_file", "list_dir", "search_files", "delete_file"),
    "shell": ("run_shell",),
    "code": ("python_exec", "python_eval", "run_tests"),
    "web": ("web_search", "fetch_url"),
    "memory": ("remember", "recall"),
}

_TOOL_TO_GROUP: dict[str, str] = {
    tool_name: group for group, names in _GROUP_TOOL_NAMES.items() for tool_name in names
}


def _check_tables_consistency() -> None:
    """两组冻结清单必须完全一致（导入时自检，失败即 bug）。

    这是最便宜的一种"守门"：把漂移暴露在 ``import`` 时刻，而不是等某个使用者发现
    "``--tools run_shell`` 说没有这个工具"时才回头怀疑人生。
    """
    grouped = tuple(name for names in _GROUP_TOOL_NAMES.values() for name in names)
    if set(grouped) != set(BUILTIN_TOOL_NAMES):
        missing = sorted(set(BUILTIN_TOOL_NAMES) - set(grouped))
        extra = sorted(set(grouped) - set(BUILTIN_TOOL_NAMES))
        raise ConfigError(
            f"BUILTIN_TOOL_NAMES and the per-group tables disagree "
            f"(missing from groups: {missing}; not in BUILTIN_TOOL_NAMES: {extra})"
        )


_check_tables_consistency()


def _resolve_element(element: str, *, what: str) -> tuple[str, str | None]:
    """把一个 include/exclude 元素解析成 ``(组名, 工具名 | None)``。

    ``what`` 只用于错误文案（"include"/"exclude"），让报错能指出是谁写错了。
    """
    if element in BUILTIN_TOOL_GROUPS:
        return element, None
    if element in _TOOL_TO_GROUP:
        return _TOOL_TO_GROUP[element], element
    raise ConfigError(
        f"unknown {what} entry {element!r}; expected a group name "
        f"({', '.join(BUILTIN_TOOL_GROUPS)}) or a builtin tool name "
        f"({', '.join(BUILTIN_TOOL_NAMES)})"
    )


def _resolve_sandbox_root(sandbox_root: str | None) -> str | None:
    """沙箱根解析：参数 > ``LITEAGENT_SANDBOX_ROOT`` > ``os.getcwd()``。"""
    if sandbox_root:
        return sandbox_root
    if sandbox_root is not None:
        # 显式传了空串：这是"我不想给沙箱根"的明确表达，不要用 cwd 悄悄补上。
        return None
    from_env = os.environ.get("LITEAGENT_SANDBOX_ROOT")
    if from_env:
        return from_env
    return os.getcwd()


def register_all(
    registry: "ToolRegistry",
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    sandbox_root: str | None = None,
    allow_shell: bool | None = None,
    allow_network: bool = True,
    memory: "MemoryManager | None" = None,
    search_backend: SearchBackend | None = None,
    web_transport: "Transport | None" = None,
) -> "ToolRegistry":
    """把内置工具注册进 ``registry`` 并返回它（便于链式调用）。

    语义见模块 docstring。本函数是唯一有"批量副作用"的地方，且必须显式调用。
    """
    if not hasattr(registry, "register"):
        raise ConfigError(
            f"register_all(registry) expects a ToolRegistry, got {type(registry).__name__}"
        )

    # ---- 1) include 展开成 {组名: 想要的工具名集合 | None（整组）} ----
    wanted: dict[str, set[str] | None] = {}
    if include is None:
        for group in BUILTIN_TOOL_GROUPS:
            if group == "memory" and memory is None:
                # 没有 MemoryManager 就没有 memory 组可言，这不是错误（§7.5）。
                continue
            wanted[group] = None
    else:
        for element in include:
            group, tool_name = _resolve_element(str(element), what="include")
            if tool_name is None:
                # 整组：覆盖掉之前可能记下的"只要其中一个工具"。
                wanted[group] = None
            elif group not in wanted:
                wanted[group] = {tool_name}
            elif wanted[group] is not None:
                wanted[group].add(tool_name)  # type: ignore[union-attr]

    # ---- 2) exclude 先解析（未命中即报错），再在展开结果上做减法 ----
    excluded_groups: set[str] = set()
    excluded_tools: set[str] = set()
    for element in exclude or ():
        group, tool_name = _resolve_element(str(element), what="exclude")
        if tool_name is None:
            excluded_groups.add(group)
        else:
            excluded_tools.add(tool_name)

    # ---- 3) 只构造真正需要的组（沙箱等参数因此按需解析） ----
    active = {group: names for group, names in wanted.items() if group not in excluded_groups}
    needs_files_or_code = bool({"files", "code"} & set(active))
    sandbox: PathSandbox | None = None
    if needs_files_or_code:
        resolved_root = _resolve_sandbox_root(sandbox_root)
        if not resolved_root:
            raise ConfigError(
                "the files/code tool groups require a sandbox root; pass sandbox_root=..., "
                "set LITEAGENT_SANDBOX_ROOT, or leave sandbox_root=None to use os.getcwd()"
            )
        sandbox = PathSandbox(resolved_root)

    registered: list[str] = []
    for group, names in active.items():
        if group == "memory" and memory is None:
            # 显式 include=["memory"] 但没给 memory 实例：注册 0 个工具 + 留痕（红线 12）。
            warnings.warn(
                "the 'memory' tool group was requested but memory=None; "
                "no remember/recall tools were registered",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        built = _build_group(
            group,
            sandbox=sandbox,
            allow_shell=allow_shell,
            allow_network=allow_network,
            memory=memory,
            search_backend=search_backend,
            web_transport=web_transport,
        )
        for tool in built:
            if names is not None and tool.name not in names:
                continue
            if tool.name in excluded_tools:
                continue
            registry.register(tool, override=False)
            registered.append(tool.name)
    return registry


def _build_group(
    group: str,
    *,
    sandbox: PathSandbox | None,
    allow_shell: bool | None,
    allow_network: bool,
    memory: "MemoryManager | None",
    search_backend: SearchBackend | None,
    web_transport: "Transport | None",
) -> list[Tool]:
    """按组名调用对应的 factory（参数传递集中在一处，便于对照 §7.5 的签名）。"""
    if group == "files":
        if sandbox is None:
            raise ConfigError("the 'files' tool group needs a sandbox root")
        return make_file_tools(sandbox)
    if group == "shell":
        return make_shell_tools(allow_shell, sandbox)
    if group == "code":
        return make_code_tools(sandbox)
    if group == "web":
        return make_web_tools(
            search_backend, allow_network=allow_network, transport=web_transport
        )
    if group == "memory":
        if memory is None:
            raise ConfigError("the 'memory' tool group needs a MemoryManager instance")
        return make_memory_tools(memory)
    raise ConfigError(f"unknown builtin tool group {group!r}")  # pragma: no cover
