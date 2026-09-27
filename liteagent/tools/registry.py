from __future__ import annotations

# liteagent/tools/registry.py —— 工具的注册表（规范 §7.3）
#
# 这个模块是"工具系统"的目录服务：名字 -> Tool 的唯一映射，以及由它派生的四种视图
# （给模型的 schema、给文本 ReAct 的摘要行、给 CLI 的详情、给测试的子集）。
#
# 三个设计点，面试常问：
#
#   1. **重名不覆盖，只报错**（`register(..., override=False)`）。工具名是模型与框架
#      之间的契约，同一个名字对应两个实现会让"模型调的是哪个 read_file"取决于
#      import 顺序 —— 这类 bug 在生产里几乎无法复现。
#
#   2. **别名不参与 `names()`**。`names()` 是"这个注册表里真正有几个工具"，
#      别名只是查找时的便利路径（`read` / `cat` -> `read_file`）。把别名混进
#      `names()` 会让 schema 导出多出一批指向同一个函数的重复定义，
#      模型会在两个等价工具之间随机选择。
#
#   3. **所有查询返回副本/不可变视图**（红线 11）：`names()` 返回新 list，
#      `list()` / `schemas()` 返回新 list，`to_prompt()` 返回新 str。
#      调用方改返回值不可能改到注册表内部状态。
#
# 默认注册表（`get_default_registry`）只为 `@tool(auto_register=True)` 服务。
# 它是**进程级全局可变状态**，所以配了 `reset_default_registry()` 让测试能隔离 ——
# 后者替换为新对象而不是清空原对象（§7.3 冻结语义），这样任何"早先拿到的旧引用"
# 不会突然变成空表。
import json
import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Callable

from liteagent.config import DEFAULT_MAX_TOOLS_IN_PROMPT
from liteagent.errors import ConfigError, ToolDefinitionError, ToolNotFoundError
from liteagent.tools.base import Tool, is_tool

__all__ = ["ToolRegistry", "get_default_registry", "reset_default_registry"]

# 工具名的合法模式（§7.3 逐字冻结）。用 `fullmatch` 而不是 `match`：
# `$` 允许尾随换行，`match` 会让 "read_file\n" 通过，而它显然不是合法工具名。
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


class ToolRegistry:
    """名字 -> `Tool` 的映射，外加别名、子集与多种导出视图。

    线程安全说明：注册表本身**不加锁**。它的可变阶段（构造 / `register_all`）
    发生在 Agent 构建期、单线程内；进入 ReAct 循环后只剩读操作，而 dict 的读
    在 CPython 下是安全的。给它加锁会让"每个工具调用都要过一把锁"成为常态开销，
    换来的却是一个现实中不存在的场景。
    """

    def __init__(
        self,
        tools: Iterable[Tool] | None = None,
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        # 用普通 dict 而不是 OrderedDict：3.7+ 的 dict 保序，且 `names()` 会排序，
        # 于是"注册顺序"只在 `to_dict()` 的 tools 列表里可见 —— 那也正是我们想要的稳定顺序。
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}
        if tools is not None:
            for item in tools:
                self.register(item)
        if aliases:
            # 别名必须在工具注册**之后**处理：`alias()` 会校验目标存在（早失败），
            # 否则 `ToolRegistry(aliases={"rf": "read_file"})` 会因为目标还没注册而炸。
            for alias_name, target in aliases.items():
                self.alias(alias_name, target)

    # --------------------------------------------------------------------------
    # 注册 / 注销
    # --------------------------------------------------------------------------

    def register(self, tool: Tool, *, override: bool = False) -> Tool:
        """注册一个工具并返回它（便于链式调用）。

        - 重名且 `override=False` -> `ToolDefinitionError`。
        - 名字不匹配 `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$` -> `ToolDefinitionError`。
        - 名字与**已有别名**撞车 -> `ToolDefinitionError`（对称地不允许静默遮蔽：
          若放任，`reg.get(name)` 走工具分支、`reg.names()` 里却没有它，
          两个视图对不上）。`override=True` 时让位给工具并丢弃该别名。
        """
        if not is_tool(tool):
            # 传进来一个裸函数是最常见的手滑。报错文案直接指路，
            # 而不是抛一个 "AttributeError: 'function' object has no attribute 'name'"。
            raise ToolDefinitionError(
                tool_name=getattr(tool, "name", "") or type(tool).__name__,
                message=(
                    f"register() expects a Tool, got {type(tool).__name__}; "
                    "use register_function(func, **kwargs) for plain functions"
                ),
            )
        name = tool.name
        if _NAME_PATTERN.fullmatch(name) is None:
            raise ToolDefinitionError(
                tool_name=name,
                message=(
                    f"invalid tool name {name!r}; it must match "
                    f"{_NAME_PATTERN.pattern}"
                ),
            )
        if name in self._tools and not override:
            # 不做"同一个对象就放行"的例外：`ToolRegistry([t, t])` 与
            # `reg.register(reg.get(n))` 都是**调用方的真 bug**（多为重复展开的工具列表），
            # 放行会把它们藏起来；而规范只给了 `override` 这一个开关（§7.3）。
            raise ToolDefinitionError(
                tool_name=name,
                message=(
                    f"tool {name!r} is already registered; "
                    "pass override=True to replace it"
                ),
            )
        if name in self._aliases:
            if not override:
                raise ToolDefinitionError(
                    tool_name=name,
                    message=(
                        f"tool name {name!r} collides with an existing alias; "
                        "pass override=True to drop the alias"
                    ),
                )
            del self._aliases[name]
        self._tools[name] = tool
        return tool

    def register_function(self, func: Callable[..., Any], **kwargs: Any) -> Tool:
        """便捷：等价 `register(Tool.from_function(func, **kwargs))`。"""
        return self.register(Tool.from_function(func, **kwargs))

    def unregister(self, name: str) -> None:
        """移除工具；不存在 -> `ToolNotFoundError`。

        同时清理**指向它**的别名：留下悬空别名会让 `get(alias)` 报
        "tool not found: alias"（而真正的错因是"目标没了"），错误信息指向错误的方向。
        """
        if name not in self._tools:
            raise ToolNotFoundError(name, available=self.names())
        del self._tools[name]
        for alias_name, target in list(self._aliases.items()):
            if target == name or alias_name == name:
                del self._aliases[alias_name]

    # --------------------------------------------------------------------------
    # 查询
    # --------------------------------------------------------------------------

    def get(self, name: str) -> Tool:
        """按名（或别名）取工具；未命中 -> `ToolNotFoundError(available=...)`。

        `available` 必须带上：executor 会把它渲染回灌给模型（§7.4.1 步骤 1），
        模型才知道"正确名字是什么"，这是自纠正能力的物质基础。
        """
        tool = self.try_get(name)
        if tool is None:
            raise ToolNotFoundError(name, available=self.names())
        return tool

    def try_get(self, name: str) -> Tool | None:
        """别名解析唯一发生的地方之一（另一个是 `get`）。未命中返回 `None`，不抛。"""
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        target = self._aliases.get(name)
        if target is None:
            return None
        return self._tools.get(target)

    def alias(self, alias: str, name: str) -> None:
        """登记一条别名；`name` 不存在 -> `ToolNotFoundError`。

        别名解析在 `get`/`try_get` 里做，**别名不进 `names()`**（§7.3 冻结）。

        `name` 本身是别名时按"跟随一跳"处理并**存成最终目标**：存成链会在
        `unregister` 时留下指向中间节点的悬空别名，而链式解析又要求
        `get` 里写循环（还得处理环）。扁平化一步到位，没有这两种问题。
        """
        if _NAME_PATTERN.fullmatch(alias) is None:
            raise ToolDefinitionError(
                tool_name=alias,
                message=(
                    f"invalid alias {alias!r}; it must match {_NAME_PATTERN.pattern}"
                ),
            )
        if alias in self._tools:
            raise ToolDefinitionError(
                tool_name=alias,
                message=(
                    f"alias {alias!r} collides with a registered tool name; "
                    "aliases must not shadow real tools"
                ),
            )
        target = self._aliases.get(name, name)
        if target not in self._tools:
            raise ToolNotFoundError(name, available=self.names())
        self._aliases[alias] = target

    def names(self) -> list[str]:
        """全部**真实**工具名，排序；不含别名。返回新 list（红线 11）。"""
        return sorted(self._tools)

    def list(
        self,
        *,
        tags: Sequence[str] | None = None,
        include_dangerous: bool = True,
    ) -> list[Tool]:
        """按 `names()` 顺序返回工具列表，可选按 tags / dangerous 过滤。

        `tags` 的语义冻结为**任一命中**（交集非空即保留）：它是给 CLI `tools list --tags`
        的粗筛，用户传 `--tags files,shell` 时期望看到两组，而不是"同时打了两个标签"的
        （通常为空）集合。空 `tags`/`None` = 不过滤。
        """
        wanted = {t for t in tags} if tags else None
        out: list[Tool] = []
        for tool in self:
            if not include_dangerous and tool.spec.dangerous:
                continue
            if wanted is not None and not wanted.intersection(tool.spec.tags):
                continue
            out.append(tool)
        return out

    def subset(self, names: Sequence[str]) -> "ToolRegistry":
        """只含指定工具的新注册表；未命中任一名字 -> `ToolNotFoundError`（早失败）。

        别名会被解析（走 `get`），重复项按**解析后的真名**去重 ——
        `subset(["read_file", "read_file"])` 与 CLI 的 `--tools read_file,read_file`
        都不该因为用户多打一遍就报重名。
        """
        out = ToolRegistry()
        seen: set[str] = set()
        for name in names:
            tool = self.get(name)  # 未命中在这里就抛，不做"部分成功"
            if tool.name in seen:
                continue
            seen.add(tool.name)
            out.register(tool)
        return out

    # --------------------------------------------------------------------------
    # 导出视图
    # --------------------------------------------------------------------------

    def schemas(self, *, fmt: str = "openai") -> list[dict[str, Any]]:
        """导出给 LLM provider 的工具定义列表。`fmt`: `"openai"` | `"anthropic"`。

        未知 fmt -> `ConfigError`（**不是**静默回退到 openai：把 anthropic 格式
        当 openai 发出去，provider 会在请求序列化阶段报一个与根因无关的错）。
        """
        if fmt == "openai":
            return [t.to_openai_schema() for t in self]
        if fmt == "anthropic":
            return [t.to_anthropic_schema() for t in self]
        raise ConfigError(
            message=f"unknown schema format {fmt!r}; available: ['anthropic', 'openai']"
        )

    def to_prompt(
        self,
        *,
        fmt: str = "text",
        max_tools: int = DEFAULT_MAX_TOOLS_IN_PROMPT,
    ) -> str:
        """渲染成可直接塞进 system prompt 的字符串（文本 ReAct 模式用）。

        [v2 变更] `max_tools` 默认 20（`DEFAULT_MAX_TOOLS_IN_PROMPT`，v1 是 64）：
        文本模式下这些摘要行**逐字**进 system prompt 的 token 预算。

        - `fmt="text"` -> 每行 `spec.summary_line()`。
        - `fmt="json"` -> 缩进 JSON 数组（每个元素是 `Tool.to_dict()`）。
        - 超过 `max_tools` 时截断并追加 `'... (N more tools omitted)'`。
          **json 分支把这条提示作为数组的最后一个字符串元素**：直接拼在 JSON 后面
          会产出非法 JSON（CLI 的 `--json` 输出要能被 `jq` 吃），
          而静默丢工具违反红线 12。合法的混合类型数组是唯一同时满足两者的形态。
        """
        if fmt not in ("text", "json"):
            raise ConfigError(
                message=f"unknown prompt format {fmt!r}; available: ['json', 'text']"
            )
        tools = self.list()
        if max_tools < 0:
            # 负数在 v1 里会被切片悄悄变成"砍掉尾部"，静默地少给工具。这里明确拒绝。
            raise ConfigError(
                message=f"max_tools must be >= 0, got {max_tools}"
            )
        omitted = len(tools) - max_tools
        shown = tools[:max_tools]

        if fmt == "text":
            lines = [t.spec.summary_line() for t in shown]
            if omitted > 0:
                lines.append(f"... ({omitted} more tools omitted)")
            return "\n".join(lines)

        payload: list[Any] = [t.to_dict() for t in shown]
        if omitted > 0:
            payload.append(f"... ({omitted} more tools omitted)")
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def describe(self, name: str, *, fmt: str = "markdown") -> str:
        """单工具详情（CLI `tools show` 用），含完整参数表。

        `fmt`: `"markdown"` | `"json"`。未知 -> `ConfigError`。
        """
        tool = self.get(name)
        if fmt == "json":
            return json.dumps(tool.to_dict(), ensure_ascii=False, indent=2)
        if fmt != "markdown":
            raise ConfigError(
                message=f"unknown describe format {fmt!r}; available: ['json', 'markdown']"
            )

        spec = tool.spec
        lines: list[str] = [f"### {spec.name}", "", spec.description or "(no description)", ""]
        lines.append(f"- tags: {', '.join(spec.tags) if spec.tags else '-'}")
        lines.append(f"- dangerous: {spec.dangerous}")
        lines.append(f"- requires_approval: {spec.requires_approval}")
        lines.append(f"- idempotent: {spec.idempotent}")
        lines.append(f"- timeout_s: {spec.timeout_s}")
        lines.append(f"- max_retries: {spec.max_retries}")
        lines.append(f"- version: {spec.version}")
        lines.append(f"- is_async: {spec.is_async}")
        lines.append("")

        # 参数表：required 判定只读 schema 的 "required" 数组，不重复推断
        # （重新推断一遍 required 就是"两处语义"，迟早漂移）。
        properties = spec.parameters.get("properties") if isinstance(spec.parameters, Mapping) else None
        required = set(spec.parameters.get("required") or ()) if isinstance(spec.parameters, Mapping) else set()
        if isinstance(properties, Mapping) and properties:
            lines.append("| parameter | type | required | default | description |")
            lines.append("|---|---|---|---|---|")
            for pname, fragment in properties.items():
                frag = fragment if isinstance(fragment, Mapping) else {}
                ptype = frag.get("type", "any")
                default = frag["default"] if "default" in frag else "-"
                lines.append(
                    "| {name} | {type} | {req} | {default} | {desc} |".format(
                        name=pname,
                        type=_escape_cell(str(ptype)),
                        req="yes" if pname in required else "no",
                        default=_escape_cell(_compact(default)),
                        desc=_escape_cell(_compact(frag.get("description", ""))),
                    )
                )
        else:
            lines.append("(no parameters)")
        lines.append("")

        if spec.warnings:
            # 红线 12：降级痕迹必须可观测 —— CLI 的 `tools show` 是最容易被看到的落点。
            lines.append("#### warnings")
            lines.append("")
            for item in spec.warnings:
                lines.append(f"- {item}")
            lines.append("")

        lines.append("```json")
        lines.append(json.dumps(spec.parameters, ensure_ascii=False, indent=2))
        lines.append("```")
        return "\n".join(lines)

    def merge(self, other: "ToolRegistry", *, override: bool = False) -> "ToolRegistry":
        """返回**新** registry（不改自身）。重名且 `override=False` -> `ToolDefinitionError`。

        只合并**工具**，不合并别名：别名是"某个注册表的本地习惯"，
        跨表合并时两个表对同名别名指向不同工具的概率不低，
        静默择一会制造"同一个别名在不同 Agent 里解析到不同工具"的幽灵 bug。
        """
        merged = ToolRegistry(self._tools.values())
        for name in other.names():
            merged.register(other.get(name), override=override)
        return merged

    # --------------------------------------------------------------------------
    # 容器协议
    # --------------------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        """与 `try_get` 同语义（**含别名**）：`__contains__` 是"能不能取到"的语法糖，
        与 `get` 用不同的判定会让人在 `in` 上产生错误的直觉。"""
        if not isinstance(name, str):
            return False
        return self.try_get(name) is not None

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        """遍历 Tool，按 `names()` 顺序（排序 -> 稳定，schema 导出不随注册顺序抖动）。"""
        for name in self.names():
            yield self._tools[name]

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化快照；字段全量输出（§2.2）。"""
        return {
            "names": self.names(),
            "count": len(self._tools),
            "aliases": dict(self._aliases),
            "tools": [t.to_dict() for t in self],
        }

    def __repr__(self) -> str:
        aliases = f", aliases={len(self._aliases)}" if self._aliases else ""
        return f"<ToolRegistry tools={len(self._tools)}{aliases}>"


def _escape_cell(text: str) -> str:
    """markdown 表格单元格转义：`|` 会截断列，换行会让表格散架。"""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _compact(value: Any) -> str:
    """把任意值压成一行短文本（表格单元格用）。

    字符串原样返回（不套 JSON 引号，表格里更好读）；其它类型走 JSON；
    不可序列化时退到 `repr` 并保持非空 —— 表格单元格为空会让人以为"没有默认值"。
    """
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - 不可序列化时退到 repr
        return repr(value)


# --------------------------------------------------------------------------------------
# 默认注册表（仅 auto_register=True 使用）
# --------------------------------------------------------------------------------------

_default_registry: ToolRegistry | None = None
# 懒创建 + reset 都在一个进程内被多线程调用（并行测试 / 多 Agent 构建期）。
# 用 threading 而不是 asyncio 原语：R-LOOP（§0.4）只禁 asyncio，
# 而且这两行代码可能在**没有事件循环**的 worker 线程里跑。
_default_lock = threading.Lock()


def get_default_registry() -> ToolRegistry:
    """模块级全局注册表（仅 `auto_register=True` 使用；测试用 `reset_default_registry()` 清理）。"""
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            # 双重检查：拿锁之前可能已经有别的线程建好了。
            # 少了这一层，两个线程会各造一个注册表，其中一个的工具从此消失。
            if _default_registry is None:
                _default_registry = ToolRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """[v2 变更] 冻结语义：把全局注册表**替换为一个全新的空 ToolRegistry**（而不是清空原对象），
    保证任何持有旧引用的代码不会看到"被清空"的状态。测试的 `tearDown` 必须调用它。

    为什么是"替换"而不是 `.clear()`：Agent 构建期常常把 registry 存进自己的字段，
    清空会让那些仍活着的 Agent 突然"工具全没了"，而这跟它们自己的生命周期毫无关系。
    """
    global _default_registry
    with _default_lock:
        _default_registry = ToolRegistry()
