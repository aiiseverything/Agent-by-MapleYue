from __future__ import annotations

# liteagent/tools/base.py —— 工具的定义与装饰器（规范 §7.2）
#
# 这个模块回答三个问题，也是面试里最值得讲的三点：
#
#   1. **"装饰器如何自动生成 JSON Schema"**：`@tool` 在**装饰期**一次性完成反射
#      （签名 + 注解 + docstring），产物是不可变的 `ToolSpec`。运行期不再反射 —— 反射很贵，
#      而 ReAct 循环每轮都要把全部 schema 塞进请求体。反射逻辑本身在 `tools/schema.py`
#      （E4 同层边），这里只负责"把反射结果装进 spec 并把它包装成可调用对象"。
#
#   2. **"装饰器为什么默认不注册"**：`auto_register=False` 是默认值，`import` 一个模块
#      **不产生任何全局副作用**。否则两个测试文件 import 同一个模块就会在第 2 次
#      import 时撞重名，测试顺序一换就红 —— 这是"测试可重复"的地基。
#
#   3. **"同步工具超时到底能不能取消"**：不能。`asyncio.wait_for` 只能取消 await 层，
#      跑在 worker 线程里的同步代码不可中断（D-09）。框架不假装它停了，而是提供一条
#      **协作式取消通道**（`cancel_scope` / `current_cancel_flag`）：executor 在超时时
#      `flag.set()`，长耗时的同步工具在自己的循环里主动查这个 flag 并提前返回。
#      为什么用 contextvars 而不是线程局部量：contextvar 的**读**能穿透
#      `run_in_executor` 到达 worker 线程（M-5 实测），于是 worker 里的工具函数
#      不需要任何参数就能拿到"我自己这次调用"的取消信号。
#
# 红线（§13）：三方库零顶层 import（本文件连可选的都没有）；一切降级必须留痕 ——
# 这里统一走 `ToolSpec.warnings`（结构化、可断言、能进 trace），不往 stderr 打警告；
# 不新增异常类型（只有 errors.py 里已定义的那些）。
import contextvars
import dataclasses
import inspect
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, overload

from liteagent.config import to_jsonable
from liteagent.errors import ToolDefinitionError
from liteagent.tools.schema import (
    SchemaResult,
    build_tool_schema,
    to_anthropic_tool,
    to_openai_tool,
)

__all__ = [
    "ToolSpec",
    "Tool",
    "tool",
    "make_function_tool",
    "is_tool",
    "current_cancel_flag",
    "cancel_scope",
    "CANCEL_FLAG_VAR_NAME",
]

# `ToolSpec.pass_style` 的两个合法取值（§7.2）。写成常量而不是散落的字符串字面量，
# 这样 `Tool._dispatch` 与 `make_function_tool` 不会漂移。
PASS_STYLE_KWARGS = "kwargs"
PASS_STYLE_MAPPING = "mapping"

# `Tool.__getattr__` 允许代理到原始函数上的属性白名单。
# 只代理这些"描述性"属性，不做无差别转发：无差别转发会让 `hasattr(tool, "anything")`
# 恒为 True，把拼写错误（`tool.spec_`）变成运行时才炸的隐蔽 bug。
_FUNC_PROXY_ATTRS = frozenset(
    {
        "__name__",
        "__qualname__",
        "__doc__",
        "__module__",
        "__wrapped__",
        "__annotations__",
        "__dict__",
    }
)

# `Tool.from_function` / `make_function_tool` 接受的 ToolSpec 字段名。
# 用一个 frozenset 做白名单：传错 kwarg（如 `require_approval=` 少了 s）必须
# **当场抛 ToolDefinitionError**，而不是被 `**kwargs` 悄悄吞掉。
_SPEC_FIELD_NAMES = frozenset(
    {
        "name",
        "description",
        "parameters",
        "is_async",
        "pass_style",
        "tags",
        "dangerous",
        "requires_approval",
        "idempotent",
        "timeout_s",
        "max_retries",
        "version",
    }
)


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的**不可变**定义（装饰期产出，运行期只读）。

    为什么 frozen：`ToolSpec` 会被多个 Agent / 多个线程共享（注册表是全局单例，
    多 Agent 协作时同一个 spec 会被并发读取）。可变定义 + 并发读取 = 最难查的一类 bug。
    """

    name: str
    description: str
    parameters: dict[str, Any]              # 参数 JSON Schema
    func: Callable[..., Any]
    is_async: bool = False
    pass_style: str = PASS_STYLE_KWARGS     # "kwargs" -> func(**args) ; "mapping" -> func(args)
    tags: tuple[str, ...] = ()
    dangerous: bool = False                 # 只影响**展示过滤**（list(include_dangerous=)）
    requires_approval: bool = False         # [v2 新增] 影响**控制流**（§7.4.1 步骤 4.5）
    idempotent: bool = True
    timeout_s: float | None = None          # None -> 继承 ExecutorConfig.default_timeout_s
                                            # NO_TIMEOUT -> 显式禁用超时
    max_retries: int | None = None          # None -> 用 ExecutorConfig.retry_policy.max_retries
    warnings: tuple[str, ...] = ()
    version: str = "1"

    # ---- schema 导出：直接转交给 tools/schema.py，避免两处实现漂移（E4）----

    def to_openai_schema(self) -> dict[str, Any]:
        """-> `{"type": "function", "function": {...}}`（§7.1 冻结形态）。"""
        return to_openai_tool(self)

    def to_anthropic_schema(self) -> dict[str, Any]:
        """-> `{"name", "description", "input_schema"}`（§7.1 冻结形态）。"""
        return to_anthropic_tool(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的全量快照（§2.2：字段全量输出，含 `None`）。

        **不含 `func`** —— 函数对象不可序列化，而 trace / CLI / 测试要的是"描述"。
        额外带一个 `"signature"`（人类可读的 `(a: int, b: str)`），它是排查
        "模型为什么传错参数"时最先看的东西。

        最后过一遍 `config.to_jsonable`：`parameters` 里的 `default` 可能是任意对象
        （例如用户写了 `Param(default=Path("x"))`），`to_dict` 是 trace 路径，
        不允许因为一个字段类型不合法而让整条 trace 丢帧。
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "is_async": self.is_async,
            "pass_style": self.pass_style,
            "tags": list(self.tags),
            "dangerous": self.dangerous,
            "requires_approval": self.requires_approval,
            "idempotent": self.idempotent,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "warnings": list(self.warnings),
            "version": self.version,
            "signature": self._signature_text(),
        }
        return to_jsonable(payload)

    def summary_line(self) -> str:
        """`'name(a: integer, b: string) - description 首行'`，给文本 ReAct 的 `{tools}` 用。

        为什么只要首行：文本模式下这些行会**逐字**进 system prompt 的 token 预算
        （§7.3 的 `DEFAULT_MAX_TOOLS_IN_PROMPT` 从 64 降到 20 就是为此）。
        """
        rendered = _render_parameters(self.parameters)
        head = self.description.strip().splitlines()
        first = head[0].strip() if head else ""
        return f"{self.name}({rendered}) - {first}" if first else f"{self.name}({rendered})"

    def _signature_text(self) -> str:
        """`"(a: int, b: str) -> int"`；不可内省时降级为占位串。

        降级**不抛异常**：`to_dict()` 跑在 trace 输出路径上，为一个不可内省的内置
        可调用对象（C 扩展实现）丢掉整条 trace 是明显更坏的选择。占位串本身
        就是可观测痕迹（红线 12）。
        """
        try:
            return str(inspect.signature(self.func))
        except (TypeError, ValueError):  # pragma: no cover - 依赖具体可调用对象
            return f"<signature unavailable for {self.name}>"


class Tool:
    """装饰器产物。可调用、带 spec、可被注册。"""

    # `__slots__` 不启用：spec 是唯一状态，但 `__slots__` 会挡掉
    # `Tool.__getattr__` 之外的一些惯用用法（例如测试给实例挂临时属性），收益不值。
    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    # ---- 只读代理：让 `tool.name` 这类写法可用（注册表与 executor 大量使用）----

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self.spec.parameters

    @property
    def is_async(self) -> bool:
        return self.spec.is_async

    @property
    def raw(self) -> Callable[..., Any]:
        """原始函数（测试常用）。"""
        return self.spec.func

    # ---- 调用路径 ----

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """直接调用原始函数（不做校验）。"""
        return self.spec.func(*args, **kwargs)

    def run(self, args: Mapping[str, Any]) -> Any:
        """[v2 新增] **同步**调用原始函数（按 pass_style）。

        - `is_async` 为 True 时抛 `TypeError("tool <name> is async; use arun()")`。
          这条不是洁癖：executor 把同步工具丢进 `run_in_executor`，若误把异步工具
          也丢进去（M-2 实测），`to_thread` 只会返回一个**没人 await 的协程对象**，
          调用方拿到的是 coroutine 而不是结果，而且没有任何异常 —— 静默失效。
        - `pass_style == "mapping"` -> `func(dict(args))`
        - `pass_style == "kwargs"`  -> `func(**dict(args))`

        同步工具在 executor 里**只走这一条路径**（§7.4.1 步骤 5.b）。
        """
        if self.spec.is_async:
            raise TypeError(f"tool {self.spec.name} is async; use arun()")
        return self._dispatch(dict(args))

    async def arun(self, args: Mapping[str, Any]) -> Any:
        """按 pass_style 调用原始函数；`is_async` 时 await。

        `is_async` 为 False 时**同步调用**原始函数（等价 `self.run(args)`）。
        **不做 schema 校验** —— 校验只在 `ToolExecutor` 里做一次，
        避免"装饰器校验一遍、执行器再校验一遍"的两处语义漂移。
        """
        data = dict(args)
        if self.spec.is_async:
            return await self._dispatch(data)
        # 同步分支刻意**不**走 `run`：`run` 会再查一次 is_async，而我们已经知道是 False。
        # 但语义完全一致（同步调用），走 run 的唯一好处是异常文案统一。
        return self.run(data)

    def _dispatch(self, data: dict[str, Any]) -> Any:
        """按 pass_style 分派；**不**做 await / 校验。`run` 与 `arun` 共用这一处。"""
        style = self.spec.pass_style
        if style == PASS_STYLE_MAPPING:
            return self.spec.func(data)
        if style == PASS_STYLE_KWARGS:
            return self.spec.func(**data)
        # 未知 pass_style 是定义期错误：静默按 kwargs 处理会让 mapping 风格的工具
        # 收到一堆意外的关键字参数，报错信息离根因十万八千里。
        raise ToolDefinitionError(
            tool_name=self.spec.name,
            message=(
                f"unknown pass_style {style!r} for tool {self.spec.name!r}; "
                f"expected {PASS_STYLE_KWARGS!r} or {PASS_STYLE_MAPPING!r}"
            ),
        )

    # ---- 导出 ----

    def to_openai_schema(self) -> dict[str, Any]:
        return self.spec.to_openai_schema()

    def to_anthropic_schema(self) -> dict[str, Any]:
        return self.spec.to_anthropic_schema()

    def to_dict(self) -> dict[str, Any]:
        return self.spec.to_dict()

    def __repr__(self) -> str:
        return f"<Tool {self.spec.name}({_render_parameters(self.spec.parameters)}) at 0x{id(self):x}>"

    def __getattr__(self, item: str) -> Any:
        """把少量描述性属性代理到原始函数（`tool.__name__` / `tool.__doc__`）。

        只代理 `_FUNC_PROXY_ATTRS` 白名单，理由见该常量的注释。
        """
        if item in _FUNC_PROXY_ATTRS:
            return getattr(self.spec.func, item)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}"
        )

    @classmethod
    def from_function(cls, func: Callable[..., Any], **kwargs: Any) -> "Tool":
        """从普通函数构建 `Tool`（`@tool` 的真正实现体）。

        对已经是 `Tool` 的入参**幂等返回本身**：`@tool` 套两次、或注册表里
        再装饰一遍时不应该产生一个"包装的包装"（那会让 `tool.raw` 指向另一个 Tool，
        递归展开就成了谁的锅说不清的问题）。
        """
        if is_tool(func):
            return func
        return cls(_build_spec(func, **kwargs))


def _build_spec(func: Callable[..., Any], **kwargs: Any) -> ToolSpec:
    """把 `func` + 装饰器参数反射成 `ToolSpec`（`Tool.from_function` 的内核）。

    反射本身全权委托 `tools/schema.py`（E4）：这里只做三件事 ——
    拆参数、判定 `is_async`、把 schema 层的 warnings 收进 spec。
    """
    docstring_style = kwargs.pop("docstring_style", "auto")
    unknown = set(kwargs) - _SPEC_FIELD_NAMES
    if unknown:
        raise ToolDefinitionError(
            tool_name=str(kwargs.get("name") or getattr(func, "__name__", "")),
            message=(
                f"unknown tool option(s) {sorted(unknown)}; "
                f"supported: {sorted(_SPEC_FIELD_NAMES)} + {{'docstring_style'}}"
            ),
        )
    if not callable(func):
        raise ToolDefinitionError(
            tool_name=str(kwargs.get("name") or ""),
            message=(
                f"@tool expects a callable, got {type(func).__name__}. "
                "If you meant to configure the tool, write @tool(...) (with parentheses)."
            ),
        )

    result: SchemaResult = build_tool_schema(
        func,
        name=kwargs.get("name"),
        description=kwargs.get("description"),
        parameters=kwargs.get("parameters"),
        docstring_style=docstring_style,
    )
    warnings_list = list(result.warnings)

    requested_is_async = kwargs.get("is_async")
    detected_is_async = inspect.iscoroutinefunction(func)
    if requested_is_async is None:
        is_async = detected_is_async
    else:
        is_async = bool(requested_is_async)
        if is_async != detected_is_async:
            # 显式声明与实际不符：多半是包装函数（functools.partial / 装饰器把
            # async 函数包成同步函数）造成的假象。留痕而不是覆盖用户的判断 ——
            # 用户可能真的知道自己在干什么。
            warnings_list.append(
                f"is_async={is_async} disagrees with "
                f"inspect.iscoroutinefunction(func)={detected_is_async}; "
                f"using the explicit value"
            )

    return ToolSpec(
        name=kwargs.get("name") or result.name,
        description=kwargs.get("description") or result.description,
        parameters=kwargs.get("parameters") or result.schema,
        func=func,
        is_async=is_async,
        pass_style=kwargs.get("pass_style", PASS_STYLE_KWARGS),
        tags=tuple(kwargs.get("tags") or ()),
        dangerous=bool(kwargs.get("dangerous", False)),
        requires_approval=bool(kwargs.get("requires_approval", False)),
        idempotent=bool(kwargs.get("idempotent", True)),
        timeout_s=kwargs.get("timeout_s"),
        max_retries=kwargs.get("max_retries"),
        warnings=tuple(warnings_list),
        version=str(kwargs.get("version", "1")),
    )


def _render_parameters(parameters: Mapping[str, Any] | None) -> str:
    """把 JSON Schema 的 `properties` 渲染成 `"a: integer, b: string"`。

    只做展示（`summary_line` / `__repr__`），**不解析**任何 JSON Schema 关键字 ——
    于是它不可能与真正的校验器漂移。没有 `type` 键的参数（`Any` / 混合 `Literal`）
    显示为 `any`，这是诚实的：`{}` 就是"任意类型"。
    """
    if not isinstance(parameters, Mapping):
        return ""
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return ""
    parts: list[str] = []
    for pname, fragment in properties.items():
        label = "any"
        if isinstance(fragment, Mapping):
            declared = fragment.get("type")
            if isinstance(declared, str):
                label = declared
        parts.append(f"{pname}: {label}")
    return ", ".join(parts)


@overload
def tool(func: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    parameters: dict[str, Any] | None = ...,
    tags: Sequence[str] = ...,
    dangerous: bool = ...,
    requires_approval: bool = ...,
    idempotent: bool = ...,
    timeout_s: float | None = ...,
    max_retries: int | None = ...,
    auto_register: bool = ...,
    docstring_style: str = ...,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(  # type: ignore[misc]  # 与上面两个 overload 一起构成冻结签名（§7.2）
    func: Callable[..., Any] | None = None, **kwargs: Any
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """同时支持 `@tool` 与 `@tool(...)` 两种用法。

    返回的 `Tool` 对象**不清除原函数**（`Tool.spec.func` / `Tool.raw` 都还在），
    因为测试与调试经常需要"拿到没有 schema 反射的那一层"。

    [v2 变更] `auto_register` 的语义**写死**：
      - `auto_register=True` 等价于 `get_default_registry().register(t, override=False)`；
        重名 -> `ToolDefinitionError`（**不静默覆盖** —— 覆盖会让"我装饰了同名工具但
        生效的是别人那个"变成只在运行期才暴露的悬案）。
      - `auto_register=False`（默认）**不得触碰任何全局状态**（受 `test_zero_dependency`
        守门的"import 无副作用"约束）。
    """
    auto_register = bool(kwargs.pop("auto_register", False))

    def _decorate(target: Callable[..., Any]) -> Tool:
        built = Tool.from_function(target, **kwargs)
        if auto_register:
            # 函数内延迟 import：`tools/registry.py` 顶层 import 本模块的 `Tool`
            # （注册表必须能判定 isinstance），顶层反向 import 会成环。
            from liteagent.tools.registry import get_default_registry

            get_default_registry().register(built, override=False)
        return built

    if func is not None:
        return _decorate(func)
    return _decorate


def make_function_tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    func: Callable[[dict[str, Any]], Any],
    is_async: bool = False,
    tags: Sequence[str] = (),
    dangerous: bool = False,
    requires_approval: bool = False,
    idempotent: bool = True,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Tool:
    """动态构建工具（用于 Hierarchical 的 delegate 工具、用户运行时造工具）。

    `pass_style` 固定为 `"mapping"`：`func` 接收一个**已校验的 args dict**。

    与 `@tool` 的关键差别：这里没有签名可反射（`func` 的签名永远是 `(args)`），
    所以 `parameters` 缺省时**不能**去反射它 —— 那会产出 `{"properties": {"args": ...}}`
    这种把模型引到沟里的 schema。缺省值是一个"接受任意对象"的宽松 schema，
    并**记一条 warning**（红线 12：宽松不是错，静默的宽松才是）。
    """
    warnings_list: list[str] = []
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "additionalProperties": True}
        warnings_list.append(
            "make_function_tool: no parameters= supplied, using a permissive schema "
            '({"type": "object", "additionalProperties": true}); '
            "the model will not learn anything about the expected keys"
        )
    built = Tool.from_function(
        func,
        name=name,
        description=description,
        parameters=parameters,
        is_async=is_async,
        pass_style=PASS_STYLE_MAPPING,
        tags=tags,
        dangerous=dangerous,
        requires_approval=requires_approval,
        idempotent=idempotent,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    if warnings_list:
        # `ToolSpec` 是 frozen 的 —— 不能原地 append。用 dataclasses.replace 造一份，
        # 既保住不可变性，又把 warning 送进 spec（测试可断言、trace 可见）。
        built = Tool(_replace_warnings(built.spec, warnings_list))
    return built


def _replace_warnings(spec: ToolSpec, extra: Sequence[str]) -> ToolSpec:
    """返回一份 `warnings` 被追加过的 spec 副本（`ToolSpec` 是 frozen，不能原地改）。"""
    return dataclasses.replace(spec, warnings=tuple(spec.warnings) + tuple(extra))


def is_tool(obj: Any) -> bool:
    """`isinstance(obj, Tool)`。**公开 API**（§7.2 冻结），供框架外部做统一的判定。

    `Tool` 与普通函数都是 callable，外部代码（调用方的工具合并、自建注册表）需要一条
    统一的判定入口。`[v3 修正]` 上一版 docstring 说"否则同一个 isinstance 检查会在 5 个
    文件里各写一遍"是**不实的**：仓库里只有两处这样的检查，而且它们就在本模块内
    （`Tool.from_function` 的幂等判定与 `ToolRegistry.register` 的类型校验），
    现在都已改为调用本函数，所以这条理由与实现一致了。
    """
    return isinstance(obj, Tool)


# --------------------------------------------------------------------------------------
# 协作式取消（配合 §7.4.1 步骤 5.c 的超时语义，D-09）
# --------------------------------------------------------------------------------------

CANCEL_FLAG_VAR_NAME = "liteagent_tool_cancel"

# 为什么是 contextvars 而不是 threading.local：executor 在**调用方线程**进入
# `cancel_scope()`，工具函数跑在**线程池的 worker 线程**里。threading.local 只认线程，
# worker 什么也看不到；contextvar 在读方向上能穿透 `run_in_executor` / `to_thread`
# （M-5 实测）。注意方向是单向的：**写**不回传，所以 `flag.set()` 只可能发生在
# 设置它的那个线程（executor 超时分支），worker 只读不写。
_CANCEL_FLAG: contextvars.ContextVar["threading.Event | None"] = contextvars.ContextVar(
    CANCEL_FLAG_VAR_NAME, default=None
)


def current_cancel_flag() -> "threading.Event | None":
    """返回当前工具调用的取消信号（`threading.Event`），无调用上下文时返回 `None`。

    长耗时的同步工具应在循环里检查 `flag is not None and flag.is_set()` 并主动返回。
    为什么需要它：`asyncio.wait_for` 无法中断已经跑在 worker 线程里的同步代码（D-09）。

    取回的是**活的 Event 对象本身**（不是副本）：worker 线程与本线程看到的是同一个
    对象，于是 `wait()` / `set()` 天然协同 —— 这正是需要 thread-safe 原语的原因，
    也是 §0.4 "threading 原语不限" 的典型用法。
    """
    return _CANCEL_FLAG.get()


@contextmanager
def cancel_scope() -> Iterator["threading.Event"]:
    """executor 内部使用：进入时造一个新 `threading.Event()` 并 set 到 contextvar，退出时 reset。

    **必须在 `asyncio.run_in_executor` / `asyncio.to_thread` 的调用方所在线程设置**
    （M-5 实测：contextvar 的**读**能穿透到 worker 线程，**写**不能回传）。

    【实测补充，3.10.12 复现，executor 实现者必读】"读能穿透"**有前提**：
    三条提交路径的行为并不一致 ——

        `asyncio.to_thread(fn)`          -> worker 能看到 flag（它内部 copy_context().run）
        `loop.run_in_executor(tp, fn)`   -> worker 看到 **None**（不复制 context）
        裸 `ThreadPoolExecutor.submit(fn)` -> worker 看到 **None**

    即 §7.4.1 步骤 5.c 里 `run_in_executor` 那行的注释（"contextvars 在此复制"）
    在本机并不成立。**走 `run_in_executor` 的同步分支必须自己包一层**：

        ctx = contextvars.copy_context()
        coro = loop.run_in_executor(tp, functools.partial(ctx.run, tool.run, args))

    否则 `current_cancel_flag()` 在同步工具里**恒为 None**，"协作式取消"会静默失效 ——
    而恰恰是同步工具（不可中断）最需要它。
    复制的是**上下文**，worker 拿到的仍是同一个 `threading.Event` 对象，
    所以之后调用方 `flag.set()` 照样能被 worker 的 `is_set()` 看到。

    [v2 变更] **每个 attempt 必须重新进入一次**（v1 允许放在重试循环外，
    会导致第 2 次尝试拿到已 set 的同一个 Event，工具一进循环就自杀，重试全部秒失败）。

    用 `reset(token)` 而不是"把 contextvar 置回 None"：前者精确还原进入前的值
    （嵌套 `cancel_scope` 时不会把外层的 flag 抹掉），后者会把外层信号一起吃掉。
    """
    flag = threading.Event()
    token = _CANCEL_FLAG.set(flag)
    try:
        yield flag
    finally:
        _CANCEL_FLAG.reset(token)
