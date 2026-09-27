from __future__ import annotations

# liteagent/tools/__init__.py —— 工具系统的公开 API（规范 §1.4）
#
# 为什么用模块级 `__getattr__` 懒加载，而不是一排顶层 `from ... import ...`：
#
#   1. `executor.py` 与 `builtin/` 的 `register_all` / `BUILTIN_*` 属于**同一份冻结文件清单**
#      （§1.2），但由不同的实现者并行交付。顶层 import 会让 `import liteagent.tools.base`
#      这种"只想用装饰器"的调用在兄弟模块尚未落地时直接 `ImportError` —— 一个与本模块
#      无关的原因，毁掉整条 import 链。
#   2. §1.4 的守门测试要求的是**行为**："`__all__` 里每个名字必须能从该包 import 到"。
#      懒加载满足它：`from liteagent.tools import ToolExecutor`、`getattr(liteagent.tools, "x")`
#      都成立（Python 的 `from ... import` 会回退到模块的 `__getattr__`），
#      而**不存在**的名字依旧老老实实抛 ImportError（不是静默的占位对象）。
#   3. 首次访问后把结果写回 `globals()`，后续访问就是普通的字典命中，
#      没有"每次都要走一遍 import 机制"的持续开销。
#
# 依赖方向（§1.1）：`X/__init__.py` 允许 import 本包任意模块 —— 这里的延迟 import
# 全部指向 `liteagent.tools.*`，没有跨包反向边。
from typing import Any

__all__ = [
    "tool", "Tool", "ToolSpec", "ToolRegistry", "ExecutorConfig", "ToolExecutor",
    "make_function_tool", "is_tool", "current_cancel_flag", "cancel_scope",
    "get_default_registry", "reset_default_registry",
    # builtin 的入口在此 re-export，便于 `from liteagent.tools import register_all`
    "register_all", "BUILTIN_TOOL_NAMES", "BUILTIN_TOOL_GROUPS",
]

# 名字 -> 其定义所在模块。拆成"谁是我自己写的"与"谁是兄弟模块"没有意义：
# 全部走同一条路径，行为一致（少一种特例 = 少一类"只有某个名字才会炸"的 bug）。
_LAZY_SOURCES: dict[str, str] = {
    "tool": "liteagent.tools.base",
    "Tool": "liteagent.tools.base",
    "ToolSpec": "liteagent.tools.base",
    "make_function_tool": "liteagent.tools.base",
    "is_tool": "liteagent.tools.base",
    "current_cancel_flag": "liteagent.tools.base",
    "cancel_scope": "liteagent.tools.base",
    "ToolRegistry": "liteagent.tools.registry",
    "get_default_registry": "liteagent.tools.registry",
    "reset_default_registry": "liteagent.tools.registry",
    "ExecutorConfig": "liteagent.tools.executor",
    "ToolExecutor": "liteagent.tools.executor",
    "register_all": "liteagent.tools.builtin",
    "BUILTIN_TOOL_NAMES": "liteagent.tools.builtin",
    "BUILTIN_TOOL_GROUPS": "liteagent.tools.builtin",
}


def __getattr__(name: str) -> Any:
    """按 `_LAZY_SOURCES` 就地解析公开名字（PEP 562）。

    解析失败时 `ImportError` 原样冒泡 —— 这是**期望**行为：某个名字取不到，
    一定是因为它所属的模块真的有问题，此时静默返回 `None` 或占位对象会把它
    变成运行到一半才炸的悬案。
    """
    module_name = _LAZY_SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # 函数内 import：既避开循环 import（executor 要 import 本包），
    # 也让"只 import base 的人"不为 builtin 的加载付代价。
    from importlib import import_module

    module = import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        # 目标模块在、名字不在（典型的"模块还没写完"或"名字被改了"）。
        # 统一转成 ImportError：`from liteagent.tools import X` 失败时抛
        # ImportError 才符合调用方的直觉（Python 自己的 `from m import n` 也是这么做的），
        # 而且文案里带上真正的来源模块，不用再去猜 `_LAZY_SOURCES`。
        raise ImportError(
            f"cannot import name {name!r} from {__name__!r}: "
            f"{module_name!r} does not define it"
        ) from exc
    globals()[name] = value  # 缓存：下次直接命中模块字典，不再走 __getattr__
    return value


def __dir__() -> list[str]:
    """让 `dir(liteagent.tools)` / 补全看得到 `__all__` 里的名字。

    没有它，懒加载的名字在 `dir()` 里不存在，IDE 补全与 `**kwargs` 式的自省
    会给出与 `__all__` 矛盾的信息。
    """
    return sorted(set(__all__) | set(globals()))
