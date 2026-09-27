from __future__ import annotations

"""内置代码工具（规范 §7.5 的 ``code.py``）：``python_eval`` / ``python_exec`` / ``run_tests``。

三个工具的安全等级**完全不同**，这份差异本身就是面试时可以说清的"分层信任"设计：

===================  ==========================================================
工具                 信任模型
===================  ==========================================================
``python_eval``      **受限表达式求值（非安全沙箱）**：AST 白名单遍历 + 自写求值器，
                     **绝不调用 builtin ``eval``**。仍然不是安全边界（资源耗尽、
                     未来节点遗漏都会破），所以文档话术统一为"受限（restricted）"。
``python_exec``      **任意代码执行**：子进程 + ``-I`` 隔离，标 ``dangerous=True``
                     + ``requires_approval=True``。它存在的理由是让 Agent 真的能跑
                     代码，因此"能否执行"必须由**审批层**而不是正则来决定。
``run_tests``        跑仓库里的测试套件（子进程），唯一特殊处理是"指向本仓库
                     ``tests/`` 时追加 WARNING"，防止一个测试调用 ``run_tests()``
                     递归跑整套用例、把测试时长变成指数增长。
===================  ==========================================================

``python_eval`` 的**复杂度闸**（面试高频追问"为什么不直接给它一个 timeout"）：
``asyncio.wait_for`` 只能让**调用方**不再等待，无法中断已经跑在 worker 线程里的
求值（§0.4 / D-09），所以 ``10**10**10`` 这类表达式必须在**静态阶段**被拦下来 ——
见 :func:`safe_eval_ast` 的 docstring 与 :data:`MAX_POW_EXPONENT` 等常量。
"""

import ast
import builtins
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liteagent.config import NO_TIMEOUT, truncate_head_tail
from liteagent.errors import ToolExecutionError, ToolTimeoutError, ToolValidationError
from liteagent.tools.base import Tool, make_function_tool

if TYPE_CHECKING:  # pragma: no cover - 只为注解
    from liteagent.tools.builtin.files import PathSandbox

__all__ = [
    "ALLOWED_AST_NODES",
    "ALLOWED_ATTRIBUTES",
    "ALLOWED_BUILTINS",
    "FORBIDDEN_NAMES",
    "MAX_AST_DEPTH",
    "MAX_EXPRESSION_CHARS",
    "MAX_POW_EXPONENT",
    "MAX_RANGE_ARG",
    "safe_eval_ast",
    "make_code_tools",
]

# --------------------------------------------------------------------------------------
# 白名单与闸门常量（§7.5 冻结）
# --------------------------------------------------------------------------------------

# 允许出现在表达式 AST 里的**全部**节点类型。不在这个集合里的节点一律拒绝 ——
# 白名单（而不是黑名单）是这里唯一正确的方向：Python 的 AST 有上百种节点，
# 黑名单漏一种就是一条任意代码执行路径。
ALLOWED_AST_NODES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.Name,
        # 运算与逻辑
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        # 运算符节点（BinOp.op 等）
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        # 比较运算符
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        # 名字上下文
        ast.Load,
        # 容器与下标
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.Subscript,
        ast.Slice,
        # 调用与属性访问（内容另有限制，见 ALLOWED_BUILTINS / ALLOWED_ATTRIBUTES）
        ast.Call,
        ast.Attribute,
        ast.keyword,
    }
)

# 允许被调用的内置函数名。
ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "float",
        "format", "hash", "hex", "int", "isinstance", "len", "list", "max", "min",
        "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
    }
)

# 明确禁止的名字。它们**本来也不在** ALLOWED_BUILTINS 里；单独列一份是为了让报错
# 文案能直指"你正在试图做什么危险的事"，而不是笼统的"unknown name"。
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__import__", "eval", "exec", "compile", "open", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
        "memoryview", "object", "type", "super",
    }
)

# 允许访问的属性：**只读方法**，且按接收者的运行时类型分派。
# 为什么按类型分：`"ab".upper` 与 `[].upper` 是两件事，前者是白名单内的只读方法，
# 后者必须失败 —— 只查"名字在不在名单里"会把这个区别抹平。
ALLOWED_ATTRIBUTES: dict[str, frozenset[str]] = {
    "str": frozenset(
        {
            "capitalize", "casefold", "count", "endswith", "find", "format", "index",
            "isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric",
            "isspace", "istitle", "isupper", "join", "lower", "lstrip", "partition",
            "removeprefix", "removesuffix", "replace", "rfind", "rindex", "rjust",
            "rpartition", "rsplit", "rstrip", "split", "splitlines", "startswith",
            "strip", "swapcase", "title", "upper", "zfill",
        }
    ),
    "list": frozenset({"count", "index"}),
    "dict": frozenset({"get", "items", "keys", "values"}),
}

# ---- 复杂度闸（静态，见模块 docstring）----
MAX_POW_EXPONENT: float = 1000.0        # `a ** b` 的 |b| 上限
MAX_RANGE_ARG: float = 1e6              # `range(n)` 里字面量 n 的上限
MAX_EXPRESSION_CHARS: int = 2000        # 表达式字符数上限
MAX_AST_DEPTH: int = 32                 # AST 嵌套深度上限

# 常量折叠的自我保护上限（见 _const_number）：超过就笼统地当作"大到不可界定"。
_FOLD_MAGNITUDE: float = 1e15           # 折叠结果超过它就归为 inf
_FOLD_POW_EXPONENT: float = 64.0        # 折叠 `**` / `<<` 时允许的指数/移位量上限
_FOLD_BASE_LIMIT: float = 1e6           # 折叠 `**` 时允许的底数上限

_EVAL_TOOL_NAME = "python_eval"


# --------------------------------------------------------------------------------------
# safe_eval_ast
# --------------------------------------------------------------------------------------


def _violation(reason: str, **context: Any) -> ToolExecutionError:
    """构造一个"违规"异常。``retryable=False``（类默认值）：表达式非法是**永久**错误，
    重试同一个表达式不会有不同结果。``context`` 里带上违规位置，便于排查是哪一段触发的。
    """
    return ToolExecutionError(
        tool_name=_EVAL_TOOL_NAME,
        message=f"expression rejected: {reason}",
        context={"reason": reason, **context},
    )


def _safe_builtins() -> dict[str, Any]:
    """把 ``ALLOWED_BUILTINS`` 映射到真正的内置函数对象。

    直接用真内置（而不是自己包一层）：``len``/``sorted``/``range`` 这些函数本身没有
    逃逸能力（拿到它们也拿不到 ``__builtins__``，因为属性访问被 ``ALLOWED_ATTRIBUTES``
    限制在 str/list/dict 的只读方法上）。
    """
    table: dict[str, Any] = {}
    for name in ALLOWED_BUILTINS:
        func = getattr(builtins, name, None)
        if func is not None:
            table[name] = func
    return table


_SAFE_BUILTINS: dict[str, Any] = _safe_builtins()


def _ast_depth(node: ast.AST) -> int:
    """AST 的最大嵌套深度（迭代实现）。

    为什么用迭代而不是递归：这个函数**自己**就必须能在深表达式上安全运行 ——
    一个为"防止递归爆栈"而写的检查器如果自己递归，就等于把爆栈点往前挪了一步。
    """
    deepest = 0
    stack: list[tuple[ast.AST, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        deepest = max(deepest, depth)
        for child in ast.iter_child_nodes(current):
            stack.append((child, depth + 1))
    return deepest


def _const_number(node: ast.AST) -> float | int | None:
    """把"纯字面量算术"折叠成一个数，用于复杂度闸。

    返回 ``None`` = "不是常数，静态阶段无法界定"（调用方按**拒绝**处理）；
    返回 ``inf`` = "是常数，但大到不必精确算"（同样是拒绝，且避免折叠本身变成攻击面）。

    **折叠必须是安全的**（否则闸门自己就是炸弹）：

    - ``**`` 只在 ``|指数| <= _FOLD_POW_EXPONENT`` 且 ``|底| <= 1e6`` 时真的去算，
      否则直接返回 ``inf`` —— ``10**10**10`` 的中间值 ``10**10`` 能算（10 位十进制），
      而 ``2**100000000`` 绝不能算。
    - ``<<`` 同理限制移位量。
    - 结果统一按 ``_FOLD_MAGNITUDE`` 截断为 ``inf``。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return _cap_number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_number(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _const_number(node.left)
        right = _const_number(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow) and (
            abs(left) > _FOLD_BASE_LIMIT or abs(right) > _FOLD_POW_EXPONENT
        ):
            return float("inf")
        if isinstance(node.op, (ast.LShift, ast.RShift)) and abs(right) > _FOLD_POW_EXPONENT:
            return float("inf")
        try:
            value = _apply_binop(node.op, left, right)
        except (ArithmeticError, TypeError, ValueError):
            # 折叠失败（如 1/0）-> 交给求值阶段按正常的 Python 语义报错。
            return None
        if isinstance(value, complex):
            return None
        return _cap_number(value)
    return None


def _cap_number(value: float | int) -> float | int:
    """超过 ``_FOLD_MAGNITUDE`` 的值统一收敛成 ``inf``：闸门只关心"是否超限"。"""
    try:
        return float("inf") if abs(value) > _FOLD_MAGNITUDE else value
    except (TypeError, OverflowError):  # pragma: no cover - 白名单内不会出现
        return float("inf")


def _check_pow(node: ast.BinOp) -> None:
    """``**`` 的复杂度闸：指数必须是字面量，且 ``|指数| <= MAX_POW_EXPONENT``。"""
    exponent = _const_number(node.right)
    if exponent is None:
        raise _violation(
            f"cannot statically bound the exponent of '**' (limit {MAX_POW_EXPONENT:g}); "
            "use a literal exponent",
            node="Pow",
        )
    if abs(exponent) > MAX_POW_EXPONENT:
        raise _violation(
            f"exponent {exponent:g} exceeds the limit {MAX_POW_EXPONENT:g}",
            node="Pow",
        )


def _check_range(node: ast.Call) -> None:
    """``range(n)`` 的字面量参数上界（防止 ``list(range(10**7))`` 这种内存炸弹）。"""
    for arg in node.args:
        value = _const_number(arg)
        if value is not None and abs(value) > MAX_RANGE_ARG:
            raise _violation(
                f"range() argument {value:g} exceeds the limit {MAX_RANGE_ARG:g}",
                node="Call",
                func="range",
            )


def _check_pow_call(node: ast.Call) -> None:
    """``pow(a, b)`` 与 ``a ** b`` 用同一把尺子。

    ``pow`` 在白名单里（它是纯函数），但 ``pow(2, 10**10)`` 与 ``2 ** (10**10)``
    是同一颗炸弹 —— 只堵运算符不堵函数等于没堵。这是对 §7.5 闸门 #2 的**保守延伸**
    （规范只点名了 ``Pow`` 节点），延伸方向与"复杂度必须在静态阶段拦截"一致。
    """
    if len(node.args) < 2:
        return
    exponent = _const_number(node.args[1])
    if exponent is None:
        raise _violation(
            "cannot statically bound the exponent of pow() "
            f"(limit {MAX_POW_EXPONENT:g}); use a literal exponent",
            node="Call",
            func="pow",
        )
    if abs(exponent) > MAX_POW_EXPONENT:
        raise _violation(
            f"pow() exponent {exponent:g} exceeds the limit {MAX_POW_EXPONENT:g}",
            node="Call",
            func="pow",
        )


def _validate_tree(tree: ast.Expression, expression: str, variables: Mapping[str, Any]) -> None:
    """三条静态规则 + 逐节点白名单（全部在求值**之前**跑完）。"""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise _violation(
            f"expression is {len(expression)} chars, over the limit {MAX_EXPRESSION_CHARS}",
            node="Expression",
        )
    depth = _ast_depth(tree)
    if depth > MAX_AST_DEPTH:
        raise _violation(f"nesting depth {depth} exceeds the limit {MAX_AST_DEPTH}", node="Expression")

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODE_TYPES:
            raise _violation(f"node type {type(node).__name__} is not allowed", node=type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise _violation(f"name {node.id!r} is forbidden", node="Name", name=node.id)
            if node.id not in ALLOWED_BUILTINS and node.id not in variables:
                raise _violation(
                    f"unknown name {node.id!r}; allowed builtins: "
                    f"{', '.join(sorted(ALLOWED_BUILTINS))}",
                    node="Name",
                    name=node.id,
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise _violation(
                    f"attribute {node.attr!r} is not allowed (names starting with '_' are never "
                    "exposed)",
                    node="Attribute",
                    attr=node.attr,
                )
            if not any(node.attr in allowed for allowed in ALLOWED_ATTRIBUTES.values()):
                raise _violation(
                    f"attribute {node.attr!r} is not in the read-only allow-list "
                    f"({', '.join(sorted(ALLOWED_ATTRIBUTES))})",
                    node="Attribute",
                    attr=node.attr,
                )
        elif isinstance(node, ast.Call):
            _validate_call(node, variables)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _check_pow(node)
        elif isinstance(node, ast.keyword) and node.arg is None:
            # `f(**kwargs)` 展开的是运行时才知道的键，静态阶段无法审。
            raise _violation("'**' keyword unpacking is not allowed", node="keyword")


def _validate_call(node: ast.Call, variables: Mapping[str, Any]) -> None:
    """只允许两种调用形态：白名单内置函数，或白名单只读方法。"""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in FORBIDDEN_NAMES:
            raise _violation(f"call to {func.id!r} is forbidden", node="Call", func=func.id)
        if func.id not in ALLOWED_BUILTINS:
            raise _violation(
                f"call to {func.id!r} is not allowed; callable names are limited to the allowed "
                "builtins",
                node="Call",
                func=func.id,
            )
        if func.id in variables:
            # 变量遮蔽了内置名：静态审查看到的是"调到 len()"，运行时却会调到变量里的
            # 那个可调用对象。两者不一致时一律拒绝（静态与运行时必须同源）。
            raise _violation(
                f"name {func.id!r} is shadowed by a variable; rename it in `variables`",
                node="Call",
                func=func.id,
            )
        if func.id == "range":
            _check_range(node)
        elif func.id == "pow":
            _check_pow_call(node)
        return
    if isinstance(func, ast.Attribute):
        # 接收者类型在静态阶段未知（`"a".upper` vs `[1].upper`），方法名先过名单，
        # 精确的类型分派留给求值器的 `_eval_call`（那里能看到真实对象）。
        if not any(func.attr in allowed for allowed in ALLOWED_ATTRIBUTES.values()):
            raise _violation(
                f"method call {func.attr!r} is not in the read-only allow-list",
                node="Call",
                func=func.attr,
            )
        return
    raise _violation(
        f"unsupported call target {type(func).__name__}",
        node="Call",
        func=type(func).__name__,
    )


_ALLOWED_NODE_TYPES: frozenset[type] = ALLOWED_AST_NODES


def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
    """运算符分派。用 ``ast`` 节点类到 lambda 的映射，而不是一串 ``if isinstance``：
    后者在"新增一个运算符但忘了加分支"时会静默走到 ``else``。
    """
    table: dict[type, Any] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
        ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b,
        ast.BitOr: lambda a, b: a | b,
        ast.BitXor: lambda a, b: a ^ b,
        ast.BitAnd: lambda a, b: a & b,
    }
    handler = table.get(type(op))
    if handler is None:  # pragma: no cover - 白名单已经挡住了
        raise _violation(f"unsupported operator {type(op).__name__}", node="BinOp")
    return handler(left, right)


def _apply_unaryop(op: ast.unaryop, value: Any) -> Any:
    table: dict[type, Any] = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
        ast.Not: lambda a: not a,
        ast.Invert: lambda a: ~a,
    }
    handler = table.get(type(op))
    if handler is None:  # pragma: no cover - 白名单已经挡住了
        raise _violation(f"unsupported unary operator {type(op).__name__}", node="UnaryOp")
    return handler(value)


def _apply_compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    table: dict[type, Any] = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Is: lambda a, b: a is b,
        ast.IsNot: lambda a, b: a is not b,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }
    handler = table.get(type(op))
    if handler is None:  # pragma: no cover - 白名单已经挡住了
        raise _violation(f"unsupported comparison {type(op).__name__}", node="Compare")
    return bool(handler(left, right))


def _lookup_attribute(value: Any, attr: str) -> Any:
    """取白名单属性，并在**运行时**按接收者类型再校验一次。

    静态检查只能看名字（``upper`` 在名单里），这里才能回答"``upper`` 是不是挂在一个
    ``str`` 上"。两次检查都必须有：静态的负责给出清晰的拒绝文案，运行时的负责堵住
    "名字合法但接收者不对"的漏洞。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，先挡掉以免误判
        raise _violation(f"attribute {attr!r} is not allowed on bool", node="Attribute", attr=attr)
    type_name: str | None = None
    if isinstance(value, str):
        type_name = "str"
    elif isinstance(value, list):
        type_name = "list"
    elif isinstance(value, dict):
        type_name = "dict"
    allowed = ALLOWED_ATTRIBUTES.get(type_name or "", frozenset())
    if attr not in allowed:
        raise _violation(
            f"attribute {attr!r} is not allowed on {type(value).__name__}",
            node="Attribute",
            attr=attr,
        )
    # 取属性后**再确认一次**它是绑定方法：属性白名单是"方法名"名单，
    # 万一某个类型上同名属性其实是数据字段，这一步会把它挡住。
    target = getattr(value, attr, None)
    if target is None or not callable(target):
        raise _violation(f"attribute {attr!r} is not callable", node="Attribute", attr=attr)
    return target


class _Evaluator:
    """受限表达式求值器：显式 AST 遍历，**不调用 builtin eval**。

    自己实现的代价是"要覆盖所有支持的节点"，收益是"每个节点都只有一种解释方式，
    不可能被 ``eval`` 的隐式行为（如导入、属性链、dunder）绕过"。
    """

    def __init__(self, variables: Mapping[str, Any]) -> None:
        self.variables: dict[str, Any] = dict(variables)

    def eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            return _SAFE_BUILTINS[node.id]
        if isinstance(node, ast.BinOp):
            return _apply_binop(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _apply_unaryop(node.op, self.eval(node.operand))
        if isinstance(node, ast.BoolOp):
            # 短路语义必须保留：`x != 0 and 10 // x` 依赖它才不会 ZeroDivisionError。
            result: Any = isinstance(node.op, ast.And)
            for value_node in node.values:
                value = self.eval(value_node)
                if isinstance(node.op, ast.And):
                    if not value:
                        return value
                    result = value
                else:
                    if value:
                        return value
                    result = value
            return result
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self.eval(comparator)
                if not _apply_compare(op, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.eval(node.body) if self.eval(node.test) else self.eval(node.orelse)
        if isinstance(node, ast.List):
            return [self.eval(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.eval(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {self.eval(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            return {
                self.eval(key): self.eval(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Subscript):
            return self.eval(node.value)[self._slice(node.slice)]
        if isinstance(node, ast.Attribute):
            return _lookup_attribute(self.eval(node.value), node.attr)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        # pragma: no cover - 白名单已经挡住了，这里是防御性兜底
        raise _violation(f"node type {type(node).__name__} is not supported", node=type(node).__name__)

    def _slice(self, node: ast.AST) -> Any:
        """把下标节点求值成 ``int`` 或 ``slice``（``a[1:2]`` 的 ``1:2`` 是 Slice 节点）。"""
        if isinstance(node, ast.Slice):
            return slice(
                self.eval(node.lower) if node.lower is not None else None,
                self.eval(node.upper) if node.upper is not None else None,
                self.eval(node.step) if node.step is not None else None,
            )
        return self.eval(node)

    def _eval_call(self, node: ast.Call) -> Any:
        func = node.func
        args = [self.eval(arg) for arg in node.args]
        kwargs = {keyword.arg: self.eval(keyword.value) for keyword in node.keywords}
        if isinstance(func, ast.Name):
            return _SAFE_BUILTINS[func.id](*args, **kwargs)
        # 静态阶段已确认 attr 在名单里；这里取到的是**绑定方法**（白名单全是只读方法，
        # 不会改变接收者的结构，因此不必担心 `sort()` 这类原地修改被当成表达式副作用）。
        method = _lookup_attribute(self.eval(func.value), func.attr)
        return method(*args, **kwargs)


def safe_eval_ast(expression: str, variables: Mapping[str, Any] | None = None) -> Any:
    """受限表达式求值（非安全沙箱）。AST 白名单求值，**绝不调用 builtin eval**。

    违规 -> ``ToolExecutionError(retryable=False)``。三条冻结规则：

    1. ``Name`` 允许 ``ALLOWED_BUILTINS`` 里的名字，**或** ``variables`` 里存在的键
       （v1 的注释写"仅白名单内置"，与 ``variables`` 参数直接冲突；v2 冻结为两者都允许）。
    2. **静态复杂度闸**（因为 ``timeout_s`` 无法中断求值，见 D-09 的诚实原则）：
       拒绝 ``Pow`` 且指数绝对值 > ``MAX_POW_EXPONENT``；拒绝 ``range`` 字面参数 > ``MAX_RANGE_ARG``；
       拒绝表达式长度 > ``MAX_EXPRESSION_CHARS``；拒绝 AST 深度 > ``MAX_AST_DEPTH``。
       **``timeout_s`` 只保证调用方不再等待，不保证中断求值，因此复杂度必须在静态检查
       阶段拦截。**
    3. 本函数**不是安全边界**（资源耗尽、未来节点遗漏都会破）。文档话术统一为
       "受限表达式求值（非安全沙箱）"。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise _violation("expression must be a non-empty string")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(
            tool_name=_EVAL_TOOL_NAME,
            message=f"expression is not valid Python: {exc.msg}",
            context={"reason": "syntax", "offset": exc.offset or 0},
            cause=exc,
        ) from exc
    table = dict(variables or {})
    _validate_tree(parsed, expression, table)
    return _Evaluator(table).eval(parsed.body)


# --------------------------------------------------------------------------------------
# 模块级私有实现（闭包注入沙箱；返回 str）
# --------------------------------------------------------------------------------------


def _python_eval(_sandbox: "PathSandbox | None", expression: str, variables_json: str = "{}") -> str:
    """Restricted expression evaluation (NOT a security sandbox)."""
    try:
        parsed_variables = json.loads(variables_json) if variables_json else {}
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            tool_name=_EVAL_TOOL_NAME,
            message=f"variables_json is not valid JSON: {exc}",
            context={"reason": "variables_json"},
            cause=exc,
        ) from exc
    if not isinstance(parsed_variables, dict):
        raise ToolExecutionError(
            tool_name=_EVAL_TOOL_NAME,
            message="variables_json must decode to a JSON object",
            context={"reason": "variables_json", "type": type(parsed_variables).__name__},
        )
    result = safe_eval_ast(expression, parsed_variables)
    # str 结果原样返回（`'hello'` 加引号会让模型以为自己拿到的是 Python 字面量），
    # 其它类型用 repr：`[1, 2]` 比 `[1, 2]（str() 的结果）` 更没有歧义，
    # 也和"表达式求值"的心智模型一致。
    return result if isinstance(result, str) else repr(result)


def _resolve_workdir(_sandbox: "PathSandbox | None", cwd: str | None) -> str:
    """解析子进程的工作目录：有沙箱时以沙箱根为默认，``cwd`` 一律先过沙箱校验。"""
    if _sandbox is None:
        return str(Path(cwd).expanduser().resolve()) if cwd else str(Path.cwd())
    if cwd is not None:
        return str(_sandbox.resolve(cwd, write=True))
    return str(_sandbox.root)


def _python_exec(
    _sandbox: "PathSandbox | None",
    code: str,
    timeout_s: float = 10.0,
    cwd: str | None = None,
) -> str:
    """Execute Python code in an isolated subprocess. Returns stdout/stderr/exit code.

    ``-I``（isolated）隔离掉 ``PYTHONPATH`` / 用户 site-packages / ``sys.path[0]``：
    子进程不该因为"调用方恰好在一个奇怪的工作目录里"而 import 到别的东西。
    代码本身**不受沙箱约束** —— 它本就是任意代码执行工具，所以标
    ``dangerous=True`` + ``requires_approval=True``，靠审批层而不是正则来把关。
    """
    workdir = _resolve_workdir(_sandbox, cwd)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=workdir,
            timeout=float(timeout_s) if timeout_s and timeout_s > 0 else None,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(
            tool_name="python_exec",
            timeout_s=float(timeout_s or 0.0),
            cause=exc,
        ) from exc
    return _render_process_output(proc, label="python_exec")


def _render_process_output(proc: "subprocess.CompletedProcess[str]", *, label: str) -> str:
    """把子进程结果渲染成"退出码 + stdout + stderr"三段（与 run_shell 保持同一副面孔）。"""
    sections = [f"exit_code: {int(proc.returncode)}"]
    if proc.stdout:
        sections.append("stdout:\n" + proc.stdout.rstrip("\n"))
    if proc.stderr:
        sections.append("stderr:\n" + proc.stderr.rstrip("\n"))
    if not proc.stdout and not proc.stderr:
        sections.append(f"({label}: no output)")
    return "\n".join(sections)


_RAN_RE = re.compile(r"Ran (\d+) tests? in ([\d.]+)s")
_RESULT_RE = re.compile(r"^(OK|FAILED)\b", re.MULTILINE)


def _run_tests(
    _sandbox: "PathSandbox | None",
    path: str = "tests",
    pattern: str = "test_*.py",
    timeout_s: float = 300.0,
    extra_args: str = "",
) -> str:
    """Run unittest discovery and return a summary.

    ``extra_args`` 走 ``shlex.split`` 而不是 ``str.split``：``-k "a and b"`` 这类
    带空格的参数是常态，简单 split 会把它们切成两半。

    [v2 冻结] ``path`` 经 ``sandbox.resolve`` 解析；**若解析结果落在本仓库的 ``tests/``
    目录内，在返回文本里追加一行 WARNING** —— 一个测试调用 ``run_tests()`` 会再跑一遍
    整套用例，递归下去测试时长是指数级的。
    """
    if _sandbox is not None:
        start_dir = _sandbox.resolve(path, write=True)
    else:
        start_dir = Path(path).expanduser().resolve()
    if not start_dir.exists():
        return f"ERROR: test directory not found: {start_dir}"
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(start_dir),
        "-p",
        pattern,
        "-v",
    ]
    if extra_args:
        command.extend(shlex.split(extra_args))
    workdir = _resolve_workdir(_sandbox, None)
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            timeout=float(timeout_s) if timeout_s and timeout_s > 0 else None,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(
            tool_name="run_tests",
            timeout_s=float(timeout_s or 0.0),
            cause=exc,
        ) from exc

    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    ran_match = _RAN_RE.search(output)
    status_match = _RESULT_RE.search(output)
    total = int(ran_match.group(1)) if ran_match else 0
    elapsed = ran_match.group(2) if ran_match else "?"
    status = status_match.group(1) if status_match else ("OK" if proc.returncode == 0 else "FAILED")

    header = [
        f"status: {status}",
        f"tests: {total}",
        f"elapsed_s: {elapsed}",
        f"exit_code: {int(proc.returncode)}",
    ]
    if _is_repo_tests_dir(start_dir):
        header.append(
            "WARNING: path resolves inside this repository's tests/ directory; running it from "
            "inside a test recursively re-runs the whole suite. Point run_tests at a scratch "
            "directory or a narrower -s path instead."
        )
    body = truncate_head_tail(output, 8000) if len(output) > 8000 else output
    return "\n".join(header) + "\n--- output ---\n" + body


def _is_repo_tests_dir(path: Path) -> bool:
    """``path`` 是否落在"本仓库的 tests/ 目录"里。

    用 ``__file__`` 往上数四级而不是硬编码路径：仓库可以被 clone 到任何地方，
    但"code.py 所属仓库的 tests/ 目录"这个关系是不变的。
    """
    try:
        repo_tests = Path(__file__).resolve().parents[3] / "tests"
        return path == repo_tests or path.is_relative_to(repo_tests)
    except (IndexError, OSError):  # pragma: no cover - 目录结构被改坏时的兜底
        return False


# --------------------------------------------------------------------------------------
# 模型可见的 schema（手写，与模块级函数签名逐字一致）
# --------------------------------------------------------------------------------------

_PYTHON_EVAL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "A single Python expression (no statements, no f-strings, no lambdas).",
        },
        "variables_json": {
            "type": "string",
            "description": 'JSON object of extra variables, e.g. \'{"x": 3}\'.',
        },
    },
    "required": ["expression"],
    "additionalProperties": False,
}

_PYTHON_EXEC_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source to run in an isolated subprocess."},
        "timeout_s": {"type": "number", "description": "Kill the subprocess after this many seconds."},
        "cwd": {"type": "string", "description": "Working directory (must stay inside the sandbox)."},
    },
    "required": ["code"],
    "additionalProperties": False,
}

_RUN_TESTS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to discover tests in."},
        "pattern": {"type": "string", "description": "unittest discovery pattern."},
        "timeout_s": {"type": "number", "description": "Kill the discovery run after this many seconds."},
        "extra_args": {"type": "string", "description": "Extra unittest CLI arguments."},
    },
    "required": [],
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


def make_code_tools(sandbox: "PathSandbox | None" = None) -> list[Tool]:
    """返回 ``[python_eval, python_exec, run_tests]``。

    ``sandbox`` 仅用于把 ``run_tests`` 的 ``path`` 解析到沙箱内（以及给 ``python_exec``
    一个默认工作目录）；``python_exec`` 的**代码**不受沙箱约束 —— 它本就是任意代码执行
    工具，因此标 ``dangerous=True`` + ``requires_approval=True``，并在文档里明确警告。
    """
    if sandbox is not None:
        from liteagent.tools.builtin.files import PathSandbox

        if not isinstance(sandbox, PathSandbox):
            from liteagent.errors import ConfigError

            raise ConfigError(
                "make_code_tools(sandbox=...) requires a PathSandbox instance or None, got "
                f"{type(sandbox).__name__}"
            )

    def python_eval(args: dict[str, Any]) -> str:
        return _python_eval(
            sandbox,
            **_call_args(
                args,
                tool_name="python_eval",
                required=("expression",),
                optional=("variables_json",),
            ),
        )

    def python_exec(args: dict[str, Any]) -> str:
        return _python_exec(
            sandbox,
            **_call_args(
                args,
                tool_name="python_exec",
                required=("code",),
                optional=("timeout_s", "cwd"),
            ),
        )

    def run_tests(args: dict[str, Any]) -> str:
        return _run_tests(
            sandbox,
            **_call_args(
                args,
                tool_name="run_tests",
                required=(),
                optional=("path", "pattern", "timeout_s", "extra_args"),
            ),
        )

    return [
        make_function_tool(
            name="python_eval",
            description=(
                "Evaluate a restricted Python expression (NOT a security sandbox). "
                "Use it for arithmetic/string/data-literal computations."
            ),
            parameters=_PYTHON_EVAL_PARAMETERS,
            func=python_eval,
            tags=("code",),
            dangerous=False,
            idempotent=True,
            timeout_s=5.0,
        ),
        make_function_tool(
            name="python_exec",
            description=(
                "Execute arbitrary Python code in an isolated subprocess. "
                "WARNING: this is real code execution on the host."
            ),
            parameters=_PYTHON_EXEC_PARAMETERS,
            func=python_exec,
            tags=("code",),
            dangerous=True,
            requires_approval=True,
            idempotent=False,
            timeout_s=NO_TIMEOUT,  # 超时由 timeout_s 参数 + subprocess 自身保证
        ),
        make_function_tool(
            name="run_tests",
            description=_run_tests.__doc__ or "Run unittest discovery and return a summary.",
            parameters=_RUN_TESTS_PARAMETERS,
            func=run_tests,
            tags=("code",),
            dangerous=True,
            idempotent=False,
            timeout_s=NO_TIMEOUT,  # 默认 300s 的测试运行不该被 executor 的 30s 抢答
        ),
    ]
