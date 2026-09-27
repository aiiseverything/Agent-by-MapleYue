from __future__ import annotations

# liteagent/tools/schema.py —— type hints -> JSON Schema 的反射（规范 §7.1）
#
# 为什么这个模块值得单独一个文件：它是"装饰器自动生成 JSON Schema"的唯一实现点，
# 也是全项目唯一同时处理 typing / enum / dataclasses / pydantic 四条反射路径的地方。
#
# 三条设计约束（§7.1 / §13）：
#   1. **顶层零三方 import**：连 pydantic 都不 import —— 可用性用 importlib.util.find_spec
#      探测，pydantic 的模型与 FieldInfo 全部走鸭子类型。这让本模块在无 pydantic 的环境里
#      行为完全一致（§1.3 只允许 transport/embeddings/config/cli 出现三方 import）。
#   2. **一切降级必须留下 warning**（红线 12）：绝不静默返回 {}，调用方（ToolSpec.warnings）
#      能看到"哪个参数被降级、为什么"。
#   3. **required 判定只有唯一一条公式**（§7.1.5），顶层参数与嵌套 dataclass 字段共用它。
#
# 3.10 实测陷阱（§0.2.1 M-1）：`isinstance(Annotated[int, "x"], Annotated)` 恒为 False
# （不报错，静默失效）。取 metadata 的唯一合法方式是 `getattr(ann, "__metadata__", ())`，
# 所以本文件里**没有**任何 `isinstance(..., Annotated)`。

import dataclasses
import enum
import importlib.util
import inspect
import json
import re
import textwrap
import types
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from liteagent.errors import ToolDefinitionError

if TYPE_CHECKING:  # E4：字符串注解 + TYPE_CHECKING，避免 base.py <-> schema.py 循环 import
    from liteagent.tools.base import ToolSpec

__all__ = [
    "SCHEMA_DRAFT",
    "MAX_SCHEMA_DEPTH",
    "PYDANTIC_AVAILABLE",
    "Param",
    "SchemaResult",
    "build_tool_schema",
    "annotation_to_schema",
    "is_optional",
    "annotated_metadata",
    "parse_docstring",
    "validate_instance",
    "unvalidatable_parameters",
    "validate_schema",
    "to_openai_tool",
    "to_anthropic_tool",
]

# 可用性常量必须定义在模块顶层供测试断言（§1.3）。
# 注意这里**不 import pydantic**：find_spec 只查 import 系统，不执行模块代码，
# 因此 schema.py 的依赖面是纯 stdlib。
try:  # pragma: no cover - 环境相关
    PYDANTIC_AVAILABLE: bool = importlib.util.find_spec("pydantic") is not None
except (ImportError, ValueError):  # pragma: no cover - sys.path 损坏等极端情况
    PYDANTIC_AVAILABLE = False

SCHEMA_DRAFT: str = "https://json-schema.org/draft/2020-12/schema"
MAX_SCHEMA_DEPTH: int = 8

# Param.default 的哨兵（模块级，必须先于 Param 定义）。
# 为什么不用 None：None 是合法默认值，用 None 会让"显式 None"与"未设置"不可区分（§7.1.3）。
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """框架自带的结构化参数元数据（零依赖，不 import pydantic）。

    用法：`def f(x: Annotated[int, Param(description="数量", ge=0)] = 1) -> None: ...`
    `default` 的语义：**若设置了，则该参数视为可选**（即使签名里没有默认值），
    schema 的 `default` 取该值，且不进入 `required`（§7.1.3 / §7.1.5）。
    """

    description: str | None = None
    default: Any = _UNSET  # 哨兵，区别于 None
    ge: float | None = None  # -> minimum
    le: float | None = None  # -> maximum
    gt: float | None = None  # -> exclusiveMinimum
    lt: float | None = None  # -> exclusiveMaximum
    min_length: int | None = None  # str -> minLength, list -> minItems
    max_length: int | None = None  # str -> maxLength, list -> maxItems
    pattern: str | None = None  # -> pattern
    examples: tuple[Any, ...] = ()  # -> examples
    enum: tuple[Any, ...] | None = None  # 覆盖 -> enum
    title: str | None = None  # **会被丢弃**，仅为 API 完整性保留


@dataclass
class SchemaResult:
    """`build_tool_schema` 的返回值：schema + 逐参描述 + 降级 warning + 名字/描述。"""

    schema: dict[str, Any]
    descriptions: dict[str, str]
    warnings: list[str]
    name: str
    description: str


# ---------------------------------------------------------------------------
# Annotated 元数据：识别与取值（§7.1.3，全部冻结）
# ---------------------------------------------------------------------------


def annotated_metadata(annotation: Any) -> tuple[Any, ...]:
    """返回 Annotated 的 metadata 元组，非 Annotated 时返回 ()。

    冻结实现（M-1）：`return getattr(annotation, "__metadata__", ())`。
    **禁止** `isinstance(annotation, Annotated)`（3.10 实测恒 False，静默失效）。
    """
    return getattr(annotation, "__metadata__", ())


def _unwrap_annotated(annotation: Any) -> Any:
    """把 `Annotated[X, meta...]` 剥成 X；非 Annotated 原样返回。

    与 `annotated_metadata` 用同一套 `__metadata__` 判据（M-1）。
    """
    if getattr(annotation, "__metadata__", ()):
        args = get_args(annotation)
        return args[0] if args else Any
    return annotation


def _is_pydantic_undefined(value: Any) -> bool:
    """判断一个值是不是 pydantic 的"未设置"哨兵（`PydanticUndefined` / v1 的 `Undefined`）。

    SPEC-AMBIGUITY：§7.1.3 的文字是"FieldInfo 鸭子类型 -> .default"，字面实现会把
    `Field(ge=1)`（未给 default）读成 `PydanticUndefined`，它 `is not _UNSET`，
    于是**必填参数被误判为可选**。按本节的语义（`_UNSET` 表示"未设置"）把 pydantic 的
    未设置哨兵折成 `_UNSET`，这不改变任何显式默认值的行为。
    """
    return type(value).__name__ in ("PydanticUndefinedType", "UndefinedType")


def _is_field_info(meta: Any) -> bool:
    """鸭子类型识别 pydantic `FieldInfo`（§7.1.3 第 2 条）：**绝不 import pydantic**。

    判据：同时有 `description` 与 `default` 属性，且类型的模块名以 "pydantic" 开头。
    `Param` 也有这两个属性，但它在识别顺序里排第 1，先被 `isinstance(meta, Param)` 截走。
    """
    if not (hasattr(meta, "description") and hasattr(meta, "default")):
        return False
    module = getattr(type(meta), "__module__", "") or ""
    return module.startswith("pydantic")


def _meta_default(meta: Any) -> Any:
    """从一条 metadata 里取 default：Param -> `.default`；pydantic FieldInfo -> `.default`；
    其它 -> `_UNSET`。这是 §7.1.5 required 公式的组成部分。"""
    if isinstance(meta, Param):
        return meta.default
    if _is_field_info(meta):
        value = getattr(meta, "default", _UNSET)
        return _UNSET if _is_pydantic_undefined(value) else value
    return _UNSET


# ---------------------------------------------------------------------------
# 类型判定 / 字面量推断
# ---------------------------------------------------------------------------


def is_optional(annotation: Any) -> bool:
    """注解是否表达"可为 None"。

    SPEC-AMBIGUITY：§7.1 的散文定义只写了"get_origin 是 Union"，但 3.10 实测：
      - `int | None`（PEP 604）的 get_origin 是 `types.UnionType`，**不是** typing.Union；
      - `Annotated[int | None, "x"]` 的 get_origin 是 `typing.Annotated`。
    按字面实现，这两种常见写法都会被判成 required，与 §7.1.1 的"Optional[X] 不进入
    required"和 D-05 直接冲突。因此这里按规范的**整体精神**放宽：先剥 Annotated，
    再把 `types.UnionType` 与 `typing.Union` 一视同仁。
    """
    base = _unwrap_annotated(annotation)
    if base is type(None):
        return True
    origin = get_origin(base)
    if origin is Union or origin is types.UnionType:
        return any(arg is type(None) for arg in get_args(base))
    return False


def _schema_type_name(value: Any) -> str | None:
    """把一个 Python 值映射成 JSON Schema 的 type 名（用于 Literal/Enum 的类型推断）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):  # bool 必须先于 int（isinstance(True, int) 为 True）
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return None


def _infer_type_from_values(values: Sequence[Any]) -> str | None:
    """枚举/字面量的 type 推断：所有值同类才给 type，混型返回 None（省略 type 键）。"""
    kinds = {_schema_type_name(v) for v in values}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    return kind


# ---------------------------------------------------------------------------
# 降级辅助：JSON 可序列化转换（Param.default / examples / enum）
# ---------------------------------------------------------------------------


def _jsonable(value: Any, warnings: list[str]) -> Any:
    """把元数据里的值转成 JSON 可序列化形态。

    为什么要做这一步：schema 最终会进 HTTP body（`json.dumps`）与 trace。
    一个 `Param(default=SomeEnum.X)` 会让整次 LLM 调用在序列化阶段炸掉，
    而降级必须留痕（红线 12），所以这里逐个转换并在真的降级时记 warning。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _jsonable(value.value, warnings)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, warnings) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, warnings) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # dataclass -> dict（asdict 递归，内部的 Enum 等再由上面的分支处理）
        try:
            return _jsonable(dataclasses.asdict(value), warnings)
        except Exception as exc:  # noqa: BLE001 - 只可能是不可深拷贝的对象
            warnings.append(
                f"metadata value {value!r} could not be converted to a dict "
                f"({type(exc).__name__}: {exc}); coerced with str()"
            )
            return str(value)
    warnings.append(
        f"metadata value of type {type(value).__name__} is not JSON serializable; coerced with str()"
    )
    return str(value)


def _length_keyword(fragment: dict[str, Any], kind: str) -> str:
    """min_length/max_length 落哪个关键字：str -> minLength，array -> minItems。

    SPEC-AMBIGUITY：§7.1.3 只写"str -> minLength, list -> minItems"，没写"注解是
    `Any`（无 type）时怎么办"。保守选择：按 **字符串** 形态落（`{}` 上的 minLength
    是无副作用的空关键字），因为绝大多数 `min_length` 用在 str 上。
    """
    if fragment.get("type") == "array":
        return "minItems" if kind == "min" else "maxItems"
    return "minLength" if kind == "min" else "maxLength"


def _looks_like_constraint(obj: Any) -> bool:
    """判断一个对象是否"看起来是可以映射到 JSON Schema 的约束"（annotated_types / pydantic）。"""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith(("annotated_types", "pydantic"))


def _apply_constraint_object(fragment: dict[str, Any], obj: Any, warnings: list[str]) -> None:
    """从 pydantic `FieldInfo.metadata` 里的约束对象（annotated_types.Ge / MinLen / ...）取值。

    为什么需要这一层：实测 pydantic 2.13 的 `FieldInfo` **没有** `ge/le/gt/lt/
    min_length/max_length/pattern` 属性（`Field(ge=1)` 把约束放进 `.metadata` 列表）。
    字面实现 §7.1.3 的 getattr 会**静默丢失** `Field(ge=1)` 的约束 —— 属于红线 12 的静默降级。
    因此这里按属性名鸭子类型补一层映射；实在映射不了的约束必须留 warning。
    """
    mapped = False
    for attr, keyword in (
        ("ge", "minimum"),
        ("gt", "exclusiveMinimum"),
        ("le", "maximum"),
        ("lt", "exclusiveMaximum"),
    ):
        value = getattr(obj, attr, None)
        if value is not None and keyword not in fragment:
            fragment[keyword] = value
            mapped = True
    for attr, kind in (("min_length", "min"), ("max_length", "max")):
        value = getattr(obj, attr, None)
        if value is not None:
            keyword = _length_keyword(fragment, kind)
            if keyword not in fragment:
                fragment[keyword] = value
                mapped = True
    pattern = getattr(obj, "pattern", None)
    if pattern is not None and "pattern" not in fragment:
        if isinstance(pattern, str):
            fragment["pattern"] = pattern
            mapped = True
        elif isinstance(getattr(pattern, "pattern", None), str):  # 已编译的正则
            fragment["pattern"] = pattern.pattern
            mapped = True
    if mapped or not _looks_like_constraint(obj):
        return
    warnings.append(
        f"pydantic constraint {type(obj).__name__} has no JSON Schema counterpart; ignored"
    )


def _apply_param(fragment: dict[str, Any], param: Param, warnings: list[str]) -> None:
    """把一条 `Param` 元数据叠加到 schema fragment 上（§7.1.3）。"""
    if param.description is not None:
        fragment["description"] = param.description
    if param.default is not _UNSET:
        fragment["default"] = _jsonable(param.default, warnings)
    if param.ge is not None:
        fragment["minimum"] = param.ge
    if param.le is not None:
        fragment["maximum"] = param.le
    if param.gt is not None:
        fragment["exclusiveMinimum"] = param.gt
    if param.lt is not None:
        fragment["exclusiveMaximum"] = param.lt
    if param.min_length is not None:
        fragment[_length_keyword(fragment, "min")] = param.min_length
    if param.max_length is not None:
        fragment[_length_keyword(fragment, "max")] = param.max_length
    if param.pattern is not None:
        fragment["pattern"] = param.pattern
    if param.examples:
        fragment["examples"] = [_jsonable(v, warnings) for v in param.examples]
    if param.enum is not None:
        values = [_jsonable(v, warnings) for v in param.enum]
        if not values:  # 空的 enum 会让生成的 schema 非法（draft 要求至少一个元素）
            warnings.append("Param(enum=()) is empty; the enum constraint is dropped")
        else:
            fragment["enum"] = values
    # param.title 按 §7.1.3 **明确丢弃**：JSON Schema 的 title 会被部分 provider 拼进
    # tool 描述里，对模型是纯噪声；保留字段只为 API 完整性。


def _apply_field_info(fragment: dict[str, Any], field_info: Any, warnings: list[str]) -> None:
    """把一条 pydantic `FieldInfo`（鸭子类型）叠加到 fragment 上。

    先读 §7.1.3 列出的属性名，再补 `.metadata` 里的约束对象（见 `_apply_constraint_object`
    的说明）。**绝不 import pydantic**。
    """
    description = getattr(field_info, "description", None)
    if isinstance(description, str) and description:
        fragment["description"] = description
    default = _meta_default(field_info)
    if default is not _UNSET:
        fragment["default"] = _jsonable(default, warnings)
    for attr, keyword in (
        ("ge", "minimum"),
        ("le", "maximum"),
        ("gt", "exclusiveMinimum"),
        ("lt", "exclusiveMaximum"),
    ):
        value = getattr(field_info, attr, None)
        if value is not None:
            fragment[keyword] = value
    for attr, kind in (("min_length", "min"), ("max_length", "max")):
        value = getattr(field_info, attr, None)
        if value is not None:
            fragment[_length_keyword(fragment, kind)] = value
    pattern = getattr(field_info, "pattern", None)
    if isinstance(pattern, str):
        fragment["pattern"] = pattern
    examples = getattr(field_info, "examples", None)
    if examples:
        fragment["examples"] = [_jsonable(v, warnings) for v in examples]
    for constraint in getattr(field_info, "metadata", ()) or ():
        _apply_constraint_object(fragment, constraint, warnings)


def _apply_metadata(fragment: dict[str, Any], metas: Sequence[Any], warnings: list[str]) -> None:
    """识别顺序（§7.1.3，冻结）：Param -> pydantic FieldInfo -> str -> 其它（忽略 + warning）。"""
    for meta in metas:
        if isinstance(meta, Param):
            _apply_param(fragment, meta, warnings)
        elif _is_field_info(meta):
            _apply_field_info(fragment, meta, warnings)
        elif isinstance(meta, str):
            # `Annotated[int, "count of items"]` 这种写法很常见，直接当 description。
            # 已有的 description（来自 Param/FieldInfo）不被裸字符串覆盖 ——
            # 结构化元数据的表达力更强，裸字符串只是快捷写法。
            if "description" not in fragment:
                fragment["description"] = meta
        else:
            warnings.append(
                f"unsupported Annotated metadata of type {type(meta).__name__}; ignored"
            )


# ---------------------------------------------------------------------------
# 注解 -> JSON Schema（§7.1.1 映射表，逐行实现）
# ---------------------------------------------------------------------------


def annotation_to_schema(
    annotation: Any,
    *,
    depth: int = 0,
    seen: frozenset[type] = frozenset(),
) -> tuple[dict[str, Any], list[str]]:
    """返回 (schema_fragment, warnings)。深度/环超限时返回 ({}, [warning])。

    `seen` 是当前递归路径上的 dataclass 类型集合（§7.1.2 的环检测）；
    `depth` 是嵌套层数，超过 `MAX_SCHEMA_DEPTH` 一律降级为 {}（任意类型）+ warning。
    """
    warnings: list[str] = []

    # 深度上限必须在最前面判定：否则深层递归会在检查前先爆栈。
    if depth > MAX_SCHEMA_DEPTH:
        return {}, [
            f"annotation nesting exceeded MAX_SCHEMA_DEPTH={MAX_SCHEMA_DEPTH} "
            f"at depth {depth}; degraded to {{}} (any)"
        ]

    # Annotated 先剥壳，metadata 最后统一应用（§7.1.3）。
    # M-1：判据是 __metadata__ 非空，不是 isinstance(..., Annotated)。
    metas = annotated_metadata(annotation)
    if metas:
        fragment, sub_warnings = annotation_to_schema(
            _unwrap_annotated(annotation), depth=depth, seen=seen
        )
        warnings.extend(sub_warnings)
        _apply_metadata(fragment, metas, warnings)
        return fragment, warnings

    if annotation is None or annotation is type(None):
        return {}, warnings  # 任意类型
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}, warnings
    # ---- 标量：bool 必须先于 int 判定（isinstance(True, int) is True）----
    if annotation is bool:
        return {"type": "boolean"}, warnings
    if annotation is int:
        return {"type": "integer"}, warnings
    if annotation is float:
        return {"type": "number"}, warnings
    if annotation is str:
        return {"type": "string"}, warnings

    # ---- 字符串注解：get_type_hints 失败（如 from __future__ import annotations +
    # 局部导入的 NameError）时的兜底路径，只能降级并留痕 ----
    if isinstance(annotation, str):
        warnings.append(
            f"unresolved string annotation {annotation!r} "
            "(typing.get_type_hints failed); degraded to {} (any)"
        )
        return {}, warnings

    # ---- Enum（IntEnum 是 int 子类，但这里判的是类对象本身，与上面的 is int 不冲突）----
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [member.value for member in annotation]  # 成员顺序 = 定义顺序
        fragment: dict[str, Any] = {"enum": [_jsonable(v, warnings) for v in values]}
        inferred = _infer_type_from_values(values)
        if inferred is not None:
            fragment = {"type": inferred, **fragment}
        return fragment, warnings

    origin = get_origin(annotation)

    # ---- Literal ----
    if origin is Literal:
        values = list(get_args(annotation))
        fragment = {"enum": [_jsonable(v, warnings) for v in values]}
        inferred = _infer_type_from_values(values)
        if inferred is not None:
            fragment = {"type": inferred, **fragment}  # 混型则省略 type（§7.1.1）
        return fragment, warnings

    # ---- 嵌套 dataclass（先于"其它类"的兜底分支）----
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _dataclass_to_schema(annotation, depth=depth, seen=seen, warnings=warnings)

    # ---- pydantic BaseModel（仅当可用；鸭子类型识别，不 import pydantic）----
    if _is_pydantic_model(annotation):
        return _pydantic_model_to_schema(annotation, warnings=warnings), warnings

    # ---- 容器 ----
    # 裸 `list` / 裸 `typing.List`：get_origin(typing.List) is list 但 get_args 为空
    if annotation is list or (origin is list and not get_args(annotation)):
        return {"type": "array"}, warnings
    if origin is list:
        items, sub_warnings = annotation_to_schema(
            get_args(annotation)[0], depth=depth + 1, seen=seen
        )
        warnings.extend(sub_warnings)
        return _with_items(items, unique=False), warnings

    if annotation is set or annotation is frozenset or (
        origin in (set, frozenset) and not get_args(annotation)
    ):
        return {"type": "array", "uniqueItems": True}, warnings
    if origin in (set, frozenset):
        items, sub_warnings = annotation_to_schema(
            get_args(annotation)[0], depth=depth + 1, seen=seen
        )
        warnings.extend(sub_warnings)
        return _with_items(items, unique=True), warnings

    if annotation is tuple or (origin is tuple and not get_args(annotation)):
        return {"type": "array"}, warnings
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:  # tuple[X, ...] 变长元组
            items, sub_warnings = annotation_to_schema(args[0], depth=depth + 1, seen=seen)
            warnings.extend(sub_warnings)
            return _with_items(items, unique=False), warnings
        # tuple[X, Y] 定长元组：不支持 prefixItems，降级为通用数组 + minItems/maxItems
        warnings.append(
            f"fixed-length tuple {annotation!r} has no draft2020 prefixItems support; "
            "degraded to a generic array with minItems/maxItems"
        )
        return {"type": "array", "items": {}, "minItems": len(args), "maxItems": len(args)}, warnings

    if annotation is dict or (origin is dict and not get_args(annotation)):
        return {"type": "object"}, warnings
    if origin is dict:
        # 只支持 dict[str, X]（JSON 对象的键恒为字符串，非 str 键类型直接忽略）
        args = get_args(annotation)
        if len(args) == 2:
            value_schema, sub_warnings = annotation_to_schema(args[1], depth=depth + 1, seen=seen)
            warnings.extend(sub_warnings)
        else:  # pragma: no cover - 3.10 下 dict[X] 只有一个参数，保守兜底
            value_schema = {}
        fragment = {"type": "object"}
        if value_schema:
            fragment["additionalProperties"] = value_schema
        return fragment, warnings

    # ---- Union / Optional ----
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != len(args):  # Optional[X]
            if len(non_none) == 1:
                # 冻结决策 D-05：**不生成** "type": ["X", "null"]，Optional 只表达"非必填"。
                # 校验器对非必填字段收到显式 None 时直接放行（validate_instance）。
                return annotation_to_schema(non_none[0], depth=depth, seen=seen)
            warnings.append(
                f"Optional union {annotation!r} has multiple non-None members; "
                "degraded to {} (any)"
            )
            return {}, warnings
        warnings.append(
            f"union type {annotation!r} is unsupported (multi-type schemas break "
            "OpenAI/Anthropic-compatible grammar converters); degraded to {} (any)"
        )
        return {}, warnings

    # ---- 其它（自定义类、Callable、bytes、datetime ...）----
    warnings.append(f"unsupported annotation {annotation!r}; degraded to {{}} (any)")
    return {}, warnings


def _with_items(items: dict[str, Any], *, unique: bool) -> dict[str, Any]:
    """构造 array fragment；items 为空时省略该键（`{}` 是空约束，写了只是噪声）。"""
    fragment: dict[str, Any] = {"type": "array"}
    if items:
        fragment["items"] = items
    if unique:
        fragment["uniqueItems"] = True
    return fragment


def _is_pydantic_model(annotation: Any) -> bool:
    """鸭子类型识别 pydantic 模型类（不 import pydantic）。"""
    if not PYDANTIC_AVAILABLE or not isinstance(annotation, type):
        return False
    module = getattr(type(annotation), "__module__", "") or ""
    return module.startswith("pydantic") and hasattr(annotation, "model_json_schema")


def _dataclass_to_schema(
    cls: type,
    *,
    depth: int,
    seen: frozenset[type],
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """嵌套 dataclass -> 递归对象 schema（§7.1.2）。"""
    name = getattr(cls, "__name__", repr(cls))
    if cls in seen:  # 环检测：当前递归路径上已经出现过这个类型
        warnings.append(f"recursive dataclass {name} truncated at depth {depth}")
        return {}, warnings

    # 解析字段注解。`from __future__ import annotations` 下 f.type 是字符串，
    # 必须靠 get_type_hints 才能拿到真正的类型对象（include_extras 保 Annotated）。
    field_types: dict[str, Any] = {}
    try:
        field_types = get_type_hints(cls, include_extras=True)
    except Exception as exc:  # noqa: BLE001 - NameError/TypeError 都可能，统一降级
        warnings.append(
            f"cannot resolve type hints of dataclass {name} "
            f"({type(exc).__name__}: {exc}); string annotations degrade to {{}} (any)"
        )

    properties: dict[str, Any] = {}
    required: list[str] = []
    nested_seen = seen | {cls}
    for field in dataclasses.fields(cls):
        annotation = field_types.get(field.name, field.type)
        fragment, sub_warnings = annotation_to_schema(
            annotation, depth=depth + 1, seen=nested_seen
        )
        warnings.extend(sub_warnings)
        # §7.1.2 规则 4：嵌套字段的描述**只**来自 Annotated[..., Param(description=...)]，
        # 不去解析嵌套 dataclass 自己的 docstring Args 段（结构不可靠且引入解析歧义）。
        # SPEC-AMBIGUITY：规则 4 的"否则取字段名"有两种读法 —— 写成 description=字段名，
        # 或"只是保留字段名、不写 description"。这里取后者，理由：
        #   1. 规则 5 明文要求环检测/深度超限时"返回 {}"，写 description 会与它直接冲突；
        #   2. 属性名本身就是字段名，`{"description": "x"}` 对模型零信息量、纯 token 噪声；
        #   3. §7.1.4 的终点约定是"不写 description 键"，这里保持一致。
        properties[field.name] = fragment
        # 默认值规则与顶层完全一致（§7.1.2 规则 3 / §7.1.5 的唯一公式）
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
            and not is_optional(annotation)
            and not any(
                _meta_default(m) is not _UNSET for m in annotated_metadata(annotation)
            )
        ):
            required.append(field.name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, warnings


def _pydantic_model_to_schema(cls: type, *, warnings: list[str]) -> dict[str, Any]:
    """pydantic 模型 -> 清洗后的 JSON Schema（内联 $defs、删 title，§7.1.2 规则 7）。"""
    name = getattr(cls, "__name__", repr(cls))
    try:
        raw = cls.model_json_schema()
    except Exception as exc:  # noqa: BLE001 - 用户的 pydantic 模型可能有自定义 hook
        warnings.append(
            f"pydantic model {name}.model_json_schema() failed "
            f"({type(exc).__name__}: {exc}); degraded to {{}} (any)"
        )
        return {}
    if not isinstance(raw, dict):
        warnings.append(
            f"pydantic model {name}.model_json_schema() returned "
            f"{type(raw).__name__}, not a dict; degraded to {{}} (any)"
        )
        return {}
    return _inline_refs(raw, warnings=warnings)


def _inline_refs(schema: dict[str, Any], *, warnings: list[str] | None = None) -> dict[str, Any]:
    """把 `{"$ref": "#/$defs/X"}` 就地内联成 `$defs["X"]`，并删除所有 `title` 键。

    为什么必须做：OpenAI/Anthropic 的 function calling 对 `$defs/$ref` 支持不一致，
    部分兼容端（vLLM/Ollama 的 JSON-schema-to-grammar）直接报错；模型看平铺 schema 也更准。
    环引用截断为 `{}`（不抛异常：schema 仍可用，只是该处变成"任意类型"）。
    `title` 删除理由：pydantic 会用字段名自动生成 title，对模型是冗余噪声。
    本函数**不 import pydantic**，只在真的存在 `$ref` 时被调用（无 pydantic 环境同样安全）。

    `warnings` 是可选的降级记录通道（§13 红线 12）：环引用/悬空引用属于降级，
    调用方传进来就能观测到；不传也不影响任何行为。
    """
    definitions = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(definitions, Mapping):  # pragma: no cover - 非 pydantic 来源的脏 schema
        definitions = {}

    def resolve(node: Any, seen_refs: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen_refs) for item in node]
        if not isinstance(node, dict):
            return node
        result: dict[str, Any] = {}
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
            key = ref.rsplit("/", 1)[-1]
            target = definitions.get(key)
            if key in seen_refs:
                if warnings is not None:
                    warnings.append(
                        f"recursive $ref {ref!r} truncated to {{}} (any) while inlining pydantic schema"
                    )
                result = {}
            elif not isinstance(target, dict):
                if warnings is not None:
                    warnings.append(f"dangling $ref {ref!r} degraded to {{}} (any)")
                result = {}
            else:
                result = resolve(target, seen_refs | {key})
        for key, value in node.items():
            if key in ("$ref", "$defs", "definitions", "title"):
                continue
            result[key] = resolve(value, seen_refs)
        result.pop("title", None)
        return result

    return resolve(schema, frozenset())


# ---------------------------------------------------------------------------
# docstring 解析（§7.1.4，冻结规则）
# ---------------------------------------------------------------------------

# Google 风格的 section 标题 / 结束标志 / 条目行（正则逐字来自 §7.1.4）
_GOOGLE_ARGS_RE = re.compile(r"^\s*(Args|Arguments|Parameters|参数)\s*:\s*$", re.IGNORECASE)
_GOOGLE_SECTION_END_RE = re.compile(
    r"^\s*(Returns|Raises|Yields|Examples|Example|Notes|Note|Attributes|返回|异常|示例)\s*:\s*$",
    re.IGNORECASE,
)
_GOOGLE_ENTRY_RE = re.compile(r"^\s*(\*{0,2}\w+)\s*(\(([^)]*)\))?\s*:\s*(.*)$")
_SPHINX_PARAM_RE = re.compile(r"^\s*:param\s+(\w+)\s*:\s*(.*)$")
_SPHINX_TYPE_RE = re.compile(r"^\s*:type\s+")
_SPHINX_FIELD_RE = re.compile(r"^\s*:(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _DocEntry:
    """一条逐参描述。`type_hint` 用于 §7.1.4 的"文档说是 optional 但签名无默认值"告警。"""

    name: str
    description: str
    type_hint: str = ""


def _squash(text: str) -> str:
    """把多行/多空格压缩成单个空格（summary 与逐参描述都用它）。"""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _prepare_doc_lines(doc: str) -> list[str]:
    """dedent + 去首尾空行（§7.1.4 步骤 2）。"""
    lines = textwrap.dedent(doc).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _doc_summary(lines: Sequence[str]) -> str:
    """summary = 到第一个空行（含）为止的所有行，行间用空格拼接后压缩空白。"""
    parts: list[str] = []
    for line in lines:
        if not line.strip():
            break
        parts.append(line.strip())
    return _squash(" ".join(parts))


def _parse_google_entries(
    lines: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Google 风格逐参解析：返回 (名字 -> 描述片段, 名字 -> 类型串)，保持出现顺序。"""
    start: int | None = None
    for index, line in enumerate(lines):
        if _GOOGLE_ARGS_RE.match(line):
            start = index
            break
    if start is None:
        return {}, {}

    header_indent = _indent_of(lines[start])
    parts: dict[str, list[str]] = {}
    hints: dict[str, str] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        # 同级或更低缩进的下一个 section 标题 = 段结束
        if line.strip() and _indent_of(line) <= header_indent and _GOOGLE_SECTION_END_RE.match(line):
            break
        match = _GOOGLE_ENTRY_RE.match(line)
        if match and line.strip():
            name = match.group(1).lstrip("*")  # 剥离 *args/**kwargs 的星号
            hints[name] = (match.group(3) or "").strip()
            parts[name] = [(match.group(4) or "").strip()]
            current = name
        elif current is not None and line.strip():
            parts[current].append(line.strip())  # 缩进更深的续行
    return parts, hints


def _parse_sphinx_entries(
    lines: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Sphinx 风格逐参解析：`:param x: desc`，续行为更深缩进；`:type x:` 忽略。"""
    parts: dict[str, list[str]] = {}
    current: str | None = None
    entry_indent = 0
    for line in lines:
        match = _SPHINX_PARAM_RE.match(line)
        if match:
            current = match.group(1)
            parts[current] = [match.group(2).strip()]
            entry_indent = _indent_of(line)
            continue
        if _SPHINX_TYPE_RE.match(line) or _SPHINX_FIELD_RE.match(line):
            current = None  # :type:/:returns:/:rtype: 等其它字段：忽略并终止当前条目
            continue
        if not line.strip():
            continue
        if current is not None and _indent_of(line) > entry_indent:
            parts[current].append(line.strip())
    return parts, {}


def _docstring_entries(
    doc: str | None, style: str
) -> tuple[str, dict[str, _DocEntry]]:
    """(summary, 逐参描述)。供 `parse_docstring` 与 `build_tool_schema` 共用。

    `build_tool_schema` 需要 `type_hint`（§7.1.4 的 optional 告警），
    而 `parse_docstring` 的冻结返回值只有描述字符串，所以这里放内部通道。
    """
    if not doc:
        return "", {}
    lines = _prepare_doc_lines(doc)
    summary = _doc_summary(lines)

    normalized = (style or "auto").strip().lower()
    if normalized == "none":
        return summary, {}

    if normalized == "google":
        parts, hints = _parse_google_entries(lines)
    elif normalized == "sphinx":
        parts, hints = _parse_sphinx_entries(lines)
    else:
        # "auto"（以及未识别的取值 —— 未知 style 不报错，按 auto 宽容处理）
        parts, hints = _parse_google_entries(lines)
        if not parts:
            parts, hints = _parse_sphinx_entries(lines)

    entries = {
        name: _DocEntry(name=name, description=_squash(" ".join(chunks)), type_hint=hints.get(name, ""))
        for name, chunks in parts.items()
    }
    return summary, entries


def parse_docstring(doc: str | None, *, style: str = "auto") -> tuple[str, dict[str, str]]:
    """返回 (summary, {param_name: description})。summary = 第一段落，压缩内部换行与多余空格。

    style="auto" 时先尝试 Google（Args:/Arguments:/参数:），失败再尝试 Sphinx（:param x:）。
    style="none" 只取 summary。

    SPEC-AMBIGUITY：§7.1.4 第 5/6 条要求"参数名不在签名里的条目丢弃 + warning"，
    但本函数的签名里没有 func/签名，拿不到参数名集合。因此**这里返回全部解析到的条目**，
    由 `build_tool_schema` 做签名过滤并记 warning（warning 的归属方在那边才成立）。
    """
    summary, entries = _docstring_entries(doc, style)
    return summary, {name: entry.description for name, entry in entries.items()}


# ---------------------------------------------------------------------------
# build_tool_schema（§7.1 主入口）
# ---------------------------------------------------------------------------


def _definition_error(tool_name: str, message: str) -> ToolDefinitionError:
    """构造 `ToolDefinitionError`。

    §3.3 只冻结了额外字段 `tool_name`；`message` 是基类的位置参数（§3.1）。
    errors.py 由另一个实现者编写，构造形态未被逐字冻结，所以这里运行时探测一次
    `message` 形参是否存在：存在就带上（日志可读），不存在就只传冻结字段。
    这不是吞异常 —— 探测失败只会让 message 缺失，异常照常抛出。
    """
    kwargs: dict[str, Any] = {"tool_name": tool_name}
    try:
        parameters = inspect.signature(ToolDefinitionError.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - 极端对象签名
        parameters = {}
    if "message" in parameters:
        kwargs["message"] = message
    return ToolDefinitionError(**kwargs)


def build_tool_schema(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    docstring_style: str = "auto",
) -> SchemaResult:
    """把函数的签名 + 注解 + docstring 反射成工具 JSON Schema。

    `parameters` 非 None 时直接短路返回（用户完全接管 schema，仍会跑一遍 `validate_schema`
    做自检，自检问题进 warnings —— 不抛异常，因为用户明确表示由自己负责）。
    """
    warnings: list[str] = []
    tool_name = name or getattr(func, "__name__", None) or type(func).__name__

    try:
        doc = inspect.getdoc(func)
    except Exception as exc:  # noqa: BLE001 - 某些代理对象（C 扩展 / 自定义 __doc__）会抛
        doc = None
        warnings.append(
            f"inspect.getdoc failed ({type(exc).__name__}: {exc}); "
            "the tool description falls back to the name"
        )
    summary, doc_entries = _docstring_entries(doc, docstring_style)
    descriptions = {pname: entry.description for pname, entry in doc_entries.items()}
    # 描述来源链（§7.1.4）：参数 description= > docstring summary > name > ""
    tool_description = description or summary or tool_name or ""

    if parameters is not None:
        for problem in validate_schema(parameters):
            warnings.append(f"user-supplied parameters self-check: {problem}")
        return SchemaResult(
            schema=parameters,
            descriptions=descriptions,
            warnings=warnings,
            name=tool_name,
            description=tool_description,
        )

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        # 拿不到签名就无法反射参数 —— 这属于"装饰器阶段无法生成 schema"（§3.3）
        raise _definition_error(
            tool_name,
            f"cannot introspect the signature of {tool_name!r} "
            f"({type(exc).__name__}: {exc}); pass parameters= to take over the schema",
        ) from exc

    hints: dict[str, Any] = {}
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:  # noqa: BLE001 - NameError 是主因（§7.2 第 2 条）
        warnings.append(
            f"typing.get_type_hints failed ({type(exc).__name__}: {exc}); "
            "falling back to __annotations__ (string annotations degrade to {} (any))"
        )
        raw = getattr(func, "__annotations__", None) or {}
        hints = dict(raw)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            # **kwargs：无法为任意键生成 schema。用户确实需要时必须显式传 parameters=（§7.1.5）
            raise _definition_error(
                tool_name,
                f"{tool_name!r} declares **{param_name}; arbitrary keys cannot be reflected "
                "into a JSON Schema. Pass parameters= explicitly to take over the schema.",
            )

        annotation = hints.get(param_name, param.annotation)

        is_varargs = param.kind is inspect.Parameter.VAR_POSITIONAL
        if is_varargs:
            # *args：映射为 array，**永不 required**（缺省即空数组）
            items, sub_warnings = annotation_to_schema(annotation, depth=0, seen=frozenset())
            warnings.extend(sub_warnings)
            fragment = _with_items(items, unique=False)
        else:
            fragment, sub_warnings = annotation_to_schema(annotation, depth=0, seen=frozenset())
            warnings.extend(sub_warnings)

        entry = doc_entries.get(param_name)
        if entry is not None:
            if (
                not is_varargs
                and "optional" in entry.type_hint.lower()
                and param.default is inspect.Parameter.empty
            ):
                # §7.1.4 第 3 条：文档里的 optional 只记 warning，**不改变 required 判定**
                warnings.append(
                    f"docstring declares {param_name!r} as {entry.type_hint!r} (optional) but the "
                    "signature has no default; required judgement is unchanged"
                )
            # 逐参描述优先级（§7.1.4）：Annotated Param(description=...) > docstring Args 段
            if entry.description and "description" not in fragment:
                fragment["description"] = entry.description
        properties[param_name] = fragment

        if is_varargs:
            continue

        # ---- required 的唯一公式（§7.1.5，v2）----
        metas = annotated_metadata(annotation)
        if (
            param.default is inspect.Parameter.empty
            and not is_optional(annotation)
            and not any(_meta_default(m) is not _UNSET for m in metas)
        ):
            required.append(param_name)

    # docstring 里写了但签名里没有的参数名 -> 丢弃 + warning（§7.1.4 第 5/6 条）
    for unknown in doc_entries:
        if unknown not in signature.parameters:
            warnings.append(
                f"docstring documents parameter {unknown!r} which is not in the signature; dropped"
            )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    for problem in validate_schema(schema):
        # 自检不该在正常路径上命中；一旦命中说明生成的 schema 有 bug，必须可见（红线 12）
        warnings.append(f"generated schema self-check: {problem}")

    return SchemaResult(
        schema=schema,
        descriptions=descriptions,
        warnings=warnings,
        name=tool_name,
        description=tool_description,
    )


# ---------------------------------------------------------------------------
# validate_instance：纯 stdlib 的 JSON Schema 子集校验器（§7.1）
# ---------------------------------------------------------------------------


def _matches_type(value: Any, expected: str) -> bool:
    """类型判定（冻结：bool 先于 int；float 不满足 "integer"）。"""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "object":
        return isinstance(value, Mapping)
    return True  # 未知 type 关键字 -> 宽容放行（外部 schema 可能用自定义类型名）


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_name(value: Any) -> str:
    """给模型看的类型名（尽量用 JSON Schema 的词汇，便于它自纠正）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_equal(left: Any, right: Any) -> bool:
    """JSON 语义的相等：bool 与 int 不相等（Python 里 `1 == True`）。

    enum/const 的判定必须用它，否则 `{"enum": [true]}` 会把 `1` 判成合法值。
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _canonical(value: Any) -> str:
    """uniqueItems/去重用的稳定键（True 与 1 必须不同）。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _allows_null(schema: dict[str, Any]) -> bool:
    """该 schema 是否显式允许 null（nullable / type: null / const null / enum 含 null）。"""
    if schema.get("nullable") is True:
        return True
    expected = schema.get("type")
    if expected == "null":
        return True
    if isinstance(expected, (list, tuple)) and "null" in expected:
        return True
    if "const" in schema and schema["const"] is None:
        return True
    enum_values = schema.get("enum")
    if isinstance(enum_values, (list, tuple)) and any(v is None for v in enum_values):
        return True
    return False


def _probe_valid(value: Any, sub_schema: Any, max_errors: int) -> bool:
    """anyOf/oneOf 的分支探测：只关心"该分支是否产生错误"，错误本身丢弃。"""
    probe: list[str] = []
    _validate_into(value, sub_schema, "$", probe, max_errors)
    return not probe


def _validate_into(
    value: Any,
    schema: Any,
    path: str,
    errors: list[str],
    max_errors: int,
) -> None:
    if len(errors) >= max_errors or not isinstance(schema, dict):
        # 非 dict 的 schema 一律忽略（宽容；schema 可能来自外部 provider 或 pydantic）
        return

    # nullable / type 含 "null" / const null / enum 含 null 时，None 不该再撞类型检查
    null_allowed = value is None and _allows_null(schema)

    expected = schema.get("type")
    if not null_allowed:
        if isinstance(expected, (list, tuple)):
            if not any(
                _matches_type(value, t) for t in expected if isinstance(t, str)
            ):
                errors.append(
                    f"{path}: expected {' or '.join(str(t) for t in expected)}, "
                    f"got {_type_name(value)}"
                )
                return
        elif isinstance(expected, str) and expected != "any":
            if not _matches_type(value, expected):
                errors.append(f"{path}: expected {expected}, got {_type_name(value)}")
                return

    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: expected const {_short(schema['const'])}, got {_short(value)}")
    enum_values = schema.get("enum")
    if isinstance(enum_values, (list, tuple)) and not any(
        _json_equal(value, v) for v in enum_values
    ):
        errors.append(f"{path}: {_short(value)} is not one of {_short(list(enum_values))}")

    # 组合关键字与值的类型无关，**必须**先于下面这条 null 早退（否则
    # `{"oneOf":[{"type":"integer"}]}` 收到 None 会被静默判为通过）
    _validate_combinators(value, schema, path, errors, max_errors)

    if value is None:
        return  # null 已由 type/const/enum 判过；"非必填字段收到 None"在父级对象分支里放行

    if isinstance(value, Mapping):
        _validate_object(value, schema, path, errors, max_errors)
    elif isinstance(value, (list, tuple)):
        _validate_array(value, schema, path, errors, max_errors)
    elif isinstance(value, str):
        _validate_string(value, schema, path, errors)
    elif _is_number(value):
        _validate_number(value, schema, path, errors)


def _validate_object(
    value: Mapping[Any, Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
    max_errors: int,
) -> None:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    required_names = [n for n in required if isinstance(n, str)] if isinstance(required, (list, tuple)) else []

    for name in required_names:
        if name not in value:
            errors.append(f"{path}: missing required property {name!r}")

    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        child_path = f"{path}.{key}"
        if key in properties:
            sub_schema = properties[key]
            if isinstance(sub_schema, dict):
                # D-05 的配套规则（§7.1）：非必填字段（不在 required 且 schema 无 default）
                # 收到显式 None 时**直接放行** —— 因为 Optional 只表达"非必填"，
                # 不生成 "type": ["X","null"]，所以 null 不是类型错误。
                if (
                    item is None
                    and key not in required_names
                    and "default" not in sub_schema
                    and not _allows_null(sub_schema)
                ):
                    continue
                _validate_into(item, sub_schema, child_path, errors, max_errors)
            else:
                _validate_into(item, {}, child_path, errors, max_errors)
        elif additional is False:
            errors.append(f"{path}: additional property {key!r} is not allowed")
        elif isinstance(additional, dict):
            _validate_into(item, additional, child_path, errors, max_errors)
        if len(errors) >= max_errors:
            return


def _validate_array(
    value: Sequence[Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
    max_errors: int,
) -> None:
    items = schema.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            _validate_into(item, items, f"{path}[{index}]", errors, max_errors)
            if len(errors) >= max_errors:
                return
    elif isinstance(items, (list, tuple)):  # draft-07 的位置化 items，宽容支持
        for index, (item, sub_schema) in enumerate(zip(value, items)):
            _validate_into(item, sub_schema, f"{path}[{index}]", errors, max_errors)
            if len(errors) >= max_errors:
                return

    minimum = schema.get("minItems")
    if isinstance(minimum, int) and not isinstance(minimum, bool) and len(value) < minimum:
        errors.append(f"{path}: expected at least {minimum} items, got {len(value)}")
    maximum = schema.get("maxItems")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and len(value) > maximum:
        errors.append(f"{path}: expected at most {maximum} items, got {len(value)}")
    if schema.get("uniqueItems") is True and len({_canonical(v) for v in value}) != len(value):
        errors.append(f"{path}: array items must be unique")


def _validate_string(
    value: str, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    minimum = schema.get("minLength")
    if isinstance(minimum, int) and not isinstance(minimum, bool) and len(value) < minimum:
        errors.append(f"{path}: string is shorter than minLength {minimum}")
    maximum = schema.get("maxLength")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and len(value) > maximum:
        errors.append(f"{path}: string is longer than maxLength {maximum}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            if re.search(pattern, value) is None:
                errors.append(f"{path}: string does not match pattern {pattern!r}")
        except re.error as exc:
            # schema 自己的正则是坏的：这是作者 bug，必须可见（不能静默吞，§13 红线 10）
            errors.append(f"{path}: schema pattern {pattern!r} is not a valid regex ({exc})")


def _validate_number(
    value: float, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    minimum = schema.get("minimum")
    if _is_number(minimum) and value < minimum:
        errors.append(f"{path}: {value} is less than the minimum {minimum}")
    maximum = schema.get("maximum")
    if _is_number(maximum) and value > maximum:
        errors.append(f"{path}: {value} is greater than the maximum {maximum}")
    exclusive_minimum = schema.get("exclusiveMinimum")
    if _is_number(exclusive_minimum) and value <= exclusive_minimum:
        errors.append(f"{path}: {value} must be greater than {exclusive_minimum}")
    exclusive_maximum = schema.get("exclusiveMaximum")
    if _is_number(exclusive_maximum) and value >= exclusive_maximum:
        errors.append(f"{path}: {value} must be less than {exclusive_maximum}")


def _validate_combinators(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
    max_errors: int,
) -> None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, (list, tuple)) and any_of:
        if not any(_probe_valid(value, sub, max_errors) for sub in any_of):
            errors.append(f"{path}: {_short(value)} matches none of the {len(any_of)} anyOf schemas")
    one_of = schema.get("oneOf")
    if isinstance(one_of, (list, tuple)) and one_of:
        # 冻结：oneOf **取第一个通过的分支**（不做"恰好一个"的严格语义，
        # 因为 provider 生成的 schema 常常并不严格互斥）
        if not any(_probe_valid(value, sub, max_errors) for sub in one_of):
            errors.append(f"{path}: {_short(value)} matches none of the {len(one_of)} oneOf schemas")


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
    max_errors: int = 32,
) -> list[str]:
    """纯 stdlib 的 JSON Schema 子集校验器。返回错误消息列表，空列表 = 通过。

    支持的关键字：type, enum, const, properties, required, additionalProperties(bool 或 schema),
                 items, minItems, maxItems, uniqueItems, minimum, maximum,
                 exclusiveMinimum, exclusiveMaximum, minLength, maxLength, pattern,
                 anyOf, oneOf(取第一个通过的分支), nullable。
    不支持的 key 一律**忽略**（宽容），且不产生错误 —— schema 可能来自外部 provider 或
    pydantic，严格失败会造成假阳性。
    路径表示：'$.a.b[0]'。
    令牌判定：bool 必须在 int 之前判定（isinstance(True, int) 为 True）；
    数字：int 满足 "number"；float **不满足** "integer"（冻结：保持实现简单可预测）。
    非必填字段（不在 required 且 schema 无 default）收到显式 None 时直接放行（D-05 的配套）。
    """
    errors: list[str] = []
    _validate_into(instance, schema, path, errors, max_errors)
    return errors


# ---------------------------------------------------------------------------
# unvalidatable_parameters：schema 的"校验覆盖率缺口"（降级成 `{}` 的参数）
# ---------------------------------------------------------------------------


#: `validate_instance` 真正会**读**的关键字。判据 `unvalidatable_parameters` 依赖它：
#: 一个片段只要和这个集合**不相交**，校验器对它就没有任何可查之处 —— 恒真。
#: 放在校验器同一个文件里是刻意的：改校验器时这张表就在眼皮底下（否则迟早漂移）。
_VALIDATING_KEYWORDS: frozenset[str] = frozenset({
    "type", "const", "enum", "nullable",
    "anyOf", "oneOf",
    "properties", "required", "additionalProperties",
    "items", "minItems", "maxItems", "uniqueItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern",
})

#: 递归进嵌套 object 片段的深度上限（防环 + 防病态 schema）。
_MAX_COVERAGE_DEPTH = 4


def unvalidatable_parameters(schema: Mapping[str, Any] | None) -> list[str]:
    """返回 `schema` 里**无法在执行前校验**的参数名（片段降级成 `{}`，即 any）。

    判据：`properties[name]` 与 `_VALIDATING_KEYWORDS` **不相交**。典型形态是空 dict `{}`
    —— `annotation_to_schema` 对无法精确表达的注解（`Union[A, B]` / `Mapping[...]` /
    `Callable[...]` / `get_type_hints` 失败后的字符串注解…）唯一的兜底产物，
    并在 `ToolSpec.warnings` 里留下 `degraded to {} (any)` 的原文。
    **但不能只判"是空 dict"**：`_function_to_schema` 随后会把 docstring 里的
    `description` 贴进同一个 dict，于是降级片段会变成 `{"description": "..."}`
    —— 依旧一个校验关键字都没有，依旧恒真。

    为什么这个函数必须存在（§13 红线 12）：schema 生成阶段把降级记进了
    `ToolSpec.warnings`，而**执行阶段**看不见它。调用方看到 `validate_instance` 返回
    `[]` 会以为"参数都校验过了"，实际上这些参数的**类型与取值永远不会被拦下**
    （README §边界 已经这么写了，但只有文档、没有运行时信号）。执行器用本函数
    把这份缺口变成 WARNING + `ToolResult.metadata`，让"降级"在 trace 里可见。

    嵌套 object 片段会以点号路径（如 `payload.inner`）一并列出，深度上限
    `_MAX_COVERAGE_DEPTH`。schema 不是 object schema / 没有 properties 时返回 `[]`。
    """
    if not isinstance(schema, Mapping):
        return []
    names: list[str] = []
    _collect_unvalidatable(schema, "", names, seen=set(), depth=0)
    return names


def _collect_unvalidatable(
    schema: Mapping[str, Any],
    prefix: str,
    out: list[str],
    *,
    seen: set[int],
    depth: int,
) -> None:
    """`unvalidatable_parameters` 的递归实现（就地往 `out` 里追加，保持 properties 顺序）。"""
    if depth > _MAX_COVERAGE_DEPTH or id(schema) in seen:
        return
    seen.add(id(schema))
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return
    for name, fragment in properties.items():
        if not isinstance(name, str):
            continue
        path = f"{prefix}{name}"
        if not isinstance(fragment, Mapping):
            # 非 dict 的片段在校验器里被当成 `{}`（宽容放行）—— 同样"无法校验"。
            out.append(path)
            continue
        if _VALIDATING_KEYWORDS.isdisjoint(fragment):
            out.append(path)
            continue
        # 有校验关键字，但嵌套 object 自己的字段可能仍然降级了 —— 继续下钻。
        _collect_unvalidatable(fragment, f"{path}.", out, seen=seen, depth=depth + 1)


# ---------------------------------------------------------------------------
# validate_schema：只校验"我们自己生成"的 schema 是否自洽（§7.1）
# ---------------------------------------------------------------------------


def _check_schema_node(
    schema: Any,
    path: str,
    problems: list[str],
    seen: set[int],
    *,
    is_root: bool = False,
) -> None:
    if not isinstance(schema, dict):
        problems.append(f"{path}: schema node must be a dict, got {type(schema).__name__}")
        return
    if id(schema) in seen:  # 自引用 schema（用户传进来的）不能无限递归
        return
    seen.add(id(schema))

    if is_root:
        expected = schema.get("type")
        if expected is not None and expected != "object":
            problems.append(
                f"{path}: tool parameters must be an object schema, got type={expected!r}"
            )

    required = schema.get("required")
    properties = schema.get("properties")
    # 只有 properties **存在**（哪怕是空对象）时才核对 required 的名字：
    # 合法的 JSON Schema 允许 `required` 与 `properties` 并存于不同分支，
    # 无 properties 的 required 并不非法，报出来是假阳性。
    properties_present = "properties" in schema and isinstance(properties, Mapping)
    property_names = set(properties) if isinstance(properties, Mapping) else set()
    if required is not None:
        if not isinstance(required, (list, tuple)):
            problems.append(f"{path}: 'required' must be a list, got {type(required).__name__}")
        else:
            seen_names: set[str] = set()
            for name in required:
                if not isinstance(name, str):
                    problems.append(
                        f"{path}: 'required' entries must be strings, got {type(name).__name__}"
                    )
                    continue
                if name in seen_names:
                    problems.append(f"{path}: 'required' lists {name!r} twice")
                seen_names.add(name)
                if properties_present and name not in property_names:
                    problems.append(
                        f"{path}: 'required' names {name!r} which is not in 'properties'"
                    )

    if properties is not None and not isinstance(properties, Mapping):
        problems.append(f"{path}: 'properties' must be an object, got {type(properties).__name__}")
    elif isinstance(properties, Mapping):
        for name, sub_schema in properties.items():
            if not isinstance(sub_schema, dict):
                problems.append(f"{path}.properties.{name}: schema node must be a dict")
            else:
                _check_schema_node(sub_schema, f"{path}.properties.{name}", problems, seen)

    enum_values = schema.get("enum")
    if isinstance(enum_values, (list, tuple)) and len(enum_values) == 0:
        problems.append(f"{path}: 'enum' must not be empty")

    items = schema.get("items")
    if isinstance(items, dict):
        _check_schema_node(items, f"{path}.items", problems, seen)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _check_schema_node(additional, f"{path}.additionalProperties", problems, seen)
    for keyword in ("anyOf", "oneOf"):
        subschemas = schema.get(keyword)
        if isinstance(subschemas, (list, tuple)):
            for index, sub_schema in enumerate(subschemas):
                if isinstance(sub_schema, dict):
                    _check_schema_node(sub_schema, f"{path}.{keyword}[{index}]", problems, seen)


def validate_schema(schema: dict[str, Any]) -> list[str]:
    """校验我们**自己生成**的 schema 是否合法（内部自检用，如 required 里的名字必须出现在
    properties 中）。返回问题列表。仅开发/测试期调用。"""
    problems: list[str] = []
    _check_schema_node(schema, "$", problems, set(), is_root=True)
    return problems


# ---------------------------------------------------------------------------
# provider 适配（§7.1，供 ToolSpec.to_openai_schema / to_anthropic_schema 调用）
# ---------------------------------------------------------------------------


def to_openai_tool(spec: "ToolSpec") -> dict[str, Any]:
    """{"type":"function","function":{"name","description","parameters"}}"""
    # 顶层做一次浅拷贝：调用方（provider / registry）可能顺手往里加 "strict" 之类的键，
    # 不该污染 ToolSpec 持有的那份 schema。
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": dict(spec.parameters),
        },
    }


def to_anthropic_tool(spec: "ToolSpec") -> dict[str, Any]:
    """{"name","description","input_schema": parameters}"""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": dict(spec.parameters),
    }
