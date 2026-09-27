from __future__ import annotations

# ``tests/test_tools_schema.py`` —— ``liteagent/tools/schema.py`` 的单元测试。
#
# 覆盖重点逐字来自 ``docs/INTERFACES.md`` §12 第 5113 行（工具层最重的两个文件之一）：
#
# * §7.1.1 映射表**逐行**断言（bool 先于 int、Literal、Enum、list[dict]、嵌套 dataclass、
#   递归截断、深度上限、set/tuple、dict[str,X]、Union 降级 warning、Optional 不进 required、
#   ``*args``、``**kwargs`` 抛 ``ToolDefinitionError``）；
# * §7.1.3 ``Param`` 元数据、pydantic ``FieldInfo`` 鸭子类型、``Annotated[str, "desc"]``；
# * §7.1.4 docstring Google/Sphinx/无 Args、summary 提取；
# * §7.1.5 **三条 Annotated required 用例**；
# * ``validate_instance`` 各关键字；``validate_schema`` 自检；两个 provider 适配器。
#
# 测试卫生（§12.1 冻结规则）：零网络、零真实时钟依赖、零真实 sleep；不碰默认注册表
# （本文件不需要 ``auto_register``，因此不需要 ``reset_default_registry``）。

import dataclasses
import enum
import unittest
from dataclasses import dataclass
from typing import Annotated, Any, Callable, List, Literal, Optional, Union

from liteagent.errors import ToolDefinitionError
from liteagent.tools.schema import (
    MAX_SCHEMA_DEPTH,
    PYDANTIC_AVAILABLE,
    SCHEMA_DRAFT,
    Param,
    SchemaResult,
    annotated_metadata,
    annotation_to_schema,
    build_tool_schema,
    is_optional,
    parse_docstring,
    to_anthropic_tool,
    to_openai_tool,
    validate_instance,
    validate_schema,
)
from tests.helpers import add as add_tool


# ======================================================================================
# 模块级 fixture 类型（必须定义在模块层：get_type_hints 只认模块全局名）
# ======================================================================================


class Color(enum.Enum):
    """普通 Enum：值全是字符串。"""

    RED = "red"
    GREEN = "green"


class Level(enum.IntEnum):
    """IntEnum：值是 int，type 推断为 integer。"""

    LOW = 1
    HIGH = 2


@dataclass
class Point:
    """嵌套 dataclass 的**内层**类型（x 必填、y 有默认值）。"""

    x: int
    y: int = 0


@dataclass
class Shape:
    """嵌套 dataclass 的**外层**类型。"""

    origin: Point
    name: str


@dataclass
class Bundle:
    """[v3 补测] 含 `field(default_factory=...)` 字段的嵌套 dataclass。

    §7.1.2 规则 3 的 required 公式对嵌套字段有两个默认值来源：`field.default is MISSING`
    与 `field.default_factory is MISSING`。v2 的测试只覆盖了前者（普通默认值），
    删掉 `default_factory` 那一行后 `field(default_factory=list)` 这类**可选**字段会被
    判成 required（生成的 JSON Schema 会告诉模型"这个字段必须提供"），
    而 1618 个测试照样全绿。
    """

    tags: List[str] = dataclasses.field(default_factory=list)
    note: str = dataclasses.field(default_factory=lambda: "n/a")
    plain: int = 0


@dataclass
class FancyPoint:
    """字段用 Annotated 携带描述的嵌套 dataclass。"""

    x: Annotated[int, Param(description="the x coordinate")] = 0
    y: int = 1


@dataclass
class TreeNode:
    """自引用 dataclass：用于测环检测。"""

    value: int
    child: Optional["TreeNode"] = None


class Custom:
    """既不是 dataclass 也没有 schema 映射的自定义类。"""


@dataclass
class _Payload:
    """端到端用例用的载荷类型（必须模块级：``get_type_hints`` 只认模块全局名）。"""

    name: str
    size: int = 1


# pydantic 是**可选**依赖：写成 try/except ImportError 的可选依赖形态（§1.3 的写法），
# 并且模型类必须定义在模块层 —— 本文件开启了 ``from __future__ import annotations``，
# 局部定义的类/局部 import 的名字在 ``get_type_hints`` 里解析不到（NameError），
# 注解会被降级成 ``{}``，用例就变成了假测试。
try:  # pragma: no cover - 环境相关
    import pydantic as _pydantic
except ImportError:  # pragma: no cover
    _pydantic = None  # type: ignore[assignment]


if _pydantic is not None:  # pragma: no cover - 环境相关
    class _PydanticModel(_pydantic.BaseModel):
        """模块级 pydantic 模型：内联 $defs / 删 title 的用例需要它。"""

        x: int
        y: str = "a"


# ======================================================================================
# §7.1.1 映射表逐行断言
# ======================================================================================


class MappingTableTests(unittest.TestCase):
    """§7.1.1 的映射表：每一行一个断言。"""

    def test_scalar_types(self) -> None:
        """str/int/float/bool 四行。"""
        self.assertEqual(annotation_to_schema(str), ({"type": "string"}, []))
        self.assertEqual(annotation_to_schema(int), ({"type": "integer"}, []))
        self.assertEqual(annotation_to_schema(float), ({"type": "number"}, []))
        self.assertEqual(annotation_to_schema(bool), ({"type": "boolean"}, []))

    def test_bool_precedes_int(self) -> None:
        """**bool 必须先于 int 判定**（isinstance(True, int) is True）。

        这条不是洁癖：若顺序写反，`bool` 会被映射成 `{"type": "integer"}`，
        模型就会收到"传 0/1"的暗示，而工具签名要的是 true/false。
        """
        fragment, warnings = annotation_to_schema(bool)
        self.assertEqual(fragment, {"type": "boolean"})
        self.assertNotEqual(fragment.get("type"), "integer")
        self.assertEqual(warnings, [])

    def test_none_and_any_and_bare_annotation(self) -> None:
        """None / type(None) / Any / 无注解 -> `{}`（任意类型），且不留 warning。"""
        self.assertEqual(annotation_to_schema(None), ({}, []))
        self.assertEqual(annotation_to_schema(type(None)), ({}, []))
        self.assertEqual(annotation_to_schema(Any), ({}, []))

    def test_literal_of_strings_and_ints(self) -> None:
        """Literal["a","b"] 与 Literal[1,2]：type 由字面量值的类型推断。"""
        self.assertEqual(
            annotation_to_schema(Literal["a", "b"]),
            ({"type": "string", "enum": ["a", "b"]}, []),
        )
        self.assertEqual(
            annotation_to_schema(Literal[1, 2]),
            ({"type": "integer", "enum": [1, 2]}, []),
        )

    def test_literal_of_booleans(self) -> None:
        """Literal[True, False] -> `{"enum": [true,false], "type": "boolean"}`。"""
        fragment, warnings = annotation_to_schema(Literal[True, False])
        self.assertEqual(fragment, {"type": "boolean", "enum": [True, False]})
        self.assertEqual(warnings, [])

    def test_literal_mixed_omits_type(self) -> None:
        """Literal["a", 1] 混型：省略 type 键（不能猜成 string 或 integer）。"""
        fragment, warnings = annotation_to_schema(Literal["a", 1])
        self.assertEqual(fragment, {"enum": ["a", 1]})
        self.assertNotIn("type", fragment)
        self.assertEqual(warnings, [])

    def test_enum_and_intenum(self) -> None:
        """Enum 子类与 IntEnum 子类：成员顺序 = 定义顺序，type 由值推断。"""
        self.assertEqual(
            annotation_to_schema(Color),
            ({"type": "string", "enum": ["red", "green"]}, []),
        )
        self.assertEqual(
            annotation_to_schema(Level),
            ({"type": "integer", "enum": [1, 2]}, []),
        )

    def test_list_and_list_of_dict(self) -> None:
        """list[int] 与 list[dict]（后者是最常见的"对象数组"写法）。"""
        self.assertEqual(
            annotation_to_schema(list[int]),
            ({"type": "array", "items": {"type": "integer"}}, []),
        )
        self.assertEqual(
            annotation_to_schema(list[dict]),
            ({"type": "array", "items": {"type": "object"}}, []),
        )
        self.assertEqual(
            annotation_to_schema(List[str]),
            ({"type": "array", "items": {"type": "string"}}, []),
        )

    def test_bare_containers(self) -> None:
        """裸 list / 裸 dict / 裸 tuple -> 无 items / additionalProperties 的容器。"""
        self.assertEqual(annotation_to_schema(list), ({"type": "array"}, []))
        self.assertEqual(annotation_to_schema(dict), ({"type": "object"}, []))
        self.assertEqual(annotation_to_schema(tuple), ({"type": "array"}, []))

    def test_set_and_frozenset_get_unique_items(self) -> None:
        """set[X] / frozenset[X] -> array + uniqueItems: true。"""
        self.assertEqual(
            annotation_to_schema(set[int]),
            ({"type": "array", "items": {"type": "integer"}, "uniqueItems": True}, []),
        )
        self.assertEqual(
            annotation_to_schema(frozenset[str]),
            ({"type": "array", "items": {"type": "string"}, "uniqueItems": True}, []),
        )

    def test_tuple_variadic(self) -> None:
        """tuple[X, ...] -> 变长数组。"""
        self.assertEqual(
            annotation_to_schema(tuple[int, ...]),
            ({"type": "array", "items": {"type": "integer"}}, []),
        )

    def test_tuple_fixed_length_degraded_with_warning(self) -> None:
        """tuple[X, Y] 降级：不支持 prefixItems -> generic array + minItems/maxItems。"""
        fragment, warnings = annotation_to_schema(tuple[int, str])
        self.assertEqual(
            fragment, {"type": "array", "items": {}, "minItems": 2, "maxItems": 2}
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("prefixItems", warnings[0])

    def test_dict_str_to_value(self) -> None:
        """dict[str, X] -> object + additionalProperties:<X>。"""
        self.assertEqual(
            annotation_to_schema(dict[str, int]),
            ({"type": "object", "additionalProperties": {"type": "integer"}}, []),
        )

    def test_optional_is_unwrapped_and_not_nullable(self) -> None:
        """Optional[X] -> `<X>`，**不生成** `type: ["X","null"]`（冻结决策 D-05）。"""
        fragment, warnings = annotation_to_schema(Optional[int])
        self.assertEqual(fragment, {"type": "integer"})
        self.assertNotIsInstance(fragment.get("type"), list)
        self.assertEqual(warnings, [])

    def test_union_of_two_types_degrades_with_warning(self) -> None:
        """非 Optional 的 Union[A, B] -> `{}` + warning（多态不支持，降级为任意）。"""
        fragment, warnings = annotation_to_schema(Union[int, str])
        self.assertEqual(fragment, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("unsupported", warnings[0])

    def test_unsupported_annotations_degrade_with_warning(self) -> None:
        """自定义类 / bytes / Callable 一律降级为 `{}` 并记 warning（红线 12）。"""
        for annotation in (Custom, bytes, Callable[[int], int]):
            fragment, warnings = annotation_to_schema(annotation)
            self.assertEqual(fragment, {}, msg=f"{annotation!r} should degrade to {{}}")
            self.assertTrue(warnings, msg=f"{annotation!r} should leave a warning")

    def test_unresolved_string_annotation_degrades(self) -> None:
        """字符串注解（get_type_hints 失败的兜底路径）-> `{}` + warning。"""
        fragment, warnings = annotation_to_schema("SomeUnresolvedName")
        self.assertEqual(fragment, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("unresolved string annotation", warnings[0])

    def test_schema_draft_constant(self) -> None:
        """§7.1 冻结的 draft 常量。"""
        self.assertEqual(SCHEMA_DRAFT, "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(MAX_SCHEMA_DEPTH, 8)


class NestedDataclassTests(unittest.TestCase):
    """§7.1.2 嵌套 dataclass：递归对象、环检测、深度上限。"""

    def test_nested_dataclass_object_schema(self) -> None:
        """嵌套 dataclass -> 递归 object schema，required 用同一条公式。"""
        fragment, warnings = annotation_to_schema(Shape)
        self.assertEqual(fragment["type"], "object")
        self.assertEqual(fragment["additionalProperties"], False)
        self.assertEqual(fragment["required"], ["origin", "name"])

        origin = fragment["properties"]["origin"]
        self.assertEqual(origin["type"], "object")
        self.assertEqual(origin["properties"], {"x": {"type": "integer"}, "y": {"type": "integer"}})
        # 内层 Point.y 有默认值 -> 不进 required
        self.assertEqual(origin["required"], ["x"])
        self.assertEqual(warnings, [])

    def test_default_factory_fields_are_not_required(self) -> None:
        """[v3 补测] `field(default_factory=...)` 是"有默认值"，绝不能进 required。

        这里的断言方向很重要：`default_factory` 字段被判 required 是**对外可见的
        schema 错误**（模型被要求提供本可省略的字段），而 v2 的套件对该分支零覆盖 ——
        删掉 schema.py 里那一行后所有测试仍全绿。
        """
        fragment, warnings = annotation_to_schema(Bundle)
        self.assertEqual(fragment["type"], "object")
        self.assertEqual(fragment["required"], [], msg=fragment)
        # 三个字段都在 properties 里（可选 != 不出现）
        self.assertEqual(sorted(fragment["properties"]), ["note", "plain", "tags"])
        self.assertEqual(warnings, [])

    def test_default_factory_branch_is_load_bearing_for_validation(self) -> None:
        """同一份 schema 也用于执行前校验：可选字段缺失时必须放行。"""

        def bundle_tool(bundle: Bundle) -> str:
            """Take a bundle.

            Args:
                bundle: the bundle.
            """
            return str(bundle)

        schema = build_tool_schema(bundle_tool).schema
        nested = schema["properties"]["bundle"]
        self.assertEqual(nested["required"], [])
        self.assertEqual(validate_instance({"bundle": {}}, schema), [])

    def test_nested_dataclass_field_description_from_param(self) -> None:
        """嵌套字段描述优先取 `Annotated[..., Param(description=...)]`（§7.1.2 规则 4）。"""
        fragment, _ = annotation_to_schema(FancyPoint)
        self.assertEqual(
            fragment["properties"]["x"]["description"], "the x coordinate"
        )

    def test_recursive_dataclass_truncated_with_warning(self) -> None:
        """环检测：当前递归路径上出现过的类型 -> `{}` + 'recursive dataclass ...'。"""
        fragment, warnings = annotation_to_schema(TreeNode)
        self.assertEqual(fragment["properties"]["child"], {})
        self.assertTrue(
            any("recursive dataclass TreeNode" in w for w in warnings), msg=warnings
        )
        self.assertTrue(any("truncated" in w for w in warnings), msg=warnings)

    def test_depth_cap(self) -> None:
        """深度上限：depth > MAX_SCHEMA_DEPTH -> `{}` + warning（不是抛异常）。"""
        # 直接调：depth 超过上限时立刻降级，与注解本身无关
        fragment, warnings = annotation_to_schema(list[int], depth=MAX_SCHEMA_DEPTH + 1)
        self.assertEqual(fragment, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("MAX_SCHEMA_DEPTH", warnings[0])

        # 递归路径：外层注解一层层包下去，最内层必然被截断
        deep: Any = int
        for _ in range(MAX_SCHEMA_DEPTH + 2):
            deep = list[deep]
        fragment, warnings = annotation_to_schema(deep)
        self.assertTrue(any("MAX_SCHEMA_DEPTH" in w for w in warnings), msg=warnings)

        node: Any = fragment
        hops = 0
        while isinstance(node, dict) and isinstance(node.get("items"), dict):
            node = node["items"]
            hops += 1
        # `_with_items` 对空 items 省略该键，所以最内层恰好是 {"type": "array"}
        self.assertEqual(hops, MAX_SCHEMA_DEPTH)
        self.assertEqual(node, {"type": "array"})


# ======================================================================================
# §7.1.3 Annotated 元数据 / §7.1.5 required 公式
# ======================================================================================


class AnnotatedTests(unittest.TestCase):
    """``Annotated`` 的三条识别路径与 Param 的字段映射。"""

    def test_annotated_metadata_uses_dunder_metadata(self) -> None:
        """M-1：**禁止** `isinstance(ann, Annotated)`（3.10 恒 False），只能用 `__metadata__`。"""
        annotation = Annotated[int, "count"]
        # 先证明陷阱真实存在，再证明我们的实现没踩它。
        self.assertFalse(isinstance(annotation, Annotated))
        self.assertEqual(annotated_metadata(annotation), ("count",))
        self.assertEqual(annotated_metadata(int), ())

    def test_annotated_str_is_treated_as_description(self) -> None:
        """`Annotated[int, "desc"]` 这种常见写法直接当 description。"""
        fragment, warnings = annotation_to_schema(Annotated[str, "the text"])
        self.assertEqual(fragment, {"type": "string", "description": "the text"})
        self.assertEqual(warnings, [])

    def test_param_metadata_maps_every_keyword(self) -> None:
        """Param 的每个字段落到哪个 JSON Schema 关键字（§7.1.3 表格）。"""
        annotation = Annotated[
            int,
            Param(
                description="d",
                ge=1,
                le=9,
                gt=0,
                lt=10,
                min_length=2,
                max_length=5,
                pattern="^x",
                examples=(7,),
                enum=(1, 2),
                title="ignored",
            ),
        ]
        fragment, warnings = annotation_to_schema(annotation)
        self.assertEqual(fragment["description"], "d")
        self.assertEqual(fragment["minimum"], 1)
        self.assertEqual(fragment["maximum"], 9)
        self.assertEqual(fragment["exclusiveMinimum"], 0)
        self.assertEqual(fragment["exclusiveMaximum"], 10)
        # int 不是 array -> min_length/max_length 落在 minLength/maxLength
        self.assertEqual(fragment["minLength"], 2)
        self.assertEqual(fragment["maxLength"], 5)
        self.assertEqual(fragment["pattern"], "^x")
        self.assertEqual(fragment["examples"], [7])
        self.assertEqual(fragment["enum"], [1, 2])
        # §7.1.3：title **会被丢弃**
        self.assertNotIn("title", fragment)
        self.assertEqual(warnings, [])

    def test_param_min_length_on_array_uses_min_items(self) -> None:
        """min_length/max_length 对 array 落在 minItems/maxItems。"""
        annotation = Annotated[list[int], Param(min_length=1, max_length=3)]
        fragment, _ = annotation_to_schema(annotation)
        self.assertEqual(fragment["minItems"], 1)
        self.assertEqual(fragment["maxItems"], 3)

    def test_param_default_is_jsonable_coerced(self) -> None:
        """Param(default=...) 里的 Enum / 不可序列化对象都会被转换并留痕。"""
        fragment, warnings = annotation_to_schema(
            Annotated[str, Param(default=Color.RED)]
        )
        self.assertEqual(fragment["default"], "red")
        self.assertEqual(warnings, [])

        fragment, warnings = annotation_to_schema(
            Annotated[str, Param(default=object())]
        )
        self.assertIsInstance(fragment["default"], str)
        self.assertTrue(any("not JSON serializable" in w for w in warnings), msg=warnings)

    def test_empty_param_enum_is_dropped_with_warning(self) -> None:
        """空 enum 会让 schema 非法（draft 要求至少一个元素）-> 丢弃 + warning。"""
        fragment, warnings = annotation_to_schema(Annotated[str, Param(enum=())])
        self.assertNotIn("enum", fragment)
        self.assertTrue(any("enum" in w for w in warnings), msg=warnings)

    def test_unsupported_metadata_object_warns(self) -> None:
        """其它类型的 metadata 对象：忽略并记 warning（识别顺序第 4 条）。"""
        fragment, warnings = annotation_to_schema(Annotated[int, object()])
        self.assertEqual(fragment, {"type": "integer"})
        self.assertTrue(any("unsupported Annotated metadata" in w for w in warnings))

    def test_annotated_required_case_one_param_default(self) -> None:
        """§7.1.5 必测第一条：`Annotated[int, Param(default=5)]` 不进 required 且带 default。"""

        def fn(x: Annotated[int, Param(default=5)]) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertNotIn("x", result.schema["required"])
        self.assertEqual(result.schema["properties"]["x"]["default"], 5)

    def test_annotated_required_case_two_bare_string(self) -> None:
        """§7.1.5 必测第二条：`Annotated[int, 'desc']` 只加 description，不改 required。"""

        def fn(x: Annotated[int, "count of items"]) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["required"], ["x"])
        self.assertEqual(
            result.schema["properties"]["x"]["description"], "count of items"
        )

    def test_annotated_required_case_three_description_beats_docstring(self) -> None:
        """§7.1.5 必测第三条：Param(description) 优先于 docstring 的 Args 段。"""

        def fn(x: Annotated[str, Param(description="from param")]) -> None:
            """S.

            Args:
                x: from docstring
            """

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["properties"]["x"]["description"], "from param")

    def test_docstring_description_used_when_param_has_none(self) -> None:
        """没有 Annotated 描述时，docstring 的 Args 段补上（优先级链的下游）。"""

        def fn(x: str) -> None:
            """S.

            Args:
                x: from docstring
            """

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["properties"]["x"]["description"], "from docstring")


class PydanticDuckTypingTests(unittest.TestCase):
    """§7.1.3 第 2 条：pydantic FieldInfo 的**鸭子类型**识别（绝不 import pydantic）。"""

    def test_fake_field_info_is_recognised(self) -> None:
        """用 `__module__` 前缀假装成 pydantic 的类型，覆盖鸭子类型分支（无需装 pydantic）。"""

        class _FakeFieldInfo:
            description = "duck typed"
            default = None
            ge = 2
            le = 8
            gt = None
            lt = None
            min_length = None
            max_length = None
            pattern = None
            examples = (3,)
            metadata = ()

        _FakeFieldInfo.__module__ = "pydantic.fields"

        fragment, warnings = annotation_to_schema(
            Annotated[int, _FakeFieldInfo()]
        )
        self.assertEqual(fragment["description"], "duck typed")
        self.assertEqual(fragment["minimum"], 2)
        self.assertEqual(fragment["maximum"], 8)
        self.assertEqual(fragment["examples"], [3])
        self.assertEqual(warnings, [])

    def test_non_pydantic_object_does_not_match_duck_type(self) -> None:
        """`hasattr(description/default)` 但模块不是 pydantic -> 不识别（只记 warning）。"""

        class _NotPydantic:
            description = "x"
            default = 1

        fragment, warnings = annotation_to_schema(Annotated[int, _NotPydantic()])
        self.assertEqual(fragment, {"type": "integer"})
        self.assertTrue(any("unsupported Annotated metadata" in w for w in warnings))

    @unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic is not installed in this environment")
    def test_real_pydantic_field_and_model(self) -> None:
        """真 pydantic：`Field(...)` 的约束与 `BaseModel` 的 schema 都被采纳。"""

        def fn(
            a: Annotated[int, _pydantic.Field(description="count", ge=1, le=10)],
            m: "_PydanticModel",
        ) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        props = result.schema["properties"]
        self.assertEqual(props["a"]["description"], "count")
        self.assertEqual(props["a"]["minimum"], 1)
        self.assertEqual(props["a"]["maximum"], 10)
        # BaseModel -> model_json_schema() 清洗后内联
        self.assertEqual(props["m"]["type"], "object")
        self.assertEqual(props["m"]["required"], ["x"])
        self.assertNotIn("$defs", props["m"])
        self.assertNotIn("title", props["m"])


# ======================================================================================
# §7.1.4 docstring -> description
# ======================================================================================


GOOGLE_DOC = """Do a thing.

    Longer explanation that is not part of the summary.

    Args:
        a (int): the first number
            continued on the next line
        b: the second number

    Returns:
        int: the sum
    """

SPHINX_DOC = """Sum two numbers.

    :param a: the first number
    :param b: the second number
    :type b: int
    :returns: the sum
    """


class DocstringTests(unittest.TestCase):
    """§7.1.4 冻结的 docstring 解析规则。"""

    def test_google_style_entries(self) -> None:
        """Google 风格的逐参描述，续行用空格拼接；Returns 段之后不再解析。"""
        summary, params = parse_docstring(GOOGLE_DOC)
        self.assertEqual(summary, "Do a thing.")
        self.assertEqual(
            params,
            {"a": "the first number continued on the next line", "b": "the second number"},
        )

    def test_google_style_forced(self) -> None:
        """style="google" 与 auto 在 Google 文档上必须一致。"""
        self.assertEqual(parse_docstring(GOOGLE_DOC), parse_docstring(GOOGLE_DOC, style="google"))

    def test_sphinx_style_entries(self) -> None:
        """:param x: 形态；`:type b:` 被忽略且不打断后续条目。"""
        summary, params = parse_docstring(SPHINX_DOC, style="sphinx")
        self.assertEqual(summary, "Sum two numbers.")
        self.assertEqual(
            params, {"a": "the first number", "b": "the second number"}
        )

    def test_auto_falls_back_to_sphinx(self) -> None:
        """style="auto" 先试 Google，失败再试 Sphinx。"""
        self.assertEqual(parse_docstring(SPHINX_DOC), parse_docstring(SPHINX_DOC, style="sphinx"))
        # 反过来：强制 google 时 Sphinx 文档解析不出条目
        self.assertEqual(parse_docstring(SPHINX_DOC, style="google")[1], {})

    def test_style_none_only_takes_summary(self) -> None:
        """style="none" -> 只取 summary，不做逐参解析。"""
        summary, params = parse_docstring(GOOGLE_DOC, style="none")
        self.assertEqual(summary, "Do a thing.")
        self.assertEqual(params, {})

    def test_docstring_without_args_section(self) -> None:
        """没有 Args 段时只有 summary，逐参描述为空 dict（不报错）。"""
        summary, params = parse_docstring("Just a summary.\n\nMore text.")
        self.assertEqual(summary, "Just a summary.")
        self.assertEqual(params, {})

    def test_none_and_empty_docstring(self) -> None:
        """None / 空串 -> ("", {})。"""
        self.assertEqual(parse_docstring(None), ("", {}))
        self.assertEqual(parse_docstring(""), ("", {}))

    def test_summary_stops_at_first_blank_line(self) -> None:
        """summary = 到第一个空行为止，行间单空格拼接、压缩连续空白。"""
        doc = "Line one\n    line two\n\n    Not part of summary.\n"
        summary, _ = parse_docstring(doc)
        self.assertEqual(summary, "Line one line two")

    def test_summary_single_paragraph_is_whole_doc(self) -> None:
        """整段就是 summary（没有空行）时，summary 就是全部内容。"""
        summary, params = parse_docstring("Only  a   summary here")
        self.assertEqual(summary, "Only a summary here")
        self.assertEqual(params, {})

    def test_docstring_documents_unknown_param_warns(self) -> None:
        """docstring 里写了签名中没有的参数 -> 丢弃 + warning（§7.1.4 第 5/6 条）。"""

        def fn(a: int) -> None:
            """S.

            Args:
                a: the a
                zzz: not in the signature
            """

        result = build_tool_schema(fn)
        self.assertNotIn("zzz", result.schema["properties"])
        self.assertTrue(
            any("zzz" in w and "not in the signature" in w for w in result.warnings),
            msg=result.warnings,
        )

    def test_docstring_optional_hint_only_warns(self) -> None:
        """文档写 optional 但签名无默认值 -> 记 warning，**不改变 required 判定**。"""

        def fn(a: int) -> None:
            """S.

            Args:
                a (int, optional): the a
            """

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["required"], ["a"])
        self.assertTrue(any("optional" in w for w in result.warnings), msg=result.warnings)

    def test_docstring_style_option_is_honoured_by_build_tool_schema(self) -> None:
        """`build_tool_schema(docstring_style=...)` 透传到解析器。"""

        def fn(a: int) -> None:
            """S.

            :param a: sphinx description
            """

        # auto：Google 段不存在 -> 回退 Sphinx
        self.assertEqual(
            build_tool_schema(fn).schema["properties"]["a"]["description"],
            "sphinx description",
        )
        # 强制 google：解析不出条目 -> 该参数没有 description 键
        self.assertNotIn(
            "description",
            build_tool_schema(fn, docstring_style="google").schema["properties"]["a"],
        )


# ======================================================================================
# build_tool_schema 主入口
# ======================================================================================


class BuildToolSchemaTests(unittest.TestCase):
    """签名反射、required 判定、`**kwargs` 拒绝、description 来源链。"""

    def test_required_only_for_missing_default_non_optional(self) -> None:
        """required 的唯一公式（§7.1.5）：无默认值 且 非 Optional 且 无 Param default。"""

        def fn(a: int, b: str = "x", c: Optional[int] = None) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["required"], ["a"])
        self.assertEqual(
            sorted(result.schema["properties"]), ["a", "b", "c"]
        )
        self.assertEqual(result.schema["type"], "object")
        self.assertIs(result.schema["additionalProperties"], False)

    def test_optional_positional_parameter_is_not_required(self) -> None:
        """`Optional[int]`（没有默认值）也不进 required（D-05：Optional 只表达非必填）。"""

        def fn(a: Optional[int], b: int) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["required"], ["b"])

    def test_pep604_union_with_none_is_optional(self) -> None:
        """`int | None`（PEP 604）与 `Optional[int]` 同义，同样不进 required。"""

        def fn(a: "int | None", b: int) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["required"], ["b"])

    def test_varargs_becomes_array_and_never_required(self) -> None:
        """`*args` -> array 且**永不** required（缺省即空数组）。"""

        def fn(*values: int) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(result.schema["properties"]["values"]["type"], "array")
        self.assertEqual(result.schema["properties"]["values"]["items"], {"type": "integer"})
        self.assertEqual(result.schema["required"], [])

    def test_varkwargs_raises_tool_definition_error(self) -> None:
        """`**kwargs` -> ToolDefinitionError（无法为任意键生成 schema）。"""

        def fn(a: int, **extra: Any) -> None:
            """Doc."""

        with self.assertRaises(ToolDefinitionError) as ctx:
            build_tool_schema(fn)
        self.assertIn("extra", str(ctx.exception))

    def test_user_supplied_parameters_short_circuits(self) -> None:
        """parameters= 非 None 时直接接管，仍跑一遍自检（问题进 warnings）。"""
        explicit = {"type": "object", "properties": {"q": {"type": "string"}}}
        result = build_tool_schema(lambda: None, name="x", description="d", parameters=explicit)
        self.assertIs(result.schema, explicit)
        self.assertEqual(result.name, "x")
        self.assertEqual(result.description, "d")
        self.assertEqual(result.warnings, [])

        bad = {"type": "object", "properties": {}, "required": ["missing"]}
        result = build_tool_schema(lambda: None, name="x", parameters=bad)
        self.assertTrue(
            any("self-check" in w for w in result.warnings), msg=result.warnings
        )

    def test_description_source_chain(self) -> None:
        """description 来源链：参数 description= > docstring summary > name > ""。"""

        def documented(a: int) -> None:
            """Summary from doc."""

        def undocumented(a: int) -> None:
            pass

        self.assertEqual(build_tool_schema(documented).description, "Summary from doc.")
        self.assertEqual(
            build_tool_schema(documented, description="explicit").description, "explicit"
        )
        self.assertEqual(build_tool_schema(undocumented).description, "undocumented")

    def test_schema_result_carries_descriptions_mapping(self) -> None:
        """SchemaResult.descriptions 是 {参数名: 描述}，且 name/description 齐全。"""

        def fn(a: int) -> None:
            """Summary.

            Args:
                a: the a
            """

        result = build_tool_schema(fn)
        self.assertIsInstance(result, SchemaResult)
        self.assertEqual(result.descriptions, {"a": "the a"})
        self.assertEqual(result.name, "fn")
        self.assertEqual(result.description, "Summary.")

    def test_generated_schema_passes_self_check(self) -> None:
        """正常路径生成的 schema 不该有自检问题（有就是实现 bug，必须可见）。"""

        def fn(a: int, b: Optional[str] = None) -> None:
            """Doc."""

        result = build_tool_schema(fn)
        self.assertEqual(validate_schema(result.schema), [])
        self.assertEqual(
            [w for w in result.warnings if "self-check" in w], []
        )


class IsOptionalTests(unittest.TestCase):
    """`is_optional` 的判定边界。"""

    def test_variants(self) -> None:
        self.assertTrue(is_optional(Optional[int]))
        self.assertTrue(is_optional(type(None)))
        self.assertTrue(is_optional(Union[int, None]))
        self.assertFalse(is_optional(int))
        self.assertFalse(is_optional(Union[int, str]))
        self.assertFalse(is_optional(Literal["a"]))
        # Annotated 先剥壳再判：`Annotated[int, "x"]` 的底层是 int -> 非 Optional
        self.assertFalse(is_optional(Annotated[int, "x"]))
        self.assertTrue(is_optional(Annotated[Optional[int], "x"]))

    def test_pep604_and_annotated(self) -> None:
        """PEP 604 的 `int | None` 与 `Annotated[int | None, ...]` 都算 Optional。

        §7.1 的散文定义只提了 `get_origin` 是 `typing.Union`，而 3.10 实测
        `int | None` 的 origin 是 `types.UnionType`。这里按 §7.1.1 "Optional[X]
        不进入 required" 的整体精神裁决为 True（已记入 ambiguities）。
        """
        self.assertTrue(is_optional(int | None))
        self.assertTrue(is_optional(Annotated[int | None, "x"]))
        self.assertFalse(is_optional(Annotated[int, "x"]))


# ======================================================================================
# validate_instance
# ======================================================================================


class ValidateInstanceTypeTests(unittest.TestCase):
    """类型关键字与 bool/int/number 的边界。"""

    def test_scalar_types(self) -> None:
        self.assertEqual(validate_instance("s", {"type": "string"}), [])
        self.assertEqual(validate_instance(5, {"type": "integer"}), [])
        self.assertEqual(validate_instance(5, {"type": "number"}), [])
        self.assertEqual(validate_instance(5.5, {"type": "number"}), [])
        self.assertEqual(validate_instance(True, {"type": "boolean"}), [])
        self.assertEqual(validate_instance([1], {"type": "array"}), [])
        self.assertEqual(validate_instance({"a": 1}, {"type": "object"}), [])
        self.assertEqual(validate_instance(None, {"type": "null"}), [])

    def test_bool_is_not_integer(self) -> None:
        """bool 先于 int：`True` 不是合法的 integer（否则 `1` 与 `true` 会被混为一谈）。"""
        self.assertTrue(validate_instance(True, {"type": "integer"}))
        self.assertTrue(validate_instance(True, {"type": "number"}))
        self.assertTrue(validate_instance(1, {"type": "boolean"}))

    def test_float_does_not_satisfy_integer(self) -> None:
        """冻结：float **不满足** "integer"（保持实现简单可预测）。"""
        self.assertTrue(validate_instance(5.0, {"type": "integer"}))

    def test_int_satisfies_number(self) -> None:
        self.assertEqual(validate_instance(7, {"type": "number"}), [])

    def test_unknown_type_keyword_is_permissive(self) -> None:
        """不认识的 type 名（外部 provider 的自定义类型）宽容放行。"""
        self.assertEqual(validate_instance(object(), {"type": "utopia"}), [])

    def test_empty_schema_accepts_everything(self) -> None:
        self.assertEqual(validate_instance(object(), {}), [])
        self.assertEqual(validate_instance(None, {}), [])


class ValidateInstanceValueTests(unittest.TestCase):
    """enum / const / 数值边界 / 字符串长度与 pattern / 数组 / 对象。"""

    def test_enum_and_const_use_json_semantics(self) -> None:
        """`1 == True` 在 Python 成立、在 JSON 不成立 —— 必须区分。"""
        self.assertEqual(validate_instance("a", {"enum": ["a", "b"]}), [])
        self.assertTrue(validate_instance("c", {"enum": ["a", "b"]}))
        self.assertTrue(validate_instance(True, {"enum": [1]}))
        self.assertTrue(validate_instance(1, {"enum": [True]}))
        self.assertEqual(validate_instance(5, {"const": 5}), [])
        self.assertTrue(validate_instance(True, {"const": 1}))

    def test_number_bounds(self) -> None:
        self.assertTrue(validate_instance(3, {"type": "integer", "minimum": 5}))
        self.assertTrue(validate_instance(9, {"type": "integer", "maximum": 8}))
        self.assertTrue(validate_instance(2, {"type": "integer", "exclusiveMinimum": 2}))
        self.assertTrue(validate_instance(2, {"type": "integer", "exclusiveMaximum": 2}))
        self.assertEqual(validate_instance(2, {"type": "integer", "minimum": 2}), [])
        # bool 是 int 的子类，但边界判定不该把它当数字
        self.assertEqual(validate_instance(True, {"type": "boolean", "minimum": 5}), [])

    def test_string_length_and_pattern(self) -> None:
        self.assertTrue(validate_instance("a", {"type": "string", "minLength": 2}))
        self.assertTrue(validate_instance("abc", {"type": "string", "maxLength": 2}))
        self.assertTrue(validate_instance("abc", {"type": "string", "pattern": "^b"}))
        self.assertEqual(validate_instance("abc", {"type": "string", "pattern": "^a"}), [])

    def test_invalid_pattern_is_reported_not_swallowed(self) -> None:
        """schema 自己的正则是坏的 -> 报错可见（红线 10：不得静默）。"""
        errors = validate_instance("x", {"type": "string", "pattern": "("})
        self.assertTrue(errors)
        self.assertIn("not a valid regex", errors[0])

    def test_array_items_and_bounds(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        self.assertEqual(validate_instance([1, 2], schema), [])
        errors = validate_instance([1, "x"], schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("[1]", errors[0])

        self.assertTrue(validate_instance([1], {"type": "array", "minItems": 2}))
        self.assertTrue(validate_instance([1, 2], {"type": "array", "maxItems": 1}))
        self.assertTrue(
            validate_instance([1, 1], {"type": "array", "uniqueItems": True})
        )
        self.assertEqual(
            validate_instance([True, 1], {"type": "array", "uniqueItems": True}), []
        )

    def test_nested_path_notation(self) -> None:
        """路径表示冻结为 '$.a.b[0]'，模型靠它定位到底错在哪个参数。"""
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {"b": {"type": "array", "items": {"type": "integer"}}},
                }
            },
        }
        errors = validate_instance({"a": {"b": [1, "x"]}}, schema)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("$.a.b[1]:"), msg=errors[0])


class ValidateInstanceObjectTests(unittest.TestCase):
    """required / additionalProperties / 非必填字段收到 None 的放行规则。"""

    def test_required_and_additional_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        self.assertEqual(validate_instance({"a": 1}, schema), [])
        errors = validate_instance({}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing required property 'a'", errors[0])

        errors = validate_instance({"a": 1, "z": 2}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("additional property 'z'", errors[0])

    def test_additional_properties_as_schema(self) -> None:
        """additionalProperties 是 schema 时校验每个额外值的类型。"""
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        self.assertEqual(validate_instance({"a": "x"}, schema), [])
        errors = validate_instance({"a": 1}, schema)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("$.a:"))

    def test_non_required_null_passes_through(self) -> None:
        """D-05 的配套：非必填且 schema 无 default 的字段收到显式 None 直接放行。"""
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        self.assertEqual(validate_instance({"a": None}, schema), [])
        # 进了 required 就不再放行
        required_schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        }
        self.assertTrue(validate_instance({"a": None}, required_schema))
        # schema 自己声明可空时也放行
        nullable_schema = {
            "type": "object",
            "properties": {"a": {"type": "integer", "nullable": True}},
        }
        self.assertEqual(validate_instance({"a": None}, nullable_schema), [])

    def test_nullable_and_multi_type_accept_none(self) -> None:
        self.assertEqual(validate_instance(None, {"type": "integer", "nullable": True}), [])
        self.assertEqual(validate_instance(None, {"type": ["integer", "null"]}), [])
        self.assertTrue(validate_instance(None, {"type": "integer"}))


class ValidateInstanceCombinatorTests(unittest.TestCase):
    """anyOf / oneOf（冻结：取第一个通过的分支）。"""

    def test_any_of(self) -> None:
        schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        self.assertEqual(validate_instance(1, schema), [])
        self.assertEqual(validate_instance("a", schema), [])
        self.assertTrue(validate_instance([1], schema))

    def test_one_of_accepts_first_matching_branch(self) -> None:
        """oneOf 不做"恰好一个"的严格语义，取第一个通过的分支。"""
        schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        self.assertEqual(validate_instance(1, schema), [])
        self.assertTrue(validate_instance([1], schema))

    def test_combinators_run_even_for_null(self) -> None:
        """组合关键字先于 null 早退：`{"oneOf":[{"type":"integer"}]}` 收到 None 必须报错。"""
        self.assertTrue(validate_instance(None, {"oneOf": [{"type": "integer"}]}))

    def test_unknown_keywords_are_ignored(self) -> None:
        """不支持的 key 一律忽略且不产生错误（schema 可能来自外部 provider）。"""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "patternProperties": {"^x": {"type": "string"}},
            "if": {"type": "object"},
            "dependencies": {"a": ["b"]},
        }
        self.assertEqual(validate_instance({"a": 1}, schema), [])

    def test_max_errors_cap(self) -> None:
        """max_errors 封顶：错误列表不会无限增长（模型侧只需要前几条）。"""
        schema = {
            "type": "object",
            "properties": {f"p{i}": {"type": "integer"} for i in range(10)},
        }
        value = {f"p{i}": "bad" for i in range(10)}
        self.assertEqual(len(validate_instance(value, schema, max_errors=3)), 3)
        self.assertEqual(len(validate_instance(value, schema)), 10)

    def test_path_argument_is_used_as_prefix(self) -> None:
        """path 参数决定错误消息的前缀。"""
        errors = validate_instance("x", {"type": "integer"}, path="$.args")
        self.assertTrue(errors[0].startswith("$.args:"))


# ======================================================================================
# validate_schema（自检）与 provider 适配器
# ======================================================================================


class ValidateSchemaTests(unittest.TestCase):
    """`validate_schema` 只校验"我们自己生成"的 schema 是否自洽。"""

    def test_required_name_must_exist_in_properties(self) -> None:
        problems = validate_schema(
            {"type": "object", "properties": {}, "required": ["missing"]}
        )
        self.assertTrue(any("not in 'properties'" in p for p in problems), msg=problems)

    def test_root_must_be_object(self) -> None:
        problems = validate_schema({"type": "array"})
        self.assertTrue(any("must be an object schema" in p for p in problems))

    def test_duplicate_required_and_empty_enum(self) -> None:
        problems = validate_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string", "enum": []}},
                "required": ["a", "a"],
            }
        )
        self.assertTrue(any("twice" in p for p in problems), msg=problems)
        self.assertTrue(any("'enum' must not be empty" in p for p in problems), msg=problems)

    def test_valid_schema_has_no_problems(self) -> None:
        self.assertEqual(
            validate_schema(
                {
                    "type": "object",
                    "properties": {"a": {"type": "array", "items": {"type": "string"}}},
                    "required": ["a"],
                    "additionalProperties": False,
                }
            ),
            [],
        )


class ProviderAdapterTests(unittest.TestCase):
    """`to_openai_tool` / `to_anthropic_tool` 的冻结形态。"""

    def test_openai_shape(self) -> None:
        payload = to_openai_tool(add_tool.spec)
        self.assertEqual(sorted(payload), ["function", "type"])
        self.assertEqual(payload["type"], "function")
        self.assertEqual(payload["function"]["name"], "add")
        self.assertEqual(
            payload["function"]["parameters"], add_tool.spec.parameters
        )

    def test_anthropic_shape(self) -> None:
        payload = to_anthropic_tool(add_tool.spec)
        self.assertEqual(sorted(payload), ["description", "input_schema", "name"])
        self.assertEqual(payload["name"], add_tool.name)
        self.assertEqual(payload["input_schema"], add_tool.spec.parameters)

    def test_parameters_dict_is_shallow_copied(self) -> None:
        """适配器做浅拷贝：调用方往里加 "strict" 不该污染 ToolSpec 持有的 schema。"""
        payload = to_openai_tool(add_tool.spec)
        self.assertIsNot(payload["function"]["parameters"], add_tool.spec.parameters)
        payload["function"]["parameters"]["injected"] = True
        self.assertNotIn("injected", add_tool.spec.parameters)


# ======================================================================================
# 端到端：装饰器 -> schema -> 校验
# ======================================================================================


class DecoratorToValidatorTests(unittest.TestCase):
    """把"反射 -> 校验"接起来跑一遍，证明两边用的是同一份契约。"""

    def test_generated_schema_accepts_valid_and_rejects_invalid_args(self) -> None:
        def deploy(payload: _Payload, force: bool = False) -> str:
            """Deploy something.

            Args:
                payload: the payload to deploy
                force: skip the safety check
            """

        result = build_tool_schema(deploy, docstring_style="google")
        schema = result.schema
        self.assertEqual(result.schema["required"], ["payload"])
        self.assertEqual(
            schema["properties"]["payload"]["properties"]["name"], {"type": "string"}
        )
        self.assertEqual(
            sorted(schema["properties"]["payload"]["required"]), ["name"]
        )
        self.assertEqual(
            validate_instance({"payload": {"name": "x"}}, schema), []
        )
        errors = validate_instance({"payload": {"name": 5}}, schema)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("$.payload.name:"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
