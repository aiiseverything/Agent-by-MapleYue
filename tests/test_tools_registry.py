from __future__ import annotations

# ``tests/test_tools_registry.py`` —— ``liteagent/tools/registry.py`` 的单元测试。
#
# 覆盖重点逐字来自 ``docs/INTERFACES.md`` §12 第 5114 行：
#
# * 注册 / 重名 / 别名 / ``subset`` 未命中抛错 / ``schemas`` 两种格式 /
#   ``to_prompt`` 截断（默认 20）/ ``merge`` 不改自身 / 非法工具名；
# * **[v2 新增] 四条 auto_register 用例**：``@tool(auto_register=True)`` 后
#   ``get_default_registry().names()`` 含该工具；同名再注册抛 ``ToolDefinitionError``；
#   ``reset_default_registry()`` 后 ``names() == []``；裸 ``@tool`` 装饰后 ``names()`` 不变。
#
# 测试卫生（§12.1 冻结规则 1）：本文件所有用例的 ``setUp``/``tearDown`` 都调用
# ``reset_default_registry()`` —— 默认注册表是**进程级全局可变状态**，不隔离就会与
# 并行运行的其他测试文件互相污染。

import json
import unittest

from liteagent.config import DEFAULT_MAX_TOOLS_IN_PROMPT
from liteagent.errors import ConfigError, ToolDefinitionError, ToolNotFoundError
from liteagent.tools.base import Tool, is_tool, make_function_tool, tool
from liteagent.tools.registry import (
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)
from tests.helpers import add, boom, echo, make_registry


def _anonymous() -> str:
    """A tool used only to exercise the registry (no arguments)."""
    return "anonymous"


def _make_tools(count: int) -> list[Tool]:
    """造 ``count`` 个动态工具（``to_prompt`` 的截断用例需要 > 20 个）。"""
    tools: list[Tool] = []
    for index in range(count):
        tools.append(
            make_function_tool(
                name=f"tool_{index:02d}",
                description=f"Tool number {index}.",
                parameters={"type": "object", "properties": {}},
                func=lambda args, index=index: index,
            )
        )
    return tools


class RegistryTestBase(unittest.TestCase):
    """所有用例共享的隔离：**默认注册表是全局状态**，进出都必须清干净。"""

    def setUp(self) -> None:
        reset_default_registry()

    def tearDown(self) -> None:
        reset_default_registry()


# ======================================================================================
# 注册 / 重名 / 非法名字
# ======================================================================================


class RegisterTests(RegistryTestBase):
    """``register`` / ``register_function`` / ``unregister``。"""

    def test_register_returns_tool_and_tracks_names(self) -> None:
        registry = make_registry(add, echo)
        self.assertEqual(registry.names(), ["add", "echo"])
        self.assertEqual(len(registry), 2)
        self.assertIs(registry.get("add"), add)

    def test_duplicate_name_raises_without_override(self) -> None:
        """重名**不静默覆盖**：工具名是模型与框架之间的契约。"""
        registry = make_registry(add)
        with self.assertRaises(ToolDefinitionError):
            registry.register(add)
        # 原工具还在
        self.assertIs(registry.get("add"), add)

    def test_override_replaces(self) -> None:
        registry = make_registry(add)
        replacement = Tool.from_function(_anonymous, name="add", description="replacement")
        registry.register(replacement, override=True)
        self.assertIs(registry.get("add"), replacement)
        self.assertEqual(len(registry), 1)

    def test_register_rejects_non_tool(self) -> None:
        """传裸函数是最常见的手滑 -> ToolDefinitionError（不是 AttributeError）。"""
        registry = ToolRegistry()
        with self.assertRaises(ToolDefinitionError):
            registry.register(_anonymous)  # type: ignore[arg-type]

    def test_register_function_convenience(self) -> None:
        registry = ToolRegistry()
        registered = registry.register_function(_anonymous, name="anon", description="d")
        self.assertIsInstance(registered, Tool)
        self.assertEqual(registry.names(), ["anon"])

    def test_invalid_tool_names_rejected(self) -> None:
        """合法模式冻结为 ``^[A-Za-z_][A-Za-z0-9_.-]{0,63}$``。"""
        for bad_name in ("1tool", "bad name", "with/slash", "x" * 65, "emoji🙂"):
            bad_tool = Tool.from_function(_anonymous, name=bad_name, description="d")
            with self.assertRaises(ToolDefinitionError, msg=f"{bad_name!r} should be rejected"):
                ToolRegistry().register(bad_tool)

    def test_empty_name_is_rejected_by_register(self) -> None:
        """空串是 falsy，`Tool.from_function(name="")` 会回退到函数名；

        真要造一个空名工具只能直接构造 spec，注册表必须拒绝它。
        """
        self.assertEqual(
            Tool.from_function(_anonymous, name="", description="d").name, "_anonymous"
        )
        empty = Tool(add.spec.__class__(name="", description="d", parameters={}, func=_anonymous))
        with self.assertRaises(ToolDefinitionError):
            ToolRegistry().register(empty)

    def test_valid_special_names_accepted(self) -> None:
        """点号与短横线是合法字符（`fs.read` / `fs-read` 这类命名）。"""
        for good_name in ("fs.read", "fs-read", "_private", "A1"):
            good_tool = Tool.from_function(_anonymous, name=good_name, description="d")
            registry = ToolRegistry()
            registry.register(good_tool)
            self.assertIn(good_name, registry.names())

    def test_unregister_removes_tool_and_dangling_aliases(self) -> None:
        """注销时清理**指向它**的别名：留下悬空别名会把错误指向错误的方向。"""
        registry = make_registry(add, echo)
        registry.alias("sum", "add")
        registry.unregister("add")
        self.assertEqual(registry.names(), ["echo"])
        self.assertNotIn("sum", registry)
        self.assertIsNone(registry.try_get("sum"))

    def test_unregister_drops_aliases_pointing_to_it(self) -> None:
        """[v3 补测] 直接钉住**别名表状态**，而不是只看查找结果。

        v2 的 `test_unregister_removes_tool_and_dangling_aliases` 只断言
        `try_get("sum") is None` —— 而悬空别名下它也返回 None（`_tools["add"]` 已没了），
        所以那条断言对别名表毫无观测力：删掉 `target == name` 这个子条件后，
        `to_dict()["aliases"]` 会残留 `{"sum": "add"}`，随后用 `sum` 当工具名注册
        会抛 `ToolDefinitionError`（用户可见的行为回归），而 1618 个测试全绿。
        更坏的是"静默改绑"：unregister 后注册一个**同名但不同实现**的新工具，
        悬空别名会悄悄指向新实现，调用方无任何报错地分派到错误的工具。
        """
        registry = make_registry(add, echo)
        registry.alias("sum", "add")
        registry.unregister("add")
        # 直接观测别名表（公开 API：to_dict）
        self.assertEqual(registry.to_dict()["aliases"], {})
        # 行为侧：旧别名可以立刻被当成一个真工具名重新注册
        registry.register_function(_anonymous, name="sum")
        self.assertEqual(sorted(registry.names()), ["echo", "sum"])

    def test_unregister_does_not_rebind_a_dangling_alias_to_a_new_tool(self) -> None:
        """[v3 补测] 悬空别名不得静默改绑到 unregister 之后新注册的同名工具。

        这是比"抛 ToolDefinitionError"更危险的一侧：模型/调用方在没有可见报错的
        情况下被分派到一个**完全不同的实现**。
        """
        registry = make_registry(add, echo)
        registry.alias("sum", "add")
        registry.unregister("add")
        registry.register_function(_anonymous, name="add")  # 同名、不同实现
        self.assertIsNone(registry.try_get("sum"))

    def test_unregister_drops_chained_aliases(self) -> None:
        """别名是扁平化的，所以 unregister 一跳就是终态（链上的别名全部消失）。"""
        registry = make_registry(add, echo)
        registry.alias("read", "add")
        registry.alias("cat", "read")
        registry.unregister("add")
        self.assertEqual(registry.to_dict()["aliases"], {})
        self.assertIsNone(registry.try_get("cat"))

    def test_is_tool_predicate_is_covered(self) -> None:
        """[v3 补测] §7.2 冻结并导出的公开谓词 `is_tool` 必须有**行为**覆盖。

        v2 的 1618 个测试里 `is_tool` 的调用点数为 0（只有字符串出现在
        `__all__` / `APPENDIX_B_NAMES` 这类清单里，守门测试只做 `getattr` 可解析性）——
        把它的函数体改坏不会有任何断言转红。
        """
        self.assertTrue(is_tool(add))
        self.assertFalse(is_tool(lambda a: a))
        self.assertFalse(is_tool(add.spec))       # ToolSpec 是 dataclass，不是 Tool
        self.assertFalse(is_tool(42))
        # 顶层懒导出必须与 tools.base 里的是同一个对象
        import liteagent

        self.assertIs(liteagent.is_tool, is_tool)
        self.assertIs(liteagent.tools.is_tool, is_tool)

    def test_unregister_unknown_raises(self) -> None:
        registry = make_registry(add)
        with self.assertRaises(ToolNotFoundError):
            registry.unregister("nope")


# ======================================================================================
# 查询：get / try_get / 别名 / list / subset
# ======================================================================================


class LookupTests(RegistryTestBase):
    """名字解析与过滤视图。"""

    def test_get_missing_lists_available(self) -> None:
        """未命中必须带上 `available`：executor 靠它回灌给模型做自纠正。"""
        registry = make_registry(add, echo)
        with self.assertRaises(ToolNotFoundError) as ctx:
            registry.get("nope")
        self.assertEqual(ctx.exception.name, "nope")
        self.assertEqual(ctx.exception.available, ["add", "echo"])

    def test_try_get_returns_none(self) -> None:
        registry = make_registry(add)
        self.assertIsNone(registry.try_get("nope"))
        self.assertIs(registry.try_get("add"), add)

    def test_alias_resolution_and_names_exclusion(self) -> None:
        """别名解析在 get/try_get 里做，**别名不进 names()**（§7.3 冻结）。"""
        registry = make_registry(add, echo)
        registry.alias("sum", "add")
        self.assertIs(registry.get("sum"), add)
        self.assertIs(registry.try_get("sum"), add)
        self.assertIn("sum", registry)
        self.assertEqual(registry.names(), ["add", "echo"])
        self.assertEqual(len(registry), 2)
        self.assertNotIn("sum", registry.to_dict()["names"])
        self.assertEqual(registry.to_dict()["aliases"], {"sum": "add"})

    def test_alias_to_unknown_target_raises(self) -> None:
        registry = make_registry(add)
        with self.assertRaises(ToolNotFoundError):
            registry.alias("sum", "nope")

    def test_alias_cannot_shadow_tool_or_be_invalid(self) -> None:
        registry = make_registry(add)
        with self.assertRaises(ToolDefinitionError):
            registry.alias("add", "add")  # 与真实工具名撞车
        with self.assertRaises(ToolDefinitionError):
            registry.alias("1bad", "add")

    def test_aliases_via_constructor(self) -> None:
        """别名必须在工具注册**之后**处理，否则构造期就会因目标不存在而炸。"""
        registry = ToolRegistry([add], aliases={"sum": "add"})
        self.assertIs(registry.get("sum"), add)

    def test_subset_returns_new_registry_with_dedup(self) -> None:
        registry = make_registry(add, echo, boom)
        subset = registry.subset(["add", "add", "echo"])
        self.assertIsNot(subset, registry)
        self.assertEqual(subset.names(), ["add", "echo"])
        # 原名不变（返回的是新注册表）
        self.assertEqual(registry.names(), ["add", "boom", "echo"])

    def test_subset_missing_name_raises_early(self) -> None:
        """未命中任一名字 -> ToolNotFoundError（早失败，不做"部分成功"）。"""
        registry = make_registry(add, echo)
        with self.assertRaises(ToolNotFoundError):
            registry.subset(["add", "nope"])

    def test_list_filters_by_tags_and_dangerous(self) -> None:
        safe = Tool.from_function(_anonymous, name="safe", description="d", tags=("files",))
        risky = Tool.from_function(
            _anonymous, name="risky", description="d", tags=("shell",), dangerous=True
        )
        registry = make_registry(safe, risky)
        self.assertEqual([t.name for t in registry.list()], ["risky", "safe"])
        self.assertEqual(
            [t.name for t in registry.list(include_dangerous=False)], ["safe"]
        )
        self.assertEqual([t.name for t in registry.list(tags=["shell"])], ["risky"])
        self.assertEqual([t.name for t in registry.list(tags=["files", "shell"])], ["risky", "safe"])
        self.assertEqual(registry.list(tags=["nope"]), [])

    def test_iteration_follows_sorted_names(self) -> None:
        """`__iter__` 按 names() 顺序 -> schema 导出不随注册顺序抖动。"""
        registry = make_registry(echo, add)
        self.assertEqual([t.name for t in registry], ["add", "echo"])

    def test_contains_rejects_non_str(self) -> None:
        registry = make_registry(add)
        self.assertIn("add", registry)
        self.assertNotIn(1, registry)
        self.assertNotIn(None, registry)


# ======================================================================================
# 导出视图：schemas / to_prompt / describe / to_dict / merge
# ======================================================================================


class ExportTests(RegistryTestBase):
    """四种导出视图与合并。"""

    def test_schemas_openai_and_anthropic(self) -> None:
        registry = make_registry(add, echo)
        openai = registry.schemas()
        self.assertEqual([item["function"]["name"] for item in openai], ["add", "echo"])
        self.assertTrue(all(item["type"] == "function" for item in openai))

        anthropic = registry.schemas(fmt="anthropic")
        self.assertEqual([item["name"] for item in anthropic], ["add", "echo"])
        self.assertTrue(all("input_schema" in item for item in anthropic))

    def test_schemas_unknown_format_raises(self) -> None:
        """未知 fmt -> ConfigError（不静默回退到 openai）。"""
        registry = make_registry(add)
        with self.assertRaises(ConfigError):
            registry.schemas(fmt="gemini")
        with self.assertRaises(ConfigError):
            registry.to_prompt(fmt="yaml")
        with self.assertRaises(ConfigError):
            registry.describe("add", fmt="html")

    def test_to_prompt_text_default_max_tools_is_20(self) -> None:
        """[v2 变更] 默认 max_tools 从 64 降到 20（DEFAULT_MAX_TOOLS_IN_PROMPT）。"""
        self.assertEqual(DEFAULT_MAX_TOOLS_IN_PROMPT, 20)
        registry = ToolRegistry(_make_tools(25))
        text = registry.to_prompt()
        lines = text.splitlines()
        self.assertEqual(len(lines), 21)  # 20 行工具 + 1 行省略提示
        self.assertEqual(lines[-1], "... (5 more tools omitted)")
        self.assertEqual(registry.to_prompt(max_tools=25).count("\n"), 24)
        self.assertNotIn("omitted", registry.to_prompt(max_tools=25))

    def test_to_prompt_text_lines_are_summary_lines(self) -> None:
        registry = make_registry(add)
        self.assertEqual(registry.to_prompt(), add.spec.summary_line())
        self.assertIn("add(a: integer, b: integer)", registry.to_prompt())

    def test_to_prompt_json_is_valid_json_with_marker_element(self) -> None:
        """json 分支把省略提示作为数组的**最后一个字符串元素**（否则不是合法 JSON）。"""
        registry = ToolRegistry(_make_tools(23))
        payload = json.loads(registry.to_prompt(fmt="json"))
        self.assertEqual(len(payload), 21)
        self.assertEqual(payload[-1], "... (3 more tools omitted)")
        self.assertEqual(payload[0]["name"], "tool_00")

    def test_to_prompt_rejects_negative_max_tools(self) -> None:
        """负数会被切片悄悄变成"砍掉尾部"，静默少给工具 -> 明确拒绝。"""
        registry = make_registry(add)
        with self.assertRaises(ConfigError):
            registry.to_prompt(max_tools=-1)
        self.assertEqual(registry.to_prompt(max_tools=0), "... (1 more tools omitted)")

    def test_describe_markdown_and_json(self) -> None:
        registry = make_registry(add)
        markdown = registry.describe("add")
        self.assertIn("### add", markdown)
        self.assertIn("| parameter | type | required | default | description |", markdown)
        self.assertIn("| a | integer | yes | - |", markdown)
        self.assertIn("```json", markdown)

        payload = json.loads(registry.describe("add", fmt="json"))
        self.assertEqual(payload["name"], "add")
        self.assertEqual(payload["parameters"], add.parameters)

    def test_merge_does_not_mutate_self(self) -> None:
        """`merge` 返回**新** registry（不改自身）。"""
        left = make_registry(add)
        right = make_registry(echo)
        merged = left.merge(right)
        self.assertIsNot(merged, left)
        self.assertIsNot(merged, right)
        self.assertEqual(merged.names(), ["add", "echo"])
        self.assertEqual(left.names(), ["add"])
        self.assertEqual(right.names(), ["echo"])

    def test_merge_conflict_and_override(self) -> None:
        left = make_registry(add)
        right = make_registry(add)
        with self.assertRaises(ToolDefinitionError):
            left.merge(right)
        merged = left.merge(right, override=True)
        self.assertEqual(merged.names(), ["add"])

    def test_merge_does_not_merge_aliases(self) -> None:
        """别名是"某个注册表的本地习惯"，跨表合并时静默择一会制造幽灵 bug。"""
        left = make_registry(add)
        right = ToolRegistry([echo], aliases={"say": "echo"})
        merged = left.merge(right)
        self.assertNotIn("say", merged)
        self.assertEqual(merged.get("echo"), echo)

    def test_to_dict_is_json_serialisable(self) -> None:
        registry = make_registry(add)
        registry.alias("sum", "add")
        payload = registry.to_dict()
        self.assertEqual(payload["names"], ["add"])
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["aliases"], {"sum": "add"})
        self.assertEqual([t["name"] for t in payload["tools"]], ["add"])
        # 能真的序列化（trace / CLI 路径不会炸）
        json.dumps(payload, ensure_ascii=False)

    def test_names_returns_a_fresh_list(self) -> None:
        """红线 11：查询返回副本，改返回值不可能改到注册表内部状态。"""
        registry = make_registry(add, echo)
        names = registry.names()
        names.append("injected")
        self.assertEqual(registry.names(), ["add", "echo"])


# ======================================================================================
# [v2 新增] 四条 auto_register 用例
# ======================================================================================


class AutoRegisterTests(RegistryTestBase):
    """`@tool(auto_register=True)` 的冻结语义（§7.2）。"""

    def test_auto_register_true_adds_to_default_registry(self) -> None:
        @tool(auto_register=True)
        def auto_tool(x: int) -> int:
            """Auto registered."""
            return x

        self.assertIn("auto_tool", get_default_registry().names())
        self.assertIs(get_default_registry().get("auto_tool"), auto_tool)

    def test_auto_register_duplicate_raises(self) -> None:
        """重名 -> ToolDefinitionError（**不静默覆盖**）。"""

        @tool(auto_register=True)
        def dup_tool(x: int) -> int:
            """First."""
            return x

        with self.assertRaises(ToolDefinitionError):

            @tool(auto_register=True)
            def dup_tool(x: int) -> int:  # noqa: F811 - 故意的同名重注册
                """Second."""
                return x

        # 第一次注册的那个还在
        self.assertIs(get_default_registry().get("dup_tool"), dup_tool)

    def test_reset_default_registry_replaces_with_empty_registry(self) -> None:
        """冻结语义：**替换为新对象**而不是清空原对象（旧引用不该看到空表）。"""

        @tool(auto_register=True)
        def before_reset(x: int) -> int:
            """Before."""
            return x

        old = get_default_registry()
        self.assertEqual(old.names(), ["before_reset"])
        reset_default_registry()
        self.assertEqual(get_default_registry().names(), [])
        # 旧引用**没有**被清空 —— 这正是"替换"而不是 ".clear()" 的意义
        self.assertEqual(old.names(), ["before_reset"])
        self.assertIsNot(get_default_registry(), old)

    def test_bare_tool_does_not_touch_global_registry(self) -> None:
        """裸 `@tool`（auto_register 默认 False）**不得触碰任何全局状态**。"""
        before = list(get_default_registry().names())

        @tool
        def bare_tool(x: int) -> int:
            """Bare."""
            return x

        self.assertEqual(get_default_registry().names(), before)
        self.assertNotIn("bare_tool", get_default_registry().names())
        # 装饰本身仍然是完整的（只是没注册）
        self.assertEqual(bare_tool.name, "bare_tool")
        self.assertEqual(bare_tool.parameters["required"], ["x"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
