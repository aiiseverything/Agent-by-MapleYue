from __future__ import annotations

"""tests/test_builtin_files.py —— 内置文件工具（§7.5 ``files.py``；§12 第 5130 行）。

§12 第 5130 行列出的覆盖点（逐条都有对应用例）：
  * ``..`` 逃逸 -> ``SandboxViolationError``
  * 绝对路径逃逸 -> ``SandboxViolationError``
  * symlink 逃逸 -> ``SandboxViolationError``
  * ``allow_read_outside``（只放宽读，写永远不许越界）
  * ``delete_file`` 需 ``confirm=True``
  * ``metadata['sandbox']`` 是 **realpath**（§2.4 的 metadata 表把写入者冻结为 builtin/files）
  * ``make_file_tools(None)`` -> ``ConfigError``

安全相关的用例**真的在真实文件系统上跑**：沙箱必须真的挡住，而不是"看起来挡住了"。
所有临时文件都落在 ``tests.helpers.make_temp_sandbox`` 与 ``tempfile.TemporaryDirectory``
里，退出即清理（可并行运行、互不干扰）。
"""

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from liteagent.errors import ConfigError, SandboxViolationError, ToolValidationError
from liteagent.tools.base import Tool
from liteagent.tools.builtin.files import PathSandbox, make_file_tools
from tests.helpers import make_temp_sandbox

#: 工具名 -> 在 ``make_file_tools`` 返回值里的下标（§7.5 冻结顺序）。
_TOOL_ORDER = ("read_file", "write_file", "list_dir", "search_files", "delete_file")


def _by_name(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


@contextmanager
def _outside_dir() -> Iterator[Path]:
    """一个**沙箱之外**的临时目录（用于逃逸/allow_read_outside 用例）。"""
    with tempfile.TemporaryDirectory(prefix="liteagent-outside-") as tmp:
        yield Path(tmp)


class PathSandboxResolveTests(unittest.TestCase):
    """`PathSandbox` 的路径解析：`..` / 绝对路径 / symlink 三条逃逸路径。"""

    def test_dotdot_escape_raises_sandbox_violation(self) -> None:
        with make_temp_sandbox() as sandbox:
            with self.assertRaises(SandboxViolationError) as ctx:
                sandbox.resolve("../escape.txt")
            # 冻结字段（§3.3）：path 是原始入参、root 是真实的沙箱根。
            self.assertEqual(ctx.exception.path, "../escape.txt")
            self.assertEqual(ctx.exception.root, str(sandbox.root))

    def test_dotdot_escape_after_descend_raises(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "sub").mkdir()
            with self.assertRaises(SandboxViolationError):
                sandbox.resolve("sub/../../escape.txt")

    def test_absolute_path_outside_raises(self) -> None:
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            secret = outside / "secret.txt"
            secret.write_text("top secret", encoding="utf-8")
            with self.assertRaises(SandboxViolationError):
                sandbox.resolve(str(secret))

    def test_absolute_path_inside_is_allowed(self) -> None:
        """绝对路径**在 root 内**是合法的（§7.5：绝对路径直接 resolve 再检查）。"""
        with make_temp_sandbox() as sandbox:
            inside = sandbox.root / "a.txt"
            inside.write_text("hi", encoding="utf-8")
            self.assertEqual(sandbox.resolve(str(inside)), inside.resolve())

    def test_symlink_escape_raises(self) -> None:
        """root 里放一个指向外部目录的软链 —— 最容易漏掉的一种逃逸。"""
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            (outside / "leak.txt").write_text("leak", encoding="utf-8")
            link = sandbox.root / "link"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:  # pragma: no cover - 平台不支持
                self.skipTest(f"symlink not supported: {exc}")
            with self.assertRaises(SandboxViolationError):
                sandbox.resolve("link/leak.txt")

    def test_symlink_to_inside_is_allowed(self) -> None:
        with make_temp_sandbox() as sandbox:
            real = sandbox.root / "real"
            real.mkdir()
            (real / "f.txt").write_text("ok", encoding="utf-8")
            try:
                os.symlink(real, sandbox.root / "alias", target_is_directory=True)
            except (OSError, NotImplementedError) as exc:  # pragma: no cover
                self.skipTest(f"symlink not supported: {exc}")
            self.assertEqual(
                sandbox.resolve("alias/f.txt"), (real / "f.txt").resolve()
            )

    def test_root_none_raises_config_error(self) -> None:
        """root=None 必须响亮地失败：静默落到 cwd 会让 delete_file 删真项目文件。"""
        with self.assertRaises(ConfigError):
            PathSandbox(None)

    def test_tilde_is_not_expanded(self) -> None:
        """`~` 不 expanduser：它是 root 下一个普通目录名，不是 home。"""
        with make_temp_sandbox() as sandbox:
            resolved = sandbox.resolve("~/x.txt")
            self.assertIn("~", resolved.parts)
            self.assertTrue(str(resolved).startswith(str(sandbox.root)))

    def test_root_is_resolved_realpath(self) -> None:
        """`root` 在构造时就被 `.resolve()`：软链形式的 root 会被还原成真实路径。"""
        with _outside_dir() as real:
            link = Path(str(real) + "-link")
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:  # pragma: no cover
                self.skipTest(f"symlink not supported: {exc}")
            try:
                sandbox = PathSandbox(link)
                self.assertEqual(sandbox.root, real.resolve())
                self.assertNotEqual(str(sandbox.root), str(link))
            finally:
                link.unlink()


class AllowReadOutsideTests(unittest.TestCase):
    """`allow_read_outside=True` 只放宽**读**（写路径永远被限制在 root 内）。"""

    def test_read_outside_allowed(self) -> None:
        with make_temp_sandbox() as tmp_sandbox, _outside_dir() as outside:
            foreign = outside / "note.txt"
            foreign.write_text("outside content", encoding="utf-8")
            sandbox = PathSandbox(tmp_sandbox.root, allow_read_outside=True)
            tools = _by_name(make_file_tools(sandbox))
            result = tools["read_file"].raw({"path": str(foreign)})
            self.assertIn("outside content", result.content)

    def test_write_outside_still_raises(self) -> None:
        with make_temp_sandbox() as tmp_sandbox, _outside_dir() as outside:
            sandbox = PathSandbox(tmp_sandbox.root, allow_read_outside=True)
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["write_file"].raw(
                    {"path": str(outside / "pwned.txt"), "content": "x"}
                )
            self.assertFalse((outside / "pwned.txt").exists())

    def test_delete_outside_still_raises(self) -> None:
        with make_temp_sandbox() as tmp_sandbox, _outside_dir() as outside:
            keep = outside / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            sandbox = PathSandbox(tmp_sandbox.root, allow_read_outside=True)
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["delete_file"].raw({"path": str(keep), "confirm": True})
            self.assertTrue(keep.exists())

    def test_default_is_deny(self) -> None:
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            foreign = outside / "note.txt"
            foreign.write_text("nope", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["read_file"].raw({"path": str(foreign)})


class MakeFileToolsGuardTests(unittest.TestCase):
    """factory 本身的安全闸：`make_file_tools(None)` -> `ConfigError`。"""

    def test_none_sandbox_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            make_file_tools(None)

    def test_non_sandbox_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                make_file_tools(tmp)  # 裸路径字符串也不算"显式沙箱"

    def test_tool_order_and_names(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = make_file_tools(sandbox)
            self.assertEqual([tool.name for tool in tools], list(_TOOL_ORDER))
            self.assertEqual(len(tools), 5)

    def test_frozen_decorator_metadata(self) -> None:
        """§7.5 冻结的元数据：dangerous/idempotent/tags。"""
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            expected = {
                "read_file": (True, False),
                "list_dir": (True, False),
                "search_files": (True, False),
                "write_file": (True, True),
                "delete_file": (False, True),
            }
            for name, (idempotent, dangerous) in expected.items():
                spec = tools[name].spec
                self.assertEqual(spec.idempotent, idempotent, name)
                self.assertEqual(spec.dangerous, dangerous, name)
                self.assertEqual(tuple(spec.tags), ("fs",), name)

    def test_hidden_sandbox_param_absent_from_schema(self) -> None:
        """模型可见的 `parameters` 里**不得**出现 `_sandbox`（§7.5 冻结总规则）。"""
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            for name, tool in tools.items():
                properties = tool.parameters.get("properties", {})
                self.assertNotIn("_sandbox", properties, name)
                self.assertNotIn("_sandbox", str(tool.parameters), name)

    def test_schema_property_names_match_frozen_signatures(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            expected = {
                "read_file": {"path", "start_line", "end_line", "max_chars"},
                "write_file": {"path", "content", "create_dirs", "overwrite"},
                "list_dir": {"path", "pattern", "recursive", "max_entries"},
                "search_files": {
                    "pattern", "path", "glob", "max_results", "case_sensitive",
                },
                "delete_file": {"path", "confirm"},
            }
            for name, keys in expected.items():
                self.assertEqual(set(tools[name].parameters["properties"]), keys, name)


class ReadFileToolTests(unittest.TestCase):
    def test_reads_text(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.txt").write_text("hello\nworld", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["read_file"].raw({"path": "a.txt"})
            self.assertEqual(result.content, "hello\nworld")
            self.assertTrue(result.ok)

    def test_line_range_is_one_based_inclusive(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.txt").write_text("l1\nl2\nl3\nl4", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["read_file"].raw(
                {"path": "a.txt", "start_line": 2, "end_line": 3}
            )
            self.assertEqual(result.content, "l2\nl3")

    def test_missing_file_returns_error_text(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            result = tools["read_file"].raw({"path": "nope.txt"})
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertIn("not found", result.content)

    def test_missing_required_arg_raises_validation_error(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(ToolValidationError):
                tools["read_file"].raw({})

    def test_metadata_sandbox_is_realpath(self) -> None:
        """§2.4：`metadata['sandbox']` 写入者 builtin/files，值是**生效沙箱根的 realpath**。"""
        with _outside_dir() as real:
            link = Path(str(real) + "-link")
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:  # pragma: no cover
                self.skipTest(f"symlink not supported: {exc}")
            try:
                sandbox = PathSandbox(link)
                (real / "a.txt").write_text("body", encoding="utf-8")
                tools = _by_name(make_file_tools(sandbox))
                result = tools["read_file"].raw({"path": "a.txt"})
                self.assertEqual(result.metadata["sandbox"], str(real.resolve()))
                self.assertEqual(result.metadata["sandbox"], str(sandbox.root))
                self.assertNotEqual(result.metadata["sandbox"], str(link))
            finally:
                link.unlink()

    def test_metadata_sandbox_present_on_every_file_tool(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            calls = {
                "read_file": {"path": "a.txt"},
                "write_file": {"path": "b.txt", "content": "x"},
                "list_dir": {"path": "."},
                "search_files": {"pattern": "x"},
                "delete_file": {"path": "b.txt", "confirm": True},
            }
            (sandbox.root / "a.txt").write_text("x", encoding="utf-8")
            for name, args in calls.items():
                result = tools[name].raw(args)
                self.assertEqual(
                    result.metadata.get("sandbox"), str(sandbox.root), name
                )


class WriteFileToolTests(unittest.TestCase):
    def test_writes_and_reports_length(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            result = tools["write_file"].raw({"path": "out/a.txt", "content": "abc"})
            self.assertEqual((sandbox.root / "out" / "a.txt").read_text(), "abc")
            self.assertIn("3 chars", result.content)

    def test_overwrite_false_refuses(self) -> None:
        with make_temp_sandbox() as sandbox:
            target = sandbox.root / "a.txt"
            target.write_text("old", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["write_file"].raw(
                {"path": "a.txt", "content": "new", "overwrite": False}
            )
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertEqual(target.read_text(), "old")

    def test_create_dirs_false_refuses(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            result = tools["write_file"].raw(
                {"path": "missing/a.txt", "content": "x", "create_dirs": False}
            )
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertFalse((sandbox.root / "missing").exists())

    def test_write_escape_through_dotdot(self) -> None:
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["write_file"].raw(
                    {"path": "../pwned.txt", "content": "x"}
                )
            self.assertFalse((outside / "pwned.txt").exists())


class ListDirToolTests(unittest.TestCase):
    def test_lists_directory_with_slash_marker(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "sub").mkdir()
            (sandbox.root / "a.txt").write_text("x", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["list_dir"].raw({"path": "."})
            self.assertIn("sub/", result.content)
            self.assertIn("a.txt", result.content)

    def test_pattern_filter(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.py").write_text("x", encoding="utf-8")
            (sandbox.root / "b.txt").write_text("x", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["list_dir"].raw({"path": ".", "pattern": "*.py"})
            self.assertIn("a.py", result.content)
            self.assertNotIn("b.txt", result.content)

    def test_list_escape_raises(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["list_dir"].raw({"path": ".."})


class SearchFilesToolTests(unittest.TestCase):
    def test_finds_matching_lines_with_line_numbers(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["search_files"].raw({"pattern": "beta"})
            self.assertIn("a.txt:2:", result.content)
            self.assertIn("beta", result.content)

    def test_no_match_reports_searched_count(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.txt").write_text("alpha\n", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["search_files"].raw({"pattern": "zzz"})
            self.assertIn("no matches", result.content)

    def test_unsupported_glob_reports_error(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            result = tools["search_files"].raw({"pattern": "x", "glob": "*.py"})
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertIn("unsupported glob", result.content)

    def test_binary_file_is_skipped(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "bin.dat").write_bytes(b"\x00\x01needle\x00")
            (sandbox.root / "text.txt").write_text("needle\n", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["search_files"].raw({"pattern": "needle"})
            self.assertIn("text.txt", result.content)
            self.assertNotIn("bin.dat", result.content)
            self.assertIn("skipped 1 binary", result.content)

    def test_oversized_file_is_skipped_and_recorded(self) -> None:
        """§7.5：超过 1MB 的文件跳过，但必须**记入返回文本**（红线 12 的留痕）。"""
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "big.txt").write_text(
                "needle\n" + "x" * (1024 * 1024 + 10), encoding="utf-8"
            )
            (sandbox.root / "small.txt").write_text("needle\n", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["search_files"].raw({"pattern": "needle"})
            self.assertIn("small.txt", result.content)
            self.assertNotIn("big.txt:1", result.content)
            self.assertIn("1 oversized", result.content)

    def test_case_insensitive_by_default(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "a.txt").write_text("Needle\n", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            self.assertIn("Needle", tools["search_files"].raw({"pattern": "needle"}).content)
            case_sensitive = tools["search_files"].raw(
                {"pattern": "needle", "case_sensitive": True}
            )
            self.assertIn("no matches", case_sensitive.content)

    def test_search_escape_raises(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["search_files"].raw({"pattern": "x", "path": ".."})


class DeleteFileToolTests(unittest.TestCase):
    def test_requires_confirm_true(self) -> None:
        """§12：`delete_file` 需 `confirm=True` —— 默认/False 都必须拒绝。"""
        with make_temp_sandbox() as sandbox:
            target = sandbox.root / "a.txt"
            target.write_text("x", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["delete_file"].raw({"path": "a.txt"})
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertIn("confirm=True", result.content)
            self.assertTrue(target.exists())

    def test_confirm_false_also_refuses(self) -> None:
        with make_temp_sandbox() as sandbox:
            target = sandbox.root / "a.txt"
            target.write_text("x", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["delete_file"].raw({"path": "a.txt", "confirm": False})
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertTrue(target.exists())

    def test_confirm_true_deletes(self) -> None:
        with make_temp_sandbox() as sandbox:
            target = sandbox.root / "a.txt"
            target.write_text("x", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            result = tools["delete_file"].raw({"path": "a.txt", "confirm": True})
            self.assertIn("Deleted", result.content)
            self.assertFalse(target.exists())

    def test_directory_is_not_deleted(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "sub").mkdir()
            tools = _by_name(make_file_tools(sandbox))
            result = tools["delete_file"].raw({"path": "sub", "confirm": True})
            self.assertTrue(result.content.startswith("ERROR:"))
            self.assertTrue((sandbox.root / "sub").exists())

    def test_delete_escape_raises(self) -> None:
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            keep = outside / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            tools = _by_name(make_file_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["delete_file"].raw({"path": str(keep), "confirm": True})
            self.assertTrue(keep.exists())

    def test_symlink_escape_through_tool(self) -> None:
        """端到端：通过工具接口走一遍 symlink 逃逸（不只是 `resolve` 单测）。"""
        with make_temp_sandbox() as sandbox, _outside_dir() as outside:
            secret = outside / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            try:
                os.symlink(outside, sandbox.root / "link", target_is_directory=True)
            except (OSError, NotImplementedError) as exc:  # pragma: no cover
                self.skipTest(f"symlink not supported: {exc}")
            tools = _by_name(make_file_tools(sandbox))
            for name, args in (
                ("read_file", {"path": "link/secret.txt"}),
                ("write_file", {"path": "link/new.txt", "content": "x"}),
                ("delete_file", {"path": "link/secret.txt", "confirm": True}),
            ):
                with self.assertRaises(SandboxViolationError, msg=name):
                    tools[name].raw(args)
            self.assertTrue(secret.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
