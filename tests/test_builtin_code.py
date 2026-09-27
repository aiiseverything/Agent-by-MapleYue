from __future__ import annotations

"""tests/test_builtin_code.py —— 内置代码工具（§7.5 ``code.py``；§12 第 5132 行）。

§12 第 5132 行列出的覆盖点（逐条都有对应用例）：
  * ``python_eval`` 合法表达式
  * **``Name`` 允许 ``variables`` 里的键**
  * 拒绝 ``__import__`` / ``eval`` / 属性白名单外访问
  * **复杂度闸（``10**10**10``、``range(10**7)``）**
  * ``python_exec`` 的 ``TimeoutExpired`` -> ``ToolTimeoutError``
  * ``run_tests`` 解析 ``Ran N tests``
  * **``run_tests`` 指向本仓库 ``tests/`` 时返回文本含 ``WARNING``**

复杂度闸的用例**只断言"被静态拒绝"**，绝不让表达式真的开始求值 —— 这正是 D-09
"timeout_s 无法中断求值，因此必须在静态检查阶段拦截"的可执行证明。
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from liteagent.errors import ConfigError, SandboxViolationError, ToolExecutionError, ToolTimeoutError
from liteagent.tools.base import Tool
from liteagent.tools.builtin.code import (
    ALLOWED_BUILTINS,
    MAX_AST_DEPTH,
    MAX_POW_EXPONENT,
    MAX_RANGE_ARG,
    safe_eval_ast,
    make_code_tools,
)
from tests.helpers import make_temp_sandbox

#: 本仓库的 `tests/` 目录（run_tests 的 WARNING 判定基准）。
REPO_TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"

#: 一个"瞬时完成"的两用例测试文件（run_tests 的成功路径用）。
_PASSING_TEST_SOURCE = """\
import unittest


class ScratchTest(unittest.TestCase):
    def test_one(self):
        self.assertEqual(1 + 1, 2)

    def test_two(self):
        self.assertTrue(True)
"""

#: 一个必然失败的测试文件（run_tests 的 FAILED 解析用）。
_FAILING_TEST_SOURCE = """\
import unittest


class ScratchTest(unittest.TestCase):
    def test_fails(self):
        self.assertEqual(1, 2)
"""


def _by_name(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


class SafeEvalAllowedTests(unittest.TestCase):
    """合法表达式：算术、容器、方法调用、变量。"""

    def test_arithmetic(self) -> None:
        self.assertEqual(safe_eval_ast("1 + 2 * 3"), 7)
        self.assertEqual(safe_eval_ast("(10 - 4) / 3"), 2.0)
        self.assertEqual(safe_eval_ast("2 ** 10"), 1024)
        self.assertEqual(safe_eval_ast("7 // 2"), 3)
        self.assertEqual(safe_eval_ast("7 % 2"), 1)

    def test_containers_and_subscript(self) -> None:
        self.assertEqual(safe_eval_ast("[1, 2, 3][1:]"), [2, 3])
        self.assertEqual(safe_eval_ast("{'a': 1}['a']"), 1)
        self.assertEqual(safe_eval_ast("(1, 2) + (3,)"), (1, 2, 3))
        self.assertEqual(safe_eval_ast("{1, 2} | {3}"), {1, 2, 3})

    def test_comparisons_and_bool_short_circuit(self) -> None:
        self.assertTrue(safe_eval_ast("1 < 2 <= 2"))
        self.assertTrue(safe_eval_ast("'a' in 'abc'"))
        # 短路：`0 and ...` 不能触发后面的 ZeroDivisionError。
        self.assertEqual(safe_eval_ast("0 != 0 and 10 // 0"), False)
        self.assertEqual(safe_eval_ast("'x' if 1 else 'y'"), "x")

    def test_allowed_builtins_are_callable(self) -> None:
        self.assertEqual(safe_eval_ast("len([1, 2, 3])"), 3)
        self.assertEqual(safe_eval_ast("sorted([3, 1, 2])"), [1, 2, 3])
        self.assertEqual(safe_eval_ast("sum(range(10))"), 45)
        self.assertEqual(safe_eval_ast("max(1, 5, 2)"), 5)
        self.assertEqual(safe_eval_ast("str(12) + 'x'"), "12x")

    def test_builtin_whitelist_snapshot(self) -> None:
        self.assertIn("len", ALLOWED_BUILTINS)
        self.assertIn("range", ALLOWED_BUILTINS)
        self.assertNotIn("eval", ALLOWED_BUILTINS)
        self.assertNotIn("open", ALLOWED_BUILTINS)

    def test_readonly_attribute_methods(self) -> None:
        self.assertEqual(safe_eval_ast("'abc'.upper()"), "ABC")
        self.assertEqual(safe_eval_ast("'ab,cd'.split(',')"), ["ab", "cd"])
        self.assertEqual(safe_eval_ast("d.get('a')", {"d": {"a": 5}}), 5)
        self.assertEqual(safe_eval_ast("[1, 2, 3].count(2)"), 1)


class SafeEvalVariablesTests(unittest.TestCase):
    """[v2 冻结] 规则 1：`Name` 允许 `ALLOWED_BUILTINS` 里的名字 **或** `variables` 的键。"""

    def test_variables_are_visible_as_names(self) -> None:
        self.assertEqual(safe_eval_ast("x + y", {"x": 1, "y": 2}), 3)
        self.assertEqual(safe_eval_ast("payload['k']", {"payload": {"k": "v"}}), "v")

    def test_name_allowed_even_if_it_looks_like_a_name_node(self) -> None:
        """逐字对齐 §12：`Name` 允许 `variables` 里的键（哪怕它不是内置名）。"""
        self.assertEqual(safe_eval_ast("Name", {"Name": 42}), 42)
        self.assertEqual(safe_eval_ast("weird_name_123", {"weird_name_123": "ok"}), "ok")

    def test_name_allowed_as_a_key_is_still_unknown_without_variables(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("Name")

    def test_variables_from_json_in_the_tool(self) -> None:
        tools = _by_name(make_code_tools())
        result = tools["python_eval"].raw(
            {"expression": "x * 2", "variables_json": '{"x": 3}'}
        )
        self.assertEqual(result, "6")

    def test_variables_shadowing_a_builtin_in_a_call_is_rejected(self) -> None:
        """静态审查看到 `len()`、运行时却调到变量 —— 两者不一致必须拒绝。"""
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("len([1])", {"len": lambda value: 999})

    def test_bad_variables_json_is_rejected(self) -> None:
        tools = _by_name(make_code_tools())
        with self.assertRaises(ToolExecutionError):
            tools["python_eval"].raw(
                {"expression": "1", "variables_json": "{not json"}
            )

    def test_variables_json_must_decode_to_object(self) -> None:
        tools = _by_name(make_code_tools())
        with self.assertRaises(ToolExecutionError):
            tools["python_eval"].raw({"expression": "1", "variables_json": "[1, 2]"})


class SafeEvalRejectionTests(unittest.TestCase):
    """违规表达式：危险名字、未授权属性、非白名单节点。"""

    def test_forbidden_names_are_rejected(self) -> None:
        for expression in (
            "__import__('os')",
            "eval('1 + 1')",
            "exec('x = 1')",
            "open('/etc/passwd')",
            "globals()",
            "locals()",
            "getattr('a', 'upper')",
            "compile('1', '<s>', 'eval')",
            "input()",
            "vars()",
            "__builtins__",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolExecutionError):
                    safe_eval_ast(expression)

    def test_unknown_names_are_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("os.system('ls')")
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("sys.exit(0)")

    def test_attributes_outside_whitelist_are_rejected(self) -> None:
        for expression in (
            "'a'.encode",
            "'a'.__len__",
            "'a'.format_map",
            "(1).real",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolExecutionError):
                    safe_eval_ast(expression)

    def test_dunder_attribute_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("'a'.__class__")

    def test_whitelisted_method_on_wrong_receiver_is_rejected_at_runtime(self) -> None:
        """静态只能看名字（`upper` 在 str 名单里），运行时按接收者类型再判一次。"""
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("[1, 2].upper()")

    def test_lambda_and_comprehensions_are_rejected(self) -> None:
        for expression in (
            "(lambda: 1)()",
            "[x for x in range(3)]",
            "{x for x in range(3)}",
            "{x: x for x in range(3)}",
            "f'{1 + 1}'",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolExecutionError):
                    safe_eval_ast(expression)

    def test_kwargs_unpacking_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("dict(**{'a': 1})")

    def test_syntax_error_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("1 +")

    def test_empty_expression_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("   ")


class ComplexityGateTests(unittest.TestCase):
    """[v2 冻结] 规则 2：复杂度必须在**静态检查阶段**拦截（timeout 拦不住求值）。"""

    def test_huge_pow_is_rejected_before_evaluation(self) -> None:
        """`10**10**10`：中间值 `10**10` 可折叠，折叠结果直接撞上指数上限。"""
        with self.assertRaises(ToolExecutionError) as ctx:
            safe_eval_ast("10**10**10")
        self.assertIn("Pow", ctx.exception.context.get("node", ""))

    def test_pow_with_literal_exponent_over_limit_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("2 ** 5000")
        self.assertGreater(MAX_POW_EXPONENT, 0)
        self.assertEqual(safe_eval_ast("2 ** 1000"), 2**1000)

    def test_pow_with_non_literal_exponent_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("2 ** n", {"n": 3})

    def test_pow_function_shares_the_same_gate(self) -> None:
        """只堵 `**` 不堵 `pow()` 等于没堵（同一颗炸弹的两种写法）。"""
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("pow(2, 10**10)")

    def test_huge_range_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError) as ctx:
            safe_eval_ast("range(10**7)")
        self.assertEqual(ctx.exception.context.get("func"), "range")
        self.assertGreater(MAX_RANGE_ARG, 1.0)

    def test_range_within_limit_is_allowed(self) -> None:
        self.assertEqual(safe_eval_ast("len(range(10))"), 10)

    def test_overlong_expression_is_rejected(self) -> None:
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast("1" * 2001)

    def test_too_deeply_nested_expression_is_rejected(self) -> None:
        expression = "-" * (MAX_AST_DEPTH + 5) + "1"
        with self.assertRaises(ToolExecutionError):
            safe_eval_ast(expression)

    def test_gate_reports_through_the_tool(self) -> None:
        tools = _by_name(make_code_tools())
        for expression in ("10**10**10", "range(10**7)"):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolExecutionError):
                    tools["python_eval"].raw({"expression": expression})


class PythonEvalToolTests(unittest.TestCase):
    def test_string_result_is_returned_raw(self) -> None:
        tools = _by_name(make_code_tools())
        result = tools["python_eval"].raw({"expression": "'hello'"})
        self.assertEqual(result, "hello")

    def test_non_string_result_uses_repr(self) -> None:
        tools = _by_name(make_code_tools())
        self.assertEqual(tools["python_eval"].raw({"expression": "[1, 2]"}), "[1, 2]")
        self.assertEqual(tools["python_eval"].raw({"expression": "1 + 1"}), "2")

    def test_missing_expression_raises_validation_error(self) -> None:
        from liteagent.errors import ToolValidationError

        tools = _by_name(make_code_tools())
        with self.assertRaises(ToolValidationError):
            tools["python_eval"].raw({})

    def test_frozen_decorator_metadata(self) -> None:
        tools = _by_name(make_code_tools())
        spec = tools["python_eval"].spec
        self.assertEqual(spec.timeout_s, 5.0)
        self.assertTrue(spec.idempotent)
        self.assertFalse(spec.dangerous)
        self.assertEqual(tuple(spec.tags), ("code",))


class PythonExecToolTests(unittest.TestCase):
    def test_executes_code_and_reports_stdout(self) -> None:
        tools = _by_name(make_code_tools())
        result = tools["python_exec"].raw({"code": "print(6 * 7)"})
        self.assertIn("exit_code: 0", result)
        self.assertIn("42", result)

    def test_nonzero_exit_code_on_exception(self) -> None:
        tools = _by_name(make_code_tools())
        result = tools["python_exec"].raw({"code": "raise SystemExit(7)"})
        self.assertIn("exit_code: 7", result)

    def test_timeout_expired_becomes_tool_timeout_error(self) -> None:
        tools = _by_name(make_code_tools())
        expired = subprocess.TimeoutExpired(cmd=[sys.executable, "-c", "x"], timeout=0.5)
        with mock.patch(
            "liteagent.tools.builtin.code.subprocess.run", side_effect=expired
        ):
            with self.assertRaises(ToolTimeoutError) as ctx:
                tools["python_exec"].raw({"code": "while True: pass", "timeout_s": 0.5})
        self.assertEqual(ctx.exception.tool_name, "python_exec")
        self.assertEqual(ctx.exception.timeout_s, 0.5)

    def test_runs_with_isolated_interpreter_flag(self) -> None:
        tools = _by_name(make_code_tools())
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with mock.patch(
            "liteagent.tools.builtin.code.subprocess.run", return_value=fake
        ) as fake_run:
            tools["python_exec"].raw({"code": "print(1)"})
        argv = fake_run.call_args.args[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertIn("-I", argv)
        self.assertIn("-c", argv)

    def test_cwd_outside_sandbox_raises(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_code_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["python_exec"].raw({"code": "print(1)", "cwd": "../.."})

    def test_frozen_decorator_metadata(self) -> None:
        tools = _by_name(make_code_tools())
        spec = tools["python_exec"].spec
        self.assertTrue(spec.dangerous)
        self.assertTrue(spec.requires_approval)
        self.assertFalse(spec.idempotent)
        self.assertEqual(tuple(spec.tags), ("code",))


class RunTestsToolTests(unittest.TestCase):
    """`run_tests`：真实跑一个瞬时 scratch 套件 + WARNING 判定（后者用 mock 保护）。"""

    def test_parses_ran_n_tests_ok(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "test_scratch.py").write_text(
                _PASSING_TEST_SOURCE, encoding="utf-8"
            )
            tools = _by_name(make_code_tools(sandbox))
            result = tools["run_tests"].raw(
                {"path": ".", "pattern": "test_scratch.py"}
            )
            self.assertIn("Ran 2 tests", result)
            self.assertIn("tests: 2", result)
            self.assertIn("status: OK", result)
            self.assertIn("exit_code: 0", result)

    def test_parses_failed_status(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "test_failing.py").write_text(
                _FAILING_TEST_SOURCE, encoding="utf-8"
            )
            tools = _by_name(make_code_tools(sandbox))
            result = tools["run_tests"].raw(
                {"path": ".", "pattern": "test_failing.py"}
            )
            self.assertIn("tests: 1", result)
            self.assertIn("status: FAILED", result)
            self.assertIn("exit_code: 1", result)

    def test_missing_directory_reports_error(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_code_tools(sandbox))
            result = tools["run_tests"].raw({"path": "no_such_dir"})
            self.assertTrue(result.startswith("ERROR:"))

    def test_path_outside_sandbox_raises(self) -> None:
        with make_temp_sandbox() as sandbox:
            tools = _by_name(make_code_tools(sandbox))
            with self.assertRaises(SandboxViolationError):
                tools["run_tests"].raw({"path": "../../.."})

    def test_repo_tests_dir_warns(self) -> None:
        """§7.5 [v2 冻结]：指向本仓库 `tests/` 时追加 WARNING（防递归跑整套用例）。

        这里用 mock 换掉 `subprocess.run`：真正跑一遍整个仓库的套件既是自指递归、
        也会让本测试的耗时引爆。WARNING 判定逻辑本身走的是**真实**的路径解析。
        """
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Ran 3 tests in 0.001s\n\nOK\n", stderr=""
        )
        with mock.patch(
            "liteagent.tools.builtin.code.subprocess.run", return_value=fake
        ):
            tools = _by_name(make_code_tools())
            result = tools["run_tests"].raw({"path": str(REPO_TESTS_DIR)})
        self.assertIn("WARNING", result)
        self.assertIn("recursively re-runs the whole suite", result)
        self.assertIn("tests: 3", result)

    def test_relative_repo_tests_path_also_warns(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Ran 1 test in 0.001s\n\nOK\n", stderr=""
        )
        with mock.patch(
            "liteagent.tools.builtin.code.subprocess.run", return_value=fake
        ):
            tools = _by_name(make_code_tools())
            result = tools["run_tests"].raw({"path": str(REPO_TESTS_DIR / ".." / "tests")})
        self.assertIn("WARNING", result)

    def test_scratch_dir_does_not_warn(self) -> None:
        with make_temp_sandbox() as sandbox:
            (sandbox.root / "test_scratch.py").write_text(
                _PASSING_TEST_SOURCE, encoding="utf-8"
            )
            tools = _by_name(make_code_tools(sandbox))
            result = tools["run_tests"].raw(
                {"path": ".", "pattern": "test_scratch.py"}
            )
            self.assertNotIn("WARNING", result)

    def test_frozen_decorator_metadata(self) -> None:
        tools = _by_name(make_code_tools())
        spec = tools["run_tests"].spec
        self.assertTrue(spec.dangerous)
        self.assertFalse(spec.idempotent)
        self.assertEqual(tuple(spec.tags), ("code",))


class MakeCodeToolsGuardTests(unittest.TestCase):
    def test_tool_names_and_order(self) -> None:
        tools = make_code_tools()
        self.assertEqual(
            [tool.name for tool in tools], ["python_eval", "python_exec", "run_tests"]
        )

    def test_bad_sandbox_type_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            make_code_tools(sandbox="/tmp")

    def test_eval_needs_no_sandbox(self) -> None:
        """`python_eval` 是纯函数求值，sandbox=None 也必须可用（§7.5 签名 default None）。"""
        tools = _by_name(make_code_tools(None))
        self.assertEqual(tools["python_eval"].raw({"expression": "1 + 1"}), "2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
