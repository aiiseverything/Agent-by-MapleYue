from __future__ import annotations

# =============================================================================
# tests/test_examples_import.py —— examples/ 的**静态**守门（§12 line 5137）
# =============================================================================
#
# 存在理由：examples/ 是"框架能不能被人照着跑起来"的门面，但它**不能全部被执行**
# （02/05/06 的联网路径在本环境里跑不通）。于是本文件只做静态检查，一个字节都不执行：
#
#   * 每个 `examples/*.py` 都能 `ast.parse` + `compile(src, path, "exec")`（**不执行**）；
#   * 每个示例都有 `if __name__ == "__main__":`（否则 `python3 examples/xx.py` 什么都不干）；
#   * 每个示例都通过 `argparse.add_argument("--offline", ...)` 声明了离线开关。
#
# 真跑（subprocess）在 `test_examples_offline.py`；端到端在 `test_e2e_code_assistant.py`。
# 静态与动态分开的理由（面试可讲）：静态检查毫秒级、零副作用，能覆盖**全部 8 个**示例；
# 动态检查只能覆盖"真的能离线跑"的那 4 个（§12 line 5136 的划分）。
# 两者互补 —— 静态保证"文件不会语法错误地躺在仓库里"，动态保证"跑起来真的对"。

import ast
import pathlib
import types
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

#: 需要真 provider / 网络、因此**只静态检查不执行**的示例（§12 line 5136 的划分）。
NETWORK_ONLY_EXAMPLES: tuple[str, ...] = (
    "02_tools_custom.py",
    "05_multiagent_sequential.py",
    "06_multiagent_hierarchical.py",
)

#: 能离线真跑的四个（与 `test_examples_offline.py` 共用同一份事实）。
OFFLINE_EXAMPLES: tuple[str, ...] = (
    "01_quickstart.py",
    "03_react_text_mode.py",
    "04_memory.py",
    "07_code_assistant.py",
)

#: 汇总器：它靠**子进程**跑别的示例，自己不 bootstrap `sys.path`（见下面的引导用例）。
AGGREGATOR = "run_all_examples.py"


def _example_paths() -> list[pathlib.Path]:
    """`examples/` 下所有示例源文件（排序，保证失败信息可复现）。"""
    return sorted(
        path for path in EXAMPLES_DIR.glob("*.py") if path.name != "__init__.py"
    )


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_main_guard(tree: ast.Module) -> bool:
    """`if __name__ == "__main__":` 的 **AST** 判定。

    刻意不搜源码文本：`"--main--"` 出现在注释或字符串里（本仓库的示例头部就大量
    提到 `__main__`）不算数，只有真的有一条比较 `__name__` 与 `"__main__"` 的 `if`
    才算 —— 这正是"守门测试用 AST 而不是子串匹配"的同一个理由。
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        left = test.left
        if not (isinstance(left, ast.Name) and left.id == "__name__"):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def _declares_offline_switch(tree: ast.Module) -> bool:
    """是否有 `xxx.add_argument("--offline", ...)`。

    只认 `add_argument` 的**第一个位置参数**是字面量 `"--offline"` 的调用：
    把开关名写进文档字符串、或拼成 f-string 都不算（读者要能一眼在 CLI 里看到它）。
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "--offline":
            return True
    return False


def _bootstraps_sys_path(tree: ast.Module) -> bool:
    """是否有 `sys.path.insert(...)`（"从任意 cwd 直接 python3 也能 import liteagent"）。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert"):
            continue
        target = func.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            return True
    return False


class ExampleInventoryTests(unittest.TestCase):
    """示例清单本身（少了文件、名字写错了都要在这里炸，而不是等 subprocess 报 FileNotFound）。"""

    def test_expected_examples_are_present(self) -> None:
        names = {path.name for path in _example_paths()}
        missing = [
            name
            for name in (*OFFLINE_EXAMPLES, *NETWORK_ONLY_EXAMPLES, AGGREGATOR)
            if name not in names
        ]
        self.assertEqual(
            [], missing, f"examples/ 下缺少这些被 §12 点名的示例：{missing}"
        )

    def test_examples_directory_is_not_empty(self) -> None:
        self.assertGreaterEqual(
            len(_example_paths()),
            len(OFFLINE_EXAMPLES) + len(NETWORK_ONLY_EXAMPLES),
            "examples/ 的文件数不足以覆盖 §12 冻结的 01|02|03|04|05|06|07",
        )


class ExampleStaticCheckTests(unittest.TestCase):
    """逐文件静态检查（**不执行任何示例**）。"""

    def test_every_example_parses_and_compiles(self) -> None:
        offenders: list[str] = []
        for path in _example_paths():
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
                code = compile(source, str(path), "exec")
            except SyntaxError as exc:
                offenders.append(f"{path.name}: {exc.msg} (line {exc.lineno})")
                continue
            if not isinstance(tree, ast.Module):
                offenders.append(f"{path.name}: ast.parse 没有返回 Module")
            elif not isinstance(code, types.CodeType):
                offenders.append(f"{path.name}: compile() 没有返回 code object")
        self.assertEqual([], offenders, f"这些示例无法编译：{offenders}")

    def test_every_example_has_main_guard(self) -> None:
        offenders = [
            path.name for path in _example_paths() if not _has_main_guard(_parse(path))
        ]
        self.assertEqual(
            [],
            offenders,
            f"这些示例缺少 `if __name__ == \"__main__\":`，直接 python3 跑不会执行 main：{offenders}",
        )

    def test_every_example_declares_offline_switch(self) -> None:
        offenders = [
            path.name
            for path in _example_paths()
            if not _declares_offline_switch(_parse(path))
        ]
        self.assertEqual(
            [],
            offenders,
            f"这些示例没有 argparse 的 `--offline` 开关，无法在 CI 里离线跑：{offenders}",
        )

    def test_every_numbered_example_bootstraps_sys_path(self) -> None:
        """01..07 必须自己把仓库根放进 `sys.path`（否则 `python3 examples/xx.py` 会 ImportError）。

        `run_all_examples.py` 是例外：它跑的是**子进程**，改为通过 `PYTHONPATH` 兜底，
        自己不 import liteagent（见文件内注释）。
        """
        offenders = [
            path.name
            for path in _example_paths()
            if path.name != AGGREGATOR and not _bootstraps_sys_path(_parse(path))
        ]
        self.assertEqual(
            [],
            offenders,
            f"这些示例没有 `sys.path.insert(...)` 引导，从任意 cwd 直接跑会 import 失败：{offenders}",
        )

    def test_static_checks_do_not_execute_the_examples(self) -> None:
        """把三条判定器喂给"坏源码"，确认它们真的会报 False（守门测试的自证）。

        没有这条自证，判定器整体写错方向（例如恒返回 True）时，"全部通过"会变成
        一句空话 —— 这正是 §12 要求守门测试用 AST 的同一精神：判据本身也要可测。
        """
        good = ast.parse(
            "import sys\n"
            "sys.path.insert(0, '/x')\n"
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            'parser.add_argument("--offline", action="store_true")\n'
            'if __name__ == "__main__":\n'
            "    raise SystemExit(0)\n"
        )
        self.assertTrue(_has_main_guard(good))
        self.assertTrue(_declares_offline_switch(good))
        self.assertTrue(_bootstraps_sys_path(good))

        bad = ast.parse(
            "# 提到 __main__ 与 --offline 只是注释，不该被认出来\n"
            'BANNER = "__main__ --offline"\n'
            "x = 1\n"
        )
        self.assertFalse(_has_main_guard(bad))
        self.assertFalse(_declares_offline_switch(bad))
        self.assertFalse(_bootstraps_sys_path(bad))

        # 反例：`--offline` 只在第二个参数位置出现 -> 不算"声明了开关"。
        shifted = ast.parse('parser.add_argument("x", "--offline")\n')
        self.assertFalse(_declares_offline_switch(shifted))


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
