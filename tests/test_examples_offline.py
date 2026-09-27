from __future__ import annotations

# =============================================================================
# tests/test_examples_offline.py —— **真跑**离线示例（§12 line 5136）
# =============================================================================
#
# 与 `test_examples_import.py` 的分工：那边只静态看文件（编译得过就算数），这边用
# `subprocess` **真的把示例当命令行程序跑一遍**，断言退出码 0 且 stdout 里出现
# 预期关键串。区别在"能证明什么"：
#
#   * 静态检查证明"文件是合法 Python"；
#   * 子进程真跑证明"示例在干净环境下、从仓库根启动、整条链路真的走通了"。
#
# 三条刻意的设计（面试可讲）：
#
#   1. **cwd 固定为仓库根**：示例自己会 `sys.path.insert` 引导，但如果它们悄悄依赖
#      "调用者的 cwd"，在这里就会暴露（`examples/` 里读相对路径会炸）。
#   2. **不依赖外部环境变量**：`_clean_env()` 主动剥掉 `*_API_KEY`，并钉死
#      `PYTHONIOENCODING=utf-8`（示例打印中文，在 LANG=C 的机器上会因 ASCII 编码炸掉
#      —— 那是环境问题，不是示例问题，不该让它冒充"示例有 bug"）。
#   3. **不看退出码了事**：每个用例都断言 stdout 里的关键串。只判退出码的话，
#      "示例把异常吞了、打印了一行错误然后 return 0" 也会变绿。
#
# 覆盖范围与 §12 line 5136 冻结的划分一致：真跑 01|03|04|07，02|05|06 只 `compile()`。
# （实测 02|05|06 的 `--offline` 也能跑通，但冻结的划分如此 —— 见 VERIFICATION 的说明。）

import os
import pathlib
import subprocess
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

#: 单个示例的超时（§12 line 5136 冻结为 60s）。07 会起 unittest 子进程，是最慢的一个。
SUBPROCESS_TIMEOUT_S = 60

#: 需要联网/真 provider 的示例：**只编译、不执行**（§12 冻结的划分）。
NETWORK_ONLY_EXAMPLES: tuple[str, ...] = (
    "02_tools_custom.py",
    "05_multiagent_sequential.py",
    "06_multiagent_hierarchical.py",
)

#: 真跑矩阵：`(示例文件, 额外 argv, stdout 必须出现的关键串)`。
#:
#: `--offline` / `--provider echo` 是两条**等价**的离线写法（各示例的 arg 解析里，
#: `--provider echo` 仅在 04/05/06/07 被当成"离线别名"；01/03 的 `--provider` 走真实
#: provider 分支，所以它们只跑 `--offline` —— 这一点在 VERIFICATION 里如实记录）。
OFFLINE_INVOCATIONS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "01_quickstart.py",
        ("--offline",),
        ("status=FINISHED", "The answer is 42."),
    ),
    (
        # 01 的 `--provider echo` 走的是"真实 provider"分支，但 echo 是纯本地的
        # 占位模型（零网络），因此仍然可以在 CI 里断言它的输出。
        "01_quickstart.py",
        ("--provider", "echo"),
        ("真实 provider 'echo'", "status=FINISHED"),
    ),
    (
        "03_react_text_mode.py",
        ("--offline",),
        ("运行模式：离线（--offline）", "Final Answer          : 84", "status          : FINISHED"),
    ),
    (
        "04_memory.py",
        ("--offline",),
        ("provider='echo'  offline=True", "三层各司其职", "全部离线跑通，输出可复现。"),
    ),
    (
        "04_memory.py",
        ("--provider", "echo"),
        ("provider='echo'", "三层各司其职", "全部离线跑通，输出可复现。"),
    ),
    (
        "07_code_assistant.py",
        ("--offline",),
        ("离线验收：4 条断言全部通过", "status=FINISHED"),
    ),
    (
        "07_code_assistant.py",
        ("--provider", "echo"),
        ("离线验收：4 条断言全部通过", "status=FINISHED"),
    ),
)


def _clean_env() -> dict[str, str]:
    """子进程环境：**剥掉所有 API key**，并钉死 UTF-8 输出编码。

    * 剥 key 的理由：本机即便恰好有 `OPENAI_API_KEY`，也不该让"离线示例"有联网的可能
      （离线用例必须证明"没有 key 也能跑"）。`LITEAGENT_ALLOW_SHELL` 之类会改变工具
      行为的开关一并清掉，避免测试结果随开发机的环境漂移。
    * 钉 `PYTHONIOENCODING=utf-8`：示例大量打印中文；在 `LANG=C` 的环境里子进程 stdout
      默认是 ASCII，会抛 `UnicodeEncodeError` —— 那会让"环境缺 locale"伪装成"示例有 bug"。
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not (key == "LITEAGENT_API_KEY" or key.endswith("_API_KEY"))
    }
    env.pop("LITEAGENT_ALLOW_SHELL", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_example(
    name: str, extra: tuple[str, ...] = (), *, timeout: float = SUBPROCESS_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    """在**仓库根**、**干净环境**下把示例当命令行程序跑一遍（同步阻塞，有超时）。"""
    command = [sys.executable, str(EXAMPLES_DIR / name), *extra]
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


def _tail(text: str, limit: int = 20) -> str:
    """失败信息里附上输出尾部 —— 只报"退出码 1"没法定位。"""
    lines = (text or "").splitlines()
    return "\n".join(lines[-limit:])


class ExampleOfflineRunTests(unittest.TestCase):
    """01|03|04|07 的 `--offline` / `--provider echo` 真跑。"""

    def _assert_offline_run(
        self, name: str, extra: tuple[str, ...], markers: tuple[str, ...]
    ) -> subprocess.CompletedProcess[str]:
        label = " ".join((name, *extra))
        completed = _run_example(name, extra)
        self.assertEqual(
            0,
            completed.returncode,
            f"{label} 退出码应为 0，实际 {completed.returncode}\n"
            f"--- stdout 尾部 ---\n{_tail(completed.stdout)}\n"
            f"--- stderr 尾部 ---\n{_tail(completed.stderr)}",
        )
        self.assertNotIn(
            "Traceback (most recent call last)",
            completed.stdout + completed.stderr,
            f"{label} 输出里出现 traceback，说明离线路径没走通：\n{_tail(completed.stderr)}",
        )
        for marker in markers:
            self.assertIn(
                marker,
                completed.stdout,
                f"{label} 的 stdout 缺少关键串 {marker!r}\n--- stdout 尾部 ---\n{_tail(completed.stdout)}",
            )
        return completed

    def test_01_quickstart_offline_and_provider_echo(self) -> None:
        for name, extra, markers in OFFLINE_INVOCATIONS:
            if name != "01_quickstart.py":
                continue
            with self.subTest(argv=extra):
                self._assert_offline_run(name, extra, markers)

    def test_03_react_text_mode_offline(self) -> None:
        for name, extra, markers in OFFLINE_INVOCATIONS:
            if name != "03_react_text_mode.py":
                continue
            with self.subTest(argv=extra):
                self._assert_offline_run(name, extra, markers)

    def test_04_memory_offline_and_provider_echo(self) -> None:
        entries = [item for item in OFFLINE_INVOCATIONS if item[0] == "04_memory.py"]
        self.assertEqual(2, len(entries), "04 应当覆盖 --offline 与 --provider echo 两种写法")
        for name, extra, markers in entries:
            with self.subTest(argv=extra):
                self._assert_offline_run(name, extra, markers)

    def test_07_code_assistant_offline_and_provider_echo(self) -> None:
        entries = [
            item for item in OFFLINE_INVOCATIONS if item[0] == "07_code_assistant.py"
        ]
        self.assertEqual(2, len(entries), "07 应当覆盖 --offline 与 --provider echo 两种写法")
        for name, extra, markers in entries:
            with self.subTest(argv=extra):
                completed = self._assert_offline_run(name, extra, markers)
                # 07 是旗舰示例：它自己就断言了"文件真的被改了、测试真的过了"。
                # 这里再钉一次那句自证，确保我们不是把"验收被跳过"当成通过。
                self.assertIn("文件内容已改变", completed.stdout)


class NetworkOnlyExampleTests(unittest.TestCase):
    """02|05|06：只 `compile()`（§12 line 5136 的冻结划分）。"""

    def test_network_only_examples_compile_without_executing(self) -> None:
        offenders: list[str] = []
        for name in NETWORK_ONLY_EXAMPLES:
            path = EXAMPLES_DIR / name
            self.assertTrue(path.is_file(), f"{name} 不存在")
            source = path.read_text(encoding="utf-8")
            try:
                compile(source, str(path), "exec")
            except SyntaxError as exc:
                offenders.append(f"{name}: {exc.msg} (line {exc.lineno})")
        self.assertEqual([], offenders, f"这些示例无法编译：{offenders}")


class OfflineRunHarnessTests(unittest.TestCase):
    """夹具自身的自证：环境清洗与超时参数不能形同虚设。"""

    def test_clean_env_drops_api_keys_and_pins_utf8(self) -> None:
        original = dict(os.environ)
        try:
            os.environ["OPENAI_API_KEY"] = "should-be-dropped"
            os.environ["LITEAGENT_API_KEY"] = "should-be-dropped"
            os.environ["LITEAGENT_ALLOW_SHELL"] = "1"
            env = _clean_env()
        finally:
            os.environ.clear()
            os.environ.update(original)

        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("LITEAGENT_API_KEY", env)
        self.assertNotIn("LITEAGENT_ALLOW_SHELL", env)
        self.assertEqual("utf-8", env["PYTHONIOENCODING"])

    def test_invocation_matrix_only_covers_the_four_offline_examples(self) -> None:
        covered = {name for name, _extra, _markers in OFFLINE_INVOCATIONS}
        self.assertEqual(
            {"01_quickstart.py", "03_react_text_mode.py", "04_memory.py",
             "07_code_assistant.py"},
            covered,
            "真跑矩阵必须恰好覆盖 §12 冻结的 01|03|04|07 四个示例",
        )


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
