from __future__ import annotations

"""``liteagent/cli.py`` 的单元测试（§12 清单行 ``tests/test_cli.py``）。

规范真值源：``docs/INTERFACES.md`` §11（4992-5091 行，CLI 的冻结用法与退出码契约）、
§12 第 5134 行（本文件的覆盖清单）、附录 B（``__main__.py`` 的逐字内容与版本号来源）。

§12 第 5134 行要求的覆盖点，逐条对应用例（自查表）：

============================================================  ==============================
覆盖点                                                        用例
============================================================  ==============================
``main(["version"]) == 0``                                     ``test_version_*`` / ``test_dashdash_version_*``
``tools list``                                                 ``test_tools_list_*``
``tools show``                                                 ``test_tools_show_*``
``tools schema --format openai`` 是合法 JSON                   ``test_tools_schema_openai_*``
``run -p ... --provider echo --json`` 可 ``json.loads``         ``test_run_json_is_loadable``
``run`` 缺 prompt -> 2                                         ``test_run_without_prompt_*``
未知子命令 -> 2                                                ``test_unknown_subcommand_*``
``trace FILE`` 与 ``trace FILE --stats --json``                ``TraceCommandTest``
``--tools read_file`` / ``files`` / ``""`` 三种结果区分         ``ToolSelectionTest``（五条）
``--no-tools`` 与 ``--no-builtin`` 冲突 -> 2                    ``test_no_tools_plus_no_builtin_*``
``chat`` 子命令                                                ``ChatCommandTest``
stdout 无 traceback                                            ``test_unexpected_exception_*`` 等
``render_result`` 两种格式                                     ``RenderResultTest``
============================================================  ==============================

**为什么全部用 ``cli.main([...])`` 而不是起子进程**：CLI 的契约是"返回 int 而不是
``sys.exit``"（§11 冻结），直接调用才能断言退出码；子进程既慢又会把"测试环境里
有没有装包"变成隐性依赖。唯一需要 ``subprocess`` 的是 ``test_main_py_is_runnable``
那条"包入口真的能跑"的守门用例。

**为什么不联网**：所有需要模型的地方一律 ``--provider echo``（离线 stub），
``--provider openai`` 之类只在"缺 key 应当报错"的用例里出现，且断言的是
**它没发请求就失败了**。所有时间断言一律避开（§2.8 禁止清单）。

**并行安全**：本文件不碰 ``get_default_registry()`` / ``auto_register``（§12.1 卫生规则 1），
因此不需要 ``reset_default_registry()``；``setUp`` 里清空环境变量，避免宿主机的
``LITEAGENT_TRACE`` / ``LITEAGENT_SANDBOX_ROOT`` 把结果带偏。
"""

import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from liteagent import __version__, cli
from liteagent.agent.callbacks import load_trace, trace_stats
from liteagent.agent.state import AgentResult, AgentStatus
from liteagent.errors import AgentError, ConfigError, LLMAuthError
from liteagent.types import TokenUsage

# ======================================================================================
# 调用夹具
# ======================================================================================


@contextlib.contextmanager
def _restore_logging():
    """把全局 logging 状态原样还原。

    ``cli.main`` 会调 ``logging.basicConfig(stream=sys.stderr, ...)``（§11 之外的
    实现细节），而我们的 ``sys.stderr`` 是一个临时 ``StringIO``：**不还原的话**，
    根 logger 上会永久挂一个写向"已经没人看的 buffer"的 handler，
    后续在同一个进程里运行的其它测试文件的日志就被吞了。这是最常见的
    "单跑绿、全量跑红"来源之一，所以在这一层统一挡住。
    """
    root = logging.getLogger()
    cli_logger = logging.getLogger("liteagent.cli")
    saved_handlers = root.handlers[:]
    saved_root_level = root.level
    saved_cli_level = cli_logger.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_root_level)
        cli_logger.setLevel(saved_cli_level)


def invoke(
    argv: list[str],
    *,
    stdin: str | None = None,
    lines: list[str] | None = None,
) -> tuple[int, str, str]:
    """调用 ``cli.main(argv)``，返回 ``(退出码, stdout, stderr)``。

    * 环境变量**全部清空**：宿主机上的 ``LITEAGENT_TRACE`` 会让一次普通的 ``run``
      偷偷写出一个 trace 文件，``LITEAGENT_SANDBOX_ROOT`` 会改变 ``--tools files``
      的沙箱根 —— 两者都会让断言随机器而变。
    * ``stdin`` 给定字符串时替换 ``sys.stdin``（测 ``--stdin`` 与 ``multi`` 的管道输入）。
    * ``lines`` 给定时替换 ``builtins.input``：队列耗尽后抛 ``EOFError``（**不是**
      ``StopIteration`` —— 后者会被 mock 吞成"测试写错了"这一类的假失败，
      而 REPL 的干净退出路径恰恰靠 ``EOFError``）。
    """
    out, err = io.StringIO(), io.StringIO()
    queue = list(lines) if lines is not None else None

    def _fake_input(prompt: str = "") -> str:
        if queue:
            return queue.pop(0)
        raise EOFError

    with _restore_logging():
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            if stdin is not None:
                stack.enter_context(mock.patch.object(sys, "stdin", io.StringIO(stdin)))
            if queue is not None:
                stack.enter_context(mock.patch("builtins.input", side_effect=_fake_input))
            code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTestCase(unittest.TestCase):
    """公共前置：清空环境变量 + 一个临时目录（``--sandbox-root`` / 输出文件用）。"""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self._tmp = tempfile.TemporaryDirectory(prefix="liteagent-cli-")
        self.addCleanup(self._tmp.cleanup)

    @property
    def tmpdir(self) -> str:
        return self._tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.tmpdir, name)

    # ---- 小工具 ----

    def registry_for(self, argv: list[str]):
        """按命令行参数造注册表（直接调冻结的装配函数，绕开 argparse 之外的噪声）。"""
        parser = cli.build_parser()
        return cli.build_registry_from_args(parser.parse_args(list(argv)))


# ======================================================================================
# 顶层分发：退出码契约（§11）
# ======================================================================================


class MainDispatchTest(CliTestCase):
    """退出码常量、``--version`` / ``-h`` / 无子命令 / 未知子命令。"""

    def test_exit_code_constants_are_frozen(self) -> None:
        self.assertEqual(0, cli.EXIT_OK)
        self.assertEqual(1, cli.EXIT_AGENT_FAILED)
        self.assertEqual(2, cli.EXIT_USAGE)
        self.assertEqual(3, cli.EXIT_PROVIDER_ERROR)
        self.assertEqual("liteagent", cli.PROG)

    def test_version_subcommand_returns_zero(self) -> None:
        code, out, err = invoke(["version"])
        self.assertEqual(0, code, msg=f"stderr={err!r}")
        self.assertIn(__version__, out)

    def test_version_subcommand_prints_only_the_version(self) -> None:
        code, out, _ = invoke(["version"])
        self.assertEqual(0, code)
        self.assertEqual(out.strip(), __version__)
        self.assertEqual(1, len(out.strip().splitlines()))

    def test_dashdash_version_returns_zero(self) -> None:
        code, out, _ = invoke(["--version"])
        self.assertEqual(0, code)
        self.assertTrue(out.startswith("liteagent "), msg=out)

    def test_help_returns_zero(self) -> None:
        code, out, _ = invoke(["-h"])
        self.assertEqual(0, code)
        self.assertIn("usage:", out)

    def test_no_command_prints_help_and_returns_usage(self) -> None:
        """§11：没有子命令 -> EXIT_USAGE，且帮助信息走 stderr（stdout 保持干净）。"""
        code, out, err = invoke([])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("usage:", err)

    def test_unknown_subcommand_returns_usage(self) -> None:
        code, out, err = invoke(["definitely-not-a-command"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("invalid choice", err)

    def test_main_returns_int_and_never_raises_systemexit(self) -> None:
        """§11 冻结："返回退出码（不调用 ``sys.exit``），便于测试直接断言返回值"。"""
        for argv in (["version"], [], ["--version"], ["-h"], ["bogus"]):
            with self.subTest(argv=argv):
                code, _, _ = invoke(argv)
                self.assertIsInstance(code, int, msg=f"{argv} 必须返回 int，不能抛 SystemExit")

    def test_main_accepts_argv_none_without_crashing(self) -> None:
        """``main(None)`` 读 ``sys.argv``；用 ``patch.object`` 给一个确定的 argv。"""
        out, err = io.StringIO(), io.StringIO()
        with _restore_logging():
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(sys, "argv", ["liteagent", "version"]):
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = cli.main(None)
        self.assertEqual(0, code)
        self.assertIn(__version__, out.getvalue())

    def test_build_parser_returns_a_fresh_parser_each_call(self) -> None:
        """每次调用都是新解析器：否则并行测试会共享 prog/子命令状态。"""
        first, second = cli.build_parser(), cli.build_parser()
        self.assertIsNot(first, second)
        self.assertEqual(cli.PROG, first.prog)

    def test_main_py_is_runnable_as_a_module(self) -> None:
        """``python3 -m liteagent version`` 必须真的能跑（``__main__.py`` 的守门用例）。"""
        proc = subprocess.run(
            [sys.executable, "-m", "liteagent", "version"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=60,
        )
        self.assertEqual(0, proc.returncode, msg=proc.stderr)
        self.assertIn(__version__, proc.stdout)


# ======================================================================================
# run 子命令
# ======================================================================================


class RunCommandTest(CliTestCase):
    """``run`` 的提示词来源、退出码映射、``--json`` stdout 纯净性。"""

    def test_run_json_is_loadable(self) -> None:
        code, out, err = invoke(["run", "-p", "hello", "--provider", "echo", "--json"])
        self.assertEqual(0, code, msg=f"stderr={err!r}")
        payload = json.loads(out)  # 解析失败即失败：这是 §12 的硬性覆盖点
        self.assertEqual("FINISHED", payload["status"])
        self.assertIn("hello", payload["output"])
        self.assertIsInstance(payload["usage"], dict)

    def test_run_json_stdout_is_pure_json(self) -> None:
        """§11：``--json`` 时 stdout **只有**那段 JSON（保证可被 ``jq`` 消费）。"""
        code, out, err = invoke(["run", "-p", "pure", "--provider", "echo", "--json"])
        self.assertEqual(0, code, msg=err)
        # 整段 stdout 必须能被 json.loads 吃完：任何一行诊断都会让它失败。
        payload = json.loads(out)
        self.assertNotIn("Traceback", out)
        self.assertIn("pure", payload["output"])
        self.assertEqual(set(payload), {
            "output", "status", "steps", "tool_calls", "tool_results",
            "usage", "error", "duration_ms", "agent_name", "metadata",
        })

    def test_run_text_output_uses_frozen_layout(self) -> None:
        code, out, _ = invoke(["run", "-p", "text-mode", "--provider", "echo"])
        self.assertEqual(0, code)
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("status: "), msg=lines[0])
        self.assertIn("---", lines)
        self.assertIn("text-mode", out)

    def test_run_without_prompt_returns_usage(self) -> None:
        code, out, err = invoke(["run", "--provider", "echo"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("no prompt", err)

    def test_run_prompt_source_flags_are_mutually_exclusive(self) -> None:
        code, _, err = invoke(["run", "-p", "a", "-f", "b", "--provider", "echo"])
        self.assertEqual(2, code)
        self.assertIn("not allowed with", err)

    def test_run_reads_prompt_from_file(self) -> None:
        prompt_file = self.path("prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write("from-a-file")
        code, out, err = invoke(["run", "-f", prompt_file, "--provider", "echo", "--json"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("from-a-file", json.loads(out)["output"])

    def test_run_missing_prompt_file_returns_usage(self) -> None:
        code, out, err = invoke(["run", "-f", self.path("nope.txt"), "--provider", "echo"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("cannot read prompt file", err)

    def test_run_reads_prompt_from_stdin(self) -> None:
        code, out, err = invoke(
            ["run", "--stdin", "--provider", "echo", "--json"], stdin="piped-input"
        )
        self.assertEqual(0, code, msg=err)
        self.assertIn("piped-input", json.loads(out)["output"])

    def test_run_trace_writes_jsonl_file(self) -> None:
        trace_path = self.path("run.jsonl")
        code, _, err = invoke(
            ["run", "-p", "traced", "--provider", "echo", "--trace", trace_path, "--json"]
        )
        self.assertEqual(0, code, msg=err)
        events = load_trace(trace_path)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual("run_started", events[0].type.value)

    def test_run_unknown_provider_is_a_config_error(self) -> None:
        """未知 provider 是 ``ConfigError`` -> EXIT_USAGE（不是 provider 错误码 3）。"""
        code, out, err = invoke(["run", "-p", "x", "--provider", "not-a-provider"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("not-a-provider", err)

    def test_run_llm_error_maps_to_provider_error_code(self) -> None:
        """§11 冻结映射：``LLMError`` -> ``EXIT_PROVIDER_ERROR``(3)。

        用 patch 而不是真 provider：``Agent`` 会把 LLM 层的失败收敛成
        ``AgentResult(status=FAILED)``（红线 6），所以"退出码 3"这条路径
        只能在装配期失败时才可见。``patch.object`` 打在 ``cli`` 上而不是
        ``liteagent.llm.registry`` 上 —— 后者会被其它测试文件同时使用。
        """
        boom = LLMAuthError("missing API key for provider 'openai'")
        with mock.patch.object(cli, "_build_llm", side_effect=boom):
            code, out, err = invoke(["run", "-p", "x", "--provider", "echo"])
        self.assertEqual(3, code)
        self.assertEqual("", out)
        self.assertIn("missing API key", err)

    def test_run_agent_failure_returns_one(self) -> None:
        """``AgentResult`` 非 ok -> EXIT_AGENT_FAILED(1)。"""
        failing = AgentResult(
            output="partial",
            status=AgentStatus.FAILED,
            error=AgentError("model output could not be parsed"),
            agent_name="fake",
        )
        with mock.patch.object(cli, "_load_agent_class") as loader:
            loader.return_value.return_value.run.return_value = failing
            code, out, err = invoke(["run", "-p", "x", "--provider", "echo"])
        self.assertEqual(1, code, msg=err)
        self.assertIn("status: FAILED", out)

    def test_unexpected_exception_returns_one_and_stdout_has_no_traceback(self) -> None:
        """§11：``main`` **保证不向 stdout 打印 traceback**（-v 时打印到 stderr）。"""
        with mock.patch.object(cli, "_load_agent_class", side_effect=RuntimeError("kaboom")):
            code, out, err = invoke(["run", "-p", "x", "--provider", "echo"])
        self.assertEqual(1, code)
        self.assertEqual("", out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("kaboom", out)
        self.assertIn("kaboom", err)
        self.assertNotIn("Traceback", err)  # 非 -v 时连 stderr 也只有一行人话

    def test_verbose_sends_traceback_to_stderr_only(self) -> None:
        with mock.patch.object(cli, "_load_agent_class", side_effect=RuntimeError("kaboom")):
            code, out, err = invoke(["run", "-p", "x", "--provider", "echo", "-v"])
        self.assertEqual(1, code)
        self.assertEqual("", out)
        self.assertNotIn("Traceback", out)
        self.assertIn("Traceback", err)

    def test_tools_show_unknown_name_keeps_stdout_clean(self) -> None:
        """同一条"stdout 无 traceback"的约束在 tools 路径上的复现。"""
        code, out, err = invoke(["tools", "show", "definitely-not-a-tool"])
        self.assertEqual(1, code)
        self.assertEqual("", out)
        self.assertNotIn("Traceback", out)
        self.assertIn("available", err)

    def test_no_tools_and_no_builtin_conflict_returns_usage(self) -> None:
        """§11：两者同时出现 -> ``EXIT_USAGE`` + 明确错误信息。"""
        code, out, err = invoke(
            ["run", "-p", "x", "--provider", "echo", "--no-tools", "--no-builtin"]
        )
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("--no-builtin", err)
        self.assertIn("--no-tools", err)

    def test_registry_builder_rejects_no_tools_plus_no_builtin(self) -> None:
        """手工构造的 ``Namespace`` 绕过 argparse 时，装配层必须**自己**报错。"""
        parser = cli.build_parser()
        args = parser.parse_args(["run", "-p", "x", "--no-tools"])
        args.no_builtin = True  # 模拟"同时给了两个旗标"的 Namespace
        with self.assertRaises(ConfigError) as ctx:
            cli.build_registry_from_args(args)
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_max_total_tokens_zero_means_unlimited(self) -> None:
        """§11：``--max-total-tokens 0`` 读作"不限预算"，而不是"立刻超预算"。"""
        parser = cli.build_parser()
        args = parser.parse_args(
            ["run", "-p", "x", "--provider", "echo", "--max-total-tokens", "0"]
        )
        with mock.patch.object(cli, "_load_agent_class") as loader:
            cli.build_agent_from_args(args)
        config = loader.return_value.call_args.kwargs["config"]
        self.assertIsNone(config.max_total_tokens)

    def test_max_wall_clock_zero_means_unlimited(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            ["run", "-p", "x", "--provider", "echo", "--max-wall-clock", "0"]
        )
        with mock.patch.object(cli, "_load_agent_class") as loader:
            cli.build_agent_from_args(args)
        config = loader.return_value.call_args.kwargs["config"]
        self.assertIsNone(config.max_wall_clock_s)


# ======================================================================================
# --tools 的三种结果（§11 冻结的"必须有三条区分用例"）
# ======================================================================================


class ToolSelectionTest(CliTestCase):
    """``--tools read_file``（1） / ``--tools files``（5） / ``--tools ""``（0） 必须可区分。

    §11 原文："**必须有三条区分用例**"。这三条是本文件里最容易被"实现偷偷改成
    同一件事"的契约 —— ``None``（全部）与 ``[]``（零个）一旦混淆，
    ``--tools ""`` 会静默退回"注册全部"，而用户以为自己把工具全关了。
    """

    def test_tools_read_file_registers_exactly_one(self) -> None:
        registry = self.registry_for(
            ["tools", "list", "--tools", "read_file", "--sandbox-root", self.tmpdir]
        )
        self.assertEqual(["read_file"], registry.names())

    def test_tools_files_group_registers_five(self) -> None:
        registry = self.registry_for(
            ["tools", "list", "--tools", "files", "--sandbox-root", self.tmpdir]
        )
        self.assertEqual(5, len(registry), msg=str(registry.names()))
        self.assertEqual(
            {"read_file", "write_file", "list_dir", "search_files", "delete_file"},
            set(registry.names()),
        )

    def test_tools_empty_string_registers_zero(self) -> None:
        """``--tools ""`` -> ``include=[]`` -> 注册 0 个工具（**不是**"全部"）。"""
        registry = self.registry_for(
            ["tools", "list", "--tools", "", "--sandbox-root", self.tmpdir]
        )
        self.assertEqual([], registry.names())
        self.assertEqual(0, len(registry))

    def test_three_forms_are_pairwise_distinct(self) -> None:
        """把三条放在一起断言"两两不同"，防止将来有人把其中两条合并成一条语义。"""
        one = self.registry_for(
            ["tools", "list", "--tools", "read_file", "--sandbox-root", self.tmpdir]
        )
        group = self.registry_for(
            ["tools", "list", "--tools", "files", "--sandbox-root", self.tmpdir]
        )
        empty = self.registry_for(
            ["tools", "list", "--tools", "", "--sandbox-root", self.tmpdir]
        )
        counts = (len(one), len(group), len(empty))
        self.assertEqual((1, 5, 0), counts)
        self.assertEqual(len(set(counts)), 3, msg=f"三类必须互不相同，得到 {counts}")

    def test_no_tools_flag_is_equivalent_to_empty_string(self) -> None:
        """§11：``--no-tools`` 等价 ``--tools ""``。"""
        flag = self.registry_for(
            ["tools", "list", "--no-tools", "--sandbox-root", self.tmpdir]
        )
        empty = self.registry_for(
            ["tools", "list", "--tools", "", "--sandbox-root", self.tmpdir]
        )
        self.assertEqual([], flag.names())
        self.assertEqual(flag.names(), empty.names())

    def test_no_builtin_registers_only_tools_declared_in_config(self) -> None:
        """§11：``--no-builtin`` 只注册 ``--config`` 声明的工具。"""
        config_path = self.path("app.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"tools": ["read_file"]}, handle)
        registry = self.registry_for(
            [
                "tools", "list", "--no-builtin", "--config", config_path,
                "--sandbox-root", self.tmpdir,
            ]
        )
        self.assertEqual(["read_file"], registry.names())

        # 没有 --config 声明时 -> 0 个工具（而不是回退成"全部内置"）。
        bare = self.registry_for(["tools", "list", "--no-builtin", "--sandbox-root", self.tmpdir])
        self.assertEqual([], bare.names())

    def test_default_registers_every_builtin_group(self) -> None:
        registry = self.registry_for(["tools", "list", "--sandbox-root", self.tmpdir])
        self.assertGreaterEqual(len(registry), 13)
        for expected in ("read_file", "write_file", "run_shell", "web_search", "remember"):
            self.assertIn(expected, registry.names())

    def test_parse_tool_list_helper(self) -> None:
        """逗号分隔 + 去空白；空串与纯分隔符都退化成空列表。"""
        self.assertEqual(["a", "b"], cli._parse_tool_list("a,b"))
        self.assertEqual(["a", "b"], cli._parse_tool_list(" a , b "))
        self.assertEqual([], cli._parse_tool_list(""))
        self.assertEqual([], cli._parse_tool_list(","))

    def test_unknown_tool_entry_is_a_config_error(self) -> None:
        code, out, err = invoke(["tools", "list", "--tools", "not_a_tool"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("unknown include entry", err)


# ======================================================================================
# tools list / show / schema
# ======================================================================================


class ToolsCommandTest(CliTestCase):
    """``tools`` 的三个动作与两种输出格式。"""

    def test_tools_without_action_returns_usage(self) -> None:
        code, out, err = invoke(["tools"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("usage", err.lower())

    def test_tools_list_table(self) -> None:
        code, out, err = invoke(["tools", "list"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("name", out)
        self.assertIn("read_file", out)
        self.assertIn("dangerous", out)

    def test_tools_list_json(self) -> None:
        code, out, err = invoke(["tools", "list", "--format", "json"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        names = {entry["name"] for entry in payload}
        self.assertIn("read_file", names)
        self.assertIn("description", payload[0])

    def test_tools_list_tags_filter(self) -> None:
        """``--tags`` 过滤；被标成 ``dangerous`` 的工具默认不可见（§7.5 的默认安全姿势）。"""
        code, out, err = invoke(["tools", "list", "--tags", "fs", "--format", "json"])
        self.assertEqual(0, code, msg=err)
        names = {entry["name"] for entry in json.loads(out)}
        self.assertEqual({"read_file", "list_dir", "search_files"}, names)

        code, out, err = invoke(
            ["tools", "list", "--tags", "fs", "--include-dangerous", "--format", "json"]
        )
        self.assertEqual(0, code, msg=err)
        names = {entry["name"] for entry in json.loads(out)}
        self.assertEqual(
            {"read_file", "write_file", "list_dir", "search_files", "delete_file"}, names
        )

    def test_tools_show_markdown(self) -> None:
        code, out, err = invoke(["tools", "show", "read_file"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("read_file", out)
        self.assertIn("| parameter | type | required | default | description |", out)

    def test_tools_show_json_format(self) -> None:
        code, out, err = invoke(["tools", "show", "read_file", "--format", "json"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertEqual("read_file", payload["name"])
        self.assertIn("signature", payload)

    def test_tools_show_with_schema_json_has_both_provider_shapes(self) -> None:
        code, out, err = invoke(["tools", "show", "read_file", "--show-schema", "--format", "json"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertEqual({"tool", "openai", "anthropic"}, set(payload))
        self.assertEqual("read_file", payload["openai"]["function"]["name"])
        self.assertEqual("read_file", payload["anthropic"]["name"])

    def test_tools_show_with_schema_markdown_embeds_openai_json(self) -> None:
        code, out, err = invoke(["tools", "show", "read_file", "--show-schema"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("## schema (openai)", out)
        self.assertIn('"parameters"', out)

    def test_tools_show_unknown_tool_returns_agent_failed(self) -> None:
        code, _, err = invoke(["tools", "show", "definitely-not-a-tool"])
        self.assertEqual(1, code)
        self.assertIn("tool not found", err)

    def test_tools_show_missing_name_returns_usage(self) -> None:
        code, out, err = invoke(["tools", "show"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("usage", err.lower())

    def test_tools_schema_openai_is_valid_json(self) -> None:
        """§12 硬性覆盖点：``tools schema --format openai`` 的输出必须是合法 JSON。"""
        code, out, err = invoke(["tools", "schema", "--format", "openai"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 13)
        self.assertEqual("function", payload[0]["type"])
        self.assertIn("name", payload[0]["function"])
        self.assertIn("parameters", payload[0]["function"])

    def test_tools_schema_anthropic_is_valid_json(self) -> None:
        code, out, err = invoke(["tools", "schema", "--format", "anthropic"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertGreaterEqual(len(payload), 13)
        for entry in payload:
            self.assertIn("input_schema", entry)
            self.assertNotIn("function", entry)  # anthropic 形态不套 "function" 外壳
        names = {entry["name"] for entry in payload}
        self.assertIn("read_file", names)

    def test_tools_schema_name_filter(self) -> None:
        code, out, err = invoke(["tools", "schema", "read_file", "--format", "openai"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertEqual(1, len(payload))
        self.assertEqual("read_file", payload[0]["function"]["name"])

    def test_tools_schema_writes_output_file_and_keeps_stdout_clean(self) -> None:
        target = self.path("schemas.json")
        code, out, err = invoke(["tools", "schema", "--format", "openai", "-o", target])
        self.assertEqual(0, code, msg=err)
        self.assertEqual("", out)
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertGreaterEqual(len(payload), 13)

    def test_tools_schema_unknown_name_returns_agent_failed(self) -> None:
        code, _, err = invoke(["tools", "schema", "definitely-not-a-tool"])
        self.assertEqual(1, code)
        self.assertIn("tool not found", err)

    def test_top_level_schema_command_returns_json(self) -> None:
        """顶层的 ``schema``（另一个子命令，与 ``tools schema`` 不同）。"""
        code, out, err = invoke(["schema", "--format", "openai"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 13)


# ======================================================================================
# trace 子命令
# ======================================================================================


class TraceCommandTest(CliTestCase):
    """``trace FILE`` / ``--stats --json`` / 各种过滤器。

    trace 文件由**一次真实的 echo run** 产生（离线、确定性），而不是手搓 JSONL：
    手搓的 fixture 只能证明"解析器读得懂我编的格式"，证明不了"CLI 写出来的东西
    CLI 自己读得回来"。
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._dir = tempfile.TemporaryDirectory(prefix="liteagent-cli-trace-")
        cls.trace_path = os.path.join(cls._dir.name, "trace.jsonl")
        out, err = io.StringIO(), io.StringIO()
        with _restore_logging():
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                stack.enter_context(contextlib.redirect_stdout(out))
                stack.enter_context(contextlib.redirect_stderr(err))
                cls.producer_code = cli.main(
                    ["run", "-p", "hi", "--provider", "echo", "--trace", cls.trace_path, "--json"]
                )
        cls.producer_stderr = err.getvalue()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(0, self.producer_code, msg=f"trace fixture 没造出来: {self.producer_stderr!r}")
        self.assertTrue(os.path.exists(self.trace_path))

    def test_trace_renders_events(self) -> None:
        code, out, err = invoke(["trace", self.trace_path])
        self.assertEqual(0, code, msg=err)
        self.assertIn("run_started", out)
        self.assertIn("run_finished", out)

    def test_trace_json_is_a_list_of_event_dicts(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--json"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertIn("run_started", [event["type"] for event in payload])

    def test_trace_stats_json_matches_the_frozen_function(self) -> None:
        """§11 冻结：输出逐字等于 ``json.dumps(trace_stats(load_trace(FILE)), ...)``。"""
        code, out, err = invoke(["trace", self.trace_path, "--stats", "--json"])
        self.assertEqual(0, code, msg=err)
        expected = json.dumps(
            trace_stats(load_trace(self.trace_path)), ensure_ascii=False, indent=2
        )
        self.assertEqual(expected, out.strip())
        stats = json.loads(out)
        self.assertEqual(1, stats["runs"])
        self.assertIn("usage", stats)

    def test_trace_stats_human_readable(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--stats"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("runs:", out)
        self.assertIn("steps:", out)

    def test_trace_type_filter(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--type", "run_started", "--json"])
        self.assertEqual(0, code, msg=err)
        payload = json.loads(out)
        self.assertTrue(payload)
        self.assertEqual({"run_started"}, {event["type"] for event in payload})

    def test_trace_agent_filter_keeps_everything_for_the_echo_agent(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--agent", "agent", "--json"])
        self.assertEqual(0, code, msg=err)
        self.assertTrue(json.loads(out))

    def test_trace_step_filter(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--step", "0", "--json"])
        self.assertEqual(0, code, msg=err)
        for event in json.loads(out):
            self.assertEqual(0, event["step"])

    def test_trace_limit_keeps_the_last_n(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--limit", "1", "--json"])
        self.assertEqual(0, code, msg=err)
        self.assertEqual(1, len(json.loads(out)))

    def test_trace_unknown_type_is_a_config_error(self) -> None:
        code, out, err = invoke(["trace", self.trace_path, "--type", "not-an-event"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("unknown event type", err)

    def test_trace_missing_file_warns_but_returns_zero(self) -> None:
        """坏/缺文件不该让 CLI 崩：给一行 warning，退出码仍是 0（只读命令失败得无害）。"""
        code, out, err = invoke(["trace", self.path("does-not-exist.jsonl")])
        self.assertEqual(0, code)
        self.assertIn("warning", err)
        self.assertIn("0 event(s)", out)

    def test_trace_missing_file_stats_json_is_still_loadable(self) -> None:
        code, out, err = invoke(["trace", self.path("does-not-exist.jsonl"), "--stats", "--json"])
        self.assertEqual(0, code, msg=err)
        self.assertEqual(0, json.loads(out)["runs"])


# ======================================================================================
# chat 子命令
# ======================================================================================


class ChatCommandTest(CliTestCase):
    """交互式 REPL：REPL 循环、内建命令、EOF 干净退出。"""

    def test_chat_quit_returns_zero_and_says_bye(self) -> None:
        code, out, err = invoke(["chat", "--provider", "echo"], lines=["/quit"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("bye", out)

    def test_chat_eof_is_a_clean_exit(self) -> None:
        """非交互 stdin（管道）下 EOF 不是错误：干净退出、退出码 0。"""
        code, out, err = invoke(["chat", "--provider", "echo"], lines=[])
        self.assertEqual(0, code, msg=err)
        self.assertIn("bye", out)
        self.assertNotIn("Traceback", out)

    def test_chat_runs_a_turn_through_the_agent(self) -> None:
        code, out, err = invoke(["chat", "--provider", "echo"], lines=["hello there"])
        self.assertEqual(0, code, msg=err)
        self.assertIn("echo: received", out)
        self.assertIn("hello there", out)

    def test_chat_help_lists_every_frozen_command(self) -> None:
        code, out, _ = invoke(["chat", "--provider", "echo"], lines=["/help"])
        self.assertEqual(0, code)
        for command in ("/help", "/reset", "/tools", "/memory", "/trace", "/stats", "/mode", "/quit"):
            self.assertIn(command, out)
        self.assertEqual(
            set(cli._CHAT_COMMANDS),
            {"/help", "/reset", "/tools", "/memory", "/trace", "/stats", "/mode", "/quit"},
        )

    def test_chat_tools_command_lists_the_registry(self) -> None:
        code, out, _ = invoke(["chat", "--provider", "echo"], lines=["/tools"])
        self.assertEqual(0, code)
        self.assertIn("read_file", out)
        self.assertIn("tool(s)", out)

    def test_chat_memory_and_stats_commands(self) -> None:
        code, out, _ = invoke(["chat", "--provider", "echo"], lines=["/memory", "/stats"])
        self.assertEqual(0, code)
        self.assertIn("calls: 0", out)
        self.assertIn("cost=", out)
        self.assertIn("long_term_items", out)

    def test_chat_mode_command_switches_and_rejects_bad_values(self) -> None:
        code, out, err = invoke(
            ["chat", "--provider", "echo"], lines=["/mode native", "/mode bogus", "/quit"]
        )
        self.assertEqual(0, code, msg=err)
        self.assertIn("mode = native", out)
        self.assertIn("/mode expects one of auto/native/text", err)

    def test_chat_reset_and_unknown_command(self) -> None:
        code, out, err = invoke(
            ["chat", "--provider", "echo"], lines=["/reset", "/definitely-not", "/quit"]
        )
        self.assertEqual(0, code, msg=err)
        self.assertIn("session reset", out)
        self.assertIn("unknown command", err)

    def test_chat_trace_command_reports_missing_trace_file(self) -> None:
        code, out, _ = invoke(["chat", "--provider", "echo"], lines=["/trace", "/quit"])
        self.assertEqual(0, code)
        self.assertIn("no trace file", out)

    def test_chat_trace_command_reports_event_count(self) -> None:
        trace_path = self.path("chat.jsonl")
        code, out, _ = invoke(
            ["chat", "--provider", "echo", "--trace", trace_path], lines=["/trace", "/quit"]
        )
        self.assertEqual(0, code)
        self.assertIn("trace file: ", out)
        self.assertIn("0 event(s)", out)

    def test_chat_command_table_is_not_empty(self) -> None:
        """守门：内建命令清单一旦被清空，REPL 就成了"什么都不能做"的循环。"""
        self.assertGreaterEqual(len(cli._CHAT_COMMANDS), 8)
        self.assertEqual(len(set(cli._CHAT_COMMANDS)), len(cli._CHAT_COMMANDS))


# ======================================================================================
# render_result（§11 冻结格式）
# ======================================================================================


class RenderResultTest(CliTestCase):
    """``render_result`` 的两种格式；文本形态必须与 §11 的样例**逐字**一致。"""

    def _result(self, **overrides) -> AgentResult:
        payload = dict(
            output="the answer",
            status=AgentStatus.FINISHED,
            steps=3,
            usage=TokenUsage(prompt_tokens=120, completion_tokens=45),
            duration_ms=1234.5,
            metadata={"cost_usd": 0.000123},
        )
        payload.update(overrides)
        return AgentResult(**payload)

    def test_text_format_matches_the_frozen_sample(self) -> None:
        rendered = cli.render_result(self._result())
        self.assertEqual(
            "status: FINISHED\n"
            "steps: 3\n"
            "tokens: prompt=120 completion=45 total=165\n"
            "cost: $0.000123\n"
            "duration: 1234.5ms\n"
            "---\n"
            "the answer",
            rendered,
        )

    def test_text_format_appends_error_line_only_when_failed(self) -> None:
        rendered = cli.render_result(self._result(status=AgentStatus.FAILED, error=AgentError("boom")))
        self.assertIn("error: AgentError: boom", rendered)
        self.assertLess(rendered.index("error:"), rendered.index("---"))
        # 成功时**不得**出现 error 行（否则与冻结样例不再逐字一致）。
        self.assertNotIn("error:", cli.render_result(self._result()))

    def test_text_format_reports_missing_cost_as_na(self) -> None:
        rendered = cli.render_result(self._result(metadata={}))
        self.assertIn("cost: n/a", rendered)

    def test_json_format_round_trips_to_dict(self) -> None:
        result = self._result()
        rendered = cli.render_result(result, as_json=True)
        payload = json.loads(rendered)
        self.assertEqual(result.to_dict(), payload)
        # 中文与缩进：ensure_ascii=False + indent=2（§11 冻结）。
        self.assertIn("\n  \"output\"", rendered)

    def test_json_format_keeps_non_ascii_readable(self) -> None:
        rendered = cli.render_result(self._result(output="中文输出"), as_json=True)
        self.assertIn("中文输出", rendered)
        self.assertNotIn("\\u4e2d", rendered)

    def test_json_format_is_deterministic_across_calls(self) -> None:
        result = self._result()
        self.assertEqual(
            cli.render_result(result, as_json=True), cli.render_result(result, as_json=True)
        )


# ======================================================================================
# 装配函数（CLI 与 examples 共用的公共入口）
# ======================================================================================


class AssemblyHelpersTest(CliTestCase):
    """``build_registry_from_args`` / ``build_memory_from_args`` / ``build_agent_from_args``。"""

    def test_build_agent_from_args_accepts_a_manual_namespace(self) -> None:
        """命令实现与装配函数被单测**直接**调用时会拿到手工构造的 Namespace。"""
        import argparse

        args = argparse.Namespace(provider="echo", prompt="x")
        agent = cli.build_agent_from_args(args)
        self.assertEqual("agent", agent.name)
        self.assertGreaterEqual(len(agent.tools), 13)
        self.assertTrue(hasattr(agent, "memory"))

    def test_build_memory_from_args_respects_memory_flags(self) -> None:
        parser = cli.build_parser()
        default = cli.build_memory_from_args(parser.parse_args(["run", "-p", "x", "--provider", "echo"]))
        self.assertTrue(default.config.long_term_enabled)

        no_memory = cli.build_memory_from_args(
            parser.parse_args(["run", "-p", "x", "--provider", "echo", "--no-memory"])
        )
        self.assertFalse(no_memory.config.long_term_enabled)
        self.assertFalse(no_memory.config.summary_enabled)

        no_long_term = cli.build_memory_from_args(
            parser.parse_args(["run", "-p", "x", "--provider", "echo", "--no-long-term"])
        )
        self.assertFalse(no_long_term.config.long_term_enabled)
        self.assertTrue(no_long_term.config.summary_enabled)

    def test_memory_max_tokens_overrides_the_window_budget(self) -> None:
        parser = cli.build_parser()
        manager = cli.build_memory_from_args(
            parser.parse_args(["run", "-p", "x", "--provider", "echo", "--memory-max-tokens", "123"])
        )
        self.assertEqual(123, manager.config.buffer_max_tokens)

    def test_approve_never_is_the_default_and_disables_approval(self) -> None:
        """§11：``never``（默认）= ``approval_policy=None``。"""
        parser = cli.build_parser()
        for argv in (["run", "-p", "x", "--provider", "echo"],
                     ["run", "-p", "x", "--provider", "echo", "--approve", "never"]):
            with self.subTest(argv=argv):
                agent = cli.build_agent_from_args(parser.parse_args(list(argv)))
                self.assertIsNone(agent.executor.config.approval_policy)

    def test_approve_write_allows_only_write_tools(self) -> None:
        parser = cli.build_parser()
        agent = cli.build_agent_from_args(
            parser.parse_args(["run", "-p", "x", "--provider", "echo", "--approve", "write"])
        )
        policy = agent.executor.config.approval_policy
        self.assertIsNotNone(policy)
        write_call = mock.Mock(name="call")
        write_call.name = "write_file"
        read_call = mock.Mock(name="call")
        read_call.name = "read_file"
        tool = mock.Mock(name="tool")
        self.assertTrue(policy(write_call, tool))
        self.assertFalse(policy(read_call, tool))

    def test_approve_all_allows_everything(self) -> None:
        parser = cli.build_parser()
        agent = cli.build_agent_from_args(
            parser.parse_args(["run", "-p", "x", "--provider", "echo", "--approve", "all"])
        )
        policy = agent.executor.config.approval_policy
        call = mock.Mock(name="call")
        call.name = "run_shell"
        self.assertTrue(policy(call, mock.Mock(name="tool")))

    def test_build_registry_from_args_returns_a_registry(self) -> None:
        parser = cli.build_parser()
        registry = cli.build_registry_from_args(parser.parse_args(["tools", "list"]))
        self.assertIn("read_file", registry.names())
        self.assertEqual(len(registry.names()), len(set(registry.names())))

    def test_run_subparser_rejects_unknown_mode(self) -> None:
        code, _, err = invoke(["run", "-p", "x", "--provider", "echo", "--mode", "psychic"])
        self.assertEqual(2, code)
        self.assertIn("invalid choice", err)

    def test_multi_requires_mode_and_agents(self) -> None:
        code, out, err = invoke(["multi"])
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("--mode", err)


class FrozenSignatureTests(unittest.TestCase):
    """§11 冻结的工厂函数签名必须与 INTERFACES.md 逐字一致（参数名、默认值、返回类型）。

    `[v3 回归]` `build_agent_from_args` 曾经被写成 `-> Any`：`Agent` 当时是延迟加载的，
    作者为了不 import 它就放宽了注解。结果是 `inspect.signature` / `typing.get_type_hints`
    都拿不到返回类型，而 §11 冻结的是 `-> Agent`（同组的 `-> ToolRegistry` /
    `-> MemoryManager` / `-> str` 都逐字一致）。修复后这里把它钉死。
    """

    def test_build_agent_from_args_returns_agent(self) -> None:
        import inspect
        import typing

        from liteagent.agent.agent import Agent

        signature = inspect.signature(cli.build_agent_from_args)
        self.assertEqual(str(signature), "(args: 'argparse.Namespace') -> 'Agent'")
        self.assertIs(typing.get_type_hints(cli.build_agent_from_args)["return"], Agent)

    def test_sibling_factories_keep_their_frozen_annotations(self) -> None:
        import typing

        from liteagent.memory.manager import MemoryManager
        from liteagent.tools.registry import ToolRegistry

        self.assertIs(
            typing.get_type_hints(cli.build_registry_from_args)["return"], ToolRegistry
        )
        self.assertIs(
            typing.get_type_hints(cli.build_memory_from_args)["return"], MemoryManager
        )
        self.assertIs(typing.get_type_hints(cli.render_result)["return"], str)


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
