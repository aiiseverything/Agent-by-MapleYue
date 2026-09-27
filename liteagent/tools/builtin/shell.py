from __future__ import annotations

"""内置 shell 工具（规范 §7.5 的 ``shell.py``，**不含 `python_exec`** —— 它在 ``code.py``）。

安全模型是"**双层闸 + 默认关闭**"：

1. **默认关闭**：``LITEAGENT_ALLOW_SHELL`` 不是 ``"1"`` 时 ``run_shell`` 依然**可见**
   （模型知道这个能力存在），但调用即返回
   ``'shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)'``。
   工具可见而不可用，是为了让模型的"我能做什么"与真实能力一致 —— 直接隐藏工具会让它
   反复尝试不存在的替代方案。
2. **拒绝列表**（``SHELL_DENY_PATTERNS``）：纯函数 ``check_command_allowed`` 逐条匹配，
   命中即抛 ``SandboxViolationError``（``path=command``、``root="denylist:<reason>"``，
   §3.3 的冻结形态）。
3. **执行目录**永远是沙箱根（``cwd`` 只做越界校验），于是"``cd /`` + 相对路径操作"
   这条最容易被忽略的逃逸路径从根上不存在。

**拒绝列表不是安全边界**（诚实声明，写进 ``docs/TOOLS.md`` 的话术一致）：正则匹配
总能被编码/变量展开绕过。它是"防手滑 + 防模型幻觉式破坏"，真正的隔离靠容器/沙箱
（例如生产里把 ``run_shell`` 放进限制权限的容器）。因此默认关闭才是主防线。
"""

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liteagent.config import NO_TIMEOUT, parse_bool, truncate_head_tail
from liteagent.errors import (
    ConfigError,
    SandboxViolationError,
    ToolTimeoutError,
    ToolValidationError,
)
from liteagent.tools.base import Tool, make_function_tool
from liteagent.types import ToolResult

if TYPE_CHECKING:  # pragma: no cover - 只为注解
    from liteagent.tools.builtin.files import PathSandbox

__all__ = ["SHELL_DENY_PATTERNS", "check_command_allowed", "make_shell_tools"]

# 冻结列表（§7.5）：大小写不敏感，匹配即拒绝。
# 每一条都对应一类"一句话毁掉机器/仓库"的命令，注意最后几条是管道注入
# （`curl ... | sh` 是"从网上拉一段脚本直接跑"，比任何单条破坏命令都危险）。
SHELL_DENY_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[a-z]*r[a-z]*f?\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",
    r"\bsudo\b",
    r"\bchmod\s+777\s+/",
    r">\s*/dev/sd",
    r"\bcurl\b[^|]*\|\s*(ba)?sh",
    r"\bwget\b[^|]*\|\s*(ba)?sh",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\s+-9\s+1\b",
)

# 冻结文案（§7.5）：未开启时 run_shell 的返回值**逐字**是它。
_SHELL_DISABLED_MESSAGE = "shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)"

# `check_command_allowed` 用到的编译后正则。模块级编译（纯数据，无 loop 依赖），
# 不在每次调用里重复 re.compile。
_DENY_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in SHELL_DENY_PATTERNS
)


def check_command_allowed(command: str) -> str | None:
    """纯函数：返回拒绝原因，``None`` 表示允许。

    - 空/纯空白命令 -> ``"empty command"``（一个空命令只会让 ``subprocess`` 白跑一次，
      早退比"执行了但什么也没发生"更容易排查）。
    - 否则逐条用 ``SHELL_DENY_PATTERNS`` 匹配（``re.search`` + ``re.IGNORECASE``）。
    """
    if not command or not command.strip():
        return "empty command"
    for pattern, regex in zip(SHELL_DENY_PATTERNS, _DENY_REGEXES):
        if regex.search(command):
            return f"matches denied pattern {pattern}"
    return None


def _execute_shell(
    _sandbox: "PathSandbox | None",
    command: str,
    cwd: str | None = None,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
    max_output_chars: int = 10000,
) -> tuple[str, int]:
    """真正跑命令，返回 ``(渲染后的文本, exit_code)``。

    拆出元组返回值只有一个目的：``metadata["exit_code"]`` 的写入者被 §2.4 冻结为
    ``builtin/shell``，而工具层唯一能带回 metadata 的通道是 ``ToolResult``。
    ``_run_shell``（冻结签名，返回 ``str``）复用本函数并丢掉 exit_code，
    这样"纯实现的返回值"与"工具结果的 metadata"不会出现两份各自解析输出的实现。
    """
    reason = check_command_allowed(command)
    if reason is not None:
        # 冻结形态（§3.3）：denylist 命中时 path=command、root="denylist:<reason>"。
        raise SandboxViolationError(command, "denylist:" + reason)

    workdir = os.getcwd()
    if _sandbox is not None:
        # 冻结行为（§7.5）：cwd 只做**越界校验**，实际执行目录固定为沙箱根。
        # 这样 `cd /` 之类的幻觉参数既不会逃出沙箱，也不会让相对路径操作落到别处。
        if cwd is not None:
            _sandbox.resolve(cwd, write=True)
        workdir = str(_sandbox.root)
    elif cwd is not None:
        # 没有沙箱可依据时，按调用方给的绝对路径执行（并且要把这件事说清楚：
        # 无沙箱的 run_shell 本身就是"任意目录任意命令"，调用方必须显式传 sandbox）。
        workdir = str(Path(cwd).expanduser().resolve())

    run_env: dict[str, str] | None = None
    if env:
        # 在 os.environ 的**副本**上叠加：只写 env 会连 PATH 一起丢掉，
        # 变成一个"所有命令都 command not found"的迷惑现场。
        run_env = {**os.environ, **{str(k): str(v) for k, v in env.items()}}

    effective_timeout: float | None
    if timeout_s is None or float(timeout_s) <= 0:
        # NO_TIMEOUT(-1.0) / 0 / None 都表示不限时；直接透传给 subprocess 会抛
        # ValueError: timeout value out of range。
        effective_timeout = None
    else:
        effective_timeout = float(timeout_s)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            env=run_env,
            timeout=effective_timeout,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(
            tool_name="run_shell",
            timeout_s=float(timeout_s or 0.0),
            cause=exc,
        ) from exc

    code = int(proc.returncode)
    sections = [f"exit_code: {code}"]
    if proc.stdout:
        sections.append("stdout:\n" + proc.stdout.rstrip("\n"))
    if proc.stderr:
        sections.append("stderr:\n" + proc.stderr.rstrip("\n"))
    text = "\n".join(sections)
    if max_output_chars is not None and max_output_chars > 0:
        text = truncate_head_tail(text, int(max_output_chars))
    return text, code


def _run_shell(
    _sandbox: "PathSandbox | None",
    command: str,
    cwd: str | None = None,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
    max_output_chars: int = 10000,
) -> str:
    """Run a shell command. Returns exit code, stdout and stderr.

    [v2 冻结] 行为见模块 docstring 与 ``_execute_shell``：
    拒绝列表 -> ``SandboxViolationError``；超时 -> ``ToolTimeoutError``；输出截断走
    ``truncate_head_tail``（头尾都保留，报错在尾部、命令回显在头部，砍掉哪头都会丢关键信息）。

    元数据：``idempotent=False, dangerous=True, requires_approval=True,
    timeout_s=NO_TIMEOUT``（由 ``timeout_s`` 参数控制）、``tags=("shell",)``。
    """
    return _execute_shell(_sandbox, command, cwd, timeout_s, env, max_output_chars)[0]


_RUN_SHELL_OPTIONAL = ("cwd", "timeout_s", "env", "max_output_chars")


def _call_args(args: dict[str, Any], *, tool_name: str, required: Sequence[str],
               optional: Sequence[str]) -> dict[str, Any]:
    """整理成可 ``**`` 到纯实现的 kwargs（与 ``files._call_args`` 同一套约定）。

    每个模块各留一份而不是共享：``tools/builtin/`` 内部的同层 import 只允许 §1.1 的
    **E8**（``shell/code -> files``，因为 ``PathSandbox`` 在那里）。为一个 12 行的参数
    整理函数新增一条同层依赖边，会让守门测试的"允许边清单"变成一句空话。
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


_RUN_SHELL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command line to execute."},
        "cwd": {
            "type": "string",
            "description": "Validated against the sandbox; execution stays at the sandbox root.",
        },
        "timeout_s": {"type": "number", "description": "Kill the command after this many seconds."},
        "env": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Extra environment variables (merged over os.environ).",
        },
        "max_output_chars": {"type": "integer", "description": "Truncate the output to this size."},
    },
    "required": ["command"],
    "additionalProperties": False,
}


def make_shell_tools(
    allow_shell: bool | None = None,
    sandbox: "PathSandbox | None" = None,
) -> list[Tool]:
    """返回 ``[run_shell]``。

    ``allow_shell`` 默认取 ``parse_bool(os.environ.get("LITEAGENT_ALLOW_SHELL"))``
    （``"1"/"true"/"yes"/"on"`` 才算开启）。**未开启时工具依然返回**，调用即返回
    失败结果 ``'shell execution is disabled (set LITEAGENT_ALLOW_SHELL=1)'`` ——
    保持工具可见，让模型知道能力存在但被禁用，而不是去猜"为什么没有 shell 工具"。

    [v2 冻结] ``allow_shell=False`` 时**不进入** ``check_command_allowed``：
    被禁用的工具不该有机会产生"拒绝列表命中"的异常，否则调用方会以为自己在处理
    一个命令安全问题，而真相是"这个能力根本没开"。
    """
    from liteagent.tools.builtin.files import PathSandbox

    effective_allow = (
        parse_bool(os.environ.get("LITEAGENT_ALLOW_SHELL"))
        if allow_shell is None
        else bool(allow_shell)
    )
    if sandbox is not None and not isinstance(sandbox, PathSandbox):
            raise ConfigError(
                "make_shell_tools(sandbox=...) requires a PathSandbox instance or None, got "
                f"{type(sandbox).__name__}"
            )

    def run_shell(args: dict[str, Any]) -> ToolResult:
        if not effective_allow:
            # 禁用形态下直接返回冻结文案：**不进** denylist、不碰 subprocess。
            return ToolResult(
                call_id="",
                name="run_shell",
                content=_SHELL_DISABLED_MESSAGE,
                ok=False,
                error=_SHELL_DISABLED_MESSAGE,
                error_type="ToolExecutionError",
            )
        cleaned = _call_args(
            args,
            tool_name="run_shell",
            required=("command",),
            optional=_RUN_SHELL_OPTIONAL,
        )
        text, exit_code = _execute_shell(sandbox, **cleaned)
        return ToolResult(
            call_id="",
            name="run_shell",
            content=text,
            metadata={"exit_code": exit_code},
        )

    return [
        make_function_tool(
            name="run_shell",
            description=_run_shell.__doc__ or "Run a shell command.",
            parameters=_RUN_SHELL_PARAMETERS,
            func=run_shell,
            tags=("shell",),
            dangerous=True,
            requires_approval=True,
            idempotent=False,
            timeout_s=NO_TIMEOUT,  # 超时由参数控制，executor 侧不再抢答
        )
    ]
