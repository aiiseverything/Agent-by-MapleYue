from __future__ import annotations

"""内置文件工具（规范 §7.5 的 ``files.py``）。

**安全模型（面试要点，一句话版）**：所有路径都必须先过 :class:`PathSandbox`，它在
``resolve()`` 里把 **`..` / 绝对路径 / symlink** 三种逃逸形态收敛到**同一条检查路径**
上（``Path.resolve()`` 解出真实路径 + ``is_relative_to(root)`` 前缀检查）。
三种写法共用一段逻辑，就不会出现"某一种忘了拦"的漏网之鱼 —— 分开写三处判定是最
常见的安全 bug 来源。

另外两条同样重要的冻结决定：

1. ``make_file_tools(sandbox)`` 的 ``sandbox`` **必填**，``None`` -> ``ConfigError``。
   理由不是洁癖：隐式落到 ``os.getcwd()`` 时，在仓库根跑一次测试里的
   ``delete_file("liteagent/x.py")`` 就会**真的删掉项目文件**。
2. 模块级函数（``_read_file`` 等）是**纯实现**，只接受一个 ``_sandbox`` 注入参数；
   模型可见的 ``Tool`` 由 ``make_file_tools`` 用闭包包装生成，隐藏参数
   （``_sandbox``）**绝不出现在 ``parameters`` 里**（§7.5 的冻结总规则）。
"""

import fnmatch
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from liteagent.config import truncate_head_tail
from liteagent.errors import ConfigError, SandboxViolationError, ToolValidationError
from liteagent.tools.base import Tool, make_function_tool
from liteagent.types import ToolResult

__all__ = ["PathSandbox", "make_file_tools"]

# 判二进制的方式：只看前 8KB 有没有 NUL（与 §7.5 的冻结描述一致）。
# 用"嗅探"而不是 `mimetypes`/扩展名：模型写的临时文件常常没有后缀。
_BINARY_SNIFF_BYTES = 8192
# 超过 1MB 的文件不参与 `search_files`（正则扫描是 CPU 大户，且命中率极低）。
_MAX_SEARCH_FILE_BYTES = 1024 * 1024
# 单行命中文本的展示上限：某行是压缩后的 JS/JSON 时，整行回灌会吃掉整个上下文窗口。
_MAX_LINE_CHARS = 240


class PathSandbox:
    """把相对路径解析到 ``root`` 下，并拒绝逃逸。

    ``root`` 为 ``None`` 时**抛 ``ConfigError``**（不允许隐式 cwd，见模块 docstring）。
    ``allow_read_outside=True`` 只放宽**读**：写路径（``write=True``）永远被限制在
    ``root`` 内 —— 否则任何一个"只读"开关都会变成"写任意文件"的后门。
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        allow_read_outside: bool = False,
    ) -> None:
        if root is None:
            # 这里必须"响亮地失败"：静默 cwd 会让 delete_file 删掉调用方的项目文件。
            raise ConfigError(
                "PathSandbox requires an explicit root; "
                "defaulting to os.getcwd() would let delete_file remove real project files"
            )
        self.root: Path = Path(str(root)).resolve()
        self.allow_read_outside: bool = bool(allow_read_outside)

    # ------------------------------------------------------------------ 路径解析

    def resolve(self, path: str, *, write: bool = False) -> Path:
        """把 ``path`` 解析为绝对路径；越出 ``root`` 时抛 ``SandboxViolationError``。

        - 绝对路径：直接 resolve；相对路径：``root / path`` 后 resolve。
        - **符号链接**：``Path.resolve()`` 会一直解到最终真实路径，因此
          "root 里放一个指向 /etc 的软链"同样被前缀检查拦住。
        - ``..``：不特判 —— 它就是路径的一部分，resolve 之后自然落到 root 外，
          于是仍走同一条前缀检查。
        - ``~`` **不做 expanduser**：让 "~" 变成 root 下一个名叫 "~" 的普通目录，
          "把 home 目录当相对路径拼进来"这条最隐蔽的逃逸路径直接消失。
        """
        raw = Path(str(path))
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        if not _is_within(resolved, self.root):
            if self.allow_read_outside and not write:
                return resolved
            raise SandboxViolationError(
                str(path),
                str(self.root),
                message=(
                    f"path {str(path)!r} escapes sandbox root {str(self.root)!r}; "
                    "使用相对于沙箱根的路径（不允许 .. / 绝对路径 / 指向外部的 symlink）"
                ),
            )
        return resolved

    def display(self, path: Path) -> str:
        """把已解析的路径渲染成"沙箱内相对路径"，越界（allow_read_outside）时用绝对路径。

        只是展示用：回灌给模型的路径越短越省 token，且相对路径天然引导它继续用相对路径。
        """
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"PathSandbox(root={str(self.root)!r}, allow_read_outside={self.allow_read_outside})"


def _is_within(child: Path, root: Path) -> bool:
    """``child`` 是否等于 ``root`` 或在其下（3.9+ 的 ``is_relative_to``）。

    单独抽函数是为了让"检查"只有一处实现 —— 测试里对 ``..`` / 绝对路径 / symlink
    三条路径的断言，本质上断言的都是这一个布尔值。
    """
    try:
        return child.is_relative_to(root)
    except ValueError:  # pragma: no cover - 只在不同盘符（Windows）下可能发生
        return False


# --------------------------------------------------------------------------------------
# 模块级私有实现（纯函数，只接受 `_sandbox` 注入；返回 str，便于直接单测）
# --------------------------------------------------------------------------------------


def _read_file(
    _sandbox: PathSandbox,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int = 20000,
) -> str:
    """Read a UTF-8 text file, optionally a line range.

    ``start_line`` / ``end_line`` 都是 **1 基、闭区间**；越界的行号被裁剪而不是报错
    （模型经常按"我猜文件有 500 行"来传参，为此失败一次不值得）。
    """
    target = _sandbox.resolve(path)
    if not target.exists():
        return f"ERROR: file not found: {_sandbox.display(target)}"
    if target.is_dir():
        return f"ERROR: {_sandbox.display(target)} is a directory; use list_dir instead"
    try:
        # errors="replace"：一个坏字节不该让整次读取失败（模型要的是内容，不是编码正确性）。
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: cannot read {_sandbox.display(target)}: {exc}"
    if start_line is not None or end_line is not None:
        lines = text.splitlines()
        first = max(1, int(start_line)) if start_line is not None else 1
        last = len(lines) if end_line is None else min(len(lines), int(end_line))
        text = "\n".join(lines[first - 1:last]) if last >= first else ""
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = truncate_head_tail(text, max_chars)
    return text


def _write_file(
    _sandbox: PathSandbox,
    path: str,
    content: str,
    create_dirs: bool = True,
    overwrite: bool = True,
) -> str:
    """Write text to a file. Returns a one-line summary.

    ``write=True``：即使沙箱开了 ``allow_read_outside``，写也永远不许越界。
    """
    target = _sandbox.resolve(path, write=True)
    if target.is_dir():
        return f"ERROR: {_sandbox.display(target)} is a directory"
    if target.exists() and not overwrite:
        return (
            f"ERROR: file exists and overwrite=False: {_sandbox.display(target)}; "
            "pass overwrite=True to replace it"
        )
    parent = target.parent
    if not parent.exists():
        if not create_dirs:
            return (
                f"ERROR: parent directory does not exist: {_sandbox.display(parent)}; "
                "pass create_dirs=True to create it"
            )
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"ERROR: cannot create directory {_sandbox.display(parent)}: {exc}"
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: cannot write {_sandbox.display(target)}: {exc}"
    return f"Wrote {len(content)} chars to {_sandbox.display(target)}"


def _list_dir(
    _sandbox: PathSandbox,
    path: str = ".",
    pattern: str | None = None,
    recursive: bool = False,
    max_entries: int = 500,
) -> str:
    """List files under a directory.

    目录后面带 ``/``：模型据此可以区分"该用 read_file 还是 list_dir"，
    而不必先失败一次再纠正。
    """
    target = _sandbox.resolve(path)
    if not target.exists():
        return f"ERROR: directory not found: {_sandbox.display(target)}"
    if not target.is_dir():
        return f"ERROR: {_sandbox.display(target)} is not a directory"
    entries: list[Path] = []
    try:
        iterator = target.rglob("*") if recursive else target.iterdir()
        for entry in iterator:
            name = entry.name
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            entries.append(entry)
    except OSError as exc:
        return f"ERROR: cannot list {_sandbox.display(target)}: {exc}"
    # 排序后再截断：否则 max_entries 截到哪一批取决于文件系统返回顺序（不可复现）。
    entries.sort(key=lambda p: str(p))
    lines: list[str] = []
    for entry in entries[: max(0, max_entries)]:
        try:
            is_dir = entry.is_dir()
        except OSError:  # pragma: no cover - 断链 symlink 等
            is_dir = False
        suffix = "/" if is_dir else ""
        lines.append(f"{_sandbox.display(entry)}{suffix}")
    if len(entries) > max_entries:
        lines.append(f"... ({len(entries) - max_entries} more entries omitted)")
    if not lines:
        return "(empty)"
    return "\n".join(lines)


def _search_files(
    _sandbox: PathSandbox,
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    max_results: int = 50,
    case_sensitive: bool = False,
) -> str:
    """Regex-search file contents; returns 'path:line: text' lines.

    ``glob`` 只支持 ``**/*``（递归）与 ``*``（仅顶层）两种形态 —— 不做完整 glob 引擎，
    因为"完整的 glob"在模型手里 90% 的用法就是这两种，多出来的语义（``?`` / ``[]`` /
    多段 ``**``）只会带来"为什么我的模式没匹配上"的排查成本。
    """
    base = _sandbox.resolve(path)
    if not base.is_dir():
        return f"ERROR: not a directory: {_sandbox.display(base)}"
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"ERROR: invalid regex {pattern!r}: {exc}"
    recursive = glob.strip() == "**/*"
    if glob.strip() not in ("**/*", "*"):
        # 兜底但不静默：按"递归"处理并在返回文本里说明（红线 12）。
        return (
            f"ERROR: unsupported glob {glob!r}; only '**/*' (recursive) and '*' (top level) "
            "are supported"
        )
    candidates = sorted(
        (p for p in (base.rglob("*") if recursive else base.iterdir()) if p.is_file()),
        key=lambda p: str(p),
    )
    lines: list[str] = []
    searched = 0
    skipped_binary = 0
    skipped_big = 0
    for candidate in candidates:
        if len(lines) >= max_results:
            break
        try:
            size = candidate.stat().st_size
        except OSError:  # pragma: no cover - 竞态删除
            continue
        if size > _MAX_SEARCH_FILE_BYTES:
            skipped_big += 1
            continue
        try:
            raw = candidate.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
            skipped_binary += 1
            continue
        searched += 1
        text = raw.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                snippet = line.strip()[:_MAX_LINE_CHARS]
                lines.append(f"{_sandbox.display(candidate)}:{lineno}: {snippet}")
                if len(lines) >= max_results:
                    break
    if not lines:
        return f"no matches for {pattern!r} in {_sandbox.display(base)} (searched {searched} files)"
    footer = [f"[searched {searched} files]"]
    if skipped_binary or skipped_big:
        footer.append(f"[skipped {skipped_binary} binary, {skipped_big} oversized (>1MB)]")
    if len(lines) >= max_results:
        footer.append(f"[stopped at max_results={max_results}]")
    return "\n".join(lines + footer)


def _delete_file(_sandbox: PathSandbox, path: str, confirm: bool = False) -> str:
    """Delete a file. Requires confirm=True.

    ``confirm`` **没有默认 True**：删除是不可逆的，而工具参数是模型生成的；
    让"删"必须显式多写一个参数，等于给模型一次"我真的要删吗"的自检，
    也让审批层（CLI 的 ``--approve``）有一个稳定的判定点。
    """
    target = _sandbox.resolve(path, write=True)
    if not confirm:
        return (
            f"ERROR: delete_file requires confirm=True; refusing to delete "
            f"{_sandbox.display(target)} (this is irreversible)"
        )
    if not target.exists():
        return f"ERROR: file not found: {_sandbox.display(target)}"
    if target.is_dir():
        return (
            f"ERROR: {_sandbox.display(target)} is a directory; only single files can be "
            "deleted (no recursive delete by design)"
        )
    try:
        target.unlink()
    except OSError as exc:
        return f"ERROR: cannot delete {_sandbox.display(target)}: {exc}"
    return f"Deleted {_sandbox.display(target)}"


# --------------------------------------------------------------------------------------
# 模型可见的 schema（手写，与模块级函数签名逐字一致，`_sandbox` 不出现）
# --------------------------------------------------------------------------------------

_PATH_DESC = "Path relative to the sandbox root (absolute paths are accepted only inside it)."

_READ_FILE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": _PATH_DESC},
        "start_line": {"type": "integer", "description": "1-based first line to include."},
        "end_line": {"type": "integer", "description": "1-based last line to include."},
        "max_chars": {"type": "integer", "description": "Truncate the result to this many chars."},
    },
    "required": ["path"],
    "additionalProperties": False,
}

_WRITE_FILE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": _PATH_DESC},
        "content": {"type": "string", "description": "Full UTF-8 content to write."},
        "create_dirs": {"type": "boolean", "description": "Create missing parent directories."},
        "overwrite": {"type": "boolean", "description": "Replace the file if it already exists."},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

_LIST_DIR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list."},
        "pattern": {"type": "string", "description": "fnmatch pattern applied to entry names."},
        "recursive": {"type": "boolean", "description": "Walk sub-directories as well."},
        "max_entries": {"type": "integer", "description": "Maximum number of entries to print."},
    },
    "required": [],
    "additionalProperties": False,
}

_SEARCH_FILES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Python regular expression to search for."},
        "path": {"type": "string", "description": "Directory to search in."},
        "glob": {
            "type": "string",
            "enum": ["**/*", "*"],
            "description": "File selection: '**/*' = recursive, '*' = top level only.",
        },
        "max_results": {"type": "integer", "description": "Maximum number of matching lines."},
        "case_sensitive": {"type": "boolean", "description": "Case-sensitive regex matching."},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}

_DELETE_FILE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": _PATH_DESC},
        "confirm": {
            "type": "boolean",
            "description": "Must be true; deleting is irreversible.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# 闭包包装（§7.5 的冻结总规则）
# --------------------------------------------------------------------------------------


def _tool_result(name: str, text: str, *, sandbox: PathSandbox) -> ToolResult:
    """把纯实现的字符串结果包成 ``ToolResult``，并附上 ``metadata["sandbox"]``。

    为什么必须走 ``ToolResult`` 而不是直接返回 str：``metadata["sandbox"]`` 的
    写入者被 §2.4 冻结为 ``builtin/files``，而工具函数的返回值是**唯一**能带 metadata
    回到 ``ToolResult`` 的通道（executor 的 ``_stringify_extras`` 会合并它的 metadata）。
    返回的 ``call_id``/``name`` 会被 executor 用真实调用覆盖，这里填空串即可。
    """
    return ToolResult(
        call_id="",
        name=name,
        content=text,
        metadata={"sandbox": str(sandbox.root)},
    )


def _call_args(
    args: dict[str, Any],
    *,
    tool_name: str,
    required: Sequence[str],
    optional: Sequence[str],
) -> dict[str, Any]:
    """把工具参数整理成"可以直接 ``**`` 到纯实现"的 kwargs。

    两个约定，都是为了让**绕过 schema 校验**的调用路径（测试直接 ``.raw(...)``、
    或自定义 executor）不至于表现成 ``KeyError`` / ``int(None)``：

    - 必填项缺失 -> ``ToolValidationError``（可恢复类：executor 会把"缺了什么"
      作为 ``feedback_kind="recoverable"`` 回灌给模型）。
    - 可选项为 ``None`` -> **丢掉这个键**，让纯实现用自己的默认值。
      "显式 null" 与 "没给" 在 JSON 语义里同义（§7.1.5 的 D-05 精神），
      而 ``int(args.get("max_chars"))`` 遇到显式 None 会抛 ``TypeError`` ——
      那是一个"框架自己的 bug"伪装成"模型的参数错误"。
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


def make_file_tools(sandbox: PathSandbox) -> list[Tool]:
    """返回 ``[read_file, write_file, list_dir, search_files, delete_file]``。

    **同名新 Tool**，通过闭包注入 ``sandbox``；模块级**不得**定义同名的公开函数
    （`read_file` 等名字在注册表里只有一份，避免"import 到哪个"成为悬案）。
    """
    if not isinstance(sandbox, PathSandbox):
        # 传 None / 字符串 / 随便一个对象都不行：这是"安全默认值"的最后一道闸。
        raise ConfigError(
            "make_file_tools(sandbox) requires a PathSandbox instance, got "
            f"{type(sandbox).__name__}; pass PathSandbox(<root>) explicitly"
        )

    def read_file(args: dict[str, Any]) -> ToolResult:
        text = _read_file(
            sandbox,
            **_call_args(
                args,
                tool_name="read_file",
                required=("path",),
                optional=("start_line", "end_line", "max_chars"),
            ),
        )
        return _tool_result("read_file", text, sandbox=sandbox)

    def write_file(args: dict[str, Any]) -> ToolResult:
        text = _write_file(
            sandbox,
            **_call_args(
                args,
                tool_name="write_file",
                required=("path", "content"),
                optional=("create_dirs", "overwrite"),
            ),
        )
        return _tool_result("write_file", text, sandbox=sandbox)

    def list_dir(args: dict[str, Any]) -> ToolResult:
        text = _list_dir(
            sandbox,
            **_call_args(
                args,
                tool_name="list_dir",
                required=(),
                optional=("path", "pattern", "recursive", "max_entries"),
            ),
        )
        return _tool_result("list_dir", text, sandbox=sandbox)

    def search_files(args: dict[str, Any]) -> ToolResult:
        text = _search_files(
            sandbox,
            **_call_args(
                args,
                tool_name="search_files",
                required=("pattern",),
                optional=("path", "glob", "max_results", "case_sensitive"),
            ),
        )
        return _tool_result("search_files", text, sandbox=sandbox)

    def delete_file(args: dict[str, Any]) -> ToolResult:
        text = _delete_file(
            sandbox,
            **_call_args(
                args,
                tool_name="delete_file",
                required=("path",),
                optional=("confirm",),
            ),
        )
        return _tool_result("delete_file", text, sandbox=sandbox)

    return [
        # 元数据冻结（§7.5）：read_file/list_dir/search_files -> dangerous=False,
        # idempotent=True；write_file -> dangerous=True, idempotent=True；
        # delete_file -> dangerous=True, idempotent=False。tags 全部是 ("fs",)。
        make_function_tool(
            name="read_file",
            description=_read_file.__doc__ or "Read a UTF-8 text file.",
            parameters=_READ_FILE_PARAMETERS,
            func=read_file,
            tags=("fs",),
            dangerous=False,
            idempotent=True,
        ),
        make_function_tool(
            name="write_file",
            description=_write_file.__doc__ or "Write text to a file.",
            parameters=_WRITE_FILE_PARAMETERS,
            func=write_file,
            tags=("fs",),
            dangerous=True,
            idempotent=True,
        ),
        make_function_tool(
            name="list_dir",
            description=_list_dir.__doc__ or "List files under a directory.",
            parameters=_LIST_DIR_PARAMETERS,
            func=list_dir,
            tags=("fs",),
            dangerous=False,
            idempotent=True,
        ),
        make_function_tool(
            name="search_files",
            description=_search_files.__doc__ or "Regex-search file contents.",
            parameters=_SEARCH_FILES_PARAMETERS,
            func=search_files,
            tags=("fs",),
            dangerous=False,
            idempotent=True,
        ),
        make_function_tool(
            name="delete_file",
            description=_delete_file.__doc__ or "Delete a file.",
            parameters=_DELETE_FILE_PARAMETERS,
            func=delete_file,
            tags=("fs",),
            dangerous=True,
            idempotent=False,
        ),
    ]
