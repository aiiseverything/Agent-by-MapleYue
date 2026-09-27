from __future__ import annotations

# =============================================================================
# scripts/check_spec_consistency.py —— 规范与仓库的一致性检查器（§12.4）
# =============================================================================
#
# 它回答一个问题：**`docs/INTERFACES.md` 说的和仓库里真的有对得上吗？**
#
# 检查项（逐条对应 §12.4 的职责描述）：
#
#   1. §1.2 的**文件清单**（41 个 `.py` 的封闭表格）与实际 `liteagent/**/*.py` 是否一致
#      —— 缺文件 / 多文件都要报。顺带核对标题里写死的"共 41 个"。
#   2. §1.2 冻结的 `liteagent/__main__.py` **逐字内容**（6 行）是否一字不差。
#   3. §1.4 各包、§10.5 multiagent、附录 B 顶层 `__init__` 的 `__all__` 清单：
#      规范里列的每个名字是否 (a) 在实现的 `__all__` 里、(b) 能 `getattr` 到。
#      **能解析**是最低要求；顺带把"实现多导出的名字"报出来（信息性）。
#   4. §1.2 里其它封闭清单（根目录交付物 / `docs/` / `examples/` / `benchmarks/` /
#      `scripts/`）的存在性。
#
# 为什么 #4 只报 warn 不算失败：这些清单由**别的 owner**负责（README/TOOLS.md/
# INTERVIEW.md 等可能还没写到），本脚本的存在意义是"发现不一致"，
# 不是"替别人宣布失败"。而 #1/#2/#3 直接对应 liteagent 内核对接口的承诺，算 error。
#
# 这个脚本**不参与测试**（§12.4：可选工具），但它必须真能跑：
#   python3 scripts/check_spec_consistency.py
#   python3 scripts/check_spec_consistency.py --json
#   echo $?     # 0 = 没有 error；1 = 有 error（warn 不影响退出码）
#
# 注意：每个 .py 的第一行必须是 `from __future__ import annotations`（仓库冻结约定），
# 所以本文件用 `#` 注释而不是模块 docstring 来写说明。

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent.config import format_ts, utc_now  # noqa: E402

INTERFACES = _REPO_ROOT / "docs" / "INTERFACES.md"

#: 所有出现在报告里的严重级别。`error` 影响退出码，`warn`/`info` 不影响。
SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"
SEVERITY_INFO = "info"


# ---------------------------------------------------------------------------
# Markdown 解析小工具（只用 stdlib re，不引 markdown 库）
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{2,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"^```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)^```[ \t]*$",
                            re.MULTILINE | re.DOTALL)
_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`", re.MULTILINE)


def section_body(text: str, heading_prefix: str) -> str | None:
    """取某个标题到「下一个同级或更高级标题」之间的正文。

    为什么要卡级别：§1.4（`###`）的正文里如果出现 `####` 子标题，它仍然属于 §1.4；
    但下一个 `## 2.` 就不属于了。按级别比较是 markdown 的结构语义，不是启发式。

    `heading_prefix` 可以带 `#`（调用方写 `"### 1.2 "` 更直观），比较时把 `#` 剥掉 ——
    因为 `_HEADING_RE` 的捕获组只含标题文字、不含井号。
    """
    wanted = heading_prefix.lstrip("#").lstrip()
    for match in _HEADING_RE.finditer(text):
        if not match.group(2).startswith(wanted):
            continue
        level = len(match.group(1))
        start = match.end()
        for later in _HEADING_RE.finditer(text, start):
            if len(later.group(1)) <= level:
                return text[start:later.start()]
        return text[start:]
    return None


def code_blocks(body: str) -> list[tuple[str, str]]:
    """返回正文里所有 ``` 围栏代码块的 `(语言, 内容)`。"""
    return [(m.group(1), m.group(2)) for m in _CODE_BLOCK_RE.finditer(body)]


def _all_from_code(code: str) -> list[str] | None:
    """从一段 python 代码里抠出 `__all__ = [...]` 的字符串名字。没有则返回 None。"""
    match = re.search(r"__all__\s*=\s*\[(.*?)\]", code, re.DOTALL)
    if match is None:
        return None
    return re.findall(r'"([^"]+)"', match.group(1))


def _backticked_after(body: str, anchor: str, suffix: str | None = None) -> list[str]:
    """取 anchor 所在那一句里 `` ` `` 包住的名字（可选按后缀过滤）。找不到 anchor 返回 []。

    **必须跨行累积**：`INTERFACES.md` 是 100 列换行的，`docs/`（7 个）与 `examples/`（8 个）
    这两行清单都被折成了 2~3 行；只看 anchor 那一行会漏掉一半名字（实测：
    docs 只捞到 2/7、examples 只捞到 3/8）。
    累积的终止条件是「这一句写完了」：行尾是中文句号，或下一行是空行，或累积到 10 行。
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if anchor not in line:
            continue
        chunk = line
        cursor = index
        while (
            not chunk.rstrip().endswith("。")
            and cursor + 1 < len(lines)
            and cursor - index < 10
        ):
            nxt = lines[cursor + 1]
            if not nxt.strip():
                break
            cursor += 1
            chunk += nxt
        names = re.findall(r"`([^`]+)`", chunk)
        if suffix is None:
            return names
        return [name for name in names if name.endswith(suffix)]
    return []


def _check(name: str, ok: bool, severity: str, detail: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "check": name,
        "ok": bool(ok),
        "severity": severity if not ok else SEVERITY_INFO,
        "detail": detail,
    }
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# 检查 1：§1.2 的文件清单
# ---------------------------------------------------------------------------
def check_file_manifest(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body = section_body(text, "### 1.2 ")
    checks: list[dict[str, Any]] = []
    if body is None:
        checks.append(_check("§1.2 存在性", False, SEVERITY_ERROR,
                             "找不到 `### 1.2` 小节"))
        return checks, {}

    declared = _TABLE_ROW_RE.findall(body)
    declared_set = set(declared)

    actual = sorted(
        str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
        for path in (_REPO_ROOT / "liteagent").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    actual_set = set(actual)

    missing = sorted(declared_set - actual_set)
    extra = sorted(actual_set - declared_set)

    # 标题里写死的"共 41 个 `.py`"也要核对：数字对不上说明解析器或文档坏了。
    # 注意要在**标题行**上找，不是在正文里 —— 正文第一句就是表格。
    declared_count = None
    for match in _HEADING_RE.finditer(text):
        if not match.group(2).startswith("1.2 "):
            continue
        count_match = re.search(r"共\s*(\d+)\s*个", match.group(2))
        if count_match:
            declared_count = int(count_match.group(1))
        break

    checks.append(_check(
        "§1.2 文件清单：声明的 %d 个文件都在仓库里" % len(declared_set),
        not missing, SEVERITY_ERROR,
        "缺失：%s" % (missing or "无"), missing=missing,
    ))
    checks.append(_check(
        "§1.2 文件清单：仓库里没有清单外的 .py",
        not extra, SEVERITY_ERROR,
        "多于清单：%s" % (extra or "无"), extra=extra,
    ))
    checks.append(_check(
        "§1.2 标题声明的数量与表格行数一致（声明的 %s / 解析到 %d）"
        % (declared_count, len(declared)),
        declared_count == len(declared), SEVERITY_ERROR,
        "标题写 %s，表格解析出 %d 行" % (declared_count, len(declared)),
    ))
    checks.append(_check(
        "§1.2 文件清单里没有重复行",
        len(declared) == len(declared_set), SEVERITY_ERROR,
        "重复项：%s" % (sorted(
            {name for name in declared if declared.count(name) > 1}) or "无"),
    ))

    # 顺带：清单里的每个路径都要能 import（名字 -> 模块），这是"文件真能被加载"
    unimportable: list[str] = []
    for rel in sorted(declared_set):
        if not rel.endswith(".py") or not rel.startswith("liteagent/"):
            continue
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        if module.endswith(".__main__"):
            continue  # `__main__` 只允许以 `python -m liteagent` 形式执行
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - 报告要用异常文本，不该在这里分类
            unimportable.append("%s: %s: %s" % (rel, type(exc).__name__, exc))
    checks.append(_check(
        "§1.2 清单里的模块都能 import",
        not unimportable, SEVERITY_ERROR,
        "失败：%s" % (unimportable or "无"), failures=unimportable,
    ))

    summary = {
        "declared_count": len(declared_set),
        "declared_count_in_heading": declared_count,
        "actual_count": len(actual_set),
        "missing": missing,
        "extra": extra,
    }
    return checks, summary


# ---------------------------------------------------------------------------
# 检查 2：`liteagent/__main__.py` 的冻结逐字内容
# ---------------------------------------------------------------------------
def check_main_module(text: str) -> list[dict[str, Any]]:
    body = section_body(text, "### 1.2 ")
    if body is None:
        return []
    anchor = "冻结逐字内容"
    anchor_index = body.find(anchor)
    if anchor_index < 0:
        return [_check("§1.2 `__main__.py` 冻结内容存在", False, SEVERITY_WARN,
                       "正文里找不到「冻结逐字内容」锚点")]

    blocks = code_blocks(body[anchor_index:])
    if not blocks:
        return [_check("§1.2 `__main__.py` 冻结内容存在", False, SEVERITY_WARN,
                       "锚点之后没有代码块")]
    frozen_lines = [line.rstrip() for line in blocks[0][1].strip("\n").splitlines()]
    while frozen_lines and not frozen_lines[-1]:
        frozen_lines.pop()

    path = _REPO_ROOT / "liteagent" / "__main__.py"
    if not path.exists():
        return [_check("§1.2 `liteagent/__main__.py` 存在", False, SEVERITY_ERROR,
                       "文件不存在")]
    actual_lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    while actual_lines and not actual_lines[-1]:
        actual_lines.pop()

    same = frozen_lines == actual_lines
    detail = "一致（%d 行）" % len(frozen_lines)
    if not same:
        diff = []
        for i in range(max(len(frozen_lines), len(actual_lines))):
            expected = frozen_lines[i] if i < len(frozen_lines) else "<缺失>"
            got = actual_lines[i] if i < len(actual_lines) else "<缺失>"
            if expected != got:
                diff.append("第 %d 行：规范 %r != 实际 %r" % (i + 1, expected, got))
        detail = "不一致：%s" % "；".join(diff)
    return [_check("§1.2 `liteagent/__main__.py` 与冻结内容逐字一致", same,
                   SEVERITY_ERROR, detail,
                   frozen=frozen_lines, actual=actual_lines)]


# ---------------------------------------------------------------------------
# 检查 3：`__all__` 清单
# ---------------------------------------------------------------------------
def _module_for_relative(rel: str) -> str:
    module = rel[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def check_all_lists(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """解析 §1.4 / §10.5 / 附录 B 的 `__all__`，逐块核对。"""
    checks: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []

    sources: list[tuple[str, str | None]] = [
        ("### 1.4 ", None),          # 模块名写在代码块的注释里
        ("### 10.5 ", "liteagent/multiagent/__init__.py"),  # 小节标题就是模块
        ("## 附录 B", "liteagent/__init__.py"),             # 顶层 `__init__`
    ]

    for heading, forced_module in sources:
        body = section_body(text, heading)
        if body is None:
            checks.append(_check("%s 存在性" % heading.strip(), False, SEVERITY_ERROR,
                                 "找不到小节"))
            continue
        found_any = False
        for _lang, code in code_blocks(body):
            names = _all_from_code(code)
            if names is None:
                continue
            module_hint = forced_module
            if module_hint is None:
                match = re.search(r"#\s*(liteagent/[\w/]+\.py)", code)
                module_hint = match.group(1) if match else None
            if module_hint is None:
                checks.append(_check("%s 里的 __all__ 块" % heading.strip(), False,
                                     SEVERITY_WARN,
                                     "解析到 __all__ 但认不出模块归属，已跳过"))
                continue
            found_any = True
            blocks.append({"source": heading.strip(), "relative_path": module_hint,
                           "expected_names": names})
        if not found_any:
            checks.append(_check("%s 里解析到 __all__ 块" % heading.strip(), False,
                                 SEVERITY_ERROR, "没有解析到任何 __all__ 块"))

    for block in blocks:
        rel = block["relative_path"]
        module_name = _module_for_relative(rel)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("`%s` 可 import（%s）" % (module_name, block["source"]),
                                 False, SEVERITY_ERROR,
                                 "%s: %s" % (type(exc).__name__, exc)))
            continue

        expected = list(block["expected_names"])
        actual_all = list(getattr(module, "__all__", []) or [])
        missing_from_all = [n for n in expected if n not in actual_all]
        unresolvable = [n for n in expected if not hasattr(module, n)]
        extra = [n for n in actual_all if n not in expected]

        block["actual_all_count"] = len(actual_all)
        block["missing_from_all"] = missing_from_all
        block["unresolvable"] = unresolvable
        block["extra_in_all"] = extra

        checks.append(_check(
            "`%s` 的 __all__ 覆盖规范列出的 %d 个名字（%s）"
            % (module_name, len(expected), block["source"]),
            not missing_from_all, SEVERITY_ERROR,
            "缺：%s" % (missing_from_all or "无"), missing=missing_from_all,
        ))
        checks.append(_check(
            "`%s` 的 %d 个规范名字全部可 getattr（%s）"
            % (module_name, len(expected), block["source"]),
            not unresolvable, SEVERITY_ERROR,
            "不可解析：%s" % (unresolvable or "无"), unresolvable=unresolvable,
        ))
        # 实现多导出名字不算失败：规范只承诺"列出来的必须能拿到"，
        # 反方向（多导出）是 info，因为 §1.4 的措辞是"未列出的视为内部"而非"禁止存在"。
        checks.append(_check(
            "`%s` 没有 __all__ 之外的多余导出（%s）" % (module_name, block["source"]),
            not extra, SEVERITY_INFO,
            "多导出：%s" % (extra or "无"), extra=extra,
        ))

    return checks, blocks


# ---------------------------------------------------------------------------
# 检查 4：§1.2 里其它封闭清单（只报 warn）
# ---------------------------------------------------------------------------
def check_other_manifests(text: str) -> list[dict[str, Any]]:
    body = section_body(text, "### 1.2 ")
    if body is None:
        return []
    checks: list[dict[str, Any]] = []

    specs: list[tuple[str, str, str, str | None]] = [
        ("根目录交付物", "**根目录交付物（封闭清单）**：", "", None),
        ("docs/ 封闭清单（7 个）", "**`docs/`（封闭清单", "docs", ".md"),
        ("examples/ 封闭清单（8 个）", "**`examples/`（8 个）**：", "examples", ".py"),
        ("benchmarks/ 封闭清单（3 个）", "**`benchmarks/`（3 个，封闭清单）**：",
         "benchmarks", ".py"),
    ]
    for label, anchor, folder, suffix in specs:
        declared = _backticked_after(body, anchor, suffix)
        if not declared:
            checks.append(_check("解析 §1.2 的 %s" % label, False, SEVERITY_WARN,
                                 "锚点 %r 没解析出任何文件名" % anchor))
            continue
        if folder:
            actual = {
                path.name
                for path in (_REPO_ROOT / folder).iterdir()
                if path.is_file() or path.is_dir()
            }
        else:
            actual = {path.name for path in _REPO_ROOT.iterdir()}
        missing = sorted(set(declared) - actual)
        checks.append(_check(
            "%s：声明的 %d 个都在（这一项不影响退出码）" % (label, len(declared)),
            not missing, SEVERITY_WARN,
            "缺：%s" % (missing or "无"),
            declared=sorted(set(declared)), missing=missing,
        ))

    # scripts/ 只允许放这一支脚本（§1.2 原文："不得放与规范无关的文件"）
    scripts_dir = _REPO_ROOT / "scripts"
    if scripts_dir.is_dir():
        found = sorted(
            path.name for path in scripts_dir.iterdir()
            if path.is_file() and path.name != "__pycache__"
        )
        unexpected = [name for name in found if name != "check_spec_consistency.py"]
        checks.append(_check(
            "scripts/ 只放 check_spec_consistency.py",
            not unexpected, SEVERITY_WARN,
            "多余文件：%s" % (unexpected or "无"), found=found,
        ))
    return checks


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    if not INTERFACES.exists():
        checks.append(_check("docs/INTERFACES.md 存在", False, SEVERITY_ERROR, "文件不存在"))
        text = ""
    else:
        text = INTERFACES.read_text(encoding="utf-8")

    checks.append(_check(
        "docs/INTERFACES.md 非空", bool(text.strip()), SEVERITY_ERROR,
        "%d 字节" % len(text.encode("utf-8")),
    ))
    if not text.strip():
        return {
            "script": "scripts/check_spec_consistency.py",
            "generated_at": format_ts(utc_now()) + " UTC",
            "interfaces": str(INTERFACES),
            "summary": {},
            "checks": checks,
            "error_count": sum(1 for c in checks if not c["ok"] and c["severity"] == "error"),
            "warn_count": sum(1 for c in checks if not c["ok"] and c["severity"] == "warn"),
        }

    manifest_checks, manifest_summary = check_file_manifest(text)
    checks.extend(manifest_checks)
    summary["file_manifest"] = manifest_summary

    checks.extend(check_main_module(text))

    all_checks, all_blocks = check_all_lists(text)
    checks.extend(all_checks)
    summary["all_lists"] = all_blocks

    checks.extend(check_other_manifests(text))

    errors = [c for c in checks if not c["ok"] and c["severity"] == SEVERITY_ERROR]
    warns = [c for c in checks if not c["ok"] and c["severity"] == SEVERITY_WARN]
    return {
        "script": "scripts/check_spec_consistency.py",
        "generated_at": format_ts(utc_now()) + " UTC",
        "interfaces": str(INTERFACES),
        "summary": summary,
        "checks": checks,
        "error_count": len(errors),
        "warn_count": len(warns),
        "info_count": sum(1 for c in checks if not c["ok"] and c["severity"] == SEVERITY_INFO),
    }


def render_text(result: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("=" * 88)
    out.append("scripts/check_spec_consistency.py —— 规范 vs 仓库一致性（§12.4，不参与测试）")
    out.append("=" * 88)
    out.append("规范文件：%s" % result["interfaces"])
    out.append("检查于：%s" % result["generated_at"])
    out.append("")

    manifest = result["summary"].get("file_manifest") or {}
    if manifest:
        out.append("[§1.2 文件清单]")
        out.append("  规范表格声明 %d 个 / 标题写 %s / 仓库实际 %d 个"
                   % (manifest["declared_count"],
                      manifest["declared_count_in_heading"], manifest["actual_count"]))
        out.append("  缺失：%s" % (manifest["missing"] or "无"))
        out.append("  多出：%s" % (manifest["extra"] or "无"))
        out.append("")

    blocks = result["summary"].get("all_lists") or []
    if blocks:
        out.append("[__all__ 清单]")
        for block in blocks:
            out.append("  %-14s %-34s 规范 %3d 个 / 实现 %3d 个"
                       % (block["source"], block["relative_path"],
                          len(block["expected_names"]), block.get("actual_all_count", 0)))
            if block.get("missing_from_all"):
                out.append("      缺：%s" % block["missing_from_all"])
            if block.get("unresolvable"):
                out.append("      不可解析：%s" % block["unresolvable"])
            if block.get("extra_in_all"):
                out.append("      （实现多导出，不算失败）：%s" % block["extra_in_all"])
        out.append("")

    out.append("[逐项结果]")
    for check in result["checks"]:
        if check["ok"]:
            mark = "ok  "
        elif check["severity"] == SEVERITY_ERROR:
            mark = "FAIL"
        elif check["severity"] == SEVERITY_WARN:
            mark = "WARN"
        else:
            mark = "info"
        out.append("  [%s] %s" % (mark, check["check"]))
        if not check["ok"] and check["detail"]:
            out.append("         %s" % check["detail"])
    out.append("")
    out.append("[结论] error=%d warn=%d info=%d（只有 error 影响退出码）"
               % (result["error_count"], result["warn_count"], result["info_count"]))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_spec_consistency",
        description="检查 docs/INTERFACES.md 的文件清单与 __all__ 是否与仓库一致",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 打印结果")
    args = parser.parse_args(argv)

    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(result))
    return 1 if result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
