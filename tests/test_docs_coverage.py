from __future__ import annotations

# tests/test_docs_coverage.py —— §12 冻结清单里的「[v2 新增，轻量版]」文档覆盖测试。
#
# 冻结要求（docs/INTERFACES.md §12 表最后一行 + §12.2）逐字：
#   「读 docs/VERIFICATION.md，断言每行 evidence 列里提到的 tests/test_*.py 文件
#     真实存在（**不做方法级断言**，避免改文档就挂测试）。」
#
# 为什么刻意做得这么轻：
#   * 这份测试的职责是抓「台账里点名了一份并不存在的测试文件」这种**证据造假/漂移**，
#     不是复读文档里写的用例数。用例数（`（66 例）`）会随重构变化，写进断言就等于
#     "改一次测试就得改文档、改一次文档就挂测试"，与 §12 的原话直接冲突。
#   * 因此 **只断言"文件在磁盘上存在"**，不 import 被测文件、不数 test_* 方法、
#     不读文件内容。跑得飞快，且与其余 36 份测试互不耦合。
#
# 解析口径（本文件唯一的实现细节，抽成纯函数 `evidence_files()` 便于自证）：
#   * 只认 markdown 表格行，且**只认表头里真的有 `evidence` 那一列的表格** ——
#     §12.2 冻结的主表结构是 `claim | evidence | status`；
#   * 表格之外的散文（§3 的 U-1..U-10、§4 的 G-1..G-3）与代码块（§5 的复现命令）
#     里出现的测试路径**不计入**，因为那些地方写的是"待办/缺口/怎么跑"，不是证据；
#   * §4 那张表的表头是 `实测证据`（中文），与冻结列名 `evidence` 不同名，天然被排除；
#   * §2.2 的支撑表与 §2.1 的主表列名一致，一并覆盖。
#
# 判定「表头」的判据：某表格行**紧跟着一行分隔行**（`|---|---:|`）。这条判据让
# "表头" 与 "普通行" 在不依赖列数的情况下可区分，避免把 §2 的任意一行当成表头。

import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFICATION_MD = REPO_ROOT / "docs" / "VERIFICATION.md"

#: 一行 markdown 表格行：`| a | b | c |`（首尾各一竖线）
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
#: 表头与内容之间的分隔行：`|---|---:|---|`（支持两端对齐冒号）
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{1,}:?\s*\|)+\s*$")
#: §4 缺口表里的裸测试模块名（`test_e2e_code_assistant` 这种，没有 `tests/` 前缀）
_MISSING_FILE_RE = re.compile(r"(?<![/\w.])(test_[A-Za-z0-9_]+)(?!\.py)")
#: evidence 列里点名的测试文件（**只认 `tests/test_*.py` 这一种形态**）
_TEST_FILE_RE = re.compile(r"tests/test_[A-Za-z0-9_]+\.py")

#: §12.2 冻结的 evidence 列表头名（小写，逐字）
EVIDENCE_HEADER = "evidence"

#: §12.2：「每个简历原子能力一行（至少 9 行）」。这个下限既是规范要求，也是本文件
#: 解析器的**防空转哨兵**：如果哪天 markdown 结构变了导致一列都没解析出来，
#: 断言会在这里响，而不是让下面那条"文件都存在"的断言空集通过。
MIN_EVIDENCE_ROWS = 9


def _split_row(line: str) -> list[str]:
    """把一行 markdown 表格切成单元格（去掉首尾竖线与每格两侧空白）。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def evidence_files(text: str) -> list[str]:
    """返回所有 `claim | evidence | status` 表格 evidence 列里出现的 `tests/test_*.py`。

    只收集「表头含 `evidence` 列」的表格；表格外与其它列里的路径一律不收集。
    返回值按**文档出现顺序**去重（保持可读的失败信息）。
    """
    found: list[str] = []
    header: list[str] | None = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _TABLE_ROW_RE.match(line):
            header = None  # 离开表格，忘掉当前表头
            index += 1
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if _TABLE_SEPARATOR_RE.match(following):
            header = _split_row(line)  # 这一行是表头
            index += 2  # 连同分隔行一起跳过
            continue
        if header is not None and EVIDENCE_HEADER in header:
            column = header.index(EVIDENCE_HEADER)
            cells = _split_row(line)
            if column < len(cells):
                for match in _TEST_FILE_RE.finditer(cells[column]):
                    if match.group(0) not in found:
                        found.append(match.group(0))
        index += 1
    return found


class EvidenceColumnParserTests(unittest.TestCase):
    """解析器的自证用例：合成的 markdown，判定既不过松也不过严（§12.2 口径）。"""

    #: 合成台账：一张证据表 + 一段散文 + 一张别名列的表 + 一个代码块，各含同一路径
    SAMPLE = "\n".join([
        "# 标题",
        "",
        "## 2. 主表",
        "",
        "| claim | evidence | status |",
        "|---|---|---|",
        "| 能力甲 | `tests/test_alpha.py`（3 例）+ `tests/test_beta.py` | offline-verified |",
        "| 能力乙 | `command: python3 -m liteagent version` | offline-verified |",
        "| 能力丙 | `tests/test_alpha.py`（重复出现，应去重） | offline-verified |",
        "",
        "散文里提到 tests/test_prose_only.py 不算证据（这一行不是表格行）。",
        "",
        "| # | 缺口 | 实测证据 | 影响 |",
        "|---|---|---|---|",
        "| G-9 | 空 | `tests/test_column_mismatch.py` | 无 |",
        "",
        "```bash",
        "python3 -m unittest tests.test_code_block_only",
        "```",
    ])

    def test_collects_paths_from_evidence_column_only(self) -> None:
        self.assertEqual(
            ["tests/test_alpha.py", "tests/test_beta.py"], evidence_files(self.SAMPLE)
        )

    def test_ignores_prose_and_code_blocks(self) -> None:
        collected = evidence_files(self.SAMPLE)
        self.assertNotIn("tests/test_prose_only.py", collected)
        self.assertNotIn("tests/test_code_block_only.py", collected)

    def test_ignores_tables_whose_header_is_not_evidence(self) -> None:
        """表头是 `实测证据`（中文别名）的表不算 evidence 列 —— §12.2 逐字冻结列名。"""
        self.assertNotIn("tests/test_column_mismatch.py", evidence_files(self.SAMPLE))

    def test_returns_empty_for_text_without_any_table(self) -> None:
        self.assertEqual([], evidence_files("没有任何表格的文本 tests/test_ghost.py。"))

    def test_header_row_itself_is_not_mined(self) -> None:
        """表头行里若写了路径也不该被当成证据行（表头不是数据行）。"""
        sample = "\n".join([
            "| claim | evidence | status |",
            "|---|---|---|",
            "| 甲 | 无 | offline-verified |",
        ])
        self.assertEqual([], evidence_files(sample))


class VerificationEvidenceTests(unittest.TestCase):
    """真实台账：`docs/VERIFICATION.md` 的 evidence 列点名的测试文件必须真实存在。"""

    def _document(self) -> str:
        self.assertTrue(
            VERIFICATION_MD.is_file(), f"{VERIFICATION_MD} 必须存在（§12.2 的落点）"
        )
        return VERIFICATION_MD.read_text(encoding="utf-8")

    def test_verification_document_exists(self) -> None:
        self.assertTrue(VERIFICATION_MD.is_file(), "docs/VERIFICATION.md 缺失")
        self.assertGreater(VERIFICATION_MD.stat().st_size, 0)

    def test_document_has_a_frozen_evidence_column(self) -> None:
        """§12.2 冻结三列表 `claim | evidence | status`，必须真的有一列叫 `evidence`。"""
        text = self._document()
        self.assertIn("| claim | evidence | status |", text)

    def test_document_has_at_least_nine_evidence_rows(self) -> None:
        files = evidence_files(self._document())
        self.assertGreaterEqual(
            len(files),
            MIN_EVIDENCE_ROWS,
            f"§12.2 要求至少 {MIN_EVIDENCE_ROWS} 行原子能力；解析到的 file 太少 "
            f"（{len(files)}），要么文档结构变了、要么解析器空转了：{files}",
        )

    def test_every_evidence_test_file_really_exists(self) -> None:
        """本文件的核心断言：evidence 列点名过的每个 `tests/test_*.py` 都必须落地。"""
        missing = [
            relative
            for relative in evidence_files(self._document())
            if not (REPO_ROOT / relative).is_file()
        ]
        self.assertEqual(
            [],
            missing,
            "docs/VERIFICATION.md 的 evidence 列点名了并不存在的测试文件"
            "（§12.2：evidence 必须写成 `tests/test_x.py`，点名的文件就得在磁盘上）",
        )

    def test_declared_missing_test_files_really_are_missing(self) -> None:
        """[v3 补测] §4 声明为「尚未落地」的测试文件必须**真的**不在磁盘上。

        这是上面那条核心断言的反方向，也是**台账自洽性**的最低要求：v2 把 §12 冻结的
        4 份测试文件写成「尚未落地」（并据此把主套件说成 `FAILED (failures=1)`），
        而它们当时已经全部落地 —— 一条 `python3 -m unittest discover` 就能看穿。
        这条断言依然不做**方法级**断言（§12 原话禁止的是数 `（66 例）` 那种用例数），
        只管"文件到底在不在"，因此不会"改一次测试就挂文档"。
        """
        text = self._document()
        declared_missing: list[str] = []
        for line in text.splitlines():
            if "尚未落地" not in line and "未落地" not in line:
                continue
            for name in _MISSING_FILE_RE.findall(line):
                declared_missing.append(f"tests/{name}.py")
        actually_present = [
            relative for relative in declared_missing if (REPO_ROOT / relative).is_file()
        ]
        self.assertEqual(
            [],
            actually_present,
            "docs/VERIFICATION.md 把已落地的测试文件仍声明为「尚未落地」"
            f"（台账漂移）：{sorted(set(actually_present))}",
        )

    def test_evidence_paths_are_normalized_and_inside_tests_dir(self) -> None:
        """防御性：解析出来的路径统一是 `tests/xxx.py`，且去掉反引号后仍在仓库内。"""
        for relative in evidence_files(self._document()):
            self.assertTrue(relative.startswith("tests/test_"), relative)
            self.assertTrue(relative.endswith(".py"), relative)
            self.assertNotIn("`", relative)


if __name__ == "__main__":  # pragma: no cover - 允许直接跑本文件
    unittest.main()
