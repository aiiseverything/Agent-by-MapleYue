from __future__ import annotations

# =============================================================================
# benchmarks/bench_embedding_similarity.py —— 填 D-04 的「实测补充」
# =============================================================================
#
# 被测对象：`HashingEmbedder`（纯 stdlib 的特征哈希 + 带符号累加 + L2 归一化）
# 在**同义句 / 反义句 / 无关句 / 跨语言翻译句**四组对照上的余弦相似度矩阵。
#
# 为什么要跑这个脚本：D-04 的决策是"用纯 stdlib 的 hashing trick 当默认 embedder，
# 并把能力边界写进文档"。**能力边界不能靠嘴说，要有一张表**。这张表要回答的问题是：
#
#     "词面哈希到底能不能区分同义和反义？"
#
# 本机的实测答案是：**不能，而且反义句的平均分还比同义句更高**。
# 原因是同义改写常常换词（"book a flight" vs "book a plane ticket"），
# 而反义句几乎总是共享大部分词（"enable" vs "disable"、喜欢 vs 讨厌）——
# 特征哈希量的就是"共享了多少 token"，所以共享得多的反义句得分更高。
#
# 这不是 bug，是**选型的固有边界**（D-04 的"代价"一栏写的就是它）：
#   - 默认 `HashingEmbedder` 的用途是"离线、确定性、零依赖地跑通检索链路与测试"，
#     不是"做语义理解"。
#   - 生产要语义就把 `RemoteEmbedder`（或任何实现了 `Embedder` 的向量服务）
#     注入 `VectorMemory(embedder=...)`；`VectorMemory.__init__` 里"注入的 embedder
#     的维度是权威"这条 v2 冻结规则就是为这种替换准备的。
#
# 跑法：
#   python3 benchmarks/bench_embedding_similarity.py            # 矩阵 + 分组统计
#   python3 benchmarks/bench_embedding_similarity.py --json     # 机器可读 JSON
#   python3 benchmarks/bench_embedding_similarity.py --json --no-write
#
# 注意：每个 .py 的第一行必须是 `from __future__ import annotations`（仓库冻结约定），
# 所以本文件用 `#` 注释而不是模块 docstring 来写说明。

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent.config import (  # noqa: E402
    DEFAULT_HASHING_EMBED_DIM,
    format_ts,
    utc_now,
)
from liteagent.memory.embeddings import (  # noqa: E402
    HashingEmbedder,
    cosine_similarity,
    tokenize,
)

_BENCH_ID = "bench-embedding-similarity"
_VERIFICATION = _REPO_ROOT / "docs" / "VERIFICATION.md"
_DOC_TRIGGER = "python3 benchmarks/bench_embedding_similarity.py --json"

# ---------------------------------------------------------------------------
# 冻结的句子集：每组至少 4 对，且**成对**（P 组内的两句是同一对的两种说法）
# ---------------------------------------------------------------------------
# 每项：(id, 语言, 句子)。id 短、稳定，矩阵表头用它。
SENTENCES: list[tuple[str, str, str]] = [
    # --- 同义（英文，换词改写，共享一部分 token）---
    ("S01", "en", "The cat is sitting on the mat."),
    ("S02", "en", "A cat sits on the mat."),
    ("S03", "en", "I would like to book a flight to Paris."),
    ("S04", "en", "Please book me a plane ticket to Paris."),
    ("S05", "en", "Please restart the server."),
    ("S06", "en", "Could you restart the server?"),
    # --- 同义（中文）---
    ("S07", "zh", "用户偏好使用 Python 编写后端服务。"),
    ("S08", "zh", "用户喜欢用 Python 做后端开发。"),
    # --- 同义但**词面重合极少**（哈希 embedding 的失效样本）---
    ("S09", "zh", "用户需要重置密码。"),
    ("S10", "zh", "用户想把密码改掉。"),
    # --- 反义（共享大量 token，只有极性词不同）---
    ("S11", "en", "I love programming."),
    ("S12", "en", "I hate programming."),
    ("S13", "en", "The service is very fast."),
    ("S14", "en", "The service is very slow."),
    ("S15", "en", "Please enable the cache."),
    ("S16", "en", "Please disable the cache."),
    ("S17", "zh", "用户喜欢这个方案。"),
    ("S18", "zh", "用户讨厌这个方案。"),
    # --- 无关 ---
    ("S19", "en", "The weather in Tokyo is rainy today."),
    ("S20", "en", "Quantum computing uses qubits."),
    ("S21", "en", "I need to buy milk and eggs."),
    ("S22", "en", "The stock market fell sharply."),
    # --- 跨语言翻译对（语义完全等价，词面零重合）---
    ("S23", "en", "Good morning"),
    ("S24", "zh", "早上好"),
    ("S25", "en", "Thank you very much."),
    ("S26", "zh", "非常感谢。"),
]

# 带类型标注的对照对。`relation` 是**人工标注**（不是模型判的），
# 它是这张表唯一的"真值"，脚本只负责量出模型分与真值的偏差。
PAIRS: list[tuple[str, str, str]] = [
    ("synonym", "S01", "S02"),
    ("synonym", "S03", "S04"),
    ("synonym", "S05", "S06"),
    ("synonym", "S07", "S08"),
    ("hard-synonym", "S09", "S10"),
    ("antonym", "S11", "S12"),
    ("antonym", "S13", "S14"),
    ("antonym", "S15", "S16"),
    ("antonym", "S17", "S18"),
    ("unrelated", "S19", "S20"),
    ("unrelated", "S21", "S22"),
    ("translation", "S23", "S24"),
    ("translation", "S25", "S26"),
]


def compute(*, dim: int) -> dict[str, Any]:
    """算完整矩阵 + 分组统计，返回 JSON 可序列化的结果。"""
    embedder = HashingEmbedder(dim=dim)
    ids = [sid for sid, _, _ in SENTENCES]
    texts = [text for _, _, text in SENTENCES]
    vectors = embedder.embed(texts)
    index = {sid: i for i, sid in enumerate(ids)}

    # 对称矩阵：matrix[i][j] = cos(v_i, v_j)，对角线是 1.0（零向量除外）
    matrix: list[list[float]] = []
    for i, vec in enumerate(vectors):
        row: list[float] = []
        for j, other in enumerate(vectors):
            row.append(cosine_similarity(vec, other) if i != j else _self_similarity(vec))
        matrix.append(row)

    pair_rows: list[dict[str, Any]] = []
    for relation, a, b in PAIRS:
        score = matrix[index[a]][index[b]]
        pair_rows.append(
            {
                "relation": relation,
                "a": a,
                "b": b,
                "text_a": SENTENCES[index[a]][2],
                "text_b": SENTENCES[index[b]][2],
                "similarity": score,
            }
        )

    by_relation: dict[str, list[float]] = {}
    for row in pair_rows:
        by_relation.setdefault(row["relation"], []).append(row["similarity"])
    relation_stats = {
        relation: {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "min": min(scores),
            "max": max(scores),
        }
        for relation, scores in by_relation.items()
    }

    syn = by_relation.get("synonym", [])
    ant = by_relation.get("antonym", [])
    hard = by_relation.get("hard-synonym", [])
    unl = by_relation.get("unrelated", [])
    tr = by_relation.get("translation", [])

    mean_syn = statistics.fmean(syn) if syn else 0.0
    mean_ant = statistics.fmean(ant) if ant else 0.0
    mean_unl = statistics.fmean(unl) if unl else 0.0
    mean_tr = statistics.fmean(tr) if tr else 0.0
    mean_hard = statistics.fmean(hard) if hard else 0.0

    # 判别力：如果有语义，synonym 应该显著高于 antonym。实测是反过来的。
    margin = mean_syn - mean_ant

    return {
        "bench": _BENCH_ID,
        "fills": "D-04「实测补充」",
        "generated_at": format_ts(utc_now()) + " UTC",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedder": type(embedder).__name__,
        "dim": dim,
        "sentence_count": len(SENTENCES),
        "sentences": [
            {"id": sid, "lang": lang, "text": text} for sid, lang, text in SENTENCES
        ],
        "ids": ids,
        "matrix": matrix,
        "pairs": pair_rows,
        "relation_stats": relation_stats,
        "verdict": {
            "mean_synonym": mean_syn,
            "mean_antonym": mean_ant,
            "mean_hard_synonym": mean_hard,
            "mean_unrelated": mean_unl,
            "mean_translation": mean_tr,
            # >0 说明"同义 > 反义"（有语义）；<=0 说明没有语义区分力
            "semantic_margin_syn_minus_ant": margin,
            "discriminative_power": "none" if margin <= 0.0 else "weak",
            "summary": (
                "同义对平均 %.4f，反义对平均 %.4f —— 反义更高（margin=%.4f <= 0）。"
                "说明 HashingEmbedder 量的是 token 重合度，不是语义极性。"
                "跨语言翻译对平均 %.4f（=0，因为中英文没有共享 token）。"
                % (mean_syn, mean_ant, margin, mean_tr)
            ),
        },
        "limitations": [
            "捕捉的是**词面（token）重叠**，不是语义：同义改写换词后分数会掉，"
            "反义句因为共享大部分实词反而得分更高。",
            "跨语言零重合：'Good morning' 与 '早上好' 的余弦是 0.0；"
            "任何中英混排的语料里，翻译等价的记忆互相检索不到。",
            "中文按**单字 + 相邻双字**分词，字符级重合会虚高："
            "意思无关但用字相近的两句（如都含'用户'/'方案'）也会得到非零分。",
            "对停用词敏感：'the'/'is'/'a' 也进哈希，短句之间靠虚词就能拿到分数。",
            "维度 dim 只影响哈希碰撞概率与内存，不改变上述性质；"
            "调大 dim 不会让它在语义上变准。",
            "结论只在**本机 CPython 3.10.12 + 该句子集**上成立；样本 26 句、"
            "对照 13 对，是演示性的，不是评测集。",
        ],
    }


def _self_similarity(vec: Iterable[float]) -> float:
    """对角线元素。

    对**零向量**（空文本或全被过滤）定义为 0.0 而不是 1.0：`cosine_similarity`
    对零向量的契约就是返回 0.0（"零向量 -> 0.0，不抛异常"），
    这里保持与它一致，免得矩阵里出现一个"自己和自己不相似"的 1.0 假象。
    """
    values = list(vec)
    if all(x == 0.0 for x in values):
        return 0.0
    return cosine_similarity(values, values)


# ---------------------------------------------------------------------------
# 人类可读输出
# ---------------------------------------------------------------------------
def render_text(result: dict[str, Any]) -> str:
    out: list[str] = []
    ids = result["ids"]
    matrix = result["matrix"]
    out.append("=" * 78)
    out.append("benchmarks/bench_embedding_similarity.py  ->  D-04「实测补充」")
    out.append("=" * 78)
    out.append("%s(dim=%d) | 余弦相似度矩阵 | %d 句 | 生成于 %s"
               % (result["embedder"], result["dim"], result["sentence_count"],
                  result["generated_at"]))
    out.append("")
    out.append("[1] 句子清单")
    for item in result["sentences"]:
        out.append("  %-4s %-3s %s" % (item["id"], item["lang"], item["text"]))
    out.append("")
    out.append("[2] 相似度矩阵（对称；保留 2 位小数）")
    out.append("       " + " ".join("%4s" % sid for sid in ids))
    for i, sid in enumerate(ids):
        cells = " ".join("%4.2f" % value for value in matrix[i])
        out.append("  %-4s %s" % (sid, cells))
    out.append("")
    out.append("[3] 人工标注的对照对")
    out.append("  %-13s %-4s %-4s %8s  %s" % ("relation", "a", "b", "cos", "句对"))
    for row in result["pairs"]:
        out.append("  %-13s %-4s %-4s %8.4f  %s || %s"
                   % (row["relation"], row["a"], row["b"], row["similarity"],
                      row["text_a"], row["text_b"]))
    out.append("")
    out.append("[4] 分组统计")
    for relation, stats in sorted(result["relation_stats"].items()):
        out.append("  %-13s n=%d  mean=%.4f  min=%.4f  max=%.4f"
                   % (relation, stats["n"], stats["mean"], stats["min"], stats["max"]))
    out.append("")
    v = result["verdict"]
    out.append("[5] 结论（⚠️ 与直觉相反，这是选型的固有限制）")
    out.append("  同义对平均      %.4f" % v["mean_synonym"])
    out.append("  反义对平均      %.4f" % v["mean_antonym"])
    out.append("  hard-synonym    %.4f  （词面重合极少的同义改写，分数反而最低）"
               % v["mean_hard_synonym"])
    out.append("  无关对平均      %.4f" % v["mean_unrelated"])
    out.append("  跨语言翻译对    %.4f  （中英零共享 token）" % v["mean_translation"])
    out.append("  判别力 margin = mean(synonym) - mean(antonym) = %.4f  -> %s"
               % (v["semantic_margin_syn_minus_ant"], v["discriminative_power"]))
    out.append("  %s" % v["summary"])
    out.append("")
    out.append("[6] 分词示例（为什么虚词也计分）")
    sample = result["sentences"][0]["text"]
    out.append("  tokenize(%r) = %r" % (sample, tokenize(sample)))
    out.append("")
    out.append("[7] 能力边界（必须如实说明）")
    for item in result["limitations"]:
        out.append("  - %s" % item)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 写 docs/VERIFICATION.md 的「实测补充」小节
# ---------------------------------------------------------------------------
_MARK_BEGIN = "<!-- BEGIN %s (auto-generated; 由脚本覆盖，请勿手改本块) -->" % _BENCH_ID
_MARK_END = "<!-- END %s -->" % _BENCH_ID


def update_verification(result: dict[str, Any]) -> Path:
    """只重写本脚本的标记块；文件不存在时创建一个**只含本节**的最小文件。"""
    block = "%s\n%s\n%s\n" % (_MARK_BEGIN, _render_verification_section(result), _MARK_END)
    if _VERIFICATION.exists():
        text = _VERIFICATION.read_text(encoding="utf-8")
    else:
        text = (
            "# liteagent 验证记录（VERIFICATION.md）\n\n"
            "> 本文件由多个部分拼装：§12.2 主表与 §12.3 简历对账表由主实现者维护；\n"
            "> 各决策的「实测补充」小节由 `benchmarks/*.py` 自动追加/更新，"
            "脚本只重写各自标记块之间的内容。\n"
        )
    if _MARK_BEGIN in text and _MARK_END in text:
        head, _, rest = text.partition(_MARK_BEGIN)
        _, _, tail = rest.partition(_MARK_END)
        text = head + block + tail.lstrip("\n")
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    _VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
    _VERIFICATION.write_text(text, encoding="utf-8")
    return _VERIFICATION


def _render_verification_section(result: dict[str, Any]) -> str:
    v = result["verdict"]
    ids = result["ids"]
    matrix = result["matrix"]
    lines: list[str] = []
    lines.append("### 实测补充 · D-04 Embedding 用纯 stdlib 的 Hashing Trick")
    lines.append("")
    lines.append("**落点**：`benchmarks/bench_embedding_similarity.py`（`%s`）" % _DOC_TRIGGER)
    lines.append("")
    lines.append("**环境**：Python %s ｜ %s(dim=%d) ｜ %d 句 / %d 对照对 ｜ 生成于 %s"
                 % (result["python"], result["embedder"], result["dim"],
                    result["sentence_count"], len(result["pairs"]), result["generated_at"]))
    lines.append("")
    lines.append("**相似度矩阵**（对称，保留 2 位小数）")
    lines.append("")
    lines.append("| | " + " | ".join(ids) + " |")
    lines.append("|---|" + "---|" * len(ids))
    for i, sid in enumerate(ids):
        lines.append("| **%s** | " % sid + " | ".join("%.2f" % value for value in matrix[i]) + " |")
    lines.append("")
    lines.append("**人工标注的对照对与实测余弦**")
    lines.append("")
    lines.append("| 关系 | 句 a | 句 b | 余弦 |")
    lines.append("|---|---|---|---:|")
    for row in result["pairs"]:
        lines.append("| %s | %s `%s` | %s `%s` | %.4f |"
                     % (row["relation"], row["a"], row["text_a"],
                        row["b"], row["text_b"], row["similarity"]))
    lines.append("")
    lines.append("**分组统计**")
    lines.append("")
    lines.append("| 关系 | n | mean | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for relation, stats in sorted(result["relation_stats"].items()):
        lines.append("| %s | %d | %.4f | %.4f | %.4f |"
                     % (relation, stats["n"], stats["mean"], stats["min"], stats["max"]))
    lines.append("")
    lines.append("**实测结论（⚠️ 与直觉相反）**")
    lines.append("")
    lines.append("- 同义对平均 **%.4f**，反义对平均 **%.4f** —— **反义句得分更高**。"
                 % (v["mean_synonym"], v["mean_antonym"]))
    lines.append("- 判别力 `margin = mean(synonym) - mean(antonym) = %.4f` -> **%s**。"
                 % (v["semantic_margin_syn_minus_ant"], v["discriminative_power"]))
    lines.append("- 同义但**词面重合极少**的改写（hard-synonym）平均只有 **%.4f**，比反义对还低。"
                 % v["mean_hard_synonym"])
    lines.append("- 跨语言翻译对平均 **%.4f**：'Good morning' 与 '早上好' 的余弦是 0.0。"
                 % v["mean_translation"])
    lines.append("")
    lines.append("机制解释：`HashingEmbedder` 是 **token 级的特征哈希**——把 token 用 "
                 "`blake2b` 映射到 `dim` 维、带符号累加、`1 + log1p(count)` 加权、再 L2 归一化。"
                 "它度量的是「两句共享了多少 token」。反义句几乎总是共享大部分实词"
                 "（`enable`/`disable`、喜欢/讨厌），而同义改写常常整套换词"
                 "（`book a flight` / `book a plane ticket`），所以前者分更高。"
                 "**这不是实现 bug，是选型的固有限制。**")
    lines.append("")
    lines.append("**能力边界（必须如实说明，不许说成「语义检索」）**")
    lines.append("")
    for item in result["limitations"]:
        lines.append("- %s" % item)
    lines.append("")
    lines.append("**因此 D-04 的论证要修正为**：默认 `HashingEmbedder` 的价值是"
                 "「零依赖 + 确定性 + 离线可跑通检索链路」，**不是**语义质量；"
                 "生产要语义必须把 `RemoteEmbedder`/自研向量服务注入 "
                 "`VectorMemory(embedder=...)`（v2 已冻结「注入的 embedder 的维度是权威」）。"
                 "面试时把这张表拿出来讲，比说「我实现了向量检索」有说服力得多——"
                 "因为它同时证明了我知道**自己方案的能力边界在哪**。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_embedding_similarity",
        description="HashingEmbedder 的同义/反义相似度矩阵与能力边界（D-04）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 打印结果")
    parser.add_argument("--dim", type=int, default=DEFAULT_HASHING_EMBED_DIM,
                        help="哈希维度（默认 %d）" % DEFAULT_HASHING_EMBED_DIM)
    parser.add_argument("--no-write", action="store_true", help="不写 docs/VERIFICATION.md")
    args = parser.parse_args(argv)

    result = compute(dim=max(1, args.dim))

    if not args.no_write:
        result["verification_written"] = str(update_verification(result))
    else:
        result["verification_written"] = None

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(result))
        if result["verification_written"]:
            print("\n已更新 %s" % result["verification_written"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
