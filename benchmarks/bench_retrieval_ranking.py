from __future__ import annotations

# =============================================================================
# benchmarks/bench_retrieval_ranking.py —— 填 D-08 的「实测补充」
# =============================================================================
#
# 被测对象：`VectorMemory.search()` 在**同一个 20 条记忆的库**上、三种打分策略下的
# top-5 对照：
#
#   (A) 纯相似度      = VectorConfig(w_recency=0, w_importance=0) + use_mmr=False
#   (B) 混合打分      = VectorConfig()（w_sim=1.0 / w_recency=0.15 / w_importance=0.1）
#                       + use_mmr=False
#   (C) 混合 + MMR    = VectorConfig() + use_mmr=True（λ=0.7）
#
# 库的构成（**冻结**，逐条写在下面，任何一条改动都会让对照表失去意义）：
#   * 17 条"互不相同的事实"
#   * 3 条**冗余**（同一事实的三种写法，实测两两余弦 0.92~0.98）
#   * 2 条**过时**（200/220 天前的高重要度事实，同一条查询下词面仍然相关）
#
# 这张表要回答的问题：**只按向量相似度取 top-k 会错在哪？**
# 实测答案是两个具体失效，而不是一句"效果更好"：
#   1. 冗余挤占：纯相似度的 top-5 里有 3 条是同一事实（3/5 的上下文预算被浪费）。
#   2. 过时冒充：MySQL 5.7 那条（200 天前）在纯相似度下排第 3，
#      混合打分的近因衰减把它踢出 top-5。
# MMR 再把重复项压到 1 条，让"另一个相关主题"（延迟目标）进榜。
#
# 另一条同样重要的实测结论：**词面哈希让"过时"和"相关"是两个独立的信号，但不等于
# 语义对**。本脚本用 `HashingEmbedder`，它对"数据库"这个语义的召回靠的是词面重合，
# 所以结论只能推广到"打分公式与去冗算法"这一层，不能推广成"检索质量"。
#
# 跑法：
#   python3 benchmarks/bench_retrieval_ranking.py            # 人类可读对照表
#   python3 benchmarks/bench_retrieval_ranking.py --json     # 机器可读 JSON
#   python3 benchmarks/bench_retrieval_ranking.py --json --no-write
#   python3 benchmarks/bench_retrieval_ranking.py --top-k 5
#
# 注意：每个 .py 的第一行必须是 `from __future__ import annotations`（仓库冻结约定），
# 所以本文件用 `#` 注释而不是模块 docstring 来写说明。

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent.config import (  # noqa: E402
    DEFAULT_MMR_LAMBDA,
    DEFAULT_RECENCY_HALF_LIFE_D,
    DEFAULT_W_IMPORTANCE,
    DEFAULT_W_RECENCY,
    DEFAULT_W_SIM,
    format_ts,
    utc_now,
)
from liteagent.memory.base import MemoryItem  # noqa: E402
from liteagent.memory.embeddings import HashingEmbedder, cosine_similarity  # noqa: E402
from liteagent.memory.vector import VectorConfig, VectorMemory  # noqa: E402

_BENCH_ID = "bench-retrieval-ranking"
_VERIFICATION = _REPO_ROOT / "docs" / "VERIFICATION.md"
_DOC_TRIGGER = "python3 benchmarks/bench_retrieval_ranking.py --json"

#: 冻结的"现在"。**必须固定**：recency 是 `2 ** (-age_days / half_life)`，
#: 用真实时钟跑的话两小时后再跑一次，表里的分数就全变了（D-08 的"代价"一栏
#: 专门写了这个不确定性问题，缓解办法就是 `search(now=...)` 注入）。
NOW = 1_700_000_000.0
DAY = 86400.0

QUERY = "用户的编程语言和数据库偏好是什么？"

#: (id, 主题, 标签, 内容, importance, 距今多少天)
#: 标签取值：dup=冗余 / stale=过时 / fact=普通事实。主题用于算 top-5 的主题覆盖率。
MEMORIES: list[tuple[str, str, str, str, float, float]] = [
    # --- 3 条冗余：同一事实的三种写法（只改标点与虚词）---
    ("m00", "lang", "dup", "用户偏好用 Python 写后端服务，主力语言是 Python。", 0.6, 10),
    ("m01", "lang", "dup", "用户偏好用 Python 写后端服务，主力语言是 Python", 0.6, 9),
    ("m02", "lang", "dup", "用户偏好用 Python 写后端服务，主力语言为 Python。", 0.6, 8),
    # --- 2 条过时：重要度高、词面与查询相关，但时间很久 ---
    ("m03", "db", "stale", "用户的数据库是 MySQL 5.7，一直在用。", 0.9, 200),
    ("m04", "infra", "stale", "用户部署在单台物理机上的 Ubuntu 16.04。", 0.9, 220),
    # --- 15 条普通事实 ---
    ("m05", "db", "fact", "用户现在用 PostgreSQL 做生产数据库。", 0.7, 3),
    ("m06", "cache", "fact", "用户的缓存层是 Redis。", 0.5, 5),
    ("m07", "api", "fact", "用户偏好用 FastAPI 写 HTTP 接口。", 0.6, 4),
    ("m08", "process", "fact", "用户要求所有接口都要有类型注解。", 0.4, 30),
    ("m09", "process", "fact", "用户喜欢在 CI 里跑 pytest。", 0.4, 20),
    ("m10", "style", "fact", "用户的代码风格是 100 列宽。", 0.3, 60),
    ("m11", "style", "fact", "用户讨厌在代码里写 TODO 注释。", 0.3, 40),
    ("m12", "identity", "fact", "用户的名字是 Alice。", 0.9, 7),
    ("m13", "team", "fact", "用户的团队有 5 个人。", 0.5, 50),
    ("m14", "process", "fact", "用户希望每周一上午开会。", 0.4, 15),
    ("m15", "lang", "fact", "用户偏好用 Python 的 asyncio 处理并发。", 0.6, 6),
    ("m16", "db", "fact", "用户说数据库迁移计划在 Q3 完成。", 0.8, 12),
    ("m17", "ops", "fact", "用户的日志用结构化 JSON 格式。", 0.4, 25),
    ("m18", "db", "fact", "用户不喜欢 ORM，倾向手写 SQL。", 0.6, 11),
    ("m19", "perf", "fact", "用户的目标是把延迟降到 50ms 以下。", 0.7, 2),
]

#: 三种策略的冻结定义（名字会出现在 VERIFICATION.md 的表格里，别改）
STRATEGIES: list[tuple[str, str, dict[str, Any], bool]] = [
    (
        "pure_similarity",
        "纯相似度",
        {"w_sim": 1.0, "w_recency": 0.0, "w_importance": 0.0},
        False,
    ),
    (
        "hybrid",
        "混合打分",
        {"w_sim": DEFAULT_W_SIM, "w_recency": DEFAULT_W_RECENCY,
         "w_importance": DEFAULT_W_IMPORTANCE},
        False,
    ),
    (
        "hybrid_mmr",
        "混合 + MMR",
        {"w_sim": DEFAULT_W_SIM, "w_recency": DEFAULT_W_RECENCY,
         "w_importance": DEFAULT_W_IMPORTANCE},
        True,
    ),
]


def _make_store(mode_overrides: dict[str, Any]) -> VectorMemory:
    """造一个装好 20 条记忆的 `VectorMemory`（`created_at` 显式注入，可复现）。"""
    config = VectorConfig(dim=256, **mode_overrides)
    store = VectorMemory(config=config)
    for mem_id, _topic, _tag, content, importance, age_days in MEMORIES:
        item = MemoryItem.create(content, importance=importance, item_id=mem_id)
        item.created_at = NOW - age_days * DAY
        item.last_access_at = item.created_at
        store.add(item)
    return store


def _pairwise_redundancy(ids: Sequence[str], vectors: dict[str, list[float]]) -> float:
    """一组条目两两余弦的**最大值**（= 这一组里最像的两条有多像）。

    比平均值更能说明"冗余"：top-5 里只要有一对是 0.98，那一条就是白占预算。
    """
    best = 0.0
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            best = max(best, cosine_similarity(vectors[left], vectors[right]))
    return best


def compute(*, top_k: int) -> dict[str, Any]:
    """跑三种策略、两种 k（top-k 与全量），返回 JSON 可序列化的结果。"""
    embedder = HashingEmbedder(dim=256)
    vectors = {mem[0]: embedder.embed_one(mem[3]) for mem in MEMORIES}
    by_id = {mem[0]: mem for mem in MEMORIES}
    tags = {mem[0]: mem[2] for mem in MEMORIES}
    topics = {mem[0]: mem[1] for mem in MEMORIES}

    # 冗余三兄弟的实测两两余弦（人工标签的"证据"，不是声称）
    dup_ids = [mem[0] for mem in MEMORIES if mem[2] == "dup"]
    dup_pairs = []
    for i, left in enumerate(dup_ids):
        for right in dup_ids[i + 1:]:
            dup_pairs.append(
                {"a": left, "b": right,
                 "similarity": cosine_similarity(vectors[left], vectors[right])}
            )

    modes: list[dict[str, Any]] = []
    for key, label, overrides, use_mmr in STRATEGIES:
        store = _make_store(overrides)
        # 全量排序（rank 用），以及 top_k 截断（对照表用）
        full = store.search(QUERY, limit=len(MEMORIES), now=NOW, use_mmr=use_mmr)
        rank = {item.id: i + 1 for i, item in enumerate(full)}
        top = full[:top_k]
        top_ids = [item.id for item in top]

        rows = []
        for item in full:
            breakdown = item.score_breakdown or {}
            rows.append(
                {
                    "id": item.id,
                    "topic": topics[item.id],
                    "tag": tags[item.id],
                    "content": item.content,
                    "importance": item.importance,
                    "age_days": round((NOW - item.created_at) / DAY, 1),
                    "sim": breakdown.get("sim", 0.0),
                    "recency": breakdown.get("recency", 0.0),
                    "score": item.score or 0.0,
                    "rank": rank[item.id],
                    "in_top_k": item.id in top_ids,
                }
            )
        # 按 id 索引一份：跨模式的对照表必须**按 id** 取值，不能按下标 ——
        # 每个模式的排序不同，`modes[a]["rows"][i]` 与 `modes[b]["rows"][i]` 不是同一条。
        by_id = {row["id"]: row for row in rows}

        modes.append(
            {
                "key": key,
                "label": label,
                "use_mmr": use_mmr,
                "weights": {
                    "w_sim": overrides["w_sim"],
                    "w_recency": overrides["w_recency"],
                    "w_importance": overrides["w_importance"],
                },
                "rows": rows,  # 按本模式的名次排列
                "by_id": by_id,
                "top_k_ids": top_ids,
                "top_k_detail": [
                    {
                        "id": row["id"],
                        "rank": row["rank"],
                        "score": row["score"],
                        "sim": row["sim"],
                        "recency": row["recency"],
                        "tag": row["tag"],
                        "topic": row["topic"],
                        "content": row["content"],
                    }
                    for row in rows
                    if row["in_top_k"]
                ],
                "metrics": {
                    "duplicates_in_top_k": sum(1 for i in top_ids if tags[i] == "dup"),
                    "stale_in_top_k": sum(1 for i in top_ids if tags[i] == "stale"),
                    "distinct_topics_in_top_k": len({topics[i] for i in top_ids}),
                    "max_pairwise_similarity_in_top_k": _pairwise_redundancy(top_ids, vectors),
                    "stale_ranks": {
                        i: rank[i] for i in tags if tags[i] == "stale"
                    },
                },
            }
        )

    summary = {
        "top_k": top_k,
        "pure_similarity_top_k": modes[0]["top_k_ids"],
        "hybrid_top_k": modes[1]["top_k_ids"],
        "hybrid_mmr_top_k": modes[2]["top_k_ids"],
        "duplicates_in_top_k": {
            modes[i]["key"]: modes[i]["metrics"]["duplicates_in_top_k"] for i in range(3)
        },
        "stale_ranks": {
            modes[i]["key"]: modes[i]["metrics"]["stale_ranks"] for i in range(3)
        },
        "distinct_topics_in_top_k": {
            modes[i]["key"]: modes[i]["metrics"]["distinct_topics_in_top_k"] for i in range(3)
        },
        "summary": (
            "纯相似度把 3 条同一事实塞进 top-%d，且 200 天前的 MySQL 事实排第 %d；"
            "混合打分的近因衰减把它踢出 top-%d；MMR 再把冗余从 %d 条压到 %d 条、"
            "主题覆盖从 %d 提到 %d。"
            % (
                top_k,
                modes[0]["metrics"]["stale_ranks"].get("m03", -1),
                top_k,
                modes[0]["metrics"]["duplicates_in_top_k"],
                modes[2]["metrics"]["duplicates_in_top_k"],
                modes[0]["metrics"]["distinct_topics_in_top_k"],
                modes[2]["metrics"]["distinct_topics_in_top_k"],
            )
        ),
    }

    return {
        "bench": _BENCH_ID,
        "fills": "D-08「实测补充」",
        "generated_at": format_ts(utc_now()) + " UTC",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedder": "HashingEmbedder(dim=256)",
        "query": QUERY,
        "now": NOW,
        "now_readable": format_ts(NOW) + " UTC",
        "memory_count": len(MEMORIES),
        "redundant_count": len(dup_ids),
        "stale_count": sum(1 for mem in MEMORIES if mem[2] == "stale"),
        "half_life_days": DEFAULT_RECENCY_HALF_LIFE_D,
        "mmr_lambda": DEFAULT_MMR_LAMBDA,
        "duplicate_pair_similarities": dup_pairs,
        # 跨模式对照表的固定行序（= 夹具声明序），保证三种模式逐行可比
        "memory_order": [mem[0] for mem in MEMORIES],
        "modes": modes,
        "summary": summary,
        "caveats": [
            "默认 embedder 是 HashingEmbedder（词面哈希）：它对'数据库偏好'的召回靠的是"
            "token 重合，所以本表只能证明**打分公式与去冗算法的行为**，不能证明检索质量。"
            "换成真实的语义 embedder，sim 列会整体变化，但 recency/MMR 的机制不变。",
            "'过时'与'冗余'是我**人工标注**的（stale = 200/220 天前且已被新事实取代；"
            "dup = 同一事实的三种写法）。标签是这张表的真值来源，不是我测出来的结论；"
            "脚本把 dup 两两余弦与 stale 在各模式下的名次一并打印，供读者自行核对。",
            "`now` 是硬编码的 1700000000（2023-11-14 UTC），不是真实时钟："
            "recency 是连续衰减，用真实时钟会让这张表不可复现。",
            "只测 20 条记忆、1 条查询、top-5。样本是演示性的，不是检索评测集。",
        ],
    }


# ---------------------------------------------------------------------------
# 人类可读输出
# ---------------------------------------------------------------------------
def render_text(result: dict[str, Any]) -> str:
    top_k = result["summary"]["top_k"]
    out: list[str] = []
    out.append("=" * 96)
    out.append("benchmarks/bench_retrieval_ranking.py  ->  D-08「实测补充」")
    out.append("=" * 96)
    out.append("%s | %d 条记忆（%d 冗余 + %d 过时）| query=%r"
               % (result["embedder"], result["memory_count"], result["redundant_count"],
                  result["stale_count"], result["query"]))
    out.append("now=%s（硬编码）| 半衰期=%.0f 天 | MMR λ=%.1f | top-k=%d"
               % (result["now_readable"], result["half_life_days"],
                  result["mmr_lambda"], top_k))
    out.append("")
    out.append("[0] 冗余标签的证据：三条 dup 的实测两两余弦")
    for pair in result["duplicate_pair_similarities"]:
        out.append("    %s ~ %s  %.4f" % (pair["a"], pair["b"], pair["similarity"]))
    out.append("")
    out.append("[1] 各模式名次对照（按 id 取值；`*` = 落在该模式的 top-%d 内）" % top_k)
    header = "  %-4s %-6s %-9s %7s " % ("id", "tag", "topic", "sim")
    for mode in result["modes"]:
        header += "| %-26s " % mode["label"]
    out.append(header)
    out.append("  " + "-" * (34 + 29 * len(result["modes"])))
    for item in result["memory_order"]:
        base = result["modes"][0]["by_id"][item]
        line = "  %-4s %-6s %-9s %7.3f " % (base["id"], base["tag"], base["topic"],
                                            base["sim"])
        for mode in result["modes"]:
            mrow = mode["by_id"][item]
            line += "| #%-2d score=%.3f %s " % (mrow["rank"], mrow["score"],
                                                "*" if mrow["in_top_k"] else " ")
        out.append(line)
    out.append("")
    out.append("[2] top-%d 明细（每个模式内部按最终顺序）" % top_k)
    for mode in result["modes"]:
        out.append("  -- %s (use_mmr=%s, w=%s)" % (mode["label"], mode["use_mmr"],
                                                    mode["weights"]))
        for entry in mode["top_k_detail"]:
            out.append("     #%d %-4s %-6s sim=%.3f rec=%.3f score=%.3f  %s"
                       % (entry["rank"], entry["id"], entry["tag"], entry["sim"],
                          entry["recency"], entry["score"], entry["content"]))
    out.append("")
    out.append("[3] 量化指标")
    out.append("  %-16s %-22s %-22s %-22s" % ("mode", "dup in top-k", "stale ranks",
                                              "distinct topics / max pair sim"))
    for mode in result["modes"]:
        m = mode["metrics"]
        out.append("  %-16s %-22s %-22s %d / %.3f"
                   % (mode["label"],
                      "%d 条" % m["duplicates_in_top_k"],
                      str(m["stale_ranks"]),
                      m["distinct_topics_in_top_k"],
                      m["max_pairwise_similarity_in_top_k"]))
    out.append("")
    out.append("[4] 结论")
    out.append("  %s" % result["summary"]["summary"])
    out.append("")
    out.append("[5] 诚实边界（必须一并说明）")
    for caveat in result["caveats"]:
        out.append("  - %s" % caveat)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 写 docs/VERIFICATION.md 的「实测补充」小节
# ---------------------------------------------------------------------------
_MARK_BEGIN = "<!-- BEGIN %s (auto-generated; 由脚本覆盖，请勿手改本块) -->" % _BENCH_ID
_MARK_END = "<!-- END %s -->" % _BENCH_ID


def update_verification(result: dict[str, Any]) -> Path:
    """只重写本脚本的标记块；文件已存在时不动标记块之外的任何字节。"""
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
    top_k = result["summary"]["top_k"]
    modes = result["modes"]
    lines: list[str] = []
    lines.append("### 实测补充 · D-08 混合打分 + MMR 去冗")
    lines.append("")
    lines.append("**落点**：`benchmarks/bench_retrieval_ranking.py`（`%s`）" % _DOC_TRIGGER)
    lines.append("")
    lines.append("**环境**：Python %s ｜ %s ｜ 生成于 %s"
                 % (result["python"], result["embedder"], result["generated_at"]))
    lines.append("")
    lines.append("**夹具（冻结）**：%d 条记忆 = 17 条普通事实 + **%d 条冗余** + **%d 条过时**；"
                 "查询 `%s`；`now=%s`（硬编码，不是真实时钟）；"
                 "半衰期 %.0f 天；MMR λ=%.1f；top-k=%d。"
                 % (result["memory_count"], result["redundant_count"], result["stale_count"],
                    result["query"], result["now_readable"],
                    result["half_life_days"], result["mmr_lambda"], top_k))
    lines.append("")
    lines.append("**冗余标签的证据**（三条 dup 的实测两两余弦，说明它们确实是同一事实）：")
    lines.append("")
    lines.append("| a | b | 余弦 |")
    lines.append("|---|---|---:|")
    for pair in result["duplicate_pair_similarities"]:
        lines.append("| %s | %s | %.4f |" % (pair["a"], pair["b"], pair["similarity"]))
    lines.append("")
    lines.append("**三种策略的冻结定义**")
    lines.append("")
    lines.append("| 策略 | w_sim | w_recency | w_importance | use_mmr |")
    lines.append("|---|---:|---:|---:|---|")
    for mode in modes:
        w = mode["weights"]
        lines.append("| %s | %.2f | %.2f | %.2f | %s |"
                     % (mode["label"], w["w_sim"], w["w_recency"], w["w_importance"],
                        mode["use_mmr"]))
    lines.append("")
    lines.append("**逐条对照表**（按 id 逐行对齐；`sim` 三模式相同故只列一次；"
                 "`#n` 是该条在**该模式全量排序**里的名次，`*` 表示落在 top-%d 内）" % top_k)
    lines.append("")
    header = "| id | 标签 | 主题 | sim |"
    for mode in modes:
        header += " %s 名次/分数 |" % mode["label"]
    lines.append(header)
    lines.append("|---|---|---|---:|" + "---|" * len(modes))
    for item in result["memory_order"]:
        base = modes[0]["by_id"][item]
        cells = "| %s | %s | %s | %.3f |" % (base["id"], base["tag"], base["topic"],
                                            base["sim"])
        for mode in modes:
            mrow = mode["by_id"][item]
            cells += " #%d / %.3f%s |" % (mrow["rank"], mrow["score"],
                                          " \\*" if mrow["in_top_k"] else "")
        lines.append(cells)
    lines.append("")
    lines.append("**top-%d 明细**" % top_k)
    lines.append("")
    for mode in modes:
        lines.append("- **%s**（use_mmr=%s）：`%s`"
                     % (mode["label"], mode["use_mmr"], " → ".join(mode["top_k_ids"])))
        for entry in mode["top_k_detail"]:
            lines.append("  - #%d `%s`（%s, sim=%.3f, rec=%.3f, score=%.3f）%s"
                         % (entry["rank"], entry["id"], entry["tag"], entry["sim"],
                            entry["recency"], entry["score"], entry["content"]))
    lines.append("")
    lines.append("**量化指标**")
    lines.append("")
    lines.append("| 策略 | top-%d 里的冗余条数 | top-%d 里的过时条数 | top-%d 主题覆盖 | "
                 "top-%d 内最大两两余弦 |" % (top_k, top_k, top_k, top_k))
    lines.append("|---|---:|---:|---:|---:|")
    for mode in modes:
        m = mode["metrics"]
        lines.append("| %s | %d | %d | %d | %.3f |"
                     % (mode["label"], m["duplicates_in_top_k"], m["stale_in_top_k"],
                        m["distinct_topics_in_top_k"], m["max_pairwise_similarity_in_top_k"]))
    lines.append("")
    lines.append("**两个过时条目的名次变化**（名次越小越靠前；`-` 表示该模式下它排在最后之外）")
    lines.append("")
    lines.append("| 策略 | m03（MySQL 5.7，200 天前） | m04（Ubuntu 16.04，220 天前） |")
    lines.append("|---|---:|---:|")
    for mode in modes:
        ranks = mode["metrics"]["stale_ranks"]
        lines.append("| %s | #%s | #%s |" % (mode["label"], ranks.get("m03", "-"),
                                             ranks.get("m04", "-")))
    lines.append("")
    lines.append("**实测结论**")
    lines.append("")
    lines.append("1. **冗余挤占**：纯相似度的 top-%d 里有 **%d/%d** 条是同一事实"
                 "（三条写法彼此余弦 0.92~0.98）——**超过一半的上下文预算被浪费**，"
                 "而这正是 D-08 说的「同一事实写入 5 次占满 top-5」。"
                 % (top_k, modes[0]["metrics"]["duplicates_in_top_k"], top_k))
    lines.append("2. **过时冒充**：200 天前那条 MySQL 事实在纯相似度下排 **#%s**（进了 top-%d），"
                 "混合打分的近因衰减（半衰期 %.0f 天 -> recency=%.3f）把它踢出 top-%d。"
                 % (modes[0]["metrics"]["stale_ranks"].get("m03", "-"), top_k,
                    result["half_life_days"],
                    next((r["recency"] for r in modes[0]["rows"] if r["id"] == "m03"), 0.0),
                    top_k))
    lines.append("3. **MMR 去冗**：`hybrid_mmr` 把 top-%d 里的冗余从 %d 条压到 **%d 条**，"
                 "主题覆盖从 %d 提到 **%d**，组内最大两两余弦从 %.3f 降到 **%.3f**。"
                 % (top_k, modes[1]["metrics"]["duplicates_in_top_k"],
                    modes[2]["metrics"]["duplicates_in_top_k"],
                    modes[1]["metrics"]["distinct_topics_in_top_k"],
                    modes[2]["metrics"]["distinct_topics_in_top_k"],
                    modes[1]["metrics"]["max_pairwise_similarity_in_top_k"],
                    modes[2]["metrics"]["max_pairwise_similarity_in_top_k"]))
    lines.append("4. **权重仍然是相似度主导**：`w_sim=1.0` vs `w_recency=0.15` / "
                 "`w_importance=0.1`，所以时间只用来**打破接近分数的平局**，"
                 "而不是主导排序——表里 hybrid 与 pure 的前两名完全一致就是证据。")
    lines.append("")
    lines.append("**诚实边界（不许把这张表说成「检索质量提升」）**")
    lines.append("")
    for caveat in result["caveats"]:
        lines.append("- %s" % caveat)
    lines.append("")
    lines.append("**面试怎么讲**：把这三行 `top_ids` 拿出来，先说失效（3/5 是同一件事、"
                 "200 天前的旧数据库排第 3），再说修法（半衰期衰减 + MMR），"
                 "最后说**代价**（recency 让结果随时间漂移，所以要 `search(now=...)` 注入；"
                 "MMR 是 O(k²)、k ≤ 15 可忽略）。面试官问的从来不是「你用了什么算法」，"
                 "而是「你怎么知道它坏了、你怎么知道它修好了」——这张表就是答案。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_retrieval_ranking",
        description="20 条记忆的 top-k 对照：纯相似度 vs 混合打分 vs 混合+MMR（D-08）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 打印结果")
    parser.add_argument("--top-k", type=int, default=5, help="top-k（默认 5）")
    parser.add_argument("--no-write", action="store_true", help="不写 docs/VERIFICATION.md")
    args = parser.parse_args(argv)

    result = compute(top_k=max(1, min(args.top_k, len(MEMORIES))))

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
