from __future__ import annotations

# =============================================================================
# examples/04_memory.py —— liteagent 三层记忆的最小可运行教案
# =============================================================================
#
# 这个示例回答一个问题：**一个 Agent 的"记忆"到底分几层、每一层解决什么**。
#
# 三层（与 liteagent/memory/manager.py 的编排一一对应）：
#
#   1. 短期滑动窗口 (BufferMemory)
#      问题：对话越聊越长，而模型的上下文是有限的。
#      做法：按 token 预算 + 消息条数**双约束**，从最新一条往前保留，超出的挤出去。
#      它保证"送进模型的永远是最近、最相关的一小段"，且 add() 是 O(1)。
#
#   2. 摘要压缩 (SummaryMemory)
#      问题：被窗口挤出去的历史如果直接扔掉，模型就忘了最初的目标。
#      做法：把"被挤出去的那批消息"压成一段自然语言摘要，作为单独一条消息拼进 prompt。
#      它永远不抛异常：LLM 挂了 / 没配 LLM 就退化成零 LLM 的抽取式摘要。
#
#   3. 长期向量记忆 (VectorMemory)
#      问题：跨会话（甚至跨进程）需要记住"用户是谁、偏好什么"这类事实。
#      做法：把事实写成向量存起来，检索时用**混合打分**（相似度 + 近因 + 重要度）排序，
#            再用 MMR 去冗，避免 5 条几乎一样的记忆塞满上下文。
#
# 本示例全程离线、无 API key、无网络：
#   - 向量化用 HashingEmbedder（纯 stdlib 的**词面**哈希，确定性，不是语义 embedding）；
#   - 摘要用 ScriptedLLM 喂一段预置文本（离线确定性），并说明真实 provider 的接法。
#
# 跑法：
#   python3 examples/04_memory.py --offline
#   python3 examples/04_memory.py --provider echo        # 等价写法
#
# 注意：每个 .py 的第一行必须是 `from __future__ import annotations`（仓库冻结约定），
# 所以本文件用 `#` 注释而不是模块 docstring 来写说明。

import argparse
import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 让示例在"源码树里直接跑"与"pip install -e . 之后跑"两种情况下都能 import
# ---------------------------------------------------------------------------
# 直接 `python3 examples/04_memory.py` 时，sys.path[0] 是 examples/ 而不是仓库根，
# `import liteagent` 会失败。把仓库根插到最前面即可 —— 这行样板在每个示例里都有，
# 是"零安装体验"的代价，也是每个示例都必须自洽的原因。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent import (  # noqa: E402  （必须在 sys.path 调整之后再 import）
    BufferConfig,
    BufferMemory,
    HashingEmbedder,
    MemoryConfig,
    MemoryManager,
    Message,
    ScriptedLLM,
    ScriptedResponse,
)

# ---------------------------------------------------------------------------
# 小工具：让输出有可读的分节标题
# ---------------------------------------------------------------------------

WIDTH = 78


def hr(title: str) -> None:
    """打印一个分节标题。教学示例的输出应当能一眼看出"现在在讲哪一层"。"""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def role_label(message: Message) -> str:
    """把 role 渲染成固定宽度的小标签，便于对齐阅读。"""
    return f"[{message.role.value:<9}]"


# ===========================================================================
# 第 1 层：短期滑动窗口
# ===========================================================================

# 预算刻意调小，好让 8 轮对话就触发裁剪 —— 教学示例不该让人跑 10 分钟才看到效果。
# 真实项目里这两个数应当由模型上下文窗口反推（MemoryConfig.context_window_tokens）。
BUFFER_MAX_TOKENS = 140
BUFFER_MAX_MESSAGES = 8

#: 一段"模拟多轮对话"，用于把窗口撑爆。中文按 1 字 1 token 估算，所以这些内容很"贵"。
CHAT_TURNS: list[tuple[str, str]] = [
    ("帮我写一个读取 CSV 的函数。", "用 csv 模块即可：with open(...) as f: ..."),
    ("如果文件特别大、内存放不下呢？", "改成逐行流式读取，或者按 chunk 分块处理。"),
    ("那编码不是 utf-8 怎么办？", "open() 传 encoding 参数，必要时先用 errors='replace'。"),
    ("顺便帮我把结果写回一个新文件。", "写回时用 csv.writer，注意 newline='' 避免空行。"),
    ("这些 CSV 文件都放在 data/ 目录下。", "那就用 pathlib.Path('data').glob('*.csv') 遍历。"),
    ("能不能加一个进度条？", "三方库 tqdm 最省事，但纯 stdlib 可以自己打印百分比。"),
    ("我还是希望零依赖。", "那就每处理 N 行打印一次进度，别引入新依赖。"),
    ("总结一下我们最后定了什么方案。", "零依赖 + 流式读取 + 显式编码 + 手写进度输出。"),
]


def build_manager() -> MemoryManager:
    """装配一个"预算很小"的三层记忆管理器。

    `embedder=` 显式传 HashingEmbedder(dim=256)：它是纯 stdlib、确定性的**词面**哈希，
    没有网络、不需要 API key —— 这是"离线也能演示向量检索"的前提。
    换成生产语义检索时，这里注入 `RemoteEmbedder(...)` 即可，其余代码一行不动
    （这正是"注入点"的价值）。
    """
    config = MemoryConfig(
        buffer_max_tokens=BUFFER_MAX_TOKENS,
        buffer_max_messages=BUFFER_MAX_MESSAGES,
        buffer_keep_last_n=2,       # 无论如何至少保留最近 2 条（最小可用上下文）
        long_term_enabled=True,
        embedder_dim=256,
        summary_enabled=True,
        summary_trigger_ratio=0.8,  # 上下文用量到 80% 就压缩
        summary_min_evict=3,        # 或者积压了 3 条待压缩消息就压缩
        max_summary_chars=400,
    )
    summarizer_llm = ScriptedLLM(
        # 离线：用一段预置摘要代替真实模型输出。真实项目里换成
        # build_llm(LLMConfig(provider="openai", ...)) 即可，SummaryMemory 完全不关心。
        [ScriptedResponse.text(
            "- 目标：写一个零依赖的 CSV 处理脚本\n"
            "- 约束：不引三方库；文件可能很大，必须流式读取；编码可能不是 utf-8\n"
            "- 已定方案：csv 模块 + 显式 encoding + 手写进度输出"
        )],
        loop=True,  # 多次压缩时循环复用同一条脚本响应
    )
    return MemoryManager.from_config(config, llm=summarizer_llm, embedder=HashingEmbedder(dim=256))


def demo_short_term(manager: MemoryManager) -> None:
    """演示 (a)：短期窗口按 token 预算裁剪，打印裁剪前后的消息数与窗口内容。"""
    hr("第 1 层 / 短期滑动窗口：预算一满就从最旧的那条开始挤出去")

    buffer = manager.buffer
    buffer.add(Message.system("你是一个严谨的 Python 助手，回答要给出可执行的代码。"))
    for user_text, assistant_text in CHAT_TURNS:
        buffer.add(Message.user(user_text))
        buffer.add(Message.assistant(assistant_text))

    print(f"预算            : max_tokens={BUFFER_MAX_TOKENS}, max_messages={BUFFER_MAX_MESSAGES}, "
          f"keep_last_n={buffer.config.keep_last_n}")
    print(f"buffer（全量）  : {len(buffer.messages())} 条消息, "
          f"约 {buffer.estimated_tokens()} tokens")
    print("  -> 全量是**完整历史**，跨轮累积、只增不减；它不是要送进模型的东西。")

    # window() 是"每次调用重算"的：它按预算裁剪 + 修复被切断的工具对，
    # 返回的才是真正会送进模型的那批消息。add() 里不做裁剪是为了让 add 保持 O(1)。
    window = buffer.window()
    print()
    print(f"window（裁剪后）: {len(window)} 条消息, 约 {buffer.window_tokens()} tokens")
    print(f"被挤出去(evicted): {len(buffer.evicted())} 条消息")
    print()
    print("窗口内容（这就是模型这一轮真正看到的历史）：")
    for message in window:
        one_line = message.content.replace("\n", " ")
        print(f"  {role_label(message)} {one_line}")

    print()
    print("注意 pinned 前缀：开头的 SYSTEM 消息被钉住（keep_system=True），"
          "它不属于'可裁的业务对话'。")


# ===========================================================================
# 第 2 层：摘要压缩
# ===========================================================================

async def async_demo_summary(manager: MemoryManager) -> None:
    """演示 (b)：灌足够多的轮次触发压缩，打印摘要文本与 compression_count。

    压缩要调用 LLM（或抽取式兜底），是一条 async 链路，所以这个 demo 是 coroutine。
    """
    hr("第 2 层 / 摘要压缩：把被挤出去的历史压成一段可携带的摘要")

    # 再灌 4 轮，把刚才被挤出去的那批之外又攒出新的一批 evicted。
    extra_turns = [
        ("那日志格式用 JSON Lines 吧。", "可以，每行一个对象，方便流式追加与 grep。"),
        ("日志里要带时间戳和级别。", "用 datetime.now(timezone.utc).isoformat() 加 level 字段。"),
        ("错误也别抛崩，全部落日志。", "包一层 try/except，把异常信息写进日志再继续。"),
        ("好，就按这个来。", "收到，方案定稿。"),
    ]
    for user_text, assistant_text in extra_turns:
        manager.buffer.add(Message.user(user_text))
        manager.buffer.add(Message.assistant(assistant_text))

    # 必须先重算一次窗口，才知道"哪些消息被挤出去了"（evicted 是 window() 的产物）。
    manager.buffer.window()
    pending = len(manager.buffer.evicted())
    print(f"当前 buffer={len(manager.buffer.messages())} 条, "
          f"window={len(manager.buffer.window())} 条, 待压缩 evicted={pending} 条")
    print("触发条件（OR）：窗口用量 >= ceil(max_tokens * 0.8) 或 待压缩 >= 3 条")

    # acompress_if_needed：drain_evicted() -> should_compress() -> acompress() -> emit 事件。
    # 冻结契约：Agent 只在**每轮末尾**调一次，绝不能每个 observation 都调 ——
    # 否则一轮 N 个工具就是 N 次摘要调用 + N 份 token 账单，而摘要质量并不会更好。
    new_summary = await manager.acompress_if_needed()

    print()
    if new_summary is None:
        print("本轮没有触发压缩（没有待压缩批次，或未达阈值）。")
    else:
        print(f"压缩产物（摘要文本）：")
        for line in new_summary.splitlines():
            print(f"  {line}")
    print()
    print(f"compression_count = {manager.summarizer.compression_count}"
          "   （只增不减：它就是'真的压过几次'的证据）")

    # 幂等性演示：没有再产生新 evicted 时再调一次，不会推高计数。
    again = await manager.acompress_if_needed()
    print(f"再调一次 acompress_if_needed() -> {again!r}；"
          f"compression_count 仍为 {manager.summarizer.compression_count}")
    print("  -> 没有待压缩批次时直接返回 None：计数不会被空转调用推高。")

    print()
    print("降级路径（面试常问）：把 llm=None 时，SummaryMemory 走**抽取式兜底** ——")
    print("  每条消息取 '[role] ' + content[:80] 逐行拼接，零 LLM、零网络、完全确定性。")
    print("  摘要的'存在性'比质量更重要：被裁掉的历史一旦彻底消失就再也找不回来了。")


# ===========================================================================
# 第 3 层：长期向量记忆
# ===========================================================================

#: 要记住的"用户事实"。注意：HashingEmbedder 是**词面**相似，所以检索时换个措辞
#: 仍然要靠词面重叠命中 —— 这正是要如实说明的能力边界，不是 bug。
FACTS: list[tuple[str, float]] = [
    ("我偏好用 uv 管理 Python 项目的依赖，不喜欢 pip。", 0.9),
    ("我的项目统一用 ruff 做 lint 和格式化。", 0.7),
    ("我们的 CI 跑在 GitHub Actions 上，每次 push 触发。", 0.6),
    ("我叫小林，是一名后端工程师。", 0.8),
]

#: 检索用的"换一种措辞"的 query：与第 1 条事实共享「依赖 / 管理 / 工具」等词面。
RETRIEVAL_QUERY = "Python 项目的依赖该用什么工具管理比较好？"


async def demo_long_term(manager: MemoryManager) -> None:
    """演示 (c)：remember 几条事实，换一种措辞检索，打印命中与 score_breakdown。"""
    hr("第 3 层 / 长期向量记忆：写入有策略，检索是混合打分")

    print("写入（aremember -> VectorMemory.upsert，带去重）：")
    for content, importance in FACTS:
        item = await manager.aremember(content, importance=importance)
        print(f"  + importance={importance:.1f}  {item.content}")
    print()
    print(f"长期库现有 {len(manager.long_term)} 条；"
          f"embedder={manager.long_term.embedder.name!r}, dim={manager.long_term.dim}")

    # `now` 注入：recency = 2 ** (-age_days / half_life)。不固定 now 的话
    # "这条排第几"会随真实时钟漂移；这里取最新一条的 created_at，让 age=0 -> recency=1.0。
    reference_now = manager.long_term.all()[-1].created_at

    print()
    print(f"检索 query（注意：**换了措辞**）：{RETRIEVAL_QUERY!r}")
    hits = manager.retrieve(RETRIEVAL_QUERY, limit=3, now=reference_now)

    print()
    print("混合打分公式（冻结，面试核心）：")
    print("  score = 1.00 * sim  +  0.15 * recency  +  0.10 * importance")
    print("  sim        = 余弦相似度(query_vec, item_vec)")
    print("  recency    = 2 ** (-age_days / half_life_days)   （half_life=7 天）")
    print("  importance = clamp(item.importance, 0, 1)")
    print()
    print(f"命中 {len(hits)} 条（已按 score 降序、并经 MMR 去冗）：")
    for rank, item in enumerate(hits, start=1):
        breakdown = item.score_breakdown or {}
        print(f"  #{rank} score={item.score:.4f}   "
              f"score_breakdown={{'sim': {breakdown.get('sim', 0.0):+.4f}, "
              f"'recency': {breakdown.get('recency', 0.0):.4f}, "
              f"'importance': {breakdown.get('importance', 0.0):.2f}}}")
        print(f"      {item.content}")

    print()
    print("为什么不是'纯相似度排序'：三天前说过的偏好会因为一个无关的近期词就输掉；")
    print("近因项让'最近确认过的事实'更靠前，重要度项让'用户显式声明重要的事'更稳。")
    print("为什么还要 MMR：5 条几乎一样的记忆 = 只检索到 1 条，却花掉 5 条的上下文预算。")


# ===========================================================================
# 收尾：三层一起拼进 prompt
# ===========================================================================

async def demo_prompt_injection(manager: MemoryManager) -> None:
    """演示：<relevant_memories> 段落被注入 prompt 的样子（三层合体）。"""
    hr("合体：MemoryManager.abuild_prompt() 把三层拼成一份 prompt")

    reference_now = manager.long_term.all()[-1].created_at
    messages = await manager.abuild_prompt(
        system="你是一个严谨的 Python 助手。",
        user_input=RETRIEVAL_QUERY,
        now=reference_now,
    )

    print("组装顺序（冻结）：")
    print("  [1] system 提示")
    print("  [2] <conversation_summary> 摘要块（第 2 层，若有）")
    print("  [3] <relevant_memories>   长期记忆块（第 3 层，若有）")
    print("  [4] 短期窗口 window（第 1 层）")
    print("  [5] extra（multiagent 注入，本例为空）")
    print("  [6] 本轮增量 user 输入")
    print()
    print(f"实际组装出 {len(messages)} 条消息：")
    for index, message in enumerate(messages, start=1):
        first_line = message.content.splitlines()[0] if message.content else ""
        kind = message.metadata.get("kind", "")
        tag = f" (kind={kind})" if kind else ""
        print(f"  {index:>2}. {role_label(message)}{tag} {first_line[:60]}")

    print()
    print("--- 第 3 段 <relevant_memories> 的原文（这就是注入 prompt 的样子）---")
    memory_blocks = [m for m in messages if m.metadata.get("kind") == "memories"]
    if memory_blocks:
        print(memory_blocks[0].content)
    else:
        print("(本次检索没有命中，未注入记忆块)")

    print()
    print("--- 第 2 段 <conversation_summary> 的原文 ---")
    summary_blocks = [m for m in messages if m.metadata.get("kind") == "summary"]
    print(summary_blocks[0].content if summary_blocks else "(还没有摘要)")

    print()
    print("可观测性：last_retrieved 就是这次组装真正用到的条目 —— 它**不会**触发第二次")
    print("检索（search 会写 access_count，重复调用会污染数据）。")
    for item in manager.last_retrieved:
        print(f"  * score={item.score:.4f}  access_count={item.access_count}  {item.content}")

    print()
    stats = manager.stats()
    print("manager.stats()：")
    for key, value in stats.items():
        print(f"  {key:<22} = {value}")


# ===========================================================================
# 入口
# ===========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。

    本示例**没有联网路径**，所以 `--offline` 只是一个显式的自我声明（也是守门测试
    检查的开关）；`--provider echo` 是等价的写法，两者都接受。
    """
    parser = argparse.ArgumentParser(
        description="liteagent 三层记忆演示（离线、确定性、无需 API key）",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="离线模式（本示例的唯一模式；这是显式声明，便于守门测试与读者确认）",
    )
    parser.add_argument(
        "--provider", default="echo",
        help="LLM provider 名；本示例只用 echo/scripted（离线），其它取值会被忽略",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    print("liteagent 三层记忆演示 —— 离线、确定性、无需 API key")
    print(f"provider={args.provider!r}  offline={bool(args.offline)}")

    manager = build_manager()
    demo_short_term(manager)
    await async_demo_summary(manager)
    await demo_long_term(manager)
    await demo_prompt_injection(manager)

    hr("完成")
    print("三层各司其职：短期窗口管'最近'，摘要管'更早的要点'，向量库管'跨会话的事实'。")
    print("全部离线跑通，输出可复现。")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
