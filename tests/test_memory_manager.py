from __future__ import annotations

"""tests/test_memory_manager.py —— §8.6 `memory/manager.py` 的单元测试。

§12 第 5122 行要求的覆盖点：
  * `abuild_prompt` 段落顺序与内容（**传 `now` 后才断言 `<relevant_memories>` 文本**）
  * `kind="memories"` 与 `memory_count`
  * `<conversation_summary>` 格式
  * `last_retrieved` 不触发第二次 search（用 `access_count` 断言）
  * `stats` 字段
  * `aclear` 两种范围
  * `build_prompt` 在 loop 内抛 `ConfigError`
  * `append_user_input=False` 时不追加第 6 段
  * `context_window_tokens` 反推 `buffer_budget_tokens`
"""

import unittest

from liteagent.config import (
    DEFAULT_REACT_SYSTEM_PROMPT,
    DEFAULT_TOKEN_CHAR_RATIO,
    format_ts,
    render_template,
)
from liteagent.errors import ConfigError
from liteagent.llm.message import Message, Role
from liteagent.memory.base import HeuristicTokenizer, MemoryConfig, MemoryItem
from liteagent.memory.embeddings import CallableEmbedder
from liteagent.memory.buffer import BufferConfig, BufferMemory
from liteagent.memory.manager import MemoryManager
from liteagent.memory.summary import SummaryConfig, SummaryMemory
from liteagent.memory.vector import VectorConfig, VectorMemory
from tests.helpers import det_embedder

_NOW = 1_700_000_000.0
_WEEK = 7.0 * 86400.0

_UNIT_X = [1.0, 0.0, 0.0, 0.0]
_UNIT_Y = [0.0, 1.0, 0.0, 0.0]
_TABLE = {"q": _UNIT_X, "alpha": _UNIT_X, "beta": _UNIT_Y}


def _long_term_item(item_id: str, content: str, created_at: float) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        content=content,
        created_at=created_at,
        last_access_at=created_at,
        importance=0.8,
    )


_MISSING = object()


def _manager(**overrides) -> MemoryManager:
    """一个完全确定的 MemoryManager：det_embedder + HeuristicTokenizer + 无 LLM 摘要器。

    `long_term` 的默认值受 `config.long_term_enabled` 约束（显式注入的层永远优先，
    所以 `long_term_enabled=False` 的用例不能顺手塞一个默认的 VectorMemory 进去）。
    """
    # 注意：不能用 `overrides.pop(...) or <默认值>` —— MemoryStore 定义了 __len__，
    # 空 store 是 falsy，`or` 会把调用方显式注入的空 buffer/空库悄悄换掉。
    config = overrides.pop("config", _MISSING)
    if config is _MISSING:
        config = MemoryConfig()
    buffer = overrides.pop("buffer", _MISSING)
    if buffer is _MISSING:
        buffer = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=20))
    long_term = overrides.pop("long_term", _MISSING)
    if long_term is _MISSING:
        long_term = (
            VectorMemory(embedder=det_embedder(_TABLE), config=VectorConfig())
            if config.long_term_enabled
            else None
        )
    summarizer = overrides.pop("summarizer", _MISSING)
    if summarizer is _MISSING:
        summarizer = SummaryMemory(llm=None, config=SummaryConfig())
    return MemoryManager(
        buffer=buffer,
        long_term=long_term,
        summarizer=summarizer,
        config=config,
        tokenizer=overrides.pop("tokenizer", None) or HeuristicTokenizer(),
        **overrides,
    )


def _permissive_embedder() -> CallableEmbedder:
    """给"写入任意文本"的用例用：所有文本都映射到同一个单位向量（不会 KeyError）。"""
    return CallableEmbedder(lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts], dim=4)


class BuildPromptAssemblyTests(unittest.IsolatedAsyncioTestCase):
    """§8.6 的 6 段组装顺序（冻结）。"""

    async def asyncSetUp(self) -> None:
        self.buffer = BufferMemory(BufferConfig(max_tokens=10_000, max_messages=20))
        self.long_term = VectorMemory(embedder=det_embedder(_TABLE), config=VectorConfig())
        self.summarizer = SummaryMemory(llm=None, config=SummaryConfig())
        self.manager = MemoryManager(
            buffer=self.buffer,
            long_term=self.long_term,
            summarizer=self.summarizer,
            config=MemoryConfig(),
            tokenizer=HeuristicTokenizer(),
        )
        self.buffer.add(Message.user("history message"))
        self.summarizer._summary = "user wants a memory layer"
        self.long_term.add(_long_term_item("m1", "alpha", _NOW - _WEEK))

    async def test_segment_order(self) -> None:
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="q", now=_NOW
        )
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0].role, Role.SYSTEM)
        self.assertEqual(messages[0].content, "SYS")
        self.assertEqual(messages[1].metadata["kind"], "summary")
        self.assertEqual(messages[2].metadata["kind"], "memories")
        self.assertEqual(messages[3].content, "history message")
        self.assertEqual(messages[4].role, Role.USER)
        self.assertEqual(messages[4].content, "q")

    async def test_conversation_summary_format(self) -> None:
        messages = await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        summary_message = messages[1]
        self.assertEqual(summary_message.role, Role.SYSTEM)
        self.assertEqual(
            summary_message.content,
            "<conversation_summary>\nuser wants a memory layer\n</conversation_summary>",
        )
        self.assertEqual(summary_message.metadata["kind"], "summary")

    async def test_relevant_memories_text_and_metadata(self) -> None:
        """**传了 `now`** 之后混合分完全确定，可以逐字断言整块文本。"""
        expected_score = 1.0 * 1.0 + 0.15 * 0.5 + 0.1 * 0.8
        date_text = format_ts(_NOW - _WEEK, fmt="%Y-%m-%d")
        expected_block = (
            "<relevant_memories>\n"
            f"- (score={expected_score:.2f}, {date_text}) alpha\n"
            "</relevant_memories>"
        )
        messages = await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        memories = messages[2]
        self.assertEqual(memories.role, Role.SYSTEM)
        self.assertEqual(memories.content, expected_block)
        self.assertEqual(memories.metadata["kind"], "memories")
        self.assertEqual(memories.metadata["memory_count"], 1)

    async def test_no_summary_and_no_memories_segments(self) -> None:
        self.summarizer._summary = ""
        self.long_term.clear()
        messages = await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        self.assertEqual([m.content for m in messages], ["SYS", "history message", "q"])

    async def test_retrieve_false_skips_the_memories_segment(self) -> None:
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="q", retrieve=False, now=_NOW
        )
        self.assertNotIn("memories", [m.metadata.get("kind") for m in messages])

    async def test_generated_segments_are_not_written_back_to_the_buffer(self) -> None:
        before = len(self.buffer)
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        self.assertEqual(len(self.buffer), before)

    async def test_extra_messages_come_before_the_user_input(self) -> None:
        extra = [Message.system("shared context")]
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="q", extra=extra, now=_NOW
        )
        self.assertEqual(messages[-2].content, "shared context")
        self.assertEqual(messages[-1].content, "q")

    async def test_append_user_input_false_omits_the_sixth_segment(self) -> None:
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="q", append_user_input=False, now=_NOW
        )
        self.assertEqual(messages[-1].content, "history message")
        self.assertNotIn("q", [m.content for m in messages])

    async def test_empty_user_input_is_not_appended(self) -> None:
        messages = await self.manager.abuild_prompt(system="SYS", user_input="", now=_NOW)
        self.assertEqual(messages[-1].content, "history message")

    async def test_empty_query_still_retrieves(self) -> None:
        """`user_input=""`（第 2 轮起）也要出长期记忆块，否则模型再也看不到召回的事实。"""
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="", append_user_input=False, now=_NOW
        )
        kinds = [m.metadata.get("kind") for m in messages]
        self.assertIn("memories", kinds)

    async def test_long_memory_block_is_capped_by_chars(self) -> None:
        table = {"q": _UNIT_X}
        big = "z" * 600
        table[big] = _UNIT_X
        manager = _manager(long_term=VectorMemory(embedder=det_embedder(table)))
        manager.long_term.add(
            MemoryItem(id="big", content=big, created_at=_NOW, importance=0.8)
        )
        messages = await manager.abuild_prompt(
            system="SYS", user_input="q", memory_query="q", now=_NOW
        )
        block = next(m.content for m in messages if m.metadata.get("kind") == "memories")
        self.assertIn("truncated", block)
        self.assertLess(block.count("z"), 600)
        self.assertLess(len(block), 4000 + 100)


class LastRetrievedTests(unittest.IsolatedAsyncioTestCase):
    """§8.6 `[v2 新增]`：`last_retrieved` 是可观测性缓存，不得触发第二次 search。"""

    async def asyncSetUp(self) -> None:
        self.manager = _manager()
        self.manager.long_term.add(_long_term_item("m1", "alpha", _NOW - _WEEK))

    async def test_reading_last_retrieved_does_not_search_again(self) -> None:
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        item = self.manager.long_term.get("m1")
        self.assertEqual(item.access_count, 1)
        for _ in range(3):
            self.assertEqual(len(self.manager.last_retrieved), 1)
        self.assertEqual(item.access_count, 1)  # 再读三次也没有第二次 search

    async def test_last_retrieved_is_a_copy(self) -> None:
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        snapshot = self.manager.last_retrieved
        snapshot.clear()
        self.assertEqual(len(self.manager.last_retrieved), 1)

    async def test_second_build_prompt_searches_again(self) -> None:
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        self.assertEqual(self.manager.long_term.get("m1").access_count, 2)

    async def test_retrieve_false_clears_last_retrieved(self) -> None:
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        self.assertEqual(len(self.manager.last_retrieved), 1)
        await self.manager.abuild_prompt(
            system="SYS", user_input="q", retrieve=False, now=_NOW
        )
        self.assertEqual(self.manager.last_retrieved, [])


class RetrieveDelegationTests(unittest.TestCase):
    """`retrieve` 的默认值来自 config，且逐级透传 `now`。"""

    def test_retrieve_uses_config_defaults(self) -> None:
        manager = _manager(config=MemoryConfig(retrieve_limit=1, retrieve_min_score=0.0))
        manager.long_term.add(_long_term_item("m1", "alpha", _NOW - _WEEK))
        manager.long_term.add(_long_term_item("m2", "beta", _NOW - _WEEK))
        results = manager.retrieve("q", now=_NOW)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "m1")

    def test_retrieve_returns_empty_without_long_term(self) -> None:
        manager = _manager(config=MemoryConfig(long_term_enabled=False))
        self.assertIsNone(manager.long_term)
        self.assertEqual(manager.retrieve("q", now=_NOW), [])

    def test_now_is_passed_through_to_the_scoring(self) -> None:
        manager = _manager()
        item = _long_term_item("m1", "alpha", _NOW - _WEEK)
        manager.long_term.add(item)
        results = manager.retrieve("q", limit=1, now=_NOW)
        self.assertEqual(results[0].score_breakdown["recency"], 0.5)
        self.assertEqual(item.last_access_at, _NOW)


class StatsAndClearTests(unittest.IsolatedAsyncioTestCase):
    """`stats` 的冻结键集与 `aclear` 的作用范围。"""

    async def asyncSetUp(self) -> None:
        self.manager = _manager()
        self.manager.buffer.add(Message.user("history"))
        self.manager.long_term.add(_long_term_item("m1", "alpha", _NOW - _WEEK))
        self.manager.summarizer._summary = "a summary"

    async def test_stats_has_the_frozen_key_set(self) -> None:
        stats = self.manager.stats()
        self.assertEqual(
            set(stats),
            {
                "short_term_messages",
                "short_term_tokens",
                "window_messages",
                "window_tokens",
                "evicted_pending",
                "long_term_items",
                "has_summary",
                "summary_tokens",
                "compressions",
                "embedder",
                "buffer_budget_tokens",
                "context_window_tokens",
            },
        )
        self.assertEqual(stats["short_term_messages"], 1)
        self.assertEqual(stats["long_term_items"], 1)
        self.assertTrue(stats["has_summary"])
        self.assertEqual(stats["embedder"], "det")
        self.assertIsNone(stats["context_window_tokens"])
        self.assertEqual(stats["buffer_budget_tokens"], self.manager.buffer.config.max_tokens)

    async def test_stats_tolerates_missing_long_term_and_distinguishes_no_summary(self) -> None:
        manager = _manager(config=MemoryConfig(long_term_enabled=False))
        stats = manager.stats()
        self.assertEqual(stats["long_term_items"], 0)
        self.assertIsNone(stats["embedder"])
        self.assertFalse(stats["has_summary"])
        self.assertEqual(stats["summary_tokens"], 0)
        self.assertEqual(stats["compressions"], 0)

    async def test_aclear_default_keeps_long_term_and_drops_summary(self) -> None:
        await self.manager.abuild_prompt(system="SYS", user_input="q", now=_NOW)
        await self.manager.aclear()
        self.assertEqual(len(self.manager.buffer), 0)
        self.assertEqual(len(self.manager.long_term), 1)   # 默认保留长期库
        self.assertEqual(self.manager.summarizer.summary, "")
        self.assertEqual(self.manager.last_retrieved, [])

    async def test_aclear_long_term_true_clears_the_vector_store(self) -> None:
        await self.manager.aclear(long_term=True)
        self.assertEqual(len(self.manager.long_term), 0)
        self.assertEqual(self.manager.summarizer.summary, "")

    async def test_aclear_summary_false_keeps_the_summary(self) -> None:
        await self.manager.aclear(summary=False)
        self.assertEqual(self.manager.summarizer.summary, "a summary")
        self.assertEqual(len(self.manager.buffer), 0)

    async def test_aclear_keeps_leading_system_messages(self) -> None:
        self.manager.buffer.clear(keep_system=False)
        self.manager.buffer.add(Message.system("pinned"))
        self.manager.buffer.add(Message.user("later"))
        await self.manager.aclear()
        self.assertEqual([m.content for m in self.manager.buffer.messages()], ["pinned"])


class SyncApiTests(unittest.IsolatedAsyncioTestCase):
    """§8.6：`build_prompt` 是 `run_sync` 包装，在运行中的 loop 内必须抛 `ConfigError`。"""

    async def test_build_prompt_inside_a_running_loop_raises_config_error(self) -> None:
        manager = _manager()
        with self.assertRaises(ConfigError):
            manager.build_prompt(system="SYS", user_input="q")

    def test_build_prompt_works_outside_a_running_loop(self) -> None:
        manager = _manager()
        messages = manager.build_prompt(system="SYS", user_input="hi")
        self.assertEqual(messages[0].content, "SYS")
        self.assertEqual(messages[-1].content, "hi")


class ContextBudgetTests(unittest.TestCase):
    """D-15：`context_window_tokens` 反推 `buffer_budget_tokens`。"""

    @staticmethod
    def _system_prompt_tokens() -> int:
        rendered = render_template(
            DEFAULT_REACT_SYSTEM_PROMPT,
            {"name": "agent", "tools": "", "tool_names": ""},
        )
        return HeuristicTokenizer().estimate(rendered)

    def test_budget_is_reverse_computed_from_the_context_window(self) -> None:
        config = MemoryConfig(
            context_window_tokens=8000,
            reserve_completion_tokens=1024,
            tools_schema_tokens_reserve=0,
        )
        manager = MemoryManager(config=config, tokenizer=HeuristicTokenizer())
        expected = max(512, 8000 - 1024 - 0 - self._system_prompt_tokens())
        self.assertEqual(manager.buffer.config.max_tokens, expected)
        self.assertEqual(manager.buffer_budget_tokens, expected)
        stats = manager.stats()
        self.assertEqual(stats["buffer_budget_tokens"], expected)
        self.assertEqual(stats["context_window_tokens"], 8000)
        self.assertEqual(manager.tokenizer_chars_budget(), expected * DEFAULT_TOKEN_CHAR_RATIO)

    def test_no_context_window_keeps_the_configured_buffer_budget(self) -> None:
        config = MemoryConfig(buffer_max_tokens=1234, context_window_tokens=None)
        manager = MemoryManager(config=config, tokenizer=HeuristicTokenizer())
        self.assertEqual(manager.buffer_budget_tokens, 1234)

    def test_too_small_window_clamps_to_512_with_a_warning(self) -> None:
        config = MemoryConfig(context_window_tokens=1000, reserve_completion_tokens=1024)
        with self.assertWarns(RuntimeWarning):
            manager = MemoryManager(config=config, tokenizer=HeuristicTokenizer())
        self.assertEqual(manager.buffer_budget_tokens, 512)

    def test_tools_schema_reserve_is_subtracted(self) -> None:
        config = MemoryConfig(
            context_window_tokens=8000,
            reserve_completion_tokens=0,
            tools_schema_tokens_reserve=2000,
        )
        manager = MemoryManager(config=config, tokenizer=HeuristicTokenizer())
        expected = max(512, 8000 - 0 - 2000 - self._system_prompt_tokens())
        self.assertEqual(manager.buffer_budget_tokens, expected)


class FromConfigTests(unittest.TestCase):
    """§8.6 的冻结映射表 + embedder 透传。"""

    def test_field_mapping_table(self) -> None:
        config = MemoryConfig(
            buffer_max_tokens=777,
            buffer_max_messages=9,
            buffer_keep_last_n=3,
            summary_enabled=False,
            summary_trigger_ratio=0.5,
            summary_min_evict=7,
            max_summary_chars=321,
            embedder_dim=64,
            write_policy="turn",
            auto_write_min_chars=11,
            dedup_threshold=0.5,
            max_items=42,
            retrieve_limit=2,
            retrieve_min_score=0.25,
            w_sim=2.0,
            w_recency=0.5,
            w_importance=0.25,
            recency_half_life_days=3.0,
            mmr_lambda=0.2,
        )
        manager = MemoryManager.from_config(config)
        self.assertEqual(manager.buffer.config.max_tokens, 777)
        self.assertEqual(manager.buffer.config.max_messages, 9)
        self.assertEqual(manager.buffer.config.keep_last_n, 3)
        self.assertTrue(manager.buffer.config.keep_system)
        self.assertFalse(manager.summarizer.config.enabled)
        self.assertEqual(manager.summarizer.config.trigger_ratio, 0.5)
        self.assertEqual(manager.summarizer.config.min_evict_batch, 7)
        self.assertEqual(manager.summarizer.config.max_summary_chars, 321)
        self.assertEqual(manager.summarizer.config.max_input_chars, 16000)
        self.assertTrue(manager.summarizer.config.update_existing)
        self.assertEqual(manager.long_term.config.dim, 64)
        self.assertEqual(manager.long_term.config.write_policy, "turn")
        self.assertEqual(manager.long_term.config.auto_write_min_chars, 11)
        self.assertEqual(manager.long_term.config.dedup_threshold, 0.5)
        self.assertEqual(manager.long_term.config.max_items, 42)
        self.assertEqual(manager.long_term.config.retrieve_limit, 2)
        self.assertEqual(manager.long_term.config.retrieve_min_score, 0.25)
        self.assertEqual(manager.long_term.config.w_sim, 2.0)
        self.assertEqual(manager.long_term.config.w_recency, 0.5)
        self.assertEqual(manager.long_term.config.w_importance, 0.25)
        self.assertEqual(manager.long_term.config.half_life_days, 3.0)
        self.assertEqual(manager.long_term.config.mmr_lambda, 0.2)
        self.assertTrue(manager.long_term.config.use_numpy)

    def test_injected_embedder_is_passed_through_and_dim_wins(self) -> None:
        config = MemoryConfig(embedder_dim=64)
        manager = MemoryManager.from_config(config, embedder=det_embedder(_TABLE))
        self.assertEqual(manager.long_term.dim, 4)
        self.assertEqual(manager.long_term.config.dim, 4)

    def test_long_term_disabled(self) -> None:
        manager = MemoryManager.from_config(MemoryConfig(long_term_enabled=False))
        self.assertIsNone(manager.long_term)
        self.assertEqual(manager.stats()["long_term_items"], 0)

    def test_tokenizer_override_reaches_all_three_layers(self) -> None:
        tokenizer = HeuristicTokenizer(char_ratio=2.0)
        manager = MemoryManager(
            buffer=BufferMemory(), summarizer=SummaryMemory(None), tokenizer=tokenizer
        )
        self.assertIs(manager.buffer.tokenizer, tokenizer)
        self.assertIs(manager.summarizer.tokenizer, tokenizer)
        self.assertIs(manager.tokenizer, tokenizer)


class AddEventTests(unittest.IsolatedAsyncioTestCase):
    """`aadd` 的三态 `auto_write` 与 `MEMORY_WRITE` 事件。"""

    async def asyncSetUp(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.manager = _manager(
            long_term=VectorMemory(embedder=_permissive_embedder(), config=VectorConfig()),
            on_event=lambda name, data: self.events.append((name, data)),
        )

    async def test_user_message_is_written_to_both_layers(self) -> None:
        written = await self.manager.aadd(Message.user("记住我用 uv 管理依赖"))
        self.assertIsNotNone(written)
        self.assertEqual(written.source, "auto")
        self.assertEqual(written.metadata["role"], "user")
        self.assertEqual(len(self.manager.buffer), 1)
        self.assertEqual(len(self.manager.long_term), 1)
        self.assertEqual(self.events[-1][0], "memory_write")

    async def test_auto_write_true_does_not_exempt_the_role_constraint(self) -> None:
        with self.assertWarns(RuntimeWarning):
            written = await self.manager.aadd(
                Message.assistant("x" * 200), auto_write=True
            )
        self.assertIsNone(written)
        self.assertEqual(len(self.manager.long_term), 0)
        self.assertEqual(len(self.manager.buffer), 1)  # 短期写入是无条件的

    async def test_auto_write_false_disables_the_long_term_write(self) -> None:
        self.assertIsNone(await self.manager.aadd(Message.user("记住这件事"), auto_write=False))
        self.assertEqual(len(self.manager.long_term), 0)

    async def test_short_unmarked_message_is_not_written(self) -> None:
        self.assertIsNone(await self.manager.aadd(Message.user("hi")))
        self.assertEqual(len(self.manager.long_term), 0)

    async def test_aremember_upserts_and_emits(self) -> None:
        item = await self.manager.aremember("记住我用 uv", importance=0.9, source="manual")
        self.assertEqual(item.importance, 0.9)
        self.assertEqual(item.metadata["role"], "user")
        self.assertEqual(len(self.manager.long_term), 1)
        await self.manager.aremember("记住我用 uv", importance=0.4)
        self.assertEqual(len(self.manager.long_term), 1)  # upsert 去重
        self.assertEqual(self.events[-1][0], "memory_write")

    async def test_aremember_without_long_term_raises_memory_store_error(self) -> None:
        manager = _manager(config=MemoryConfig(long_term_enabled=False))
        with self.assertRaises(Exception) as ctx:
            await manager.aremember("x")
        self.assertIn("long-term", str(ctx.exception).lower())

    async def test_aadd_turn_returns_only_the_user_item(self) -> None:
        written = await self.manager.aadd_turn(
            Message.user("记住我用 uv 管理依赖"), Message.assistant("好的")
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(len(self.manager.buffer), 2)


class CompressTests(unittest.IsolatedAsyncioTestCase):
    """`acompress_if_needed` 的编排（drain -> 判定 -> 摘要 -> 事件）。"""

    async def asyncSetUp(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.buffer = BufferMemory(BufferConfig(max_tokens=8, max_messages=2, keep_last_n=0))
        self.manager = _manager(
            buffer=self.buffer, on_event=lambda name, data: self.events.append((name, data))
        )

    async def test_force_compresses_the_evicted_batch(self) -> None:
        for index in range(5):
            self.buffer.add(Message.user(f"message {index}"))
        self.buffer.window()
        with self.assertWarns(RuntimeWarning):  # 没有配 LLM -> 抽取式兜底必须留痕
            result = await self.manager.acompress_if_needed(force=True)
        self.assertIsNotNone(result)
        self.assertIn("messages summarized", result)
        self.assertEqual(self.manager.summarizer.compression_count, 1)
        self.assertEqual(self.events[-1][0], "memory_compress")
        self.assertTrue(self.events[-1][1]["compressed"])

    async def test_nothing_evicted_means_no_compression(self) -> None:
        self.buffer.add(Message.user("only one"))
        self.buffer.window()
        self.assertIsNone(await self.manager.acompress_if_needed(force=True))
        self.assertEqual(self.manager.summarizer.compression_count, 0)

    async def test_disabled_summary_config_skips_compression(self) -> None:
        for index in range(5):
            self.buffer.add(Message.user(f"message {index}"))
        self.buffer.window()
        manager = _manager(
            buffer=self.buffer, summarizer=SummaryMemory(None, SummaryConfig(enabled=False))
        )
        self.assertIsNone(await manager.acompress_if_needed())
        self.assertEqual(manager.summarizer.compression_count, 0)

    async def test_acompress_is_an_alias(self) -> None:
        for index in range(5):
            self.buffer.add(Message.user(f"message {index}"))
        self.buffer.window()
        with self.assertWarns(RuntimeWarning):
            self.assertIsNotNone(await self.manager.acompress(force=True))


class ContextTruncationTests(unittest.IsolatedAsyncioTestCase):
    """D-15 的最终闸门：整段超预算时按"长期记忆块 -> 窗口最旧消息"裁剪并留痕。"""

    async def asyncSetUp(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.manager = _manager(
            long_term=VectorMemory(
                embedder=_permissive_embedder(), config=VectorConfig(retrieve_limit=5)
            ),
            config=MemoryConfig(context_window_tokens=800, reserve_completion_tokens=0),
            on_event=lambda name, data: self.events.append((name, data)),
        )
        for index in range(5):
            self.manager.long_term.add(
                MemoryItem(
                    id=f"m{index}",
                    content=f"q{index} " + "z" * 500,
                    created_at=1.0,
                    importance=0.5,
                )
            )
        for index in range(30):
            self.manager.buffer.add(Message.user(f"history {index} " + "x" * 100))

    async def test_memory_block_is_dropped_first_and_event_is_emitted(self) -> None:
        messages = await self.manager.abuild_prompt(
            system="SYS", user_input="q", memory_query="q0"
        )
        truncated = [data for name, data in self.events if name == "context_truncated"]
        self.assertEqual(len(truncated), 1)
        self.assertGreater(truncated[0]["before"], truncated[0]["after"])
        self.assertEqual(truncated[0]["dropped_messages"], 1)
        # 第一刀切长期记忆块（最可有可无的一段）。
        self.assertNotIn("memories", [m.metadata.get("kind") for m in messages])
        # 不动系统提示，也不动最后一条 user。
        self.assertEqual(messages[0].content, "SYS")
        self.assertEqual(messages[-1].content, "q")
        self.assertLess(
            self.manager.tokenizer.estimate_many([m.text() for m in messages]), 800
        )

    async def test_under_budget_emits_nothing(self) -> None:
        manager = _manager(config=MemoryConfig(context_window_tokens=100_000))
        manager.buffer.add(Message.user("short"))
        await manager.abuild_prompt(system="SYS", user_input="q")
        self.assertEqual(
            [name for name, _ in self.events if name == "context_truncated"], []
        )


class ToDictTests(unittest.TestCase):
    """`to_dict` 供 trace/CLI 使用，**不含 embedding**（§2.2 的 opt-out）。"""

    def test_to_dict_shape(self) -> None:
        manager = _manager()
        manager.long_term.add(_long_term_item("m1", "alpha", _NOW - _WEEK))
        payload = manager.to_dict()
        self.assertEqual(set(payload), {"stats", "summary", "last_retrieved", "long_term"})
        self.assertIn("stats", payload)
        self.assertEqual(len(payload["long_term"]), 1)
        self.assertNotIn("embedding", payload["long_term"][0])
        self.assertEqual(payload["long_term"][0]["id"], "m1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
