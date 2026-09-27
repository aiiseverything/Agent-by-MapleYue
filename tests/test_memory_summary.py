from __future__ import annotations

"""tests/test_memory_summary.py —— §8.4 `memory/summary.py` 的单元测试。

§12 第 5119 行要求的覆盖点：
  * `should_compress` 三条判定
  * LLM 摘要成功路径
  * LLM 失败走抽取式兜底且不抛
  * 超长截断
  * `compression_count`
  * `previous_summary` 被并入 prompt
"""

import unittest
from types import SimpleNamespace
from typing import Any, Sequence

from liteagent.llm.message import Message
from liteagent.memory.summary import (
    SUMMARY_PROMPT_TEMPLATE,
    SummaryConfig,
    SummaryMemory,
)


class _RecordingLLM:
    """鸭子类型的假 LLM：记录调用参数，可回放内容或抛异常（零网络、零时钟）。"""

    def __init__(self, *, content: str = "SUMMARY", error: BaseException | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> Any:
        self.calls.append((list(messages), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)

    @property
    def last_prompt(self) -> str:
        return self.calls[-1][0][0].content


def _msgs(count: int = 3) -> list[Message]:
    return [Message.user(f"hello {index}") for index in range(count)]


class ShouldCompressTests(unittest.TestCase):
    """§8.4 的冻结判定（OR 关系，任一成立即压缩）。"""

    def test_disabled_never_compresses(self) -> None:
        memory = SummaryMemory(config=SummaryConfig(enabled=False))
        self.assertFalse(
            memory.should_compress(current_tokens=10**6, max_tokens=100, pending_evicted=10**6)
        )

    def test_token_threshold_uses_ceil_of_ratio(self) -> None:
        memory = SummaryMemory(config=SummaryConfig(trigger_ratio=0.8, min_evict_batch=99))
        # ceil(100 * 0.8) == 80
        self.assertFalse(
            memory.should_compress(current_tokens=79, max_tokens=100, pending_evicted=0)
        )
        self.assertTrue(
            memory.should_compress(current_tokens=80, max_tokens=100, pending_evicted=0)
        )

    def test_pending_evicted_threshold(self) -> None:
        memory = SummaryMemory(config=SummaryConfig(trigger_ratio=10.0, min_evict_batch=4))
        self.assertFalse(
            memory.should_compress(current_tokens=0, max_tokens=100, pending_evicted=3)
        )
        self.assertTrue(
            memory.should_compress(current_tokens=0, max_tokens=100, pending_evicted=4)
        )

    def test_neither_condition_holds(self) -> None:
        memory = SummaryMemory(config=SummaryConfig(trigger_ratio=0.8, min_evict_batch=4))
        self.assertFalse(
            memory.should_compress(current_tokens=10, max_tokens=100, pending_evicted=1)
        )


class AcompressTests(unittest.IsolatedAsyncioTestCase):
    """`acompress` 的四条路径：成功 / 失败兜底 / 空输入 / 超长截断。"""

    async def test_llm_success_path(self) -> None:
        llm = _RecordingLLM(content="用户想给 liteagent 加记忆层。")
        memory = SummaryMemory(llm, SummaryConfig())
        result = await memory.acompress(_msgs(3))
        self.assertEqual(result, "用户想给 liteagent 加记忆层。")
        self.assertEqual(memory.summary, result)
        self.assertEqual(memory.compression_count, 1)
        self.assertEqual(len(llm.calls), 1)
        # prompt 走冻结模板：对话文本被渲染进 {transcript}。
        self.assertIn("Conversation to compress:", llm.last_prompt)
        self.assertIn("[user] hello 0", llm.last_prompt)

    async def test_llm_failure_falls_back_and_does_not_raise(self) -> None:
        llm = _RecordingLLM(error=RuntimeError("provider exploded"))
        memory = SummaryMemory(llm, SummaryConfig())
        with self.assertWarns(RuntimeWarning):
            result = await memory.acompress(_msgs(2))
        self.assertIn("[user] hello 0", result)
        self.assertIn("(2 messages summarized)", result)
        self.assertEqual(memory.compression_count, 1)

    async def test_no_llm_uses_extractive_fallback(self) -> None:
        memory = SummaryMemory(None, SummaryConfig())
        with self.assertWarns(RuntimeWarning):
            result = await memory.acompress(_msgs(2))
        self.assertEqual(result, memory.summary)
        self.assertIn("(2 messages summarized)", result)

    async def test_empty_llm_content_falls_back(self) -> None:
        llm = _RecordingLLM(content="   ")
        memory = SummaryMemory(llm, SummaryConfig())
        with self.assertWarns(RuntimeWarning):
            result = await memory.acompress(_msgs(1))
        self.assertIn("(1 messages summarized)", result)

    async def test_empty_input_returns_empty_and_keeps_previous_summary(self) -> None:
        memory = SummaryMemory(_RecordingLLM(content="OLD"))
        await memory.acompress(_msgs(1))
        self.assertEqual(memory.summary, "OLD")
        result = await memory.acompress([])
        self.assertEqual(result, "")
        self.assertEqual(memory.summary, "OLD")  # 空摘要绝不覆盖有效摘要
        self.assertEqual(memory.compression_count, 2)

    async def test_overlong_result_is_truncated_head_and_tail(self) -> None:
        memory = SummaryMemory(
            _RecordingLLM(content="x" * 500), SummaryConfig(max_summary_chars=100)
        )
        result = await memory.acompress(_msgs(1))
        self.assertLess(len(result), 500)
        self.assertIn("truncated", result)
        # 头尾都保留（head_ratio=0.7 -> 70 字符头、30 字符尾）。
        self.assertTrue(result.startswith("x" * 70), result[:80])
        self.assertTrue(result.endswith("x" * 30), result[-40:])
        self.assertEqual(memory.summary, result)

    async def test_compression_count_is_monotonic(self) -> None:
        memory = SummaryMemory(_RecordingLLM())
        self.assertEqual(memory.compression_count, 0)
        for expected in (1, 2, 3):
            await memory.acompress(_msgs(1))
            self.assertEqual(memory.compression_count, expected)


class PreviousSummaryTests(unittest.IsolatedAsyncioTestCase):
    """§8.4：`previous_summary` 被并入 prompt（滚动更新）。"""

    async def test_explicit_previous_summary_is_rendered_into_prompt(self) -> None:
        llm = _RecordingLLM(content="MERGED")
        memory = SummaryMemory(llm, SummaryConfig(update_existing=True))
        await memory.acompress(_msgs(2), previous_summary="OLD FACTS")
        self.assertIn("Existing summary so far (merge, do not repeat):\nOLD FACTS\n\n", llm.last_prompt)

    async def test_falling_back_to_stored_summary(self) -> None:
        llm = _RecordingLLM(content="FIRST")
        memory = SummaryMemory(llm, SummaryConfig(update_existing=True))
        await memory.acompress(_msgs(1))
        await memory.acompress(_msgs(1))
        self.assertIn("FIRST", llm.last_prompt)

    async def test_update_existing_false_omits_previous_block(self) -> None:
        llm = _RecordingLLM(content="FRESH")
        memory = SummaryMemory(llm, SummaryConfig(update_existing=False))
        memory._summary = "SHOULD NOT APPEAR"
        await memory.acompress(_msgs(1), previous_summary="ALSO NOT")
        self.assertNotIn("Existing summary so far", llm.last_prompt)
        self.assertNotIn("SHOULD NOT APPEAR", llm.last_prompt)
        self.assertNotIn("ALSO NOT", llm.last_prompt)

    async def test_transcript_with_braces_is_not_re_parsed(self) -> None:
        """`render_template` 用 format_map(_SafeDict)：JSON 里的 `{}` 不会被二次解析。"""
        llm = _RecordingLLM(content="OK")
        memory = SummaryMemory(llm, SummaryConfig())
        tricky = Message.user('Action Input: {"a": 1, "b": "{c}"}')
        await memory.acompress([tricky])
        self.assertIn('{"a": 1, "b": "{c}"}', llm.last_prompt)

    async def test_max_input_chars_truncates_the_transcript(self) -> None:
        llm = _RecordingLLM(content="OK")
        memory = SummaryMemory(llm, SummaryConfig(max_input_chars=60, update_existing=False))
        await memory.acompress([Message.user("y" * 400)])
        self.assertIn("truncated", llm.last_prompt)
        self.assertLess(len(llm.last_prompt), 60 + len(SUMMARY_PROMPT_TEMPLATE))


class FallbackExtractiveTests(unittest.TestCase):
    """§8.4 的抽取式兜底格式。"""

    def test_format_lines_and_stats_suffix(self) -> None:
        memory = SummaryMemory(None)
        text = memory.fallback_extractive_summary(_msgs(2))
        lines = text.splitlines()
        self.assertEqual(lines[0], "[user] hello 0")
        self.assertEqual(lines[1], "[user] hello 1")
        self.assertEqual(lines[2], "(2 messages summarized)")

    def test_each_line_keeps_only_first_80_chars(self) -> None:
        memory = SummaryMemory(None)
        text = memory.fallback_extractive_summary([Message.user("z" * 200)])
        first = text.splitlines()[0]
        self.assertEqual(first, "[user] " + "z" * 80)

    def test_roles_are_rendered_as_plain_strings(self) -> None:
        memory = SummaryMemory(None)
        text = memory.fallback_extractive_summary(
            [Message.system("sys"), Message.assistant("reply")]
        )
        self.assertEqual(text.splitlines()[0], "[system] sys")
        self.assertEqual(text.splitlines()[1], "[assistant] reply")

    def test_summary_tokens_follow_the_tokenizer(self) -> None:
        memory = SummaryMemory(None)
        self.assertEqual(memory.summary_tokens, 0)
        memory._summary = "abcd"
        self.assertEqual(memory.summary_tokens, 1)
        memory._summary = "你好"
        self.assertEqual(memory.summary_tokens, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
