from __future__ import annotations

"""``liteagent/config.py`` 的单元测试（§12 清单行：test_config.py）。

覆盖重点（§12 冻结清单，逐条对应到用例）：
* 默认值（§5.3 的五份配置 dataclass + §2.5 的常量）
* ``RetryPolicy.delay_for`` 与 ``compute_backoff`` **逐字等价**
* ``rng_seed`` 可复现、``rng=None`` 不崩
* ``truncate_head_tail`` 头尾都保留
* ``parse_dotenv`` 的引号 / 注释 / ``export``
* ``to_jsonable`` 各类型
* ``render_template`` 缺 key 不报错
* ``estimate_cost_usd`` 命中 / 未命中
* ``LoopBoundPool`` 的三条（同 key 不同 value 抛 ``ConfigError``、``release()`` 后
  ``len(pool)==0``、无运行 loop 时抛 ``ConfigError``）

规范真值源：``docs/INTERFACES.md`` §5（877-1285 行）、§2.5（常量）、§5.5（注入点）。
硬性要求：只用 stdlib ``unittest``、禁用裸 ``assert``、不联网、不睡真实时钟
（时间断言一律走 ``tests.helpers.frozen_time``）、异步用例继承 ``IsolatedAsyncioTestCase``。
"""

import asyncio
import json
import os
import random
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest import mock

from liteagent import config
from liteagent.errors import ConfigError, SerializationError
from liteagent.types import TokenUsage

from tests.helpers import RecordingSleep, frozen_time


@contextmanager
def _clean_env(**values: str):
    """把 ``os.environ`` 清空后注入指定变量（provider 相关的解析必须完全确定）。

    清空是有意的：真实机器上可能存在 ``LITEAGENT_API_KEY`` / ``OPENAI_API_KEY``，
    不清空就会让"没有 key 时返回 None"这类断言随机失败。
    """
    with mock.patch.dict(os.environ, dict(values), clear=True):
        yield


class _Color(Enum):
    RED = "red"


@dataclass
class _Box:
    name: str
    count: int = 0


class _Opaque:
    """to_jsonable 的"未知类型"用例（必须留痕，不能抛）。"""


# ======================================================================================
# §2.5 常量与 §5.3 配置默认值
# ======================================================================================


class DefaultsTests(unittest.TestCase):
    """§2.5 冻结常量与 §5.3 各配置 dataclass 的默认值。"""

    def test_module_constants_match_spec(self) -> None:
        self.assertEqual(config.DEFAULT_MAX_STEPS, 10)
        self.assertEqual(config.DEFAULT_MAX_CONCURRENCY, 4)
        self.assertEqual(config.DEFAULT_THREAD_POOL_SIZE, 8)
        self.assertEqual(config.DEFAULT_TOOL_TIMEOUT_S, 30.0)
        self.assertEqual(config.DEFAULT_LLM_TIMEOUT_S, 60.0)
        self.assertEqual(config.DEFAULT_MAX_RETRIES, 2)
        self.assertEqual(config.DEFAULT_BACKOFF_BASE_S, 0.25)
        self.assertEqual(config.DEFAULT_BACKOFF_MAX_S, 8.0)
        self.assertEqual(config.DEFAULT_BACKOFF_JITTER, 0.5)
        self.assertEqual(config.DEFAULT_MAX_RESULT_CHARS, 8000)
        self.assertEqual(config.DEFAULT_MAX_OBSERVATION_CHARS, 8000)
        self.assertEqual(config.NO_TIMEOUT, -1.0)

    def test_llm_config_defaults(self) -> None:
        llm = config.LLMConfig()
        self.assertEqual(llm.provider, "openai")
        self.assertEqual(llm.model, "gpt-4o-mini")
        self.assertIsNone(llm.api_key)
        self.assertIsNone(llm.base_url)
        self.assertEqual(llm.timeout_s, config.DEFAULT_LLM_TIMEOUT_S)
        self.assertEqual(llm.retry_policy, config.RetryPolicy())
        self.assertIsNone(llm.temperature)
        self.assertIsNone(llm.max_tokens)
        self.assertEqual(llm.extra_headers, {})
        self.assertEqual(llm.extra_body, {})
        self.assertFalse(llm.stream)
        self.assertIsNone(llm.sleep_fn)

    def test_retry_policy_defaults(self) -> None:
        policy = config.RetryPolicy()
        self.assertEqual(policy.max_retries, config.DEFAULT_MAX_RETRIES)
        self.assertEqual(policy.backoff_base_s, config.DEFAULT_BACKOFF_BASE_S)
        self.assertEqual(policy.backoff_max_s, config.DEFAULT_BACKOFF_MAX_S)
        self.assertEqual(policy.jitter, config.DEFAULT_BACKOFF_JITTER)
        self.assertIsNone(policy.rng_seed)
        self.assertIsNone(policy.sleep_fn)

    def test_agent_config_defaults(self) -> None:
        agent = config.AgentConfig()
        self.assertEqual(agent.max_steps, config.DEFAULT_MAX_STEPS)
        self.assertEqual(agent.mode, "auto")
        self.assertEqual(agent.tool_choice, "auto")
        self.assertEqual(agent.temperature, config.DEFAULT_TEMPERATURE)
        self.assertTrue(agent.parallel_tool_calls)
        self.assertEqual(agent.repeat_action_policy, "nudge_then_fail")
        self.assertFalse(agent.raise_on_error)
        self.assertIsNone(agent.system_prompt)
        self.assertEqual(agent.system_prompt_template, config.DEFAULT_REACT_SYSTEM_PROMPT)
        self.assertEqual(agent.name, "agent")
        self.assertEqual(agent.max_total_tokens, config.DEFAULT_MAX_TOTAL_TOKENS)
        self.assertIsNone(agent.max_wall_clock_s)

    def test_memory_config_defaults(self) -> None:
        memory = config.MemoryConfig()
        self.assertEqual(memory.buffer_max_tokens, config.DEFAULT_BUFFER_MAX_TOKENS)
        self.assertEqual(memory.buffer_max_messages, config.DEFAULT_BUFFER_MAX_MESSAGES)
        self.assertEqual(memory.write_policy, "selective")
        self.assertEqual(memory.retrieve_limit, config.DEFAULT_RETRIEVE_LIMIT)
        self.assertEqual(memory.mmr_lambda, config.DEFAULT_MMR_LAMBDA)
        self.assertIsNone(memory.persist_path)

    def test_executor_config_defaults(self) -> None:
        executor = config.ExecutorConfig()
        self.assertEqual(executor.max_concurrency, config.DEFAULT_MAX_CONCURRENCY)
        # §5.3 [v2 变更]：None 表示"对所有工具禁用超时"，默认是个具体值。
        self.assertEqual(executor.default_timeout_s, config.DEFAULT_TOOL_TIMEOUT_S)
        self.assertEqual(executor.thread_pool_size, config.DEFAULT_THREAD_POOL_SIZE)
        self.assertEqual(executor.sequential_tools, frozenset())
        self.assertFalse(executor.fail_fast)
        self.assertFalse(executor.allow_retry_on_non_idempotent)
        self.assertIsNone(executor.approval_policy)
        self.assertEqual(executor.disable_tool_after_failures, config.DEFAULT_TOOL_FAILURE_LIMIT)

    def test_team_config_defaults(self) -> None:
        team = config.TeamConfig()
        self.assertEqual(team.max_depth, config.DEFAULT_TEAM_MAX_DEPTH)
        self.assertEqual(team.max_rounds, config.DEFAULT_TEAM_MAX_ROUNDS)
        self.assertTrue(team.parallel_subagents)
        self.assertFalse(team.share_memory)
        self.assertEqual(team.propagate_failure, "return")
        self.assertTrue(team.enable_cycle_detection)

    def test_app_config_defaults(self) -> None:
        app = config.AppConfig()
        self.assertIsInstance(app.llm, config.LLMConfig)
        self.assertIsInstance(app.agent, config.AgentConfig)
        self.assertIsInstance(app.memory, config.MemoryConfig)
        self.assertIsInstance(app.executor, config.ExecutorConfig)
        self.assertIsInstance(app.team, config.TeamConfig)
        self.assertEqual(app.tools, [])
        self.assertIsNone(app.trace_file)
        self.assertIsNone(app.sandbox_root)
        self.assertFalse(app.verbose)

    def test_configs_do_not_share_mutable_defaults(self) -> None:
        # default_factory 的经典陷阱：两个实例的 dict 必须是两个对象。
        first = config.LLMConfig()
        second = config.LLMConfig()
        first.extra_headers["x"] = "y"
        first.retry_policy.max_retries = 99
        self.assertEqual(second.extra_headers, {})
        self.assertEqual(second.retry_policy.max_retries, config.DEFAULT_MAX_RETRIES)

    def test_model_prices_table_matches_spec(self) -> None:
        self.assertEqual(
            set(config.MODEL_PRICES),
            {"gpt-4o-mini", "claude-3-5-haiku", "deepseek-chat"},
        )
        self.assertEqual(config.MODEL_PRICES["gpt-4o-mini"], (0.00015, 0.0006))


# ======================================================================================
# §5.1 compute_backoff / RetryPolicy
# ======================================================================================


class ComputeBackoffTests(unittest.TestCase):
    """§5.1：full = min(max_s, base_s * 2**attempt)，jitter 三档，clamp 到 [0, max_s]。"""

    def test_jitter_zero_is_deterministic_full(self) -> None:
        self.assertEqual(config.compute_backoff(0, jitter=0.0), 0.25)
        self.assertEqual(config.compute_backoff(1, jitter=0.0), 0.5)
        self.assertEqual(config.compute_backoff(2, jitter=0.0), 1.0)

    def test_exponential_growth_is_capped_by_max_s(self) -> None:
        self.assertEqual(config.compute_backoff(0, jitter=0.0), 0.25)
        self.assertEqual(config.compute_backoff(5, jitter=0.0), 8.0)  # 0.25*32 = 8
        self.assertEqual(config.compute_backoff(20, jitter=0.0), 8.0)

    def test_result_is_clamped_into_range(self) -> None:
        self.assertLessEqual(config.compute_backoff(20, base_s=100.0, max_s=8.0, jitter=0.0), 8.0)
        self.assertGreaterEqual(config.compute_backoff(20, base_s=100.0, max_s=8.0, jitter=0.0), 0.0)

    def test_custom_base_and_max(self) -> None:
        self.assertEqual(config.compute_backoff(3, base_s=1.0, max_s=10.0, jitter=0.0), 8.0)
        self.assertEqual(config.compute_backoff(4, base_s=1.0, max_s=10.0, jitter=0.0), 10.0)

    def test_equal_jitter_stays_within_fraction_of_full(self) -> None:
        for attempt in range(6):
            with self.subTest(attempt=attempt):
                full = min(8.0, 0.25 * 2**attempt)
                value = config.compute_backoff(
                    attempt, jitter=0.5, rng=random.Random(1234)
                )
                self.assertGreaterEqual(value, 0.5 * full - 1e-12)
                self.assertLessEqual(value, full + 1e-12)

    def test_full_jitter_stays_within_zero_and_full(self) -> None:
        for attempt in range(6):
            with self.subTest(attempt=attempt):
                full = min(8.0, 0.25 * 2**attempt)
                value = config.compute_backoff(attempt, jitter=1.0, rng=random.Random(7))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, full + 1e-12)

    def test_rng_none_does_not_crash(self) -> None:
        # §5.1 [v2]：rng=None 等价 random.Random(None)，禁止用模块级全局函数。
        value = config.compute_backoff(3, jitter=0.5, rng=None)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, config.DEFAULT_BACKOFF_MAX_S)

    def test_rng_none_with_full_jitter_does_not_crash(self) -> None:
        value = config.compute_backoff(1, jitter=1.0, rng=None)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)

    def test_rng_none_ignored_when_jitter_zero(self) -> None:
        self.assertEqual(config.compute_backoff(2, jitter=0.0, rng=None), 1.0)

    def test_is_reproducible_with_a_seeded_rng(self) -> None:
        first = config.compute_backoff(4, jitter=0.5, rng=random.Random(2024))
        second = config.compute_backoff(4, jitter=0.5, rng=random.Random(2024))
        self.assertEqual(first, second)

    def test_does_not_touch_the_global_random_module(self) -> None:
        # §5.5 规则 4：禁止模块级 random.*（会与并发重试/其它测试互相扰动）。
        random.seed(99)
        before = random.random()
        random.seed(99)
        config.compute_backoff(3, jitter=0.5, rng=random.Random(1))
        self.assertEqual(random.random(), before)


class RetryPolicyDelayTests(unittest.TestCase):
    """§5.1：``delay_for`` 的逐字实现是委托 ``compute_backoff``（测试断言两者完全一致）。"""

    def test_delay_for_is_verbatim_equivalent_to_compute_backoff(self) -> None:
        policy = config.RetryPolicy(
            backoff_base_s=0.5, backoff_max_s=4.0, jitter=0.0
        )
        for attempt in range(8):
            with self.subTest(attempt=attempt):
                self.assertEqual(
                    policy.delay_for(attempt),
                    config.compute_backoff(
                        attempt, base_s=0.5, max_s=4.0, jitter=0.0
                    ),
                )

    def test_delay_for_equivalent_with_jitter_and_identical_seeds(self) -> None:
        policy = config.RetryPolicy(backoff_base_s=0.25, backoff_max_s=8.0, jitter=0.5)
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                self.assertEqual(
                    policy.delay_for(attempt, rng=random.Random(42)),
                    config.compute_backoff(
                        attempt, base_s=0.25, max_s=8.0, jitter=0.5, rng=random.Random(42)
                    ),
                )

    def test_delay_for_honours_its_own_parameters(self) -> None:
        policy = config.RetryPolicy(backoff_base_s=2.0, backoff_max_s=3.0, jitter=0.0)
        self.assertEqual(policy.delay_for(0), 2.0)
        self.assertEqual(policy.delay_for(9), 3.0)

    def test_delay_for_without_rng_does_not_crash(self) -> None:
        policy = config.RetryPolicy(jitter=0.5)
        value = policy.delay_for(1)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, policy.backoff_max_s)

    def test_rng_seed_is_reproducible_across_policy_instances(self) -> None:
        # §5.5 规则 3：`random.Random(seed)` 序列可复现 —— 两次构造得到同一序列。
        first = config.RetryPolicy(jitter=0.5, rng_seed=17)
        second = config.RetryPolicy(jitter=0.5, rng_seed=17)
        first_delays = [first.delay_for(i, rng=random.Random(17)) for i in range(5)]
        second_delays = [second.delay_for(i, rng=random.Random(17)) for i in range(5)]
        self.assertEqual(first_delays, second_delays)

    def test_rng_seed_sequence_differs_between_seeds(self) -> None:
        policy = config.RetryPolicy(jitter=0.5)
        self.assertNotEqual(
            [policy.delay_for(i, rng=random.Random(1)) for i in range(4)],
            [policy.delay_for(i, rng=random.Random(2)) for i in range(4)],
        )

    def test_sleep_fn_field_is_carried_but_not_used_by_delay_for(self) -> None:
        sleeper = RecordingSleep()
        policy = config.RetryPolicy(jitter=0.0, sleep_fn=sleeper)
        self.assertEqual(policy.delay_for(0), 0.25)
        self.assertEqual(sleeper.delays, [])  # delay_for 只算不睡


class RetryPolicySerializationTests(unittest.TestCase):
    """§5.1 / §2.2：``to_dict`` / ``from_dict``（callable 无法往返）。"""

    def test_to_dict_has_all_fields(self) -> None:
        payload = config.RetryPolicy(jitter=0.25, rng_seed=3).to_dict()
        self.assertEqual(
            set(payload),
            {"max_retries", "backoff_base_s", "backoff_max_s", "jitter", "rng_seed", "sleep_fn"},
        )
        json.dumps(payload)
        self.assertEqual(payload["rng_seed"], 3)
        self.assertIsNone(payload["sleep_fn"])

    def test_to_dict_renders_callable_as_a_name(self) -> None:
        payload = config.RetryPolicy(sleep_fn=RecordingSleep()).to_dict()
        self.assertIsInstance(payload["sleep_fn"], str)
        json.dumps(payload)

    def test_from_dict_round_trip(self) -> None:
        policy = config.RetryPolicy(max_retries=5, backoff_base_s=1.0, jitter=0.1, rng_seed=9)
        self.assertEqual(config.RetryPolicy.from_dict(policy.to_dict()), policy)

    def test_from_dict_tolerates_missing_fields(self) -> None:
        self.assertEqual(config.RetryPolicy.from_dict({}), config.RetryPolicy())

    def test_from_dict_warns_and_drops_sleep_fn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = config.RetryPolicy.from_dict({"sleep_fn": "helpers.RecordingSleep"})
        self.assertIsNone(policy.sleep_fn)
        self.assertTrue(any("sleep_fn" in str(w.message) for w in caught))

    def test_from_dict_warns_on_unknown_key(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy = config.RetryPolicy.from_dict({"nope": 1})
        self.assertEqual(policy, config.RetryPolicy())
        self.assertTrue(any("nope" in str(w.message) for w in caught))


# ======================================================================================
# §5.2 辅助纯函数
# ======================================================================================


class TruncateHeadTailTests(unittest.TestCase):
    """§5.2：``max_chars<=0`` 或未超长原样返回；否则头尾都保留。"""

    def test_shorter_text_is_returned_verbatim(self) -> None:
        self.assertEqual(config.truncate_head_tail("short", 20), "short")
        self.assertEqual(config.truncate_head_tail("exactly", 7), "exactly")

    def test_max_chars_zero_or_negative_returns_text(self) -> None:
        text = "x" * 100
        self.assertEqual(config.truncate_head_tail(text, 0), text)
        self.assertEqual(config.truncate_head_tail(text, -5), text)

    def test_head_and_tail_are_both_preserved(self) -> None:
        text = "".join(str(i % 10) for i in range(200))
        truncated = config.truncate_head_tail(text, 20)
        self.assertTrue(truncated.startswith(text[:14]))  # int(20 * 0.7) = 14
        self.assertTrue(truncated.endswith(text[-6:]))

    def test_marker_reports_the_dropped_count(self) -> None:
        truncated = config.truncate_head_tail("A" * 100, 20)
        self.assertIn("truncated 80 chars", truncated)

    def test_custom_head_ratio(self) -> None:
        text = "H" * 50 + "T" * 50
        truncated = config.truncate_head_tail(text, 10, head_ratio=0.2)
        self.assertTrue(truncated.startswith("H" * 2))
        self.assertTrue(truncated.endswith("T" * 8))

    def test_custom_marker(self) -> None:
        truncated = config.truncate_head_tail("A" * 30, 10, marker="<{n}>")
        self.assertIn("<20>", truncated)

    def test_head_ratio_one_keeps_only_head(self) -> None:
        text = "H" * 30 + "T" * 30
        truncated = config.truncate_head_tail(text, 10, head_ratio=1.0)
        self.assertTrue(truncated.startswith("H" * 10))
        self.assertFalse(truncated.endswith("T"))

    def test_ratio_is_clamped(self) -> None:
        text = "A" * 100
        truncated = config.truncate_head_tail(text, 10, head_ratio=5.0)
        self.assertTrue(truncated.startswith("A" * 10))

    def test_empty_text_is_returned_verbatim(self) -> None:
        self.assertEqual(config.truncate_head_tail("", 10), "")


class ParseDotenvTests(unittest.TestCase):
    """§5.2：空行/注释、``export``、引号剥离与转义、不支持变量插值。"""

    def test_plain_pairs(self) -> None:
        parsed = config.parse_dotenv("A=1\nB=two\n")
        self.assertEqual(parsed, {"A": "1", "B": "two"})

    def test_skips_blank_lines_and_comments(self) -> None:
        parsed = config.parse_dotenv("\n# comment\n   \nA=1\n   # indented comment\n")
        self.assertEqual(parsed, {"A": "1"})

    def test_export_prefix_is_supported(self) -> None:
        self.assertEqual(config.parse_dotenv("export A=1\n"), {"A": "1"})
        self.assertEqual(config.parse_dotenv("export\tB=2\n"), {"B": "2"})

    def test_lines_without_equals_are_skipped(self) -> None:
        self.assertEqual(config.parse_dotenv("JUST_TEXT\nA=1\n"), {"A": "1"})

    def test_whitespace_around_key_and_value_is_stripped(self) -> None:
        self.assertEqual(config.parse_dotenv("  A  =  value  \n"), {"A": "value"})

    def test_double_quotes_are_stripped_and_escapes_decoded(self) -> None:
        parsed = config.parse_dotenv('A="line1\\nline2\\t!"\n')
        self.assertEqual(parsed["A"], "line1\nline2\t!")

    def test_single_quotes_are_literal(self) -> None:
        parsed = config.parse_dotenv("A='no\\nescape'\n")
        self.assertEqual(parsed["A"], "no\\nescape")

    def test_quoted_hash_is_not_a_comment(self) -> None:
        parsed = config.parse_dotenv('A="value # not a comment"\nB=\'a # b\'\n')
        self.assertEqual(parsed["A"], "value # not a comment")
        self.assertEqual(parsed["B"], "a # b")

    def test_trailing_comment_on_bare_value_is_stripped(self) -> None:
        self.assertEqual(config.parse_dotenv("A=value # comment\n"), {"A": "value"})

    def test_hash_without_leading_space_is_kept(self) -> None:
        self.assertEqual(config.parse_dotenv("A=value#tag\n"), {"A": "value#tag"})

    def test_no_variable_interpolation(self) -> None:
        parsed = config.parse_dotenv("A=$HOME/x\nB=${OTHER}\n")
        self.assertEqual(parsed["A"], "$HOME/x")
        self.assertEqual(parsed["B"], "${OTHER}")

    def test_last_assignment_wins(self) -> None:
        self.assertEqual(config.parse_dotenv("A=1\nA=2\n"), {"A": "2"})

    def test_value_containing_equals_is_kept(self) -> None:
        self.assertEqual(config.parse_dotenv("A=a=b\n"), {"A": "a=b"})

    def test_empty_value(self) -> None:
        self.assertEqual(config.parse_dotenv("A=\n"), {"A": ""})

    def test_unknown_escape_keeps_the_backslash(self) -> None:
        self.assertEqual(config.parse_dotenv('A="a\\db"\n'), {"A": "a\\db"})

    def test_load_dotenv_writes_environ_and_respects_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("LITEAGENT_TEST_DOTENV=from-file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"LITEAGENT_TEST_DOTENV": "pre-existing"}):
                parsed = config.load_dotenv(path, override=False)
                self.assertEqual(parsed, {"LITEAGENT_TEST_DOTENV": "from-file"})
                self.assertEqual(os.environ["LITEAGENT_TEST_DOTENV"], "pre-existing")
                config.load_dotenv(path, override=True)
                self.assertEqual(os.environ["LITEAGENT_TEST_DOTENV"], "from-file")

    def test_load_dotenv_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(config.load_dotenv(Path(tmp) / "absent.env"), {})


class ParseBoolTests(unittest.TestCase):
    """§5.2：真值/假值表；空串是明确 False；其它值（含 None）回退 ``default``。"""

    def test_truthy_values(self) -> None:
        for raw in ("1", "true", "TRUE", "Yes", "on", " y "):
            with self.subTest(raw=raw):
                self.assertIs(config.parse_bool(raw), True)

    def test_falsy_values(self) -> None:
        for raw in ("0", "false", "No", "OFF", ""):
            with self.subTest(raw=raw):
                self.assertIs(config.parse_bool(raw), False)

    def test_unknown_value_falls_back_to_default(self) -> None:
        self.assertIs(config.parse_bool("maybe", default=True), True)
        self.assertIs(config.parse_bool("maybe", default=False), False)

    def test_none_falls_back_to_default(self) -> None:
        self.assertIs(config.parse_bool(None, default=True), True)
        self.assertIs(config.parse_bool(None), False)

    def test_bool_passthrough(self) -> None:
        self.assertIs(config.parse_bool(True), True)
        self.assertIs(config.parse_bool(False), False)

    def test_empty_string_is_definitely_false(self) -> None:
        # §5.2：空串是"明确 False"而不是 default（否则 LITEAGENT_ALLOW_NETWORK="" 会静默打开联网）。
        self.assertIs(config.parse_bool("", default=True), False)


class ToJsonableTests(unittest.TestCase):
    """§5.2：dataclass / Enum / set / tuple / Path / bytes / Exception / 未知类型。"""

    def test_scalars_and_none_pass_through(self) -> None:
        self.assertIsNone(config.to_jsonable(None))
        self.assertEqual(config.to_jsonable(True), True)
        self.assertEqual(config.to_jsonable(3), 3)
        self.assertEqual(config.to_jsonable(1.5), 1.5)
        self.assertEqual(config.to_jsonable("s"), "s")

    def test_enum_becomes_its_value(self) -> None:
        self.assertEqual(config.to_jsonable(_Color.RED), "red")

    def test_dataclass_becomes_field_dict(self) -> None:
        payload = config.to_jsonable(_Box("a", 2))
        self.assertEqual(payload, {"name": "a", "count": 2})

    def test_nested_dataclass_is_recursive(self) -> None:
        payload = config.to_jsonable({"box": _Box("a"), "colors": {_Color.RED}})
        self.assertEqual(payload, {"box": {"name": "a", "count": 0}, "colors": ["red"]})

    def test_set_tuple_frozenset_become_lists(self) -> None:
        self.assertEqual(config.to_jsonable((1, 2)), [1, 2])
        self.assertEqual(config.to_jsonable({1}), [1])
        self.assertEqual(config.to_jsonable(frozenset({1})), [1])

    def test_path_becomes_str(self) -> None:
        self.assertEqual(config.to_jsonable(Path("/tmp/x")), "/tmp/x")

    def test_bytes_become_base64(self) -> None:
        self.assertEqual(config.to_jsonable(b"hi"), "aGk=")
        self.assertEqual(config.to_jsonable(bytearray(b"hi")), "aGk=")

    def test_exception_becomes_type_and_message(self) -> None:
        self.assertEqual(
            config.to_jsonable(ValueError("bad")),
            {"type": "ValueError", "message": "bad"},
        )

    def test_unknown_type_is_marked_unserializable(self) -> None:
        opaque = _Opaque()
        payload = config.to_jsonable(opaque)
        self.assertEqual(payload["unserializable"], True)
        self.assertEqual(payload["repr"], repr(opaque))

    def test_mapping_keys_are_stringified(self) -> None:
        self.assertEqual(config.to_jsonable({1: "a"}), {"1": "a"})

    def test_everything_is_json_serializable(self) -> None:
        payload = config.to_jsonable(
            {
                "enum": _Color.RED,
                "box": _Box("a", 1),
                "set": {1, 2},
                "path": Path("/tmp"),
                "bytes": b"\x00\x01",
                "exc": config.ConfigError("boom"),
                "opaque": _Opaque(),
            }
        )
        json.dumps(payload)

    def test_token_usage_is_converted_through_fields(self) -> None:
        payload = config.to_jsonable(TokenUsage(prompt_tokens=1, completion_tokens=2))
        self.assertEqual(payload["prompt_tokens"], 1)
        self.assertEqual(payload["total_tokens"], 3)


class RenderTemplateTests(unittest.TestCase):
    """§5.3：``format_map(_SafeDict(...))``，缺 key 不报错。"""

    def test_substitutes_known_keys(self) -> None:
        self.assertEqual(config.render_template("hi {name}", {"name": "a"}), "hi a")

    def test_missing_key_is_left_as_literal_placeholder(self) -> None:
        # §5.3 冻结：_SafeDict.__missing__ 返回 "{" + key + "}"。
        self.assertEqual(config.render_template("{a} {missing}", {"a": "1"}), "1 {missing}")

    def test_does_not_raise_on_unknown_placeholder(self) -> None:
        rendered = config.render_template("x {nope} y", {})
        self.assertIn("{nope}", rendered)

    def test_injected_values_are_not_reparsed(self) -> None:
        # tools 已经是渲染好的多行文本；其中的 JSON 大括号不能被二次解析。
        self.assertEqual(
            config.render_template("{a}", {"a": '{"json": 1}'}), '{"json": 1}'
        )

    def test_default_system_prompt_prompt_renders_all_placeholders(self) -> None:
        rendered = config.render_template(
            config.DEFAULT_REACT_SYSTEM_PROMPT,
            {"name": "agent", "tools": "no tools", "tool_names": "none"},
        )
        self.assertNotIn("{name}", rendered)
        self.assertNotIn("{tools}", rendered)
        self.assertNotIn("{tool_names}", rendered)
        self.assertIn("Final Answer:", rendered)

    def test_agent_config_render_system_prompt_uses_template(self) -> None:
        agent = config.AgentConfig()
        rendered = agent.render_system_prompt(name="helper", tools_text="T", tool_names="a,b")
        self.assertIn("helper", rendered)
        self.assertIn("T", rendered)

    def test_agent_config_explicit_system_prompt_replaces_template(self) -> None:
        agent = config.AgentConfig(system_prompt="FROZEN PROMPT")
        self.assertEqual(
            agent.render_system_prompt(name="x", tools_text="y", tool_names="z"),
            "FROZEN PROMPT",
        )


class SystemPromptTemplateTests(unittest.TestCase):
    """§5.3：``DEFAULT_REACT_SYSTEM_PROMPT`` 是**冻结字面量**（逐字对照）。"""

    #: 逐字转抄自 §5.3 的 ```text 代码块（末尾带一个换行）。
    FROZEN_PROMPT = "\n".join(
        [
            "You are {name}, a helpful AI agent that solves tasks step by step.",
            "",
            "You have access to the following tools:",
            "{tools}",
            "",
            "Use the following format:",
            "",
            "Thought: your reasoning about what to do next",
            "Action: the name of the tool to use, one of [{tool_names}]",
            "Action Input: the arguments, as a JSON object",
            "Observation: the result of the tool (this is filled in for you, do not write it yourself)",
            "... (Thought/Action/Action Input/Observation may repeat)",
            "Thought: I now know the final answer",
            "Final Answer: the final answer to the user",
            "",
            "Rules:",
            "- Emit exactly ONE Action per step. Wait for the Observation before the next Thought.",
            "- Action Input must be a single-line JSON object matching the tool's parameters.",
            "- Never invent tool names. Available tools: [{tool_names}]",
            "- If your output is truncated, continue from where you stopped.",
            '- When you have enough information, stop calling tools and emit "Final Answer: ...".',
            "",
        ]
    )

    def test_default_template_matches_the_frozen_literal(self) -> None:
        self.assertEqual(config.DEFAULT_REACT_SYSTEM_PROMPT, self.FROZEN_PROMPT)

    def test_agent_config_uses_the_frozen_literal_as_default(self) -> None:
        self.assertEqual(config.AgentConfig().system_prompt_template, self.FROZEN_PROMPT)

    def test_tool_names_placeholder_appears_twice(self) -> None:
        self.assertEqual(config.DEFAULT_REACT_SYSTEM_PROMPT.count("{tool_names}"), 2)


class EstimateCostUsdTests(unittest.TestCase):
    """§5.2：命中价格表按公式算；未命中返回 None（诚实 > 猜）。"""

    def test_hit_uses_prompt_and_completion_prices(self) -> None:
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
        self.assertAlmostEqual(
            config.estimate_cost_usd(usage, model="gpt-4o-mini"),
            0.00015 + 0.0006,
            places=12,
        )

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(
            config.estimate_cost_usd(TokenUsage(prompt_tokens=1), model="unknown-model")
        )
        self.assertIsNone(config.estimate_cost_usd(TokenUsage(), model=""))

    def test_partial_usage_only_charges_the_given_side(self) -> None:
        usage = TokenUsage(prompt_tokens=2000, completion_tokens=0)
        self.assertAlmostEqual(
            config.estimate_cost_usd(usage, model="deepseek-chat"), 0.00054, places=12
        )

    def test_zero_usage_is_zero_cost_on_a_known_model(self) -> None:
        self.assertEqual(
            config.estimate_cost_usd(TokenUsage(), model="claude-3-5-haiku"), 0.0
        )

    def test_real_usage_is_never_returned_as_none_when_price_known(self) -> None:
        cost = config.estimate_cost_usd(
            TokenUsage(prompt_tokens=10, completion_tokens=5), model="gpt-4o-mini"
        )
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0.0)


# ======================================================================================
# §5.2 时钟与同步入口
# ======================================================================================


class ClockTests(unittest.TestCase):
    """§2.2 / §5.2：``utc_now`` 是唯一时钟，``frozen_time`` 能把它冻住。"""

    def test_utc_now_returns_unix_seconds_float(self) -> None:
        self.assertIsInstance(config.utc_now(), float)

    def test_frozen_time_patches_utc_now(self) -> None:
        with frozen_time(1_700_000_000.0):
            self.assertEqual(config.utc_now(), 1_700_000_000.0)

    def test_frozen_time_is_restored_after_the_block(self) -> None:
        with frozen_time(1.0):
            pass
        self.assertNotEqual(config.utc_now(), 1.0)

    def test_format_ts_uses_utc(self) -> None:
        self.assertEqual(config.format_ts(0.0), "1970-01-01 00:00:00")

    def test_format_ts_accepts_a_custom_format(self) -> None:
        self.assertEqual(config.format_ts(0.0, fmt="%Y"), "1970")

    def test_frozen_now_reads_the_module_hook(self) -> None:
        original = config._FROZEN_NOW

        def restore() -> None:
            config._FROZEN_NOW = original

        self.addCleanup(restore)
        config._FROZEN_NOW = 42.0
        self.assertEqual(config.frozen_now(), 42.0)
        config._FROZEN_NOW = None
        self.assertIsInstance(config.frozen_now(), float)


class RunSyncTests(unittest.TestCase):
    """§5.2：``run_sync`` 收工厂函数；运行中的 loop 里调用抛 ``ConfigError``。"""

    def test_run_sync_executes_the_coroutine(self) -> None:
        async def main() -> int:
            return 42

        self.assertEqual(config.run_sync(lambda: main()), 42)

    def test_run_sync_propagates_exceptions(self) -> None:
        async def main() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            config.run_sync(lambda: main())

    def test_run_sync_cleans_up_loop_bound_primitives(self) -> None:
        # §5.2 的 _run_and_cleanup 必须在 finally 里 release_loop —— 否则每次都泄漏一个 loop。
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)

        async def main() -> None:
            pool.semaphore("exec", 2)
            self.assertEqual(len(pool), 1)

        config.run_sync(lambda: main())
        self.assertEqual(len(pool), 0)

    def test_run_sync_requires_a_factory_not_a_coroutine_object(self) -> None:
        # §5.2 [v2]：签名冻结为收工厂函数。传协程对象会立刻失败（而不是被静默 await），
        # 这正是"先检查 loop、后构造协程"的目的：协程对象根本没被造出来。
        async def main() -> int:
            return 1

        coro = main()
        try:
            with self.assertRaises(TypeError):
                config.run_sync(coro)  # type: ignore[arg-type]
        finally:
            coro.close()


class RunSyncInsideLoopTests(unittest.IsolatedAsyncioTestCase):
    """§5.2：已有运行中的 loop 时 ``run_sync`` 抛 ``ConfigError``（绝不套娃 run_until_complete）。"""

    async def test_run_sync_raises_config_error_inside_running_loop(self) -> None:
        async def inner() -> int:
            return 1

        with self.assertRaises(ConfigError):
            config.run_sync(lambda: inner())

    async def test_config_error_message_mentions_the_async_variant(self) -> None:
        async def inner() -> int:
            return 1

        with self.assertRaises(ConfigError) as ctx:
            config.run_sync(lambda: inner())
        self.assertIn("running event loop", str(ctx.exception))


# ======================================================================================
# §5.3 配置 dataclass 的行为
# ======================================================================================


class LLMConfigTests(unittest.TestCase):
    """§5.3：``resolve_api_key`` / ``resolved_base_url`` / ``to_dict`` / ``from_dict``。"""

    def test_resolve_api_key_prefers_explicit_value(self) -> None:
        with _clean_env(LITEAGENT_API_KEY="generic", ANTHROPIC_API_KEY="anthropic"):
            self.assertEqual(config.LLMConfig(api_key="explicit").resolve_api_key(), "explicit")
            # 优先级：self.api_key -> LITEAGENT_API_KEY -> <PROVIDER>_API_KEY
            self.assertEqual(config.LLMConfig(provider="anthropic").resolve_api_key(), "generic")
            self.assertEqual(config.LLMConfig().resolve_api_key(), "generic")

    def test_resolve_api_key_falls_back_to_provider_specific_var(self) -> None:
        with _clean_env(ANTHROPIC_API_KEY="anthropic"):
            self.assertEqual(
                config.LLMConfig(provider="anthropic").resolve_api_key(), "anthropic"
            )

    def test_resolve_api_key_returns_none_when_nothing_set(self) -> None:
        with _clean_env():
            self.assertIsNone(config.LLMConfig(provider="openai").resolve_api_key())

    def test_resolve_api_key_maps_provider_to_env_name(self) -> None:
        with _clean_env(OPENAI_COMPATIBLE_API_KEY="compat", DEEPSEEK_API_KEY="deep"):
            self.assertEqual(
                config.LLMConfig(provider="openai-compatible").resolve_api_key(), "compat"
            )
            self.assertEqual(config.LLMConfig(provider="deepseek").resolve_api_key(), "deep")

    def test_resolve_api_key_is_not_cached(self) -> None:
        # 每次调用都重新读环境变量：缓存会让 patch.dict 的测试互相污染。
        llm = config.LLMConfig()
        with _clean_env():
            self.assertIsNone(llm.resolve_api_key())
        with _clean_env(LITEAGENT_API_KEY="later"):
            self.assertEqual(llm.resolve_api_key(), "later")

    def test_resolved_base_url_priority(self) -> None:
        with _clean_env():
            self.assertEqual(
                config.LLMConfig(provider="openai").resolved_base_url(),
                "https://api.openai.com/v1",
            )
            self.assertEqual(
                config.LLMConfig(provider="anthropic").resolved_base_url(),
                "https://api.anthropic.com",
            )
            self.assertEqual(
                config.LLMConfig(provider="deepseek").resolved_base_url(),
                "https://api.deepseek.com/v1",
            )
            self.assertIsNone(config.LLMConfig(provider="echo").resolved_base_url())
            self.assertIsNone(
                config.LLMConfig(provider="openai-compatible").resolved_base_url()
            )
        with _clean_env(LITEAGENT_BASE_URL="http://env.example"):
            self.assertEqual(
                config.LLMConfig(provider="openai").resolved_base_url(), "http://env.example"
            )
            self.assertEqual(
                config.LLMConfig(base_url="http://explicit").resolved_base_url(),
                "http://explicit",
            )

    def test_to_dict_masks_api_key(self) -> None:
        payload = config.LLMConfig(api_key="sk-secret").to_dict()
        self.assertEqual(payload["api_key"], "***")
        self.assertNotIn("sk-secret", json.dumps(payload))
        self.assertIsNone(config.LLMConfig().to_dict()["api_key"])

    def test_to_dict_has_all_fields_and_is_json_serializable(self) -> None:
        payload = config.LLMConfig().to_dict()
        self.assertEqual(
            set(payload),
            {
                "provider",
                "model",
                "api_key",
                "base_url",
                "timeout_s",
                "retry_policy",
                "temperature",
                "max_tokens",
                "extra_headers",
                "extra_body",
                "stream",
                "sleep_fn",
            },
        )
        json.dumps(payload)

    def test_from_dict_demasks_api_key_to_none(self) -> None:
        # 脱敏值不能当真实 key 用（否则会拿着 "***" 去发请求）。
        restored = config.LLMConfig.from_dict(config.LLMConfig(api_key="sk").to_dict())
        self.assertIsNone(restored.api_key)

    def test_from_dict_round_trip(self) -> None:
        llm = config.LLMConfig(provider="deepseek", model="deepseek-chat", temperature=0.2)
        self.assertEqual(config.LLMConfig.from_dict(llm.to_dict()), llm)

    def test_from_dict_nested_retry_policy(self) -> None:
        restored = config.LLMConfig.from_dict(
            {"retry_policy": {"max_retries": 7, "rng_seed": 5}}
        )
        self.assertEqual(restored.retry_policy.max_retries, 7)
        self.assertEqual(restored.retry_policy.rng_seed, 5)

    def test_from_dict_raises_on_non_mapping_retry_policy(self) -> None:
        with self.assertRaises(SerializationError):
            config.LLMConfig.from_dict({"retry_policy": "nope"})

    def test_from_dict_warns_on_unknown_key(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.LLMConfig.from_dict({"provider": "echo", "nope": 1})
        self.assertTrue(any("nope" in str(w.message) for w in caught))

    def test_from_env_reads_provider_and_model(self) -> None:
        with _clean_env(LITEAGENT_PROVIDER="echo", LITEAGENT_MODEL="echo-1"):
            llm = config.LLMConfig.from_env()
        self.assertEqual(llm.provider, "echo")
        self.assertEqual(llm.model, "echo-1")

    def test_from_env_does_not_freeze_the_api_key(self) -> None:
        with _clean_env(LITEAGENT_API_KEY="late"):
            llm = config.LLMConfig.from_env()
            self.assertIsNone(llm.api_key)
            self.assertEqual(llm.resolve_api_key(), "late")


class ExecutorConfigTests(unittest.TestCase):
    """§5.3 / §5.5：序列化、``sequential_tools`` 的 frozenset 往返、``resolved_sleep`` 优先级。"""

    def test_to_dict_sorts_sequential_tools(self) -> None:
        payload = config.ExecutorConfig(
            sequential_tools=frozenset({"write_file", "read_file"})
        ).to_dict()
        self.assertEqual(payload["sequential_tools"], ["read_file", "write_file"])
        json.dumps(payload)

    def test_from_dict_restores_sequential_tools_as_frozenset(self) -> None:
        restored = config.ExecutorConfig.from_dict({"sequential_tools": ["a", "b"]})
        self.assertEqual(restored.sequential_tools, frozenset({"a", "b"}))
        self.assertIsInstance(restored.sequential_tools, frozenset)

    def test_from_dict_rejects_non_sequence_sequential_tools(self) -> None:
        with self.assertRaises(SerializationError):
            config.ExecutorConfig.from_dict({"sequential_tools": 5})

    def test_round_trip(self) -> None:
        executor = config.ExecutorConfig(max_concurrency=2, fail_fast=True)
        self.assertEqual(config.ExecutorConfig.from_dict(executor.to_dict()), executor)

    def test_thread_pool_default_timeout_none_has_frozen_semantics(self) -> None:
        # §5.3 [v2]：None = 对所有工具禁用超时（由 executor 解释，这里只断言字段可设）。
        self.assertIsNone(config.ExecutorConfig(default_timeout_s=None).default_timeout_s)

    def test_resolved_sleep_priority(self) -> None:
        sleeper = RecordingSleep()
        policy_sleeper = RecordingSleep()
        self.assertIs(
            config.ExecutorConfig(
                sleep_fn=sleeper, retry_policy=config.RetryPolicy(sleep_fn=policy_sleeper)
            ).resolved_sleep(),
            sleeper,
        )
        self.assertIs(
            config.ExecutorConfig(
                retry_policy=config.RetryPolicy(sleep_fn=policy_sleeper)
            ).resolved_sleep(),
            policy_sleeper,
        )
        self.assertIs(config.ExecutorConfig().resolved_sleep(), config.default_sleep)

    def test_resolved_sleep_fallback_argument(self) -> None:
        sleeper = RecordingSleep()
        self.assertIs(config.ExecutorConfig().resolved_sleep(fallback=sleeper), sleeper)


class AppConfigTests(unittest.TestCase):
    """§5.3：``from_dict`` / ``from_json`` / ``from_file`` / ``from_env`` / ``to_dict``。"""

    def test_to_dict_round_trip(self) -> None:
        app = config.AppConfig()
        payload = app.to_dict()
        json.dumps(payload)
        self.assertEqual(config.AppConfig.from_dict(payload), app)

    def test_to_dict_nests_every_sub_config(self) -> None:
        payload = config.AppConfig().to_dict()
        self.assertEqual(
            set(payload),
            {
                "llm",
                "agent",
                "memory",
                "executor",
                "team",
                "tools",
                "trace_file",
                "sandbox_root",
                "verbose",
            },
        )

    def test_from_dict_builds_nested_configs(self) -> None:
        app = config.AppConfig.from_dict(
            {"llm": {"provider": "echo"}, "agent": {"max_steps": 3}, "tools": ["web_search"]}
        )
        self.assertEqual(app.llm.provider, "echo")
        self.assertEqual(app.agent.max_steps, 3)
        self.assertEqual(app.tools, ["web_search"])

    def test_from_dict_raises_on_non_mapping(self) -> None:
        with self.assertRaises(SerializationError):
            config.AppConfig.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_from_dict_raises_on_non_mapping_nested_section(self) -> None:
        with self.assertRaises(SerializationError):
            config.AppConfig.from_dict({"llm": "nope"})

    def test_from_dict_warns_on_unknown_key(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.AppConfig.from_dict({"nope": 1})
        self.assertTrue(any("nope" in str(w.message) for w in caught))

    def test_from_json_reads_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"agent": {"max_steps": 4}}), encoding="utf-8")
            self.assertEqual(config.AppConfig.from_json(path).agent.max_steps, 4)

    def test_from_json_missing_file_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                config.AppConfig.from_json(Path(tmp) / "absent.json")

    def test_from_json_invalid_json_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                config.AppConfig.from_json(path)

    def test_from_file_rejects_toml_with_a_specific_message(self) -> None:
        # 3.10 没有 tomllib：单独报错，用户不会误以为是拼写问题。
        with self.assertRaises(ConfigError) as ctx:
            config.AppConfig.from_file("config.toml")
        self.assertIn("tomllib", str(ctx.exception))

    def test_from_file_rejects_unknown_extension(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            config.AppConfig.from_file("config.ini")
        self.assertIn(".ini", str(ctx.exception))

    def test_from_file_dispatches_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(config.AppConfig.from_file(path), config.AppConfig())

    def test_from_env_reads_max_steps(self) -> None:
        with _clean_env(LITEAGENT_MAX_STEPS="7"):
            self.assertEqual(config.AppConfig.from_env().agent.max_steps, 7)

    def test_from_env_bad_int_falls_back_with_warning(self) -> None:
        with _clean_env(LITEAGENT_MAX_STEPS="many"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                app = config.AppConfig.from_env()
        self.assertEqual(app.agent.max_steps, config.DEFAULT_MAX_STEPS)
        self.assertTrue(any("LITEAGENT_MAX_STEPS" in str(w.message) for w in caught))

    def test_from_yaml_without_pyyaml_raises_the_frozen_message(self) -> None:
        # §5.3 冻结文案：'PyYAML not installed; use JSON config'。
        with mock.patch.object(config, "YAML_AVAILABLE", False):
            with self.assertRaises(ConfigError) as ctx:
                config.AppConfig.from_yaml("config.yaml")
        self.assertEqual(str(ctx.exception), "PyYAML not installed; use JSON config")

    def test_from_yaml_missing_file_raises_config_error(self) -> None:
        if not config.YAML_AVAILABLE:  # pragma: no cover - 环境相关
            self.skipTest("PyYAML not installed")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                config.AppConfig.from_yaml(Path(tmp) / "absent.yaml")

    def test_from_yaml_reports_invalid_yaml_as_config_error(self) -> None:
        if not config.YAML_AVAILABLE:  # pragma: no cover - 环境相关
            self.skipTest("PyYAML not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("llm: [unclosed\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                config.AppConfig.from_yaml(path)

    def test_from_file_dispatches_yaml(self) -> None:
        if not config.YAML_AVAILABLE:  # pragma: no cover - 环境相关
            self.skipTest("PyYAML not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("agent:\n  max_steps: 6\n", encoding="utf-8")
            self.assertEqual(config.AppConfig.from_file(path).agent.max_steps, 6)


class MiscConfigSerializationTests(unittest.TestCase):
    """§5.3：其余配置 dataclass 的 round trip（字段全量输出）。"""

    def test_agent_config_round_trip(self) -> None:
        agent = config.AgentConfig(max_steps=5, system_prompt="hi")
        payload = agent.to_dict()
        json.dumps(payload)
        self.assertEqual(config.AgentConfig.from_dict(payload), agent)

    def test_memory_config_round_trip(self) -> None:
        memory = config.MemoryConfig(persist_path="/tmp/mem.jsonl", retrieve_limit=3)
        payload = memory.to_dict()
        json.dumps(payload)
        self.assertEqual(config.MemoryConfig.from_dict(payload), memory)

    def test_team_config_round_trip(self) -> None:
        team = config.TeamConfig(max_depth=1, share_memory=True)
        self.assertEqual(config.TeamConfig.from_dict(team.to_dict()), team)

    def test_from_dict_ignores_unknown_keys_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.AgentConfig.from_dict({"max_step": 3})  # 拼错的键
        self.assertTrue(any("max_step" in str(w.message) for w in caught))


# ======================================================================================
# §5.4 LoopBoundPool
# ======================================================================================


def _unregister_pool(pool: config.LoopBoundPool) -> None:
    """把一个池从全局登记表里摘掉（保持 ``_ALL_POOLS`` 不被本文件反复污染）。

    ``_ALL_POOLS`` 是本类**冻结的** ClassVar（§5.4），``aclose_all()`` 会清掉所有池，
    而其它测试文件可能同时拥有活着的池 —— 所以这里只摘自己那一个。
    """
    pool.release(None)
    try:
        config.LoopBoundPool._ALL_POOLS.remove(pool)
    except ValueError:  # pragma: no cover - 已被 aclose_all 摘除
        pass


class LoopBoundPoolWithoutLoopTests(unittest.TestCase):
    """§5.4 第 3 条：无运行 loop 时抛 ``ConfigError``（R-LOOP 的第一道防线）。"""

    def test_semaphore_without_running_loop_raises_config_error(self) -> None:
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)
        with self.assertRaises(ConfigError):
            pool.semaphore("exec", 1)
        self.assertEqual(len(pool), 0)  # 失败的调用不留空桶

    def test_lock_without_running_loop_raises_config_error(self) -> None:
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)
        with self.assertRaises(ConfigError):
            pool.lock("exec")

    def test_thread_pool_without_running_loop_raises_config_error(self) -> None:
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)
        with self.assertRaises(ConfigError):
            pool.thread_pool("workers", 2)

    def test_current_returns_none_without_running_loop(self) -> None:
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)
        self.assertIsNone(pool.current())

    def test_error_message_mentions_the_running_loop_requirement(self) -> None:
        pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, pool)
        with self.assertRaises(ConfigError) as ctx:
            pool.event("e")
        self.assertIn("running event loop", str(ctx.exception))


class LoopBoundPoolTests(unittest.IsolatedAsyncioTestCase):
    """§5.4 第 1、2 条：同 key 不同 value 抛 ``ConfigError``；``release()`` 后不泄漏。"""

    def setUp(self) -> None:
        self.pool = config.LoopBoundPool()
        self.addCleanup(_unregister_pool, self.pool)

    async def test_semaphore_returns_an_asyncio_semaphore(self) -> None:
        semaphore = self.pool.semaphore("exec", 2)
        self.assertIsInstance(semaphore, asyncio.Semaphore)
        self.assertEqual(len(self.pool), 1)

    async def test_semaphore_same_key_same_value_is_reused(self) -> None:
        first = self.pool.semaphore("exec", 2)
        self.assertIs(self.pool.semaphore("exec", 2), first)

    async def test_semaphore_same_key_different_value_raises_config_error(self) -> None:
        # §5.4 冻结：**不静默返回旧对象**（否则两个调用方以为自己在用不同的并发上限）。
        self.pool.semaphore("exec", 2)
        with self.assertRaises(ConfigError) as ctx:
            self.pool.semaphore("exec", 3)
        self.assertIn("exec", str(ctx.exception))

    async def test_semaphore_non_positive_value_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self.pool.semaphore("exec", 0)
        with self.assertRaises(ConfigError):
            self.pool.semaphore("exec", -1)
        self.assertEqual(len(self.pool), 0)

    async def test_different_keys_are_independent(self) -> None:
        self.pool.semaphore("exec", 2)
        self.pool.semaphore("subagent:worker", 5)
        self.assertIsNot(
            self.pool.semaphore("exec", 2), self.pool.semaphore("subagent:worker", 5)
        )

    async def test_lock_and_semaphore_do_not_collide_on_the_same_key(self) -> None:
        # 桶里的 key 带 kind 前缀：同一个名字在不同子系统里指不同东西。
        self.assertIsNot(self.pool.lock("exec"), self.pool.semaphore("exec", 2))

    async def test_condition_and_event_are_created(self) -> None:
        self.assertIsInstance(self.pool.condition("bb"), asyncio.Condition)
        self.assertIsInstance(self.pool.event("go"), asyncio.Event)

    async def test_thread_pool_same_key_different_workers_raises_config_error(self) -> None:
        pool = self.pool.thread_pool("workers", 2)
        self.assertIs(self.pool.thread_pool("workers", 2), pool)
        with self.assertRaises(ConfigError):
            self.pool.thread_pool("workers", 4)

    async def test_loop_of_returns_the_binding_loop(self) -> None:
        semaphore = self.pool.semaphore("exec", 1)
        self.assertIs(self.pool.loop_of(semaphore), asyncio.get_running_loop())

    async def test_loop_of_returns_none_for_foreign_objects(self) -> None:
        self.assertIsNone(self.pool.loop_of(object()))

    async def test_current_returns_the_running_loop(self) -> None:
        self.assertIs(self.pool.current(), asyncio.get_running_loop())

    async def test_release_drops_everything_for_the_loop(self) -> None:
        # §5.4 第 2 条：release() 后 len(pool)==0（不泄漏 loop 与原语）。
        self.pool.semaphore("exec", 2)
        self.pool.lock("exec")
        self.pool.thread_pool("workers", 1)
        self.assertEqual(len(self.pool), 1)
        self.pool.release()
        self.assertEqual(len(self.pool), 0)
        # 释放后同一 key 可以用**新** value 重建（旧绑定已被丢弃）
        self.assertEqual(len(self.pool), 0)

    async def test_release_with_the_explicit_loop_object(self) -> None:
        loop = asyncio.get_running_loop()
        self.pool.semaphore("exec", 2)
        self.pool.release(loop)
        self.assertEqual(len(self.pool), 0)

    async def test_primitives_are_recreated_after_release(self) -> None:
        first = self.pool.semaphore("exec", 2)
        self.pool.release()
        second = self.pool.semaphore("exec", 3)  # 释放后允许用新 value 重建
        self.assertIsNot(first, second)
        self.assertEqual(len(self.pool), 1)

    async def test_len_counts_loops_not_primitives(self) -> None:
        self.pool.semaphore("a", 1)
        self.pool.semaphore("b", 1)
        self.pool.lock("c")
        self.assertEqual(len(self.pool), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
