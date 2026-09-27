from __future__ import annotations

"""``BaseLLMClient._with_retry`` 的重试语义（§6.3 / §6.4 / §3.4）。

覆盖 §12 对 ``test_llm_retry.py`` 冻结的覆盖点：

1. 429 重试后成功；
2. ``Retry-After`` 被尊重（**用 ``tests.helpers.RecordingSleep`` 断言 delays，不真睡**）；
3. 重试耗尽抛 ``LLMRateLimitError``；
4. 不可重试错误只尝试 1 次；
5. ``LLMTimeoutError`` **不重试**（``retry_on_timeout=False``）。

**没有任何真实 sleep**：延迟全部落在 ``RecordingSleep.delays`` 上（§5.5 规则 1）。
"""

import asyncio
import json
import unittest

from liteagent.config import RetryPolicy, compute_backoff
from liteagent.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from liteagent.llm.message import Message
from liteagent.llm.providers import OpenAIChatClient
from liteagent.llm.transport import HTTPResponse
from liteagent.config import LLMConfig
from tests.helpers import FakeTransport, RecordingSleep

_OK_BODY = json.dumps(
    {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
)


def _rate_limited(retry_after: float | None = None) -> LLMRateLimitError:
    """一条"服务端限流"的响应。

    **为什么排队异常实例而不是 ``HTTPResponse(status_code=429)``**：``Transport`` 的
    冻结契约是"失败必须以 ``LiteAgentError`` 子类抛出"（§6.2），映射由各 transport 的
    ``map_http_error`` 完成；``FakeTransport`` 只是回放队列，不做映射。所以 429 在
    LLM 层的正确表达就是 ``LLMRateLimitError`` 实例（``FakeTransport`` 的 docstring
    也把"抛异常实例"列为错误路径的用法）。
    """
    if retry_after is None:
        return LLMRateLimitError(status_code=429, message="slow down")
    return LLMRateLimitError(status_code=429, retry_after_s=retry_after, message="slow down")


def _client(
    responses,
    *,
    max_retries: int = 2,
    jitter: float = 0.0,
    base_s: float = 1.0,
    max_s: float = 60.0,
    rng_seed: int | None = None,
    on_event=None,
    client_type=OpenAIChatClient,
):
    """一个带 ``RecordingSleep`` 的 OpenAI client（无网络、无真实等待）。"""
    sleep = RecordingSleep()
    policy = RetryPolicy(
        max_retries=max_retries,
        backoff_base_s=base_s,
        backoff_max_s=max_s,
        jitter=jitter,
        rng_seed=rng_seed,
    )
    config = LLMConfig(
        provider="openai", api_key="sk-test", retry_policy=policy, sleep_fn=sleep
    )
    transport = FakeTransport(list(responses))
    return client_type(config, transport=transport, on_event=on_event), transport, sleep


class RateLimitRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_429_then_success_retries_once(self) -> None:
        client, transport, sleep = _client([_rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)])
        response = await client.achat([Message.user("hi")])
        self.assertEqual("ok", response.content)
        self.assertEqual(2, len(transport.requests))
        # jitter=0 -> delay = min(max_s, base_s * 2**0) = 1.0
        self.assertEqual([1.0], sleep.delays)

    async def test_retry_after_header_is_respected(self) -> None:
        client, transport, sleep = _client([_rate_limited(5.0), HTTPResponse(status_code=200, text=_OK_BODY)])
        await client.achat([Message.user("hi")])
        # 退避 1.0 与服务端要求的 5.0 取更大者（不能密集打点）。
        self.assertEqual([5.0], sleep.delays)
        self.assertEqual(2, len(transport.requests))

    async def test_backoff_grows_across_attempts(self) -> None:
        client, transport, sleep = _client(
            [_rate_limited(), _rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)],
            max_retries=3,
        )
        await client.achat([Message.user("hi")])
        self.assertEqual([1.0, 2.0], sleep.delays)
        self.assertEqual(3, len(transport.requests))

    async def test_delays_match_compute_backoff_exactly(self) -> None:
        policy = RetryPolicy(max_retries=3, backoff_base_s=1.0, backoff_max_s=60.0, jitter=0.0)
        client, _, sleep = _client(
            [_rate_limited(), _rate_limited(), _rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)],
            max_retries=3,
        )
        await client.achat([Message.user("hi")])
        self.assertEqual(
            [policy.delay_for(i) for i in range(3)],
            [
                compute_backoff(i, base_s=policy.backoff_base_s, max_s=policy.backoff_max_s, jitter=0.0)
                for i in range(3)
            ],
        )
        self.assertEqual([policy.delay_for(i) for i in range(3)], sleep.delays)

    async def test_exhausted_retries_raise_rate_limit_error(self) -> None:
        client, transport, sleep = _client([_rate_limited(), _rate_limited(), _rate_limited()])
        with self.assertRaises(LLMRateLimitError) as ctx:
            await client.achat([Message.user("hi")])
        self.assertEqual(429, ctx.exception.status_code)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(3, len(transport.requests))  # 1 + max_retries(2)
        self.assertEqual([1.0, 2.0], sleep.delays)

    async def test_max_retries_zero_still_raises_after_one_attempt(self) -> None:
        client, transport, sleep = _client([_rate_limited()], max_retries=0)
        with self.assertRaises(LLMRateLimitError):
            await client.achat([Message.user("hi")])
        self.assertEqual(1, len(transport.requests))
        self.assertEqual([], sleep.delays)

    async def test_connection_error_is_retried_then_succeeds(self) -> None:
        client, transport, sleep = _client(
            [
                LLMConnectionError(status_code=503, message="unavailable"),
                HTTPResponse(status_code=200, text=_OK_BODY),
            ]
        )
        response = await client.achat([Message.user("hi")])
        self.assertEqual("ok", response.content)
        self.assertEqual([1.0], sleep.delays)
        self.assertEqual(2, len(transport.requests))


class NonRetryableTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_retryable_error_attempts_exactly_once(self) -> None:
        cases = [
            (LLMAuthError(401, message="denied"), LLMAuthError),
            (LLMBadRequestError(400, message="bad request"), LLMBadRequestError),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                client, transport, sleep = _client(
                    [error, HTTPResponse(status_code=200, text=_OK_BODY)]
                )
                with self.assertRaises(expected):
                    await client.achat([Message.user("hi")])
                self.assertEqual(1, len(transport.requests))
                self.assertEqual([], sleep.delays)

    async def test_timeout_is_not_retried_by_default(self) -> None:
        """§6.4：``HTTPChatClient.retry_on_timeout = False``（请求幂等性未知）。"""
        self.assertFalse(OpenAIChatClient.retry_on_timeout)
        client, transport, sleep = _client(
            [LLMTimeoutError(timeout_s=3.0), HTTPResponse(status_code=200, text=_OK_BODY)]
        )
        with self.assertRaises(LLMTimeoutError) as ctx:
            await client.achat([Message.user("hi")])
        self.assertEqual(3.0, ctx.exception.timeout_s)
        self.assertEqual(1, len(transport.requests))
        self.assertEqual([], sleep.delays)

    async def test_timeout_is_retried_when_the_client_opts_in(self) -> None:
        """反证上一条：把 `retry_on_timeout` 打开后同样的超时就**会**重试。"""

        class _RetryOnTimeout(OpenAIChatClient):
            retry_on_timeout = True

        client, transport, sleep = _client(
            [LLMTimeoutError(timeout_s=3.0), HTTPResponse(status_code=200, text=_OK_BODY)],
            client_type=_RetryOnTimeout,
        )
        response = await client.achat([Message.user("hi")])
        self.assertEqual("ok", response.content)
        self.assertEqual(2, len(transport.requests))
        self.assertEqual([1.0], sleep.delays)


class RetryEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_error_events_carry_the_retry_index(self) -> None:
        events: list[tuple[str, dict]] = []
        client, _, _ = _client(
            [_rate_limited(), _rate_limited(), _rate_limited()],
            on_event=lambda name, data: events.append((name, data)),
        )
        with self.assertRaises(LLMRateLimitError):
            await client.achat([Message.user("hi")], tools=[{"name": "add"}])

        self.assertEqual(
            [
                ("llm_request", 0),
                ("llm_error", 0),
                ("llm_request", 1),
                ("llm_error", 1),
                ("llm_request", 2),
                ("llm_error", 2),
            ],
            [(name, data["retry"]) for name, data in events],
        )
        first_request = events[0][1]
        self.assertEqual(1, first_request["messages_count"])
        self.assertEqual(1, first_request["tools_count"])
        self.assertEqual("gpt-4o-mini", first_request["model"])

        first_error = events[1][1]
        self.assertEqual("LLMRateLimitError", first_error["error_type"])
        self.assertTrue(first_error["message"])

    async def test_no_events_leak_dataclass_instances(self) -> None:
        events: list[tuple[str, dict]] = []
        client, _, _ = _client(
            [HTTPResponse(status_code=200, text=_OK_BODY)],
            on_event=lambda name, data: events.append((name, data)),
        )
        await client.achat([Message.user("hi")])
        for name, data in events:
            with self.subTest(event=name):
                json.dumps(data)  # 必须整体可 JSON 序列化（data 已过 to_jsonable）

    async def test_success_on_first_try_emits_request_then_response_only(self) -> None:
        events: list[tuple[str, dict]] = []
        client, _, _ = _client(
            [HTTPResponse(status_code=200, text=_OK_BODY)],
            on_event=lambda name, data: events.append((name, data)),
        )
        await client.achat([Message.user("hi")])
        self.assertEqual(["llm_request", "llm_response"], [name for name, _ in events])


class RetryPolicyReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_same_rng_seed_reproduces_the_delay_sequence(self) -> None:
        """§5.5 规则 3：rng 在构造时创建一次并复用整个生命周期。"""
        responses = [_rate_limited(), _rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)]
        first = _client(responses, max_retries=3, jitter=0.5, rng_seed=1234)
        second = _client(responses, max_retries=3, jitter=0.5, rng_seed=1234)
        await first[0].achat([Message.user("hi")])
        await second[0].achat([Message.user("hi")])
        self.assertEqual(first[2].delays, second[2].delays)
        self.assertEqual(2, len(first[2].delays))

    async def test_jittered_delays_stay_within_bounds(self) -> None:
        client, _, sleep = _client(
            [_rate_limited() for _ in range(4)] + [HTTPResponse(status_code=200, text=_OK_BODY)],
            max_retries=4,
            jitter=1.0,
            base_s=1.0,
            max_s=8.0,
            rng_seed=7,
        )
        await client.achat([Message.user("hi")])
        self.assertEqual(4, len(sleep.delays))
        for attempt, delay in enumerate(sleep.delays):
            with self.subTest(attempt=attempt):
                self.assertGreaterEqual(delay, 0.0)
                self.assertLessEqual(delay, min(8.0, 1.0 * 2 ** attempt))

    async def test_latency_and_call_accounting_are_unaffected_by_retries(self) -> None:
        client, transport, _ = _client([_rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)])
        response = await client.achat([Message.user("hi")])
        self.assertGreaterEqual(response.latency_ms, 0.0)
        # 两次请求的发往同一个 endpoint（重试用的是同一个 request 对象内容）。
        self.assertEqual(
            {request.url for request in transport.requests},
            {"https://api.openai.com/v1/chat/completions"},
        )


class SleepFnResolutionTests(unittest.IsolatedAsyncioTestCase):
    """§5.5 规则 2 的解析优先级：``LLMConfig.sleep_fn`` > ``RetryPolicy.sleep_fn`` > ``default_sleep``。"""

    def _config_with(self, policy_sleep, config_sleep):
        return LLMConfig(
            provider="openai",
            api_key="sk-test",
            retry_policy=RetryPolicy(
                max_retries=1, backoff_base_s=1.0, jitter=0.0, sleep_fn=policy_sleep
            ),
            sleep_fn=config_sleep,
        )

    def _client(self, config):
        return OpenAIChatClient(
            config, transport=FakeTransport([_rate_limited(), HTTPResponse(status_code=200, text=_OK_BODY)])
        )

    async def test_retry_policy_sleep_fn_is_used_when_the_config_has_none(self) -> None:
        policy_sleep = RecordingSleep()
        client = self._client(self._config_with(policy_sleep, None))
        self.assertIs(policy_sleep, client._sleep)
        await client.achat([Message.user("hi")])
        self.assertEqual([1.0], policy_sleep.delays)

    async def test_config_sleep_fn_wins_over_the_retry_policy_one(self) -> None:
        policy_sleep = RecordingSleep()
        config_sleep = RecordingSleep()
        client = self._client(self._config_with(policy_sleep, config_sleep))
        self.assertIs(config_sleep, client._sleep)
        await client.achat([Message.user("hi")])
        self.assertEqual([1.0], config_sleep.delays)
        self.assertEqual([], policy_sleep.delays)

    async def test_default_sleep_is_used_when_nothing_is_injected(self) -> None:
        from liteagent.config import default_sleep

        client = OpenAIChatClient(
            LLMConfig(provider="openai", api_key="k", retry_policy=RetryPolicy(max_retries=0))
        )
        self.assertIs(default_sleep, client._sleep)


class WithRetryCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_error_is_reraised_without_retrying(self) -> None:
        """M-5 / §6.3：宽 ``except`` 的首行必须是 ``except CancelledError: raise``。"""
        client, _, sleep = _client([], max_retries=5)
        attempts = {"n": 0}

        async def boom():
            attempts["n"] += 1
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await client._with_retry(boom, what="test")
        self.assertEqual(1, attempts["n"])
        self.assertEqual([], sleep.delays)

    async def test_retry_helper_returns_the_value_on_success(self) -> None:
        client, _, _ = _client([], max_retries=0)

        async def ok():
            return "value"

        self.assertEqual("value", await client._with_retry(ok, what="test"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
