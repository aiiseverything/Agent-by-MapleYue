from __future__ import annotations

"""``liteagent/llm/transport.py`` 的单元测试（§6.2）。

覆盖 §12 对 ``test_transport.py`` 冻结的四个覆盖点：

1. ``FakeTransport`` 记录请求 / 回放响应（§12.1 的冻结签名，本文件也顺带充当它的自测）；
2. ``map_http_error`` **全表**（401/403/429 带 ``Retry-After``/400/404/500/未知）；
3. ``wrap_transport_exception``（**含 ``asyncio.CancelledError`` 原样上抛**）；
4. ``UrllibTransport`` 只测请求构造与错误映射（用 ``unittest.mock.patch`` 替换 ``urlopen``）。

**零网络**：所有 ``urlopen`` 调用都走 ``mock.patch``；``FakeTransport`` 不碰 socket。
"""

import asyncio
import io
import json
import unittest
import urllib.error
from unittest import mock

from liteagent.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
)
from liteagent.config import LLMConfig, RetryPolicy
from liteagent.llm.message import Message
from liteagent.llm.providers import OpenAIChatClient
from liteagent.llm.transport import (
    HTTPRequest,
    HTTPResponse,
    Transport,
    UrllibTransport,
    default_transport,
    map_http_error,
    wrap_transport_exception,
)
from tests.helpers import FakeTransport


# --------------------------------------------------------------------------------------
# HTTP 数据类
# --------------------------------------------------------------------------------------


class HTTPDataTypesTests(unittest.TestCase):
    def test_ok_property_boundaries(self) -> None:
        self.assertTrue(HTTPResponse(status_code=200).ok)
        self.assertTrue(HTTPResponse(status_code=299).ok)
        self.assertFalse(HTTPResponse(status_code=199).ok)
        self.assertFalse(HTTPResponse(status_code=300).ok)
        self.assertFalse(HTTPResponse(status_code=500).ok)

    def test_json_parses_body(self) -> None:
        self.assertEqual({"a": 1}, HTTPResponse(status_code=200, text='{"a": 1}').json())

    def test_json_failure_raises_response_format_error_with_body(self) -> None:
        response = HTTPResponse(status_code=200, text="<html>nope", url="http://x")
        with self.assertRaises(LLMResponseFormatError) as ctx:
            response.json()
        self.assertEqual("<html>nope", ctx.exception.body)

    def test_request_defaults(self) -> None:
        request = HTTPRequest()
        self.assertEqual("POST", request.method)
        self.assertEqual({}, request.headers)
        self.assertIsNone(request.json_body)
        self.assertIsNone(request.data)
        self.assertGreater(request.timeout_s, 0)


# --------------------------------------------------------------------------------------
# FakeTransport（§12.1 冻结签名）
# --------------------------------------------------------------------------------------


class FakeTransportTests(unittest.TestCase):
    def test_records_every_request_in_order(self) -> None:
        transport = FakeTransport(
            [
                HTTPResponse(status_code=200, text="{}"),
                HTTPResponse(status_code=200, text="{}"),
            ]
        )
        first = HTTPRequest(url="http://a")
        second = HTTPRequest(url="http://b")
        transport.send(first)
        transport.send(second)
        self.assertEqual([first, second], transport.requests)

    def test_replays_queued_responses_fifo(self) -> None:
        transport = FakeTransport(
            [HTTPResponse(status_code=200, text="one"), HTTPResponse(status_code=201, text="two")]
        )
        self.assertEqual("one", transport.send(HTTPRequest()).text)
        self.assertEqual("two", transport.send(HTTPRequest()).text)

    def test_reraises_queued_exception_instance(self) -> None:
        error = LLMTimeoutError(timeout_s=1.5)
        transport = FakeTransport([error])
        with self.assertRaises(LLMTimeoutError) as ctx:
            transport.send(HTTPRequest(url="http://x"))
        self.assertIs(error, ctx.exception)
        # 先记录后回放：抛异常的请求也在 requests 里（复盘时序最需要它）。
        self.assertEqual(1, len(transport.requests))

    def test_queue_appends_after_construction(self) -> None:
        transport = FakeTransport()
        transport.queue(HTTPResponse(status_code=200, text="later"))
        self.assertEqual("later", transport.send(HTTPRequest()).text)

    def test_exhausted_queue_without_default_raises_assertion_error(self) -> None:
        transport = FakeTransport([HTTPResponse(status_code=200, text="{}")])
        transport.send(HTTPRequest())
        with self.assertRaises(AssertionError) as ctx:
            transport.send(HTTPRequest())
        self.assertIn("ran out of queued responses", str(ctx.exception))

    def test_explicit_default_replays_forever(self) -> None:
        """显式给 ``status``/``body`` 时队列耗尽后**永久回放**（这里用 2xx 避开错误映射）。"""
        transport = FakeTransport(status=201, body="created")
        for _ in range(3):
            response = transport.send(HTTPRequest(url="http://x"))
            self.assertEqual(201, response.status_code)
            self.assertEqual("created", response.text)

    def test_explicit_default_error_status_becomes_a_permanent_failure(self) -> None:
        """§6.2 契约：假传输层也要把 429 映射成 ``LLMRateLimitError``（不带 Retry-After）。"""
        transport = FakeTransport(status=429, body="slow down")
        for _ in range(3):
            with self.assertRaises(LLMRateLimitError) as ctx:
                transport.send(HTTPRequest(url="http://x"))
            self.assertEqual(429, ctx.exception.status_code)
            self.assertIsNone(ctx.exception.retry_after_s)
        # 失败请求同样"先记录后回放"，三次都在。
        self.assertEqual(3, len(transport.requests))

    def test_queued_error_status_is_mapped_and_raised(self) -> None:
        """旧行为（把 429 body 当 200 载荷回放）是错的：状态码必须变成异常。"""
        transport = FakeTransport([HTTPResponse(status_code=500, text="boom")])
        with self.assertRaises(LLMConnectionError) as ctx:
            transport.send(HTTPRequest(url="http://x"))
        self.assertEqual(500, ctx.exception.status_code)
        self.assertEqual(1, len(transport.requests))

    def test_queued_429_carries_retry_after_from_headers(self) -> None:
        """headers 必须透传给 ``map_http_error``，否则 ``Retry-After`` 解析不出来。"""
        transport = FakeTransport(
            [HTTPResponse(status_code=429, headers={"Retry-After": "2"}, text="slow down")]
        )
        with self.assertRaises(LLMRateLimitError) as ctx:
            transport.send(HTTPRequest(url="http://x"))
        self.assertEqual(2.0, ctx.exception.retry_after_s)
        self.assertTrue(ctx.exception.retryable)

    def test_queued_bad_request_and_auth_statuses_map_to_their_classes(self) -> None:
        cases = [(400, LLMBadRequestError), (401, LLMAuthError), (403, LLMAuthError)]
        for status, expected in cases:
            with self.subTest(status=status):
                transport = FakeTransport([HTTPResponse(status_code=status, text="nope")])
                with self.assertRaises(expected) as ctx:
                    transport.send(HTTPRequest(url="http://x"))
                self.assertEqual(status, ctx.exception.status_code)

    def test_is_a_transport_and_closes_quietly(self) -> None:
        transport = FakeTransport()
        self.assertIsInstance(transport, Transport)
        self.assertIsNone(transport.close())
        self.assertEqual("fake", transport.name)


# --------------------------------------------------------------------------------------
# map_http_error（冻结表）
# --------------------------------------------------------------------------------------


class MapHttpErrorTests(unittest.TestCase):
    def test_401_and_403_map_to_auth_error(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status):
                error = map_http_error(status, "denied", url="http://x")
                self.assertIsInstance(error, LLMAuthError)
                self.assertEqual(status, error.status_code)
                self.assertIn("http://x", str(error))

    def test_429_with_retry_after_header(self) -> None:
        error = map_http_error(429, "slow", headers={"Retry-After": "2"}, url="http://x")
        self.assertIsInstance(error, LLMRateLimitError)
        self.assertEqual(429, error.status_code)
        self.assertEqual(2.0, error.retry_after_s)
        self.assertTrue(error.retryable)

    def test_429_retry_after_lookup_is_case_insensitive_and_accepts_float(self) -> None:
        error = map_http_error(429, "slow", headers={"retry-after": " 1.25 "})
        self.assertEqual(1.25, error.retry_after_s)

    def test_429_without_retry_after_keeps_class_default(self) -> None:
        error = map_http_error(429, "slow")
        self.assertIsInstance(error, LLMRateLimitError)
        self.assertIsNone(error.retry_after_s)

    def test_429_with_http_date_retry_after_is_not_parsed(self) -> None:
        """冻结：只支持纯数字秒；HTTP-date 形态退回 None（好过猜一个错的等待时间）。"""
        error = map_http_error(429, "slow", headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
        self.assertIsInstance(error, LLMRateLimitError)
        self.assertIsNone(error.retry_after_s)

    def test_bad_request_table(self) -> None:
        for status in (400, 404, 413, 422):
            with self.subTest(status=status):
                error = map_http_error(status, "bad", url="http://x")
                self.assertIsInstance(error, LLMBadRequestError)
                self.assertEqual(status, error.status_code)
                self.assertEqual("bad", error.body)
                self.assertFalse(error.retryable)

    def test_other_4xx_maps_to_bad_request(self) -> None:
        error = map_http_error(418, "teapot")
        self.assertIsInstance(error, LLMBadRequestError)
        self.assertEqual(418, error.status_code)

    def test_connection_retryable_table(self) -> None:
        for status in (408, 409, 425, 500, 502, 503, 504):
            with self.subTest(status=status):
                error = map_http_error(status, "boom")
                self.assertIsInstance(error, LLMConnectionError)
                self.assertEqual(status, error.status_code)
                self.assertTrue(error.retryable)

    def test_unmapped_status_returns_base_llm_error(self) -> None:
        """表外的 3xx 不能凭空造语义：返回基类 ``LLMError`` 且 message 里带状态码。"""
        error = map_http_error(302, "moved")
        self.assertIs(type(error), LLMError)
        self.assertFalse(error.retryable)
        self.assertIn("302", str(error))

    def test_body_is_truncated_in_context_and_timeout_recorded(self) -> None:
        error = map_http_error(500, "x" * 5000, url="http://x", timeout_s=9.0)
        self.assertEqual("http://x", error.context["url"])
        self.assertEqual(9.0, error.context["timeout_s"])
        self.assertEqual(1000, len(error.context["body"]))
        self.assertNotIsInstance(error, LLMTimeoutError)


# --------------------------------------------------------------------------------------
# wrap_transport_exception
# --------------------------------------------------------------------------------------


class WrapTransportExceptionTests(unittest.TestCase):
    def test_timeout_like_exceptions_map_to_llm_timeout_error(self) -> None:
        for exc in (TimeoutError("slow"), asyncio.TimeoutError()):
            with self.subTest(exc=type(exc).__name__):
                error = wrap_transport_exception(exc, timeout_s=3.0)
                self.assertIsInstance(error, LLMTimeoutError)
                self.assertEqual(3.0, error.timeout_s)
                self.assertIs(exc, error.cause)

    def test_connection_like_exceptions_map_to_llm_connection_error(self) -> None:
        for exc in (
            urllib.error.URLError("dns"),
            ConnectionError("reset"),
            OSError("broken pipe"),
        ):
            with self.subTest(exc=type(exc).__name__):
                error = wrap_transport_exception(exc, timeout_s=None)
                self.assertIsInstance(error, LLMConnectionError)
                self.assertIsNone(error.status_code)
                self.assertTrue(error.retryable)

    def test_unknown_exception_maps_to_base_llm_error(self) -> None:
        error = wrap_transport_exception(ValueError("weird"), timeout_s=None)
        self.assertIs(type(error), LLMError)
        self.assertIn("ValueError", str(error))

    def test_cancelled_error_is_reraised_unchanged(self) -> None:
        """M-5 / §6.2 [v2 变更]：``CancelledError`` 不得被包装，否则取消传播被破坏。"""
        cancelled = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError) as ctx:
            wrap_transport_exception(cancelled, timeout_s=1.0)
        self.assertIs(cancelled, ctx.exception)
        # 再确认它不是被"包装后抛出"（CancelledError 不是 LiteAgentError 子类）。
        self.assertNotIsInstance(ctx.exception, LLMError)


# --------------------------------------------------------------------------------------
# UrllibTransport（只测请求构造与错误映射，urlopen 全程被 mock）
# --------------------------------------------------------------------------------------


class _FakeUrlopenResponse:
    """``urlopen`` 的返回值替身（只需要 status/headers/read/geturl/close）。"""

    def __init__(self, *, status: int = 200, body: bytes = b"{}", url: str = "http://x") -> None:
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body
        self._url = url
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


def _http_error(code: int, body: bytes = b"", headers=None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://x", code, "err", headers if headers is not None else {}, io.BytesIO(body)
    )


def _raise(exc: BaseException):
    """造一个"调用即抛``exc``"的 ``urlopen`` 替身（比 lambda 里的生成器技巧可读）。"""

    def urlopen(request, timeout=None):
        raise exc

    return urlopen


class UrllibTransportTests(unittest.TestCase):
    def _send_with(self, urlopen_side_effect, request: HTTPRequest) -> tuple[HTTPResponse, mock.Mock]:
        patcher = mock.patch("urllib.request.urlopen", side_effect=urlopen_side_effect)
        fake = patcher.start()
        self.addCleanup(patcher.stop)
        return UrllibTransport().send(request), fake

    def test_builds_a_post_request_with_json_body_and_timeout(self) -> None:
        captured = {}

        def urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeUrlopenResponse(body=b'{"ok": true}')

        response, _ = self._send_with(
            urlopen,
            HTTPRequest(
                url="https://api.example.com/v1/chat/completions",
                headers={"Authorization": "Bearer k"},
                json_body={"model": "gpt-4o-mini"},
                timeout_s=12.5,
            ),
        )
        request = captured["request"]
        self.assertEqual("https://api.example.com/v1/chat/completions", request.full_url)
        self.assertEqual("POST", request.get_method())
        self.assertEqual(json.dumps({"model": "gpt-4o-mini"}).encode("utf-8"), request.data)
        self.assertEqual("application/json", request.get_header("Content-type"))
        self.assertEqual("Bearer k", request.get_header("Authorization"))
        self.assertEqual(12.5, captured["timeout"])
        self.assertEqual(200, response.status_code)
        self.assertEqual('{"ok": true}', response.text)
        self.assertEqual({"Content-Type": "application/json"}, response.headers)

    def test_explicit_content_type_is_not_overwritten(self) -> None:
        captured = {}

        def urlopen(request, timeout=None):
            captured["request"] = request
            return _FakeUrlopenResponse()

        self._send_with(
            urlopen,
            HTTPRequest(
                url="http://x",
                headers={"Content-Type": "application/json; charset=utf-8"},
                json_body={"a": 1},
            ),
        )
        self.assertEqual(
            "application/json; charset=utf-8", captured["request"].get_header("Content-type")
        )

    def test_params_are_appended_to_the_query_string(self) -> None:
        captured = {}

        def urlopen(request, timeout=None):
            captured["request"] = request
            return _FakeUrlopenResponse()

        self._send_with(
            urlopen,
            HTTPRequest(url="http://x/v1/chat", params={"key": "a b", "v": "1"}),
        )
        self.assertEqual("http://x/v1/chat?key=a+b&v=1", captured["request"].full_url)

    def test_non_positive_timeout_is_sent_as_no_timeout(self) -> None:
        captured = {}

        def urlopen(request, timeout=None):
            captured["timeout"] = timeout
            return _FakeUrlopenResponse()

        self._send_with(urlopen, HTTPRequest(url="http://x", timeout_s=-1.0))
        self.assertIsNone(captured["timeout"])

    def test_http_error_is_mapped_to_llm_error(self) -> None:
        cases = [
            (401, {}, LLMAuthError),
            (404, {}, LLMBadRequestError),
            (500, {}, LLMConnectionError),
            (429, {"Retry-After": "4"}, LLMRateLimitError),
        ]
        for status, headers, expected in cases:
            with self.subTest(status=status):
                response = _http_error(status, b'{"error": "nope"}', headers)
                with self.assertRaises(expected) as ctx:
                    self._send_with(_raise(response), HTTPRequest(url="http://x"))
                self.assertIn("HTTP %d" % status, str(ctx.exception))

    def test_http_error_retry_after_header_reaches_the_exception(self) -> None:
        with self.assertRaises(LLMRateLimitError) as ctx:
            self._send_with(
                _raise(_http_error(429, b"slow", {"Retry-After": "7"})),
                HTTPRequest(url="http://x"),
            )
        self.assertEqual(7.0, ctx.exception.retry_after_s)

    def test_oserror_from_urlopen_is_wrapped_as_connection_error(self) -> None:
        with self.assertRaises(LLMConnectionError):
            self._send_with(_raise(OSError("connection reset")), HTTPRequest(url="http://x"))

    def test_cancelled_error_propagates_through_send(self) -> None:
        """``except Exception`` 抓不住 BaseException -> 取消原样上抛（M-5）。"""
        with self.assertRaises(asyncio.CancelledError):
            self._send_with(_raise(asyncio.CancelledError()), HTTPRequest(url="http://x"))


class DefaultTransportTests(unittest.TestCase):
    def test_default_transport_is_a_transport_with_a_name(self) -> None:
        transport = default_transport()
        self.assertIsInstance(transport, Transport)
        self.assertTrue(transport.name)

    def test_default_transport_is_cached(self) -> None:
        """``functools.cache``：两次取到的是同一个对象（进程级单例）。"""
        self.assertIs(default_transport(), default_transport())


class FakeTransportThroughAchatTests(unittest.IsolatedAsyncioTestCase):
    """``FakeTransport`` 造出的失败后端，经 ``OpenAIChatClient.achat`` 走完整链路。

    这一步才是"照着 docstring 写限流测试"的人真正会写的代码：错误必须一路
    冒泡成对应异常，而不是被当成 200 载荷送进 ``_parse_response`` 抛
    ``LLMResponseFormatError``。
    """

    @staticmethod
    def _client(transport: FakeTransport) -> OpenAIChatClient:
        # max_retries=0 -> 只尝试一次：429 是可重试的，否则这里会真的退避重试。
        return OpenAIChatClient(
            LLMConfig(
                provider="openai",
                api_key="sk-test",
                retry_policy=RetryPolicy(max_retries=0),
            ),
            transport=transport,
        )

    async def test_permanent_401_becomes_llm_auth_error(self) -> None:
        transport = FakeTransport(status=401, body='{"error": {"message": "invalid key"}}')
        with self.assertRaises(LLMAuthError) as ctx:
            await self._client(transport).achat([Message.user("hi")])
        self.assertEqual(401, ctx.exception.status_code)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(1, len(transport.requests))

    async def test_permanent_429_without_retry_after_becomes_rate_limit_error(self) -> None:
        transport = FakeTransport(status=429, body="slow down")
        with self.assertRaises(LLMRateLimitError) as ctx:
            await self._client(transport).achat([Message.user("hi")])
        self.assertEqual(429, ctx.exception.status_code)
        self.assertIsNone(ctx.exception.retry_after_s)

    async def test_queued_429_retry_after_reaches_the_rate_limit_error(self) -> None:
        transport = FakeTransport(
            [HTTPResponse(status_code=429, headers={"Retry-After": "3"}, text="slow down")]
        )
        with self.assertRaises(LLMRateLimitError) as ctx:
            await self._client(transport).achat([Message.user("hi")])
        self.assertEqual(3.0, ctx.exception.retry_after_s)
        self.assertTrue(ctx.exception.retryable)


class _RecorderTransport(Transport):
    """只记录 ``send`` 调用的最小 Transport 实现（验证 ``asend`` 的默认委托）。"""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[HTTPRequest] = []

    def send(self, request: HTTPRequest) -> HTTPResponse:
        self.seen.append(request)
        return HTTPResponse(status_code=200, text="ok")


class AsendDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_asend_calls_send_in_a_worker_thread(self) -> None:
        """``Transport.asend`` 的默认实现 = ``await asyncio.to_thread(self.send, request)``。"""
        transport = _RecorderTransport()
        request = HTTPRequest(url="http://x")
        response = await transport.asend(request)
        self.assertEqual(200, response.status_code)
        self.assertEqual([request], transport.seen)

    async def test_asend_maps_exceptions_like_the_sync_path(self) -> None:
        class _Boom(_RecorderTransport):
            def send(self, request: HTTPRequest) -> HTTPResponse:
                raise LLMConnectionError(status_code=None, message="down")

        with self.assertRaises(LLMConnectionError):
            await _Boom().asend(HTTPRequest(url="http://x"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
