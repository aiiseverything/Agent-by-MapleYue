from __future__ import annotations

"""tests/test_builtin_web.py —— 内置联网工具（§7.5 ``web.py``；§12 第 5133 行）。

§12 第 5133 行列出的覆盖点（逐条都有对应用例）：
  * ``HTMLTextExtractor`` 跳过 ``script`` / ``style``
  * ``html.unescape``（实体解码）
  * ``NullSearchBackend`` 返回 ``[]`` 且 ``web_search`` 返回**可读的失败串**
  * ``href`` 只留 http/https
  * ``allow_network=False``
  * **``make_web_tools(transport=FakeTransport(...))`` 的成功路径**

本文件**零真实网络**：所有传输都经 ``tests.helpers.FakeTransport``，所有搜索后端要么是
``NullSearchBackend``，要么是 ``DuckDuckGoHTMLBackend`` + 注入的 ``FakeTransport``
（这正是 §7.5 把 ``transport`` 写进构造签名的理由）。
"""

import os
import unittest
from contextlib import contextmanager
from typing import Iterator
from unittest import mock

from liteagent.config import DEFAULT_MAX_RETRIES
from liteagent.llm.transport import HTTPResponse
from liteagent.tools.base import Tool
from liteagent.tools.builtin.web import (
    DuckDuckGoHTMLBackend,
    HTMLTextExtractor,
    NullSearchBackend,
    SearchHit,
    default_search_backend,
    html_to_text,
    make_web_tools,
)
from tests.helpers import FakeSearchBackend, FakeTransport

NETWORK_DISABLED_MESSAGE = "network access is disabled"

#: 一张"DDG 结果页"的最小样本：两条结果，一条 uddg 跳转链接、一条直接链接。
_DDG_HTML = """\
<html><body>
<div class="result results_links">
  <h2><a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.invalid%2Fone">One Title</a></h2>
  <a class="result__snippet" href="https://example.invalid/one">First snippet</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://example.invalid/two">Two Title</a>
  <a class="result__snippet" href="https://example.invalid/two">Second snippet</a>
</div>
<a class="result__a" href="javascript:alert(1)">Not fetchable</a>
</body></html>
"""

#: 环境变量的"干净视图"：清掉所有会让被测模块改变分支的键。
_ENV_KEYS = (
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "LITEAGENT_ALLOW_NETWORK",
    "LITEAGENT_SANDBOX_ROOT",
    "LITEAGENT_ALLOW_SHELL",
)


@contextmanager
def clean_env(**overrides: str) -> Iterator[None]:
    """只保留系统必需的变量，再叠加 ``overrides``（避免被 ambient 环境改变分支）。"""
    base = {key: value for key, value in os.environ.items() if key not in _ENV_KEYS}
    base.update(overrides)
    with mock.patch.dict(os.environ, base, clear=True):
        yield


def _by_name(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def _html_response(body: str, content_type: str | None = "text/html; charset=utf-8") -> HTTPResponse:
    headers = {"content-type": content_type} if content_type else {}
    return HTTPResponse(status_code=200, headers=headers, text=body, url="https://x.invalid/")


class HTMLTextExtractorTests(unittest.TestCase):
    """纯 stdlib 的 HTML -> 文本。"""

    def test_skips_script_style_and_head(self) -> None:
        document = (
            "<html><head><title>Page Title</title>"
            "<style>body { color: red; }</style></head>"
            "<body><script>alert('injected')</script>"
            "<p>Visible paragraph</p></body></html>"
        )
        text = html_to_text(document)
        self.assertIn("Visible paragraph", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("color: red", text)
        self.assertNotIn("Page Title", text)

    def test_unescapes_entities(self) -> None:
        text = html_to_text("<p>a &amp; b &lt;tag&gt; &quot;q&quot; &#233;</p>")
        self.assertIn("a & b <tag>", text)
        self.assertIn('"q"', text)
        self.assertIn("é", text)

    def test_extractor_get_text_is_accessible_directly(self) -> None:
        parser = HTMLTextExtractor()
        parser.feed("<div>hello</div><script>bad()</script>")
        parser.close()
        self.assertEqual(parser.get_text(), "hello")

    def test_block_tags_produce_newlines_and_blank_lines_collapse(self) -> None:
        text = html_to_text("<div>a</div><div></div><div>b</div>")
        self.assertEqual(text, "a\n\nb")

    def test_links_keep_only_http_and_https(self) -> None:
        parser = HTMLTextExtractor()
        parser.feed(
            '<a href="javascript:void(0)">js</a>'
            '<a href="mailto:a@example.invalid">mail</a>'
            '<a href="/relative/path">rel</a>'
            '<a href="ftp://files.invalid/x">ftp</a>'
            '<a href="https://ok.example/page">ok</a>'
            '<a href="http://plain.example/page">plain</a>'
        )
        parser.close()
        self.assertEqual(
            parser.links, ["https://ok.example/page", "http://plain.example/page"]
        )

    def test_script_content_inside_link_is_skipped(self) -> None:
        parser = HTMLTextExtractor()
        parser.feed("<script><a href='https://evil.invalid/'>x</a></script>")
        parser.close()
        self.assertEqual(parser.links, [])
        self.assertEqual(parser.get_text(), "")

    def test_max_chars_truncates_head_and_tail(self) -> None:
        document = "<p>" + ("A" * 4000) + ("B" * 4000) + "</p>"
        text = html_to_text(document, max_chars=200)
        self.assertIn("truncated", text)
        self.assertLess(len(text), 1000)

    def test_malformed_html_does_not_raise(self) -> None:
        text = html_to_text("<p>unclosed <b>bold</p></div>")
        self.assertIn("unclosed", text)


class SearchBackendTests(unittest.TestCase):
    def test_null_backend_returns_empty_list(self) -> None:
        backend = NullSearchBackend()
        self.assertEqual(backend.name, "null")
        self.assertEqual(backend.search("anything"), [])

    def test_search_hit_to_dict(self) -> None:
        hit = SearchHit(title="t", url="https://u.invalid/", snippet="s")
        self.assertEqual(
            hit.to_dict(), {"title": "t", "url": "https://u.invalid/", "snippet": "s"}
        )

    def test_default_backend_priority(self) -> None:
        with clean_env(TAVILY_API_KEY="k"):
            self.assertEqual(default_search_backend().name, "tavily")
        with clean_env(SERPER_API_KEY="k"):
            self.assertEqual(default_search_backend().name, "serper")
        with clean_env(LITEAGENT_ALLOW_NETWORK="1"):
            self.assertEqual(default_search_backend().name, "duckduckgo")
        with clean_env(LITEAGENT_ALLOW_NETWORK="0"):
            # 没有 key + 联网被关掉 -> 零依赖兜底（并留下 warning 痕迹）。
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(default_search_backend().name, "null")


class WebSearchToolTests(unittest.TestCase):
    def test_null_backend_returns_readable_failure_string(self) -> None:
        tools = _by_name(make_web_tools(NullSearchBackend()))
        text = tools["web_search"].raw({"query": "liteagent"})
        self.assertIsInstance(text, str)
        self.assertIn("no search results", text)
        self.assertIn("'null'", text)
        self.assertIn("liteagent", text)
        self.assertNotEqual(text.strip(), "")

    def test_success_path_renders_title_url_snippet(self) -> None:
        backend = FakeSearchBackend()
        tools = _by_name(make_web_tools(backend))
        text = tools["web_search"].raw({"query": "ReAct"})
        self.assertEqual(backend.queries, ["ReAct"])
        self.assertIn("1. liteagent 框架文档", text)
        self.assertIn("https://example.invalid/liteagent/docs", text)
        self.assertIn("[3 result(s) from backend='fake']", text)

    def test_max_results_is_forwarded(self) -> None:
        tools = _by_name(make_web_tools(FakeSearchBackend()))
        text = tools["web_search"].raw({"query": "x", "max_results": 1})
        self.assertIn("[1 result(s) from backend='fake']", text)

    def test_backend_exception_becomes_readable_failure(self) -> None:
        class ExplodingBackend(NullSearchBackend):
            name = "exploding"

            def search(self, query, *, max_results=5, timeout_s=15.0):
                raise RuntimeError("backend down")

        tools = _by_name(make_web_tools(ExplodingBackend()))
        text = tools["web_search"].raw({"query": "x"})
        self.assertIn("search failed", text)
        self.assertIn("RuntimeError", text)
        self.assertIn("backend down", text)

    def test_empty_query_is_rejected(self) -> None:
        tools = _by_name(make_web_tools(FakeSearchBackend()))
        text = tools["web_search"].raw({"query": "   "})
        self.assertTrue(text.startswith("ERROR:"))

    def test_duckduckgo_backend_success_path_is_offline(self) -> None:
        transport = FakeTransport([_html_response(_DDG_HTML)])
        backend = DuckDuckGoHTMLBackend(transport=transport)
        tools = _by_name(make_web_tools(backend, transport=transport))
        text = tools["web_search"].raw({"query": "react"})
        self.assertIn("1. One Title", text)
        self.assertIn("https://example.invalid/one", text)
        self.assertIn("First snippet", text)
        self.assertIn("https://example.invalid/two", text)
        self.assertNotIn("javascript:", text)
        self.assertIn("from backend='duckduckgo'", text)
        # 请求确实发给了注入的传输层（而不是真网络）。
        self.assertEqual(len(transport.requests), 1)
        request = transport.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertTrue(request.url.startswith("https://html.duckduckgo.com/html/"))
        self.assertEqual(request.params.get("q"), "react")
        self.assertIn("User-Agent", request.headers)

    def test_duckduckgo_http_error_degrades_with_reason(self) -> None:
        transport = FakeTransport([HTTPResponse(status_code=503, text="nope")])
        backend = DuckDuckGoHTMLBackend(transport=transport)
        tools = _by_name(make_web_tools(backend))
        text = tools["web_search"].raw({"query": "x"})
        self.assertIn("no search results", text)
        self.assertIn("503", text)

    def test_duckduckgo_transport_error_degrades_with_reason(self) -> None:
        transport = FakeTransport([OSError("connection refused")])
        backend = DuckDuckGoHTMLBackend(transport=transport)
        tools = _by_name(make_web_tools(backend))
        text = tools["web_search"].raw({"query": "x"})
        self.assertIn("no search results", text)
        self.assertIn("OSError", text)

    def test_duckduckgo_max_results(self) -> None:
        transport = FakeTransport([_html_response(_DDG_HTML)])
        backend = DuckDuckGoHTMLBackend(transport=transport)
        hits = backend.search("q", max_results=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "One Title")


class FetchUrlToolTests(unittest.TestCase):
    def test_html_success_path(self) -> None:
        """§12 点名：`make_web_tools(transport=FakeTransport(...))` 的成功路径。"""
        transport = FakeTransport(
            [
                _html_response(
                    "<html><head><style>p{color:red}</style></head>"
                    "<body><h1>Title</h1><p>Body &amp; more</p>"
                    "<script>evil()</script></body></html>"
                )
            ]
        )
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/page"})
        self.assertTrue(text.startswith("# https://example.invalid/page\n"))
        self.assertIn("Title", text)
        self.assertIn("Body & more", text)
        self.assertNotIn("evil()", text)
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(transport.requests[0].url, "https://example.invalid/page")

    def test_plain_text_success_path(self) -> None:
        transport = FakeTransport(
            [_html_response("plain body text", content_type="text/plain")]
        )
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/robots.txt"})
        self.assertIn("plain body text", text)

    def test_html_sniffed_when_content_type_missing(self) -> None:
        transport = FakeTransport(
            [_html_response("<html><body><p>sniffed</p></body></html>", content_type=None)]
        )
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/"})
        self.assertIn("sniffed", text)
        self.assertNotIn("<p>", text)

    def test_non_text_content_type_is_rejected(self) -> None:
        transport = FakeTransport(
            [_html_response("%PDF-1.4 binary", content_type="application/pdf")]
        )
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/a.pdf"})
        self.assertTrue(text.startswith("ERROR:"))
        self.assertIn("unsupported content-type", text)

    def test_http_error_status_is_reported(self) -> None:
        transport = FakeTransport([HTTPResponse(status_code=404, text="not found")])
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/missing"})
        self.assertTrue(text.startswith("ERROR:"))
        self.assertIn("404", text)

    def test_transport_exception_is_reported(self) -> None:
        transport = FakeTransport([OSError("network unreachable")])
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/"})
        self.assertIn("fetch failed", text)
        self.assertIn("OSError", text)

    def test_empty_body_is_reported(self) -> None:
        transport = FakeTransport([_html_response("", content_type="text/plain")])
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw({"url": "https://example.invalid/empty"})
        self.assertIn("empty body", text)

    def test_non_http_scheme_is_rejected_without_any_request(self) -> None:
        transport = FakeTransport([_html_response("<p>x</p>")])
        tools = _by_name(make_web_tools(transport=transport))
        for url in ("file:///etc/passwd", "data:text/html,<b>x</b>", "ftp://h/x", "not a url"):
            with self.subTest(url=url):
                text = tools["fetch_url"].raw({"url": url})
                self.assertTrue(text.startswith("ERROR:"))
                self.assertIn("only http/https", text)
        self.assertEqual(transport.requests, [])

    def test_max_chars_truncates(self) -> None:
        transport = FakeTransport(
            [_html_response("<p>" + "A" * 5000 + "</p>", content_type="text/plain")]
        )
        tools = _by_name(make_web_tools(transport=transport))
        text = tools["fetch_url"].raw(
            {"url": "https://example.invalid/big", "max_chars": 200}
        )
        self.assertIn("truncated", text)
        self.assertLess(len(text), 2000)


class NetworkDisabledTests(unittest.TestCase):
    """`allow_network=False`：两个工具都返回冻结文案，但**仍然可见**。"""

    def test_both_tools_return_frozen_message(self) -> None:
        tools = _by_name(make_web_tools(allow_network=False))
        self.assertEqual(
            tools["web_search"].raw({"query": "x"}), NETWORK_DISABLED_MESSAGE
        )
        self.assertEqual(
            tools["fetch_url"].raw({"url": "https://example.invalid/"}),
            NETWORK_DISABLED_MESSAGE,
        )

    def test_tools_stay_visible(self) -> None:
        tools = make_web_tools(allow_network=False)
        self.assertEqual([tool.name for tool in tools], ["web_search", "fetch_url"])
        self.assertIn("query", tools[0].parameters["properties"])
        self.assertIn("url", tools[1].parameters["properties"])

    def test_no_request_is_made_when_disabled(self) -> None:
        transport = FakeTransport([_html_response("<p>x</p>")])
        tools = _by_name(make_web_tools(transport=transport, allow_network=False))
        tools["fetch_url"].raw({"url": "https://example.invalid/"})
        self.assertEqual(transport.requests, [])

    def test_disabled_never_constructs_a_default_backend(self) -> None:
        """`allow_network=False` 时**不得**调用 `default_search_backend()`。

        否则"关掉联网"的进程里仍会构造一个会发请求的后端（DDG / Tavily / Serper），
        开关就只是"调用前挡一下"，而不是"根本不具备联网能力"。
        """
        with mock.patch(
            "liteagent.tools.builtin.web.default_search_backend",
            return_value=NullSearchBackend(),
        ) as fake_default:
            make_web_tools(allow_network=False)
            fake_default.assert_not_called()
        with mock.patch(
            "liteagent.tools.builtin.web.default_search_backend",
            return_value=NullSearchBackend(),
        ) as fake_default:
            make_web_tools(allow_network=True)
            fake_default.assert_called_once()

    def test_env_var_can_disable_network(self) -> None:
        with clean_env(LITEAGENT_ALLOW_NETWORK="0"):
            tools = _by_name(make_web_tools())
            self.assertEqual(
                tools["web_search"].raw({"query": "x"}), NETWORK_DISABLED_MESSAGE
            )

    def test_env_var_enabling_network_keeps_tools_usable(self) -> None:
        with clean_env(LITEAGENT_ALLOW_NETWORK="1"):
            tools = _by_name(make_web_tools(FakeSearchBackend()))
            self.assertIn("fake", tools["web_search"].raw({"query": "x"}))


class MakeWebToolsMetadataTests(unittest.TestCase):
    def test_frozen_decorator_metadata(self) -> None:
        tools = _by_name(make_web_tools(NullSearchBackend()))
        for name in ("web_search", "fetch_url"):
            spec = tools[name].spec
            with self.subTest(tool=name):
                self.assertTrue(spec.idempotent)
                self.assertFalse(spec.dangerous)
                self.assertEqual(tuple(spec.tags), ("web",))
                # §7.5 冻结：网络类工具 retryable=True（在 ToolSpec 里由 max_retries 表达）。
                self.assertEqual(spec.max_retries, DEFAULT_MAX_RETRIES)

    def test_tool_order(self) -> None:
        self.assertEqual(
            [tool.name for tool in make_web_tools()], ["web_search", "fetch_url"]
        )

    def test_missing_required_argument_raises_validation_error(self) -> None:
        from liteagent.errors import ToolValidationError

        tools = _by_name(make_web_tools(NullSearchBackend()))
        with self.assertRaises(ToolValidationError):
            tools["fetch_url"].raw({})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
