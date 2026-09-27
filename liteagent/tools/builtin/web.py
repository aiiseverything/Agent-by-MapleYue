from __future__ import annotations

"""内置联网工具（规范 §7.5 的 ``web.py``）：``web_search`` / ``fetch_url``。

**可测性是这一节的设计主线**（面试时值得展开的一点）：联网工具的测试在**离线环境**
里怎么跑？答案是**注入 Transport**：

    make_web_tools(transport=FakeTransport([HTTPResponse(200, body="<html>...")]))

``SearchBackend`` 的每个实现都把 ``transport`` 作为构造参数（``DuckDuckGoHTMLBackend``
甚至允许改 ``endpoint``），于是"解析 duckduckgo HTML / tavily JSON"这两条**最容易写错**
的逻辑可以在完全离线的条件下被逐字段断言。生产路径则走 ``default_transport()``
（urllib -> requests -> httpx 依次降级，见 ``llm/transport.py``）。

另外三件事：

1. **纯 stdlib 的 HTML -> 文本**（:class:`HTMLTextExtractor`）：不引入 BeautifulSoup
   （环境里也没有），用 ``html.parser`` 实现"跳过 script/style/head + 块级标签补换行 +
   压缩连续空行 + 解码实体"。
2. **降级必须留痕**（红线 12）：搜索后端不可用、HTTP 非 2xx、响应不是 HTML，
   都返回**可读的**失败字符串（而不是空串）—— 模型据此才能换策略。
3. ``allow_network=False`` 时工具**仍然可见**，调用返回 ``'network access is disabled'``。
"""

import html
import os
import urllib.parse
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from liteagent.config import DEFAULT_MAX_RETRIES, parse_bool, truncate_head_tail
from liteagent.errors import ToolValidationError
from liteagent.tools.base import Tool, make_function_tool

if TYPE_CHECKING:  # pragma: no cover - 只为注解（避免新增跨包顶层 import 边）
    from liteagent.llm.transport import HTTPRequest, HTTPResponse, Transport

__all__ = [
    "HTMLTextExtractor",
    "NullSearchBackend",
    "SearchBackend",
    "SearchHit",
    "default_search_backend",
    "html_to_text",
    "make_web_tools",
]

# 冻结文案（§7.5）：未开启联网时两个工具的返回值都是它。
_NETWORK_DISABLED_MESSAGE = "network access is disabled"

_USER_AGENT = "Mozilla/5.0 (compatible; liteagent/0.1; +https://example.invalid/liteagent)"

# 只接受这两种 scheme。`file://` / `data:` / `javascript:` 一律过滤掉 ——
# `fetch_url` 是"模型给的 URL 直接发请求"，scheme 白名单是这里最便宜也最有效的闸门。
_ALLOWED_SCHEMES = ("http", "https")

_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_SERPER_ENDPOINT = "https://google.serper.dev/search"


def _is_fetchable_url(url: str) -> bool:
    """只放行 http/https（大小写不敏感）。"""
    try:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:  # pragma: no cover - 畸形 URL
        return False
    return scheme in _ALLOWED_SCHEMES


def _http_request(
    *,
    method: str,
    url: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    timeout_s: float = 15.0,
) -> "HTTPRequest":
    """构造 ``HTTPRequest``。

    函数体内 import ``liteagent.llm.transport``：§1.1 给 ``tools/builtin/*.py`` 列出的
    允许依赖是 ``errors/types/config/tools/*``，并没有列 ``llm/*``。用函数体内延迟 import
    既满足"顶层 import 边必须出现在白名单表里"的守门测试，又不必放弃复用 LLM 层那套
    已经写好错误映射的传输层（``map_http_error`` / 超时归一化 / 三种适配器降级）。
    """
    from liteagent.llm.transport import HTTPRequest

    return HTTPRequest(
        method=method,
        url=url,
        params=dict(params or {}),
        headers=dict(headers or {}),
        json_body=json_body,
        timeout_s=timeout_s,
    )


# --------------------------------------------------------------------------------------
# 搜索结果的数据结构
# --------------------------------------------------------------------------------------


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出（§2.2）。"""
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchBackend(ABC):
    """搜索后端抽象：子类只需实现同步 ``search``。"""

    name: str = "abstract"

    @abstractmethod
    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
    ) -> list[SearchHit]:
        """返回命中的搜索结果（失败时返回 ``[]``，**不抛异常**）。

        为什么把"失败"设计成空列表而不是异常：搜索是**尽力而为**的能力，
        三个后端（DDG 无 key / Tavily / Serper）可用性各不相同；把失败统一成
        "空结果 + 可读原因"，调用方（``_web_search``）就能用同一段文案回复模型。
        """
        raise NotImplementedError


class NullSearchBackend(SearchBackend):
    """零依赖兜底：始终返回 ``[]``，并让 ``web_search`` 返回可读的失败说明。"""

    name = "null"

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
    ) -> list[SearchHit]:
        return []


def _send(transport: "Transport", request: "HTTPRequest") -> "HTTPResponse":
    """薄封装 ``transport.send``，单独一层是为了让所有后端共用同一个调用点。"""
    return transport.send(request)


def _resolve_transport(transport: "Transport | None") -> "Transport":
    """注入优先，否则用 LLM 层的默认传输（urllib -> requests -> httpx 降级）。"""
    if transport is not None:
        return transport
    from liteagent.llm.transport import default_transport

    return default_transport()


# --------------------------------------------------------------------------------------
# DuckDuckGo（无 key，解析 HTML）
# --------------------------------------------------------------------------------------


def _clean_ddg_href(href: str) -> str:
    """还原 DuckDuckGo 的跳转链接（``//duckduckgo.com/l/?uddg=<urlencoded>``）。"""
    if "uddg=" not in href:
        return href
    query = urllib.parse.urlsplit(href).query
    for key, value in urllib.parse.parse_qsl(query):
        if key == "uddg" and value:
            return value
    return href


class _DDGResultParser(HTMLParser):
    """从 DuckDuckGo 的 HTML 结果页里抽取 (title, url, snippet)。

    用"class 里含 ``result__a`` / ``result__snippet``"这种**宽松匹配**而不是精确相等：
    DDG 的 class 名在不同版本里加过后缀（``result__a js-result-title-link``），
    精确相等会让整个解析在某次改版后静默返回 0 条结果。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchHit] = []
        self._current: SearchHit | None = None
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs}
        classes = attributes.get("class", "")
        if tag == "a" and "result__a" in classes:
            href = _clean_ddg_href(attributes.get("href", ""))
            self._current = SearchHit(title="", url=href)
            self._in_title = True
        elif "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current is not None:
                self.results.append(self._current)
                self._current = None
        if self._in_snippet and tag in ("a", "div", "span"):
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title and self._current is not None:
            self._current.title = f"{self._current.title} {text}".strip()
        elif self._in_snippet and self._current is not None:
            self._current.snippet = f"{self._current.snippet} {text}".strip()
        elif self._in_snippet and self.results:
            # snippet 有时出现在结果锚点**闭合之后**（DDG 的两种排版），
            # 补写到最近一条结果上，比丢掉它更接近用户预期。
            last = self.results[-1]
            last.snippet = f"{last.snippet} {text}".strip()


class DuckDuckGoHTMLBackend(SearchBackend):
    """用 Transport 请求 ``https://html.duckduckgo.com/html/?q=...``，纯 stdlib 解析结果。

    [v2 冻结] ``__init__(self, transport=None, *, endpoint=...)`` —— transport 可注入
    是离线测试成功路径的**唯一**手段。
    """

    name = "duckduckgo"

    def __init__(
        self,
        transport: "Transport | None" = None,
        *,
        endpoint: str = _DDG_ENDPOINT,
    ) -> None:
        self.transport = transport
        self.endpoint = endpoint

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
    ) -> list[SearchHit]:
        try:
            transport = _resolve_transport(self.transport)
            request = _http_request(
                method="GET",
                url=self.endpoint,
                params={"q": query},
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
                timeout_s=timeout_s,
            )
            response = _send(transport, request)
        except Exception as exc:  # noqa: BLE001 - 网络失败一律降级为空结果
            # 降级必须留痕（红线 12）：把原因返回给调用方渲染进失败文案。
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        if not response.ok:
            self.last_error = f"HTTP {response.status_code} from {self.endpoint}"
            return []
        self.last_error = ""
        return _parse_ddg_html(response.text, max_results)


def _parse_ddg_html(body: str, max_results: int) -> list[SearchHit]:
    """解析 DDG 结果页；主解析器没抽到就退化为"所有 http(s) 外链"。"""
    parser = _DDGResultParser()
    try:
        parser.feed(body)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - 畸形 HTML 不能连累整个搜索
        warnings.warn(f"duckduckgo HTML parse failed: {exc!r}", RuntimeWarning, stacklevel=2)
    hits = [hit for hit in parser.results if hit.url and _is_fetchable_url(hit.url)]
    return hits[:max_results]


# --------------------------------------------------------------------------------------
# Tavily / Serper（都需要 API key，走 JSON API）
# --------------------------------------------------------------------------------------


class TavilyBackend(SearchBackend):
    """Tavily 搜索 API（需要 ``TAVILY_API_KEY``）。

    [v2 冻结] ``__init__(self, api_key=None, *, transport=None)``。
    """

    name = "tavily"

    def __init__(self, api_key: str | None = None, *, transport: "Transport | None" = None) -> None:
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            from liteagent.errors import ConfigError

            raise ConfigError(
                "TavilyBackend requires an api key; pass api_key=... or set TAVILY_API_KEY"
            )
        self.api_key = key
        self.transport = transport
        self.endpoint = _TAVILY_ENDPOINT

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
    ) -> list[SearchHit]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            transport = _resolve_transport(self.transport)
            response = _send(
                transport,
                _http_request(
                    method="POST",
                    url=self.endpoint,
                    headers={"User-Agent": _USER_AGENT},
                    json_body=payload,
                    timeout_s=timeout_s,
                ),
            )
            if not response.ok:
                self.last_error = f"HTTP {response.status_code} from {self.endpoint}"
                return []
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        self.last_error = ""
        results = data.get("results") if isinstance(data, dict) else None
        hits: list[SearchHit] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", ""))
            if not _is_fetchable_url(url):
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title", "")),
                    url=url,
                    snippet=str(item.get("content", "") or item.get("snippet", "")),
                )
            )
        return hits[:max_results]


class SerperBackend(SearchBackend):
    """Serper（Google）搜索 API（需要 ``SERPER_API_KEY``）。签名同 ``TavilyBackend``。"""

    name = "serper"

    def __init__(self, api_key: str | None = None, *, transport: "Transport | None" = None) -> None:
        key = api_key or os.environ.get("SERPER_API_KEY")
        if not key:
            from liteagent.errors import ConfigError

            raise ConfigError(
                "SerperBackend requires an api key; pass api_key=... or set SERPER_API_KEY"
            )
        self.api_key = key
        self.transport = transport
        self.endpoint = _SERPER_ENDPOINT

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 15.0
    ) -> list[SearchHit]:
        try:
            transport = _resolve_transport(self.transport)
            response = _send(
                transport,
                _http_request(
                    method="POST",
                    url=self.endpoint,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json_body={"q": query, "num": max_results},
                    timeout_s=timeout_s,
                ),
            )
            if not response.ok:
                self.last_error = f"HTTP {response.status_code} from {self.endpoint}"
                return []
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        self.last_error = ""
        organic = data.get("organic") if isinstance(data, dict) else None
        hits: list[SearchHit] = []
        for item in organic or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("link", "") or item.get("url", ""))
            if not _is_fetchable_url(url):
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title", "")),
                    url=url,
                    snippet=str(item.get("snippet", "")),
                )
            )
        return hits[:max_results]


def default_search_backend() -> SearchBackend:
    """优先级：``TAVILY_API_KEY`` -> ``SERPER_API_KEY`` -> duckduckgo -> ``NullSearchBackend``。

    最后一步（Null）只在**联网被关掉**时才会走到 —— DuckDuckGo 后端不需要 key，
    所以"没有 key"并不构成降级理由。把这一步显式写出来（而不是让 DuckDuckGo 兜底后
    再无条件返回），是为了让"搜索能力为 Null"这件事在 trace/日志里可见（红线 12）。
    """
    if os.environ.get("TAVILY_API_KEY"):
        return TavilyBackend()
    if os.environ.get("SERPER_API_KEY"):
        return SerperBackend()
    if not parse_bool(os.environ.get("LITEAGENT_ALLOW_NETWORK"), default=True):
        warnings.warn(
            "LITEAGENT_ALLOW_NETWORK is off; using NullSearchBackend (search returns no results)",
            RuntimeWarning,
            stacklevel=2,
        )
        return NullSearchBackend()
    return DuckDuckGoHTMLBackend()


# --------------------------------------------------------------------------------------
# HTML -> 文本（纯 stdlib）
# --------------------------------------------------------------------------------------

# 这些标签的**内容**不是正文：script/style 是代码，head/title 是元数据，
# noscript/template/svg 在抽取正文时只会带来噪声。
_SKIP_CONTENT_TAGS = frozenset({"script", "style", "head", "noscript", "template", "svg"})

# 块级标签：进入/离开时补换行，否则整页 HTML 会被压成一整行、段落关系全丢。
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
        "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
)


class HTMLTextExtractor(HTMLParser):
    """纯 stdlib 的 HTML -> 文本：跳过 script/style/head，块级标签补换行，
    压缩连续空行，解码实体（用 ``html.unescape``）。

    ``links`` 是额外产物：只收集 ``href`` 为 http/https 的链接（``javascript:`` /
    ``mailto:`` / 相对路径全部丢弃）。Agent 常需要"从这一页找下一页"，
    而正文里往往没有 URL 文本。
    """

    def __init__(self) -> None:
        # convert_charrefs=True：parser 自己就把 `&amp;` 解成 "&"。
        # 我们仍在 get_text 里再过一遍 html.unescape —— 属性值里的实体不会被
        # convert_charrefs 处理，且这一步是幂等的（`&amp;` 不会被二次解码成别的东西）。
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_stack: list[str] = []
        self.links: list[str] = []

    # ---- HTMLParser 回调 ----

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_CONTENT_TAGS:
            self._skip_stack.append(lowered)
            return
        if lowered == "a":
            href = dict(attrs).get("href") or ""
            if href and _is_fetchable_url(href):
                self.links.append(href)
        if lowered in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_CONTENT_TAGS:
            # 容错：畸形 HTML 里出现多余的 </script> 时不能把 skip 状态弄反。
            if lowered in self._skip_stack:
                self._skip_stack.remove(lowered)
            return
        if lowered in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        self._parts.append(data)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """自闭合标签（``<br/>`` / ``<img/>``）也走一遍 start 逻辑。"""
        self.handle_starttag(tag, attrs)

    # ---- 输出 ----

    def get_text(self) -> str:
        """渲染成文本：解实体 -> 去行尾空白 -> 压缩连续空行。"""
        raw = html.unescape("".join(self._parts))
        lines = [line.strip() for line in raw.splitlines()]
        rendered: list[str] = []
        for line in lines:
            if not line:
                # 连续空行只保留一个：HTML 里 `<div></div>` 连锁产生的空行非常多，
                # 原样保留会让文本的 token 成本翻倍、信息密度减半。
                if rendered and rendered[-1] == "":
                    continue
                rendered.append("")
            else:
                rendered.append(line)
        return "\n".join(rendered).strip()


def html_to_text(html: str, *, max_chars: int = 20000) -> str:
    """HTML -> 正文文本（纯 stdlib）。超长时头尾保留截断（尾部常有关键的"下一页"信息）。"""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - 畸形 HTML 不该让抓取整体失败
        warnings.warn(
            "HTML parsing failed; returning whatever was extracted so far",
            RuntimeWarning,
            stacklevel=2,
        )
    text = parser.get_text()
    if max_chars > 0 and len(text) > max_chars:
        text = truncate_head_tail(text, max_chars)
    return text


# --------------------------------------------------------------------------------------
# 模块级私有实现（闭包注入后端/传输/开关）
# --------------------------------------------------------------------------------------


def _web_search(
    _backend: SearchBackend,
    _allow_network: bool,
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web. Returns title/url/snippet lines."""
    if not _allow_network:
        return _NETWORK_DISABLED_MESSAGE
    if not query or not query.strip():
        return "ERROR: query must be a non-empty string"
    backend_name = getattr(_backend, "name", type(_backend).__name__)
    try:
        hits = _backend.search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 - 后端不该把异常抛到工具层
        return (
            f"search failed (backend={backend_name!r}): {type(exc).__name__}: {exc}; "
            "try a different query or answer from what you already know"
        )
    if not hits:
        detail = getattr(_backend, "last_error", "") or ""
        hint = (
            "no search backend is configured (set TAVILY_API_KEY or SERPER_API_KEY)"
            if backend_name == "null"
            else "the backend returned no results"
        )
        return (
            f"no search results (backend={backend_name!r}, query={query!r}): {hint}"
            + (f" [{detail}]" if detail else "")
        )
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. {hit.title}".rstrip())
        lines.append(f"   {hit.url}")
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
    lines.append(f"[{len(hits)} result(s) from backend={backend_name!r}]")
    return "\n".join(lines)


def _fetch_url(
    _backend: SearchBackend,
    _transport: "Transport | None",
    _allow_network: bool,
    url: str,
    max_chars: int = 20000,
    timeout_s: float = 15.0,
) -> str:
    """Fetch a URL and return its main text content."""
    if not _allow_network:
        return _NETWORK_DISABLED_MESSAGE
    if not _is_fetchable_url(url):
        return f"ERROR: only http/https URLs are supported, got {url!r}"
    try:
        transport = _resolve_transport(_transport)
        response = _send(
            transport,
            _http_request(
                method="GET",
                url=url,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html, text/plain;q=0.9"},
                timeout_s=timeout_s,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: fetch failed for {url}: {type(exc).__name__}: {exc}; "
            "check the URL or try a different source"
        )
    if not response.ok:
        return f"ERROR: HTTP {response.status_code} for {url}"
    content_type = ""
    for key, value in (response.headers or {}).items():
        if key.lower() == "content-type":
            content_type = value.lower()
            break
    body = response.text or ""
    if "html" in content_type or (not content_type and _looks_like_html(body)):
        text = html_to_text(body, max_chars=max_chars)
    elif content_type and not content_type.startswith("text/"):
        # 非文本响应（PDF/图片/二进制）：直说，别把二进制当文本回灌给模型。
        return (
            f"ERROR: unsupported content-type {content_type!r} for {url}; "
            "only text/html and text/* are converted to text"
        )
    else:
        text = body
        if max_chars > 0 and len(text) > max_chars:
            text = truncate_head_tail(text, max_chars)
    if not text.strip():
        return f"(empty body from {url}; content-type={content_type or 'unknown'})"
    return f"# {url}\n{text}"


def _looks_like_html(body: str) -> bool:
    """没有 Content-Type 时的嗅探：前 512 个字符里出现 ``<html`` / ``<!doctype`` / ``<body``。"""
    head = body[:512].lower()
    return any(marker in head for marker in ("<html", "<!doctype", "<body", "<div", "<p>"))


# --------------------------------------------------------------------------------------
# 模型可见的 schema（手写）
# --------------------------------------------------------------------------------------

_WEB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {"type": "integer", "description": "Maximum number of results to return."},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_FETCH_URL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Absolute http/https URL to fetch."},
        "max_chars": {"type": "integer", "description": "Maximum number of characters to return."},
        "timeout_s": {"type": "number", "description": "Request timeout in seconds."},
    },
    "required": ["url"],
    "additionalProperties": False,
}


def _call_args(
    args: dict[str, Any],
    *,
    tool_name: str,
    required: Sequence[str],
    optional: Sequence[str],
) -> dict[str, Any]:
    """整理成可 ``**`` 到纯实现的 kwargs（与 ``files._call_args`` 同一套约定）。

    每个模块各留一份而不是共享：``tools/builtin/`` 内部的同层 import 只允许 §1.1 的
    **E8**（``shell/code -> files``，因为 ``PathSandbox`` 在那里）。为一个 12 行的
    参数整理函数新增一条同层依赖边，会让守门测试的"允许边清单"变成一句空话。
    """
    missing = [key for key in required if args.get(key) is None]
    if missing:
        raise ToolValidationError(
            errors=[f"{key!r} is a required property" for key in missing],
            tool_name=tool_name,
        )
    cleaned = {key: args[key] for key in required}
    for key in optional:
        value = args.get(key)
        if value is not None:
            cleaned[key] = value
    return cleaned


def make_web_tools(
    backend: SearchBackend | None = None,
    *,
    allow_network: bool = True,
    transport: "Transport | None" = None,
) -> list[Tool]:
    """返回 ``[web_search, fetch_url]``。

    [v2 变更] ``transport`` 是**离线测试的唯一注入点**：把 ``FakeTransport`` 传进来，
    ``fetch_url`` 与 ``DuckDuckGoHTMLBackend`` 都不会真的碰网络。

    ``allow_network=False``（或环境变量 ``LITEAGENT_ALLOW_NETWORK=0``）时两者都返回
    ``'network access is disabled'`` 失败结果（仍可见）。
    """
    effective_allow = bool(allow_network) and parse_bool(
        os.environ.get("LITEAGENT_ALLOW_NETWORK"), default=True
    )
    if backend is None:
        backend = default_search_backend() if effective_allow else NullSearchBackend()

    def web_search(args: dict[str, Any]) -> str:
        cleaned = _call_args(
            args,
            tool_name="web_search",
            required=("query",),
            optional=("max_results",),
        )
        return _web_search(backend, effective_allow, **cleaned)

    def fetch_url(args: dict[str, Any]) -> str:
        cleaned = _call_args(
            args,
            tool_name="fetch_url",
            required=("url",),
            optional=("max_chars", "timeout_s"),
        )
        return _fetch_url(backend, transport, effective_allow, **cleaned)

    return [
        make_function_tool(
            name="web_search",
            description=(
                "Search the web and return title/url/snippet lines. "
                "Use it to discover sources; follow up with fetch_url."
            ),
            parameters=_WEB_SEARCH_PARAMETERS,
            func=web_search,
            tags=("web",),
            dangerous=False,
            idempotent=True,
            # 网络类工具重试有意义（幂等、失败多为瞬时）。§7.5 冻结"两者 retryable=True"，
            # 而在 ToolSpec 里表达"这个工具值得重试"的唯一字段就是显式的 max_retries。
            max_retries=DEFAULT_MAX_RETRIES,
        ),
        make_function_tool(
            name="fetch_url",
            description="Fetch an http/https URL and return its main text content.",
            parameters=_FETCH_URL_PARAMETERS,
            func=fetch_url,
            tags=("web",),
            dangerous=False,
            idempotent=True,
            max_retries=DEFAULT_MAX_RETRIES,
        ),
    ]
