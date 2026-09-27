from __future__ import annotations

# HTTP 传输层（§6.2）。
#
# 分层目的：把"怎么发 HTTP"与"怎么拼某个 provider 的请求体"彻底分开 ——
# `Transport` 只负责请求/响应与**错误映射**，provider 只负责 body 与响应解码。
# 这样离线测试可以塞一个 `FakeTransport`（§12.1）进来，整条 LLM 链路零网络可测。
#
# 红线（§6.2）：
# - `UrllibTransport` 是**唯一**必须实现的传输层（纯 stdlib，永远可用）。
# - `RequestsTransport` / `HttpxTransport` 只在库可用时定义（`if XXX_AVAILABLE:` 包裹），
#   否则模块仍可 import；因此**不要**在这两个类外面引用它们。
# - `asyncio.CancelledError` 绝不能在本模块被包装（M-5 / §6.2 [v2 变更]）。
#
# 为什么头部写注释而不是模块 docstring：§2.1 冻结"每个 .py 文件**第一行**必须是
# `from __future__ import annotations`"，两者不可兼得 —— 规范优先。

import asyncio
import contextlib
import functools
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from liteagent.config import DEFAULT_LLM_TIMEOUT_S
from liteagent.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
)

try:  # pragma: no cover - 环境相关（§1.3 冻结写法）
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

try:  # pragma: no cover - 环境相关（§1.3 冻结写法）
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

#: `json_body` 非 None 时自动补的 Content-Type（只在调用方没显式给时补，见 `UrllibTransport.send`）。
JSON_CONTENT_TYPE = "application/json"

#: `map_http_error` 放进 `LiteAgentError.context` 的 body 截断长度（避免异常对象里挂几 MB 的 HTML）。
_ERROR_BODY_CONTEXT_CHARS = 1000

# 异常构造约定（与 liteagent/errors.py 逐字对齐，别写反）：
#   `errors.py` 把 §3.3 表里的**额外字段排在 `message` 之前**（位置或关键字皆可），
#   而 `message` 是 keyword-only：`LLMAuthError(status_code=401, message="...", context={...})`。
#   §3.3 本身没有冻结子类 `__init__` 的签名，所以本文件一律**全关键字**构造 ——
#   对"message 在前/在后"两种约定都成立，也不依赖参数顺序
#   （`LLMAuthError("HTTP 401")` 会把字符串塞进 `status_code`，是个静默的类型错误）。
#   为什么不在 `llm/message.py` 里共用同一约定注释：§1.1 冻结 `transport.py <- errors, config`，
#   transport 不得 import `llm/message.py`。


@dataclass
class HTTPRequest:
    """一次 HTTP 请求的描述（纯数据，不持有连接）。"""

    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    json_body: Any = None  # 与 data 互斥；非 None 时序列化并设 Content-Type
    data: bytes | None = None
    params: dict[str, str] = field(default_factory=dict)
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S


@dataclass
class HTTPResponse:
    """一次 HTTP 响应的最小投影（只保留 provider 解码需要的三样东西）。"""

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    url: str = ""

    @property
    def ok(self) -> bool:
        """`200 <= status < 300`。"""
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """解析 body；失败抛 `LLMResponseFormatError`（它带 `body` 字段，便于排查）。

        这里用 `json.loads` 而不是 `to_jsonable`：反序列化方向没有 dataclass 复原需求，
        多一层反而会掩盖"body 根本不是 JSON"这个事实。
        """
        try:
            return json.loads(self.text)
        except (ValueError, TypeError) as exc:
            raise LLMResponseFormatError(
                body=self.text,
                message=f"response body is not valid JSON (status={self.status_code}, url={self.url})",
                cause=exc,
            ) from exc


class Transport(ABC):
    """传输层抽象：子类只需实现同步 `send`，异步版本由基类统一提供。"""

    name: str = "abstract"

    @abstractmethod
    def send(self, request: HTTPRequest) -> HTTPResponse:
        """发一次请求。**失败必须以 `LiteAgentError` 子类抛出**（见 `map_http_error`）。"""
        raise NotImplementedError

    async def asend(self, request: HTTPRequest) -> HTTPResponse:
        """默认实现：`await asyncio.to_thread(self.send, request)`。

        所有 Transport 只实现同步 `send`，异步版本由此默认实现提供。

        [v2 变更] 取消语义：调用方取消时 `asend` 会被取消，但 HTTP 线程同样成为孤儿。
        冻结：**不重试已发送的请求**（`LLMTimeoutError` 的 retryable 在传输层保持 True，
        但 `ToolSpec` 之外的 LLM 调用在 timeout 后是否重发由 `_with_retry` 决定，
        而 `HTTPChatClient` 对 `LLMTimeoutError` **不重试**（幂等性未知）——
        见 §6.4 的 `retry_on_timeout = False` 类属性）。

        `to_thread` 里抛出的 `CancelledError` 不会被 `send` 的 `except Exception` 吃掉
        （它继承 `BaseException`，M-5），因此取消能原样传播到调用方。
        """
        return await asyncio.to_thread(self.send, request)

    def close(self) -> None:
        """默认 no-op：无连接池的实现不需要释放任何东西。"""
        return None


def _resolve_timeout(timeout_s: float | None) -> float | None:
    """把超时哨兵归一化为"各自库认识的形态"。

    `NO_TIMEOUT = -1.0`（§2.5）表示显式禁用超时；`None` 同样表示不限时。
    直接把它透传给 `urlopen(timeout=-1)` / `requests(timeout=-1)` 会抛
    `ValueError: Timeout value out of range`，所以统一映射为 `None`（阻塞式）。
    """
    if timeout_s is None or timeout_s <= 0:
        return None
    return float(timeout_s)


def _decode(raw: bytes | str) -> str:
    """字节 -> 文本。用 `errors="replace"` 而不是严格模式：一个坏字节不该让整次调用失败。"""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


def _headers_to_dict(headers: Any) -> dict[str, str]:
    """把 `http.client.HTTPMessage` / `CaseInsensitiveDict` / dict 统一成 `dict[str, str]`。"""
    if headers is None:
        return {}
    items = getattr(headers, "items", None)
    if items is None:
        return {}
    try:
        return {str(key): str(value) for key, value in items()}
    except Exception as exc:  # pragma: no cover - 罕见：畸形 headers 对象
        warnings.warn(
            f"failed to normalize response headers ({type(headers).__name__}): {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}


def _close_quietly(obj: Any, what: str) -> None:
    """尽力关闭响应/错误对象。关闭失败只留 warning，绝不掩盖原始异常。"""
    close = getattr(obj, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:  # pragma: no cover - 罕见：socket 已断
        warnings.warn(f"failed to close {what}: {exc!r}", RuntimeWarning, stacklevel=2)


def _apply_params(url: str, params: Mapping[str, str]) -> str:
    """把 query 参数合并进 URL（urllib 没有 params 参数）。"""
    if not params:
        return url
    query = urllib.parse.urlencode(dict(params))
    if not query:
        return url
    separator = "&" if urllib.parse.urlsplit(url).query else "?"
    return f"{url}{separator}{query}"


class UrllibTransport(Transport):
    """纯 stdlib 实现，零依赖，永远可用（§6.2 红线：唯一必须实现的传输层）。"""

    name = "urllib"

    def send(self, request: HTTPRequest) -> HTTPResponse:
        url = _apply_params(request.url, request.params)
        headers = dict(request.headers)
        body: bytes | None = request.data
        if request.json_body is not None:
            # 与 data 互斥：json_body 优先（约定见 §6.2 字段注释）。
            body = json.dumps(request.json_body).encode("utf-8")
            # 用 setdefault 而不是覆盖：调用方显式给的 Content-Type 优先级更高
            # （例如 Anthropic 的 "application/json" 之外还可能带 charset）。
            headers.setdefault("Content-Type", JSON_CONTENT_TYPE)

        url_request = urllib.request.Request(
            url, data=body, headers=headers, method=request.method
        )
        try:
            # `urlopen(timeout=...)` 的超时同时覆盖连接与读取；必须关闭连接（closing）。
            with contextlib.closing(
                urllib.request.urlopen(url_request, timeout=_resolve_timeout(request.timeout_s))
            ) as response:
                status = getattr(response, "status", None)
                if status is None:  # pragma: no cover - file:// 等非 HTTP 响应
                    status = response.getcode()
                return HTTPResponse(
                    status_code=int(status),
                    headers=_headers_to_dict(getattr(response, "headers", None)),
                    text=_decode(response.read()),
                    url=response.geturl() or url,
                )
        except urllib.error.HTTPError as exc:
            # HTTPError 是 URLError 子类，**同时是一个可读的响应对象**（§6.2 易错点冻结）：
            # 必须先读 body 才能交给 map_http_error，否则错误信息只剩 "HTTP Error 4xx"。
            try:
                body_text = _decode(exc.read())
            except Exception as read_exc:  # pragma: no cover - body 已被消费等
                warnings.warn(
                    f"failed to read HTTP {getattr(exc, 'code', '?')} error body: {read_exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                body_text = ""
            finally:
                _close_quietly(exc, "HTTPError")
            raise map_http_error(
                exc.code,
                body_text,
                url=url,
                headers=_headers_to_dict(getattr(exc, "headers", None)),
                timeout_s=request.timeout_s,
            ) from exc
        except Exception as exc:
            # BaseException（含 CancelledError / KeyboardInterrupt）不走这里 —— 原样上抛。
            raise wrap_transport_exception(exc, timeout_s=request.timeout_s) from exc


if REQUESTS_AVAILABLE:

    class RequestsTransport(Transport):
        """基于 `requests.Session`（连接复用）。仅在 requests 可 import 时定义。"""

        name = "requests"

        def __init__(self, *, session: Any | None = None) -> None:
            """`session` 可注入（测试可传假 session）；未注入时懒创建并复用。

            用 `threading.Lock` 而不是 `asyncio.Lock`：`send` 会在线程池里被调用
            （`asend` -> `to_thread`），跨线程生效的互斥只能用 threading 原语（§13 红线 13）。
            """
            self._session = session
            self._lock = threading.Lock()

        def _ensure_session(self) -> Any:
            with self._lock:
                if self._session is None:
                    self._session = requests.Session()
                return self._session

        def send(self, request: HTTPRequest) -> HTTPResponse:
            session = self._ensure_session()
            try:
                response = session.request(
                    request.method,
                    request.url,
                    headers=dict(request.headers),
                    params=dict(request.params) if request.params else None,
                    json=request.json_body if request.json_body is not None else None,
                    # json_body 与 data 互斥：同时传会被 requests 拒绝，这里按字段注释取 json_body。
                    data=None if request.json_body is not None else request.data,
                    timeout=_resolve_timeout(request.timeout_s),
                )
            except Exception as exc:
                raise wrap_transport_exception(exc, timeout_s=request.timeout_s) from exc

            body_text = getattr(response, "text", "") or ""
            if response.status_code >= 400:
                raise map_http_error(
                    response.status_code,
                    body_text,
                    url=getattr(response, "url", "") or request.url,
                    headers=getattr(response, "headers", None),
                    timeout_s=request.timeout_s,
                )
            return HTTPResponse(
                status_code=int(response.status_code),
                headers=_headers_to_dict(getattr(response, "headers", None)),
                text=body_text,
                url=getattr(response, "url", "") or request.url,
            )

        def close(self) -> None:
            with self._lock:
                session, self._session = self._session, None
            if session is not None:
                session.close()


if HTTPX_AVAILABLE:

    class HttpxTransport(Transport):
        """基于 `httpx.Client`（连接复用 + HTTP/2 可选）。仅在 httpx 可 import 时定义。"""

        name = "httpx"

        def __init__(self, *, client: Any | None = None, verify: bool | None = None) -> None:
            """`client` 可注入；`verify=False` 时关闭证书校验（自签名的本地 vLLM 场景）。"""
            self._client = client
            self._verify = verify
            self._lock = threading.Lock()

        def _ensure_client(self) -> Any:
            with self._lock:
                if self._client is None:
                    kwargs: dict[str, Any] = {}
                    if self._verify is not None:
                        kwargs["verify"] = self._verify
                    self._client = httpx.Client(**kwargs)
                return self._client

        def send(self, request: HTTPRequest) -> HTTPResponse:
            client = self._ensure_client()
            try:
                response = client.request(
                    request.method,
                    request.url,
                    headers=dict(request.headers),
                    params=dict(request.params) if request.params else None,
                    json=request.json_body if request.json_body is not None else None,
                    # json_body 与 data 互斥（同 RequestsTransport）。
                    content=None if request.json_body is not None else request.data,
                    timeout=_resolve_timeout(request.timeout_s),
                )
            except Exception as exc:
                raise wrap_transport_exception(exc, timeout_s=request.timeout_s) from exc

            body_text = getattr(response, "text", "") or ""
            if response.status_code >= 400:
                raise map_http_error(
                    response.status_code,
                    body_text,
                    url=str(getattr(response, "url", "") or request.url),
                    headers=getattr(response, "headers", None),
                    timeout_s=request.timeout_s,
                )
            return HTTPResponse(
                status_code=int(response.status_code),
                headers=_headers_to_dict(getattr(response, "headers", None)),
                text=body_text,
                url=str(getattr(response, "url", "") or request.url),
            )

        def close(self) -> None:
            with self._lock:
                client, self._client = self._client, None
            if client is not None:
                client.close()


@functools.cache
def default_transport() -> Transport:
    """优先级：httpx -> requests -> urllib。结果缓存（`functools.cache`）。

    构造失败时降级到下一档并留 `warnings` 痕迹（红线 12：降级不得静默）——
    例如 httpx 装在但底层 SSL 后端缺失时，`httpx.Client()` 会抛异常，
    此时用 requests/urllib 比让整个框架起不来合理。
    """
    candidates: list[tuple[str, Any]] = []
    if HTTPX_AVAILABLE:
        candidates.append(("httpx", HttpxTransport))
    if REQUESTS_AVAILABLE:
        candidates.append(("requests", RequestsTransport))
    candidates.append(("urllib", UrllibTransport))

    failures: list[str] = []
    for name, factory in candidates:
        if name == "urllib":
            # urllib 失败就没有退路了，让异常直接冒泡（不吞）。
            return factory()
        try:
            return factory()
        except Exception as exc:  # pragma: no cover - 依赖环境
            failures.append(f"{name}: {exc!r}")
            warnings.warn(
                f"{name} transport is unavailable ({exc!r}); falling back",
                RuntimeWarning,
                stacklevel=2,
            )
    raise LLMConnectionError(  # pragma: no cover - 上面的 urllib 分支已保证可达
        f"no usable transport could be built: {failures}"
    )


def _get_header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """大小写不敏感的 header 查找（HTTP header 名不区分大小写）。"""
    if not headers:
        return None
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return value
    return None


def _parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """解析 `Retry-After`。

    规范冻结为"支持纯数字秒"（§6.2），因此 HTTP-date 形态**不作解析**，返回 None
    （让调用方退回自己的退避策略，好过猜一个错的等待时间）。
    """
    raw = _get_header(headers, "Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def map_http_error(
    status: int,
    body: str,
    *,
    url: str = "",
    headers: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> LLMError:
    """统一错误映射（冻结表，§6.2）：

      401/403        -> LLMAuthError
      429            -> LLMRateLimitError（retry_after_s 解析 Retry-After 头，支持纯数字秒）
      408/409/425    -> LLMConnectionError（可按 retryable 处理）
      400/404/413/422-> LLMBadRequestError
      5xx            -> LLMConnectionError
      其它 4xx       -> LLMBadRequestError

    表外状态（1xx/2xx/3xx）不在这张表里，也不能凭空造异常语义：
    返回基类 `LLMError`，message 里带状态码，`context` 里带状态码与 URL。

    返回值**不是**抛出的：调用方 `raise map_http_error(...)` —— 这样测试可以直接断言
    返回类型与字段，不必用 `assertRaises` 拆包（§12 `test_transport.py` 就是这么用的）。
    """
    where = url or "endpoint"
    message = f"HTTP {status} from {where}"
    context: dict[str, Any] = {"status_code": status, "url": url}
    if timeout_s is not None:
        context["timeout_s"] = timeout_s
    if body:
        context["body"] = body[:_ERROR_BODY_CONTEXT_CHARS]

    if status in (401, 403):
        return LLMAuthError(status_code=status, message=message, context=context)

    if status == 429:
        retry_after = _parse_retry_after(headers)
        if retry_after is not None:
            context["retry_after_s"] = retry_after
            return LLMRateLimitError(
                status_code=status,
                retry_after_s=retry_after,
                message=message,
                context=context,
            )
        return LLMRateLimitError(status_code=status, message=message, context=context)

    if status in (408, 409, 425):
        return LLMConnectionError(status_code=status, message=message, context=context)

    if status in (400, 404, 413, 422):
        return LLMBadRequestError(
            status_code=status, body=body, message=message, context=context
        )

    if 500 <= status < 600:
        return LLMConnectionError(status_code=status, message=message, context=context)

    if 400 <= status < 500:
        # 表里的"其它 4xx"
        return LLMBadRequestError(
            status_code=status, body=body, message=message, context=context
        )

    return LLMError(
        message=f"{message} (unexpected status; not present in the retryable mapping table)",
        context=context,
    )


def wrap_transport_exception(exc: BaseException, *, timeout_s: float | None) -> LLMError:
    """把传输层底层异常映射成 LLM 层异常。

      socket.timeout/TimeoutError/asyncio.TimeoutError -> LLMTimeoutError
      urllib.error.URLError / ConnectionError / OSError / httpx.TransportError -> LLMConnectionError
      其它 -> LLMError

    [v2 变更] **asyncio.CancelledError 必须原样向上抛**（它是 BaseException，M-5），
    绝不能被本函数包装（否则取消传播被破坏）。注意 `asyncio.TimeoutError is not
    builtins.TimeoutError`（M-5），两者都要显式列出。
    """
    # 顺序关键：CancelledError 必须第一个判（它 IS-A BaseException，任何宽 except 都会误伤）。
    # 用 raise 而不是 return：调用方写 `raise wrap_transport_exception(...)` 时，
    # 这里的 raise 会先一步把取消抛出去，语义与"原样上抛"完全一致。
    if isinstance(exc, asyncio.CancelledError):
        raise exc

    timeout_value = 0.0 if timeout_s is None else float(timeout_s)
    timeout_detail = (
        f"after {timeout_s}s" if timeout_s is not None else "(timeout budget unknown)"
    )

    # 关于 `socket.timeout`：3.10 起它就是内建 `TimeoutError` 的别名
    # （已在本机实测 `socket.timeout is TimeoutError == True`），显式列出纯属冗余；
    # 而 §1.3 冻结的 3.11-API AST 检查把**任何** `attr == "timeout"` 的属性访问判为违规，
    # 写 `socket.timeout` 会误伤自己 —— 故只列 `TimeoutError` + `asyncio.TimeoutError`
    # （M-5 实测 `asyncio.TimeoutError is not builtins.TimeoutError`，两者都必须留）。
    #
    # requests/httpx 的超时类不是 builtin TimeoutError，必须单独判（且要在 OSError 之前：
    # requests.RequestException 继承自 IOError == OSError，会被后面的分支吃掉）。
    is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
    if not is_timeout and REQUESTS_AVAILABLE:
        is_timeout = isinstance(exc, requests.exceptions.Timeout)
    if not is_timeout and HTTPX_AVAILABLE:
        is_timeout = isinstance(exc, httpx.TimeoutException)
    if is_timeout:
        return LLMTimeoutError(
            timeout_s=timeout_value,
            message=f"request timed out {timeout_detail}: {exc}",
            cause=exc,
            context={"timeout_s": timeout_value, "exception_type": type(exc).__name__},
        )

    context: dict[str, Any] = {"exception_type": type(exc).__name__}
    if timeout_s is not None:
        context["timeout_s"] = timeout_s

    # URLError 是 OSError 子类，一起判；httpx.TransportError 不在 OSError 体系里，补一条。
    is_connection = isinstance(exc, (urllib.error.URLError, ConnectionError, OSError))
    if not is_connection and HTTPX_AVAILABLE:
        is_connection = isinstance(exc, httpx.TransportError)
    if not is_connection and REQUESTS_AVAILABLE:
        is_connection = isinstance(exc, requests.exceptions.RequestException)
    if is_connection:
        return LLMConnectionError(
            status_code=None,
            message=f"connection failed: {type(exc).__name__}: {exc}",
            cause=exc,
            context=context,
        )

    return LLMError(
        message=f"transport failure: {type(exc).__name__}: {exc}",
        cause=exc,
        context=context,
    )
