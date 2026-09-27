from __future__ import annotations

"""共享测试夹具（签名逐字对齐规范 §12.1，10 个测试实现者共用本文件）。

本模块的存在理由是"唯一性"（§12.1 原文）：没有它，``test_transport.py`` 与
``test_llm_providers.py`` 会各写一套不兼容的 ``FakeTransport`` 并在合并时冲突。
因此**签名冻结**，只允许在实现细节上补强，不允许改名/改参数。

冻结的测试卫生规则（§12.1）：
1. 任何 ``setUp``/``tearDown`` 里用到 ``auto_register`` 的测试，必须在 ``tearDown``
   调 ``reset_default_registry()``（本模块的三个 ``@tool`` 都是 ``auto_register=False``，
   只有 import 期零副作用这一条约束）。
2. 不得依赖真实网络、真实时钟、真实 sleep —— ``FakeTransport`` 断网、
   ``RecordingSleep`` 不睡、``frozen_time`` 冻结 ``config.utc_now``。
3. 断言不确定字段前先查 §2.8 的禁止清单。
"""

import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence
from unittest import mock

from liteagent.agent.callbacks import CallbackManager, EventType, TraceEvent
from liteagent.config import LoopBoundPool
from liteagent.llm.transport import (
    HTTPRequest,
    HTTPResponse,
    Transport,
    map_http_error,
)
from liteagent.memory.embeddings import CallableEmbedder
from liteagent.tools.base import Tool, tool
from liteagent.tools.builtin.files import PathSandbox
from liteagent.tools.builtin.web import SearchBackend, SearchHit
from liteagent.tools.registry import ToolRegistry

__all__ = [
    "FakeTransport",
    "FakeSearchBackend",
    "RecordingSleep",
    "frozen_time",
    "make_registry",
    "collect_events",
    "make_temp_sandbox",
    "det_embedder",
    "assert_no_error_events",
    "liteagent_worker_threads",
    "close_loop_bound_pools",
    "add",
    "echo",
    "boom",
    "DEFAULT_SEARCH_HITS",
    "ERROR_EVENT_TYPES",
]


# ======================================================================================
# 传输层夹具（§6.2 / §12.1）
# ======================================================================================

#: 默认响应被"显式提供"的判据（见 FakeTransport.__init__ 的说明）。
_DEFAULT_STATUS = 200
_DEFAULT_BODY = "{}"


class FakeTransport(Transport):
    """可编程的假传输层。``requests`` 记录每一次 ``send`` 的请求。

    **错误映射与真传输层一致**（§6.2 契约："失败必须以 ``LiteAgentError`` 子类抛出"）：
    回放出来的 ``HTTPResponse.status_code >= 400`` 时，``send`` 调用
    ``map_http_error(...)``（带上该响应的 headers 与请求的 url/timeout，因此
    ``Retry-After`` 会生效）并**抛出**该异常 —— 假传输层不是"只回放不映射"的例外。

    设计取舍（§12.1 只写了"队列为空且未给默认响应时抛 AssertionError"，
    而签名里的 ``status``/``body`` 有字面默认值 ``200``/``"{}"``，无法区分
    "没传"与"传了同一个值"）：

    * 队列里的元素按 FIFO 回放；元素是 ``BaseException`` 时**原样抛出**
      （测错误路径，例如 ``LLMTimeoutError`` / ``asyncio.CancelledError``）；
      元素是 ``HTTPResponse`` 时，``status_code < 400`` 原样返回，
      ``status_code >= 400`` 走 :func:`map_http_error` 抛异常（见上）。
    * 队列耗尽后，若调用方**显式**给了 ``status``/``body``（即与字面默认值不同），
      则永久回放这个默认响应（同样对 ``>= 400`` 走映射/抛异常）；否则抛
      ``AssertionError("FakeTransport ran out of queued responses")``。

    两种用法（都对应真实后端行为）：

    (a) 造一个"永久 429 的后端"（每次 ``send`` 都抛 ``LLMRateLimitError``，
        无 ``Retry-After``）：只写两个 kwarg 即可，队列为空时回放默认响应::

            FakeTransport(status=429, body="slow down")

    (b) 要带 ``Retry-After`` 之类的头，就排队一个 ``HTTPResponse``
        （``retry_after_s`` 会从头上解析出来）::

            FakeTransport([HTTPResponse(
                status_code=429, headers={"Retry-After": "2"}, text="slow down",
            )])
            # -> 抛 LLMRateLimitError(retry_after_s=2.0)

    于是 ``FakeTransport([r])`` 在第二次 ``send`` 时抛 AssertionError。
    """

    name = "fake"

    def __init__(
        self,
        responses: Sequence[HTTPResponse | BaseException] = (),
        *,
        status: int = _DEFAULT_STATUS,
        body: str = _DEFAULT_BODY,
    ) -> None:
        #: 每次 send 追加（**先记录后回放**，抛异常的请求也在记录里 —— 复盘时序时最需要它）。
        self.requests: list[HTTPRequest] = []
        self._queue: list[HTTPResponse | BaseException] = list(responses)
        self._status = status
        self._body = body
        self._has_default = (status, body) != (_DEFAULT_STATUS, _DEFAULT_BODY)

    def queue(self, response: HTTPResponse | BaseException) -> None:
        """追加一个响应（``HTTPResponse`` 或要抛出的异常实例）。"""
        self._queue.append(response)

    @property
    def queued(self) -> int:
        """队列里还剩几条（不是冻结签名的一部分，只是给断言用的便利属性）。"""
        return len(self._queue)

    def send(self, request: HTTPRequest) -> HTTPResponse:
        # 先记录后回放：抛异常的请求同样留在 requests 里（复盘时序最需要它）。
        self.requests.append(request)
        if self._queue:
            item = self._queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            response = item
        elif self._has_default:
            response = HTTPResponse(
                status_code=self._status,
                headers={},
                text=self._body,
                url=request.url,
            )
        else:
            raise AssertionError("FakeTransport ran out of queued responses")

        # 与真传输层同一条契约（§6.2）：失败必须以 LiteAgentError 子类抛出。
        # headers 必须带上，否则 Retry-After 解析不出来、429 会丢掉 retry_after_s。
        if response.status_code >= 400:
            raise map_http_error(
                response.status_code,
                response.text,
                url=response.url or request.url,
                headers=response.headers,
                timeout_s=request.timeout_s,
            )
        return response

    def __repr__(self) -> str:  # pragma: no cover - 只为失败信息可读
        return f"<FakeTransport queued={len(self._queue)} sent={len(self.requests)}>"


# ======================================================================================
# 搜索后端夹具（§7.5 / §12.1）
# ======================================================================================

#: ``FakeSearchBackend()`` 不给 hits 时回放的固定 3 条（"成功路径用"）。
DEFAULT_SEARCH_HITS: tuple[SearchHit, ...] = (
    SearchHit(
        title="liteagent 框架文档",
        url="https://example.invalid/liteagent/docs",
        snippet="ReAct 循环、工具系统与记忆层的离线文档。",
    ),
    SearchHit(
        title="ReAct: Synergizing Reasoning and Acting",
        url="https://example.invalid/react-paper",
        snippet="Thought-Action-Observation 循环的原始论文。",
    ),
    SearchHit(
        title="Function calling 与 JSON Schema",
        url="https://example.invalid/function-calling",
        snippet="把 Python 签名反射成工具 schema 的实践。",
    ),
)


class FakeSearchBackend(SearchBackend):
    """固定回放 ``hits``（默认 3 条）的搜索后端，**零网络**。

    ``hits`` 为空（或没传）时回放 :data:`DEFAULT_SEARCH_HITS`；要"空结果"语义请用
    ``NullSearchBackend``（§12 line 5133 就是这么测的）—— 两个夹具分工明确，
    免得"传了空列表却拿到 3 条"这种最费时间的误解。

    额外的可观测字段（不是冻结签名的一部分，但对断言很有用）：
    ``queries`` 按调用顺序记下每次查询串。
    """

    name = "fake"

    def __init__(self, hits: Sequence[SearchHit] = ()) -> None:
        self.hits: tuple[SearchHit, ...] = tuple(hits) if hits else DEFAULT_SEARCH_HITS
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout_s: float = 15.0,
    ) -> list[SearchHit]:
        self.queries.append(query)
        hits = list(self.hits)
        # max_results < 0 视为"不限"（与 NO_TIMEOUT 的哨兵精神一致）；0 返回空列表。
        if max_results is not None and int(max_results) >= 0:
            return hits[: int(max_results)]
        return hits

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FakeSearchBackend hits={len(self.hits)} queries={len(self.queries)}>"


# ======================================================================================
# 睡眠夹具（§5.5 / §12.1）
# ======================================================================================


class RecordingSleep:
    """替代真实 sleep 的 ``SleepFn`` 记录器：**默认立即返回、不真睡**。

    用法：把它塞给 ``RetryPolicy(sleep_fn=...)`` / ``ExecutorConfig(sleep_fn=...)`` /
    ``LLMConfig(sleep_fn=...)``，然后断言 ``delays`` 的序列 —— 从"墙钟耗时"变成
    "延迟序列"，既准确又快（D-26）。
    """

    delays: list[float]

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """只 append，不 await sleep（**这里绝不能出现 asyncio.sleep**）。"""
        self.delays.append(float(seconds))

    @property
    def total(self) -> float:
        """累计延迟（便利属性；断言总和比手写 sum 更可读）。"""
        return sum(self.delays)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RecordingSleep delays={self.delays!r}>"


# ======================================================================================
# 时钟夹具（§2.2 / §12.1）
# ======================================================================================

@contextmanager
def frozen_time(ts: float) -> Iterator[None]:
    """在 with 块内 patch ``liteagent.config.utc_now`` 返回传入的 ``ts``。

    实现**逐字**是 §12.1 冻结的那一行：
    ``unittest.mock.patch("liteagent.config.utc_now", return_value=ts)``。
    **不 patch ``time.time``**（D-26：避免干扰 logging / asyncio 自身的计时）。

    冻结能生效的写法只有两种（§12.1 / §2.2）：
    ``config.utc_now()`` 属性访问，与 ``field(default_factory=lambda: config.utc_now())``
    （``MemoryItem`` / ``MemoryItem.last_access_at`` 就是这个形态）。

    **刻意不冻结的两种形态**（两种都是 §12.1 明文认可的写法，不要用它们断言时间戳）：
    1. ``from liteagent.config import utc_now`` 的**模块级裸名**：import 期就把函数对象
       固定了，patch 模块属性够不到它。``blackboard.BlackboardEntry.created_at``、
       ``AgentState.started_at``、``TraceEvent.timestamp`` 都是这种形态 —— 想断言它们，
       要么显式传 ``now=`` / ``timestamp=``，要么在测试里自己
       ``mock.patch.object(<module>, "utc_now", return_value=ts)``。
    2. ``max_wall_clock_s`` 这类**真实墙钟**判定：冻结后 elapsed 恒为 0，
       要触发超时请直接给一个极小的 ``max_wall_clock_s``（依赖真实耗时），
       或自己用 ``side_effect`` 造一个递增的假时钟。
    """
    with mock.patch("liteagent.config.utc_now", return_value=ts):
        yield


# ======================================================================================
# 注册表 / 事件 / 沙箱 / 向量夹具（§12.1）
# ======================================================================================


def make_registry(*tools: Tool) -> ToolRegistry:
    """一个**全新的** ``ToolRegistry``（不碰 ``get_default_registry()``）。

    刻意不返回全局注册表：并行运行的两个测试文件若共用它，就互相污染了。
    默认注册表只在测 ``auto_register`` 的用例里出现（那种用例必须在 tearDown 里
    ``reset_default_registry()``，见 §12.1 冻结的测试卫生规则 1）。
    """
    return ToolRegistry(list(tools))


def collect_events() -> tuple[CallbackManager, list[TraceEvent]]:
    """返回 ``(manager, events)``：``events`` 是 manager 分发出去的**全部**事件的实时列表。

    ``list.append`` 是原子的，多线程 emit 下无需再加锁（与 ``TraceRecorder`` 同一理由）。
    """
    events: list[TraceEvent] = []
    manager = CallbackManager()
    manager.subscribe(events.append)
    return manager, events


@contextmanager
def make_temp_sandbox() -> Iterator[PathSandbox]:
    """在 ``tempfile.TemporaryDirectory`` 里造一个 ``PathSandbox``，退出时清理。

    传出 ``PathSandbox`` 而不是目录字符串：调用方要的是"沙箱"（``resolve`` /
    ``display`` / 逃逸判定），拿到裸路径还得自己再包一层。
    """
    with tempfile.TemporaryDirectory(prefix="liteagent-sandbox-") as tmp:
        yield PathSandbox(tmp)


def det_embedder(table: Mapping[str, Sequence[float]]) -> CallableEmbedder:
    """按"文本 -> 向量"的查找表构造确定性 embedder（``dim`` 由表里的向量长度决定）。

    * 所有向量的长度必须一致，否则抛 ``ValueError``（长度不齐会在检索时变成
      难查的矩阵错位）。
    * 不在表里的文本**大声抛 ``KeyError``**（列出表里的键）：静默返回零向量会让
      "忘了给查询串配向量"伪装成"相似度为 0"，是最耗时的一类假象。
    * 注入的 ``CallableEmbedder`` 会按契约做 L2 归一化，所以断言时拿到的可能是
      单位向量（表的作者请按"方向"设计，别依赖未归一化前的模长）。
    """
    if not table:
        raise ValueError("det_embedder requires a non-empty 'text -> vector' table")
    dims = {len(vector) for vector in table.values()}
    if len(dims) != 1:
        raise ValueError(f"det_embedder: inconsistent vector dims {sorted(dims)}")
    dim = dims.pop()
    if dim <= 0:
        raise ValueError(f"det_embedder: vectors must be non-empty, got dim={dim}")

    lookup: dict[str, list[float]] = {
        str(text): [float(x) for x in vector] for text, vector in table.items()
    }

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            key = str(text)
            if key not in lookup:
                raise KeyError(
                    f"det_embedder: no vector for text {key!r}; "
                    f"table keys are {sorted(lookup)}"
                )
            out.append(list(lookup[key]))
        return out

    return CallableEmbedder(_embed, dim=dim, name="det")


#: "错误事件"的冻结集合：有它们就说明某条路径没走通。
ERROR_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.RUN_FAILED,
    EventType.LLM_ERROR,
    EventType.TOOL_ERROR,
)


def assert_no_error_events(
    testcase: unittest.TestCase, events: Sequence[TraceEvent]
) -> None:
    """断言 ``events`` 里没有 ``run_failed`` / ``llm_error`` / ``tool_error``。

    用 ``testcase.assertFalse`` 而不是裸 ``assert``（§12 硬性要求：禁用裸 assert），
    并在消息里带上前几条的具体内容 —— "有错误事件"本身不足以定位，错误类型与
    error_type 才是。
    """
    offenders = [event for event in events if event.type in ERROR_EVENT_TYPES]
    if not offenders:
        return
    preview = "; ".join(
        f"{event.type.value}({event.data.get('tool_name') or event.data.get('model') or ''}"
        f"{':' if event.data.get('error_type') else ''}"
        f"{event.data.get('error_type') or event.data.get('error') or event.data.get('message') or ''})"
        for event in offenders[:5]
    )
    testcase.assertFalse(
        offenders,
        msg=(
            f"expected no error events, found {len(offenders)} of "
            f"{len(events)}: {preview}"
        ),
    )


# ======================================================================================
# 测试卫生：LoopBoundPool 的 per-loop 线程池收尾（[v3]）
# ======================================================================================
#
# 为什么需要这一节（这是一个**真实的**、被手动变异测试证实过的洞）：
#
#   1. `Agent` / `ToolExecutor` 各自持有一个 `LoopBoundPool`，同步工具在它的
#      `"exec"` 线程池（`ThreadPoolExecutor(thread_name_prefix="liteagent-exec")`）里跑；
#   2. 线程池的 worker 是**非 daemon** 线程，只有 `shutdown()` 能让它们退出；
#   3. `config._run_and_cleanup` 会在 `run_sync` / `execute_sync` 的 `finally` 里
#      `release_loop(loop)` —— 但 `IsolatedAsyncioTestCase` **直接关闭每个用例的 event
#      loop**，根本不走那条路径，于是 `release_loop` 从来没被调用过；
#   4. 池对象又被 `LoopBoundPool._ALL_POOLS` 强引用，连 GC 回收都指望不上。
#
# 复现（把 `tests/test_agent_{features,react_native,react_text}.py` 的
# `tearDownModule` 改成 pass）：跑完整套件后残留 **33** 个 `liteagent-exec_*` 线程
# （`python3 -c "import threading; print(len([t for t in threading.enumerate() if 'liteagent' in t.name]))"`
# 在进程内为 0 才叫干净）。解释器的 atexit 会 join **空闲**池，所以进程仍能秒退 ——
# 代价是**进程内**的线程随模块数线性增长，且这套测试失去了"跑完不留线程"的性质。
#
# 这里提供**唯一**的收尾入口：模块级 `tearDownModule` 调 `close_loop_bound_pools()`。
# 它顺手把"跑完还剩几个 liteagent 线程"变成断言（而不是靠人肉去数）——
# 断言失败 = 有人加了新的 Agent 测试却忘了收尾，这正是要拦的那种回归。

#: worker 线程名前缀（`LoopBoundPool.thread_pool()` 的 `f"liteagent-{key}"` 产物）。
LITEAGENT_THREAD_PREFIX = "liteagent"

#: 关闭后等 worker 退出的宽限（`shutdown(wait=False)` 是**异步**的：worker 要先被
#: 唤醒再去读停止标志）。线程实际退出在微秒级，这里给足冗余，绝不写 `sleep(0.01)` 碰运气。
DEFAULT_THREAD_EXIT_GRACE_S = 5.0
_POLL_INTERVAL_S = 0.01


def liteagent_worker_threads() -> list[threading.Thread]:
    """当前存活的 `liteagent-*` 工作线程快照（测试卫生断言用）。"""
    return [t for t in threading.enumerate() if LITEAGENT_THREAD_PREFIX in t.name]


def close_loop_bound_pools(*, grace_s: float = DEFAULT_THREAD_EXIT_GRACE_S) -> None:
    """**统一的**测试收尾：关掉所有 `LoopBoundPool` 的线程池，并断言没有线程残留。

    干什么：
      1. `LoopBoundPool.aclose_all()` —— `release(None)` 所有实例（对每个 per-loop
         `ThreadPoolExecutor` 调 `shutdown(wait=False, cancel_futures=True)`），
         并把它们从 `_ALL_POOLS` 注销（否则那个列表会随用例数线性增长）；
      2. 轮询等 worker 真正退出（上限 `grace_s`），仍然存活就 `AssertionError` ——
         带上线程名，让失败信息直接指向漏收尾的那个模块。

    为什么是 `AssertionError` 而不是 `logger.warning`：这一步在 `tearDownModule` 里跑，
    抛异常会让**套件变红**。线程泄漏是"测试卫生"级别的问题，但它是**可复现**的、
    且修法只有一行 —— 让它变红比让它静默累积便宜得多（§13 红线 12 的同一个道理）。

    `grace_s` 可调是为了让不确定的环境（CI 上 CPU 被抢）不误报；
    **不要**把它当"多等一会儿就绿了"的旋钮 —— 真正的修法是给新模块加 `tearDownModule`。
    """
    LoopBoundPool.aclose_all()
    deadline = time.monotonic() + max(0.0, float(grace_s))
    leftover = liteagent_worker_threads()
    while leftover and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        leftover = liteagent_worker_threads()
    if leftover:
        raise AssertionError(
            "LoopBoundPool.aclose_all() 之后仍有 "
            f"{len(leftover)} 个 liteagent 工作线程存活（等待 {grace_s}s）："
            f"{sorted(t.name for t in leftover)} —— "
            "说明有池没被 release（模块级 tearDownModule 调 close_loop_bound_pools() 即可）"
        )


# ======================================================================================
# 常用工具（§12.1 "EchoTool 家族"）
# ======================================================================================


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@tool
def boom(msg: str = "boom") -> str:
    """Always raise ValueError (exception-path fixture)."""
    raise ValueError(msg)
