from __future__ import annotations

"""LLM 客户端的最小契约与公共基类（§6.3）。

为什么需要这一层：provider 的实现差异只应该体现在"请求体/响应体怎么拼、怎么解"，
而 **重试、退避、超时策略、事件发射、token 估算** 是每一家 provider 都一样的横切逻辑。
把这些放进 `BaseLLMClient` 之后，新增第三家 provider（见 `providers.DeepSeekChatClient`）
只需要 5 行代码 —— 这是"LLM 抽象层统一多模型 API"最直接的证据。

**环依赖规避（冻结）**：本模块**不得** import `agent/callbacks.py`，因此事件回调签名
是轻量的 `LowLevelEvent`（事件类型是字符串，如 `"llm_request"`），
`agent/callbacks.as_llm_callback(manager)` 是唯一的适配器（§2.7）。
"""

import asyncio
import logging
import math
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from liteagent.config import (
    DEFAULT_CJK_CHAR_COST,
    DEFAULT_TOKEN_CHAR_RATIO,
    LLMConfig,
    SleepFn,
    default_sleep,
    run_sync,
    to_jsonable,
)
from liteagent.errors import LiteAgentError, LLMTimeoutError
from liteagent.llm.message import Message
from liteagent.types import LLMResponse

if TYPE_CHECKING:  # pragma: no cover - 只为注解
    # 运行期**不** import transport：§1.1 的 L 表里 `llm/base.py` 的依赖是
    # errors/types/config/llm-message，没有 transport（那是 providers 的依赖）。
    # 规范明确放行 `if TYPE_CHECKING:` 里的 import（§1.3），因此注释保持与 §6.3
    # 冻结签名逐字一致，而依赖边不新增。
    from liteagent.llm.transport import Transport

__all__ = ["LLMClient", "BaseLLMClient", "LLMStreamChunk", "LowLevelEvent"]

T = TypeVar("T")

#: §2.7 的冻结定义式 `LowLevelEvent = Callable[[str, dict[str, Any]], None]`。
# SPEC-AMBIGUITY: §2.7 只给了定义式，没有指定它归哪个模块，而 config.py 里也没有这个名字
# （§6.3/§7.4/§10.2 都只写 `on_event: LowLevelEvent | None`）。裁决：按**字面**在本文件
# 定义一次并在 llm 包内复用 —— 类型别名在运行期不存在（只是注解糖），其它层自行定义结构
# 等价的别名不会冲突；而"从 config import"在今天会直接 ImportError 打死整个 llm 包。
LowLevelEvent = Callable[[str, dict[str, Any]], None]

_logger = logging.getLogger("liteagent.llm")

#: token 估算缓存的容量上限。缓存是为了让"同一个长 prompt 每轮都被估算"不反复扫字符串；
#: 有上限是为了让长驻进程不会因为历史消息不断变化而无界增长（超过就整体丢弃重来）。
_TOKEN_CACHE_MAX_ENTRIES = 2048

#: CJK 区间（与 memory/base.py 的 `CJK_RANGES`、`HeuristicTokenizer` 逐条一致，D-06）。
#: llm 层**不能** import memory 层（反向依赖），所以这里按值复制一份；两处对"CJK"的
#: 定义必须永远一致，改动时两边同时改。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
    (0xAC00, 0xD7AF),
    (0x3040, 0x30FF),
)


def _is_cjk_char(ch: str) -> bool:
    """`ch` 是否落在冻结的 CJK 区间内（按码点硬编码，不用 `unicodedata` 名字匹配）。"""
    code_point = ord(ch)
    for low, high in _CJK_RANGES:
        if low <= code_point <= high:
            return True
    return False


def _heuristic_tokens(text: str) -> int:
    """零依赖的 token 近似（D-06 混合估算）：CJK 1 字 1 token，其余 4 字符 1 token。

    纯 `len/4` 对中文严重低估（会让窗口超预算），引入 tiktoken 又会违反零依赖红线；
    代价是 ±20% 的误差，靠"预算留余量 + 允许注入 `CallableTokenizer`"缓解。
    **空文本 -> 0**（与 `HeuristicTokenizer("") == 0` 对齐；<1 的非空估算钳到 1）。
    """
    if text == "":
        return 0
    n_cjk = 0
    for ch in text:
        if _is_cjk_char(ch):
            n_cjk += 1
    n_other = len(text) - n_cjk
    estimate = math.ceil(n_cjk * DEFAULT_CJK_CHAR_COST + n_other / DEFAULT_TOKEN_CHAR_RATIO)
    return estimate if estimate >= 1 else 1


@dataclass
class LLMStreamChunk:
    """流式响应的一小片（§6.3）。`index` 从 0 递增，`finish_reason` 只在最后一片非 None。"""

    delta: str = ""
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    index: int = 0


class LLMClient(ABC):
    """所有 LLM 客户端的最小契约。实现者应继承 BaseLLMClient 而非直接继承本类。"""

    #: 该 provider 是否支持原生 function calling / tool use
    supports_tool_calling: bool = False
    #: 是否需要 api_key（echo/scripted 不需要）
    requires_api_key: bool = True

    @property
    def model(self) -> str:
        """模型名。默认从 `self.config` 取（`BaseLLMClient` 的子类都走这条路径）。"""
        config = getattr(self, "config", None)
        return getattr(config, "model", "") if config is not None else ""

    def count_tokens(self, text: str) -> int:
        """token 估算（默认走 heuristic，空串 -> 0）。

        只是"给人看的近似值"，**不用**做预算判定的唯一依据：memory 层有更强的
        `Tokenizer` 家族，预算判定走它（llm 层不能 import memory 层）。
        """
        if not text:
            return 0
        return _heuristic_tokens(text)

    def resolve_mode(self, *, has_tools: bool) -> str:
        """[v2 新增] `'native' if (self.supports_tool_calling and has_tools) else 'text'`。

        Agent 的 `mode='auto'` 就用这一个函数，保证"原生 vs 文本"的判定只有一处
        （§9.4.5）。分散判定是 v1 的 bug 来源：两处实现迟早会分叉。
        """
        return "native" if (self.supports_tool_calling and has_tools) else "text"

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发一次对话请求。失败必须以 `LiteAgentError` 子类抛出（可重试性见 §3.4）。"""
        raise NotImplementedError

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """`achat` 的同步镜像。

        收的是**工厂函数**而不是协程对象（§5.2 冻结）：如果先构造协程再抛
        `ConfigError`，那个协程永远不会被 await，3.10 会打
        `RuntimeWarning: coroutine ... was never awaited`，在 `-W error` 下直接失败。
        在已有运行中的 loop 里调用 -> `ConfigError`（由 `run_sync` 抛）。
        """
        return run_sync(lambda: self.achat(messages, **kwargs))

    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> AsyncIterator[LLMStreamChunk]:
        """流式版本。默认实现抛 `NotImplementedError`；调用方**必须**捕获并回退到 `achat`。

        [v2 变更] `ScriptedLLM` **必须**实现它（§6.6 冻结），否则流式路径零覆盖。
        真实 provider 是否实现由子类决定：`HTTPChatClient` 不实现，因为 `Transport`
        只提供"整段响应"，没有流级别的接口（伪流式只会假装在流）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement astream_chat; fall back to achat"
        )

    async def aclose(self) -> None:
        """释放底层资源（连接池等）。默认 no-op。"""
        return None


class BaseLLMClient(LLMClient):
    """提供：重试（RetryPolicy）、trace 事件发射、token 计数缓存、温度/最大 token 默认值合并。"""

    config: LLMConfig
    on_event: LowLevelEvent | None

    #: 超时后是否重发请求。基类默认 **True**（§3.4 的重试白名单里 `LLMTimeoutError`
    #: 的"默认是否重试"就是"是"）；`HTTPChatClient` 覆盖为 `False`，因为对真实 HTTP
    #: 端点无法确认请求是否已经到达服务端并产生副作用/计费（§6.4）。
    retry_on_timeout: bool = True

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: Transport | None = None,
        on_event: LowLevelEvent | None = None,
    ) -> None:
        """[v2 变更] `on_event` 类型统一为 `LowLevelEvent`。

        `transport` 在这里只是"存下来"：不使用 HTTP 的客户端（echo/scripted）不关心它。
        """
        self.config = config if config is not None else LLMConfig()
        self.transport = transport
        self.on_event = on_event
        # §5.5 规则 3：rng 与 sleep 在构造时**各解析一次**并复用整个生命周期。
        # 每次重试重建 rng 会让退避序列不可复现（D-07 的断言直接依赖可复现）。
        self._rng: random.Random = random.Random(self.config.retry_policy.rng_seed)
        self._sleep: SleepFn = (
            self.config.sleep_fn or self.config.retry_policy.sleep_fn or default_sleep
        )
        self._token_cache: dict[str, int] = {}
        # LLM_REQUEST 事件的必需键里有 messages_count / tools_count，而 `_with_retry` 的
        # 签名被 §6.3 冻结成 `(fn, *, what)`，没有位置传这两个数。因此由发起请求的一方
        # （各 `achat`）先调 `_prepare_request` 记下"本次请求画像"，重试时读取。
        self._request_messages_count = 0
        self._request_tools_count = 0

    def count_tokens(self, text: str) -> int:
        """带缓存的启发式估算（**空串 -> 0，且空串不进缓存**）。

        缓存按整段文本做键：ReAct 循环里 system prompt 与历史前缀在轮次之间几乎不变，
        命中率很高；文本一变就是一个新键，所以必须有容量上限（见模块常量）。
        """
        if not text:
            return 0
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached
        value = super().count_tokens(text)
        if len(self._token_cache) >= _TOKEN_CACHE_MAX_ENTRIES:
            # 简单粗暴但可预测：整体丢弃重来，不引入 LRU 依赖、也不做逐条淘汰。
            self._token_cache.clear()
        self._token_cache[text] = value
        return value

    def _resolve_params(
        self, temperature: float | None, max_tokens: int | None
    ) -> tuple[float | None, int | None]:
        """调用参数 -> 生效参数：**显式传入优先，否则回落到 config 的默认值**。

        顺序写死为"参数 > config"，让同一个 client 可以被不同 Agent 以不同温度复用。
        """
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        return resolved_temperature, resolved_max_tokens

    def _should_retry(self, exc: LiteAgentError, attempt: int) -> bool:
        """是否再试一次（§3.4 白名单 + `RetryPolicy.max_retries` + timeout 例外）。"""
        if not exc.retryable:
            # 无法自愈的错误（401/400/格式错）重试只是浪费时间与配额。
            return False
        if attempt >= self.config.retry_policy.max_retries:
            return False
        if isinstance(exc, LLMTimeoutError) and not self.retry_on_timeout:
            # 请求可能已经到达服务端并计费/产生副作用，重发不幂等（§6.4）。
            return False
        return True

    def _prepare_request(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None
    ) -> None:
        """记下"本次请求画像"，供 `_with_retry` 填充 LLM_REQUEST 事件的必需键。

        已知代价（诚实记录）：同一个 client 实例被多个 Agent 并发共享时，重试事件的
        counts 可能来自另一个并发请求。只影响事件的展示数据，不影响请求内容 ——
        换用参数传递会违反 §6.3 对 `_with_retry` 签名的冻结。
        """
        self._request_messages_count = len(messages)
        self._request_tools_count = 0 if tools is None else len(tools)

    async def _with_retry(self, fn: Callable[[], Awaitable[T]], *, what: str) -> T:
        """对 retryable 的 `LiteAgentError` 重试。

        退避用 `config.compute_backoff`（经 `RetryPolicy.delay_for`，`rng=self._rng`），
        等待走 `self._sleep`（§5.5 冻结：**禁止**直接 `await asyncio.sleep`，否则
        `RecordingSleep` 之类的测试替身失效、且真实测试会真的睡满退避时间）。

        事件：每次**尝试**发一条 `LLM_REQUEST`（`retry` 字段 = 已失败次数，首次为 0），
        每次失败发一条 `LLM_ERROR`。发射者是本函数（§2.7 的唯一发射者矩阵），
        Agent 侧不得重复发。
        """
        policy = self.config.retry_policy
        attempt = 0  # 已失败的次数；同时就是 LLM_REQUEST 的 retry 字段
        while True:
            self._emit(
                "llm_request",
                messages_count=self._request_messages_count,
                tools_count=self._request_tools_count,
                retry=attempt,
                model=self.model,
            )
            try:
                return await fn()
            except asyncio.CancelledError:
                # M-5：CancelledError 继承 BaseException，本来就不会被下面的 except 抓住。
                # 显式写出来是为了挡住"后来者把 except 改成 BaseException"时吞掉取消。
                raise
            except LiteAgentError as exc:
                self._emit(
                    "llm_error",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    retry=attempt,
                )
                if not self._should_retry(exc, attempt):
                    raise
                delay = policy.delay_for(attempt, rng=self._rng)
                retry_after = exc.retry_after_s
                if retry_after is not None:
                    # 服务端明确要求稍后再来（429 的 Retry-After）：尊重它，
                    # 但退避本身也不小于它 —— 取两者更大，避免"服务端说 5 秒、
                    # 本地退避 0.25 秒"的密集打点。
                    delay = max(delay, float(retry_after))
                attempt += 1
                _logger.debug(
                    "%s failed (%s); retry %d/%d in %.3fs",
                    what,
                    type(exc).__name__,
                    attempt,
                    policy.max_retries,
                    delay,
                )
                await self._sleep(delay)

    def _emit(self, event_type: str, **data: Any) -> None:
        """低层事件发射（§2.7 冻结签名）。

        实现就是一行转发：`data` 里的值**必须**已经过 `to_jsonable`，这里再过一次是
        为了兜底 —— `TraceEvent.to_json()` 里的 `json.dumps` 永不因类型而抛 `TypeError`。
        **严禁**在 data 里放 `TokenUsage` 实例（必须 `to_dict()`），见 `_emit_response`。
        """
        if self.on_event is None:
            return
        self.on_event(event_type, to_jsonable(data))

    def _emit_response(self, resp: LLMResponse) -> None:
        """发 `LLM_RESPONSE`（§6.3 冻结的 data 键）。

        `usage` 写成 `resp.usage.to_dict()`（3 个 int 的纯 dict）而不是 TokenUsage 实例：
        事件 data 的唯一约束就是"已经 JSON 可序列化"，dataclass 实例会破坏这一约束。
        """
        self._emit(
            "llm_response",
            content_len=len(resp.content),
            tool_calls=len(resp.tool_calls),
            finish_reason=resp.finish_reason,
            latency_ms=float(resp.latency_ms),
            usage=resp.usage.to_dict(),
        )

    async def aclose(self) -> None:
        """默认 no-op：不持有连接的客户端（echo）无需释放任何东西。"""
        return None
