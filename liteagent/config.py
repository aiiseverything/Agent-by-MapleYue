from __future__ import annotations

"""跨层基础设施：配置 dataclass + 跨层共享的纯函数（§1.1 的 L1，只依赖 errors/types）。

为什么不放在各自的包里：``run_sync`` / ``LoopBoundPool`` / ``utc_now`` 被 llm 与
tools **同时**需要，放在任意一方都会让另一方写"反向 import"；文件清单又是封闭的
（§1.2 共 41 个 ``.py``，不能新增文件），所以这里是唯一不会成环的最小公倍数。

三条贯穿全项目的冻结规则（也是本文件存在的理由）：

1. **一切退避等待走 ``sleep_fn``**（§5.5）：测试靠注入 ``RecordingSleep`` 断言退避
   序列，直接 ``await asyncio.sleep`` 会让测试真睡 8 秒。
2. **一切"取当前时间"走 ``utc_now()``**（§2.2）：否则 ``tests.helpers.frozen_time``
   冻结不了时间，带时间戳的断言全会 flaky。
3. **任何 asyncio 同步原语不得在 ``__init__`` / 模块级创建**（R-LOOP，§0.4）：
   一律经 :class:`LoopBoundPool` 在**运行中的 loop 内**懒创建，并由
   :func:`_run_and_cleanup` 在 ``finally`` 里 ``release_loop()`` —— 实测（M-3）
   不清理的话每次 ``asyncio.run`` 泄漏一个 loop 与一个原语
   （``asyncio.Semaphore`` 争用后会强引用 loop，弱引用字典救不了）。
"""

import asyncio
import base64
import json
import os
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Mapping,
    TypeVar,
)

from liteagent.errors import ConfigError, SerializationError

if TYPE_CHECKING:  # 仅供类型检查；运行期不需要（注解全部是字符串）
    from liteagent.types import TokenUsage

# ---------------------------------------------------------------------------
# 可选三方依赖（§1.3：顶层不得裸 import 三方库，一律 try/except + XXX_AVAILABLE）
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 环境相关
    import yaml as _yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境相关
    _yaml = None
    YAML_AVAILABLE = False

# 本模块**刻意不定义 `__all__`**：`__all__` 的冻结清单只覆盖各**包**的 `__init__.py`
# （§1.4），而 config.py 是"所有层都能拿的公共抽屉"——写一份白名单反而会
# 与 `liteagent/__init__.py` 的附录 B 清单形成第二个真值源。未列出的名字视为内部
# （靠单下划线前缀表达，§2.3）。

T = TypeVar("T")

# ===========================================================================
# §2.5 全局常量（冻结数值，实现者不得改动）
# ===========================================================================

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_THREAD_POOL_SIZE = 8
DEFAULT_TOOL_TIMEOUT_S = 30.0
DEFAULT_LLM_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_S = 0.25
DEFAULT_BACKOFF_MAX_S = 8.0
DEFAULT_BACKOFF_JITTER = 0.5
DEFAULT_MAX_RESULT_CHARS = 8000
DEFAULT_MAX_OBSERVATION_CHARS = 8000
DEFAULT_MAX_TRANSCRIPT = 1000
# 超时哨兵：显式禁用超时（见 §5.3 / §7.4.1 步骤 2）。
# 之所以需要它：v1 的候选值链里**没有任何取值**能表达"这个工具不要超时"，
# 传 None 会被链上的 `is not None` 过滤掉，从而静默退回 config 的默认超时。
NO_TIMEOUT = -1.0

# [v2 新增]
DEFAULT_MAX_TOTAL_TOKENS = 200000  # AgentConfig.max_total_tokens 的建议默认（None=不限）
DEFAULT_RESERVE_COMPLETION_TOKENS = 1024  # 上下文预算里给回复预留的 token
DEFAULT_MAX_TRUNCATION_RETRIES = 1  # finish_reason=="length" 的续写重试上限
DEFAULT_TOOL_FAILURE_LIMIT = 3  # 同一工具连续 infrastructure 失败多少次后熔断
DEFAULT_MAX_TOOLS_IN_PROMPT = 20  # 文本模式 system prompt 里最多渲染几个工具
DEFAULT_PERSIST_BATCH = 500  # save() 每批写多少条（内存友好）

# memory
DEFAULT_BUFFER_MAX_TOKENS = 3000
DEFAULT_BUFFER_MAX_MESSAGES = 50
DEFAULT_SUMMARY_TRIGGER_RATIO = 0.8
DEFAULT_SUMMARY_MIN_EVICT = 4
DEFAULT_MAX_SUMMARY_CHARS = 1200
DEFAULT_HASHING_EMBED_DIM = 256
DEFAULT_RECENCY_HALF_LIFE_D = 7.0
DEFAULT_W_SIM = 1.0
DEFAULT_W_RECENCY = 0.15
DEFAULT_W_IMPORTANCE = 0.1
DEFAULT_DEDUP_THRESHOLD = 0.95
DEFAULT_MAX_MEMORY_ITEMS = 10000
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_RETRIEVE_LIMIT = 5
DEFAULT_TOKEN_CHAR_RATIO = 4.0  # ASCII
DEFAULT_CJK_CHAR_COST = 1.0  # 每 CJK 字符算 1 token
DEFAULT_MEMORY_QUERY_CHARS = 500  # <relevant_memories> 单条 content 截断
DEFAULT_MEMORY_BLOCK_CHARS = 4000  # <relevant_memories> 整体截断

# agent
DEFAULT_PARSE_MAX_RETRIES = 2
DEFAULT_REPEAT_THRESHOLD = 2
DEFAULT_TEMPERATURE = 0.0

# multiagent
DEFAULT_TEAM_MAX_DEPTH = 3
DEFAULT_TEAM_MAX_ROUNDS = 10
DEFAULT_SUBAGENT_CONCURRENCY = 3
DEFAULT_SUBAGENT_MAX_CHARS = 2000
SUBAGENT_HEAD_RATIO = 0.7  # head+tail 压缩：头 70%

# 各 provider 的内置默认 base_url（``LLMConfig.resolved_base_url`` 的最后一档）。
# 为什么在 config.py 里再抄一份而不 import providers：providers 是 L2，
# config 是 L1，import 它就是反向依赖。两份值必须保持一致（providers 的
# ``default_base_url`` 类属性是权威定义，这里只是它的镜像）。
_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com/v1",
    # "openai-compatible" 必须显式给 base_url（vLLM/Ollama 没有默认地址）
    # "echo" 不联网，没有 base_url
}


# ===========================================================================
# §5.5 时间与随机性的注入点（为可测性冻结）
# ===========================================================================

#: 一切退避等待的注入点：``async def f(seconds: float) -> None``。
SleepFn = Callable[[float], Awaitable[None]]

#: HITL 审批回调：(call, tool) -> 是否放行。
#: 用字符串前向引用而不是 import：config 是 L1，tools 是 L3，运行期不能依赖它。
ApprovalPolicy = Callable[["ToolCall", "Tool"], bool]  # noqa: F821 - 见上


async def default_sleep(seconds: float) -> None:
    """默认的 sleep 实现。生产路径唯一允许出现 ``asyncio.sleep`` 的地方。"""
    await asyncio.sleep(seconds)


# ===========================================================================
# §5.1 RetryPolicy 与退避计算（被 executor 与 llm 复用）
# ===========================================================================


def compute_backoff(
    attempt: int,
    *,
    base_s: float = DEFAULT_BACKOFF_BASE_S,
    max_s: float = DEFAULT_BACKOFF_MAX_S,
    jitter: float = DEFAULT_BACKOFF_JITTER,
    rng: random.Random | None = None,
) -> float:
    """指数退避 + 抖动。``attempt`` 从 0 开始（第 1 次失败后调 ``compute_backoff(0)``）。

    ``full = min(max_s, base_s * 2 ** attempt)``，再按 ``jitter`` 分三档：

    * ``jitter <= 0``      -> ``full``（确定性，测试用）
    * ``jitter >= 1``      -> ``rng.uniform(0, full)``（full jitter）
    * ``0 < jitter < 1``   -> ``rng.uniform(1 - jitter, 1.0) * full``（equal jitter，默认 0.5）

    返回值 clamp 到 ``[0.0, max_s]`` —— full jitter 的上界理论上就是 full，
    但调用方可能传 ``jitter < 0`` 或 ``base_s > max_s``，显式 clamp 让契约闭合。

    ``rng`` 冻结规则（v1 完全未定义，三个实现都"符合文档"）：
    ``rng is None`` 等价于 ``random.Random(None)``（即不可复现），
    **禁止**使用模块级 ``random.random()`` / ``random.uniform()`` ——
    模块级全局状态会让并发重试互相扰动、测试无法复现（§5.5 规则 4）。
    调用点应显式传 ``rng=self._rng``（实例级，见 §5.5 规则 3）。
    """
    full = min(max_s, base_s * (2 ** attempt))
    if jitter <= 0.0:
        delay = full
    else:
        if rng is None:
            # 显式造一个不可复现的实例，而不是用模块级全局函数 —— 语义等价
            # 但在代码里可见，能被 code review 与 grep 抓到。
            rng = random.Random(None)
        if jitter >= 1.0:
            delay = rng.uniform(0.0, full)
        else:
            delay = rng.uniform(1.0 - jitter, 1.0) * full
    return max(0.0, min(delay, max_s))


@dataclass
class RetryPolicy:
    """重试与退避策略。被 ``ExecutorConfig`` 与 ``LLMConfig`` 共用（§5.1）。

    ``max_retries`` 是**额外**尝试次数：总尝试数 = ``1 + max_retries``。
    """

    max_retries: int = DEFAULT_MAX_RETRIES  # 额外尝试次数（总尝试 = 1 + max_retries）
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    backoff_max_s: float = DEFAULT_BACKOFF_MAX_S
    jitter: float = DEFAULT_BACKOFF_JITTER  # 0.0 <= jitter <= 1.0
    rng_seed: int | None = None  # 非 None 时退避可复现（测试用）
    sleep_fn: "SleepFn | None" = None  # [v2 新增] 见 §5.5

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """等价于 ``compute_backoff(attempt, base_s=..., max_s=..., jitter=..., rng=rng)``。

        **两个方法必须调用同一个函数（禁止各自手写）**：测试断言两者对同一组参数
        返回完全相同的值，任何"顺手优化一下"的分支都会让断言失败。
        """
        return compute_backoff(
            attempt,
            base_s=self.backoff_base_s,
            max_s=self.backoff_max_s,
            jitter=self.jitter,
            rng=rng,
        )

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出（§2.2）。``sleep_fn`` 是 callable，序列化为限定名或 None
        —— 保留字段让 trace 结构稳定，但诚实：可调用对象无法 JSON 往返，
        ``from_dict`` 忽略这一项（调用方必须在代码里显式注入 sleep_fn）。"""
        return {
            "max_retries": self.max_retries,
            "backoff_base_s": self.backoff_base_s,
            "backoff_max_s": self.backoff_max_s,
            "jitter": self.jitter,
            "rng_seed": self.rng_seed,
            "sleep_fn": _callable_name(self.sleep_fn),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetryPolicy":
        """容忍缺失字段（取默认值）与 ``sleep_fn`` 的字符串形式（丢弃 + 警告）。"""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                warnings.warn(f"RetryPolicy.from_dict: ignoring unknown key {key!r}", stacklevel=2)
                continue
            if key == "sleep_fn":
                if value is not None:
                    warnings.warn(
                        "RetryPolicy.from_dict: sleep_fn cannot be restored from data; "
                        "inject it explicitly (got %r)" % (value,),
                        stacklevel=2,
                    )
                continue
            kwargs[key] = value
        return cls(**kwargs)

def _callable_name(fn: Any) -> str | None:
    """把（可能不存在的）可调用对象表示成可 JSON 序列化的字符串。"""
    if fn is None:
        return None
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _env_int(name: str, raw: str, default: int) -> int:
    """解析环境变量里的整数；坏值 -> 默认值 + 警告（降级必须可观测，§13 红线 12）。"""
    try:
        return int(raw)
    except (TypeError, ValueError):
        warnings.warn(f"{name}={raw!r} is not an int; falling back to {default!r}", stacklevel=2)
        return default


# ===========================================================================
# §5.2 辅助纯函数
# ===========================================================================


def run_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """在同步上下文跑协程。

    * 当前线程无运行中的 loop -> :func:`_run_and_cleanup`
    * 已有运行中的 loop -> ``ConfigError``（**绝不** ``run_until_complete`` 套娃 /
      ``nest_asyncio``：那会让同一个 loop 重入，把并发语义彻底破坏）

    **签名冻结为收工厂函数**，不收协程对象。理由：调用方若写
    ``run_sync(self.achat(...))``，协程对象在检查之前就已构造，抛 ``ConfigError``
    后该协程永远不会被 await，3.10 会打
    ``RuntimeWarning: coroutine ... was never awaited`` —— 污染测试输出
    （``-W error`` 下直接失败）。**所有调用点必须写成 ``run_sync(lambda: self.achat(...))``。**
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的 loop —— 这是正常的同步入口路径
        return _run_and_cleanup(factory)
    # 先检查、后构造协程：factory 不会被调用，所以不会有"从未被 await 的协程"
    raise ConfigError(
        "cannot call sync API from a running event loop; use the 'a*' variant"
    )


def _run_and_cleanup(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """内部：新建 loop -> 跑协程 -> 在 ``finally`` 里
    ``LoopBoundPool.release_loop(loop)``。

    ``ToolExecutor.execute_sync`` / ``Agent.run`` 等同步入口共用它，这是"连续 3 次同步
    调用后池内条数为 0"成立的前提 —— 也是"谁负责清理 per-loop 原语"这个问题的唯一答案。
    对应的回归测试是 `tests/test_tools_executor.py::ExecuteSyncTests` 里那条
    "每次 execute_sync 之后 `len(executor._pool) == 0`"（`[v3 修正]` 上一版这里引用的
    ``test_cross_run_no_leak`` 在仓库里**并不存在** —— 规范文档与实现 docstring
    共同引用了一个不存在的测试作为证据）。

    为什么不用 ``asyncio.run``：``asyncio.run`` 内部自建 loop 且返回时已 close，
    调用方拿不到 loop 对象，而 ``release_loop`` 需要一个可作 key 的 loop。
    这里复刻 ``asyncio.run`` 的收尾动作（取消遗留任务 / shutdown asyncgens /
    shutdown 默认 executor），并额外拿到 loop 做清理。
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(factory())
    finally:
        try:
            _cancel_all_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except BaseException as exc:  # pragma: no cover - 收尾失败不应掩盖主异常
            _note(f"_run_and_cleanup: loop shutdown raised: {exc!r}")
        finally:
            try:
                # 释放该 loop 的全部 asyncio 原语与私有线程池（R-LOOP / M-3）
                LoopBoundPool.release_loop(loop)
            finally:
                asyncio.set_event_loop(None)
                loop.close()


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """复刻 stdlib ``asyncio.runners._cancel_all_tasks``：取消并回收残留任务。

    没有它，被遗忘的 task 会在 ``loop.close()`` 时打
    "Task was destroyed but it is pending"，污染测试输出。
    """
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return
    for task in to_cancel:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))
    for task in to_cancel:
        if task.cancelled():
            continue
        # 已经抛出过异常的 task 必须显式取回异常，否则打 "exception was never retrieved"
        if task.exception() is not None:
            _note(f"_run_and_cleanup: unretrieved exception from task {task!r}")


def _note(message: str) -> None:
    """清理路径上的"留痕"（§13 红线 10：禁止吞异常而不记录）。

    为什么不用 ``warnings.warn``：清理发生在 ``finally`` 里，而 ``-W error`` 下
    ``warn`` 会**变成异常**，把主异常顶掉、还跳过必须执行的释放动作。
    这里直接写 stderr —— 同样可观测，但永远不会抛。
    """
    sys.stderr.write(f"liteagent.config: {message}\n")


def utc_now() -> float:
    """**唯一时钟**（§2.2）：Unix 秒（``float``）。

    实现就是 ``time.time()``；命名保留 UTC 语义只为可读性。
    L1 及以上所有模块里凡是"取当前时间"的地方（含所有 ``default_factory``）
    都必须走它，这样 ``tests.helpers.frozen_time`` 才能通过 patch
    ``liteagent.config.utc_now`` 冻结时间。
    """
    return time.time()


#: 测试钩子：非 None 时 :func:`frozen_now` 返回它。
_FROZEN_NOW: float | None = None


def frozen_now() -> float:  # 仅供测试覆盖，不在 __all__
    """被冻结的时间；未冻结时等价 :func:`utc_now`。

    ``frozen_now`` 与 ``utc_now`` 是两条独立的注入路径：§12.1 的 ``frozen_time``
    patch 的是 ``utc_now``（模块属性查找），而这里读的是模块变量 ``_FROZEN_NOW``，
    测试可以直接 ``config._FROZEN_NOW = 123.0`` 得到"永远不推进的时钟"。
    """
    return _FROZEN_NOW if _FROZEN_NOW is not None else time.time()


def format_ts(ts: float, *, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """把 Unix 秒格式化成可读字符串（UTC）。

    用 ``datetime.timezone.utc`` 而不是 ``datetime.UTC``：后者是 3.11 新增（§0.1）。
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


def to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / Enum / set / tuple / Path / bytes / Exception 转成 JSON 可序列化结构。

    * ``dataclass`` -> ``{f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}``
    * ``Enum`` -> ``obj.value``；``set`` / ``tuple`` / ``frozenset`` -> ``list``
    * ``Path`` -> ``str``；``bytes`` -> base64 ``str``
    * ``Exception`` -> ``{"type": ..., "message": ...}``
    * 未知类型 -> ``{"repr": repr(obj), "unserializable": True}``（留痕，禁止静默丢弃）

    **重要**：``to_jsonable(MemoryItem)`` 的输出**包含** ``embedding`` —— 它不感知
    opt-out（§2.2 的三个显式开关只在对象的 ``to_dict`` 上生效）。要省略请显式调
    ``obj.to_dict(include_embedding=False)`` 再过 ``to_jsonable``。

    §2.7 要求"进入事件 ``data`` 的值必须已过 ``to_jsonable``"，所以这是事件管线的
    最后一道防线：**返回值必须能被 ``json.dumps`` 接受**（未知类型也不能抛）。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, BaseException):
        # 只输出 type/message（§5.2 冻结）；异常的额外字段由各异常的 to_dict 负责。
        return {"type": type(obj).__name__, "message": str(obj)}
    # 未知类型：不抛异常（事件管线不能因为一个奇怪的对象整条挂掉），
    # 但必须留下可观测痕迹，禁止静默（§13 红线 12）。
    return {"repr": repr(obj), "unserializable": True}


def truncate_head_tail(
    text: str,
    max_chars: int,
    *,
    head_ratio: float = 0.7,
    marker: str = "\n...[truncated {n} chars]...\n",
) -> str:
    """头尾保留式截断：``max_chars<=0`` 或 ``len(text)<=max_chars`` 时原样返回。

    否则保留前 ``int(max_chars*head_ratio)`` 与后 ``max_chars - 前`` 个字符，
    中间插入 ``marker``（``{n}`` 替换为被丢弃的字符数）。

    为什么头多尾少：LLM 的输入通常"头是 prompt/结构，尾是刚发生的事"，
    中间是最可牺牲的细节。工具结果、子 Agent 输出（``SUBAGENT_HEAD_RATIO``）、
    观察值截断都复用这一个函数，保证截断行为只有一处定义。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    ratio = min(1.0, max(0.0, head_ratio))
    head_chars = int(max_chars * ratio)
    tail_chars = max_chars - head_chars
    dropped = len(text) - max_chars
    head = text[:head_chars]
    tail = text[len(text) - tail_chars:] if tail_chars > 0 else ""
    return head + marker.replace("{n}", str(dropped)) + tail


def parse_dotenv(text: str) -> dict[str, str]:
    """手写 ``.env`` 解析（本环境没有 ``python-dotenv``，§0.2 实测不可用）。

    规则：
    * 跳过空行与 ``#`` 注释行
    * ``KEY=VALUE``；键两侧空白剥除；没有 ``=`` 的行跳过
    * 支持 ``export`` 前缀（``export KEY=VALUE`` 等价 ``KEY=VALUE``）
    * 值两侧的单/双引号剥离；双引号内支持 ``\\n`` / ``\\t`` / ``\\\\`` / ``\\"`` 转义
    * **不支持**变量插值（``$HOME`` 原样保留）
    * 非引号值尾部形如 `` # 注释`` 的片段按注释剥除（与 python-dotenv 一致）

    返回解析结果（不写 ``os.environ``；写入由 :func:`load_dotenv` 负责）。
    """
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export"):].strip()
        if "=" not in line:
            # 不是 KEY=VALUE 的行：静默跳过（容忍 .env 里混入的说明文字），
            # 这是刻意的宽容而不是吞异常 —— 没有异常被吞。
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        result[key] = _parse_dotenv_value(value.strip())
    return result


def _parse_dotenv_value(value: str) -> str:
    """解析单个 ``.env`` 值：引号剥离与转义处理（:func:`parse_dotenv` 的私有部分）。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            return _unescape_double_quoted(inner)
        # 单引号：字面量，不做任何转义（与 shell 语义一致）
        return inner
    # 非引号值：剥掉尾部注释（仅当 '#' 前有空白，避免破坏 URL 里的 '#'）
    hash_index = value.find(" #")
    if hash_index == -1:
        hash_index = value.find("\t#")
    if hash_index != -1:
        value = value[:hash_index].rstrip()
    return value


def _unescape_double_quoted(inner: str) -> str:
    """处理双引号内的转义序列；未知转义（如 ``\\d``）原样保留反斜杠。"""
    out: list[str] = []
    idx = 0
    while idx < len(inner):
        char = inner[idx]
        if char == "\\" and idx + 1 < len(inner):
            nxt = inner[idx + 1]
            if nxt == "n":
                out.append("\n")
                idx += 2
                continue
            if nxt == "t":
                out.append("\t")
                idx += 2
                continue
            if nxt == "r":
                out.append("\r")
                idx += 2
                continue
            if nxt in ('"', "\\", "'"):
                out.append(nxt)
                idx += 2
                continue
            # 未知转义：保留原文，不猜
            out.append(char)
            idx += 1
            continue
        out.append(char)
        idx += 1
    return "".join(out)


def load_dotenv(
    path: str | os.PathLike[str] = ".env", *, override: bool = False
) -> dict[str, str]:
    """读 ``.env`` 并写入 ``os.environ``，返回解析结果。

    文件不存在 -> 返回 ``{}``（**不抛异常**：``.env`` 是可选的开发便利，
    缺它不该让 CLI 挂掉）。其它读取错误（权限/编码）-> 警告 + 返回 ``{}``
    （降级必须可观测，§13 红线 12）。
    ``override`` 为 False 时不覆盖已存在的环境变量（真实环境变量优先）。
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            parsed = parse_dotenv(handle.read())
    except FileNotFoundError:
        return {}
    except OSError as exc:
        warnings.warn(f"load_dotenv: cannot read {path!s}: {exc}", stacklevel=2)
        return {}
    except UnicodeDecodeError as exc:  # pragma: no cover - 环境相关
        warnings.warn(f"load_dotenv: {path!s} is not valid UTF-8: {exc}", stacklevel=2)
        return {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY = frozenset({"0", "false", "no", "off", "", "n", "f"})


def parse_bool(value: str | bool | None, *, default: bool = False) -> bool:
    """``'1'/'true'/'yes'/'on'``（大小写不敏感）-> True；
    ``'0'/'false'/'no'/'off'/''`` -> False；其它值（含 None）-> ``default``。

    注意空串是**明确 False** 而不是 ``default``：``LITEAGENT_ALLOW_NETWORK=""``
    应该关掉联网，而不是退回"默认开着"。
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default


class _SafeDict(dict[str, Any]):
    """``format_map`` 用的字典：缺 key 时返回字面量 ``{key}`` 而不是抛 ``KeyError``。

    用户自定义的 ``system_prompt_template`` 很可能带我们不认识的占位符
    （或带 ``{}`` 的 JSON 片段），缺 key 直接炸会让框架在用户手上很难用。
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: str, values: Mapping[str, Any]) -> str:
    """用 ``template.format_map(_SafeDict(values))`` 渲染模板（§5.3 冻结规则）。

    传进来的 ``tools`` 值已经是渲染好的**多行文本**，因此工具 description 里的
    ``{}`` 不会被二次解析 —— 二次解析是模板渲染最经典的坑（用户写了一个
    ``Action Input: {"a": 1}`` 的示例就炸了）。
    """
    return template.format_map(_SafeDict(values))


DEFAULT_REACT_SYSTEM_PROMPT = """\
You are {name}, a helpful AI agent that solves tasks step by step.

You have access to the following tools:
{tools}

Use the following format:

Thought: your reasoning about what to do next
Action: the name of the tool to use, one of [{tool_names}]
Action Input: the arguments, as a JSON object
Observation: the result of the tool (this is filled in for you, do not write it yourself)
... (Thought/Action/Action Input/Observation may repeat)
Thought: I now know the final answer
Final Answer: the final answer to the user

Rules:
- Emit exactly ONE Action per step. Wait for the Observation before the next Thought.
- Action Input must be a single-line JSON object matching the tool's parameters.
- Never invent tool names. Available tools: [{tool_names}]
- If your output is truncated, continue from where you stopped.
- When you have enough information, stop calling tools and emit "Final Answer: ...".
"""


MODEL_PRICES: dict[str, tuple[float, float]] = {
    # 每 1k token 的 (输入价, 输出价)，美元。[v2 新增]
    # 冻结说明：这是**示例价格表**，用于让 `AgentResult.metadata["cost_usd"]` 有值可展示；
    # 价格会变，生产使用前请自行更新。未列出的模型一律返回 None（诚实 > 猜）。
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-5-haiku": (0.0008, 0.004),
    "deepseek-chat": (0.00027, 0.0011),
}


def estimate_cost_usd(usage: "TokenUsage", *, model: str) -> float | None:
    """按 :data:`MODEL_PRICES` 估算成本（美元）。

    公式：``prompt_tokens/1000 * price_in + completion_tokens/1000 * price_out``。
    ``model`` 未命中价格表 -> ``None``。**返回 None 而不是 0.0** 是刻意的：
    0.0 会被读成"这次调用免费"，None 才是"不知道"。
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    price_in, price_out = price
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt_tokens / 1000.0 * price_in + completion_tokens / 1000.0 * price_out


# ===========================================================================
# §5.3 配置 dataclass（全部字段有默认值，保证 XxxConfig() 可无参构造）
# ===========================================================================


def _known_fields(cls: type) -> set[str]:
    """``dataclasses.fields`` 的名字集合（``from_dict`` 的容忍度判定用）。"""
    return {f.name for f in fields(cls)}  # type: ignore[arg-type]


def _warn_unknown(cls: type, data: Mapping[str, Any]) -> None:
    """未知配置键 -> 警告。

    为什么不抛异常：配置来源（JSON/YAML/CLI）可能比代码版本新，
    抛异常会让"多写一个无害字段"直接炸掉运行。为什么不静默：写错键名
    （``max_step`` vs ``max_steps``）如果静默，用户会花一小时 debug 配置没生效。
    """
    unknown = sorted(set(data) - _known_fields(cls))
    if unknown:
        warnings.warn(
            f"{cls.__name__}.from_dict: ignoring unknown keys {unknown}", stacklevel=3
        )


def _as_mapping(cls: type, key: str, value: Any) -> Mapping[str, Any]:
    """``from_dict`` 的嵌套字段必须是 Mapping，否则抛 ``SerializationError``。"""
    if not isinstance(value, Mapping):
        raise SerializationError(
            target=f"{cls.__name__}.{key}",
            message=f"expected a mapping, got {type(value).__name__}",
        )
    return value


@dataclass
class LLMConfig:
    """LLM 层配置（§5.3）。所有字段有默认值，``LLMConfig()`` 可用。"""

    provider: str = "openai"  # "openai" | "anthropic" | "openai-compatible"
    # | "deepseek" [v2 新增] | "echo"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    temperature: float | None = None
    max_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    sleep_fn: "SleepFn | None" = None  # [v2 新增] 覆盖 retry_policy.sleep_fn

    def resolve_api_key(self) -> str | None:
        """优先级：``self.api_key`` -> ``LITEAGENT_API_KEY`` ->
        ``<PROVIDER>_API_KEY``（provider 大写、``-`` -> ``_``）。

        每次调用都重新读环境变量（**不缓存**）：测试经常用 ``patch.dict(os.environ)``
        临时换 key，缓存会让它们互相污染。
        """
        if self.api_key:
            return self.api_key
        generic = os.environ.get("LITEAGENT_API_KEY")
        if generic:
            return generic
        env_name = f"{self.provider.upper().replace('-', '_')}_API_KEY"
        return os.environ.get(env_name) or None

    def resolved_base_url(self) -> str | None:
        """``self.base_url`` -> ``LITEAGENT_BASE_URL`` -> 各 provider 的内置默认。

        内置默认表见 ``_DEFAULT_BASE_URLS``；``openai-compatible`` 与 ``echo``
        没有默认值（前者必须由用户显式指定，后者不联网）。
        """
        if self.base_url:
            return self.base_url
        env_url = os.environ.get("LITEAGENT_BASE_URL")
        if env_url:
            return env_url
        return _DEFAULT_BASE_URLS.get(self.provider)

    def to_dict(self) -> dict[str, Any]:
        """字段全量输出，但 ``api_key`` **脱敏**（有值 -> ``"***"``，无值 -> ``None``）。

        这是 §2.2 三个显式 opt-out 之外的**第四个**脱敏点，理由与它们不同：
        不是省体积，而是防止 key 顺着 trace/JSONL/CLI ``--json`` 泄漏到日志里。
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "retry_policy": self.retry_policy.to_dict(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_headers": dict(self.extra_headers),
            "extra_body": dict(self.extra_body),
            "stream": self.stream,
            "sleep_fn": _callable_name(self.sleep_fn),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMConfig":
        """``to_dict`` 的逆操作（``api_key="***"`` 视作"已设置但值未知"）。

        ``sleep_fn`` / ``retry_policy.sleep_fn`` 是 callable，无法从数据恢复，
        这里丢弃并警告 —— 调用方要在代码里显式注入。
        """
        _warn_unknown(cls, data)
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("retry_policy",):
                continue
            if key == "sleep_fn":
                if value is not None:
                    warnings.warn(
                        "LLMConfig.from_dict: sleep_fn cannot be restored from data; "
                        "inject it explicitly",
                        stacklevel=2,
                    )
                continue
            if key in _known_fields(cls):
                kwargs[key] = value
        if "api_key" in kwargs and kwargs["api_key"] == "***":
            # 脱敏值不能当真实 key 用（否则会拿着 "***" 去发请求）
            kwargs["api_key"] = None
        if "retry_policy" in data:
            raw_policy = _as_mapping(cls, "retry_policy", data["retry_policy"])
            kwargs["retry_policy"] = RetryPolicy.from_dict(raw_policy)
        return cls(**kwargs)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从 ``LITEAGENT_PROVIDER`` / ``LITEAGENT_MODEL`` 读取 provider 与 model。

        **刻意不读 API key**：§2.6 把 ``LITEAGENT_API_KEY`` 的读取者指派给
        ``resolve_api_key``（每次调用动态解析），在 ``from_env`` 里固化反而会让
        "先 from_env 后改环境变量"的测试失效。key/base_url 的解析留到使用时。
        """
        config = cls()
        provider = os.environ.get("LITEAGENT_PROVIDER")
        if provider:
            config.provider = provider.strip()
        model = os.environ.get("LITEAGENT_MODEL")
        if model:
            config.model = model.strip()
        return config


@dataclass
class MemoryConfig:
    """记忆层配置（跨层：``MemoryManager`` 与各层 ``Config`` 的映射见 §8.6）。"""

    buffer_max_tokens: int = DEFAULT_BUFFER_MAX_TOKENS
    buffer_max_messages: int = DEFAULT_BUFFER_MAX_MESSAGES
    buffer_keep_last_n: int = 2
    summary_enabled: bool = True
    summary_trigger_ratio: float = DEFAULT_SUMMARY_TRIGGER_RATIO
    summary_min_evict: int = DEFAULT_SUMMARY_MIN_EVICT
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    long_term_enabled: bool = True
    embedder_dim: int = DEFAULT_HASHING_EMBED_DIM
    write_policy: str = "selective"  # "selective" | "turn" | "manual"
    auto_write_min_chars: int = 40
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    max_items: int = DEFAULT_MAX_MEMORY_ITEMS
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    retrieve_min_score: float = 0.0
    w_sim: float = DEFAULT_W_SIM
    w_recency: float = DEFAULT_W_RECENCY
    w_importance: float = DEFAULT_W_IMPORTANCE
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_D
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    token_char_ratio: float = DEFAULT_TOKEN_CHAR_RATIO
    cjk_char_cost: float = DEFAULT_CJK_CHAR_COST
    # [v2 新增] 上下文窗口反推预算（D-15）
    context_window_tokens: int | None = None  # 非 None 时反推 buffer_max_tokens
    reserve_completion_tokens: int = DEFAULT_RESERVE_COMPLETION_TOKENS
    tools_schema_tokens_reserve: int = 0
    persist_path: str | None = None  # 非 None 时 MemoryManager 启动即 restore

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryConfig":
        _warn_unknown(cls, data)
        return cls(**{k: v for k, v in data.items() if k in _known_fields(cls)})


@dataclass
class ExecutorConfig:
    """工具执行器配置（§7.4）。**定义在 config.py**（不是 tools/executor.py）。"""

    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    # [v2 变更] default_timeout_s 的语义写死为：None = **对所有工具禁用超时**；
    # 想给单个工具禁用超时请用 NO_TIMEOUT（§7.4.1 步骤 2）。
    default_timeout_s: float | None = DEFAULT_TOOL_TIMEOUT_S
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    thread_pool_size: int = DEFAULT_THREAD_POOL_SIZE  # 每个 loop 的**私有**线程池大小
    sequential_tools: frozenset[str] = frozenset()  # 这些工具永不并发执行（跨线程生效）
    fail_fast: bool = False
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    allow_retry_on_non_idempotent: bool = False
    sleep_fn: "SleepFn | None" = None  # [v2 新增] 优先于 retry_policy.sleep_fn
    approval_policy: "ApprovalPolicy | None" = None  # [v2 新增] HITL，见 §7.4.1 步骤 4.5
    disable_tool_after_failures: int = DEFAULT_TOOL_FAILURE_LIMIT  # [v2 新增] 0 = 关闭熔断

    def to_dict(self) -> dict[str, Any]:
        """``sequential_tools`` 是 frozenset，用 :func:`to_jsonable` 统一成有序 list
        （排序是为了让 trace 抖动消失：同一份配置每次输出必须逐字相同）。"""
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "retry_policy":
                payload[f.name] = self.retry_policy.to_dict()
            elif f.name == "sequential_tools":
                payload[f.name] = sorted(value)
            elif f.name in ("sleep_fn", "approval_policy"):
                payload[f.name] = _callable_name(value)
            else:
                payload[f.name] = to_jsonable(value)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutorConfig":
        _warn_unknown(cls, data)
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("sleep_fn", "approval_policy"):
                if value is not None:
                    warnings.warn(
                        f"ExecutorConfig.from_dict: {key} cannot be restored from data; "
                        "inject it explicitly",
                        stacklevel=2,
                    )
                continue
            if key == "retry_policy":
                raw_policy = _as_mapping(cls, "retry_policy", value)
                kwargs["retry_policy"] = RetryPolicy.from_dict(raw_policy)
                continue
            if key == "sequential_tools":
                if not isinstance(value, (list, tuple, set, frozenset)):
                    raise SerializationError(
                        target="ExecutorConfig.sequential_tools",
                        message="expected a list of tool names",
                    )
                kwargs["sequential_tools"] = frozenset(str(v) for v in value)
                continue
            if key in _known_fields(cls):
                kwargs[key] = value
        return cls(**kwargs)

    def resolved_sleep(self, *, fallback: SleepFn | None = None) -> SleepFn:
        """冻结的解析优先级（§5.5 规则 2）：
        ``config.sleep_fn`` > ``retry_policy.sleep_fn`` > ``default_sleep``。

        调用点在 ``__init__`` 里**解析一次**并存成 ``self._sleep``；
        每次重试重新解析会让"运行中替换策略"变成隐式行为。
        """
        return self.sleep_fn or self.retry_policy.sleep_fn or fallback or default_sleep


@dataclass
class AgentConfig:
    """ReAct 状态机配置（**类定义在 config.py**，``agent/state.py`` 只 re-export）。"""

    max_steps: int = DEFAULT_MAX_STEPS
    mode: str = "auto"  # "auto" | "native" | "text"
    tool_choice: str | dict[str, Any] = "auto"
    temperature: float | None = DEFAULT_TEMPERATURE
    max_tokens: int | None = None
    parallel_tool_calls: bool = True
    max_parse_retries: int = DEFAULT_PARSE_MAX_RETRIES
    repeat_action_policy: str = "nudge_then_fail"  # "off"|"nudge"|"fail"|"nudge_then_fail"
    repeat_action_threshold: int = DEFAULT_REPEAT_THRESHOLD
    raise_on_error: bool = False
    system_prompt: str | None = None  # 非 None 时完全替换默认模板
    system_prompt_template: str = DEFAULT_REACT_SYSTEM_PROMPT
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    max_transcript_messages: int = DEFAULT_MAX_TRANSCRIPT
    include_thought_in_history: bool = True
    name: str = "agent"
    description: str = ""
    # [v2 新增] 成本与循环防护（面试必问的三件事）
    max_total_tokens: int | None = DEFAULT_MAX_TOTAL_TOKENS  # None = 不限
    max_prompt_tokens: int | None = None  # 单次 prompt 上限
    max_wall_clock_s: float | None = None  # 整轮墙钟上限
    max_truncation_retries: int = DEFAULT_MAX_TRUNCATION_RETRIES

    def to_dict(self) -> dict[str, Any]:
        """``system_prompt_template`` 是几百字符的模板，输出会污染 trace ——
        但仍然全量输出（§2.2 字段全量），只是模板字段照原样给出，
        这是"字段全量"与"可读性"之间的冻结取舍：不静默丢字段。"""
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        _warn_unknown(cls, data)
        return cls(**{k: v for k, v in data.items() if k in _known_fields(cls)})

    def render_system_prompt(self, *, name: str, tools_text: str, tool_names: str) -> str:
        """``system_prompt`` 非 None 时完全替换默认模板，否则渲染 ``system_prompt_template``。

        只做**模板渲染**这一件事，不碰工具列表的组装（那是 agent 层的业务逻辑）：
        ``tools_text`` 必须是已经渲染好的多行文本，``tool_names`` 是逗号分隔的名字。
        ``render_template`` 缺 key 不报错，所以用户写错占位符不会炸。
        """
        if self.system_prompt is not None:
            return self.system_prompt
        return render_template(
            self.system_prompt_template,
            {"name": name, "tools": tools_text, "tool_names": tool_names},
        )


@dataclass
class TeamConfig:
    """多 Agent 协作配置（``multiagent/base.py`` 只 re-export 本类）。"""

    max_depth: int = DEFAULT_TEAM_MAX_DEPTH
    max_rounds: int = DEFAULT_TEAM_MAX_ROUNDS
    parallel_subagents: bool = True
    subagent_concurrency: int = DEFAULT_SUBAGENT_CONCURRENCY
    share_memory: bool = False
    compress_subagent_output: bool = True
    subagent_output_max_chars: int = DEFAULT_SUBAGENT_MAX_CHARS
    propagate_failure: str = "return"  # "return" | "raise" | "continue"
    enable_cycle_detection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeamConfig":
        _warn_unknown(cls, data)
        return cls(**{k: v for k, v in data.items() if k in _known_fields(cls)})


_NESTED_CONFIGS: dict[str, type] = {
    "llm": LLMConfig,
    "agent": AgentConfig,
    "memory": MemoryConfig,
    "executor": ExecutorConfig,
    "team": TeamConfig,
}


@dataclass
class AppConfig:
    """应用级配置：把五份子配置 + 运行期开关打成一包（CLI ``--config`` 的载体）。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    tools: list[str] = field(default_factory=list)  # 空 = 注册全部内置工具，见 §7.5
    trace_file: str | None = None
    sandbox_root: str | None = None
    verbose: bool = False

    # ---- 构造 ----

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        """嵌套 dict -> 各子配置；未知键警告并忽略（见 :func:`_warn_unknown`）。"""
        if not isinstance(data, Mapping):
            raise SerializationError(
                target="AppConfig",
                message=f"expected a mapping, got {type(data).__name__}",
            )
        _warn_unknown(cls, data)
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            nested = _NESTED_CONFIGS.get(key)
            if nested is not None:
                kwargs[key] = nested.from_dict(_as_mapping(cls, key, value))
            elif key in _known_fields(cls):
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """读 JSON 配置；内容非法 -> ``ConfigError``（带路径，便于定位）。"""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {os.fspath(path)}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {os.fspath(path)}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read config file {os.fspath(path)}: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """读 YAML 配置；需要 PyYAML，不可用时抛
        ``ConfigError('PyYAML not installed; use JSON config')``（§1.3 的可选依赖红线）。"""
        if not YAML_AVAILABLE:
            raise ConfigError("PyYAML not installed; use JSON config")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = _yaml.safe_load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {os.fspath(path)}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read config file {os.fspath(path)}: {exc}") from exc
        except Exception as exc:  # yaml.YAMLError 的类名不在此处 import（可选依赖）
            raise ConfigError(f"invalid YAML in {os.fspath(path)}: {exc}") from exc
        if payload is None:
            payload = {}
        return cls.from_dict(payload)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """按扩展名分派：``.json`` / ``.yaml`` / ``.yml`` / ``.toml``（-> ConfigError）。

        ``.toml`` 单独报错而不是"未知扩展名"，因为 3.10 没有 ``tomllib``（§0.1），
        用户看到 "unsupported extension" 会以为是拼写问题。
        """
        suffix = Path(os.fspath(path)).suffix.lower()
        if suffix == ".json":
            return cls.from_json(path)
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if suffix == ".toml":
            raise ConfigError(
                "TOML config is not supported on Python 3.10 (no tomllib); use JSON or YAML"
            )
        raise ConfigError(
            f"unsupported config extension {suffix!r}; expected .json/.yaml/.yml"
        )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量构造（§2.6 的读取者归属表严格照抄）。

        只读 ``LITEAGENT_PROVIDER`` / ``LITEAGENT_MODEL`` / ``LITEAGENT_MAX_STEPS``。
        ``LITEAGENT_TRACE`` 的读取者是 ``cli``（透传）；``LITEAGENT_SANDBOX_ROOT`` 的
        读取者是 ``liteagent/tools/builtin/__init__.py`` 的 ``register_all`` /
        ``_resolve_sandbox_root``（优先级：显式参数 > 环境变量 > ``os.getcwd()``，
        最终构造 ``PathSandbox`` 传给 ``make_file_tools``）。
        `[v3 修正]` 上一版这里写的是"``LITEAGENT_SANDBOX_ROOT`` 的读取者是 cli 与
        ``builtin/files.py``" —— 两处都不对：``cli.py`` 只是透传 ``args.sandbox_root``，
        从不直接读该变量；``builtin/files.py`` 对它**零引用**，而且它**不该**读
        （§7.5 冻结了 ``make_file_tools(sandbox)`` 必填、``None`` -> ``ConfigError``；
        若 files.py 自己回退到 cwd，在仓库根跑测试时 ``delete_file`` 会真的删项目文件）。
        本方法**不越权**读取它们的值 —— 同一个环境变量有两个读取者时，
        行为会随调用顺序变化。
        """
        config = cls()
        config.llm = LLMConfig.from_env()
        raw_steps = os.environ.get("LITEAGENT_MAX_STEPS")
        if raw_steps:
            config.agent.max_steps = _env_int(
                "LITEAGENT_MAX_STEPS", raw_steps, config.agent.max_steps
            )
        return config

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        """全部字段输出；``llm.api_key`` 已被 :meth:`LLMConfig.to_dict` 脱敏。"""
        return {
            "llm": self.llm.to_dict(),
            "agent": self.agent.to_dict(),
            "memory": self.memory.to_dict(),
            "executor": self.executor.to_dict(),
            "team": self.team.to_dict(),
            "tools": list(self.tools),
            "trace_file": self.trace_file,
            "sandbox_root": self.sandbox_root,
            "verbose": self.verbose,
        }


# ===========================================================================
# §5.4 LoopBoundPool（解决 R-LOOP 与实测 M-3 的 loop 泄漏）
# ===========================================================================

_KIND_SEMAPHORE = "semaphore"
_KIND_LOCK = "lock"
_KIND_CONDITION = "condition"
_KIND_EVENT = "event"
_KIND_THREAD_POOL = "thread_pool"


class LoopBoundPool:
    """按运行中的 event loop 懒创建 asyncio 原语与线程池的容器（解决 §0.4 的 R-LOOP）。

    **实现不用 WeakKeyDictionary**（v1 的写法）：实测（M-3）``asyncio.Semaphore``
    一旦发生争用就把 loop 存进 ``self._loop`` —— 原语**强引用** loop，于是
    ``WeakKeyDictionary[loop]`` 的 key 永不失效，3 次 ``asyncio.run`` 就泄漏
    3 个 loop 与 3 个原语。冻结实现用普通 dict：``self._by_loop: dict[int, 桶]``
    （key 是 ``id(loop)``）+ ``self._loops: dict[int, loop]``（保活，供 release 使用）。

    两条不变式：

    1. **必须在运行中的 loop 内使用**：没有运行 loop 时入口抛 ``ConfigError``。
       否则拿到的原语会在第一次 ``await`` 时绑到别的 loop 上，报
       "bound to a different event loop" —— 且只有第二次同步调用才炸，极难定位。
    2. **所有同步入口的 ``finally`` 必须 ``release(loop)``**：``config._run_and_cleanup``
       已代办；用户手写 ``asyncio.run`` 的场景靠 ``release_loop`` / ``aclose`` 兜底。

    用法（冻结）::

        pool = LoopBoundPool()
        sem = pool.semaphore("exec", config.max_concurrency)

    ``id(loop)`` 作 key 为什么安全：``_loops`` 持有强引用，loop 在册期间不会被回收，
    因此 id 不会被新 loop 复用（这也正是不能用弱引用的另一个理由）。
    """

    #: 全局登记（模块级）：让 ``config._run_and_cleanup`` 能清理**任意实例**持有的资源
    _ALL_POOLS: ClassVar[list["LoopBoundPool"]] = []

    def __init__(self) -> None:
        # 全部是同步数据结构；本类**不**在 __init__ 里创建任何 asyncio 原语（R-LOOP）
        self._by_loop: dict[int, dict[str, Any]] = {}
        self._loops: dict[int, asyncio.AbstractEventLoop] = {}
        # 记录值的来源参数（semaphore 的 value / thread_pool 的 max_workers），
        # 用于"同 key 不同 value -> ConfigError"的判定
        self._values: dict[int, dict[str, Any]] = {}
        # 多线程保护（threading 原语不受 R-LOOP 限制，§0.4）：
        # 同一个 pool 可能被多个 worker 线程访问（seq lock 就依赖跨线程生效）
        self._lock = threading.RLock()
        LoopBoundPool._ALL_POOLS.append(self)

    # ---- 内部辅助 ----

    @staticmethod
    def _running_loop(what: str) -> asyncio.AbstractEventLoop:
        """取运行中的 loop；没有则抛 ``ConfigError``（R-LOOP 的第一道防线）。"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ConfigError(
                f"LoopBoundPool.{what}() requires a running event loop; "
                "call it inside an async entry point (R-LOOP, see INTERFACES.md §0.4)"
            ) from exc

    def _bucket(self, loop: asyncio.AbstractEventLoop) -> dict[str, Any]:
        """取（必要时创建）该 loop 的原语桶，同时登记 loop 保活。"""
        loop_id = id(loop)
        bucket = self._by_loop.get(loop_id)
        if bucket is None:
            bucket = {}
            self._by_loop[loop_id] = bucket
            self._loops[loop_id] = loop  # 强引用保活：id 不被复用 + release 时能拿到对象
            self._values[loop_id] = {}
        return bucket

    def _get_or_create(self, kind: str, key: str, factory: Callable[[], Any],
                       value: Any = None) -> Any:
        """同 key 同 value -> 复用；同 key 不同 value -> ``ConfigError``。

        为什么不能"静默返回旧对象"：两个调用方都以为自己拿到了期望的并发上限，
        实际只有先到的那个生效 —— 这类 bug 表现为"偶发地不串行"，几乎无法复现。

        桶里的 key 带 kind 前缀，所以 ``lock("exec")`` 与 ``semaphore("exec", 2)``
        是**两个**对象、互不冲突（同一个名字在不同子系统里指不同东西很常见）。
        """
        loop = self._running_loop(f"{kind}({key!r})")
        qualified = f"{kind}:{key}"
        # 参数非法时**先**失败：不能因为一次失败的调用就在 self._by_loop 里
        # 留下一个空桶（否则 len(pool) 会莫名其妙变成 1，泄漏断言直接失效）
        if value is not None and isinstance(value, int) and value <= 0:
            raise ConfigError(
                f"LoopBoundPool: {kind} {key!r} requires a positive value, got {value!r}"
            )
        with self._lock:
            bucket = self._bucket(loop)
            existing = bucket.get(qualified)
            if existing is not None:
                if value is not None:
                    # 桶与 _values 同生共死；用 .get 防御"释放与创建交错"的极端时序
                    old = self._values.get(id(loop), {}).get(qualified)
                    if old != value:
                        raise ConfigError(
                            f"LoopBoundPool: {kind} {key!r} already bound with value {old!r}, "
                            f"cannot rebind to {value!r}"
                        )
                return existing
            created = factory()
            bucket[qualified] = created
            if value is not None:
                self._values[id(loop)][qualified] = value
            return created

    # ---- asyncio 原语 ----

    def semaphore(self, key: str, value: int) -> asyncio.Semaphore:
        """本 loop 私有的信号量。``value <= 0`` -> ``ConfigError``。

        调用方必须用 owner 前缀构造 key（如 ``"exec"`` / ``"subagent:worker_name"``），
        否则两个子系统会抢同一个信号量。
        """
        return self._get_or_create(
            _KIND_SEMAPHORE, key, lambda: asyncio.Semaphore(value), value
        )

    def lock(self, key: str) -> asyncio.Lock:
        """本 loop 私有的互斥锁。"""
        return self._get_or_create(_KIND_LOCK, key, asyncio.Lock)

    def condition(self, key: str) -> asyncio.Condition:
        """本 loop 私有的条件变量（Blackboard 的 watch 用）。"""
        return self._get_or_create(_KIND_CONDITION, key, asyncio.Condition)

    def event(self, key: str) -> asyncio.Event:
        """本 loop 私有的事件。"""
        return self._get_or_create(_KIND_EVENT, key, asyncio.Event)

    def thread_pool(self, key: str, max_workers: int) -> ThreadPoolExecutor:
        """懒建的本 loop 私有线程池（异步工具 / 同步工具走 ``run_in_executor``）。

        同 key 不同 ``max_workers`` -> ``ConfigError``（理由同 :meth:`semaphore`）。

        线程池**不能**在 ``__init__`` 建：它的生命周期必须与 loop 对齐
        （loop 结束就 ``shutdown(wait=False, cancel_futures=True)``），
        否则跨两次 ``asyncio.run`` 复用 executor 时，第二次的线程池会是上一个
        loop 留下的僵尸池。
        """
        return self._get_or_create(
            _KIND_THREAD_POOL,
            key,
            lambda: ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix=f"liteagent-{key}"
            ),
            max_workers,
        )

    # ---- 查询 ----

    def loop_of(self, primitive: Any) -> asyncio.AbstractEventLoop | None:
        """返回该原语绑定的 loop（未绑定时 ``None``）。

        Blackboard 用它做唤醒转发：写入方在 worker 线程里拿到原语，
        需要知道该往哪个 loop ``call_soon_threadsafe``。

        先查自己的登记表（权威、不受争用状态影响），查不到再退回读
        ``primitive._loop``（3.10 的私有属性，只有发生争用后才有值）——
        这里只是**读**，不会因此长期持有引用（M-3 的泄漏是"存"造成的，不是"读"）。
        """
        with self._lock:
            for loop_id, bucket in self._by_loop.items():
                for stored in bucket.values():
                    if stored is primitive:
                        return self._loops.get(loop_id)
        bound = getattr(primitive, "_loop", None)
        if isinstance(bound, asyncio.AbstractEventLoop):
            return bound
        return None

    def current(self) -> asyncio.AbstractEventLoop | None:
        """当前运行中的 loop；没有则 ``None``（**不抛**）。"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def __len__(self) -> int:
        """当前持有的 loop 个数。**测试用它断言不泄漏**（§12）。"""
        return len(self._by_loop)

    # ---- 释放 ----

    def release(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """丢弃该 loop 的全部原语，并对它的全部线程池
        ``shutdown(wait=False, cancel_futures=True)``。``loop`` 为 None 时对所有 loop 执行。

        必须由 ``run_sync`` / ``execute_sync`` / ``Agent.run`` 在 ``asyncio.run``
        返回后的 ``finally`` 里调用（§5.2 的 ``_run_and_cleanup`` 已代办）。
        只丢引用不 shutdown 线程池是不够的：池里的 worker 线程是**非 daemon** 线程，
        会一直存活到进程结束。

        `[v3 修正]` 上一版这里写「不关会让解释器在退出时**挂住**」—— 那句**不准确**：
        ``concurrent.futures.thread`` 自己注册了 ``atexit`` 的 join，所以**空闲**的泄漏池
        不会让解释器卡住（实测残留 30 个 worker 线程时进程仍能 rc=0 秒退）。
        真实代价是**进程内**的累积：线程与 loop 对象随进程寿命线性增长，
        per-request 新建 executor 的长驻服务必须显式 ``aclose()`` / ``aclose_all()``。
        """
        with self._lock:
            if loop is None:
                targets = list(self._loops.values())
            else:
                target = self._loops.get(id(loop))
                targets = [target] if target is not None else [loop]
            for target in targets:
                self._release_one(target)
            if loop is None:
                self._by_loop.clear()
                self._loops.clear()
                self._values.clear()
            else:
                loop_id = id(loop)
                self._by_loop.pop(loop_id, None)
                self._loops.pop(loop_id, None)
                self._values.pop(loop_id, None)

    def _release_one(self, loop: asyncio.AbstractEventLoop) -> None:
        """关掉一个 loop 的线程池（原语只需要丢引用）。"""
        bucket = self._by_loop.get(id(loop))
        if not bucket:
            return
        for qualified, stored in list(bucket.items()):
            if not qualified.startswith(_KIND_THREAD_POOL):
                continue
            try:
                stored.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:  # pragma: no cover - shutdown 不应失败
                # 释放失败必须留痕（禁止 `except: pass`，§13 红线 10），
                # 否则表现为"进程退出时莫名其妙卡住"。用 _note 而不是 warn：
                # release 常在 finally 里被调用，-W error 下 warn 会顶掉主异常。
                _note(f"LoopBoundPool: thread pool {qualified!r} shutdown failed: {exc!r}")

    # ---- 全局登记 ----

    @classmethod
    def release_loop(cls, loop: asyncio.AbstractEventLoop | None) -> None:
        """遍历 ``_ALL_POOLS`` 对每个实例调 ``release(loop)``。

        由 ``config._run_and_cleanup`` 在 ``asyncio.run`` 返回后的 ``finally`` 里调用
        —— 这是"谁负责清理"这个问题的唯一答案（v1 用 WeakKeyDictionary 但没有清理点，
        实测每 3 次 ``asyncio.run`` 就泄漏 3 个 loop 与 3 个原语）。
        """
        for pool in list(cls._ALL_POOLS):
            pool.release(loop)

    @classmethod
    def aclose_all(cls) -> None:
        """``release(None)`` 所有实例并从 ``_ALL_POOLS`` 注销它们。

        给**长驻进程**用（pytest/unittest 之外的 server、REPL）：
        每次 ``ToolExecutor()`` / ``BaseLLMClient()`` 都会往 ``_ALL_POOLS`` 里加一项，
        不注销就是一个随进程寿命线性增长的强引用列表（`[v3]` 实测跑完整套测试后
        这里会积 400+ 个池对象 —— 请把它当成"测试卫生 + 长驻进程必须显式收尾"的
        已知代价，而不是"自动回收"）。

        **注意 ``_run_and_cleanup`` 不管这里**：它只调 ``release_loop(loop)``，
        不注销实例。而且任何**自己关 loop 而不走 ``_run_and_cleanup``** 的代码
        （例如 ``unittest.IsolatedAsyncioTestCase``，§12 强制要求异步用例用它）
        都要自己负责收尾 —— 本项目的做法是在相关测试模块的 ``tearDownModule`` 里
        调一次本方法（见 ``tests/test_agent_features.py``）。
        """
        for pool in list(cls._ALL_POOLS):
            pool.release(None)
        cls._ALL_POOLS.clear()
