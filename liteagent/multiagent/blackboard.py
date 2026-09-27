from __future__ import annotations

# liteagent/multiagent/blackboard.py —— 跨 Agent 共享的键值黑板（冻结规范 §10.2 / §10.2.1）
#
# 这个文件里有三个值得在面试里讲的点：
#
#   1. **同步优先（§2.3 的 "sync-first 例外"）**：写入方常常是跑在 worker 线程里的
#      *同步* 工具（hierarchical 的 delegate 闭包、executor 的 run_in_executor 分支），
#      那里**没有运行中的事件循环**。所以规范实现全部是同步方法 + `threading.RLock`，
#      `a*` 版本只是薄包装。反过来（async-first + `run_sync`）会在 worker 线程里直接崩
#      —— 就是 §0.4 的 R-LOOP。`threading` 锁与事件循环无关，因此可以在 `__init__` 里建。
#
#   2. **乐观并发（CAS）**：`write(..., if_version=N)` 是"仅当版本仍是 N 才写"。
#      版本号是唯一的冲突判据，失败时抛 `VersionConflictError(key, expected, actual)`
#      把三个数都带出去，调用方可以重读-重算-重试。这样多个 Agent 并发改同一条记录
#      不会丢更新（本文件末尾的并发验证脚本会真的跑 8 线程 × 100 次自增）。
#
#   3. **跨线程唤醒 watcher 是雷区（实测 M-4）**：worker 线程既不能直接
#      `await cond.wait()`，也不能直接 `cond.notify_all()`（未定义行为）。正确姿势是
#      "锁内取 watcher 快照 -> 出锁 -> 逐个 `loop.is_closed()` 判定 -> 把唤醒转发进目标 loop"，
#      并且对已关闭的 loop 必须**吞掉 RuntimeError 并记日志**：否则一次普通的同步 `write`
#      就会把异常扔给调用方，直接违反 §13 红线 6/10。
#
# SPEC-AMBIGUITY（实测记录，针对 §10.2.1 第 2 步的冻结片段）：
#   规范冻结的唤醒写法是 `loop.call_soon_threadsafe(cond.notify_all)`。本机 3.10.12 实测
#   **它唤醒不了任何 watcher**：`asyncio.Condition.notify_all()` 要求调用方**持锁**
#   （Lib/asyncio/locks.py: `if not self.locked(): raise RuntimeError('cannot notify on
#   un-acquired lock')`），而 `wait_for(cond.wait())` 期间锁是**释放**的。实测输出：
#       ERROR:asyncio:Exception in callback Condition.notify_all()
#       RuntimeError: cannot notify on un-acquired lock        # 等待方最后等到 timeout
#   裁决：**保留冻结语义的全部可观测形态**（RLock 内取快照 -> 出锁 -> `loop.is_closed()`
#   判定 -> `try/except RuntimeError` -> 逐字相同的日志文案），只把"转发的载荷"从
#   `cond.notify_all` 换成"在目标 loop 里创建一次唤醒任务"：唤醒任务在 loop 线程内先
#   `async with cond` 再 `notify_all`，满足 asyncio 的持锁要求，实测能唤醒（见文件末尾
#   的验证脚本）。
#
# [v3 变更] 上面那条改动**并没有**真的保住"每次写入都能收到"：唤醒是提示式的
#   （notify_all 在没有等待者时是空操作），而 v2 的 awatch 只在 `cond.wait()` **返回之后**
#   才比对版本 —— 只要一次写入落在"消费者正挂在 yield 上处理上一条 entry"到"重新进入
#   wait()"之间，那次唤醒被丢弃，版本比对再也不会执行，消费者永久停在旧值。v3 把
#   谓词判定移进 `async with cond` 临界区、放在 wait() 之前（标准 monitor 的
#   "持锁判谓词"），使"判脏"与"登记为等待者"和写入侧的 notify 互斥，窗口关闭。
#   同一轮改动还把判脏依据从会回退的 `entry.version` 换成单调的 `_write_seq_of(key)`，
#   修掉 TTL 过期 / delete / clear 后重写时"版本号撞回旧值、写入被静默吞掉"的第二个洞。
#   回归测试见 tests/test_blackboard.py::WatchTests。

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from liteagent.config import LoopBoundPool, to_jsonable, utc_now
from liteagent.errors import ConfigError, VersionConflictError

__all__ = ["Blackboard", "BlackboardEntry"]

# §2.7 的冻结定义式 `LowLevelEvent = Callable[[str, dict[str, Any]], None]`。
# SPEC-AMBIGUITY: §2.7 只给了定义式、没写它的归属模块，而 config.py 里也没有它；本文件
# 的层级是 L1（只依赖 errors/types/config），**不能**从 L2 的 llm/base.py import，否则
# 违反 §1.1 的依赖方向。裁决：按字面在本文件定义一次（与 memory/embeddings.py 同一裁决）。
LowLevelEvent = Callable[[str, dict[str, Any]], None]

# 与 §10.2.1 的日志要求一致：唤醒失败的痕迹必须落在 multiagent 这个 logger 上。
logger = logging.getLogger("liteagent.multiagent")

# 变更日志（`history()`）的内存上限。规范只冻结了 `history(limit=100)` 的读取语义、
# 没冻结内部保留多少条；不设上限的话长期运行的进程会把每一条写过的 entry 永久留住
# （§13 红线：降级/取舍必须可观测 —— 裁剪时记一次 WARNING）。
_MAX_HISTORY_ENTRIES = 1000


@dataclass
class BlackboardEntry:
    """黑板里的一条记录。**不可变使用**：每次写入都产生新对象，旧版本留在 history 里。"""

    key: str
    value: Any
    author: str = ""
    version: int = 1
    created_at: float = field(default_factory=utc_now)  # [v2] 唯一时钟
    updated_at: float = field(default_factory=utc_now)  # [v2]
    tags: tuple[str, ...] = ()
    expires_at: float | None = None

    def is_expired(self, *, now: float | None = None) -> bool:
        """是否已过期。`now` 可注入（测试用 frozen_time 固定时钟）。

        判定用 `>=`：到 `expires_at` 那一刻就算过期（"到期即失效"），
        这样 `ttl_s=0` 的写入行为是确定的（写进去就是过期态），不需要额外特判。
        """
        if self.expires_at is None:
            return False
        current = utc_now() if now is None else now
        return current >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """**字段全量输出**（§2.2），`tags` 是 tuple 所以走 `to_jsonable` 变成 list。

        `value` 也过一次 `to_jsonable`：黑板里放的可能是 dataclass / Enum / set，
        直接塞进 trace 的 `json.dumps` 会炸（§2.7 要求 data 里的值已可序列化）。
        """
        return {
            "key": self.key,
            "value": to_jsonable(self.value),
            "author": self.author,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": to_jsonable(self.tags),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlackboardEntry":
        """**宽容**反序列化（用于 restore / 手工构造的场景）：缺字段用默认值，绝不抛。

        为什么刻意不抛 `SerializationError`：本类没有 §2.2 的 opt-out 字段，
        调用点全部是"从持久化/快照里读回"的恢复路径，一条坏记录不该打死整个恢复流程；
        缺 `key` 时记一条 debug 痕迹（§13 红线 10：不静默）。
        """
        now = utc_now()
        created = data.get("created_at")
        updated = data.get("updated_at")
        expires_at = data.get("expires_at")
        raw_tags = data.get("tags") or ()
        if isinstance(raw_tags, str):  # 手工写的快照常见：tags 写成单个字符串
            raw_tags = (raw_tags,)
        key = data.get("key")
        if not isinstance(key, str) or not key:
            key = ""
            logger.debug("BlackboardEntry.from_dict: missing/invalid 'key' -> ''")
        return cls(
            key=key,
            value=data.get("value"),
            author=str(data.get("author") or ""),
            version=int(data.get("version") or 1),
            created_at=float(created) if created is not None else now,
            updated_at=float(updated) if updated is not None else now,
            tags=tuple(str(t) for t in raw_tags),
            expires_at=float(expires_at) if expires_at is not None else None,
        )


class Blackboard:
    """跨 Agent 共享的键值黑板。**并发安全**（§10.2.1）。

    - 唯一锁 `self._lock = threading.RLock()`，`__init__` 里创建（threading 与 loop 无关）。
    - 同步方法在临界区内**没有 await、没有 I/O**，只操作内存 dict。
    - `keys()` / `list()` / `snapshot()` 一律返回**快照副本**，不泄漏内部 dict；
      `value` 不做深拷贝（性能），因此**写入方不应在写入后再改可变对象**。
    """

    def __init__(self, *, max_entries: int = 1000, on_event: LowLevelEvent | None = None) -> None:
        # max_entries <= 0 会让"写进去立刻被淘汰"变成静默丢数据，这里选择**大声失败**
        # （§13 红线 12 的精神：降级可以是设计，但静默不行）。
        try:
            limit = int(max_entries)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Blackboard.max_entries must be an int, got {max_entries!r}") from exc
        if limit <= 0:
            raise ConfigError(f"Blackboard.max_entries must be positive, got {max_entries!r}")
        self._max_entries = limit
        self._on_event = on_event

        self._lock = threading.RLock()
        # watch 用：按 loop 懒创建的原语容器（R-LOOP，§5.4；禁止在 __init__ 建 asyncio 原语）
        self._pools = LoopBoundPool()
        # key -> [(loop, condition)]，由 awatch 自己在运行中的 loop 内登记/注销
        self._watch_loops: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Condition]]] = {}
        # key -> 保护该 key 的 _watch_loops 条目的锁（写入方只读快照，不必拿它）
        self._watch_locks: dict[str, threading.Lock] = {}

        # key -> 该 key 的**单调写入序号**（只增不减，条目过期/删除/清空都不重置）。
        # [v3 新增] awatch 的判脏依据必须是"这个 key 被写过几次"，而不是 entry.version ——
        # 后者是"entry 生命周期内的版本"，条目被 TTL 判过期、被 delete、被 clear、
        # 被 max_entries 淘汰后，同 key 再写的 version 会从 1 重新开始（见 _commit_locked），
        # 于是 `version != last` 可能永远为假，写入发生了却一次都不 yield（静默失联）。
        self._write_seq: dict[str, int] = {}
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: list[BlackboardEntry] = []
        self._subscribers: list[Callable[[str, BlackboardEntry], None]] = []
        self._history_trimmed = False  # 裁剪只 WARNING 一次，避免刷屏

    # ------------------------------------------------------------------
    # 同步 API —— 规范实现（全部在 threading.RLock 内完成）
    # ------------------------------------------------------------------

    def write(self, key: str, value: Any, *, author: str = "",
              tags: Sequence[str] = (), ttl_s: float | None = None,
              if_version: int | None = None) -> BlackboardEntry:
        """写入一条记录并返回**新版本**的 entry。

        - `key` 非空字符串，否则 `ConfigError`。
        - 已存在 -> version+1、`updated_at` 刷新、`created_at` 保留、`author` 覆盖。
        - 不存在（或被 TTL 判为过期）-> version=1。
        - `if_version` 非 None 且与当前版本不符 -> `VersionConflictError(key, expected, actual)`；
          不存在的 key 的当前版本按 **0** 计，因此 `if_version=0` 就是"仅当不存在时创建"。
        - 超过 `max_entries` -> 按 `updated_at` 淘汰最旧的并记 WARNING。
        - 订阅回调与 `on_event("blackboard_write", ...)` 在**锁外**调用；
          唤醒 watcher 的顺序见 §10.2.1（本文件顶部有实测说明）。
        """
        self._validate_key(key)
        now = utc_now()
        with self._lock:
            entry, subscribers, targets, evicted = self._commit_locked(
                key, value, author=author, tags=tags, expires_at=_expiry_from_ttl(now, ttl_s),
                if_version=if_version, now=now,
            )
        self._after_write(entry, subscribers, targets, evicted)
        return entry

    def read(self, key: str, *, default: Any = None) -> Any:
        """读值；不存在或已过期时返回 `default`（惰性过期）。

        注意：本方法把"是否命中"的事件发射收敛在 `read_entry` 里，避免一次读发两条事件。
        """
        entry = self.read_entry(key)
        return default if entry is None else entry.value

    def read_entry(self, key: str) -> BlackboardEntry | None:
        """读整条 entry（含版本号/作者/过期时间）。**过期条目按不存在处理**。"""
        now = utc_now()
        with self._lock:
            entry = self._live_entry_locked(key, now)
        # on_event 在锁外调用（§10.2）：回调里再读黑板不会撞自身锁
        self._emit("blackboard_read", {"key": key, "hit": entry is not None})
        return entry

    def delete(self, key: str, *, author: str = "") -> bool:
        """删除；返回是否**确实删掉了一条未过期记录**（过期条目按不存在处理，返回 False）。

        刻意不追加 history、也不唤醒 watcher：§10.2 冻结的是"每次该 key 被**写入**时 yield"，
        删除不是写入。所以 awatch **观测不到删除本身**（`read_entry` 返回 None 时直接跳过，
        不会 yield）；它只会在同 key **再次被写入**时收到新值 —— 判脏依据是单调的
        `_write_seq`，因此"删除后重写"（version 从 1 重新开始）也不会漏投递。
        """
        now = utc_now()
        with self._lock:
            entry = self._live_entry_locked(key, now)
            if entry is None:
                self._entries.pop(key, None)  # 顺手清掉过期残留（惰性过期的写侧）
                return False
            self._entries.pop(key, None)
        logger.debug("blackboard delete key=%s (version=%d, author=%r)", key, entry.version, author)
        return True

    def keys(self, *, prefix: str = "", tags: Sequence[str] = ()) -> list[str]:
        """匹配的 key 列表（**副本**），顺序与 `list()` 一致（updated_at 升序）。"""
        return [entry.key for entry in self.list(prefix=prefix, tags=tags)]

    def list(self, *, prefix: str = "", tags: Sequence[str] = (),
             include_expired: bool = False) -> list[BlackboardEntry]:
        """按 `updated_at` **升序**返回 entry 快照（新 list，不是内部 dict 的视图）。

        `include_expired=True` 时把过期条目也带上（调试/审计用）。注意过期条目会在
        **下一次写入**时被惰性清出内部 dict，所以 include_expired 不是"永久可见"的保证。
        """
        now = utc_now()
        wanted = tuple(tags)
        with self._lock:
            matched = [
                entry
                for entry in self._entries.values()
                if (include_expired or not entry.is_expired(now=now))
                and (not prefix or entry.key.startswith(prefix))
                and all(tag in entry.tags for tag in wanted)
            ]
        # 稳定排序：updated_at 相等时保持插入顺序，避免"同一 tick 内写多条"的断言 flaky
        matched.sort(key=lambda entry: entry.updated_at)
        return matched

    def snapshot(self) -> dict[str, Any]:
        """`{key: value}` 的**浅拷贝**（§10.2.1：不返回内部 dict 引用）。

        浅拷贝是刻意的：深拷贝一条大对象会拖慢每次 `AgentResult.metadata["blackboard"]`
        的组装，而 value 本来就约定"写入后不再修改"。
        """
        now = utc_now()
        with self._lock:
            return {
                key: entry.value
                for key, entry in self._entries.items()
                if not entry.is_expired(now=now)
            }

    def update(self, key: str, patch: Mapping[str, Any], *, author: str = "") -> BlackboardEntry:
        """把 `patch` **浅合并**进 `key` 的 dict 值，返回新版本 entry。

        语义裁决（规范只写了"value 必须是 dict；浅合并。否则 -> ConfigError"）:
        - key 不存在（或已过期）时从空 dict 起步，等价于"创建"；
        - 只 patch `value`：`tags` 与剩余 TTL 沿袭旧版本 —— 否则一次字段更新会
          静默丢掉标签或让带 TTL 的记录变成永不过期，这是更危险的默认；
        - `author` 为空串时沿袭旧作者，非空则覆盖。
        """
        self._validate_key(key)
        if not isinstance(patch, Mapping):
            raise ConfigError(
                f"Blackboard.update() expects a mapping patch, got {type(patch).__name__}"
            )
        now = utc_now()
        with self._lock:
            current = self._live_entry_locked(key, now)
            base = current.value if current is not None else {}
            if not isinstance(base, dict):
                raise ConfigError(
                    f"Blackboard.update() needs a dict value at key {key!r}, "
                    f"got {type(base).__name__}"
                )
            merged = {**base, **dict(patch)}
            entry, subscribers, targets, evicted = self._commit_locked(
                key, merged,
                author=author or (current.author if current is not None else ""),
                tags=current.tags if current is not None else (),
                expires_at=current.expires_at if current is not None else None,
                if_version=None, now=now,
            )
        self._after_write(entry, subscribers, targets, evicted)
        return entry

    def increment(self, key: str, *, amount: float = 1, author: str = "") -> float:
        """原子自增并返回**新值**（float）。不存在的 key 从 0 起步。

        `value` 非数值 -> `ConfigError`。**`bool` 也算非数值**：`True + 1` 在 Python 里
        合法但几乎必然是 bug（谁会把计数器存成布尔），静默算出 2 才是更坏的结果。
        整个"读-加-写"在 `self._lock` 内完成，所以并发自增不丢更新（见文件末尾验证脚本）。
        """
        self._validate_key(key)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ConfigError(f"Blackboard.increment() amount must be numeric, got {amount!r}")
        now = utc_now()
        with self._lock:
            current = self._live_entry_locked(key, now)
            base = 0 if current is None else current.value
            if isinstance(base, bool) or not isinstance(base, (int, float)):
                raise ConfigError(
                    f"Blackboard.increment() needs a numeric value at key {key!r}, "
                    f"got {type(base).__name__}"
                )
            new_value = base + amount
            entry, subscribers, targets, evicted = self._commit_locked(
                key, new_value,
                author=author or (current.author if current is not None else ""),
                tags=current.tags if current is not None else (),
                expires_at=current.expires_at if current is not None else None,
                if_version=None, now=now,
            )
        self._after_write(entry, subscribers, targets, evicted)
        return float(new_value)

    def clear(self) -> None:
        """清空全部条目与变更日志。**订阅者与 watcher 保留**（它们是注册关系，不是状态）。"""
        with self._lock:
            self._entries.clear()
            self._history.clear()

    def subscribe(self, fn: Callable[[str, BlackboardEntry], None]) -> Callable[[], None]:
        """订阅写入；返回**幂等**的取消函数。

        订阅回调在**锁外**、按注册顺序同步调用；单个回调抛错只记日志，绝不影响
        写入成功与其它订阅者（§13 红线 10：吞异常必须留痕）。
        """
        if not callable(fn):
            raise ConfigError(f"Blackboard.subscribe() expects a callable, got {type(fn).__name__}")
        with self._lock:
            self._subscribers.append(fn)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(fn)
                except ValueError:
                    # 幂等：重复取消不是错误，但留个痕迹便于排查"为什么回调没触发"
                    logger.debug("Blackboard.unsubscribe(): callback was not subscribed")
                    return
            logger.debug("Blackboard.unsubscribe(): removed callback %r", getattr(fn, "__name__", fn))

        return _unsubscribe

    def __len__(self) -> int:
        """**未过期**的条目数（与 `read`/`keys` 的口径一致，避免"读不到但数得到"）。"""
        now = utc_now()
        with self._lock:
            return sum(1 for entry in self._entries.values() if not entry.is_expired(now=now))

    def __contains__(self, key: object) -> bool:
        """`key in bb` == "存在且未过期"。刻意不发 `blackboard_read` 事件：成员测试不是读值。"""
        if not isinstance(key, str):
            return False
        now = utc_now()
        with self._lock:
            return self._live_entry_locked(key, now) is not None

    def to_dict(self) -> dict[str, Any]:
        """给 trace / CLI / debug 用的整体快照（不含 history，避免体积失控）。"""
        now = utc_now()
        with self._lock:
            entries = {
                key: entry.to_dict()
                for key, entry in self._entries.items()
                if not entry.is_expired(now=now)
            }
        return {
            "count": len(entries),
            "max_entries": self._max_entries,
            "entries": entries,
        }

    def __repr__(self) -> str:
        return f"<Blackboard entries={len(self)} max_entries={self._max_entries}>"

    # ------------------------------------------------------------------
    # 异步 API —— 薄包装（sync-first，§2.3）+ 唯一的真异步成员 awatch
    # ------------------------------------------------------------------

    async def awrite(self, *args: Any, **kwargs: Any) -> BlackboardEntry:
        """`write` 的 async 镜像。**直接调用同步实现**：临界区没有 await / I/O，
        真正耗时的是用户订阅回调，而它与 `CallbackManager.emit` 一样是同步契约。
        """
        return self.write(*args, **kwargs)

    async def aread(self, key: str, *, default: Any = None) -> Any:
        return self.read(key, default=default)

    async def aread_entry(self, key: str) -> BlackboardEntry | None:
        return self.read_entry(key)

    async def adelete(self, key: str, *, author: str = "") -> bool:
        return self.delete(key, author=author)

    async def alist(self, **kwargs: Any) -> list[BlackboardEntry]:
        return self.list(**kwargs)

    async def asnapshot(self) -> dict[str, Any]:
        return self.snapshot()

    async def aclear(self) -> None:
        self.clear()

    async def awatch(self, key: str, *, timeout_s: float | None = None) -> AsyncIterator[BlackboardEntry]:
        """监听 `key`：每次写入后 yield **最新的** entry（**合并语义**）。

        - 多次写入快于消费时只 yield 最新一条，中间版本去 `history()` 取
          （规范刻意不做成逐版本队列：那会让慢消费者把内存吃到无界）。
        - `timeout_s` 是**总超时**：从进入 async generator 起算，到点抛
          `asyncio.TimeoutError`（在消费者的 `async for` 处收到），yield 后不重新计时。
          `timeout_s=None` 永不超时。
        - 实现是 loop-bound `asyncio.Condition` + **版本号比对**，**不用轮询**。

        `[v2 变更]` 为什么 watcher 必须自己登记 loop：写入方可能在工作线程（同步工具里写
        黑板），那里没有运行中的 loop，`LoopBoundPool.condition()` 会直接抛 `ConfigError`。
        所以 loop 只能由 awatch 自己从 `asyncio.get_running_loop()` 拿。
        """
        # 必须在运行中的 loop 内调用（R-LOOP）：没有 loop 时 LoopBoundPool 会抛 ConfigError
        cond = self._pools.condition("bbwatch")
        loop = asyncio.get_running_loop()
        # §10.2.1 第 1 步的冻结顺序是"**先登记、再读基线**"：登记之后的写入一定在
        # 快照里（会被转发唤醒），登记之前的写入会被紧随其后的基线读取吸收 ——
        # 两种落在"读基线 / 登记"之间的写入都不会漏。顺序反过来会制造一段
        # "既不转发通知、又不在基线里"的空洞，写入永久丢失。
        # CPython 的 dict.setdefault 是原子的，所以同一 key 只会有一个锁对象胜出
        guard = self._watch_locks.setdefault(key, threading.Lock())
        with guard:
            self._watch_loops.setdefault(key, []).append((loop, cond))
        deadline = None if timeout_s is None else (loop.time() + timeout_s)
        try:
            # 基线用**单调写入序号**（`_write_seq_of`），不用 entry.version：
            # 后者在条目过期/删除后会回退到 1，导致 `!= last` 恒假、写入永不 yield。
            last = self._write_seq_of(key)
            while True:
                async with cond:
                    remaining = None if deadline is None else max(0.0, deadline - loop.time())
                    if deadline is not None and remaining == 0.0:
                        raise asyncio.TimeoutError()
                    # **持 cond 锁判定谓词**（标准 monitor 写法）：写入侧的唤醒
                    # （`_wake_watchers` 的 `async with cond: notify_all()`）必须先拿到同一把
                    # cond 锁，因此"版本比对"与"登记为等待者"之间不存在窗口 —— 要么在
                    # wait() 之前就看到新序号（本次不停车，直接在锁外 yield），要么本次
                    # wait() 一定在 notify_all 之前进入、必然被唤醒。
                    # 只在仍干净时才停车：这样"消费者处理上一条 entry 期间发生的写入"
                    # 也不会丢（notify_all 无人可唤醒的窗口被关掉）。
                    if self._write_seq_of(key) == last:
                        try:
                            await asyncio.wait_for(cond.wait(), remaining)
                        except asyncio.TimeoutError:
                            # 显式重抛（不吞）：总超时是契约的一部分，消费者必须看到
                            raise
                # 出锁后再读一次（yield 的用户代码绝不能跑在持锁状态下）
                seq = self._write_seq_of(key)
                if seq != last:
                    entry = self.read_entry(key)
                    if entry is not None:
                        yield entry  # 合并语义：只 yield 最新
                    last = seq
        finally:
            # 注销必须放在 finally：消费者 break/取消/超时都要把条目摘掉，否则
            # self._watch_loops 会无限增长，且会往死 loop 上转发唤醒
            with self._watch_locks.get(key, self._lock):
                watchers = self._watch_loops.get(key)
                if watchers is not None:
                    try:
                        watchers.remove((loop, cond))
                    except ValueError:
                        logger.debug("blackboard watcher already deregistered key=%s", key)

    # ------------------------------------------------------------------
    # 变更日志（给 trace / 调试用）
    # ------------------------------------------------------------------

    def history(self, *, limit: int = 100) -> list[BlackboardEntry]:
        """最近 `limit` 次写入（**时间升序**，最新的在最后）。`limit <= 0` 返回空列表。"""
        if limit <= 0:
            return []
        with self._lock:
            return list(self._history[-limit:])

    # ------------------------------------------------------------------
    # 内部实现（全部是 "调用方已持锁" 或 "自持锁" 的小工具）
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_key(key: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ConfigError(f"blackboard key must be a non-empty string, got {key!r}")

    def _live_entry_locked(self, key: str, now: float) -> BlackboardEntry | None:
        """持锁读一条**未过期** entry；过期视为不存在（**惰性过期，不在这里删**）。"""
        entry = self._entries.get(key)
        if entry is None or entry.is_expired(now=now):
            return None
        return entry

    def _version_of(self, key: str) -> int:
        """当前版本号；不存在或已过期 -> 0（与 `if_version=0` 的"仅创建"语义对齐）。"""
        now = utc_now()
        with self._lock:
            entry = self._live_entry_locked(key, now)
            return entry.version if entry is not None else 0

    def _write_seq_of(self, key: str) -> int:
        """该 key 的**单调写入序号**（0 表示从未写过）。

        [v3 新增] 与 `_version_of` 的区别：`_version_of` 是 entry 生命周期内的版本号，
        条目过期/删除后回到 0；`_write_seq_of` 永不回退，是 awatch 唯一正确的判脏依据。
        """
        with self._lock:
            return self._write_seq.get(key, 0)

    def _purge_expired_locked(self, now: float) -> list[str]:
        """把过期条目从内部 dict 里清掉（只在**写入路径**调用，避免给读路径加写操作）。"""
        if not self._entries:
            return []
        dead = [key for key, entry in self._entries.items() if entry.is_expired(now=now)]
        for key in dead:
            self._entries.pop(key, None)
        if dead:
            logger.debug("blackboard purged %d expired entries", len(dead))
        return dead

    def _append_history_locked(self, entry: BlackboardEntry) -> None:
        self._history.append(entry)
        overflow = len(self._history) - _MAX_HISTORY_ENTRIES
        if overflow > 0:
            del self._history[:overflow]
            if not self._history_trimmed:
                self._history_trimmed = True
                logger.warning(
                    "blackboard history trimmed to %d entries (dropped %d oldest); "
                    "increase max_entries/log rotation if you need a full audit trail",
                    _MAX_HISTORY_ENTRIES, overflow,
                )

    def _enforce_max_entries_locked(self) -> list[str]:
        """超过 `max_entries` 时按 `updated_at` 淘汰最旧的（同值时淘汰更早插入的）。"""
        evicted: list[str] = []
        while len(self._entries) > self._max_entries:
            # min() 稳定：updated_at 相同时取最先遇到的（dict 保持插入序）
            oldest = min(self._entries, key=lambda key: self._entries[key].updated_at)
            self._entries.pop(oldest, None)
            evicted.append(oldest)
        if evicted:
            logger.warning(
                "blackboard max_entries=%d exceeded; evicted %d oldest: %s",
                self._max_entries, len(evicted), evicted,
            )
        return evicted

    def _commit_locked(self, key: str, value: Any, *, author: str, tags: Sequence[str],
                       expires_at: float | None, if_version: int | None,
                       now: float) -> tuple[BlackboardEntry, list[Any], list[Any], list[str]]:
        """写入的**唯一**临界区：版本判定 -> 构造 entry -> 记账 -> 取回调/唤醒快照。

        返回 `(entry, subscribers, watcher_targets, evicted)` 四元组，
        调用方**出锁后**再跑 `_after_write` —— 这是 §10.2 冻结的"回调在锁外调用"。
        """
        self._purge_expired_locked(now)
        current = self._entries.get(key)
        if if_version is not None:
            actual = current.version if current is not None else 0
            if actual != if_version:
                raise VersionConflictError(key, if_version, actual)
        version = 1 if current is None else current.version + 1
        entry = BlackboardEntry(
            key=key,
            value=value,
            author=author,
            version=version,
            created_at=now if current is None else current.created_at,
            updated_at=now,
            tags=tuple(tags),
            expires_at=expires_at,
        )
        self._entries[key] = entry
        # [v3 新增] 单调写入序号：只增不减，**不受**条目生命周期影响（delete/过期/clear/
        # 淘汰都不重置），awatch 的判脏依据换成它之后，"每次写入都 yield" 才真正成立。
        self._write_seq[key] = self._write_seq.get(key, 0) + 1
        self._append_history_locked(entry)
        evicted = self._enforce_max_entries_locked()
        return entry, list(self._subscribers), self._snapshot_watchers(key), evicted

    def _snapshot_watchers(self, key: str) -> list[tuple[asyncio.AbstractEventLoop, asyncio.Condition]]:
        """取 watcher 快照（**调用方必须已持 `self._lock`**，§10.2.1 第 2 步）。"""
        return list(self._watch_loops.get(key, ()))

    def _after_write(self, entry: BlackboardEntry, subscribers: Sequence[Any],
                     targets: Sequence[Any], evicted: Sequence[str]) -> None:
        """出锁后的收尾：订阅回调 -> 低层事件 -> 转发唤醒。任何一步都不允许打断写入。"""
        for fn in subscribers:
            try:
                fn(entry.key, entry)
            except Exception:  # noqa: BLE001 - 回调是用户代码，必须隔离（但绝不静默）
                logger.exception("blackboard subscriber failed for key=%s", entry.key)
        self._emit("blackboard_write",
                   {"key": entry.key, "version": entry.version, "author": entry.author})
        if evicted:
            logger.debug("blackboard write evicted keys=%s (key=%s)", list(evicted), entry.key)
        for loop, cond in targets:
            self._forward_notify(loop, cond, entry.key)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """`on_event` 的低层事件出口（§2.7：Blackboard 是这两条事件的**唯一**发射者）。"""
        if self._on_event is None:
            return
        try:
            self._on_event(event_type, data)
        except Exception:  # noqa: BLE001 - 观测失败不能拖垮业务写入
            logger.exception("blackboard on_event handler failed for %s", event_type)

    def _forward_notify(self, loop: asyncio.AbstractEventLoop, cond: asyncio.Condition,
                        key: str) -> None:
        """把唤醒转发到 watcher 所在的 loop（**绝不在 worker 线程里直接 notify**）。

        M-4 实测：对**已关闭**的 loop 调 `call_soon_threadsafe` 会抛
        `RuntimeError: Event loop is closed`，所以必须先 `is_closed()` 判定；
        竞态窗口里仍可能抛，于是再包一层 try/except —— 吞掉但记日志（§13 红线 10）。
        """
        if loop.is_closed():
            logger.debug("blackboard watcher loop is closed; skip notify key=%s", key)
            return
        try:
            # 载荷是 `_schedule_wake` 而不是 `cond.notify_all`：后者要求调用方持锁
            # （本机 3.10.12 实测会抛 RuntimeError 且**唤醒不了**，见文件顶部 SPEC-AMBIGUITY）。
            # 唤醒任务在 loop 线程内先 `async with cond` 再 notify_all，满足持锁要求。
            loop.call_soon_threadsafe(_schedule_wake, loop, cond, key)
        except RuntimeError:  # 已关闭的 loop
            logger.info(
                "blackboard watcher loop is closed; skip notify key=%s", key)


# ---------------------------------------------------------------------------
# 模块级小工具
# ---------------------------------------------------------------------------


def _expiry_from_ttl(now: float, ttl_s: float | None) -> float | None:
    """TTL -> 绝对过期时间。`None` 表示永不过期；`ttl_s <= 0` 表示"写进去就是过期态"。"""
    return None if ttl_s is None else (now + ttl_s)


async def _wake_watchers(cond: asyncio.Condition) -> None:
    """在目标 loop 的线程内唤醒该 loop 上所有等待者。

    先 `async with cond` 拿锁再 `notify_all()` —— `asyncio.Condition.notify*` 的
    **持锁要求**（3.10 起就是硬要求，见文件顶部 SPEC-AMBIGUITY）。
    """
    async with cond:
        cond.notify_all()


def _schedule_wake(loop: asyncio.AbstractEventLoop, cond: asyncio.Condition, key: str) -> None:
    """`call_soon_threadsafe` 的载荷：在**目标 loop 内**创建一次唤醒任务。

    必须是"在 loop 线程里创建 task"而不是在写入线程里创建协程再传进来：
    写入线程创建协程却因 loop 关闭而未能调度时，会留下
    `RuntimeWarning: coroutine ... was never awaited`（§5.2 明确要避免的噪音）。
    """
    task = loop.create_task(_wake_watchers(cond))
    task.add_done_callback(lambda done: _report_wake_failure(done, key))


def _report_wake_failure(task: "asyncio.Task[None]", key: str) -> None:
    """唤醒任务的失败**必须留痕**（§13 红线 10），同时避免 asyncio 的
    "Task exception was never retrieved" 噪音。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.info("blackboard watcher notify failed key=%s: %r", key, exc)
