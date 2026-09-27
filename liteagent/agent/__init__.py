from __future__ import annotations

"""`liteagent.agent` —— ReAct 循环层（状态机 / 事件 / 文本解析）。

`__all__` 逐字照 §1.4 的冻结清单：**未列出的名字视为内部**，但"内部"只表示非公开 API，
不表示不可测 —— 测试可以直接 `from liteagent.agent.agent import <私有名>`（§2.3）。

依赖方向（§1.1）：本包是 L4，只允许向下 import（errors/types/config/llm/memory/tools）。
四个模块之间可以互相 import（同包自由），但**不得**反向被低层 import ——
`llm/tools/memory` 层的低层事件因此走 `LowLevelEvent`（字符串事件名）而不是 `TraceEvent`。
"""

from liteagent.agent.agent import Agent
from liteagent.agent.callbacks import (
    Callback,
    CallbackLike,
    CallbackManager,
    EventType,
    FunctionCallback,
    JsonlTraceCallback,
    LoggingCallback,
    MemoryTraceCallback,
    RichCallback,
    TokenCounterCallback,
    TraceEvent,
    TraceRecorder,
    as_llm_callback,
    events_of_type,
    load_trace,
    render_trace,
    total_usage,
    trace_stats,
)
from liteagent.agent.parser import ParsedAction, ReActParser
from liteagent.agent.state import AgentConfig, AgentResult, AgentState, AgentStatus

__all__ = [
    "Agent", "AgentConfig", "AgentResult", "AgentState", "AgentStatus",
    "EventType", "TraceEvent", "Callback", "CallbackManager", "CallbackLike",
    "FunctionCallback", "LoggingCallback", "JsonlTraceCallback", "RichCallback",
    "TokenCounterCallback", "MemoryTraceCallback", "TraceRecorder",
    "load_trace", "total_usage", "events_of_type", "render_trace", "as_llm_callback",
    "trace_stats",
    "ReActParser", "ParsedAction",
]
