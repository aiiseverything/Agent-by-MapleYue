from __future__ import annotations

"""LLM 抽象层（§6）的公开导出。

`__all__` 是**冻结清单**（§1.4）：公开 API 白名单由它显式列出，未列出的视为内部。
本文件只做 re-export 与聚合，**不写任何逻辑**（避免"import 一个模块就发请求/读环境"）。

分层：`llm` 是 L2（只依赖 errors/types/config），**不得** import memory / tools / agent /
multiagent —— 所以 `messages_tokens` 收的是 `counter` 函数而不是 `Tokenizer` 对象。
"""

from liteagent.config import LLMConfig
from liteagent.llm.base import LLMClient, BaseLLMClient, LLMStreamChunk
from liteagent.llm.message import (
    Message,
    Role,
    drop_orphan_tool_messages,
    messages_tokens,
    render_transcript,
)
from liteagent.llm.registry import LLMRegistry, build_llm, get_llm
from liteagent.llm.scripted import ScriptedLLM, ScriptedResponse
from liteagent.types import LLMResponse, ScriptedCall

__all__ = [
    "LLMClient", "BaseLLMClient", "LLMStreamChunk", "Message", "Role", "LLMResponse",
    "LLMConfig", "ScriptedLLM", "ScriptedResponse", "ScriptedCall", "LLMRegistry",
    "build_llm", "get_llm", "messages_tokens", "render_transcript",
    "drop_orphan_tool_messages",
]
