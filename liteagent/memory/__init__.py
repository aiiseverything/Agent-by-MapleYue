from __future__ import annotations

# 记忆包的公共 API 出口（§1.4 的冻结清单，逐字实现）。
#
# 为什么要显式列 `__all__` 而不是靠"导入即导出"：这份清单是**跨模块契约** ——
# agent / tools / multiagent 都从这里取类型；显式白名单让"哪些名字是公开的"有一个
# 唯一真值源，也让守门测试（§12 的 test_zero_dependency）能逐名断言可解析。
# 名字一个都不能少、一个都不能多：漏了会让上层 import 失败，多了会让"内部实现"
# 悄悄变成别人依赖的公开 API。
#
# 本文件只做 re-export，不含任何逻辑（§1.2 对 __init__.py 的职责边界要求）。

from liteagent.memory.base import (
    CallableTokenizer,
    HeuristicTokenizer,
    MemoryConfig,
    MemoryItem,
    MemoryStore,
    Tokenizer,
    get_default_tokenizer,
)
from liteagent.memory.buffer import BufferConfig, BufferMemory
from liteagent.memory.embeddings import (
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    RandomProjectionEmbedder,
    RemoteEmbedder,
    cosine_similarity,
    cosine_similarity_matrix,
    default_embedder,
)
from liteagent.memory.manager import MemoryManager
from liteagent.memory.summary import SummaryConfig, SummaryMemory
from liteagent.memory.vector import VectorConfig, VectorMemory

try:
    # numpy 是**可选**依赖（§1.3）：embeddings.py 在 NUMPY_AVAILABLE 为假时根本不定义
    # 这个类。写成 try/except 而不是条件分支赋值，是因为 import 失败必须在"名字不存在"
    # 这条路径上被接住 —— 而 `import` 语句本身只能放在 try 里（同样的形态被
    # test_zero_dependency 的 AST 判定放行）。
    from liteagent.memory.embeddings import NumpyHashingEmbedder
except ImportError:  # pragma: no cover - 本环境有 numpy，这条分支只为可移植性存在
    # 保持 `__all__` 里的名字始终可解析（`getattr(liteagent.memory, name)` 不抛），
    # 代价只是拿到的不是可用的类。users 想判断能力请查 embeddings.NUMPY_AVAILABLE。
    NumpyHashingEmbedder = None  # type: ignore[assignment]

__all__ = [
    "MemoryItem", "MemoryStore", "MemoryConfig", "Tokenizer", "HeuristicTokenizer",
    "CallableTokenizer", "get_default_tokenizer",
    "Embedder", "HashingEmbedder", "NumpyHashingEmbedder", "RandomProjectionEmbedder",
    "CallableEmbedder", "RemoteEmbedder", "default_embedder", "cosine_similarity",
    "cosine_similarity_matrix",
    "BufferMemory", "BufferConfig", "SummaryMemory", "SummaryConfig",
    "VectorMemory", "VectorConfig", "MemoryManager",
]
