from __future__ import annotations

# liteagent/multiagent/__init__.py —— 多 Agent 层的公共 API（冻结清单见 §10.5）
#
# 只做 re-export，不写逻辑（§1.2 的职责边界）。`TeamConfig` 的唯一归属地是
# `config.py`（§5.3），这里只是把它抬到多 Agent 包的门面上 —— 用户写
# `SequentialAgent([...], config=TeamConfig(max_depth=2))` 时不必知道它住在哪一层。
#
# `build_team` / `compress_subagent_output` 由 `base.py` 提供：前者是 CLI 与 examples
# 的统一工厂（内部函数内 import 两个编排器，避免 `base -> sequential -> base` 成环），
# 后者是"子 Agent 输出如何回到父 Agent"的唯一实现。

from liteagent.multiagent.base import (
    AgentLike,
    DelegationContext,
    MultiAgent,
    TeamConfig,
    build_team,
    compress_subagent_output,
)
from liteagent.multiagent.blackboard import Blackboard, BlackboardEntry
from liteagent.multiagent.hierarchical import HierarchicalAgent, Plan, SubTask
from liteagent.multiagent.sequential import SequentialAgent, SequentialStep

__all__ = [
    "AgentLike", "MultiAgent", "TeamConfig", "DelegationContext",
    "Blackboard", "BlackboardEntry", "SequentialAgent", "SequentialStep",
    "HierarchicalAgent", "Plan", "SubTask", "build_team", "compress_subagent_output",
]
