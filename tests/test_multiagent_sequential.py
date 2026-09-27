from __future__ import annotations

"""tests/test_multiagent_sequential.py —— §10.3 `multiagent/sequential.py` 的单元测试。

§12 第 5128 行要求的覆盖点（逐条对应到测试类）：

  * 三步流水线                          -> PipelineTests.test_three_step_pipeline_...
  * input_template 渲染 {input}/{prev}/{steps[x]} -> TemplateTests
  * 黑板写入                            -> BlackboardWriteTests
  * propagate_failure 三种 + 两条 output 公式 -> FailureHandlingTests
  * optional                            -> FailureHandlingTests.test_optional_...
  * usage 聚合                          -> ResultAggregationTests
  * metadata["steps"]                   -> ResultAggregationTests
  * 输出截断                            -> TruncationTests
  * share_memory=True 时 worker.memory is manager.memory -> ShareMemoryTests

另外补了 §10.3/§10.1 里与顺序编排直接相关的点：AGENT_DELEGATE/AGENT_RETURN 配对、
三道闸（深度/预算/环）、嵌套 MultiAgent 才收 `context=`、`build_team` 工厂。

全部用假的 `StubAgent`（只要求 `name` + `arun`），不拉起 LLM/工具栈，也不碰网络。
"""

import unittest
from typing import Any, Callable

from liteagent.agent.callbacks import EventType
from liteagent.agent.state import AgentResult, AgentState, AgentStatus
from liteagent.config import TeamConfig
from liteagent.errors import ConfigError, CycleDetectedError, DelegationError
from liteagent.errors import MaxDepthExceededError
from liteagent.multiagent import (
    DelegationContext,
    MultiAgent,
    SequentialAgent,
    SequentialStep,
    TeamConfig as TeamConfigReexport,
    build_team,
)
from liteagent.types import TokenUsage


# ======================================================================================
# 夹具
# ======================================================================================


class StubAgent:
    """最小可编排对象：`name` + `arun`（`AgentLike` 是结构协议，不需要继承）。"""

    def __init__(
        self,
        name: str,
        *,
        output: str | Callable[[str], str] = "out",
        status: AgentStatus = AgentStatus.FINISHED,
        error: Any = None,
        steps: int = 1,
        usage: TokenUsage | None = None,
        memory: Any = None,
    ) -> None:
        self.name = name
        self.description = f"{name} stub"
        self._output = output
        self.status = status
        self.error = error
        self.steps = steps
        self.usage = usage if usage is not None else TokenUsage()
        self.memory = memory
        self.calls: list[dict[str, Any]] = []

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult:
        self.calls.append(
            {"input": input, "state": state, "context": context, "kwargs": dict(kwargs)}
        )
        text = self._output(input) if callable(self._output) else self._output
        return AgentResult(
            output=text,
            status=self.status,
            steps=self.steps,
            usage=self.usage,
            error=self.error,
            state=state,
            agent_name=self.name,
        )


class RecordingMultiAgent(MultiAgent):
    """记录收到的 `context=` 的嵌套编排器（验证 §10.3 步骤 e 的 `isinstance` 分支）。"""

    def __init__(self, name: str) -> None:
        super().__init__([], name=name)
        self.calls: list[dict[str, Any]] = []

    async def arun(self, input: str, *, state: AgentState | None = None,
                   context: DelegationContext | None = None, **kwargs: Any) -> AgentResult:
        self.calls.append({"input": input, "state": state, "context": context})
        return AgentResult(
            output=f"[ma:{self.name}]",
            status=AgentStatus.FINISHED,
            agent_name=self.name,
            state=state,
        )


def _failed_agent(name: str, *, output: str = "", error: Exception | None = None) -> StubAgent:
    return StubAgent(
        name, output=output, status=AgentStatus.FAILED,
        error=error or DelegationError(from_agent="upstream", to_agent=name),
    )


# ======================================================================================
# 三步流水线 / 模板
# ======================================================================================


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_step_pipeline_chains_outputs_by_default(self) -> None:
        first = StubAgent("a", output="A")
        second = StubAgent("b", output="B")
        third = StubAgent("c", output="C")
        team = SequentialAgent([first, second, third])

        result = await team.arun("start")

        self.assertEqual(first.calls[0]["input"], "start")   # 第 0 步吃 {input}
        self.assertEqual(second.calls[0]["input"], "A")      # 其余吃 {prev}
        self.assertEqual(third.calls[0]["input"], "B")
        self.assertEqual(result.output, "C")
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.agent_name, "sequential")

    async def test_each_stage_gets_a_fresh_state_and_delegation_context(self) -> None:
        first = StubAgent("a", output="A")
        second = StubAgent("b", output="B")
        team = SequentialAgent([first, second])

        await team.arun("start")

        first_state = first.calls[0]["state"]
        second_state = second.calls[0]["state"]
        self.assertIsNotNone(first_state)
        self.assertIsNotNone(second_state)
        self.assertIsNot(first_state, second_state)          # 父 state 不复用
        self.assertEqual(first_state.input, "start")
        self.assertEqual(second_state.input, "A")
        self.assertEqual(first_state.agent_name, "a")

        first_ctx = first_state.scratchpad["delegation"]
        second_ctx = second_state.scratchpad["delegation"]
        self.assertIsInstance(first_ctx, DelegationContext)
        self.assertEqual(first_ctx.stack, ["sequential", "a"])
        self.assertEqual(second_ctx.stack, ["sequential", "b"])
        self.assertEqual(first_ctx.depth, 1)
        self.assertEqual(second_ctx.depth, 1)

    async def test_plain_agent_does_not_receive_context_keyword(self) -> None:
        """§10.3 步骤 e：只有 `MultiAgent` 才额外收 `context=`（普通 Agent 会当非法 override）。"""
        plain = StubAgent("plain", output="P")
        await SequentialAgent([plain]).arun("x")
        self.assertIsNone(plain.calls[0]["context"])
        self.assertEqual(plain.calls[0]["kwargs"], {})

    async def test_nested_multiagent_step_receives_context(self) -> None:
        nested = RecordingMultiAgent("nested")
        team = SequentialAgent([nested])
        await team.arun("x")
        context = nested.calls[0]["context"]
        self.assertIsInstance(context, DelegationContext)
        self.assertEqual(context.stack, ["sequential", "nested"])

    async def test_empty_pipeline_returns_finished_empty_result(self) -> None:
        team = SequentialAgent([], name="empty")
        result = await team.arun("nothing")
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.output, "")
        self.assertEqual(result.metadata["steps"], [])
        self.assertEqual(result.agent_name, "empty")

    async def test_bare_agents_are_wrapped_and_steps_exposed(self) -> None:
        team = SequentialAgent([StubAgent("a", output="A")])
        self.assertEqual(len(team.steps), 1)
        self.assertIsInstance(team.steps[0], SequentialStep)
        self.assertEqual(team.steps[0].name, "a")

    async def test_non_agent_step_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            SequentialAgent([object()])  # type: ignore[list-item]

    async def test_events_delegate_and_return_are_paired(self) -> None:
        events: list[Any] = []
        team = SequentialAgent(
            [StubAgent("a", output="A"), StubAgent("b", output="B")],
            callbacks=[events.append],
        )
        await team.arun("start")
        delegate = [e for e in events if e.type == EventType.AGENT_DELEGATE]
        returned = [e for e in events if e.type == EventType.AGENT_RETURN]
        self.assertEqual(len(delegate), 2)
        self.assertEqual(len(returned), 2)
        self.assertEqual([e.data["to"] for e in delegate], ["a", "b"])
        self.assertEqual([e.step for e in delegate], [0, 1])
        self.assertFalse(any(e.data["refused"] for e in delegate))
        self.assertEqual([e.data["status"] for e in returned], ["FINISHED", "FINISHED"])
        self.assertEqual([e.data["failed"] for e in returned], [False, False])
        self.assertEqual([e.data["output_len"] for e in returned], [1, 1])


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_template_renders_input_prev_and_steps(self) -> None:
        first = StubAgent("first", output="FIRSTOUT")
        second = StubAgent("second", output="SECONDOUT")
        second_step = SequentialStep(
            agent=second,
            input_template="raw={input} prev={prev} first={steps[first]}",
        )
        team = SequentialAgent([first, second_step])
        await team.arun("TASK")
        self.assertEqual(
            second.calls[0]["input"], "raw=TASK prev=FIRSTOUT first=FIRSTOUT"
        )

    async def test_steps_map_sees_only_completed_stages(self) -> None:
        first = StubAgent("first", output="F")
        second = StubAgent("second", output="S")
        third = StubAgent("third", output="T")
        third_step = SequentialStep(
            agent=third, input_template="{steps[first]}+{steps[second]}"
        )
        team = SequentialAgent([first, second, third_step])
        await team.arun("x")
        self.assertEqual(third.calls[0]["input"], "F+S")

    async def test_unknown_top_level_key_is_left_literal(self) -> None:
        """`render_template` 用 `_SafeDict`：缺的顶层 key 保留 `{key}` 字面量而不是抛。"""
        agent = StubAgent("a", output="A")
        step = SequentialStep(agent=agent, input_template="{input} {nope}")
        await SequentialAgent([step]).arun("hello")
        self.assertEqual(agent.calls[0]["input"], "hello {nope}")

    async def test_unrenderable_template_raises_config_error(self) -> None:
        agent = StubAgent("a", output="A")
        step = SequentialStep(agent=agent, input_template="{steps[missing]}")
        with self.assertRaises(ConfigError):
            await SequentialAgent([step]).arun("x")

    async def test_first_step_can_override_default_input_template(self) -> None:
        agent = StubAgent("a", output="A")
        step = SequentialStep(agent=agent, input_template="pre:{input}:post")
        await SequentialAgent([step]).arun("X")
        self.assertEqual(agent.calls[0]["input"], "pre:X:post")


# ======================================================================================
# 黑板
# ======================================================================================


class BlackboardWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_stage_is_written_to_the_blackboard(self) -> None:
        team = SequentialAgent([StubAgent("a", output="A"), StubAgent("b", output="B")])
        await team.arun("x")
        entry = team.blackboard.read_entry("a")
        self.assertEqual(entry.value, "A")
        self.assertEqual(entry.tags, ("stage",))
        self.assertEqual(entry.author, "a")
        self.assertEqual(team.blackboard.read("b"), "B")

    async def test_output_key_overrides_the_blackboard_key(self) -> None:
        step = SequentialStep(agent=StubAgent("a", output="A"), output_key="shared")
        team = SequentialAgent([step])
        await team.arun("x")
        self.assertEqual(team.blackboard.read("shared"), "A")
        self.assertNotIn("a", team.blackboard)

    async def test_shared_blackboard_is_visible_outside(self) -> None:
        from liteagent.multiagent import Blackboard

        board = Blackboard()
        team = SequentialAgent([StubAgent("a", output="A")], blackboard=board)
        await team.arun("x")
        self.assertIs(team.blackboard, board)
        self.assertEqual(board.read("a"), "A")

    async def test_metadata_carries_a_blackboard_snapshot(self) -> None:
        team = SequentialAgent([StubAgent("a", output="A"), StubAgent("b", output="B")])
        result = await team.arun("x")
        self.assertEqual(result.metadata["blackboard"], {"a": "A", "b": "B"})


# ======================================================================================
# usage / steps / metadata 聚合
# ======================================================================================


class ResultAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_is_summed_over_all_stages(self) -> None:
        first = StubAgent("a", output="A", usage=TokenUsage(prompt_tokens=1, completion_tokens=2))
        second = StubAgent("b", output="B", usage=TokenUsage(prompt_tokens=3, completion_tokens=4))
        third = StubAgent("c", output="C", usage=TokenUsage(prompt_tokens=5, completion_tokens=6))
        result = await SequentialAgent([first, second, third]).arun("x")
        self.assertEqual(result.usage.prompt_tokens, 9)
        self.assertEqual(result.usage.completion_tokens, 12)
        self.assertEqual(result.usage.total_tokens, 21)

    async def test_child_usage_objects_are_not_mutated(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
        agent = StubAgent("a", output="A", usage=usage)
        await SequentialAgent([agent]).arun("x")
        self.assertEqual(usage.prompt_tokens, 1)
        self.assertEqual(usage.total_tokens, 2)

    async def test_steps_counter_is_summed(self) -> None:
        team = SequentialAgent([
            StubAgent("a", output="A", steps=2),
            StubAgent("b", output="B", steps=3),
        ])
        result = await team.arun("x")
        self.assertEqual(result.steps, 5)

    async def test_metadata_steps_summarises_each_stage(self) -> None:
        team = SequentialAgent([
            StubAgent("a", output="A", steps=2),
            StubAgent("b", output="B", steps=3),
        ])
        result = await team.arun("x")
        steps = result.metadata["steps"]
        self.assertEqual([item["name"] for item in steps], ["a", "b"])
        self.assertEqual([item["status"] for item in steps], ["FINISHED", "FINISHED"])
        self.assertEqual([item["steps"] for item in steps], [2, 3])
        self.assertEqual([item["output"] for item in steps], ["A", "B"])
        for item in steps:
            self.assertEqual(
                set(item), {"name", "status", "steps", "output", "duration_ms"}
            )
        self.assertGreaterEqual(steps[0]["duration_ms"], 0.0)

    async def test_result_state_is_the_last_child_state(self) -> None:
        first = StubAgent("a", output="A")
        second = StubAgent("b", output="B")
        result = await SequentialAgent([first, second]).arun("x")
        self.assertIs(result.state, second.calls[0]["state"])
        self.assertIsNot(result.state, first.calls[0]["state"])


# ======================================================================================
# 输出截断
# ======================================================================================


class TruncationTests(unittest.IsolatedAsyncioTestCase):
    async def test_max_chars_truncates_before_handing_to_the_next_step(self) -> None:
        long_output = "H" * 60 + "T" * 60  # 120 字符
        first = StubAgent("a", output=long_output)
        second = StubAgent("b", output="B")
        step = SequentialStep(agent=first, max_chars=40)
        team = SequentialAgent([step, second])

        await team.arun("x")

        handoff = second.calls[0]["input"]
        self.assertLess(len(handoff), len(long_output))
        self.assertIn("[truncated", handoff)
        self.assertTrue(handoff.startswith("H" * 28))   # head 70%
        self.assertTrue(handoff.endswith("T" * 12))     # tail 30%
        self.assertEqual(team.blackboard.read("a"), handoff)  # 黑板书的是截断后的版本

    async def test_max_chars_zero_means_no_truncation(self) -> None:
        long_output = "x" * 500
        first = StubAgent("a", output=long_output)
        second = StubAgent("b", output="B")
        team = SequentialAgent([SequentialStep(agent=first, max_chars=0), second])
        await team.arun("x")
        self.assertEqual(second.calls[0]["input"], long_output)

    async def test_short_output_is_not_touched(self) -> None:
        first = StubAgent("a", output="short")
        second = StubAgent("b", output="B")
        team = SequentialAgent([SequentialStep(agent=first, max_chars=1000), second])
        await team.arun("x")
        self.assertEqual(second.calls[0]["input"], "short")


# ======================================================================================
# 失败处理：propagate_failure 三种 + 两条 output 公式 + optional
# ======================================================================================


class FailureHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_raise_mode_raises_delegation_error(self) -> None:
        failing = _failed_agent("b", output="partial")
        team = SequentialAgent(
            [StubAgent("a", output="A"), failing],
            config=TeamConfig(propagate_failure="raise"),
        )
        with self.assertRaises(DelegationError) as ctx:
            await team.arun("x")
        self.assertEqual(ctx.exception.from_agent, "sequential")
        self.assertEqual(ctx.exception.to_agent, "b")

    async def test_return_mode_stops_and_uses_the_failing_stage_partial_output(self) -> None:
        """output 公式第一支：失败阶段有部分输出 -> 用它。"""
        third = StubAgent("c", output="C")
        failing = _failed_agent("b", output="PARTIAL")
        team = SequentialAgent(
            [StubAgent("a", output="A"), failing, third],
            config=TeamConfig(propagate_failure="return"),
        )
        result = await team.arun("x")

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.output, "PARTIAL")      # 优先失败阶段的部分输出
        self.assertEqual(result.metadata["failed_stage"], "b")
        self.assertEqual([item["name"] for item in result.metadata["steps"]], ["a"])
        self.assertEqual(third.calls, [])               # 立即停止

    async def test_return_mode_falls_back_to_previous_output_when_output_is_empty(self) -> None:
        """output 公式第二支：失败阶段输出为空串 -> 回退上一步输出。"""
        failing = _failed_agent("b", output="")
        team = SequentialAgent(
            [StubAgent("a", output="PREVIOUS"), failing],
            config=TeamConfig(propagate_failure="return"),
        )
        result = await team.arun("x")
        self.assertEqual(result.output, "PREVIOUS")
        self.assertEqual(result.status, AgentStatus.FAILED)

    async def test_return_mode_falls_back_to_empty_string_at_the_first_stage(self) -> None:
        failing = _failed_agent("a", output="")
        team = SequentialAgent(
            [failing], config=TeamConfig(propagate_failure="return")
        )
        result = await team.arun("x")
        self.assertEqual(result.output, "")
        self.assertEqual(result.status, AgentStatus.FAILED)

    async def test_continue_mode_injects_an_observation_and_keeps_going(self) -> None:
        failing = _failed_agent("b", output="")
        third = StubAgent("c", output="C")
        team = SequentialAgent(
            [StubAgent("a", output="A"), failing, third],
            config=TeamConfig(propagate_failure="continue"),
        )
        result = await team.arun("x")

        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.output, "C")
        forwarded = third.calls[0]["input"]
        self.assertTrue(forwarded.startswith("[stage b failed:"), msg=forwarded)
        names = [item["name"] for item in result.metadata["steps"]]
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(result.metadata["steps"][1]["status"], "FAILED")

    async def test_optional_stage_ignores_raise_mode(self) -> None:
        failing = _failed_agent("b", output="")
        third = StubAgent("c", output="C")
        team = SequentialAgent(
            [
                StubAgent("a", output="A"),
                SequentialStep(agent=failing, optional=True),
                third,
            ],
            config=TeamConfig(propagate_failure="raise"),
        )
        result = await team.arun("x")
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(result.output, "C")
        self.assertTrue(third.calls[0]["input"].startswith("[stage b failed:"))

    async def test_invalid_propagate_failure_mode_raises_config_error(self) -> None:
        team = SequentialAgent(
            [_failed_agent("a", output="")],
            config=TeamConfig(propagate_failure="bogus"),
        )
        with self.assertRaises(ConfigError):
            await team.arun("x")

    async def test_last_stage_failure_in_return_mode_keeps_failed_status(self) -> None:
        team = SequentialAgent(
            [StubAgent("a", output="A"), _failed_agent("b", output="")],
            config=TeamConfig(propagate_failure="return"),
        )
        result = await team.arun("x")
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.output, "A")


# ======================================================================================
# 三道闸：深度 / 预算 / 环（§10.1 的 `_check_depth`）
# ======================================================================================


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_max_rounds_zero_makes_the_budget_exhausted(self) -> None:
        team = SequentialAgent(
            [StubAgent("a", output="A")], config=TeamConfig(max_rounds=0)
        )
        with self.assertRaises(MaxDepthExceededError):
            await team.arun("x")

    async def test_max_depth_is_enforced(self) -> None:
        team = SequentialAgent(
            [StubAgent("a", output="A")], config=TeamConfig(max_depth=-1)
        )
        with self.assertRaises(MaxDepthExceededError):
            await team.arun("x")

    async def test_cycle_detection_rejects_a_repeated_ancestor(self) -> None:
        team = SequentialAgent([StubAgent("a", output="A")])
        incoming = DelegationContext(stack=["sequential", "a"], depth=1, budget=5)
        with self.assertRaises(CycleDetectedError):
            await team.arun("x", context=incoming)

    async def test_cycle_detection_can_be_disabled(self) -> None:
        agent = StubAgent("a", output="A")
        team = SequentialAgent(
            [agent], config=TeamConfig(enable_cycle_detection=False)
        )
        incoming = DelegationContext(stack=["sequential", "a"], depth=1, budget=5)
        result = await team.arun("x", context=incoming)
        self.assertEqual(result.status, AgentStatus.FINISHED)
        self.assertEqual(len(agent.calls), 1)

    async def test_incoming_context_is_used_as_the_parent(self) -> None:
        agent = StubAgent("a", output="A")
        team = SequentialAgent([agent])
        incoming = DelegationContext(
            stack=["root"], depth=2, budget=4, root_run_id="run_root"
        )
        await team.arun("x", context=incoming)
        child_ctx = agent.calls[0]["state"].scratchpad["delegation"]
        self.assertEqual(child_ctx.stack, ["root", "a"])
        self.assertEqual(child_ctx.depth, 3)
        self.assertEqual(child_ctx.budget, 3)


# ======================================================================================
# share_memory
# ======================================================================================


class ShareMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_share_memory_true_injects_one_memory_into_every_worker(self) -> None:
        first = StubAgent("a", output="A")
        second = StubAgent("b", output="B")
        team = SequentialAgent([first, second], config=TeamConfig(share_memory=True))

        await team.arun("x")

        self.assertIsNotNone(first.memory)
        self.assertIs(first.memory, second.memory)
        self.assertIs(first.memory, team._shared_memory)

    async def test_share_memory_false_leaves_worker_memory_untouched(self) -> None:
        sentinel = object()
        first = StubAgent("a", output="A", memory=sentinel)
        second = StubAgent("b", output="B")
        team = SequentialAgent([first, second])  # 默认 share_memory=False

        await team.arun("x")

        self.assertIs(first.memory, sentinel)
        self.assertIsNone(second.memory)

    async def test_explicit_shared_memory_overrides_lazy_creation(self) -> None:
        sentinel = object()
        agent = StubAgent("a", output="A")
        team = SequentialAgent([agent], config=TeamConfig(share_memory=True))
        team._shared_memory = sentinel
        await team.arun("x")
        self.assertIs(agent.memory, sentinel)


# ======================================================================================
# 工厂 + describe
# ======================================================================================


class BuildTeamTests(unittest.IsolatedAsyncioTestCase):
    def test_team_config_is_reexported_from_multiagent(self) -> None:
        self.assertIs(TeamConfigReexport, TeamConfig)

    def test_build_team_creates_a_sequential_agent(self) -> None:
        team = build_team([StubAgent("a")], mode="Sequential")
        self.assertIsInstance(team, SequentialAgent)
        self.assertEqual(team.name, "team")

    def test_build_team_unknown_mode_lists_valid_modes(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            build_team([], mode="wat")
        self.assertIn("sequential", str(ctx.exception))
        self.assertIn("hierarchical", str(ctx.exception))

    def test_build_team_hierarchical_requires_manager(self) -> None:
        with self.assertRaises(ConfigError):
            build_team([StubAgent("a")], mode="hierarchical")

    def test_describe_lists_agents_and_config(self) -> None:
        team = build_team([StubAgent("a"), StubAgent("b")])
        info = team.describe()
        self.assertEqual(info["type"], "SequentialAgent")
        self.assertEqual(info["agents"], ["a", "b"])
        self.assertEqual(info["blackboard_entries"], 0)
        self.assertIn("propagate_failure", info["config"])

    async def test_run_inside_a_running_loop_raises_config_error(self) -> None:
        """`run()` 是 `run_sync` 包装：已经在 loop 里时**必须**抛（§5.2 的 R-LOOP 红线）。"""
        team = SequentialAgent([StubAgent("a", output="A")])
        with self.assertRaises(ConfigError):
            team.run("x")


class SyncEntryPointTests(unittest.TestCase):
    def test_run_sync_wrapper_delegates_to_arun(self) -> None:
        team = SequentialAgent([StubAgent("a", output="A")])
        result = team.run("x")
        self.assertEqual(result.output, "A")
        self.assertEqual(result.status, AgentStatus.FINISHED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
