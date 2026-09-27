from __future__ import annotations

"""``liteagent/errors.py`` 的单元测试（§12 清单行：test_errors.py）。

覆盖重点（§12 冻结清单，逐条对应到用例）：
* 继承关系（§3.2 的完整异常树，逐条对照）
* ``retryable`` 类属性表（**逐条对照 §3.4 重试白名单**）
* ``to_dict``（§3.1 的 5 个键）
* ``__str__`` 含 context
* ``ToolSkippedError`` / ``ToolApprovalDeniedError`` / ``BudgetExceededError`` /
  ``RunTimeoutError`` 的字段

规范真值源：``docs/INTERFACES.md`` §3（549-693 行）、§3.2 异常树、§3.3 字段表、§3.4 白名单。
硬性要求：只用 stdlib ``unittest``、禁用裸 ``assert``、不联网、不睡时钟、不碰默认注册表。
"""

import unittest

from liteagent import errors as errors_module
from liteagent.errors import (
    AgentAbortedError,
    AgentError,
    BudgetExceededError,
    ConfigError,
    CycleDetectedError,
    DelegationError,
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
    LiteAgentError,
    MaxDepthExceededError,
    MaxStepsExceededError,
    MemoryStoreError,
    MultiAgentError,
    ReActParseError,
    RepeatedActionError,
    RunTimeoutError,
    SandboxViolationError,
    ScriptedExhaustedError,
    SerializationError,
    ToolApprovalDeniedError,
    ToolDefinitionError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRetryExhaustedError,
    ToolSkippedError,
    ToolTimeoutError,
    ToolValidationError,
    VersionConflictError,
)

# --------------------------------------------------------------------------------------
# §3.2 异常树（逐条转抄：类 -> 直接父类）
# --------------------------------------------------------------------------------------

PARENT_BY_CLASS: dict[type, type] = {
    LiteAgentError: Exception,
    ConfigError: LiteAgentError,
    SerializationError: LiteAgentError,
    ScriptedExhaustedError: LiteAgentError,
    SandboxViolationError: LiteAgentError,
    LLMError: LiteAgentError,
    LLMAuthError: LLMError,
    LLMBadRequestError: LLMError,
    LLMResponseFormatError: LLMError,
    LLMRateLimitError: LLMError,
    LLMTimeoutError: LLMError,
    LLMConnectionError: LLMError,
    ToolError: LiteAgentError,
    ToolNotFoundError: ToolError,
    ToolValidationError: ToolError,
    ToolDefinitionError: ToolError,
    ToolExecutionError: ToolError,
    ToolTimeoutError: ToolError,
    ToolSkippedError: ToolError,
    ToolApprovalDeniedError: ToolError,
    ToolRetryExhaustedError: ToolError,
    MemoryStoreError: LiteAgentError,
    ReActParseError: LiteAgentError,
    AgentError: LiteAgentError,
    MaxStepsExceededError: AgentError,
    RepeatedActionError: AgentError,
    BudgetExceededError: AgentError,
    RunTimeoutError: AgentError,
    AgentAbortedError: AgentError,
    MultiAgentError: LiteAgentError,
    DelegationError: MultiAgentError,
    MaxDepthExceededError: MultiAgentError,
    CycleDetectedError: MultiAgentError,
    VersionConflictError: MultiAgentError,
}

# --------------------------------------------------------------------------------------
# §3.4 重试白名单（逐条转抄；**唯一真值源**）
# --------------------------------------------------------------------------------------

RETRYABLE_TRUE = (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError,
    ToolTimeoutError,
)

# --------------------------------------------------------------------------------------
# §3.3 各异常的额外字段与默认值（逐条转抄）
# --------------------------------------------------------------------------------------

EXTRA_FIELD_DEFAULTS: dict[type, dict[str, object]] = {
    ConfigError: {},
    SerializationError: {"target": ""},
    ScriptedExhaustedError: {"consumed": 0},
    SandboxViolationError: {"path": "", "root": ""},
    LLMAuthError: {"status_code": 0},
    LLMBadRequestError: {"status_code": 0, "body": ""},
    LLMResponseFormatError: {"body": ""},
    LLMRateLimitError: {"status_code": 0, "retry_after_s": None},
    LLMTimeoutError: {"timeout_s": 0.0},
    LLMConnectionError: {"status_code": None},
    ToolNotFoundError: {"name": "", "available": []},
    ToolValidationError: {"errors": [], "tool_name": ""},
    ToolDefinitionError: {"tool_name": ""},
    ToolExecutionError: {"tool_name": "", "call_id": ""},
    ToolTimeoutError: {"tool_name": "", "timeout_s": 0.0},
    ToolSkippedError: {"tool_name": "", "reason": ""},
    ToolApprovalDeniedError: {"tool_name": "", "reason": ""},
    ToolRetryExhaustedError: {"attempts": 0, "last_error": None, "tool_name": ""},
    MemoryStoreError: {"store": ""},
    ReActParseError: {"raw": "", "offset": 0, "reason": ""},
    MaxStepsExceededError: {"max_steps": 0},
    RepeatedActionError: {"action_key": "", "count": 0},
    BudgetExceededError: {"limit": 0, "used": 0, "kind": ""},
    RunTimeoutError: {"timeout_s": 0.0, "elapsed_s": 0.0},
    AgentAbortedError: {"reason": ""},
    DelegationError: {"from_agent": "", "to_agent": ""},
    MaxDepthExceededError: {"depth": 0, "max_depth": 0},
    CycleDetectedError: {"stack": []},
    VersionConflictError: {"key": "", "expected": 0, "actual": 0},
}


# ======================================================================================
# 3.2 继承关系
# ======================================================================================


class ExceptionTreeTests(unittest.TestCase):
    """§3.2 的异常树逐条对照（35 个类，含基类）。"""

    def test_every_frozen_class_has_the_frozen_parent(self) -> None:
        for cls, parent in PARENT_BY_CLASS.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__bases__, (parent,), f"{cls.__name__} 的父类不对")

    def test_all_exported_errors_subclass_liteagent_error(self) -> None:
        for name in errors_module.__all__:
            with self.subTest(name=name):
                cls = getattr(errors_module, name)
                self.assertTrue(issubclass(cls, LiteAgentError))

    def test_public_exports_match_the_frozen_tree(self) -> None:
        self.assertEqual(set(errors_module.__all__), {c.__name__ for c in PARENT_BY_CLASS})

    def test_liteagent_error_is_not_plain_exception_subclass_of_errors(self) -> None:
        # 直接继承 Exception（异常体系的最底层，不引入任何 liteagent 依赖）。
        self.assertEqual(LiteAgentError.__mro__[1], Exception)

    def test_sandbox_violation_is_not_a_tool_error(self) -> None:
        # §3.3 特别标注：SandboxViolationError 直接继承 LiteAgentError，
        # executor 对它是"原样保留类型"而不是包成 ToolExecutionError。
        self.assertFalse(issubclass(SandboxViolationError, ToolError))

    def test_agent_and_multiagent_are_sibling_subtrees(self) -> None:
        self.assertFalse(issubclass(AgentError, MultiAgentError))
        self.assertFalse(issubclass(MultiAgentError, AgentError))

    def test_tool_errors_are_not_llm_errors(self) -> None:
        self.assertFalse(issubclass(ToolError, LLMError))
        self.assertFalse(issubclass(LLMError, ToolError))


# ======================================================================================
# 3.4 重试白名单
# ======================================================================================


class RetryableAttributeTests(unittest.TestCase):
    """§3.4：``retryable`` 只对 4 个异常为 True，其余全部 False。"""

    def test_retryable_table_matches_spec(self) -> None:
        for cls, parent in PARENT_BY_CLASS.items():
            expected = cls in RETRYABLE_TRUE
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.retryable, expected)

    def test_only_four_exceptions_are_retryable(self) -> None:
        retryable = [
            getattr(errors_module, name)
            for name in errors_module.__all__
            if getattr(errors_module, name).retryable
        ]
        self.assertEqual(
            sorted(c.__name__ for c in retryable),
            sorted(c.__name__ for c in RETRYABLE_TRUE),
        )

    def test_retryable_is_a_class_attribute_not_instance(self) -> None:
        # 类属性是"异常性质"的声明，实例不应各自复制一份（LLMRateLimitError 也不覆盖它）。
        self.assertEqual(LLMRateLimitError.retryable, True)
        self.assertNotIn("retryable", LLMRateLimitError(429).__dict__)

    def test_retry_after_s_defaults_to_none_everywhere(self) -> None:
        for cls in PARENT_BY_CLASS:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.retry_after_s, None)

    def test_llm_rate_limit_accepts_retry_after_s(self) -> None:
        exc = LLMRateLimitError(429, retry_after_s="1.5")  # type: ignore[arg-type]
        self.assertEqual(exc.retry_after_s, 1.5)

    def test_llm_rate_limit_without_retry_after_keeps_class_default(self) -> None:
        exc = LLMRateLimitError(429)
        self.assertIsNone(exc.retry_after_s)
        self.assertNotIn("retry_after_s", exc.__dict__)

    def test_llm_rate_limit_direct_assignment_works(self) -> None:
        exc = LLMRateLimitError(429)
        exc.retry_after_s = 2.5
        self.assertEqual(exc.retry_after_s, 2.5)

    def test_tool_execution_error_can_be_marked_retryable_by_author(self) -> None:
        # §3.4 表格备注：工具作者对**幂等**操作可显式置 True（实例级覆盖）。
        exc = ToolExecutionError("read_file", "call_1")
        self.assertFalse(exc.retryable)
        exc.retryable = True
        self.assertTrue(exc.retryable)
        self.assertFalse(ToolExecutionError.retryable)  # 类属性未被污染


# ======================================================================================
# 3.3 字段表
# ======================================================================================


class ExtraFieldTests(unittest.TestCase):
    """§3.3：每个异常的额外字段（名字 + 默认值）逐条对照。"""

    def test_defaults_of_every_frozen_field(self) -> None:
        for cls, expected_fields in EXTRA_FIELD_DEFAULTS.items():
            exc = cls()  # 全部字段都有默认值
            for field, default in expected_fields.items():
                with self.subTest(cls=cls.__name__, field=field):
                    self.assertTrue(hasattr(exc, field), f"{cls.__name__} 缺字段 {field}")
                    self.assertEqual(getattr(exc, field), default)

    def test_no_frozen_class_gains_unexpected_public_fields(self) -> None:
        # 反向检查：实例 __dict__ 里的键只允许是 message/context/cause + §3.3 的字段
        # （retry_after_s 这类"未显式给出就继承类属性"的字段允许不在 __dict__ 里）。
        allowed = {"message", "context", "cause"}
        for cls, expected_fields in EXTRA_FIELD_DEFAULTS.items():
            with self.subTest(cls=cls.__name__):
                exc = cls()
                self.assertLessEqual(set(exc.__dict__) - allowed, set(expected_fields))

    def test_positional_field_construction(self) -> None:
        # §3.1 规则 1：字段排在 message 之前，因此 `ToolNotFoundError("read_file")` 合法。
        exc = ToolNotFoundError("read_file")
        self.assertEqual(exc.name, "read_file")
        self.assertEqual(exc.available, [])

    def test_tool_not_found_copies_available_list(self) -> None:
        names = ["read_file", "write_file"]
        exc = ToolNotFoundError("nope", names)
        names.append("mutated")
        self.assertEqual(exc.available, ["read_file", "write_file"])

    def test_serialization_error_target_round_trip(self) -> None:
        exc = SerializationError("ToolCall.id", message="missing")
        self.assertEqual(exc.target, "ToolCall.id")
        self.assertEqual(exc.message, "missing")

    def test_scripted_exhausted_consumed(self) -> None:
        self.assertEqual(ScriptedExhaustedError(consumed=3).consumed, 3)

    def test_sandbox_violation_denylist_shape(self) -> None:
        # §3.3：denylist 命中时 path=command、root="denylist:<reason>"。
        exc = SandboxViolationError(path="rm -rf /", root="denylist:destructive")
        self.assertEqual(exc.path, "rm -rf /")
        self.assertEqual(exc.root, "denylist:destructive")

    def test_tool_skipped_error_fields(self) -> None:
        # §12 点名的 v2 异常：fail_fast 取消的兄弟调用。
        exc = ToolSkippedError(tool_name="write_file", reason="cancelled_by_fail_fast")
        self.assertEqual(exc.tool_name, "write_file")
        self.assertEqual(exc.reason, "cancelled_by_fail_fast")
        self.assertFalse(exc.retryable)

    def test_tool_approval_denied_error_fields(self) -> None:
        exc = ToolApprovalDeniedError(tool_name="run_shell", reason="denied by policy")
        self.assertEqual(exc.tool_name, "run_shell")
        self.assertEqual(exc.reason, "denied by policy")
        self.assertFalse(exc.retryable)

    def test_tool_approval_denied_error_message_is_the_frozen_text(self) -> None:
        # §7.4.1 步骤 4.5 逐字冻结：ERROR(ToolApprovalDeniedError): this tool requires human approval
        exc = ToolApprovalDeniedError(tool_name="run_shell")
        self.assertEqual(exc.message, "this tool requires human approval")
        self.assertEqual(str(exc), "this tool requires human approval")

    def test_budget_exceeded_error_fields(self) -> None:
        exc = BudgetExceededError(limit=200000, used=200001, kind="total_tokens")
        self.assertEqual(exc.limit, 200000)
        self.assertEqual(exc.used, 200001)
        self.assertEqual(exc.kind, "total_tokens")
        self.assertFalse(exc.retryable)

    def test_run_timeout_error_fields(self) -> None:
        exc = RunTimeoutError(timeout_s=30.0, elapsed_s=31.5)
        self.assertEqual(exc.timeout_s, 30.0)
        self.assertEqual(exc.elapsed_s, 31.5)
        self.assertFalse(exc.retryable)

    def test_cycle_detected_error_copies_stack(self) -> None:
        stack = ["a", "b"]
        exc = CycleDetectedError(stack)
        stack.append("c")
        self.assertEqual(exc.stack, ["a", "b"])

    def test_version_conflict_error_fields(self) -> None:
        exc = VersionConflictError(key="plan", expected=1, actual=2)
        self.assertEqual((exc.key, exc.expected, exc.actual), ("plan", 1, 2))

    def test_tool_retry_exhausted_error_holds_last_error(self) -> None:
        root = LLMTimeoutError(timeout_s=5.0)
        exc = ToolRetryExhaustedError(attempts=3, last_error=root, tool_name="search")
        self.assertEqual(exc.attempts, 3)
        self.assertIs(exc.last_error, root)
        self.assertEqual(exc.tool_name, "search")

    def test_tool_validation_error_copies_errors_list(self) -> None:
        problems = ["a must be int"]
        exc = ToolValidationError(problems, tool_name="add")
        problems.append("extra")
        self.assertEqual(exc.errors, ["a must be int"])
        self.assertEqual(exc.tool_name, "add")

    def test_react_parse_error_fields(self) -> None:
        exc = ReActParseError(raw="{bad", offset=1, reason="not json")
        self.assertEqual((exc.raw, exc.offset, exc.reason), ("{bad", 1, "not json"))

    def test_default_messages_are_non_empty_for_field_rich_errors(self) -> None:
        # 空 message 会让 ToolResult.failure() 回灌给模型的文本退化成 "ERROR(X): "。
        for cls in (
            ToolSkippedError,
            ToolApprovalDeniedError,
            BudgetExceededError,
            RunTimeoutError,
            ToolExecutionError,
            ToolNotFoundError,
            LLMTimeoutError,
        ):
            with self.subTest(cls=cls.__name__):
                exc = cls()
                self.assertNotEqual(exc.message, "", f"{cls.__name__} 的默认 message 为空")


# ======================================================================================
# 3.1 基类语义
# ======================================================================================


class LiteAgentErrorBaseTests(unittest.TestCase):
    """§3.1：构造规则、``to_dict``、``__str__``。"""

    def test_message_is_first_positional_with_empty_default(self) -> None:
        self.assertEqual(LiteAgentError().message, "")
        self.assertEqual(LiteAgentError("boom").message, "boom")

    def test_context_defaults_to_empty_dict(self) -> None:
        self.assertEqual(LiteAgentError().context, {})
        self.assertIsNone(LiteAgentError().cause)

    def test_context_is_copied(self) -> None:
        source = {"a": 1}
        exc = LiteAgentError("x", context=source)
        source["a"] = 2
        self.assertEqual(exc.context, {"a": 1})

    def test_args_only_contains_the_message(self) -> None:
        # 额外字段是普通属性，不污染 args（否则日志里会多出一长串元组）。
        exc = LLMAuthError(401, context={"k": "v"}, cause=ValueError("x"))
        self.assertEqual(len(exc.args), 1)
        self.assertEqual(exc.args, (exc.message,))
        self.assertEqual(exc.context, {"k": "v"})
        self.assertEqual(exc.status_code, 401)

    def test_cause_is_stored_without_setting_dunder_cause(self) -> None:
        root = ValueError("root")
        exc = LiteAgentError("x", cause=root)
        self.assertIs(exc.cause, root)
        self.assertIsNone(exc.__cause__)

    def test_to_dict_has_the_five_frozen_keys(self) -> None:
        exc = LiteAgentError("boom", context={"a": 1})
        payload = exc.to_dict()
        self.assertEqual(
            set(payload), {"type", "message", "context", "retryable", "cause"}
        )
        self.assertEqual(payload["type"], "LiteAgentError")
        self.assertEqual(payload["message"], "boom")
        self.assertEqual(payload["context"], {"a": 1})
        self.assertIs(payload["retryable"], False)
        self.assertIsNone(payload["cause"])

    def test_to_dict_type_is_the_concrete_class_name(self) -> None:
        self.assertEqual(ToolNotFoundError("read_file").to_dict()["type"], "ToolNotFoundError")
        self.assertEqual(BudgetExceededError().to_dict()["type"], "BudgetExceededError")

    def test_to_dict_retryable_reflects_the_class_attribute(self) -> None:
        self.assertIs(LLMTimeoutError().to_dict()["retryable"], True)
        self.assertIs(ToolValidationError().to_dict()["retryable"], False)

    def test_to_dict_cause_is_stringified(self) -> None:
        exc = LiteAgentError("x", cause=ValueError("root cause"))
        self.assertEqual(exc.to_dict()["cause"], "root cause")

    def test_to_dict_context_is_a_copy(self) -> None:
        exc = LiteAgentError("x", context={"a": 1})
        payload = exc.to_dict()
        payload["context"]["a"] = 99
        self.assertEqual(exc.context, {"a": 1})

    def test_to_dict_does_not_expand_extra_fields(self) -> None:
        # §3.1：to_dict 是给 trace 看的，额外字段是给代码看的，两者读者不同。
        payload = BudgetExceededError(limit=1, used=2, kind="total_tokens").to_dict()
        self.assertNotIn("limit", payload)
        self.assertNotIn("kind", payload)

    def test_str_without_context_is_the_message(self) -> None:
        self.assertEqual(str(LiteAgentError("boom")), "boom")

    def test_str_includes_context_pairs(self) -> None:
        exc = LiteAgentError("boom", context={"a": 1, "b": "x"})
        self.assertEqual(str(exc), "boom (a=1, b=x)")

    def test_str_with_empty_message_starts_with_the_parenthesis(self) -> None:
        self.assertEqual(str(LiteAgentError("", context={"a": 1})), "(a=1)")

    def test_str_of_subclass_with_default_message(self) -> None:
        # 子类的默认 message 由自身字段渲染，因此 str() 一定可读、可用于回灌。
        exc = ToolNotFoundError("read_file", ["read_file", "write_file"])
        self.assertIn("read_file", str(exc))
        self.assertIn("write_file", str(exc))

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        json.dumps(BudgetExceededError(limit=1, used=2, kind="total_tokens").to_dict())

    def test_catching_by_base_class_works(self) -> None:
        with self.assertRaises(LiteAgentError) as ctx:
            raise ToolSkippedError(tool_name="f", reason="cancelled_by_fail_fast")
        self.assertIsInstance(ctx.exception, ToolSkippedError)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
