from __future__ import annotations

"""``liteagent/llm/registry.py`` 的单元测试（§6.5）。

覆盖 §12 对 ``test_llm_registry.py`` 冻结的覆盖点：

* ``build_llm`` / ``get_llm`` 解析 ``provider:model@base_url``；
* 未知 provider 抛 ``ConfigError`` **且消息里列出 available**；
* 缺 key 抛 ``LLMAuthError``；
* ``reset_default_registry``。

用到默认注册表的用例在 ``tearDown`` 里调 ``reset_default_registry()``
（§12.1 冻结的测试卫生规则 1），保证与并行运行的其它文件互不污染。
"""

import os
import unittest
from unittest import mock

from liteagent.config import LLMConfig, RetryPolicy
from liteagent.errors import ConfigError, LLMAuthError
from liteagent.llm.base import LLMClient
from liteagent.llm.message import Message
from liteagent.llm.providers import (
    AnthropicChatClient,
    DeepSeekChatClient,
    EchoLLM,
    OpenAIChatClient,
    OpenAICompatibleClient,
)
from liteagent.llm.registry import (
    LLMRegistry,
    build_llm,
    get_default_registry,
    get_llm,
    reset_default_registry,
)
from liteagent.llm.transport import HTTPResponse
from tests.helpers import FakeTransport

BUILTIN_PROVIDERS = {"openai", "openai-compatible", "deepseek", "anthropic", "echo"}


class DefaultRegistryIsolationMixin:
    """凡是有可能改动默认注册表的用例都在 tearDown 里复原它。"""

    def tearDown(self) -> None:  # noqa: N802 - unittest 约定
        reset_default_registry()


class BuildLLMTests(DefaultRegistryIsolationMixin, unittest.TestCase):
    def test_build_llm_uses_the_config_provider(self) -> None:
        client = build_llm(LLMConfig(provider="echo", model="echo-1"))
        self.assertIsInstance(client, EchoLLM)
        self.assertEqual("echo-1", client.model)

    def test_build_llm_applies_overrides(self) -> None:
        client = build_llm(LLMConfig(provider="openai", api_key="k"), model="gpt-4o")
        self.assertIsInstance(client, OpenAIChatClient)
        self.assertEqual("gpt-4o", client.model)

    def test_build_llm_rejects_unknown_override_fields(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            build_llm(LLMConfig(provider="echo"), bas_url="http://typo")
        message = str(ctx.exception)
        self.assertIn("bas_url", message)
        self.assertIn("valid fields", message)
        self.assertIn("provider", message)

    def test_build_llm_deep_copies_the_config(self) -> None:
        config = LLMConfig(
            provider="echo", model="m1", extra_body={"a": 1}, retry_policy=RetryPolicy(max_retries=1)
        )
        client = build_llm(config)
        config.model = "m2"
        config.extra_body["a"] = 2
        config.retry_policy.max_retries = 99
        self.assertEqual("m1", client.model)
        self.assertEqual({"a": 1}, client.config.extra_body)
        self.assertEqual(1, client.config.retry_policy.max_retries)

    def test_build_llm_without_a_config_falls_back_to_environment(self) -> None:
        """``config=None`` 时用 ``LLMConfig.from_env()``（§6.5）。"""
        with mock.patch.dict(
            os.environ,
            {"LITEAGENT_PROVIDER": "echo", "LITEAGENT_MODEL": "env-model"},
            clear=True,
        ):
            client = build_llm()
        self.assertIsInstance(client, EchoLLM)
        self.assertEqual("env-model", client.model)

    def test_build_llm_unknown_provider_lists_available(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            build_llm(LLMConfig(provider="does-not-exist"))
        message = str(ctx.exception)
        self.assertIn("does-not-exist", message)
        self.assertIn("available:", message)
        for name in BUILTIN_PROVIDERS:
            with self.subTest(name=name):
                self.assertIn(name, message)


class RegistryTests(unittest.TestCase):
    def test_create_unknown_name_lists_available(self) -> None:
        registry = LLMRegistry({"alpha": lambda **kw: EchoLLM(**kw)})
        with self.assertRaises(ConfigError) as ctx:
            registry.create("beta")
        self.assertIn("available: alpha", str(ctx.exception))

    def test_register_rejects_duplicates_unless_override(self) -> None:
        registry = LLMRegistry()
        first = lambda **kw: EchoLLM(**kw)  # noqa: E731
        second = lambda **kw: EchoLLM(**kw)  # noqa: E731
        registry.register("p", first)
        with self.assertRaises(ConfigError):
            registry.register("p", second)
        registry.register("p", second, override=True)
        self.assertIs(second, registry._factories["p"])

    def test_register_rejects_empty_name(self) -> None:
        with self.assertRaises(ConfigError):
            LLMRegistry().register("", lambda **kw: EchoLLM(**kw))

    def test_available_is_sorted_and_contains_works(self) -> None:
        registry = LLMRegistry({"z": lambda **kw: EchoLLM(**kw), "a": lambda **kw: EchoLLM(**kw)})
        self.assertEqual(["a", "z"], registry.available())
        self.assertIn("a", registry)
        self.assertNotIn("b", registry)
        self.assertEqual(2, len(registry))

    def test_factories_mapping_is_copied(self) -> None:
        source = {"a": lambda **kw: EchoLLM(**kw)}
        registry = LLMRegistry(source)
        source["b"] = lambda **kw: EchoLLM(**kw)
        self.assertEqual(["a"], registry.available())


class GetLLMSpecTests(DefaultRegistryIsolationMixin, unittest.TestCase):
    def test_bare_provider(self) -> None:
        self.assertIsInstance(get_llm("echo"), EchoLLM)

    def test_provider_and_model(self) -> None:
        client = get_llm("openai:gpt-4o-mini")
        self.assertIsInstance(client, OpenAIChatClient)
        self.assertEqual("gpt-4o-mini", client.model)

    def test_provider_model_and_base_url(self) -> None:
        client = get_llm("openai-compatible:qwen@http://host:8000/v1")
        self.assertIsInstance(client, OpenAICompatibleClient)
        self.assertEqual("qwen", client.model)
        self.assertEqual("http://host:8000/v1", client.config.base_url)
        self.assertEqual("http://host:8000/v1/chat/completions", client._endpoint())

    def test_base_url_may_contain_a_colon(self) -> None:
        """先切 ``@`` 再切 ``:`` —— 否则 host:port 会被当成 model。"""
        client = get_llm("openai-compatible:qwen@http://localhost:11434/v1")
        self.assertEqual("qwen", client.model)
        self.assertEqual("http://localhost:11434/v1", client.config.base_url)

    def test_deepseek_uses_its_default_base_url(self) -> None:
        client = get_llm("deepseek:deepseek-chat")
        self.assertIsInstance(client, DeepSeekChatClient)
        self.assertEqual("deepseek-chat", client.model)
        self.assertEqual("https://api.deepseek.com/v1", client.config.resolved_base_url())

    def test_whitespace_is_trimmed(self) -> None:
        client = get_llm("  openai : gpt-4o-mini  ")
        self.assertIsInstance(client, OpenAIChatClient)
        self.assertEqual("gpt-4o-mini", client.model)

    def test_an_existing_client_passes_through_unchanged(self) -> None:
        client = get_llm("echo")
        self.assertIs(client, get_llm(client))

    def test_empty_spec_raises_config_error(self) -> None:
        for spec in ("", "   "):
            with self.subTest(spec=repr(spec)):
                with self.assertRaises(ConfigError):
                    get_llm(spec)

    def test_empty_provider_name_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            get_llm(":model")
        self.assertIn("provider name is empty", str(ctx.exception))

    def test_unknown_provider_message_lists_available(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            get_llm("gemini:pro")
        message = str(ctx.exception)
        self.assertIn("gemini", message)
        self.assertIn("available:", message)
        self.assertIn("anthropic", message)

    def test_kwargs_are_forwarded_to_the_config(self) -> None:
        client = get_llm("echo", temperature=0.5, timeout_s=7.0)
        self.assertEqual(0.5, client.config.temperature)
        self.assertEqual(7.0, client.config.timeout_s)

    def test_anthropic_spec(self) -> None:
        client = get_llm("anthropic:claude-3-5-haiku")
        self.assertIsInstance(client, AnthropicChatClient)
        self.assertEqual("claude-3-5-haiku", client.model)


class DefaultRegistryTests(DefaultRegistryIsolationMixin, unittest.TestCase):
    def test_lazy_load_registers_the_builtin_providers(self) -> None:
        registry = get_default_registry()
        self.assertEqual(sorted(BUILTIN_PROVIDERS), registry.available())
        # 单例：再次取用拿到同一个对象。
        self.assertIs(registry, get_default_registry())

    def test_reset_rebuilds_from_provider_classes_and_drops_custom_entries(self) -> None:
        registry = get_default_registry()
        registry.register("mine", lambda **kw: EchoLLM(**kw))
        self.assertIn("mine", get_default_registry().available())

        reset_default_registry()
        rebuilt = get_default_registry()
        self.assertIsNot(registry, rebuilt)
        self.assertEqual(sorted(BUILTIN_PROVIDERS), rebuilt.available())
        # 重置不等于"删掉内置 provider"：build_llm 依然可用。
        self.assertIsInstance(build_llm(LLMConfig(provider="echo")), EchoLLM)

    def test_providers_module_is_imported_lazily(self) -> None:
        """E2：provider 的导入必须写在函数体内（顶层 import 会被 AST 检查抓住）。"""
        import inspect

        from liteagent.llm import registry as registry_module

        source = inspect.getsource(registry_module.get_default_registry)
        self.assertIn("from liteagent.llm.providers import", source)
        # 顶层没有这条 import：模块级名字表里不应出现 PROVIDER_CLASSES。
        self.assertNotIn("PROVIDER_CLASSES", vars(registry_module))


class MissingApiKeyTests(DefaultRegistryIsolationMixin, unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_raises_llm_auth_error(self) -> None:
        transport = FakeTransport([HTTPResponse(status_code=200, text="{}")])
        client = build_llm(LLMConfig(provider="openai", api_key=None))
        client.transport = transport

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMAuthError) as ctx:
                await client.achat([Message.user("hi")])
        self.assertIn("missing API key", str(ctx.exception))
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertEqual(0, ctx.exception.status_code)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual([], transport.requests)

    async def test_api_key_from_environment_is_used(self) -> None:
        transport = FakeTransport(
            [
                HTTPResponse(
                    status_code=200,
                    text='{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}',
                )
            ]
        )
        client = build_llm(LLMConfig(provider="openai", api_key=None))
        client.transport = transport
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            await client.achat([Message.user("hi")])
        self.assertEqual("Bearer env-key", transport.requests[0].headers["Authorization"])

    async def test_echo_does_not_need_a_key(self) -> None:
        client = get_llm("echo")
        self.assertFalse(client.requires_api_key)
        with mock.patch.dict(os.environ, {}, clear=True):
            response = await client.achat([Message.user("hi")])
        self.assertIn("echo: received 1 message(s)", response.content)


class RegistryClientContractTests(unittest.TestCase):
    def test_every_builtin_factory_returns_an_llm_client(self) -> None:
        for name in sorted(BUILTIN_PROVIDERS):
            with self.subTest(provider=name):
                kwargs = {"config": LLMConfig(provider=name, api_key="k")}
                if name == "openai-compatible":
                    kwargs["config"] = LLMConfig(
                        provider=name, api_key="k", base_url="http://localhost:8000/v1"
                    )
                client = get_default_registry().create(name, **kwargs)
                self.assertIsInstance(client, LLMClient)
                self.assertTrue(client.name)
        reset_default_registry()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
