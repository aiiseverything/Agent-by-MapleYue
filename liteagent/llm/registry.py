from __future__ import annotations

"""provider 注册表与装配入口（§6.5）。

**为什么要有注册表**：把"provider 名 -> 构造方式"这张表从调用点里抽出来，带来三件事：
1. `Agent`/CLI/示例只认识字符串（`get_llm("deepseek")`），不认识具体类 —— 加一家 provider
   不需要改任何调用点；
2. 测试可以注册假 provider（`registry.register("fake", FakeClient)`）而不碰全局状态；
3. 未知 provider 能给出**可行动的错误**（列出 available），而不是 `AttributeError`/`KeyError`。

`providers` 是**函数内延迟导入**的（§1.1 的 E2）：只 import 注册表不应该拉起所有 provider
（那会把 transport 的三方库探测、warnings 等副作用都带进来）。
"""

import copy
import threading
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any

from liteagent.config import LLMConfig
from liteagent.errors import ConfigError
from liteagent.llm.base import LLMClient

__all__ = [
    "LLMRegistry",
    "get_default_registry",
    "reset_default_registry",
    "build_llm",
    "get_llm",
]

#: 默认注册表的模块级单例。`None` 表示"尚未加载或已被 reset"（下次取用时重建）。
_DEFAULT_REGISTRY: "LLMRegistry | None" = None
#: 用 threading.Lock 而不是 asyncio.Lock：注册表可能被同步代码（CLI/测试 setUp）访问，
#: 且跨线程生效的互斥只能用 threading 原语（§13 红线 13）。
_REGISTRY_LOCK = threading.Lock()


class LLMRegistry:
    """`provider 名 -> 工厂` 的注册表。

    工厂的调用约定是 `factory(**kwargs)`，`build_llm` 传的是 `config=<LLMConfig>`；
    `PROVIDER_CLASSES` 里的类恰好满足这个约定（它们都收 `config` 关键字）。
    """

    def __init__(self, factories: Mapping[str, Callable[..., LLMClient]] | None = None) -> None:
        # 复制一份：调用方之后改自己的 dict 不应影响注册表（§13 红线 11 的精神）。
        self._factories: dict[str, Callable[..., LLMClient]] = dict(factories or {})

    def register(
        self, name: str, factory: Callable[..., LLMClient], *, override: bool = False
    ) -> None:
        """注册一个 provider 工厂。重名且 `override=False` -> `ConfigError`。

        默认不允许覆盖是刻意的：静默覆盖会让"我注册的 fake 为什么没生效"变成一个
        需要读源码才能回答的问题。要替换就显式写 `override=True`。
        """
        if not name:
            raise ConfigError("provider name must be a non-empty string")
        if name in self._factories and not override:
            raise ConfigError(
                f"LLM provider {name!r} is already registered; "
                f"pass override=True to replace it (available: {self._available_text()})"
            )
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> LLMClient:
        """用注册的工厂造一个 client。

        未知 `name` -> `ConfigError`（消息里**必须**列出 available()，
        [v2 变更] 原先漏了这句 —— 没有可用列表时用户只能猜拼写）。
        """
        factory = self._factories.get(name)
        if factory is None:
            raise ConfigError(
                f"unknown LLM provider {name!r}; available: {self._available_text()}"
            )
        return factory(**kwargs)

    def available(self) -> list[str]:
        """已注册的 provider 名（**已排序**，保证错误消息与断言稳定）。"""
        return sorted(self._factories)

    def _available_text(self) -> str:
        names = self.available()
        return ", ".join(names) if names else "<none registered>"

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)


def get_default_registry() -> LLMRegistry:
    """首次调用时从 `providers.PROVIDER_CLASSES` 懒加载（函数内 import providers）。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY
    with _REGISTRY_LOCK:
        # 双重检查：并发首次调用只会构建一次，且后来的线程一定看到完整注册表。
        if _DEFAULT_REGISTRY is None:
            registry = LLMRegistry()
            from liteagent.llm.providers import PROVIDER_CLASSES  # E2：延迟导入

            for name, factory in PROVIDER_CLASSES.items():
                registry.register(name, factory)
            _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """[v2 新增] 清空默认 LLM 注册表（测试隔离用，与 tools 侧同名函数对称）。

    实现是**置空 + 下次取用时按 `PROVIDER_CLASSES` 重建**，而不是"删掉内置 provider"：
    这样 `reset_default_registry()` 之后 `build_llm()` 依然可用（测试的 tearDown 经常
    无脑调它，若把内置 provider 也清掉，后面的用例会以"未知 provider"的方式失败 ——
    那是测试相互污染，不是被测代码的问题）。
    """
    global _DEFAULT_REGISTRY
    with _REGISTRY_LOCK:
        _DEFAULT_REGISTRY = None


def _copy_config(config: LLMConfig) -> LLMConfig:
    """深一层地复制配置，避免调用方之后改 config 影响已造好的 client。

    `dataclasses.replace` 只做浅拷贝，`retry_policy`/`extra_headers`/`extra_body`
    仍与外部共享同一对象 —— agent 侧改一次温度就顺手改了别人的配置，是最难复现的
    那类串扰，所以这里显式再 copy 一层。
    """
    return replace(
        config,
        retry_policy=replace(config.retry_policy),
        extra_headers=dict(config.extra_headers),
        extra_body=copy.deepcopy(config.extra_body),
    )


def build_llm(config: LLMConfig | None = None, **overrides: Any) -> LLMClient:
    """由 `LLMConfig` 造 client；`config` 为 None 时用 `LLMConfig.from_env()`。

    `overrides` 是**LLMConfig 的字段名**（`provider` / `model` / `base_url` / `api_key` ...），
    写错字段名 -> `ConfigError` 并列出合法取值（静默忽略拼错的 `bas_url` 会让人以为
    自己配了地址，然后花一小时排查 401/404）。
    """
    resolved = _copy_config(config if config is not None else LLMConfig.from_env())
    if overrides:
        known = sorted(field.name for field in fields(LLMConfig))
        unknown = sorted(set(overrides) - set(known))
        if unknown:
            raise ConfigError(
                f"unknown LLMConfig field(s): {', '.join(unknown)}; valid fields: {', '.join(known)}"
            )
        resolved = replace(resolved, **overrides)
    return get_default_registry().create(resolved.provider, config=resolved)


def _parse_spec(spec: str) -> tuple[str, str | None, str | None]:
    """解析 `provider[:model][@base_url]`。

    顺序很重要：先切 `@`（base_url 里可能带 `:` 与 `/`，例如
    `openai-compatible:qwen@http://host:8000/v1`），再切 `:`。
    """
    head, _, base_url = spec.partition("@")
    provider, _, model = head.partition(":")
    provider = provider.strip()
    if not provider:
        raise ConfigError(
            f"invalid LLM spec {spec!r}: provider name is empty "
            "(expected 'provider[:model][@base_url]')"
        )
    return provider, model.strip() or None, base_url.strip() or None


def get_llm(spec: str | LLMClient, **kwargs: Any) -> LLMClient:
    """`spec` 支持 `'openai'`、`'openai:gpt-4o-mini'`、`'openai-compatible:qwen@http://host/v1'`。

    `':'` 后为 model；`'@'` 后为 base_url。已是 `LLMClient` 实例则原样返回
    （让"用户传了个 client"与"用户传了个名字"在装配代码里长得一样 —— 这是
    `Agent(llm=...)` 能同时接受两种写法的原因）。
    """
    if isinstance(spec, LLMClient):
        return spec
    text = str(spec).strip()
    if not text:
        raise ConfigError("empty LLM spec; expected 'provider[:model][@base_url]'")
    provider, model, base_url = _parse_spec(text)

    options: dict[str, Any] = dict(kwargs)
    config = options.pop("config", None)
    options["provider"] = provider
    if model is not None:
        options["model"] = model
    if base_url is not None:
        options["base_url"] = base_url
    return build_llm(config, **options)
