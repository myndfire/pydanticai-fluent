# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-agnostic LLM model configuration for ManagedAgent.

Typed ModelConfig replaces the old prefix-string convention.  ManagedAgent
delegates to `build_model()` which either returns a plain string (letting
pydantic_ai infer the right Model + Provider from a ``provider:model``
string) or constructs explicit Provider + Model instances when the user
supplies an api_key / base_url.

Usage:
    from agent_harness.model_config import ModelConfig

    # Simple – env-var based auth, pydantic_ai handles everything
    config = ModelConfig(provider="openai", model_name="gpt-4o")
    agent = ManagedAgent(model=config)

    # Explicit auth
    config = ModelConfig(provider="anthropic", model_name="claude-sonnet-4-20250514",
                         api_key="sk-ant-...")
    agent = ManagedAgent().with_model(config)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

ProviderType = Literal[
    "ollama",
    "openai",
    "anthropic",
    "google",
    "groq",
    "mistral",
    "bedrock",
    "cohere",
    "huggingface",
    "openrouter",
    "grok",
    "deepseek",
    "cerebras",
    "fireworks",
    "together",
    "azure",
    "vercel",
    "moonshotai",
    "github",
    "heroku",
]

# Which wire field the generic `max_tokens` ModelSetting maps to for
# OpenAI-compatible Chat Completions providers:
#   "auto"                  -> provider/model profile (Ollama => max_tokens)
#   "max_tokens"            -> force the legacy field (OpenRouter, some compatible APIs)
#   "max_completion_tokens" -> force the OpenAI field (incl. o-series reasoning models)
MaxTokensField = Literal["auto", "max_tokens", "max_completion_tokens"]


@dataclass
class ModelConfig:
    """Configure which LLM provider and model an agent should use.

    Attributes:
        provider:   Provider name (e.g. ``"ollama"``, ``"openai"``,
                    ``"anthropic"``).
        model_name: Model identifier without provider prefix
                    (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``,
                    ``"gemini-2.0-flash"``).
        api_key:    API key for the provider (falls back to provider env
                    var when not set).
        base_url:   Custom endpoint URL (e.g.
                    ``"https://api.openai.com/v1"``).
        max_tokens_field: Which wire field the generic ``max_tokens`` setting
                    maps to for OpenAI-compatible providers. ``"auto"`` (default)
                    uses the pydantic-ai profile (Ollama => ``max_tokens``);
                    force ``"max_tokens"`` or ``"max_completion_tokens"`` to
                    override. Ignored by native (non-OpenAI) providers.
    """

    provider: ProviderType = "ollama"
    model_name: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens_field: MaxTokensField = "auto"


# ── Lazy builder functions (one per supported provider) ────────────────


def _resolve_max_tokens_flag(provider: str, config: ModelConfig) -> Optional[bool]:
    """Resolve the desired ``openai_chat_supports_max_completion_tokens`` flag.

    Returns ``True`` to send ``max_completion_tokens``, ``False`` to send the
    legacy ``max_tokens`` field, or ``None`` to leave the provider/profile
    default untouched.
    """
    field = getattr(config, "max_tokens_field", "auto") or "auto"
    if field == "max_tokens":
        return False
    if field == "max_completion_tokens":
        return True
    # "auto": Ollama's OpenAI-compatible endpoint honors `max_tokens` but
    # ignores `max_completion_tokens`, so route to the legacy field. Other
    # providers keep whatever their pydantic-ai profile declares.
    if provider == "ollama":
        return False
    return None


def _openai_chat_model(config: ModelConfig, provider: Any, provider_name: str) -> Any:
    """Build an ``OpenAIChatModel`` applying the max_tokens field routing profile."""
    from pydantic_ai.models.openai import OpenAIChatModel

    kwargs: dict[str, Any] = {"provider": provider}
    flag = _resolve_max_tokens_flag(provider_name, config)
    if flag is not None:
        from pydantic_ai.profiles.openai import OpenAIModelProfile

        kwargs["profile"] = OpenAIModelProfile(
            openai_chat_supports_max_completion_tokens=flag
        )
    return OpenAIChatModel(config.model_name, **kwargs)


def _build_ollama(config: ModelConfig) -> Any:
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    kwargs: dict[str, Any] = {"base_url": base_url}
    if config.api_key:
        kwargs["api_key"] = config.api_key

    flag = _resolve_max_tokens_flag("ollama", config)
    if flag is None:
        provider = OllamaProvider(**kwargs)
    else:
        # Override the provider profile so the generic `max_tokens` setting is
        # routed to the field Ollama actually honors. Using a provider subclass
        # (rather than passing `profile=` to OllamaModel) keeps OllamaModel's
        # own Ollama-Cloud json-schema guard intact.
        # See pydantic-ai #5186 / PR #5926.
        from pydantic_ai.profiles import merge_profile
        from pydantic_ai.profiles.openai import OpenAIModelProfile

        class _HarnessOllamaProvider(OllamaProvider):
            @staticmethod
            def model_profile(model_name: str):
                base = OllamaProvider.model_profile(model_name) or {}
                return merge_profile(
                    base,
                    OpenAIModelProfile(
                        openai_chat_supports_max_completion_tokens=flag,
                        # Ollama's /v1/chat/completions endpoint enforces
                        # `response_format: json_schema` at generation time via
                        # llama.cpp's grammar-constrained decoder, for every
                        # model. pydantic-ai's default output mode is `'tool'`,
                        # which small/fast local models handle unreliably (they
                        # emit no tool call, and structured output then retries
                        # until exhausted). Prefer native structured output so
                        # plain `BaseModel` output types stay model-agnostic.
                        # Explicit `ToolOutput`/`PromptedOutput` from callers
                        # still win.
                        default_structured_output_mode="native",
                    ),
                )

        provider = _HarnessOllamaProvider(**kwargs)

    return OllamaModel(config.model_name, provider=provider)


def _build_openai(config: ModelConfig) -> Any:
    from pydantic_ai.providers.openai import OpenAIProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return _openai_chat_model(config, OpenAIProvider(**kwargs), "openai")


def _build_anthropic(config: ModelConfig) -> Any:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return AnthropicModel(config.model_name, provider=AnthropicProvider(**kwargs))


def _build_google(config: ModelConfig) -> Any:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return GoogleModel(config.model_name, provider=GoogleProvider(**kwargs))


def _build_groq(config: ModelConfig) -> Any:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return GroqModel(config.model_name, provider=GroqProvider(**kwargs))


def _build_mistral(config: ModelConfig) -> Any:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return MistralModel(config.model_name, provider=MistralProvider(**kwargs))


def _build_bedrock(config: ModelConfig) -> Any:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    return BedrockConverseModel(config.model_name, provider=BedrockProvider())


def _build_cohere(config: ModelConfig) -> Any:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return CohereModel(config.model_name, provider=CohereProvider(**kwargs))


def _build_huggingface(config: ModelConfig) -> Any:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return HuggingFaceModel(config.model_name, provider=HuggingFaceProvider(**kwargs))


def _build_openrouter(config: ModelConfig) -> Any:
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, OpenRouterProvider(**kwargs), "openrouter")


def _build_grok(config: ModelConfig) -> Any:
    from pydantic_ai.providers.grok import GrokProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, GrokProvider(**kwargs), "grok")


def _build_deepseek(config: ModelConfig) -> Any:
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, DeepSeekProvider(**kwargs), "deepseek")


def _build_cerebras(config: ModelConfig) -> Any:
    from pydantic_ai.providers.cerebras import CerebrasProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, CerebrasProvider(**kwargs), "cerebras")


def _build_fireworks(config: ModelConfig) -> Any:
    from pydantic_ai.providers.fireworks import FireworksProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, FireworksProvider(**kwargs), "fireworks")


def _build_together(config: ModelConfig) -> Any:
    from pydantic_ai.providers.together import TogetherProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, TogetherProvider(**kwargs), "together")


def _build_azure(config: ModelConfig) -> Any:
    from pydantic_ai.providers.azure import AzureProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, AzureProvider(**kwargs), "azure")


def _build_vercel(config: ModelConfig) -> Any:
    from pydantic_ai.providers.vercel import VercelProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, VercelProvider(**kwargs), "vercel")


def _build_moonshotai(config: ModelConfig) -> Any:
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, MoonshotAIProvider(**kwargs), "moonshotai")


def _build_github(config: ModelConfig) -> Any:
    from pydantic_ai.providers.github import GitHubProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, GitHubProvider(**kwargs), "github")


def _build_heroku(config: ModelConfig) -> Any:
    from pydantic_ai.providers.heroku import HerokuProvider

    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return _openai_chat_model(config, HerokuProvider(**kwargs), "heroku")


_PROVIDER_BUILDERS: dict[str, Callable[[ModelConfig], Any]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "google": _build_google,
    "groq": _build_groq,
    "mistral": _build_mistral,
    "bedrock": _build_bedrock,
    "cohere": _build_cohere,
    "huggingface": _build_huggingface,
    "openrouter": _build_openrouter,
    "grok": _build_grok,
    "deepseek": _build_deepseek,
    "cerebras": _build_cerebras,
    "fireworks": _build_fireworks,
    "together": _build_together,
    "azure": _build_azure,
    "vercel": _build_vercel,
    "moonshotai": _build_moonshotai,
    "github": _build_github,
    "heroku": _build_heroku,
}


def build_model(config: ModelConfig) -> Union[str, Any]:
    """Build a pydantic_ai model from a typed ModelConfig.

    When no explicit auth (api_key / base_url) is supplied and the provider
    is supported by pydantic_ai's built-in ``infer_model()``, returns a
    plain ``"provider:model_name"`` string so pydantic_ai handles everything
    automatically (model class, provider, profile).

    Delegates to the provider-specific lazy builder when explicit auth is
    needed or when the provider (e.g. ollama) is not in the inference list.
    """
    if (
        (getattr(config, "max_tokens_field", "auto") or "auto") == "auto"
        and not config.api_key
        and not config.base_url
        and config.provider not in ("ollama", "google")
    ):
        return f"{config.provider}:{config.model_name}"

    builder = _PROVIDER_BUILDERS.get(config.provider)
    if builder is None:
        raise ValueError(
            f"Unsupported provider '{config.provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_BUILDERS))}"
        )
    return builder(config)


def build_model_ref(ref: Union[str, ModelConfig]) -> Any:
    """Resolve a model reference (``ModelConfig`` or ``"provider:model"`` string).

    Used by internal components (evaluation judges, retry fallback models) so
    they inherit the same provider/profile handling as ``ManagedAgent`` — in
    particular Ollama's ``max_tokens`` field routing. Bare or unknown strings
    are returned unchanged for pydantic-ai to infer.
    """
    if isinstance(ref, ModelConfig):
        return build_model(ref)
    if isinstance(ref, str) and ":" in ref:
        provider, _, model_name = ref.partition(":")
        if provider in _PROVIDER_BUILDERS:
            return build_model(
                ModelConfig(provider=provider, model_name=model_name)  # type: ignore[arg-type]
            )
    return ref
