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
    """

    provider: ProviderType = "ollama"
    model_name: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


# ── Lazy builder functions (one per supported provider) ────────────────


def _build_ollama(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    kwargs: dict[str, Any] = {"base_url": base_url}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=OllamaProvider(**kwargs))


def _build_openai(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIChatModel(config.model_name, provider=OpenAIProvider(**kwargs))


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
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=OpenRouterProvider(**kwargs))


def _build_grok(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.grok import GrokProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=GrokProvider(**kwargs))


def _build_deepseek(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=DeepSeekProvider(**kwargs))


def _build_cerebras(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.cerebras import CerebrasProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=CerebrasProvider(**kwargs))


def _build_fireworks(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.fireworks import FireworksProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=FireworksProvider(**kwargs))


def _build_together(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.together import TogetherProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=TogetherProvider(**kwargs))


def _build_azure(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=AzureProvider(**kwargs))


def _build_vercel(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.vercel import VercelProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=VercelProvider(**kwargs))


def _build_moonshotai(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=MoonshotAIProvider(**kwargs))


def _build_github(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.github import GitHubProvider

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=GitHubProvider(**kwargs))


def _build_heroku(config: ModelConfig) -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.heroku import HerokuProvider

    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAIChatModel(config.model_name, provider=HerokuProvider(**kwargs))


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
    if not config.api_key and not config.base_url and config.provider not in ("ollama", "google"):
        return f"{config.provider}:{config.model_name}"

    builder = _PROVIDER_BUILDERS.get(config.provider)
    if builder is None:
        raise ValueError(
            f"Unsupported provider '{config.provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_BUILDERS))}"
        )
    return builder(config)
