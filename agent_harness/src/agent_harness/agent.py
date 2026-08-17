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

"""Core ManagedAgent with fluent API for crosscutting concerns."""

import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Optional, TypeVar, Union

from .log_enrichment import LogContext, LogEnrichmentProvider

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, UserContent

from .memory import (
    MemoryProvider,
    TurnData,
    UsageData,
    MessageHistory,
    filter_thinking_parts,
)
from .prompts import PromptProvider, StaticPrompts
from .observability import Observability
from .tools import ToolRegistry
from .guards import (
    GuardConfig,
    GuardRunner,
    ErrorContext,
    AgentRunResult,
    AgentRetryConfig,
    ToolRetryConfig,
    ResultValidatorRetryConfig,
    ContentFilterConfig,
    PIIDetectionConfig,
    TokenLimitsConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
    TurnLimitsConfig,
)
from .model_config import ModelConfig, build_model
from .errorhandling import ErrorHandlingConfig, ErrorHandler
from .evaluators import Evaluator


AgentDepsT = TypeVar("AgentDepsT")


def prompt_to_text(prompt: Union[str, Sequence[UserContent]]) -> str:
    """Reduce a possibly-multimodal prompt to a plain text summary.

    ``run()`` accepts either a plain string or a sequence of pydantic_ai
    ``UserContent`` (text plus images/audio/documents). The multimodal form is
    passed through to the model untouched, but crosscutting consumers —
    evaluators, error contexts, logs — are text-only. This produces a readable
    stand-in for those, substituting a short placeholder for binary parts
    rather than dumping base64 into logs.

    Args:
        prompt: String prompt, or sequence of UserContent parts.

    Returns:
        str: Text representation. Non-text parts appear as "[image]",
             "[audio]", "[document]", "[video]" or "[binary <media_type>]".
    """
    if isinstance(prompt, str):
        return prompt

    kind_by_suffix = {
        "ImageUrl": "[image]",
        "AudioUrl": "[audio]",
        "DocumentUrl": "[document]",
        "VideoUrl": "[video]",
    }

    parts: list[str] = []
    for item in prompt:
        if isinstance(item, str):
            parts.append(item)
            continue

        type_name = type(item).__name__
        if type_name in kind_by_suffix:
            parts.append(kind_by_suffix[type_name])
        elif type_name == "BinaryContent":
            parts.append(f"[binary {getattr(item, 'media_type', 'unknown')}]")
        elif hasattr(item, "content"):
            parts.append(str(item.content))
        else:
            parts.append(f"[{type_name}]")

    return " ".join(parts)


def extract_clean_output(result) -> str:
    """Extract clean text from result, reusing filter_thinking_parts."""
    if not hasattr(result, "new_messages"):
        return str(result.output) if hasattr(result, "output") else str(result)

    new_messages = (
        result.new_messages() if callable(result.new_messages) else result.new_messages
    )
    filtered = filter_thinking_parts(new_messages)

    for msg in reversed(filtered):
        if msg.get("kind") == "response" and msg.get("parts"):
            for part in msg["parts"]:
                if part.get("type") == "TextPart":
                    return part.get("content", "")

    return str(result.output) if hasattr(result, "output") else str(result)


class ManagedAgent:
    """
    Elegant agent with fluent configuration API.

    Usage:
        from agent_harness import ManagedAgent
        from agent_harness.model_config import ModelConfig

        agent = ManagedAgent(
            model=ModelConfig(provider="openai", model_name="gpt-4o", api_key="sk-...")
        )

        # Or configure fluently
        agent = ManagedAgent().with_model(
            ModelConfig(provider="anthropic", model_name="claude-sonnet-4-20250514")
        )

        # Run with explicit message history and save targets
        history = MessageHistory()
        await history.load("session_123", from_memory=in_memory_provider)

        result = await agent.run(
            "question",
            message_history=history,
            session_id="session_123",
            save_to=[in_memory_provider]
        )
    """

    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        prompts: Optional[PromptProvider] = None,
        observability: Optional[Observability] = None,
        tools: Optional[ToolRegistry] = None,
        evaluators: Optional[list[Evaluator]] = None,
        guards: Optional[GuardConfig] = None,
        deps_type: Optional[type] = None,
        model_settings: Optional[Any] = None,
    ):
        """
        Initialize managed agent with optional components.

        Args:
            model: ModelConfig (default: ollama with gpt-oss:20b)
            prompts: Prompt provider (default: StaticPrompts)
            observability: Observability (logging, tracing, metrics)
            tools: Tool registry (default: empty ToolRegistry)
            evaluators: List of evaluators (default: empty list)
            guards: Guard configuration (default: GuardConfig with defaults)
            deps_type: Type for dependency injection
            model_settings: Optional model settings (pydantic_ai ModelSettings dict)
        """
        self._model_settings = model_settings
        self._output_type: Optional[Any] = None
        self._output_retries: int = 3
        model_config = model or ModelConfig(provider="ollama", model_name="gpt-oss:20b")
        self._agent: Agent[Any, Any] = Agent(
            model=build_model(model_config), deps_type=deps_type,
            model_settings=model_settings,
        )
        self.model = f"{model_config.provider}:{model_config.model_name}"
        self._deps_type = deps_type

        self.prompts = prompts or StaticPrompts()
        self.observability = observability or Observability()
        self.tools = tools or ToolRegistry()
        self.evaluators = evaluators or []
        self.guards = guards or GuardConfig()
        self.error_handling = ErrorHandlingConfig()
        self._enrichment: list[LogEnrichmentProvider] = []

        self._guard_runner = GuardRunner(self.guards)
        self._error_handler = ErrorHandler(self.error_handling)
        self._last_turn: Optional[TurnData] = None
        self._short_term_memory: Optional[MemoryProvider] = None
        self._long_term_memory: Optional[MemoryProvider] = None
        self._rabbitmq_config: dict = {}
        self._input_queue: Optional[str] = None
        self._input_exchange: Optional[str] = None
        self._output_queue: Optional[str] = None
        self._output_exchange: Optional[str] = None
        self._dead_letter_queue: Optional[str] = None
        self._dead_letter_exchange: Optional[str] = None
        self._turn_counts: dict[str, int] = {}

        if self.tools.get_tools():
            self.tools.register_to_agent(self._agent)

    def with_model(
        self,
        model: ModelConfig,
    ) -> "ManagedAgent":
        """Set the model using a ModelConfig object.

        Args:
            model: ModelConfig specifying provider, model_name, api_key, base_url.
        """
        kwargs: dict[str, Any] = {}
        if self._model_settings is not None:
            kwargs["model_settings"] = self._model_settings
        if self._output_type is not None:
            kwargs["output_type"] = self._output_type
            kwargs["output_retries"] = self._output_retries
        self._agent = Agent(model=build_model(model), **kwargs)
        self.model = f"{model.provider}:{model.model_name}"
        return self

    def with_model_settings(self, model_settings: Any) -> "ManagedAgent":
        """Set model settings (e.g. thinking, temperature, max_tokens).

        Args:
            model_settings: pydantic_ai ModelSettings dict or callable.
        """
        self._model_settings = model_settings
        kwargs: dict[str, Any] = {
            "model": self._agent._model,
            "toolsets": list(self._agent.toolsets),
        }
        if model_settings is not None:
            kwargs["model_settings"] = model_settings
        if self._output_type is not None:
            kwargs["output_type"] = self._output_type
            kwargs["output_retries"] = self._output_retries
        self._agent = Agent(**kwargs)
        return self

    def with_log_enrichment(self, *providers: LogEnrichmentProvider) -> "ManagedAgent":
        """Add log enrichment providers for this agent.

        Each provider's enrich() output is merged into the log context
        on every run() call, automatically appearing in all log entries,
        trace spans, and metric labels.

        Args:
            *providers: One or more LogEnrichmentProvider instances
                (LogContext, EnvEnricher, custom implementations, etc.)

        Returns:
            Self for chaining
        """
        self._enrichment.extend(providers)
        return self

    def with_short_term_memory(self, provider: MemoryProvider) -> "ManagedAgent":
        """Set short-term memory provider."""
        self._short_term_memory = provider
        return self

    @property
    def last_turn(self) -> Optional["TurnData"]:
        """Get the last turn data from the most recent run."""
        return self._last_turn

    def with_long_term_memory(
        self, provider: Optional[MemoryProvider] = None
    ) -> "ManagedAgent":
        """Set long-term memory provider."""
        self._long_term_memory = provider
        return self

    def with_deps_type(self, deps_type: type) -> "ManagedAgent":
        """Set the dependency injection type."""
        self._deps_type = deps_type
        self._agent._deps_type = deps_type
        return self

    def with_prompts(self, provider: PromptProvider) -> "ManagedAgent":
        """Set prompt provider."""
        self.prompts = provider
        return self

    def with_observability(self, observability: Observability) -> "ManagedAgent":
        """Set observability."""
        self.observability = observability
        return self

    def with_tools(self, registry: ToolRegistry) -> "ManagedAgent":
        """Set tool registry."""
        self.tools = registry
        self.tools.register_to_agent(self._agent)
        return self

    def with_mcp_server(self, url: str, **kwargs) -> "ManagedAgent":
        """Add an MCP server as a toolset."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        tool_prefix = kwargs.get("tool_prefix")
        mcp_server = (
            MCPServerStreamableHTTP(url, tool_prefix=tool_prefix)
            if tool_prefix
            else MCPServerStreamableHTTP(url)
        )

        current_toolsets = list(self._agent.toolsets)
        kwargs: dict[str, Any] = {
            "model": self._agent._model,
            "toolsets": current_toolsets + [mcp_server],
        }
        if self._model_settings is not None:
            kwargs["model_settings"] = self._model_settings
        if self._output_type is not None:
            kwargs["output_type"] = self._output_type
            kwargs["output_retries"] = self._output_retries
        self._agent = Agent(**kwargs)
        return self

    def with_mcp_servers(
        self, *urls: str, tool_prefix: Optional[str] = None
    ) -> "ManagedAgent":
        """Add multiple MCP servers as toolsets."""
        for url in urls:
            self = self.with_mcp_server(url, tool_prefix=tool_prefix)
        return self

    def with_evaluators(self, *evaluators: Evaluator) -> "ManagedAgent":
        """Add evaluators."""
        self.evaluators.extend(evaluators)
        return self

    def with_error_handling(self, config: ErrorHandlingConfig) -> "ManagedAgent":
        """Set error handling configuration."""
        self.error_handling = config
        self._error_handler = ErrorHandler(config)
        return self

    def with_agent_retries(self, config: AgentRetryConfig) -> "ManagedAgent":
        """Set agent-level retry configuration."""
        self.guards.agent = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_tool_retries(self, config: ToolRetryConfig) -> "ManagedAgent":
        """Set tool-level retry configuration."""
        self.guards.tool = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_result_validator_retries(self, config: ResultValidatorRetryConfig) -> "ManagedAgent":
        """Set result validator retry configuration."""
        self.guards.result_validator = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_content_filter(self, config: ContentFilterConfig) -> "ManagedAgent":
        """Set content filter configuration."""
        self.guards.content_filter = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_pii_detection(self, config: PIIDetectionConfig) -> "ManagedAgent":
        """Set PII detection configuration."""
        self.guards.pii_detection = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_token_limits(self, config: TokenLimitsConfig) -> "ManagedAgent":
        """Set token limits configuration."""
        self.guards.token_limits = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_cost_limits(self, config: CostLimitsConfig) -> "ManagedAgent":
        """Set cost limits configuration."""
        self.guards.cost_limits = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_circuit_breaker(self, config: CircuitBreakerConfig) -> "ManagedAgent":
        """Set circuit breaker configuration."""
        self.guards.circuit_breaker = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_turn_limits(self, config: TurnLimitsConfig) -> "ManagedAgent":
        """Set turn limits configuration."""
        self.guards.turn_limits = config
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_guardrails(
        self,
        content_filter: Optional[ContentFilterConfig] = None,
        pii_detection: Optional[PIIDetectionConfig] = None,
        token_limits: Optional[TokenLimitsConfig] = None,
        cost_limits: Optional[CostLimitsConfig] = None,
    ) -> "ManagedAgent":
        """Set multiple guardrail configurations at once."""
        if content_filter:
            self.guards.content_filter = content_filter
        if pii_detection:
            self.guards.pii_detection = pii_detection
        if token_limits:
            self.guards.token_limits = token_limits
        if cost_limits:
            self.guards.cost_limits = cost_limits
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_output(self, output_type: Any, output_retries: int = 3) -> "ManagedAgent":
        """
        Set the output type for structured responses.

        Args:
            output_type: The Pydantic model for structured output
            output_retries: Number of retries for output validation (default: 3)
        """
        self._output_type = output_type
        self._output_retries = output_retries
        kwargs: dict[str, Any] = {
            "model": self._agent._model,
            "output_type": output_type,
            "output_retries": output_retries,
        }
        if self._model_settings is not None:
            kwargs["model_settings"] = self._model_settings
        self._agent = Agent(**kwargs)
        return self

    def with_rabbitmq(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        virtual_host: Optional[str] = None,
    ) -> "ManagedAgent":
        """Configure RabbitMQ messaging service."""
        self._rabbitmq_config = {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "virtual_host": virtual_host,
        }
        return self

    def with_input_queue(self, queue_name: str) -> "ManagedAgent":
        """Set the input queue name."""
        self._input_queue = queue_name
        return self

    def with_input_exchange(self, exchange_name: str) -> "ManagedAgent":
        """Set the input exchange name."""
        self._input_exchange = exchange_name
        return self

    def with_output_queue(self, queue_name: str) -> "ManagedAgent":
        """Set the output queue name."""
        self._output_queue = queue_name
        return self

    def with_output_exchange(self, exchange_name: str) -> "ManagedAgent":
        """Set the output exchange name."""
        self._output_exchange = exchange_name
        return self

    def with_dead_letter_queue(self, queue_name: str) -> "ManagedAgent":
        """Set the dead letter queue name."""
        self._dead_letter_queue = queue_name
        return self

    def with_dead_letter_exchange(self, exchange_name: str) -> "ManagedAgent":
        """Set the dead letter exchange name."""
        self._dead_letter_exchange = exchange_name
        return self

    @property
    def has_queue_config(self) -> bool:
        """Check if queue configuration is present."""
        return hasattr(self, "_rabbitmq_config") and self._rabbitmq_config

    async def run(
        self,
        prompt: Union[str, Sequence[UserContent]],
        message_history: MessageHistory,
        session_id: str,
        save_to: Optional[list[MemoryProvider]] = None,
        deps: Any = None,
        enrichment: Optional[LogContext] = None,
        **kwargs,
    ) -> Any:
        """
        Run agent with explicit message history and save options.

        Args:
            prompt: User prompt. Either a plain string, or a sequence of
                pydantic_ai UserContent parts for multimodal input, e.g.
                ``["What is this?", ImageUrl(url="data:image/jpeg;base64,...")]``.
                Multimodal prompts require a model that accepts that media type.
            message_history: MessageHistory object with loaded history (required)
            session_id: Session ID (required - key for saving turns)
            save_to: Optional list of memory providers to save the turn to
            deps: Dependencies for dependency injection
            enrichment: Optional LogContext with per-run enrichment keys.
                Merged with agent-level enrichment providers set via
                with_log_enrichment(). All keys appear in log entries,
                trace spans, and metric labels.
            **kwargs: Additional context for prompt rendering

        Returns:
            Agent result
        """
        start_time = time.time()

        prompt_id = kwargs.pop("prompt_id", "default")
        prompt_vars = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        # Text-only view of the prompt for evaluators, error contexts and logs.
        # The original prompt is what reaches the model.
        prompt_text = prompt_to_text(prompt)

        context = {
            "session_id": session_id,
            "prompt_id": prompt_id,
            "model": self.model,
            "model_settings": self._model_settings,
        }

        # Merge agent-level enrichment providers
        for provider in self._enrichment:
            context.update(provider.enrich())

        # Merge per-run enrichment (wins over agent-level on conflicts)
        if enrichment:
            context.update(enrichment.enrich())

        try:
            async with self.observability.observe("agent_run", **context):
                try:
                    if self._short_term_memory:
                        await message_history.load(session_id, self._short_term_memory)
                    if self._long_term_memory:
                        await message_history.load(session_id, self._long_term_memory)
                except Exception as e:
                    e._error_source = "memory"
                    raise

                # ── Turn limits check ───────────────────────────
                if self.guards.turn_limits:
                    tl = self.guards.turn_limits
                    count = self._turn_counts.get(session_id, 0) + 1
                    self._turn_counts[session_id] = count
                    if tl.max_turns is not None and count > tl.max_turns:
                        error_ctx = ErrorContext(
                            error_type="TurnLimitExceeded",
                            error_message=(
                                f"Turn {count} exceeds max {tl.max_turns}"
                            ),
                            source="guardrail",
                            session_id=session_id,
                        )
                        if tl._on_turn_limit:
                            return tl._on_turn_limit(error_ctx)
                        raise RuntimeError(error_ctx.error_message)

                history = message_history.messages

                try:
                    system_prompt = await self.prompts.get_system_prompt(
                        prompt_id=prompt_id, **prompt_vars
                    )
                    if system_prompt:
                        self._agent._system_prompts = (system_prompt,)
                except Exception as e:
                    e._error_source = "prompt"
                    raise

                result = await self._guard_runner.run_with_guards(
                    agent=self._agent,
                    prompt=prompt,
                    message_history=history,
                    deps=deps,
                )

                duration = time.time() - start_time
                status = "success" if result.success else "error"
                if hasattr(result, "used_fallback") and result.used_fallback:
                    status = "fallback"

                new_messages = []
                if hasattr(result, "new_messages"):
                    nm = result.new_messages
                    new_messages = nm() if callable(nm) else nm

                serialized_messages = filter_thinking_parts(new_messages)

                try:
                    usage = None
                    if hasattr(result, "usage") and result.usage:
                        u = result.usage
                        if (
                            hasattr(u, "requests")
                            and isinstance(getattr(u, "requests", None), list)
                            and u.requests
                        ):
                            u = u.requests[0]
                        usage = UsageData(
                            input_tokens=getattr(u, "input_tokens", 0) or 0,
                            output_tokens=getattr(u, "output_tokens", 0) or 0,
                            total_tokens=getattr(u, "total_tokens", 0) or 0,
                            prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                            completion_tokens=getattr(u, "output_tokens", 0) or 0,
                        )
                    else:
                        for msg in new_messages:
                            if isinstance(msg, ModelResponse) and getattr(msg, "usage", None):
                                u = msg.usage
                                usage = UsageData(
                                    input_tokens=getattr(u, "input_tokens", 0) or 0,
                                    output_tokens=getattr(u, "output_tokens", 0) or 0,
                                    total_tokens=getattr(u, "total_tokens", 0) or 0,
                                    prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                                    completion_tokens=getattr(u, "output_tokens", 0) or 0,
                                )
                                break

                    turn = TurnData(
                        turn_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        completed_at=datetime.now(),
                        messages=serialized_messages,
                        usage=usage,
                        duration_seconds=duration,
                        model=self.model,
                        status=status,
                    )

                    self._last_turn = turn

                    if usage:
                        self.observability.log_info(
                            "token_usage",
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            total_tokens=usage.total_tokens,
                            **context,
                        )
                except Exception as e:
                    e._error_source = "output"
                    raise

                if save_to:
                    providers = save_to if isinstance(save_to, list) else [save_to]
                    try:
                        for provider in providers:
                            await provider.save_turn(session_id, turn)
                    except Exception as e:
                        e._error_source = "memory"
                        raise

                for evaluator in self.evaluators:
                    try:
                        await evaluator.evaluate(prompt_text, result, context)
                    except Exception as e:
                        e._error_source = "evaluator"
                        raise

                try:
                    if self._agent._output_type is None:
                        result.output = extract_clean_output(result)
                except Exception as e:
                    e._error_source = "output"
                    raise

                return result

        except Exception as e:
            source = getattr(e, "_error_source", "unknown")
            error_result = self._error_handler.handle_error(
                exception=e,
                source=source,
                session_id=session_id,
                prompt=prompt_text,
            )
            if error_result:
                self.observability.error(
                    "error_handled",
                    exception=e,
                    error_source=source,
                    session_id=session_id,
                )
                return error_result
            raise

    async def run_sync(
        self,
        prompt: Union[str, Sequence[UserContent]],
        message_history: MessageHistory,
        session_id: str,
        save_to: Optional[list[MemoryProvider]] = None,
        **kwargs,
    ) -> Any:
        """Synchronous wrapper for run()."""
        return await self.run(prompt, message_history, session_id, save_to, **kwargs)

    def get_agent(self) -> Agent:
        """Get the underlying PydanticAI agent."""
        return self._agent
