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
from datetime import datetime
from typing import Any, Optional, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse

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
    CostLimitsConfig,
    CircuitBreakerConfig,
)
from .model_config import ModelConfig, build_model
from .errorhandling import ErrorHandlingConfig, ErrorHandler
from .evaluators import Evaluator


AgentDepsT = TypeVar("AgentDepsT")


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
        """
        model_config = model or ModelConfig(provider="ollama", model_name="gpt-oss:20b")
        self._agent: Agent[Any, Any] = Agent(
            model=build_model(model_config), deps_type=deps_type
        )
        self.model = f"{model_config.provider}:{model_config.model_name}"
        self._deps_type = deps_type

        self.prompts = prompts or StaticPrompts()
        self.observability = observability or Observability()
        self.tools = tools or ToolRegistry()
        self.evaluators = evaluators or []
        self.guards = guards or GuardConfig()
        self.error_handling = ErrorHandlingConfig()

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
        self._agent = Agent(model=build_model(model))
        self.model = f"{model.provider}:{model.model_name}"
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
        self._agent = Agent(
            model=self._agent._model,
            toolsets=current_toolsets + [mcp_server],
        )
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
        self.guards.enable_content_filter = config.enabled
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_pii_detection(self, config: PIIDetectionConfig) -> "ManagedAgent":
        """Set PII detection configuration."""
        self.guards.enable_pii_detection = config.enabled
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_cost_limits(self, config: CostLimitsConfig) -> "ManagedAgent":
        """Set cost limits configuration."""
        self.guards.enable_cost_limits = config.max_tokens_per_request is not None
        self.guards.max_tokens_per_request = config.max_tokens_per_request
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_circuit_breaker(self, config: CircuitBreakerConfig) -> "ManagedAgent":
        """Set circuit breaker configuration."""
        self.guards.enable_circuit_breaker = config.enabled
        self.guards.failure_threshold = config.failure_threshold
        self.guards.circuit_timeout = config.circuit_timeout
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_guardrails(
        self,
        content_filter: Optional[ContentFilterConfig] = None,
        pii_detection: Optional[PIIDetectionConfig] = None,
        cost_limits: Optional[CostLimitsConfig] = None,
    ) -> "ManagedAgent":
        """Set multiple guardrail configurations at once."""
        if content_filter:
            self.guards.enable_content_filter = content_filter.enabled
        if pii_detection:
            self.guards.enable_pii_detection = pii_detection.enabled
        if cost_limits:
            self.guards.enable_cost_limits = cost_limits.max_tokens_per_request is not None
            self.guards.max_tokens_per_request = cost_limits.max_tokens_per_request
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_output(self, output_type: Any, output_retries: int = 3) -> "ManagedAgent":
        """
        Set the output type for structured responses.

        Args:
            output_type: The Pydantic model for structured output
            output_retries: Number of retries for output validation (default: 3)
        """
        self._agent = Agent(
            model=self._agent._model,
            output_type=output_type,
            output_retries=output_retries,
        )
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
        prompt: str,
        message_history: MessageHistory,
        session_id: str,
        save_to: Optional[list[MemoryProvider]] = None,
        deps: Any = None,
        **kwargs,
    ) -> Any:
        """
        Run agent with explicit message history and save options.

        Args:
            prompt: User prompt
            message_history: MessageHistory object with loaded history (required)
            session_id: Session ID (required - key for saving turns)
            save_to: Optional list of memory providers to save the turn to
            deps: Dependencies for dependency injection
            **kwargs: Additional context for prompt rendering

        Returns:
            Agent result
        """
        start_time = time.time()

        prompt_id = kwargs.pop("prompt_id", "default")
        prompt_vars = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        context = {
            "session_id": session_id,
            "prompt_id": prompt_id,
            "model": self.model,
        }

        try:
            async with self.observability.observe("agent_run", **context):
                if self._short_term_memory:
                    await message_history.load(session_id, self._short_term_memory)
                if self._long_term_memory:
                    await message_history.load(session_id, self._long_term_memory)

                history = message_history.messages

                system_prompt = await self.prompts.get_system_prompt(
                    prompt_id=prompt_id, **prompt_vars
                )
                if system_prompt:
                    self._agent._system_prompts = (system_prompt,)

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

            if save_to:
                providers = save_to if isinstance(save_to, list) else [save_to]
                for provider in providers:
                    await provider.save_turn(session_id, turn)

            if usage:
                self.observability.log_info(
                    "token_usage",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    **context,
                )

            for evaluator in self.evaluators:
                try:
                    await evaluator.evaluate(prompt, result, context)
                except Exception as e:
                    self.observability.log_warning(
                        f"Evaluator failed: {str(e)}",
                        evaluator=type(evaluator).__name__,
                    )

            if self._agent._output_type is None:
                result.output = extract_clean_output(result)
            return result

        except Exception as e:
            error_type = type(e).__name__.lower()
            error_msg = str(e).lower()

            if any(
                x in error_type or x in error_msg
                for x in [
                    "mongo",
                    "redis",
                    "elastic",
                    "motor",
                    "database",
                    "connection",
                ]
            ):
                source = "memory"
            elif any(
                x in error_type or x in error_msg
                for x in ["timeout", "model", "llm", "ollama", "openai", "anthropic", "google", "groq", "mistral", "cohere", "openrouter", "grok", "deepseek", "cerebras", "bedrock", "huggingface", "api"]
            ):
                source = "llm"
            elif any(
                x in error_type or x in error_msg for x in ["tool", "mcp", "function"]
            ):
                source = "tool"
            else:
                source = "unknown"

            result = self._error_handler.handle_error(
                exception=e,
                source=source,
                session_id=session_id,
                prompt=prompt,
            )

            if result:
                return result

            raise

    async def run_sync(
        self,
        prompt: str,
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
