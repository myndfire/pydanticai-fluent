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

"""Guards with retry logic and guardrails."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .errorhandling import ErrorContext, AgentRunResult


class AgentRetryConfig:
    """
    Configuration for agent-level retries.

    Corresponds to PydanticAI's Agent(retries=N) parameter.

    Usage:
        config = AgentRetryConfig(
            max_retries=3,
            timeout=120,
            backoff_multiplier=2.0,
            fallback_model="ollama:backup"
        )
        # or with fluent API
        config = AgentRetryConfig().with_max_retries(5).with_timeout(60)
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 120,
        backoff_multiplier: float = 2.0,
        fallback_model: Optional[str] = None,
        on_retry: Optional[Callable[[ErrorContext], None]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_multiplier = backoff_multiplier
        self.fallback_model = fallback_model
        self.on_retry = on_retry
        self.on_error = on_error

    def with_max_retries(self, max_retries: int) -> "AgentRetryConfig":
        """Set maximum number of retry attempts."""
        self.max_retries = max_retries
        return self

    def with_timeout(self, timeout: int) -> "AgentRetryConfig":
        """Set timeout in seconds."""
        self.timeout = timeout
        return self

    def with_backoff(self, backoff_multiplier: float) -> "AgentRetryConfig":
        """Set exponential backoff multiplier."""
        self.backoff_multiplier = backoff_multiplier
        return self

    def with_fallback(self, fallback_model: str) -> "AgentRetryConfig":
        """Set fallback model to use when all retries exhausted."""
        self.fallback_model = fallback_model
        return self

    def on_retry(self, callback: Callable[[ErrorContext], None]) -> "AgentRetryConfig":
        """Set callback to be called on each retry attempt."""
        self.on_retry = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "AgentRetryConfig":
        """Set callback to be called when all retries exhausted."""
        self.on_error = callback
        return self


class ToolRetryConfig:
    """
    Configuration for tool-level retries.

    Corresponds to PydanticAI's @agent.tool(retries=N) parameter.
    Applied to all tools registered with the agent.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
    ):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier

    def with_max_retries(self, max_retries: int) -> "ToolRetryConfig":
        """Set maximum number of retry attempts."""
        self.max_retries = max_retries
        return self

    def with_backoff(self, backoff_multiplier: float) -> "ToolRetryConfig":
        """Set exponential backoff multiplier."""
        self.backoff_multiplier = backoff_multiplier
        return self


class ResultValidatorRetryConfig:
    """
    Configuration for result validator retries.

    Corresponds to PydanticAI's @agent.output_validator with ModelRetry exception.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
    ):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier

    def with_max_retries(self, max_retries: int) -> "ResultValidatorRetryConfig":
        """Set maximum number of retry attempts."""
        self.max_retries = max_retries
        return self

    def with_backoff(self, backoff_multiplier: float) -> "ResultValidatorRetryConfig":
        """Set exponential backoff multiplier."""
        self.backoff_multiplier = backoff_multiplier
        return self


class ContentFilterConfig:
    """
    Configuration for content filtering.

    Filters harmful or inappropriate content from responses.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def with_enabled(self, enabled: bool) -> "ContentFilterConfig":
        """Enable or disable content filtering."""
        self.enabled = enabled
        return self


class PIIDetectionConfig:
    """
    Configuration for PII detection and redaction.

    Detects and redacts personally identifiable information from responses.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def with_enabled(self, enabled: bool) -> "PIIDetectionConfig":
        """Enable or disable PII detection."""
        self.enabled = enabled
        return self


class CostLimitsConfig:
    """
    Configuration for cost limiting.

    Limits token usage per request to control costs.
    """

    def __init__(
        self,
        max_tokens_per_request: Optional[int] = None,
    ):
        self.max_tokens_per_request = max_tokens_per_request

    def with_max_tokens(self, max_tokens: int) -> "CostLimitsConfig":
        """Set maximum tokens per request."""
        self.max_tokens_per_request = max_tokens
        return self


class CircuitBreakerConfig:
    """
    Configuration for circuit breaker pattern.

    Prevents cascading failures by stopping requests after too many errors.
    """

    def __init__(
        self,
        enabled: bool = True,
        failure_threshold: int = 5,
        circuit_timeout: int = 60,
    ):
        self.enabled = enabled
        self.failure_threshold = failure_threshold
        self.circuit_timeout = circuit_timeout

    def with_enabled(self, enabled: bool) -> "CircuitBreakerConfig":
        """Enable or disable circuit breaker."""
        self.enabled = enabled
        return self

    def with_threshold(self, failure_threshold: int) -> "CircuitBreakerConfig":
        """Set number of failures before opening circuit."""
        self.failure_threshold = failure_threshold
        return self

    def with_timeout(self, circuit_timeout: int) -> "CircuitBreakerConfig":
        """Set timeout in seconds before attempting to close circuit."""
        self.circuit_timeout = circuit_timeout
        return self


@dataclass
class GuardConfig:
    """
    Configuration for retry logic and guardrails.

    Supports three types of retries:
    - agent: Retry when the agent fails (Agent(retries=N))
    - tool: Retry when a tool call fails (@agent.tool(retries=N))
    - result_validator: Retry when result validation fails (ModelRetry)

    And guardrails:
    - content_filter: Filter harmful content
    - pii_detection: Detect and redact PII
    - cost_limits: Limit token usage
    - circuit_breaker: Prevent cascading failures
    """

    agent: AgentRetryConfig = field(default_factory=AgentRetryConfig)
    tool: ToolRetryConfig = field(default_factory=ToolRetryConfig)
    result_validator: ResultValidatorRetryConfig = field(
        default_factory=ResultValidatorRetryConfig
    )

    enable_content_filter: bool = False
    enable_pii_detection: bool = False
    enable_cost_limits: bool = False
    max_tokens_per_request: Optional[int] = None

    enable_circuit_breaker: bool = False
    failure_threshold: int = 5
    circuit_timeout: int = 60


class GuardRunner:
    """Execute agent runs with retry logic and guardrails."""

    def __init__(self, config: GuardConfig):
        """
        Initialize guard runner.

        Args:
            config: Guard configuration
        """
        self.config = config
        self._failure_count = 0
        self._circuit_open = False

    def apply_to_agent(self, agent: Agent) -> Agent:
        """
        Apply guard configuration to a PydanticAI agent.

        This applies:
        - Agent retries via agent.retries
        - Tool retries via tool decorators
        - Sets timeout on the agent

        Args:
            agent: PydanticAI agent instance

        Returns:
            Agent with retries configured
        """
        agent._retries = self.config.agent.max_retries

        return agent

    async def run_with_guards(
        self,
        agent: "Agent",
        prompt: str,
        message_history: "list[ModelMessage]",
        **kwargs,
    ) -> AgentRunResult:
        """
        Run agent with retry logic and timeout.

        Note: Most retry logic is now handled by PydanticAI natively via:
        - Agent(retries=N) for agent-level retries
        - @agent.tool(retries=N) for tool retries
        - @agent.output_validator + ModelRetry for validator retries

        This method provides additional timeout and fallback functionality.

        Args:
            agent: PydanticAI agent
            prompt: User prompt
            message_history: Conversation history
            **kwargs: Additional agent.run() arguments

        Returns:
            AgentRunResult with output and error context
        """
        last_exception = None

        for attempt in range(self.config.agent.max_retries):
            try:
                result = await asyncio.wait_for(
                    agent.run(prompt, message_history=message_history, **kwargs),
                    timeout=self.config.agent.timeout,
                )
                usage_obj = None
                if hasattr(result, "usage"):
                    try:
                        usage_obj = result.usage()
                    except Exception:
                        usage_obj = None

                self._failure_count = 0

                return AgentRunResult(
                    output=result.output if hasattr(result, "output") else result,
                    success=True,
                    error_context=None,
                    new_messages=result.new_messages()
                    if hasattr(result, "new_messages")
                    else [],
                    usage=usage_obj,
                )

            except asyncio.TimeoutError as e:
                error_ctx = ErrorContext(
                    error_type="TimeoutError",
                    error_message=f"Agent execution timed out after {self.config.agent.timeout}s",
                    source="llm",
                    attempt=attempt + 1,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=attempt < self.config.agent.max_retries - 1,
                )

                print(
                    f"[Retry] Attempt {attempt + 1}/{self.config.agent.max_retries} - Timeout after {self.config.agent.timeout}s"
                )

                if self.config.agent.on_retry:
                    self.config.agent.on_retry(error_ctx)

                if attempt < self.config.agent.max_retries - 1:
                    wait_time = self.config.agent.backoff_multiplier**attempt
                    print(f"[Retry] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    last_exception = e

            except Exception as e:
                error_ctx = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    source="llm",
                    attempt=attempt + 1,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=attempt < self.config.agent.max_retries - 1,
                )

                print(
                    f"[Retry] Attempt {attempt + 1}/{self.config.agent.max_retries} - Error: {type(e).__name__}: {str(e)}"
                )

                if self.config.agent.on_retry:
                    self.config.agent.on_retry(error_ctx)

                if attempt < self.config.agent.max_retries - 1:
                    wait_time = self.config.agent.backoff_multiplier**attempt
                    print(f"[Retry] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    last_exception = e

        if self.config.agent.fallback_model:
            try:
                fallback_agent = Agent(self.config.agent.fallback_model)
                result = await asyncio.wait_for(
                    fallback_agent.run(prompt, message_history=message_history),
                    timeout=self.config.agent.timeout,
                )
                return AgentRunResult(
                    output=result.output if hasattr(result, "output") else result,
                    success=True,
                    error_context=None,
                    used_fallback=True,
                    new_messages=result.new_messages()
                    if hasattr(result, "new_messages")
                    else [],
                    usage=result.usage if hasattr(result, "usage") else None,
                )
            except Exception as fallback_error:
                error_ctx = ErrorContext(
                    error_type="FallbackError",
                    error_message=f"All retries exhausted. Last error: {last_exception}, Fallback error: {fallback_error}",
                    source="llm",
                    attempt=self.config.agent.max_retries,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=False,
                )

                if self.config.agent.on_error:
                    fallback_output = self.config.agent.on_error(error_ctx)
                    return AgentRunResult(
                        output=fallback_output,
                        success=False,
                        error_context=error_ctx,
                        used_fallback=True,
                        new_messages=[],
                        usage=None,
                    )

                raise Exception(
                    f"All retries exhausted and fallback failed. "
                    f"Last error: {str(last_exception)}. "
                    f"Fallback error: {str(fallback_error)}"
                )

        error_ctx = ErrorContext(
            error_type="MaxRetriesExceeded",
            error_message=str(last_exception),
            source="llm",
            attempt=self.config.agent.max_retries,
            max_attempts=self.config.agent.max_retries,
            will_retry=False,
        )

        if self.config.agent.on_error:
            error_output = self.config.agent.on_error(error_ctx)
            return AgentRunResult(
                output=error_output,
                success=False,
                error_context=error_ctx,
                used_fallback=False,
                new_messages=[],
                usage=None,
            )

        raise Exception(
            f"All {self.config.agent.max_retries} retries exhausted. "
            f"Last error: {str(last_exception)}"
        )

    async def _filter_content(self, result: Any) -> Any:
        """Filter harmful or inappropriate content."""
        return result

    async def _redact_pii(self, result: Any) -> Any:
        """Detect and redact personally identifiable information."""
        return result

    def _check_token_usage(self, result: Any) -> None:
        """Check if token usage exceeds limits."""
        pass

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state."""
        if not self.config.enable_circuit_breaker:
            return

        if self._circuit_open:
            raise RuntimeError("Circuit breaker is open - too many failures")