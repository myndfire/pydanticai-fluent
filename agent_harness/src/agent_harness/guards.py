"""Guards with retry logic and guardrails."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .errorhandling import ErrorContext, AgentRunResult


@dataclass
class AgentRetryConfig:
    """
    Configuration for agent-level retries.

    Corresponds to PydanticAI's Agent(retries=N) parameter.
    """

    max_retries: int = 3
    timeout: int = 120
    backoff_multiplier: float = 2.0
    fallback_model: Optional[str] = None


@dataclass
class ToolRetryConfig:
    """
    Configuration for tool-level retries.

    Corresponds to PydanticAI's @agent.tool(retries=N) parameter.
    Applied to all tools registered with the agent.
    """

    max_retries: int = 3
    backoff_multiplier: float = 2.0


@dataclass
class ResultValidatorRetryConfig:
    """
    Configuration for result validator retries.

    Corresponds to PydanticAI's @agent.output_validator with ModelRetry exception.
    """

    max_retries: int = 3
    backoff_multiplier: float = 2.0


@dataclass
class GuardConfig:
    """
    Configuration for retry logic and guardrails.

    Supports three types of retries:
    - agent: Retry when the agent fails (Agent(retries=N))
    - tool: Retry when a tool call fails (@agent.tool(retries=N))
    - result_validator: Retry when result validation fails (ModelRetry)
    """

    # Agent-level retry configuration
    agent: AgentRetryConfig = field(default_factory=AgentRetryConfig)

    # Tool-level retry configuration
    tool: ToolRetryConfig = field(default_factory=ToolRetryConfig)

    # Result validator retry configuration
    result_validator: ResultValidatorRetryConfig = field(
        default_factory=ResultValidatorRetryConfig
    )

    # Error callback - called on each retry attempt
    # Signature: Callable[[ErrorContext], None]
    on_retry: Optional[Callable[[ErrorContext], None]] = None

    # Final error callback - called when all retries exhausted
    # Signature: Callable[[ErrorContext], Any] - can return fallback output
    on_error: Optional[Callable[[ErrorContext], Any]] = None

    # Guardrails (Placeholders for future implementation)
    enable_content_filter: bool = False
    enable_pii_detection: bool = False
    enable_cost_limits: bool = False
    max_tokens_per_request: Optional[int] = None

    # Circuit Breaker (Placeholder)
    enable_circuit_breaker: bool = False
    failure_threshold: int = 5
    circuit_timeout: int = 60

    # Global error handler - called when any error occurs in the agent run
    # Signature: Callable[[ErrorContext, Exception], bool]
    #   - Returns True → continue with wrapped error result
    #   - Returns False or doesn't return True → rethrow original
    on_error_handler: Optional[Callable[["ErrorContext", Exception], bool]] = None

    def with_agent_retries(
        self,
        max_retries: int,
        timeout: int = 120,
        backoff_multiplier: float = 2.0,
        fallback_model: Optional[str] = None,
        on_retry: Optional[Callable[[ErrorContext], None]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ) -> "GuardConfig":
        """Configure agent-level retries with optional callbacks."""
        self.agent = AgentRetryConfig(
            max_retries=max_retries,
            timeout=timeout,
            backoff_multiplier=backoff_multiplier,
            fallback_model=fallback_model,
        )
        if on_retry:
            self.on_retry = on_retry
        if on_error:
            self.on_error = on_error
        return self

    def with_tool_retries(
        self, max_retries: int, backoff_multiplier: float = 2.0
    ) -> "GuardConfig":
        """Configure tool-level retries."""
        self.tool = ToolRetryConfig(
            max_retries=max_retries,
            backoff_multiplier=backoff_multiplier,
        )
        return self

    def with_result_validator_retries(
        self, max_retries: int, backoff_multiplier: float = 2.0
    ) -> "GuardConfig":
        """Configure result validator retries."""
        self.result_validator = ResultValidatorRetryConfig(
            max_retries=max_retries,
            backoff_multiplier=backoff_multiplier,
        )
        return self

    def on_retry_callback(
        self, callback: Callable[[ErrorContext], None]
    ) -> "GuardConfig":
        """Set callback to be called on each retry attempt."""
        self.on_retry = callback
        return self

    def on_error_callback(
        self, callback: Callable[[ErrorContext], Any]
    ) -> "GuardConfig":
        """Set callback to be called when all retries exhausted."""
        self.on_error = callback
        return self

    def with_error_handler(
        self, handler: Callable[["ErrorContext", Exception], bool]
    ) -> "GuardConfig":
        """Set global error handler for agent run errors."""
        self.on_error_handler = handler
        return self


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
        # Apply agent-level retries
        agent._retries = self.config.agent.max_retries

        # Note: Tool retries and result validator retries are handled
        # by PydanticAI's built-in mechanisms when tools are registered
        # with @agent.tool(retries=N) and @agent.output_validator

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
                # Apply timeout
                result = await asyncio.wait_for(
                    agent.run(prompt, message_history=message_history, **kwargs),
                    timeout=self.config.agent.timeout,
                )
                # Extract usage from the bound method if present
                usage_obj = None
                if hasattr(result, "usage"):
                    try:
                        usage_obj = result.usage()
                    except Exception:
                        usage_obj = None

                # Reset failure count on success
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

                # Call on_retry callback if configured
                if self.config.on_retry:
                    self.config.on_retry(error_ctx)

                if attempt < self.config.agent.max_retries - 1:
                    # Wait before retry with exponential backoff
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

                # Call on_retry callback if configured
                if self.config.on_retry:
                    self.config.on_retry(error_ctx)

                if attempt < self.config.agent.max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = self.config.agent.backoff_multiplier**attempt
                    print(f"[Retry] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    last_exception = e

        # All retries exhausted - try fallback if configured
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
                # Fallback also failed - call on_error if configured
                error_ctx = ErrorContext(
                    error_type="FallbackError",
                    error_message=f"All retries exhausted. Last error: {last_exception}, Fallback error: {fallback_error}",
                    source="llm",
                    attempt=self.config.agent.max_retries,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=False,
                )

                if self.config.on_error:
                    fallback_output = self.config.on_error(error_ctx)
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

        # No fallback - call on_error if configured
        error_ctx = ErrorContext(
            error_type="MaxRetriesExceeded",
            error_message=str(last_exception),
            source="llm",
            attempt=self.config.agent.max_retries,
            max_attempts=self.config.agent.max_retries,
            will_retry=False,
        )

        if self.config.on_error:
            error_output = self.config.on_error(error_ctx)
            return AgentRunResult(
                output=error_output,
                success=False,
                error_context=error_ctx,
                used_fallback=False,
                new_messages=[],
                usage=None,
            )

        # No fallback, no on_error - raise exception
        raise Exception(
            f"All {self.config.agent.max_retries} retries exhausted. "
            f"Last error: {str(last_exception)}"
        )

    # Placeholder methods for future guardrails

    async def _filter_content(self, result: Any) -> Any:
        """
        Filter harmful or inappropriate content.

        TODO: Implement content filtering using:
        - OpenAI Moderation API
        - Custom content policy rules
        - Third-party content filtering services

        Args:
            result: Agent result

        Returns:
            Filtered result
        """
        # Placeholder
        return result

    async def _redact_pii(self, result: Any) -> Any:
        """
        Detect and redact personally identifiable information.

        TODO: Implement PII detection using:
        - Regex patterns for emails, phone numbers, SSNs, etc.
        - NER models for names, addresses, etc.
        - Third-party PII detection services

        Args:
            result: Agent result

        Returns:
            Result with PII redacted
        """
        # Placeholder
        return result

    def _check_token_usage(self, result: Any) -> None:
        """
        Check if token usage exceeds limits.

        TODO: Implement token usage tracking and limits

        Args:
            result: Agent result with usage information

        Raises:
            ValueError: If token limit exceeded
        """
        # Placeholder
        pass

    def _check_circuit_breaker(self) -> None:
        """
        Check circuit breaker state.

        TODO: Implement circuit breaker pattern to prevent cascading failures

        Raises:
            RuntimeError: If circuit is open
        """
        if not self.config.enable_circuit_breaker:
            return

        # Placeholder for circuit breaker logic
        if self._circuit_open:
            raise RuntimeError("Circuit breaker is open - too many failures")
