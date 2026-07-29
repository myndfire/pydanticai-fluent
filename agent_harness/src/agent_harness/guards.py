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

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, UserContent

from .errorhandling import ErrorContext, AgentRunResult


class AgentRetryConfig:
    """Configuration for agent-level retries.

    Corresponds to PydanticAI's Agent(retries=N) parameter.

    Usage:
        config = AgentRetryConfig(
            max_retries=3,
            timeout=120,
            backoff_multiplier=2.0,
            fallback_model="ollama:backup",
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
        self._on_retry = on_retry
        self._on_error = on_error

    def with_max_retries(self, max_retries: int) -> "AgentRetryConfig":
        self.max_retries = max_retries
        return self

    def with_timeout(self, timeout: int) -> "AgentRetryConfig":
        self.timeout = timeout
        return self

    def with_backoff(self, backoff_multiplier: float) -> "AgentRetryConfig":
        self.backoff_multiplier = backoff_multiplier
        return self

    def with_fallback(self, fallback_model: str) -> "AgentRetryConfig":
        self.fallback_model = fallback_model
        return self

    def on_retry(self, callback: Callable[[ErrorContext], None]) -> "AgentRetryConfig":
        self._on_retry = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "AgentRetryConfig":
        self._on_error = callback
        return self


class ToolRetryConfig:
    """Configuration for tool-level retries.

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
        self.max_retries = max_retries
        return self

    def with_backoff(self, backoff_multiplier: float) -> "ToolRetryConfig":
        self.backoff_multiplier = backoff_multiplier
        return self


class ResultValidatorRetryConfig:
    """Configuration for result validator retries.

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
        self.max_retries = max_retries
        return self

    def with_backoff(self, backoff_multiplier: float) -> "ResultValidatorRetryConfig":
        self.backoff_multiplier = backoff_multiplier
        return self


class ContentFilterConfig:
    """Configuration for content filtering.

    Filters harmful or inappropriate content from responses via a user-provided callback.
    """

    def __init__(
        self,
        on_filter: Optional[Callable[[str], str]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self._on_filter = on_filter
        self._on_error = on_error

    def on_filter(self, callback: Callable[[str], str]) -> "ContentFilterConfig":
        self._on_filter = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "ContentFilterConfig":
        self._on_error = callback
        return self


class PIIDetectionConfig:
    """Configuration for PII detection and redaction.

    Detects and redacts personally identifiable information via a user-provided callback.
    """

    def __init__(
        self,
        on_redact: Optional[Callable[[str], str]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self._on_redact = on_redact
        self._on_error = on_error

    def on_redact(self, callback: Callable[[str], str]) -> "PIIDetectionConfig":
        self._on_redact = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "PIIDetectionConfig":
        self._on_error = callback
        return self


class TokenLimitsConfig:
    """Configuration for token usage limits.

    Caps token usage per request to control processing cost and latency.
    """

    def __init__(
        self,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        on_token_limit: Optional[Callable[[ErrorContext], Any]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self._on_token_limit = on_token_limit
        self._on_error = on_error

    def with_max_input_tokens(self, n: int) -> "TokenLimitsConfig":
        self.max_input_tokens = n
        return self

    def with_max_output_tokens(self, n: int) -> "TokenLimitsConfig":
        self.max_output_tokens = n
        return self

    def with_max_total_tokens(self, n: int) -> "TokenLimitsConfig":
        self.max_total_tokens = n
        return self

    def on_token_limit(self, callback: Callable[[ErrorContext], Any]) -> "TokenLimitsConfig":
        self._on_token_limit = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "TokenLimitsConfig":
        self._on_error = callback
        return self


class CostLimitsConfig:
    """Configuration for cost limiting.

    Caps dollar cost per request using per-token pricing.
    """

    def __init__(
        self,
        max_input_cost: Optional[float] = None,
        max_output_cost: Optional[float] = None,
        max_total_cost: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        on_cost_limit: Optional[Callable[[ErrorContext], Any]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_input_cost = max_input_cost
        self.max_output_cost = max_output_cost
        self.max_total_cost = max_total_cost
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self._on_cost_limit = on_cost_limit
        self._on_error = on_error

    def with_max_input_cost(self, n: float) -> "CostLimitsConfig":
        self.max_input_cost = n
        return self

    def with_max_output_cost(self, n: float) -> "CostLimitsConfig":
        self.max_output_cost = n
        return self

    def with_max_total_cost(self, n: float) -> "CostLimitsConfig":
        self.max_total_cost = n
        return self

    def with_cost_per_input_token(self, n: float) -> "CostLimitsConfig":
        self.cost_per_input_token = n
        return self

    def with_cost_per_output_token(self, n: float) -> "CostLimitsConfig":
        self.cost_per_output_token = n
        return self

    def on_cost_limit(self, callback: Callable[[ErrorContext], Any]) -> "CostLimitsConfig":
        self._on_cost_limit = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "CostLimitsConfig":
        self._on_error = callback
        return self


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern.

    Prevents cascading failures by stopping requests after too many errors.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        circuit_timeout: int = 60,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.circuit_timeout = circuit_timeout
        self._on_error = on_error

    def with_threshold(self, failure_threshold: int) -> "CircuitBreakerConfig":
        self.failure_threshold = failure_threshold
        return self

    def with_timeout(self, circuit_timeout: int) -> "CircuitBreakerConfig":
        self.circuit_timeout = circuit_timeout
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "CircuitBreakerConfig":
        self._on_error = callback
        return self


class TurnLimitsConfig:
    """Configuration for session turn limits.

    Caps the number of agent invocations per session to prevent
    runaway loops, control cost, and limit abuse in multi-turn conversations.
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        on_turn_limit: Optional[Callable[[ErrorContext], Any]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_turns = max_turns
        self._on_turn_limit = on_turn_limit
        self._on_error = on_error

    def with_max_turns(self, n: int) -> "TurnLimitsConfig":
        self.max_turns = n
        return self

    def on_turn_limit(self, callback: Callable[[ErrorContext], Any]) -> "TurnLimitsConfig":
        self._on_turn_limit = callback
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "TurnLimitsConfig":
        self._on_error = callback
        return self


@dataclass
class GuardConfig:
    """Configuration for retry logic and guardrails.

    Supports three types of retries:
    - agent: Retry when the agent fails (Agent(retries=N))
    - tool: Retry when a tool call fails (@agent.tool(retries=N))
    - result_validator: Retry when result validation fails (ModelRetry)

    And guardrails (None to disable, set config to enable):
    - content_filter: Filter harmful content
    - pii_detection: Detect and redact PII
    - token_limits: Cap token usage
    - cost_limits: Cap dollar cost
    - circuit_breaker: Prevent cascading failures
    - turn_limits: Cap turns per session
    """

    agent: AgentRetryConfig = field(default_factory=AgentRetryConfig)
    tool: ToolRetryConfig = field(default_factory=ToolRetryConfig)
    result_validator: ResultValidatorRetryConfig = field(
        default_factory=ResultValidatorRetryConfig
    )

    content_filter: Optional[ContentFilterConfig] = None
    pii_detection: Optional[PIIDetectionConfig] = None
    token_limits: Optional[TokenLimitsConfig] = None
    cost_limits: Optional[CostLimitsConfig] = None
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    turn_limits: Optional[TurnLimitsConfig] = None


class GuardRunner:
    """Execute agent runs with retry logic and guardrails."""

    def __init__(self, config: GuardConfig):
        self.config = config
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_opened_at: Optional[float] = None
        self._half_open_pending = False

    def apply_to_agent(self, agent: Agent) -> Agent:
        """Apply guard configuration to a PydanticAI agent."""
        agent._retries = self.config.agent.max_retries
        return agent

    async def run_with_guards(
        self,
        agent: "Agent",
        prompt: Union[str, Sequence[UserContent]],
        message_history: "list[ModelMessage]",
        **kwargs,
    ) -> AgentRunResult:
        """Run agent with retry logic, timeout, and guardrails.

        Guardrails applied in order: circuit breaker (gateway), retries,
        token limits, cost limits, content filter, PII detection.

        ``prompt`` is forwarded to the model unchanged, so it may be a plain
        string or a sequence of pydantic_ai UserContent parts for multimodal
        input.
        """
        # ── Circuit breaker gateway check ──────────────────────────
        if self.config.circuit_breaker and self._circuit_open:
            cb = self.config.circuit_breaker
            if self._circuit_opened_at is not None:
                elapsed = time.time() - self._circuit_opened_at
                if elapsed >= cb.circuit_timeout:
                    self._half_open_pending = True
                else:
                    error_ctx = ErrorContext(
                        error_type="CircuitBreakerOpen",
                        error_message=(
                            f"Circuit breaker is open after {self._failure_count} "
                            f"failures. Retry in {cb.circuit_timeout - int(elapsed)}s"
                        ),
                        source="guardrail",
                        attempt=self._failure_count,
                        max_attempts=cb.failure_threshold,
                        will_retry=False,
                    )
                    if cb._on_error:
                        return AgentRunResult(
                            output=cb._on_error(error_ctx),
                            success=False,
                            error_context=error_ctx,
                        )
                    raise RuntimeError(error_ctx.error_message)

        # ── Retry loop with timeout ────────────────────────────────
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

                # ── Circuit breaker: reset on success ──────────────
                if self.config.circuit_breaker:
                    self._failure_count = 0
                    if self._half_open_pending:
                        self._circuit_open = False
                        self._half_open_pending = False

                # ── Extract output ─────────────────────────────────
                output = result.output if hasattr(result, "output") else result

                # ── Token limits check ─────────────────────────────
                if self.config.token_limits and usage_obj:
                    tl = self.config.token_limits

                    input_tok = getattr(usage_obj, "input_tokens", 0) or 0
                    output_tok = getattr(usage_obj, "output_tokens", 0) or 0
                    total_tok = getattr(usage_obj, "total_tokens", 0) or 0

                    if tl.max_input_tokens is not None and input_tok > tl.max_input_tokens:
                        error_ctx = ErrorContext(
                            error_type="TokenLimitExceeded",
                            error_message=(
                                f"Input tokens {input_tok} > {tl.max_input_tokens}"
                            ),
                            source="guardrail",
                        )
                        if tl._on_token_limit:
                            return AgentRunResult(
                                output=tl._on_token_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                    if tl.max_output_tokens is not None and output_tok > tl.max_output_tokens:
                        error_ctx = ErrorContext(
                            error_type="TokenLimitExceeded",
                            error_message=(
                                f"Output tokens {output_tok} > {tl.max_output_tokens}"
                            ),
                            source="guardrail",
                        )
                        if tl._on_token_limit:
                            return AgentRunResult(
                                output=tl._on_token_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                    if tl.max_total_tokens is not None and total_tok > tl.max_total_tokens:
                        error_ctx = ErrorContext(
                            error_type="TokenLimitExceeded",
                            error_message=(
                                f"Total tokens {total_tok} > {tl.max_total_tokens}"
                            ),
                            source="guardrail",
                        )
                        if tl._on_token_limit:
                            return AgentRunResult(
                                output=tl._on_token_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                # ── Cost limits check ──────────────────────────────
                if self.config.cost_limits and usage_obj:
                    cl = self.config.cost_limits

                    input_tok = getattr(usage_obj, "input_tokens", 0) or 0
                    output_tok = getattr(usage_obj, "output_tokens", 0) or 0

                    input_cost = input_tok * (cl.cost_per_input_token or 0)
                    output_cost = output_tok * (cl.cost_per_output_token or 0)
                    total_cost = input_cost + output_cost

                    if cl.max_input_cost is not None and input_cost > cl.max_input_cost:
                        error_ctx = ErrorContext(
                            error_type="CostLimitExceeded",
                            error_message=(
                                f"Input cost ${input_cost:.6f} > ${cl.max_input_cost:.6f}"
                            ),
                            source="guardrail",
                        )
                        if cl._on_cost_limit:
                            return AgentRunResult(
                                output=cl._on_cost_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                    if cl.max_output_cost is not None and output_cost > cl.max_output_cost:
                        error_ctx = ErrorContext(
                            error_type="CostLimitExceeded",
                            error_message=(
                                f"Output cost ${output_cost:.6f} "
                                f"> ${cl.max_output_cost:.6f}"
                            ),
                            source="guardrail",
                        )
                        if cl._on_cost_limit:
                            return AgentRunResult(
                                output=cl._on_cost_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                    if cl.max_total_cost is not None and total_cost > cl.max_total_cost:
                        error_ctx = ErrorContext(
                            error_type="CostLimitExceeded",
                            error_message=(
                                f"Total cost ${total_cost:.6f} > ${cl.max_total_cost:.6f}"
                            ),
                            source="guardrail",
                        )
                        if cl._on_cost_limit:
                            return AgentRunResult(
                                output=cl._on_cost_limit(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise RuntimeError(error_ctx.error_message)

                # ── Content filter ─────────────────────────────────
                if self.config.content_filter and self.config.content_filter._on_filter:
                    try:
                        output = self.config.content_filter._on_filter(output)
                    except Exception as e:
                        cf = self.config.content_filter
                        error_ctx = ErrorContext(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            source="guardrail",
                        )
                        if cf._on_error:
                            return AgentRunResult(
                                output=cf._on_error(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise

                # ── PII detection ──────────────────────────────────
                if self.config.pii_detection and self.config.pii_detection._on_redact:
                    try:
                        output = self.config.pii_detection._on_redact(output)
                    except Exception as e:
                        pd = self.config.pii_detection
                        error_ctx = ErrorContext(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            source="guardrail",
                        )
                        if pd._on_error:
                            return AgentRunResult(
                                output=pd._on_error(error_ctx),
                                success=False,
                                error_context=error_ctx,
                            )
                        raise

                return AgentRunResult(
                    output=output,
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
                    error_message=(
                        f"Agent execution timed out after {self.config.agent.timeout}s"
                    ),
                    source="llm",
                    attempt=attempt + 1,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=attempt < self.config.agent.max_retries - 1,
                )

                print(
                    f"[Retry] Attempt {attempt + 1}/{self.config.agent.max_retries} "
                    f"- Timeout after {self.config.agent.timeout}s"
                )

                if self.config.agent._on_retry:
                    self.config.agent._on_retry(error_ctx)

                self._track_circuit_failure(type(e).__name__, str(e))

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
                    f"[Retry] Attempt {attempt + 1}/{self.config.agent.max_retries} "
                    f"- Error: {type(e).__name__}: {str(e)}"
                )

                if self.config.agent._on_retry:
                    self.config.agent._on_retry(error_ctx)

                self._track_circuit_failure(type(e).__name__, str(e))

                if attempt < self.config.agent.max_retries - 1:
                    wait_time = self.config.agent.backoff_multiplier**attempt
                    print(f"[Retry] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    last_exception = e

        # ── Fallback ───────────────────────────────────────────────
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
                    error_message=(
                        f"All retries exhausted. "
                        f"Last error: {last_exception}, "
                        f"Fallback error: {fallback_error}"
                    ),
                    source="llm",
                    attempt=self.config.agent.max_retries,
                    max_attempts=self.config.agent.max_retries,
                    will_retry=False,
                )

                if self.config.agent._on_error:
                    fallback_output = self.config.agent._on_error(error_ctx)
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

        # ── Exhaustion ─────────────────────────────────────────────
        error_ctx = ErrorContext(
            error_type="MaxRetriesExceeded",
            error_message=str(last_exception),
            source="llm",
            attempt=self.config.agent.max_retries,
            max_attempts=self.config.agent.max_retries,
            will_retry=False,
        )

        if self.config.agent._on_error:
            error_output = self.config.agent._on_error(error_ctx)
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

    def _track_circuit_failure(self, error_type: str, message: str) -> None:
        """Track failure for circuit breaker, opening circuit if threshold reached."""
        cb = self.config.circuit_breaker
        if cb is None:
            return

        self._failure_count += 1
        if self._failure_count >= cb.failure_threshold and not self._circuit_open:
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            self._half_open_pending = False
            print(
                f"[CircuitBreaker] OPEN after {self._failure_count} consecutive "
                f"failures (threshold: {cb.failure_threshold}, "
                f"timeout: {cb.circuit_timeout}s). "
                f"Last error: {error_type}: {message}"
            )
