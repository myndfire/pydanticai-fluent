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
import structlog
import time
import traceback
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.usage import UsageLimits, UsageLimitExceeded

from .errorhandling import ErrorContext, AgentRunResult, TokenUsageInfo
from .model_config import build_model_ref
from ._agent_factory import build_harness_agent

# Forward reference for Observability to avoid circular imports at type-check time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .observability import Observability

from .observability import _truncate_traceback


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
        max_reasoning_tokens: Optional[int] = None,
        billing_mode: str = "output_plus_reasoning",
        auto_estimate_reasoning: bool = True,
        on_token_limit: Optional[Callable[[ErrorContext], Any]] = None,
        on_streaming_token_limit: Optional[Callable[[ErrorContext, str], Any]] = None,
        capture_reasoning_traces: bool = False,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.max_reasoning_tokens = max_reasoning_tokens
        self.billing_mode = billing_mode
        self.auto_estimate_reasoning = auto_estimate_reasoning
        self._on_token_limit = on_token_limit
        self._on_streaming_token_limit = on_streaming_token_limit
        self._capture_reasoning_traces = capture_reasoning_traces
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

    def with_max_reasoning_tokens(self, n: int) -> "TokenLimitsConfig":
        self.max_reasoning_tokens = n
        return self

    def with_billing_mode(self, mode: str) -> "TokenLimitsConfig":
        """Set billing mode: "output_plus_reasoning" (default) or "output_only"."""
        self.billing_mode = mode
        return self

    def with_auto_estimate_reasoning(self, enabled: bool = True) -> "TokenLimitsConfig":
        """Estimate reasoning tokens from text if API doesn't report them."""
        self.auto_estimate_reasoning = enabled
        return self

    def on_token_limit(self, callback: Callable[[ErrorContext], Any]) -> "TokenLimitsConfig":
        self._on_token_limit = callback
        return self

    def on_streaming_token_limit(self, callback: Callable[[ErrorContext, str], Any]) -> "TokenLimitsConfig":
        """Set callback for streaming token limit events.

        Receives (ErrorContext, partial_output) where partial_output is the
        text collected before the limit was hit.
        """
        self._on_streaming_token_limit = callback
        return self

    def with_reasoning_traces(self, enabled: bool = True) -> "TokenLimitsConfig":
        """Capture reasoning traces (thinking parts) in error context for debugging."""
        self._capture_reasoning_traces = enabled
        return self

    def on_error(self, callback: Callable[[ErrorContext], Any]) -> "TokenLimitsConfig":
        self._on_error = callback
        return self


class RateLimitError(RuntimeError):
    """Raised when a token/request rate window has no remaining capacity."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float,
        requested_tokens: int,
        available_tokens: int,
        window_seconds: float,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.requested_tokens = requested_tokens
        self.available_tokens = available_tokens
        self.window_seconds = window_seconds
        self._error_source = "rate_limit"


class TokenRateLimitConfig:
    """Sliding-window request/token limiter for downstream model providers."""

    def __init__(
        self,
        tokens_per_window: Optional[int] = None,
        requests_per_window: Optional[int] = None,
        window_seconds: float = 60.0,
        reserve_output_tokens: int = 0,
        on_exceeded: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.tokens_per_window = tokens_per_window
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.reserve_output_tokens = reserve_output_tokens
        self._on_exceeded = on_exceeded

    def with_tokens_per_window(self, value: int) -> "TokenRateLimitConfig":
        self.tokens_per_window = value
        return self

    def with_requests_per_window(self, value: int) -> "TokenRateLimitConfig":
        self.requests_per_window = value
        return self

    def with_window(self, seconds: float) -> "TokenRateLimitConfig":
        self.window_seconds = seconds
        return self

    def with_reserve_output_tokens(self, value: int) -> "TokenRateLimitConfig":
        self.reserve_output_tokens = value
        return self

    def on_exceeded(self, callback: Callable[[ErrorContext], Any]) -> "TokenRateLimitConfig":
        self._on_exceeded = callback
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
        cost_per_reasoning_token: Optional[float] = None,
        billing_mode: str = "output_plus_reasoning",
        on_cost_limit: Optional[Callable[[ErrorContext], Any]] = None,
        on_error: Optional[Callable[[ErrorContext], Any]] = None,
    ):
        self.max_input_cost = max_input_cost
        self.max_output_cost = max_output_cost
        self.max_total_cost = max_total_cost
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.cost_per_reasoning_token = cost_per_reasoning_token
        self.billing_mode = billing_mode
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

    def with_cost_per_reasoning_token(self, n: float) -> "CostLimitsConfig":
        self.cost_per_reasoning_token = n
        return self

    def with_billing_mode(self, mode: str) -> "CostLimitsConfig":
        """Set billing mode: "output_plus_reasoning" (default) or "output_only"."""
        self.billing_mode = mode
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
    - token_rate_limit: Cap requests/tokens over a sliding window
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
    token_rate_limit: Optional[TokenRateLimitConfig] = None
    cost_limits: Optional[CostLimitsConfig] = None
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    turn_limits: Optional[TurnLimitsConfig] = None
    observability: Optional["Observability"] = None


class _GuardrailHandled(Exception):
    """Internal signal that a guardrail callback produced a handled result."""

    def __init__(self, result: AgentRunResult):
        self.result = result


class GuardRunner:
    """Execute agent runs with retry logic and guardrails."""

    def __init__(self, config: GuardConfig):
        self.config = config
        self._observability = config.observability
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_opened_at: Optional[float] = None
        self._half_open_pending = False
        self._rate_limit_events: deque[tuple[float, int]] = deque()
        self._cumulative_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _extract_usage(self, result: Any) -> Any:
        """Return a result's usage object (calling it when it is a method)."""
        if result is None or not hasattr(result, "usage"):
            return None
        try:
            usage = result.usage
            return usage() if callable(usage) else usage
        except Exception:
            return None

    def _accumulate_usage(self, result: Any) -> None:
        """Accumulate a result's token usage into the run totals."""
        usage = self._extract_usage(result)
        if usage is None:
            return
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        cum = self._cumulative_usage
        cum["input_tokens"] += input_tok
        cum["output_tokens"] += output_tok
        cum["reasoning_tokens"] += getattr(usage, "reasoning_tokens", 0) or 0
        cum["total_tokens"] += getattr(usage, "total_tokens", 0) or (input_tok + output_tok)
        cum["prompt_tokens"] += input_tok
        cum["completion_tokens"] += output_tok

    def _build_token_usage_info(
        self, limit_type: str, limit_value: int, actual: int, usage_obj: Any = None
    ) -> TokenUsageInfo:
        """Build structured token usage info for limit debugging."""
        from .errorhandling import TokenUsageInfo

        input_tok = getattr(usage_obj, "input_tokens", 0) or 0 if usage_obj else None
        output_tok = getattr(usage_obj, "output_tokens", 0) or 0 if usage_obj else None
        reasoning_tok = getattr(usage_obj, "reasoning_tokens", 0) or 0 if usage_obj else None

        exceeded = max(0, actual - limit_value)
        pct = (actual / limit_value * 100) if limit_value > 0 else 0.0

        return TokenUsageInfo(
            limit_type=limit_type,
            limit_value=limit_value,
            actual_tokens=actual,
            output_tokens=output_tok,
            reasoning_tokens=reasoning_tok,
            input_tokens=input_tok,
            exceeded_by=exceeded,
            percentage_of_limit=pct,
            billing_mode=self.config.token_limits.billing_mode if self.config.token_limits else "output_plus_reasoning",
        )

    def _guard_error_context(
        self,
        error_type: str,
        error_message: str,
        *,
        source: str = "guardrail",
        session_id: Any = None,
        attempt: int = 0,
        max_attempts: Optional[int] = None,
        stack_trace: Optional[str] = None,
        **extra: Any,
    ) -> ErrorContext:
        """Build an ErrorContext enriched with session, attempt, and stack context.

        Centralizes the extra fields so every guardrail failure carries the same
        context (session id + attempt + stack trace), which callers and log
        pipelines can surface uniformly.
        """
        return ErrorContext(
            error_type=error_type,
            error_message=error_message,
            source=source,
            session_id=session_id,
            attempt=attempt + 1,
            max_attempts=(
                max_attempts
                if max_attempts is not None
                else self.config.agent.max_retries
            ),
            stack_trace=stack_trace,
            **extra,
        )

    def _log(self, level: str, event: str, **kwargs) -> None:
        """Log via observability if available, otherwise bootstrap print fallback."""
        if self._observability:
            kwargs.setdefault("component", "guards")
            getattr(self._observability, f"log_{level}")(event, **kwargs)
        # else: silently drop — Observability is initialized by ManagedAgent before run

    def _record_token_limit_metric(self, session_id: Any = None, limit_type: str = "unknown") -> None:
        """Emit a counter when a token limit is enforced (post-hoc or streaming)."""
        if self._observability:
            try:
                self._observability.record_metric(
                    "counter",
                    "agent_token_limit_exceeded",
                    1,
                    limit_type=limit_type,
                    session_id=session_id,
                    **{
                        "error.type": "TokenLimitExceeded",
                        "error.source": "guardrail",
                        "error.handled": True,
                    },
                )
            except Exception:
                pass


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
        execution = kwargs.pop("execution_context", None)
        session_id = kwargs.get("conversation_id")

        gated = self._circuit_breaker_gate(session_id)
        if gated is not None:
            return gated

        self._reset_cumulative_usage()
        usage_limits = self._build_usage_limits()
        last_exception: Optional[BaseException] = None

        # ``max_retries`` means retries after the first attempt. This matches
        # the public configuration and PydanticAI terminology.
        for attempt in range(self.config.agent.max_retries + 1):
            result = None
            try:
                self._check_rate_limit(prompt, session_id, attempt)
                if execution is not None:
                    execution.budget.check()
                    execution.budget.consume_model_request()
                timeout = self.config.agent.timeout
                if execution is not None:
                    remaining = execution.budget.remaining_seconds()
                    if remaining is not None:
                        timeout = min(timeout, remaining)
                result = await asyncio.wait_for(
                    agent.run(
                        prompt,
                        message_history=message_history,
                        usage_limits=usage_limits,
                        **kwargs,
                    ),
                    timeout=timeout,
                )
                usage_obj = self._extract_usage(result)
                self._reset_circuit_on_success()
                output = result.output if hasattr(result, "output") else result

                self._enforce_token_limits(usage_obj, session_id, attempt)
                self._enforce_cost_limits(usage_obj, session_id, attempt)
                output = self._apply_content_filter(output, session_id, attempt)
                output = self._apply_pii_detection(output, session_id, attempt)

                return self._build_success_result(result, output, usage_obj)

            except _GuardrailHandled as handled:
                return handled.result
            except RateLimitError as e:
                if await self._handle_retryable(
                    e, "rate_limit", result, session_id, attempt
                ):
                    continue
                last_exception = e
            except UsageLimitExceeded as e:
                return self._handle_usage_limit_exceeded(e, result, session_id, attempt)
            except asyncio.TimeoutError as e:
                if await self._handle_retryable(e, "timeout", result, session_id, attempt):
                    continue
                last_exception = e
            except Exception as e:
                if await self._handle_retryable(e, "error", result, session_id, attempt):
                    continue
                last_exception = e

        if self.config.agent.fallback_model:
            return await self._run_fallback(prompt, message_history, session_id, last_exception)

        return self._terminal_failure(session_id, last_exception)

    # ── Circuit breaker ────────────────────────────────────────────
    def _circuit_breaker_gate(self, session_id: Any) -> Optional[AgentRunResult]:
        """Return a handled result when the circuit is open, otherwise None."""
        cb = self.config.circuit_breaker
        if not (cb and self._circuit_open):
            return None
        if self._circuit_opened_at is None:
            return None
        elapsed = time.time() - self._circuit_opened_at
        if elapsed >= cb.circuit_timeout:
            self._half_open_pending = True
            return None
        error_ctx = ErrorContext(
            error_type="CircuitBreakerOpen",
            error_message=(
                f"Circuit breaker is open after {self._failure_count} "
                f"failures. Retry in {cb.circuit_timeout - int(elapsed)}s"
            ),
            source="guardrail",
            session_id=session_id,
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

    def _reset_circuit_on_success(self) -> None:
        if self.config.circuit_breaker:
            self._failure_count = 0
            if self._half_open_pending:
                self._circuit_open = False
                self._half_open_pending = False

    # ── Usage helpers ──────────────────────────────────────────────
    def _reset_cumulative_usage(self) -> None:
        self._cumulative_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _build_usage_limits(self) -> Optional[UsageLimits]:
        """Return PydanticAI UsageLimits, or None.

        The harness deliberately does NOT hand usage limits to PydanticAI:
        PydanticAI raises ``UsageLimitExceeded`` inside its own asyncio task
        (losing the caller frame and recording a second, differently-typed
        error on the child span). Instead the harness enforces limits after
        each response and emits one consistent ``*_limit_exceeded`` event.
        """
        return None

    def _check_rate_limit(self, prompt: Any, session_id: Any, attempt: int) -> None:
        """Reserve estimated request capacity before calling the model."""
        config = self.config.token_rate_limit
        if config is None:
            return
        now = time.monotonic()
        cutoff = now - config.window_seconds
        while self._rate_limit_events and self._rate_limit_events[0][0] <= cutoff:
            self._rate_limit_events.popleft()

        request_count = len(self._rate_limit_events)
        estimated_tokens = max(1, len(str(prompt)) // 4) + config.reserve_output_tokens
        used_tokens = sum(tokens for _, tokens in self._rate_limit_events)
        request_exceeded = (
            config.requests_per_window is not None
            and request_count >= config.requests_per_window
        )
        token_exceeded = (
            config.tokens_per_window is not None
            and used_tokens + estimated_tokens > config.tokens_per_window
        )
        if request_exceeded or token_exceeded:
            retry_after = config.window_seconds
            if self._rate_limit_events:
                retry_after = max(
                    0.0,
                    self._rate_limit_events[0][0] + config.window_seconds - now,
                )
            available = max(
                0,
                (config.tokens_per_window or used_tokens + estimated_tokens) - used_tokens,
            )
            error = RateLimitError(
                "model token/request rate limit exceeded",
                retry_after=retry_after,
                requested_tokens=estimated_tokens,
                available_tokens=available,
                window_seconds=config.window_seconds,
            )
            if config._on_exceeded:
                config._on_exceeded(
                    ErrorContext(
                        error_type=type(error).__name__,
                        error_message=str(error),
                        source="rate_limit",
                        session_id=session_id,
                        attempt=attempt + 1,
                        max_attempts=self.config.agent.max_retries + 1,
                        will_retry=attempt < self.config.agent.max_retries,
                    )
                )
            raise error
        self._rate_limit_events.append((now, estimated_tokens))

    # ── Token / cost limits ────────────────────────────────────────
    def _enforce_token_limits(self, usage_obj: Any, session_id: Any, attempt: int) -> None:
        """Raise when a token limit is exceeded (handled or as RuntimeError)."""
        tl = self.config.token_limits
        if not (tl and usage_obj):
            return
        input_tok = getattr(usage_obj, "input_tokens", 0) or 0
        output_tok = getattr(usage_obj, "output_tokens", 0) or 0
        reasoning_tok = getattr(usage_obj, "reasoning_tokens", 0) or 0
        checks = (
            ("reasoning", "Reasoning", tl.max_reasoning_tokens, reasoning_tok),
            ("input", "Input", tl.max_input_tokens, input_tok),
            ("output", "Output", tl.max_output_tokens, output_tok),
            ("total", "Total", tl.max_total_tokens, input_tok + output_tok),
        )
        for limit_type, label, limit, actual in checks:
            if limit is not None and actual > limit:
                self._raise_token_limit(
                    limit_type, label, limit, actual, usage_obj, tl, session_id, attempt
                )

    def _raise_token_limit(
        self, limit_type, label, limit, actual, usage_obj, tl, session_id, attempt
    ) -> None:
        error_ctx = self._guard_error_context(
            "TokenLimitExceeded",
            f"{label} tokens {actual} > {limit}",
            session_id=session_id,
            attempt=attempt,
            token_usage=self._build_token_usage_info(limit_type, limit, actual, usage_obj),
        )
        self._record_token_limit_metric(session_id, limit_type=limit_type)
        error_ctx.handled = True
        self._log(
            "error",
            "token_limit_exceeded",
            error_type="TokenLimitExceeded",
            error_message=error_ctx.error_message,
            error_source="guardrail",
            error_handled=True,
            session_id=session_id,
            attempt=attempt + 1,
        )
        output = (
            tl._on_token_limit(error_ctx)
            if tl._on_token_limit
            else f"Token limit exceeded: {error_ctx.error_message}"
        )
        raise _GuardrailHandled(
            AgentRunResult(output=output, success=False, error_context=error_ctx)
        )

    def _enforce_cost_limits(self, usage_obj: Any, session_id: Any, attempt: int) -> None:
        """Raise when a cost limit is exceeded (handled or as RuntimeError)."""
        cl = self.config.cost_limits
        if not (cl and usage_obj):
            return
        input_tok = getattr(usage_obj, "input_tokens", 0) or 0
        output_tok = getattr(usage_obj, "output_tokens", 0) or 0
        reasoning_tok = getattr(usage_obj, "reasoning_tokens", 0) or 0

        input_cost = input_tok * (cl.cost_per_input_token or 0)
        if cl.billing_mode == "output_plus_reasoning":
            output_cost = (output_tok + reasoning_tok) * (cl.cost_per_output_token or 0)
        else:
            output_cost = output_tok * (cl.cost_per_output_token or 0)
        total_cost = input_cost + output_cost

        checks = (
            ("Input", cl.max_input_cost, input_cost),
            ("Output", cl.max_output_cost, output_cost),
            ("Total", cl.max_total_cost, total_cost),
        )
        for label, limit, cost in checks:
            if limit is not None and cost > limit:
                self._raise_cost_limit(label, limit, cost, cl, session_id, attempt)

    def _raise_cost_limit(self, label, limit, cost, cl, session_id, attempt) -> None:
        error_ctx = self._guard_error_context(
            "CostLimitExceeded",
            f"{label} cost ${cost:.6f} > ${limit:.6f}",
            session_id=session_id,
            attempt=attempt,
        )
        error_ctx.handled = True
        self._log(
            "error",
            "cost_limit_exceeded",
            error_type="CostLimitExceeded",
            error_message=error_ctx.error_message,
            error_source="guardrail",
            error_handled=True,
            session_id=session_id,
            attempt=attempt + 1,
        )
        output = (
            cl._on_cost_limit(error_ctx)
            if cl._on_cost_limit
            else f"Cost limit exceeded: {error_ctx.error_message}"
        )
        raise _GuardrailHandled(
            AgentRunResult(output=output, success=False, error_context=error_ctx)
        )

    # ── Content transforms ─────────────────────────────────────────
    def _apply_content_filter(self, output: Any, session_id: Any, attempt: int) -> Any:
        cf = self.config.content_filter
        if not (cf and cf._on_filter):
            return output
        try:
            return cf._on_filter(output)
        except Exception as e:
            error_ctx = self._guard_error_context(
                type(e).__name__,
                str(e),
                session_id=session_id,
                attempt=attempt,
                stack_trace=traceback.format_exc(),
            )
            if cf._on_error:
                raise _GuardrailHandled(
                    AgentRunResult(
                        output=cf._on_error(error_ctx),
                        success=False,
                        error_context=error_ctx,
                    )
                )
            raise

    def _apply_pii_detection(self, output: Any, session_id: Any, attempt: int) -> Any:
        pd = self.config.pii_detection
        if not (pd and pd._on_redact):
            return output
        try:
            return pd._on_redact(output)
        except Exception as e:
            error_ctx = self._guard_error_context(
                type(e).__name__,
                str(e),
                session_id=session_id,
                attempt=attempt,
                stack_trace=traceback.format_exc(),
            )
            if pd._on_error:
                raise _GuardrailHandled(
                    AgentRunResult(
                        output=pd._on_error(error_ctx),
                        success=False,
                        error_context=error_ctx,
                    )
                )
            raise

    # ── Results ────────────────────────────────────────────────────
    def _build_success_result(self, result: Any, output: Any, usage_obj: Any) -> AgentRunResult:
        self._accumulate_usage(result)
        token_usage_info = None
        if usage_obj and self.config.token_limits:
            token_usage_info = self._build_token_usage_info("success", 0, 0, usage_obj)
        return AgentRunResult(
            output=output,
            success=True,
            error_context=None,
            new_messages=result.new_messages()
            if hasattr(result, "new_messages")
            else [],
            usage=usage_obj,
            cumulative_usage=self._cumulative_usage.copy(),
            token_usage=token_usage_info,
        )

    def _handle_usage_limit_exceeded(self, e, result, session_id, attempt) -> AgentRunResult:
        self._accumulate_usage(result)
        error_ctx = self._guard_error_context(
            "TokenLimitExceeded", str(e), session_id=session_id, attempt=attempt
        )
        self._log(
            "error",
            "token_limit_exceeded",
            error_type="TokenLimitExceeded",
            error_message=str(e),
            error_source="guardrail",
            error_handled=True,
            session_id=session_id,
            attempt=attempt + 1,
        )
        self._record_token_limit_metric(session_id, limit_type="usage_limits")
        error_ctx.handled = True
        tl = self.config.token_limits
        output = (
            tl._on_token_limit(error_ctx)
            if tl and tl._on_token_limit
            else f"Token limit exceeded: {error_ctx.error_message}"
        )
        return AgentRunResult(output=output, success=False, error_context=error_ctx)

    async def _handle_retryable(self, e, reason, result, session_id, attempt) -> bool:
        """Record a retryable failure; return True when a retry should happen."""
        self._accumulate_usage(result)
        will_retry = attempt < self.config.agent.max_retries
        max_attempts = self.config.agent.max_retries + 1
        error_ctx = ErrorContext(
            error_type="TimeoutError" if reason == "timeout" else type(e).__name__,
            error_message=(
                f"Agent execution timed out after {self.config.agent.timeout}s"
                if reason == "timeout"
                else str(e)
            ),
            source="rate_limit" if reason == "rate_limit" else "llm",
            session_id=session_id,
            attempt=attempt + 1,
            max_attempts=max_attempts,
            will_retry=will_retry,
            stack_trace=traceback.format_exc(),
        )

        log_fields: dict[str, Any] = {
            "attempt": attempt + 1,
            "max_attempts": max_attempts,
            "wait_seconds": (
                self.config.agent.backoff_multiplier**attempt if will_retry else 0.0
            ),
        }
        if reason == "timeout":
            log_fields["reason"] = "timeout"
            log_fields["timeout_seconds"] = self.config.agent.timeout
        elif reason == "rate_limit":
            log_fields["reason"] = "rate_limit"
            log_fields["retry_after_seconds"] = getattr(e, "retry_after", 0.0)
            log_fields["requested_tokens"] = getattr(e, "requested_tokens", 0)
            log_fields["available_tokens"] = getattr(e, "available_tokens", 0)
        else:
            log_fields["reason"] = "error"
            log_fields["error_type"] = type(e).__name__
            log_fields["error_message"] = str(e)[:200]
        self._log("info", "retry_attempt", **log_fields)

        if self.config.agent._on_retry:
            self.config.agent._on_retry(error_ctx)
        self._track_circuit_failure(type(e).__name__, str(e))

        if not will_retry:
            return False
        wait_time = (
            getattr(e, "retry_after", 0.0)
            if reason == "rate_limit"
            else self.config.agent.backoff_multiplier**attempt
        )
        self._log("debug", "retry_wait", wait_seconds=wait_time, attempt=attempt + 1)
        await asyncio.sleep(wait_time)
        return True

    async def _run_fallback(
        self, prompt, message_history, session_id, last_exception
    ) -> AgentRunResult:
        try:
            fallback_agent = build_harness_agent(
                build_model_ref(self.config.agent.fallback_model),
                observability_getter=lambda: self._observability,
            )
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
                source=getattr(last_exception, "_error_source", "llm"),
                session_id=session_id,
                attempt=self.config.agent.max_retries + 1,
                max_attempts=self.config.agent.max_retries + 1,
                will_retry=False,
                stack_trace=traceback.format_exc(),
            )
            if self.config.agent._on_error:
                return AgentRunResult(
                    output=self.config.agent._on_error(error_ctx),
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

    def _terminal_failure(self, session_id, last_exception) -> AgentRunResult:
        error_ctx = ErrorContext(
            error_type="MaxRetriesExceeded",
            error_message=str(last_exception),
            source=getattr(last_exception, "_error_source", "llm"),
            session_id=session_id,
            attempt=self.config.agent.max_retries + 1,
            max_attempts=self.config.agent.max_retries + 1,
            will_retry=False,
            stack_trace=(
                "".join(
                    traceback.format_exception(
                        type(last_exception),
                        last_exception,
                        last_exception.__traceback__,
                    )
                )
                if last_exception is not None
                else None
            ),
        )
        if self.config.agent._on_error:
            return AgentRunResult(
                output=self.config.agent._on_error(error_ctx),
                success=False,
                error_context=error_ctx,
                used_fallback=False,
                new_messages=[],
                usage=None,
            )
        exc = Exception(
            f"All {self.config.agent.max_retries} retries exhausted. "
            f"Last error: {str(last_exception)}"
        )
        exc._error_source = getattr(last_exception, "_error_source", None) or "llm"
        exc._cumulative_usage = self._cumulative_usage.copy()
        if self._observability and self._observability.traceback_frame_limit is not None:
            exc.__traceback__ = _truncate_traceback(
                exc.__traceback__, self._observability.traceback_frame_limit
            )
        raise exc

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
            self._log(
                "error",
                "circuit_breaker_open",
                failure_count=self._failure_count,
                threshold=cb.failure_threshold,
                timeout_seconds=cb.circuit_timeout,
                error_type=error_type,
                error_message=message,
            )
