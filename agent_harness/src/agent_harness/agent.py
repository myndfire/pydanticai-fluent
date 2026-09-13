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

import os
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Optional, TypeVar, Union

from .log_enrichment import LogContext, LogEnrichmentProvider
from ._agent_factory import build_harness_agent
from .logging import caller_code_location, set_harness_call_site

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
from .observability import Observability, ObservabilityBuilder, HARNESS_SETTINGS, _truncate_traceback
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
    TokenRateLimitConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
    TurnLimitsConfig,
    _GuardrailHandled,
)
from .model_config import ModelConfig, build_model
from .errorhandling import ErrorHandlingConfig, ErrorHandler
from .evaluators import Evaluator
from .compaction import (
    clear_tool_results_from_env,
    make_clear_tool_results,
    make_report_context_usage,
    make_sliding_window,
    make_tiered,
    make_warn_near_limits,
    report_context_usage_from_env,
    sliding_window_from_env,
    tiered_from_env,
    warn_near_limits_from_env,
)
from .persistence import (
    make_file_store,
    make_memory_store,
    make_mongo_store,
    make_sqlite_store,
    make_step_persistence,
    step_persistence_from_env,
)
from .execution import CURRENT_EXECUTION, ExecutionContext


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


class _RunTimeline:
    """Collects ordered segment timestamps for an agent run.

    Produces a non-overlapping latency breakdown: each segment is measured from
    the previous mark, and the remainder becomes ``overhead_seconds``.
    """

    _SEGMENTS = (
        ("memory_load", "memory_load_seconds"),
        ("prompt_fetch", "prompt_fetch_seconds"),
        ("agent_core", "agent_core_seconds"),
        ("memory_save", "memory_save_seconds"),
        ("evaluators", "evaluator_seconds"),
    )

    def __init__(self, start: float):
        self._start = start
        self._marks: dict[str, float] = {"start": start}

    def mark(self, name: str) -> None:
        self._marks[name] = time.time()

    def breakdown(self) -> dict:
        total = time.time() - self._start
        out: dict = {"total_seconds": total}
        previous = self._start
        measured = 0.0
        for name, key in self._SEGMENTS:
            current = self._marks.get(name, previous)
            segment = max(0.0, current - previous)
            out[key] = segment
            measured += segment
            previous = current
        out["overhead_seconds"] = max(0.0, total - measured)
        return out


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
        self._agent: Agent[Any, Any] = build_harness_agent(
            build_model(model_config), deps_type=deps_type,
            model_settings=model_settings,
            observability_getter=lambda: self._observability,
        )
        self.model = f"{model_config.provider}:{model_config.model_name}"
        self._deps_type = deps_type

        self.prompts = prompts or StaticPrompts()
        self._observability = observability  # Could be None; created lazily via property
        self.tools = tools or ToolRegistry(self._observability)
        self.evaluators = evaluators or []
        self._attach_observability_to_evaluators()
        self.guards = guards or GuardConfig()
        self.guards.observability = self._observability  # defer; .with_observability() propagates later
        self.error_handling = ErrorHandlingConfig()
        self._enrichment: list[LogEnrichmentProvider] = []
        self._workflow_context: dict[str, str] = {}
        # Max innermost traceback frames; unset/None/0 keeps the full traceback.
        self.traceback_frame_limit = HARNESS_SETTINGS.default_traceback_frames
        if self.traceback_frame_limit is not None and self._observability is not None:
            self._observability.traceback_frame_limit = self.traceback_frame_limit

        self._guard_runner = GuardRunner(self.guards)
        self._error_handler = ErrorHandler(self.error_handling)
        self._last_turn: Optional[TurnData] = None
        self._short_term_memory: Optional[MemoryProvider] = None
        self._long_term_memory: Optional[MemoryProvider] = None
        self._compaction: list[Any] = []
        self._persistence: Optional[Any] = None
        self._extra_toolsets: list[Any] = []
        self._rabbitmq_config: dict = {}
        self._input_queue: Optional[str] = None
        self._input_exchange: Optional[str] = None
        self._output_queue: Optional[str] = None
        self._output_exchange: Optional[str] = None
        self._dead_letter_queue: Optional[str] = None
        self._dead_letter_exchange: Optional[str] = None
        self._turn_counts: dict[str, int] = {}

        if self.tools.get_tools():
            self.tools.register_to_agent(
                self._agent, retries=self.guards.tool.max_retries
            )

    def _propagate_observability(self, observability: Observability) -> None:
        """Attach an observability stack to all interested components.

        Centralizes propagation so a lazily-created default stack reaches
        tools, guards, and the guard runner exactly like an explicitly
        provided one. Assigns ``_observability`` directly to avoid recursing
        through the ``observability`` property.
        """
        self._observability = observability
        if self.traceback_frame_limit is not None:
            observability.traceback_frame_limit = self.traceback_frame_limit
        if getattr(self, "tools", None) is not None:
            self.tools._observability = observability
        if getattr(self, "guards", None) is not None:
            self.guards.observability = observability
        guard_runner = getattr(self, "_guard_runner", None)
        if guard_runner is not None:
            guard_runner._observability = observability
        self._attach_observability_to_evaluators()

    @property
    def observability(self) -> Observability:
        """Lazy-init observability: creates default OTEL backends on first access."""
        if self._observability is None:
            self._propagate_observability(
                Observability(
                    builder=ObservabilityBuilder(service_name="agent")
                    .with_otel_observability(
                        otlp_endpoint=os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317"),
                    )
                )
            )
        return self._observability

    @observability.setter
    def observability(self, value: Observability):
        self._propagate_observability(value)

    def _attach_observability_to_evaluators(self) -> None:
        """Give opt-in evaluators (e.g. ``QualityCheck``) the observability stack.

        Agent-backed evaluators build their own PydanticAI agent; handing them
        the stack lets that agent correlate its spans to logs like the main one.
        """
        for evaluator in getattr(self, "evaluators", []):
            if hasattr(evaluator, "_observability"):
                evaluator._observability = self._observability

    def _warn_if_provider_ignored_max_tokens(self, result: Any, context: dict) -> None:
        """Canary: warn when a provider generated more than the configured max_tokens.

        Detects silent provider/API drift (e.g. the token cap being routed to a
        field the backend ignores), which would otherwise only show up as
        runaway generation.
        """
        max_tokens = None
        settings = self._model_settings
        if isinstance(settings, dict):
            max_tokens = settings.get("max_tokens")
        elif settings is not None:
            getter = getattr(settings, "get", None)
            if callable(getter):
                max_tokens = getter("max_tokens")
        if not max_tokens:
            return
        usage = getattr(result, "usage", None)
        completion = getattr(usage, "output_tokens", 0) or 0
        if completion and completion > max_tokens:
            self.observability.log_warning(
                "max_tokens_exceeded_by_provider",
                configured_max_tokens=max_tokens,
                completion_tokens=completion,
                **context,
            )
            self.observability.record_metric(
                "counter",
                "agent_max_tokens_exceeded_by_provider",
                1,
                **{
                    k: str(v)
                    for k, v in context.items()
                    if k in ("model", "session_id")
                },
            )

    def _rebuild_agent(self, model: Any = None, extra_toolsets: Any = None) -> None:
        """Rebuild the underlying Agent, preserving prior configuration.

        The Agent is immutable-by-replacement in this harness: model, settings,
        output and MCP changes each build a new instance. This helper carries
        over constructor toolsets, compaction capabilities, model settings,
        output type and dependency type so no ``with_*`` call silently drops
        them. Function tools are always re-registered fresh from the
        ToolRegistry: ``agent.tool()`` mutates the agent's function toolset
        in place, so carrying ``agent.toolsets`` by reference would register
        them twice.
        """
        if extra_toolsets:
            self._extra_toolsets.extend(extra_toolsets)
        kwargs: dict[str, Any] = {
            "model": model if model is not None else self._agent._model,
            "toolsets": list(self._extra_toolsets),
            "capabilities": list(self._compaction)
            + ([self._persistence] if self._persistence is not None else []),
        }
        if self._deps_type is not None:
            kwargs["deps_type"] = self._deps_type
        if self._model_settings is not None:
            kwargs["model_settings"] = self._model_settings
        if self._output_type is not None:
            kwargs["output_type"] = self._output_type
            kwargs["retries"] = self._output_retries
        self._agent = build_harness_agent(
            observability_getter=lambda: self._observability, **kwargs
        )
        if self.tools.get_tools():
            self.tools.register_to_agent(
                self._agent, retries=self.guards.tool.max_retries
            )

    def with_model(
        self,
        model: ModelConfig,
    ) -> "ManagedAgent":
        """Set the model using a ModelConfig object.

        Args:
            model: ModelConfig specifying provider, model_name, api_key, base_url.
        """
        self._rebuild_agent(model=build_model(model))
        self.model = f"{model.provider}:{model.model_name}"
        return self

    def with_model_settings(self, model_settings: Any) -> "ManagedAgent":
        """Set model settings (e.g. thinking, temperature, max_tokens).

        Args:
            model_settings: pydantic_ai ModelSettings dict or callable.
        """
        self._model_settings = model_settings
        self._rebuild_agent()
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

    def with_workflow_context(
        self, workflow: str, step: Optional[str] = None
    ) -> "ManagedAgent":
        """Attach generic workflow scope to every execution record."""
        self._workflow_context = {"workflow.name": workflow}
        if step:
            self._workflow_context["workflow.step"] = step
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
        # Propagate observability to the registry, but do not materialize the
        # lazy default here: .with_observability() must be able to replace it
        # later without leaving a discarded default stack behind.
        if self._observability is not None:
            self.tools._observability = self._observability
        self._rebuild_agent()
        return self

    def with_mcp_server(self, url: str, **kwargs) -> "ManagedAgent":
        """Add an MCP server as a toolset."""
        from pydantic_ai.mcp import MCPToolset, FastMCPClient

        tool_prefix = kwargs.get("tool_prefix")
        mcp_server = MCPToolset(FastMCPClient(url))
        if tool_prefix:
            mcp_server = mcp_server.prefixed(tool_prefix)

        self._rebuild_agent(extra_toolsets=[mcp_server])
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
        self._attach_observability_to_evaluators()
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
        self._rebuild_agent()
        return self

    def with_result_validator_retries(self, config: ResultValidatorRetryConfig) -> "ManagedAgent":
        """Set result validator retry configuration."""
        self.guards.result_validator = config
        self._output_retries = config.max_retries
        if self._output_type is not None:
            self._rebuild_agent()
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

    def with_token_rate_limit(self, config: TokenRateLimitConfig) -> "ManagedAgent":
        """Limit model requests/tokens over a sliding time window."""
        self.guards.token_rate_limit = config
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
        token_rate_limit: Optional[TokenRateLimitConfig] = None,
        cost_limits: Optional[CostLimitsConfig] = None,
    ) -> "ManagedAgent":
        """Set multiple guardrail configurations at once."""
        if content_filter:
            self.guards.content_filter = content_filter
        if pii_detection:
            self.guards.pii_detection = pii_detection
        if token_limits:
            self.guards.token_limits = token_limits
        if token_rate_limit:
            self.guards.token_rate_limit = token_rate_limit
        if cost_limits:
            self.guards.cost_limits = cost_limits
        self._guard_runner = GuardRunner(self.guards)
        return self

    def with_traceback_frame_limit(self, limit: Optional[int]) -> "ManagedAgent":
        """Set max traceback frames shown (None = full tracebacks)."""
        self.traceback_frame_limit = limit
        # Only touch an existing stack; otherwise the limit is stored and
        # applied when observability is provided (or lazily created).
        if self._observability is not None:
            self._observability.traceback_frame_limit = limit
        return self

    def with_minimal_traceback(self) -> "ManagedAgent":
        """Show only error message, no traceback frames."""
        return self.with_traceback_frame_limit(0)

    def with_full_traceback(self) -> "ManagedAgent":
        """Show full tracebacks (default)."""
        return self.with_traceback_frame_limit(None)

    def with_output(self, output_type: Any, output_retries: int = 3) -> "ManagedAgent":
        """
        Set the output type for structured responses.

        Args:
            output_type: The Pydantic model for structured output
            output_retries: Number of retries for output validation (default: 3)
        """
        self._output_type = output_type
        self._output_retries = output_retries
        self._rebuild_agent()
        return self

    def with_compaction(self, *caps: Any) -> "ManagedAgent":
        """Attach compaction capabilities (escape hatch for any strategy).

        Accepts upstream ``pydantic-ai-harness`` capabilities
        (ClearToolResults, SlidingWindowCompaction, TieredCompaction,
        WarnNearLimits, ReportContextUsage, ...) or any custom
        ``CompactionStrategy``. Order is preserved.

        Example:
            agent.with_compaction(
                ClearToolResults(max_fraction=0.7, keep_pairs=3),
                WarnNearLimits(max_context_fraction=0.9),
            )
        """
        self._compaction.extend(caps)
        self._rebuild_agent()
        return self

    def clear_compaction(self) -> "ManagedAgent":
        """Remove all compaction capabilities."""
        self._compaction = []
        self._rebuild_agent()
        return self

    def with_clear_tool_results(
        self,
        keep_pairs: int,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_fraction: Optional[float] = None,
    ) -> "ManagedAgent":
        """Clear old tool results, keeping the last ``keep_pairs`` pairs."""
        return self.with_compaction(
            make_clear_tool_results(
                keep_pairs=keep_pairs,
                max_messages=max_messages,
                max_tokens=max_tokens,
                max_fraction=max_fraction,
            )
        )

    def with_clear_tool_results_from_env(self) -> "ManagedAgent":
        """Clear old tool results using ``HARNESS_COMPACTION_*`` env vars."""
        return self.with_compaction(clear_tool_results_from_env())

    def with_sliding_window(
        self,
        keep_messages: int,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_fraction: Optional[float] = None,
    ) -> "ManagedAgent":
        """Keep only the recent ``keep_messages`` tail of the history."""
        return self.with_compaction(
            make_sliding_window(
                keep_messages=keep_messages,
                max_messages=max_messages,
                max_tokens=max_tokens,
                max_fraction=max_fraction,
            )
        )

    def with_sliding_window_from_env(self) -> "ManagedAgent":
        """Sliding-window compaction using ``HARNESS_COMPACTION_*`` env vars."""
        return self.with_compaction(sliding_window_from_env())

    def with_warn_near_limits(
        self,
        warning_threshold: float,
        max_iterations: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        max_context_fraction: Optional[float] = None,
    ) -> "ManagedAgent":
        """Warn the model (no history edits) as limits approach."""
        return self.with_compaction(
            make_warn_near_limits(
                warning_threshold=warning_threshold,
                max_iterations=max_iterations,
                max_context_tokens=max_context_tokens,
                max_context_fraction=max_context_fraction,
            )
        )

    def with_warn_near_limits_from_env(self) -> "ManagedAgent":
        """Warn-near-limits using ``HARNESS_COMPACTION_*`` env vars."""
        return self.with_compaction(warn_near_limits_from_env())

    def with_report_context_usage(self, on_usage: Any) -> "ManagedAgent":
        """Report live context usage via an ``on_usage`` callback."""
        return self.with_compaction(make_report_context_usage(on_usage))

    def with_report_context_usage_from_env(self, on_usage: Any) -> "ManagedAgent":
        """Report context usage; window config from env, callback explicit."""
        return self.with_compaction(report_context_usage_from_env(on_usage))

    def with_tiered_compaction(
        self,
        tiers: Any,
        target_tokens: Optional[int] = None,
        target_fraction: Optional[float] = None,
    ) -> "ManagedAgent":
        """Escalate cheap-to-expensive tiers until under target (recommended)."""
        return self.with_compaction(
            make_tiered(
                tiers=tiers,
                target_tokens=target_tokens,
                target_fraction=target_fraction,
            )
        )

    def with_tiered_compaction_from_env(self, tiers: Any) -> "ManagedAgent":
        """Tiered compaction; target budget from env, tiers explicit."""
        return self.with_compaction(tiered_from_env(tiers))

    def with_step_persistence(self, capability: Any) -> "ManagedAgent":
        """Attach a StepPersistence capability (replaces any existing one).

        Accepts an upstream ``StepPersistence`` built by
        ``agent_harness.persistence.make_step_persistence`` (or constructed
        directly). One slot per agent: attaching again replaces the previous
        capability, since multiple instances need explicit upstream ids.
        """
        self._persistence = capability
        self._rebuild_agent()
        return self

    def clear_step_persistence(self) -> "ManagedAgent":
        """Remove the StepPersistence capability."""
        self._persistence = None
        self._rebuild_agent()
        return self

    def with_memory_steps(
        self,
        agent_name: Optional[str] = None,
        max_snapshots_per_run: Optional[int] = None,
    ) -> "ManagedAgent":
        """Persist steps to a process-local in-memory store (great for tests)."""
        return self.with_step_persistence(
            make_step_persistence(
                make_memory_store(max_snapshots_per_run=max_snapshots_per_run),
                agent_name=agent_name,
            )
        )

    def with_file_steps(
        self,
        directory: str,
        agent_name: Optional[str] = None,
        max_snapshots_per_run: Optional[int] = None,
    ) -> "ManagedAgent":
        """Persist steps to a directory-backed file store."""
        return self.with_step_persistence(
            make_step_persistence(
                make_file_store(
                    directory, max_snapshots_per_run=max_snapshots_per_run
                ),
                agent_name=agent_name,
            )
        )

    def with_sqlite_steps(
        self,
        database: str,
        agent_name: Optional[str] = None,
        max_snapshots_per_run: Optional[int] = None,
    ) -> "ManagedAgent":
        """Persist steps to a single-file SQLite store."""
        return self.with_step_persistence(
            make_step_persistence(
                make_sqlite_store(
                    database, max_snapshots_per_run=max_snapshots_per_run
                ),
                agent_name=agent_name,
            )
        )

    def with_mongo_steps(
        self,
        database: str,
        db_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        max_snapshots_per_run: Optional[int] = None,
    ) -> "ManagedAgent":
        """Persist steps to MongoDB (needs the ``mongodb`` harness extra)."""
        return self.with_step_persistence(
            make_step_persistence(
                make_mongo_store(
                    database,
                    db_url=db_url,
                    max_snapshots_per_run=max_snapshots_per_run,
                ),
                agent_name=agent_name,
            )
        )

    def with_step_persistence_from_env(self, store: Any = None) -> "ManagedAgent":
        """Step persistence from ``HARNESS_PERSISTENCE_*`` env vars.

        Backend, paths, retention bound and agent name all come from env;
        pass an explicit ``store`` to skip backend selection.
        """
        return self.with_step_persistence(step_persistence_from_env(store))

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

    # ── Run helpers (shared by run and run_stream) ─────────────────
    def _build_run_context(
        self,
        session_id: str,
        prompt_id: str,
        enrichment: Optional[LogContext],
        execution: Optional[ExecutionContext] = None,
    ) -> dict:
        """Build the per-run log/trace context."""
        context = {
            "session_id": session_id,
            "model": self.model,
            "model_settings": self._model_settings,
        }
        provider, _, model_name = self.model.partition(":")
        context.update(
            {
                "model.requested.name": model_name or self.model,
                "model.requested.provider": provider,
            }
        )
        if execution is not None:
            context.update(execution.as_dict())
        context.update(self._workflow_context)
        if prompt_id != "default":
            context["prompt_id"] = prompt_id
        for provider in self._enrichment:
            context.update(provider.enrich())
        if enrichment:
            context.update(enrichment.enrich())
        return context

    async def _load_history(
        self, message_history: MessageHistory, session_id: str
    ) -> None:
        """Load short- and long-term memory into the message history."""
        try:
            if self._short_term_memory:
                await message_history.load(session_id, self._short_term_memory)
            if self._long_term_memory:
                await message_history.load(session_id, self._long_term_memory)
        except Exception as e:
            e._error_source = "memory"
            raise

    async def _apply_system_prompt(self, prompt_id: str, prompt_vars: dict) -> None:
        """Fetch the system prompt and attach it to the underlying agent."""
        try:
            system_prompt = await self.prompts.get_system_prompt(
                prompt_id=prompt_id, **prompt_vars
            )
            if system_prompt:
                self._agent._system_prompts = (system_prompt,)
        except Exception as e:
            e._error_source = "prompt"
            raise

    async def _run_evaluators(
        self, prompt_text: str, evaluation_target: Any, context: dict
    ) -> list[Any]:
        """Run evaluators and return structured results with lifecycle logging."""
        self._attach_observability_to_evaluators()
        session_id = context.get("session_id")
        results: list[Any] = []
        for evaluator in self.evaluators:
            try:
                evaluator_name = getattr(evaluator, "name", type(evaluator).__name__)
                self.observability.log_info(
                    "evaluator_started",
                    component="evaluators",
                    evaluator=evaluator_name,
                    session_id=session_id,
                )
                started = time.time()
                evaluation = await evaluator.evaluate(
                    prompt_text, evaluation_target, context
                )
                if evaluation is not None:
                    results.append(evaluation)
                self.observability.log_info(
                    "evaluator_completed",
                    component="evaluators",
                    evaluator=evaluator_name,
                    session_id=session_id,
                    performance={"duration_seconds": time.time() - started},
                    evaluation=(
                        evaluation.__dict__
                        if hasattr(evaluation, "__dict__")
                        else evaluation
                    ),
                )
            except Exception as e:
                e._error_source = "evaluator"
                self.observability.log_error(
                    "evaluator_failed",
                    component="evaluators",
                    exception=e,
                    evaluator=getattr(evaluator, "name", type(evaluator).__name__),
                    session_id=session_id,
                )
                raise
        return results

    def _turn_usage_from_summary(
        self, turn_summary: dict
    ) -> tuple[Optional[UsageData], Optional[float], dict, dict, int]:
        """Derive (usage, cost, cost_breakdown, latency_breakdown, turn_count)."""
        if not turn_summary.get("turn_count"):
            return None, None, {}, {}, 0
        tok = turn_summary["token_usage"]
        usage = UsageData(
            input_tokens=tok["input_tokens"],
            output_tokens=tok["output_tokens"],
            reasoning_tokens=tok["reasoning_tokens"],
            total_tokens=tok["total_tokens"],
            prompt_tokens=tok["input_tokens"],
            completion_tokens=tok["output_tokens"],
        )
        cost = turn_summary["cost"].get("total_usd")
        if not self.observability.granularity.standard:
            return usage, cost, {}, {}, 0
        return (
            usage,
            cost,
            dict(turn_summary["cost"]),
            dict(turn_summary["latency"]),
            turn_summary["turn_count"],
        )

    def _build_turn_data(
        self,
        messages: list,
        usage: Optional[UsageData],
        cost: Optional[float],
        cost_breakdown: dict,
        latency_breakdown: dict,
        turn_count: int,
        duration: float,
        status: str,
        error: Optional[dict] = None,
    ) -> TurnData:
        """Assemble a conversation-turn record (shared by run and run_stream)."""
        return TurnData(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            completed_at=datetime.now(),
            messages=messages,
            usage=usage,
            duration_seconds=duration,
            cost=cost,
            cost_breakdown=cost_breakdown,
            billing_mode=self.guards.token_limits.billing_mode
            if self.guards.token_limits
            else "output_plus_reasoning",
            latency_breakdown=latency_breakdown,
            turn_count=turn_count,
            model=self.model,
            status=status,
            error=error,
        )

    def _enforce_turn_limit(self, session_id: str) -> Any:
        """Return a handled result when the conversation turn limit is hit."""
        tl = self.guards.turn_limits
        if not tl:
            return None
        count = self._turn_counts.get(session_id, 0) + 1
        self._turn_counts[session_id] = count
        if tl.max_turns is not None and count > tl.max_turns:
            error_ctx = ErrorContext(
                error_type="TurnLimitExceeded",
                error_message=f"Turn {count} exceeds max {tl.max_turns}",
                source="guardrail",
                session_id=session_id,
            )
            if tl._on_turn_limit:
                return tl._on_turn_limit(error_ctx)
            raise RuntimeError(error_ctx.error_message)
        return None

    def _emit_run_summary(
        self, turn_summary: dict, context: dict, latency_breakdown: dict, status: str
    ) -> None:
        """Emit the run summary and attach the latency breakdown to the turn."""
        self.observability.log_run_summary(
            turn_summary, context, latency_breakdown, status=status
        )
        if self._last_turn is not None and self.observability.granularity.standard:
            self._last_turn.latency_breakdown = {
                **self._last_turn.latency_breakdown,
                **latency_breakdown,
            }

    async def run(
        self,
        prompt: Union[str, Sequence[UserContent]],
        message_history: MessageHistory,
        session_id: str,
        save_to: Optional[list[MemoryProvider]] = None,
        deps: Any = None,
        enrichment: Optional[LogContext] = None,
        conversation_id: Optional[str] = None,
        execution: Optional[ExecutionContext] = None,
        workflow: Optional[str] = None,
        step: Optional[str] = None,
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
            conversation_id: Upstream dialogue grouping for step persistence
                (multi-turn runs share one id; defaults to ``session_id``).
            **kwargs: Additional context for prompt rendering

        Returns:
            Agent result
        """
        start_time = time.time()
        token_usage_logged = False
        # Capture the caller's location so failures raised inside pydantic-ai's
        # own asyncio task (which drops the caller frame) still attribute back
        # to the application. Propagates to hooks/spans via the contextvar.
        callsite = caller_code_location()
        set_harness_call_site(callsite)

        prompt_id = kwargs.pop("prompt_id", "default")
        prompt_vars = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        # Text-only view of the prompt for evaluators, error contexts and logs.
        # The original prompt is what reaches the model.
        prompt_text = prompt_to_text(prompt)

        execution = execution or ExecutionContext(
            session_id=session_id,
            conversation_id=conversation_id or session_id,
        )
        context = self._build_run_context(session_id, prompt_id, enrichment, execution)
        if workflow:
            context["workflow.name"] = workflow
        if step:
            context["workflow.step"] = step
        timeline = _RunTimeline(start_time)
        execution_token = CURRENT_EXECUTION.set(execution)

        try:
            async with self.observability.observe("agent_run", **context):
                execution.budget.check()
                execution.budget.consume_iteration()
                await self._load_history(message_history, session_id)
                timeline.mark("memory_load")

                handled = self._enforce_turn_limit(session_id)
                if handled is not None:
                    CURRENT_EXECUTION.reset(execution_token)
                    return handled

                history = message_history.messages

                await self._apply_system_prompt(prompt_id, prompt_vars)
                timeline.mark("prompt_fetch")

                result = await self._guard_runner.run_with_guards(
                    agent=self._agent,
                    prompt=prompt,
                    message_history=history,
                    deps=deps,
                    conversation_id=conversation_id or session_id,
                    execution_context=execution,
                )

                timeline.mark("agent_core")
                duration = time.time() - start_time
                status = "success" if result.success else "error"
                if hasattr(result, "used_fallback") and result.used_fallback:
                    status = "fallback"

                new_messages = []
                if hasattr(result, "new_messages"):
                    nm = result.new_messages
                    new_messages = nm() if callable(nm) else nm

                serialized_messages = filter_thinking_parts(new_messages)

                # Log per-turn metrics and collect the run aggregate.
                turn_summary = self.observability.log_turns(result, context)
                token_usage_logged = True
                self._warn_if_provider_ignored_max_tokens(result, context)

                try:
                    usage, cost, cost_breakdown, latency_breakdown, turn_count = (
                        self._turn_usage_from_summary(turn_summary)
                    )
                    turn = self._build_turn_data(
                        serialized_messages,
                        usage,
                        cost,
                        cost_breakdown,
                        latency_breakdown,
                        turn_count,
                        duration,
                        status,
                        error=None
                        if result.success
                        else {
                            "error_type": result.error_context.error_type
                            if result.error_context
                            else "Unknown",
                            "error_message": result.error_context.error_message
                            if result.error_context
                            else "Unknown",
                        },
                    )

                    self._last_turn = turn
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

                timeline.mark("memory_save")

                await self._run_evaluators(prompt_text, result, context)
                timeline.mark("evaluators")

                try:
                    if self._agent._output_type is None:
                        result.output = extract_clean_output(result)
                except Exception as e:
                    e._error_source = "output"
                    raise

                self._emit_run_summary(
                    turn_summary, context, timeline.breakdown(), status=status
                )

                CURRENT_EXECUTION.reset(execution_token)
                return result

        except Exception as e:
            # Apply traceback frame limit to terminal output
            if self.traceback_frame_limit is not None and self.traceback_frame_limit >= 0:
                e.__traceback__ = _truncate_traceback(e.__traceback__, self.traceback_frame_limit)
            source = getattr(e, "_error_source", "unknown")
            # Capture turn metrics even on failure (if a result was produced).
            if not token_usage_logged:
                if "result" in locals() and result is not None:
                    try:
                        fail_summary = self.observability.log_turns(result, context)
                        self.observability.log_run_summary(
                            fail_summary,
                            context,
                            {"total_seconds": time.time() - start_time},
                            status="error",
                        )
                    except Exception:
                        pass
                else:
                    cumulative = getattr(e, "_cumulative_usage", None)
                    if cumulative:
                        self.observability.log_info(
                            "run_summary",
                            token_usage=cumulative,
                            run={"status": "error", "turn_count": 0},
                            **context,
                        )
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
                CURRENT_EXECUTION.reset(execution_token)
                return error_result

            # Create error turn even on failure
            error_turn = TurnData(
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                completed_at=datetime.now(),
                messages=[
                    {"role": "system", "content": f"Error: {type(e).__name__}: {str(e)}"}
                ],
                usage=None,
                duration_seconds=time.time() - start_time,
                model=self.model,
                status="error",
            )
            self._last_turn = error_turn

            # Save error turn to memory providers
            if save_to:
                providers = save_to if isinstance(save_to, list) else [save_to]
                for provider in providers:
                    try:
                        await provider.save_turn(session_id, error_turn)
                    except Exception:
                        pass  # Don't let save failure mask the original error

            CURRENT_EXECUTION.reset(execution_token)
            raise

    async def run_stream(
        self,
        prompt: Union[str, Sequence[UserContent]],
        message_history: MessageHistory,
        session_id: str,
        save_to: Optional[list[MemoryProvider]] = None,
        deps: Optional[Any] = None,
        enrichment: Optional[LogContext] = None,
        conversation_id: Optional[str] = None,
        execution: Optional[ExecutionContext] = None,
        workflow: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs,
    ):
        """Run agent with streaming output, yielding text chunks in real-time.

        Mirrors the behavior of run() but yields text chunks as they arrive
        from the model. Token limits are enforced via pydantic-ai's UsageLimits.

        Args:
            prompt: User prompt text
            message_history: MessageHistory object with loaded history (required)
            session_id: Session ID (required - key for saving turns)
            save_to: Optional list of memory providers to save the turn to
            deps: Dependencies for dependency injection
            enrichment: Optional LogContext with per-run enrichment keys
            conversation_id: Upstream dialogue grouping for step persistence
                (defaults to ``session_id``).
            **kwargs: Additional context for prompt rendering

        Yields:
            Text chunks as they are generated by the model

        Raises:
            UsageLimitExceeded: When token limits are exceeded (if no callback)
            RuntimeError: When other errors occur
        """
        from pydantic_ai.usage import UsageLimits, UsageLimitExceeded

        start_time = time.time()
        callsite = caller_code_location()
        set_harness_call_site(callsite)

        prompt_id = kwargs.pop("prompt_id", "default")
        prompt_vars = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        prompt_text = prompt_to_text(prompt)

        execution = execution or ExecutionContext(
            session_id=session_id,
            conversation_id=conversation_id or session_id,
        )
        context = self._build_run_context(session_id, prompt_id, enrichment, execution)
        if workflow:
            context["workflow.name"] = workflow
        if step:
            context["workflow.step"] = step

        async with self.observability.observe("agent_run_stream", **context):
            execution.budget.check()
            execution.budget.consume_iteration()
            await self._load_history(message_history, session_id)

            history = message_history.messages

            await self._apply_system_prompt(prompt_id, prompt_vars)

            # Limits are enforced by the harness after each response so a
            # violation surfaces as one consistent event (see guards). Passing
            # pydantic-ai UsageLimits would raise UsageLimitExceeded mid-run
            # inside its own task, losing the caller frame and duplicating the
            # error across the child span and the log record.
            usage_limits = None

            try:
                async with self._agent.run_stream(
                    prompt,
                    message_history=history,
                    usage_limits=usage_limits,
                    conversation_id=conversation_id or session_id,
                ) as result:
                    collected = ""
                    buffer_stream = bool(
                        self.guards.content_filter or self.guards.pii_detection
                    )
                    async for chunk in result.stream_text(delta=True):
                        collected += chunk
                        if not buffer_stream:
                            yield chunk

                    usage_obj = self._guard_runner._extract_usage(result)
                    self._guard_runner._enforce_token_limits(
                        usage_obj, session_id, 0
                    )
                    self._guard_runner._enforce_cost_limits(
                        usage_obj, session_id, 0
                    )
                    safe_output = self._guard_runner._apply_content_filter(
                        collected, session_id, 0
                    )
                    safe_output = self._guard_runner._apply_pii_detection(
                        safe_output, session_id, 0
                    )
                    if buffer_stream:
                        yield safe_output

                    # Log per-turn metrics after stream completes
                    duration = time.time() - start_time
                    turn_summary = self.observability.log_turns(result, context)
                    self._warn_if_provider_ignored_max_tokens(result, context)

                    # Capture reasoning traces if enabled
                    reasoning_traces = None
                    if (
                        self.guards.token_limits
                        and self.guards.token_limits._capture_reasoning_traces
                        and hasattr(result, "new_messages")
                    ):
                        try:
                            nm = result.new_messages
                            new_messages = nm() if callable(nm) else nm
                            reasoning_msgs = filter_thinking_parts(new_messages)
                            reasoning_traces = str(reasoning_msgs) if reasoning_msgs else None
                        except Exception:
                            pass

                    # Build TurnData for memory
                    usage, cost, cost_breakdown, latency_breakdown, turn_count = (
                        self._turn_usage_from_summary(turn_summary)
                    )

                    messages = []
                    if reasoning_traces:
                        messages.append({"role": "system", "content": f"Reasoning traces: {reasoning_traces}"})
                    messages.append({"role": "assistant", "content": safe_output})

                    turn = self._build_turn_data(
                        messages,
                        usage,
                        cost,
                        cost_breakdown,
                        latency_breakdown,
                        turn_count,
                        duration,
                        "success",
                    )
                    self._last_turn = turn

                    if save_to:
                        providers = save_to if isinstance(save_to, list) else [save_to]
                        for provider in providers:
                            try:
                                await provider.save_turn(session_id, turn)
                            except Exception:
                                pass

                    await self._run_evaluators(prompt_text, safe_output, context)

                    self._emit_run_summary(
                        turn_summary,
                        context,
                        {"total_seconds": time.time() - start_time},
                        status="success",
                    )

            except _GuardrailHandled as handled:
                # A post-stream guard may recover with a safe replacement.
                # The raw stream has already been withheld when a transform is
                # configured, so yielding this value is safe.
                yield handled.result.output
                CURRENT_EXECUTION.reset(execution_token)
                return
            except UsageLimitExceeded as e:
                duration = time.time() - start_time
                error_message = str(e)

                # Capture reasoning traces if enabled
                reasoning_traces = None
                if (
                    self.guards.token_limits
                    and self.guards.token_limits._capture_reasoning_traces
                    and hasattr(result, "new_messages")
                ):
                    try:
                        nm = result.new_messages
                        new_messages = nm() if callable(nm) else nm
                        reasoning_msgs = filter_thinking_parts(new_messages)
                        reasoning_traces = str(reasoning_msgs) if reasoning_msgs else None
                    except Exception:
                        pass

                # Log truncation event with full context
                self.observability.log_error(
                    "token_limit_exceeded",
                    error_message=error_message,
                    session_id=session_id,
                    partial_output_length=len(collected),
                    reasoning_traces_length=len(reasoning_traces) if reasoning_traces else 0,
                    duration_seconds=duration,
                )
                self.observability.record_metric(
                    "counter",
                    "agent_token_limit_exceeded",
                    1,
                    limit_type="streaming",
                    session_id=session_id,
                )

                error_ctx = ErrorContext(
                    error_type="TokenLimitExceeded",
                    error_message=error_message,
                    source="guardrail",
                    session_id=session_id,
                    partial_output=collected,
                    reasoning_traces=reasoning_traces,
                )

                # Save error turn to memory
                if save_to:
                    error_turn = TurnData(
                        turn_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        completed_at=datetime.now(),
                        messages=[
                            {"role": "system", "content": f"Error: {error_message}"},
                            {"role": "assistant", "content": collected},
                        ],
                        usage=None,
                        duration_seconds=duration,
                        model=self.model,
                        status="error",
                    )
                    providers = save_to if isinstance(save_to, list) else [save_to]
                    for provider in providers:
                        try:
                            await provider.save_turn(session_id, error_turn)
                        except Exception:
                            pass

                # Call streaming-specific callback if set, otherwise fall back to regular callback
                if (
                    self.guards.token_limits
                    and self.guards.token_limits._on_streaming_token_limit
                ):
                    callback_result = self.guards.token_limits._on_streaming_token_limit(
                        error_ctx, collected
                    )
                    if callback_result is not None:
                        yield callback_result
                    else:
                        yield f"\n[TRUNCATED: {error_message}]\nPartial output: {collected[:200]}..."
                    CURRENT_EXECUTION.reset(execution_token)
                    return
                elif (
                    self.guards.token_limits
                    and self.guards.token_limits._on_token_limit
                ):
                    callback_result = self.guards.token_limits._on_token_limit(error_ctx)
                    if callback_result is not None:
                        yield callback_result
                    else:
                        yield f"\n[TRUNCATED: {error_message}]"
                    CURRENT_EXECUTION.reset(execution_token)
                    return
                else:
                    raise RuntimeError(error_ctx.error_message) from e

            except Exception as e:
                if self.traceback_frame_limit is not None and self.traceback_frame_limit >= 0:
                    e.__traceback__ = _truncate_traceback(e.__traceback__, self.traceback_frame_limit)
                CURRENT_EXECUTION.reset(execution_token)
                raise

            CURRENT_EXECUTION.reset(execution_token)

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
