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

"""Unified observability facade combining logging, tracing, and metrics."""

import os
import socket
import traceback
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional, Union

from pydantic_settings import BaseSettings
from dotenv import find_dotenv

from .logging import NoOpLogger, Logger, OTELLogger, app_code_location, get_harness_call_site
from .errorhandling import ErrorContext
from .tracing import Tracer, NoOpTracer, OTELTracer
from .metrics import MetricsCollector, NoOpMetrics, MetricNames, OTELMetrics
from .telemetry_runtime import TelemetryRuntime
from .telemetry_schema import (
    TelemetryFields,
    bounded_metric_attributes,
    execution_context,
    new_error_id,
)


from pydantic import Field

# Granularity ladder for telemetry emission across logs, metrics and traces.
TELEMETRY_LEVELS = ("minimal", "standard", "verbose")


class HarnessSettings(BaseSettings):
    """Harness environment settings read from .env at module load."""

    app_env: str = "development"
    service_name: str = "agent-harness"
    traceback_frame_limit: Optional[int] = None
    # Max innermost traceback frames kept in error stacktraces. Unset, None or
    # <= 0 means the full traceback; a positive N keeps the last N frames.
    default_traceback_frames: Optional[int] = Field(
        default=None,
        validation_alias="HARNESS_DEFAULT_TRACEBACK_FRAMES",
    )
    # Granularity of emitted telemetry (logs, metrics, traces, memory):
    #   minimal  -> run summary + warnings/errors only
    #   standard -> + per-turn records/metrics and enriched TurnData
    #   verbose  -> + retries, lifecycle events, trace content, TTFT
    telemetry_level: str = Field(
        default="standard",
        validation_alias="HARNESS_TELEMETRY_LEVEL",
    )
    telemetry_enabled: bool = Field(
        default=True,
        validation_alias="HARNESS_TELEMETRY_ENABLED",
    )
    # Render OTel records to the local console via the OTel console exporters.
    telemetry_console: bool = Field(
        default=True,
        validation_alias="HARNESS_TELEMETRY_CONSOLE",
    )

    class Config:
        env_file = find_dotenv() or ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Read once at module load — changes require restart
HARNESS_SETTINGS = HarnessSettings()


class TelemetryGranularity:
    """Resolve and compare the configured telemetry granularity level."""

    _RANK = {"minimal": 0, "standard": 1, "verbose": 2}

    def __init__(self, level: str | None = None):
        level = (level or HARNESS_SETTINGS.telemetry_level or "standard").lower()
        if level not in self._RANK:
            level = "standard"
        self.level = level

    @property
    def rank(self) -> int:
        return self._RANK[self.level]

    def at_least(self, level: str) -> bool:
        return self.rank >= self._RANK[level]

    @property
    def minimal(self) -> bool:
        return self.level == "minimal"

    @property
    def standard(self) -> bool:
        return self.rank >= self._RANK["standard"]

    @property
    def verbose(self) -> bool:
        return self.level == "verbose"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TelemetryGranularity({self.level!r})"


def _truncate_traceback(tb, limit: int):
    """Truncate traceback chain to show at most `limit` innermost frames.

    Args:
        tb: Traceback object (or None)
        limit: Max innermost frames to keep. ``None`` or ``<= 0`` means no
            truncation (full traceback).

    Returns:
        Truncated traceback, the original, or None.
    """
    if tb is None:
        return tb
    if limit is None or limit <= 0:
        return tb

    import types

    # Collect all frames
    frames = []
    current = tb
    while current is not None:
        frames.append(current)
        current = current.tb_next

    # Keep only the last `limit` frames
    keep = frames[-limit:]
    if not keep:
        return None

    # Rebuild chain from bottom up
    prev = None
    for frame in reversed(keep):
        prev = types.TracebackType(
            tb_next=prev,
            tb_frame=frame.tb_frame,
            tb_lasti=frame.tb_lasti,
            tb_lineno=frame.tb_lineno,
        )
    return prev


def _format_traceback(exception: Optional[BaseException], limit: Optional[int]) -> Optional[str]:
    """Format an exception traceback, honoring the frame limit.

    ``limit`` follows :func:`_truncate_traceback`: ``None``/``<= 0`` is full,
    otherwise keep the innermost ``limit`` frames (``traceback`` uses a negative
    limit for innermost frames).
    """
    if exception is None:
        return None
    fmt_limit = None if (limit is None or limit <= 0) else -abs(limit)
    return "".join(
        traceback.format_exception(
            type(exception), exception, exception.__traceback__, limit=fmt_limit
        )
    )


def build_error_attributes(
    ctx: Optional[ErrorContext] = None,
    *,
    exception: Optional[BaseException] = None,
    callsite: Optional[Mapping] = None,
    limit: Optional[int] = None,
) -> dict:
    """Project an ``ErrorContext`` (+ exception) into the canonical error schema.

    This is the single source of truth for error fields emitted to logs, spans
    and metric labels: ``error.type``/``.message``/``.stacktrace``/``.source``/
    ``.handled``, the ``exception.*`` mirror, and ``code.*`` for the caller.

    ``code.*`` prefers the deepest user frame of ``exception``; when the stack
    is entirely internal (errors raised inside pydantic-ai's asyncio task) it
    falls back to ``callsite`` -- the caller recorded at ``ManagedAgent.run()``.
    """
    error_type = getattr(ctx, "error_type", None)
    error_message = getattr(ctx, "error_message", None)
    source = getattr(ctx, "source", None)
    handled = bool(getattr(ctx, "handled", False))
    stack_trace = getattr(ctx, "stack_trace", None)

    if exception is not None:
        error_type = error_type or type(exception).__name__
        error_message = error_message or str(exception)
        source = source or getattr(exception, "_error_source", None)
        if not stack_trace:
            stack_trace = _format_traceback(exception, limit)

    attrs: dict = {}
    if error_type:
        attrs["error.type"] = error_type
        attrs["exception.type"] = error_type
    if error_message:
        attrs["error.message"] = error_message
        attrs["exception.message"] = error_message
    if stack_trace:
        attrs["error.stacktrace"] = stack_trace
        attrs["exception.stacktrace"] = stack_trace
    attrs["error.source"] = source or "unknown"
    attrs["error.handled"] = handled
    error_id = getattr(ctx, "error_id", None) if ctx is not None else None
    if exception is not None:
        error_id = error_id or getattr(exception, "_telemetry_error_id", None)
        if error_id is None:
            error_id = new_error_id()
            try:
                setattr(exception, "_telemetry_error_id", error_id)
            except Exception:
                pass
    if error_id:
        attrs[TelemetryFields.ERROR_ID] = error_id

    location: dict = {}
    if exception is not None:
        location = app_code_location(exception.__traceback__)
    if not location and callsite:
        location = dict(callsite)
    attrs.update(location)
    return attrs


def _exception_record(exception: BaseException, limit: Optional[int] = None) -> dict:
    """Backwards-compatible wrapper: canonical attrs for a caught exception."""
    return build_error_attributes(exception=exception, callsite=get_harness_call_site(), limit=limit)


def _error_attrs(
    exception: Optional[BaseException],
    context: dict,
    limit: Optional[int] = None,
) -> dict:
    """Collect canonical ``error.*``/``exception.*``/``code.*`` attrs.

    Handles both an exception object and context-provided error details
    (``error_type``/``error_message``/``stack_trace`` from guardrails, or an
    ``error={"type", "message"}`` mapping from tools).
    """
    error = context.get("error")
    error = error if isinstance(error, Mapping) else {}
    error_type = context.get("error_type") or error.get("type")
    error_message = context.get("error_message") or error.get("message")
    stack_trace = context.get("stack_trace") or error.get("stacktrace")
    error_source = context.get("error_source") or context.get("source")

    ctx = None
    if error_type or error_message or stack_trace or error_source or "error_handled" in context:
        ctx = ErrorContext(
            error_type=error_type or "",
            error_message=error_message or "",
            source=error_source or "unknown",
            stack_trace=stack_trace,
            handled=bool(context.get("error_handled", False)),
        )
    callsite = context.get("_error_callsite") or get_harness_call_site() or None
    attrs = build_error_attributes(
        ctx, exception=exception, callsite=callsite, limit=limit
    )
    if TelemetryFields.ERROR_ID not in attrs:
        attrs[TelemetryFields.ERROR_ID] = context.get(
            TelemetryFields.ERROR_ID,
            context.get("error_id", new_error_id()),
        )
    return attrs


def _error_body(event_name: str, attrs: dict) -> str:
    """Build a human-readable message body for an error record."""
    error_type = attrs.get("error.type")
    error_message = attrs.get("error.message")
    if not (error_type or error_message):
        return event_name
    prefix = f"{error_type}: " if error_type else ""
    return f"{event_name}: {prefix}{error_message or ''}".rstrip()



def _usage_metrics(usage: Any) -> dict:
    """Normalize a pydantic-ai ``UsageBase`` into a flat token-usage dict."""
    if usage is None:
        return {}

    def _get(name: str) -> int:
        return getattr(usage, name, 0) or 0

    input_tokens = _get("input_tokens")
    output_tokens = _get("output_tokens")
    cache_read = _get("cache_read_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": _get("reasoning_tokens"),
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": _get("cache_write_tokens"),
        "cache_hit_ratio": round(cache_read / input_tokens, 4) if input_tokens else 0.0,
    }


def _cost_metrics(usage: Any, response: Any) -> dict:
    """Best-effort USD cost breakdown for a model response via genai-prices."""
    if usage is None:
        return {}
    result: dict = {}
    total = getattr(usage, "cost", None)
    if total is not None:
        result["total_usd"] = float(total)
    try:
        from pydantic_ai._genai_prices import best_effort_price

        calc = best_effort_price(
            usage,
            model_name=getattr(response, "model_name", None),
            provider_api_url=getattr(response, "provider_url", None),
            provider_name=getattr(response, "provider_name", None),
            genai_request_timestamp=getattr(response, "timestamp", None),
        )
    except Exception:
        calc = None
    if calc is not None:
        result.setdefault("total_usd", float(calc.total_price))
        result["input_usd"] = float(calc.input_price)
        result["output_usd"] = float(calc.output_price)
        result["source"] = "genai-prices"
    elif "total_usd" in result:
        result["source"] = "provider"
    else:
        result["source"] = "unpriced"
    return result


def _model_messages(result: Any) -> list:
    """Return a result's new messages as a list (method or attribute)."""
    try:
        nm = getattr(result, "new_messages", None)
        return list(nm() if callable(nm) else (nm or []))
    except Exception:
        return []


def _attach_tool_latency(turn: dict, request_msg: Any) -> None:
    """Record tool names/count and elapsed tool time for a turn."""
    try:
        from pydantic_ai.messages import ToolReturnPart
    except Exception:  # pragma: no cover - pydantic-ai always available
        return
    returns = [
        p for p in getattr(request_msg, "parts", [])
        if isinstance(p, ToolReturnPart)
    ]
    if not returns:
        return
    stamps = [
        p.timestamp for p in returns if getattr(p, "timestamp", None) is not None
    ]
    response_ts = turn.get("response_ts")
    if stamps and response_ts:
        turn["tool_seconds"] = max(0.0, (max(stamps) - response_ts).total_seconds())
    turn["tool_call_count"] = len(returns)
    turn["tool_names"] = [getattr(p, "tool_name", None) for p in returns]


def _extract_turns(result: Any, detect_phase) -> list[dict]:
    """Extract per-turn metrics from a run result's new messages.

    A "turn" is one model-iteration: the ``ModelResponse`` and any tool returns
    that follow it before the next request.
    """
    try:
        from pydantic_ai.messages import ModelRequest, ModelResponse
    except Exception:  # pragma: no cover - pydantic-ai always available
        return []

    turns: list[dict] = []
    last_request_ts = None
    current: dict | None = None

    for msg in _model_messages(result):
        if isinstance(msg, ModelRequest):
            last_request_ts = getattr(msg, "timestamp", None)
            if current is not None:
                _attach_tool_latency(current, msg)
        elif isinstance(msg, ModelResponse):
            response_ts = getattr(msg, "timestamp", None)
            model_seconds = None
            if response_ts and last_request_ts:
                model_seconds = max(
                    0.0, (response_ts - last_request_ts).total_seconds()
                )
            current = {
                "response": msg,
                "response_ts": response_ts,
                "usage": getattr(msg, "usage", None),
                "model_seconds": model_seconds,
                "tool_seconds": 0.0,
                "tool_call_count": 0,
                "tool_names": [],
                "model_name": getattr(msg, "model_name", None),
                "provider_name": getattr(msg, "provider_name", None),
            }
            turns.append(current)

    total = len(turns)
    for index, turn in enumerate(turns):
        turn["index"] = index + 1
        turn["phase"] = detect_phase(index, total)
    return turns



class Observability:
    """
    Unified observability combining logging, tracing, and metrics.

    Accepts multiple loggers, tracers, and metrics backends.
    Each is called in sequence, enabling multi-destination observability
    with single-responsibility components.

    Example:
        obs = Observability(
            loggers=[OTELLogger(...)],
            tracers=[OTELTracer(...)],
            metrics=[OTELMetrics(...)],
        )

    Or via builder injection (recommended):
        obs = Observability(
            builder=ObservabilityBuilder("agent").with_otel_observability()
        )
    """

    @classmethod
    def configure(cls, service_name: str = "agent", **kwargs) -> "Observability":
        """Create the canonical OTLP-backed observability stack."""
        if "endpoint" in kwargs and "otlp_endpoint" not in kwargs:
            kwargs["otlp_endpoint"] = kwargs.pop("endpoint")
        return ObservabilityBuilder(service_name=service_name).with_otel_observability(
            **kwargs
        ).build()

    def __init__(
        self,
        logger: Optional[Logger] = None,
        tracer: Optional[Tracer] = None,
        metrics: Optional[MetricsCollector] = None,
        service_name: str = "agent",
        loggers: Optional[list[Logger]] = None,
        tracers: Optional[list[Tracer]] = None,
        metrics_list: Optional[list[MetricsCollector]] = None,
        traceback_frame_limit: Optional[int] = None,
        builder: Optional["ObservabilityBuilder"] = None,
        granularity: Optional[str] = None,
    ):
        """
        Initialize observability with pluggable backends.

        Args:
            logger: Single structured logging backend (for convenience)
            tracer: Single tracing backend (for convenience)
            metrics: Single metrics backend (for convenience)
            service_name: Service name for all observability data
            loggers: Multiple logging backends
            tracers: Multiple tracing backends
            metrics_list: Multiple metrics backends
            traceback_frame_limit: Max traceback frames (None = full)
            builder: ObservabilityBuilder to pull backends from directly
            granularity: Telemetry granularity (``minimal``/``standard``/``verbose``);
                defaults to ``HARNESS_TELEMETRY_LEVEL``.
        """
        if builder:
            self.service_name = builder.service_name
            self._loggers: list[Logger] = list(builder._loggers)
            self._tracers: list[Tracer] = list(builder._tracers)
            self._metrics: list[MetricsCollector] = list(builder._metrics)
            self._runtime = builder._runtime
            if granularity is None:
                granularity = builder.granularity
        else:
            self.service_name = service_name
            # Build lists from single or multiple args
            self._loggers = loggers or []
            if logger:
                self._loggers.append(logger)
            self._tracers = tracers or []
            if tracer:
                self._tracers.append(tracer)
            self._metrics = metrics_list or []
            if metrics:
                self._metrics.append(metrics)
            self._runtime = None

        # Telemetry granularity drives what is emitted across logs/metrics/traces.
        self.granularity = TelemetryGranularity(granularity)
        if self._runtime is None:
            self._runtime = TelemetryRuntime(
                self.service_name,
                environment=HARNESS_SETTINGS.app_env,
                host=socket.gethostname(),
                telemetry_level=self.granularity.level,
            )

        # Priority: passed arg > env var > None (full)
        self.traceback_frame_limit = (
            traceback_frame_limit
            if traceback_frame_limit is not None
            else HARNESS_SETTINGS.traceback_frame_limit
        )

        self._apply_otel_defaults()

        # Base context injected into every log entry. Deployment-wide facts
        # (service, environment, host) live on the OTel Resource instead of
        # being repeated on every record. Harness-owned logs default to the
        # ``agent`` component; subsystems override it per call.
        self._base_context: dict = {"component": "agent"}

    def _apply_otel_defaults(self) -> None:
        """Fill empty backend lists with the default OTel backends."""
        if not HARNESS_SETTINGS.telemetry_enabled:
            if not self._loggers:
                self._loggers = [NoOpLogger()]
            if not self._tracers:
                self._tracers = [NoOpTracer()]
            if not self._metrics:
                self._metrics = [NoOpMetrics()]
            return
        if not self._loggers:
            self._loggers = [
                OTELLogger(
                    service_name=self.service_name,
                    environment=HARNESS_SETTINGS.app_env,
                    host=socket.gethostname(),
                    console=HARNESS_SETTINGS.telemetry_console,
                    telemetry_level=self.granularity.level,
                    runtime=self._runtime,
                )
            ]
        if not self._tracers:
            self._tracers = [
                OTELTracer(
                    service_name=self.service_name,
                    telemetry_level=self.granularity.level,
                    runtime=self._runtime,
                )
            ]
        # Metrics are not exported at the minimal level (low-noise mode).
        if not self._metrics and not self.granularity.minimal:
            self._metrics = [
                OTELMetrics(
                    service_name=self.service_name,
                    telemetry_level=self.granularity.level,
                    runtime=self._runtime,
                )
            ]

    # Convenience properties — delegate to first backend
    @property
    def logger(self) -> Logger:
        return self._loggers[0]

    @property
    def tracer(self) -> Tracer:
        return self._tracers[0]

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics[0]

    def debug(self, message: str, **context) -> None:
        for lg in self._loggers:
            lg.debug(message, **context)

    def info(self, message: str, **context) -> None:
        for lg in self._loggers:
            lg.info(message, **context)

    def warning(self, message: str, **context) -> None:
        for lg in self._loggers:
            lg.warning(message, **context)

    def error(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **context,
    ) -> None:
        attrs = _error_attrs(exception, context, self.traceback_frame_limit)
        # Reserved key: used for attribution, never emitted as an attribute.
        context.pop("_error_callsite", None)
        context = {**context, **attrs}
        self._set_span_error_attributes(attrs, exception)
        body = _error_body(message, attrs)
        for lg in self._loggers:
            lg.error(body, event_name=message, **context)

    def _set_span_error_attributes(
        self, attrs: dict, exception: Optional[BaseException] = None
    ) -> None:
        """Write the canonical error attrs onto the current recording span.

        Called from error logging so PydanticAI's ``chat``/``invoke_agent``
        spans (which are current when the ``_span_logs`` hooks log) carry the
        same fields as the log record -- keeping logs and traces consistent.
        """
        if not attrs:
            return
        try:
            from opentelemetry import trace as _otel_trace

            span = _otel_trace.get_current_span()
            if span is None or not span.is_recording():
                return
            for key, value in attrs.items():
                span.set_attribute(key, value)
        except Exception:
            pass


    @asynccontextmanager
    async def observe(self, operation: str, **context):
        """
        Observe an operation with logging, tracing, and metrics.

        Fires all loggers, all tracers, and all metrics backends.
        Structlog contextvars are bound/unbound automatically so any
        structlog call within the span inherits the enrichment keys.
        """
        import structlog as _structlog

        context = execution_context(context)
        context.setdefault(TelemetryFields.OPERATION, operation)
        _structlog.contextvars.bind_contextvars(**context)
        start_time = datetime.now()
        try:
            self._emit_lifecycle("debug", "started", operation, context)
            self._record_lifecycle_counter(operation, context)

            async with self._chain_tracers(operation, **context) as trace_contexts:
                trace_context = self._trace_context(trace_contexts)
                _structlog.contextvars.bind_contextvars(**trace_context)
                try:
                    yield {
                        **context,
                        **trace_context,
                        "tool_call": context.get(
                            "tool_call", {"tool": None, "parameters": {}}
                        ),
                    }
                    duration = (datetime.now() - start_time).total_seconds()
                    self._emit_lifecycle(
                        "info",
                        "completed",
                        operation,
                        context,
                        trace_context,
                        performance={"duration_seconds": duration},
                    )
                    self._record_lifecycle_duration(operation, duration, context)
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    error_attrs = _exception_record(e, self.traceback_frame_limit)
                    self._emit_lifecycle(
                        "error",
                        "failed",
                        operation,
                        context,
                        trace_context,
                        body=_error_body(f"{operation}_failed", error_attrs),
                        performance={"duration_seconds": duration},
                        **error_attrs,
                    )
                    self._record_lifecycle_error(operation, duration, e)
                    raise
        finally:
            _structlog.contextvars.clear_contextvars()

    def _emit_lifecycle(
        self,
        level: str,
        suffix: str,
        operation: str,
        context: dict,
        trace_context: Optional[dict] = None,
        body: Optional[str] = None,
        **extra,
    ) -> None:
        """Emit ``<operation>_<suffix>`` on every logger.

        ``body`` overrides the log message while ``event_name`` stays
        ``<operation>_<suffix>`` (used for human-readable failure messages).
        """
        event_name = f"{operation}_{suffix}"
        merged = {**self._base_context, **context, **(trace_context or {}), **extra}
        message = body if body is not None else event_name
        if body is not None:
            merged["event_name"] = event_name
        for logger in self._loggers:
            getattr(logger, level)(message, **merged)

    def _record_lifecycle_counter(self, operation: str, context: dict) -> None:
        name = (
            MetricNames.AGENT_RUNS
            if operation == "agent_run"
            else f"{operation}_total"
        )
        labels = {
            k: str(v)
            for k, v in execution_context(context).items()
            if k in (TelemetryFields.MODEL, TelemetryFields.PROVIDER)
        }
        for m in self._metrics:
            m.counter(name, **labels)

    def _record_lifecycle_duration(
        self, operation: str, duration: float, context: dict
    ) -> None:
        name = (
            MetricNames.AGENT_DURATION
            if operation == "agent_run"
            else f"{operation}_duration_seconds"
        )
        labels = {
            k: str(v)
            for k, v in execution_context(context).items()
            if k in (TelemetryFields.MODEL, TelemetryFields.PROVIDER, TelemetryFields.STATUS)
        }
        for m in self._metrics:
            m.histogram(name, duration, **labels)

    def _record_lifecycle_error(
        self, operation: str, duration: float, e: BaseException
    ) -> None:
        for m in self._metrics:
            m.counter(
                MetricNames.AGENT_ERRORS,
                **{
                    "error.type": type(e).__name__,
                    "error.source": getattr(e, "_error_source", "unknown"),
                    "error.handled": False,
                    "operation": operation,
                },
            )
            m.histogram(f"{operation}_duration_seconds", duration, status="error")

    @staticmethod
    def _trace_context(trace_contexts: list) -> dict:
        """Extract trace_id/span_id from the primary tracer's span context."""
        if not trace_contexts:
            return {}
        primary = trace_contexts[0]
        if not primary:
            return {}
        try:
            ctx = primary if hasattr(primary, "trace_id") else primary.context
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
        except (AttributeError, TypeError):
            return {}

    @asynccontextmanager
    async def _chain_tracers(self, operation: str, **context):
        """Run all tracers in sequence, collecting their span contexts."""
        span_contexts = []
        active_spans = []

        for t in self._tracers:
            cm = t.span(operation, **context)
            span = await cm.__aenter__()
            active_spans.append((cm, span))
            span_contexts.append(span)

        try:
            yield span_contexts
            for cm, _ in active_spans:
                await cm.__aexit__(None, None, None)
        except Exception as e:
            for cm, _ in active_spans:
                await cm.__aexit__(type(e), e, e.__traceback__)
            raise

    def log_debug(self, message: str, **context):
        enriched = {**self._base_context, **execution_context(context)}
        for lg in self._loggers:
            lg.debug(message, **enriched)

    def log_info(self, message: str, **context):
        enriched = {**self._base_context, **execution_context(context)}
        for lg in self._loggers:
            lg.info(message, **enriched)

    def log_warning(self, message: str, **context):
        enriched = {**self._base_context, **execution_context(context)}
        for lg in self._loggers:
            lg.warning(message, **enriched)

    def log_error(self, message: str, exception: Optional[BaseException] = None, **context):
        attrs = _error_attrs(exception, context, self.traceback_frame_limit)
        context.pop("_error_callsite", None)
        enriched = {**self._base_context, **context, **attrs}
        self._set_span_error_attributes(attrs, exception)
        body = _error_body(message, attrs)
        for lg in self._loggers:
            lg.error(body, event_name=message, **enriched)

    def collect_turns(self, result: Any) -> dict:
        """Extract per-turn token/cost/latency metrics and aggregate them."""
        turns = _extract_turns(result, self._detect_phase)
        summary = {
            "turn_count": len(turns),
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "cost": {"total_usd": 0.0, "input_usd": 0.0, "output_usd": 0.0},
            "latency": {"model_seconds": 0.0, "tool_seconds": 0.0, "total_seconds": 0.0},
            "tool_call_count": 0,
            "turns": [],
        }
        for turn in turns:
            tokens = _usage_metrics(turn["usage"])
            cost = _cost_metrics(turn["usage"], turn["response"])
            latency = {
                "model_seconds": turn["model_seconds"] or 0.0,
                "tool_seconds": turn["tool_seconds"] or 0.0,
            }
            latency["total_seconds"] = latency["model_seconds"] + latency["tool_seconds"]
            entry = {
                "index": turn["index"],
                "phase": turn["phase"],
                "token_usage": tokens,
                "cost": cost,
                "latency": latency,
                "tool_call_count": turn["tool_call_count"],
                "tool_names": turn["tool_names"],
                "model_name": turn["model_name"],
            }
            summary["turns"].append(entry)
            for key in summary["token_usage"]:
                summary["token_usage"][key] += tokens.get(key, 0)
            for key in summary["cost"]:
                summary["cost"][key] += cost.get(key, 0.0)
            for key in summary["latency"]:
                summary["latency"][key] += latency.get(key, 0.0)
            summary["tool_call_count"] += turn["tool_call_count"]
        return summary

    def log_turns(self, result: Any, context: dict) -> dict:
        """Log one ``agent_turn`` per model iteration; return the run summary.

        Per-turn records are emitted at the ``standard`` granularity and above.
        The returned aggregate is used to build the run-level ``run_summary``
        at every granularity.
        """
        summary = self.collect_turns(result)
        if self.granularity.standard:
            for turn in summary["turns"]:
                ctx = {**self._base_context, **context}
                ctx.pop("cumulative_usage", None)
                ctx["model"] = turn.get("model_name") or ctx.get("model")
                self.log_info(
                    "agent_turn",
                    turn={
                        "index": turn["index"],
                        "phase": turn["phase"],
                        "tool_names": turn["tool_names"],
                        "tool_call_count": turn["tool_call_count"],
                    },
                    token_usage=turn["token_usage"],
                    cost=turn["cost"],
                    latency=turn["latency"],
                    **ctx,
                )
                self.record_metric(
                    "histogram",
                    "gen_ai.client.operation.duration",
                    turn["latency"]["total_seconds"],
                    operation="chat",
                    phase=turn["phase"],
                    model=str(turn.get("model_name") or context.get("model") or ""),
                )
        return summary

    def log_run_summary(
        self,
        summary: dict,
        context: dict,
        latency_breakdown: Optional[dict] = None,
        status: str = "success",
    ) -> None:
        """Emit the per-run ``run_summary`` aggregate."""
        turn_count = summary.get("turn_count", 0) or 0
        latency = {**summary.get("latency", {}), **(latency_breakdown or {})}
        run = {
            "turn_count": turn_count,
            "request_count": turn_count,
            "tool_call_count": summary.get("tool_call_count", 0),
            "status": status,
        }
        if turn_count:
            run["avg_turn_latency_seconds"] = round(
                summary["latency"]["total_seconds"] / turn_count, 6
            )
            run["avg_turn_cost_usd"] = round(
                summary["cost"]["total_usd"] / turn_count, 8
            )
            run["max_turn_latency_seconds"] = round(
                max(
                    (t["latency"]["total_seconds"] for t in summary.get("turns", [])),
                    default=0.0,
                ),
                6,
            )
        ctx = {**self._base_context, **context}
        ctx.pop("cumulative_usage", None)
        self.log_info(
            "run_summary",
            run=run,
            token_usage=summary.get("token_usage", {}),
            cost=summary.get("cost", {}),
            latency=latency,
            **ctx,
        )
        if latency.get("total_seconds") is not None:
            self.record_metric(
                "histogram",
                "gen_ai.client.operation.duration",
                float(latency.get("total_seconds") or 0.0),
                operation="invoke_agent",
                model=str(context.get("model") or ""),
            )

    @staticmethod
    def _detect_phase(turn_index: int, total_turns: int) -> str:
        """Determine the phase of a turn."""
        if turn_index == 0 and total_turns > 1:
            return "tool_decision"
        elif turn_index == total_turns - 1:
            return "final_response"
        return "intermediate"

    def record_metric(
        self, metric_type: str, name: str, value: Union[float, int], **labels
    ):
        labels = bounded_metric_attributes(labels)
        for m in self._metrics:
            if metric_type == "counter":
                m.counter(name, int(value), **labels)
            elif metric_type == "gauge":
                m.gauge(name, float(value), **labels)
            elif metric_type == "histogram":
                m.histogram(name, float(value), **labels)
            elif metric_type == "summary":
                m.summary(name, float(value), **labels)

    def record_error(
        self,
        exception: BaseException,
        *,
        source: str = "unknown",
        handled: bool = False,
        **context,
    ) -> dict:
        """Record one correlated error across logs, spans, and metrics."""
        if not getattr(exception, "_error_source", None):
            exception._error_source = source
        context = {
            **context,
            "error_source": source,
            "error_handled": handled,
        }
        attrs = _error_attrs(exception, context, self.traceback_frame_limit)
        self._set_span_error_attributes(attrs, exception)
        self.log_error(_error_body("agent.error", attrs), exception=exception, **attrs)
        self.record_metric(
            "counter",
            MetricNames.AGENT_ERRORS,
            1,
            operation=context.get("operation.name", "agent.run"),
            **attrs,
        )
        return attrs

    def flush(self) -> None:
        """Flush all three signals through the shared runtime."""
        self._runtime.flush()

    async def shutdown(self) -> None:
        """Flush and shut down all providers exactly once."""
        self._runtime.shutdown()

    async def __aenter__(self) -> "Observability":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    def add_span_event(self, name: str, **attributes):
        for t in self._tracers:
            if hasattr(t, "add_event"):
                t.add_event(name, **attributes)

    def set_span_attribute(self, key: str, value: any):
        for t in self._tracers:
            if hasattr(t, "set_attribute"):
                t.set_attribute(key, value)


class ObservabilityBuilder:
    """Fluent builder for observability configuration.

    Provides one convenience method for the supported stack:

    - ``with_otel_observability()`` — OpenTelemetry (logging + tracing + metrics)

    All parameters are optional with sensible defaults; pass only what you
    need to override.
    """

    def __init__(self, service_name: str = "agent", granularity: Optional[str] = None):
        self.service_name = service_name
        self.granularity = granularity
        self._loggers: list[Logger] = []
        self._tracers: list[Tracer] = []
        self._metrics: list[MetricsCollector] = []
        self._runtime: TelemetryRuntime | None = None

    def with_otel_observability(
        self,
        otlp_endpoint: str = "localhost:4317",
        sample_rate: float = 1.0,
        create_spans: bool = False,
        record_failures: bool = True,
        headers: Optional[dict[str, str]] = None,
        export_interval_ms: int = 5000,
        flush_on_exit: bool = True,
        shutdown_on_exit: bool = True,
        granularity: Optional[str] = None,
        console: Optional[bool] = None,
        environment: Optional[str] = None,
        host: Optional[str] = None,
    ) -> "ObservabilityBuilder":
        """Add complete OpenTelemetry observability (logging + tracing + metrics).

        All signals are exported via OTLP gRPC to the same collector endpoint.

        Args:
            otlp_endpoint: OTel Collector OTLP gRPC endpoint
            sample_rate: Trace sampling ratio (0.0–1.0, default 1.0)
            create_spans: When True, export harness-owned spans in addition to
                PydanticAI native spans (default False)
            record_failures: Record exceptions in the trace stream (default True)
            headers: Optional gRPC metadata headers for authenticated endpoints
                (e.g. ``{"Authorization": "Bearer <token>"}``).  The standard
                ``OTEL_EXPORTER_OTLP_HEADERS`` env var is also supported by the
                SDK automatically.
            export_interval_ms: BatchSpanProcessor export interval in
                milliseconds (default 5000). Lower values make traces appear
                in the backend faster during development.
            flush_on_exit: Register atexit handlers that call ``force_flush()``
                on all OTEL providers before exit (default True). Ensures
                buffered telemetry is sent even for short-lived scripts.
            shutdown_on_exit: Register atexit handlers that call ``shutdown()``
                on all OTEL providers (default True). Implies ``flush_on_exit``.
            granularity: Telemetry granularity (``minimal``/``standard``/``verbose``).
                Defaults to ``HARNESS_TELEMETRY_LEVEL``.
            console: Render logs/traces/metrics to the local console via the
                OTel console exporters. Defaults to ``HARNESS_TELEMETRY_CONSOLE``.

        Returns:
            Self for chaining
        """
        from .logging import OTELLogger
        from .tracing import OTELTracer
        from .metrics import OTELMetrics

        if not HARNESS_SETTINGS.telemetry_enabled:
            self._loggers.append(NoOpLogger())
            self._tracers.append(NoOpTracer())
            self._metrics.append(NoOpMetrics())
            return self

        if granularity is not None:
            self.granularity = granularity
        resolved = TelemetryGranularity(self.granularity)
        if console is None:
            console = HARNESS_SETTINGS.telemetry_console
        self._runtime = TelemetryRuntime(
            self.service_name,
            environment=environment or HARNESS_SETTINGS.app_env,
            host=host or socket.gethostname(),
            telemetry_level=resolved.level,
            flush_on_exit=flush_on_exit,
            shutdown_on_exit=shutdown_on_exit,
        )

        self._loggers.append(
            OTELLogger(
                service_name=self.service_name,
                otlp_endpoint=otlp_endpoint,
                flush_on_exit=flush_on_exit,
                shutdown_on_exit=shutdown_on_exit,
                environment=HARNESS_SETTINGS.app_env,
                host=socket.gethostname(),
                console=console,
                telemetry_level=resolved.level,
                headers=headers,
                runtime=self._runtime,
            )
        )
        self._tracers.append(
            OTELTracer(
                service_name=self.service_name,
                otlp_endpoint=otlp_endpoint,
                sample_rate=sample_rate,
                create_spans=create_spans,
                record_failures=record_failures,
                export_interval_ms=export_interval_ms,
                flush_on_exit=flush_on_exit,
                shutdown_on_exit=shutdown_on_exit,
                telemetry_level=resolved.level,
                console=console,
                headers=headers,
                runtime=self._runtime,
            )
        )
        self._metrics.append(
            OTELMetrics(
                service_name=self.service_name,
                otlp_endpoint=otlp_endpoint,
                flush_on_exit=flush_on_exit,
                shutdown_on_exit=shutdown_on_exit,
                telemetry_level=resolved.level,
                console=console,
                headers=headers,
                runtime=self._runtime,
            )
        )
        return self

    def build(self) -> Observability:
        return Observability(builder=self)
