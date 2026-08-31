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
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Union

from pydantic_settings import BaseSettings
from dotenv import find_dotenv

from .logging import Logger, ConsoleLogger, LogfireLogger
from .tracing import Tracer, LogfireTracer, NoOpTracer
from .metrics import MetricsCollector, NoOpMetrics, MetricNames, LogfireMetrics


from pydantic import Field

class HarnessSettings(BaseSettings):
    """Harness environment settings read from .env at module load."""

    app_env: str = "development"
    service_name: str = "agent-harness"
    traceback_frame_limit: Optional[int] = None
    default_traceback_frames: Optional[int] = Field(
        default=None,
        validation_alias="HARNESS_DEFAULT_TRACEBACK_FRAMES",
    )

    class Config:
        env_file = find_dotenv() or ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Read once at module load — changes require restart
HARNESS_SETTINGS = HarnessSettings()


def _truncate_traceback(tb, limit: int):
    """Truncate traceback chain to show at most `limit` frames.

    Args:
        tb: Traceback object (or None)
        limit: Max frames to keep (0 = None, 1 = last frame, etc.)

    Returns:
        Truncated traceback or None
    """
    if tb is None or limit < 0:
        return tb
    if limit == 0:
        return None

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


def _exception_record(e: BaseException, limit: Optional[int] = None) -> dict:
    """Build error.* + raise-site code.* fields for a caught exception.

    Args:
        e: The exception to record
        limit: Max traceback frames to include (None = all)
    """
    formatted = traceback.format_exception(type(e), e, e.__traceback__, limit=limit)
    record = {
        "error.type": type(e).__name__,
        "error.message": str(e),
        "error.stacktrace": "".join(formatted),
    }
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is not None:
        frame = tb.tb_frame
        record["code.file.path"] = os.path.relpath(frame.f_code.co_filename)
        record["code.function"] = frame.f_code.co_name
        record["code.line.number"] = frame.f_lineno
    return record


class Observability:
    """
    Unified observability combining logging, tracing, and metrics.

    Accepts multiple loggers, tracers, and metrics backends.
    Each is called in sequence, enabling multi-destination observability
    with single-responsibility components.

    Example:
        obs = Observability(
            loggers=[ConsoleLogger(), ElasticsearchLogger(...)],
            tracers=[LogfireTracer(...), OTELTracer(...)],
            metrics=[InMemoryMetrics(), OTELMetrics(...)],
        )
    """

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
        """
        self.service_name = service_name
        # Priority: passed arg > env var > None (full)
        self.traceback_frame_limit = (
            traceback_frame_limit
            if traceback_frame_limit is not None
            else HARNESS_SETTINGS.traceback_frame_limit
        )

        # Build lists from single or multiple args
        self._loggers: list[Logger] = loggers or []
        if logger:
            self._loggers.append(logger)
        if not self._loggers:
            self._loggers = [ConsoleLogger()]

        self._tracers: list[Tracer] = tracers or []
        if tracer:
            self._tracers.append(tracer)
        if not self._tracers:
            self._tracers = [NoOpTracer()]

        self._metrics: list[MetricsCollector] = metrics_list or []
        if metrics:
            self._metrics.append(metrics)
        if not self._metrics:
            self._metrics = [NoOpMetrics()]

        # Base context injected into every log entry
        self._base_context = {
            "service": HARNESS_SETTINGS.service_name,
            "environment": HARNESS_SETTINGS.app_env,
            "host": socket.gethostname(),
            "traceback_frame_limit": self.traceback_frame_limit,
        }

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
        if exception is not None:
            context = {**context, **_exception_record(exception, self.traceback_frame_limit)}
        for lg in self._loggers:
            lg.error(message, **context)

    @asynccontextmanager
    async def observe(self, operation: str, **context):
        """
        Observe an operation with logging, tracing, and metrics.

        Fires all loggers, all tracers, and all metrics backends.
        Structlog contextvars are bound/unbound automatically so any
        structlog call within the span inherits the enrichment keys.
        """
        import structlog as _structlog

        _structlog.contextvars.bind_contextvars(**context)
        try:
            start_time = datetime.now()

            # Log start on all loggers
            for lg in self._loggers:
                lg.info(f"{operation}_started", **{**self._base_context, **context})

            # Increment counter on all metrics
            for m in self._metrics:
                m.counter(
                    MetricNames.AGENT_RUNS
                    if operation == "agent_run"
                    else f"{operation}_total",
                    **{
                        k: str(v)
                        for k, v in context.items()
                        if k in ["model", "session_id"]
                    },
                )

            # Chain all tracers
            async with self._chain_tracers(operation, **context) as trace_contexts:
                try:
                    trace_context = {}
                    if trace_contexts:
                        primary_ctx = trace_contexts[0]
                        if primary_ctx:
                            try:
                                if hasattr(primary_ctx, "trace_id"):
                                    ctx = primary_ctx
                                else:
                                    ctx = primary_ctx.context
                                trace_context = {
                                    "trace_id": format(ctx.trace_id, "032x"),
                                    "span_id": format(ctx.span_id, "016x"),
                                }
                            except (AttributeError, TypeError):
                                pass

                    _structlog.contextvars.bind_contextvars(**trace_context)

                    yield {**context, **trace_context, "tool_call": context.get("tool_call", {"tool": None, "parameters": {}})}

                    duration = (datetime.now() - start_time).total_seconds()

                    for lg in self._loggers:
                        merged = {**self._base_context, **context, **trace_context}
                        merged["performance"] = {"duration_seconds": duration}
                        lg.info(f"{operation}_completed", **merged)

                    for m in self._metrics:
                        m.histogram(
                            MetricNames.AGENT_DURATION
                            if operation == "agent_run"
                            else f"{operation}_duration_seconds",
                            duration,
                            **{
                                k: str(v)
                                for k, v in context.items()
                                if k in ["model", "status"]
                            },
                        )

                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()

                    for lg in self._loggers:
                        merged = {**self._base_context, **context, **trace_context}
                        merged["performance"] = {"duration_seconds": duration}
                        merged.update(_exception_record(e, self.traceback_frame_limit))
                        lg.error(f"{operation}_failed", **merged)

                    for m in self._metrics:
                        m.counter(
                            MetricNames.AGENT_ERRORS,
                            error_type=type(e).__name__,
                            operation=operation,
                        )
                        m.histogram(
                            f"{operation}_duration_seconds", duration, status="error"
                        )

                    raise
        finally:
            _structlog.contextvars.clear_contextvars()

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
        enriched = {**self._base_context, **context}
        for lg in self._loggers:
            lg.debug(message, **enriched)

    def log_info(self, message: str, **context):
        enriched = {**self._base_context, **context}
        for lg in self._loggers:
            lg.info(message, **enriched)

    def log_warning(self, message: str, **context):
        enriched = {**self._base_context, **context}
        for lg in self._loggers:
            lg.warning(message, **enriched)

    def log_error(self, message: str, exception: Optional[BaseException] = None, **context):
        if exception is not None:
            context = {**context, **_exception_record(exception, self.traceback_frame_limit)}
        enriched = {**self._base_context, **context}
        for lg in self._loggers:
            lg.error(message, **enriched)

    def log_token_usage(self, result: Any, context: dict) -> None:
        """Extract and log token usage for all internal model requests."""
        from .memory import UsageData, ModelResponse

        usage_list = []

        # Path 1: result.usage.requests (multiple internal calls)
        if hasattr(result, "usage") and result.usage:
            u = result.usage
            if (
                hasattr(u, "requests")
                and isinstance(getattr(u, "requests", None), list)
                and u.requests
            ):
                for i, req in enumerate(u.requests):
                    usage_list.append({
                        "turn": i + 1,
                        "phase": self._detect_phase(i, len(u.requests)),
                        "usage": UsageData(
                            input_tokens=getattr(req, "input_tokens", 0) or 0,
                            output_tokens=getattr(req, "output_tokens", 0) or 0,
                            reasoning_tokens=getattr(req, "reasoning_tokens", 0) or 0,
                            total_tokens=getattr(req, "total_tokens", 0) or 0,
                            prompt_tokens=getattr(req, "input_tokens", 0) or 0,
                            completion_tokens=getattr(req, "output_tokens", 0) or 0,
                        )
                    })
            else:
                # Path 2: result.usage direct access (single call)
                usage_list.append({
                    "turn": 1,
                    "phase": "final_response",
                    "usage": UsageData(
                        input_tokens=getattr(u, "input_tokens", 0) or 0,
                        output_tokens=getattr(u, "output_tokens", 0) or 0,
                        reasoning_tokens=getattr(u, "reasoning_tokens", 0) or 0,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                        prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                        completion_tokens=getattr(u, "output_tokens", 0) or 0,
                    )
                })

        # Log each request separately
        for entry in usage_list:
            ctx = {**context, "turn": entry["turn"], "phase": entry["phase"]}
            # Include cumulative usage if provided in context
            cumulative = context.get("cumulative_usage")
            if cumulative:
                ctx["cumulative_usage"] = cumulative
            self.log_info(
                "token_usage",
                token_usage={
                    "input_tokens": entry["usage"].input_tokens,
                    "output_tokens": entry["usage"].output_tokens,
                    "reasoning_tokens": entry["usage"].reasoning_tokens,
                    "total_tokens": entry["usage"].total_tokens,
                    "prompt_tokens": entry["usage"].prompt_tokens,
                    "completion_tokens": entry["usage"].completion_tokens,
                },
                **ctx,
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
        for m in self._metrics:
            if metric_type == "counter":
                m.counter(name, int(value), **labels)
            elif metric_type == "gauge":
                m.gauge(name, float(value), **labels)
            elif metric_type == "histogram":
                m.histogram(name, float(value), **labels)
            elif metric_type == "summary":
                m.summary(name, float(value), **labels)

    def add_span_event(self, name: str, **attributes):
        for t in self._tracers:
            if hasattr(t, "add_event"):
                t.add_event(name, **attributes)

    def set_span_attribute(self, key: str, value: any):
        for t in self._tracers:
            if hasattr(t, "set_attribute"):
                t.set_attribute(key, value)


class ObservabilityBuilder:
    """Fluent builder for observability configuration."""

    def __init__(self, service_name: str = "agent"):
        self.service_name = service_name
        self._loggers: list[Logger] = []
        self._tracers: list[Tracer] = []
        self._metrics: list[MetricsCollector] = []

    def with_console_logging(self) -> "ObservabilityBuilder":
        from .logging import ConsoleLogger

        self._loggers.append(ConsoleLogger())
        return self

    def with_elasticsearch_logging(
        self, endpoint: str, index_prefix: str = "agent-logs"
    ) -> "ObservabilityBuilder":
        from .logging import ElasticsearchLogger

        self._loggers.append(
            ElasticsearchLogger(
                endpoint=endpoint,
                index_prefix=index_prefix,
                service_name=self.service_name,
            )
        )
        return self

    def with_file_logging(self, log_file: str = "agent.log") -> "ObservabilityBuilder":
        from .logging import FileLogger

        self._loggers.append(FileLogger(log_file=log_file))
        return self

    def with_logfire_logging(self) -> "ObservabilityBuilder":
        """Add Logfire as a logging backend."""
        from .logging import LogfireLogger

        self._loggers.append(LogfireLogger(service_name=self.service_name))
        return self

    def with_otel_logging(
        self, otlp_endpoint: str = "localhost:4317"
    ) -> "ObservabilityBuilder":
        from .logging import OTELLogger

        self._loggers.append(
            OTELLogger(service_name=self.service_name, otlp_endpoint=otlp_endpoint)
        )
        return self

    def with_logfire_tracing(
        self,
        send_to_logfire: bool = True,
        instrument_pydantic_ai: bool = True,
    ) -> "ObservabilityBuilder":
        from .tracing import LogfireTracer

        self._tracers.append(
            LogfireTracer(
                service_name=self.service_name,
                send_to_logfire=send_to_logfire,
                instrument_pydantic_ai=instrument_pydantic_ai,
            )
        )
        return self

    def with_otel_tracing(
        self,
        otlp_endpoint: str = "localhost:4317",
        sample_rate: float = 1.0,
        create_spans: bool = False,
        record_failures: bool = True,
    ) -> "ObservabilityBuilder":
        from .tracing import OTELTracer

        self._tracers.append(
            OTELTracer(
                service_name=self.service_name,
                otlp_endpoint=otlp_endpoint,
                sample_rate=sample_rate,
                create_spans=create_spans,
                record_failures=record_failures,
            )
        )
        return self

    def with_jaeger_tracing(
        self, jaeger_host: str = "localhost", jaeger_port: int = 6831
    ) -> "ObservabilityBuilder":
        from .tracing import JaegerTracer

        self._tracers.append(
            JaegerTracer(
                service_name=self.service_name,
                jaeger_host=jaeger_host,
                jaeger_port=jaeger_port,
            )
        )
        return self

    def with_prometheus_metrics(
        self, push_gateway: Optional[str] = None
    ) -> "ObservabilityBuilder":
        from .metrics import PrometheusMetrics

        self._metrics.append(
            PrometheusMetrics(namespace=self.service_name, push_gateway=push_gateway)
        )
        return self

    def with_statsd_metrics(
        self, host: str = "localhost", port: int = 8125
    ) -> "ObservabilityBuilder":
        from .metrics import StatsdMetrics

        self._metrics.append(
            StatsdMetrics(host=host, port=port, prefix=self.service_name)
        )
        return self

    def with_in_memory_metrics(self) -> "ObservabilityBuilder":
        from .metrics import InMemoryMetrics

        self._metrics.append(InMemoryMetrics())
        return self

    def with_logfire_metrics(self) -> "ObservabilityBuilder":
        """Add Logfire as a metrics backend."""
        from .metrics import LogfireMetrics

        self._metrics.append(LogfireMetrics(service_name=self.service_name))
        return self

    def with_logfire_observability(
        self,
        send_to_logfire: bool = True,
        include_tracing: bool = True,
        include_metrics: bool = True,
    ) -> "ObservabilityBuilder":
        """
        Add complete Logfire observability (logging, tracing, metrics).

        Args:
            send_to_logfire: Send to Logfire cloud or local only
            include_tracing: Enable Logfire tracing
            include_metrics: Enable Logfire metrics

        Returns:
            Self for chaining
        """
        # Add Logfire logging (structured logging to Logfire)
        self._loggers.append(LogfireLogger(service_name=self.service_name))

        # Add Logfire tracing
        if include_tracing:
            from .tracing import LogfireTracer

            self._tracers.append(
                LogfireTracer(
                    service_name=self.service_name,
                    send_to_logfire=send_to_logfire,
                )
            )

        # Add Logfire metrics
        if include_metrics:
            from .metrics import LogfireMetrics

            self._metrics.append(LogfireMetrics(service_name=self.service_name))

        return self

    def build(self) -> Observability:
        return Observability(
            service_name=self.service_name,
            loggers=self._loggers or None,
            tracers=self._tracers or None,
            metrics_list=self._metrics or None,
        )
