"""Unified observability facade combining logging, tracing, and metrics."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Union

from .logging import Logger, ConsoleLogger, LogfireLogger
from .tracing import Tracer, LogfireTracer, NoOpTracer
from .metrics import MetricsCollector, NoOpMetrics, MetricNames, LogfireMetrics


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
            metrics=[InMemoryMetrics(), OTLPMetrics(...)],
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
        """
        self.service_name = service_name

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

    def error(self, message: str, **context) -> None:
        for lg in self._loggers:
            lg.error(message, **context)

    @asynccontextmanager
    async def observe(self, operation: str, **context):
        """
        Observe an operation with logging, tracing, and metrics.

        Fires all loggers, all tracers, and all metrics backends.
        """
        start_time = datetime.now()

        # Log start on all loggers
        for lg in self._loggers:
            lg.info(f"{operation}_started", **context)

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
                            trace_context = {
                                "trace_id": format(
                                    primary_ctx.context.trace_id, "032x"
                                ),
                                "span_id": format(primary_ctx.context.span_id, "016x"),
                            }
                        except (AttributeError, TypeError):
                            pass

                yield {**context, **trace_context}

                duration = (datetime.now() - start_time).total_seconds()

                for lg in self._loggers:
                    lg.info(
                        f"{operation}_completed",
                        duration_seconds=duration,
                        **context,
                        **trace_context,
                    )

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
                    lg.error(
                        f"{operation}_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        duration_seconds=duration,
                        **context,
                    )

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
        for lg in self._loggers:
            lg.debug(message, **context)

    def log_info(self, message: str, **context):
        for lg in self._loggers:
            lg.info(message, **context)

    def log_warning(self, message: str, **context):
        for lg in self._loggers:
            lg.warning(message, **context)

    def log_error(self, message: str, **context):
        for lg in self._loggers:
            lg.error(message, **context)

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
        self, otlp_endpoint: str = "localhost:4317", sample_rate: float = 1.0
    ) -> "ObservabilityBuilder":
        from .tracing import OTELTracer

        self._tracers.append(
            OTELTracer(
                service_name=self.service_name,
                otlp_endpoint=otlp_endpoint,
                sample_rate=sample_rate,
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
