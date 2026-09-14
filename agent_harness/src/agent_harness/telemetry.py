"""Explicit application-owned telemetry setup helpers."""

from __future__ import annotations

import os
import socket
from typing import Optional

from .logging import ConsoleLogger, OTELLogger
from .metrics import NoOpMetrics, OTELMetrics
from .observability import Observability, TelemetryGranularity
from .tracing import NoOpTracer, OTELTracer


def configure_console(
    *,
    stream=None,
    granularity: Optional[str] = None,
) -> Observability:
    """Create a console-only observability composition.

    This configures no OpenTelemetry providers or exporters. Application logs
    remain under application control; the returned logger handles harness
    records only.
    """
    return Observability(
        logger=ConsoleLogger(stream=stream),
        tracer=NoOpTracer(),
        metrics=NoOpMetrics(),
        granularity=granularity,
    )


def configure_otlp(
    *,
    service_name: str = "agent",
    endpoint: str = "localhost:4317",
    headers: Optional[dict[str, str]] = None,
    sample_rate: float = 1.0,
    create_spans: bool = False,
    record_failures: bool = True,
    export_interval_ms: int = 5000,
    granularity: Optional[str] = None,
    console: bool = False,
    environment: Optional[str] = None,
    host: Optional[str] = None,
    resource_attributes: Optional[dict[str, object]] = None,
) -> Observability:
    """Explicitly create and own an OTLP observability composition.

    The application must call this function deliberately. Provider creation,
    global registration, and provider shutdown belong to this explicit setup
    boundary; importing or constructing a ``ManagedAgent`` does none of them.
    """
    from opentelemetry import metrics, trace
    from opentelemetry._logs import get_logger_provider, set_logger_provider
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
        SimpleLogRecordProcessor,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace import ProxyTracerProvider

    attributes: dict[str, object] = {
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
    }
    if environment:
        attributes["deployment.environment"] = environment
    if host:
        attributes["host.name"] = host
    if resource_attributes:
        attributes.update(resource_attributes)
    resource = Resource.create(attributes)

    if not isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
        raise RuntimeError("A global TracerProvider is already configured")
    meter_provider = metrics.get_meter_provider()
    if type(meter_provider).__name__ != "_ProxyMeterProvider":
        raise RuntimeError("A global MeterProvider is already configured")
    if "Proxy" not in type(get_logger_provider()).__name__:
        raise RuntimeError("A global LoggerProvider is already configured")

    tracer_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(sample_rate),
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers or None,
                insecure=True,
            ),
            schedule_delay_millis=export_interval_ms,
        )
    )
    if console:
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )

    metric_readers = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=endpoint,
                headers=headers or None,
                insecure=True,
            ),
            export_interval_millis=export_interval_ms,
        )
    ]
    if console:
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=export_interval_ms,
            )
        )
    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=endpoint,
                headers=headers or None,
                insecure=True,
            )
        )
    )
    if console:
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(ConsoleLogExporter())
        )

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    set_logger_provider(logger_provider)

    resolved = TelemetryGranularity(granularity)
    return Observability(
        logger=OTELLogger(logger_provider, service_name, resolved.level),
        tracer=OTELTracer(
            tracer_provider,
            service_name,
            create_spans=create_spans,
            record_failures=record_failures,
        ),
        metrics=OTELMetrics(meter_provider, service_name),
        service_name=service_name,
        granularity=resolved.level,
    ).own_providers(tracer_provider, meter_provider, logger_provider)
