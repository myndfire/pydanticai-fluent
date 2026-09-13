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

"""Metrics collection and monitoring."""

from typing import Protocol, Any, Optional
from collections import defaultdict
from datetime import datetime


def _is_proxy_provider(provider: Any) -> bool:
    """Return True only if the global provider is still OTEL's unset default proxy.

    OpenTelemetry installs a proxy provider until a real one is registered, and
    rejects any attempt to override an already-registered real provider. We must
    distinguish OTEL's *internal* no-op default (``_ProxyMeterProvider``) from
    other proxies, which are real providers that must not be overridden.
    """
    return type(provider).__name__ == "_ProxyMeterProvider"


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def counter(self, name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        ...

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge metric."""
        ...

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram value."""
        ...

    def summary(self, name: str, value: float, **labels):
        """Record a summary value."""
        ...


class NoOpMetrics:
    """No-op metrics collector (default, for development)."""

    def counter(self, name: str, value: int = 1, **labels):
        """No-op counter."""
        pass

    def gauge(self, name: str, value: float, **labels):
        """No-op gauge."""
        pass

    def histogram(self, name: str, value: float, **labels):
        """No-op histogram."""
        pass

    def summary(self, name: str, value: float, **labels):
        """No-op summary."""
        pass


class InMemoryMetrics:
    """In-memory metrics collector (for development/testing)."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._counters = defaultdict(int)
        self._gauges = {}
        self._histograms = defaultdict(list)
        self._summaries = defaultdict(list)

    def counter(self, name: str, value: int = 1, **labels):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram value."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

    def summary(self, name: str, value: float, **labels):
        """Record a summary value."""
        key = self._make_key(name, labels)
        self._summaries[key].append(value)

    def get_metrics(self) -> dict:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": dict(self._histograms),
            "summaries": dict(self._summaries),
        }

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._summaries.clear()

    def _make_key(self, name: str, labels: dict) -> str:
        """Create a key from metric name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class OTELMetrics:
    """OTLP metrics exporter - sends metrics to an OTel Collector via OTLP gRPC."""

    def __init__(
        self,
        service_name: str = "agent",
        otlp_endpoint: str = "localhost:4317",
        headers: dict[str, str] | None = None,
        runtime: Any = None,
        flush_on_exit: bool = True,
        shutdown_on_exit: bool = True,
        telemetry_level: str = "standard",
        console: bool = False,
    ):
        """
        Initialize OTEL metrics.

        Args:
            service_name: Service name for metrics
            otlp_endpoint: OTel Collector OTLP gRPC endpoint (default: localhost:4317)
            flush_on_exit: Register an atexit handler that calls
                ``force_flush()`` on the MeterProvider before exit (default True).
            shutdown_on_exit: Register an atexit handler that calls
                ``shutdown()`` on the MeterProvider (default True). Implies
                ``flush_on_exit``.
            telemetry_level: Granularity level exported as the
                ``harness.telemetry.level`` resource attribute.
            console: Also render metrics to the local console via the OTel
                ``ConsoleMetricExporter``.
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.headers = headers or {}
        self.runtime = runtime
        self.telemetry_level = telemetry_level
        self.console = console
        self._flush_on_exit = flush_on_exit or shutdown_on_exit
        self._shutdown_on_exit = shutdown_on_exit
        self._meter = None
        self._provider = None
        self._shut_down = False

        self._setup_otlp()

    def _setup_otlp(self):
        """Setup OTLP metrics exporter."""
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            from ._otel import build_resource, register_atexit

            existing = metrics.get_meter_provider()
            if not _is_proxy_provider(existing):
                # Never construct an unused provider/exporter when another
                # application already owns the process-wide meter provider.
                self._meter = metrics.get_meter(self.service_name)
                self._provider = existing
                self._counters = {}
                self._gauges = {}
                self._histograms = {}
                return

            resource = self.runtime.resource if self.runtime else build_resource(
                self.service_name,
                telemetry_level=self.telemetry_level,
                extra={"service.version": "0.1.0"},
            )

            exporter = OTLPMetricExporter(
                endpoint=self.otlp_endpoint,
                headers=self.headers or None,
                insecure=True,
            )
            readers = [
                PeriodicExportingMetricReader(
                    exporter, export_interval_millis=5000
                )
            ]
            if self.console:
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

                readers.append(
                    PeriodicExportingMetricReader(
                        ConsoleMetricExporter(),
                        export_interval_millis=5000,
                    )
                )

            provider = MeterProvider(resource=resource, metric_readers=readers)

            # OpenTelemetry allows only one global MeterProvider per process.
            # Reuse an already-registered provider instead of overriding it
            # (which OTEL rejects with "Overriding of current MeterProvider").
            if _is_proxy_provider(existing):
                metrics.set_meter_provider(provider)
                self._meter = metrics.get_meter(self.service_name)
                self._provider = provider
                print(f"✅ OTLP metrics initialized: {self.otlp_endpoint}")
                if self.runtime:
                    self.runtime.register(provider)
                else:
                    register_atexit(
                        provider,
                        flush_on_exit=self._flush_on_exit,
                        shutdown_on_exit=self._shutdown_on_exit,
                        is_shut_down=lambda: self._shut_down,
                    )

            self._counters = {}
            self._gauges = {}
            self._histograms = {}

        except Exception as e:
            print(f"⚠️  Failed to setup OTLP metrics: {str(e)}")
            self._meter = None

    def counter(self, name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        if not self._meter:
            return

        metric_name = f"{self.service_name}_{name}"
        if metric_name not in self._counters:
            self._counters[metric_name] = self._meter.create_counter(
                metric_name, unit="1", description=f"Counter for {name}"
            )
        self._counters[metric_name].add(value, attributes=labels or None)

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge metric."""
        if not self._meter:
            return

        metric_name = f"{self.service_name}_{name}"
        if metric_name not in self._gauges:
            self._gauges[metric_name] = self._meter.create_gauge(
                metric_name, unit="1", description=f"Gauge for {name}"
            )
        self._gauges[metric_name].set(value, attributes=labels or None)

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram value."""
        if not self._meter:
            return

        # Semantic-convention names (e.g. gen_ai.*) are used verbatim so they
        # align with pydantic-ai's native metrics; other names are namespaced.
        metric_name = name if name.startswith("gen_ai.") else f"{self.service_name}_{name}"
        if metric_name not in self._histograms:
            self._histograms[metric_name] = self._meter.create_histogram(
                metric_name, unit="1", description=f"Histogram for {name}"
            )
        self._histograms[metric_name].record(value, attributes=labels or None)

    def summary(self, name: str, value: float, **labels):
        """Record a summary value (as histogram)."""
        self.histogram(name, value, **labels)

    def shutdown(self):
        """Explicitly flush and shut down the OTLP metrics provider.

        Call this for deterministic cleanup in long-running processes or
        when ``flush_on_exit=False``.
        """
        if self._provider:
            try:
                self._provider.force_flush()
                self._provider.shutdown()
            except Exception:
                pass
            self._provider = None
            self._meter = None
            self._shut_down = True


class MetricNames:
    """Standard metric names for agent operations."""

    # Counters
    AGENT_RUNS = "agent_runs_total"
    AGENT_ERRORS = "agent_errors_total"
    TOOL_CALLS = "tool_calls_total"
    EVALUATIONS = "evaluations_total"
    RETRIES = "retries_total"

    # Gauges
    ACTIVE_SESSIONS = "active_sessions"
    MEMORY_SIZE = "memory_size_bytes"

    # Histograms/Summaries
    AGENT_DURATION = "agent_duration_seconds"
    TOOL_DURATION = "tool_duration_seconds"
    TOKEN_USAGE = "token_usage_total"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    RESPONSE_SIZE = "response_size_bytes"
    REASONING_COST = "reasoning_cost"
