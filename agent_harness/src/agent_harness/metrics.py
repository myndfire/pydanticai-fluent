"""Metrics collection and monitoring."""

from typing import Protocol, Any, Optional
from collections import defaultdict
from datetime import datetime


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


class OTLPMetrics:
    """OTLP metrics exporter - sends metrics to an OTel Collector via OTLP gRPC."""

    def __init__(
        self,
        service_name: str = "agent",
        otlp_endpoint: str = "localhost:4319",
    ):
        """
        Initialize OTLP metrics.

        Args:
            service_name: Service name for metrics
            otlp_endpoint: OTel Collector OTLP gRPC endpoint (default: localhost:4319)
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self._meter = None

        self._setup_otlp()

    def _setup_otlp(self):
        """Setup OTLP metrics exporter."""
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            resource = Resource.create({"service.name": self.service_name})

            exporter = OTLPMetricExporter(endpoint=self.otlp_endpoint, insecure=True)
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=5000
            )

            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)

            self._meter = metrics.get_meter(self.service_name)
            self._counters = {}
            self._gauges = {}
            self._histograms = {}

            print(f"✅ OTLP metrics initialized: {self.otlp_endpoint}")

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

        metric_name = f"{self.service_name}_{name}"
        if metric_name not in self._histograms:
            self._histograms[metric_name] = self._meter.create_histogram(
                metric_name, unit="1", description=f"Histogram for {name}"
            )
        self._histograms[metric_name].record(value, attributes=labels or None)

    def summary(self, name: str, value: float, **labels):
        """Record a summary value (as histogram)."""
        self.histogram(name, value, **labels)


class PrometheusMetrics:
    """Prometheus metrics collector."""

    def __init__(self, namespace: str = "agent", push_gateway: Optional[str] = None):
        """
        Initialize Prometheus metrics.

        Args:
            namespace: Metric namespace
            push_gateway: Prometheus push gateway URL (optional)
        """
        self.namespace = namespace
        self.push_gateway = push_gateway
        self._metrics = {}

        self._setup_prometheus()

    def _setup_prometheus(self):
        """Setup Prometheus client."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary

            self.Counter = Counter
            self.Gauge = Gauge
            self.Histogram = Histogram
            self.Summary = Summary

            print(f"✅ Prometheus metrics initialized (namespace: {self.namespace})")

        except Exception as e:
            print(f"⚠️  Failed to setup Prometheus: {str(e)}")
            self.Counter = None
            self.Gauge = None
            self.Histogram = None
            self.Summary = None

    def counter(self, name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        if not self.Counter:
            return

        metric_name = f"{self.namespace}_{name}"

        if metric_name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[metric_name] = self.Counter(
                metric_name, f"Counter for {name}", label_names
            )

        if labels:
            self._metrics[metric_name].labels(**labels).inc(value)
        else:
            self._metrics[metric_name].inc(value)

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge metric."""
        if not self.Gauge:
            return

        metric_name = f"{self.namespace}_{name}"

        if metric_name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[metric_name] = self.Gauge(
                metric_name, f"Gauge for {name}", label_names
            )

        if labels:
            self._metrics[metric_name].labels(**labels).set(value)
        else:
            self._metrics[metric_name].set(value)

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram value."""
        if not self.Histogram:
            return

        metric_name = f"{self.namespace}_{name}"

        if metric_name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[metric_name] = self.Histogram(
                metric_name, f"Histogram for {name}", label_names
            )

        if labels:
            self._metrics[metric_name].labels(**labels).observe(value)
        else:
            self._metrics[metric_name].observe(value)

    def summary(self, name: str, value: float, **labels):
        """Record a summary value."""
        if not self.Summary:
            return

        metric_name = f"{self.namespace}_{name}"

        if metric_name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[metric_name] = self.Summary(
                metric_name, f"Summary for {name}", label_names
            )

        if labels:
            self._metrics[metric_name].labels(**labels).observe(value)
        else:
            self._metrics[metric_name].observe(value)

    def push_to_gateway(self, job_name: str = "agent"):
        """Push metrics to Prometheus push gateway."""
        if not self.push_gateway:
            return

        try:
            from prometheus_client import push_to_gateway as push

            push(self.push_gateway, job=job_name, registry=None)
        except Exception as e:
            print(f"⚠️  Failed to push to gateway: {str(e)}")


class StatsdMetrics:
    """StatsD metrics collector."""

    def __init__(
        self, host: str = "localhost", port: int = 8125, prefix: str = "agent"
    ):
        """
        Initialize StatsD metrics.

        Args:
            host: StatsD server host
            port: StatsD server port
            prefix: Metric prefix
        """
        self.host = host
        self.port = port
        self.prefix = prefix
        self.client = None

        self._setup_statsd()

    def _setup_statsd(self):
        """Setup StatsD client."""
        try:
            from statsd import StatsClient

            self.client = StatsClient(
                host=self.host, port=self.port, prefix=self.prefix
            )

            print(f"✅ StatsD metrics initialized: {self.host}:{self.port}")

        except Exception as e:
            print(f"⚠️  Failed to setup StatsD: {str(e)}")
            self.client = None

    def counter(self, name: str, value: int = 1, **labels):
        """Increment a counter."""
        if self.client:
            metric_name = self._format_name(name, labels)
            self.client.incr(metric_name, count=value)

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge value."""
        if self.client:
            metric_name = self._format_name(name, labels)
            self.client.gauge(metric_name, value)

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram value (timing in StatsD)."""
        if self.client:
            metric_name = self._format_name(name, labels)
            self.client.timing(metric_name, value)

    def summary(self, name: str, value: float, **labels):
        """Record a summary value (timing in StatsD)."""
        if self.client:
            metric_name = self._format_name(name, labels)
            self.client.timing(metric_name, value)

    def _format_name(self, name: str, labels: dict) -> str:
        """Format metric name with labels."""
        if not labels:
            return name

        # StatsD doesn't support labels natively, so we append them to the name
        label_str = ".".join(f"{k}.{v}" for k, v in sorted(labels.items()))
        return f"{name}.{label_str}"


# Common metric names (constants for consistency)
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
    RESPONSE_SIZE = "response_size_bytes"
