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
        meter_provider: Any | None = None,
        service_name: str = "agent",
    ):
        """
        Initialize OTEL metrics.

        Args:
            meter_provider: Application-owned OTel MeterProvider. ``None``
                disables metrics for this adapter.
            service_name: Instrumentation scope name for metrics.
        """
        self.service_name = service_name
        self._provider = meter_provider
        self._meter = (
            meter_provider.get_meter(service_name)
            if meter_provider is not None
            else None
        )
        self._counters = {}
        self._gauges = {}
        self._histograms = {}

    @property
    def provider(self) -> Any | None:
        """Return the application-owned meter provider, if configured."""
        return self._provider

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
        """Release this adapter without shutting down the app-owned provider."""
        self._provider = None
        self._meter = None


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
