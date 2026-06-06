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

"""Tracing & Metrics — InMemoryTracer, InMemoryMetrics, span/measurement inspection.

Demonstrates:
  - InMemoryTracer: records spans to a list for inspection via get_spans()/reset()
  - InMemoryMetrics: counters, histograms, gauges stored in dicts via get_metrics()/reset()
  - MetricNames constants: AGENT_RUNS, AGENT_DURATION, AGENT_ERRORS, etc.

Usage:
    uv run python 02_tracing_metrics.py

Setup
-----
    1. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/02_tracing_metrics.py
"""

import asyncio

from agent_harness.tracing import InMemoryTracer
from agent_harness.metrics import InMemoryMetrics, MetricNames


async def main():
    print("=" * 60)
    print("Tracing & Metrics — InMemory Backends")
    print("=" * 60)

    # ── InMemoryTracer: Span recording ──────────────────────────
    print("\n--- InMemoryTracer: Span Recording ---")

    tracer = InMemoryTracer()
    print(f"  Initial spans: {len(tracer.get_spans())}")

    # Create spans with attributes
    async with tracer.span("agent_run", model="gpt-4o", session_id="s1") as span:
        span["extra"] = "additional context"
        async with tracer.span("tool_call", tool="search", query="python patterns") as inner:
            inner["result"] = "found 42 docs"

    async with tracer.span("evaluation", evaluator="quality_check", score=0.95):
        pass

    spans = tracer.get_spans()
    print(f"  Spans recorded: {len(spans)}")
    for s in spans:
        print(f"    - {s['name']}: {s['attributes']}")

    # ── InMemoryTracer: reset ───────────────────────────────────
    print(f"\n  Before reset: {len(tracer.get_spans())} spans")
    tracer.reset()
    print(f"  After reset:  {len(tracer.get_spans())} spans")

    # ── InMemoryMetrics: Counters ───────────────────────────────
    print("\n--- InMemoryMetrics: Counters ---")

    metrics = InMemoryMetrics()
    metrics.counter(MetricNames.AGENT_RUNS, model="gpt-4o", session_id="s1")
    metrics.counter(MetricNames.AGENT_RUNS, model="gpt-4o", session_id="s1")
    metrics.counter(MetricNames.AGENT_RUNS, model="claude-3", session_id="s2")
    metrics.counter(MetricNames.AGENT_ERRORS, error_type="TimeoutError")
    metrics.counter(MetricNames.AGENT_ERRORS, error_type="ValidationError")

    # ── InMemoryMetrics: Histograms ─────────────────────────────
    print("\n--- InMemoryMetrics: Histograms ---")
    metrics.histogram(MetricNames.AGENT_DURATION, 1.2, model="gpt-4o", status="success")
    metrics.histogram(MetricNames.AGENT_DURATION, 2.5, model="gpt-4o", status="success")
    metrics.histogram(MetricNames.AGENT_DURATION, 0.8, model="claude-3", status="error")
    metrics.histogram(MetricNames.AGENT_DURATION, 3.1, model="gpt-4o", status="success")

    # ── InMemoryMetrics: Gauges ─────────────────────────────────
    print("\n--- InMemoryMetrics: Gauges ---")
    metrics.gauge(MetricNames.ACTIVE_SESSIONS, 5)
    metrics.gauge(MetricNames.MEMORY_SIZE, 1024 * 1024 * 128, component="short_term")
    metrics.gauge(MetricNames.ACTIVE_SESSIONS, 3)

    # ── Inspect metrics ─────────────────────────────────────────
    print("\n--- Inspect metrics ---")
    all_metrics = metrics.get_metrics()

    print("  Counters:")
    for key, value in all_metrics["counters"].items():
        print(f"    {key} = {value}")

    print("  Gauges:")
    for key, value in all_metrics["gauges"].items():
        print(f"    {key} = {value}")

    print("  Histograms:")
    for key, values in all_metrics["histograms"].items():
        avg = sum(values) / len(values) if values else 0
        print(f"    {key} = {values} (avg: {avg:.2f})")

    # ── Reset metrics ───────────────────────────────────────────
    print(f"\n  Before reset: {len(metrics.get_metrics()['counters'])} counter keys")
    metrics.reset()
    print(f"  After reset:  {len(metrics.get_metrics()['counters'])} counter keys")

    # ── MetricNames reference ───────────────────────────────────
    print("\n--- MetricNames Constants ---")
    for attr in sorted(dir(MetricNames)):
        if not attr.startswith("_"):
            print(f"  MetricNames.{attr} = '{getattr(MetricNames, attr)}'")


if __name__ == "__main__":
    asyncio.run(main())
