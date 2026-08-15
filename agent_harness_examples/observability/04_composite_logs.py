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

"""Composite — multi-destination fan-out with CompositeLogger and Observability.

Demonstrates:
  - CompositeLogger: fan-out log messages to multiple loggers
  - Observability with multiple loggers, tracers, and metrics backends
  - Constructor-based multi-backend: Observability(loggers=[...], tracers=[...], metrics_list=[...])
  - Convenience properties: .logger, .tracer, .metrics (delegate to first backend)
  - Fan-out in observe() context manager: all backends receive events

Usage:
    uv run python 04_composite_logs.py

Setup
-----
    1. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/04_composite_logs.py
"""

import asyncio

from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger, FileLogger, CompositeLogger
from agent_harness.tracing import InMemoryTracer, NoOpTracer
from agent_harness.metrics import InMemoryMetrics, MetricNames, NoOpMetrics


async def main():
    print("=" * 60)
    print("Composite — Multi-Destination Fan-Out")
    print("=" * 60)

    # ── Example 1: CompositeLogger ──────────────────────────────
    print("\n--- Example 1: CompositeLogger ---")

    console = ConsoleLogger()
    file_log = FileLogger(log_file="composite_example.log")
    composite = CompositeLogger(console, file_log)

    print(f"  CompositeLogger fans out to {len(composite.loggers)} loggers:")
    for lg in composite.loggers:
        print(f"    - {type(lg).__name__}")

    # Single call → both loggers receive it
    composite.info("composite_test", source="example", targets=2)

    # ── Example 2: Observability with multiple loggers ──────────
    print("\n--- Example 2: Observability(loggers=[...]) ---")

    obs = Observability(
        loggers=[
            ConsoleLogger(),
            FileLogger(log_file="multi_logger.log"),
        ],
    )
    print(f"  Loggers: {[type(l).__name__ for l in obs._loggers]}")
    print(f"  .logger (convenience): {type(obs.logger).__name__}")

    # All loggers receive the message
    obs.info("multi_target_log", source="observability", fan_out=True)
    obs.warning("disk_space_low", available_gb=2, threshold_gb=5)
    obs.error("api_key_expired", provider="openai")

    # ── Example 3: Observability with multiple tracers ──────────
    print("\n--- Example 3: Observability(tracers=[...]) ---")

    mem_tracer = InMemoryTracer()

    obs2 = Observability(
        loggers=[ConsoleLogger()],
        tracers=[mem_tracer, NoOpTracer()],
    )
    print(f"  Tracers: {[type(t).__name__ for t in obs2._tracers]}")

    # observe() chains ALL tracers, logs to ALL loggers, emits to ALL metrics
    async with obs2.observe("multi_tracer_test", op="demo", run=1):
        obs2.info("inside_observe", step="setup")
        await asyncio.sleep(0.01)

    spans = mem_tracer.get_spans()
    print(f"  Spans captured by InMemoryTracer: {len(spans)}")
    for s in spans:
        print(f"    - {s['name']}: {s['attributes']}")

    # ── Example 4: Full multi-backend ───────────────────────────
    print("\n--- Example 4: Full multi-backend (loggers + tracers + metrics) ---")

    mem_metrics = InMemoryMetrics()

    obs3 = Observability(
        service_name="multi-backend-demo",
        loggers=[ConsoleLogger(), FileLogger(log_file="full_multi.log")],
        tracers=[InMemoryTracer()],
        metrics_list=[mem_metrics],
    )

    print(f"  Loggers: {[type(l).__name__ for l in obs3._loggers]}")
    print(f"  Tracers: {[type(t).__name__ for t in obs3._tracers]}")
    print(f"  Metrics: {[type(m).__name__ for m in obs3._metrics]}")

    # observe() fans out to ALL backends simultaneously
    async with obs3.observe("full_operation", session_id="full-1", model="test"):
        obs3.info("processing", items=100)
        obs3.metrics.counter("items_processed", count=100)
        await asyncio.sleep(0.01)

    # ── Inspect results ─────────────────────────────────────────
    print("\n--- Results ---")
    spans3 = obs3._tracers[0].get_spans() if hasattr(obs3._tracers[0], "get_spans") else []
    print(f"  Spans: {len(spans3)}")

    if isinstance(obs3.metrics, InMemoryMetrics) or hasattr(obs3._metrics[0], "get_metrics"):
        metrics_data = mem_metrics.get_metrics()
        for key, value in metrics_data["counters"].items():
            print(f"  {key} = {value}")
        for key, values in metrics_data["histograms"].items():
            print(f"  {key} = {values}")

    # ── Example 5: Convenience properties ───────────────────────
    print("\n--- Example 5: Convenience properties ---")
    obs5 = Observability(
        loggers=[ConsoleLogger(), FileLogger(log_file="conv.log")],
        tracers=[InMemoryTracer()],
        metrics_list=[InMemoryMetrics()],
    )
    print(f"  obs.logger  → first logger:  {type(obs5.logger).__name__}")
    print(f"  obs.tracer  → first tracer:  {type(obs5.tracer).__name__}")
    print(f"  obs.metrics → first metrics: {type(obs5.metrics).__name__}")

    # Use convenience properties directly
    obs5.logger.info("via_convenience_property")
    # obs5.tracer.span() would work here too
    obs5.metrics.counter(MetricNames.AGENT_RUNS, model="convenience")


if __name__ == "__main__":
    asyncio.run(main())
