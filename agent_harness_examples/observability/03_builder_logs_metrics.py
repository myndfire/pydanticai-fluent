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

"""ObservabilityBuilder — fluent builder to compose logging, tracing, and metrics.

Demonstrates:
  - ObservabilityBuilder.with_console_logging()
  - ObservabilityBuilder.with_file_logging()
  - ObservabilityBuilder.with_in_memory_metrics()
  - ObservabilityBuilder.build() → Observability
  - with_observability() on ManagedAgent
  - Multiple loggers/tracers/metrics in a single Observability instance

Usage:
    uv run python 03_builder_logs_metrics.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/03_builder_logs_metrics.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.observability import ObservabilityBuilder
from agent_harness.metrics import InMemoryMetrics, MetricNames


async def main():
    print("=" * 60)
    print("ObservabilityBuilder — Fluent Compose & Agent Integration")
    print("=" * 60)

    # ── Example 1: Console + File + InMemoryMetrics ─────────────
    print("\n--- Example 1: Console + File + InMemoryMetrics ---")
    obs = (
        ObservabilityBuilder(service_name="example-agent")
        .with_console_logging()
        .with_file_logging(log_file="builder_example.log")
        .with_in_memory_metrics()
        .build()
    )

    print(f"  Service: {obs.service_name}")
    print(f"  Loggers: {len(obs._loggers)} ({[type(l).__name__ for l in obs._loggers]})")
    print(f"  Tracers: {len(obs._tracers)} ({[type(t).__name__ for t in obs._tracers]})")
    print(f"  Metrics: {len(obs._metrics)} ({[type(m).__name__ for m in obs._metrics]})")

    # Log through the observability facade
    obs.info("builder_test", source="example", builders=1)
    obs.warning("test_warning", threshold=0.9)
    obs.metrics.counter(MetricNames.AGENT_RUNS, model="test")

    # ── Example 2: Builder defaults ─────────────────────────────
    print("\n--- Example 2: Builder defaults (no methods called) ---")
    obs2 = ObservabilityBuilder(service_name="minimal-agent").build()
    print(f"  Loggers: {len(obs2._loggers)} ({[type(l).__name__ for l in obs2._loggers]})")
    print(f"  Tracers: {len(obs2._tracers)} ({[type(t).__name__ for t in obs2._tracers]})")
    print(f"  Metrics: {len(obs2._metrics)} ({[type(m).__name__ for m in obs2._metrics]})")

    # ── Example 3: Agent with observability ─────────────────────
    print("\n--- Example 3: Agent with Observability ---")

    metrics_store = InMemoryMetrics()

    agent_obs = (
        ObservabilityBuilder(service_name="demo-agent")
        .with_console_logging()
        .with_file_logging(log_file="agent_demo.log")
        .build()
    )
    # Add InMemoryMetrics directly (builder doesn't have to do everything)
    agent_obs._metrics.append(metrics_store)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_observability(agent_obs)
    )

    memory = InMemoryProvider()

    # Run the agent — observe() automatically logs start/complete,
    # increments AGENT_RUNS counter, records AGENT_DURATION histogram
    history = await MessageHistory().load("obs-demo", memory)
    result = await agent.run(
        "What is 2+2?",
        history,
        "obs-demo",
        save_to=[memory],
    )
    print(f"  Agent response: {result.output[:80]}...")

    # ── Inspect auto-collected metrics ──────────────────────────
    print("\n--- Auto-collected metrics ---")
    all_m = metrics_store.get_metrics()

    print("  Counters:")
    for key, value in all_m["counters"].items():
        print(f"    {key} = {value}")

    print("  Histograms:")
    for key, values in all_m["histograms"].items():
        print(f"    {key} = {values}")

    # ── Example 4: Chained builder methods ──────────────────────
    print("\n--- Example 4: Chained builder ---")
    obs4 = (
        ObservabilityBuilder("chained-demo")
        .with_console_logging()
        .with_in_memory_metrics()
        .build()
    )

    # Use the observe() context manager directly
    async with obs4.observe("custom_operation", step="data_processing", batch_size=32):
        await asyncio.sleep(0.01)  # simulated work
        obs4.info("processing_chunk", chunks=8)

    # Check metrics after observe()
    if isinstance(obs4.metrics, InMemoryMetrics):
        metrics_data = obs4.metrics.get_metrics()
        for key, value in metrics_data["counters"].items():
            print(f"  {key} = {value}")
        for key, values in metrics_data["histograms"].items():
            print(f"  {key} = {values}")


if __name__ == "__main__":
    asyncio.run(main())
