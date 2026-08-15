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

"""Live agent with full observability — logs, traces, metrics, auto-instrumentation.

Demonstrates:
  - ObservabilityBuilder to compose logging + tracing + metrics
  - observe() context manager wrapping agent.run() automatically
  - Automatic metric collection: agent_runs_total, agent_duration_seconds, agent_errors_total
  - InMemoryTracer span inspection after agent runs
  - InMemoryMetrics counters, histograms, gauges after runs
  - Structured logging with session_id, model, prompt_id
  - Multi-turn conversation with per-turn observability

No external services required — all backends are in-memory/console.

Usage:
    uv run python 08_live_agent_logs_metrics.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/08_live_agent_logs_metrics.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.observability import ObservabilityBuilder
from agent_harness.metrics import InMemoryMetrics, MetricNames


async def main():
    print("=" * 60)
    print("Live Agent — Full Observability")
    print("=" * 60)

    # ── Build observability ─────────────────────────────────────
    mem_metrics = InMemoryMetrics()

    obs = (
        ObservabilityBuilder(service_name="live-agent-demo")
        .with_console_logging()
        .with_file_logging(log_file="live_agent.log")
        .with_in_memory_metrics()
        .build()
    )
    # Add a second metrics backend for comparison
    obs._metrics.append(mem_metrics)

    print(f"\nObservability backends:")
    print(f"  Loggers: {[type(l).__name__ for l in obs._loggers]}")
    print(f"  Tracers: {[type(t).__name__ for t in obs._tracers]}")
    print(f"  Metrics: {[type(m).__name__ for m in obs._metrics]}")

    # ── Build agent ─────────────────────────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    session = "live-obs-demo"

    # ── Multi-turn conversation ─────────────────────────────────
    conversations = [
        "My name is Carol and I live in Tokyo.",
        "What is 7 * 8? Just the number.",
        "Based on our conversation, what is my name and where do I live?",
    ]

    print(f"\n--- Multi-turn conversation ({session}) ---")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, memory)

        # agent.run() internally calls obs.observe("agent_run", ...)
        # which automatically:
        #  - Logs agent_run_started / agent_run_completed
        #  - Increments agent_runs_total counter (with model, session_id labels)
        #  - Records agent_duration_seconds histogram
        #  - Creates a trace span
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[memory],
        )

        status = "success" if result.success else "error"
        print(f"\n  Turn {i} [{status}]: {result.output[:100]}")
        obs.info("turn_completed", turn=i, session_id=session,
                 status=status, model=agent.model)

    # ── Inspect collected metrics ───────────────────────────────
    print("\n--- Auto-collected metrics ---")

    all_metrics = mem_metrics.get_metrics()

    print("\n  Counters:")
    for key, value in sorted(all_metrics["counters"].items()):
        print(f"    {key} = {value}")

    print("\n  Histograms:")
    for key, values in sorted(all_metrics["histograms"].items()):
        if values:
            avg = sum(values) / len(values)
            print(f"    {key} = {values} (avg: {avg:.2f}s)")

    # ── Demonstrate observe() context manager ───────────────────
    print("\n--- Manual observe() context manager ---")
    async with obs.observe("custom_workflow", workflow="data_processing", batch_count=5):
        obs.info("workflow_step_1", action="fetching", source="database")
        await asyncio.sleep(0.02)
        obs.info("workflow_step_2", action="processing", items=100)
        await asyncio.sleep(0.02)
        obs.metrics.counter("items_processed", count=100)

    # After observe(), check metrics again
    final_metrics = mem_metrics.get_metrics()
    print("\n  Metrics after custom workflow:")
    for key, value in sorted(final_metrics["counters"].items()):
        print(f"    {key} = {value}")
    for key, values in sorted(final_metrics["histograms"].items()):
        if values:
            print(f"    {key} = {values}")

    # ── Token usage logging (logged by agent.run()) ─────────────
    print("\n--- Token usage was logged automatically ---")
    print("  agent.run() calls obs.log_info('token_usage', ...) with:")
    print("    - input_tokens")
    print("    - output_tokens")
    print("    - total_tokens")
    print("  Check live_agent.log for the structured log entries.")

    print("\n" + "=" * 60)
    print("Observability Summary:")
    print(f"  Log file:  live_agent.log")
    print(f"  Metrics keys tracked: {len(all_metrics['counters'])} counters, "
          f"{len(all_metrics['histograms'])} histograms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
