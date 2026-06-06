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

"""PrometheusMetrics — Prometheus metrics with push gateway support.

Demonstrates:
  - PrometheusMetrics(namespace, push_gateway)
  - counter(), gauge(), histogram(), summary()
  - Label-based metrics: model, session_id, status, error_type
  - push_to_gateway() to push metrics to Prometheus pushgateway
  - Agent run with Prometheus metrics auto-collection via observe()

Prerequisite:
    docker compose -f agent_harness_examples/observability/docker-compose.prometheus.yml up -d

Prometheus pushgateway: http://localhost:9091

Usage:
    python 07_prometheus.py
"""

import asyncio

from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger
from agent_harness.metrics import PrometheusMetrics, InMemoryMetrics, MetricNames
from agent_harness.tracing import InMemoryTracer
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

PUSH_GATEWAY = "http://localhost:9091"


async def check_pushgateway(url: str) -> bool:
    """Check if Prometheus pushgateway is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/metrics")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("PrometheusMetrics — Push Gateway")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking Prometheus pushgateway at {PUSH_GATEWAY} ...")
    gw_ok = await check_pushgateway(PUSH_GATEWAY)
    print(f"  {'Reachable' if gw_ok else 'NOT reachable'}")

    # ── Example 1: PrometheusMetrics (standalone) ───────────────
    print("\n--- Example 1: PrometheusMetrics (standalone) ---")

    prom = PrometheusMetrics(
        namespace="example_agent",
        push_gateway=PUSH_GATEWAY,
    )
    print(f"  Namespace: {prom.namespace}")
    print(f"  Push gateway: {prom.push_gateway}")

    # Record metrics
    prom.counter(MetricNames.AGENT_RUNS, model="gpt-4o")
    prom.counter(MetricNames.AGENT_RUNS, model="gpt-4o")
    prom.counter(MetricNames.AGENT_RUNS, model="claude-3")
    prom.counter(MetricNames.AGENT_ERRORS, error_type="TimeoutError")
    prom.counter(MetricNames.AGENT_ERRORS, error_type="ValidationError")

    prom.gauge(MetricNames.ACTIVE_SESSIONS, 3)
    prom.gauge(MetricNames.MEMORY_SIZE, 128 * 1024 * 1024, component="short_term")

    prom.histogram(MetricNames.AGENT_DURATION, 0.8, model="gpt-4o", status="success")
    prom.histogram(MetricNames.AGENT_DURATION, 1.2, model="gpt-4o", status="success")
    prom.histogram(MetricNames.AGENT_DURATION, 2.5, model="claude-3", status="error")

    # Push metrics
    if gw_ok:
        prom.push_to_gateway(job_name="example_agent")
        print(f"  Metrics pushed to {PUSH_GATEWAY}")
        print(f"  View at: {PUSH_GATEWAY}/metrics")

    # ── Example 2: PrometheusMetrics in Observability ───────────
    print("\n--- Example 2: PrometheusMetrics in Observability ---")

    prom_obs = PrometheusMetrics(
        namespace="agent_obs",
        push_gateway=PUSH_GATEWAY if gw_ok else None,
    )

    obs = Observability(
        service_name="prometheus-agent",
        loggers=[ConsoleLogger()],
        tracers=[InMemoryTracer()],
        metrics_list=[prom_obs, InMemoryMetrics()],
    )

    print(f"  Metrics backends: {[type(m).__name__ for m in obs._metrics]}")

    # observe() automatically records AGENT_RUNS counter and
    # AGENT_DURATION histogram through ALL metrics backends
    async with obs.observe("prom_test_operation", session_id="test-1"):
        obs.info("testing_prometheus", step="inside_observe")
        await asyncio.sleep(0.05)

    # Push after observe
    if gw_ok:
        prom_obs.push_to_gateway(job_name="agent_obs")
        print(f"  Metrics pushed after observe()")

    # ── Example 3: Agent run with Prometheus ────────────────────
    print("\n--- Example 3: Agent run with Prometheus ---")

    agent_prom = PrometheusMetrics(
        namespace="live_agent",
        push_gateway=PUSH_GATEWAY if gw_ok else None,
    )

    agent_obs = Observability(
        service_name="live-prom-agent",
        loggers=[ConsoleLogger()],
        metrics_list=[agent_prom],
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_observability(agent_obs)
    )

    memory = InMemoryProvider()

    # Run a few turns — each turn auto-records AGENT_RUNS + AGENT_DURATION
    for i in range(1, 4):
        history = await MessageHistory().load(f"prom-agent-{i}", memory)
        result = await agent.run(
            f"Say 'turn {i}' in exactly 3 words.",
            history,
            f"prom-agent-{i}",
            save_to=[memory],
        )
        print(f"  Turn {i}: {result.output}")

    if gw_ok:
        agent_prom.push_to_gateway(job_name="live_agent")
        print(f"\n  All metrics pushed. View at: {PUSH_GATEWAY}/metrics")


if __name__ == "__main__":
    asyncio.run(main())
