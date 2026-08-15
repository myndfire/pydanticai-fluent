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

"""All-in-one OTEL → Elasticsearch — logs, metrics, and traces via OpenTelemetry.

Demonstrates:
  - OTELLogger:  structured logs exported via OTLP gRPC
  - OTELTracer:  traces exported via OTLP gRPC
  - OTELMetrics: metrics exported via OTLP gRPC
  - All three signals → OpenTelemetry Collector → Elasticsearch
  - Log-trace correlation: log records emitted inside a span carry trace_id/span_id
  - Agent run instrumented end-to-end with a single Observability facade

Architecture:
    agent_harness  --OTLP gRPC:14317-->  otel-collector  --elasticsearch-->  Elasticsearch
    (logs + metrics + traces)          (elasticsearchexporter)         (data streams:
                                                                        logs-*, metrics-*, traces-*,
                                                                        e.g. logs-generic.otel-default-<date>)
                                        (traces also fan out)
                                        otlp/tempo-->  Tempo (Grafana trace backend)

Prerequisite:
    docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana tempo

Usage:
    uv run python 09_otel_logs_traces_metrics_elasticsearch.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Start Elasticsearch + the OpenTelemetry Collector + Kibana + Grafana + Tempo:
        docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana tempo
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/09_otel_logs_traces_metrics_elasticsearch.py

Visualize
---------
    Kibana:  http://localhost:5601
      Data views (Stack Management → Data Views):
        - logs-generic.otel-default-*      (time field: @timestamp)
        - metrics-generic.otel-default-*   (time field: @timestamp)
        - traces-generic.otel-default-*    (time field: @timestamp)
      Then open Discover and filter  service.name: all-in-one-es-demo

    Grafana: http://localhost:3000   (login admin/admin, datasources auto-provisioned)
      Explore → Logs (Elasticsearch datasource):  service.name: all-in-one-es-demo
      Metrics:         ES aggregations (sum/count) over metrics.* fields
      Explore → Tempo: native trace waterfall (search service.name or trace_id)
      Select a span → "View in logs" jumps to the correlated ES log records by trace_id
"""

import asyncio

from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger, OTELLogger
from agent_harness.tracing import OTELTracer
from agent_harness.metrics import OTELMetrics
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

ES_ENDPOINT = "http://localhost:9200"
OTEL_ENDPOINT = "localhost:14317"
OTEL_TRACER_ENDPOINT = "http://localhost:14317"
SERVICE_NAME = "all-in-one-es-demo"


async def check_port(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=2.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def check_elasticsearch(endpoint: str) -> bool:
    """Check if Elasticsearch is reachable (plain HTTP check, no es client needed)."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{endpoint}/")
            return resp.status_code == 200
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("All-in-One OTEL → Elasticsearch")
    print("=" * 60)

    # ── Connection checks ───────────────────────────────────────
    print(f"\nChecking Elasticsearch at {ES_ENDPOINT} ...")
    es_ok = await check_elasticsearch(ES_ENDPOINT)
    print(f"  {'Reachable' if es_ok else 'NOT reachable'}")

    print(f"Checking OTel Collector at {OTEL_ENDPOINT} ...")
    otel_ok = await check_port("localhost", 14317)
    print(f"  {'Reachable' if otel_ok else 'NOT reachable'}")

    if not (es_ok and otel_ok):
        print("\n  Start both with:")
        print("    docker compose -f docker-compose.yml up -d elasticsearch otel-collector grafana tempo")
        return

    # ── Observability: logs + metrics + traces via OTLP ─────────
    print("\n--- Building Observability (OTEL backends) ---")

    otel_logger = OTELLogger(service_name=SERVICE_NAME, otlp_endpoint=OTEL_ENDPOINT)
    obs = Observability(
        service_name=SERVICE_NAME,
        loggers=[ConsoleLogger(), otel_logger],
        tracers=[
            OTELTracer(
                service_name=SERVICE_NAME,
                otlp_endpoint=OTEL_TRACER_ENDPOINT,
                sample_rate=1.0,
            ),
        ],
        metrics_list=[
            OTELMetrics(service_name=SERVICE_NAME, otlp_endpoint=OTEL_ENDPOINT),
        ],
    )
    print(f"  Loggers: {[type(lg).__name__ for lg in obs._loggers]}")
    print(f"  Tracers: {[type(t).__name__ for t in obs._tracers]}")
    print(f"  Metrics: {[type(m).__name__ for m in obs._metrics]}")

    # ── Structured logs ─────────────────────────────────────────
    print("\n--- Sending structured logs (→ ES via OTLP) ---")

    obs.info("agent_initialized", version="0.1.0", pid=12345)
    obs.info("agent_run_started", model="gpt-oss:20b", session_id="all-in-one-1")
    obs.warning("rate_limit_approaching", remaining=10, limit=500)
    obs.error("token_limit_exceeded", actual=5000, max_tokens=4096)
    await asyncio.sleep(1)

    # ── Manual span with log-trace correlation ──────────────────
    print("\n--- Manual span (log records inherit trace context) ---")

    async with obs.observe("manual_span", op="test", value=42):
        obs.info("inside_span", detail="This log is correlated with the span")
        obs.add_span_event("checkpoint_reached", step=1)
        await asyncio.sleep(0.5)

    # ── Agent run ───────────────────────────────────────────────
    print("\n--- Agent run (logs + spans + metrics) ---")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("all-in-one-es-session", memory)
    result = await agent.run(
        "What is 10 divided by 2?",
        history,
        "all-in-one-es-session",
        save_to=[memory],
    )
    print(f"  Agent response: {result.output}")

    # ── Flush batch exporters ───────────────────────────────────
    print("\n--- Flushing OTLP batch exporters ---")
    await asyncio.sleep(7)  # span/metric readers + log processor export in background
    otel_logger.close()
    print("  Flushed.")

    # ── How to inspect in Elasticsearch ─────────────────────────
    print("\n" + "=" * 60)
    print("View data in Elasticsearch (data streams are auto-created):")
    print("  curl -s http://localhost:9200/_cat/indices/*generic.otel-default*")
    print("  Logs:    curl -s 'http://localhost:9200/logs-generic.otel-default-*/_search?q=service.name:%s'"
          % SERVICE_NAME)
    print("  Metrics: curl -s 'http://localhost:9200/metrics-generic.otel-default-*/_search?q=service.name:%s'"
          % SERVICE_NAME)
    print("  Traces:  curl -s 'http://localhost:9200/traces-generic.otel-default-*/_search?q=service.name:%s'"
          % SERVICE_NAME)
    print("\n  Log-trace correlation: log records emitted inside a span carry")
    print("  top-level trace_id/span_id fields, e.g. filter by trace_id:")
    print("    curl -s 'http://localhost:9200/logs-generic.otel-default-*/_search?q=trace_id:<span-trace-id>'")

    print("\n  Visualize in Kibana (http://localhost:5601):")
    print("    1. Stack Management → Data Views → create:")
    print("       - logs-generic.otel-default-*      (time field: @timestamp)")
    print("       - metrics-generic.otel-default-*   (time field: @timestamp)")
    print("       - traces-generic.otel-default-*    (time field: @timestamp)")
    print("    2. Discover → filter  service.name: all-in-one-es-demo")
    print("\n  Visualize in Grafana (http://localhost:3000, admin/admin):")
    print("    Elasticsearch + Tempo datasources are auto-provisioned.")
    print("    Explore → Logs (Elasticsearch datasource):  service.name: all-in-one-es-demo")
    print("    Metrics:         ES aggregations (sum/count) over metrics.* fields")
    print("    Explore → Tempo: native trace waterfall (search service.name or trace_id)")
    print("    Select a span → 'View in logs' jumps to correlated ES log records by trace_id")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
