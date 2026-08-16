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

"""All-in-one OTLP — logs, metrics, and traces via OpenTelemetry.

Demonstrates:
  - OTELLogger:  structured logs exported via OTLP gRPC
  - OTELTracer:  traces exported via OTLP gRPC
  - OTELMetrics: metrics exported via OTLP gRPC
  - One OTLP receiver → three separate OTLP backends:
      logs → Elasticsearch (OTLP/HTTP /_otlp/v1/logs)
      metrics → Prometheus (OTLP receiver /api/v1/otlp/v1/metrics)
      traces → Jaeger (OTLP gRPC :4317)
  - Log-trace correlation: log records emitted inside a span carry trace_id/span_id
  - Agent run instrumented end-to-end with a single Observability facade

Architecture:
    agent_harness  --OTLP gRPC:14317-->  otel-collector  --otlphttp-->  Elasticsearch (logs)
    (logs + metrics + traces)          (otlp receiver:          --otlphttp-->  Prometheus (metrics)
                                         grpc :4317, http :4318) +--otlp gRPC-->  Jaeger (traces)

Prerequisite:
    docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana jaeger prometheus

Usage:
    uv run python 09_otel_oltp_logs_traces_metrics.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Start the observability stack:
        docker compose -f docker-compose.yml up -d elasticsearch otel-collector grafana jaeger prometheus
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/09_otel_oltp_logs_traces_metrics.py

Visualize (single pane: Grafana)
--------------------------------
    Grafana: http://localhost:3000   (login admin/admin)
      Datasources (Elasticsearch, Prometheus, Jaeger) and the dashboard
      "Agent Harness — OTel Telemetry" are auto-provisioned
      (Dashboards → OTel).

      Logs like Kibana:    Logs Drilldown /a/explore-logs (Elasticsearch datasource)
      Metrics:             Prometheus (PromQL) — e.g.
                           sum(all_in_one_observability_demo_agent_runs_total)
      Traces like Jaeger:  Jaeger UI http://localhost:16686 or Explore → Jaeger —
                           native waterfall; select a span → "View in logs" jumps to
                           correlated ES log records by trace_id

    Kibana (optional specialist): http://localhost:5601
      Log levels dashboard "Agent Harness — Log Levels" (bar by severity,
      volume-over-time, donut share, recent-logs table) is provisioned via:
        ./kibana/provision-log-levels-dashboard.sh
      Data view: logs-generic.otel-default*  (time field: @timestamp)
      Or use Discover and filter  service.name: all-in-one-observability-demo
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
SERVICE_NAME = "all-in-one-observability-demo"


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
    print("All-in-One OTLP → ES logs + Prometheus metrics + Jaeger traces")
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
        print("    docker compose -f docker-compose.yml up -d elasticsearch otel-collector grafana jaeger prometheus")
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
    obs.info("agent_run_started", model="gpt-oss:20b", session_id="all-in-one-observability-1")
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
    history = await MessageHistory().load("all-in-one-observability-session", memory)
    result = await agent.run(
        "What is 10 divided by 2?",
        history,
        "all-in-one-observability-session",
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
    print("View data per backend (all via one OTLP collector):")
    print("\n  Logs → Elasticsearch (data streams are auto-created):")
    print("  curl -s http://localhost:9200/_cat/indices/*generic.otel-default*")
    print("  curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=service.name:%s'"
          % SERVICE_NAME)
    print("\n  Metrics → Prometheus (OTLP receiver):")
    print("  curl -s 'http://localhost:9090/api/v1/query?query=%s'"
          % "sum({__name__=~\\\"all_in_one_observability_demo_agent_runs_total\\\"})")
    print("\n  Traces → Jaeger:")
    print("  Open http://localhost:16686 and search service: all-in-one-observability-demo")
    print("\n  Log-trace correlation: log records emitted inside a span carry")
    print("  top-level trace_id/span_id fields, e.g. filter by trace_id:")
    print("    curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=trace_id:<span-trace-id>'")

    print("\n  Visualize in Kibana (http://localhost:5601):")
    print("    Provision once:  ./kibana/provision-log-levels-dashboard.sh")
    print("    Dashboard:       /app/dashboards#/view/log-levels-dashboard")
    print("                     ('Agent Harness — Log Levels' — bar by severity,")
    print("                      volume-over-time, donut share, recent-logs table)")
    print("    Discover:        data view logs-generic.otel-default*,")
    print("                     filter  service.name: all-in-one-observability-demo")
    print("\n  Visualize (single pane → Grafana, http://localhost:3000, admin/admin):")
    print("    Datasources (Elasticsearch, Prometheus, Jaeger) + dashboard")
    print("    'Agent Harness — OTel Telemetry' auto-provisioned (Dashboards → OTel).")
    print("    Logs like Kibana:    /a/explore-logs (Elasticsearch datasource)")
    print("    Metrics (PromQL):   sum({__name__=\"all_in_one_observability_demo_agent_runs_total\"}),")
    print("                         sum(..._duration_seconds_sum) / sum(..._duration_seconds_count)")
    print("    Explore → Jaeger:    native trace waterfall; select a span →")
    print("                         'View in logs' jumps to correlated ES log records by trace_id")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())