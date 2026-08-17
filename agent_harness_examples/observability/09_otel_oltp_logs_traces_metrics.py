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
  - Tool calls (RunContext so they can log to ES) with the trace stream being
    PydanticAI's native spans only (invoke_agent / execute_tool <tool> / chat <model>);
    the default ``create_spans=False`` keeps the harness from adding its own
    agent_run/manual_span spans, so querying on PydanticAI's canonical span names
    and gen_ai.* attributes is stable
  - Failure telemetry (default ``record_failures=True``): exceptions escaping a
    span() block surface in the trace stream as status=ERROR + an exception event.
    PydanticAI-owned failures land on the canonical spans
    (invoke_agent/execute_tool/chat); harness-owned failures that occur outside
    those spans emit ``<service>.<operation>:failed`` spans carrying
    ``error.type`` / ``error.source``. Guarded by ``DEMO_FAILURES`` at the bottom
    of main().

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
from dataclasses import dataclass

from pydantic_ai import RunContext

from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger, OTELLogger
from agent_harness.tracing import OTELTracer
from agent_harness.metrics import OTELMetrics
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.tools import ToolRegistry

ES_ENDPOINT = "http://localhost:9200"
OTEL_ENDPOINT = "localhost:14317"
OTEL_TRACER_ENDPOINT = "http://localhost:14317"
SERVICE_NAME = "all-in-one-observability-demo"

# When True, main() additionally exercises failure paths so the resulting
# traces contain ERROR spans / exception events + harness <service>.<op>:failed
# spans (see the "Failure telemetry demo" section of main()).
DEMO_FAILURES = True


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


@dataclass
class ToolDeps:
    """Dependency container injected into context-aware tools via RunContext."""

    observability: Observability
    session_id: str


def get_weather(ctx: RunContext[ToolDeps], city: str) -> str:
    """Get current weather conditions for a city.

    Args:
        city: Name of the city to check weather for.
    """
    ctx.deps.observability.info(
        "tool_call", tool="get_weather", city=city, session_id=ctx.deps.session_id
    )
    conditions = {
        "tokyo": "Clear skies, 22°C",
        "london": "Light rain, 15°C",
        "new york": "Partly cloudy, 18°C",
    }
    payload = conditions.get(city.lower())
    if payload is None:
        if DEMO_FAILURES:
            err = ValueError(f"Unknown city: {city}")
            err._error_source = "tool"
            raise err
        payload = f"Unknown city: {city}"
    ctx.deps.observability.info(
        "tool_result",
        tool="get_weather",
        city=city,
        result=payload,
        session_id=ctx.deps.session_id,
    )
    return payload


def calculator(ctx: RunContext[ToolDeps], expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '(22 + 15 + 18) / 3'.
    """
    ctx.deps.observability.info(
        "tool_call",
        tool="calculator",
        expression=expression,
        session_id=ctx.deps.session_id,
    )
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        result = f"Error: {e}"
    ctx.deps.observability.info(
        "tool_result",
        tool="calculator",
        expression=expression,
        result=result,
        session_id=ctx.deps.session_id,
    )
    return f"Result: {result}"


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
                create_spans=False,  # canonical — no harness spans; native PydanticAI spans only
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

    # ── Agent run ───────────────────────────────────────────────
    print("\n--- Agent run (logs + spans + metrics) ---")

    session_id = "all-in-one-observability-session"

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(
            StaticPrompts(
                "You are a helpful assistant with weather and calculator tools. "
                "You MUST use the tools to answer questions. "
                "Call get_weather for each city, then calculator to compute the average. "
                "Never provide answers from memory."
            )
        )
        .with_deps_type(ToolDeps)
        .with_observability(obs)
        .with_tools(ToolRegistry().add_many(get_weather, calculator))
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load(session_id, memory)
    deps = ToolDeps(observability=obs, session_id=session_id)
    result = await agent.run(
        "What is the average temperature (in °C) of Tokyo, London, and New York? "
        "Use get_weather for each city, then calculator.",
        history,
        session_id,
        deps=deps,
        save_to=[memory],
    )
    print(f"  Agent response: {result.output}")

    # ── Failure telemetry demo (guarded) ────────────────────────
    if DEMO_FAILURES:
        print("\n--- Failure telemetry demo ---")
        print("  (1) harness-owned failure → <service>.guardrail_eval:failed")

        async def _fail_guardrail():
            raise_runtime = RuntimeError("output guard rejected: hallucination score 0.87")
            raise_runtime._error_source = "guardrail"
            raise raise_runtime

        try:
            async with obs.observe(
                "guardrail_eval", guard="output", session_id=session_id
            ):
                await _fail_guardrail()
        except RuntimeError as e:
            print(f"      (expected) raised: {type(e).__name__}: {e}")

        print("  (2) PydanticAI-owned in-graph failure → invoke_agent/execute_tool "
              "ERROR + exception events")
        try:
            session_id2 = "all-in-one-observability-failure-session"
            memory2 = InMemoryProvider()
            history2 = await MessageHistory().load(session_id2, memory2)
            deps2 = ToolDeps(observability=obs, session_id=session_id2)
            await agent.run(
                "get_weather called on 'nowhere' will fail — call it and report "
                "whatever happens.",
                history2,
                session_id2,
                deps=deps2,
                save_to=[],
            )
            print("      (unexpected) run succeeded, the model avoided the failing tool")
        except Exception as e:
            print(f"      (expected) run failed: {type(e).__name__}: {e}")

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

    print("\n  Failure telemetry (traces carrying status=ERROR / exception events):")
    print("  Trace data view:  traces-generic.otel-default*  (time field: @timestamp)")
    print("    curl -s 'http://localhost:9200/traces-generic.otel-default*/_search?_source=name,status,attributes.error.type,attributes.error.source&q=name:%22*:failed%22'")
    print("    curl -s 'http://localhost:9200/traces-generic.otel-default*/_search?_source=name,status,events&q=events.name:exception'")
    print("  Kibana/ES filter for the Errors dashboard:")
    print("    status.code: \"STATUS_CODE_ERROR\"                     # all failed spans")
    print("    name: *:failed                                        # harness-owned failures only")
    print("    error.type: builtins.ValueError / attributes.error.source: tool  # drill into cause")

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