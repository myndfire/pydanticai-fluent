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
  - Tool calls with RunContext for structured logging
  - Failure telemetry (record_failures=True): exceptions surface in traces
  - ObservabilityBuilder.with_otel_observability() convenience method

Architecture:
    agent_harness  --OTLP gRPC:4317-->  otel-collector  --otlphttp-->  Elasticsearch (logs)
    (logs + metrics + traces)          (otlp receiver:          --otlphttp-->  Prometheus (metrics)
                                         grpc :4317, http :4318) +--otlp gRPC-->  Jaeger (traces)

Prerequisite:
    docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana jaeger prometheus

Usage:
    uv run python 02_otel_full_stack.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Start the observability stack:
        docker compose -f docker-compose.yml up -d elasticsearch otel-collector grafana jaeger prometheus
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/02_otel_full_stack.py

Visualize (single pane: Grafana)
--------------------------------
    Grafana:      http://localhost:3000  (login admin/admin)
                  Datasources (Elasticsearch, Prometheus, Jaeger) and the
                  dashboard "Agent Harness — OTel Telemetry" are
                  auto-provisioned (Dashboards → OTel).
    Jaeger:       http://localhost:16686 → search service <SERVICE_NAME>
    Prometheus:   http://localhost:9090  (PromQL under Prometheus datasource)
    Kibana:       http://localhost:5601
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import structlog

from pydantic_ai import RunContext

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.tools import ToolRegistry

load_dotenv()

log = structlog.get_logger()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
OTEL_ENDPOINT = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")
SERVICE_NAME = os.getenv("OBSERVABILITY_SERVICE_NAME", "all-in-one-observability-demo")

DEMO_FAILURES = True


async def check_port(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=2.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


@dataclass
class ToolDeps:
    observability: Observability
    session_id: str


def get_weather(ctx: RunContext[ToolDeps], city: str) -> str:
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
        "tool_result", tool="get_weather", city=city, result=payload,
        session_id=ctx.deps.session_id,
    )
    return payload


def calculator(ctx: RunContext[ToolDeps], expression: str) -> str:
    ctx.deps.observability.info(
        "tool_call", tool="calculator", expression=expression,
        session_id=ctx.deps.session_id,
    )
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        result = f"Error: {e}"
    ctx.deps.observability.info(
        "tool_result", tool="calculator", expression=expression, result=result,
        session_id=ctx.deps.session_id,
    )
    return f"Result: {result}"


async def main():
    log.debug("separator", char="=", count=60)
    log.debug("title", title="All-in-One OTLP → ES logs + Prometheus metrics + Jaeger traces")
    log.debug("separator", char="=", count=60)

    log.debug("checking_collector", endpoint=OTEL_ENDPOINT)
    otel_ok = await check_port("localhost", 4317)
    log.debug("collector_status", reachable=otel_ok)

    if not otel_ok:
        log.debug("start_instructions")
        log.debug("docker_command", command="docker compose -f docker-compose.yml up -d elasticsearch otel-collector grafana jaeger prometheus")
        return

    obs = Observability(
        builder=ObservabilityBuilder(service_name=SERVICE_NAME)
        .with_otel_observability(otlp_endpoint=OTEL_ENDPOINT, sample_rate=1.0)
    )

    log.debug("loggers", loggers=[type(lg).__name__ for lg in obs._loggers])
    log.debug("tracers", tracers=[type(t).__name__ for t in obs._tracers])
    log.debug("metrics", metrics=[type(m).__name__ for m in obs._metrics])

    obs.info("agent_initialized", version="0.1.0", pid=12345)
    obs.warning("rate_limit_approaching", remaining=10, limit=500)
    await asyncio.sleep(1)

    session_id = "all-in-one-observability-session"
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_prompts(StaticPrompts(
            "You are a helpful assistant with weather and calculator tools. "
            "You MUST use the tools to answer questions. "
            "Call get_weather for each city, then calculator to compute the average. "
            "Never provide answers from memory."
        ))
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
    log.debug("agent_response", output=result.output)

    if DEMO_FAILURES:
        log.debug("section", title="Failure telemetry demo")

        async def _fail_guardrail():
            raise RuntimeError("output guard rejected: hallucination score 0.87")

        try:
            async with obs.observe("guardrail_eval", guard="output", session_id=session_id):
                await _fail_guardrail()
        except RuntimeError as e:
            log.debug("expected_error", error_type=type(e).__name__, error=str(e))

    log.debug("section", title="Flushing OTLP batch exporters")
    await asyncio.sleep(7)
    for lg in obs._loggers:
        if hasattr(lg, "close"):
            lg.close()
    log.debug("flushed")

    log.debug("separator", char="=", count=60)
    log.debug("view_data")
    log.debug("grafana", url="http://localhost:3000", credentials="admin/admin")
    log.debug("jaeger", url="http://localhost:16686", service_name=SERVICE_NAME)
    log.debug("prometheus", url="http://localhost:9090")
    log.debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
