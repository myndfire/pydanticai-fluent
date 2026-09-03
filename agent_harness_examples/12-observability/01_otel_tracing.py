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

"""OTEL tracing — OTLP gRPC export via OTel Collector to Jaeger.

Demonstrates:
  - OTELTracer: OTLP gRPC export (→ OTel Collector → Jaeger)
  - Span attributes and span context (trace_id, span_id) in observe()
  - Manual span events and attributes
  - Agent run with automatic trace spans

Prerequisite:
    docker compose -f docker-compose.yml up -d otel-collector jaeger

Usage:
    uv run python 01_otel_tracing.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Start the OTel Collector + Jaeger:
        docker compose -f docker-compose.yml up -d otel-collector jaeger
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/01_otel_tracing.py

Visualize:
    Jaeger UI: http://localhost:16686 → search service "otel-tracing-demo"
"""

import asyncio
import os
from dotenv import load_dotenv
import structlog

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

log = structlog.get_logger()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
OTEL_COLLECTOR = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")
SERVICE_NAME = "otel-tracing-demo"


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


async def main():
    log.debug("separator", char="=", count=60)
    log.debug("title", title="OTEL Tracing — OTLP → Collector → Jaeger")
    log.debug("separator", char="=", count=60)

    log.debug("checking_collector", endpoint=OTEL_COLLECTOR)
    otlp_ok = await check_port("localhost", 4317)
    log.debug("collector_status", reachable=otlp_ok)

    if not otlp_ok:
        log.debug("start_instructions")
        log.debug("docker_command", command="docker compose -f docker-compose.yml up -d otel-collector jaeger")
        return

    obs = Observability(
        builder=ObservabilityBuilder(service_name=SERVICE_NAME)
        .with_otel_observability(
            otlp_endpoint=OTEL_COLLECTOR,
            sample_rate=1.0,
            create_spans=True,
        )
    )

    async with obs.observe("manual_span", op="test", value=42):
        obs.info("inside_span", detail="This appears in the trace")
        obs.add_span_event("checkpoint_reached", step=1)
        obs.set_span_attribute("custom_key", "custom_value")
        await asyncio.sleep(0.05)

    log.debug("section", title="Agent run with OTEL tracing")
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("otel-tracing", memory)
    result = await agent.run(
        "What is 10 divided by 2?",
        history,
        "otel-tracing",
        save_to=[memory],
    )
    log.debug("response", output=result.output)

    log.debug("separator", char="=", count=60)
    log.debug("view_traces", url="http://localhost:16686")
    log.debug("service_filter", service_name=SERVICE_NAME)
    log.debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
