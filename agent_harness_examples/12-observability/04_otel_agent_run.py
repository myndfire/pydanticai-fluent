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

"""Live agent with OTEL observability — multi-turn conversation.

Demonstrates:
  - Agent run with automatic OTEL instrumentation (logs + traces + metrics)
  - Multi-turn conversation with per-turn observability
  - PydanticAI native spans (invoke_agent, execute_tool, chat) via OTEL
  - Default create_spans=False: only PydanticAI canonical spans, no harness spans
  - Automatic metric collection: agent_runs_total, agent_duration_seconds
  - Log-trace correlation: log records carry trace_id/span_id

Prerequisite:
    docker compose -f docker-compose.yml up -d otel-collector

Usage:
    uv run python 04_otel_agent_run.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Start the OTel Collector:
        docker compose -f docker-compose.yml up -d otel-collector
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/04_otel_agent_run.py

Visualize:
    Jaeger:    http://localhost:16686 → search service "otel-agent-demo"
    Prometheus: http://localhost:9090
    Grafana:   http://localhost:3000
"""

import asyncio
import os
from dotenv import load_dotenv
import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.observability import Observability, ObservabilityBuilder

load_dotenv()

log = structlog.get_logger()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
OTEL_COLLECTOR = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")
SERVICE_NAME = "otel-agent-demo"


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
    log.debug("title", title="Live Agent — OTEL Observability (Multi-Turn)")
    log.debug("separator", char="=", count=60)

    log.debug("checking_collector", endpoint=OTEL_COLLECTOR)
    otel_ok = await check_port("localhost", 4317)
    log.debug("collector_status", reachable=otel_ok)

    if not otel_ok:
        log.debug("start_instructions")
        log.debug("docker_command", command="docker compose -f docker-compose.yml up -d otel-collector")
        return

    obs = Observability(
        builder=ObservabilityBuilder(service_name=SERVICE_NAME)
        .with_otel_observability(otlp_endpoint=OTEL_COLLECTOR)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    session = "otel-agent-session"

    conversations = [
        "My name is Carol and I live in Tokyo.",
        "What is 7 * 8? Just the number.",
        "Based on our conversation, what is my name and where do I live?",
    ]

    log.debug("section", title="Multi-turn conversation", session=session)
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[memory],
        )
        status = "success" if result.success else "error"
        log.debug("turn", turn=i, status=status, output=result.output[:100])
        obs.info("turn_completed", turn=i, session_id=session, status=status)

    log.debug("separator", char="=", count=60)
    log.debug("view_traces", url="http://localhost:16686")
    log.debug("service_filter", service_name=SERVICE_NAME)
    log.debug("info", detail="Each agent.run() + tool call creates spans automatically.")
    log.debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
