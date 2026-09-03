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

"""Logfire + OTEL — Dual-Destination Observability.

Demonstrates:
  - Chaining with_logfire_observability() + with_otel_observability()
    to send telemetry to both Logfire cloud and a local OTel Collector
  - PydanticAI auto-instrumentation covers both destinations
  - Log-trace correlation across Logfire and OTEL backends
  - All logging (demo output + agent observability) flows through OTEL

Prerequisite:
    Set LOGFIRE_TOKEN in .env (token from https://logfire.pydantic.dev)
    OTel Collector running on localhost:4317 (or set OTEL_COLLECTOR_ENDPOINT)

Usage:
    uv run python 05_logfire.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Set your Logfire token:
        echo "LOGFIRE_TOKEN=pylf_v1_us_..." >> .env
    3. Start the OTel Collector (optional, for OTEL backend):
        docker compose up -d otel-collector
    4. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/05_logfire.py

Visualize:
    Logfire: https://logfire.pydantic.dev → your project dashboard
    Jaeger:  http://localhost:16686 (if collector exports to Jaeger)
"""

import asyncio
import os
from dotenv import load_dotenv

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.logging import ConsoleLogger
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
SERVICE_NAME = "logfire-agent-demo"


async def main():
    builder = (
        ObservabilityBuilder(service_name=SERVICE_NAME)
        .with_logfire_observability()
        .with_otel_observability(
            otlp_endpoint=os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317"),
        )
    )
    builder._loggers.append(ConsoleLogger())
    obs = Observability(builder=builder)

    obs.log_debug("separator", char="=", count=60)
    obs.log_debug("title", title="Logfire + OTEL — Dual-Destination Observability")
    obs.log_debug("separator", char="=", count=60)

    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if not logfire_token:
        obs.log_debug("logfire_token_missing")
        obs.log_debug("get_token_url", url="https://logfire.pydantic.dev")
        obs.log_debug("add_to_env")
        obs.log_debug("env_example", env="LOGFIRE_TOKEN=pylf_v1_us_...")
        return

    obs.log_debug("logfire_token", suffix=logfire_token[-8:])

    obs.log_debug("loggers", loggers=[type(l).__name__ for l in obs._loggers])
    obs.log_debug("tracers", tracers=[type(t).__name__ for t in obs._tracers])
    obs.log_debug("metrics", metrics=[type(m).__name__ for m in obs._metrics])

    obs.log_debug("section", title="Agent run with dual-destination observability")
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    session = "logfire-agent-session"

    conversations = [
        "My name is Carol and I live in Tokyo.",
        "What is 7 * 8? Just the number.",
        "Based on our conversation, what is my name and where do I live?",
    ]

    obs.log_debug("section", title="Multi-turn conversation", session=session)
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[memory],
        )
        status = "success" if result.success else "error"
        obs.log_debug("turn", turn=i, status=status, output=result.output[:100])
        obs.log_info("turn_completed", turn=i, session_id=session, status=status)

    obs.log_debug("separator", char="=", count=60)
    obs.log_debug("view_traces_logfire", url="https://logfire.pydantic.dev")
    obs.log_debug("view_traces_otel", url="http://localhost:16686 (Jaeger)")
    obs.log_debug("service_name", service_name=SERVICE_NAME)
    obs.log_debug("info", detail="Each agent.run() + tool call creates spans on both backends.")
    obs.log_debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
