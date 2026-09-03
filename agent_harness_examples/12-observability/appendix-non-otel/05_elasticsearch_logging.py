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

"""ElasticsearchLogger — structured logging to Elasticsearch with daily indices.

Demonstrates:
  - ElasticsearchLogger(endpoint, index_prefix, service_name)
  - Daily index pattern: agent-logs-YYYY.MM.DD
  - Structured document format: timestamp, service_name, level, message, context
  - Lazy connection with graceful fallback
  - close() to flush pending tasks and clean up the connection

Prerequisite:
    docker compose -f docker-compose.yml up -d elasticsearch

Usage:
    uv run python 05_elasticsearch_logging.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start Elasticsearch:
        docker compose -f docker-compose.yml up -d elasticsearch
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/05_elasticsearch_logging.py
"""

import asyncio
import os
import time
from dotenv import load_dotenv

from agent_harness.observability import ObservabilityBuilder
from agent_harness.logging import ElasticsearchLogger
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
ES_ENDPOINT = os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")


async def main():
    """Run the Elasticsearch logging example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start Elasticsearch: docker compose -f docker-compose.yml up -d elasticsearch
        4. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("ElasticsearchLogger — ES Structured Logging")
    print("=" * 60)

    # ── ElasticsearchLogger ─────────────────────────────────────
    es_logger = ElasticsearchLogger(
        endpoint=ES_ENDPOINT,
        index_prefix="agent-logs",
        service_name="es-logging-demo",
    )
    print(f"  Logger: ElasticsearchLogger(endpoint={ES_ENDPOINT}, index_prefix=agent-logs)")

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking Elasticsearch at {ES_ENDPOINT} ...")
    if es_logger.es_client is None:
        print(f"  Elasticsearch not reachable at {ES_ENDPOINT}")
        print("  Start with:")
        print("    docker compose -f docker-compose.yml up -d elasticsearch")
        return
    print("  Elasticsearch is reachable.")

    # ── Send structured logs ────────────────────────────────────
    print("\n--- Sending structured logs ---")
    print("  Each log call writes to BOTH console (structlog) and Elasticsearch.")

    es_logger.info("agent_initialized", version="0.1.0", pid=12345)
    await asyncio.sleep(0.2)  # give async task time to fire

    es_logger.info("agent_run_started", model="gpt-4o", session_id="es-demo-1")
    await asyncio.sleep(0.2)

    es_logger.info("tool_invoked", tool="web_search", query="latest AI news",
                   latency_ms=342, result_count=12)
    await asyncio.sleep(0.2)

    es_logger.warning("rate_limit_approaching", remaining=10, limit=500,
                      model="gpt-4o", session_id="es-demo-1")
    await asyncio.sleep(0.2)

    es_logger.error("token_limit_exceeded", actual=5000, max_tokens=4096,
                    session_id="es-demo-1", model="gpt-4o")
    await asyncio.sleep(0.2)

    # ── Agent run with Elasticsearch logging ────────────────────
    print("\n--- Agent run with Elasticsearch logging ---")

    obs_builder = ObservabilityBuilder(service_name="es-agent-demo")
    obs_builder._loggers.append(es_logger)
    obs = obs_builder.build()

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_observability(obs)
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("es-agent-session", memory)
    result = await agent.run(
        "What is the capital of Japan? Keep it brief.",
        history,
        "es-agent-session",
        save_to=[memory],
    )
    print(f"  Agent response: {result.output}")

    # ── Wait for all pending ES writes ──────────────────────────
    print("\n--- Flushing pending writes ---")
    await asyncio.sleep(1)  # give tasks time to complete

    # ── Close connection ────────────────────────────────────────
    print("\n--- Closing Elasticsearch logger ---")
    await es_logger.close()
    print("  Logger closed.")

    print(f"\n  View logs in Elasticsearch:")
    print(f"    GET /_cat/indices/agent-logs-*")
    print(f"    GET /agent-logs-*/_search?q=service_name:es-agent-demo")


if __name__ == "__main__":
    asyncio.run(main())
