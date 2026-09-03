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

"""Log enrichment — agent-level and per-run enrichment with LogContext.

Demonstrates:
  - with_log_enrichment() to attach persistent enrichment to an agent
  - Per-run enrichment via agent.run(enrichment=LogContext().with_(...))
  - Enriched log context flowing to _started, _completed, token_usage events
  - OTEL log records carrying pipeline, agent_role, and stage keys

Usage:
    uv run python logging/01_basic_log_context.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python logging/01_basic_log_context.py
"""

import asyncio
import os

import structlog

from agent_harness import ManagedAgent, LogContext, EnvEnricher
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

log = structlog.get_logger()


async def main():
    log.debug("separator")
    log.debug("section", title="Log Enrichment — Basic LogContext")
    log.debug("separator")

    model = ModelConfig(
        provider=os.getenv("LOGGING_MODEL_PROVIDER", "ollama"),
        model_name=os.getenv("LOGGING_MODEL_NAME", "gpt-oss:20b"),
    )
    memory = InMemoryProvider()

    # ── Agent with persistent enrichment (A) ────────────────────
    # Every log from this agent automatically carries pipeline + agent_role
    agent = (
        ManagedAgent()
        .with_model(model)
        .with_log_enrichment(
            LogContext()
            .with_("pipeline", os.getenv("LOGGING_01_PIPELINE", "content-qa"))
            .with_(
                "agent_role", os.getenv("LOGGING_01_AGENT_ROLE", "assistant")
            ),
            EnvEnricher(),  # auto-attaches host, env, pid
        )
    )

    log.debug("section", title="Agent enrichment providers")
    log.debug(
        "enrichment_providers",
        log_context=f"pipeline={os.getenv('LOGGING_01_PIPELINE', 'content-qa')}, "
        f"agent_role={os.getenv('LOGGING_01_AGENT_ROLE', 'assistant')}",
    )
    log.debug("enricher", name="EnvEnricher", details="host, env, pid (automatic)")

    # ── Run 1: with per-run enrichment (B) ──────────────────────
    log.debug("section", title="Run 1: Per-run enrichment")
    h1 = await MessageHistory().load("log-1", memory)
    r1 = await agent.run(
        "What is 2+2? Answer in one word.",
        h1,
        "log-1",
        enrichment=LogContext().with_(
            "stage", os.getenv("LOGGING_01_STAGE_1", "math-check")
        ),
    )
    log.debug("output", result=str(r1.output))

    # ── Run 2: different per-run context ────────────────────────
    log.debug("section", title="Run 2: Different per-run context")
    h2 = await MessageHistory().load("log-2", memory)
    r2 = await agent.run(
        "What color is the sky? Answer in one word.",
        h2,
        "log-2",
        enrichment=LogContext().with_(
            "stage", os.getenv("LOGGING_01_STAGE_2", "knowledge-check")
        ),
    )
    log.debug("output", result=str(r2.output))

    log.debug("done", message="Check the JSON log output above.")
    log.debug(
        "entry_fields",
        pipeline=os.getenv('LOGGING_01_PIPELINE', 'content-qa'),
        agent_role=os.getenv('LOGGING_01_AGENT_ROLE', 'assistant'),
        fields="stage, host, env, pid",
    )


if __name__ == "__main__":
    asyncio.run(main())
