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

"""Log enrichment — PipelineContext with run_stage() and auto-logging.

Demonstrates:
  - PipelineContext.with_observability() — auto-logs each post()
  - PipelineContext.with_memory() — auto-creates MessageHistory per stage
  - ctx.run_stage() — simplified pipeline orchestration
  - pipeline_stage_completed log events
  - ctx.display_trace() — formatted trace table
  - Agent-level enrichment flowing through the pipeline

Usage:
    uv run python logging/03_pipeline_logging.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python logging/03_pipeline_logging.py
"""

import asyncio
import os

import structlog

from agent_harness import (
    ManagedAgent,
    LogContext,
    EnvEnricher,
    PipelineContext,
)
from agent_harness.memory import InMemoryProvider
from agent_harness.model_config import ModelConfig
from agent_harness.observability import Observability, ObservabilityBuilder

log = structlog.get_logger()


async def main():
    log.debug("separator")
    log.debug("section", title="Log Enrichment — Pipeline with Auto-Logging")
    log.debug("separator")

    model = ModelConfig(
        provider=os.getenv("LOGGING_MODEL_PROVIDER", "ollama"),
        model_name=os.getenv("LOGGING_MODEL_NAME", "gpt-oss:20b"),
    )
    obs = Observability(
        builder=ObservabilityBuilder(service_name="pipeline-logging")
        .with_otel_observability()
    )

    # ── Agents with persistent enrichment ───────────────────────
    base = LogContext().with_(
        "pipeline", os.getenv("LOGGING_03_PIPELINE", "content-qa")
    )

    researcher = (
        ManagedAgent()
        .with_model(model)
        .with_log_enrichment(
            base.with_(
                "agent_role",
                os.getenv("LOGGING_03_ROLE_RESEARCHER", "researcher"),
            ),
            EnvEnricher(),
        )
    )

    writer = (
        ManagedAgent()
        .with_model(model)
        .with_log_enrichment(
            base.with_(
                "agent_role", os.getenv("LOGGING_03_ROLE_WRITER", "writer")
            ),
            EnvEnricher(),
        )
    )

    editor = (
        ManagedAgent()
        .with_model(model)
        .with_log_enrichment(
            base.with_(
                "agent_role", os.getenv("LOGGING_03_ROLE_EDITOR", "editor")
            ),
            EnvEnricher(),
        )
    )

    # ── Pipeline context — handles history, enrichment, logging ─
    ctx = (
        PipelineContext()
        .with_observability(obs)
        .with_memory(InMemoryProvider())
    )

    # ── Run the pipeline ────────────────────────────────────────
    log.debug("section", title="Stage 1: Research")
    r1 = await ctx.run_stage(
        researcher, "Research",
        "What are embeddings in machine learning? Give 2 key facts.",
    )

    log.debug("section", title="Stage 2: Write")
    r2 = await ctx.run_stage(
        writer, "Write",
        f"Write one sentence about: {r1.output}",
    )

    log.debug("section", title="Stage 3: Edit")
    r3 = await ctx.run_stage(
        editor, "Edit",
        f"Polish this sentence: {r2.output}",
    )

    # ── Display trace ───────────────────────────────────────────
    ctx.display_trace()

    log.debug("log_entry_fields", fields="pipeline, agent_role, stage, host, env, pid")
    log.debug("auto_logged_events", events="agent_run_started, token_usage, agent_run_completed")
    log.debug("pipeline_events", event="pipeline_stage_completed for each stage")


if __name__ == "__main__":
    asyncio.run(main())
