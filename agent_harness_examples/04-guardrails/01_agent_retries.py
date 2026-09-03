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

"""Agent-level retries with timeout, backoff, fallback model, and callbacks.

Demonstrates:
  - Retry attempts on timeout/error with exponential backoff
  - on_retry callback called on each retry attempt
  - Fallback model used after all retries exhausted
  - on_error callback for graceful failure handling

Usage:
    uv run python 01_agent_retries.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/01_agent_retries.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig


load_dotenv()
log = structlog.get_logger()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

AGENT_RETRIES_MAX_RETRIES = int(os.getenv("AGENT_RETRIES_MAX_RETRIES", "3"))
AGENT_RETRIES_TIMEOUT = int(os.getenv("AGENT_RETRIES_TIMEOUT", "10"))
AGENT_RETRIES_BACKOFF = float(os.getenv("AGENT_RETRIES_BACKOFF", "2.0"))
AGENT_RETRIES_FALLBACK_MODEL = os.getenv(
    "AGENT_RETRIES_FALLBACK_MODEL", "ollama:gpt-oss:20b"
)


async def main():
    """Run the agent-retries demonstration.

    Setup:
        - A `.env` file is loaded via `load_dotenv()` at import time. The
          following variables are read, each falling back to a default:
          `GUARDRAILS_MODEL_PROVIDER` (default "ollama"),
          `GUARDRAILS_MODEL_NAME` (default "gpt-oss:20b"),
          `AGENT_RETRIES_MAX_RETRIES` (default 3),
          `AGENT_RETRIES_TIMEOUT` (default 10),
          `AGENT_RETRIES_BACKOFF` (default 2.0),
          `AGENT_RETRIES_FALLBACK_MODEL` (default "ollama:gpt-oss:20b").
        - A model provider must be reachable: if `GUARDRAILS_MODEL_PROVIDER`
          is "ollama", Ollama must be running (`ollama serve`); if "openai",
          `OPENAI_API_KEY` must be set and network access is required.
        - Project dependencies (`dotenv`, `agent_harness`) must be installed,
          e.g. via `uv sync`.
    """
    log.debug("separator")
    log.debug("section", title="Agent-Level Retries with Fallback & Callbacks")
    log.debug("separator")

    # ── Setup memory ────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("retry-demo", memory)

    # ── Configure agent retries with all options ────────────────
    retry_config = (
        AgentRetryConfig()
        .with_max_retries(AGENT_RETRIES_MAX_RETRIES)          # attempts total
        .with_timeout(AGENT_RETRIES_TIMEOUT)                  # per attempt
        .with_backoff(AGENT_RETRIES_BACKOFF)                  # exponential backoff
        .with_fallback(AGENT_RETRIES_FALLBACK_MODEL)          # fallback model
        .on_retry(lambda ctx: log.debug("retry_callback",  # called on each retry
            error_type=ctx.error_type,
            attempt=ctx.attempt,
            max_attempts=ctx.max_attempts,
            will_retry=ctx.will_retry,
        ))
        .on_error(lambda ctx: f"Fallback response: all attempts failed "
                               f"({ctx.error_type}: {ctx.error_message})")
    )

    # ── Build agent ────────────────────────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_agent_retries(retry_config)
    )

    # ── Run ────────────────────────────────────────────────────
    log.debug("section", title="Configuration")
    log.debug("retry_config", max_retries=retry_config.max_retries)
    log.debug("retry_config", timeout=retry_config.timeout)
    log.debug("retry_config", backoff_multiplier=retry_config.backoff_multiplier)
    log.debug("retry_config", fallback_model=retry_config.fallback_model)
    log.debug("section", title="Sending prompt: 'What is 2+2?'")

    result = await agent.run(
        "What is 2+2?",
        history,
        "retry-demo",
    )

    log.debug("separator")
    log.debug("result", success=result.success)
    log.debug("result", used_fallback=result.used_fallback)
    log.debug("result", output=result.output)
    log.debug("result", error_context=result.error_context)


if __name__ == "__main__":
    asyncio.run(main())
