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

"""Content filtering guardrail via user-provided callback.

Demonstrates:
  - on_filter callback that transforms agent output
  - Filtering profanity or unwanted patterns from responses
  - on_error callback for graceful failure when the filter raises

Usage:
    uv run python 04_content_filter.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/04_content_filter.py
"""

import asyncio
import os
import re

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import ContentFilterConfig


load_dotenv()
log = structlog.get_logger()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")


def content_filter(text: str) -> str:
    """Filter unwanted words from the response.

    In production, this could call an external moderation API.
    """
    # Redact common profanity patterns (case-insensitive)
    profanities = [
        r"\b(damn)\b",
        r"\b(hell)\b",
        r"\b(crap)\b",
    ]
    for pattern in profanities:
        text = re.sub(pattern, "***", text, flags=re.IGNORECASE)
    return text


def on_filter_error(ctx):
    """Fallback when the filter callback itself raises an exception."""
    log.debug("filter_error", error_message=ctx.error_message)
    return f"[Content filtered - error]: {ctx.error_message}"


async def main():
    log.debug("separator")
    log.debug("section", title="Content Filter Guardrail")
    log.debug("separator")

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("content-filter-demo", memory)

    filter_config = (
        ContentFilterConfig()
        .on_filter(content_filter)
        .on_error(on_filter_error)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_content_filter(filter_config)
    )

    # ── Run ────────────────────────────────────────────────────
    log.debug("filter_config", active=filter_config._on_filter is not None)
    log.debug("section", title="Sending prompt: angry rant using profanity")

    result = await agent.run(
        "Write a short angry rant about losing a video game. "
        "Use words like 'damn', 'hell', and 'crap' in your response.",
        history,
        "content-filter-demo",
    )

    log.debug("separator")
    log.debug("result", success=result.success)
    log.debug("result", filtered_output=result.output)

    # ── Demonstrate filter error handling ──────────────────────
    log.debug("separator")
    log.debug("section", title="Demonstration: broken filter callback")
    broken_filter = ContentFilterConfig().on_filter(
        lambda t: 1 / 0  # raises ZeroDivisionError
    ).on_error(on_filter_error)

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_content_filter(broken_filter)
    )

    history2 = await MessageHistory().load("filter-error-demo", memory)
    result2 = await agent2.run(
        "Say hello",
        history2,
        "filter-error-demo",
    )

    log.debug("result", success=result2.success)
    log.debug("result", error_output=result2.output)


if __name__ == "__main__":
    asyncio.run(main())
