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
"""

import asyncio
import re

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import ContentFilterConfig


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
    print(f"  [on_error] Filter failed: {ctx.error_message}")
    return f"[Content filtered - error]: {ctx.error_message}"


async def main():
    print("=" * 60)
    print("Content Filter Guardrail")
    print("=" * 60)

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
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_content_filter(filter_config)
    )

    # ── Run ────────────────────────────────────────────────────
    print(f"\nContent filter active: {filter_config._on_filter is not None}")
    print(f"\nSending prompt: 'Write a short angry rant about losing a game "
          f"using words like damn, hell, crap.'...\n")

    result = await agent.run(
        "Write a short angry rant about losing a video game. "
        "Use words like 'damn', 'hell', and 'crap' in your response.",
        history,
        "content-filter-demo",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Filtered output: {result.output}")
    print()

    # ── Demonstrate filter error handling ──────────────────────
    print("-" * 40)
    print("Demonstration: broken filter callback")
    broken_filter = ContentFilterConfig().on_filter(
        lambda t: 1 / 0  # raises ZeroDivisionError
    ).on_error(on_filter_error)

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_content_filter(broken_filter)
    )

    history2 = await MessageHistory().load("filter-error-demo", memory)
    result2 = await agent2.run(
        "Say hello",
        history2,
        "filter-error-demo",
    )

    print(f"Success: {result2.success}")
    print(f"Error output: {result2.output}")


if __name__ == "__main__":
    asyncio.run(main())
