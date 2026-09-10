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

"""ClearToolResults — blank old tool results, keep the recent pairs.

Demonstrates:
  - with_clear_tool_results(keep_pairs, max_messages): cheapest compaction tier
  - tool outputs dominate context; old results are blanked in place while
    tool-call / tool-return pairing stays valid

All values come from the environment (no hardcoded numbers):

    COMPACTION_CLEAR_MAX_MESSAGES / COMPACTION_CLEAR_KEEP_PAIRS
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 01_clear_tool_results.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.tools import ToolRegistry


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")

CLEAR_MAX_MESSAGES = int(os.getenv("COMPACTION_CLEAR_MAX_MESSAGES", "10"))
CLEAR_KEEP_PAIRS = int(os.getenv("COMPACTION_CLEAR_KEEP_PAIRS", "2"))


def bulky_lookup(city: str) -> str:
    """Return a bulky weather report for a city (fills context fast)."""
    return ("Sunny, 21C. " + f"Details for {city}. ") * 40


async def main():
    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_tools(ToolRegistry().add(bulky_lookup))
        .with_short_term_memory(memory)
        .with_clear_tool_results(
            keep_pairs=CLEAR_KEEP_PAIRS, max_messages=CLEAR_MAX_MESSAGES
        )
    )
    log.debug(
        "compaction_config",
        strategy="clear_tool_results",
        max_messages=CLEAR_MAX_MESSAGES,
        keep_pairs=CLEAR_KEEP_PAIRS,
    )

    session = "compaction-clear-demo"
    for i, city in enumerate(["Paris", "Rome", "Madrid"], start=1):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(
            f"Look up the weather in {city} and reply in one sentence.",
            history,
            session,
            save_to=[memory],
        )
        log.debug("turn", number=i, output=str(result.output)[:100])


if __name__ == "__main__":
    asyncio.run(main())
