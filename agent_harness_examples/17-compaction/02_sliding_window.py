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

"""SlidingWindowCompaction — keep only the recent tail of the history.

Demonstrates:
  - with_sliding_window(keep_messages, max_messages): drop oldest messages
    once the history exceeds the trigger, preserving pairing

All values come from the environment (no hardcoded numbers):

    COMPACTION_SLIDING_MAX_MESSAGES / COMPACTION_SLIDING_KEEP_MESSAGES
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 02_sliding_window.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")

SLIDING_MAX_MESSAGES = int(os.getenv("COMPACTION_SLIDING_MAX_MESSAGES", "12"))
SLIDING_KEEP_MESSAGES = int(os.getenv("COMPACTION_SLIDING_KEEP_MESSAGES", "6"))


async def main():
    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_sliding_window(
            keep_messages=SLIDING_KEEP_MESSAGES, max_messages=SLIDING_MAX_MESSAGES
        )
    )
    log.debug(
        "compaction_config",
        strategy="sliding_window",
        max_messages=SLIDING_MAX_MESSAGES,
        keep_messages=SLIDING_KEEP_MESSAGES,
    )

    session = "compaction-sliding-demo"
    topics = ["apples", "bridges", "clouds", "deserts", "engines", "forests"]
    for i, topic in enumerate(topics, start=1):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(
            f"Tell me one fact about {topic}.",
            history,
            session,
            save_to=[memory],
        )
        log.debug("turn", number=i, output=str(result.output)[:100])


if __name__ == "__main__":
    asyncio.run(main())
