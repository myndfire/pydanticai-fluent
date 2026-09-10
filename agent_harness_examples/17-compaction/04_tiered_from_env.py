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

"""TieredCompaction from env — cheap tiers first, budget from shared keys.

Demonstrates:
  - with_tiered_compaction_from_env(tiers): the recommended default.
    Tiers stay explicit in code (cheap-to-expensive); the stop budget comes
    from shared ``HARNESS_COMPACTION_*`` env keys.
  - with_compaction(...) escape hatch for any upstream capability.

Env keys (shared, no hardcoded numbers):

    HARNESS_COMPACTION_TARGET_TOKENS / HARNESS_COMPACTION_TARGET_FRACTION
    HARNESS_COMPACTION_KEEP_PAIRS / HARNESS_COMPACTION_KEEP_MESSAGES
    HARNESS_COMPACTION_MAX_MESSAGES
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 04_tiered_from_env.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.compaction import (
    make_clear_tool_results,
    make_sliding_window,
)
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")


async def main():
    memory = InMemoryProvider()
    # NOTE: tier triggers are bypassed inside TieredCompaction (it drives
    # each tier's compact() directly), so max_messages=1 here is only a
    # placeholder to satisfy the constructor — the upstream-documented dummy.
    tiers = [
        make_clear_tool_results(
            keep_pairs=int(os.getenv("HARNESS_COMPACTION_KEEP_PAIRS", "2")),
            max_messages=1,
        ),
        make_sliding_window(
            keep_messages=int(os.getenv("HARNESS_COMPACTION_KEEP_MESSAGES", "8")),
            max_messages=1,
        ),
    ]
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_tiered_compaction_from_env(tiers)
    )
    log.debug(
        "compaction_config",
        strategy="tiered_from_env",
        tiers=[type(t).__name__ for t in tiers],
        target_tokens=os.getenv("HARNESS_COMPACTION_TARGET_TOKENS"),
        target_fraction=os.getenv("HARNESS_COMPACTION_TARGET_FRACTION"),
    )

    session = "compaction-tiered-demo"
    history = await MessageHistory().load(session, memory)
    result = await agent.run("Say hello in one sentence.", history, session)
    log.debug("output", text=str(result.output)[:120])


if __name__ == "__main__":
    asyncio.run(main())
