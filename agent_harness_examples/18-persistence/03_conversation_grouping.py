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

"""Conversation grouping — one dialogue, many runs, chronological lookup.

Demonstrates:
  - conversation_id defaults to session_id, so consecutive turns group
    automatically without caller changes
  - store.list_runs(conversation_id=...): all turns of a dialogue, oldest first
  - with_step_persistence_from_env(): backend + agent name from shared
    HARNESS_PERSISTENCE_* keys

Env keys (no hardcoded numbers):

    HARNESS_PERSISTENCE_BACKEND / HARNESS_PERSISTENCE_AGENT_NAME
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    HARNESS_PERSISTENCE_BACKEND=memory uv run python 03_conversation_grouping.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.persistence import make_memory_store, make_step_persistence
from agent_harness.prompts import StaticPrompts


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")


async def main():
    memory = InMemoryProvider()
    store = make_memory_store()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_step_persistence_from_env(store)
    )

    session = "persistence-conv-demo"
    for i, question in enumerate(
        ["Name a red fruit.", "Name a yellow fruit.", "Name a green fruit."],
        start=1,
    ):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(question, history, session, save_to=[memory])
        log.debug("turn", number=i, output=str(result.output)[:80])

    turns = await store.list_runs(conversation_id=session)
    log.debug(
        "dialogue",
        turns=len(turns),
        run_ids=[r.run_id for r in turns],
        note="three distinct run_ids, one conversation_id",
    )


if __name__ == "__main__":
    asyncio.run(main())
