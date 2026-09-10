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

"""In-memory step persistence — record, inspect, and continue a run.

Demonstrates:
  - with_step_persistence(make_step_persistence(store, ...)): attach a
    process-local step store (great for tests)
  - store.list_runs / list_events: inspect the recorded run tree
  - continue_run(store, run_id=...): resume from the latest settled snapshot
    by seeding a fresh MessageHistory

All values come from the environment (no hardcoded numbers):

    PERSISTENCE_AGENT_NAME
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 01_memory_steps.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.persistence import (
    continue_run,
    make_memory_store,
    make_step_persistence,
)
from agent_harness.prompts import StaticPrompts


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")
AGENT_NAME = os.getenv("PERSISTENCE_AGENT_NAME", "persistence-demo")


async def main():
    memory = InMemoryProvider()
    store = make_memory_store()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_step_persistence(
            make_step_persistence(store, agent_name=AGENT_NAME)
        )
    )

    session = "persistence-memory-demo"
    history = await MessageHistory().load(session, memory)
    result = await agent.run(
        "Name one moon of Jupiter.", history, session, save_to=[memory]
    )
    log.debug("turn_1", output=str(result.output)[:100])

    # ── Inspect what was recorded ────────────────────────────────
    runs = await store.list_runs(conversation_id=session)
    log.debug("recorded_runs", count=len(runs), run_id=runs[-1].run_id)
    events = await store.list_events(run_id=runs[-1].run_id)
    log.debug("recorded_events", count=len(events))

    # ── Continue from the latest settled snapshot ────────────────
    continued = MessageHistory()
    continued.messages.extend(
        await continue_run(store, run_id=runs[-1].run_id)
    )
    result2 = await agent.run(
        "And one more moon, different from before.",
        continued,
        session,
        save_to=[memory],
    )
    log.debug("turn_2", output=str(result2.output)[:100])


if __name__ == "__main__":
    asyncio.run(main())
