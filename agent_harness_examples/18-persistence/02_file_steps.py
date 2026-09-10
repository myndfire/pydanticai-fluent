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

"""File step store — durable run history across processes.

Demonstrates:
  - with_file_steps(directory, agent_name): directory-backed store that
    survives process restarts
  - list_unresolved_tool_effects: check for tools left `started` with no
    terminal update before deciding to resume
  - fork_run(store, run_id=...): branch a fresh attempt from a snapshot

All values come from the environment (no hardcoded numbers):

    PERSISTENCE_FILE_DIRECTORY / PERSISTENCE_AGENT_NAME
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 02_file_steps.py
"""

import asyncio
import os
import tempfile

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.persistence import fork_run, make_file_store, make_step_persistence
from agent_harness.prompts import StaticPrompts


load_dotenv()
log = structlog.get_logger()

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")
AGENT_NAME = os.getenv("PERSISTENCE_AGENT_NAME", "persistence-demo")
DIRECTORY = os.getenv("PERSISTENCE_FILE_DIRECTORY") or os.path.join(
    tempfile.gettempdir(), "step-persistence-demo"
)


async def main():
    memory = InMemoryProvider()
    store = make_file_store(DIRECTORY)
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_step_persistence(
            make_step_persistence(store, agent_name=AGENT_NAME)
        )
    )
    log.debug("store", directory=DIRECTORY)

    session = "persistence-file-demo"
    history = await MessageHistory().load(session, memory)
    result = await agent.run(
        "Name one planet with rings.", history, session, save_to=[memory]
    )
    log.debug("turn_1", output=str(result.output)[:100])

    runs = await store.list_runs(conversation_id=session)
    run_id = runs[-1].run_id
    unresolved = await store.list_unresolved_tool_effects(run_id=run_id)
    log.debug("unresolved_tool_effects", count=len(unresolved))

    # Branch a fresh attempt from the recorded snapshot.
    branched = MessageHistory()
    branched.messages.extend(await fork_run(store, run_id=run_id))
    result2 = await agent.run(
        "Name a different ringed planet.", branched, session, save_to=[memory]
    )
    log.debug("turn_2", output=str(result2.output)[:100])


if __name__ == "__main__":
    asyncio.run(main())
