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

"""Memory CRUD operations — get, delete, clear, and load with limit.

Demonstrates:
  - load_turns(session_id, limit=N): retrieve the N most recent turns
  - get_turn(session_id, turn_id): fetch a specific turn by ID
  - delete_turn(session_id, turn_id): remove a single turn
  - clear(session_id): wipe all turns for a session
  - last_turn property for direct access to the most recent turn

Usage:
    uv run python 04_memory_operations.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/04_memory_operations.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))


async def main():
    """Run the memory CRUD operations example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="Memory CRUD Operations")
    log.debug("separator", width=60)

    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )

    session = "crud-demo"

    # ── Populate: save 5 turns ──────────────────────────────────
    log.debug("section", title="Populating 5 turns")
    turn_ids = []
    for i in range(1, 6):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(
            f"Say 'I am turn number {i}' and nothing else.",
            history,
            session,
            save_to=[memory],
        )
        turn_ids.append(agent.last_turn.turn_id)
        log.debug("turn_saved", index=i, turn_id=turn_ids[-1][:12], status=agent.last_turn.status)

    # ── load_turns with limit ───────────────────────────────────
    log.debug("section", title="load_turns with limit")
    all_turns = await memory.load_turns(session)
    log.debug("all_turns", count=len(all_turns))

    last_2 = await memory.load_turns(session, limit=2)
    log.debug("limited_turns", count=len(last_2))
    for t in last_2:
        log.debug("turn_id", turn_id=t.turn_id[:12])

    # ── get_turn by ID ──────────────────────────────────────────
    log.debug("section", title="get_turn by ID")
    target_id = turn_ids[2]  # third turn
    turn = await memory.get_turn(session, target_id)
    if turn:
        log.debug("found", turn_id=turn.turn_id[:12], status=turn.status)
        msg_count = len(turn.messages)
        log.debug("messages", count=msg_count)
    else:
        log.debug("not_found")

    # Also check for a non-existent turn ID
    missing = await memory.get_turn(session, "nonexistent-id")
    log.debug("missing_returns", value=missing)

    # ── delete_turn ─────────────────────────────────────────────
    log.debug("section", title="delete_turn")
    delete_id = turn_ids[1]  # second turn
    log.debug("before_delete", turns=len(await memory.load_turns(session)))

    deleted = await memory.delete_turn(session, delete_id)
    log.debug("deleted", turn_id=delete_id[:12], result=deleted)

    log.debug("after_delete", turns=len(await memory.load_turns(session)))

    # Verify it's gone
    gone = await memory.get_turn(session, delete_id)
    log.debug("verify_gone", is_none=gone is None)

    # Delete non-existent — no error
    not_deleted = await memory.delete_turn(session, "fake-id")
    log.debug("delete_fake", result=not_deleted, expected=False)

    # ── clear ───────────────────────────────────────────────────
    log.debug("section", title="clear session")
    log.debug("before_clear", turns=len(await memory.load_turns(session)))
    await memory.clear(session)
    log.debug("after_clear", turns=len(await memory.load_turns(session)))

    # ── Prove clear doesn't affect other sessions ───────────────
    log.debug("section", title="clear isolation")
    # Populate another session
    memory2 = InMemoryProvider()
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )
    for i in range(1, 4):
        h = await MessageHistory().load("session-keep", memory2)
        await agent2.run(f"Say 'keep {i}'.", h, "session-keep", save_to=[memory2])

    # Populate and clear a different session
    for i in range(1, 3):
        h = await MessageHistory().load("session-delete", memory2)
        await agent2.run(f"Say 'del {i}'.", h, "session-delete", save_to=[memory2])

    await memory2.clear("session-delete")

    keep_turns = await memory2.load_turns("session-keep")
    del_turns = await memory2.load_turns("session-delete")
    log.debug("session_turns", session="session-keep", count=len(keep_turns), expected=3)
    log.debug("session_turns", session="session-delete", count=len(del_turns), expected=0)

    # ── last_turn property ──────────────────────────────────────
    log.debug("section", title="last_turn property")
    h = await MessageHistory().load("last-turn-demo", memory)
    await agent.run("Say hello.", h, "last-turn-demo", save_to=[memory])
    last = agent.last_turn
    log.debug("last_turn", turn_id=last.turn_id[:12], status=last.status, timestamp=last.timestamp.isoformat(), not_none=last is not None)


if __name__ == "__main__":
    asyncio.run(main())
