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
    python 04_memory_operations.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


async def main():
    print("=" * 60)
    print("Memory CRUD Operations")
    print("=" * 60)

    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    )

    session = "crud-demo"

    # ── Populate: save 5 turns ──────────────────────────────────
    print("\n--- Populating 5 turns ---")
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
        print(f"  Turn {i} saved: {turn_ids[-1][:12]}... "
              f"status={agent.last_turn.status}")

    # ── load_turns with limit ───────────────────────────────────
    print("\n--- load_turns with limit ---")
    all_turns = await memory.load_turns(session)
    print(f"  All turns: {len(all_turns)}")

    last_2 = await memory.load_turns(session, limit=2)
    print(f"  Last 2: {len(last_2)}")
    for t in last_2:
        print(f"    {t.turn_id[:12]}...")

    # ── get_turn by ID ──────────────────────────────────────────
    print("\n--- get_turn by ID ---")
    target_id = turn_ids[2]  # third turn
    turn = await memory.get_turn(session, target_id)
    if turn:
        print(f"  Found: {turn.turn_id[:12]}... status={turn.status}")
        msg_count = len(turn.messages)
        print(f"  Messages: {msg_count}")
    else:
        print("  Not found")

    # Also check for a non-existent turn ID
    missing = await memory.get_turn(session, "nonexistent-id")
    print(f"  Missing ID returns: {missing}")

    # ── delete_turn ─────────────────────────────────────────────
    print("\n--- delete_turn ---")
    delete_id = turn_ids[1]  # second turn
    print(f"  Before delete: {len(await memory.load_turns(session))} turns")

    deleted = await memory.delete_turn(session, delete_id)
    print(f"  Deleted turn {delete_id[:12]}...: {deleted}")

    print(f"  After delete:  {len(await memory.load_turns(session))} turns")

    # Verify it's gone
    gone = await memory.get_turn(session, delete_id)
    print(f"  Verify gone: {gone is None}")

    # Delete non-existent — no error
    not_deleted = await memory.delete_turn(session, "fake-id")
    print(f"  Delete fake ID: {not_deleted} (should be False)")

    # ── clear ───────────────────────────────────────────────────
    print("\n--- clear session ---")
    print(f"  Before clear: {len(await memory.load_turns(session))} turns")
    await memory.clear(session)
    print(f"  After clear:  {len(await memory.load_turns(session))} turns")

    # ── Prove clear doesn't affect other sessions ───────────────
    print("\n--- clear isolation ---")
    # Populate another session
    memory2 = InMemoryProvider()
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
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
    print(f"  session-keep turns:   {len(keep_turns)} (should be 3)")
    print(f"  session-delete turns: {len(del_turns)} (should be 0)")

    # ── last_turn property ──────────────────────────────────────
    print("\n--- last_turn property ---")
    h = await MessageHistory().load("last-turn-demo", memory)
    await agent.run("Say hello.", h, "last-turn-demo", save_to=[memory])
    last = agent.last_turn
    print(f"  last_turn.turn_id:    {last.turn_id[:12]}...")
    print(f"  last_turn.status:     {last.status}")
    print(f"  last_turn.timestamp:  {last.timestamp.isoformat()}")
    print(f"  last_turn is not None: {last is not None}")


if __name__ == "__main__":
    asyncio.run(main())
