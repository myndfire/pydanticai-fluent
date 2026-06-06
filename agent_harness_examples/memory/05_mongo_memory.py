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

"""MongoMemory — persistent long-term memory via MongoDB.

Demonstrates:
  - Connection check with graceful fallback if MongoDB is unreachable
  - MongoMemory as long-term memory with InMemoryProvider for short-term
  - Multi-turn conversation persistence to MongoDB
  - Context restoration — new agent instance loads history from MongoDB
  - CRUD operations: load_turns, get_turn, delete_turn
  - Cleanup prompt

Prerequisite:
    docker compose -f agent_harness_examples/memory/docker-compose.mongo.yml up -d

Usage:
    uv run python 05_mongo_memory.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, MongoMemory
from agent_harness.model_config import ModelConfig

MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "agent_memory"
MONGO_COLLECTION = "conversations"


async def check_mongo(uri: str) -> bool:
    """Check if MongoDB is reachable."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=2000)
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("MongoMemory — MongoDB Long-Term Memory")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking MongoDB at {MONGO_URI} ...")
    if not await check_mongo(MONGO_URI):
        print(f"  MongoDB not reachable at {MONGO_URI}")
        print("  Start with:")
        print("    docker compose -f agent_harness_examples/memory/docker-compose.mongo.yml up -d")
        return
    print("  MongoDB is reachable.")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=10)
    mongo = MongoMemory(
        uri=MONGO_URI,
        database=MONGO_DATABASE,
        collection=MONGO_COLLECTION,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(mongo)
    )

    print(f"  Short-term: InMemoryProvider(max_turns=10)")
    print(f"  Long-term:  MongoMemory({MONGO_URI}, db={MONGO_DATABASE}, coll={MONGO_COLLECTION})")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "mongo-demo"
    conversations = [
        "My name is Alice and I work at Acme Corp.",
        "What is 25 * 4? Just the number.",
        "What is my name and where do I work?",
    ]

    print(f"\n--- Multi-turn conversation (session: {session}) ---")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, mongo],
        )
        print(f"  Turn {i}: {result.output}")

    # ── Context restoration ─────────────────────────────────────
    print("\n--- Context restoration (new agent, load from MongoDB) ---")
    new_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    )
    # Load only from mongo — prove persistence works independently
    history_restored = await MessageHistory().load(session, mongo)
    msg_count = len(history_restored.messages)
    print(f"  Messages loaded from MongoDB: {msg_count}")

    result_restore = await new_agent.run(
        "Based on our conversation, what did I tell you my name was?",
        history_restored,
        session,
    )
    print(f"  Response: {result_restore.output}")

    # ── CRUD operations ─────────────────────────────────────────
    print("\n--- CRUD operations ---")

    turns = await mongo.load_turns(session)
    print(f"  load_turns: {len(turns)} turns")

    if turns:
        target = turns[1]  # second turn
        fetched = await mongo.get_turn(session, target.turn_id)
        print(f"  get_turn: {fetched.turn_id[:12] if fetched else 'NOT FOUND'}...")

        deleted = await mongo.delete_turn(session, target.turn_id)
        print(f"  delete_turn: {deleted}")
        print(f"  After delete: {len(await mongo.load_turns(session))} turns")

    # ── Cleanup ─────────────────────────────────────────────────
    print("\n--- Cleanup ---")
    answer = input("Delete demo data from MongoDB? (y/n) ").strip().lower()
    if answer == "y":
        await mongo.clear(session)
        remaining = await mongo.load_turns(session)
        print(f"  Remaining turns for '{session}': {len(remaining)}")
    else:
        print("  Skipped cleanup. Data preserved.")


if __name__ == "__main__":
    asyncio.run(main())
