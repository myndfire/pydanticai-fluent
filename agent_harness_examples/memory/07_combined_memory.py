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

"""Combined memory — Redis (short-term) + MongoDB (long-term).

Demonstrates:
  - RedisMemory for fast short-term context retrieval
  - MongoMemory for durable long-term archival
  - Context union: run() loads from BOTH providers, giving the agent
    the combined conversation history
  - Cross-provider CRUD: delete from Redis, verify MongoDB still has it
  - Provider comparison: identical turn counts across both stores

Prerequisite:
    docker compose -f agent_harness_examples/memory/docker-compose.yml up -d

Usage:
    python 07_combined_memory.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import (
    InMemoryProvider,
    MessageHistory,
    MongoMemory,
    RedisMemory,
)
from agent_harness.model_config import ModelConfig

MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "agent_memory"
MONGO_COLLECTION = "conversations"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_KEY_PREFIX = "agent:memory:"


async def check_mongo(uri: str) -> bool:
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=2000)
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def check_redis(host: str, port: int) -> bool:
    try:
        import redis.asyncio as redis
        r = redis.Redis(host=host, port=port, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("Combined Memory — Redis (short) + MongoDB (long)")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking MongoDB at {MONGO_URI} ...")
    mongo_ok = await check_mongo(MONGO_URI)
    print(f"  {'Reachable' if mongo_ok else 'NOT reachable'}")

    print(f"Checking Redis at {REDIS_HOST}:{REDIS_PORT} ...")
    redis_ok = await check_redis(REDIS_HOST, REDIS_PORT)
    print(f"  {'Reachable' if redis_ok else 'NOT reachable'}")

    if not (mongo_ok and redis_ok):
        print("\n  Both services required. Start with:")
        print("    docker compose -f agent_harness_examples/memory/docker-compose.yml up -d")
        return

    # ── Setup providers ─────────────────────────────────────────
    redis_mem = RedisMemory(
        host=REDIS_HOST,
        port=REDIS_PORT,
        key_prefix=REDIS_KEY_PREFIX,
    )
    mongo = MongoMemory(
        uri=MONGO_URI,
        database=MONGO_DATABASE,
        collection=MONGO_COLLECTION,
    )

    # Redis = short-term (fast reads), Mongo = long-term (durable archive)
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(redis_mem)
        .with_long_term_memory(mongo)
    )

    print(f"\n  Short-term: RedisMemory ({REDIS_HOST}:{REDIS_PORT})")
    print(f"  Long-term:  MongoMemory ({MONGO_URI})")
    print(f"  On each run(): agent loads context from BOTH providers (union)")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "combined-demo"
    conversations = [
        "Remember: the launch code is 'Sierra-7-Alpha'.",
        "What is 100 divided by 4? Just the number.",
        "What was the launch code I told you?",
        "Add 50 to the number you calculated earlier. What is the total?",
    ]

    print(f"\n--- Multi-turn conversation (session: {session}) ---")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, redis_mem)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[redis_mem, mongo],
        )
        print(f"  Turn {i}: {result.output}")

    # ── Provider comparison ─────────────────────────────────────
    print("\n--- Provider comparison ---")
    redis_turns = await redis_mem.load_turns(session)
    mongo_turns = await mongo.load_turns(session)
    print(f"  Redis turns: {len(redis_turns)}")
    print(f"  MongoDB turns: {len(mongo_turns)}")

    # ── Context union verification ──────────────────────────────
    print("\n--- Context union: load only from MongoDB ---")
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    )
    # Load from Mongo only — proves Redis is not required for persistence
    mongo_only_history = await MessageHistory().load(session, mongo)
    print(f"  Messages from MongoDB alone: {len(mongo_only_history.messages)}")

    result_union = await agent2.run(
        "Based on everything we discussed, what was the launch code "
        "and what math result did you calculate?",
        mongo_only_history,
        session,
    )
    print(f"  Response: {result_union.output}")

    # ── Cross-provider CRUD ─────────────────────────────────────
    print("\n--- Cross-provider: delete from Redis, verify MongoDB still has it ---")
    if mongo_turns:
        target_id = mongo_turns[0].turn_id
        redis_deleted = await redis_mem.delete_turn(session, target_id)
        print(f"  Deleted from Redis: {redis_deleted}")
        mongo_still = await mongo.get_turn(session, target_id)
        print(f"  MongoDB still has it: {mongo_still is not None}")
        print(f"  Redis turns after delete: {len(await redis_mem.load_turns(session))}")
        print(f"  MongoDB turns unchanged:  {len(await mongo.load_turns(session))}")

    # ── Cleanup ─────────────────────────────────────────────────
    print("\n--- Cleanup ---")
    answer = input("Delete demo data from both providers? (y/n) ").strip().lower()
    if answer == "y":
        await redis_mem.clear(session)
        await mongo.clear(session)
        print(f"  Redis '{session}' remaining: {len(await redis_mem.load_turns(session))}")
        print(f"  MongoDB '{session}' remaining: {len(await mongo.load_turns(session))}")
    else:
        print("  Skipped cleanup. Data preserved.")


if __name__ == "__main__":
    asyncio.run(main())
