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
    docker compose -f docker-compose.yml up -d mongo redis

Usage:
    uv run python 07_combined_memory.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start MongoDB:
        docker compose -f docker-compose.yml up -d mongo
    3. (Optional) Start Redis:
        docker compose -f docker-compose.yml up -d redis
    4. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/07_combined_memory.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import (
    InMemoryProvider,
    MessageHistory,
    MongoMemory,
    RedisMemory,
)
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.getenv("MONGODB_DATABASE", "agent_memory")
MONGO_COLLECTION = os.getenv("MONGODB_COLLECTION", "conversations")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "agent:memory:")


async def check_mongo(uri: str) -> bool:
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=int(os.getenv("MEMORY_MONGODB_TIMEOUT_MS", "2000")))
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def check_redis(host: str, port: int) -> bool:
    try:
        import redis.asyncio as redis
        r = redis.Redis(host=host, port=port, socket_connect_timeout=int(os.getenv("MEMORY_REDIS_CONNECT_TIMEOUT", "2")))
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def main():
    """Run the combined memory example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start MongoDB: docker compose -f docker-compose.yml up -d mongo
        4. Start Redis: docker compose -f docker-compose.yml up -d redis
        5. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="Combined Memory — Redis (short) + MongoDB (long)")
    log.debug("separator", width=60)

    # ── Connection check ────────────────────────────────────────
    log.debug("checking", service="MongoDB", uri=MONGO_URI)
    mongo_ok = await check_mongo(MONGO_URI)
    log.debug("reachable", service="MongoDB", ok=mongo_ok)

    log.debug("checking", service="Redis", host=REDIS_HOST, port=REDIS_PORT)
    redis_ok = await check_redis(REDIS_HOST, REDIS_PORT)
    log.debug("reachable", service="Redis", ok=redis_ok)

    if not (mongo_ok and redis_ok):
        log.debug("both_required", hint="docker compose -f docker-compose.yml up -d mongo redis")
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
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(redis_mem)
        .with_long_term_memory(mongo)
    )

    log.debug("short_term", provider=f"RedisMemory ({REDIS_HOST}:{REDIS_PORT})")
    log.debug("long_term", provider=f"MongoMemory ({MONGO_URI})")
    log.debug("context_union", message="On each run(): agent loads context from BOTH providers (union)")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "combined-demo"
    conversations = [
        "Remember: the launch code is 'Sierra-7-Alpha'.",
        "What is 100 divided by 4? Just the number.",
        "What was the launch code I told you?",
        "Add 50 to the number you calculated earlier. What is the total?",
    ]

    log.debug("section", title=f"Multi-turn conversation (session: {session})")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, redis_mem)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[redis_mem, mongo],
        )
        log.debug("turn", index=i, output=result.output)

    # ── Provider comparison ─────────────────────────────────────
    log.debug("section", title="Provider comparison")
    redis_turns = await redis_mem.load_turns(session)
    mongo_turns = await mongo.load_turns(session)
    log.debug("turn_counts", redis=len(redis_turns), mongodb=len(mongo_turns))

    # ── Context union verification ──────────────────────────────
    log.debug("section", title="Context union: load only from MongoDB")
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )
    # Load from Mongo only — proves Redis is not required for persistence
    mongo_only_history = await MessageHistory().load(session, mongo)
    log.debug("loaded", source="MongoDB", messages=len(mongo_only_history.messages))

    result_union = await agent2.run(
        "Based on everything we discussed, what was the launch code "
        "and what math result did you calculate?",
        mongo_only_history,
        session,
    )
    log.debug("response", output=result_union.output)

    # ── Cross-provider CRUD ─────────────────────────────────────
    log.debug("section", title="Cross-provider: delete from Redis, verify MongoDB still has it")
    if mongo_turns:
        target_id = mongo_turns[0].turn_id
        redis_deleted = await redis_mem.delete_turn(session, target_id)
        log.debug("deleted_from_redis", result=redis_deleted)
        mongo_still = await mongo.get_turn(session, target_id)
        log.debug("mongodb_still_has", exists=mongo_still is not None)
        log.debug("redis_turns_after_delete", count=len(await redis_mem.load_turns(session)))
        log.debug("mongodb_turns_unchanged", count=len(await mongo.load_turns(session)))

    # ── Cleanup ─────────────────────────────────────────────────
    log.debug("section", title="Cleanup")
    answer = input("Delete demo data from both providers? (y/n) ").strip().lower()
    if answer == "y":
        await redis_mem.clear(session)
        await mongo.clear(session)
        log.debug("remaining", provider="Redis", session=session, count=len(await redis_mem.load_turns(session)))
        log.debug("remaining", provider="MongoDB", session=session, count=len(await mongo.load_turns(session)))
    else:
        log.debug("skipped_cleanup")


if __name__ == "__main__":
    asyncio.run(main())
