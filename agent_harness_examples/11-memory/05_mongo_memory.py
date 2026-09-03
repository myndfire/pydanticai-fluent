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
    docker compose -f docker-compose.yml up -d mongo

Usage:
    uv run python 05_mongo_memory.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start MongoDB:
        docker compose -f docker-compose.yml up -d mongo
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/05_mongo_memory.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, MongoMemory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.getenv("MONGODB_DATABASE", "agent_memory")
MONGO_COLLECTION = os.getenv("MONGODB_COLLECTION", "conversations")


async def check_mongo(uri: str) -> bool:
    """Check if MongoDB is reachable."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=int(os.getenv("MEMORY_MONGODB_TIMEOUT_MS", "2000")))
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def main():
    """Run the MongoMemory example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start MongoDB: docker compose -f docker-compose.yml up -d mongo
        4. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="MongoMemory — MongoDB Long-Term Memory")
    log.debug("separator", width=60)

    # ── Connection check ────────────────────────────────────────
    log.debug("checking", service="MongoDB", uri=MONGO_URI)
    if not await check_mongo(MONGO_URI):
        log.debug("not_reachable", service="MongoDB", uri=MONGO_URI)
        log.debug("start_hint", command="docker compose -f docker-compose.yml up -d mongo")
        return
    log.debug("reachable", service="MongoDB")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "10")))
    mongo = MongoMemory(
        uri=MONGO_URI,
        database=MONGO_DATABASE,
        collection=MONGO_COLLECTION,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(short_term)
        .with_long_term_memory(mongo)
    )

    log.debug("short_term", provider=f"InMemoryProvider(max_turns=10)")
    log.debug("long_term", provider=f"MongoMemory({MONGO_URI}, db={MONGO_DATABASE}, coll={MONGO_COLLECTION})")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "mongo-demo"
    conversations = [
        "My name is Alice and I work at Acme Corp.",
        "What is 25 * 4? Just the number.",
        "What is my name and where do I work?",
    ]

    log.debug("section", title=f"Multi-turn conversation (session: {session})")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, mongo],
        )
        log.debug("turn", index=i, output=result.output)

    # ── Context restoration ─────────────────────────────────────
    log.debug("section", title="Context restoration (new agent, load from MongoDB)")
    new_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )
    # Load only from mongo — prove persistence works independently
    history_restored = await MessageHistory().load(session, mongo)
    msg_count = len(history_restored.messages)
    log.debug("loaded", source="MongoDB", messages=msg_count)

    result_restore = await new_agent.run(
        "Based on our conversation, what did I tell you my name was?",
        history_restored,
        session,
    )
    log.debug("response", output=result_restore.output)

    # ── CRUD operations ─────────────────────────────────────────
    log.debug("section", title="CRUD operations")

    turns = await mongo.load_turns(session)
    log.debug("load_turns", count=len(turns))

    if turns:
        target = turns[1]  # second turn
        fetched = await mongo.get_turn(session, target.turn_id)
        log.debug("get_turn", turn_id=fetched.turn_id[:12] if fetched else "NOT FOUND")

        deleted = await mongo.delete_turn(session, target.turn_id)
        log.debug("delete_turn", result=deleted)
        log.debug("after_delete", turns=len(await mongo.load_turns(session)))

    # ── Cleanup ─────────────────────────────────────────────────
    log.debug("section", title="Cleanup")
    answer = input("Delete demo data from MongoDB? (y/n) ").strip().lower()
    if answer == "y":
        await mongo.clear(session)
        remaining = await mongo.load_turns(session)
        log.debug("remaining", session=session, count=len(remaining))
    else:
        log.debug("skipped_cleanup")


if __name__ == "__main__":
    asyncio.run(main())
