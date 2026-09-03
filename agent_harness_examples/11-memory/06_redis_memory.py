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

"""RedisMemory — persistent memory via Redis with key prefix isolation.

Demonstrates:
  - Connection check with graceful fallback if Redis is unreachable
  - RedisMemory as long-term memory with InMemoryProvider for short-term
  - Key prefix isolation (agent:memory:<session_id>)
  - Multi-turn conversation persistence to Redis
  - CRUD operations: load_turns, get_turn, delete_turn, clear
  - Cleanup prompt

Prerequisite:
    docker compose -f docker-compose.yml up -d redis

Usage:
    uv run python 06_redis_memory.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start Redis:
        docker compose -f docker-compose.yml up -d redis
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/06_redis_memory.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, RedisMemory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "agent:memory:")


async def check_redis(host: str, port: int) -> bool:
    """Check if Redis is reachable."""
    try:
        import redis.asyncio as redis

        r = redis.Redis(host=host, port=port, socket_connect_timeout=int(os.getenv("MEMORY_REDIS_CONNECT_TIMEOUT", "2")))
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def main():
    """Run the RedisMemory example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start Redis: docker compose -f docker-compose.yml up -d redis
        4. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="RedisMemory — Redis Long-Term Memory")
    log.debug("separator", width=60)

    # ── Connection check ────────────────────────────────────────
    log.debug("checking", service="Redis", host=REDIS_HOST, port=REDIS_PORT)
    if not await check_redis(REDIS_HOST, REDIS_PORT):
        log.debug("not_reachable", service="Redis", host=REDIS_HOST, port=REDIS_PORT)
        log.debug("start_hint", command="docker compose -f docker-compose.yml up -d redis")
        return
    log.debug("reachable", service="Redis")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "10")))
    redis_mem = RedisMemory(
        host=REDIS_HOST,
        port=REDIS_PORT,
        key_prefix=REDIS_KEY_PREFIX,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(short_term)
        .with_long_term_memory(redis_mem)
    )

    log.debug("short_term", provider=f"InMemoryProvider(max_turns=10)")
    log.debug("long_term", provider=f"RedisMemory({REDIS_HOST}:{REDIS_PORT}, key_prefix={REDIS_KEY_PREFIX})")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "redis-demo"
    conversations = [
        "Remember this: the project codename is 'Nightingale'.",
        "What is the capital of Japan? Just the city name.",
        "What was the project codename I told you?",
    ]

    log.debug("section", title=f"Multi-turn conversation (session: {session})")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, redis_mem],
        )
        log.debug("turn", index=i, output=result.output)

    # ── Key prefix inspection ───────────────────────────────────
    log.debug("section", title="Key prefix inspection")
    log.debug("prefix", prefix=REDIS_KEY_PREFIX)
    log.debug("redis_key", key=f"{REDIS_KEY_PREFIX}{session}")
    log.debug("isolation_note", message="Each session gets its own Redis list, isolated by prefix.")

    # Verify isolation — another session should be empty
    other_turns = await redis_mem.load_turns("other-session")
    log.debug("isolation_check", session="other-session", turns=len(other_turns), expected=0)

    # ── CRUD operations ─────────────────────────────────────────
    log.debug("section", title="CRUD operations")

    turns = await redis_mem.load_turns(session)
    log.debug("load_turns", count=len(turns))

    if turns:
        target = turns[1]
        fetched = await redis_mem.get_turn(session, target.turn_id)
        log.debug("get_turn", turn_id=fetched.turn_id[:12] if fetched else "NOT FOUND")

        deleted = await redis_mem.delete_turn(session, target.turn_id)
        log.debug("delete_turn", result=deleted)
        log.debug("after_delete", turns=len(await redis_mem.load_turns(session)))

    # ── Cleanup ─────────────────────────────────────────────────
    log.debug("section", title="Cleanup")
    answer = input("Delete demo data from Redis? (y/n) ").strip().lower()
    if answer == "y":
        await redis_mem.clear(session)
        remaining = await redis_mem.load_turns(session)
        log.debug("remaining", session=session, count=len(remaining))
    else:
        log.debug("skipped_cleanup")


if __name__ == "__main__":
    asyncio.run(main())
