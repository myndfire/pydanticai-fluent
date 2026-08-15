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

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, RedisMemory
from agent_harness.model_config import ModelConfig

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_KEY_PREFIX = "agent:memory:"


async def check_redis(host: str, port: int) -> bool:
    """Check if Redis is reachable."""
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
    print("RedisMemory — Redis Long-Term Memory")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking Redis at {REDIS_HOST}:{REDIS_PORT} ...")
    if not await check_redis(REDIS_HOST, REDIS_PORT):
        print(f"  Redis not reachable at {REDIS_HOST}:{REDIS_PORT}")
        print("  Start with:")
        print("    docker compose -f docker-compose.yml up -d redis")
        return
    print("  Redis is reachable.")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=10)
    redis_mem = RedisMemory(
        host=REDIS_HOST,
        port=REDIS_PORT,
        key_prefix=REDIS_KEY_PREFIX,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(redis_mem)
    )

    print(f"  Short-term: InMemoryProvider(max_turns=10)")
    print(f"  Long-term:  RedisMemory({REDIS_HOST}:{REDIS_PORT}, key_prefix={REDIS_KEY_PREFIX})")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "redis-demo"
    conversations = [
        "Remember this: the project codename is 'Nightingale'.",
        "What is the capital of Japan? Just the city name.",
        "What was the project codename I told you?",
    ]

    print(f"\n--- Multi-turn conversation (session: {session}) ---")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, redis_mem],
        )
        print(f"  Turn {i}: {result.output}")

    # ── Key prefix inspection ───────────────────────────────────
    print(f"\n--- Key prefix inspection ---")
    print(f"  Prefix: {REDIS_KEY_PREFIX}")
    print(f"  This session's Redis key: {REDIS_KEY_PREFIX}{session}")
    print(f"  Each session gets its own Redis list, isolated by prefix.")

    # Verify isolation — another session should be empty
    other_turns = await redis_mem.load_turns("other-session")
    print(f"  'other-session' turns: {len(other_turns)} (should be 0)")

    # ── CRUD operations ─────────────────────────────────────────
    print("\n--- CRUD operations ---")

    turns = await redis_mem.load_turns(session)
    print(f"  load_turns: {len(turns)} turns")

    if turns:
        target = turns[1]
        fetched = await redis_mem.get_turn(session, target.turn_id)
        print(f"  get_turn: {fetched.turn_id[:12] if fetched else 'NOT FOUND'}...")

        deleted = await redis_mem.delete_turn(session, target.turn_id)
        print(f"  delete_turn: {deleted}")
        print(f"  After delete: {len(await redis_mem.load_turns(session))} turns")

    # ── Cleanup ─────────────────────────────────────────────────
    print("\n--- Cleanup ---")
    answer = input("Delete demo data from Redis? (y/n) ").strip().lower()
    if answer == "y":
        await redis_mem.clear(session)
        remaining = await redis_mem.load_turns(session)
        print(f"  Remaining turns for '{session}': {len(remaining)}")
    else:
        print("  Skipped cleanup. Data preserved.")


if __name__ == "__main__":
    asyncio.run(main())
