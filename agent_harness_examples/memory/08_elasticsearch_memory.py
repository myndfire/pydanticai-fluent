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

"""ElasticsearchMemory — persistent memory with full-text search.

Demonstrates:
  - Connection check with graceful fallback if Elasticsearch is unreachable
  - ElasticsearchMemory as long-term memory with InMemoryProvider for short-term
  - Auto-index creation with session_id, turn_id, timestamp mappings
  - Document ID format: {session_id}:{turn_id}
  - Multi-turn conversation persistence to Elasticsearch
  - CRUD operations: load_turns (with search), get_turn, delete_turn
  - Cleanup prompt

Prerequisite:
    docker compose -f agent_harness_examples/memory/docker-compose.elastic.yml up -d

Usage:
    uv run python 08_elasticsearch_memory.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, ElasticsearchMemory
from agent_harness.model_config import ModelConfig

ES_ENDPOINT = "http://localhost:9200"
ES_INDEX = "agent-memory"


async def check_elasticsearch(endpoint: str) -> bool:
    """Check if Elasticsearch is reachable."""
    try:
        from elasticsearch import AsyncElasticsearch

        es = AsyncElasticsearch([endpoint])
        info = await es.info()
        await es.close()
        return True
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("ElasticsearchMemory — Elasticsearch Long-Term Memory")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking Elasticsearch at {ES_ENDPOINT} ...")
    if not await check_elasticsearch(ES_ENDPOINT):
        print(f"  Elasticsearch not reachable at {ES_ENDPOINT}")
        print("  Start with:")
        print("    docker compose -f agent_harness_examples/memory/docker-compose.elastic.yml up -d")
        return
    print("  Elasticsearch is reachable.")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=10)
    es_mem = ElasticsearchMemory(
        endpoint=ES_ENDPOINT,
        index=ES_INDEX,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(es_mem)
    )

    print(f"  Short-term: InMemoryProvider(max_turns=10)")
    print(f"  Long-term:  ElasticsearchMemory({ES_ENDPOINT}, index={ES_INDEX})")
    print(f"  On first use, index '{ES_INDEX}' is auto-created with mappings:")
    print(f"    session_id: keyword")
    print(f"    turn_id:    keyword")
    print(f"    timestamp:  date")
    print(f"    turn_data:  object")
    print(f"  Document ID format: {{session_id}}:{{turn_id}}")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "es-demo"
    conversations = [
        "Remember this fact: the speed of light is 299,792,458 m/s.",
        "What is the capital of France? Just the city name.",
        "What was the scientific fact I told you about light?",
        "Now tell me: what year did World War II end?",
    ]

    print(f"\n--- Multi-turn conversation (session: {session}) ---")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, es_mem],
        )
        print(f"  Turn {i}: {result.output}")

    # ── Context restoration ─────────────────────────────────────
    print("\n--- Context restoration (new agent, load from Elasticsearch) ---")
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    )
    restored_history = await MessageHistory().load(session, es_mem)
    print(f"  Messages loaded from Elasticsearch: {len(restored_history.messages)}")

    result_restore = await agent2.run(
        "Based on our conversation history, summarize the facts "
        "I've shared with you so far.",
        restored_history,
        session,
    )
    print(f"  Response: {result_restore.output}")

    # ── CRUD operations ─────────────────────────────────────────
    print("\n--- CRUD operations ---")

    turns = await es_mem.load_turns(session)
    print(f"  load_turns: {len(turns)} turns (sorted by timestamp asc)")

    if turns:
        # get_turn by document ID
        target = turns[1]
        fetched = await es_mem.get_turn(session, target.turn_id)
        print(f"  get_turn({target.turn_id[:12]}...): "
              f"{'found' if fetched else 'NOT FOUND'}")

        # delete_turn
        deleted = await es_mem.delete_turn(session, target.turn_id)
        print(f"  delete_turn: {deleted}")

        # Verify via search
        remaining = await es_mem.load_turns(session)
        print(f"  After delete: {len(remaining)} turns remain")

    # ── Cleanup ─────────────────────────────────────────────────
    print("\n--- Cleanup ---")
    answer = input("Delete demo data from Elasticsearch? (y/n) ").strip().lower()
    if answer == "y":
        await es_mem.clear(session)
        remaining = await es_mem.load_turns(session)
        print(f"  Remaining turns for '{session}': {len(remaining)}")
    else:
        print("  Skipped cleanup. Data preserved.")


if __name__ == "__main__":
    asyncio.run(main())
