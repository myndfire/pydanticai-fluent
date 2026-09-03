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
    docker compose -f docker-compose.yml up -d elasticsearch

Usage:
    uv run python 08_elasticsearch_memory.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start Elasticsearch:
        docker compose -f docker-compose.yml up -d elasticsearch
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/08_elasticsearch_memory.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, ElasticsearchMemory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))
ES_ENDPOINT = os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "agent-memory")


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
    """Run the ElasticsearchMemory example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start Elasticsearch: docker compose -f docker-compose.yml up -d elasticsearch
        4. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="ElasticsearchMemory — Elasticsearch Long-Term Memory")
    log.debug("separator", width=60)

    # ── Connection check ────────────────────────────────────────
    log.debug("checking", service="Elasticsearch", endpoint=ES_ENDPOINT)
    if not await check_elasticsearch(ES_ENDPOINT):
        log.debug("not_reachable", service="Elasticsearch", endpoint=ES_ENDPOINT)
        log.debug("start_hint", command="docker compose -f docker-compose.yml up -d elasticsearch")
        return
    log.debug("reachable", service="Elasticsearch")

    # ── Setup providers ─────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "10")))
    es_mem = ElasticsearchMemory(
        endpoint=ES_ENDPOINT,
        index=ES_INDEX,
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(short_term)
        .with_long_term_memory(es_mem)
    )

    log.debug("short_term", provider=f"InMemoryProvider(max_turns=10)")
    log.debug("long_term", provider=f"ElasticsearchMemory({ES_ENDPOINT}, index={ES_INDEX})")
    log.debug("auto_index", index=ES_INDEX, mappings="session_id: keyword, turn_id: keyword, timestamp: date, turn_data: object")
    log.debug("document_id_format", format="{session_id}:{turn_id}")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "es-demo"
    conversations = [
        "Remember this fact: the speed of light is 299,792,458 m/s.",
        "What is the capital of France? Just the city name.",
        "What was the scientific fact I told you about light?",
        "Now tell me: what year did World War II end?",
    ]

    log.debug("section", title=f"Multi-turn conversation (session: {session})")
    for i, prompt in enumerate(conversations, 1):
        history = await MessageHistory().load(session, short_term)
        result = await agent.run(
            prompt,
            history,
            session,
            save_to=[short_term, es_mem],
        )
        log.debug("turn", index=i, output=result.output)

    # ── Context restoration ─────────────────────────────────────
    log.debug("section", title="Context restoration (new agent, load from Elasticsearch)")
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )
    restored_history = await MessageHistory().load(session, es_mem)
    log.debug("loaded", source="Elasticsearch", messages=len(restored_history.messages))

    result_restore = await agent2.run(
        "Based on our conversation history, summarize the facts "
        "I've shared with you so far.",
        restored_history,
        session,
    )
    log.debug("response", output=result_restore.output)

    # ── CRUD operations ─────────────────────────────────────────
    log.debug("section", title="CRUD operations")

    turns = await es_mem.load_turns(session)
    log.debug("load_turns", count=len(turns), sorted_by="timestamp asc")

    if turns:
        # get_turn by document ID
        target = turns[1]
        fetched = await es_mem.get_turn(session, target.turn_id)
        log.debug("get_turn", turn_id=target.turn_id[:12], found=fetched is not None)

        # delete_turn
        deleted = await es_mem.delete_turn(session, target.turn_id)
        log.debug("delete_turn", result=deleted)

        # Verify via search
        remaining = await es_mem.load_turns(session)
        log.debug("after_delete", turns=len(remaining))

    # ── Cleanup ─────────────────────────────────────────────────
    log.debug("section", title="Cleanup")
    answer = input("Delete demo data from Elasticsearch? (y/n) ").strip().lower()
    if answer == "y":
        await es_mem.clear(session)
        remaining = await es_mem.load_turns(session)
        log.debug("remaining", session=session, count=len(remaining))
    else:
        log.debug("skipped_cleanup")


if __name__ == "__main__":
    asyncio.run(main())
