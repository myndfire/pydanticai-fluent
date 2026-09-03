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

"""InMemoryProvider — ephemeral memory for dev, testing, and short-term cache.

Demonstrates:
  - Short-term memory: fast cache for current conversation context
  - Long-term memory: separate store for persistent history
  - save_to: persist turn to one or more providers after each run()
  - last_turn property: inspect the most recent TurnData
  - max_turns: automatic trim to keep memory bounded
  - Session isolation: each session_id has its own turn list

Usage:
    uv run python 01_in_memory.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/01_in_memory.py
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
    """Run the InMemoryProvider example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="InMemoryProvider — Short & Long Term Memory")
    log.debug("separator", width=60)

    # ── Create providers ────────────────────────────────────────
    short_term = InMemoryProvider(max_turns=10)   # recent turns for context
    long_term = InMemoryProvider(max_turns=100)    # full conversation archive

    # ── Build agent with both ───────────────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
    )

    log.debug("providers", short_term=f"InMemoryProvider(max_turns={short_term._max_turns})")
    log.debug("providers", long_term=f"InMemoryProvider(max_turns={long_term._max_turns})")

    # ── Turn 1 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 1: Greeting")
    history = await MessageHistory().load("session-42", short_term)
    result = await agent.run(
        "My name is Alice and I live in Portland.",
        history,
        "session-42",
        save_to=[short_term, long_term],   # persist to both
    )
    log.debug("output", output=result.output)
    log.debug("last_turn_status", status=agent.last_turn.status)
    log.debug("last_turn_model", model=agent.last_turn.model)
    if agent.last_turn.usage:
        u = agent.last_turn.usage
        log.debug("usage", input_tokens=u.input_tokens, output_tokens=u.output_tokens, total_tokens=u.total_tokens)

    # ── Turn 2 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 2: Recall (uses history from Turn 1)")
    history2 = await MessageHistory().load("session-42", short_term)
    result2 = await agent.run(
        "What is my name and where do I live?",
        history2,
        "session-42",
        save_to=[short_term, long_term],
    )
    log.debug("output", output=result2.output)
    log.debug("duration", seconds=agent.last_turn.duration_seconds)

    # ── Turn 3 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 3: More context building")
    history3 = await MessageHistory().load("session-42", short_term)
    result3 = await agent.run(
        "What did I tell you my name was? Also, add that I'm a software engineer.",
        history3,
        "session-42",
        save_to=[short_term, long_term],
    )
    log.debug("output", output=result3.output)

    # ── Inspect stored turns ────────────────────────────────────
    log.debug("section", title="Stored turns")
    short_turns = await short_term.load_turns("session-42")
    long_turns = await long_term.load_turns("session-42")
    log.debug("turn_counts", short_term=len(short_turns), long_term=len(long_turns))

    for i, turn in enumerate(short_turns, 1):
        msg_count = len(turn.messages)
        log.debug("turn_detail", index=i, turn_id=turn.turn_id[:8], msgs=msg_count, status=turn.status, duration=turn.duration_seconds)

    # ── max_turns trimming ──────────────────────────────────────
    log.debug("section", title="max_turns trimming")
    small = InMemoryProvider(max_turns=3)
    # Generate 5 turns
    dummy_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
    )
    for i in range(1, 6):
        h = await MessageHistory().load("trim-test", small)
        await dummy_agent.run(
            f"Say 'turn {i}'.",
            h,
            "trim-test",
            save_to=[small],
        )
    turns = await small.load_turns("trim-test")
    log.debug("trim_result", turns_stored=len(turns), max_turns=3, total_saved=5)
    log.debug("turn_ids", ids=[t.turn_id[:8] for t in turns])

    # ── Session isolation ───────────────────────────────────────
    log.debug("section", title="Session isolation")
    session_a = await short_term.load_turns("session-a")
    session_b = await short_term.load_turns("session-b")
    log.debug("session_turns", session_a=len(session_a), expected=0)
    log.debug("session_turns", session_b=len(session_b), expected=0)
    log.debug("session_turns", session_42=len(await short_term.load_turns('session-42')))


if __name__ == "__main__":
    asyncio.run(main())
