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
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

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
    print("=" * 60)
    print("InMemoryProvider — Short & Long Term Memory")
    print("=" * 60)

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

    print(f"\nShort-term: InMemoryProvider(max_turns={short_term._max_turns})")
    print(f"Long-term:  InMemoryProvider(max_turns={long_term._max_turns})")

    # ── Turn 1 ──────────────────────────────────────────────────
    print("\n--- Turn 1: Greeting ---")
    history = await MessageHistory().load("session-42", short_term)
    result = await agent.run(
        "My name is Alice and I live in Portland.",
        history,
        "session-42",
        save_to=[short_term, long_term],   # persist to both
    )
    print(f"  Output: {result.output}")
    print(f"  last_turn.status: {agent.last_turn.status}")
    print(f"  last_turn.model: {agent.last_turn.model}")
    if agent.last_turn.usage:
        u = agent.last_turn.usage
        print(f"  last_turn.usage: in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")

    # ── Turn 2 ──────────────────────────────────────────────────
    print("\n--- Turn 2: Recall (uses history from Turn 1) ---")
    history2 = await MessageHistory().load("session-42", short_term)
    result2 = await agent.run(
        "What is my name and where do I live?",
        history2,
        "session-42",
        save_to=[short_term, long_term],
    )
    print(f"  Output: {result2.output}")
    print(f"  last_turn.duration: {agent.last_turn.duration_seconds:.2f}s")

    # ── Turn 3 ──────────────────────────────────────────────────
    print("\n--- Turn 3: More context building ---")
    history3 = await MessageHistory().load("session-42", short_term)
    result3 = await agent.run(
        "What did I tell you my name was? Also, add that I'm a software engineer.",
        history3,
        "session-42",
        save_to=[short_term, long_term],
    )
    print(f"  Output: {result3.output}")

    # ── Inspect stored turns ────────────────────────────────────
    print("\n--- Stored turns ---")
    short_turns = await short_term.load_turns("session-42")
    long_turns = await long_term.load_turns("session-42")
    print(f"  Short-term turns: {len(short_turns)}")
    print(f"  Long-term turns:  {len(long_turns)}")

    for i, turn in enumerate(short_turns, 1):
        msg_count = len(turn.messages)
        print(f"  Turn {i}: id={turn.turn_id[:8]}... "
              f"msgs={msg_count} status={turn.status} "
              f"dur={turn.duration_seconds:.2f}s")

    # ── max_turns trimming ──────────────────────────────────────
    print("\n--- max_turns trimming ---")
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
    print(f"  After 5 saves with max_turns=3: {len(turns)} turns stored")
    print(f"  Turn IDs: {[t.turn_id[:8] for t in turns]}")

    # ── Session isolation ───────────────────────────────────────
    print("\n--- Session isolation ---")
    session_a = await short_term.load_turns("session-a")
    session_b = await short_term.load_turns("session-b")
    print(f"  session-a turns: {len(session_a)} (should be 0)")
    print(f"  session-b turns: {len(session_b)} (should be 0)")
    print(f"  session-42 turns: {len(await short_term.load_turns('session-42'))}")


if __name__ == "__main__":
    asyncio.run(main())
