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

"""MessageHistory — load, rebuild, and inspect conversation context.

Demonstrates:
  - MessageHistory.load(session_id, from_memory=provider) for context restoration
  - How prior turns are reconstructed as ModelRequest/ModelResponse objects
  - filter_thinking_parts() for serializing messages without internal parts
  - Multi-turn conversation context across separate run() calls
  - Inspecting message count and content in loaded history

Usage:
    uv run python 02_message_history.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/02_message_history.py
"""

import asyncio
import os
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import (
    InMemoryProvider,
    MessageHistory,
    filter_thinking_parts,
)
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))


async def main():
    """Run the MessageHistory example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("MessageHistory — Conversation Context")
    print("=" * 60)

    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
    )

    # ── Build a multi-turn conversation ─────────────────────────
    session = "msg-history-demo"

    conversations = [
        "My name is Bob and I'm a data scientist.",
        "What is 15 * 7? Just give me the number.",
        "Based on what I told you, what is my profession?",
        "Add 100 to the previous number you calculated. What is it?",
    ]

    print("\nBuilding conversation history...")
    for i, prompt in enumerate(conversations, 1):
        # Load prior context before each turn
        history = await MessageHistory().load(session, memory)
        msg_count = len(history.messages)
        print(f"\n--- Turn {i} ({msg_count} prior messages in context) ---")
        print(f"  Prompt: {prompt}")

        result = await agent.run(prompt, history, session, save_to=[memory])
        print(f"  Response: {result.output}")

    # ── Inspect the full history ────────────────────────────────
    full_history = await MessageHistory().load(session, memory)
    print(f"\n--- Full History: {len(full_history.messages)} messages ---")

    for i, msg in enumerate(full_history.messages):
        kind = msg.__class__.__name__
        if kind == "ModelRequest":
            content = [p.content for p in msg.parts if hasattr(p, "content")]
            print(f"  [{i}] Request: {content}")
        elif kind == "ModelResponse":
            content = [p.content for p in msg.parts if hasattr(p, "content")]
            print(f"  [{i}] Response: {content}")

    # ── filter_thinking_parts demonstration ─────────────────────
    print("\n--- filter_thinking_parts ---")
    serialized = filter_thinking_parts(full_history.messages)
    print(f"  Messages after filtering: {len(serialized)}")
    for m in serialized[:3]:
        kind = m.get("kind", "?")
        parts = m.get("parts", [])
        preview = [p.get("content", "")[:50] for p in parts[:2]]
        print(f"  {kind}: {preview}")

    # ── Fresh session (no prior context) ────────────────────────
    print("\n--- Fresh session (no history) ---")
    fresh_history = await MessageHistory().load("brand-new-session", memory)
    print(f"  Messages loaded: {len(fresh_history.messages)} (should be 0)")

    fresh_result = await agent.run(
        "Say hello in exactly 3 words.",
        fresh_history,
        "brand-new-session",
        save_to=[memory],
    )
    print(f"  Response: {fresh_result.output}")


if __name__ == "__main__":
    asyncio.run(main())
