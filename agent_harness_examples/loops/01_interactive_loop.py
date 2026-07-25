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

"""Interactive loop — a conversational REPL with persistent memory.

This example demonstrates the most common agent loop pattern: a continuous
back-and-forth chat where the agent remembers the full conversation across
multiple turns. It uses a simple ``while True`` loop that prompts the user,
loads the conversation history, runs the agent, and prints the response.

Key Concepts Demonstrated
-----------------------
- **Persistent Memory**: Each turn loads ``MessageHistory`` from an
  ``InMemoryProvider`` so the agent retains full conversational context.
- **Session Management**: A fixed ``session_id`` (``"interactive-session"``)
  ties all turns together in the memory provider.
- **Graceful Exit**: The loop exits cleanly when the user types ``quit``,
  ``exit``, or ``bye`` (case-insensitive).
- **Async I/O**: Uses ``asyncio`` and ``await`` for all agent operations,
  matching the async-first design of ``ManagedAgent``.

What You Will See
-----------------
When you run the script, it prints a banner and waits for your input::

    $ uv run python loops/01_interactive_loop.py
    ============================================================
    Interactive Agent Loop
    Model: qwen2.5:3b
    ============================================================
    Type 'quit', 'exit', or 'bye' to end the conversation.

    You: What is the capital of France?
      [turn 1] thinking...
    Agent: The capital of France is Paris.

    You: What is its population?
      [turn 2] thinking...
    Agent: Paris has a population of approximately 2.1 million.

    You: quit
    Agent: Goodbye! Have a great day.

Because the agent loads history each turn, it correctly interprets
"its" in the second question as referring to Paris.

Architecture
------------
::

    User input
        │
        ▼
    Load MessageHistory (prior turns)
        │
        ▼
    ManagedAgent.run(prompt, history, session_id)
        │
        ├──► LLM processes prompt + history
        │
        └──► Result saved to InMemoryProvider
        │
        ▼
    Print result.output
        │
        ▼
    Loop back to user input

Configuration
-------------
The model name is read from the ``MODEL_NAME`` environment variable (set in
``.env``). It defaults to ``qwen2.5:3b``. Change it by editing ``.env``::

    MODEL_NAME=qwen3.5:9b

Usage
-----
Run from the ``agent_harness_examples`` directory::

    uv run python loops/01_interactive_loop.py

Setup
-----
1. Start Ollama (or your preferred local LLM server)::

       ollama serve

2. Install dependencies::

       cd agent_harness_examples
       uv sync

3. (Optional) Edit ``.env`` to change the model.

Tips
----
- Empty inputs are ignored so you don't waste a turn by pressing Enter.
- ``EOFError`` and ``KeyboardInterrupt`` (Ctrl-C) are caught and trigger a
  graceful goodbye message.
- The ``InMemoryProvider`` is ephemeral; conversation history disappears when
  the process exits. Swap it for ``MongoMemory`` or ``RedisMemory`` to make
  sessions persistent across restarts.
"""

import os
import asyncio

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))


async def main():
    print("=" * 60)
    print("Interactive Agent Loop")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

    memory = InMemoryProvider()
    session_id = "interactive-session"

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_prompts(
            StaticPrompts(
                "You are a helpful assistant. "
                "Keep responses concise and conversational."
            )
        )
        .with_observability(Observability())
        .with_short_term_memory(memory)
    )

    turn = 0
    while True:
        # --- Prompt user ---
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # --- Exit conditions ---
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Agent: Goodbye! Have a great day.")
            print("\n--- Session Summary ---")
            print(f"  Total turns: {turn}")
            print(f"  Memory retained: {turn_count} turn(s) in InMemoryProvider")
            print("  Concepts: persistent memory, session management, async I/O")
            break

        if not user_input:
            continue

        # --- Load conversation history ---
        history = await MessageHistory().load(session_id, memory)
        turns = await memory.load_turns(session_id)
        turn_count = len(turns)
        print(f"  [Memory] Loaded {turn_count} prior turn(s)")

        # --- Run agent ---
        turn += 1
        print(f"  [turn {turn}] thinking...")
        result = await agent.run(user_input, history, session_id, model_settings={"max_tokens": MAX_TOKENS}, save_to=[memory])

        # --- Print response ---
        print(f"Agent: {result.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
