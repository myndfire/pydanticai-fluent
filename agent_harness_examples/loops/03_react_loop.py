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

"""ReAct loop — Reason, Act, Observe cycles driven by the agent.

This example demonstrates the classic **ReAct** (Reason → Act → Observe)
pattern where the *agent itself* drives the loop. Given a multi-step task,
the LLM reasons about what information it needs, calls a tool to obtain it,
observes the result, and repeats until it has enough data to synthesize a
final answer. The entire cycle happens inside **pydantic-ai's built-in
tool loop**; the outer Python code makes a single ``agent.run()`` call.

Key Concepts Demonstrated
-------------------------
- **Agent-Driven Loop**: The LLM decides *which* tool to call, *when* to
call it, and *when* to stop — no external loop logic required.
- **Multi-Step Reasoning**: The task requires three weather lookups and one
calculation; the agent sequences these automatically.
- **Tool Registry**: Plain functions are registered via ``ToolRegistry`` and
attached to the agent with ``.with_tools()``.
- **Context Preservation**: ``MessageHistory`` ensures the agent remembers
intermediate observations (e.g., "Tokyo is 22°C") across internal turns.

What You Will See
-----------------
When you run the script, it prints the task and then shows each internal
tool invocation as the agent works through the problem::

    $ uv run python loops/03_react_loop.py
    ============================================================
    ReAct Loop — Reason, Act, Observe
    Model: qwen3.5:4b
    ============================================================

    Task: What is the average temperature (in °C) of Tokyo, London, and New York?

    Starting ReAct loop (agent will reason → act → observe → repeat)...

        [tool:get_weather] city: Tokyo
        [tool:get_weather] city: London
        [tool:get_weather] city: New York
        [tool:calculator] expression: (22 + 15 + 18) / 3

    ============================================================
    FINAL ANSWER
    ============================================================
    The average temperature of Tokyo (22°C), London (15°C), and New York (18°C)
    is 18.33°C.

Architecture
------------
::

    User task (single prompt)
        │
        ▼
    ┌──────────────────────────────────────┐
    │  pydantic-ai internal tool loop        │
    │  (driven by LLM reasoning)             │
    │                                      │
    │  LLM: "I need Tokyo's weather"        │
    │      │                               │
    │      ▼                               │
    │  Tool call: get_weather("Tokyo")    │
    │      │                               │
    │      ▼                               │
    │  Observation: "Clear skies, 22°C"    │
    │      │                               │
    │  LLM: "Now I need London..."         │
    │      │                               │
    │      ▼                               │
    │  (repeat for London, New York)      │
    │      │                               │
    │  LLM: "I have all data; calculate"  │
    │      │                               │
    │      ▼                               │
    │  Tool call: calculator(...)          │
    │      │                               │
    │      ▼                               │
    │  LLM: "Done. Final answer: ..."     │
    │      │                               │
    └──────┼───────────────────────────────┘
           │
           ▼
    AgentRunResult.output

Configuration
-------------
- Model name is read from ``MODEL_NAME`` in ``.env`` (defaults to
  ``qwen3.5:4b``).
- The ``get_weather`` and ``calculator`` tools are simulated (no external
  APIs). Add real API calls inside the tool functions for production use.

Usage
-----
Run from the ``agent_harness_examples`` directory::

    uv run python loops/03_react_loop.py

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
- If the model calls tools in the wrong order or skips a step, improve the
  system prompt in ``StaticPrompts`` to be more explicit about the required
  sequence.
- You can add an ``InMemoryTracer`` to capture internal spans and inspect
  exactly how many reasoning cycles occurred.
- For tasks with many steps, consider increasing the model's
  ``max_tokens`` via ``model_settings`` on ``agent.run()``.
"""

import os
import asyncio

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability
from agent_harness.tools import ToolRegistry


load_dotenv()

MODEL_NAME = os.getenv("REACT_MODEL_NAME", "qwen3.5:4b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


# ── Tools ────────────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    """Get the current weather for a city (simulated).

    Args:
        city: Name of the city to check weather for.
    """
    print(f"    [tool:get_weather] city: {city}")
    conditions = {
        "tokyo": "Clear skies, 22°C",
        "london": "Light rain, 15°C",
        "new york": "Partly cloudy, 18°C",
        "paris": "Sunny, 20°C",
        "sydney": "Windy, 25°C",
    }
    return conditions.get(city.lower(), f"Unknown city: {city}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '(22 + 15 + 18) / 3'.
    """
    print(f"    [tool:calculator] expression: {expression}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("ReAct Loop — Reason, Act, Observe")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    memory = InMemoryProvider()
    session_id = "react-session"

    tools = ToolRegistry().add_many(get_weather, calculator)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_prompts(
            StaticPrompts(
                "You are a helpful assistant with access to weather and calculator tools. "
                "You MUST use these tools to answer questions. Do not guess or make up answers. "
                "Call get_weather for each city, then call calculator to compute the average. "
                "Always use the tools — never provide information from memory. "
                "Show your reasoning briefly, then give the final result clearly."
            )
        )
        .with_observability(Observability())
        .with_short_term_memory(memory)
        .with_tools(tools)
    )

    task = (
        "What is the average temperature (in °C) of Tokyo, London, and New York? "
        "Use the weather tool for each city, then calculate the average."
    )

    print(f"\nTask: {task}\n")
    print("Starting ReAct loop (agent will reason → act → observe → repeat)...\n")

    history = await MessageHistory().load(session_id, memory)
    turns = await memory.load_turns(session_id)
    turn_count = len(turns)
    print(f"  [Memory] Loaded {turn_count} prior turn(s)")

    # A single agent.run() call triggers the internal ReAct loop:
    #   LLM reasons → decides to call tool → tool executes →
    #   LLM observes → decides next tool or final answer → repeat
    result = await agent.run(task, history, session_id, model_settings={"max_tokens": MAX_TOKENS}, save_to=[memory])

    print("\n--- ReAct cycle complete (single agent.run() call) ---")

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER")
    print(f"{'=' * 60}")
    print(result.output)

    print(f"\n{'=' * 60}")
    print("CONCEPTS DEMONSTRATED")
    print(f"{'=' * 60}")
    print("✓ Agent-driven loop: LLM decided which tools to call and when")
    print("✓ Multi-step reasoning: 3 weather lookups + 1 calculation")
    print("✓ Tool registry: plain functions registered with .with_tools()")
    print("✓ Context preservation: intermediate results saved to MessageHistory")
    print("✓ Single agent.run() triggered all internal reasoning cycles")


if __name__ == "__main__":
    asyncio.run(main())
