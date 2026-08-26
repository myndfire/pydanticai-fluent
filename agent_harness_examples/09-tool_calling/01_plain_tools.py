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

"""Plain function tools — register stateless callables the agent can invoke.

Demonstrates:
  - ToolRegistry.add() for a single tool
  - ToolRegistry.add_many() for bulk registration
  - Tool function signatures with type hints drive the schema
  - Fluent chaining: .add(tool1).add_many(tool2, tool3)

Plain tools are registered via pydantic_ai's agent.tool_plain().
They do NOT receive RunContext or agent dependencies.

Usage:
    uv run python 01_plain_tools.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python tools/01_plain_tools.py
"""

import asyncio
import json
import math
import os
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry

load_dotenv()

MODEL_NAME = os.getenv("TOOL_CALLING_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("TOOL_CALLING_MAX_TOKENS", "512"))


# ── Tool definitions ────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression like '2 + 3 * 4'.
    """
    print(f"  [tool:calculator] evaluating: {expression}")
    try:
        allowed = set("0123456789+-*/().% sqrtpi")
        sanitized = expression.lower()
        sanitized = sanitized.replace("sqrt", "math.sqrt")
        sanitized = sanitized.replace("pi", "math.pi")
        result = eval(sanitized, {"__builtins__": {}}, {"math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get the current weather for a city (simulated).

    Args:
        city: Name of the city to check weather for.
    """
    print(f"  [tool:get_weather] city: {city}")
    conditions = {
        "new york": "Partly cloudy, 72F",
        "london": "Rain, 58F",
        "tokyo": "Clear, 68F",
        "sydney": "Sunny, 80F",
    }
    return conditions.get(city.lower(), f"Unknown city: {city}")


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (simulated).

    Args:
        amount: The amount to convert.
        from_currency: Three-letter currency code (e.g., USD).
        to_currency: Three-letter currency code (e.g., EUR).
    """
    print(f"  [tool:convert_currency] {amount} {from_currency} -> {to_currency}")
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
    }
    key = (from_currency.upper(), to_currency.upper())
    rate = rates.get(key)
    if rate is None:
        return f"No exchange rate found for {from_currency} -> {to_currency}"
    converted = amount * rate
    return f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    """Run the plain function tools example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Plain Function Tools")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Single tool with .add() ──────────────────────
    print("\n--- Example 1: Single tool ---")
    tools1 = ToolRegistry().add(calculator)

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(tools1)
    )

    history1 = await MessageHistory().load("tools-ex1", memory)
    result1 = await agent1.run(
        "What is the square root of 144 divided by 2? Use the calculator.",
        history1,
        "tools-ex1",
    )
    print(f"  Output: {result1.output}")

    # ── Example 2: Multiple tools with .add_many() ──────────────
    print("\n--- Example 2: add_many() - multiple tools ---")
    tools2 = ToolRegistry().add_many(calculator, get_weather, convert_currency)

    print(f"  Registered {len(tools2.get_tools())} tools:")
    for t in tools2.get_tools():
        print(f"    - {t.__name__}")

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(tools2)
    )

    history2 = await MessageHistory().load("tools-ex2", memory)
    result2 = await agent2.run(
        "What's the weather in Tokyo? Also, convert 100 USD to JPY.",
        history2,
        "tools-ex2",
    )
    print(f"  Output: {result2.output}")

    # ── Example 3: Fluent chaining (add + add_many) ─────────────
    print("\n--- Example 3: Fluent chaining ---")
    tools3 = (
        ToolRegistry()
        .add(calculator)
        .add_many(get_weather, convert_currency)
    )
    print(f"  Registered {len(tools3.get_tools())} tools via chaining")

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(tools3)
    )

    history3 = await MessageHistory().load("tools-ex3", memory)
    result3 = await agent3.run(
        "Calculate 42 * 3, then tell me the weather in London.",
        history3,
        "tools-ex3",
    )
    print(f"  Output: {result3.output}")

    # ── Example 4: Clear and rebuild tools ──────────────────────
    print("\n--- Example 4: clear() then rebuild ---")
    tools4 = ToolRegistry().add_many(calculator, get_weather)
    print(f"  Before clear: {len(tools4.get_tools())} tools")
    tools4.clear()
    print(f"  After clear: {len(tools4.get_tools())} tools")
    tools4.add(convert_currency)
    print(f"  After re-add: {len(tools4.get_tools())} tools")


if __name__ == "__main__":
    asyncio.run(main())
