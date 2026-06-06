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

"""Human-in-the-loop with multi-step tools — approve tool output between calls.

The human reviews BETWEEN tool calls, not just at the end. Each tool
prompts the human to approve or correct its output before returning to
the LLM. This gives the human fine-grained control over a multi-step
workflow.

Scenario
────────
You ask the agent to plan a trip. The agent calls three tools in sequence:

  1. get_flight_price("Tokyo")    → human sees the price, can approve or correct
  2. calculate_total(flight, hotel, days) → human sees the breakdown, can approve or correct
  3. apply_discount(total, 10%)   → applies discount, no human needed

After all tools complete, the LLM composes a final response.

Pipeline:
  ┌──────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────┐
  │ agent.run()  │ ──→ │  tool 1  │ ──→ │  human   │ ──→ │  tool 2  │ ──→ │ LLM │
  │ (prompt)     │     │ executes │     │ approves │     │ executes │     │     │
  └──────────────┘     └──────────┘     └──────────┘     └──────────┘     └─────┘
                               ↑                             ↑
                         human approves               human approves
                         flight price                 total cost

Usage
─────
    uv run python 03_multistep_tools.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


# ── Tool 1: Human approves price before it's used downstream ────────

def get_flight_price(destination: str) -> str:
    """Look up flight price. Human approves or corrects before returning."""
    print(f"\n[tool:get_flight_price] Looking up price for {destination}...")

    prices = {"tokyo": "$1,200.00", "new york": "$450.00", "london": "$800.00"}
    price = prices.get(destination.lower(), "$999.00")

    print(f"[tool:get_flight_price] Found: {price}")
    choice = input("[tool:get_flight_price] Approve? [Y]es  enter correction: ").strip()

    if choice.upper() == "Y":
        return price
    return choice


# ── Tool 2: Human approves total before discount is applied ──────────

def calculate_total(flight_price: str, hotel_per_night: str, nights: int) -> str:
    """Calculate total trip cost. Human approves before tool returns."""
    flight = float(flight_price.replace("$", "").replace(",", ""))
    hotel = float(hotel_per_night.replace("$", "").replace(",", ""))
    total = flight + (hotel * nights)

    result = f"${total:,.2f}"

    print(f"\n[tool:calculate_total] Breakdown:")
    print(f"  Flight:       {flight_price}")
    print(f"  Hotel:        {hotel_per_night} × {nights} nights")
    print(f"  Total before discount: {result}")

    choice = input("[tool:calculate_total] Approve? [Y]es  enter correction: ").strip()

    if choice.upper() == "Y":
        return result
    return choice


# ── Tool 3: Automatic — no human review needed ───────────────────────

def apply_discount(total: str, discount_percent: float) -> str:
    """Apply a loyalty discount. Returns the final price."""
    amount = float(total.replace("$", "").replace(",", ""))
    discounted = amount * (1 - discount_percent / 100)
    return f"${discounted:,.2f} ({discount_percent}% off)"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    tools = ToolRegistry().add_many(get_flight_price, calculate_total, apply_discount)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(tools)
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("hitl-multistep-demo", memory)
    print("[agent] Generating response...")
    result = await agent.run(
        "I need to plan a 5-night trip to Tokyo. "
        "First, look up the flight price to Tokyo. "
        "Then calculate the total cost with a $200/night hotel. "
        "Finally, apply my 10% loyalty discount.",
        history,
        "hitl-multistep-demo",
    )

    print(f"\nFinal output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
