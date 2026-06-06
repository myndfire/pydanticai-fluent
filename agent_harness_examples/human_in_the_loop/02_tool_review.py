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

"""Human-in-the-loop with tool calling — review agent output that includes tool results.

This example builds on 01_review_approval.py by adding a tool the agent
can call. The human reviews the final composed response after the tool
has been invoked and the LLM has incorporated its results.

Scenario
────────
You ask the agent to calculate a mortgage payment. The agent:
  1. Sees the prompt asking for a calculation
  2. Calls the mortgage_calculator tool with the loan parameters
  3. The tool returns a monthly payment amount
  4. The LLM composes a final response incorporating the tool's result
  5. The human_review callback fires — you see the response
  6. You [A]pprove it as-is, or [M]odify it (e.g. correct the math)
  7. The (possibly modified) response is returned

This demonstrates that the full pipeline runs transparently:
  ┌──────────────┐     ┌──────────┐     ┌─────┐     ┌──────────┐
  │ agent.run()  │ ──→ │ tool     │ ──→ │ LLM │ ──→ │ human    │
  │ (prompt)     │     │ executes │     │     │     │ reviews  │
  └──────────────┘     └──────────┘     └─────┘     └──────────┘
                                                         │
                                               ┌─────────┴─────────┐
                                               │ [A]pprove         │
                                               │   → return as-is  │
                                               │ [M]odify          │
                                               │   → return edited │
                                               └───────────────────┘

Key points
──────────
  - The tool is a plain Python function registered via ToolRegistry
  - ContentFilterConfig.on_filter intercepts the response post-generation
  - The response already includes the tool's computed result
  - The human can verify or correct the tool's output before accepting
  - Try modifying the response with a different payment amount to see
    how the human's edit replaces the agent's output entirely

Usage
─────
    uv run python 02_tool_review.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry
from agent_harness.guards import ContentFilterConfig


# ── Tool ────────────────────────────────────────────────────────────

def mortgage_calculator(loan_amount: float, annual_rate: float, years: int) -> str:
    """Calculate monthly mortgage payment.

    Args:
        loan_amount: Total loan amount in dollars (e.g. 500000)
        annual_rate: Annual interest rate as a decimal (e.g. 0.065 for 6.5%)
        years: Loan term in years (e.g. 30)

    Returns:
        Formatted monthly payment string.
    """
    monthly_rate = annual_rate / 12
    months = years * 12
    payment = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)
    result = f"${payment:,.2f}/month"
    print(f"[tool:mortgage_calculator] ${loan_amount:,.0f} at {annual_rate*100}% for {years}yr = {result}")
    return result


# ── Human review callback ───────────────────────────────────────────

def human_review(text: str) -> str:
    """Show the agent's response to a human for approval or modification.

    The response already includes the tool's computed result.
    """
    print(f"\nAgent says: {text}\n")

    choice = input("Approve or modify? [A]pprove  [M]odify: ").strip().upper()

    if choice == "M":
        return input("Your version: ").strip()

    return text


# ── Main ────────────────────────────────────────────────────────────

async def main():
    tools = ToolRegistry().add(mortgage_calculator)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(tools)
        .with_content_filter(ContentFilterConfig().on_filter(human_review))
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("hitl-tool-demo", memory)
    print("[agent] Generating response...")
    result = await agent.run(
        "Calculate the monthly payment for a $500,000 mortgage "
        "at 6.5% annual interest over 30 years. "
        "Use the mortgage_calculator tool.",
        history,
        "hitl-tool-demo",
    )

    print(f"\nFinal output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
