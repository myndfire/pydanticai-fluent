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

"""Cost limits guardrail — cap dollar cost per request using token pricing.

Demonstrates:
  - Per-token pricing configuration (cost_per_input_token, cost_per_output_token)
  - Input cost limit, output cost limit, and total cost limit
  - on_cost_limit callback for graceful handling when budget exceeded
  - on_error callback for unexpected failures

Cost formula:
  input_cost  = input_tokens  * cost_per_input_token
  output_cost = output_tokens * cost_per_output_token
  total_cost  = input_cost + output_cost

Usage:
    python 07_cost_limits.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import CostLimitsConfig


def on_cost_limit_handler(ctx):
    """Graceful fallback when a cost limit is hit."""
    print(f"  [on_cost_limit] {ctx.error_type}: {ctx.error_message}")
    return (
        f"Response unavailable: cost limit exceeded. "
        f"({ctx.error_type}: {ctx.error_message})"
    )


async def main():
    print("=" * 60)
    print("Cost Limits Guardrail")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()

    # ── GPT-4o pricing (approximate) ───────────────────────────
    # $3.00  per 1M input tokens  → $0.000003  per input token
    # $15.00 per 1M output tokens → $0.000015  per output token
    INPUT_COST = 0.000003
    OUTPUT_COST = 0.000015

    print("\nPricing (GPT-4o approx):")
    print(f"  Input:  ${INPUT_COST:.7f} per token")
    print(f"  Output: ${OUTPUT_COST:.7f} per token")

    # ── Example 1: Strict total cost limit ─────────────────────
    print("\n--- Example 1: Total cost limit ($0.005) ---")
    limits1 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_total_cost(0.005)
        .on_cost_limit(on_cost_limit_handler)
    )

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_cost_limits(limits1)
    )

    history1 = await MessageHistory().load("cost-limit-demo-1", memory)

    print(f"Max total cost: ${limits1.max_total_cost}")
    print("\nSending prompt: 'Explain the theory of relativity.'...\n")

    result1 = await agent1.run(
        "Explain Einstein's theory of relativity in detail.",
        history1,
        "cost-limit-demo-1",
    )

    print(f"\nSuccess: {result1.success}")
    print(f"Output: {result1.output}")

    # ── Example 2: Separate input/output cost limits ───────────
    print("\n--- Example 2: Separate input + output cost limits ---")
    limits2 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_input_cost(0.001)
        .with_max_output_cost(0.003)
        .on_cost_limit(on_cost_limit_handler)
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_cost_limits(limits2)
    )

    history2 = await MessageHistory().load("cost-limit-demo-2", memory)

    print(f"Max input cost: ${limits2.max_input_cost}")
    print(f"Max output cost: ${limits2.max_output_cost}")
    print("\nSending prompt: 'Write a short poem about the ocean.'...\n")

    result2 = await agent2.run(
        "Write a short poem about the ocean.",
        history2,
        "cost-limit-demo-2",
    )

    print(f"\nSuccess: {result2.success}")
    print(f"Output: {result2.output}")

    # ── Example 3: Extremely tight budget (will trigger) ───────
    print("\n--- Example 3: Extremely tight budget ($0.000001) ---")
    limits3 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_total_cost(0.000001)  # nearly impossible
        .on_cost_limit(on_cost_limit_handler)
    )

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_cost_limits(limits3)
    )

    history3 = await MessageHistory().load("cost-limit-demo-3", memory)

    print(f"Max total cost: ${limits3.max_total_cost}")
    print("No on_cost_limit callback set — will raise RuntimeError...\n")

    result3 = await agent3.run(
        "Say hello.",
        history3,
        "cost-limit-demo-3",
    )

    print(f"\nSuccess: {result3.success}")
    print(f"Output: {result3.output}")


if __name__ == "__main__":
    asyncio.run(main())
