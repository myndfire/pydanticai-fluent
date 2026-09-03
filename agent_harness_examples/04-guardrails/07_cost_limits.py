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
    uv run python 07_cost_limits.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/07_cost_limits.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import CostLimitsConfig


load_dotenv()
log = structlog.get_logger()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

COST_LIMITS_INPUT_COST = float(os.getenv("COST_LIMITS_INPUT_COST", "0.000003"))
COST_LIMITS_OUTPUT_COST = float(os.getenv("COST_LIMITS_OUTPUT_COST", "0.000015"))
COST_LIMITS_EX1_MAX_TOTAL = float(os.getenv("COST_LIMITS_EX1_MAX_TOTAL", "0.005"))
COST_LIMITS_EX2_MAX_INPUT = float(os.getenv("COST_LIMITS_EX2_MAX_INPUT", "0.001"))
COST_LIMITS_EX2_MAX_OUTPUT = float(os.getenv("COST_LIMITS_EX2_MAX_OUTPUT", "0.003"))
COST_LIMITS_EX3_MAX_TOTAL = float(os.getenv("COST_LIMITS_EX3_MAX_TOTAL", "0.000001"))


def on_cost_limit_handler(ctx):
    """Graceful fallback when a cost limit is hit."""
    log.debug("cost_limit_exceeded", error_type=ctx.error_type, error_message=ctx.error_message)
    return (
        f"Response unavailable: cost limit exceeded. "
        f"({ctx.error_type}: {ctx.error_message})"
    )


async def main():
    log.debug("separator")
    log.debug("section", title="Cost Limits Guardrail")
    log.debug("separator")

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()

    # ── GPT-4o pricing (approximate) ───────────────────────────
    # $3.00  per 1M input tokens  → $0.000003  per input token
    # $15.00 per 1M output tokens → $0.000015  per output token
    INPUT_COST = COST_LIMITS_INPUT_COST
    OUTPUT_COST = COST_LIMITS_OUTPUT_COST

    log.debug("section", title="Pricing (GPT-4o approx)")
    log.debug("pricing", cost_per_input_token=INPUT_COST)
    log.debug("pricing", cost_per_output_token=OUTPUT_COST)

    # ── Example 1: Strict total cost limit ─────────────────────
    log.debug("example", example=1, title="Total cost limit ($0.005)")
    limits1 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_total_cost(COST_LIMITS_EX1_MAX_TOTAL)
        .on_cost_limit(on_cost_limit_handler)
    )

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_cost_limits(limits1)
    )

    history1 = await MessageHistory().load("cost-limit-demo-1", memory)

    log.debug("cost_limits", max_total_cost=limits1.max_total_cost)
    log.debug("section", title="Sending prompt: explain theory of relativity")

    result1 = await agent1.run(
        "Explain Einstein's theory of relativity in detail.",
        history1,
        "cost-limit-demo-1",
    )

    log.debug("separator")
    log.debug("result", success=result1.success)
    log.debug("result", output=result1.output)

    # ── Example 2: Separate input/output cost limits ───────────
    log.debug("example", example=2, title="Separate input + output cost limits")
    limits2 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_input_cost(COST_LIMITS_EX2_MAX_INPUT)
        .with_max_output_cost(COST_LIMITS_EX2_MAX_OUTPUT)
        .on_cost_limit(on_cost_limit_handler)
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_cost_limits(limits2)
    )

    history2 = await MessageHistory().load("cost-limit-demo-2", memory)

    log.debug("cost_limits", max_input_cost=limits2.max_input_cost)
    log.debug("cost_limits", max_output_cost=limits2.max_output_cost)
    log.debug("section", title="Sending prompt: write a short poem about the ocean")

    result2 = await agent2.run(
        "Write a short poem about the ocean.",
        history2,
        "cost-limit-demo-2",
    )

    log.debug("separator")
    log.debug("result", success=result2.success)
    log.debug("result", output=result2.output)

    # ── Example 3: Extremely tight budget (will trigger) ───────
    log.debug("example", example=3, title="Extremely tight budget ($0.000001)")
    limits3 = (
        CostLimitsConfig()
        .with_cost_per_input_token(INPUT_COST)
        .with_cost_per_output_token(OUTPUT_COST)
        .with_max_total_cost(COST_LIMITS_EX3_MAX_TOTAL)  # nearly impossible
        .on_cost_limit(on_cost_limit_handler)
    )

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_cost_limits(limits3)
    )

    history3 = await MessageHistory().load("cost-limit-demo-3", memory)

    log.debug("cost_limits", max_total_cost=limits3.max_total_cost)
    log.debug("section", title="No on_cost_limit callback set - will raise RuntimeError")

    result3 = await agent3.run(
        "Say hello.",
        history3,
        "cost-limit-demo-3",
    )

    log.debug("separator")
    log.debug("result", success=result3.success)
    log.debug("result", output=result3.output)


if __name__ == "__main__":
    asyncio.run(main())
