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

"""Token limits guardrail — cap input, output, and total tokens per request.

Demonstrates:
  - max_input_tokens: limit input token count
  - max_output_tokens: limit output token count
  - max_total_tokens: limit combined token count
  - on_token_limit callback for graceful handling when limit exceeded
  - on_error callback for unexpected failures

Usage:
    uv run python 06_token_limits.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import TokenLimitsConfig


def on_token_limit_handler(ctx):
    """Graceful fallback when a token limit is hit."""
    print(f"  [on_token_limit] {ctx.error_type}: {ctx.error_message}")
    return (
        f"Response truncated: token limit reached. "
        f"({ctx.error_type}: {ctx.error_message})"
    )


async def main():
    print("=" * 60)
    print("Token Limits Guardrail")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()

    # ── Example 1: Strict total token limit ────────────────────
    print("\n--- Example 1: Strict total token limit (50 tokens) ---")
    limits = (
        TokenLimitsConfig()
        .with_max_total_tokens(50)
        .with_max_output_tokens(200)
        .on_token_limit(on_token_limit_handler)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_token_limits(limits)
    )

    history = await MessageHistory().load("token-limit-demo-1", memory)

    print(f"Max total tokens: {limits.max_total_tokens}")
    print(f"Max output tokens: {limits.max_output_tokens}")
    print("\nSending prompt: 'Explain quantum computing in detail.'...\n")

    result = await agent.run(
        "Explain quantum computing in detail.",
        history,
        "token-limit-demo-1",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Output: {result.output}")

    # ── Example 2: All limits set ──────────────────────────────
    print("\n--- Example 2: Input + output + total limits ---")
    limits2 = (
        TokenLimitsConfig()
        .with_max_input_tokens(500)
        .with_max_output_tokens(100)
        .with_max_total_tokens(600)
        .on_token_limit(on_token_limit_handler)
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_token_limits(limits2)
    )

    history2 = await MessageHistory().load("token-limit-demo-2", memory)

    print(f"Max input tokens: {limits2.max_input_tokens}")
    print(f"Max output tokens: {limits2.max_output_tokens}")
    print(f"Max total tokens: {limits2.max_total_tokens}")
    print("\nSending prompt: 'List the 50 US state capitals.'...\n")

    result2 = await agent2.run(
        "List the 50 US state capitals.",
        history2,
        "token-limit-demo-2",
    )

    print(f"\nSuccess: {result2.success}")
    print(f"Output: {result2.output}")

    # ── Example 3: Token limit without callback (raises) ───────
    print("\n--- Example 3: Token limit exceeded (no callback, will raise) ---")
    limits3 = TokenLimitsConfig().with_max_total_tokens(5)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_token_limits(limits3)
    )

    history3 = await MessageHistory().load("token-limit-demo-3", memory)

    print(f"Max total tokens: {limits3.max_total_tokens}")
    print("No on_token_limit callback set — will raise RuntimeError...\n")

    try:
        await agent3.run(
            "What is the capital of France?",
            history3,
            "token-limit-demo-3",
        )
    except RuntimeError as e:
        print(f"  Caught RuntimeError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
