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

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/06_token_limits.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import TokenLimitsConfig


load_dotenv()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

TOKEN_LIMITS_EX1_MAX_TOTAL = int(os.getenv("TOKEN_LIMITS_EX1_MAX_TOTAL", "50"))
TOKEN_LIMITS_EX1_MAX_OUTPUT = int(os.getenv("TOKEN_LIMITS_EX1_MAX_OUTPUT", "200"))
TOKEN_LIMITS_EX2_MAX_INPUT = int(os.getenv("TOKEN_LIMITS_EX2_MAX_INPUT", "500"))
TOKEN_LIMITS_EX2_MAX_OUTPUT = int(os.getenv("TOKEN_LIMITS_EX2_MAX_OUTPUT", "100"))
TOKEN_LIMITS_EX2_MAX_TOTAL = int(os.getenv("TOKEN_LIMITS_EX2_MAX_TOTAL", "600"))
TOKEN_LIMITS_EX3_MAX_TOTAL = int(os.getenv("TOKEN_LIMITS_EX3_MAX_TOTAL", "5"))


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
        .with_max_total_tokens(TOKEN_LIMITS_EX1_MAX_TOTAL)
        .with_max_output_tokens(TOKEN_LIMITS_EX1_MAX_OUTPUT)
        .on_token_limit(on_token_limit_handler)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
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
        .with_max_input_tokens(TOKEN_LIMITS_EX2_MAX_INPUT)
        .with_max_output_tokens(TOKEN_LIMITS_EX2_MAX_OUTPUT)
        .with_max_total_tokens(TOKEN_LIMITS_EX2_MAX_TOTAL)
        .on_token_limit(on_token_limit_handler)
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
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
    limits3 = TokenLimitsConfig().with_max_total_tokens(TOKEN_LIMITS_EX3_MAX_TOTAL)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
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
    except Exception as e:
        print(f"  Caught exception: {e}")

    # ── Example 4: Streaming with token limits ──────────────
    print("\n--- Example 4: Streaming with token limits ---")

    limits4 = (
        TokenLimitsConfig()
        .with_max_output_tokens(20)
        .with_reasoning_traces(True)  # capture thinking parts for debugging
        .on_streaming_token_limit(
            lambda ctx, partial: (
                f"\n[TRUNCATED after {len(partial)} chars] "
                f"{ctx.error_message}\n"
                f"Partial output: {partial[:100]}..."
            )
        )
    )

    agent4 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_token_limits(limits4)
    )

    history4 = await MessageHistory().load("token-limit-demo-4", memory)

    print(f"Output tokens limit: {limits4.max_output_tokens}")
    print("Streaming response (tokens arrive in real-time)...\n")

    collected = ""
    try:
        async for chunk in agent4.run_stream(
            "Write a haiku about the sea.",
            history4,
            "token-limit-demo-4",
        ):
            collected += chunk
            print(f"  [{len(collected):3d} chars] {chunk!r}")
        print(f"\nFull output: {collected}")
    except RuntimeError as e:
        print(f"  Caught RuntimeError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
