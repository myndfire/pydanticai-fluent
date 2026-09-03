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

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import TokenLimitsConfig


load_dotenv()
log = structlog.get_logger()

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
    log.debug("token_limit_exceeded", error_type=ctx.error_type, error_message=ctx.error_message)
    return (
        f"Response truncated: token limit reached. "
        f"({ctx.error_type}: {ctx.error_message})"
    )


async def main():
    log.debug("separator")
    log.debug("section", title="Token Limits Guardrail")
    log.debug("separator")

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()

    # ── Example 1: Strict total token limit ────────────────────
    log.debug("example", example=1, title="Strict total token limit (50 tokens)")
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

    log.debug("token_limits", max_total_tokens=limits.max_total_tokens)
    log.debug("token_limits", max_output_tokens=limits.max_output_tokens)
    log.debug("section", title="Sending prompt: explain quantum computing")

    result = await agent.run(
        "Explain quantum computing in detail.",
        history,
        "token-limit-demo-1",
    )

    log.debug("separator")
    log.debug("result", success=result.success)
    log.debug("result", output=result.output)

    # ── Example 2: All limits set ──────────────────────────────
    log.debug("example", example=2, title="Input + output + total limits")
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

    log.debug("token_limits", max_input_tokens=limits2.max_input_tokens)
    log.debug("token_limits", max_output_tokens=limits2.max_output_tokens)
    log.debug("token_limits", max_total_tokens=limits2.max_total_tokens)
    log.debug("section", title="Sending prompt: list 50 US state capitals")

    result2 = await agent2.run(
        "List the 50 US state capitals.",
        history2,
        "token-limit-demo-2",
    )

    log.debug("separator")
    log.debug("result", success=result2.success)
    log.debug("result", output=result2.output)

    # ── Example 3: Token limit without callback (raises) ───────
    log.debug("example", example=3, title="Token limit exceeded (no callback, will raise)")
    limits3 = TokenLimitsConfig().with_max_total_tokens(TOKEN_LIMITS_EX3_MAX_TOTAL)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_token_limits(limits3)
    )

    history3 = await MessageHistory().load("token-limit-demo-3", memory)

    log.debug("token_limits", max_total_tokens=limits3.max_total_tokens)
    log.debug("section", title="No on_token_limit callback set - will raise RuntimeError")

    try:
        await agent3.run(
            "What is the capital of France?",
            history3,
            "token-limit-demo-3",
        )
    except Exception as e:
        log.debug("exception", caught=str(e))

    # ── Example 4: Streaming with token limits ──────────────
    log.debug("example", example=4, title="Streaming with token limits")

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

    log.debug("token_limits", max_output_tokens=limits4.max_output_tokens)
    log.debug("section", title="Streaming response (tokens arrive in real-time)")

    collected = ""
    try:
        async for chunk in agent4.run_stream(
            "Write a haiku about the sea.",
            history4,
            "token-limit-demo-4",
        ):
            collected += chunk
            log.debug("stream_chunk", chars=len(collected), chunk=chunk)
        log.debug("result", full_output=collected)
    except RuntimeError as e:
        log.debug("exception", caught_runtime_error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
