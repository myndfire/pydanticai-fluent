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

"""Agent-level retries with timeout, backoff, fallback model, and callbacks.

Demonstrates:
  - Retry attempts on timeout/error with exponential backoff
  - on_retry callback called on each retry attempt
  - Fallback model used after all retries exhausted
  - on_error callback for graceful failure handling

Usage:
    python 01_agent_retries.py
"""

import asyncio
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig


async def main():
    print("=" * 60)
    print("Agent-Level Retries with Fallback & Callbacks")
    print("=" * 60)

    # ── Setup memory ────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("retry-demo", memory)

    # ── Configure agent retries with all options ────────────────
    retry_config = (
        AgentRetryConfig()
        .with_max_retries(3)          # 3 attempts total
        .with_timeout(10)             # 10s per attempt
        .with_backoff(2.0)            # exponential backoff
        .with_fallback("ollama:gpt-oss:20b")  # fallback model
        .on_retry(lambda ctx: print(  # called on each retry
            f"  [on_retry] type={ctx.error_type} "
            f"attempt={ctx.attempt}/{ctx.max_attempts} "
            f"will_retry={ctx.will_retry}"
        ))
        .on_error(lambda ctx: f"Fallback response: all attempts failed "
                               f"({ctx.error_type}: {ctx.error_message})")
    )

    # ── Build agent ────────────────────────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_agent_retries(retry_config)
    )

    # ── Run ────────────────────────────────────────────────────
    print(f"\nMax retries: {retry_config.max_retries}")
    print(f"Timeout: {retry_config.timeout}s")
    print(f"Backoff multiplier: {retry_config.backoff_multiplier}x")
    print(f"Fallback model: {retry_config.fallback_model}")
    print(f"\nSending prompt: 'What is 2+2?'...\n")

    result = await agent.run(
        "What is 2+2?",
        history,
        "retry-demo",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Used fallback: {result.used_fallback}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error_context}")


if __name__ == "__main__":
    asyncio.run(main())
