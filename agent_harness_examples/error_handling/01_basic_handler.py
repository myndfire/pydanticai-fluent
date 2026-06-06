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

"""Basic error handler — suppress or re-raise errors from agent runs.

Demonstrates:
  - ErrorHandlingConfig().on_llm_error(cb) and other per-source callbacks
  - ErrorContext inspection: error_type, error_message, source, session_id, stack_trace
  - Return a value to suppress → AgentRunResult(success=False, output=value)
  - Return None to re-raise → exception propagates
  - Catch-all on_error for unhandled sources

Usage:
    uv run python 01_basic_handler.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext


# ── Handler: suppress LLM errors ────────────────────────────────────

def suppress_llm_handler(ctx: ErrorContext) -> str | None:
    """Suppress the LLM error and return a graceful failure message."""
    print(f"\n  [on_llm_error] Error intercepted:")
    print(f"    Source:      {ctx.source}")
    print(f"    Type:        {ctx.error_type}")
    print(f"    Message:     {ctx.error_message}")
    print(f"    Session:     {ctx.session_id}")
    print(f"    Prompt:      {ctx.prompt}")
    print(f"    Has stack:   {ctx.stack_trace is not None}")
    return f"Graceful fallback: LLM call failed ({ctx.error_type})"


# ── Handler: re-raise ──────────────────────────────────────────────

def re_raise_handler(ctx: ErrorContext) -> str | None:
    """Log the error but allow it to propagate."""
    print(f"\n  [on_error] Logging but will RE-RAISE:")
    print(f"    Source:  {ctx.source}")
    print(f"    Type:    {ctx.error_type}")
    print(f"    Message: {ctx.error_message[:100]}")
    return None  # re-raise


async def main():
    print("=" * 60)
    print("Error Handling — Per-Source Callbacks")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Suppress LLM errors (return a value) ─────────
    print("\n--- Example 1: Suppress LLM errors (return value) ---")

    config = (
        ErrorHandlingConfig()
        .on_llm_error(suppress_llm_handler)
    )

    bad_agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="nonexistent-model-xyz",  # will fail → source="llm"
        ))
        .with_error_handling(config)
    )

    history1 = await MessageHistory().load("err-llm", memory)
    result1 = await bad_agent.run("Say hello.", history1, "err-llm")

    print(f"\n  Result: success={result1.success}")
    print(f"  Output: {result1.output}")
    if result1.error_context:
        ec = result1.error_context
        print(f"  Error: {ec.error_type}: {ec.error_message}")

    # ── Example 2: Re-raise (return None) ───────────────────────
    print("\n--- Example 2: Re-raise (return None) ---")

    re_raise_config = ErrorHandlingConfig().on_llm_error(re_raise_handler)

    bad_agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="also-broken-model",
        ))
        .with_error_handling(re_raise_config)
    )

    print("  About to run an agent that will fail...")
    try:
        history2 = await MessageHistory().load("err-re-raise", memory)
        await bad_agent2.run("Say hello.", history2, "err-re-raise")
    except Exception as e:
        print(f"  Exception re-raised: {type(e).__name__}: {e}")

    # ── Example 3: Catch-all on_error ───────────────────────────
    print("\n--- Example 3: Catch-all on_error ---")

    catch_all_config = ErrorHandlingConfig().on_error(
        lambda ctx: f"Caught {ctx.source} error: {ctx.error_message[:50]}"
    )

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="will-definitely-fail",
        ))
        .with_error_handling(catch_all_config)
    )

    history3 = await MessageHistory().load("err-catchall", memory)
    result3 = await agent3.run("Hello.", history3, "err-catchall")
    print(f"  Suppressed: success={result3.success}, output={result3.output}")

    # ── Example 4: Successful run (handler not triggered) ───────
    print("\n--- Example 4: Successful run (handler not triggered) ---")

    good_config = ErrorHandlingConfig().on_llm_error(suppress_llm_handler)

    good_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_error_handling(good_config)
    )

    history4 = await MessageHistory().load("err-good", memory)
    result4 = await good_agent.run("What is 2+2?", history4, "err-good")
    print(f"  Success: {result4.success}")
    print(f"  Output: {result4.output}")

    # ── Example 5: Per-source routing ───────────────────────────
    print("\n--- Example 5: Per-source routing ---")

    routing_config = (
        ErrorHandlingConfig()
        .on_llm_error(lambda ctx: f"LLM failed: {ctx.error_message[:40]}")
        .on_memory_error(lambda ctx: f"Memory failed: {ctx.error_message[:40]}")
        .on_prompt_error(lambda ctx: f"Prompt failed: {ctx.error_message[:40]}")
        .on_output_error(lambda ctx: f"Output failed: {ctx.error_message[:40]}")
        .on_error(
            lambda ctx: f"Unhandled ({ctx.source}): {ctx.error_message[:40]}"
        )
    )

    agent5 = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="routing-test-fail",
        ))
        .with_error_handling(routing_config)
    )

    history5 = await MessageHistory().load("err-routing", memory)
    result5 = await agent5.run("Hi", history5, "err-routing")
    print(f"  Routed through on_llm_error: {result5.output}")


if __name__ == "__main__":
    asyncio.run(main())
