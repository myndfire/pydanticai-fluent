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

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/01_basic_handler.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Handler: suppress LLM errors ────────────────────────────────────

def suppress_llm_handler(ctx: ErrorContext) -> str | None:
    """Suppress the LLM error and return a graceful failure message."""
    log.debug(
        "error_intercepted",
        source="llm",
        error_type=ctx.error_type,
        error_message=ctx.error_message,
        session_id=ctx.session_id,
        prompt=ctx.prompt,
        has_stack=ctx.stack_trace is not None,
    )
    return f"Graceful fallback: LLM call failed ({ctx.error_type})"


# ── Handler: re-raise ──────────────────────────────────────────────

def re_raise_handler(ctx: ErrorContext) -> str | None:
    """Log the error but allow it to propagate."""
    log.debug(
        "error_re_raise",
        source=ctx.source,
        error_type=ctx.error_type,
        error_message=ctx.error_message[:100],
    )
    return None  # re-raise


async def main():
    log.debug("separator")
    log.debug("section", title="Error Handling — Per-Source Callbacks")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Suppress LLM errors (return a value) ─────────
    log.debug("example", example=1, title="Suppress LLM errors (return value)")

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

    log.debug("result", success=result1.success, output=result1.output)
    if result1.error_context:
        ec = result1.error_context
        log.debug("error_context", error_type=ec.error_type, error_message=ec.error_message)

    # ── Example 2: Re-raise (return None) ───────────────────────
    log.debug("example", example=2, title="Re-raise (return None)")

    re_raise_config = ErrorHandlingConfig().on_llm_error(re_raise_handler)

    bad_agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="also-broken-model",
        ))
        .with_error_handling(re_raise_config)
    )

    log.debug("status", message="About to run an agent that will fail...")
    try:
        history2 = await MessageHistory().load("err-re-raise", memory)
        await bad_agent2.run("Say hello.", history2, "err-re-raise")
    except Exception as e:
        log.debug("exception_caught", exception_type=type(e).__name__, message=str(e))

    # ── Example 3: Catch-all on_error ───────────────────────────
    log.debug("example", example=3, title="Catch-all on_error")

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
    log.debug("result", suppressed=True, success=result3.success, output=result3.output)

    # ── Example 4: Successful run (handler not triggered) ───────
    log.debug("example", example=4, title="Successful run (handler not triggered)")

    good_config = ErrorHandlingConfig().on_llm_error(suppress_llm_handler)

    good_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_error_handling(good_config)
    )

    history4 = await MessageHistory().load("err-good", memory)
    result4 = await good_agent.run("What is 2+2?", history4, "err-good")
    log.debug("result", success=result4.success, output=result4.output)

    # ── Example 5: Per-source routing ───────────────────────────
    log.debug("example", example=5, title="Per-source routing")

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
    log.debug("result", routed_to="on_llm_error", output=result5.output)


if __name__ == "__main__":
    asyncio.run(main())
