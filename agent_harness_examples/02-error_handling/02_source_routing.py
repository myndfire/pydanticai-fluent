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

"""Source routing — different handling per error source.

Demonstrates:
  - Per-source callbacks: on_llm_error, on_tool_error, on_validation_error,
    on_guardrail_error, on_memory_error, on_prompt_error,
    on_evaluator_error, on_output_error
  - Source set explicitly by agent.run() at each origination point
  - Different recovery strategies per source
  - Catch-all on_error for unhandled sources

Eight error sources tracked by agent.run():
  "llm"        — model call fails (network, auth, rate limit, timeout)
  "tool"       — tool function execution fails
  "validation" — output validator ModelRetry exhausted
  "guardrail"  — circuit breaker, token/cost limits, content filter, turns
  "memory"     — load/save to memory providers
  "prompt"     — get_system_prompt() / Jinja2 render fails
  "evaluator"  — evaluator.evaluate() raises
  "output"     — usage parsing, turn construction, output extraction

Usage:
    uv run python 02_source_routing.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/02_source_routing.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Per-source handlers ─────────────────────────────────────────────

def on_llm(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="llm", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"LLM fallback: model unavailable ({ctx.error_type})"

def on_tool(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="tool", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"Tool fallback: operation could not be completed"

def on_validation(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="validation", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"Validation fallback: response format was invalid"

def on_guardrail(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="guardrail", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"Guardrail fallback: request blocked ({ctx.error_type})"

def on_memory(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="memory", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    # Memory errors: suppress (continue without persistence)
    return f"Memory fallback: persistence unavailable — continuing anyway"

def on_prompt(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="prompt", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return None  # re-raise — prompts are critical

def on_evaluator(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="evaluator", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    # Evaluator errors: suppress (non-critical)
    return f"Evaluator error: {ctx.error_type} — continuing"

def on_output(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="output", error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"Output processing fallback: {ctx.error_type}"

def catch_all(ctx: ErrorContext) -> str | None:
    log.debug("source_handler", source="catch-all", error_source=ctx.source, error_type=ctx.error_type, error_message=ctx.error_message[:60])
    return f"Unhandled: {ctx.source} error"


async def main():
    log.debug("separator")
    log.debug("section", title="Source Routing — Per-Source Error Handlers")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Build config with all 8 sources + catch-all ─────────────
    config = (
        ErrorHandlingConfig()
        .on_llm_error(on_llm)
        .on_tool_error(on_tool)
        .on_validation_error(on_validation)
        .on_guardrail_error(on_guardrail)
        .on_memory_error(on_memory)
        .on_prompt_error(on_prompt)
        .on_evaluator_error(on_evaluator)
        .on_output_error(on_output)
        .on_error(catch_all)
    )

    log.debug("status", message="Configured handlers")
    for name in ["on_llm_error", "on_tool_error", "on_validation_error",
                 "on_guardrail_error", "on_memory_error", "on_prompt_error",
                 "on_evaluator_error", "on_output_error", "on_error"]:
        present = getattr(config, f"_{name}") is not None
        log.debug("handler_status", handler=name, present=present)

    # ── Demonstrate LLM error routing ───────────────────────────
    log.debug("example", title="LLM error (broken model)")
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="this-model-does-not-exist",
        ))
        .with_error_handling(config)
    )

    history = await MessageHistory().load("route-llm", memory)
    result = await agent.run("Hello.", history, "route-llm")
    log.debug("result", suppressed=True, output=result.output)

    # ── Source reference ────────────────────────────────────────
    log.debug("example", title="Error source reference")
    log.debug("source_reference", source="llm", set_at="agent.run() inside asyncio.wait_for")
    log.debug("source_reference", source="tool", set_at="tool function execution")
    log.debug("source_reference", source="validation", set_at="output validator ModelRetry")
    log.debug("source_reference", source="guardrail", set_at="circuit breaker, token/cost, content filter")
    log.debug("source_reference", source="memory", set_at="message_history.load(), save_to save_turn()")
    log.debug("source_reference", source="prompt", set_at="get_system_prompt(), Jinja2 render")
    log.debug("source_reference", source="evaluator", set_at="evaluator.evaluate()")
    log.debug("source_reference", source="output", set_at="usage parsing, TurnData, extract_clean_output")


if __name__ == "__main__":
    asyncio.run(main())
