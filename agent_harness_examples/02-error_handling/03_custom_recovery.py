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

"""Custom recovery — fallback responses with stack trace capture.

Demonstrates:
  - Custom error handler that returns fallback output
  - Stack trace inspection from ErrorContext
  - Per-source fallback messages (LLM, memory, output, etc.)
  - Combined recovery strategy: suppress all, log, return degraded response

Usage:
    uv run python 03_custom_recovery.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/03_custom_recovery.py
"""

import asyncio
from datetime import datetime

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Recovery state ──────────────────────────────────────────────────

error_log: list[dict] = []


def log_recovery(ctx: ErrorContext, action: str) -> None:
    """Log the recovery action."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": ctx.source,
        "error_type": ctx.error_type,
        "error_message": ctx.error_message,
        "action": action,
    }
    error_log.append(entry)
    log.debug("recovery_action", count=len(error_log), source=ctx.source, error_type=ctx.error_type, action=action)


def stack_inspector(ctx: ErrorContext) -> str | None:
    """Inspect the stack trace on an error."""
    stack = ctx.stack_trace or ""
    lines = stack.strip().split("\n")
    log.debug("stack_trace", frame_count=len(lines))
    for line in lines[:4]:
        log.debug("stack_frame", frame=line.strip())
    return None  # re-raise — stack inspection only, let other handlers decide


def llm_recovery(ctx: ErrorContext) -> str | None:
    """Recover from LLM failures with a fallback message."""
    log_recovery(ctx, "suppress")
    return (
        f"I'm sorry, the model is currently unavailable. "
        f"Error: {ctx.error_type}. Please try again later."
    )


def memory_recovery(ctx: ErrorContext) -> str | None:
    """Memory errors — suppress, agent continues without persistence."""
    log_recovery(ctx, "suppress — continue without persistence")
    return f"[Memory warning]: persistence unavailable, continuing anyway"


def output_recovery(ctx: ErrorContext) -> str | None:
    """Output processing failures — return raw text."""
    log_recovery(ctx, "suppress — return raw output marker")
    return f"[Output handling error]: {ctx.error_type}"


async def main():
    log.debug("separator")
    log.debug("section", title="Custom Recovery — Fallback & Stack Traces")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Full recovery config ─────────────────────────
    log.debug("example", example=1, title="Per-source recovery")

    config = (
        ErrorHandlingConfig()
        .on_llm_error(llm_recovery)
        .on_memory_error(memory_recovery)
        .on_output_error(output_recovery)
        .on_error(
            lambda ctx: f"Caught {ctx.source} error: {ctx.error_message[:50]}"
        )
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="recovery-test-model-fail",
        ))
        .with_error_handling(config)
    )

    history = await MessageHistory().load("recover-1", memory)
    result = await agent.run(
        "What is the capital of France?",
        history,
        "recover-1",
    )

    log.debug("result", success=result.success, output=result.output)
    if result.error_context:
        ec = result.error_context
        log.debug("error_context", error_type=ec.error_type, error_message=ec.error_message[:80], source=ec.source)

    # ── Example 2: Multiple agent failures ──────────────────────
    log.debug("example", example=2, title="Multiple failures (same config)")

    for i in range(1, 4):
        agent_i = (
            ManagedAgent()
            .with_model(ModelConfig(
                provider="ollama",
                model_name=f"fail-{i}",
            ))
            .with_error_handling(config)
        )
        history_i = await MessageHistory().load(f"recover-multi-{i}", memory)
        result_i = await agent_i.run(
            f"Say turn {i}.",
            history_i,
            f"recover-multi-{i}",
        )
        log.debug("result", turn=i, success=result_i.success, output=result_i.output[:60])

    # ── Example 3: Stack trace capture ──────────────────────────
    log.debug("example", example=3, title="Stack trace capture")

    stack_config = (
        ErrorHandlingConfig()
        .on_llm_error(stack_inspector)
        .on_error(lambda ctx: f"Fallback after stack inspection: {ctx.error_type}")
    )

    stack_agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="stack-trace-test-model",
        ))
        .with_error_handling(stack_config)
    )

    history_s = await MessageHistory().load("recover-stack", memory)
    result_s = await stack_agent.run("Hello.", history_s, "recover-stack")
    log.debug("result", output=result_s.output)

    # ── Error log summary ───────────────────────────────────────
    log.debug("recovery_log_summary", entry_count=len(error_log))
    for entry in error_log:
        log.debug("recovery_log_entry", timestamp=entry['timestamp'][:19], source=entry['source'], error_type=entry['error_type'], action=entry['action'])


if __name__ == "__main__":
    asyncio.run(main())
