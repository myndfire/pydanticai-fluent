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
"""

import asyncio
from datetime import datetime

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext


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
    print(f"  [recovery] #{len(error_log)}: {ctx.source}/{ctx.error_type} → {action}")


def stack_inspector(ctx: ErrorContext) -> str | None:
    """Inspect the stack trace on an error."""
    stack = ctx.stack_trace or ""
    lines = stack.strip().split("\n")
    print(f"  [stack] {len(lines)} frames in traceback")
    for line in lines[:4]:
        print(f"      {line.strip()}")
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
    print("=" * 60)
    print("Custom Recovery — Fallback & Stack Traces")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Full recovery config ─────────────────────────
    print("\n--- Example 1: Per-source recovery ---")

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

    print(f"\n  Result: success={result.success}")
    print(f"  Output: {result.output}")
    if result.error_context:
        ec = result.error_context
        print(f"  ErrorContext:")
        print(f"    error_type:    {ec.error_type}")
        print(f"    error_message: {ec.error_message[:80]}")
        print(f"    source:        {ec.source}")

    # ── Example 2: Multiple agent failures ──────────────────────
    print("\n--- Example 2: Multiple failures (same config) ---")

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
        print(f"  Turn {i}: success={result_i.success}, output={result_i.output[:60]}...")

    # ── Example 3: Stack trace capture ──────────────────────────
    print("\n--- Example 3: Stack trace capture ---")

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
    print(f"  Result: {result_s.output}")

    # ── Error log summary ───────────────────────────────────────
    print(f"\n--- Recovery log ({len(error_log)} entries) ---")
    for entry in error_log:
        print(f"  [{entry['timestamp'][:19]}] {entry['source']}: "
              f"{entry['error_type']} → {entry['action']}")


if __name__ == "__main__":
    asyncio.run(main())
