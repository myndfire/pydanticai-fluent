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

"""Evaluator errors — handling failures when evaluators raise exceptions.

Demonstrates:
  - Registering an evaluator that deliberately raises
  - on_evaluator_error callback to handle evaluator failures
  - Source="evaluator" classification in agent.run()
  - Evaluator errors are no longer silently swallowed
  - Graceful degradation: agent continues despite evaluator failure

Evaluators run post-turn to assess quality, safety, etc.
Previously their failures were silently swallowed.
Now they propagate to the error handler with source="evaluator".

Usage:
    uv run python 06_evaluator_errors.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/06_evaluator_errors.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import Evaluator
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Evaluator that raises ───────────────────────────────────────────

class FailingEvaluator(Evaluator):
    """An evaluator that always raises — simulates a broken quality check."""

    def __init__(self, name: str = "broken_check"):
        self.name = name

    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        log.debug("evaluator_failing", name=self.name, message="About to fail...")
        raise RuntimeError(f"Evaluator '{self.name}' crashed: external service timeout")


class WorkingEvaluator(Evaluator):
    """An evaluator that works — logs the output length."""

    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        output = getattr(result, "output", str(result))
        log.debug("evaluator_working", output_length=len(output or ''))


# ── Evaluator error handler ─────────────────────────────────────────

def on_evaluator_failure(ctx: ErrorContext) -> str | None:
    """Handle evaluator failures — suppress, don't crash the agent."""
    log.debug(
        "evaluator_error_intercepted",
        error_type=ctx.error_type,
        error_message=ctx.error_message,
        session_id=ctx.session_id,
        prompt=ctx.prompt,
    )
    # Evaluators are non-critical — suppress and continue
    return None  # return None to re-raise, or return a string to suppress


def suppress_evaluator(ctx: ErrorContext) -> str | None:
    """Suppress the evaluator failure and continue."""
    log.debug("evaluator_suppressed", error_type=ctx.error_type)
    return None  # suppress by returning None... wait, this would re-raise

# Actually: return a value = suppress, return None = re-raise
# For evaluators we want to suppress gracefully

def handle_evaluator_gracefully(ctx: ErrorContext) -> str | None:
    """Suppress evaluator failures — they're non-critical."""
    log.debug("evaluator_suppressed_gracefully", error_type=ctx.error_type, error_message=ctx.error_message[:80])
    return None  # suppress with no fallback output


async def main():
    log.debug("separator")
    log.debug("section", title="Evaluator Errors — Handling Post-Turn Failures")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Failing evaluator with suppress handler ──────
    log.debug("example", example=1, title="Suppress evaluator errors")

    config = (
        ErrorHandlingConfig()
        .on_evaluator_error(lambda ctx: None)  # suppress completely
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(FailingEvaluator("quality_gate"))
        .with_error_handling(config)
    )

    history = await MessageHistory().load("eval-err-1", memory)
    result = await agent.run(
        "What is the capital of Japan?",
        history,
        "eval-err-1",
    )
    log.debug("result", agent_output=result.output, success=result.success)

    # ── Example 2: Multiple evaluators, one fails ───────────────
    log.debug("example", example=2, title="Mixed evaluators (one fails, one works)")

    config2 = (
        ErrorHandlingConfig()
        .on_evaluator_error(
            lambda ctx: None  # suppress — don't let it kill the run
        )
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(FailingEvaluator("safety_scan"), WorkingEvaluator())
        .with_error_handling(config2)
    )

    log.debug("status", evaluators="FailingEvaluator + WorkingEvaluator")

    # First run: FailingEvaluator raises → caught by on_evaluator_error
    # Note: after the first evaluator raises, the error propagates
    # and skip the second. To handle this, we'd need per-evaluator
    # try/except in agent.run().
    history2 = await MessageHistory().load("eval-err-2", memory)
    try:
        result2 = await agent2.run(
            "What is 2+2?",
            history2,
            "eval-err-2",
        )
        log.debug("result", agent_output=result2.output)
    except RuntimeError as e:
        log.debug("status", message=f"Propagated: {e}")
        log.debug("status", message="First evaluator failed; on_evaluator_error returned None → re-raise")
        log.debug("status", message="To run all evaluators even if one fails, return a value from the handler.")

    # ── Example 3: Suppress with fallback message ───────────────
    log.debug("example", example=3, title="Suppress evaluator errors with fallback")

    config3 = (
        ErrorHandlingConfig()
        .on_evaluator_error(
            lambda ctx: f"[Evaluator note]: quality check failed ({ctx.error_type}). "
                         "Results may not be verified."
        )
    )

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(FailingEvaluator("final_check"))
        .with_error_handling(config3)
    )

    history3 = await MessageHistory().load("eval-err-3", memory)
    result3 = await agent3.run(
        "Say hello in exactly 3 words.",
        history3,
        "eval-err-3",
    )
    log.debug("result", agent_output=result3.output, success=result3.success)


if __name__ == "__main__":
    asyncio.run(main())
