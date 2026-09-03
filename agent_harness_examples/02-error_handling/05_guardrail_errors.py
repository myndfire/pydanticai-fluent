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

"""Guardrail errors — handling circuit breaker and token limit violations.

Demonstrates:
  - Circuit breaker opens after N consecutive failures
  - on_guardrail_error callback handles circuit open, token limit, and
    content filter failures gracefully
  - Different guardrail types all route through the same handler
  - Source="guardrail" set by GuardRunner.run_with_guards()

Usage:
    uv run python 05_guardrail_errors.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/05_guardrail_errors.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import CircuitBreakerConfig, ContentFilterConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Guardrail error handler ─────────────────────────────────────────

def on_guardrail_failure(ctx: ErrorContext) -> str | None:
    """Handle guardrail violations with a graceful fallback."""
    log.debug(
        "guardrail_triggered",
        error_type=ctx.error_type,
        error_message=ctx.error_message,
        source=ctx.source,
        session_id=ctx.session_id,
    )

    if "CircuitBreakerOpen" in ctx.error_type:
        return f"Service temporarily unavailable: circuit breaker is open."
    elif "TokenLimit" in ctx.error_type:
        return f"Response truncated: token limit reached."
    else:
        return f"Request blocked by guardrail: {ctx.error_type}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    log.debug("separator")
    log.debug("section", title="Guardrail Errors — Circuit Breaker & Limits")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Circuit breaker opens ────────────────────────
    log.debug("example", example=1, title="Circuit breaker opens")

    config = ErrorHandlingConfig().on_guardrail_error(on_guardrail_failure)

    bad_agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="this-will-fail-immediately",
        ))
        .with_circuit_breaker(
            CircuitBreakerConfig()
            .with_threshold(1)      # open after 1 failure
            .with_timeout(30)
        )
        .with_error_handling(config)
    )

    log.debug("status", circuit_breaker="threshold=1, timeout=30s", handler="on_guardrail_error → on_guardrail_failure")

    # First run: model fails → circuit opens
    log.debug("run", run=1, description="model fails → 1st failure → circuit opens")
    history1 = await MessageHistory().load("guard-err-1", memory)
    try:
        result1 = await bad_agent.run("Hello.", history1, "guard-err-1")
        log.debug("result", output=result1.output)
    except RuntimeError as e:
        log.debug("status", message="Raised (no CB on_error callback set on CircuitBreakerConfig)")

    # Second run: circuit is open → guardrail error
    log.debug("run", run=2, description="circuit open → guardrail error")
    history2 = await MessageHistory().load("guard-err-2", memory)
    try:
        result2 = await bad_agent.run("Hello again.", history2, "guard-err-2")
        log.debug("result", output=result2.output)
    except RuntimeError as e:
        log.debug("guardrail_error", error=str(e))

    # ── Example 2: Circuit breaker with on_error on CB config ───
    log.debug("example", example=2, title="Circuit breaker with on_error callback")

    def cb_on_error(ctx):
        log.debug("circuit_breaker_error", error_type=ctx.error_type, error_message=ctx.error_message)
        return f"CB fallback: service paused ({ctx.error_type})"

    cb_config = (
        CircuitBreakerConfig()
        .with_threshold(1)
        .with_timeout(30)
        .on_error(cb_on_error)
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="another-broken-model",
        ))
        .with_circuit_breaker(cb_config)
        .with_error_handling(
            ErrorHandlingConfig().on_guardrail_error(on_guardrail_failure)
        )
    )

    log.debug("status", message="CB on_error callback set on CircuitBreakerConfig directly")
    log.debug("status", message="Also have on_guardrail_error on ErrorHandlingConfig")
    log.debug("status", message="CB on_error fires first (before it reaches ErrorHandler)")

    # Run to open the circuit
    history_a = await MessageHistory().load("guard-err-a", memory)
    try:
        await agent2.run("Hi.", history_a, "guard-err-a")
    except Exception:
        pass

    # Now circuit is open
    history_b = await MessageHistory().load("guard-err-b", memory)
    result_b = await agent2.run("Hi again.", history_b, "guard-err-b")
    log.debug("result", output=result_b.output)

    # ── Example 3: Content filter callback raises ───────────────
    log.debug("example", example=3, title="Content filter callback raises → guardrail")

    def broken_filter(text: str) -> str:
        raise RuntimeError("Filter service unavailable")

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_content_filter(
            ContentFilterConfig()
            .on_filter(broken_filter)
        )
        .with_error_handling(
            ErrorHandlingConfig().on_guardrail_error(on_guardrail_failure)
        )
    )

    history3 = await MessageHistory().load("guard-err-3", memory)
    try:
        result3 = await agent3.run("Say hello.", history3, "guard-err-3")
        log.debug("result", output=result3.output)
    except RuntimeError as e:
        log.debug("status", message=f"Propagated (no on_error set on ContentFilterConfig): {e}")

    # ── Summary ─────────────────────────────────────────────────
    log.debug("example", title="Guardrail error types")
    log.debug("guardrail_type", type="CircuitBreakerOpen", description="too many failures, circuit tripped")
    log.debug("guardrail_type", type="TokenLimitExceeded", description="token usage cap reached")
    log.debug("guardrail_type", type="CostLimitExceeded", description="dollar cost cap reached")
    log.debug("guardrail_type", type="TurnLimitExceeded", description="session turn cap reached")
    log.debug("guardrail_type", type="Callback exceptions", description="content filter / PII redact raises")


if __name__ == "__main__":
    asyncio.run(main())
