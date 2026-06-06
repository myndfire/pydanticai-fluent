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
    python 05_guardrail_errors.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import CircuitBreakerConfig, ContentFilterConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext


# ── Guardrail error handler ─────────────────────────────────────────

def on_guardrail_failure(ctx: ErrorContext) -> str | None:
    """Handle guardrail violations with a graceful fallback."""
    print(f"\n  [on_guardrail_error] Guardrail triggered!")
    print(f"    Type:    {ctx.error_type}")
    print(f"    Message: {ctx.error_message}")
    print(f"    Source:  {ctx.source}")
    print(f"    Session: {ctx.session_id}")

    if "CircuitBreakerOpen" in ctx.error_type:
        return f"Service temporarily unavailable: circuit breaker is open."
    elif "TokenLimit" in ctx.error_type:
        return f"Response truncated: token limit reached."
    else:
        return f"Request blocked by guardrail: {ctx.error_type}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Guardrail Errors — Circuit Breaker & Limits")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Circuit breaker opens ────────────────────────
    print("\n--- Example 1: Circuit breaker opens ---")

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

    print("  Circuit breaker: threshold=1, timeout=30s")
    print("  Error handler: on_guardrail_error → on_guardrail_failure")

    # First run: model fails → circuit opens
    print("\n  --- Run 1: model fails → 1st failure → circuit opens ---")
    history1 = await MessageHistory().load("guard-err-1", memory)
    try:
        result1 = await bad_agent.run("Hello.", history1, "guard-err-1")
        print(f"  Result: {result1.output}")
    except RuntimeError as e:
        print(f"  Raised (no CB on_error callback set on CircuitBreakerConfig)")

    # Second run: circuit is open → guardrail error
    print("\n  --- Run 2: circuit open → guardrail error ---")
    history2 = await MessageHistory().load("guard-err-2", memory)
    try:
        result2 = await bad_agent.run("Hello again.", history2, "guard-err-2")
        print(f"  Result: {result2.output}")
    except RuntimeError as e:
        print(f"  Guardrail error propagated: {e}")

    # ── Example 2: Circuit breaker with on_error on CB config ───
    print("\n--- Example 2: Circuit breaker with on_error callback ---")

    def cb_on_error(ctx):
        print(f"\n  [CircuitBreaker.on_error] {ctx.error_type}: {ctx.error_message}")
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

    print("  CB on_error callback set on CircuitBreakerConfig directly")
    print("  Also have on_guardrail_error on ErrorHandlingConfig")
    print("  CB on_error fires first (before it reaches ErrorHandler)")

    # Run to open the circuit
    history_a = await MessageHistory().load("guard-err-a", memory)
    try:
        await agent2.run("Hi.", history_a, "guard-err-a")
    except Exception:
        pass

    # Now circuit is open
    history_b = await MessageHistory().load("guard-err-b", memory)
    result_b = await agent2.run("Hi again.", history_b, "guard-err-b")
    print(f"  Result: {result_b.output}")

    # ── Example 3: Content filter callback raises ───────────────
    print("\n--- Example 3: Content filter callback raises → guardrail ---")

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
        print(f"  Result: {result3.output}")
    except RuntimeError as e:
        print(f"  Propagated (no on_error set on ContentFilterConfig): {e}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n--- Guardrail error types ---")
    print("  CircuitBreakerOpen    — too many failures, circuit tripped")
    print("  TokenLimitExceeded    — token usage cap reached")
    print("  CostLimitExceeded     — dollar cost cap reached")
    print("  TurnLimitExceeded     — session turn cap reached")
    print("  Callback exceptions   — content filter / PII redact raises")


if __name__ == "__main__":
    asyncio.run(main())
