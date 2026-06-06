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

"""Circuit breaker guardrail — prevent cascading failures.

Demonstrates:
  - Circuit states: CLOSED (healthy) → OPEN (blocked) → HALF_OPEN (testing)
  - failure_threshold: consecutive failures before circuit opens
  - circuit_timeout: seconds before attempting a half-open trial
  - on_error callback for graceful handling when circuit is open
  - Automatic recovery: successful trial resets the circuit

The circuit breaker protects downstream services by stopping requests
after a configurable number of consecutive failures. After a cooldown
period, a single trial request is allowed (half-open). If it succeeds,
the circuit closes. If it fails, the circuit re-opens.

Usage:
    uv run python 08_circuit_breaker.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/08_circuit_breaker.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import CircuitBreakerConfig


def on_circuit_open(ctx):
    """Graceful fallback when the circuit breaker is open."""
    print(f"  [on_error] Circuit is OPEN: {ctx.error_message}")
    return f"Service unavailable: {ctx.error_message}"


async def run_with_possible_failure(agent, prompt, session_id, memory, step_label):
    """Run an agent step, catching circuit breaker or runtime errors."""
    history = await MessageHistory().load(session_id, memory)
    try:
        result = await agent.run(prompt, history, session_id)
        print(f"  [{step_label}] SUCCESS: {result.output[:80]}...")
        return True
    except RuntimeError as e:
        print(f"  [{step_label}] BLOCKED: {e}")
        return False
    except Exception as e:
        print(f"  [{step_label}] ERROR: {type(e).__name__}: {e}")
        return False


async def main():
    print("=" * 60)
    print("Circuit Breaker Guardrail")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Configure circuit breaker ──────────────────────────────
    cb = (
        CircuitBreakerConfig()
        .with_threshold(3)      # open after 3 consecutive failures
        .with_timeout(5)        # wait 5s before half-open trial
        .on_error(on_circuit_open)
    )

    # ── Agent that will fail (invalid model) ───────────────────
    print(f"\nCircuit config: threshold={cb.failure_threshold}, timeout={cb.circuit_timeout}s")
    print("\nUsing an intentionally broken model to trigger failures...\n")

    bad_agent = (
        ManagedAgent()
        .with_model(ModelConfig(
            provider="ollama",
            model_name="this-model-does-not-exist"  # will fail
        ))
        .with_circuit_breaker(cb)
    )

    # ── Generate failures to trip the circuit ──────────────────
    print("--- Phase 1: Generating failures ---")
    for i in range(1, 6):
        print(f"\nRequest {i}:")
        await run_with_possible_failure(
            bad_agent, "Hello", f"cb-demo-fail-{i}", memory, f"req-{i}"
        )

    print("\n--- Phase 2: Circuit should be OPEN ---")

    # ── Good agent, but circuit breaker is shared? No ──────────
    # The circuit breaker state is per GuardRunner instance.
    # Since bad_agent has its own GuardRunner, only its requests are blocked.

    print("\n--- Phase 3: Demonstrate HALF-OPEN recovery ---")
    print("Waiting for circuit timeout (5s)...")
    await asyncio.sleep(6)

    # After timeout, first request should be allowed (half-open)
    good_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_circuit_breaker(
            CircuitBreakerConfig()
            .with_threshold(3)
            .with_timeout(5)
        )
    )

    print("\nTrial request after timeout (half-open):")
    success = await run_with_possible_failure(
        good_agent, "What is 1+1?", "cb-demo-recover", memory, "half-open"
    )

    if success:
        print("  Circuit is now CLOSED (recovered)")
    else:
        print("  Circuit is still OPEN")


if __name__ == "__main__":
    asyncio.run(main())
