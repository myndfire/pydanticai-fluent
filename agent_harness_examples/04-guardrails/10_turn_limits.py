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

"""Turn limits guardrail — cap agent invocations per session.

Demonstrates:
  - max_turns: maximum number of agent.turn() invocations per session
  - on_turn_limit callback for graceful handling when limit exceeded
  - Turn counting tracked per session_id across calls
  - Multi-session isolation (each session has its own counter)

Useful for:
  - Preventing runaway conversation loops
  - Controlling per-session cost budgets
  - Limiting abuse in multi-turn agents

Usage:
    uv run python 10_turn_limits.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/10_turn_limits.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import TurnLimitsConfig


load_dotenv()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

TURN_LIMITS_EX1_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX1_MAX_TURNS", "3"))
TURN_LIMITS_EX2_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX2_MAX_TURNS", "2"))
TURN_LIMITS_EX3_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX3_MAX_TURNS", "1"))


def on_turn_limit_handler(ctx):
    """Graceful fallback when the turn limit is reached."""
    session = ctx.session_id or "unknown"
    print(f"  [on_turn_limit] Session limit hit: {ctx.error_message}")
    return (
        f"Session {session}: maximum turns reached. "
        f"({ctx.error_message})"
    )


async def run_turn(agent, prompt, session_id, memory, label):
    """Run a single turn, returning True if successful."""
    history = await MessageHistory().load(session_id, memory)
    try:
        result = await agent.run(prompt, history, session_id)
        print(f"  [{label}] Success: {str(result)[:80]}")
        return True
    except RuntimeError as e:
        print(f"  [{label}] Blocked: {e}")
        return False


async def main():
    print("=" * 60)
    print("Turn Limits Guardrail")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Strict turn limit ────────────────────────────
    print("\n--- Example 1: max_turns=3 on a single session ---")
    turn_config = (
        TurnLimitsConfig()
        .with_max_turns(TURN_LIMITS_EX1_MAX_TURNS)
        .on_turn_limit(on_turn_limit_handler)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config)
    )

    print(f"Max turns: {turn_config.max_turns}")
    session = "turn-limit-demo-1"

    for i in range(1, 6):
        print(f"\nTurn {i}:")
        success = await run_turn(
            agent,
            f"Say 'turn number {i}' and nothing else.",
            session,
            memory,
            f"t{i}",
        )
        if not success:
            break

    # ── Example 2: Multi-session isolation ──────────────────────
    print("\n--- Example 2: Multi-session isolation (max_turns=2) ---")
    turn_config2 = TurnLimitsConfig().with_max_turns(TURN_LIMITS_EX2_MAX_TURNS)

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config2)
    )

    # Session A
    print("\nSession A:")
    for i in range(1, 4):
        success = await run_turn(
            agent2,
            f"Say 'session A turn {i}'.",
            "session-a",
            memory,
            f"A-{i}",
        )

    # Session B (separate counter — should start fresh)
    print("\nSession B:")
    for i in range(1, 4):
        success = await run_turn(
            agent2,
            f"Say 'session B turn {i}'.",
            "session-b",
            memory,
            f"B-{i}",
        )

    # ── Example 3: No callback (raises RuntimeError) ────────────
    print("\n--- Example 3: No on_turn_limit callback (raises) ---")
    turn_config3 = TurnLimitsConfig().with_max_turns(TURN_LIMITS_EX3_MAX_TURNS)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config3)
    )

    print(f"Max turns: {turn_config3.max_turns}")
    print("No on_turn_limit callback set — will raise RuntimeError...")

    # First turn should pass
    await run_turn(agent3, "Say hi.", "raise-demo", memory, "t1")

    # Second turn should raise
    print("\nTurn 2:")
    try:
        history = await MessageHistory().load("raise-demo", memory)
        await agent3.run("Say hi again.", history, "raise-demo")
    except RuntimeError as e:
        print(f"  Caught RuntimeError: {e}")

    # ── Example 4: Unlimited (no limit) ─────────────────────────
    print("\n--- Example 4: Unlimited turns (max_turns=None) ---")
    turn_config4 = TurnLimitsConfig()

    agent4 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config4)
    )

    print(f"Max turns: {turn_config4.max_turns} (unlimited)")
    for i in range(1, 4):
        await run_turn(
            agent4,
            f"Say 'unlimited turn {i}'.",
            "unlimited-demo",
            memory,
            f"u{i}",
        )

    print("\nAll turn limits examples complete.")


if __name__ == "__main__":
    asyncio.run(main())
