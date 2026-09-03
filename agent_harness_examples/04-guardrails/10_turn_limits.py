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

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import TurnLimitsConfig


load_dotenv()
log = structlog.get_logger()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

TURN_LIMITS_EX1_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX1_MAX_TURNS", "3"))
TURN_LIMITS_EX2_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX2_MAX_TURNS", "2"))
TURN_LIMITS_EX3_MAX_TURNS = int(os.getenv("TURN_LIMITS_EX3_MAX_TURNS", "1"))


def on_turn_limit_handler(ctx):
    """Graceful fallback when the turn limit is reached."""
    session = ctx.session_id or "unknown"
    log.debug("turn_limit_exceeded", session=session, error_message=ctx.error_message)
    return (
        f"Session {session}: maximum turns reached. "
        f"({ctx.error_message})"
    )


async def run_turn(agent, prompt, session_id, memory, label):
    """Run a single turn, returning True if successful."""
    history = await MessageHistory().load(session_id, memory)
    try:
        result = await agent.run(prompt, history, session_id)
        log.debug("turn_result", label=label, success=True, result=str(result)[:80])
        return True
    except RuntimeError as e:
        log.debug("turn_result", label=label, success=False, blocked=str(e))
        return False


async def main():
    log.debug("separator")
    log.debug("section", title="Turn Limits Guardrail")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Strict turn limit ────────────────────────────
    log.debug("example", example=1, title="max_turns=3 on a single session")
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

    log.debug("turn_config", max_turns=turn_config.max_turns)
    session = "turn-limit-demo-1"

    for i in range(1, 6):
        log.debug("turn", number=i)
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
    log.debug("example", example=2, title="Multi-session isolation (max_turns=2)")
    turn_config2 = TurnLimitsConfig().with_max_turns(TURN_LIMITS_EX2_MAX_TURNS)

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config2)
    )

    # Session A
    log.debug("section", title="Session A")
    for i in range(1, 4):
        success = await run_turn(
            agent2,
            f"Say 'session A turn {i}'.",
            "session-a",
            memory,
            f"A-{i}",
        )

    # Session B (separate counter — should start fresh)
    log.debug("section", title="Session B")
    for i in range(1, 4):
        success = await run_turn(
            agent2,
            f"Say 'session B turn {i}'.",
            "session-b",
            memory,
            f"B-{i}",
        )

    # ── Example 3: No callback (raises RuntimeError) ────────────
    log.debug("example", example=3, title="No on_turn_limit callback (raises)")
    turn_config3 = TurnLimitsConfig().with_max_turns(TURN_LIMITS_EX3_MAX_TURNS)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config3)
    )

    log.debug("turn_config", max_turns=turn_config3.max_turns)
    log.debug("section", title="No on_turn_limit callback set - will raise RuntimeError")

    # First turn should pass
    await run_turn(agent3, "Say hi.", "raise-demo", memory, "t1")

    # Second turn should raise
    log.debug("turn", number=2)
    try:
        history = await MessageHistory().load("raise-demo", memory)
        await agent3.run("Say hi again.", history, "raise-demo")
    except RuntimeError as e:
        log.debug("exception", caught_runtime_error=str(e))

    # ── Example 4: Unlimited (no limit) ─────────────────────────
    log.debug("example", example=4, title="Unlimited turns (max_turns=None)")
    turn_config4 = TurnLimitsConfig()

    agent4 = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_turn_limits(turn_config4)
    )

    log.debug("turn_config", max_turns=turn_config4.max_turns, note="unlimited")
    for i in range(1, 4):
        await run_turn(
            agent4,
            f"Say 'unlimited turn {i}'.",
            "unlimited-demo",
            memory,
            f"u{i}",
        )

    log.debug("separator")
    log.debug("section", title="All turn limits examples complete.")
    log.debug("separator")


if __name__ == "__main__":
    asyncio.run(main())
