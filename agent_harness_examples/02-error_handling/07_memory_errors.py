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

"""Memory errors — handling failures when memory providers fail.

Demonstrates:
  - Custom MemoryProvider that raises on save_turn()
  - on_memory_error callback to handle storage failures gracefully
  - Source="memory" from agent.run() _error_source tagging
  - Two memory error origins: load (line 425-428) and save (line 517-520)
  - Agent continues without persistence when memory is unavailable

Usage:
    uv run python 07_memory_errors.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/07_memory_errors.py
"""

import asyncio
from typing import Optional

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory, TurnData, MemoryProvider
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext


# ── Mock MemoryProvider that raises on save ─────────────────────────

class FailingMemoryProvider:
    """A MemoryProvider that raises on save_turn() — simulates DB outage."""

    def __init__(self, name: str = "broken-db"):
        self.name = name
        self._storage: dict[str, list[TurnData]] = {}

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        print(f"  [memory:{self.name}] save_turn FAILING deliberately!")
        raise ConnectionError(f"{self.name}: database connection refused")

    async def load_turns(self, session_id: str, limit: Optional[int] = None):
        return self._storage.get(session_id, [])

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        return None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        return False

    async def clear(self, session_id: str) -> None:
        pass


# ── Mock MemoryProvider that raises on load ─────────────────────────

class FailingLoadProvider:
    """A MemoryProvider that raises on load_turns() — simulates connection loss."""

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        pass

    async def load_turns(self, session_id: str, limit: Optional[int] = None):
        raise ConnectionError(f"Connection pool exhausted for session {session_id}")

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        return None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        return False

    async def clear(self, session_id: str) -> None:
        pass


# ── Memory error handler ────────────────────────────────────────────

def on_memory_failure(ctx: ErrorContext) -> str | None:
    """Handle memory failures — suppress, continue without persistence."""
    print(f"\n  [on_memory_error] Memory failure detected!")
    print(f"    Type:    {ctx.error_type}")
    print(f"    Message: {ctx.error_message}")
    print(f"    Session: {ctx.session_id}")
    return None  # suppress — agent continues, just without persistence


def on_memory_with_log(ctx: ErrorContext) -> str | None:
    """Handle memory failures with a warning message in the output."""
    print(f"  [on_memory_error] {ctx.error_type}: {ctx.error_message[:80]}")
    return (
        f"[Warning: persistence unavailable — {ctx.error_type}] "
        f"Agent will continue without saving this turn."
    )


async def main():
    print("=" * 60)
    print("Memory Errors — Handling Storage Failures")
    print("=" * 60)

    # ── Example 1: save_turn() fails ────────────────────────────
    print("\n--- Example 1: save_turn() fails (save_to provider raises) ---")

    broken_save = FailingMemoryProvider("mongo-primary")
    short_term = InMemoryProvider(max_turns=10)

    config1 = ErrorHandlingConfig().on_memory_error(on_memory_failure)

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_error_handling(config1)
    )

    # Short-term memory (load) works, but save_to includes the broken one
    history1 = await MessageHistory().load("mem-err-1", short_term)
    result1 = await agent1.run(
        "What is 2+2? Just the number.",
        history1,
        "mem-err-1",
        save_to=[broken_save],  # this will fail during save
    )
    print(f"\n  Result: success={result1.success}")
    print(f"  Output: {result1.output}")

    # ── Example 2: load_turns() fails ───────────────────────────
    print("\n--- Example 2: load_turns() fails (history load raises) ---")

    broken_load = FailingLoadProvider()

    config2 = ErrorHandlingConfig().on_memory_error(
        lambda ctx: f"Memory unavailable at startup: {ctx.error_type}. Continuing with empty context."
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(broken_load)
        .with_error_handling(config2)
    )

    history2 = await MessageHistory().load("mem-err-2", broken_load)
    result2 = await agent2.run(
        "Say hello in 3 words.",
        history2,
        "mem-err-2",
    )
    print(f"\n  Result: success={result2.success}")
    print(f"  Output: {result2.output}")

    # ── Example 3: Save failure with fallback output ────────────
    print("\n--- Example 3: Save failure with fallback output ---")

    broken_save2 = FailingMemoryProvider("redis-cache")

    config3 = ErrorHandlingConfig().on_memory_error(on_memory_with_log)

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_error_handling(config3)
    )

    history3 = await MessageHistory().load("mem-err-3", InMemoryProvider())
    result3 = await agent3.run(
        "What is the capital of France?",
        history3,
        "mem-err-3",
        save_to=[broken_save2],
    )
    print(f"\n  Result: success={result3.success}")
    print(f"  Output: {result3.output}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n--- Memory error origination points ---")
    print("  1. message_history.load(session_id, memory_provider)")
    print("     → _error_source = 'memory'")
    print("     → Routed to on_memory_error")
    print()
    print("  2. provider.save_turn(session_id, turn)")
    print("     → _error_source = 'memory'")
    print("     → Routed to on_memory_error")


if __name__ == "__main__":
    asyncio.run(main())
