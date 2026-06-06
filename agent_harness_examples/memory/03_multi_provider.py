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

"""Multiple memory providers — split persistence across backends.

Demonstrates:
  - save_to=[short_term, long_term, audit]: persist turn to multiple providers
  - Short-term: InMemoryProvider for fast context retrieval
  - Long-term: separate InMemoryProvider for full archive
  - Audit log: third provider for compliance/analytics
  - TurnData inspection: turn_id, timestamp, usage, status, duration
  - UsageData: token breakdown by turn
  - last_turn property after each run

Usage:
    python 03_multi_provider.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


async def inspect_turn(label: str, turn):
    """Print TurnData details."""
    if turn is None:
        print(f"  {label}: None")
        return
    has_usage = turn.usage is not None
    msg_count = len(turn.messages)
    print(f"  {label}:")
    print(f"    turn_id:    {turn.turn_id}")
    print(f"    status:     {turn.status}")
    print(f"    model:      {turn.model}")
    print(f"    duration:   {turn.duration_seconds:.2f}s")
    print(f"    messages:   {msg_count}")
    if has_usage:
        print(f"    usage:      in={turn.usage.input_tokens} "
              f"out={turn.usage.output_tokens} "
              f"total={turn.usage.total_tokens}")
    print(f"    timestamp:  {turn.timestamp.isoformat()}")


async def main():
    print("=" * 60)
    print("Multiple Memory Providers")
    print("=" * 60)

    # ── Three providers, three purposes ──────────────────────────
    short_term = InMemoryProvider(max_turns=10)    # fast context
    long_term = InMemoryProvider(max_turns=1000)   # full archive
    audit_log = InMemoryProvider(max_turns=10000)  # compliance trail

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
    )

    session = "multi-prov-demo"
    print(f"\nSession: {session}")
    print(f"Save targets: short_term + long_term + audit_log")

    # ── Turn 1 ──────────────────────────────────────────────────
    print("\n--- Turn 1 ---")
    history = await MessageHistory().load(session, short_term)
    result = await agent.run(
        "Remember this: the secret code is 'XY-42-ALPHA'.",
        history,
        session,
        save_to=[short_term, long_term, audit_log],
    )

    inspect_turn("last_turn", agent.last_turn)

    # ── Turn 2 ──────────────────────────────────────────────────
    print("\n--- Turn 2 ---")
    history2 = await MessageHistory().load(session, short_term)
    result2 = await agent.run(
        "What was the secret code I told you?",
        history2,
        session,
        save_to=[short_term, long_term, audit_log],
    )
    print(f"  Response: {result2.output}")

    # ── Turn 3 ──────────────────────────────────────────────────
    print("\n--- Turn 3 ---")
    history3 = await MessageHistory().load(session, short_term)
    result3 = await agent.run(
        "Confirm the secret code and tell me the current time.",
        history3,
        session,
        save_to=[short_term, long_term, audit_log],
    )
    print(f"  Response: {result3.output}")

    # ── Compare storage across providers ────────────────────────
    print("\n--- Storage comparison ---")
    for name, provider in [
        ("short_term", short_term),
        ("long_term", long_term),
        ("audit_log", audit_log),
    ]:
        turns = await provider.load_turns(session)
        total_tokens = sum(
            (t.usage.total_tokens if t.usage else 0) for t in turns
        )
        total_dur = sum(t.duration_seconds for t in turns)
        print(f"  {name}: {len(turns)} turns, "
              f"{total_tokens} total tokens, "
              f"{total_dur:.2f}s total duration")

    # ── Inspect a specific turn from the audit log ──────────────
    print("\n--- Audit trail (first turn) ---")
    audit_turns = await audit_log.load_turns(session, limit=1)
    if audit_turns:
        inspect_turn("audit[0]", audit_turns[0])

    # ── to_dict / from_dict round-trip ──────────────────────────
    print("\n--- TurnData.to_dict() / from_dict() round-trip ---")
    if audit_turns:
        turn_dict = audit_turns[0].to_dict()
        print(f"  Serialized keys: {list(turn_dict.keys())}")
        print(f"  timestamp as string: {turn_dict['timestamp']}")

        rebuilt = audit_turns[0].__class__.from_dict(turn_dict)
        print(f"  Round-trip OK: {rebuilt.turn_id == audit_turns[0].turn_id}")


if __name__ == "__main__":
    asyncio.run(main())
