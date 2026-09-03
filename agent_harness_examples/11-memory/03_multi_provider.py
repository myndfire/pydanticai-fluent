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
    uv run python 03_multi_provider.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/03_multi_provider.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MEMORY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))


async def inspect_turn(label: str, turn):
    """Print TurnData details."""
    if turn is None:
        log.debug("turn_inspect", label=label, value=None)
        return
    has_usage = turn.usage is not None
    msg_count = len(turn.messages)
    log.debug("turn_inspect", label=label, turn_id=turn.turn_id, status=turn.status,
              model=turn.model, duration=turn.duration_seconds, messages=msg_count)
    if has_usage:
        log.debug("turn_usage", input_tokens=turn.usage.input_tokens,
                  output_tokens=turn.usage.output_tokens, total_tokens=turn.usage.total_tokens)
    log.debug("turn_timestamp", timestamp=turn.timestamp.isoformat())


async def main():
    """Run the multi-provider example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="Multiple Memory Providers")
    log.debug("separator", width=60)

    # ── Three providers, three purposes ──────────────────────────
    short_term = InMemoryProvider(max_turns=int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "10")))    # fast context
    long_term = InMemoryProvider(max_turns=int(os.getenv("MEMORY_LONG_TERM_MAX_TURNS", "100")))     # full archive
    audit_log = InMemoryProvider(max_turns=int(os.getenv("MEMORY_AUDIT_MAX_TURNS", "10000")))       # compliance trail

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
    )

    session = "multi-prov-demo"
    log.debug("session", session=session)
    log.debug("save_targets", targets="short_term + long_term + audit_log")

    # ── Turn 1 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 1")
    history = await MessageHistory().load(session, short_term)
    result = await agent.run(
        "Remember this: the secret code is 'XY-42-ALPHA'.",
        history,
        session,
        save_to=[short_term, long_term, audit_log],
    )

    inspect_turn("last_turn", agent.last_turn)

    # ── Turn 2 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 2")
    history2 = await MessageHistory().load(session, short_term)
    result2 = await agent.run(
        "What was the secret code I told you?",
        history2,
        session,
        save_to=[short_term, long_term, audit_log],
    )
    log.debug("response", output=result2.output)

    # ── Turn 3 ──────────────────────────────────────────────────
    log.debug("section", title="Turn 3")
    history3 = await MessageHistory().load(session, short_term)
    result3 = await agent.run(
        "Confirm the secret code and tell me the current time.",
        history3,
        session,
        save_to=[short_term, long_term, audit_log],
    )
    log.debug("response", output=result3.output)

    # ── Compare storage across providers ────────────────────────
    log.debug("section", title="Storage comparison")
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
        log.debug("storage", provider=name, turns=len(turns), total_tokens=total_tokens, total_duration=total_dur)

    # ── Inspect a specific turn from the audit log ──────────────
    log.debug("section", title="Audit trail (first turn)")
    audit_turns = await audit_log.load_turns(session, limit=1)
    if audit_turns:
        inspect_turn("audit[0]", audit_turns[0])

    # ── to_dict / from_dict round-trip ──────────────────────────
    log.debug("section", title="TurnData.to_dict() / from_dict() round-trip")
    if audit_turns:
        turn_dict = audit_turns[0].to_dict()
        log.debug("serialized", keys=list(turn_dict.keys()), timestamp=turn_dict['timestamp'])

        rebuilt = audit_turns[0].__class__.from_dict(turn_dict)
        log.debug("round_trip", ok=rebuilt.turn_id == audit_turns[0].turn_id)


if __name__ == "__main__":
    asyncio.run(main())
