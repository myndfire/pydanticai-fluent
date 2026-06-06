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

"""Validation retries — custom business rules with output validators.

Pydantic validates field types and constraints (int, str, ge, le, etc.).
For business rules (e.g. "end_time must be after start_time"), use a
custom output validator that raises ModelRetry. Each ModelRetry triggers
the agent to try again with a new response.

Usage:
    uv run python 04_validation_retries.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python structured_output/04_validation_retries.py
"""

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


class MeetingScheduler(BaseModel):
    """A scheduled meeting with time validation."""
    title: str = Field(description="Meeting title or subject")
    start_time: str = Field(description="Start time in HH:MM format, e.g. '14:00'")
    end_time: str = Field(description="End time in HH:MM format, e.g. '15:00'")
    attendees: list[str] = Field(default_factory=list, description="List of attendee names")


async def main():
    print("=" * 60)
    print("Validation Retries — Business Rules with ModelRetry")
    print("=" * 60)

    # ── Build agent with output validator ───────────────────────
    # The validator runs after Pydantic schema validation.
    # It enforces business rules that Pydantic can't express as type constraints.

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="ollama:phi4-mini-reasoning"))
        .with_output(MeetingScheduler, output_retries=5)
    )

    # Attach a custom output validator to the underlying PydanticAI agent
    raw_agent = agent.get_agent()

    @raw_agent.output_validator
    async def validate_meeting(data: MeetingScheduler) -> MeetingScheduler:
        """Validate business rules: end_time must be after start_time."""
        start_h, start_m = map(int, data.start_time.split(":"))
        end_h, end_m = map(int, data.end_time.split(":"))

        if end_h < start_h or (end_h == start_h and end_m <= start_m):
            raise ModelRetry(
                f"End time {data.end_time} must be after start time {data.start_time}. "
                f"Please suggest a valid end time."
            )

        return data

    memory = InMemoryProvider()

    # ── Run 1: Valid meeting ────────────────────────────────────
    print("\n--- Valid meeting (should pass) ---")
    history = await MessageHistory().load("valid-1", memory)
    result = await agent.run(
        "Schedule a 1-hour project review meeting starting at 2pm "
        "with Alice, Bob, and Charlie.",
        history, "valid-1",
    )
    m = result.output
    print(f"  Title:     {m.title}")
    print(f"  Start:     {m.start_time}")
    print(f"  End:       {m.end_time}")
    print(f"  Attendees: {', '.join(m.attendees)}")

    # ── Run 2: Impossible meeting ───────────────────────────────
    print("\n--- Impossible meeting (should fail validation) ---")
    history2 = await MessageHistory().load("valid-2", memory)
    try:
        result2 = await agent.run(
            "Schedule a meeting from 3pm to 1pm with the marketing team.",
            history2, "valid-2",
        )
        m2 = result2.output
        print(f"  Unexpectedly succeeded: {m2}")
    except Exception as e:
        print(f"  ✗ Validation exhausted after retries: {type(e).__name__}")

    # ── How ModelRetry works ────────────────────────────────────
    print(f"\n--- How ModelRetry works ---")
    print(f"  1. Agent generates a response matching the MeetingScheduler schema")
    print(f"  2. Pydantic validates field types (str, list[str])")
    print(f"  3. Custom validator checks business rules (end > start)")
    print(f"  4. If rules fail → raise ModelRetry('reason')")
    print(f"  5. Agent sees the error message and tries again")
    print(f"  6. After output_retries=5 attempts → gives up, raises exception")


if __name__ == "__main__":
    asyncio.run(main())
