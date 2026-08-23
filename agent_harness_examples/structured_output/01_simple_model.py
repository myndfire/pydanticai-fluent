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

"""Basic structured output — constraining responses to a Pydantic model.

.with_output(WeatherReport) tells the agent to always return a validated
WeatherReport object. The LLM generates JSON, pydantic validates it against
the model schema. result.output is a typed instance — no manual parsing.

Compared to unstructured output, structured output guarantees the response
has exactly the fields you expect, with the right types.

Usage:
    uv run python 01_simple_model.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python structured_output/01_simple_model.py
"""

import asyncio
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


class WeatherReport(BaseModel):
    """A weather report for a city."""
    city: str = Field(description="The city name")
    temperature_f: int = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="e.g. sunny, rainy, cloudy")
    humidity_percent: int = Field(description="Humidity as a percentage (0-100)")
    wind_mph: int = Field(description="Wind speed in miles per hour")


async def main():
    print("=" * 60)
    print("Structured Output — Typed Weather Report")
    print("=" * 60)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="phi4-mini"))
        .with_output(WeatherReport, output_retries=3)
    )

    memory = InMemoryProvider()

    # ── Run 1: Tokyo ────────────────────────────────────────────
    print("\n--- Tokyo ---")
    history = await MessageHistory().load("struct-tokyo", memory)
    result = await agent.run(
        "What is the current weather in Tokyo?",
        history, "struct-tokyo",
    )

    report = result.output
    print(f"  City:      {report.city}")
    print(f"  Temp:      {report.temperature_f}F")
    print(f"  Conditions:{report.conditions}")
    print(f"  Humidity:  {report.humidity_percent}%")
    print(f"  Wind:      {report.wind_mph} mph")

    # ── Run 2: London ───────────────────────────────────────────
    print("\n--- London ---")
    history2 = await MessageHistory().load("struct-london", memory)
    result2 = await agent.run(
        "What is the current weather in London?",
        history2, "struct-london",
    )

    report2 = result2.output
    print(f"  City:      {report2.city}")
    print(f"  Temp:      {report2.temperature_f}F")
    print(f"  Conditions:{report2.conditions}")

    # ── Comparison ──────────────────────────────────────────────
    print(f"\n--- Typed access ---")
    print(f"  result.output is a WeatherReport: {isinstance(result.output, WeatherReport)}")
    print(f"  Fields are typed: temperature_f is int = {report.temperature_f}")
    print(f"  IDE autocomplete works on {type(report).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
