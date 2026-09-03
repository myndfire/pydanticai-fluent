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
import os
from dotenv import load_dotenv
import structlog
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

log = structlog.get_logger()

MODEL_NAME = os.getenv("STRUCTURED_OUTPUT_MODEL_NAME", "phi4-mini")
MAX_TOKENS = int(os.getenv("STRUCTURED_OUTPUT_MAX_TOKENS", "512"))


class WeatherReport(BaseModel):
    """A weather report for a city."""
    city: str = Field(description="The city name")
    temperature_f: int = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="e.g. sunny, rainy, cloudy")
    humidity_percent: int = Field(description="Humidity as a percentage (0-100)")
    wind_mph: int = Field(description="Wind speed in miles per hour")


async def main():
    """Run the simple structured output example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull phi4-mini
        3. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator")
    log.debug("title", title="Structured Output — Typed Weather Report")
    log.debug("separator")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_output(WeatherReport, output_retries=3)
    )

    memory = InMemoryProvider()

    # ── Run 1: Tokyo ────────────────────────────────────────────
    log.debug("section", title="Tokyo")
    history = await MessageHistory().load("struct-tokyo", memory)
    result = await agent.run(
        "What is the current weather in Tokyo?",
        history, "struct-tokyo",
    )

    report = result.output
    log.debug("field", field="city", value=report.city)
    log.debug("field", field="temp", value=f"{report.temperature_f}F")
    log.debug("field", field="conditions", value=report.conditions)
    log.debug("field", field="humidity", value=f"{report.humidity_percent}%")
    log.debug("field", field="wind", value=f"{report.wind_mph} mph")

    # ── Run 2: London ───────────────────────────────────────────
    log.debug("section", title="London")
    history2 = await MessageHistory().load("struct-london", memory)
    result2 = await agent.run(
        "What is the current weather in London?",
        history2, "struct-london",
    )

    report2 = result2.output
    log.debug("field", field="city", value=report2.city)
    log.debug("field", field="temp", value=f"{report2.temperature_f}F")
    log.debug("field", field="conditions", value=report2.conditions)

    # ── Comparison ──────────────────────────────────────────────
    log.debug("section", title="Typed access")
    log.debug("type_check", is_weather_report=isinstance(result.output, WeatherReport))
    log.debug("field", field="temperature_f", value=report.temperature_f, note="Fields are typed")
    log.debug("field", field="type", value=type(report).__name__, note="IDE autocomplete works")


if __name__ == "__main__":
    asyncio.run(main())
