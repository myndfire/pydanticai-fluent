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

"""Result validator retries for structured output validation failures.

Demonstrates:
  - Retries when PydanticAI's output validation fails
  - Structured output with a Pydantic model
  - Combined agent + validator retry configuration

Result validator retries correspond to PydanticAI's
@agent.output_validator with ModelRetry exception.

Usage:
    python 03_result_validator_retries.py
"""

import asyncio
from pydantic import BaseModel

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig, ResultValidatorRetryConfig


class WeatherReport(BaseModel):
    """Structured weather report."""
    temperature_f: int
    conditions: str
    humidity_percent: int


async def main():
    print("=" * 60)
    print("Result Validator Retries with Structured Output")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("validator-demo", memory)

    agent_retry = AgentRetryConfig().with_max_retries(2).with_timeout(30)
    validator_retry = (
        ResultValidatorRetryConfig()
        .with_max_retries(3)      # retry up to 3 times on validation failure
        .with_backoff(2.0)        # 2x exponential backoff
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_output(WeatherReport, output_retries=validator_retry.max_retries)
        .with_agent_retries(agent_retry)
        .with_result_validator_retries(validator_retry)
    )

    # ── Run ────────────────────────────────────────────────────
    print(f"\nOutput type: WeatherReport")
    print(f"Validator max retries: {validator_retry.max_retries}")
    print(f"Validator backoff: {validator_retry.backoff_multiplier}x")
    print(f"\nSending prompt: 'What's the weather like in San Francisco? "
          f"Return temperature, conditions, and humidity.'...\n")

    result = await agent.run(
        "What's the weather like in San Francisco? "
        "Return temperature in Fahrenheit, conditions (sunny/rainy/cloudy), "
        "and humidity as a percentage.",
        history,
        "validator-demo",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Output type: {type(result.output).__name__}")
    if isinstance(result.output, WeatherReport):
        print(f"  Temperature: {result.output.temperature_f}F")
        print(f"  Conditions: {result.output.conditions}")
        print(f"  Humidity: {result.output.humidity_percent}%")


if __name__ == "__main__":
    asyncio.run(main())
