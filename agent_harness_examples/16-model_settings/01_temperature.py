"""
Model Settings — Temperature

Demonstrates how temperature controls response randomness.
  - 0.0 → deterministic, focused
  - 0.5 → balanced
  - 1.0 → creative, varied

Run:
    uv run python 01_temperature.py
"""

import asyncio
import os

from dotenv import load_dotenv
from agent_harness import ManagedAgent
from agent_harness.model_config import ModelConfig
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.prompts import StaticPrompts
from pydantic_ai.settings import ModelSettings

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "qwen2.5:3b")
PROVIDER = os.getenv("PROVIDER", "ollama")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")

SYSTEM_PROMPT = "You are a concise assistant. Answer in one or two sentences."

PROMPT = "What is the capital of France?"


def build_agent(temperature: float) -> ManagedAgent:
    return (
        ManagedAgent()
        .with_model(
            ModelConfig(
                provider=PROVIDER,
                model_name=MODEL,
                base_url=BASE_URL,
            )
        )
        .with_model_settings(
            ModelSettings(
                temperature=temperature,
                max_tokens=128,
            )
        )
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(StaticPrompts(SYSTEM_PROMPT))
    )


async def run_with_temperature(label: str, temperature: float) -> None:
    agent = build_agent(temperature)
    history = await MessageHistory().load(f"temp-{temperature}", agent._short_term_memory)
    result = await agent.run(PROMPT, history, f"temp-{temperature}")
    print(f"  [{label}] temperature={temperature}")
    print(f"  → {result.output}\n")


async def main() -> None:
    print("=== Temperature Demo ===\n")

    await run_with_temperature("deterministic", 0.0)
    await run_with_temperature("balanced", 0.5)
    await run_with_temperature("creative", 1.0)


if __name__ == "__main__":
    asyncio.run(main())
