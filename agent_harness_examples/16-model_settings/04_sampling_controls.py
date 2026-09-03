"""
Model Settings — Sampling Controls

Demonstrates fine-grained sampling parameters:
  - top_p            → nucleus sampling (0.1 = narrow, 1.0 = all tokens)
  - top_k            → limit to top K token candidates
  - seed             → reproducible output (provider-dependent)
  - presence_penalty → penalize tokens already present in output
  - frequency_penalty → penalize frequently used tokens

Run:
    uv run python 04_sampling_controls.py
"""

import asyncio
import os

from dotenv import load_dotenv
from agent_harness import ManagedAgent
from agent_harness.model_config import ModelConfig
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.prompts import StaticPrompts
from pydantic_ai.settings import ModelSettings
import structlog

load_dotenv()

log = structlog.get_logger()

MODEL = os.getenv("MODEL_NAME", "qwen2.5:3b")
PROVIDER = os.getenv("PROVIDER", "ollama")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT = "Write a short paragraph about the ocean."


def build_agent(**kwargs) -> ManagedAgent:
    settings = {"max_tokens": 200, "temperature": 0.7}
    settings.update(kwargs)
    return (
        ManagedAgent()
        .with_model(
            ModelConfig(
                provider=PROVIDER,
                model_name=MODEL,
                base_url=BASE_URL,
            )
        )
        .with_model_settings(ModelSettings(settings))
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(StaticPrompts(SYSTEM_PROMPT))
    )


async def run_config(label: str, settings: dict) -> None:
    agent = build_agent(**settings)
    session = f"sampling-{label}"
    history = await MessageHistory().load(session, agent._short_term_memory)
    result = await agent.run(PROMPT, history, session)
    log.debug("run_result", label=label, settings=str(settings), output=str(result.output))


async def main() -> None:
    log.debug("section_header", title="Sampling Controls Demo")

    await run_config("top_p narrow", {"top_p": 0.1})
    await run_config("top_p wide", {"top_p": 1.0})
    await run_config("top_k=10", {"top_k": 10})
    await run_config("seed=42", {"seed": 42})
    await run_config("presence_penalty=1.5", {"presence_penalty": 1.5})
    await run_config("frequency_penalty=1.0", {"frequency_penalty": 1.0})


if __name__ == "__main__":
    asyncio.run(main())
