"""
Model Settings — Max Tokens

Demonstrates how max_tokens controls output length.
  - 30  → heavily truncated
  - 128 → moderate
  - 512 → longer, more complete

Run:
    uv run python 02_max_tokens.py
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

SYSTEM_PROMPT = "You are a helpful assistant. Be thorough in your explanations."

PROMPT = "Explain what a neural network is in simple terms."


def build_agent(max_tokens: int) -> ManagedAgent:
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
                max_tokens=max_tokens,
                temperature=0.3,
            )
        )
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(StaticPrompts(SYSTEM_PROMPT))
    )


async def run_with_max_tokens(label: str, max_tokens: int) -> None:
    agent = build_agent(max_tokens)
    history = await MessageHistory().load(f"mt-{max_tokens}", agent._short_term_memory)
    result = await agent.run(PROMPT, history, f"mt-{max_tokens}")
    word_count = len(result.output.split())
    log.debug("run_result", label=label, max_tokens=max_tokens, word_count=word_count, output=str(result.output))


async def main() -> None:
    log.debug("section_header", title="Max Tokens Demo")

    await run_with_max_tokens("short", 30)
    await run_with_max_tokens("medium", 128)
    await run_with_max_tokens("long", 512)


if __name__ == "__main__":
    asyncio.run(main())
