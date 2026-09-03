"""
Model Settings — Combined Settings

Practical example combining multiple model settings for different use cases:
  1. Creative writer → high temperature, high top_p, reasoning enabled
  2. Factual analyst → low temperature, low top_p, max_tokens capped

Also demonstrates per-run override via the model_settings kwarg on run().

Run:
    uv run python 05_combined_settings.py
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


def build_creative_agent() -> ManagedAgent:
    """Agent tuned for creative, varied output."""
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
                temperature=0.9,
                top_p=0.95,
                max_tokens=300,
            )
        )
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(
            StaticPrompts(
                "You are a creative writer. Use vivid language and metaphors."
            )
        )
    )


def build_factual_agent() -> ManagedAgent:
    """Agent tuned for precise, factual output."""
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
                temperature=0.0,
                top_p=0.5,
                max_tokens=150,
                seed=42,
            )
        )
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(
            StaticPrompts(
                "You are a factual analyst. Be precise and concise. Cite specifics."
            )
        )
    )


async def main() -> None:
    log.debug("section_header", title="Combined Settings Demo")

    # --- 1. Creative writer ---
    creative = build_creative_agent()
    history = await MessageHistory().load("creative-1", creative._short_term_memory)
    result = await creative.run(
        "Write a short poem about the night sky.",
        history,
        "creative-1",
    )
    log.debug("run_result", label="creative writer", temperature=0.9, top_p=0.95, max_tokens=300, output=str(result.output))

    # --- 2. Factual analyst ---
    factual = build_factual_agent()
    history = await MessageHistory().load("factual-1", factual._short_term_memory)
    result = await factual.run(
        "What are the three laws of thermodynamics? Summarize each in one sentence.",
        history,
        "factual-1",
    )
    log.debug("run_result", label="factual analyst", temperature=0.0, top_p=0.5, max_tokens=150, seed=42, output=str(result.output))

    # --- 3. Per-run override ---
    # Same agent, but override settings at run time
    history = await MessageHistory().load("override-1", creative._short_term_memory)
    result = await creative.run(
        "Describe the color blue.",
        history,
        "override-1",
        model_settings=ModelSettings(
            temperature=0.0,
            max_tokens=80,
        ),
    )
    log.debug("run_result", label="per-run override", temperature=0.0, max_tokens=80, output=str(result.output))


if __name__ == "__main__":
    asyncio.run(main())
