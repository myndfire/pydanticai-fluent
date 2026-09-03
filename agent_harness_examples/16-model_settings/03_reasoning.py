"""
Model Settings — Reasoning / Thinking

Demonstrates how reasoning models behave differently from regular models.

On Ollama (and most local providers):
  - The `thinking` parameter is silently ignored — the model either reasons
    naturally or it doesn't, based on its training.
  - `reasoning_tokens` is always 0 because Ollama doesn't report reasoning
    separately from regular output tokens.
  - Reasoning models (e.g., phi4-mini-reasoning, deepseek-r1) produce
    step-by-step thinking traces even without being asked.

On OpenAI / Anthropic:
  - `thinking=True/False` actually controls provider-level reasoning.
  - `reasoning_tokens` is populated separately from `output_tokens`.

This example uses Ollama. To see provider-level thinking control, switch to
OpenAI (gpt-5.3, gpt-5.4) or Anthropic (Claude) and set PROVIDER=openai.

Run:
    uv run python 03_reasoning.py
"""

import asyncio
import os

from dotenv import load_dotenv
from agent_harness import ManagedAgent
from agent_harness.model_config import ModelConfig
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.prompts import StaticPrompts
import structlog

load_dotenv()

log = structlog.get_logger()

# Default to the reasoning model the user already has locally.
# A non-reasoning model (e.g., phi4-mini, qwen2.5:3b) gives a short direct answer.
# A reasoning model (e.g., phi4-mini-reasoning, deepseek-r1) produces step-by-step traces.
REASONING_MODEL = os.getenv("REASONING_MODEL_NAME", "phi4-mini-reasoning")
REGULAR_MODEL = os.getenv("MODEL_NAME", "phi4-mini")
PROVIDER = os.getenv("PROVIDER", "ollama")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly."

PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep does the farmer have left?"
)


def build_agent(model_name: str, thinking=None) -> ManagedAgent:
    settings: dict = {"max_tokens": 512, "temperature": 0.3}
    if thinking is not None:
        settings["thinking"] = thinking

    return (
        ManagedAgent()
        .with_model(
            ModelConfig(
                provider=PROVIDER,
                model_name=model_name,
                base_url=BASE_URL,
            )
        )
        .with_model_settings(settings)
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(StaticPrompts(SYSTEM_PROMPT))
    )


async def run_with_config(label: str, model_name: str, thinking=None) -> None:
    agent = build_agent(model_name, thinking)
    session = f"reasoning-{label}"
    history = await MessageHistory().load(session, agent._short_term_memory)
    result = await agent.run(PROMPT, history, session)

    usage = getattr(result, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

    log.debug("run_result", label=label, model=model_name, thinking=thinking, output_tokens=output_tokens, total_tokens=total_tokens, reasoning_tokens=reasoning_tokens, output=str(result.output))


async def main() -> None:
    log.debug("section_header", title="Reasoning / Thinking Demo (Ollama)")
    log.debug("note", message="Ollama ignores the `thinking` parameter. Reasoning is determined by the model's training, not the setting. reasoning_tokens is always 0 on Ollama.")

    # 1. Regular model → short, direct answer
    await run_with_config("regular", REGULAR_MODEL)

    # 2. Reasoning model → long, step-by-step trace
    await run_with_config("reasoning", REASONING_MODEL)

    # 3. Reasoning model + thinking=False → same as #2 (parameter ignored)
    await run_with_config("reasoning-thinking-off", REASONING_MODEL, thinking=False)


if __name__ == "__main__":
    asyncio.run(main())
