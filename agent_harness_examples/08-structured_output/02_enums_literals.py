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

"""Classification with Literal types — constrained enum output.

Literal[...] types restrict the LLM to a fixed set of valid values. If the
LLM returns "happy" instead of "positive", pydantic rejects it and the
agent retries. This makes classification tasks reliable and consistent.

Usage:
    uv run python 02_enums_literals.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python structured_output/02_enums_literals.py
"""

import asyncio
import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("STRUCTURED_OUTPUT_REASONING_MODEL", "phi4-mini-reasoning")
MAX_TOKENS = int(os.getenv("STRUCTURED_OUTPUT_MAX_TOKENS", "512"))


class SentimentResult(BaseModel):
    """Sentiment analysis result with constrained output."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0"
    )
    keywords: list[str] = Field(
        description="Key words or phrases that influenced the sentiment"
    )
    reasoning: str = Field(
        description="Brief explanation of why this sentiment was chosen"
    )


async def main():
    """Run the enums/literals structured output example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull phi4-mini-reasoning
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Classification with Literal Types")
    print("=" * 60)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_output(SentimentResult, output_retries=3)
    )

    memory = InMemoryProvider()

    # ── Positive ────────────────────────────────────────────────
    print("\n--- Positive review ---")
    history = await MessageHistory().load("sent-pos", memory)
    result = await agent.run(
        "Analyze this review: 'Absolutely love this product! "
        "It arrived early and works perfectly. Best purchase ever.'",
        history, "sent-pos",
    )
    r = result.output
    print(f"  Sentiment:  {r.sentiment}")
    print(f"  Confidence: {r.confidence:.2f}")
    print(f"  Keywords:   {', '.join(r.keywords)}")
    print(f"  Reasoning:  {r.reasoning}")

    # ── Negative ────────────────────────────────────────────────
    print("\n--- Negative review ---")
    history2 = await MessageHistory().load("sent-neg", memory)
    result2 = await agent.run(
        "Analyze this review: 'Terrible experience. It broke after "
        "two days and customer service ignored my emails.'",
        history2, "sent-neg",
    )
    r2 = result2.output
    print(f"  Sentiment:  {r2.sentiment}")
    print(f"  Confidence: {r2.confidence:.2f}")
    print(f"  Keywords:   {', '.join(r2.keywords)}")

    # ── Neutral ─────────────────────────────────────────────────
    print("\n--- Neutral review ---")
    history3 = await MessageHistory().load("sent-neu", memory)
    result3 = await agent.run(
        "Analyze this review: 'Product works as described. Nothing "
        "special but gets the job done. Average quality.'",
        history3, "sent-neu",
    )
    r3 = result3.output
    print(f"  Sentiment:  {r3.sentiment}")
    print(f"  Confidence: {r3.confidence:.2f}")

    # ── How Literal works ──────────────────────────────────────
    print(f"\n--- How Literal types work ---")
    print(f"  The model can only return: positive, negative, or neutral")
    print(f"  If it returns 'happy' → pydantic rejects → agent retries")
    print(f"  This guarantees consistent classification labels")


if __name__ == "__main__":
    asyncio.run(main())
