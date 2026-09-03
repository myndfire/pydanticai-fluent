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
import structlog
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

log = structlog.get_logger()

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
    log.debug("separator")
    log.debug("title", title="Classification with Literal Types")
    log.debug("separator")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_output(SentimentResult, output_retries=3)
    )

    memory = InMemoryProvider()

    # ── Positive ────────────────────────────────────────────────
    log.debug("section", title="Positive review")
    history = await MessageHistory().load("sent-pos", memory)
    result = await agent.run(
        "Analyze this review: 'Absolutely love this product! "
        "It arrived early and works perfectly. Best purchase ever.'",
        history, "sent-pos",
    )
    r = result.output
    log.debug("field", field="sentiment", value=r.sentiment)
    log.debug("field", field="confidence", value=f"{r.confidence:.2f}")
    log.debug("field", field="keywords", value=", ".join(r.keywords))
    log.debug("field", field="reasoning", value=r.reasoning)

    # ── Negative ────────────────────────────────────────────────
    log.debug("section", title="Negative review")
    history2 = await MessageHistory().load("sent-neg", memory)
    result2 = await agent.run(
        "Analyze this review: 'Terrible experience. It broke after "
        "two days and customer service ignored my emails.'",
        history2, "sent-neg",
    )
    r2 = result2.output
    log.debug("field", field="sentiment", value=r2.sentiment)
    log.debug("field", field="confidence", value=f"{r2.confidence:.2f}")
    log.debug("field", field="keywords", value=", ".join(r2.keywords))

    # ── Neutral ─────────────────────────────────────────────────
    log.debug("section", title="Neutral review")
    history3 = await MessageHistory().load("sent-neu", memory)
    result3 = await agent.run(
        "Analyze this review: 'Product works as described. Nothing "
        "special but gets the job done. Average quality.'",
        history3, "sent-neu",
    )
    r3 = result3.output
    log.debug("field", field="sentiment", value=r3.sentiment)
    log.debug("field", field="confidence", value=f"{r3.confidence:.2f}")

    # ── How Literal works ──────────────────────────────────────
    log.debug("section", title="How Literal types work")
    log.debug("info", message="The model can only return: positive, negative, or neutral")
    log.debug("info", message="If it returns 'happy' → pydantic rejects → agent retries")
    log.debug("info", message="This guarantees consistent classification labels")


if __name__ == "__main__":
    asyncio.run(main())
