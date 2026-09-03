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

"""QualityCheck — LLM-as-judge quality evaluation after each turn.

Demonstrates:
  - QualityCheck(threshold=7.0, judge_model="...")
  - A second LLM call rates the response 0–10
  - Logs a warning if score < threshold, info if passed
  - Runs after every agent.turn() — read-only, never modifies output
  - Multiple QualityCheck instances with different thresholds

Usage:
    uv run python 01_quality_check.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python evaluators/01_quality_check.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import QualityCheck

load_dotenv()
log = structlog.get_logger()


async def main():
    log.debug("separator", separator="=" * 60)
    log.debug("section", title="QualityCheck — LLM-as-Judge Evaluation")
    log.debug("separator", separator="=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Default threshold (7.0) ──────────────────────
    log.debug("example", example=1, title="QualityCheck with default threshold (7.0)")

    quality = QualityCheck(
        threshold=7.0,
        judge_model=os.getenv("QUALITY_CHECK_MODEL", "ollama:gpt-oss:20b"),
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(quality)
    )

    log.debug("evaluator_config", evaluator="QualityCheck", threshold=quality.threshold)
    log.debug("judge_model", judge_model=quality.judge_model)

    # Turn 1: should produce a high-quality answer
    history = await MessageHistory().load("quality-1", memory)
    result = await agent.run(
        "What is 2+2? Answer in exactly one word.",
        history,
        "quality-1",
    )
    log.debug("agent_output", output=result.output)

    # Turn 2: ask for a short answer (judge might rate differently)
    history2 = await MessageHistory().load("quality-2", memory)
    result2 = await agent.run(
        "Explain quantum computing in 200 words.",
        history2,
        "quality-2",
    )
    log.debug("agent_output", output=result2.output[:80], truncated=True)

    # ── Example 2: Stricter threshold ───────────────────────────
    log.debug("example", example=2, title="Stricter threshold (9.5) — almost always warns")

    strict = QualityCheck(
        threshold=9.5,
        judge_model=os.getenv("QUALITY_CHECK_MODEL", "ollama:gpt-oss:20b"),
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(strict)
    )

    history3 = await MessageHistory().load("quality-3", memory)
    result3 = await agent2.run(
        "What color is the sky? Answer briefly.",
        history3,
        "quality-3",
    )
    log.debug("agent_output", output=result3.output)

    # ── Example 3: Multiple QualityChecks with different thresholds ──
    log.debug("example", example=3, title="Two QualityCheck evaluators")

    lenient = QualityCheck(threshold=3.0, judge_model=os.getenv("QUALITY_CHECK_MODEL", "ollama:gpt-oss:20b"))
    moderate = QualityCheck(threshold=7.0, judge_model=os.getenv("QUALITY_CHECK_MODEL", "ollama:gpt-oss:20b"))

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(lenient, moderate)
    )

    log.debug("evaluator_config", evaluator="lenient", threshold=lenient.threshold)
    log.debug("evaluator_config", evaluator="moderate", threshold=moderate.threshold)

    history4 = await MessageHistory().load("quality-4", memory)
    result4 = await agent3.run(
        "Tell me a short joke.",
        history4,
        "quality-4",
    )
    log.debug("agent_output", output=result4.output)

    # ── How it works ────────────────────────────────────────────
    log.debug("section", title="How QualityCheck works")
    log.debug("explanation", step=1, description="After each agent.run(), the evaluator fires")
    log.debug("explanation", step=2, description="It sends a second LLM call asking to rate the response 0-10")
    log.debug("explanation", step=3, description="If score < threshold → log warning")
    log.debug("explanation", step=4, description="If score >= threshold → log info")
    log.debug("explanation", step=5, description="Never modifies the agent output — read-only observer")


if __name__ == "__main__":
    asyncio.run(main())
