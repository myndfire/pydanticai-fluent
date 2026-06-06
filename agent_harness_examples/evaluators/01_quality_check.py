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

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import QualityCheck


async def main():
    print("=" * 60)
    print("QualityCheck — LLM-as-Judge Evaluation")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Default threshold (7.0) ──────────────────────
    print("\n--- Example 1: QualityCheck with default threshold (7.0) ---")

    quality = QualityCheck(
        threshold=7.0,
        judge_model="ollama:gpt-oss:20b",
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(quality)
    )

    print(f"  Evaluator: QualityCheck(threshold={quality.threshold})")
    print(f"  Judge model: {quality.judge_model}")
    print()

    # Turn 1: should produce a high-quality answer
    history = await MessageHistory().load("quality-1", memory)
    result = await agent.run(
        "What is 2+2? Answer in exactly one word.",
        history,
        "quality-1",
    )
    print(f"  Output: {result.output}")

    # Turn 2: ask for a short answer (judge might rate differently)
    history2 = await MessageHistory().load("quality-2", memory)
    result2 = await agent.run(
        "Explain quantum computing in 200 words.",
        history2,
        "quality-2",
    )
    print(f"  Output: {result2.output[:80]}...")

    # ── Example 2: Stricter threshold ───────────────────────────
    print("\n--- Example 2: Stricter threshold (9.5) — almost always warns ---")

    strict = QualityCheck(
        threshold=9.5,
        judge_model="ollama:gpt-oss:20b",
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(strict)
    )

    history3 = await MessageHistory().load("quality-3", memory)
    result3 = await agent2.run(
        "What color is the sky? Answer briefly.",
        history3,
        "quality-3",
    )
    print(f"  Output: {result3.output}")

    # ── Example 3: Multiple QualityChecks with different thresholds ──
    print("\n--- Example 3: Two QualityCheck evaluators ---")

    lenient = QualityCheck(threshold=3.0, judge_model="ollama:gpt-oss:20b")
    moderate = QualityCheck(threshold=7.0, judge_model="ollama:gpt-oss:20b")

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(lenient, moderate)
    )

    print(f"  Lenient:  threshold={lenient.threshold}")
    print(f"  Moderate: threshold={moderate.threshold}")
    print()

    history4 = await MessageHistory().load("quality-4", memory)
    result4 = await agent3.run(
        "Tell me a short joke.",
        history4,
        "quality-4",
    )
    print(f"  Output: {result4.output}")

    # ── How it works ────────────────────────────────────────────
    print("\n--- How QualityCheck works ---")
    print("  1. After each agent.run(), the evaluator fires")
    print("  2. It sends a second LLM call asking to rate the response 0-10")
    print("  3. If score < threshold → log warning")
    print("  4. If score >= threshold → log info")
    print("  5. Never modifies the agent output — read-only observer")


if __name__ == "__main__":
    asyncio.run(main())
