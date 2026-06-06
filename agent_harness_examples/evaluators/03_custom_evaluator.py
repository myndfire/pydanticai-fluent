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

"""CustomEvaluator — subclass with built-in logging helpers.

Demonstrates:
  - CustomEvaluator base class with log_info(), log_warning(), log_error()
  - Subclassing to create domain-specific evaluators
  - Automatic [name] prefix on all log messages
  - Multiple custom evaluators running together

Usage:
    uv run python 03_custom_evaluator.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import CustomEvaluator


# ── Custom evaluator: response length ───────────────────────────────

class ResponseLengthEvaluator(CustomEvaluator):
    """Warns if the response is too short or too long."""

    def __init__(self, min_words: int = 3, max_words: int = 500):
        super().__init__(name="length_check")
        self.min_words = min_words
        self.max_words = max_words

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)
        word_count = len(output.split()) if output else 0

        if word_count < self.min_words:
            self.log_warning(
                "Response too short",
                word_count=word_count,
                min_required=self.min_words,
                session_id=context.get("session_id"),
            )
        elif word_count > self.max_words:
            self.log_warning(
                "Response too long",
                word_count=word_count,
                max_allowed=self.max_words,
                session_id=context.get("session_id"),
            )
        else:
            self.log_info(
                "Response length OK",
                word_count=word_count,
                session_id=context.get("session_id"),
            )


# ── Custom evaluator: keyword detection ─────────────────────────────

class KeywordEvaluator(CustomEvaluator):
    """Checks if the response contains required keywords."""

    def __init__(self, keywords: list[str]):
        super().__init__(name="keyword_check")
        self.keywords = keywords

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)
        output_lower = output.lower() if output else ""

        found = [kw for kw in self.keywords if kw.lower() in output_lower]
        missing = [kw for kw in self.keywords if kw.lower() not in output_lower]

        if found:
            self.log_info(
                "Keywords found",
                found=found,
                session_id=context.get("session_id"),
            )
        if missing:
            self.log_warning(
                "Keywords missing",
                missing=missing,
                session_id=context.get("session_id"),
            )


# ── Custom evaluator: turn counter ──────────────────────────────────

class TurnCounterEvaluator(CustomEvaluator):
    """Counts how many turns have been evaluated."""

    def __init__(self):
        super().__init__(name="turn_counter")
        self.count = 0

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        self.count += 1
        self.log_info(
            f"Turn #{self.count} completed",
            session_id=context.get("session_id"),
            status="success" if getattr(result, "success", True) else "error",
        )


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("CustomEvaluator — Subclass with Logging Helpers")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Response length evaluator ────────────────────
    print("\n--- Example 1: ResponseLengthEvaluator ---")

    length_eval = ResponseLengthEvaluator(min_words=5, max_words=500)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(length_eval)
    )

    print(f"  Evaluator: ResponseLengthEvaluator(min_words=5, max_words=500)")
    print()

    # Short answer (may trigger too-short warning)
    history = await MessageHistory().load("custom-1", memory)
    result = await agent.run(
        "What is 2+2? Answer in one word only.",
        history,
        "custom-1",
    )
    print(f"  Output: {result.output}")

    # Longer answer
    history2 = await MessageHistory().load("custom-2", memory)
    result2 = await agent.run(
        "Explain the water cycle in detail.",
        history2,
        "custom-2",
    )
    print(f"  Output: {result2.output[:80]}...")

    # ── Example 2: Keyword evaluator ────────────────────────────
    print("\n--- Example 2: KeywordEvaluator ---")

    keyword_eval = KeywordEvaluator(
        keywords=["Paris", "France", "Eiffel"]
    )

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(keyword_eval)
    )

    print(f"  Evaluator: KeywordEvaluator(keywords=['Paris', 'France', 'Eiffel'])")
    print()

    history3 = await MessageHistory().load("custom-3", memory)
    result3 = await agent2.run(
        "What is the capital of France and name its most famous landmark.",
        history3,
        "custom-3",
    )
    print(f"  Output: {result3.output}")

    # ── Example 3: Multiple custom evaluators ───────────────────
    print("\n--- Example 3: All three custom evaluators together ---")

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(
            ResponseLengthEvaluator(min_words=3, max_words=300),
            KeywordEvaluator(keywords=["Python", "programming"]),
            TurnCounterEvaluator(),
        )
    )

    print("  Evaluators: ResponseLength + Keyword + TurnCounter")
    print()

    for i in range(1, 4):
        history4 = await MessageHistory().load("custom-4", memory)
        result4 = await agent3.run(
            f"Say 'turn {i}' and mention Python programming.",
            history4,
            "custom-4",
        )
        print(f"  Turn {i}: {result4.output[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
