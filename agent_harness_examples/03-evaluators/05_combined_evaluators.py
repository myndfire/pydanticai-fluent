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

"""Combined evaluators — all evaluator types running together.

Demonstrates:
  - QualityCheck + SafetyCheck + CustomEvaluator + protocol evaluator
  - Each evaluator receives the same (prompt, result, context)
  - Evaluators run sequentially — order matters for side effects
  - Fluent chaining: .with_evaluators(eval1, eval2, eval3)
  - Each evaluator is independent — failures in one don't affect others
    (evaluator exceptions now route to on_evaluator_error)

Usage:
    uv run python 05_combined_evaluators.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python evaluators/05_combined_evaluators.py
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import QualityCheck, CustomEvaluator
from agent_harness.errorhandling import ErrorHandlingConfig

load_dotenv()


# ── Protocol evaluator: timing logger ───────────────────────────────

class TimingLogger:
    """Logs response metadata after each turn."""

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)
        chars = len(output) if output else 0
        words = len(output.split()) if output else 0
        print(f"  [timing] {words} words, {chars} chars | "
              f"model={context.get('model')} "
              f"session={context.get('session_id')}")


# ── Custom evaluator: response diversity ────────────────────────────

class DiversityEvaluator(CustomEvaluator):
    """Tracks whether responses are getting repetitive."""

    def __init__(self):
        super().__init__(name="diversity")
        self._previous_outputs: list[str] = []

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)

        # Check if this output is similar to any previous
        for i, prev in enumerate(self._previous_outputs):
            if output.strip() == prev.strip():
                self.log_warning(
                    "Duplicate response detected",
                    matches_turn=i + 1,
                    session_id=context.get("session_id"),
                )
                break

        self._previous_outputs.append(output)

        if len(self._previous_outputs) >= 3:
            unique = len(set(o.strip() for o in self._previous_outputs))
            self.log_info(
                "Response diversity",
                total_turns=len(self._previous_outputs),
                unique_responses=unique,
            )


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Combined Evaluators — All Types Together")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Build agent with 4 evaluators ───────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(
            # Built-in: LLM-as-judge quality scoring
            QualityCheck(threshold=6.0, judge_model=os.getenv("QUALITY_CHECK_MODEL", "ollama:gpt-oss:20b")),
            # Custom: response diversity tracking
            DiversityEvaluator(),
            # Protocol: timing/metadata logging
            TimingLogger(),
        )
        # Add error handling so evaluator failures don't crash the agent
        .with_error_handling(
            ErrorHandlingConfig().on_evaluator_error(
                lambda ctx: None  # suppress evaluator failures
            )
        )
    )

    print("\n  Evaluators (4 total):")
    print("    1. QualityCheck(threshold=6.0)    — LLM-as-judge")
    print("    2. DiversityEvaluator()           — duplicate detection")
    print("    3. TimingLogger()                 — word/char counts")
    print("  Error handler: on_evaluator_error → suppress")

    # ── Multi-turn conversation ─────────────────────────────────
    session = "combined-eval"

    prompts = [
        "What is machine learning? Answer in 2-3 sentences.",
        "What is machine learning? Answer in 2-3 sentences.",  # duplicate test
        "How is it different from deep learning?",
        "Give me a practical example of machine learning in healthcare.",
    ]

    print(f"\n--- Multi-turn ({len(prompts)} turns) ---")
    for i, prompt in enumerate(prompts, 1):
        history = await MessageHistory().load(session, memory)
        result = await agent.run(prompt, history, session, save_to=[memory])
        print(f"\n  Turn {i}: {result.output[:100]}...")

    # ── Example 2: Builder-style evaluator chain ────────────────
    print(f"\n--- Example 2: Builder-style chain ---")
    print("  Evaluators are just objects — build them fluently:")
    print()
    print("  agent.with_evaluators(")
    print("      QualityCheck(threshold=7.0),")
    print("      DiversityEvaluator(),")
    print("      TimingLogger(),")
    print("  )")

    # ── Evaluator execution order ───────────────────────────────
    print(f"\n--- Execution order ---")
    print("  Evaluators run sequentially in the order they are registered:")
    print("    1st → QualityCheck runs")
    print("    2nd → DiversityEvaluator runs")
    print("    3rd → TimingLogger runs")
    print()
    print("  Each receives the same (prompt, result, context).")
    print("  Evaluators are read-only observers — they cannot modify the output.")


if __name__ == "__main__":
    asyncio.run(main())
