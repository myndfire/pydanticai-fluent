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

"""SafetyCheck — OpenAI moderation API for content safety evaluation.

Demonstrates:
  - SafetyCheck() evaluates both the user prompt AND the agent response
  - Checks for: hate, harassment, self-harm, sexual, violence content
  - Logs warnings for flagged categories
  - Gracefully handles missing OpenAI library
  - Runs after every turn — read-only, never modifies output

Requires: uv add openai
Usage:    python 02_safety_check.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python evaluators/02_safety_check.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import SafetyCheck

load_dotenv()


async def main():
    print("=" * 60)
    print("SafetyCheck — OpenAI Moderation API")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── SafetyCheck setup ───────────────────────────────────────
    print("\n--- SafetyCheck setup ---")

    safety = SafetyCheck()

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(safety)
    )

    print("  Evaluator: SafetyCheck()")
    print("  Checks both prompt and response via OpenAI moderation API")
    print("  Categories checked: hate, harassment, self-harm, sexual, violence")
    print()

    # ── Example 1: Safe content ─────────────────────────────────
    print("--- Example 1: Safe content ---")
    history = await MessageHistory().load("safety-1", memory)
    result = await agent.run(
        "What is the capital of France?",
        history,
        "safety-1",
    )
    print(f"  Output: {result.output}")
    print(f"  (If OpenAI is available, moderation API checked both prompt and response)")

    # ── Example 2: Potentially sensitive topic ──────────────────
    print("\n--- Example 2: Potentially sensitive topic ---")
    history2 = await MessageHistory().load("safety-2", memory)
    result2 = await agent.run(
        "Explain what happens to the human body during extreme physical trauma. "
        "Keep it clinical and brief.",
        history2,
        "safety-2",
    )
    print(f"  Output: {result2.output}")
    print(f"  (Moderation API checks for violence/self-harm categories)")

    # ── Example 3: Multiple turns with safety check ─────────────
    print("\n--- Example 3: Multi-turn conversation ---")

    agent3 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(SafetyCheck())
    )

    prompts = [
        "What are the symptoms of the common cold?",
        "Is it dangerous if left untreated?",
        "What should someone do if they can't afford a doctor?",
    ]

    for i, prompt in enumerate(prompts, 1):
        history3 = await MessageHistory().load("safety-3", memory)
        result3 = await agent3.run(prompt, history3, "safety-3")
        print(f"  Turn {i}: {result3.output[:80]}...")

    # ── How it works ────────────────────────────────────────────
    print("\n--- How SafetyCheck works ---")
    print("  1. After each agent.run(), the evaluator fires")
    print("  2. Calls openai.moderations.create(input=[prompt, response])")
    print("  3. For each flagged category → log warning with category names")
    print("  4. If categories are clean → log debug")
    print("  5. If openai not installed → log warning and skip")
    print("  6. Never modifies output — read-only observer")

    print("\n--- Checking OpenAI availability ---")
    try:
        import openai
        print("  openai package is installed")
    except ImportError:
        print("  openai package is NOT installed — safety checks will be skipped")
        print("  Install: uv add openai")
        print("  Set key: export OPENAI_API_KEY=sk-...")


if __name__ == "__main__":
    asyncio.run(main())
