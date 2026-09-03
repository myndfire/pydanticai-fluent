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

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.evaluators import SafetyCheck

load_dotenv()
log = structlog.get_logger()


async def main():
    log.debug("separator", separator="=" * 60)
    log.debug("section", title="SafetyCheck — OpenAI Moderation API")
    log.debug("separator", separator="=" * 60)

    memory = InMemoryProvider()

    # ── SafetyCheck setup ───────────────────────────────────────
    log.debug("section", title="SafetyCheck setup")

    safety = SafetyCheck()

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(safety)
    )

    log.debug("evaluator_config", evaluator="SafetyCheck()")
    log.debug("evaluator_info", description="Checks both prompt and response via OpenAI moderation API")
    log.debug("evaluator_info", description="Categories checked: hate, harassment, self-harm, sexual, violence")

    # ── Example 1: Safe content ─────────────────────────────────
    log.debug("example", example=1, title="Safe content")
    history = await MessageHistory().load("safety-1", memory)
    result = await agent.run(
        "What is the capital of France?",
        history,
        "safety-1",
    )
    log.debug("agent_output", output=result.output)
    log.debug("note", description="If OpenAI is available, moderation API checked both prompt and response")

    # ── Example 2: Potentially sensitive topic ──────────────────
    log.debug("example", example=2, title="Potentially sensitive topic")
    history2 = await MessageHistory().load("safety-2", memory)
    result2 = await agent.run(
        "Explain what happens to the human body during extreme physical trauma. "
        "Keep it clinical and brief.",
        history2,
        "safety-2",
    )
    log.debug("agent_output", output=result2.output)
    log.debug("note", description="Moderation API checks for violence/self-harm categories")

    # ── Example 3: Multiple turns with safety check ─────────────
    log.debug("example", example=3, title="Multi-turn conversation")

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
        log.debug("turn_output", turn=i, output=result3.output[:80], truncated=True)

    # ── How it works ────────────────────────────────────────────
    log.debug("section", title="How SafetyCheck works")
    log.debug("explanation", step=1, description="After each agent.run(), the evaluator fires")
    log.debug("explanation", step=2, description="Calls openai.moderations.create(input=[prompt, response])")
    log.debug("explanation", step=3, description="For each flagged category → log warning with category names")
    log.debug("explanation", step=4, description="If categories are clean → log debug")
    log.debug("explanation", step=5, description="If openai not installed → log warning and skip")
    log.debug("explanation", step=6, description="Never modifies output — read-only observer")

    log.debug("section", title="Checking OpenAI availability")
    try:
        import openai
        log.debug("openai_status", installed=True)
    except ImportError:
        log.debug("openai_status", installed=False, note="safety checks will be skipped")
        log.debug("install_hint", command="uv add openai")
        log.debug("env_hint", env_var="OPENAI_API_KEY", value="sk-...")


if __name__ == "__main__":
    asyncio.run(main())
