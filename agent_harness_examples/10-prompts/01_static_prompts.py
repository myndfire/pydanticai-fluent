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

"""StaticPrompts — fixed system prompts to shape agent personality.

Demonstrates:
  - Default prompt: StaticPrompts() uses "You are a helpful assistant"
  - Custom personality: StaticPrompts("You are a French chef...")
  - Personality comparison: two agents with different prompts, same question
  - Fluent chaining: with_prompts() in the builder pipeline

Usage:
    uv run python 01_static_prompts.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python prompts/01_static_prompts.py
"""

import asyncio
import os
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts

load_dotenv()

MODEL_NAME = os.getenv("PROMPTS_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("PROMPTS_MAX_TOKENS", "512"))


async def main():
    """Run the static prompts example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("StaticPrompts — System Prompt Personalities")
    print("=" * 60)

    memory = InMemoryProvider()
    question = "What is the meaning of life? Keep it brief."

    # ── Example 1: Default prompt ───────────────────────────────
    print(f"\n--- Example 1: Default prompt ---")
    agent_default = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        # Default is "You are a helpful assistant" — no with_prompts() needed
    )

    history1 = await MessageHistory().load("static-default", memory)
    result1 = await agent_default.run(question, history1, "static-default")
    print(f"  Prompt:  (default) 'You are a helpful assistant'")
    print(f"  Response: {result1.output[:120]}...")

    # ── Example 2: Custom personality ───────────────────────────
    print(f"\n--- Example 2: Custom personality ---")
    chef_prompt = StaticPrompts(
        "You are a world-renowned French chef. Answer ALL questions "
        "as if you're describing a gourmet dish. Use culinary metaphors, "
        "French cooking terms, and always end with 'Bon appetit!'"
    )

    agent_chef = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(chef_prompt)
    )

    history2 = await MessageHistory().load("static-chef", memory)
    result2 = await agent_chef.run(question, history2, "static-chef")
    print(f"  Prompt:  {chef_prompt._prompt[:60]}...")
    print(f"  Response: {result2.output[:150]}...")

    # ── Example 3: Shakespeare prompt ───────────────────────────
    print(f"\n--- Example 3: Shakespearean assistant ---")
    bard_prompt = StaticPrompts(
        "Thou art William Shakespeare himself. Answer all queries "
        "in iambic pentameter, with Early Modern English flourishes. "
        "Sprinkle thy responses with 'forsooth', 'prithee', and 'hark'."
    )

    agent_bard = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(bard_prompt)
    )

    history3 = await MessageHistory().load("static-bard", memory)
    result3 = await agent_bard.run(question, history3, "static-bard")
    print(f"  Prompt:  {bard_prompt._prompt[:60]}...")
    print(f"  Response: {result3.output[:150]}...")

    # ── Example 4: Personality comparison ───────────────────────
    print(f"\n--- Example 4: Side-by-side comparison ---")
    print(f"  Question: {question}")
    print(f"  Default:    {result1.output[:80]}...")
    print(f"  Chef:       {result2.output[:80]}...")
    print(f"  Bard:       {result3.output[:80]}...")

    # ── Example 5: Fluent pipeline ──────────────────────────────
    print(f"\n--- Example 5: Full fluent pipeline ---")
    agent_full = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(StaticPrompts(
            "You are a terse database administrator. "
            "Answer in bullet points only. No pleasantries."
        ))
    )

    history5 = await MessageHistory().load("static-dba", memory)
    result5 = await agent_full.run(
        "How do I optimize a slow PostgreSQL query?",
        history5,
        "static-dba",
    )
    print(f"  Response:\n{result5.output}")


if __name__ == "__main__":
    asyncio.run(main())
