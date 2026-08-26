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

"""Prompt variables — prompt_id switching and kwargs as template vars.

Demonstrates:
  - How run() passes kwargs as template rendering variables
  - prompt_id kwarg selects which prompt to use
  - Multi-turn conversation switching prompt IDs mid-session
  - Default behavior: no prompt_id → uses "default"
  - Combining StaticPrompts (no rendering) vs variables in kwargs
  - How to pass structured data (lists, dicts) as template variables

Usage:
    uv run python 03_prompt_variables.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start MongoDB:
        docker compose -f docker-compose.yml up -d mongo
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python prompts/03_prompt_variables.py
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
    """Run the prompt variables example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Prompt Variables — prompt_id & kwargs Flow")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Default prompt_id behavior ───────────────────
    print(f"\n--- Example 1: Default prompt_id ('default') ---")
    print("  No prompt_id in kwargs → uses 'default'")

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(StaticPrompts("You are a helpful assistant. Be concise."))
    )

    history1 = await MessageHistory().load("var-default", memory)
    # No prompt_id kwarg — defaults to "default"
    result1 = await agent1.run(
        "What is the capital of Australia? One word only.",
        history1,
        "var-default",
    )
    print(f"  Response: {result1.output}")

    # ── Example 2: prompt_id switching ──────────────────────────
    print(f"\n--- Example 2: prompt_id switching mid-session ---")
    print("  Same agent, different prompt_id each turn")

    # StaticPrompts ignores the template variables,
    # but demonstrates how prompt_id flows through
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(StaticPrompts("You are a helpful assistant."))
    )

    # Turn 1: Use "formal" prompt_id
    history2a = await MessageHistory().load("var-switch", memory)
    result2a = await agent2.run(
        "Introduce yourself.",
        history2a,
        "var-switch",
        save_to=[memory],
        prompt_id="formal",  # selects a different prompt
    )
    print(f"  Turn 1 (prompt_id='formal'): {result2a.output[:80]}...")

    # Turn 2: Use "casual" prompt_id
    history2b = await MessageHistory().load("var-switch", memory)
    result2b = await agent2.run(
        "Explain quantum computing in one sentence.",
        history2b,
        "var-switch",
        save_to=[memory],
        prompt_id="casual",
    )
    print(f"  Turn 2 (prompt_id='casual'): {result2b.output[:80]}...")

    # Turn 3: Back to default
    history2c = await MessageHistory().load("var-switch", memory)
    result2c = await agent2.run(
        "What did I just ask you about?",
        history2c,
        "var-switch",
        save_to=[memory],
    )
    print(f"  Turn 3 (prompt_id=default): {result2c.output[:80]}...")

    # ── Example 3: Variables flow into prompts ──────────────────
    print(f"\n--- Example 3: Variables flow (conceptual with StaticPrompts) ---")
    print("  With MongoPrompts, kwargs become Jinja2 template variables.")
    print("  Example API pattern:")
    print("    agent.run(")
    print("        'What is the weather?',")
    print("        history,")
    print("        session_id,")
    print("        prompt_id='weather_expert',")
    print("        city='Tokyo',           # → {{city}} in template")
    print("        units='metric',          # → {{units}} in template")
    print("        format='concise_report', # → {{format}} in template")
    print("    )")

    # ── Example 4: Structured data as variables ─────────────────
    print(f"\n--- Example 4: Structured data (lists, dicts) as template vars ---")
    print("  Jinja2 supports complex variables in templates:")
    print("    Template: 'You support these languages: {% for lang in languages %}- {{lang}}{% endfor %}'")
    print("    kwargs: languages=['Python', 'Rust', 'TypeScript']")
    print()
    print("    Template: 'Your config: endpoint={{config.host}}:{{config.port}}'")
    print("    kwargs: config={'host': 'api.example.com', 'port': 443}")

    # ── Example 5: Variables ignored by StaticPrompts ───────────
    print(f"\n--- Example 5: Variables ignored by StaticPrompts ---")
    print("  StaticPrompts ignores template variables — they don't cause errors.")

    agent5 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(StaticPrompts("You are a brief assistant."))
    )

    # Pass variables that StaticPrompts will simply ignore
    history5 = await MessageHistory().load("var-ignored", memory)
    result5 = await agent5.run(
        "Say hello.",
        history5,
        "var-ignored",
        # These are ignored by StaticPrompts — no error
        role="doctor",
        domain="cardiology",
        language="simple",
        any_key="any_value",
    )
    print(f"  Response: {result5.output}")
    print(f"  (Extra kwargs passed but ignored — no errors)")

    # ── Example 6: Reserved kwargs ──────────────────────────────
    print(f"\n--- Example 6: Reserved kwargs ---")
    print("  These kwargs are consumed by run() and NOT passed as template vars:")
    print("    - prompt_id   : selects the prompt template")
    print("    - _prefix keys: prefixed with underscore, filtered out")
    print("  All other kwargs → template rendering variables")

    agent6 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_prompts(StaticPrompts("You are a helpful assistant."))
    )

    history6 = await MessageHistory().load("var-reserved", memory)
    result6 = await agent6.run(
        "Say hello in 5 words.",
        history6,
        "var-reserved",
        prompt_id="custom-id",  # reserved — selects prompt
        role="engineer",          # template var
        _internal_meta="hidden", # underscore-prefixed → filtered out
    )
    print(f"  prompt_id='custom-id', role='engineer', _internal_meta='hidden' → filtered")
    print(f"  Response: {result6.output}")


if __name__ == "__main__":
    asyncio.run(main())
