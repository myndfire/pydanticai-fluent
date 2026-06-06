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

"""MongoPrompts — MongoDB-backed Jinja2 templates with CRUD and caching.

Demonstrates:
  - Connection check with graceful fallback
  - create_prompt(): seed templates into MongoDB
  - list_prompts(): browse available prompts
  - get_system_prompt(): fetch + render a Jinja2 template with variables
  - Jinja2 template syntax: {{role}}, {{domain}}, conditionals, loops
  - update_prompt(): modify a template (invalidates cache)
  - clear_cache(): force cache refresh
  - Integration with run(): prompt_id + **kwargs flow through to get_system_prompt()

MongoDB schema:
  {
      "_id": "prompt_id",
      "template": "You are a {{role}} specialized in {{domain}}...",
      "active": true,
      "version": 1,
      "metadata": {"tags": [...]}
  }

Prerequisite:
    docker compose -f agent_harness_examples/prompts/docker-compose.mongo.yml up -d

Usage:
    uv run python 02_mongo_prompts.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import MongoPrompts

MONGO_URI = "mongodb://localhost:27017"
PROMPTS_DB = "agent_prompts"
PROMPTS_COLLECTION = "prompts"


async def check_mongo(uri: str) -> bool:
    """Check if MongoDB is reachable."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=2000)
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def main():
    print("=" * 60)
    print("MongoPrompts — MongoDB Jinja2 Templates")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking MongoDB at {MONGO_URI} ...")
    if not await check_mongo(MONGO_URI):
        print(f"  MongoDB not reachable at {MONGO_URI}")
        print("  Start with:")
        print("    docker compose -f agent_harness_examples/prompts/docker-compose.mongo.yml up -d")
        return
    print("  MongoDB is reachable.")

    # ── Initialize MongoPrompts ─────────────────────────────────
    prompts = MongoPrompts(
        uri=MONGO_URI,
        database=PROMPTS_DB,
        collection=PROMPTS_COLLECTION,
    )

    # ── Seed templates ──────────────────────────────────────────
    print(f"\n--- Seeding prompt templates ---")

    await prompts.create_prompt(
        prompt_id="doctor",
        template=(
            "You are a {{specialty}} medical professional. "
            "Answer health questions with evidence-based advice. "
            "{% if language == 'simple' %}Use plain, non-technical language.{% endif %} "
            "Always include a disclaimer to consult a real doctor."
        ),
        version=1,
        metadata={"category": "healthcare", "tone": "professional"},
    )
    print("  Created: 'doctor' (healthcare template with specialty + language vars)")

    await prompts.create_prompt(
        prompt_id="coder",
        template=(
            "You are an expert {{language}} programmer with {{years}} years of experience. "
            "Answer coding questions with code examples. "
            "{% for rule in rules %}- {{rule}}\n{% endfor %}"
        ),
        version=1,
        metadata={"category": "engineering", "tone": "technical"},
    )
    print("  Created: 'coder' (engineering template with language, years, rules vars)")

    await prompts.create_prompt(
        prompt_id="poet",
        template=(
            "You are a {{style}} poet. Respond to all prompts in the form of a {{style}} poem. "
            "Use {{meter}} meter and the tone should be {{tone}}."
        ),
        version=1,
        metadata={"category": "creative", "tone": "artistic"},
    )
    print("  Created: 'poet' (creative template with style, meter, tone vars)")

    # ── list_prompts ────────────────────────────────────────────
    print(f"\n--- Available prompts ---")
    all_prompts = await prompts.list_prompts(active_only=True)
    for p in all_prompts:
        print(f"  - {p['prompt_id']} (v{p['version']}, {p['metadata'].get('category', '?')})")

    # ── Render templates directly ───────────────────────────────
    print(f"\n--- Rendering templates (direct) ---")

    doctor_prompt = await prompts.get_system_prompt(
        prompt_id="doctor",
        specialty="cardiology",
        language="simple",
    )
    print(f"  Doctor prompt (rendered):\n      {doctor_prompt[:150]}...")

    coder_prompt = await prompts.get_system_prompt(
        prompt_id="coder",
        language="Python",
        years=10,
        rules=[
            "Always use type hints",
            "Prefer async/await over threading",
            "Write docstrings for all functions",
        ],
    )
    print(f"  Coder prompt (rendered):\n      {coder_prompt[:200]}...")

    # ── update_prompt (cache invalidation) ──────────────────────
    print(f"\n--- Updating prompt (cache invalidation) ---")

    # First render to populate cache
    await prompts.get_system_prompt(prompt_id="poet", style="haiku", meter="5-7-5", tone="serene")
    print(f"  Cache before update: {'poet' in prompts._cache} (should be True)")

    # Update the template
    await prompts.update_prompt(
        prompt_id="poet",
        template=(
            "You are a {{style}} poet. Respond to all prompts in {{style}} form. "
            "Use {{meter}} meter. The mood should be {{tone}} and {{mood}}."
        ),
    )
    print(f"  Template updated (added 'mood' variable)")

    # Cache should be invalidated
    print(f"  Cache after update: {'poet' in prompts._cache} (should be False)")

    # Render again with new variable
    poet_prompt = await prompts.get_system_prompt(
        prompt_id="poet",
        style="haiku",
        meter="5-7-5",
        tone="serene",
        mood="reflective",
    )
    print(f"  Poet prompt (updated, rendered):\n      {poet_prompt}")

    # ── Integration with agent.run() ────────────────────────────
    print(f"\n--- Integration with agent.run() ---")
    print("  run() pops 'prompt_id' from kwargs, passes remaining kwargs as template vars")

    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(prompts)
    )

    # Use 'coder' prompt with template variables via kwargs
    history = await MessageHistory().load("mongo-prompt-coder", memory)
    result = await agent.run(
        "Write a function that calculates the Fibonacci sequence.",
        history,
        "mongo-prompt-coder",
        prompt_id="coder",    # selects the 'coder' template
        language="Python",    # Jinja2 variable
        years=5,              # Jinja2 variable
        rules=[               # Jinja2 variable (iterable for the {% for %} loop)
            "Use type hints",
            "Add error handling",
            "Include unit tests as doctests",
        ],
    )
    print(f"\n  Prompt ID used: coder")
    print(f"  Template vars: language=Python, years=5, rules=[...]")
    print(f"  Response:\n{result.output[:200]}...")

    # ── clear_cache ─────────────────────────────────────────────
    print(f"\n--- Cache management ---")
    cache_size = len(prompts._cache)
    print(f"  Cache entries before clear: {cache_size}")
    prompts.clear_cache()
    print(f"  Cache entries after clear: {len(prompts._cache)}")

    # ── Cleanup ─────────────────────────────────────────────────
    print(f"\n--- Cleanup ---")
    answer = input("Delete demo prompts from MongoDB? (y/n) ").strip().lower()
    if answer == "y":
        for pid in ["doctor", "coder", "poet"]:
            try:
                await prompts.update_prompt(pid, active=False)
            except ValueError:
                pass
        print("  Demo prompts deactivated.")
    else:
        print("  Skipped cleanup. Data preserved.")


if __name__ == "__main__":
    asyncio.run(main())
