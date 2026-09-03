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
    docker compose -f docker-compose.yml up -d mongo

Usage:
    uv run python 02_mongo_prompts.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start MongoDB:
        docker compose -f docker-compose.yml up -d mongo
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python prompts/02_mongo_prompts.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import MongoPrompts

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("PROMPTS_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("PROMPTS_MAX_TOKENS", "512"))
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
PROMPTS_DB = os.getenv("MONGODB_DATABASE", "agent_prompts")
PROMPTS_COLLECTION = os.getenv("MONGODB_COLLECTION", "prompts")


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
    """Run the MongoDB prompts example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start MongoDB: docker compose -f docker-compose.yml up -d mongo
        4. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", title="MongoPrompts — MongoDB Jinja2 Templates")

    # ── Connection check ────────────────────────────────────────
    log.debug("connection_check", uri=MONGO_URI)
    if not await check_mongo(MONGO_URI):
        log.debug("connection_failed", uri=MONGO_URI, hint="Start with: docker compose -f docker-compose.yml up -d mongo")
        return
    log.debug("connection_ok", uri=MONGO_URI)

    # ── Initialize MongoPrompts ─────────────────────────────────
    prompts = MongoPrompts(
        uri=MONGO_URI,
        database=PROMPTS_DB,
        collection=PROMPTS_COLLECTION,
    )

    # ── Seed templates ──────────────────────────────────────────
    log.debug("section", title="Seeding prompt templates")

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
    log.debug("prompt_created", prompt_id="doctor", category="healthcare", description="healthcare template with specialty + language vars")

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
    log.debug("prompt_created", prompt_id="coder", category="engineering", description="engineering template with language, years, rules vars")

    await prompts.create_prompt(
        prompt_id="poet",
        template=(
            "You are a {{style}} poet. Respond to all prompts in the form of a {{style}} poem. "
            "Use {{meter}} meter and the tone should be {{tone}}."
        ),
        version=1,
        metadata={"category": "creative", "tone": "artistic"},
    )
    log.debug("prompt_created", prompt_id="poet", category="creative", description="creative template with style, meter, tone vars")

    # ── list_prompts ────────────────────────────────────────────
    log.debug("section", title="Available prompts")
    all_prompts = await prompts.list_prompts(active_only=True)
    for p in all_prompts:
        log.debug("prompt_listed", prompt_id=p['prompt_id'], version=p['version'], category=p['metadata'].get('category', '?'))

    # ── Render templates directly ───────────────────────────────
    log.debug("section", title="Rendering templates (direct)")

    doctor_prompt = await prompts.get_system_prompt(
        prompt_id="doctor",
        specialty="cardiology",
        language="simple",
    )
    log.debug("prompt_rendered", prompt_id="doctor", preview=doctor_prompt[:150], truncation="...")

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
    log.debug("prompt_rendered", prompt_id="coder", preview=coder_prompt[:200], truncation="...")

    # ── update_prompt (cache invalidation) ──────────────────────
    log.debug("section", title="Updating prompt (cache invalidation)")

    # First render to populate cache
    await prompts.get_system_prompt(prompt_id="poet", style="haiku", meter="5-7-5", tone="serene")
    log.debug("cache_check", prompt_id="poet", in_cache='poet' in prompts._cache, expected=True)

    # Update the template
    await prompts.update_prompt(
        prompt_id="poet",
        template=(
            "You are a {{style}} poet. Respond to all prompts in {{style}} form. "
            "Use {{meter}} meter. The mood should be {{tone}} and {{mood}}."
        ),
    )
    log.debug("template_updated", prompt_id="poet", change="added 'mood' variable")

    # Cache should be invalidated
    log.debug("cache_check", prompt_id="poet", in_cache='poet' in prompts._cache, expected=False)

    # Render again with new variable
    poet_prompt = await prompts.get_system_prompt(
        prompt_id="poet",
        style="haiku",
        meter="5-7-5",
        tone="serene",
        mood="reflective",
    )
    log.debug("prompt_rendered", prompt_id="poet", preview=poet_prompt)

    # ── Integration with agent.run() ────────────────────────────
    log.debug("section", title="Integration with agent.run()")
    log.debug("api_info", description="run() pops 'prompt_id' from kwargs, passes remaining kwargs as template vars")

    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
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
    log.debug("agent_result", prompt_id="coder", template_vars="language=Python, years=5, rules=[...]", response=result.output[:200], truncation="...")

    # ── clear_cache ─────────────────────────────────────────────
    log.debug("section", title="Cache management")
    cache_size = len(prompts._cache)
    log.debug("cache_entries", count=cache_size, label="before clear")
    prompts.clear_cache()
    log.debug("cache_entries", count=len(prompts._cache), label="after clear")

    # ── Cleanup ─────────────────────────────────────────────────
    log.debug("section", title="Cleanup")
    answer = input("Delete demo prompts from MongoDB? (y/n) ").strip().lower()
    if answer == "y":
        for pid in ["doctor", "coder", "poet"]:
            try:
                await prompts.update_prompt(pid, active=False)
            except ValueError:
                pass
        log.debug("cleanup", action="prompts deactivated")
    else:
        log.debug("cleanup", action="skipped", data="preserved")


if __name__ == "__main__":
    asyncio.run(main())
