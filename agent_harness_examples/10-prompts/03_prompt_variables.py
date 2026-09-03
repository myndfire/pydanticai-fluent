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
import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts

load_dotenv()
log = structlog.get_logger()

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
    log.debug("separator", title="Prompt Variables — prompt_id & kwargs Flow")

    memory = InMemoryProvider()

    # ── Example 1: Default prompt_id behavior ───────────────────
    log.debug("example", example=1, title="Default prompt_id ('default')")
    log.debug("api_info", description="No prompt_id in kwargs → uses 'default'")

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_prompts(StaticPrompts("You are a helpful assistant. Be concise."))
    )

    history1 = await MessageHistory().load("var-default", memory)
    # No prompt_id kwarg — defaults to "default"
    result1 = await agent1.run(
        "What is the capital of Australia? One word only.",
        history1,
        "var-default",
    )
    log.debug("response", response=result1.output)

    # ── Example 2: prompt_id switching ──────────────────────────
    log.debug("example", example=2, title="prompt_id switching mid-session")
    log.debug("api_info", description="Same agent, different prompt_id each turn")

    # StaticPrompts ignores the template variables,
    # but demonstrates how prompt_id flows through
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
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
    log.debug("turn", turn=1, prompt_id="formal", response=result2a.output[:80], truncation="...")

    # Turn 2: Use "casual" prompt_id
    history2b = await MessageHistory().load("var-switch", memory)
    result2b = await agent2.run(
        "Explain quantum computing in one sentence.",
        history2b,
        "var-switch",
        save_to=[memory],
        prompt_id="casual",
    )
    log.debug("turn", turn=2, prompt_id="casual", response=result2b.output[:80], truncation="...")

    # Turn 3: Back to default
    history2c = await MessageHistory().load("var-switch", memory)
    result2c = await agent2.run(
        "What did I just ask you about?",
        history2c,
        "var-switch",
        save_to=[memory],
    )
    log.debug("turn", turn=3, prompt_id="default", response=result2c.output[:80], truncation="...")

    # ── Example 3: Variables flow into prompts ──────────────────
    log.debug("example", example=3, title="Variables flow (conceptual with StaticPrompts)")
    log.debug("api_info", description="With MongoPrompts, kwargs become Jinja2 template variables")
    log.debug("api_example", pattern="agent.run('What is the weather?', history, session_id, prompt_id='weather_expert', city='Tokyo', units='metric', format='concise_report')")

    # ── Example 4: Structured data as variables ─────────────────
    log.debug("example", example=4, title="Structured data (lists, dicts) as template vars")
    log.debug("template_example", template="You support these languages: {% for lang in languages %}- {{lang}}{% endfor %}", kwargs="languages=['Python', 'Rust', 'TypeScript']")
    log.debug("template_example", template="Your config: endpoint={{config.host}}:{{config.port}}", kwargs="config={'host': 'api.example.com', 'port': 443}")

    # ── Example 5: Variables ignored by StaticPrompts ───────────
    log.debug("example", example=5, title="Variables ignored by StaticPrompts")
    log.debug("api_info", description="StaticPrompts ignores template variables — they don't cause errors")

    agent5 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
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
    log.debug("response", response=result5.output, note="Extra kwargs passed but ignored — no errors")

    # ── Example 6: Reserved kwargs ──────────────────────────────
    log.debug("example", example=6, title="Reserved kwargs")
    log.debug("api_info", description="These kwargs are consumed by run() and NOT passed as template vars: prompt_id (selects prompt), _prefix keys (filtered out)")
    log.debug("api_info", description="All other kwargs → template rendering variables")

    agent6 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
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
    log.debug("kwargs_filtered", prompt_id="custom-id", role="engineer", _internal_meta="hidden")
    log.debug("response", response=result6.output)


if __name__ == "__main__":
    asyncio.run(main())
