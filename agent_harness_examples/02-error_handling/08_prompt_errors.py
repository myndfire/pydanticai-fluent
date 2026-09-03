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

"""Prompt errors — handling failures when prompt providers fail.

Demonstrates:
  - Custom PromptProvider that raises on get_system_prompt()
  - on_prompt_error callback to handle template/render failures
  - Source="prompt" from agent.run() _error_source tagging
  - Jinja2 render failure simulation
  - Fallback to a default prompt when the template provider fails

Usage:
    uv run python 08_prompt_errors.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start MongoDB:
        docker compose -f docker-compose.yml up -d mongo
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/08_prompt_errors.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import PromptProvider
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Mock PromptProvider that raises ──────────────────────────────────

class FailingPromptProvider:
    """A PromptProvider that raises — simulates MongoDB down for MongoPrompts."""

    def __init__(self, name: str = "broken-prompts"):
        self.name = name

    async def get_system_prompt(self, **context) -> str:
        prompt_id = context.get("prompt_id", "default")
        log.debug("prompt_provider_fail", provider=self.name, prompt_id=prompt_id, message="get_system_prompt FAILING!")
        raise RuntimeError(
            f"{self.name}: Failed to connect to prompt database "
            f"for prompt_id='{prompt_id}'"
        )


# ── Prompt error handler ────────────────────────────────────────────

def on_prompt_failure_re_raise(ctx: ErrorContext) -> str | None:
    """Log the failure and re-raise — prompts are critical."""
    log.debug(
        "prompt_critical_failure",
        error_type=ctx.error_type,
        error_message=ctx.error_message,
        session_id=ctx.session_id,
        prompt=ctx.prompt,
    )
    return None  # re-raise — prompts are critical, can't continue


def on_prompt_fallback(ctx: ErrorContext) -> str | None:
    """Suppress the prompt error by returning a fallback."""
    log.debug(
        "prompt_error_suppressed",
        error_type=ctx.error_type,
        error_message=ctx.error_message[:80],
    )
    # Note: this suppresses the error but the system prompt wasn't set.
    # The agent will run with whatever default it had.
    return None  # suppress — agent runs without system prompt


async def main():
    log.debug("separator")
    log.debug("section", title="Prompt Errors — Handling Template/Render Failures")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Example 1: Re-raise on prompt failure ───────────────────
    log.debug("example", example=1, title="Re-raise on prompt failure")

    broken_prompts = FailingPromptProvider("mongo-prompts-db")

    config1 = ErrorHandlingConfig().on_prompt_error(on_prompt_failure_re_raise)

    agent1 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(broken_prompts)
        .with_error_handling(config1)
    )

    log.debug("status", handler="on_prompt_error → re-raise (return None)", prompt_provider="FailingPromptProvider")

    try:
        history1 = await MessageHistory().load("prompt-err-1", memory)
        await agent1.run(
            "Say hello.",
            history1,
            "prompt-err-1",
            prompt_id="critical_template",
        )
    except RuntimeError as e:
        log.debug("exception_caught", error=str(e))

    # ── Example 2: Suppress with fallback ───────────────────────
    log.debug("example", example=2, title="Suppress prompt errors (run without prompt)")

    config2 = ErrorHandlingConfig().on_prompt_error(on_prompt_fallback)

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(FailingPromptProvider("mongo-secondary"))
        .with_error_handling(config2)
    )

    log.debug("status", handler="on_prompt_error → suppress (return None)")

    history2 = await MessageHistory().load("prompt-err-2", memory)
    result2 = await agent2.run(
        "What is 2+2?",
        history2,
        "prompt-err-2",
        prompt_id="healthcare_expert",
        specialty="cardiology",
    )
    log.debug("result", success=result2.success, output=result2.output, note="Agent ran without system prompt due to prompt provider failure")

    # ── Example 3: Jinja2 render failure simulation ─────────────
    log.debug("example", example=3, title="Jinja2 render failure (conceptual)")
    log.debug("prompt_failure_mode", mode="MongoDB unreachable", error_type="ConnectionError")
    log.debug("prompt_failure_mode", mode="Invalid Jinja2 template", error_type="ValueError from jinja2")
    log.debug("status", message="Both route to on_prompt_error with source='prompt'.")
    log.debug("status", message="The callback receives:")
    log.debug("status", detail="error_type: 'ConnectionError' or 'ValueError'")
    log.debug("status", detail="error_message: the specific failure detail")
    log.debug("status", detail="session_id, prompt, stack_trace")
    log.debug("status", message="Example template that would fail:")
    log.debug("status", template='You are a {{role}} with {{undefined_variable}}', error="jinja2.exceptions.UndefinedError")


if __name__ == "__main__":
    asyncio.run(main())
