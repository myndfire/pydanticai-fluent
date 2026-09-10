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

"""WarnNearLimits + ReportContextUsage — observe without rewriting history.

Demonstrates:
  - with_warn_near_limits(...): inject an URGENT/CRITICAL user turn as limits
    approach (never edits history)
  - with_report_context_usage(on_usage): live context gauge via callback

All values come from the environment (no hardcoded numbers):

    COMPACTION_WARN_MAX_TOKENS / COMPACTION_WARN_THRESHOLD
    MODEL_NAME / LLM_PROVIDER (shared test model)

Usage:
    uv run python 03_warn_and_report.py
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

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "granite4.1:8b")

WARN_MAX_TOKENS = int(os.getenv("COMPACTION_WARN_MAX_TOKENS", "4000"))
WARN_THRESHOLD = float(os.getenv("COMPACTION_WARN_THRESHOLD", "0.7"))


async def print_usage(usage):
    log.debug(
        "context_usage",
        used_tokens=usage.used_tokens,
        window_tokens=usage.window_tokens,
        fraction=round(usage.fraction, 3),
        resolved=usage.resolved,
    )


async def main():
    memory = InMemoryProvider()
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=MODEL_PROVIDER, model_name=MODEL_NAME))
        .with_prompts(StaticPrompts("You are a helpful assistant. Reply briefly."))
        .with_short_term_memory(memory)
        .with_warn_near_limits(
            warning_threshold=WARN_THRESHOLD, max_context_tokens=WARN_MAX_TOKENS
        )
        .with_report_context_usage(print_usage)
    )
    log.debug(
        "compaction_config",
        strategy="warn_and_report",
        max_context_tokens=WARN_MAX_TOKENS,
        warning_threshold=WARN_THRESHOLD,
    )

    session = "compaction-warn-demo"
    history = await MessageHistory().load(session, memory)
    result = await agent.run(
        "Summarize the water cycle in two sentences.", history, session
    )
    log.debug("output", text=str(result.output)[:120])


if __name__ == "__main__":
    asyncio.run(main())
