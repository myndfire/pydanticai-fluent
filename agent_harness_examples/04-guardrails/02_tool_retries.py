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

"""Tool-level retries for individual tool call failures.

Demonstrates:
  - Retry attempts for individual tool executions
  - Exponential backoff between tool retries
  - Combined with agent-level retry for full coverage

Tool retries correspond to PydanticAI's @agent.tool(retries=N) parameter.

Usage:
    uv run python 02_tool_retries.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/02_tool_retries.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig, ToolRetryConfig
from agent_harness.tools import ToolRegistry


load_dotenv()
log = structlog.get_logger()

GUARDRAILS_MODEL_PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
GUARDRAILS_MODEL_NAME = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")

TOOL_RETRIES_AGENT_MAX_RETRIES = int(os.getenv("TOOL_RETRIES_AGENT_MAX_RETRIES", "2"))
TOOL_RETRIES_AGENT_TIMEOUT = int(os.getenv("TOOL_RETRIES_AGENT_TIMEOUT", "30"))
TOOL_RETRIES_MAX_RETRIES = int(os.getenv("TOOL_RETRIES_MAX_RETRIES", "3"))
TOOL_RETRIES_BACKOFF = float(os.getenv("TOOL_RETRIES_BACKOFF", "1.5"))


def echo(text: str) -> str:
    """Echo the given text back."""
    log.debug("tool", tool="echo", received=text)
    return f"Echo: {text}"


def reverse(text: str) -> str:
    """Reverse the given text."""
    log.debug("tool", tool="reverse", received=text)
    return text[::-1]


async def main():
    log.debug("separator")
    log.debug("section", title="Tool-Level Retries")
    log.debug("separator")

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("tool-retry-demo", memory)

    tools = ToolRegistry().add_many(echo, reverse)

    agent_retry = AgentRetryConfig().with_max_retries(TOOL_RETRIES_AGENT_MAX_RETRIES).with_timeout(TOOL_RETRIES_AGENT_TIMEOUT)
    tool_retry = (
        ToolRetryConfig()
        .with_max_retries(TOOL_RETRIES_MAX_RETRIES)      # retry tool calls up to N times
        .with_backoff(TOOL_RETRIES_BACKOFF)              # exponential backoff between tool retries
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=GUARDRAILS_MODEL_PROVIDER, model_name=GUARDRAILS_MODEL_NAME))
        .with_tools(tools)
        .with_agent_retries(agent_retry)
        .with_tool_retries(tool_retry)
    )

    # ── Run ────────────────────────────────────────────────────
    log.debug("section", title="Configuration")
    log.debug("tool_retry", max_retries=tool_retry.max_retries)
    log.debug("tool_retry", backoff_multiplier=tool_retry.backoff_multiplier)
    log.debug("agent_retry", max_retries=agent_retry.max_retries)
    log.debug("section", title="Sending prompt: Use echo then reverse on result")

    result = await agent.run(
        "Use the echo tool to repeat 'hello world', "
        "then use the reverse tool on the result.",
        history,
        "tool-retry-demo",
    )

    log.debug("separator")
    log.debug("result", success=result.success)
    log.debug("result", output=result.output)


if __name__ == "__main__":
    asyncio.run(main())
