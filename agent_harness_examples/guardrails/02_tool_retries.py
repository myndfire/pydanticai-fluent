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
"""

import asyncio
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig, ToolRetryConfig
from agent_harness.tools import ToolRegistry


def echo(text: str) -> str:
    """Echo the given text back."""
    print(f"  [tool:echo] received: {text}")
    return f"Echo: {text}"


def reverse(text: str) -> str:
    """Reverse the given text."""
    print(f"  [tool:reverse] received: {text}")
    return text[::-1]


async def main():
    print("=" * 60)
    print("Tool-Level Retries")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("tool-retry-demo", memory)

    tools = ToolRegistry().add_many(echo, reverse)

    agent_retry = AgentRetryConfig().with_max_retries(2).with_timeout(30)
    tool_retry = (
        ToolRetryConfig()
        .with_max_retries(3)      # retry tool calls up to 3 times
        .with_backoff(1.5)        # 1.5x exponential backoff between tool retries
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(tools)
        .with_agent_retries(agent_retry)
        .with_tool_retries(tool_retry)
    )

    # ── Run ────────────────────────────────────────────────────
    print(f"\nTool max retries: {tool_retry.max_retries}")
    print(f"Tool backoff: {tool_retry.backoff_multiplier}x")
    print(f"Agent max retries: {agent_retry.max_retries}")
    print(f"\nSending prompt: 'Use the echo tool to repeat hello world, "
          f"then use the reverse tool on the result.'...\n")

    result = await agent.run(
        "Use the echo tool to repeat 'hello world', "
        "then use the reverse tool on the result.",
        history,
        "tool-retry-demo",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
