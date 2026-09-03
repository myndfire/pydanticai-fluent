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

"""Human-in-the-loop — review and optionally edit the agent's final response.

The agent generates a response, then you review it before it's returned.
You can approve it as-is or type your own version.

Scenario
────────
You ask the agent to write a company mission statement. The agent:
  1. Generates a response using the full agent_harness pipeline
  2. The human_review callback fires — you see the response
  3. You [A]pprove it as-is, or [M]odify it with your own text
  4. The (possibly modified) response is returned

This is the simplest human-in-the-loop pattern — review at the end.
For step-by-step review between tool calls, see 03_multistep_tools.py.

Pipeline:
  ┌──────────────┐     ┌─────┐     ┌──────────┐
  │ agent.run()  │ ──→ │ LLM │ ──→ │  human   │
  │ (prompt)     │     │     │     │ reviews  │
  └──────────────┘     └─────┘     └──────────┘
                                        │
                              ┌─────────┴─────────┐
                              │ [A]pprove         │
                              │   → return as-is  │
                              │ [M]odify          │
                              │   → return edited │
                              └───────────────────┘

Key points
──────────
  - ContentFilterConfig.on_filter intercepts the response post-generation
  - The full agent pipeline (memory, prompts, retries, observability) runs
    before the human sees the output
  - Returning a modified string from the callback replaces the output entirely
  - No new code needed in agent_harness — uses existing guardrail infrastructure

Usage
─────
    uv run python 01_review_approval.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python human_in_the_loop/01_review_approval.py
"""

import asyncio

import structlog
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import ContentFilterConfig

log = structlog.get_logger()


def human_review(text: str) -> str:
    """Show output to human. Approve or modify."""

    print(f"\nAgent says: {text}\n")

    choice = input("Approve or modify? [A]pprove  [M]odify: ").strip().upper()

    if choice == "M":
        return input("Your version: ").strip()

    return text


async def main():
    """Run the human-review demo.

    Setup:
        - Ollama must be running (`ollama serve`).
        - Model `gpt-oss:20b` must be pulled (`ollama pull gpt-oss:20b`).
        - `OLLAMA_BASE_URL` may be set to configure the Ollama endpoint
          (default: `http://localhost:11434/v1`).
    """
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_content_filter(ContentFilterConfig().on_filter(human_review))
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("hitl-demo", memory)
    log.debug("generating_response")
    result = await agent.run(
        "Write a one-sentence company mission statement "
        "for a sustainable fashion startup called Green Threads.",
        history,
        "hitl-demo",
    )

    print(f"\nFinal output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
