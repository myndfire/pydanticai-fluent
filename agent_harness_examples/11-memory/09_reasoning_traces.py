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

"""Inspect the model's hidden reasoning — what happens before the answer.

When a model runs with thinking=True, it produces TWO kinds of output
inside each response message:

  ┌─────────────────────────────────────────────────────────┐
  │  MODEL RESPONSE (one "turn")                           │
  │                                                         │
  │  🧠 ThinkingPart   ← hidden reasoning (chain-of-thought)│
  │     "Let me think about this... Seattle's climate..."   │
  │                                                         │
  │  📝 TextPart       ← visible answer                    │
  │     "Yes, you should bring an umbrella."               │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

The thinking parts are the model's internal monologue. agent_harness
strips them from stored turns and cleaned output — but the raw messages
on result.new_messages still contain everything. This example shows both.

Usage:
    uv run python 09_think_parts.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python memory/09_reasoning_traces.py
"""

import asyncio
import os
import structlog
from dotenv import load_dotenv

from pydantic_ai.messages import ThinkingPart, TextPart, ModelResponse

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig

load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "phi4-mini-reasoning")
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "512"))


async def main():
    """Run the reasoning traces example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull phi4-mini-reasoning
        3. Install deps: cd agent_harness_examples && uv sync
    """
    log.debug("separator", width=60)
    log.debug("section", title="Model Hidden Reasoning — Thinking Parts vs Visible Response")
    log.debug("separator", width=60)
    log.debug("note", message="thinking=True requires a model that supports it.")
    log.debug("note", message="If the model does not support thinking, the call will hang.")
    log.debug("note", message="A 60-second timeout is set to prevent indefinite waits.")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_agent_retries(AgentRetryConfig().with_timeout(int(os.getenv("MEMORY_REASONING_TIMEOUT", "60"))))
    )

    memory = InMemoryProvider()
    history = await MessageHistory().load("think-demo", memory)

    # thinking=True tells the model to produce internal reasoning
    try:
        result = await agent.run(
            "Should I bring an umbrella to Seattle in November? "
            "Explain your reasoning.",
            history,
            "think-demo",
            model_settings={"thinking": True},
        )
    except Exception as e:
        log.debug("agent_run_failed", error=str(e))
        log.debug("thinking_not_supported", model=MODEL_NAME)
        log.debug("try_models", models="phi4-mini-reasoning, llama3.2, gpt-oss:20b")
        return

    # ── Separate thinking parts from visible text ───────────────
    thinking_blocks: list[str] = []
    response_blocks: list[str] = []

    messages = result.new_messages
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart):
                    thinking_blocks.append(part.content)
                elif isinstance(part, TextPart):
                    response_blocks.append(part.content)

    # ── Display: visible answer ─────────────────────────────────
    log.debug("separator", width=60)
    log.debug("section", title="FINAL OUTPUT (what a user would see in a chat app)")
    log.debug("separator", width=60)
    log.debug("output", output=result.output)

    # ── Display: hidden reasoning ───────────────────────────────
    log.debug("separator", width=60)
    log.debug("section", title="HIDDEN REASONING (what the model thought before answering)")
    log.debug("separator", width=60)

    if thinking_blocks:
        for i, block in enumerate(thinking_blocks, 1):
            log.debug("reasoning_block", index=i, chars=len(block), content=block)
    else:
        log.debug("no_thinking_parts", message="The model you're using may not support thinking=True.")
        log.debug("try_model", suggestion="qwen2.5, deepseek-r1")

    # ── Summary ─────────────────────────────────────────────────
    thinking_chars = sum(len(b) for b in thinking_blocks)
    response_chars = sum(len(b) for b in response_blocks)

    log.debug("separator", width=60)
    log.debug("section", title="SUMMARY")
    log.debug("separator", width=60)
    log.debug("summary", reasoning_blocks=len(thinking_blocks), reasoning_chars=thinking_chars, visible_response_chars=response_chars)
    if response_chars > 0:
        ratio = f"{thinking_chars}:{response_chars}"
        log.debug("think_response_ratio", ratio=ratio)

    # ── How agent_harness handles these ─────────────────────────
    log.debug("separator", width=60)
    log.debug("section", title="HOW AGENT_HARNESS STRIPS THINKING PARTS")
    log.debug("separator", width=60)
    log.debug("hidden_in", result_output="extract_clean_output() removes them", turn_data_messages="filter_thinking_parts() removes them")
    log.debug("preserved_in", result_new_messages="this is what we inspected above")


if __name__ == "__main__":
    asyncio.run(main())
