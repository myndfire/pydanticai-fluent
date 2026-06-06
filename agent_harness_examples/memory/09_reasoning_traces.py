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
"""

import asyncio

from pydantic_ai.messages import ThinkingPart, TextPart, ModelResponse

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import AgentRetryConfig


async def main():
    print("=" * 60)
    print("Model Hidden Reasoning — Thinking Parts vs Visible Response")
    print("=" * 60)
    print()
    print("Note: thinking=True requires a model that supports it.")
    print("If the model does not support thinking, the call will hang.")
    print("A 60-second timeout is set to prevent indefinite waits.")
    print()

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="phi4-mini-reasoning"))
        .with_agent_retries(AgentRetryConfig().with_timeout(60))
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
        print(f"  ❌ Agent run failed: {e}")
        print()
        print("  This model may not support thinking=True.")
        print("  Try one of these Ollama models that support reasoning:")
        print("    phi4-mini-reasoning")
        print("    llama3.2")
        print("    gpt-oss:20b")
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
    print()
    print("─" * 60)
    print("📝 FINAL OUTPUT")
    print("   (what a user would see in a chat app)")
    print("─" * 60)
    print(result.output)

    # ── Display: hidden reasoning ───────────────────────────────
    print()
    print("─" * 60)
    print("🧠 HIDDEN REASONING")
    print("   (what the model thought before answering)")
    print("─" * 60)

    if thinking_blocks:
        for i, block in enumerate(thinking_blocks, 1):
            print(f"\n  [Reasoning block {i} — {len(block)} chars]")
            print(f"  {block}")
    else:
        print()
        print("  No thinking parts found.")
        print("  The model you're using may not support thinking=True.")
        print("  Try a model that supports reasoning, e.g. qwen2.5, deepseek-r1.")

    # ── Summary ─────────────────────────────────────────────────
    thinking_chars = sum(len(b) for b in thinking_blocks)
    response_chars = sum(len(b) for b in response_blocks)

    print()
    print("─" * 60)
    print("📊 SUMMARY")
    print("─" * 60)
    print(f"  Reasoning blocks:  {len(thinking_blocks)}")
    print(f"  Reasoning chars:   {thinking_chars}")
    print(f"  Visible response:  {response_chars} chars")
    if response_chars > 0:
        ratio = f"{thinking_chars}:{response_chars}"
        print(f"  Think:Response:    {ratio}")

    # ── How agent_harness handles these ─────────────────────────
    print()
    print("─" * 60)
    print("🔧 HOW AGENT_HARNESS STRIPS THINKING PARTS")
    print("─" * 60)
    print()
    print("  Thinking parts are hidden in:")
    print("    result.output       → extract_clean_output() removes them")
    print("    TurnData.messages   → filter_thinking_parts() removes them")
    print()
    print("  Raw messages are preserved in:")
    print("    result.new_messages → this is what we inspected above")


if __name__ == "__main__":
    asyncio.run(main())
