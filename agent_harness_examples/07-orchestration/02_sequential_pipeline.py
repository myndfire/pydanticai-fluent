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

"""Orchestration Pattern 2: Sequential Pipeline Chain.

Three agents run in sequence — each agent's output feeds directly
into the next agent as its prompt. No tools are involved; the flow
is fully programmatic.

Flow:
  User question
    │
    ▼
  Researcher Agent ──(facts)──▶ Writer Agent ──(draft)──▶ Editor Agent
                                                                   │
                                                                   ▼
                                                          Final polished answer

A SharedContext dataclass holds intermediate outputs at each stage.

Usage:
    uv run python orchestration/02_sequential_pipeline.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python orchestration/02_sequential_pipeline.py
"""

import os
import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


# ── Shared context ───────────────────────────────────────────────────

@dataclass
class SharedContext:
    """Holds the output of each pipeline stage."""
    research: str = ""
    draft: str = ""
    final: str = ""


# ── Main ────────────────────────────────────────────────────────────

async def main():
    """Run the sequential pipeline demo.

    Setup:
        - Ollama must be running (`ollama serve`).
        - Model must be pulled (default: `ollama pull gpt-oss:20b`).
        - `MODEL_NAME` may override the model in `.env`.
        - `LLM_PROVIDER` may override the provider in `.env`.
        - `OLLAMA_BASE_URL` may configure the Ollama endpoint
          (default: `http://localhost:11434/v1`).
    """
    print("=" * 60)
    print("Pattern 2: Sequential Pipeline Chain")
    print("=" * 60)

    model = ModelConfig(provider=LLM_PROVIDER, model_name=MODEL_NAME)

    researcher = (
        ManagedAgent()
        .with_model(model)
    )

    writer = (
        ManagedAgent()
        .with_model(model)
    )

    editor = (
        ManagedAgent()
        .with_model(model)
    )

    memory = InMemoryProvider()
    ctx = SharedContext()

    # ── Pipeline run ─────────────────────────────────────────────

    question = "What is vector search and why is it important for RAG systems?"

    # Stage 1: Research
    print(f"\n── Stage 1: Researcher ──")
    print(f"  Input: {question}")
    h1 = await MessageHistory().load("pipe-r", memory)
    r1 = await researcher.run(
        f"Research and list 3-5 key facts about this topic. Be concise: {question}",
        h1,
        "pipe-r",
    )
    ctx.research = str(r1.output or "")
    print(f"  Research output: {ctx.research[:200]}...")

    # Stage 2: Write
    print(f"\n── Stage 2: Writer ──")
    h2 = await MessageHistory().load("pipe-w", memory)
    r2 = await writer.run(
        f"Write a 2-paragraph explanation using these research notes:\n\n{ctx.research}",
        h2,
        "pipe-w",
    )
    ctx.draft = str(r2.output or "")
    print(f"  Draft output: {ctx.draft[:200]}...")

    # Stage 3: Edit
    print(f"\n── Stage 3: Editor ──")
    h3 = await MessageHistory().load("pipe-e", memory)
    r3 = await editor.run(
        f"Edit the following draft for clarity and conciseness. "
        f"Fix any issues and polish the prose:\n\n{ctx.draft}",
        h3,
        "pipe-e",
    )
    ctx.final = str(r3.output or "")

    print(f"\n{'='*60}")
    print("FINAL OUTPUT (after Researcher → Writer → Editor)")
    print("=" * 60)
    print(ctx.final)


if __name__ == "__main__":
    asyncio.run(main())
