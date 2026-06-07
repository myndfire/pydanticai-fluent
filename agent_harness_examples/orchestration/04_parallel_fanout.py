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

"""Orchestration Pattern 4: Parallel Fan-Out / Fan-In.

A coordinator agent has a tool that fans out a question to multiple
specialist agents concurrently, then aggregates their perspectives
into a single response.

Flow:
  Coordinator Agent (tool: gather_perspectives)
         │
         │ tool fires: runs 3 specialists concurrently
         │
         ├──▶ Legal Analyst   ──┐
         ├──▶ Tech Analyst    ──┤── asyncio.gather ──▶ aggregate ▶ return to coordinator
         ├──▶ Business Analyst ──┘
         │
         ▼
  SharedContext ←── updated with all perspectives

Usage:
    uv run python orchestration/04_parallel_fanout.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python orchestration/04_parallel_fanout.py
"""

import asyncio
import uuid
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


# ── Shared context ───────────────────────────────────────────────────

@dataclass
class SharedContext:
    """Accumulates all specialist perspectives across turns."""
    perspectives: list[dict] = field(default_factory=list)

    def record(self, label: str, question: str, output: str) -> None:
        self.perspectives.append({
            "specialist": label,
            "question": question,
            "result": output[:200],
        })


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Pattern 4: Parallel Fan-Out / Fan-In")
    print("=" * 60)

    model = ModelConfig(provider="ollama", model_name="gpt-oss:20b")

    # ── Specialist agents ─────────────────────────────────────────
    legal = (
        ManagedAgent()
        .with_model(model)
    )

    tech = (
        ManagedAgent()
        .with_model(model)
    )

    business = (
        ManagedAgent()
        .with_model(model)
    )

    # ── Fan-out tool ──────────────────────────────────────────────
    async def gather_perspectives(
        ctx: RunContext[SharedContext],
        question: str,
    ) -> str:
        """Gather perspectives from three specialists concurrently.

        Args:
            question: The question to ask all specialists.
        """
        print(f"\n  [fan-out] Asking 3 specialists: \"{question}\"")

        async def ask_legal():
            print("  [fan-out] → Legal Analyst (running)...")
            h = await MessageHistory().load(f"sub-l-{uuid.uuid4().hex[:6]}", memory)
            r = await legal.run(
                f"You are a legal analyst. Give one paragraph: {question}",
                h,
                f"sub-l-{uuid.uuid4().hex[:6]}",
            )
            return ("Legal", str(r.output or ""))

        async def ask_tech():
            print("  [fan-out] → Tech Analyst (running)...")
            h = await MessageHistory().load(f"sub-t-{uuid.uuid4().hex[:6]}", memory)
            r = await tech.run(
                f"You are a technology analyst. Give one paragraph: {question}",
                h,
                f"sub-t-{uuid.uuid4().hex[:6]}",
            )
            return ("Technical", str(r.output or ""))

        async def ask_business():
            print("  [fan-out] → Business Analyst (running)...")
            h = await MessageHistory().load(f"sub-b-{uuid.uuid4().hex[:6]}", memory)
            r = await business.run(
                f"You are a business analyst. Give one paragraph: {question}",
                h,
                f"sub-b-{uuid.uuid4().hex[:6]}",
            )
            return ("Business", str(r.output or ""))

        # Run all three concurrently
        results = await asyncio.gather(ask_legal(), ask_tech(), ask_business())

        # Aggregate
        report_parts = []
        for label, output in results:
            ctx.deps.record(label, question, output)
            print(f"  [fan-in] ← {label}: {output[:100]}...")
            report_parts.append(f"### {label} Perspective\n{output}")

        return "\n\n".join(report_parts)

    # ── Coordinator agent ─────────────────────────────────────────
    tools = ToolRegistry().add(gather_perspectives)
    coordinator = (
        ManagedAgent(deps_type=SharedContext)
        .with_model(model)
        .with_tools(tools)
    )

    memory = InMemoryProvider()

    # ── Turn 1 ───────────────────────────────────────────────────
    ctx1 = SharedContext()
    print("\n── Turn 1: Remote Work Policy ──")
    h1 = await MessageHistory().load("fan-r1", memory)
    r1 = await coordinator.run(
        "We're considering a permanent remote work policy. "
        "Gather perspectives from all specialists on the pros and cons.",
        h1,
        "fan-r1",
        deps=ctx1,
    )
    print(f"\n  Coordinator final answer:\n{r1.output}")

    # ── Turn 2 ───────────────────────────────────────────────────
    ctx2 = SharedContext()
    print("\n── Turn 2: AI in Customer Support ──")
    h2 = await MessageHistory().load("fan-r2", memory)
    r2 = await coordinator.run(
        "Should we replace our customer support team with AI chatbots? "
        "Get all specialist opinions.",
        h2,
        "fan-r2",
        deps=ctx2,
    )
    print(f"\n  Coordinator final answer:\n{r2.output}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SharedContext summary:")
    for p in ctx2.perspectives:
        print(f"  [{p['specialist']}] {p['question'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
