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

import os
import asyncio
import uuid
from dataclasses import dataclass, field

import structlog
from dotenv import load_dotenv
from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


load_dotenv()
log = structlog.get_logger()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


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
    """Run the parallel fan-out demo.

    Setup:
        - Ollama must be running (`ollama serve`).
        - Model must be pulled (default: `ollama pull gpt-oss:20b`).
        - `MODEL_NAME` may override the model in `.env`.
        - `LLM_PROVIDER` may override the provider in `.env`.
        - `OLLAMA_BASE_URL` may configure the Ollama endpoint
          (default: `http://localhost:11434/v1`).
    """
    log.debug("separator", char="=", count=60)
    log.debug("pattern", pattern=4, title="Parallel Fan-Out / Fan-In")
    log.debug("separator", char="=", count=60)

    model = ModelConfig(provider=LLM_PROVIDER, model_name=MODEL_NAME)

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
        log.debug("fan_out", question=question)

        async def ask_legal():
            log.debug("fan_out", specialist="Legal Analyst", status="running")
            h = await MessageHistory().load(f"sub-l-{uuid.uuid4().hex[:6]}", memory)
            r = await legal.run(
                f"You are a legal analyst. Give one paragraph: {question}",
                h,
                f"sub-l-{uuid.uuid4().hex[:6]}",
            )
            return ("Legal", str(r.output or ""))

        async def ask_tech():
            log.debug("fan_out", specialist="Tech Analyst", status="running")
            h = await MessageHistory().load(f"sub-t-{uuid.uuid4().hex[:6]}", memory)
            r = await tech.run(
                f"You are a technology analyst. Give one paragraph: {question}",
                h,
                f"sub-t-{uuid.uuid4().hex[:6]}",
            )
            return ("Technical", str(r.output or ""))

        async def ask_business():
            log.debug("fan_out", specialist="Business Analyst", status="running")
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
            log.debug("fan_in", specialist=label, output=output[:100])
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
    log.debug("turn", turn=1, title="Remote Work Policy")
    h1 = await MessageHistory().load("fan-r1", memory)
    r1 = await coordinator.run(
        "We're considering a permanent remote work policy. "
        "Gather perspectives from all specialists on the pros and cons.",
        h1,
        "fan-r1",
        deps=ctx1,
    )
    log.debug("coordinator_answer", result=str(r1.output))

    # ── Turn 2 ───────────────────────────────────────────────────
    ctx2 = SharedContext()
    log.debug("turn", turn=2, title="AI in Customer Support")
    h2 = await MessageHistory().load("fan-r2", memory)
    r2 = await coordinator.run(
        "Should we replace our customer support team with AI chatbots? "
        "Get all specialist opinions.",
        h2,
        "fan-r2",
        deps=ctx2,
    )
    log.debug("coordinator_answer", result=str(r2.output))

    # ── Summary ─────────────────────────────────────────────────
    log.debug("separator", char="=", count=60)
    log.debug("section", title="SharedContext summary")
    for p in ctx2.perspectives:
        log.debug("perspective", specialist=p['specialist'], question=p['question'][:60])


if __name__ == "__main__":
    asyncio.run(main())
