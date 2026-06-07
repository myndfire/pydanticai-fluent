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

"""Orchestration Pattern 1: Tool-Driven Agent Delegation.

A coordinator agent delegates tasks to a specialist agent via a tool.
Shared context tracks all delegations across turns so the coordinator
can recall what the specialist has already done.

Flow:
  User → Coordinator Agent (tool: delegate_to_specialist)
              │
              │ calls tool with task
              ▼
         Specialist Agent (e.g. finance analyst)
              │
              │ returns result
              ▼
         SharedContext ←── updated with delegation log
              │
              ▼
  Coordinator returns final answer incorporating specialist output

Usage:
    uv run python orchestration/01_delegation.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python orchestration/01_delegation.py
"""

import asyncio
import uuid
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


# ── Shared context between agents ────────────────────────────────────

@dataclass
class SharedContext:
    """Mutable state shared across coordinator and specialist interactions."""
    delegation_log: list[dict] = field(default_factory=list)

    def record(self, task: str, result: str) -> None:
        self.delegation_log.append({"task": task, "result": result})


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Pattern 1: Tool-Driven Agent Delegation")
    print("=" * 60)

    # ── Specialist agent ──────────────────────────────────────────
    specialist = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    )

    # ── Delegation tool ───────────────────────────────────────────
    async def delegate_to_specialist(ctx: RunContext[SharedContext], task: str) -> str:
        """Delegate a task to the specialist agent and record the result.

        Args:
            task: The specific task for the specialist to handle.
        """
        print(f"\n  [delegate] Coordinator → Specialist: {task}")
        sub_history = MessageHistory()
        result = await specialist.run(
            f"You are a finance specialist. Answer precisely: {task}",
            sub_history,
            f"sub-{uuid.uuid4().hex[:8]}",
        )
        output = str(result.output or "")
        ctx.deps.record(task, output)
        print(f"  [delegate] Specialist → Coordinator: {output[:120]}...")
        return f"[Specialist Report]\nTask: {task}\nResult: {output}"

    # ── Coordinator agent ─────────────────────────────────────────
    tools = ToolRegistry().add(delegate_to_specialist)
    coordinator = (
        ManagedAgent(deps_type=SharedContext)
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(tools)
    )

    memory = InMemoryProvider()
    ctx = SharedContext()

    # ── Turn 1: Delegate a calculation ────────────────────────────
    print("\n── Turn 1: Q3 Revenue ──")
    h1 = await MessageHistory().load("del-r1", memory)
    r1 = await coordinator.run(
        "What was the Q3 revenue if we had $1.2M in July, $1.5M in August, "
        "and $1.1M in September? Use the specialist to calculate.",
        h1,
        "del-r1",
        deps=ctx,
    )
    print(f"  Output: {r1.output}")

    # ── Turn 2: Delegate an analysis ──────────────────────────────
    print("\n── Turn 2: Expense Trends ──")
    h2 = await MessageHistory().load("del-r2", memory)
    r2 = await coordinator.run(
        "Our expenses are: Marketing $200K, R&D $450K, Operations $300K. "
        "Have the specialist analyze which department has the highest spend.",
        h2,
        "del-r2",
        deps=ctx,
    )
    print(f"  Output: {r2.output}")

    # ── Turn 3: Ask about prior delegations ───────────────────────
    print("\n── Turn 3: Recall Past Delegations ──")
    h3 = await MessageHistory().load("del-r3", memory)
    r3 = await coordinator.run(
        "What work has the specialist done so far? Summarize based on what you know.",
        h3,
        "del-r3",
        deps=ctx,
    )
    print(f"  Output: {r3.output}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\nDelegations recorded in SharedContext: {len(ctx.delegation_log)}")
    for i, entry in enumerate(ctx.delegation_log):
        print(f"  {i+1}. {entry['task']}")


if __name__ == "__main__":
    asyncio.run(main())
