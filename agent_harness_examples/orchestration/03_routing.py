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

"""Orchestration Pattern 3: Classify and Route.

A router agent classifies the user's request via a tool, then the
program routes to the appropriate specialist agent. Classification
result is stored in SharedContext for later inspection.

Flow:
  User query
    │
    ▼
  Router Agent (tool: classify_request)
    │
    │ returns category: "billing" / "tech-support" / "general"
    │
    ▼
  Program reads classification from SharedContext
    │
    ├──"billing"────▶ Billing Specialist
    ├──"tech-support"▶ Tech Support Specialist
    └──"general"────▶ General Specialist
                          │
                          ▼
                    Specialist output

Usage:
    uv run python orchestration/03_routing.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python orchestration/03_routing.py
"""

import asyncio
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


# ── Shared context ───────────────────────────────────────────────────

@dataclass
class SharedContext:
    """Tracks classification and routing decisions."""
    classification: str = ""
    specialist_used: str = ""
    routing_log: list[dict] = field(default_factory=list)

    def record(self, query: str, classification: str, specialist: str) -> None:
        self.routing_log.append({
            "query": query[:80],
            "classification": classification,
            "specialist": specialist,
        })


# ── Classify tool ────────────────────────────────────────────────────

def classify_request(ctx: RunContext[SharedContext], text: str) -> str:
    """Classify a user request into a category.

    Args:
        text: The user's request text to classify.
    """
    text_lower = text.lower()

    billing_words = ["bill", "invoice", "payment", "charge", "refund", "subscription", "price"]
    tech_words = ["error", "bug", "crash", "login", "password", "install", "slow", "broken"]

    if any(w in text_lower for w in billing_words):
        category = "billing"
    elif any(w in text_lower for w in tech_words):
        category = "tech-support"
    else:
        category = "general"

    ctx.deps.classification = category
    print(f"  [classify] \"{text[:60]}...\" → {category}")
    return category


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Pattern 3: Classify and Route")
    print("=" * 60)

    model = ModelConfig(provider="ollama", model_name="gpt-oss:20b")
    memory = InMemoryProvider()

    # ── Router agent ──────────────────────────────────────────────
    tools = ToolRegistry().add(classify_request)
    router = (
        ManagedAgent(deps_type=SharedContext)
        .with_model(model)
        .with_tools(tools)
    )

    # ── Specialist agents ─────────────────────────────────────────
    billing = (
        ManagedAgent()
        .with_model(model)
    )

    tech = (
        ManagedAgent()
        .with_model(model)
    )

    general = (
        ManagedAgent()
        .with_model(model)
    )

    # ── Run with different queries ────────────────────────────────

    queries = [
        "I got charged twice for my subscription. Can I get a refund?",
        "I can't log into my account. It says 'invalid password' every time.",
        "What are your office hours? I'd like to visit the team.",
    ]

    for i, query in enumerate(queries):
        ctx = SharedContext()
        print(f"\n{'─'*60}")
        print(f"Turn {i+1}: {query}")

        # Step 1: Classify
        print(f"\n  [Step 1] Router classifies...")
        h1 = await MessageHistory().load(f"route-r{i}", memory)
        await router.run(
            f"Classify this request: {query}",
            h1,
            f"route-r{i}",
            deps=ctx,
        )

        # Step 2: Route to specialist
        specialist_map = {
            "billing": (billing, "Billing Specialist"),
            "tech-support": (tech, "Tech Support Specialist"),
            "general": (general, "General Specialist"),
        }
        agent, label = specialist_map.get(ctx.classification, specialist_map["general"])
        ctx.specialist_used = label
        print(f"\n  [Step 2] Routing to: {label}")

        h2 = await MessageHistory().load(f"route-s{i}", memory)
        r2 = await agent.run(
            f"You are a {label}. Handle this request: {query}",
            h2,
            f"route-s{i}",
        )
        print(f"  Specialist output: {r2.output}")

        # Record
        ctx.record(query, ctx.classification, label)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Routing Log (from SharedContext)")
    print("=" * 60)
    for entry in ctx.routing_log:
        print(f"  [{entry['classification']}] → {entry['specialist']}")


if __name__ == "__main__":
    asyncio.run(main())
