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

"""Combining tools with guards, evaluators, and structured output.

Demonstrates:
  - Tools + agent-level retries
  - Tools + tool-level retries
  - Tools + content filtering on tool responses
  - Tools + evaluators that inspect tool usage
  - Tools + structured output (tools populate a Pydantic model)

Usage:
    uv run python 04_tool_combinations.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python tools/04_tool_combinations.py
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry
from agent_harness.guards import (
    AgentRetryConfig,
    ToolRetryConfig,
    ContentFilterConfig,
    TokenLimitsConfig,
)
from agent_harness.evaluators import Evaluator

load_dotenv()

MODEL_NAME = os.getenv("TOOL_CALLING_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("TOOL_CALLING_MAX_TOKENS", "512"))


# ── Tools ────────────────────────────────────────────────────────────

def search_docs(query: str) -> str:
    """Search the documentation for a given topic (simulated).

    Args:
        query: The search query string.
    """
    print(f"  [tool:search_docs] query: {query}")
    docs = {
        "flask": "Flask is a micro web framework for Python.",
        "fastapi": "FastAPI is a modern, fast web framework for Python 3.7+.",
        "django": "Django is a high-level Python web framework.",
        "pydantic": "Pydantic is a data validation library using Python type annotations.",
        "definitely": "Most definitely!",
        "absolutely": "Absolutely correct!",
    }
    return docs.get(query.lower(), f"No docs found for: {query}")


def calculate_rating(reviews: str) -> str:
    """Calculate an average rating from comma-separated review scores.

    Args:
        reviews: Comma-separated numeric ratings, e.g. '4,5,3,5'.
    """
    print(f"  [tool:calculate_rating] reviews: {reviews}")
    try:
        scores = [int(r.strip()) for r in reviews.split(",")]
        avg = sum(scores) / len(scores)
        return f"Average rating: {avg:.1f}/5 from {len(scores)} reviews"
    except Exception as e:
        return f"Error: {e}"


# ── Dependency container for context-aware tool ──────────────────────

@dataclass
class SearchDeps:
    """Dependencies for search operations."""
    search_engine: str = "default"
    max_results: int = 5
    queries_made: int = 0


def search_with_context(
    ctx: RunContext[SearchDeps],
    query: str,
    limit: int = 3,
) -> str:
    """Search with context — tracks query count and respects limits.

    Args:
        query: The search term.
        limit: Maximum results to return.
    """
    deps = ctx.deps
    deps.queries_made += 1
    print(
        f"  [tool:search_with_context] "
        f"engine={deps.search_engine}, "
        f"query={query}, "
        f"queries_so_far={deps.queries_made}"
    )
    docs = {
        "python": "Python is a high-level programming language.",
        "rust": "Rust is a systems programming language focused on safety.",
        "typescript": "TypeScript is a typed superset of JavaScript.",
    }
    return docs.get(query.lower(), f"No results for: {query}") + \
        f" (query #{deps.queries_made})"


# ── Content filter for tool responses ────────────────────────────────

def clean_tool_output(text: str) -> str:
    """Remove any 'definitely' or 'absolutely' filler from tool responses."""
    text = re.sub(r"\bdefinitely\b", "certainly", text, flags=re.IGNORECASE)
    text = re.sub(r"\babsolutely\b", "indeed", text, flags=re.IGNORECASE)
    return text


# ── Structured output model ──────────────────────────────────────────

class SearchResult(BaseModel):
    """Structured search result."""
    topic: str = Field(description="The search topic")
    result: str = Field(description="The search result or summary")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    sources_consulted: int = Field(description="Number of sources checked", default=1)


# ── Evaluator ────────────────────────────────────────────────────────

class ToolUsageEvaluator(Evaluator):
    """Evaluator that inspects tool usage after each turn."""

    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        output_text = getattr(result, "output", str(result))
        print(f"  [evaluator] Prompt: {prompt[:60]}...")
        print(f"  [evaluator] Output length: {len(output_text or '')} chars")
        print(f"  [evaluator] Session: {context.get('session_id', 'unknown')}")


# ── Main ────────────────────────────────────────────────────────────

async def main():
    """Run the tool combinations example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Tools + Guards + Evaluators + Structured Output")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Build the agent with everything ─────────────────────────
    deps = SearchDeps(search_engine="vector-db", max_results=5)

    tools = (
        ToolRegistry()
        .add(search_docs)
        .add(calculate_rating)
        .add(search_with_context)
    )

    agent = (
        ManagedAgent(deps_type=SearchDeps)
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(tools)
        # Agent retries with timeout
        .with_agent_retries(
            AgentRetryConfig()
            .with_max_retries(2)
            .with_timeout(30)
        )
        # Tool-level retries
        .with_tool_retries(
            ToolRetryConfig()
            .with_max_retries(2)
            .with_backoff(1.5)
        )
        # Content filter on output
        .with_content_filter(
            ContentFilterConfig()
            .on_filter(clean_tool_output)
        )
        # Token limits
        .with_token_limits(
            TokenLimitsConfig()
            .with_max_total_tokens(5000)
        )
        # Evaluator
        .with_evaluators(ToolUsageEvaluator())
    )

    # ── Print configuration ─────────────────────────────────────
    print(f"\nTools registered: {len(tools.get_tools())}")
    for t in tools.get_tools():
        has_ctx = "RunContext" in str(next(iter(t.__annotations__.values()), ""))
        print(f"  - {t.__name__} {'(context-aware)' if has_ctx else '(plain)'}")

    # ── Run 1: Plain tool via search_docs ───────────────────────
    print("\n--- Run 1: Search documentation ---")
    history1 = await MessageHistory().load("combo-1", memory)
    result1 = await agent.run(
        "What is Flask? Search the docs.",
        history1,
        "combo-1",
        deps=deps,
    )
    print(f"  Output: {result1.output}")

    # ── Run 2: Context-aware tool ───────────────────────────────
    print("\n--- Run 2: Context-aware search ---")
    history2 = await MessageHistory().load("combo-2", memory)
    result2 = await agent.run(
        "Search for information about Rust.",
        history2,
        "combo-2",
        deps=deps,
    )
    print(f"  Output: {result2.output}")

    # ── Run 3: Calculate rating tool ────────────────────────────
    print("\n--- Run 3: Calculate ratings ---")
    history3 = await MessageHistory().load("combo-3", memory)
    result3 = await agent.run(
        "Calculate the average of these review scores: 4, 5, 3, 5, 4, 2",
        history3,
        "combo-3",
        deps=deps,
    )
    print(f"  Output: {result3.output}")

    # ── Run 4: Tool that triggers content filter ────────────────
    print("\n--- Run 4: Content filter demonstration ---")
    history4 = await MessageHistory().load("combo-4", memory)
    result4 = await agent.run(
        "Is Flask definitely a good framework? Search the docs for 'definitely'.",
        history4,
        "combo-4",
        deps=deps,
    )
    print(f"  Output: {result4.output}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\nTotal context-aware queries: {deps.queries_made}")


if __name__ == "__main__":
    asyncio.run(main())
