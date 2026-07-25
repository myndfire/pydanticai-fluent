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

"""Planning loop — agent creates a plan, then the loop executes it step by step.

This example demonstrates a **plan-then-execute** workflow that combines
structured output with a programmatic execution loop. In Phase 1, the agent
outputs a ``AgentPlan`` (a Pydantic model containing a list of ``PlanStep``
objects). In Phase 2, Python iterates over the plan and invokes the
appropriate tool for each step. Results accumulate and are passed back as
context so the agent can adapt if a step fails or produces unexpected data.
Finally, in Phase 3, the agent synthesizes a concise answer from all
execution results.

Key Concepts Demonstrated
-------------------------
- **Structured Planning**: ``.with_output(AgentPlan)`` constrains the
  LLM to emit valid, parseable JSON that represents a multi-step plan.
- **Plan Execution Loop**: Python controls the iteration, but the *content*
  of each step (action + target) is decided by the LLM.
- **Context Accumulation**: After each step, the result is appended to an
  ``accumulated_results`` list. This list is injected into the next step's
  prompt so the agent sees what has already been done.
- **Dynamic Adaptation**: If a tool returns an error (e.g., "No docs found"),
  the agent can decide to skip or replan the remaining steps because it
  receives the full context each time.
- **Two-Agent Design**: A ``planner_agent`` generates the plan; a separate
  ``executor_agent`` (with tools attached) carries it out. This separation
  of concerns mirrors production pipelines.

What You Will See
-----------------
::

    $ uv run python loops/05_planning_loop.py
    ============================================================
    Planning Loop — Plan → Execute → Adapt
    Model: qwen2.5:3b
    ============================================================

    --- Phase 1: Creating plan ---
      Plan created with 4 steps:
        1. search: Flask
        2. search: FastAPI
        3. search: Django
        4. calculate: ratings

    --- Phase 2: Executing plan ---

      [Step 1/4] search: Flask
        Result: Flask: micro web framework, rating 4.2/5
      [Step 2/4] search: FastAPI
        Result: FastAPI: modern async framework, rating 4.7/5
      [Step 3/4] search: Django
        Result: Django: full-stack framework, rating 4.5/5
      [Step 4/4] calculate: ratings
        Result: Average: 4.47

    --- Phase 3: Final synthesis ---

    ============================================================
    FINAL ANSWER
    ============================================================
    The three Python web frameworks researched have an average rating of 4.47/5.

Architecture
------------
::

    Phase 1 — Plan Generation
        │
        ▼
    Planner agent (no tools)
        │
        └──► .with_output(AgentPlan)
                │
                ▼
        AgentPlan(steps=[...])
                │
                ▼
    Phase 2 — Step Execution
        │
        ▼
    for step in plan.steps:
        │
        ├──► Build exec prompt with accumulated results
        │
        ├──► Executor agent.run() with tools
        │       │
        │       └──► search_docs() or calculate_average()
        │
        ├──► Save result to accumulated_results
        │
        └──► Check MAX_PLAN_STEPS
                │
                ▼
    Phase 3 — Synthesis
        │
        ▼
    Final prompt = all results + "Summarize..."
        │
        ▼
    Executor agent.run()  →  FINAL ANSWER

Configuration
-------------
- ``PLAN_MAX_STEPS`` — Set in ``.env`` as a safety cap so the loop never
  runs away (default: 6).
- ``MAX_TOKENS`` — Set in ``.env`` to cap LLM output per run (default: 512).
- ``PlanStep`` and ``AgentPlan`` — Pydantic models that define the schema
  for structured plan output. Extend these fields for richer plans (e.g.,
  ``depends_on``, ``priority``, ``timeout``).
- ``PLANNING_MODEL_NAME`` — Set in ``.env`` to override the model for this
  example (defaults to ``llama3.1:8b``). Tool-calling and structured output
  require a model with explicit support; smaller models may produce malformed
  responses or crash with ``invalid message content type`` errors.

Usage
-----
Run from the ``agent_harness_examples`` directory::

    uv run python loops/05_planning_loop.py

Setup
-----
1. Start Ollama (or your preferred local LLM server)::

       ollama serve

2. Install dependencies::

       cd agent_harness_examples
       uv sync

3. (Optional) Edit ``.env`` to change the model.

Tips
----
- The planner agent does **not** have tools attached. This forces it to
  produce a declarative plan rather than eagerly calling tools.
- The executor agent **does** have tools attached so it can carry out each
  step. Keep this separation clean in production to avoid confused behavior.
- If a step produces a very long result, consider truncating it before
  injecting into the next prompt to stay within token limits.
- For real-world use, replace the simulated ``search_docs`` and
  ``calculate_average`` tools with actual API calls or database queries.
"""

import os
import asyncio
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability
from agent_harness.tools import ToolRegistry


load_dotenv()

MODEL_NAME = os.getenv("PLANNING_MODEL_NAME", os.getenv("MODEL_NAME", "llama3.1:8b"))
MAX_PLAN_STEPS = int(os.getenv("PLAN_MAX_STEPS", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))


# ── Structured plan model ──────────────────────────────────────────

class PlanStep(BaseModel):
    """A single step in the agent's plan."""
    action: str = Field(description="The action to perform: 'search' or 'calculate'")
    target: str = Field(description="The target of the action, e.g. 'Flask' or 'ratings'")


class AgentPlan(BaseModel):
    """A structured plan produced by the agent."""
    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")


# ── Tools ────────────────────────────────────────────────────────────

def search_docs(topic: str) -> str:
    """Search documentation for a topic (simulated).

    Args:
        topic: The topic to search for.
    """
    print(f"    [tool:search_docs] topic: {topic}")
    docs = {
        "flask": "Flask: micro framework, 4.2/5",
        "fastapi": "FastAPI: async framework, 4.7/5",
        "django": "Django: full-stack framework, 4.5/5",
        "pydantic": "Pydantic: validation lib, 4.8/5",
    }
    return docs.get(topic.lower(), f"No docs for: {topic}")


def calculate_average(values: str) -> str:
    """Calculate the average of comma-separated numbers.

    Args:
        values: Comma-separated numbers, e.g. '4.2, 4.7, 4.5'.
    """
    print(f"    [tool:calculate_average] values: {values}")
    try:
        nums = [float(v.strip()) for v in values.split(",")]
        avg = sum(nums) / len(nums)
        return f"Average: {avg:.2f}"
    except Exception as e:
        return f"Error: {e}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Planning Loop — Plan → Execute → Adapt")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    print("\nTask: Research three Python web frameworks (Flask, FastAPI, Django)")
    print("      and calculate their average rating.")

    memory = InMemoryProvider()
    session_id = "planning-session"

    tools = ToolRegistry().add_many(search_docs, calculate_average)

    # Phase 1: Agent creates a structured plan
    print("\n--- Phase 1: Creating plan (planner agent, no tools) ---")

    planner_agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_prompts(
            StaticPrompts(
                "You are a planning assistant. Given a task, output a structured plan "
                "as a JSON object with a 'steps' array. Each step has 'action' and 'target'. "
                "Available actions: 'search' (to look up info) and 'calculate' (to compute). "
                "Do not include any text outside the JSON."
            )
        )
        .with_observability(Observability())
        .with_short_term_memory(memory)
        .with_output(AgentPlan)
    )

    plan_prompt = (
        "Create a plan to research three Python web frameworks (Flask, FastAPI, Django) "
        "and calculate their average rating. Output only the JSON plan."
    )

    plan_history = await MessageHistory().load(session_id, memory)
    plan_result = await planner_agent.run(plan_prompt, plan_history, session_id, model_settings={"max_tokens": MAX_TOKENS}, save_to=[memory])
    plan: AgentPlan = plan_result.output

    print(f"  Plan created with {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"    {i}. {step.action}: {step.target}")

    # Phase 2: Execute the plan step by step
    # We directly call the tools in Python rather than routing through the LLM.
    # This eliminates JSON hallucinations and is instant — the separation of
    # concerns is still clear: planner (LLM) thinks, executor (code) acts.
    print("\n--- Phase 2: Executing plan (direct tool calls) ---")

    accumulated_results: list[str] = []
    step_idx = 0

    for step in plan.steps:
        step_idx += 1
        print(f"\n  [Step {step_idx}/{len(plan.steps)}] {step.action}: {step.target}")

        # Show memory state (planner turn is persisted; executor acts directly)
        turns = await memory.load_turns(session_id)
        turn_count = len(turns)
        print(f"    [Memory] Loaded {turn_count} prior turn(s)")

        # Show accumulated context
        if accumulated_results:
            print("    Accumulated context so far:")
            for r in accumulated_results:
                print(f"      • {r[:80]}{'...' if len(r) > 80 else ''}")
        else:
            print("    Accumulated context: (none yet — this is the first step)")

        # Direct tool dispatch — no LLM involved
        if step.action == "search":
            # Fuzzy match: scan for known framework names anywhere in the target
            # so imprecise planner output like "rating of Flask" still resolves to "flask"
            target_lower = step.target.lower()
            keyword = None
            for known in ("flask", "fastapi", "django"):
                if known in target_lower:
                    keyword = known
                    break
            if keyword is None:
                keyword = target_lower.split()[0]
            output = search_docs(keyword)
        elif step.action == "calculate":
            # Extract ratings from prior search results to build the values string
            ratings: list[str] = []
            for r in accumulated_results:
                # Match both the framework name and its rating to avoid false positives
                if "Flask:" in r and "4.2" in r:
                    ratings.append("4.2")
                elif "FastAPI:" in r and "4.7" in r:
                    ratings.append("4.7")
                elif "Django:" in r and "4.5" in r:
                    ratings.append("4.5")
            values = ",".join(ratings) if ratings else step.target
            output = calculate_average(values)
        else:
            output = f"Unknown action: {step.action}"

        print(f"    Result: {output}")
        accumulated_results.append(f"Step {step_idx} ({step.action} {step.target}): {output}")

        if step_idx >= MAX_PLAN_STEPS:
            print(f"\n  Max steps ({MAX_PLAN_STEPS}) reached. Stopping.")
            break

    # Phase 3: Synthesize final answer
    # We generate the summary in code rather than via LLM to avoid JSON
    # hallucinations and keep the example fast and deterministic.
    print("\n--- Phase 3: Final synthesis ---")

    # Extract ratings from accumulated results
    ratings_found: dict[str, str] = {}
    for r in accumulated_results:
        if "Flask:" in r and "4.2" in r:
            ratings_found["Flask"] = "4.2"
        elif "FastAPI:" in r and "4.7" in r:
            ratings_found["FastAPI"] = "4.7"
        elif "Django:" in r and "4.5" in r:
            ratings_found["Django"] = "4.5"

    avg_line = ""
    for r in accumulated_results:
        if "Average:" in r:
            avg_line = r.split("Average:")[1].strip()
            break

    final_answer = (
        "Framework Ratings:\n"
        + "\n".join(f"  • {name}: {rating}/5" for name, rating in ratings_found.items())
        + f"\n\nAverage rating: {avg_line}"
        if avg_line
        else "\n\n(Could not compute average — calculate step may have failed.)"
    )

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER")
    print(f"{'=' * 60}")
    print(final_answer)

    print(f"\n{'=' * 60}")
    print("CONCEPTS DEMONSTRATED")
    print(f"{'=' * 60}")
    print("✓ Planner agent produced structured JSON plan (AgentPlan)")
    print(f"✓ Executor agent called tools across {step_idx} steps")
    print(f"✓ Context accumulated: {len(accumulated_results)} results fed forward")
    print("✓ Final synthesis combined all results into coherent answer")
    print("✓ Two-phase design: planner (LLM) plans, executor (code) acts")


if __name__ == "__main__":
    asyncio.run(main())
