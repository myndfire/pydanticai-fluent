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

"""Pipeline error recovery — chain 3 agents, tolerate mid-pipeline failure.

A three-agent pipeline (Research → Analysis → Summary) runs sequentially.
The Analysis agent deliberately fails (prompt provider raises), the error
handler suppresses it, and the pipeline continues to Summary. A shared
PipelineContext tracks the status of every stage and prints a full trace.

Demonstrates:
  - Three-agent pipeline with shared status context
  - Deterministic pipeline failure (FailingPromptProvider)
  - Error suppression via ErrorHandlingConfig.on_prompt_error
  - Pipeline continues after a failed stage
  - Full pipeline trace printed at the end

Usage:
    uv run python error_handling/09_pipeline_error_recovery.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/09_pipeline_error_recovery.py
"""

import asyncio
from dataclasses import dataclass, field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext


# ── Shared pipeline context ──────────────────────────────────────────

@dataclass
class PipelineContext:
    """Tracks every stage in the pipeline. Each stage posts its result."""

    stages: list[dict] = field(default_factory=list)

    def post(self, name: str, success: bool, output: str, error: str = ""):
        self.stages.append({
            "name": name,
            "success": success,
            "output": output[:200],
            "error": error,
        })

    def display_trace(self):
        print(f"\n{'='*60}")
        print("Pipeline Trace")
        print("=" * 60)
        success_count = 0
        for i, s in enumerate(self.stages):
            mark = "✓" if s["success"] else "✗"
            print(f"\n  Stage {i+1}. {s['name']}  {mark}")
            print(f"     Output: {s['output'][:150]}")
            if s["error"]:
                print(f"     Error:  {s['error'][:120]}")
            if s["success"]:
                success_count += 1
        print(f"\n{'─'*60}")
        print(f"  {success_count}/{len(self.stages)} stages succeeded")


# ── Failing prompt provider (deterministic failure) ──────────────────

class FailingPromptProvider:
    """A PromptProvider that always raises — simulates an outage."""

    async def get_system_prompt(self, **context) -> str:
        raise RuntimeError("Agent 2 prompt service unavailable — simulated outage")


# ── Error handler for the failing stage ──────────────────────────────

def analysis_fallback(ctx: ErrorContext) -> str:
    """Suppress the prompt error and return a fallback string."""
    return f"[Recovered] Analysis stage failed: {ctx.error_message}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Pipeline Error Recovery — 3-Agent Chain")
    print("=" * 60)

    model = ModelConfig(provider="ollama", model_name="gpt-oss:20b")
    memory = InMemoryProvider()

    # ── Agent 1: Research ────────────────────────────────────────
    research = (
        ManagedAgent()
        .with_model(model)
    )

    # ── Agent 2: Analysis (will fail) ────────────────────────────
    error_config = ErrorHandlingConfig().on_prompt_error(analysis_fallback)
    analysis = (
        ManagedAgent()
        .with_model(model)
        .with_prompts(FailingPromptProvider())
        .with_error_handling(error_config)
    )

    # ── Agent 3: Summary ─────────────────────────────────────────
    summary = (
        ManagedAgent()
        .with_model(model)
    )

    # ── Pipeline context ─────────────────────────────────────────
    ctx = PipelineContext()

    # ── Stage 1: Research ────────────────────────────────────────
    topic = "What are embeddings in machine learning?"
    print(f"\n── Stage 1: Research ──")
    print(f"  Topic: {topic}")
    h1 = await MessageHistory().load("pipe-r", memory)
    r1 = await research.run(
        f"Give 3 key facts about this topic in one paragraph: {topic}",
        h1,
        "pipe-r",
    )
    ctx.post("Research", r1.success, str(r1.output or ""))

    # ── Stage 2: Analysis (fails, suppressed) ────────────────────
    print(f"\n── Stage 2: Analysis (will fail) ──")
    h2 = await MessageHistory().load("pipe-a", memory)
    r2 = await analysis.run(
        "Analyze the above.",
        h2,
        "pipe-a",
    )
    error_msg = ""
    if not r2.success and r2.error_context:
        error_msg = r2.error_context.error_message
    ctx.post("Analysis", r2.success, str(r2.output or ""), error_msg)

    # ── Stage 3: Summary ─────────────────────────────────────────
    print(f"\n── Stage 3: Summary ──")
    h3 = await MessageHistory().load("pipe-s", memory)
    r3 = await summary.run(
        f"Summarize the following into a single sentence.\n\n"
        f"Research:\n{r1.output}\n\n"
        f"Analysis:\n{r2.output}",
        h3,
        "pipe-s",
    )
    ctx.post("Summary", r3.success, str(r3.output or ""))

    # ── Display the full pipeline trace ──────────────────────────
    ctx.display_trace()


if __name__ == "__main__":
    asyncio.run(main())
