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

"""Protocol evaluator — implement the Evaluator protocol directly.

Demonstrates:
  - Implementing Evaluator protocol without subclassing CustomEvaluator
  - The protocol only requires: async def evaluate(prompt, result, context) -> None
  - Context dict includes: session_id, prompt_id, model
  - Accessing result.output, result.success, result.usage
  - Using any logging/tracing/metrics library inside the evaluator

Usage:
    uv run python 04_protocol_evaluator.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python evaluators/04_protocol_evaluator.py
"""

import asyncio
import json
import time
from pathlib import Path

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


# ── Protocol evaluator: JSON audit log ──────────────────────────────

class AuditLogEvaluator:
    """Writes every turn to a JSON audit log file — no base class needed."""

    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = Path(log_path)

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)
        entry = {
            "timestamp": time.time(),
            "session_id": context.get("session_id"),
            "prompt_id": context.get("prompt_id"),
            "model": context.get("model"),
            "prompt": prompt,
            "response": output,
            "success": getattr(result, "success", True),
            "error": str(result.error_context) if getattr(result, "error_context", None) else None,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Protocol evaluator: latency tracker ─────────────────────────────

class LatencyTracker:
    """Tracks and prints response latency per turn."""

    def __init__(self):
        self.start_time = None

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        # The agent.run() already measures duration.
        # We can access it from the context or from the result
        session = context.get("session_id", "unknown")
        output = result.output if hasattr(result, "output") else str(result)
        output_len = len(output) if output else 0
        print(f"  [latency_tracker] session={session} | "
              f"response_length={output_len} chars")


# ── Protocol evaluator: PII scanner ─────────────────────────────────

class PiiScanner:
    """Scans the response for potential PII patterns and logs findings."""

    def __init__(self):
        import re
        self.email_re = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        self.phone_re = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.ssn_re = re.compile(r'\d{3}-\d{2}-\d{4}')

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = result.output if hasattr(result, "output") else str(result)

        emails = self.email_re.findall(output)
        phones = self.phone_re.findall(output)
        ssns = self.ssn_re.findall(output)

        findings = []
        if emails:
            findings.append(f"emails={emails}")
        if phones:
            findings.append(f"phones={phones}")
        if ssns:
            findings.append(f"SSNs={len(ssns)}")

        if findings:
            session = context.get("session_id", "?")
            print(f"  [pii_scanner] ⚠️ PII detected in session {session}: "
                  f"{', '.join(findings)}")
        else:
            print(f"  [pii_scanner] ✓ No PII detected")


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Protocol Evaluator — Direct Protocol Implementation")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Audit log evaluator ──────────────────────────
    print("\n--- Example 1: AuditLogEvaluator (writes JSONL) ---")

    audit = AuditLogEvaluator(log_path="audit_log.jsonl")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(audit)
    )

    print(f"  Evaluator: AuditLogEvaluator → {audit.log_path}")
    print()

    history = await MessageHistory().load("proto-1", memory)
    result = await agent.run(
        "What is the speed of light in m/s?",
        history,
        "proto-1",
    )
    print(f"  Output: {result.output}")

    history2 = await MessageHistory().load("proto-2", memory)
    result2 = await agent.run(
        "What is Planck's constant?",
        history2,
        "proto-2",
    )
    print(f"  Output: {result2.output}")

    # Inspect the audit log
    if audit.log_path.exists():
        lines = audit.log_path.read_text().strip().split("\n")
        print(f"\n  Audit log entries written: {len(lines)}")
        for line in lines:
            entry = json.loads(line)
            print(f"  [{entry['session_id']}] {entry['prompt'][:50]}... "
                  f"→ {entry['response'][:50]}...")

    # ── Example 2: All three protocol evaluators ────────────────
    print("\n--- Example 2: All three protocol evaluators ---")

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_evaluators(
            AuditLogEvaluator(log_path="full_audit.jsonl"),
            LatencyTracker(),
            PiiScanner(),
        )
    )

    print("  Evaluators: AuditLog + LatencyTracker + PiiScanner")
    print()

    history3 = await MessageHistory().load("proto-3", memory)
    result3 = await agent2.run(
        "Create a sample user profile with name John Doe, "
        "email john@example.com, and phone 555-123-4567.",
        history3,
        "proto-3",
    )
    print(f"  Output: {result3.output}")

    # ── The Protocol ────────────────────────────────────────────
    print("\n--- The Evaluator Protocol ---")
    print("  Any class with this method is an evaluator:")
    print()
    print("  class MyEvaluator:")
    print("      async def evaluate(self, prompt: str, result: Any, context: dict) -> None:")
    print("          # inspect prompt, result, context")
    print("          # log, trace, store, alert — anything")
    print("          # never modify result — read-only")
    print()
    print("  The context dict contains:")
    print("    - session_id: str")
    print("    - prompt_id: str")
    print("    - model: str")


if __name__ == "__main__":
    asyncio.run(main())
