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

Two approaches to creating evaluators:
  1. Subclass CustomEvaluator (see 03_custom_evaluator.py) — structured logging, automatic [name] prefix
  2. Implement the Evaluator protocol directly (this file)

This file demonstrates the **protocol approach**.

When to use the Protocol approach:
  - You need full control over output (write to files, external APIs, custom formats)
  - You want to use a specific logging library (structlog, loguru, etc.) directly
  - You're integrating with existing monitoring systems (Prometheus, StatsD, etc.)
  - Zero framework dependency is important

When to use CustomEvaluator instead (see 03_custom_evaluator.py):
  - You want structured logging with self.log_info(), log_warning(), log_error()
  - Automatic [name] prefix on all log messages for easy filtering
  - Consistent logging format across multiple evaluators
  - Domain-specific evaluators that benefit from logging helpers

The Evaluator protocol only requires:
    async def evaluate(self, prompt: str, result: Any, context: dict) -> None

No base class, no imports from evaluators module — just implement this method.

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
import os
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()
log = structlog.get_logger()


# ── Protocol evaluator: JSON audit log ──────────────────────────────

class AuditLogEvaluator:
    """Writes every turn to a JSON audit log file — no base class needed.

    Demonstrates:
      - Writing directly to files (no framework logging helpers)
      - Accessing context keys: session_id, prompt_id, model
      - Building structured JSON entries manually
      - Using standard library (json, pathlib) for output
    """

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
    """Tracks and prints response latency per turn.

    Demonstrates:
      - Using structlog directly (no framework logging)
      - Accessing result.output for response content
      - Minimal evaluator implementation
    """

    def __init__(self):
        self.start_time = None

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        session = context.get("session_id", "unknown")
        output = result.output if hasattr(result, "output") else str(result)
        output_len = len(output) if output else 0
        log.debug("latency_tracker", session=session, response_length=output_len, unit="chars")


# ── Protocol evaluator: PII scanner ─────────────────────────────────

class PiiScanner:
    """Scans the response for potential PII patterns and logs findings.

    Demonstrates:
      - Using standard library (re) for pattern matching
      - Custom structlog output with structured data
      - No framework dependency — pure Python regex
    """

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
            log.warning("pii_detected", session=session, findings=findings)
        else:
            log.debug("pii_scan_clean", note="No PII detected")


# ── Main ────────────────────────────────────────────────────────────

async def main():
    log.debug("separator", separator="=" * 60)
    log.debug("section", title="Protocol Evaluator — Direct Protocol Implementation")
    log.debug("separator", separator="=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Audit log evaluator ──────────────────────────
    log.debug("example", example=1, title="AuditLogEvaluator (writes JSONL)")

    audit = AuditLogEvaluator(log_path="audit_log.jsonl")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(audit)
    )

    log.debug("evaluator_config", evaluator="AuditLogEvaluator", log_path=str(audit.log_path))

    history = await MessageHistory().load("proto-1", memory)
    result = await agent.run(
        "What is the speed of light in m/s?",
        history,
        "proto-1",
    )
    log.debug("agent_output", output=result.output)

    history2 = await MessageHistory().load("proto-2", memory)
    result2 = await agent.run(
        "What is Planck's constant?",
        history2,
        "proto-2",
    )
    log.debug("agent_output", output=result2.output)

    # Inspect the audit log
    if audit.log_path.exists():
        lines = audit.log_path.read_text().strip().split("\n")
        log.debug("audit_log_info", entries_written=len(lines))
        for line in lines:
            entry = json.loads(line)
            log.debug("audit_entry", session_id=entry['session_id'], prompt=entry['prompt'][:50], response=entry['response'][:50])

    # ── Example 2: All three protocol evaluators ────────────────
    log.debug("example", example=2, title="All three protocol evaluators")

    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "gpt-oss:20b")))
        .with_evaluators(
            AuditLogEvaluator(log_path="full_audit.jsonl"),
            LatencyTracker(),
            PiiScanner(),
        )
    )

    log.debug("evaluator_list", evaluators="AuditLog + LatencyTracker + PiiScanner")

    history3 = await MessageHistory().load("proto-3", memory)
    result3 = await agent2.run(
        "Create a sample user profile with name John Doe, "
        "email john@example.com, and phone 555-123-4567.",
        history3,
        "proto-3",
    )
    log.debug("agent_output", output=result3.output)

    # ── The Protocol ────────────────────────────────────────────
    log.debug("section", title="The Evaluator Protocol")
    log.debug("protocol_intro", description="Any class with this method is an evaluator:")
    log.debug("protocol_method", class_name="MyEvaluator", method="async def evaluate(self, prompt: str, result: Any, context: dict) -> None")
    log.debug("protocol_comment", comment="inspect prompt, result, context")
    log.debug("protocol_comment", comment="log, trace, store, alert — anything")
    log.debug("protocol_comment", comment="never modify result — read-only")
    log.debug("context_keys", session_id="str", prompt_id="str", model="str")


if __name__ == "__main__":
    asyncio.run(main())
