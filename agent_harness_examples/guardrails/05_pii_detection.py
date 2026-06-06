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

"""PII detection and redaction guardrail via user-provided callback.

Demonstrates:
  - on_redact callback that redacts PII from agent output
  - Common patterns: emails, phone numbers, SSNs, credit cards
  - on_error callback for graceful failure when the redactor raises

Usage:
    uv run python 05_pii_detection.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python guardrails/05_pii_detection.py
"""

import asyncio
import re

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import PIIDetectionConfig


def redact_pii(text: str) -> str:
    """Redact common PII patterns from text.

    In production, this could use a dedicated PII detection service
    like Microsoft Presidio, AWS Comprehend, or Google DLP.
    """
    patterns = [
        # Email addresses
        (r'[\w.+-]+@[\w-]+\.[\w.-]+', '[EMAIL]'),
        # US phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567
        (r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]'),
        # SSN: 123-45-6789
        (r'\d{3}-\d{2}-\d{4}', '[SSN]'),
        # Credit card (simplified): 4111-1111-1111-1111
        (r'\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}', '[CREDIT_CARD]'),
        # IPv4 addresses
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def on_redact_error(ctx):
    """Fallback when the redaction callback raises an exception."""
    print(f"  [on_error] Redaction failed: {ctx.error_message}")
    return f"[PII redaction error]: {ctx.error_message}"


async def main():
    print("=" * 60)
    print("PII Detection Guardrail")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    memory = InMemoryProvider()
    history = await MessageHistory().load("pii-demo", memory)

    pii_config = (
        PIIDetectionConfig()
        .on_redact(redact_pii)
        .on_error(on_redact_error)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_pii_detection(pii_config)
    )

    # ── Run ────────────────────────────────────────────────────
    print(f"\nPII detection active: {pii_config._on_redact is not None}")
    print(f"\nSending prompt: 'Generate a fake user profile with name, "
          f"email, phone, SSN, and credit card number.'...\n")

    result = await agent.run(
        "Generate a fake user profile for a character named Jane Smith. "
        "Include these details: email jane.smith@example.com, "
        "phone (555) 123-4567, SSN 123-45-6789, "
        "credit card 4111-2222-3333-4444, "
        "and a note about her IP address 192.168.1.100.",
        history,
        "pii-demo",
    )

    print(f"\nSuccess: {result.success}")
    print(f"Redacted output:\n{result.output}")
    print()

    # ── Demonstrate that PII was redacted ──────────────────────
    print("-" * 40)
    print("Verification: checking for redacted patterns...")
    raw = result.output or ""
    checks = {
        "No raw emails": "jane.smith@example.com" not in raw,
        "No raw phone": "(555) 123-4567" not in raw,
        "No raw SSN": "123-45-6789" not in raw,
        "No raw CC": "4111-2222-3333-4444" not in raw,
        "No raw IP": "192.168.1.100" not in raw,
        "Has [EMAIL]": "[EMAIL]" in raw,
        "Has [PHONE]": "[PHONE]" in raw,
    }
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {label}")


if __name__ == "__main__":
    asyncio.run(main())
