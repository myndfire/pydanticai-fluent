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

"""Log enrichment — custom enrichment providers from external libraries.

Demonstrates:
  - LogEnrichmentProvider protocol — any class with enrich() can be a provider
  - EnvEnricher — built-in provider that reads host, env, pid
  - Custom RequestIdEnricher — reads from Python contextvars (simulating
    an async request context from a web framework)
  - Layering multiple providers on a single agent

Usage:
    uv run python logging/02_custom_enricher.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python logging/02_custom_enricher.py
"""

import asyncio
import contextvars
import os
import uuid

from agent_harness import ManagedAgent, LogContext, EnvEnricher
from agent_harness.log_enrichment import LogEnrichmentProvider
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

# ── Request ID context variable (simulating web framework context) ───

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


# ── Custom enrichment providers ──────────────────────────────────────

class RequestIdEnricher:
    """Pull a request ID from async context — use with web frameworks."""

    def enrich(self) -> dict[str, str]:
        req_id = request_id_var.get()
        return {"request_id": req_id} if req_id else {}


class VersionEnricher:
    """Inject the application version into every log entry."""

    def __init__(self, version: str):
        self._version = version

    def enrich(self) -> dict[str, str]:
        return {"version": self._version}


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Log Enrichment — Custom Providers")
    print("=" * 60)

    model = ModelConfig(
        provider=os.getenv("LOGGING_MODEL_PROVIDER", "ollama"),
        model_name=os.getenv("LOGGING_MODEL_NAME", "gpt-oss:20b"),
    )
    memory = InMemoryProvider()

    # ── Agent with multiple enrichment sources ──────────────────
    agent = (
        ManagedAgent()
        .with_model(model)
        .with_log_enrichment(
            LogContext().with_(
                "pipeline", os.getenv("LOGGING_02_PIPELINE", "enrichment-demo")
            ),
            EnvEnricher(),  # host, env, pid
            VersionEnricher(
                os.getenv("LOGGING_02_APP_VERSION", "2.1.0")
            ),  # application version
            RequestIdEnricher(),  # async request context
        )
    )

    # ── Run 1: simulate a request with a request ID ─────────────
    request_id_var.set(f"req-{uuid.uuid4().hex[:8]}")
    print(f"\n── Run 1: request_id={request_id_var.get()} ──")
    h1 = await MessageHistory().load("enrich-1", memory)
    r1 = await agent.run(
        "Say hello in one word.",
        h1,
        "enrich-1",
    )
    print(f"  Output: {r1.output}")

    # ── Run 2: simulate a DIFFERENT request ─────────────────────
    request_id_var.set(f"req-{uuid.uuid4().hex[:8]}")
    print(f"\n── Run 2: request_id={request_id_var.get()} ──")
    h2 = await MessageHistory().load("enrich-2", memory)
    r2 = await agent.run(
        "Say goodbye in one word.",
        h2,
        "enrich-2",
    )
    print(f"  Output: {r2.output}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n✓ Every log entry carries:")
    print(
        f"  pipeline={os.getenv('LOGGING_02_PIPELINE', 'enrichment-demo')} (from LogContext)"
    )
    print(f"  host, env, pid (from EnvEnricher)")
    print(
        f"  version={os.getenv('LOGGING_02_APP_VERSION', '2.1.0')} (from VersionEnricher)"
    )
    print(f"  request_id=<uuid> (from RequestIdEnricher)")
    print(f"\n  Custom providers implement: def enrich(self) -> dict[str, str]")


if __name__ == "__main__":
    asyncio.run(main())
