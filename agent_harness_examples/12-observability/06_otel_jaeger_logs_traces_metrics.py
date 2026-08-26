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

"""OTEL — OpenTelemetry and Jaeger v2 distributed tracing via OTLP.

Demonstrates:
  - OTELTracer: OTLP gRPC export (→ Jaeger collector at localhost:4317)
  - InMemoryTracer: chained alongside for local span inspection
  - Span attributes, span context (trace_id, span_id) captured in observe()
  - Observability with multiple tracers chained together

Prerequisite:
    docker compose -f docker-compose.yml up -d jaeger

Jaeger UI: http://localhost:16686

Usage:
    uv run python 06_otel_jaeger_logs_traces_metrics.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. (Optional) Start Jaeger:
        docker compose -f docker-compose.yml up -d jaeger
    3. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/06_otel_jaeger_logs_traces_metrics.py
"""

import asyncio
import os
from dotenv import load_dotenv

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.logging import ConsoleLogger
from agent_harness.tracing import OTELTracer, InMemoryTracer, NoOpTracer
from agent_harness.metrics import InMemoryMetrics
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("OBSERVABILITY_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("OBSERVABILITY_MAX_TOKENS", "512"))
JAEGER_OTLP = os.getenv("JAEGER_OTLP_ENDPOINT", "localhost:4317")


async def check_port(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=2.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def main():
    """Run the OTEL + Jaeger tracing example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Start Jaeger: docker compose -f docker-compose.yml up -d jaeger
        4. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("OTEL + Jaeger — Distributed Tracing")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"Checking Jaeger OTLP at {JAEGER_OTLP} ...")
    otlp_ok = await check_port("localhost", 4317)
    print(f"  {'Reachable' if otlp_ok else 'NOT reachable'}")

    if not otlp_ok:
        print("\n  Jaeger not reachable. Start with:")
        print("    docker compose -f docker-compose.yml up -d jaeger")
        print("  Then open http://localhost:16686 to view traces.")
        return

    # ── Example 1: OTELTracer (OTLP → Jaeger) ───────────────────
    if otlp_ok:
        print("\n--- Example 1: OTELTracer (OTLP gRPC → Jaeger) ---")

        mem_tracer = InMemoryTracer()

        obs_otel = Observability(
            service_name="otel-jaeger-demo",
            loggers=[ConsoleLogger()],
            tracers=[
                OTELTracer(
                    service_name="otel-jaeger-demo",
                    otlp_endpoint=f"http://{JAEGER_OTLP}",
                    sample_rate=1.0,
                    create_spans=True,  # this demo explicitly exercises manual OTel spans
                ),
                mem_tracer,  # chain: OTEL + InMemory for local inspection
            ],
            metrics_list=[InMemoryMetrics()],
        )

        print(f"  Tracers: {[type(t).__name__ for t in obs_otel._tracers]}")

        # Create spans manually
        async with obs_otel.observe("manual_span", op="test", value=42):
            obs_otel.info("inside_span", detail="This appears in the trace")
            obs_otel.add_span_event("checkpoint_reached", step=1)
            obs_otel.set_span_attribute("custom_key", "custom_value")
            await asyncio.sleep(0.05)

        # Inspect local spans
        spans = mem_tracer.get_spans()
        print(f"  InMemory spans captured: {len(spans)}")
        for s in spans:
            print(f"    - {s['name']}: {s['attributes']}")

        # ── Run agent with OTEL tracing ─────────────────────────
        print("\n  --- Agent run with OTEL tracing ---")
        agent_otel = (
            ManagedAgent()
            .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
            .with_observability(obs_otel)
        )

        memory = InMemoryProvider()
        history = await MessageHistory().load("otel-agent", memory)
        result = await agent_otel.run(
            "What is 10 divided by 2?",
            history,
            "otel-agent",
            save_to=[memory],
        )
        print(f"  Response: {result.output}")

        all_spans = mem_tracer.get_spans()
        print(f"  Total InMemory spans: {len(all_spans)}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("View traces at: http://localhost:16686")
    print("  Service filter: otel-jaeger-demo")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
