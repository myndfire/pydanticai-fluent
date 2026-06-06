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

"""OTEL & Jaeger — OpenTelemetry and Jaeger distributed tracing.

Demonstrates:
  - OTELTracer: OTLP gRPC export (→ Jaeger collector at localhost:4317)
  - JaegerTracer: native Jaeger UDP agent (localhost:6831)
  - InMemoryTracer: chained alongside for local span inspection
  - Span attributes, span context (trace_id, span_id) captured in observe()
  - Observability with multiple tracers chained together

Prerequisite:
    docker compose -f agent_harness_examples/observability/docker-compose.jaeger.yml up -d

Jaeger UI: http://localhost:16686

Usage:
    uv run python 06_otel_jaeger.py
"""

import asyncio
import socket

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.logging import ConsoleLogger
from agent_harness.tracing import OTELTracer, JaegerTracer, InMemoryTracer, NoOpTracer
from agent_harness.metrics import InMemoryMetrics
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

JAEGER_OTLP = "localhost:4317"
JAEGER_UDP_HOST = "localhost"
JAEGER_UDP_PORT = 6831


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
    print("=" * 60)
    print("OTEL + Jaeger — Distributed Tracing")
    print("=" * 60)

    # ── Connection check ────────────────────────────────────────
    print(f"\nChecking Jaeger OTLP at {JAEGER_OTLP} ...")
    otlp_ok = await check_port("localhost", 4317)
    print(f"  {'Reachable' if otlp_ok else 'NOT reachable'}")

    print(f"Checking Jaeger UDP at {JAEGER_UDP_HOST}:{JAEGER_UDP_PORT} ...")
    udp_ok = await check_port("localhost", 6831)
    print(f"  {'Reachable' if udp_ok else 'NOT reachable'}")

    if not (otlp_ok or udp_ok):
        print("\n  Jaeger not reachable. Start with:")
        print("    docker compose -f agent_harness_examples/observability/docker-compose.jaeger.yml up -d")
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
                    otlp_endpoint="http://localhost:4317",
                    sample_rate=1.0,
                ),
                mem_tracer,  # chain: OTEL + InMemory for local inspection
            ],
            metrics_list=[InMemoryMetrics()],
        )

        print(f"  Tracers: {[type(t).__name__ for t in obs_otel._tracers]}")

        # Create spans manually
        async with obs_otel.observe("manual_span", operation="test", value=42):
            obs_otel.info("inside_span", message="This appears in the trace")
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
            .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
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

    # ── Example 2: JaegerTracer (UDP agent) ─────────────────────
    if udp_ok:
        print("\n--- Example 2: JaegerTracer (UDP agent) ---")

        mem_tracer2 = InMemoryTracer()

        obs_jaeger = Observability(
            service_name="jaeger-native-demo",
            loggers=[ConsoleLogger()],
            tracers=[
                JaegerTracer(
                    service_name="jaeger-native-demo",
                    jaeger_host=JAEGER_UDP_HOST,
                    jaeger_port=JAEGER_UDP_PORT,
                ),
                mem_tracer2,
            ],
            metrics_list=[InMemoryMetrics()],
        )

        print(f"  Tracers: {[type(t).__name__ for t in obs_jaeger._tracers]}")

        async with obs_jaeger.observe("jaeger_native_span", source="example", run=1):
            obs_jaeger.info("inside_jaeger_span", tracer="JaegerTracer")
            await asyncio.sleep(0.05)

        print(f"  InMemory spans: {len(mem_tracer2.get_spans())}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("View traces at: http://localhost:16686")
    print("  Service filter: otel-jaeger-demo or jaeger-native-demo")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
