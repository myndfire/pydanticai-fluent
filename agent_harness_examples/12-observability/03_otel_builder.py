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

"""Observability configuration — explicit OTLP setup and composition.

Demonstrates:
  - configure_otlp(): the application explicitly creates and owns the OTLP
    providers/exporters for logs + traces + metrics
  - ObservabilityBuilder: compose application-owned backends
  - observe() context manager for manual instrumentation
  - Graceful behavior when the OTel Collector is not running

Prerequisite:
    docker compose -f docker-compose.yml up -d otel-collector

Usage:
    uv run python 03_otel_builder.py

Setup
-----
    1. Start the OTel Collector:
        docker compose -f docker-compose.yml up -d otel-collector
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python observability/03_otel_builder.py
"""

import asyncio
import os
from dotenv import load_dotenv
import structlog

from agent_harness.logging import ConsoleLogger
from agent_harness.metrics import InMemoryMetrics
from agent_harness.observability import ObservabilityBuilder
from agent_harness.telemetry import configure_otlp
from agent_harness.tracing import InMemoryTracer

load_dotenv()

log = structlog.get_logger()

OTEL_COLLECTOR = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")


async def check_port(host: str, port: int) -> bool:
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
    log.debug("separator", char="=", count=60)
    log.debug("title", title="Observability — Explicit Setup and Composition")
    log.debug("separator", char="=", count=60)

    # ── Example 1: Compose application-owned backends (no OTel) ──
    log.debug("example", example=1, title="Compose application-owned backends")
    composed = (
        ObservabilityBuilder(service_name="builder-demo")
        .with_logger(ConsoleLogger())
        .with_tracer(InMemoryTracer())
        .with_metrics(InMemoryMetrics())
        .build()
    )
    log.debug("service_name", service_name=composed.service_name)
    log.debug("loggers", loggers=[type(l).__name__ for l in composed._loggers])
    log.debug("tracers", tracers=[type(t).__name__ for t in composed._tracers])
    log.debug("metrics", metrics=[type(m).__name__ for m in composed._metrics])
    composed.info("composed_record", note="no OTel providers created")

    # ── Example 2: Explicit OTLP setup (collector required) ──────
    log.debug("example", example=2, title="configure_otlp()")
    log.debug("checking_collector", endpoint=OTEL_COLLECTOR)
    otel_host, _, otel_port = OTEL_COLLECTOR.partition(":")
    otel_ok = await check_port(otel_host or "localhost", int(otel_port or 4317))
    log.debug("collector_status", reachable=otel_ok)

    if not otel_ok:
        log.debug("start_instructions")
        log.debug(
            "docker_command",
            command="docker compose -f docker-compose.yml up -d otel-collector",
        )
        return

    obs = configure_otlp(
        service_name="builder-demo",
        endpoint=OTEL_COLLECTOR,
        sample_rate=1.0,
        create_spans=True,
    )
    try:
        log.debug("loggers", loggers=[type(l).__name__ for l in obs._loggers])
        log.debug("tracers", tracers=[type(t).__name__ for t in obs._tracers])
        log.debug("metrics", metrics=[type(m).__name__ for m in obs._metrics])

        # ── Example 3: observe() context manager ─────────────────
        log.debug("example", example=3, title="Manual observe()")
        async with obs.observe(
            "custom_operation", step="data_processing", batch_size=32
        ):
            obs.info("processing_chunk", chunks=8)
            await asyncio.sleep(0.02)
            obs.info("chunk_complete", chunks_done=8)

        log.debug(
            "observe_info",
            detail="observe() auto-logs _started/_completed, records duration + metrics",
        )

        # ── Example 4: Auth headers for cloud OTLP endpoints ─────
        log.debug(
            "example",
            example=4,
            title="Auth headers (cloud OTLP endpoints)",
        )
        log.debug(
            "auth_headers",
            detail="configure_otlp(headers={'Authorization': 'Bearer <token>'})",
        )
    finally:
        await obs.shutdown()

    log.debug("separator", char="=", count=60)
    log.debug("builder_methods")
    log.debug("method", method="configure_otlp", params="service_name, endpoint, sample_rate, create_spans, headers, granularity, console")
    log.debug("usage")
    log.debug("usage_example", example="obs = configure_otlp(service_name='agent', endpoint='localhost:4317')")
    log.debug("usage_example", example="builder = ObservabilityBuilder().with_logger(my_logger).with_tracer(my_tracer).with_metrics(my_metrics)")
    log.debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
