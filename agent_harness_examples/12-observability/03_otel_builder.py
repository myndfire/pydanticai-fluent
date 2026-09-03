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

"""ObservabilityBuilder — fluent builder for OTEL observability.

Demonstrates:
  - ObservabilityBuilder.with_otel_observability() — one call configures
    logging + tracing + metrics via OTLP gRPC
  - Customizing endpoint, sample_rate, create_spans
  - build() → Observability instance
  - observe() context manager for manual instrumentation
  - Chaining builder methods

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

from agent_harness.observability import Observability, ObservabilityBuilder

load_dotenv()

log = structlog.get_logger()

OTEL_COLLECTOR = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")


async def main():
    log.debug("separator", char="=", count=60)
    log.debug("title", title="ObservabilityBuilder — OTEL Fluent Configuration")
    log.debug("separator", char="=", count=60)

    # ── Example 1: Default OTEL observability ────────────────────
    log.debug("example", example=1, title="Default with_otel_observability()")
    obs = Observability(
        builder=ObservabilityBuilder(service_name="builder-demo")
        .with_otel_observability()
    )
    log.debug("service_name", service_name=obs.service_name)
    log.debug("loggers", loggers=[type(l).__name__ for l in obs._loggers])
    log.debug("tracers", tracers=[type(t).__name__ for t in obs._tracers])
    log.debug("metrics", metrics=[type(m).__name__ for m in obs._metrics])

    # ── Example 2: Custom endpoint and sampling ──────────────────
    log.debug("example", example=2, title="Custom endpoint + sampling")
    obs2 = Observability(
        builder=ObservabilityBuilder(service_name="custom-demo")
        .with_otel_observability(
            otlp_endpoint=OTEL_COLLECTOR,
            sample_rate=0.5,
            create_spans=True,
        )
    )
    log.debug("endpoint", endpoint=OTEL_COLLECTOR)
    log.debug("sample_rate", rate=0.5)
    log.debug("create_spans", enabled=True)

    # ── Example 3: observe() context manager ─────────────────────
    log.debug("example", example=3, title="Manual observe()")
    async with obs.observe("custom_operation", step="data_processing", batch_size=32):
        obs.info("processing_chunk", chunks=8)
        await asyncio.sleep(0.02)
        obs.info("chunk_complete", chunks_done=8)

    log.debug("observe_info", detail="observe() auto-logs _started/_completed, records duration + metrics")

    # ── Example 4: Auth headers for cloud endpoints ──────────────
    log.debug("example", example=4, title="Auth headers (for cloud OTLP endpoints)")
    obs4 = Observability(
        builder=ObservabilityBuilder(service_name="cloud-demo")
        .with_otel_observability(
            otlp_endpoint="otlp.grafana-cloud.com:4317",
            headers={"Authorization": "Bearer <token>"},
        )
    )
    log.debug("auth_headers", detail="Built with auth headers (collector not running — gracefully no-ops)")

    log.debug("separator", char="=", count=60)
    log.debug("builder_methods")
    log.debug("method", method="with_otel_observability", params="endpoint, sample_rate, create_spans, headers")
    log.debug("method", method="with_logfire_observability", params="send_to_logfire")
    log.debug("usage")
    log.debug("usage_example", example="obs = Observability(builder=ObservabilityBuilder().with_otel_observability())")
    log.debug("usage_example", example="obs = ObservabilityBuilder().with_otel_observability().build()")
    log.debug("separator", char="=", count=60)


if __name__ == "__main__":
    asyncio.run(main())
