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

"""Logging — ConsoleLogger, FileLogger, structured context, log levels.

Demonstrates:
  - ConsoleLogger: structlog-based console output (default in Observability)
  - FileLogger: file-based logging with daily or size-based rotation
  - Structured context: pass key-value pairs as **kwargs
  - Log levels: debug, info, warning, error across all loggers

Usage:
    python 01_logging.py
"""

import asyncio

from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.logging import ConsoleLogger, FileLogger


async def main():
    print("=" * 60)
    print("Logging — ConsoleLogger + FileLogger + Structured Context")
    print("=" * 60)

    # ── Example 1: ConsoleLogger (default) ──────────────────────
    print("\n--- Example 1: ConsoleLogger (default) ---")
    console = ConsoleLogger()
    print("  ConsoleLogger sends structured logs to stderr via structlog.")

    console.info("agent_started", model="gpt-4o", session_id="abc123")
    console.warning("rate_limit_approaching", remaining=100, limit=500)
    console.error("token_limit_exceeded", actual=4097, max=4096)

    # ── Example 2: FileLogger with daily rotation ───────────────
    print("\n--- Example 2: FileLogger (daily rotation) ---")
    file_log = FileLogger(
        log_file="agent_example.log",
        rotation="daily",
        retention=7,
    )
    print(f"  FileLogger writing to: {file_log.log_file} (daily rotation, 7 days retention)")

    file_log.info("agent_run_started", session_id="xyz789")
    file_log.info("tool_invoked", tool="search", query="quantum computing")
    file_log.warning("slow_response", duration_seconds=5.2, threshold=3.0)
    file_log.error("tool_failed", tool="api_call", error="ConnectionError")

    # ── Example 3: FileLogger with size rotation ────────────────
    print("\n--- Example 3: FileLogger (size rotation) ---")
    size_log = FileLogger(
        log_file="agent_size.log",
        rotation="size",
        retention=5,
    )
    print(f"  FileLogger writing to: {size_log.log_file} (10MB rotation, 5 files retention)")
    size_log.info("size_rotated_log", example=True)

    # ── Example 4: Structured context (key-value pairs) ─────────
    print("\n--- Example 4: Structured context ---")
    print("  All log methods accept **kwargs as structured key-value pairs:")
    print("    logger.info('event_name', user_id='u1', action='login', ip='10.0.0.1')")

    console.info("user_login", user_id="usr_42", action="login", ip="10.0.0.1")
    console.info("llm_call", provider="openai", model="gpt-4o",
                 input_tokens=150, output_tokens=80, duration_ms=1200)
    console.error("pipeline_failure", step="rag_retrieval",
                  error="timeout", retry_count=3)

    # ── Example 5: Observability with console + file ────────────
    print("\n--- Example 5: Observability with ConsoleLogger + FileLogger ---")
    obs = Observability(
        loggers=[ConsoleLogger(), FileLogger(log_file="combined.log")],
    )
    print("  Loggers: ConsoleLogger + FileLogger(combined.log)")

    obs.info("observability_initialized", loggers=2)
    obs.warning("memory_usage_high", percent=85, threshold=80)
    obs.error("connection_lost", endpoint="localhost:9200", retries=5)

    # ── Example 6: Log levels demonstration ─────────────────────
    print("\n--- Example 6: All log levels ---")
    for level_name, method in [
        ("debug", console.debug),
        ("info", console.info),
        ("warning", console.warning),
        ("error", console.error),
    ]:
        print(f"  Calling logger.{level_name}('test_{level_name}_message')...")
        method(f"test_{level_name}_message", source="example", level=level_name)

    print(f"\n  Check agent_example.log, agent_size.log, and combined.log for file output.")


if __name__ == "__main__":
    asyncio.run(main())
