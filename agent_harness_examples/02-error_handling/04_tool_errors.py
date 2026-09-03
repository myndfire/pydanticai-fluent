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

"""Tool call errors — handling failures when tool functions raise exceptions.

Demonstrates:
  - Registering a tool that deliberately raises
  - on_tool_error callback to suppress tool failures gracefully
  - Source="tool" classification when a tool function crashes
  - agent.run() tagging the exception with _error_source="tool"
  - Fallback response when tools are unavailable

Usage:
    uv run python 04_tool_errors.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python error_handling/04_tool_errors.py
"""

import asyncio

import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext

log = structlog.get_logger()


# ── Tool that raises ─────────────────────────────────────────────────

def broken_divider(a: float, b: float) -> str:
    """Divide a by b — but fails on certain inputs.

    Args:
        a: Numerator
        b: Denominator
    """
    log.debug("tool_call", tool="broken_divider", a=a, b=b)
    if b == 0:
        log.debug("tool_error", tool="broken_divider", error="division by zero")
        raise ValueError(f"Cannot divide {a} by zero")
    return f"{a} / {b} = {a / b}"


def stable_echo(message: str) -> str:
    """Echo a message — this tool always works."""
    log.debug("tool_call", tool="stable_echo", message=message)
    return f"Echo: {message}"


# ── Tool error handler ──────────────────────────────────────────────

def on_tool_failure(ctx: ErrorContext) -> str | None:
    """Handle tool execution failures with a graceful fallback."""
    log.debug(
        "tool_error_intercepted",
        error_type=ctx.error_type,
        error_message=ctx.error_message,
        source=ctx.source,
        session_id=ctx.session_id,
        has_stack=ctx.stack_trace is not None,
    )
    return (
        f"I'm sorry, the calculation tool encountered an error "
        f"({ctx.error_type}). Using echo instead."
    )


async def main():
    log.debug("separator")
    log.debug("section", title="Tool Errors — Handling Tool Function Failures")
    log.debug("separator")

    memory = InMemoryProvider()

    # ── Build agent with error-broken tool ──────────────────────
    tools = ToolRegistry().add_many(broken_divider, stable_echo)

    config = ErrorHandlingConfig().on_tool_error(on_tool_failure)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(tools)
        .with_error_handling(config)
    )

    log.debug("tools_registered", count=len(tools.get_tools()))
    for t in tools.get_tools():
        log.debug("tool", name=t.__name__)

    log.debug("status", handler="on_tool_error → on_tool_failure")

    # ── Run: ask the agent to divide by zero ────────────────────
    log.debug("example", title="Run: divide by zero (tool will raise)")
    history = await MessageHistory().load("tool-err-1", memory)
    result = await agent.run(
        "Use the broken_divider tool to divide 10 by 0. "
        "If it fails, use stable_echo to say 'division failed'.",
        history,
        "tool-err-1",
    )
    log.debug("result", success=result.success, output=result.output)

    # ── Run: valid division (tool succeeds) ─────────────────────
    log.debug("example", title="Run: valid division (tool succeeds, no error)")
    history2 = await MessageHistory().load("tool-err-2", memory)
    result2 = await agent.run(
        "Use the broken_divider tool to divide 42 by 6.",
        history2,
        "tool-err-2",
    )
    log.debug("result", success=result2.success, output=result2.output)

    # ── Summary ─────────────────────────────────────────────────
    log.debug("example", title="Summary")
    log.debug("summary", message="Tool errors are caught and routed to on_tool_error.")
    log.debug("summary", message="The callback receives ErrorContext with:")
    log.debug("summary", detail="error_type: the Python exception class name")
    log.debug("summary", detail="error_message: the exception message")
    log.debug("summary", detail="stack_trace: full traceback for debugging")
    log.debug("summary", message="Return a string to suppress, None to re-raise.")


if __name__ == "__main__":
    asyncio.run(main())
