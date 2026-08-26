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

"""Context-aware tools with RunContext and dependency injection.

Demonstrates:
  - Tools that receive RunContext to access dependencies and state
  - Dependency injection via deps_type / with_deps_type
  - How ToolRegistry detects RunContext and registers via agent.tool()
  - Combining context-aware and plain tools in one registry

When a tool's first parameter is annotated with RunContext,
ToolRegistry.register_to_agent() uses agent.tool() instead of
agent.tool_plain(), giving the tool access to the agent's
dependency container and run context metadata.

Usage:
    uv run python 02_context_tools.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python tools/02_context_tools.py
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import RunContext

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry

load_dotenv()

MODEL_NAME = os.getenv("TOOL_CALLING_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("TOOL_CALLING_MAX_TOKENS", "512"))


# ── Dependency container ────────────────────────────────────────────

@dataclass
class UserDeps:
    """Dependency container injected into context-aware tools."""

    user_id: str
    username: str
    role: str = "member"
    api_calls: list[str] = field(default_factory=list)

    def log_call(self, tool_name: str, params: str) -> None:
        """Log a tool invocation for auditing."""
        self.api_calls.append(f"{tool_name}({params})")
        print(f"  [deps:audit] user={self.username} invoked {tool_name}({params})")


# ── Context-aware tools ─────────────────────────────────────────────

def get_profile(ctx: RunContext[UserDeps]) -> str:
    """Get the current user's profile information.

    The tool uses RunContext to access the UserDeps dependency container.
    """
    deps = ctx.deps
    deps.log_call("get_profile", "")
    return (
        f"Profile:\n"
        f"  User ID: {deps.user_id}\n"
        f"  Username: {deps.username}\n"
        f"  Role: {deps.role}\n"
        f"  Total API calls: {len(deps.api_calls)}"
    )


def set_role(ctx: RunContext[UserDeps], new_role: str) -> str:
    """Change the current user's role (simulated).

    Args:
        new_role: The new role to assign (e.g., 'admin', 'member', 'viewer').
    """
    deps = ctx.deps
    deps.log_call("set_role", new_role)
    old_role = deps.role
    deps.role = new_role
    return f"Role changed from '{old_role}' to '{new_role}'"


def get_audit_log(ctx: RunContext[UserDeps], count: int = 5) -> str:
    """Retrieve the most recent API call audit log entries.

    Args:
        count: Number of recent entries to return (default 5).
    """
    deps = ctx.deps
    deps.log_call("get_audit_log", str(count))
    entries = deps.api_calls[-count:] if count > 0 else deps.api_calls
    if not entries:
        return "No audit log entries yet."
    return "Recent API calls:\n  " + "\n  ".join(entries)


# ── Plain tool (no context needed) ───────────────────────────────────

def echo(message: str) -> str:
    """Echo a message back to the user."""
    print(f"  [tool:echo] {message}")
    return f"Echo: {message}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    """Run the context-aware tools example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Context-Aware Tools with RunContext")
    print("=" * 60)

    # ── Create dependency container ─────────────────────────────
    deps = UserDeps(
        user_id="usr_abc123",
        username="alice",
        role="member",
    )

    # ── Register context-aware + plain tools ────────────────────
    # ToolRegistry inspects function signatures:
    #   - get_profile(ctx: RunContext[UserDeps]) → agent.tool()
    #   - set_role(ctx: RunContext[UserDeps], ...) → agent.tool()
    #   - echo(message: str) → agent.tool_plain()
    tools = ToolRegistry().add_many(get_profile, set_role, get_audit_log, echo)

    print("\nTool inspection:")
    for t in tools.get_tools():
        sig_info = "context-aware" if "RunContext" in str(t.__annotations__.get(list(t.__annotations__.keys())[0] if t.__annotations__ else "", "")) else "plain"
        print(f"  {t.__name__}: {sig_info}")

    # ── Build agent with dependency type ────────────────────────
    agent = (
        ManagedAgent(deps_type=UserDeps)
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(tools)
    )

    memory = InMemoryProvider()

    # ── Run 1: Get profile ──────────────────────────────────────
    print("\n--- Run 1: Get user profile ---")
    history1 = await MessageHistory().load("ctx-tools-1", memory)
    result1 = await agent.run(
        "Check my user profile.",
        history1,
        "ctx-tools-1",
        deps=deps,
    )
    print(f"  Output: {result1.output}")

    # ── Run 2: Change role ──────────────────────────────────────
    print("\n--- Run 2: Change role to admin ---")
    history2 = await MessageHistory().load("ctx-tools-2", memory)
    result2 = await agent.run(
        "Please change my role to 'admin'.",
        history2,
        "ctx-tools-2",
        deps=deps,
    )
    print(f"  Output: {result2.output}")

    # ── Run 3: Verify role changed ──────────────────────────────
    print("\n--- Run 3: Verify updated profile ---")
    history3 = await MessageHistory().load("ctx-tools-3", memory)
    result3 = await agent.run(
        "What is my current role and profile?",
        history3,
        "ctx-tools-3",
        deps=deps,
    )
    print(f"  Output: {result3.output}")

    # ── Run 4: Check audit log ──────────────────────────────────
    print("\n--- Run 4: Get audit log ---")
    history4 = await MessageHistory().load("ctx-tools-4", memory)
    result4 = await agent.run(
        "Show me the last 3 audit log entries.",
        history4,
        "ctx-tools-4",
        deps=deps,
    )
    print(f"  Output: {result4.output}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\nTotal deps API calls: {len(deps.api_calls)}")
    print(f"Final role: {deps.role}")


if __name__ == "__main__":
    asyncio.run(main())
