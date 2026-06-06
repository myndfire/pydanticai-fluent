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

"""MCP (Model Context Protocol) server integration.

Demonstrates:
  - with_mcp_server() — add a single MCP server as a toolset
  - with_mcp_servers() — add multiple MCP servers at once
  - tool_prefix option for disambiguating tool names across servers
  - Combining MCP toolsets with custom tools via ToolRegistry

MCP servers expose tools over HTTP that the agent can discover and
invoke at runtime (e.g., filesystem access, web browsing, databases).

NOTE: MCP support uses pydantic_ai's MCPServerStreamableHTTP.
Requires a running MCP server (e.g., via mcp-server-sdk or
an SSE-compatible MCP implementation).

Usage:
    uv run python 03_mcp_server.py
"""

import asyncio

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry


# ── Example custom tool to complement MCP tools ─────────────────────

def echo(message: str) -> str:
    """Echo a message back."""
    print(f"  [tool:echo] {message}")
    return f"Echo: {message}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("MCP Server Integration")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Example 1: Single MCP server ────────────────────────────
    print("\n--- Example 1: Single MCP server ---")
    print("  NOTE: MCP server must be running at the given URL.")
    print("  This example shows the fluent API pattern.")
    print()

    # In production, replace with a real MCP server URL:
    #   agent.with_mcp_server("http://localhost:8000")
    # Common MCP servers:
    #   - filesystem: file read/write operations
    #   - browserbase: web browsing automation
    #   - postgres: database queries
    #   - github: repository operations

    # API pattern (requires running MCP server):
    # agent1 = (
    #     ManagedAgent()
    #     .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    #     .with_mcp_server("http://localhost:8000/mcp/filesystem")
    # )

    print("  API: agent.with_mcp_server(\"http://localhost:8000/mcp/filesystem\")")

    # ── Example 2: Multiple MCP servers ─────────────────────────
    print("\n--- Example 2: Multiple MCP servers ---")
    print("  API: agent.with_mcp_servers(\n"
          "    \"http://localhost:8000/mcp/filesystem\",\n"
          "    \"http://localhost:8001/mcp/postgres\",\n"
          "    \"http://localhost:8002/mcp/github\",\n"
          "  )")

    # ── Example 3: MCP with tool_prefix for disambiguation ──────
    print("\n--- Example 3: tool_prefix for disambiguation ---")
    print("  Use tool_prefix to avoid name collisions when multiple")
    print("  servers expose tools with the same name.\n")
    print("  API: agent.with_mcp_server(\n"
          "    \"http://localhost:8000/mcp/filesystem\",\n"
          "    tool_prefix=\"fs_\"\n"
          "  )")
    print("  API: agent.with_mcp_server(\n"
          "    \"http://localhost:8001/mcp/s3\",\n"
          "    tool_prefix=\"s3_\"\n"
          "  )")

    # ── Example 4: MCP + custom tools ───────────────────────────
    print("\n--- Example 4: MCP toolsets + custom tools ---")
    # Combine MCP servers with custom ToolRegistry tools:
    # custom_tools = ToolRegistry().add_many(echo)
    # agent = (
    #     ManagedAgent()
    #     .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    #     .with_tools(custom_tools)
    #     .with_mcp_servers(
    #         "http://localhost:8000/mcp/filesystem",
    #         "http://localhost:8001/mcp/web-search",
    #         tool_prefix="mcp_",
    #     )
    # )
    print("  API: agent.with_tools(ToolRegistry().add(echo))\n"
          "       .with_mcp_servers(\n"
          "           \"http://localhost:8000/mcp/filesystem\",\n"
          "           \"http://localhost:8001/mcp/web-search\",\n"
          "           tool_prefix=\"mcp_\",\n"
          "       )")

    # ── Example 5: Run agent with MCP + custom tool (live demo) ─
    print("\n" + "=" * 60)
    print("Live Demo: Agent with custom tool (no MCP server needed)")
    print("=" * 60)

    custom_tools = ToolRegistry().add(echo)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_tools(custom_tools)
    )
    # In production, you would also add:
    # .with_mcp_server("http://localhost:8000")

    history = await MessageHistory().load("mcp-demo-live", memory)
    result = await agent.run(
        "Use the echo tool to say 'Hello from MCP demo!'",
        history,
        "mcp-demo-live",
    )
    print(f"\n  Output: {result.output}")

    # ── Example 6: MCP server discovery (placeholder) ───────────
    print("\n--- Example 6: MCP server discovery ---")
    from agent_harness.tools import discover_mcp_servers, get_mcp_server_info

    servers = discover_mcp_servers()
    print(f"  Discoverable servers: {servers}")
    for server in servers:
        info = get_mcp_server_info(server)
        print(f"    {info['name']}: {info['description']} "
              f"(endpoint: {info['endpoint']}, available: {info['available']})")
    print("  NOTE: These are placeholder stubs for future MCP SDK integration.")


if __name__ == "__main__":
    asyncio.run(main())
