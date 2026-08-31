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

NOTE: MCP support uses pydantic_ai's MCPToolset and FastMCPClient.
Requires a running MCP server (e.g., via mcp-server-sdk or
an SSE-compatible MCP implementation).

Usage:
    uv run python 03_mcp_server.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python tools/03_mcp_server.py
"""

import asyncio
import os
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry

load_dotenv()

MODEL_NAME = os.getenv("TOOL_CALLING_MODEL_NAME", "gpt-oss:20b")
MAX_TOKENS = int(os.getenv("TOOL_CALLING_MAX_TOKENS", "512"))
MCP_HTTP_URL = os.getenv("MCP_HTTP_URL", "https://mcp.context7.com/mcp")
MCP_COMPLEX_URL = os.getenv("MCP_COMPLEX_URL", "https://mcpplaygroundonline.com/mcp-complex-server")


# ── Example custom tool to complement MCP tools ─────────────────────

def echo(message: str) -> str:
    """Echo a message back."""
    print(f"  [tool:echo] {message}")
    return f"Echo: {message}"


# ── Main ────────────────────────────────────────────────────────────

async def main():
    """Run the MCP server integration example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull gpt-oss:20b
        3. Install deps: cd agent_harness_examples && uv sync
    """
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
    #   agent.with_mcp_server("MCP_HTTP_URL")
    # Common MCP servers:
    #   - filesystem: file read/write operations
    #   - browserbase: web browsing automation
    #   - postgres: database queries
    #   - github: repository operations

    # API pattern (requires running MCP server):
    # agent1 = (
    #     ManagedAgent()
    #     .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    #     .with_mcp_server("MCP_HTTP_URL/mcp/filesystem")
    # )

    print("  API: agent.with_mcp_server(\"MCP_HTTP_URL/mcp/filesystem\")")

    # ── Example 2: Multiple MCP servers ─────────────────────────
    print("\n--- Example 2: Multiple MCP servers ---")
    print("  API: agent.with_mcp_servers(\n"
          "    \"MCP_HTTP_URL/mcp/filesystem\",\n"
          "    \"http://localhost:8001/mcp/postgres\",\n"
          "    \"http://localhost:8002/mcp/github\",\n"
          "  )")

    # ── Example 3: MCP with tool_prefix for disambiguation ──────
    print("\n--- Example 3: tool_prefix for disambiguation ---")
    print("  Use tool_prefix to avoid name collisions when multiple")
    print("  servers expose tools with the same name.\n")
    print("  API: agent.with_mcp_server(\n"
          "    \"MCP_HTTP_URL/mcp/filesystem\",\n"
          "    tool_prefix=\"fs_\"\n"
          "  )")
    print("  API: agent.with_mcp_server(\n"
          "    \"http://localhost:8001/mcp/s3\",\n"
          "    tool_prefix=\"s3_\"\n"
          "  )")

    # ── Example 4: MCP + custom tools ───────────────────────────
    print("\n--- Example 4: MCP toolsets + custom tools ---")
    # Combine two MCP servers with custom ToolRegistry tools.
    # Context7 provides library documentation tools.
    # Complex server provides data/user/order tools.
    # tool_prefix disambiguates tool names across servers.
    custom_tools = ToolRegistry().add(echo)
    agent4 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(custom_tools)
        .with_mcp_server(MCP_HTTP_URL, tool_prefix="ctx7_")
        .with_mcp_server(MCP_COMPLEX_URL, tool_prefix="complex_")
    )
    print("  API: agent.with_tools(ToolRegistry().add(echo))\n"
          "       .with_mcp_server(MCP_HTTP_URL, tool_prefix=\"ctx7_\")\n"
          "       .with_mcp_server(MCP_COMPLEX_URL, tool_prefix=\"complex_\")")
    print("  Tools available: ctx7_*, complex_*, echo")

    # ── Example 5: Run agent with MCP + custom tool (live demo) ─
    print("\n" + "=" * 60)
    print("Live Demo: Agent with MCP + custom tool")
    print("=" * 60)

    custom_tools = ToolRegistry().add(echo)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_tools(custom_tools)
        .with_mcp_server(MCP_HTTP_URL)
    )

    history = await MessageHistory().load("mcp-demo-live", memory)
    result = await agent.run(
        "Use the MCP-provided tools to resolve the library id for 'fastapi' and fetch a docs excerpt.",
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
