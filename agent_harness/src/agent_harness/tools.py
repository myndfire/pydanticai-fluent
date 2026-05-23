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

"""Tool registry with MCP support (placeholder)."""

from typing import Callable, Optional
from pydantic_ai import Agent


class ToolRegistry:
    """Fluent tool registry supporting multiple tool sources."""

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: list[Callable] = []

    def add(self, func: Callable) -> "ToolRegistry":
        """
        Add a custom function tool.

        Args:
            func: Function to register as a tool

        Returns:
            Self for chaining
        """
        self._tools.append(func)
        return self

    def add_many(self, *funcs: Callable) -> "ToolRegistry":
        """
        Add multiple function tools.

        Args:
            *funcs: Functions to register

        Returns:
            Self for chaining
        """
        self._tools.extend(funcs)
        return self

    def add_mcp(self, server: str, endpoint: Optional[str] = None) -> "ToolRegistry":
        """
        Add MCP (Model Context Protocol) server tools.

        NOTE: This is a placeholder for MCP integration.

        When MCP SDK is available, this will:
        1. Connect to MCP server at endpoint
        2. Discover available tools
        3. Wrap them as PydanticAI-compatible functions
        4. Add to registry

        Example usage:
            registry.add_mcp("filesystem")
            registry.add_mcp("browserbase", "http://localhost:3000")

        Args:
            server: MCP server name ("filesystem", "browserbase", etc.)
            endpoint: Optional custom endpoint URL

        Returns:
            Self for chaining
        """
        # Placeholder implementation
        print(f"[MCP Placeholder] Would connect to MCP server: {server}")
        if endpoint:
            print(f"[MCP Placeholder] Using endpoint: {endpoint}")

        # TODO: Implement actual MCP integration when SDK is available
        # Example pseudocode:
        #
        # from mcp import MCPClient
        #
        # client = MCPClient(endpoint or discover_endpoint(server))
        # tools = await client.list_tools()
        #
        # for tool in tools:
        #     wrapped_func = self._wrap_mcp_tool(tool, client)
        #     self._tools.append(wrapped_func)

        return self

    def _wrap_mcp_tool(self, tool_spec: dict, client: any) -> Callable:
        """
        Wrap an MCP tool as a PydanticAI-compatible function.

        NOTE: Placeholder implementation.

        Args:
            tool_spec: MCP tool specification
            client: MCP client instance

        Returns:
            Wrapped function
        """

        # Placeholder - would create a function that calls MCP tool
        async def mcp_tool_wrapper(**kwargs):
            # return await client.call_tool(tool_spec["name"], kwargs)
            pass

        return mcp_tool_wrapper

    def register_to_agent(self, agent: Agent):
        """
        Register all tools to a PydanticAI agent.

        Uses tool_plain for functions without RunContext parameter.
        Uses tool for functions with RunContext parameter.

        Args:
            agent: PydanticAI agent instance
        """
        import inspect

        for func in self._tools:
            # Check if function has RunContext parameter
            sig = inspect.signature(func)
            params = sig.parameters

            # If first param is RunContext, use tool, otherwise use tool_plain
            if params:
                first_param = list(params.values())[0]
                # Check if first param annotation contains RunContext
                annotation = str(first_param.annotation)
                if "RunContext" in annotation:
                    agent.tool(func)
                else:
                    agent.tool_plain(func)
            else:
                agent.tool_plain(func)

    def get_tools(self) -> list[Callable]:
        """
        Get all registered tools.

        Returns:
            List of tool functions
        """
        return self._tools.copy()

    def clear(self) -> "ToolRegistry":
        """
        Clear all registered tools.

        Returns:
            Self for chaining
        """
        self._tools.clear()
        return self


# Helper function for MCP server discovery (placeholder)
def discover_mcp_servers() -> list[str]:
    """
    Discover available MCP servers.

    NOTE: Placeholder implementation.

    Returns:
        List of available MCP server names
    """
    # Placeholder - would query MCP registry
    return ["filesystem", "browserbase", "web-search"]


def get_mcp_server_info(server_name: str) -> dict:
    """
    Get information about an MCP server.

    NOTE: Placeholder implementation.

    Args:
        server_name: MCP server name

    Returns:
        Server info dict
    """
    # Placeholder - would query MCP server metadata
    return {
        "name": server_name,
        "endpoint": f"http://localhost:3000/{server_name}",
        "description": f"MCP server for {server_name}",
        "available": False,  # Placeholder
    }
