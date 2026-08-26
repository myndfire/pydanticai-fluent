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

import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional
from pydantic_ai import Agent


# ── Tool call logging ─────────────────────────────────────────────────


def _log_tool_call(observability, tool_name: str, params: Dict[str, Any]) -> None:
    """Log tool call invocation via Observability."""
    if observability:
        observability.log_info(
            "tool_call",
            func_name=tool_name,
            tool={"name": tool_name, "parameters": params},
        )
    else:
        # Bootstrap fallback — Observability not yet configured
        print(f"[tool_call] {tool_name}({params})")


def _log_tool_result(observability, tool_name: str, params: Dict[str, Any], result: Any, duration: float) -> None:
    """Log tool result via Observability."""
    if observability:
        observability.log_info(
            "tool_result",
            func_name=tool_name,
            tool={"name": tool_name, "parameters": params},
            result=str(result)[:200],  # Truncate long results
            performance={"duration_seconds": duration},
        )
    else:
        print(f"[tool_result] {tool_name}({params}) = {str(result)[:200]}")


def _log_tool_error(observability, tool_name: str, params: Dict[str, Any], error: Exception, duration: float) -> None:
    """Log tool error via Observability."""
    if observability:
        observability.log_error(
            "tool_error",
            func_name=tool_name,
            tool={"name": tool_name, "parameters": params},
            error={"type": type(error).__name__, "message": str(error)},
            performance={"duration_seconds": duration},
        )
    else:
        print(f"[tool_error] {tool_name}({params}) → {type(error).__name__}: {error}")


# ── ToolRegistry ───────────────────────────────────────────────────


class ToolRegistry:
    """Fluent tool registry supporting multiple tool sources."""

    def __init__(self, observability=None):
        """Initialize empty tool registry.
        
        Args:
            observability: Observability instance for structured logging
        """
        self._tools: list[Callable] = []
        self._observability = observability

    def _wrap_tool(self, func: Callable) -> Callable:
        """Wrap a tool function to log invocations, results, and errors.
        
        Reads self._observability at call time so it always uses the
        current value (e.g. after with_tools() injects observability).
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            _log_tool_call(self._observability, func.__name__, kwargs)
            try:
                # Support both sync and async tools
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                duration = time.time() - start
                _log_tool_result(self._observability, func.__name__, kwargs, result, duration)
                return result
            except Exception as e:
                duration = time.time() - start
                _log_tool_error(self._observability, func.__name__, kwargs, e, duration)
                raise

        return wrapper

    def add(self, func: Callable) -> "ToolRegistry":
        """
        Add a custom function tool with structured logging wrapper.

        Args:
            func: Function to register as a tool

        Returns:
            Self for chaining
        """
        self._tools.append(self._wrap_tool(func))
        return self

    def add_many(self, *funcs: Callable) -> "ToolRegistry":
        """
        Add multiple function tools with structured logging wrappers.

        Args:
            *funcs: Functions to register

        Returns:
            Self for chaining
        """
        for func in funcs:
            self._tools.append(self._wrap_tool(func))
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

        return self

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
