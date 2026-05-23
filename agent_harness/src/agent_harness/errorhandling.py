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

"""Error handling for agent runs with global error handler support."""

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic_ai.messages import ModelMessage


@dataclass
class ErrorContext:
    """Context about a failure for the agent to inspect."""

    error_type: str
    error_message: str
    source: str  # tool, memory, llm, unknown
    session_id: Optional[str] = None
    prompt: Optional[str] = None
    stack_trace: Optional[str] = None
    partial_result: Optional["AgentRunResult"] = None
    attempt: int = 1
    max_attempts: int = 1
    will_retry: bool = False


@dataclass
class AgentRunResult:
    """Result from agent run with error context."""

    output: Any
    success: bool
    error_context: Optional[ErrorContext] = None
    used_fallback: bool = False
    new_messages: list[ModelMessage] = field(default_factory=list)
    usage: Any = None


@dataclass
class AgentErrorContext:
    """Pure context about the agent when error occurred."""

    session_id: str
    prompt: str
    source: str
    error_context: ErrorContext


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling in agent runs."""

    on_error_handler: Optional[Callable[[AgentErrorContext, Exception], bool]] = None

    def with_error_handler(
        self, handler: Callable[[AgentErrorContext, Exception], bool]
    ) -> "ErrorHandlingConfig":
        """Set global error handler for agent run errors."""
        self.on_error_handler = handler
        return self


class ErrorHandler:
    """Handles errors in agent runs with configurable callbacks."""

    def __init__(self, config: ErrorHandlingConfig):
        self.config = config

    def handle_error(
        self,
        exception: Exception,
        source: str,
        session_id: str,
        prompt: str,
    ) -> Optional[AgentRunResult]:
        """
        Handle an error from agent run.

        Args:
            exception: The exception that was raised
            source: Error source (memory, llm, tool, unknown)
            session_id: Session ID
            prompt: User prompt

        Returns:
            AgentRunResult if handler returns True, None to rethrow
        """
        error_ctx = ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            source=source,
            session_id=session_id,
            prompt=prompt,
            stack_trace=traceback.format_exc(),
            partial_result=None,
        )

        agent_ctx = AgentErrorContext(
            session_id=session_id,
            prompt=prompt,
            source=source,
            error_context=error_ctx,
        )

        if self.config.on_error_handler:
            try:
                should_continue = self.config.on_error_handler(agent_ctx, exception)
                if should_continue:
                    return AgentRunResult(
                        output=None,
                        success=False,
                        error_context=error_ctx,
                        used_fallback=False,
                        new_messages=[],
                        usage=None,
                    )
            except Exception:
                pass

        return None

    @staticmethod
    def determine_error_source(exception: Exception) -> str:
        """Determine the error source based on exception type."""
        error_type = type(exception).__name__.lower()
        error_msg = str(exception).lower()

        if any(
            x in error_type or x in error_msg
            for x in ["mongo", "redis", "elastic", "motor", "database", "connection"]
        ):
            return "memory"
        elif any(
            x in error_type or x in error_msg
            for x in ["timeout", "model", "llm", "ollama", "openai", "api"]
        ):
            return "llm"
        elif any(
            x in error_type or x in error_msg for x in ["tool", "mcp", "function"]
        ):
            return "tool"
        else:
            return "unknown"
