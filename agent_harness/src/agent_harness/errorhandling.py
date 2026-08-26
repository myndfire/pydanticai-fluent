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

"""Error handling for agent runs with per-source callbacks."""

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic_ai.messages import ModelMessage


@dataclass
class ErrorContext:
    """Context about a failure for the agent to inspect."""

    error_type: str
    error_message: str
    source: str  # llm, tool, validation, guardrail, memory, prompt, evaluator, output
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
    cumulative_usage: Optional[dict] = None


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling with per-source callbacks.

    Each source has a dedicated callback. The callback receives an
    ErrorContext and returns:
      - Some(value) → suppress the error, value becomes AgentRunResult.output
      - None        → re-raise the exception

    If no source-specific callback is set, on_error is used as a catch-all.
    If neither is set, the exception propagates normally.
    """

    _on_llm_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_tool_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_validation_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_guardrail_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_memory_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_prompt_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_evaluator_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_output_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None
    _on_error: Optional[Callable[[ErrorContext], Optional[Any]]] = None

    def on_llm_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_llm_error = cb
        return self

    def on_tool_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_tool_error = cb
        return self

    def on_validation_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_validation_error = cb
        return self

    def on_guardrail_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_guardrail_error = cb
        return self

    def on_memory_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_memory_error = cb
        return self

    def on_prompt_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_prompt_error = cb
        return self

    def on_evaluator_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_evaluator_error = cb
        return self

    def on_output_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_output_error = cb
        return self

    def on_error(self, cb: Callable[[ErrorContext], Optional[Any]]) -> "ErrorHandlingConfig":
        self._on_error = cb
        return self


class ErrorHandler:
    """Handles errors in agent runs with source-based callback routing."""

    def __init__(self, config: ErrorHandlingConfig):
        self.config = config

    def handle_error(
        self,
        exception: Exception,
        source: str,
        session_id: str,
        prompt: str,
    ) -> Optional[AgentRunResult]:
        """Handle an error from agent run.

        Routes to the source-specific callback (e.g. on_llm_error for "llm"),
        then falls back to on_error. If the callback returns a value, the error
        is suppressed and that value becomes the output. If it returns None,
        the exception propagates.

        Args:
            exception: The exception that was raised
            source: Error source (llm, tool, validation, guardrail, memory,
                    prompt, evaluator, output)
            session_id: Session ID
            prompt: User prompt

        Returns:
            AgentRunResult if suppressed, None to re-raise
        """
        error_ctx = ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            source=source,
            session_id=session_id,
            prompt=prompt,
            stack_trace=traceback.format_exc(),
        )

        source_map: dict[str, Optional[Callable[[ErrorContext], Optional[Any]]]] = {
            "llm":        self.config._on_llm_error,
            "tool":       self.config._on_tool_error,
            "validation": self.config._on_validation_error,
            "guardrail":  self.config._on_guardrail_error,
            "memory":     self.config._on_memory_error,
            "prompt":     self.config._on_prompt_error,
            "evaluator":  self.config._on_evaluator_error,
            "output":     self.config._on_output_error,
        }
        handler = source_map.get(source) or self.config._on_error

        if handler:
            try:
                output = handler(error_ctx)
                if output is not None:
                    return AgentRunResult(
                        output=output,
                        success=False,
                        error_context=error_ctx,
                    )
            except Exception:
                pass

        return None
