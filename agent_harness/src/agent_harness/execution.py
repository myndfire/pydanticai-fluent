"""Shared execution identity, budgets, and cancellation state."""

from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class RunStatus:
    """Terminal statuses shared by all execution paths."""

    SUCCESS = "success"
    ERROR = "error"
    HANDLED_ERROR = "handled_error"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionBudget:
    """Optional limits shared by model, tool, evaluator, and orchestration work."""

    deadline: Optional[float] = None
    max_iterations: Optional[int] = None
    max_model_requests: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    _iterations: int = field(default=0, init=False, repr=False)
    _model_requests: int = field(default=0, init=False, repr=False)
    _tool_calls: int = field(default=0, init=False, repr=False)

    @classmethod
    def with_timeout(cls, timeout: Optional[float], **kwargs: Any) -> "ExecutionBudget":
        """Build a budget using a relative timeout in seconds."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        return cls(deadline=deadline, **kwargs)

    def remaining_seconds(self) -> Optional[float]:
        """Return remaining deadline time, or ``None`` when unlimited."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def check(self) -> None:
        """Raise ``TimeoutError`` when the execution deadline has elapsed."""
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("execution deadline exceeded")

    def consume_iteration(self) -> None:
        self._iterations += 1
        if self.max_iterations is not None and self._iterations > self.max_iterations:
            raise RuntimeError("execution iteration budget exceeded")

    def consume_model_request(self) -> None:
        self._model_requests += 1
        if (
            self.max_model_requests is not None
            and self._model_requests > self.max_model_requests
        ):
            raise RuntimeError("model request budget exceeded")

    def consume_tool_call(self) -> None:
        self._tool_calls += 1
        if self.max_tool_calls is not None and self._tool_calls > self.max_tool_calls:
            raise RuntimeError("tool call budget exceeded")


@dataclass
class ExecutionContext:
    """Identity and controls propagated through one agent execution."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    tenant_id: Optional[str] = None
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    metadata: dict[str, Any] = field(default_factory=dict)

    def child(self, **metadata: Any) -> "ExecutionContext":
        """Create a child identity while preserving the parent run relationship."""
        return ExecutionContext(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            parent_run_id=self.run_id,
            tenant_id=self.tenant_id,
            budget=self.budget,
            metadata={**self.metadata, **metadata},
        )

    def as_dict(self) -> dict[str, Any]:
        """Return canonical fields for logs, traces, and callbacks."""
        return {
            "run.id": self.run_id,
            "session.id": self.session_id,
            "conversation.id": self.conversation_id,
            "parent.run.id": self.parent_run_id,
            "tenant.id": self.tenant_id,
            **self.metadata,
        }


CURRENT_EXECUTION: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar(
    "agent_harness_execution", default=None
)


def current_execution() -> Optional[ExecutionContext]:
    """Return the execution context active in the current async task."""
    return CURRENT_EXECUTION.get()
