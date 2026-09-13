"""Canonical attributes and helpers shared by all telemetry signals."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from typing import Any


class TelemetryFields:
    """Stable field names used in logs, spans, and metric attributes."""

    ERROR_ID = "error.id"
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_SOURCE = "error.source"
    ERROR_HANDLED = "error.handled"
    ERROR_RETRYABLE = "error.retryable"
    RUN_ID = "run.id"
    SESSION_ID = "session.id"
    CONVERSATION_ID = "conversation.id"
    OPERATION = "operation.name"
    MODEL = "model.name"
    PROVIDER = "model.provider"
    REQUESTED_MODEL = "model.requested.name"
    REQUESTED_PROVIDER = "model.requested.provider"
    RESPONSE_MODEL = "model.response.name"
    RESPONSE_PROVIDER = "model.response.provider"
    COMPONENT = "component"
    WORKFLOW = "workflow.name"
    WORKFLOW_STEP = "workflow.step"
    STATUS = "status"


class TelemetryEvents:
    """Canonical event names for drill-down queries."""

    RUN_STARTED = "agent.run.started"
    RUN_COMPLETED = "agent.run.completed"
    RUN_FAILED = "agent.run.failed"
    TURN_COMPLETED = "agent.turn.completed"
    MODEL_COMPLETED = "model.request.completed"
    MODEL_FAILED = "model.request.failed"
    TOOL_STARTED = "tool.call.started"
    TOOL_COMPLETED = "tool.call.completed"
    TOOL_FAILED = "tool.call.failed"
    RETRY = "retry.attempted"


def new_error_id() -> str:
    """Return a sortable-enough identifier for one logical failure."""
    return uuid.uuid4().hex


def execution_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy context keys into the canonical telemetry schema."""
    aliases = {
        "session_id": TelemetryFields.SESSION_ID,
        "conversation_id": TelemetryFields.CONVERSATION_ID,
        "run_id": TelemetryFields.RUN_ID,
        "model": TelemetryFields.MODEL,
        "model_name": TelemetryFields.MODEL,
        "provider": TelemetryFields.PROVIDER,
        "provider_name": TelemetryFields.PROVIDER,
    }
    normalized = dict(context)
    for old, new in aliases.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]
    return normalized


def bounded_metric_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded dimensions safe for metric cardinality."""
    allowed = {
        TelemetryFields.COMPONENT,
        TelemetryFields.OPERATION,
        TelemetryFields.MODEL,
        TelemetryFields.PROVIDER,
        TelemetryFields.STATUS,
        TelemetryFields.ERROR_TYPE,
        TelemetryFields.ERROR_SOURCE,
        TelemetryFields.ERROR_HANDLED,
        "phase",
    }
    return {
        key: value
        for key, value in execution_context(attributes).items()
        if key in allowed and value is not None
    }


def service_version() -> str:
    """Resolve the deployed package version without importing the package."""
    return os.getenv("SERVICE_VERSION", "0.1.0")
