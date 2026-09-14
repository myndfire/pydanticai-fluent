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

"""Composable structured logging backends.

The application chooses destinations and composes them before injecting a
``Logger`` into ``Observability``. This module does not configure global
logging or structlog state.
"""

import contextvars
import json
import math
import os
import sys
import sysconfig
from collections.abc import Mapping
from typing import Protocol, Any

def _normalize_otel_attr(v: Any) -> Any:
    """Keep OTel-supported primitive types; stringify everything else."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v if isinstance(v, int) or math.isfinite(v) else str(v)
    if v is None:
        return "None"
    return str(v)


def _flatten_telemetry_attrs(context: dict, prefix: str = "") -> dict:
    """Flatten a log context into dotted primitive attributes.

    Nested mappings are expanded into dotted keys (``token_usage.total_tokens``,
    ``performance.duration_seconds``, ``model_settings.max_tokens``) so backends
    such as Elasticsearch index them as queryable, numeric fields instead of
    opaque ``str(dict)`` values. Sequences are JSON-encoded under their own key;
    ``None`` values are dropped to reduce noise.
    """
    flat: dict = {}
    for key, value in context.items():
        dotted = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, Mapping):
            flat.update(_flatten_telemetry_attrs(dict(value), f"{dotted}."))
        elif isinstance(value, (list, tuple, set)):
            flat[dotted] = json.dumps([_normalize_otel_attr(v) for v in value])
        else:
            flat[dotted] = _normalize_otel_attr(value)
    return flat


_HARNESS_ROOT = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__))
)
_STDLIB_DIRS = tuple(
    p
    for p in {
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
    }
    if p
)
_SITE_PACKAGES = sysconfig.get_path("purelib") or ""


def _is_harness_or_internal_frame(filename: str) -> bool:
    """True if a frame is inside agent_harness, stdlib, or third-party deps."""
    norm = os.path.normpath(filename or "")
    if not norm or norm.startswith("<"):
        return True
    if norm.startswith(_HARNESS_ROOT):
        return True
    for base in _STDLIB_DIRS:
        if base and norm.startswith(base):
            return True
    if _SITE_PACKAGES and norm.startswith(_SITE_PACKAGES):
        return True
    return False


def _app_callsite() -> dict:
    """Return the first non-harness/non-stdlib frame for OTel ``code.*``.

    Walks the stack from the caller of the logger outward, skipping any frame
    that lives inside the agent_harness package, the Python stdlib, or a
    third-party site-package.  The first "user" frame encountered is treated as
    the callsite. Uses a fast frame walk (no ``inspect.stack()``).

    Returns
    -------
    dict
        ``code.file.path`` / ``code.function`` / ``code.line.number`` /
        ``code.namespace`` or an empty dict if no suitable frame is found.
    """
    frame = sys._getframe(1)
    while frame is not None:
        if not _is_harness_or_internal_frame(frame.f_code.co_filename):
            return {
                "code.file.path": os.path.relpath(frame.f_code.co_filename),
                "code.function": frame.f_code.co_name,
                "code.line.number": frame.f_lineno,
                "code.namespace": frame.f_globals.get("__name__", ""),
            }
        frame = frame.f_back
    return {}


def app_code_location(tb) -> dict:
    """Return ``code.*`` for the DEEPEST user frame in a traceback.

    Walks the traceback and keeps the last frame that is not inside
    agent_harness, the Python stdlib, or a third-party site-package -- i.e. the
    user-code frame closest to where the exception was raised. Returns ``{}``
    when the stack is entirely internal (e.g. an error raised inside
    pydantic-ai, whose asyncio task drops the caller's frame).
    """
    deepest: dict = {}
    current = tb
    while current is not None:
        frame = current.tb_frame
        if not _is_harness_or_internal_frame(frame.f_code.co_filename):
            deepest = {
                "code.file.path": os.path.relpath(frame.f_code.co_filename),
                "code.function": frame.f_code.co_name,
                "code.line.number": frame.f_lineno,
                "code.namespace": frame.f_globals.get("__name__", ""),
            }
        current = current.tb_next
    return deepest


def caller_code_location() -> dict:
    """Return ``code.*`` for the first user frame on the CURRENT stack.

    Used to capture the caller of ``ManagedAgent.run()`` so failures raised
    inside pydantic-ai's separate asyncio task can still be attributed back to
    the application. Fast (frame walk, no ``inspect.stack()``).
    """
    frame = sys._getframe(1)
    while frame is not None:
        if not _is_harness_or_internal_frame(frame.f_code.co_filename):
            return {
                "code.file.path": os.path.relpath(frame.f_code.co_filename),
                "code.function": frame.f_code.co_name,
                "code.line.number": frame.f_lineno,
                "code.namespace": frame.f_globals.get("__name__", ""),
            }
        frame = frame.f_back
    return {}


# Per-run caller call site, read by observability/tracing when an error has no
# user frame of its own (pydantic-ai runs the agent in its own asyncio task, so
# the caller frame is not on the failing stack).
_HARNESS_CALL_SITE: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "harness_call_site", default={}
)


def set_harness_call_site(location: dict | None) -> None:
    """Record the current run's caller call site for error attribution."""
    _HARNESS_CALL_SITE.set(location or {})


def get_harness_call_site() -> dict:
    """Return the current run's caller call site (``{}`` if unset)."""
    return _HARNESS_CALL_SITE.get()



class Logger(Protocol):
    """Protocol for structured logging."""

    def debug(self, message: str, **context):
        """Log debug message."""
        ...

    def info(self, message: str, **context):
        """Log info message."""
        ...

    def warning(self, message: str, **context):
        """Log warning message."""
        ...

    def error(self, message: str, **context):
        """Log error message."""
        ...


class NoOpLogger:
    """No-op structured logger used when telemetry is disabled."""

    def debug(self, message: str, **context):
        pass

    def info(self, message: str, **context):
        pass

    def warning(self, message: str, **context):
        pass

    def error(self, message: str, **context):
        pass

    def close(self):
        pass

    def shutdown(self):
        pass


class ConsoleLogger:
    """Structured logger that writes records to the application console."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr

    def _write(self, level: str, message: str, context: dict) -> None:
        suffix = " " + json.dumps(context, default=str, sort_keys=True) if context else ""
        print(f"[{level.upper()}] {message}{suffix}", file=self.stream, flush=True)

    def debug(self, message: str, **context):
        self._write("debug", message, context)

    def info(self, message: str, **context):
        self._write("info", message, context)

    def warning(self, message: str, **context):
        self._write("warning", message, context)

    def error(self, message: str, **context):
        self._write("error", message, context)


class OTELLogger:
    """OpenTelemetry structured logging through an injected provider.

    The application owns the provider and its exporters. Records emitted inside
    an active span automatically carry trace_id/span_id for log-trace
    correlation.

    Records are emitted under an instrumentation-scope named after the
    ``component`` attribute (falling back to ``service_name``), so backends can
    group by subsystem.
    """

    def __init__(
        self,
        logger_provider: Any,
        service_name: str = "agent",
        telemetry_level: str = "standard",
    ):
        """
        Initialize OTEL logging.

        Args:
            logger_provider: Application-owned OTel LoggerProvider.
            service_name: Instrumentation scope name for log records.
            telemetry_level: Granularity level exported as the
                ``harness.telemetry.level`` resource attribute.
        """
        self.service_name = service_name
        self._provider = logger_provider
        self.telemetry_level = telemetry_level
        self._loggers: dict[str, Any] = {}
        from opentelemetry._logs import SeverityNumber
        self._severity_map = {
            "debug": SeverityNumber.DEBUG,
            "info": SeverityNumber.INFO,
            "warning": SeverityNumber.WARN,
            "error": SeverityNumber.ERROR,
        }

    def _logger_for(self, component: str):
        """Return (and cache) the OTel logger for an instrumentation scope."""
        if self._provider is None:
            return None
        if component not in self._loggers:
            self._loggers[component] = self._provider.get_logger(component)
        return self._loggers[component]

    def _emit(self, message: str, severity: str, **context):
        """Emit a structured log record via OTLP."""
        if self._provider is None:
            return

        component = context.pop("component", None) or self.service_name
        logger = self._logger_for(component)
        if logger is None:
            return

        # Callers may keep a stable `event_name` while the message body carries
        # richer, human-readable detail (e.g. an error summary).
        event_name = context.pop("event_name", None) or message

        severity_number = self._severity_map.get(severity)
        attrs = _flatten_telemetry_attrs(context)
        attrs["component"] = component
        # OTel semantic convention for a log event name; Elasticsearch maps this
        # to the aggregatable top-level `event_name` keyword field.
        attrs["event.name"] = event_name
        # Attach the application callsite to every record, unless an
        # exception's raise-site ``code.*`` fields were already projected in
        # (errors carry the deepest user frame from the traceback). Uses a fast
        # frame walk, so the cost is acceptable even at DEBUG/INFO.
        if "code.file.path" not in attrs:
            attrs.update(_app_callsite())
        logger.emit(
            severity_number=severity_number,
            severity_text=severity.upper(),
            body=message,
            attributes=attrs or None,
        )

    def debug(self, message: str, **context):
        """Log debug message."""
        self._emit(message, "debug", **context)

    def info(self, message: str, **context):
        """Log info message."""
        self._emit(message, "info", **context)

    def warning(self, message: str, **context):
        """Log warning message."""
        self._emit(message, "warning", **context)

    def error(self, message: str, **context):
        """Log error message."""
        self._emit(message, "error", **context)

    def close(self):
        """Clear adapter state without shutting down the app-owned provider."""
        self._loggers = {}

    def shutdown(self):
        """Do not shut down the application-owned provider."""
        self.close()


class CompositeLogger:
    """Composite logger that writes to multiple backends."""

    def __init__(self, *loggers: Logger):
        """
        Initialize composite logger.

        Args:
            *loggers: Logger instances to compose
        """
        self.loggers = loggers

    def debug(self, message: str, **context):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.debug(message, **context)

    def info(self, message: str, **context):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.info(message, **context)

    def warning(self, message: str, **context):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.warning(message, **context)

    def error(self, message: str, **context):
        """Log to all loggers."""
        for logger in self.loggers:
            logger.error(message, **context)
