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

"""OpenTelemetry-only structured logging.

Every log record is emitted through OpenTelemetry (OTLP export to the
collector) so the telemetry pipeline has a single egress. Local console output
is rendered by the OTel ``ConsoleLogExporter`` rather than a separate
structlog ``PrintLogger`` backend.

Existing ``structlog.get_logger()`` call sites keep working: they are bridged
to OTel by :func:`configure_structlog_otel_bridge`, which routes structlog
events into the active :class:`OTELLogger`.
"""

import contextvars
import json
import math
import os
import socket
import sys
import sysconfig
from collections.abc import Mapping
from typing import Protocol, Any

import structlog


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
    the callsite.

    Returns
    -------
    dict
        ``code.file.path`` / ``code.function`` / ``code.line.number`` /
        ``code.namespace`` or an empty dict if no suitable frame is found.
    """
    import inspect

    for frame_info in inspect.stack():
        filename = frame_info.filename
        if not _is_harness_or_internal_frame(filename):
            return {
                "code.file.path": os.path.relpath(filename),
                "code.function": frame_info.function,
                "code.line.number": frame_info.lineno,
                "code.namespace": frame_info.frame.f_globals.get("__name__", ""),
            }
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


# ── structlog → OTel bridge ────────────────────────────────────────────
#
# structlog remains the authoring API used across the examples, but all
# records are funneled through the active OTELLogger so the only egress is
# OTLP. ``ReturnLoggerFactory`` prevents structlog from rendering/printing on
# its own.
class _StructlogBridge:
    """Routes structlog events into the active OTel logger."""

    def __init__(self) -> None:
        self._active: "OTELLogger | None" = None
        self._fallback: "OTELLogger | None" = None
        self._configured = False

    def set_active(self, logger: "OTELLogger") -> None:
        """Make ``logger`` the sink for structlog events and configure structlog."""
        self._active = logger
        self.configure()

    def resolve(self) -> "OTELLogger | None":
        """Return the active logger, lazily creating a default if needed.

        The fallback lets ``structlog`` calls emitted before the harness builds
        its own ``Observability`` still reach OTel (and the console exporter)
        instead of a non-OTel print backend.
        """
        if self._active is not None:
            return self._active
        if self._fallback is None:
            try:
                console = os.getenv("HARNESS_TELEMETRY_CONSOLE", "true").lower() not in (
                    "0",
                    "false",
                    "no",
                )
                self._fallback = OTELLogger(
                    service_name=os.getenv("OBSERVABILITY_SERVICE_NAME", "agent"),
                    otlp_endpoint=os.getenv(
                        "OTEL_COLLECTOR_ENDPOINT", "localhost:4317"
                    ),
                    environment=os.getenv("APP_ENV", "development"),
                    host=socket.gethostname(),
                    console=console,
                )
            except Exception:
                self._fallback = None
        return self._fallback

    def process(self, logger, method_name: str, event_dict: dict) -> dict:
        """structlog processor that forwards the event to the active logger."""
        event = event_dict.pop("event", "")
        active = self.resolve()
        if active is not None:
            level = (
                method_name
                if method_name in ("debug", "info", "warning", "error")
                else "info"
            )
            context = {
                k: v
                for k, v in event_dict.items()
                if k not in ("level", "timestamp")
            }
            active._emit(str(event), level, **context)
        return event_dict

    def configure(self) -> None:
        """Install the bridge processors once (idempotent)."""
        if self._configured:
            return
        try:
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    self.process,
                ],
                context_class=dict,
                logger_factory=structlog.ReturnLoggerFactory(),
                cache_logger_on_first_use=False,
            )
            self._configured = True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️  Failed to configure structlog→OTel bridge: {exc}")


_BRIDGE = _StructlogBridge()


def configure_structlog_otel_bridge() -> None:
    """Configure structlog to emit through OpenTelemetry (idempotent)."""
    _BRIDGE.configure()


class OTELLogger:
    """OpenTelemetry structured logging via OTLP gRPC export.

    Sends log records to an OTel Collector (or any OTLP endpoint). Records
    emitted inside an active span automatically carry trace_id/span_id for
    log-trace correlation. Optionally also renders records to the local
    console through the OTel ``ConsoleLogExporter``.

    Records are emitted under an instrumentation-scope named after the
    ``component`` attribute (falling back to ``service_name``), so backends can
    group by subsystem.
    """

    def __init__(
        self,
        service_name: str = "agent",
        otlp_endpoint: str = "localhost:4317",
        headers: dict[str, str] | None = None,
        runtime: Any = None,
        flush_on_exit: bool = True,
        shutdown_on_exit: bool = True,
        environment: str = "development",
        host: str | None = None,
        console: bool = False,
        telemetry_level: str = "standard",
    ):
        """
        Initialize OTEL logging.

        Args:
            service_name: Service name for log records
            otlp_endpoint: OTel Collector OTLP gRPC endpoint (default: localhost:4317)
            flush_on_exit: Register an atexit handler that calls
                ``force_flush()`` on the LoggerProvider before exit (default True).
            shutdown_on_exit: Register an atexit handler that calls
                ``shutdown()`` on the LoggerProvider (default True). Implies
                ``flush_on_exit``.
            environment: Deployment environment, exported once as the
                ``deployment.environment`` resource attribute.
            host: Hostname, exported once as the ``host.name`` resource attribute.
            console: Also render records to the local console via the OTel
                ``ConsoleLogExporter``.
            telemetry_level: Granularity level exported as the
                ``harness.telemetry.level`` resource attribute.
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.headers = headers or {}
        self.runtime = runtime
        self.environment = environment
        self.host = host
        self.console = console
        self.telemetry_level = telemetry_level
        self._flush_on_exit = flush_on_exit or shutdown_on_exit
        self._shutdown_on_exit = shutdown_on_exit
        self._provider = None
        self._loggers: dict[str, Any] = {}
        self._shut_down = False

        self._setup_otlp()
        _BRIDGE.set_active(self)

    def _setup_otlp(self):
        """Setup OTLP log exporter."""
        try:
            from opentelemetry._logs import SeverityNumber
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk._logs.export import (
                BatchLogRecordProcessor,
                ConsoleLogExporter,
                SimpleLogRecordProcessor,
            )
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )

            from ._otel import build_resource, register_atexit

            resource = self.runtime.resource if self.runtime else build_resource(
                self.service_name,
                environment=self.environment,
                host=self.host,
                telemetry_level=self.telemetry_level,
                extra={"service.version": os.getenv("SERVICE_VERSION", "0.1.0")},
            )

            exporter = OTLPLogExporter(
                endpoint=self.otlp_endpoint,
                headers=self.headers or None,
                insecure=True,
            )
            self._provider = LoggerProvider(resource=resource)
            self._provider.add_log_record_processor(
                BatchLogRecordProcessor(exporter)
            )
            if self.console:
                self._provider.add_log_record_processor(
                    SimpleLogRecordProcessor(ConsoleLogExporter())
                )

            self._severity_map = {
                "debug": SeverityNumber.DEBUG,
                "info": SeverityNumber.INFO,
                "warning": SeverityNumber.WARN,
                "error": SeverityNumber.ERROR,
            }

            print(f"✅ OTEL logging initialized: {self.otlp_endpoint}")

            if self.runtime:
                self.runtime.register(self._provider)
            else:
                register_atexit(
                    self._provider,
                    flush_on_exit=self._flush_on_exit,
                    shutdown_on_exit=self._shutdown_on_exit,
                    is_shut_down=lambda: self._shut_down,
                )

        except Exception as e:
            print(f"⚠️  Failed to setup OTEL logging: {str(e)}")
            self._provider = None

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
        # Attach the application callsite only for actionable severities, and
        # never clobber an exception's raise-site `code.*` fields. This keeps
        # info-level records lean and avoids the per-emit `inspect.stack()` cost.
        if severity in ("warning", "error") and "code.file.path" not in attrs:
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
        """Flush and shut down the OTLP log provider."""
        if self._provider:
            try:
                self._provider.force_flush()
                self._provider.shutdown()
            except Exception:
                pass
            self._provider = None
            self._loggers = {}
            self._shut_down = True

    def shutdown(self):
        """Explicitly flush and shut down the OTLP log provider.

        Call this for deterministic cleanup in long-running processes or
        when ``flush_on_exit=False``.
        """
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
