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

"""Distributed tracing with Logfire (default) and OpenTelemetry."""

import os
from pathlib import Path
from typing import Protocol, Any
from contextlib import asynccontextmanager

# Load .env from common locations before other imports
_env_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent.parent / ".env",
    Path.cwd().parent / ".env",
]
for env_path in _env_paths:
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        break

# Module-level guard: prevent Agent.instrument_all(True) from being called
# multiple times across OTELTracer instances. Each call registers atexit
# handlers on the global TracerProvider, causing "shutdown can only be called
# once" warnings.
_instrumentation_enabled = False


class Tracer(Protocol):
    """Protocol for distributed tracing."""

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """Create a tracing span."""
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        ...

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current span."""
        ...

    def debug(self, message: str, **context) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **context) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **context) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **context) -> None:
        """Log error message."""
        ...


class InMemoryTracer:
    """In-memory tracer that records spans for display (for development/testing)."""

    def __init__(self):
        """Initialize in-memory storage for spans."""
        self._spans = []

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """Record a span with name and attributes."""
        span_record = {
            "name": name,
            "attributes": attributes,
        }
        self._spans.append(span_record)
        yield span_record

    def get_spans(self) -> list[dict]:
        """Get all recorded spans."""
        return list(self._spans)

    def reset(self):
        """Reset all recorded spans."""
        self._spans.clear()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, **attributes) -> None:
        pass

    def debug(self, message: str, **context) -> None:
        pass

    def info(self, message: str, **context) -> None:
        pass

    def warning(self, message: str, **context) -> None:
        pass

    def error(self, message: str, **context) -> None:
        pass


class NoOpTracer:
    """No-op tracer (minimal overhead)."""

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """No-op span - does nothing."""
        yield None

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, **attributes) -> None:
        pass

    def debug(self, message: str, **context) -> None:
        pass

    def info(self, message: str, **context) -> None:
        pass

    def warning(self, message: str, **context) -> None:
        pass

    def error(self, message: str, **context) -> None:
        pass


class LogfireTracer:
    """
    Logfire distributed tracing (default).

    Logfire is the official observability platform for PydanticAI,
    built by the Pydantic team. It provides:
    - Automatic PydanticAI instrumentation
    - Beautiful trace visualization

    For multi-destination tracing (e.g., Logfire + Jaeger), chain
    LogfireTracer and OTELTracer using with_observability_chained().
    """

    def __init__(
        self,
        service_name: str,
        send_to_logfire: bool = True,
        instrument_pydantic_ai: bool = True,
    ):
        """
        Initialize Logfire tracer.

        Args:
            service_name: Service name for traces
            send_to_logfire: Send traces to Logfire cloud (default: True)
            instrument_pydantic_ai: Automatically instrument PydanticAI (default: True)

        Examples:
            # Default: Send to Logfire cloud
            tracer = LogfireTracer("my-agent")

            # Logfire cloud disabled (local only)
            tracer = LogfireTracer("my-agent", send_to_logfire=False)
        """
        self.service_name = service_name
        self.send_to_logfire = send_to_logfire
        self.logfire = None

        self._setup_logfire()

        if instrument_pydantic_ai:
            self._instrument_pydantic_ai()

    def _setup_logfire(self):
        """Setup Logfire."""
        try:
            import logfire
            from opentelemetry import trace

            # Skip if already configured in this process
            if getattr(logfire, "_configured", False):
                self.logfire = logfire
                destination = "Logfire cloud" if self.send_to_logfire else "local only"
                print(f"✅ Logfire tracing initialized (reuse): {destination}")
                return

            config_kwargs = {
                "service_name": self.service_name,
                "send_to_logfire": self.send_to_logfire,
                "console": False,
                "scrubbing": False,
            }

            # If a TracerProvider is already registered (e.g. by OTEL), Logfire
            # attempts to override it and OpenTelemetry logs a
            # "Overriding of current TracerProvider is not allowed" warning via
            # the `logging` module (not `warnings`). Suppress that logger while
            # configuring Logfire, which then attaches to the existing provider.
            import logging

            otel_loggers = [
                logging.getLogger("opentelemetry.trace"),
                logging.getLogger("opentelemetry.metrics"),
                logging.getLogger("opentelemetry.metrics._internal"),
            ]
            saved_levels = [(lg, lg.level) for lg in otel_loggers]
            for lg in otel_loggers:
                lg.setLevel(logging.ERROR)

            try:
                logfire.configure(**config_kwargs)
            finally:
                for lg, level in saved_levels:
                    lg.setLevel(level)

            logfire._configured = True
            self.logfire = logfire

            destination = "Logfire cloud" if self.send_to_logfire else "local only"
            print(f"✅ Logfire tracing initialized: {destination}")

        except Exception as e:
            print(f"⚠️  Failed to setup Logfire: {str(e)}")
            self.logfire = None

    def _instrument_pydantic_ai(self):
        """Automatically instrument PydanticAI."""
        if self.logfire:
            try:
                self.logfire.instrument_pydantic_ai()
                print("✅ PydanticAI instrumentation enabled")
            except Exception as e:
                print(f"⚠️  Failed to instrument PydanticAI: {str(e)}")

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """
        Create a Logfire span.

        Args:
            name: Span name (e.g., "agent_run", "tool_call")
            **attributes: Span attributes as key-value pairs

        Yields:
            Logfire span object

        Example:
            async with tracer.span("agent_run", session_id="123", model="gpt-4"):
                result = await agent.run(prompt)
        """
        if not self.logfire:
            yield None
            return

        with self.logfire.span(f"{self.service_name}.{name}", **attributes) as span:
            try:
                yield span
            except Exception as e:
                # Logfire automatically captures exceptions
                raise

    def debug(self, message: str, **context):
        """Log debug message to Logfire."""
        if self.logfire:
            self.logfire.debug(message, **context)

    def info(self, message: str, **context):
        """Log info message to Logfire."""
        if self.logfire:
            self.logfire.info(message, **context)

    def notice(self, message: str, **context):
        """Log notice message to Logfire."""
        if self.logfire:
            self.logfire.notice(message, **context)

    def warning(self, message: str, **context):
        """Log warning message to Logfire."""
        if self.logfire:
            self.logfire.warning(message, **context)

    def error(self, message: str, **context):
        """Log error message to Logfire."""
        if self.logfire:
            self.logfire.error(message, **context)

    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        # Logfire handles this automatically in the span context
        pass

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current span."""
        # Logfire handles events automatically via span context
        pass


class OTELTracer:
    """
    Pure OpenTelemetry distributed tracing (without Logfire).

    Use this if you want direct OTLP export without Logfire.

    By default (``create_spans=False``) this tracer does NOT create its own
    harness spans. It only configures the global OTLP provider and lets
    PydanticAI's native instrumentation emit the canonical span tree
    (``invoke_agent <name>``, ``execute_tool <tool>``, ``chat <model>``).
    ``Observability.span()`` then yields the current span's context (or
    ``None``) so in-run log records still correlate with that tree.

    With ``record_failures=True`` (the default), failures thrown out of a
    ``span()`` block are still surfaced in the trace stream. If a recording
    span is current it is marked ERROR and records the exception; otherwise a
    harness-owned failure span ``<service>.<operation>:failed`` is emitted with
    ``status=ERROR``, ``error.type``, ``error.source`` and the exception event.
    The ``:failed`` suffix only exists to distinguish/query these harness spans
    — the failure semantics come from the standard OTel status + exception
    event. Successes never produce a harness span in this mode.

    Set ``create_spans=True`` to restore the legacy behavior of exporting a
    harness-managed ``<service>.<name>`` span for every ``Observability.span()``
    call (e.g. to explicitly demo manual OTel spans).
    """

    def __init__(
        self,
        service_name: str,
        otlp_endpoint: str = "localhost:4317",
        sample_rate: float = 1.0,
        create_spans: bool = False,
        record_failures: bool = True,
        export_interval_ms: int = 5000,
        flush_on_exit: bool = True,
        shutdown_on_exit: bool = True,
    ):
        """
        Initialize OTEL tracer.

        Args:
            service_name: Service name for traces
            otlp_endpoint: OTLP collector endpoint (gRPC)
            sample_rate: Sampling rate (0.0 to 1.0, default 1.0 = trace everything)
            create_spans: When True, every span() call starts/exports a
                harness span. When False (default), no harness spans are
                created — PydanticAI native spans are the trace content.
            record_failures: When True (default), exceptions escaping span()
                blocks are recorded in the trace (ERROR status + exception event),
                enriching a live span or emitting ``<service>.<operation>:failed``.
            export_interval_ms: BatchSpanProcessor export interval in
                milliseconds (default 5000). Lower values make traces appear
                in the backend faster during development.
            flush_on_exit: Register an atexit handler that calls
                ``force_flush()`` on the TracerProvider before the process
                exits (default True). Ensures buffered spans are sent even
                for short-lived scripts.
            shutdown_on_exit: Register an atexit handler that calls
                ``shutdown()`` on the TracerProvider (default True). Implies
                ``flush_on_exit``.
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.sample_rate = sample_rate
        self.create_spans = create_spans
        self.record_failures = record_failures
        self._export_interval_ms = export_interval_ms
        self._flush_on_exit = flush_on_exit or shutdown_on_exit
        self._shutdown_on_exit = shutdown_on_exit
        self.tracer = None
        self._provider = None
        self._shut_down = False

        self._setup_otel()

    def _enable_pydanticai_instrumentation(self) -> None:
        """Auto-instrument PydanticAI to emit native run/model/tool spans.

        PydanticAI parents its spans to the currently active span, so they nest
        under the harness's ``agent_run`` umbrella and export through the global
        OTLP tracer provider configured by ``OTELTracer``.
        """
        global _instrumentation_enabled
        if _instrumentation_enabled:
            return
        try:
            from pydantic_ai.agent import Agent

            Agent.instrument_all(True)
            _instrumentation_enabled = True
            print("✅ PydanticAI native instrumentation enabled (OTLP)")
        except Exception as e:
            print(f"⚠️  Failed to enable PydanticAI instrumentation: {str(e)}")

    def _setup_otel(self):
        """Setup OpenTelemetry tracing."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.trace import ProxyTracerProvider

            # OpenTelemetry allows only one global TracerProvider per process.
            # Reuse any already-registered provider (including Logfire's) instead
            # of overriding it (which OTEL rejects with
            # "Overriding of current TracerProvider"). We can still attach our
            # OTLP span processor to the existing provider.
            existing_provider = trace.get_tracer_provider()
            if not isinstance(existing_provider, ProxyTracerProvider):
                # Reuse existing provider, just add our exporter
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint, insecure=True, timeout=5
                )
                processor = BatchSpanProcessor(
                    otlp_exporter,
                    schedule_delay_millis=self._export_interval_ms,
                )
                existing_provider.add_span_processor(processor)
                self.tracer = trace.get_tracer(__name__)
                self._provider = existing_provider
                print(
                    f"✅ OTEL tracing initialized (reusing existing provider): {self.otlp_endpoint}"
                )
                self._enable_pydanticai_instrumentation()
                # Don't register atexit — the original provider owner already did
                return

            # No existing provider — create one
            resource = Resource.create(
                {"service.name": self.service_name, "service.version": "0.1.0"}
            )

            sampler = TraceIdRatioBased(self.sample_rate)
            provider = TracerProvider(resource=resource, sampler=sampler)

            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint, insecure=True, timeout=5
            )
            processor = BatchSpanProcessor(
                otlp_exporter,
                schedule_delay_millis=self._export_interval_ms,
            )
            provider.add_span_processor(processor)

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            self._provider = provider

            self._enable_pydanticai_instrumentation()
            print(f"✅ OTEL tracing initialized: {self.otlp_endpoint}")
            self._register_atexit(provider)

        except Exception as e:
            print(f"⚠️  Failed to setup OTEL tracing: {str(e)}")
            self.tracer = None

    def _register_atexit(self, provider):
        """Register atexit handler to flush/shutdown the TracerProvider."""
        if self._flush_on_exit or self._shutdown_on_exit:
            import atexit

            # The SDK's TracerProvider.__init__ registers its own atexit
            # handler that calls provider.shutdown(). Unregister it to avoid
            # "shutdown can only be called once" when our handler also fires.
            if getattr(provider, "_atexit_handler", None) is not None:
                atexit.unregister(provider._atexit_handler)
                provider._atexit_handler = None

            def _cleanup():
                try:
                    if self._shut_down:
                        return
                    if self._shutdown_on_exit:
                        provider.shutdown()
                    elif self._flush_on_exit:
                        provider.force_flush()
                except Exception:
                    pass

            atexit.register(_cleanup)

    def shutdown(self):
        """Explicitly flush and shut down the OTLP trace provider.

        Call this for deterministic cleanup in long-running processes or
        when ``flush_on_exit=False``.
        """
        if self._provider:
            try:
                self._provider.force_flush()
                self._provider.shutdown()
            except Exception:
                pass
            self._provider = None
            self.tracer = None
            self._shut_down = True

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """Create an OTEL span.

        With ``create_spans=False`` (default) no harness span is created;
        the current PydanticAI span's context is yielded (or ``None``) so
        in-run log records still carry its trace id. Failures escaping the
        block are recorded via ``_record_failure`` (see ``record_failures``).
        """
        if not self.tracer:
            yield None
            return

        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import Status, StatusCode

        if not self.create_spans:
            try:
                current = otel_trace.get_current_span()
                span_context = current.get_span_context()
                yield span_context if span_context.is_valid else None
            except Exception as e:
                if self.record_failures:
                    self._record_failure(name, e, **attributes)
                raise
            return

        # Start span and make it the current span so nested spans and
        # OTel log records inherit its trace context.
        span = self.tracer.start_span(f"{self.service_name}.{name}")
        span_context = span.get_span_context()

        # Add attributes
        for key, value in attributes.items():
            span.set_attribute(key, str(value))

        try:
            with otel_trace.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                yield span_context
                span.set_status(Status(StatusCode.OK))

        except Exception as e:
            self._annotate_span_failure(span, e)
            raise

        finally:
            span.end()

    def _record_failure(self, operation: str, error: Exception, **context) -> None:
        """Record a failure escaping a span() block.

        If a recording span is current, enrich it with ERROR status and the
        exception event. Otherwise emit a harness-owned failure span
        ``<service>.<operation>:failed`` carrying ``error.type``/``error.source``
        plus the operation context attributes. ``exception.type/message/stacktrace``
        come only from ``record_exception``.
        """
        if not self.tracer:
            return

        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanKind, Status, StatusCode

        current = otel_trace.get_current_span()
        if current.is_recording():
            current.set_status(Status(StatusCode.ERROR, str(error)))
            current.record_exception(error, escaped=True)
            return

        span = self.tracer.start_span(
            f"{self.service_name}.{operation}:failed", kind=SpanKind.INTERNAL
        )
        self._annotate_span_failure(span, error)
        for key, value in context.items():
            span.set_attribute(key, str(value))
        span.end()

    @staticmethod
    def _annotate_span_failure(span, error: Exception) -> None:
        """Apply standard OTel failure fields to an open span (ERROR + exception)."""
        from opentelemetry.trace import Status, StatusCode

        error_type = type(error)
        error_type_name = (
            f"{error_type.__module__}.{error_type.__qualname__}"
            if error_type.__module__ != "builtins"
            else error_type.__qualname__
        )
        span.set_attribute("error.type", error_type_name)
        span.set_attribute("error.source", getattr(error, "_error_source", "unknown"))
        span.record_exception(error, escaped=True)
        span.set_status(Status(StatusCode.ERROR, str(error)))

    def add_event(self, name: str, **attributes):
        """Add an event to the current span."""
        if self.tracer:
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span."""
        if self.tracer:
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(key, str(value))

    def debug(self, message: str, **context):
        pass

    def info(self, message: str, **context):
        pass

    def warning(self, message: str, **context):
        pass

    def error(self, message: str, **context):
        pass


class JaegerTracer:
    """
    Jaeger distributed tracing (legacy, use LogfireTracer with Jaeger export instead).

    Note: This uses the Jaeger client library directly.
    Consider using LogfireTracer with jaeger_endpoint for better integration.
    """

    def __init__(
        self, service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 6831
    ):
        """
        Initialize Jaeger tracer.

        Args:
            service_name: Service name for traces
            jaeger_host: Jaeger agent host
            jaeger_port: Jaeger agent port (UDP)
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.tracer = None

        self._setup_jaeger()

    def _setup_jaeger(self):
        """Setup Jaeger tracing."""
        try:
            from jaeger_client import Config

            config = Config(
                config={
                    "sampler": {"type": "const", "param": 1},
                    "local_agent": {
                        "reporting_host": self.jaeger_host,
                        "reporting_port": self.jaeger_port,
                    },
                    "logging": True,
                },
                service_name=self.service_name,
                validate=True,
            )

            self.tracer = config.initialize_tracer()
            print(
                f"✅ Jaeger tracing initialized: {self.jaeger_host}:{self.jaeger_port}"
            )

        except Exception as e:
            print(f"⚠️  Failed to setup Jaeger: {str(e)}")
            self.tracer = None

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """Create a Jaeger span."""
        if not self.tracer:
            yield None
            return

        with self.tracer.start_span(f"{self.service_name}.{name}") as span:
            # Add tags (attributes)
            for key, value in attributes.items():
                span.set_tag(key, str(value))

            try:
                yield span
            except Exception as e:
                span.set_tag("error", True)
                span.log_kv({"event": "error", "message": str(e)})
                raise

    def set_attribute(self, key: str, value: Any):
        pass

    def add_event(self, name: str, **attributes):
        pass

    def debug(self, message: str, **context):
        pass

    def info(self, message: str, **context):
        pass

    def warning(self, message: str, **context):
        pass

    def error(self, message: str, **context):
        pass
