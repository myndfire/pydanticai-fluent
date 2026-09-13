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

"""Distributed tracing over OpenTelemetry (OTLP)."""

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


class OTELTracer:
    """
    Pure OpenTelemetry distributed tracing.

    Use this for direct OTLP export to the collector.

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
        telemetry_level: str = "standard",
        console: bool = False,
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
            telemetry_level: Granularity level; ``verbose`` enables native
                prompt/completion content on spans.
            console: Also render spans to the local console via the OTel
                ``ConsoleSpanExporter``.
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.sample_rate = sample_rate
        self.create_spans = create_spans
        self.record_failures = record_failures
        self.telemetry_level = telemetry_level
        self.console = console
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
            from pydantic_ai.models.instrumented import InstrumentationSettings

            # Prompt/completion content is only attached to native spans at the
            # verbose level; lower levels keep spans lean.
            include_content = self.telemetry_level == "verbose"
            Agent.instrument_all(
                InstrumentationSettings(include_content=include_content)
            )
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
            from opentelemetry.trace import ProxyTracerProvider

            # OpenTelemetry allows only one global TracerProvider per process.
            # Reuse any already-registered provider instead
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
                if self.console:
                    from opentelemetry.sdk.trace.export import (
                        ConsoleSpanExporter,
                        SimpleSpanProcessor,
                    )

                    existing_provider.add_span_processor(
                        SimpleSpanProcessor(ConsoleSpanExporter())
                    )
                self.tracer = trace.get_tracer(__name__)
                self._provider = existing_provider
                print(
                    f"✅ OTEL tracing initialized (reusing existing provider): {self.otlp_endpoint}"
                )
                self._enable_pydanticai_instrumentation()
                # Don't register atexit — the original provider owner already did
                return

            # No existing provider — create one
            from ._otel import build_resource, register_atexit

            resource = build_resource(
                self.service_name,
                telemetry_level=self.telemetry_level,
                extra={"service.version": "0.1.0"},
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
            if self.console:
                from opentelemetry.sdk.trace.export import (
                    ConsoleSpanExporter,
                    SimpleSpanProcessor,
                )

                provider.add_span_processor(
                    SimpleSpanProcessor(ConsoleSpanExporter())
                )

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            self._provider = provider

            self._enable_pydanticai_instrumentation()
            print(f"✅ OTEL tracing initialized: {self.otlp_endpoint}")
            register_atexit(
                provider,
                flush_on_exit=self._flush_on_exit,
                shutdown_on_exit=self._shutdown_on_exit,
                is_shut_down=lambda: self._shut_down,
            )

        except Exception as e:
            print(f"⚠️  Failed to setup OTEL tracing: {str(e)}")
            self.tracer = None

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
    def _extract_source_location(error: Exception) -> dict | None:
        """Extract the raise-site user-code frame from an exception's traceback.

        Walks the full traceback chain and returns the DEEPEST frame that is
        not inside agent_harness, the Python stdlib, or third-party
        site-packages -- i.e. the user-code frame closest to where the
        exception was actually raised. Reuses the same frame filter as
        ``logging._app_callsite()`` so ``code.*`` on spans stays aligned with
        ``_exception_record()`` used for log records.

        Returns None if no suitable frame is found.
        """
        from .logging import app_code_location

        loc = app_code_location(error.__traceback__)
        if not loc:
            return None
        return {
            "file": loc["code.file.path"],
            "function": loc["code.function"],
            "line": loc["code.line.number"],
        }

    @staticmethod
    def _annotate_span_failure(span, error: Exception) -> None:
        """Apply the canonical error schema to an open span (ERROR + exception).

        Uses ``observability.build_error_attributes`` so spans carry the same
        ``error.*`` / ``exception.*`` / ``code.*`` fields as log records.
        ``code.*`` prefers the exception's user frame, falling back to the
        caller recorded at ``ManagedAgent.run()`` (the asyncio task boundary
        otherwise drops it).
        """
        from opentelemetry.trace import Status, StatusCode

        from .errorhandling import ErrorContext
        from .logging import get_harness_call_site
        from .observability import build_error_attributes

        ctx = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            source=getattr(error, "_error_source", "unknown"),
            handled=False,
        )
        attrs = build_error_attributes(
            ctx, exception=error, callsite=get_harness_call_site()
        )
        for key, value in attrs.items():
            span.set_attribute(key, value)

        span.record_exception(error, escaped=True)

        status_msg = str(error)
        file = attrs.get("code.file.path")
        if file:
            status_msg = f"{status_msg} | at {file}:{attrs.get('code.line.number')}"
        span.set_status(Status(StatusCode.ERROR, status_msg))

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


