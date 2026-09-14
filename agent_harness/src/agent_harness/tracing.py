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

from typing import Protocol, Any
from contextlib import asynccontextmanager


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

    Use this with an application-owned OpenTelemetry provider.

    By default (``create_spans=False``) this tracer does NOT create its own
    harness spans. PydanticAI's per-agent instrumentation emits the canonical span tree
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
        tracer_provider: Any | None = None,
        service_name: str = "agent",
        create_spans: bool = False,
        record_failures: bool = True,
    ):
        """
        Initialize OTEL tracer.

        Args:
            tracer_provider: Application-owned OTel TracerProvider. ``None``
                disables tracing for this adapter.
            service_name: Instrumentation scope name for traces.
            create_spans: When True, every span() call starts/exports a
                harness span. When False (default), no harness spans are
                created — PydanticAI native spans are the trace content.
            record_failures: When True (default), exceptions escaping span()
                blocks are recorded in the trace (ERROR status + exception event),
                enriching a live span or emitting ``<service>.<operation>:failed``.
        """
        self.service_name = service_name
        self._provider = tracer_provider
        self.create_spans = create_spans
        self.record_failures = record_failures
        self.tracer = (
            tracer_provider.get_tracer(service_name)
            if tracer_provider is not None
            else None
        )

    @property
    def provider(self) -> Any | None:
        """Return the application-owned tracer provider, if configured."""
        return self._provider

    def _enable_pydanticai_instrumentation(self) -> None:
        """Deprecated no-op; instrumentation is attached per agent."""
        return

    def shutdown(self):
        """Release this adapter without shutting down the app-owned provider."""
        self._provider = None
        self.tracer = None

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
