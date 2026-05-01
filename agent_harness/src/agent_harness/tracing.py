"""Distributed tracing with Logfire (default) and OpenTelemetry."""

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

            # If a TracerProvider already exists, Logfire will warn but still work
            # Suppress the warning since we expect this in multi-example runs
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Overriding of current TracerProvider"
                )
                logfire.configure(**config_kwargs)

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
    """

    def __init__(
        self,
        service_name: str,
        otlp_endpoint: str = "http://localhost:4317",
        sample_rate: float = 1.0,
    ):
        """
        Initialize OTEL tracer.

        Args:
            service_name: Service name for traces
            otlp_endpoint: OTLP collector endpoint (gRPC)
            sample_rate: Sampling rate (0.0 to 1.0, default 1.0 = trace everything)
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.sample_rate = sample_rate
        self.tracer = None

        self._setup_otel()

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

            # Check if a TracerProvider is already set
            existing_provider = trace.get_tracer_provider()
            if isinstance(existing_provider, TracerProvider):
                # Reuse existing provider, just add our exporter
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint, insecure=True, timeout=5
                )
                processor = BatchSpanProcessor(otlp_exporter)
                existing_provider.add_span_processor(processor)
                self.tracer = trace.get_tracer(__name__)
                print(
                    f"✅ OTEL tracing initialized (reusing existing provider): {self.otlp_endpoint}"
                )
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
            processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(processor)

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)

            print(f"✅ OTEL tracing initialized: {self.otlp_endpoint}")

        except Exception as e:
            print(f"⚠️  Failed to setup OTEL tracing: {str(e)}")
            self.tracer = None

    @asynccontextmanager
    async def span(self, name: str, **attributes):
        """Create an OTEL span."""
        if not self.tracer:
            yield None
            return

        from opentelemetry.trace import Status, StatusCode

        # Start span
        span = self.tracer.start_span(f"{self.service_name}.{name}")
        span_context = span.get_span_context()

        # Add attributes
        for key, value in attributes.items():
            span.set_attribute(key, str(value))

        try:
            yield span_context
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        finally:
            span.end()

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
