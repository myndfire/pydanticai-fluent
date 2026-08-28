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

"""Structured logging to Elasticsearch and other backends."""

import asyncio
import math
import os
import sysconfig
from typing import Protocol, Any
from datetime import datetime, date
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
    """Return the first non-harness/non-stdlib frame for OTel code.* attributes.

    Walks the stack from the caller of the logger outward, skipping any frame
    that lives inside the agent_harness package, the Python stdlib, or a
    third-party site-package.  The first "user" frame encountered is treated as
    the callsite.

    Returns
    -------
    dict
        ``{"code.file.path": ..., "code.function": ..., "code.line.number": ...}``
        or an empty dict if no suitable frame is found.
    """
    import inspect

    for frame_info in inspect.stack():
        filename = frame_info.filename
        if not _is_harness_or_internal_frame(filename):
            return {
                "code.file.path": os.path.relpath(filename),
                "code.function": frame_info.function,
                "code.line.number": frame_info.lineno,
            }
    return {}


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


class LogfireLogger:
    """Logfire structured logging with automatic tracing integration."""

    def __init__(self, service_name: str = "agent", logfire_instance: Any = None):
        """
        Initialize Logfire logger.

        Args:
            service_name: Service name for log entries
            logfire_instance: Optional Logfire instance (if None, will initialize)
        """
        self.service_name = service_name
        self.logfire = logfire_instance
        self._setup_logfire()
        self._setup_structlog()

    def _setup_logfire(self):
        """Setup Logfire if not already configured."""
        if self.logfire is not None:
            return

        try:
            import logfire

            # Skip if already configured
            if getattr(logfire, "_configured", False):
                self.logfire = logfire
                return

            logfire.configure(
                service_name=self.service_name,
                send_to_logfire=True,
                console=False,
            )
            logfire._configured = True
            self.logfire = logfire
            print(f"✅ Logfire logger initialized")

        except Exception as e:
            print(f"⚠️  Failed to setup Logfire logger: {str(e)}")
            # Fallback to console logger
            self.logfire = None

    def _setup_structlog(self):
        """Configure structlog to use Logfire."""
        try:
            # Configure structlog with automatic call site tracking
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.CallsiteParameterAdder(
                        parameters=[
                            structlog.processors.CallsiteParameter.FUNC_NAME,
                            structlog.processors.CallsiteParameter.PATHNAME,
                            structlog.processors.CallsiteParameter.LINENO,
                        ]
                    ),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
                cache_logger_on_first_use=True,
            )

        except Exception as e:
            print(f"⚠️  Failed to configure structlog: {str(e)}")

    def _log_to_logfire(self, level: str, message: str, context: dict):
        """Send log to Logfire."""
        if not self.logfire:
            return

        try:
            log_method = getattr(self.logfire, level, self.logfire.info)
            log_method(message, **context)
        except Exception as e:
            print(f"⚠️  Failed to log to Logfire: {str(e)}")

    def debug(self, message: str, **context):
        """Log debug message."""
        if self.logfire:
            self._log_to_logfire("debug", message, context)

    def info(self, message: str, **context):
        """Log info message."""
        if self.logfire:
            self._log_to_logfire("info", message, context)
        else:
            # Fallback to console
            import structlog
            logger = structlog.get_logger()
            logger.info(message, **context)

    def warning(self, message: str, **context):
        """Log warning message."""
        if self.logfire:
            self._log_to_logfire("warning", message, context)

    def error(self, message: str, **context):
        """Log error message."""
        if self.logfire:
            self._log_to_logfire("error", message, context)


class ConsoleLogger:
    """Simple console logger (default)."""

    def __init__(self):
        """Initialize console logger."""
        self.logger = structlog.get_logger()

    def debug(self, message: str, **context):
        """Log debug message to console."""
        self.logger.debug(message, **context)

    def info(self, message: str, **context):
        """Log info message to console."""
        self.logger.info(message, **context)

    def warning(self, message: str, **context):
        """Log warning message to console."""
        self.logger.warning(message, **context)

    def error(self, message: str, **context):
        """Log error message to console."""
        self.logger.error(message, **context)


class ElasticsearchLogger:
    """Elasticsearch structured logging with daily indices."""

    def __init__(
        self,
        endpoint: str,
        index_prefix: str = "agent-logs",
        service_name: str = "agent",
    ):
        """
        Initialize Elasticsearch logger.

        Args:
            endpoint: Elasticsearch endpoint URL
            index_prefix: Index prefix (creates daily indices: prefix-YYYY.MM.DD)
            service_name: Service name for log entries
        """
        self.endpoint = endpoint
        self.index_prefix = index_prefix
        self.service_name = service_name
        self.es_client = None
        self.logger = structlog.get_logger()
        self._pending_tasks: list = []

        self._setup_elasticsearch()

    def _setup_elasticsearch(self):
        """Setup Elasticsearch client."""
        try:
            from elasticsearch import AsyncElasticsearch

            self.es_client = AsyncElasticsearch([self.endpoint])
            self.logger.info("Elasticsearch logger initialized", endpoint=self.endpoint)

        except Exception as e:
            self.logger.warning(f"Failed to setup Elasticsearch: {str(e)}")
            self.es_client = None

    def debug(self, message: str, **context):
        """Log debug message."""
        self.logger.debug(message, **context)
        if self.es_client:
            import asyncio

            task = asyncio.create_task(self._log_to_es("debug", message, context))
            self._pending_tasks.append(task)

    def info(self, message: str, **context):
        """Log info message."""
        self.logger.info(message, **context)
        if self.es_client:
            import asyncio

            task = asyncio.create_task(self._log_to_es("info", message, context))
            self._pending_tasks.append(task)

    def warning(self, message: str, **context):
        """Log warning message."""
        self.logger.warning(message, **context)
        if self.es_client:
            import asyncio

            task = asyncio.create_task(self._log_to_es("warning", message, context))
            self._pending_tasks.append(task)

    def error(self, message: str, **context):
        """Log error message."""
        self.logger.error(message, **context)
        if self.es_client:
            import asyncio

            task = asyncio.create_task(self._log_to_es("error", message, context))
            self._pending_tasks.append(task)

    async def _log_to_es(self, level: str, message: str, context: dict):
        """Log to Elasticsearch with daily indices."""
        if not self.es_client:
            return

        try:
            # Create daily index name
            index_name = f"{self.index_prefix}-{date.today():%Y.%m.%d}"

            # Prepare document
            document = {
                "timestamp": datetime.now().isoformat(),
                "service_name": self.service_name,
                "level": level,
                "message": message,
                **context,
            }

            # Suppress Logfire instrumentation for ES calls to avoid noisy "index" spans
            import logfire

            with logfire.suppress_instrumentation():
                await self.es_client.index(index=index_name, document=document)

        except Exception as e:
            # Fail gracefully - don't break application
            # Only log warning once per session to avoid spam
            if not getattr(self, "_connection_error_logged", False):
                self.logger.warning(f"Failed to log to Elasticsearch: {str(e)}")
                self._connection_error_logged = True

    async def close(self):
        """Wait for pending tasks and close Elasticsearch connection."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        if self.es_client:
            await self.es_client.close()


class FileLogger:
    """File-based structured logging."""

    def __init__(
        self, log_file: str = "agent.log", rotation: str = "daily", retention: int = 7
    ):
        """
        Initialize file logger with rotation.

        Args:
            log_file: Log file path
            rotation: Rotation strategy ("daily", "size")
            retention: Days/files to retain
        """
        self.log_file = log_file
        self.rotation = rotation
        self.retention = retention
        self.logger = structlog.get_logger()

        self._setup_file_logger()

    def _setup_file_logger(self):
        """Setup file logging with rotation."""
        try:
            import logging
            from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

            # Create handler based on rotation strategy
            if self.rotation == "daily":
                handler = TimedRotatingFileHandler(
                    self.log_file, when="D", interval=1, backupCount=self.retention
                )
            else:
                handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=self.retention,
                )

            # Configure structlog to use file handler
            logging.basicConfig(handlers=[handler], level=logging.INFO)

            self.logger.info(f"File logger initialized: {self.log_file}")

        except Exception as e:
            self.logger.warning(f"Failed to setup file logger: {str(e)}")

    def debug(self, message: str, **context):
        """Log debug message to file."""
        self.logger.debug(message, **context)

    def info(self, message: str, **context):
        """Log info message to Elasticsearch."""
        self.logger.info(message, **context)

    def warning(self, message: str, **context):
        """Log warning message to Elasticsearch."""
        self.logger.warning(message, **context)

    def error(self, message: str, **context):
        """Log error message to Elasticsearch."""
        self.logger.error(message, **context)


class OTELLogger:
    """OpenTelemetry structured logging via OTLP gRPC export.

    Sends log records to an OTel Collector (or any OTLP endpoint). Records
    emitted inside an active span automatically carry trace_id/span_id for
    log-trace correlation.
    """

    def __init__(
        self,
        service_name: str = "agent",
        otlp_endpoint: str = "localhost:4317",
    ):
        """
        Initialize OTEL logging.

        Args:
            service_name: Service name for log records
            otlp_endpoint: OTel Collector OTLP gRPC endpoint (default: localhost:4317)
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self._provider = None
        self._logger = None

        self._setup_otlp()

    def _setup_otlp(self):
        """Setup OTLP log exporter."""
        try:
            from opentelemetry._logs import SeverityNumber
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})

            exporter = OTLPLogExporter(endpoint=self.otlp_endpoint, insecure=True)
            self._provider = LoggerProvider(resource=resource)
            self._provider.add_log_record_processor(
                BatchLogRecordProcessor(exporter)
            )

            self._logger = self._provider.get_logger(self.service_name)
            self._severity_map = {
                "debug": SeverityNumber.DEBUG,
                "info": SeverityNumber.INFO,
                "warning": SeverityNumber.WARN,
                "error": SeverityNumber.ERROR,
            }

            print(f"✅ OTEL logging initialized: {self.otlp_endpoint}")

        except Exception as e:
            print(f"⚠️  Failed to setup OTEL logging: {str(e)}")
            self._provider = None
            self._logger = None

    def _emit(self, message: str, severity: str, **context):
        """Emit a structured log record via OTLP."""
        if not self._logger:
            return

        severity_number = self._severity_map.get(severity)
        attrs = {k: _normalize_otel_attr(v) for k, v in context.items()}
        attrs.update(_app_callsite())
        self._logger.emit(
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
            self._provider.shutdown()
            self._provider = None
            self._logger = None


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
