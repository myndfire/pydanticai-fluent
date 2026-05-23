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
from typing import Protocol, Any
from datetime import datetime, date
import structlog


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
            # Configure structlog (simplified)
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
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
        """Log info message to file."""
        self.logger.info(message, **context)

    def warning(self, message: str, **context):
        """Log warning message to file."""
        self.logger.warning(message, **context)

    def error(self, message: str, **context):
        """Log error message to file."""
        self.logger.error(message, **context)


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
