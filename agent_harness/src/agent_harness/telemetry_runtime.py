"""Shared lifecycle and resource ownership for the three OTel signals."""

from __future__ import annotations

import atexit
import threading
from typing import Any, Optional

from .telemetry_schema import service_version


class TelemetryRuntime:
    """Own one process-level OTel runtime and all providers created by it.

    The runtime deliberately owns lifecycle. Individual signal adapters must not
    register independent shutdown handlers when they receive a runtime.
    """

    def __init__(
        self,
        service_name: str,
        *,
        environment: Optional[str] = None,
        host: Optional[str] = None,
        telemetry_level: Optional[str] = None,
        version: Optional[str] = None,
        flush_on_exit: bool = True,
        shutdown_on_exit: bool = True,
    ) -> None:
        from opentelemetry.sdk.resources import Resource

        attributes: dict[str, Any] = {
            "service.name": service_name,
            "service.version": version or service_version(),
        }
        if environment:
            attributes["deployment.environment"] = environment
        if host:
            attributes["host.name"] = host
        if telemetry_level:
            attributes["harness.telemetry.level"] = telemetry_level
        self.resource = Resource.create(attributes)
        self._providers: list[Any] = []
        self._closed = False
        self._lock = threading.Lock()
        self._shutdown_on_exit = shutdown_on_exit
        if flush_on_exit or shutdown_on_exit:
            atexit.register(self._cleanup)

    def register(self, provider: Any) -> None:
        """Register a provider for coordinated cleanup exactly once."""
        with self._lock:
            if provider not in self._providers:
                self._providers.append(provider)

    def flush(self) -> None:
        """Flush every provider, tolerating independent exporter failures."""
        for provider in tuple(self._providers):
            try:
                provider.force_flush()
            except Exception:
                continue

    def shutdown(self) -> None:
        """Flush and shut down every provider once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for provider in tuple(self._providers):
            try:
                provider.force_flush()
            except Exception:
                pass
            try:
                provider.shutdown()
            except Exception:
                pass

    def _cleanup(self) -> None:
        if self._shutdown_on_exit:
            self.shutdown()
        else:
            self.flush()
