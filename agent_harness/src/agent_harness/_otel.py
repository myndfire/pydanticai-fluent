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

"""Small shared helpers for the harness's OpenTelemetry providers.

Logs, traces, and metrics each build a ``Resource`` and register an atexit
cleanup handler. Centralizing that logic keeps the provider classes focused and
avoids repeating the SDK's "unregister its own atexit handler" workaround.
"""

from typing import Callable, Optional


def build_resource(
    service_name: str,
    *,
    environment: Optional[str] = None,
    host: Optional[str] = None,
    telemetry_level: Optional[str] = None,
    extra: Optional[dict] = None,
):
    """Build an OTel ``Resource`` with the harness's standard attributes."""
    from opentelemetry.sdk.resources import Resource

    attributes = {"service.name": service_name}
    if environment:
        attributes["deployment.environment"] = environment
    if host:
        attributes["host.name"] = host
    if telemetry_level:
        attributes["harness.telemetry.level"] = telemetry_level
    if extra:
        attributes.update(extra)
    return Resource.create(attributes)


def register_atexit(
    provider,
    *,
    flush_on_exit: bool = True,
    shutdown_on_exit: bool = True,
    is_shut_down: Optional[Callable[[], bool]] = None,
) -> None:
    """Register a flush/shutdown handler for a provider at interpreter exit.

    The OTel SDK registers its own atexit handler at construction; we unregister
    it so shutdown is not attempted twice.
    """
    if not (flush_on_exit or shutdown_on_exit):
        return

    import atexit

    for attr in ("_at_exit_handler", "_atexit_handler"):
        handler = getattr(provider, attr, None)
        if handler is not None:
            atexit.unregister(handler)
            setattr(provider, attr, None)

    def _cleanup() -> None:
        try:
            if is_shut_down is not None and is_shut_down():
                return
            if shutdown_on_exit:
                provider.shutdown()
            else:
                provider.force_flush()
        except Exception:
            pass

    atexit.register(_cleanup)
