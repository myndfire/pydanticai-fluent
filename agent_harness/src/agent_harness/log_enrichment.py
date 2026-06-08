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

"""Pluggable log enrichment — attach structured context to every log entry."""

import os
import socket
from dataclasses import dataclass, field
from typing import Protocol


class LogEnrichmentProvider(Protocol):
    """Any object that supplies key-value pairs to enrich log context.

    Implementations can read from env vars, OTel baggage, async contextvars,
    static configuration, or any other source.
    """

    def enrich(self) -> dict[str, str]:
        """Return key-value pairs to merge into log context."""
        ...


@dataclass
class LogContext:
    """Fluent key-value store for log enrichment.

    Usage:
        LogContext().with_("pipeline", "qa").with_many(stage="research", version="1.0")
    """

    _data: dict[str, str] = field(default_factory=dict)

    def with_(self, key: str, value: str) -> "LogContext":
        """Add one key-value pair."""
        self._data[key] = value
        return self

    def with_many(self, **kwargs: str) -> "LogContext":
        """Add multiple key-value pairs at once."""
        self._data.update(kwargs)
        return self

    def merge(self, other: "LogContext") -> "LogContext":
        """Merge another LogContext into this one (other wins on conflict)."""
        self._data.update(other._data)
        return self

    def enrich(self) -> dict[str, str]:
        """LogEnrichmentProvider protocol — return the data."""
        return dict(self._data)

    def as_dict(self) -> dict[str, str]:
        """Export as a plain dict."""
        return dict(self._data)


class EnvEnricher:
    """Auto-attach host, environment, and runtime metadata to every log."""

    def __init__(self):
        self._host = socket.gethostname()
        self._env = os.getenv("APP_ENV", "local")
        self._pid = str(os.getpid())

    def enrich(self) -> dict[str, str]:
        return {
            "host": self._host,
            "env": self._env,
            "pid": self._pid,
        }
