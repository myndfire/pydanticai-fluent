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

"""Step-persistence builders.

Single responsibility: build the upstream ``pydantic-ai-harness``
``StepPersistence`` capability and its stores. No state, no subclassing
(composition only).

Each backend has two helpers: an explicit builder (all values passed by the
caller) and a ``*_from_env`` variant that reads the same values from
``HARNESS_PERSISTENCE_*`` environment variables. No tuning numbers live in
this file — the only fallbacks are named constants matching the upstream
defaults, used when an env key is unset.
"""

import os
from typing import Any, Optional

from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import (
    FileStepStore,
    InMemoryStepStore,
    MongoStepStore,
    SqliteStepStore,
    annotate_tool_effect,
    continue_run,
    fork_run,
)

__all__ = [
    # Re-exports (escape hatch + resume helpers — use upstream directly)
    "StepPersistence",
    "InMemoryStepStore",
    "FileStepStore",
    "SqliteStepStore",
    "MongoStepStore",
    "continue_run",
    "fork_run",
    "annotate_tool_effect",
    # Explicit builders
    "make_memory_store",
    "make_file_store",
    "make_sqlite_store",
    "make_mongo_store",
    "make_step_persistence",
    # Env builders
    "memory_store_from_env",
    "file_store_from_env",
    "sqlite_store_from_env",
    "mongo_store_from_env",
    "step_persistence_from_env",
]

# Fallbacks used ONLY by *_from_env when the env key is unset.
# Each matches the upstream pydantic-ai-harness default, so unset env ==
# upstream default behaviour. Set the env key to override.
_FALLBACK_BACKEND = "memory"
_FALLBACK_FILE_DIRECTORY = ".step-persistence"
_FALLBACK_SQLITE_DATABASE = ".step-persistence.db"
# Local Mongo default, matching this repo's docker-compose/examples convention
# (MONGODB_URI=mongodb://localhost:27017). Upstream requires exactly one of
# client=/db_url=, so env-missing means local rather than an error.
_FALLBACK_MONGO_URL = "mongodb://localhost:27017"


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    return int(raw) if raw else None


def make_memory_store(
    max_snapshots_per_run: Optional[int] = None,
) -> InMemoryStepStore:
    """Build a process-local step store (great for tests)."""
    return InMemoryStepStore(max_snapshots_per_run=max_snapshots_per_run)


def memory_store_from_env() -> InMemoryStepStore:
    """Build a memory store; retention bound from env (unset = unbounded)."""
    return make_memory_store(
        max_snapshots_per_run=_env_int("HARNESS_PERSISTENCE_MAX_SNAPSHOTS_PER_RUN"),
    )


def make_file_store(
    directory: str,
    max_snapshots_per_run: Optional[int] = None,
) -> FileStepStore:
    """Build a directory-backed step store."""
    return FileStepStore(directory, max_snapshots_per_run=max_snapshots_per_run)


def file_store_from_env() -> FileStepStore:
    """Build a file store from ``HARNESS_PERSISTENCE_*`` env vars."""
    return make_file_store(
        directory=os.getenv("HARNESS_PERSISTENCE_DIRECTORY")
        or _FALLBACK_FILE_DIRECTORY,
        max_snapshots_per_run=_env_int("HARNESS_PERSISTENCE_MAX_SNAPSHOTS_PER_RUN"),
    )


def make_sqlite_store(
    database: str,
    max_snapshots_per_run: Optional[int] = None,
) -> SqliteStepStore:
    """Build a single-file SQLite step store."""
    return SqliteStepStore(database=database, max_snapshots_per_run=max_snapshots_per_run)


def sqlite_store_from_env() -> SqliteStepStore:
    """Build a SQLite store from ``HARNESS_PERSISTENCE_*`` env vars."""
    return make_sqlite_store(
        database=os.getenv("HARNESS_PERSISTENCE_DATABASE")
        or _FALLBACK_SQLITE_DATABASE,
        max_snapshots_per_run=_env_int("HARNESS_PERSISTENCE_MAX_SNAPSHOTS_PER_RUN"),
    )


def make_mongo_store(
    database: str,
    db_url: Optional[str] = None,
    client: Any = None,
    max_snapshots_per_run: Optional[int] = None,
) -> MongoStepStore:
    """Build a MongoDB step store (needs the ``mongodb`` harness extra)."""
    return MongoStepStore(
        client=client,
        db_url=db_url,
        database=database,
        max_snapshots_per_run=max_snapshots_per_run,
    )


def mongo_store_from_env() -> MongoStepStore:
    """Build a Mongo store from ``HARNESS_PERSISTENCE_*`` env vars."""
    database = os.getenv("HARNESS_PERSISTENCE_MONGO_DATABASE", "agent_runs")
    return make_mongo_store(
        database=database,
        db_url=os.getenv("HARNESS_PERSISTENCE_MONGO_URL") or _FALLBACK_MONGO_URL,
        max_snapshots_per_run=_env_int("HARNESS_PERSISTENCE_MAX_SNAPSHOTS_PER_RUN"),
    )


def make_step_persistence(
    store: Any,
    agent_name: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
) -> StepPersistence:
    """Build a StepPersistence capability around an explicit store.

    ``run_id`` is intentionally not exposed: an explicit id reused across
    ``.run()`` calls raises ``ValueError`` upstream (the tool-effect ledger
    keys on ``(run_id, tool_call_id)``). Leave it unset so each run derives a
    distinct id from ``(agent_name, ctx.run_id)``.
    """
    return StepPersistence(
        store=store, agent_name=agent_name, metadata=metadata or {},
    )


def step_persistence_from_env(store: Any = None) -> StepPersistence:
    """Build StepPersistence from ``HARNESS_PERSISTENCE_*`` env vars.

    The backend comes from ``HARNESS_PERSISTENCE_BACKEND``
    (``memory`` | ``file`` | ``sqlite`` | ``mongo``); pass an explicit
    ``store`` to skip backend selection. Unknown backend values raise
    ``ValueError`` rather than silently falling back to memory.
    """
    if store is None:
        backend = (
            os.getenv("HARNESS_PERSISTENCE_BACKEND") or _FALLBACK_BACKEND
        ).lower()
        if backend == "memory":
            store = memory_store_from_env()
        elif backend == "file":
            store = file_store_from_env()
        elif backend == "sqlite":
            store = sqlite_store_from_env()
        elif backend == "mongo":
            store = mongo_store_from_env()
        else:
            raise ValueError(
                f"unknown HARNESS_PERSISTENCE_BACKEND {backend!r}; "
                "expected `memory`, `file`, `sqlite` or `mongo`"
            )
    return make_step_persistence(
        store=store,
        agent_name=os.getenv("HARNESS_PERSISTENCE_AGENT_NAME"),
    )
