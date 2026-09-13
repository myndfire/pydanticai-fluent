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

"""Single construction path for every PydanticAI ``Agent`` the harness creates.

All harness-constructed agents -- the ``ManagedAgent``'s own agent, the
``QualityCheck`` LLM judge, and the guard fallback agent -- go through
:func:`build_harness_agent` so they consistently carry the span-logging
capability that correlates ``chat``/``invoke_agent`` spans to logs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from pydantic_ai import Agent

from ._span_logs import build_span_logging_capability

__all__ = ["build_harness_agent"]


def build_harness_agent(
    model: Any,
    *,
    observability_getter: Callable[[], Any],
    capabilities: Sequence[Any] = (),
    **kwargs: Any,
) -> Agent[Any, Any]:
    """Construct a PydanticAI ``Agent`` with the harness span-logging capability.

    Args:
        model: Model instance or identifier passed straight to ``Agent``.
        observability_getter: Zero-arg callable returning the harness
            ``Observability`` (or ``None``); read lazily so a stack attached
            after construction is still seen.
        capabilities: Extra capabilities to attach (compaction, persistence,
            ...). The span-logging capability is appended after these.
        **kwargs: Remaining ``Agent`` constructor arguments.

    Returns:
        The configured ``Agent``.
    """
    return Agent(
        model=model,
        capabilities=[*capabilities, build_span_logging_capability(observability_getter)],
        **kwargs,
    )
