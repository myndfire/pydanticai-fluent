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

"""Span-correlated log records for PydanticAI's run/model spans.

Why this exists
---------------
Backends such as OpenObserve correlate a trace span to logs with a *span-scoped*
lookup (``span_id='<span>' AND trace_id='<trace>'``). The harness emits logs
inside its own spans (``scenario``/``agent_run``) and inside tool execution, but
never while PydanticAI's ``chat`` (model request) or ``invoke_agent`` (run) spans
are open, so "View Logs" from those spans finds nothing.

How it works
------------
PydanticAI 2.42 opens those spans in its ``Instrumentation`` capability:
``wrap_run`` opens the ``invoke_agent`` span and ``wrap_model_request`` opens the
``chat`` span (see ``pydantic_ai/capabilities/instrumentation.py``). OpenTelemetry
log records inherit the active span context, so a capability ordered
``wrapped_by=[Instrumentation]`` runs *inside* those spans and any log it emits
carries their ``trace_id``/``span_id``. That ordering is the whole point of this
module -- without it the records would attach to the parent harness span.

Errors are logged inside the wrapper (not via ``run_error``/
``model_request_error``): ``on_run_error`` runs after the run span has closed
(``pydantic_ai/agent/__init__.py``), so a record emitted there would lose the
``invoke_agent`` span. Catching, logging and re-raising keeps the record in-span
and never suppresses the original exception.

These records are deliberately compact correlation markers. The rich model data
(model, provider, tokens, cost, and at verbose level the messages) already lives
on the spans as ``gen_ai.*`` attributes; no payloads are logged here. Attribute
names follow the harness convention (nested ``performance.*`` / ``token_usage.*``),
flattened by ``logging._flatten_telemetry_attrs``.

Notes
-----
* ``Hooks`` constructor kwargs map to capability methods: ``run`` -> ``wrap_run``,
  ``model_request`` -> ``wrap_model_request``.
* Output is gated on the observability granularity (``HARNESS_TELEMETRY_LEVEL``):
  ``minimal`` emits nothing, ``standard``/``verbose`` emit one record per run and
  one per model request (including failures).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic_ai.capabilities import CapabilityOrdering, Hooks, Instrumentation
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest

__all__ = ["build_span_logging_capability"]


def _tokens(usage: Any, name: str) -> Any:
    return getattr(usage, name, None) if usage is not None else None


def _agent_name(ctx: Any) -> Any:
    return getattr(getattr(ctx, "agent", None), "name", None)


def _ctx_model_name(ctx: Any) -> Any:
    return getattr(getattr(ctx, "model", None), "model_name", None)


def _run_fields(ctx: Any) -> dict[str, Any]:
    model = _ctx_model_name(ctx)
    provider = getattr(getattr(ctx, "model", None), "system", None)
    return {
        "agent_name": _agent_name(ctx),
        "run_id": getattr(ctx, "run_id", None),
        "model": model,
        "model.requested.name": model,
        "model.requested.provider": provider,
    }


def build_span_logging_capability(
    observability_getter: Callable[[], Any],
) -> Hooks:
    """Build a PydanticAI capability that logs inside run/model spans.

    Args:
        observability_getter: Zero-arg callable returning the harness
            ``Observability`` (or ``None``). Read at call time, not build time,
            so the capability follows a lazily-attached or later-replaced stack.

    Returns:
        A ``Hooks`` capability ordered inside PydanticAI's instrumentation spans.
    """

    def _emitter() -> Any:
        """Return the Observability when logging is enabled, else ``None``."""
        obs = observability_getter()
        if obs is None:
            return None
        if not obs.granularity.at_least("standard"):
            return None
        return obs

    async def _run(ctx: Any, *, handler: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await handler()
        except Exception as exc:
            obs = _emitter()
            if obs is not None:
                obs.log_error(
                    "agent_run",
                    exception=exc,
                    component="agent",
                    status="error",
                    error_source="llm",
                    performance={"duration_seconds": round(time.perf_counter() - start, 4)},
                    **_run_fields(ctx),
                )
            raise
        obs = _emitter()
        if obs is not None:
            obs.log_info(
                "agent_run",
                component="agent",
                status="ok",
                performance={"duration_seconds": round(time.perf_counter() - start, 4)},
                **_run_fields(ctx),
            )
        return result

    async def _model_request(ctx: Any, *, request_context: Any, handler: Any) -> Any:
        start = time.perf_counter()
        try:
            response = await handler(request_context)
        except (ModelRetry, SkipModelRequest):
            # Control flow, not an error: let the graph retry/skip.
            raise
        except Exception as exc:
            obs = _emitter()
            if obs is not None:
                obs.log_error(
                    "model_request",
                    exception=exc,
                    component="model",
                    status="error",
                    error_source="llm",
                    model=_ctx_model_name(ctx),
                    **{
                        "model.requested.name": _ctx_model_name(ctx),
                        "model.requested.provider": getattr(
                            getattr(ctx, "model", None), "system", None
                        ),
                    },
                    performance={"duration_seconds": round(time.perf_counter() - start, 4)},
                )
            raise

        obs = _emitter()
        if obs is not None:
            usage = getattr(response, "usage", None)
            obs.log_info(
                "model_request",
                component="model",
                status="ok",
                model=getattr(response, "model_name", None) or _ctx_model_name(ctx),
                **{
                    "model.requested.name": _ctx_model_name(ctx),
                    "model.requested.provider": getattr(
                        getattr(ctx, "model", None), "system", None
                    ),
                    "model.response.name": getattr(response, "model_name", None),
                    "model.response.provider": getattr(response, "provider_name", None),
                },
                provider=getattr(response, "provider_name", None)
                or getattr(getattr(ctx, "model", None), "system", None),
                finish_reason=getattr(response, "finish_reason", None),
                performance={"duration_seconds": round(time.perf_counter() - start, 4)},
                token_usage={
                    "input_tokens": _tokens(usage, "input_tokens"),
                    "output_tokens": _tokens(usage, "output_tokens"),
                    "total_tokens": _tokens(usage, "total_tokens"),
                    "cache_read_tokens": _tokens(usage, "cache_read_tokens"),
                },
            )
        return response

    return Hooks(
        ordering=CapabilityOrdering(wrapped_by=[Instrumentation]),
        run=_run,
        model_request=_model_request,
    )
