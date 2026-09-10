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

"""Compaction builders (Phase 1 basics).

Single responsibility: build upstream ``pydantic-ai-harness`` compaction
capabilities. No state, no subclassing (composition only).

Each strategy has two helpers: an explicit builder (all values passed by the
caller) and a ``*_from_env`` variant that reads the same values from
``HARNESS_COMPACTION_*`` environment variables. No tuning numbers live in
this file — missing env keys mean "trigger disabled" (``None``) or the
upstream default for keep_* settings.
"""

import os
from typing import Any, Awaitable, Callable, Optional, Sequence, Union

from pydantic_ai_harness import (
    ClearToolResults,
    ReportContextUsage,
    SlidingWindowCompaction,
    TieredCompaction,
    WarnNearLimits,
)
from pydantic_ai_harness.compaction import ContextUsage

__all__ = [
    # Re-exports (escape hatch — use upstream classes directly)
    "ClearToolResults",
    "SlidingWindowCompaction",
    "TieredCompaction",
    "WarnNearLimits",
    "ReportContextUsage",
    "ContextUsage",
    # Explicit builders
    "make_clear_tool_results",
    "make_sliding_window",
    "make_warn_near_limits",
    "make_report_context_usage",
    "make_tiered",
    # Env builders
    "clear_tool_results_from_env",
    "sliding_window_from_env",
    "warn_near_limits_from_env",
    "report_context_usage_from_env",
    "tiered_from_env",
]

# Fallbacks used ONLY by *_from_env when the env key is unset.
# Each matches the upstream pydantic-ai-harness default for that parameter,
# so unset env == upstream default behaviour. Set the env key to override.
_FALLBACK_KEEP_PAIRS = 3
_FALLBACK_KEEP_MESSAGES = 40
_FALLBACK_WARNING_THRESHOLD = 0.7
_FALLBACK_CONTEXT_WINDOW_TOKENS = 200_000


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    return float(raw) if raw else None


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    return int(raw) if raw else None


def make_clear_tool_results(
    keep_pairs: int,
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_fraction: Optional[float] = None,
    exclude_tools: frozenset[str] = frozenset(),
) -> ClearToolResults:
    """Build a ClearToolResults capability from explicit values."""
    return ClearToolResults(
        max_messages=max_messages,
        max_tokens=max_tokens,
        max_fraction=max_fraction,
        keep_pairs=keep_pairs,
        exclude_tools=exclude_tools,
    )


def clear_tool_results_from_env() -> ClearToolResults:
    """Build ClearToolResults from ``HARNESS_COMPACTION_*`` env vars."""
    keep_pairs = _env_int("HARNESS_COMPACTION_KEEP_PAIRS") or _FALLBACK_KEEP_PAIRS
    return make_clear_tool_results(
        max_messages=_env_int("HARNESS_COMPACTION_MAX_MESSAGES"),
        max_tokens=_env_int("HARNESS_COMPACTION_MAX_TOKENS"),
        max_fraction=_env_float("HARNESS_COMPACTION_MAX_FRACTION"),
        keep_pairs=keep_pairs,
    )


def make_sliding_window(
    keep_messages: int,
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_fraction: Optional[float] = None,
) -> SlidingWindowCompaction:
    """Build a SlidingWindowCompaction capability from explicit values."""
    return SlidingWindowCompaction(
        max_messages=max_messages,
        max_tokens=max_tokens,
        max_fraction=max_fraction,
        keep_messages=keep_messages,
    )


def sliding_window_from_env() -> SlidingWindowCompaction:
    """Build SlidingWindowCompaction from ``HARNESS_COMPACTION_*`` env vars."""
    keep_messages = _env_int("HARNESS_COMPACTION_KEEP_MESSAGES") or _FALLBACK_KEEP_MESSAGES
    return make_sliding_window(
        max_messages=_env_int("HARNESS_COMPACTION_MAX_MESSAGES"),
        max_tokens=_env_int("HARNESS_COMPACTION_MAX_TOKENS"),
        max_fraction=_env_float("HARNESS_COMPACTION_MAX_FRACTION"),
        keep_messages=keep_messages,
    )


def make_warn_near_limits(
    warning_threshold: float,
    max_iterations: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    max_context_fraction: Optional[float] = None,
) -> WarnNearLimits:
    """Build a WarnNearLimits capability from explicit values."""
    return WarnNearLimits(
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        max_context_fraction=max_context_fraction,
        warning_threshold=warning_threshold,
    )


def warn_near_limits_from_env() -> WarnNearLimits:
    """Build WarnNearLimits from ``HARNESS_COMPACTION_*`` env vars."""
    threshold = _env_float("HARNESS_COMPACTION_WARN_FRACTION") or _FALLBACK_WARNING_THRESHOLD
    return make_warn_near_limits(
        max_iterations=_env_int("HARNESS_COMPACTION_MAX_ITERATIONS"),
        max_context_tokens=_env_int("HARNESS_COMPACTION_MAX_TOKENS"),
        max_context_fraction=_env_float("HARNESS_COMPACTION_MAX_FRACTION"),
        warning_threshold=threshold,
    )


OnUsage = Union[
    Callable[[ContextUsage], None],
    Callable[[ContextUsage], Awaitable[None]],
]


def make_report_context_usage(on_usage: OnUsage) -> ReportContextUsage:
    """Build a ReportContextUsage capability around an ``on_usage`` callback."""
    return ReportContextUsage(on_usage=on_usage)


def report_context_usage_from_env(on_usage: OnUsage) -> ReportContextUsage:
    """Build ReportContextUsage; thresholds come from env, callback is explicit."""
    window = _env_int("HARNESS_COMPACTION_CONTEXT_WINDOW")
    fallback = _env_int("HARNESS_COMPACTION_FALLBACK_WINDOW") or _FALLBACK_CONTEXT_WINDOW_TOKENS
    if window is not None:
        return ReportContextUsage(
            on_usage=on_usage, context_window=window,
            fallback_context_window=fallback,
        )
    return ReportContextUsage(on_usage=on_usage, fallback_context_window=fallback)


def make_tiered(
    tiers: Sequence[Any],
    target_tokens: Optional[int] = None,
    target_fraction: Optional[float] = None,
) -> TieredCompaction:
    """Build the recommended TieredCompaction from explicit values + tiers."""
    return TieredCompaction(
        tiers=tiers,
        target_tokens=target_tokens,
        target_fraction=target_fraction,
    )


def tiered_from_env(tiers: Sequence[Any]) -> TieredCompaction:
    """Build TieredCompaction from ``HARNESS_COMPACTION_*`` env vars."""
    return make_tiered(
        tiers=tiers,
        target_tokens=_env_int("HARNESS_COMPACTION_TARGET_TOKENS"),
        target_fraction=_env_float("HARNESS_COMPACTION_TARGET_FRACTION"),
    )
