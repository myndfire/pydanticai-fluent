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

"""PipelineContext — track multi-agent pipeline stages with auto-logging."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING, Union

from pydantic_ai.messages import UserContent

if TYPE_CHECKING:
    from .agent import ManagedAgent
    from .observability import Observability
    from .memory import MemoryProvider, MessageHistory


@dataclass
class PipelineContext:
    """Tracks pipeline stages, auto-logs results, and prints a full trace.

    Usage:
        ctx = (
            PipelineContext()
            .with_observability(obs)
            .with_memory(memory)
        )

        r1 = await ctx.run_stage(researcher, "Research", "What is X?")
        r2 = await ctx.run_stage(analyst, "Analysis", f"Analyze: {r1.output}")
        ctx.display_trace()
    """

    _stages: list[dict[str, Any]] = field(default_factory=list)
    _observability: Optional[Any] = None  # Observability
    _memory: Optional[Any] = None  # InMemoryProvider / MemoryProvider
    _enrichment: list[Any] = field(default_factory=list)  # LogEnrichmentProvider

    def with_observability(self, obs: Any) -> "PipelineContext":
        """Attach observability for auto-logging on each post()."""
        self._observability = obs
        return self

    def with_memory(self, provider: Any) -> "PipelineContext":
        """Attach a memory provider for run_stage() session management."""
        self._memory = provider
        return self

    def with_log_enrichment(self, *providers: Any) -> "PipelineContext":
        """Add log enrichment providers for run_stage() calls."""
        self._enrichment.extend(providers)
        return self

    def post(
        self,
        name: str,
        success: bool,
        output: str,
        error: str = "",
    ) -> None:
        """Record a stage result and auto-log if observability is attached."""
        self._stages.append({
            "name": name,
            "success": success,
            "output": output[:200],
            "error": error,
        })

        if self._observability:
            self._observability.log_info(
                "pipeline_stage_completed",
                stage=name,
                success=success,
                output_preview=output[:200] if output else "",
                error=error,
            )

    async def run_stage(
        self,
        agent: Any,  # ManagedAgent
        name: str,
        prompt: Union[str, Sequence[UserContent]],
        **kwargs: Any,
    ) -> Any:  # AgentRunResult
        """Run an agent as a pipeline stage.

        Handles MessageHistory creation, session ID, enrichment,
        and calls post() automatically.

        Args:
            agent: ManagedAgent instance
            name: Stage name (e.g., "Research")
            prompt: Prompt for the agent. String, or a sequence of pydantic_ai
                UserContent parts for multimodal input.
            **kwargs: Passed through to agent.run()
        """
        from .log_enrichment import LogContext
        from .memory import MessageHistory

        session_id = f"stage-{name.lower().replace(' ', '-')}"

        # Build enrichment for this stage
        stage_enrichment = LogContext().with_("stage", name)
        for provider in self._enrichment:
            for k, v in provider.enrich().items():
                stage_enrichment = stage_enrichment.with_(k, v)

        # Build history
        history = MessageHistory()
        if self._memory:
            await history.load(session_id, self._memory)

        # Merge stage enrichment with any per-call enrichment
        call_enrichment = kwargs.pop("enrichment", None)
        if call_enrichment:
            stage_enrichment.merge(call_enrichment)

        result = await agent.run(
            prompt,
            history,
            session_id,
            enrichment=stage_enrichment,
            **kwargs,
        )

        error_msg = ""
        if not result.success and result.error_context:
            error_msg = result.error_context.error_message

        self.post(name, result.success, str(result.output or ""), error_msg)
        return result

    def display_trace(self) -> None:
        """Print a formatted table of all pipeline stages."""
        print(f"\n{'='*60}")
        print("Pipeline Trace")
        print("=" * 60)
        success_count = 0
        for i, s in enumerate(self._stages):
            mark = "✓" if s["success"] else "✗"
            print(f"\n  Stage {i+1}. {s['name']}  {mark}")
            print(f"     Output: {s['output'][:150]}")
            if s["error"]:
                print(f"     Error:  {s['error'][:120]}")
            if s["success"]:
                success_count += 1
        print(f"\n{'─'*60}")
        print(f"  {success_count}/{len(self._stages)} stages succeeded")
