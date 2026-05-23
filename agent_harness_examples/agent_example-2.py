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

import os
import asyncio
import uuid

from colorama import init as colorama_init
from dotenv import load_dotenv
from agent_harness.agent import ManagedAgent
from agent_harness.memory import MessageHistory, InMemoryProvider, MongoMemory
from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger
from agent_harness.tracing import LogfireTracer, OTELTracer
from agent_harness.metrics import OTLPMetrics
from agent_harness.prompts import StaticPrompts
from agent_harness.errorhandling import ErrorHandlingConfig, AgentErrorContext
from agent_harness.model_config import ModelConfig
from pydantic_ai.settings import ModelSettings


load_dotenv()
colorama_init()


def create_memory_providers():
    """Create short and long-term memory from .env config."""
    short_term = InMemoryProvider(max_turns=10)
    long_term = None

    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri:
        long_term = MongoMemory(
            uri=mongo_uri,
            database=os.getenv("MONGODB_DATABASE", "agent_memory"),
            collection=os.getenv("MONGODB_COLLECTION", "conversations"),
        )
    return short_term, long_term


class AgentErrorHandler:
    """Error handler for the agent."""

    def __init__(self, obs: Observability):
        self._obs = obs

    def __call__(self, ctx: AgentErrorContext, exc: Exception) -> bool:
        print(
            f"[ERROR] {ctx.source}: {ctx.error_context.error_type} - {ctx.error_context.error_message}"
        )
        print(f"  Session: {ctx.session_id}")
        print(
            f"  Prompt: {ctx.prompt[:100]}..."
            if ctx.prompt and len(ctx.prompt) > 100
            else f"  Prompt: {ctx.prompt}"
        )

        if hasattr(self._obs, "tracer"):
            self._obs.tracer.error(
                f"{ctx.error_context.error_type}: {ctx.error_context.error_message}",
                source=ctx.source,
                session_id=ctx.session_id or "unknown",
                prompt=ctx.prompt or "unknown",
                stack_trace=ctx.error_context.stack_trace or "",
            )

        return False


async def main():
    # Setup
    short_term, long_term = create_memory_providers()
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    model_settings = ModelSettings(
        thinking=True,
        max_tokens=16384,
        temperature=0.2,
        top_p=0.9,
        timeout=30.0,
    )

    obs = Observability(
        tracer=LogfireTracer(service_name="agent_example-2_service"),
        loggers=[ConsoleLogger()],
        tracers=[
            OTELTracer(
                service_name="agent_example-2_service",
                otlp_endpoint="http://localhost:4317",
                sample_rate=1.0,
            ),
        ],
        metrics_list=[
            OTLPMetrics(
                service_name="agent_example-2_service",
                otlp_endpoint="http://localhost:4319",
            ),
        ],
    )
    obs.logger.info("Starting agent execution")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(StaticPrompts("You are a helpful assistant"))
        .with_observability(obs)
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_error_handling(
            ErrorHandlingConfig().with_error_handler(AgentErrorHandler(obs))
        )
    )

    # Run agents directly
    save_to = [p for p in [short_term, long_term] if p]

    output1 = await run_agent_step(
        agent,
        "what 2+2?",
        session_id,
        save_to,
        model_settings=model_settings,
    )
    print(f"Agent 1 run 1: {output1}")

    output2 = await run_agent_step(
        agent,
        "add 1. what is the total?",
        session_id,
        save_to,
        model_settings=model_settings,
    )
    print(f"Agent 1 run 2: {output2}")

    output3 = await run_agent_step(
        agent,
        "add 2 more and tel me the total",
        session_id,
        save_to,
    )
    print(f"Agent 2, run 1: {output3}")

    obs.logger.info("Agent execution completed")


async def run_agent_step(agent, prompt, session_id, save_to, model_settings=None):
    """Execute a single agent step and return the output."""
    result = await agent.run(
        prompt,
        MessageHistory(),
        session_id,
        save_to=save_to,
        model_settings=model_settings,
    )
    return result.output


if __name__ == "__main__":
    asyncio.run(main())
