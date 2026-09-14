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
import structlog
from agent_harness.agent import ManagedAgent
from agent_harness.memory import MessageHistory, InMemoryProvider
from agent_harness.observability import Observability
from agent_harness.prompts import StaticPrompts
from agent_harness.errorhandling import ErrorHandlingConfig, ErrorContext
from agent_harness.model_config import ModelConfig
from pydantic_ai.settings import ModelSettings


load_dotenv()
colorama_init()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)


class StructlogLogger:
    """Adapt the application's structlog logger to the harness logger port."""

    def __init__(self, logger):
        self._logger = logger

    def debug(self, message: str, **context):
        self._logger.debug(message, **context)

    def info(self, message: str, **context):
        self._logger.info(message, **context)

    def warning(self, message: str, **context):
        self._logger.warning(message, **context)

    def error(self, message: str, **context):
        self._logger.error(message, **context)


log = structlog.get_logger()


def create_memory_providers():
    """Create short and long-term in-memory providers."""
    return InMemoryProvider(max_turns=10), InMemoryProvider(max_turns=100)


class AgentErrorHandler:
    """Error handler for the agent."""

    def __init__(self, obs: Observability):
        self._obs = obs

    def __call__(self, ctx: ErrorContext) -> str | None:
        log.debug(
            "agent_error",
            source=ctx.source,
            error_type=ctx.error_type,
            error_message=ctx.error_message,
        )
        log.debug("error_session", session_id=ctx.session_id)
        log.debug(
            "error_prompt",
            prompt=(ctx.prompt[:100] + "..."
                    if ctx.prompt and len(ctx.prompt) > 100
                    else ctx.prompt),
        )

        if hasattr(self._obs, "tracer"):
            self._obs.tracer.error(
                f"{ctx.error_type}: {ctx.error_message}",
                source=ctx.source,
                session_id=ctx.session_id or "unknown",
                prompt=ctx.prompt or "unknown",
                stack_trace=ctx.stack_trace or "",
            )

        return None  # re-raise


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

    obs = Observability(logger=StructlogLogger(log))
    obs.info("Starting agent execution")

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "qwen2.5:3b")))
        .with_prompts(StaticPrompts("You are a helpful assistant"))
        .with_observability(obs)
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_error_handling(
            ErrorHandlingConfig().on_error(AgentErrorHandler(obs))
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
    log.debug("agent_step", step="agent_1_run_1", output=output1)

    output2 = await run_agent_step(
        agent,
        "add 1. what is the total?",
        session_id,
        save_to,
        model_settings=model_settings,
    )
    log.debug("agent_step", step="agent_1_run_2", output=output2)

    output3 = await run_agent_step(
        agent,
        "add 2 more and tel me the total",
        session_id,
        save_to,
    )
    log.debug("agent_step", step="agent_2_run_1", output=output3)

    obs.info("Agent execution completed")


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
