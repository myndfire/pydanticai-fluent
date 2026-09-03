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
from dotenv import load_dotenv
import structlog
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig
from agent_harness.evaluators import Evaluator


def repeat(text: str) -> str:
    """Simple repeat tool that returns the provided text unchanged.
Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python agent_example-1.py
"""
    log.debug("tool_repeat_params", text=text)
    return text


def shout(text: str) -> str:
    """Simple shout tool that returns the text in uppercase."""
    log.debug("tool_shout_params", text=text)
    return text.upper()


class PrintEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        log.debug("evaluator_prompt", prompt=prompt)
        log.debug("evaluator_result", result=getattr(result, "output", result))


load_dotenv()
log = structlog.get_logger()


async def main():
    short_term = InMemoryProvider()
    long_term = InMemoryProvider()
    # Register both tools
    tools = ToolRegistry().add_many(repeat, shout)
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=os.getenv("MODEL_NAME", "qwen2.5:3b")))
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_tools(tools)
        .with_prompts(StaticPrompts("You are a helpful bot. Use the provided tools when instructed."))
        .with_error_handling(ErrorHandlingConfig())
        .with_evaluators(PrintEvaluator())
    )
    history = await MessageHistory().load("demo-session", short_term)
    # Prompt that requests use of both tools
    result = await agent.run(
        "First, use the repeat tool to repeat the phrase 'hello world'. Then, use the shout tool on the result.",
        history,
        "demo-session",
    )
    log.debug("agent_response", output=result.output)

if __name__ == "__main__":
    asyncio.run(main())