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

from pathlib import Path
from dotenv import load_dotenv
import structlog

# Load .env BEFORE any imports that trigger observability backends
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

log = structlog.get_logger()

import asyncio
from datetime import datetime
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.guards import (
    AgentRetryConfig,
    ToolRetryConfig,
    ResultValidatorRetryConfig,
    ContentFilterConfig,
    PIIDetectionConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
)
from agent_harness.errorhandling import ErrorHandlingConfig
from agent_harness.evaluators import Evaluator
from agent_harness.model_config import ModelConfig


def get_labs(category: str) -> list[str]:
    """get_labs tool that returns retrived labs for the specified category."""
    log.debug("tool_call", tool="get_labs", category=category)
    result = [
    "Total Cholesterol: 192 ( <200 )",
    "Triglyceride: 200 ( <150 )",
    "HDL-Cholesterol: 50 ( >45 )",
    "LDL-Cholesterol: 102 ( <100 )",
    "VLDL-Cholesterol: 40 ( 5-40 )",
    "Non-HDL-Cholesterol: 142 ( <130 )"
    ]
    return result


def get_diagnosis(category: str) -> list[str]:
    """get_diagnosis tool that returns retrived diagnosis for the specified category."""
    log.debug("tool_call", tool="get_diagnosis", category=category)
    result = [
    "Hepatic abnormality",
    "Liver damage", 
    "Liver disease",
    "Bilirubin elevation",
    "Albumin decrease"
    ]
    return result

def get_findings(category: str) -> list[str]:
    """get_findings tool that returns retrived findings for the specified category."""
    log.debug("tool_call", tool="get_findings", category=category)
    result = [
        "Bilateral lung fields show no obvious parenchymal lesion.",
        "Cardiac size is normal.",
        "Hila are unremarkable.",
        "Both domes of diaphragm are normal.",
        "Both cardiophrenic and costophrenic angles are normal.",
        "Bony thoracic cage appears normal."
    ]
    return result

class LLMJudgeEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:
        output = getattr(result, "output", result)
        
        # Logic to call a judge LLM
        judge_prompt = f"Rate the following response based on accuracy and helpfulness.\nPrompt: {prompt}\nResponse: {output}"
        # score = await judge_llm.run(judge_prompt) 
        
        # Log to a production monitoring system (e.g., Prometheus, LangSmith, or a DB)
        # await telemetry_client.log_metric("eval_score", score)
        log.debug("judge_eval", prompt=prompt, response=str(output))
        # print(f"[Judge] Score: {score}")  # Uncomment when judge LLM implemented

class PrintEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        log.debug("evaluator_result", context=str(context), prompt=prompt, result=str(getattr(result, "output", result)))


async def chat_loop():
    session_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    short_term = InMemoryProvider()
    long_term = InMemoryProvider()

    tools = ToolRegistry().add_many(get_labs, get_diagnosis, get_findings)
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_tools(tools)
        .with_prompts(StaticPrompts("You are a medical assistant. When user asks about labs, ALWAYS call get_labs with category='lipid panel'. When user asks about diagnosis, ALWAYS call get_diagnosis with category='general'. When user asks about imaging/films, ALWAYS call get_findings with category='chest xray'. Provide concise answers based on tool results."))
        .with_observability(
            Observability(builder=ObservabilityBuilder().with_otel_observability())
        )
        .with_error_handling(ErrorHandlingConfig())
        .with_agent_retries(AgentRetryConfig(max_retries=3, timeout=120))
        .with_tool_retries(ToolRetryConfig(max_retries=3))
        .with_result_validator_retries(ResultValidatorRetryConfig(max_retries=3))
    )

    history = await MessageHistory().load(session_id, short_term)
    log.debug("session_started", session_id=session_id)
    log.debug("chat_instructions")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            log.debug("exiting")
            break

        if user_input.lower() in ("quit", "exit"):
            log.debug("goodbye")
            break

        if not user_input:
            continue

        result = await agent.run(user_input, history, session_id)
        log.debug("agent_response", output=str(result.output))


if __name__ == "__main__":
    asyncio.run(chat_loop())