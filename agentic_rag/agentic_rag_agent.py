from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any imports that might trigger logfire
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

import asyncio
from datetime import datetime
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability, ObservabilityBuilder
from agent_harness.logging import ConsoleLogger
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


def get_labs(category: str) -> str:
    """get_labs tool that returns retrived labs for the specified category."""
    print("[tool:get_labs] params:", category)
    result = [
    "Total Cholesterol: 192 ( <200 )",
    "Triglyceride: 200 ( <150 )",
    "HDL-Cholesterol: 50 ( >45 )",
    "LDL-Cholesterol: 102 ( <100 )",
    "VLDL-Cholesterol: 40 ( 5-40 )",
    "Non-HDL-Cholesterol: 142 ( <130 )"
    ]
    return result


def get_diagnosis(category: str) -> str:
    """get_diagnosis tool that returns retrived diagnosis for the specified category."""
    print("[tool:get_diagnosis] params:", category)
    result = [
    "Hepatic abnormality",
    "Liver damage", 
    "Liver disease",
    "Bilirubin elevation",
    "Albumin decrease"
    ]
    return result

def get_findings(category: str) -> str:
    """get_findings tool that returns retrived findings for the specified category."""
    print("[tool:get_findings] params:", category)
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
        print(f"[Judge] Prompt: {prompt}")
        print(f"[Judge] Response: {output}")
        # print(f"[Judge] Score: {score}")  # Uncomment when judge LLM implemented

class PrintEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        print("[Evaluator] Context:", context)
        print("[Evaluator] Prompt:", prompt)
        print("[Evaluator] Result:", getattr(result, "output", result))


async def chat_loop():
    session_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    short_term = InMemoryProvider()
    long_term = InMemoryProvider()

    tools = ToolRegistry().add_many(get_labs, get_diagnosis, get_findings)
    agent = (
        ManagedAgent()
        .with_model("ollama:gpt-oss:20b")
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_tools(tools)
        .with_prompts(StaticPrompts("You are a medical assistant. When user asks about labs, ALWAYS call get_labs with category='lipid panel'. When user asks about diagnosis, ALWAYS call get_diagnosis with category='general'. When user asks about imaging/films, ALWAYS call get_findings with category='chest xray'. Provide concise answers based on tool results."))
        .with_observability(
            ObservabilityBuilder()
            .with_logfire_tracing()
            .with_console_logging()
            .build())
        .with_error_handling(ErrorHandlingConfig())
        .with_agent_retries(AgentRetryConfig(max_retries=3, timeout=120))
        .with_tool_retries(ToolRetryConfig(max_retries=3))
        .with_result_validator_retries(ResultValidatorRetryConfig(max_retries=3))
    )

    history = await MessageHistory().load(session_id, short_term)
    print(f"[Session: {session_id}]")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        result = await agent.run(user_input, history, session_id)
        print(f"\nAgent: {result.output}")


if __name__ == "__main__":
    asyncio.run(chat_loop())