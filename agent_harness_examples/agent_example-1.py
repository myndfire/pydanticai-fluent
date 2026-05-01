import asyncio
from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger
from agent_harness.guards import GuardConfig
from agent_harness.errorhandling import ErrorHandlingConfig
from agent_harness.evaluators import Evaluator


def repeat(text: str) -> str:
    """Simple repeat tool that returns the provided text unchanged."""
    print("[tool:repeat] params:", text)
    return text


def shout(text: str) -> str:
    """Simple shout tool that returns the text in uppercase."""
    print("[tool:shout] params:", text)
    return text.upper()


class PrintEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        print("[Evaluator] Prompt:", prompt)
        print("[Evaluator] Result:", getattr(result, "output", result))


async def main():
    short_term = InMemoryProvider()
    long_term = InMemoryProvider()
    # Register both tools
    tools = ToolRegistry().add_many(repeat, shout)
    agent = (
        ManagedAgent()
        .with_model("ollama:gpt-oss:20b")
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_tools(tools)
        .with_prompts(StaticPrompts("You are a helpful bot. Use the provided tools when instructed."))
        .with_observability(Observability())
        .with_guards(GuardConfig())
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
    print("\nAgent response:", result.output)

if __name__ == "__main__":
    asyncio.run(main())