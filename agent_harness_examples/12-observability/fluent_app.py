"""Fluent Agent Demo — pydanticai-fluent with tools, guardrails, evaluators, OTel observability.

This script demonstrates a complete agent pipeline using the agent_harness library's
fluent API. It showcases inventory lookup tools, risk scoring, content filtering,
response validation, and full OpenTelemetry observability.

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  User Prompt │────▶│ ManagedAgent │────▶│   Response  │
    └─────────────┘     └──────┬───────┘     └─────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Tools   │    │Guardrails│    │Evaluators│
        └──────────┘    └──────────┘    └──────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                     ┌─────────────────┐
                     │   OTel Stack    │
                     │ (logs/traces/m) │
                     └─────────────────┘

Prerequisites:
    - Python 3.9+
    - Required packages: agent_harness, structlog, python-dotenv
    - An OpenAI API key (or compatible provider)
    - The local observability stack running (OTel Collector + Langfuse +
      Elasticsearch + Kibana), started with: docker compose up -d

Setup:
    1. Install dependencies:
       $ pip install agent_harness structlog python-dotenv

    2. Create a .env file in the project root:
       FLUENT_MODEL_PROVIDER=openai
       FLUENT_MODEL_NAME=gpt-4o-mini
       OTEL_COLLECTOR_ENDPOINT=localhost:4317
       OBSERVABILITY_SERVICE_NAME=fluent-agent-demo
       FLUENT_MAX_TOKENS=512
       LANGFUSE_PROJECT_ID=local-project
       LANGFUSE_UI_URL=http://localhost:3000
       OPENAI_API_KEY=sk-...  # Your API key

    3. Start the observability stack:
       $ docker compose up -d   # from the repo root
       The collector receives OTLP on localhost:4317:
         - traces  -> Langfuse
         - logs    -> Elasticsearch (data stream logs-generic.otel-default)
         - metrics -> collector debug output

    4. (Optional) Provision the Kibana data view that renders trace_id as a
       clickable "View in Langfuse" link:
       $ ./kibana/provision-dashboards.sh

Usage:
    $ python fluent_app.py

    The script runs three scenarios:
    1. Look up "office mouse" (low price, low risk → approved)
    2. Look up "high-end blade server" (high price, high risk → flagged)
    3. Filter-error scenario: a deliberately broken content filter exercises
       the on_filter_error callback and guardrail error logging

    Inspect the results:
      - Langfuse traces:  http://localhost:3000
        Log in with the seeded admin user from .env
        (LANGFUSE_INIT_USER_EMAIL / LANGFUSE_INIT_USER_PASSWORD).
      - Elasticsearch:    http://localhost:9200
        curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=event_name:filter_error'
      - Kibana:           http://localhost:5601
        Discover -> data view "logs-generic.otel-default*".
        Because the harness owns the run span (create_spans=True), every
        in-run log record carries trace_id; Kibana renders it as a clickable
        "View in Langfuse" link to the trace where the event happened.

Demonstrates:
    - ManagedAgent fluent API for agent construction
    - Plain function tools (get_inventory_price, calculate_risk)
    - Guardrails: ContentFilterConfig + TokenLimitsConfig
    - Evaluators: Custom ResponseLengthEvaluator
    - OTel observability: logs -> Elasticsearch, traces -> Langfuse,
      metrics via OTLP gRPC to localhost:4317
    - Trace-linked structured logs (Kibana trace_id -> Langfuse trace)
    - Guardrail error logging via on_filter_error callback
"""

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
import structlog

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.guards import ContentFilterConfig, TokenLimitsConfig
from agent_harness.evaluators import CustomEvaluator
from agent_harness.observability import Observability
from agent_harness.errorhandling import ErrorHandlingConfig

load_dotenv()
log = structlog.get_logger()

# ── Config from .env ──────────────────────────────────────────────────────
# Environment variables loaded from .env file. All have sensible defaults.
# PROVIDER:      LLM provider name (e.g., "openai", "anthropic", "ollama")
# MODEL_NAME:    Model identifier for the chosen provider
# OTEL_ENDPOINT: OTel Collector gRPC endpoint for observability data
# OBSERVABILITY_SERVICE_NAME:  Logical service name for OTel resource attribution
# MAX_TOKENS:    Maximum tokens the LLM can generate per response
PROVIDER = os.getenv("FLUENT_MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("FLUENT_MODEL_NAME", "gpt-4o-mini")
OTEL_ENDPOINT = os.getenv("OTEL_COLLECTOR_ENDPOINT", "localhost:4317")
OBSERVABILITY_SERVICE_NAME = os.getenv("OBSERVABILITY_SERVICE_NAME", "fluent-agent-demo")
MAX_TOKENS = int(os.getenv("FLUENT_MAX_TOKENS", "512"))
# Browser-reachable Langfuse UI + project id, used to build a deep link from an
# error record to the trace where it happened (see langfuse_trace_url()).
LANGFUSE_UI_URL = os.getenv("LANGFUSE_UI_URL", "http://localhost:3000")
LANGFUSE_PROJECT_ID = os.getenv("LANGFUSE_PROJECT_ID", "local-project")


def langfuse_trace_url() -> str:
    """Build a Langfuse trace URL for the currently active span.

    Returns an empty string when no recording span is active, so callers can
    safely include it as a log attribute.
    """
    from opentelemetry import trace

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return ""
    trace_id = format(span_context.trace_id, "032x")
    base = LANGFUSE_UI_URL.rstrip("/")
    return f"{base}/project/{LANGFUSE_PROJECT_ID}/traces/{trace_id}"


# ── Tools ─────────────────────────────────────────────────────────────────
# Simulated inventory database. Each item maps to a record containing:
#   - price:    Unit price in USD
#   - sku:      Stock Keeping Unit identifier
#   - in_stock: Whether the item is currently available
INVENTORY = {
    "office mouse": {"price": 45.0, "sku": "SKU-110B", "in_stock": True},
    "mechanical keyboard": {"price": 129.99, "sku": "SKU-220K", "in_stock": True},
    "ultrawide monitor": {"price": 899.00, "sku": "SKU-333M", "in_stock": False},
    "high-end blade server": {"price": 1250.00, "sku": "SKU-992A", "in_stock": True},
}

# Risk threshold: items with a calculated risk score above this value
# are flagged by the guardrail system for manual review.
GUARDRAIL_THRESHOLD = 1000.0


def get_inventory_price(item: str) -> dict:
    """Look up an item's price and stock status from the inventory database.

    This function is registered as a tool for the ManagedAgent. It performs
    a case-insensitive lookup in the INVENTORY dictionary.

    Args:
        item: The item name to look up. Case-insensitive, whitespace-trimmed.

    Returns:
        A dictionary containing:
            - price (float): Unit price in USD
            - sku (str): Stock Keeping Unit identifier
            - in_stock (bool): Whether the item is currently available
            - error (str): Error message if the item is not found

    Example:
        >>> get_inventory_price("Office Mouse")
        {'price': 45.0, 'sku': 'SKU-110B', 'in_stock': True}

        >>> get_inventory_price("nonexistent item")
        {'error': "Item 'nonexistent item' not found in inventory"}
    """
    log.info("tool_get_inventory_price", item=item)
    key = item.lower().strip()
    record = INVENTORY.get(key)
    if record is None:
        return {"error": f"Item '{item}' not found in inventory"}
    return record


def calculate_risk(price: float) -> int:
    """Calculate a risk score (0-100) based on item price.

    This function is registered as a tool for the ManagedAgent. It uses a
    piecewise linear scoring algorithm where higher prices yield higher risk
    scores. The score is capped at 100.

    Scoring tiers:
        - price < $100:     score = 0 (no risk)
        - $100 ≤ price < $500:  score = (price - 100) / 4  (0–100 range)
        - $500 ≤ price < $1000: score = 50 + (price - 500) / 10  (50–100 range)
        - price ≥ $1000:    score = 75 + min(25, (price - 1000) / 100)  (75–100 range)

    Args:
        price: The item price in USD. Must be non-negative.

    Returns:
        An integer risk score between 0 (no risk) and 100 (maximum risk).

    Example:
        >>> calculate_risk(45.0)
        0

        >>> calculate_risk(250)
        37

        >>> calculate_risk(1250)
        77
    """
    log.info("tool_calculate_risk", price=price)
    if price < 100:
        score = 0
    elif price < 500:
        score = int((price - 100) / 4)
    elif price < 1000:
        score = 50 + int((price - 500) / 10)
    else:
        score = 75 + min(25, int((price - 1000) / 100))
    return min(score, 100)


# ── Custom Evaluator ──────────────────────────────────────────────────────
class ResponseLengthEvaluator(CustomEvaluator):
    """Evaluator that validates agent response word count.

    Checks whether the agent's response falls within an acceptable word count
    range. Logs warnings for responses that are too short or too long, and
    info-level messages for acceptable responses.

    This evaluator helps ensure the agent produces responses that are
    neither trivially brief nor excessively verbose.

    Attributes:
        min_words (int): Minimum acceptable word count (default: 3)
        max_words (int): Maximum acceptable word count (default: 300)

    Example:
        >>> evaluator = ResponseLengthEvaluator(min_words=5, max_words=200)
        >>> # Use with ManagedAgent
        >>> agent = ManagedAgent().with_evaluators(evaluator)
    """

    def __init__(self, min_words: int = 3, max_words: int = 300):
        """Initialize the ResponseLengthEvaluator.

        Args:
            min_words: Minimum acceptable word count. Responses shorter
                       than this trigger a warning.
            max_words: Maximum acceptable word count. Responses longer
                       than this trigger a warning.
        """
        super().__init__(name="length_check")
        self.min_words = min_words
        self.max_words = max_words

    async def evaluate(self, prompt: str, result, context: dict) -> None:
        """Evaluate the agent's response word count.

        Args:
            prompt: The original user prompt that triggered the response.
            result: The agent's response object. Must have an 'output'
                    attribute or be convertible to string.
            context: A dictionary containing execution context, including
                     optional 'session_id' for tracing.
        """
        output = result.output if hasattr(result, "output") else str(result)
        word_count = len(output.split()) if output else 0
        session_id = context.get("session_id", "unknown")

        if word_count < self.min_words:
            self.log_warning(
                "Response too short",
                word_count=word_count,
                min_required=self.min_words,
                session_id=session_id,
            )
        elif word_count > self.max_words:
            self.log_warning(
                "Response too long",
                word_count=word_count,
                max_allowed=self.max_words,
                session_id=session_id,
            )
        else:
            self.log_info(
                "Response length OK",
                word_count=word_count,
                session_id=session_id,
            )


# ── Guardrail callbacks ───────────────────────────────────────────────────
# These functions are callbacks invoked by the guardrail system when
# specific conditions are met during agent execution.

def content_filter(text: str) -> str:
    """Replace profanity-like words with asterisks.

    This callback is registered with ContentFilterConfig and is called
    whenever the agent's output passes through the content filter.
    It uses regex patterns to detect and mask inappropriate words.

    Args:
        text: The agent's output text to filter.

    Returns:
        The filtered text with prohibited words replaced by '***'.

    Example:
        >>> content_filter("What the hell is this?")
        'What the *** is this?'
    """
    import re
    patterns = [r"\b(hell)\b", r"\b(damn)\b", r"\b(crap)\b"]
    for pat in patterns:
        text = re.sub(pat, "***", text, flags=re.IGNORECASE)
    return text


def broken_content_filter(text: str) -> str:
    """Deliberately failing content filter used to exercise error logging.

    This function is registered as the on_filter callback for a dedicated
    broken-filter agent scenario. It always raises to simulate a filter
    failure and trigger the on_filter_error callback, demonstrating the
    guardrail error-logging path.

    Args:
        text: The agent's output text (unused; always raises).

    Raises:
        RuntimeError: Always, to simulate a content-filter failure.

    Example:
        >>> broken_content_filter("anything")
        Traceback (most recent call last):
            ...
        RuntimeError: simulated content filter failure
    """
    raise RuntimeError("simulated content filter failure")


def _guardrail_log_context(ctx, guardrail: str) -> dict:
    """Build an enriched, flattened log context for a guardrail ErrorContext.

    Non-primitive values (e.g. TokenUsageInfo) are flattened to primitives so
    they survive OTel attribute normalization and are queryable in Kibana.
    """
    context = {
        "guardrail": guardrail,
        "error_type": getattr(ctx, "error_type", None),
        "error_message": getattr(ctx, "error_message", None),
        "source": getattr(ctx, "source", None),
        "session_id": getattr(ctx, "session_id", None),
        "attempt": getattr(ctx, "attempt", None),
        "max_attempts": getattr(ctx, "max_attempts", None),
        "stack_trace": getattr(ctx, "stack_trace", None),
        "langfuse_trace_url": langfuse_trace_url(),
    }
    token = getattr(ctx, "token_usage", None)
    if token is not None:
        context.update(
            token_limit_type=getattr(token, "limit_type", None),
            token_limit_value=getattr(token, "limit_value", None),
            actual_tokens=getattr(token, "actual_tokens", None),
            exceeded_by=getattr(token, "exceeded_by", None),
            percentage_of_limit=getattr(token, "percentage_of_limit", None),
        )
    return context


def on_token_limit(ctx, observability=None):
    """Graceful fallback when a token limit is hit.

    This callback is registered with TokenLimitsConfig and is invoked
    when the agent exceeds the configured token limits. It returns a
    user-friendly truncated response instead of raising an error.

    Args:
        ctx: An ErrorContext with the limit error plus session, attempt, and
            token-usage detail.
        observability: Optional Observability stack. When provided, the warning
            is emitted through it so the structured record reaches the OTLP log
            pipeline (and Kibana, with a Langfuse trace link); otherwise it falls
            back to structlog.

    Returns:
        A fallback string indicating truncation occurred.
    """
    if observability is not None:
        observability.warning("token_limit_exceeded", **_guardrail_log_context(ctx, "token_limit"))
    else:
        log.warning("token_limit_exceeded", **_guardrail_log_context(ctx, "token_limit"))
    return f"[Truncated: {ctx.error_message}]"


def on_filter_error(ctx, observability=None):
    """Handle errors that occur during content filtering.

    This callback is registered with ContentFilterConfig and is invoked
    when the content filter encounters an error (e.g., regex compilation
    failure). It logs the error and returns a graceful fallback message.

    Args:
        ctx: An ErrorContext with the filter error plus session, attempt, and
            stack-trace detail.
        observability: Optional Observability stack. When provided, the error is
            emitted through it so the structured record reaches the OTLP log
            pipeline (and Kibana, with a Langfuse trace link); otherwise it falls
            back to structlog.

    Returns:
        A fallback string indicating a filter error occurred.
    """
    if observability is not None:
        observability.error("filter_error", **_guardrail_log_context(ctx, "content_filter"))
    else:
        log.error("filter_error", **_guardrail_log_context(ctx, "content_filter"))
    return f"[Filter error]: {ctx.error_message}"


# ── Agent builder ─────────────────────────────────────────────────────────
def build_agent(
    content_filter_config: Optional[ContentFilterConfig] = None,
    observability: Optional[Observability] = None,
) -> ManagedAgent:
    """Construct and configure the ManagedAgent with all components.

    This function builds the agent using the fluent API, wiring together:
    - Model configuration (provider, model name, token limits)
    - Short-term memory (InMemoryProvider)
    - Tools (get_inventory_price, calculate_risk)
    - System prompt (inventory assistant instructions)
    - Observability stack (console + OTel logging, tracing, metrics)
    - Guardrails (content filter, token limits)
    - Evaluators (response length validation)
    - Error handling configuration

    The agent uses a fluent/chainable API where each method returns the
    agent instance, allowing method chaining for clean configuration.

    Args:
        content_filter_config: Optional ContentFilterConfig to use instead of
            the default. Pass a broken filter config to exercise error logging
            via on_filter_error.
        observability: Optional existing Observability stack to reuse. When
            None a new stack is created.

    Returns:
        A tuple of (ManagedAgent, Observability):
            - ManagedAgent: The fully configured agent instance
            - Observability: The observability stack for manual logging

    Example:
        >>> agent, obs = build_agent()
        >>> result = await agent.run("Hello", history, session_id, save_to=[memory])
    """
    # Register tools that the agent can call during execution
    tools = ToolRegistry().add_many(get_inventory_price, calculate_risk)

    # Configure observability: OTel only (console rendered by the OTel exporter)
    if observability is None:
        observability = Observability.configure(
            service_name=OBSERVABILITY_SERVICE_NAME,
            endpoint=OTEL_ENDPOINT,
            sample_rate=1.0,
            create_spans=True,
            console=True,
        )

    # Content filter: masks inappropriate words in agent output
    if content_filter_config is None:
        content_filter_config = (
            ContentFilterConfig()
            .on_filter(content_filter)
            .on_error(lambda ctx: on_filter_error(ctx, observability))
        )

    # Token limits: prevents runaway token usage with graceful fallback
    token_limits_cfg = (
        TokenLimitsConfig()
        .with_max_total_tokens(2000)
        .with_max_output_tokens(400)
        .on_token_limit(lambda ctx: on_token_limit(ctx, observability))
    )

    # Build the agent using fluent API chaining
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=PROVIDER, model_name=MODEL_NAME))
        .with_model_settings({"max_tokens": MAX_TOKENS})
        .with_short_term_memory(InMemoryProvider())
        .with_tools(tools)
        .with_prompts(StaticPrompts(
            "You are an inventory assistant. Use get_inventory_price to look up items "
            "and calculate_risk to score the price. If the risk score exceeds 50, "
            "flag the item as 'flagged_by_guardrail'. Be concise."
        ))
        .with_observability(observability)
        .with_error_handling(ErrorHandlingConfig())
        .with_evaluators(ResponseLengthEvaluator(min_words=3, max_words=300))
        .with_content_filter(content_filter_config)
        .with_token_limits(token_limits_cfg)
    )

    return agent, observability


# ── Scenarios ──────────────────────────────────────────────────────────────
# Predefined test scenarios that demonstrate the agent's capabilities.
# Each scenario is a natural language prompt that the agent processes.
SCENARIOS = [
    "Look up the office mouse in the inventory, calculate its risk score, and tell me if it is approved.",
    "Look up the high-end blade server in the inventory, calculate its risk score, and tell me if it is approved.",
]


async def run_scenario(agent: ManagedAgent, observability, session_id: str, prompt: str):
    """Execute a single test scenario with the agent.

    This function loads conversation history, runs the agent with the
    given prompt, and logs the results to observability.

    Args:
        agent: The configured ManagedAgent instance to use.
        observability: The Observability stack for logging results.
        session_id: Unique identifier for this conversation session.
        prompt: The natural language prompt to send to the agent.

    The function:
        1. Creates an InMemoryProvider for conversation history
        2. Loads any existing history for the session
        3. Runs the agent inside a scenario span (so logs share the run's trace)
        4. Emits ``scenario_completed`` on success (via the enclosing span) and an
           enriched ``scenario_failed`` record on failure (session, prompt, output,
           error context, stack trace) plus an ``agent_errors_total`` metric and a
           Langfuse trace link
    """
    memory = InMemoryProvider()
    history = await MessageHistory().load(session_id, memory)

    async with observability.observe("scenario", component="app", session_id=session_id, prompt=prompt):
        result = await agent.run(prompt, history, session_id, save_to=[memory])

        # The enclosing observe() emits ``scenario_completed`` (with duration),
        # so there is no separate ``scenario_complete`` record here.
        if not result.success:
            ec = result.error_context
            observability.error(
                "scenario_failed",
                session_id=session_id,
                prompt=prompt,
                output=result.output,
                error_type=getattr(ec, "error_type", None),
                error_message=getattr(ec, "error_message", None),
                source=getattr(ec, "source", None),
                attempt=getattr(ec, "attempt", None),
                max_attempts=getattr(ec, "max_attempts", None),
                will_retry=getattr(ec, "will_retry", None),
                stack_trace=getattr(ec, "stack_trace", None),
                partial_output=getattr(ec, "partial_output", None),
                langfuse_trace_url=langfuse_trace_url(),
            )
            observability.record_metric(
                "counter",
                "agent_errors_total",
                1,
                error_type=getattr(ec, "error_type", None),
                source=getattr(ec, "source", None),
            )

    log.info(
        "scenario_result",
        session_id=session_id,
        output=result.output,
        success=result.success,
    )


async def _collector_reachable(endpoint: str, timeout: float = 2.0) -> bool:
    """Return True when the OTLP collector endpoint accepts a connection."""
    host, _, port_text = endpoint.partition(":")
    try:
        port = int(port_text or "4317")
    except ValueError:
        port = 4317
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host or "localhost", port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def main():
    """Main entry point that orchestrates the demo execution.

    This function:
        1. Verifies the OTel Collector is reachable before creating exporters
        2. Builds the agent with all components
        3. Runs each predefined scenario with a unique session ID
        4. Waits for OTel batch exporters to flush data
        5. Shuts down the OTLP providers (even if a scenario fails)
    """
    if not await _collector_reachable(OTEL_ENDPOINT):
        log.warning("collector_unreachable", endpoint=OTEL_ENDPOINT)
        log.info(
            "start_instructions",
            command="docker compose -f docker-compose.yml up -d otel-collector",
        )
        return

    agent, observability = build_agent()

    try:
        for idx, prompt in enumerate(SCENARIOS, start=1):
            session_id = f"fluent-session-{idx}"
            await run_scenario(agent, observability, session_id, prompt)

        # Dedicated scenario to exercise on_filter_error logging
        broken_filter_cfg = (
            ContentFilterConfig()
            .on_filter(broken_content_filter)
            .on_error(lambda ctx: on_filter_error(ctx, observability))
        )
        broken_agent, _ = build_agent(
            content_filter_config=broken_filter_cfg,
            observability=observability,
        )
        await run_scenario(broken_agent, observability, "fluent-session-filter-error",
                           "Say hello in one sentence.")

        # Allow OTel batch exporters time to flush remaining data
        await asyncio.sleep(5)
    finally:
        await observability.shutdown()

    log.info("all_scenarios_complete")
    log.info("view_traces_in_langfuse", url="http://localhost:3000")


if __name__ == "__main__":
    asyncio.run(main())
