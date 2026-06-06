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

"""All guardrails combined — retries, content filter, PII, token/cost limits.

Demonstrates:
  - Agent retries with timeout and fallback
  - Content filtering via on_filter callback
  - PII detection and redaction via on_redact callback
  - Token limits with on_token_limit callback
  - Cost limits with on_cost_limit callback
  - Circuit breaker for failure protection
  - All guardrails configured fluently on a single agent

Usage:
    uv run python 09_all_guardrails.py
"""

import asyncio
import re

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.guards import (
    AgentRetryConfig,
    ContentFilterConfig,
    PIIDetectionConfig,
    TokenLimitsConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
)


# ── Callback implementations ────────────────────────────────────────

def content_filter(text: str) -> str:
    """Filter profanity from response text."""
    for word in ["damn", "hell", "crap"]:
        text = re.sub(rf"\b{word}\b", "***", text, flags=re.IGNORECASE)
    return text


def redact_pii(text: str) -> str:
    """Redact emails and phone numbers from response text."""
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '[EMAIL]', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
    return text


def on_retry_callback(ctx):
    print(f"  [on_retry] {ctx.error_type} (attempt {ctx.attempt}/{ctx.max_attempts})")


def on_retry_error(ctx):
    return f"All retries exhausted: {ctx.error_message}"


def on_filter_error(ctx):
    return f"Content filter failed: {ctx.error_message}"


def on_redact_error(ctx):
    return f"PII redaction failed: {ctx.error_message}"


def on_token_limit(ctx):
    return f"Token limit reached: {ctx.error_message}"


def on_cost_limit(ctx):
    return f"Cost limit reached: {ctx.error_message}"


def on_circuit_error(ctx):
    return f"Circuit breaker open: {ctx.error_message}"


async def main():
    print("=" * 60)
    print("All Guardrails Combined")
    print("=" * 60)

    memory = InMemoryProvider()

    # ── Build agent with all guardrails ────────────────────────
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        # Retries
        .with_agent_retries(
            AgentRetryConfig()
            .with_max_retries(3)
            .with_timeout(30)
            .with_backoff(2.0)
            .on_retry(on_retry_callback)
            .on_error(on_retry_error)
        )
        # Content filter
        .with_content_filter(
            ContentFilterConfig()
            .on_filter(content_filter)
            .on_error(on_filter_error)
        )
        # PII detection
        .with_pii_detection(
            PIIDetectionConfig()
            .on_redact(redact_pii)
            .on_error(on_redact_error)
        )
        # Token limits
        .with_token_limits(
            TokenLimitsConfig()
            .with_max_total_tokens(500)
            .on_token_limit(on_token_limit)
        )
        # Cost limits (GPT-4o pricing)
        .with_cost_limits(
            CostLimitsConfig()
            .with_cost_per_input_token(0.000003)
            .with_cost_per_output_token(0.000015)
            .with_max_total_cost(0.01)
            .on_cost_limit(on_cost_limit)
        )
        # Circuit breaker
        .with_circuit_breaker(
            CircuitBreakerConfig()
            .with_threshold(5)
            .with_timeout(30)
            .on_error(on_circuit_error)
        )
    )

    # ── Print configuration ────────────────────────────────────
    print("\nGuard Configuration:")
    print(f"  Agent retries: {agent.guards.agent.max_retries} (timeout: {agent.guards.agent.timeout}s)")
    print(f"  Content filter: {agent.guards.content_filter is not None}")
    print(f"  PII detection:  {agent.guards.pii_detection is not None}")
    print(f"  Token limits:   {agent.guards.token_limits is not None}")
    print(f"  Cost limits:    {agent.guards.cost_limits is not None}")
    print(f"  Circuit breaker:{agent.guards.circuit_breaker is not None}")
    print(f"  Circuit threshold: {agent.guards.circuit_breaker.failure_threshold if agent.guards.circuit_breaker else 'N/A'}")

    # ── Run 1: Simple prompt ───────────────────────────────────
    print("\n--- Run 1: Simple greeting ---")
    history = await MessageHistory().load("all-guards-1", memory)

    result = await agent.run(
        "Say hello and introduce yourself.",
        history,
        "all-guards-1",
    )
    print(f"  Success: {result.success}")
    print(f"  Output: {result.output}")

    # ── Run 2: Content that triggers filters ───────────────────
    print("\n--- Run 2: Profile with PII and strong language ---")
    history2 = await MessageHistory().load("all-guards-2", memory)

    result2 = await agent.run(
        "Create a character profile for John Doe with email john@example.com, "
        "phone (212) 555-0199. Make him say 'damn, this is hell!' in a quote.",
        history2,
        "all-guards-2",
    )
    print(f"  Success: {result2.success}")
    print(f"  Output: {result2.output}")

    # ── Run 3: Using with_guardrails bulk setter ────────────────
    print("\n--- Run 3: Bulk guardrail setter ---")
    agent2 = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_agent_retries(AgentRetryConfig().with_max_retries(2))
        .with_guardrails(
            content_filter=ContentFilterConfig().on_filter(content_filter),
            pii_detection=PIIDetectionConfig().on_redact(redact_pii),
            token_limits=TokenLimitsConfig().with_max_total_tokens(300),
            cost_limits=CostLimitsConfig()
            .with_cost_per_input_token(0.000003)
            .with_cost_per_output_token(0.000015)
            .with_max_total_cost(0.005),
        )
        .with_circuit_breaker(CircuitBreakerConfig().with_threshold(3))
    )

    history3 = await MessageHistory().load("all-guards-3", memory)
    result3 = await agent2.run(
        "What is the weather like today in New York? Answer in 1-2 sentences.",
        history3,
        "all-guards-3",
    )
    print(f"  Success: {result3.success}")
    print(f"  Output: {result3.output}")

    print("\n" + "=" * 60)
    print("All guardrail demonstrations complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
