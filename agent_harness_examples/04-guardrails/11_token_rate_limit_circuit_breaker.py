"""Token/request rate limiting protected by a circuit breaker.

This example deliberately configures a one-request sliding window so the
second immediate request receives a deterministic ``RateLimitError``. Repeated
rate-limit failures open the circuit; after the cooldown, one half-open trial
is allowed and a successful model response closes the circuit again.

Demonstrates:
  - ``TokenRateLimitConfig`` for request/token windows
  - ``Retry-After``-style wait information
  - rate-limit failures as a distinct error source
  - CLOSED -> OPEN -> HALF_OPEN -> CLOSED circuit transitions
  - graceful handling while the circuit is open

Usage:
    uv run python 04-guardrails/11_token_rate_limit_circuit_breaker.py

Setup:
    1. Start the configured model provider, for example ``ollama serve``.
    2. From ``agent_harness_examples`` run the command above.
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.guards import (
    AgentRetryConfig,
    CircuitBreakerConfig,
    TokenRateLimitConfig,
)
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig


load_dotenv()
log = structlog.get_logger()

PROVIDER = os.getenv("GUARDRAILS_MODEL_PROVIDER", "ollama")
MODEL = os.getenv("GUARDRAILS_MODEL_NAME", "gpt-oss:20b")
RATE_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "1.5"))
RATE_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "1"))
RATE_TOKENS = int(os.getenv("RATE_LIMIT_TOKENS", "100"))
RESERVE_OUTPUT = int(os.getenv("RATE_LIMIT_RESERVE_OUTPUT", "16"))
CIRCUIT_THRESHOLD = int(os.getenv("RATE_LIMIT_CIRCUIT_THRESHOLD", "2"))
CIRCUIT_TIMEOUT = float(os.getenv("RATE_LIMIT_CIRCUIT_TIMEOUT", "2.0"))


def on_rate_limit(ctx):
    """Record a rate-limit event before the retry/circuit policy handles it."""
    log.warning(
        "rate_limit_exceeded",
        source=ctx.source,
        attempt=ctx.attempt,
        max_attempts=ctx.max_attempts,
        error=ctx.error_message,
    )


def on_circuit_open(ctx):
    """Return a safe response while the downstream quota is unavailable."""
    log.warning(
        "circuit_open",
        state="OPEN",
        attempt=ctx.attempt,
        error=ctx.error_message,
    )
    return f"Rate limit protection active: {ctx.error_message}"


async def request(agent, memory, number: int):
    session_id = f"rate-limit-demo-{number}"
    history = await MessageHistory().load(session_id, memory)
    try:
        result = await agent.run("Reply with exactly one short sentence.", history, session_id)
        log.info(
            "request_result",
            request=number,
            status="success" if result.success else "handled_error",
            output=str(result.output)[:120],
            used_fallback=result.used_fallback,
        )
        return result
    except Exception as exc:
        log.error(
            "request_failed",
            request=number,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


async def main():
    log.info(
        "configuration",
        provider=PROVIDER,
        model=MODEL,
        requests_per_window=RATE_REQUESTS,
        tokens_per_window=RATE_TOKENS,
        window_seconds=RATE_WINDOW_SECONDS,
        circuit_threshold=CIRCUIT_THRESHOLD,
        circuit_timeout=CIRCUIT_TIMEOUT,
    )

    rate_limit = (
        TokenRateLimitConfig(
            tokens_per_window=RATE_TOKENS,
            requests_per_window=RATE_REQUESTS,
            window_seconds=RATE_WINDOW_SECONDS,
            reserve_output_tokens=RESERVE_OUTPUT,
        )
        .on_exceeded(on_rate_limit)
    )
    circuit = (
        CircuitBreakerConfig()
        .with_threshold(CIRCUIT_THRESHOLD)
        .with_timeout(CIRCUIT_TIMEOUT)
        .on_error(on_circuit_open)
    )

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider=PROVIDER, model_name=MODEL))
        # No agent retries here: each immediate request should count once
        # toward the rate-limit circuit threshold.
        .with_agent_retries(AgentRetryConfig().with_max_retries(0))
        .with_token_rate_limit(rate_limit)
        .with_circuit_breaker(circuit)
    )
    memory = InMemoryProvider()

    log.info("state", circuit="CLOSED", detail="first request consumes the window")
    await request(agent, memory, 1)

    log.info("state", circuit="CLOSED", detail="rate limit begins failing requests")
    await request(agent, memory, 2)
    await request(agent, memory, 3)

    log.info("state", circuit="OPEN", detail="next request is rejected without model work")
    await request(agent, memory, 4)

    log.info("cooldown", seconds=CIRCUIT_TIMEOUT)
    await asyncio.sleep(CIRCUIT_TIMEOUT + 0.1)

    log.info("state", circuit="HALF_OPEN", detail="one trial request is allowed")
    recovered = await request(agent, memory, 5)
    log.info(
        "state",
        circuit="CLOSED" if recovered and recovered.success else "OPEN",
        detail="successful half-open trial closes the circuit",
    )


if __name__ == "__main__":
    asyncio.run(main())
