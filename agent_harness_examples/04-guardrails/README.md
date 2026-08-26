# 4. Guardrails

Example scripts demonstrating the guardrail features of the `agent_harness`
(`pydanticai-fluent`) framework. Each script is self-contained and shows how to
configure one or more guardrails fluently on a `ManagedAgent`.

## What these examples demonstrate

```
Guardrail types at a glance:
─────────────────────────────────────────────────────────────────

  Retries               Limits              Protection
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ 01 Agent     │     │ 06 Token     │     │ 08 Circuit   │
  │ 02 Tool      │     │ 07 Cost      │     │    Breaker   │
  │ 03 Validator │     │ 10 Turn      │     │              │
  └──────────────┘     └──────────────┘     └──────────────┘

  Filtering
  ┌──────────────┐
  │ 04 Content   │
  │ 05 PII       │
  └──────────────┘

  Combined
  ┌──────────────┐
  │ 09 All       │
  └──────────────┘
```

| #  | Script                          | Guardrail(s) shown                                                                 |
|----|---------------------------------|------------------------------------------------------------------------------------|
| 01 | `01_agent_retries.py`           | Agent-level retries: timeout, exponential backoff, fallback model, `on_retry` / `on_error` callbacks |
| 02 | `02_tool_retries.py`            | Tool-level retries (per tool call) with backoff, combined with agent retries       |
| 03 | `03_result_validator_retries.py`| Output-validation retries via a Pydantic model + result validator                  |
| 04 | `04_content_filter.py`          | Content filtering (profanity redaction) via `on_filter` / `on_error`               |
| 05 | `05_pii_detection.py`           | PII detection/redaction (email, phone, SSN, credit card, IP) via `on_redact` / `on_error` |
| 06 | `06_token_limits.py`            | Token limits (input/output/total) with `on_token_limit`; raises when no callback   |
| 07 | `07_cost_limits.py`             | Cost limits using per-token pricing, with `on_cost_limit`                          |
| 08 | `08_circuit_breaker.py`         | Circuit breaker (CLOSED/OPEN/HALF_OPEN) protecting downstream failures              |
| 09 | `09_all_guardrails.py`          | All guardrails combined on one agent + bulk `with_guardrails` setter               |
| 10 | `10_turn_limits.py`             | Turn/session limits with per-session isolation; raises when no callback            |

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Dependencies (from `agent_harness_examples/pyproject.toml`):
  `pydanticai-fluent`, `pydantic-ai`, `beanie`, `colorama`, and `python-dotenv`
  (imported via `dotenv`)
- A model provider reachable from the host:
  - **Ollama (default):** run `ollama serve` and have the model pulled
    (default `gpt-oss:20b`)
  - **OpenAI:** set `GUARDRAILS_MODEL_PROVIDER=openai` and provide
    `OPENAI_API_KEY`; network access required

## Configuration

All scripts call `load_dotenv()` at import and read variables from a `.env`
file in the `agent_harness_examples/` directory (see `.env.example` there).

Common variables:

- `GUARDRAILS_MODEL_PROVIDER` (default `ollama`)
- `GUARDRAILS_MODEL_NAME` (default `gpt-oss:20b`)

Each script exposes its own tuning knobs via environment variables (all have
sensible defaults), for example:

- `01`: `AGENT_RETRIES_MAX_RETRIES`, `AGENT_RETRIES_TIMEOUT`, `AGENT_RETRIES_BACKOFF`, `AGENT_RETRIES_FALLBACK_MODEL`
- `02`: `TOOL_RETRIES_AGENT_MAX_RETRIES`, `TOOL_RETRIES_AGENT_TIMEOUT`, `TOOL_RETRIES_MAX_RETRIES`, `TOOL_RETRIES_BACKOFF`
- `03`: `VALIDATOR_AGENT_MAX_RETRIES`, `VALIDATOR_AGENT_TIMEOUT`, `VALIDATOR_MAX_RETRIES`, `VALIDATOR_BACKOFF`
- `06`: `TOKEN_LIMITS_EX1_*`, `TOKEN_LIMITS_EX2_*`, `TOKEN_LIMITS_EX3_*`
- `07`: `COST_LIMITS_INPUT_COST`, `COST_LIMITS_OUTPUT_COST`, `COST_LIMITS_EX*_*`
- `08`: `CIRCUIT_BREAKER_BAD_MODEL_NAME`, `CIRCUIT_BREAKER_THRESHOLD`, `CIRCUIT_BREAKER_TIMEOUT`
- `09`: `ALL_GUARDRAILS_*`
- `10`: `TURN_LIMITS_EX1_MAX_TURNS`, `TURN_LIMITS_EX2_MAX_TURNS`, `TURN_LIMITS_EX3_MAX_TURNS`

See each file's module docstring for the full list and defaults.

## How to run

From the repository root of the examples package:

```bash
cd agent_harness_examples
uv sync
uv run python 4-guardrails/01_agent_retries.py
```

Replace the script name to run any other example (`01`–`10`). Output is printed
to stdout; each script is independent and uses in-memory message history.

## Retry Flow

Agent and tool retries follow the same pattern:

```
agent.run()
    │
    ├──▶ attempt 1 ──▶ timeout / error
    │                     │
    │                     ▼
    │               on_retry callback
    │               wait 2s (backoff)
    │                     │
    ├──▶ attempt 2 ──▶ timeout / error
    │                     │
    │                     ▼
    │               on_retry callback
    │               wait 4s (backoff × 2)
    │                     │
    ├──▶ attempt 3 ──▶ timeout / error
    │                     │
    │                     ▼
    │               on_error callback
    │               (fallback model? or suppress)
    │
    └──▶ final result
```

## Circuit Breaker State Machine

```
            ┌──────────────────────────────────────────┐
            │                                          │
            ▼                                          │
       ┌─────────┐    N failures    ┌──────────┐       │
       │ CLOSED  │ ──────────────▶  │   OPEN   │       │
       │(healthy)│                   │ (blocked)│       │
       └─────────┘                   └──────────┘       │
            ▲                          │                │
            │                     timeout               │
            │                          │                │
            │                          ▼                │
            │    success         ┌───────────┐          │
            └─────────────────── │ HALF_OPEN │──────────┘
                  (reset)        │  (trial)  │  failure
                                 └───────────┘
```

## Token & Cost Limit Flow

```
agent.run()
    │
    ▼
┌─────────────────────────────────────────┐
│         GuardRunner.run_with_guards()    │
│                                          │
│  1. Check token limits                   │
│     ├── input_tokens  > max_input?       │
│     ├── output_tokens > max_output?      │
│     └── total_tokens  > max_total?       │
│              │                           │
│              ▼                           │
│     on_token_limit callback              │
│                                          │
│  2. Check cost limits                    │
│     └── total_cost > max_cost?           │
│              │                           │
│              ▼                           │
│     on_cost_limit callback               │
│                                          │
│  3. Check turn limits                    │
│     └── turns_this_session > max_turns?  │
│              │                           │
│              ▼                           │
│     on_turn_limit callback               │
└─────────────────────────────────────────┘
```

## Content Filter & PII Flow

```
agent.run()
    │
    ▼
LLM produces output
    │
    ▼
┌──────────────────────────┐
│  ContentFilterConfig     │
│    .on_filter(callback)  │
│                          │
│  callback(output)        │
│    ├── regex replace     │
│    ├── external API      │
│    └── return filtered   │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  PIIDetectionConfig      │
│    .on_redact(callback)  │
│                          │
│  callback(output)        │
│    ├── detect patterns   │
│    ├── redact matches    │
│    └── return redacted   │
└──────────────────────────┘
    │
    ▼
Final output returned to caller
```

## Expected output

Each script prints a banner, a summary of its active guardrail configuration,
sends one or more prompts to the agent, and prints the result
(`Success`, `Output`, and sometimes `Error` / `used_fallback`).

- Retry, filter, PII, and limit examples show graceful handling via the
  configured callbacks.
- Scripts that omit a callback for a hard limit (e.g. `06` example 3,
  `10` example 3) demonstrate the resulting `RuntimeError` being caught and
  printed.
- `05` additionally prints PASS/FAIL checks confirming PII patterns were
  redacted.
- `08` demonstrates the circuit opening after repeated failures and recovering
  after the cooldown (includes a ~6s sleep).
