# Evaluator Examples

Five examples demonstrating the `agent_harness` evaluator system — post-turn, **read-only observers** that inspect an agent's response after every `agent.run()` call. Evaluators score quality, check content safety, run custom domain logic, and can write to files, metrics, or external systems.

## Core Concept

An *evaluator* is a callable that runs automatically after each turn. It receives the same three inputs every time:

```
evaluate(prompt: str, result: AgentRunResult, context: dict) -> None
```

- **`prompt`** — the user input for this turn
- **`result`** — the agent's `AgentRunResult` (`.output`, `.success`, `.error_context`, `.usage`)
- **`context`** — a dict with `session_id`, `prompt_id`, and `model`

Key properties:

- Registered with `.with_evaluators(eval1, eval2, ...)` on a `ManagedAgent`.
- Run **sequentially** in registration order, each receiving the same `(prompt, result, context)`.
- **Read-only** — they can never modify the agent's output; they only observe, log, store, or alert.
- Failures in one evaluator do not affect the others; an evaluator exception is routed to `on_evaluator_error`.

Two authoring styles:

1. **`CustomEvaluator` base class** (file `03`) — provides `self.log_info()`, `self.log_warning()`, `self.log_error()` with an automatic `[name]` prefix.
2. **The raw `Evaluator` protocol** (file `04`) — any class with an `async def evaluate(...)` method, no base class or imports required.

## Prerequisites

| Requirement | Setup |
|---|---|
| **Ollama** | Install from [ollama.com](https://ollama.com), then run `ollama serve` |
| **Model** | `ollama pull gpt-oss:20b` (the model used by all examples) |
| **Dependencies** | `cd agent_harness_examples && uv sync` |
| **(Optional) OpenAI** | `uv add openai` and set `OPENAI_API_KEY` — only required for `SafetyCheck` (`02_safety_check.py`). Without it, safety checks are skipped gracefully. |
| **MongoDB / Redis** | Not required — all examples use `InMemoryProvider`. |

> **Python version:** the `agent_harness_examples` project requires Python ≥ 3.11. Core dependencies are `pydanticai-fluent`, `pydantic-ai`, `beanie`, `colorama`, and `python-dotenv` (the latter is used by every example via `load_dotenv()`).

## Examples

### 01 — QualityCheck (LLM-as-Judge)

`uv run 01_quality_check.py`

Uses `QualityCheck(threshold, judge_model)` to rate each response 0–10 with a second LLM call:

- **Example 1:** default threshold `7.0`.
- **Example 2:** strict threshold `9.5` (almost always warns).
- **Example 3:** two `QualityCheck` instances with different thresholds (`3.0` lenient + `7.0` moderate) on the same agent.

Logs a warning if `score < threshold`, else logs info. Never modifies the response.

---

### 02 — SafetyCheck (OpenAI Moderation)

`uv run 02_safety_check.py`

Uses `SafetyCheck()` to evaluate **both** the user prompt and the agent response via the OpenAI moderation API. Checks the categories: `hate`, `harassment`, `self-harm`, `sexual`, `violence`. Logs a warning for any flagged category.

- If the `openai` package is not installed, it logs a warning and skips — the agent still runs.
- **Example 3** runs a 3-turn conversation through the safety check.

> The moderation model is read from `SAFETY_CHECK_MODEL` (default `omni-moderation-2024-09-26`).

---

### 03 — CustomEvaluator (Base Class)

`uv run 03_custom_evaluator.py`

Demonstrates subclassing `CustomEvaluator` with the built-in logging helpers (`log_info` / `log_warning`, automatic `[name]` prefix). Three domain evaluators:

- **`ResponseLengthEvaluator`** — warns if the response is too short or too long.
- **`KeywordEvaluator`** — checks that required keywords appear in the response.
- **`TurnCounterEvaluator`** — maintains `self.count` state across turns.

**When to use `CustomEvaluator`:** you want structured logging, an automatic `[name]` prefix, and consistent format across evaluators.

---

### 04 — Protocol Evaluator (Direct Implementation)

`uv run 04_protocol_evaluator.py`

Implements the `Evaluator` protocol directly — no base class, no framework imports, just an `async def evaluate(...)` method. Three examples:

- **`AuditLogEvaluator`** — appends a JSON line per turn to `audit_log.jsonl` / `full_audit.jsonl` (includes `timestamp`, `session_id`, `prompt_id`, `model`, `prompt`, `response`, `success`, `error`).
- **`LatencyTracker`** — prints response length metadata per turn.
- **`PiiScanner`** — regex-scans the response for emails, phone numbers, and SSNs.

The `context` dict exposes `session_id`, `prompt_id`, and `model`.

**When to use the protocol approach:** you need full control over output (files, external APIs, custom formats), want a specific logging library, or require zero framework dependency.

---

### 05 — Combined Evaluators

`uv run 05_combined_evaluators.py`

Runs multiple evaluator styles together on one agent:

- `QualityCheck(threshold=6.0)` — LLM-as-judge
- `DiversityEvaluator` (a `CustomEvaluator`) — detects duplicate/repetitive responses
- `TimingLogger` (a protocol evaluator) — word/char counts

Also attaches error handling so evaluator failures don't crash the agent:

```python
.with_error_handling(
    ErrorHandlingConfig().on_evaluator_error(lambda ctx: None)  # suppress
)
```

Demonstrates that evaluators run sequentially, each receiving the same `(prompt, result, context)`, and that they cannot modify the output.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_NAME` | No | `gpt-oss:20b` | Model used by the agent in every example |
| `QUALITY_CHECK_MODEL` | No | `ollama:gpt-oss:20b` | Judge model for `QualityCheck` (provider:model_name format) |
| `OPENAI_API_KEY` | Only for `02_safety_check.py` | — | Enables the OpenAI moderation API used by `SafetyCheck` |
| `SAFETY_CHECK_MODEL` | No | `omni-moderation-2024-09-26` | OpenAI moderation model (options: `omni-moderation-latest`, `text-moderation-latest`) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama endpoint (only needed if not on localhost) |

Example `.env` (placeholders only — never commit real keys):

```dotenv
MODEL_NAME=gpt-oss:20b
QUALITY_CHECK_MODEL=ollama:gpt-oss:20b
# OPENAI_API_KEY=sk-...
# SAFETY_CHECK_MODEL=omni-moderation-2024-09-26
# OLLAMA_BASE_URL=http://localhost:11434/v1
```

A working `.env.example` is provided in the parent `agent_harness_examples/` directory.

## Running

All commands are run from the `agent_harness_examples/3-evaluators/` directory (this folder):

```bash
cd agent_harness_examples/3-evaluators
uv sync

uv run 01_quality_check.py
uv run 02_safety_check.py
uv run 03_custom_evaluator.py
uv run 04_protocol_evaluator.py
uv run 05_combined_evaluators.py
```

> `uv sync` only needs to run once from the `agent_harness_examples` project root (it creates the shared `.venv` and installs the editable `pydanticai-fluent` package). Running an example with `uv run` inside `3-evaluators/` picks up that project environment.

## Expected Output

Each script prints a banner and per-example sections. Representative behavior:

- **`01_quality_check.py`** — after each turn, a judge score is logged; e.g. `QualityCheck(threshold=7.0)` logs info when the score passes and a warning when it falls below. The strict `9.5` threshold typically warns every time.
- **`02_safety_check.py`** — for safe prompts, moderation reports clean categories (debug). If `openai` is unavailable, it prints `openai package is NOT installed — safety checks will be skipped`.
- **`03_custom_evaluator.py`** — `[length_check]`, `[keyword_check]`, and `[turn_counter]` prefixed log lines show pass/warn states.
- **`04_protocol_evaluator.py`** — writes `audit_log.jsonl` (and `full_audit.jsonl`); the script then reads them back and prints a summary of entries. `PiiScanner` prints `⚠️ PII detected ...` when the response contains an email/phone/SSN, otherwise `✓ No PII detected`.
- **`05_combined_evaluators.py`** — combined `[diversity]`, `[timing]` logs plus the `QualityCheck` judgment; duplicate prompts trigger the "Duplicate response detected" warning.

Exact scores and responses are dynamic (LLM-generated) and will vary between runs.

## How It Works

For a single `agent.run(prompt, history, session_id)` call:

1. The agent produces an `AgentRunResult` (text in `.output`, plus `.success`, `.error_context`, `.usage`).
2. Each registered evaluator's `evaluate(prompt, result, context)` is invoked **in order**.
3. The evaluator inspects the inputs and either logs, writes a file, updates internal state, or calls an external API — but never alters `result.output`.
4. If an evaluator raises, the exception is routed to `on_evaluator_error` (when configured); otherwise the agent run continues unaffected.
5. Control returns to the script, which prints the (unmodified) agent output.

The `context` dict always contains `session_id`, `prompt_id`, and `model`, allowing evaluators to correlate or persist data across turns.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SafetyCheck` prints "openai package is NOT installed" | `openai` missing | `uv add openai` and set `OPENAI_API_KEY` |
| `Connection refused` / model errors | Ollama not running or model not pulled | `ollama serve` then `ollama pull gpt-oss:20b` |
| `QualityCheck` never logs a score | `QUALITY_CHECK_MODEL` not pulled or Ollama down | Pull the judge model or check `OLLAMA_BASE_URL` |
| Evaluator exception crashes the run | No `on_evaluator_error` handler | Attach `ErrorHandlingConfig().on_evaluator_error(...)` (see `05_combined_evaluators.py`) |
| Moderation always skipped | `OPENAI_API_KEY` unset | Set the key in `.env` and `load_dotenv()` (already called) |
