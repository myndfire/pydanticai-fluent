# Error Handling Examples

Nine examples demonstrating the `agent_harness` error handling system — per-source callbacks, recovery strategies, guardrails, and pipeline resilience.

## Core Concept

`ErrorHandlingConfig` lets you register a callback for each of the 8 error sources. The callback receives an `ErrorContext` with details about the failure. **Return a value to suppress the error** (agent gets `AgentRunResult(success=False, output=value)`). **Return `None` to re-raise** (exception propagates).

```
agent.run()
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Error Handler Chain                 │
│                                                  │
│  ErrorHandlingConfig()                           │
│      .on_llm_error(handler)         ──▶ llm      │
│      .on_tool_error(handler)        ──▶ tool     │
│      .on_validation_error(handler)  ──▶ validation│
│      .on_guardrail_error(handler)   ──▶ guardrail│
│      .on_memory_error(handler)      ──▶ memory   │
│      .on_prompt_error(handler)      ──▶ prompt   │
│      .on_evaluator_error(handler)   ──▶ evaluator│
│      .on_output_error(handler)      ──▶ output   │
│      .on_error(catch_all)           ──▶ any      │
└─────────────────────────────────────────────────┘
    │
    ├──▶ return value  →  suppress (result.success=False)
    └──▶ return None   →  re-raise (exception propagates)
```

## Error Sources at a Glance

```
agent.run() execution flow:
─────────────────────────────────────────────────────────────

  prompt ──▶ system prompt render ──▶ LLM call ──▶ tool calls
      │            │                      │            │
      ▼            ▼                      ▼            ▼
   on_prompt   on_prompt              on_llm       on_tool
                         │                │
                         ▼                ▼
                    output parse    guardrail checks
                         │          (CB, tokens, cost)
                         ▼                │
                    on_output             ▼
                                    on_guardrail

  memory load ──▶ ... ──▶ memory save
       │                    │
       ▼                    ▼
   on_memory            on_memory

  evaluator runs after each turn
       │
       ▼
  on_evaluator
```

## Prerequisites

| Requirement | Setup |
|---|---|
| **Ollama** | Install from [ollama.com](https://ollama.com), then run `ollama serve` |
| **Model** | `ollama pull gpt-oss:20b` (only real working model used) |
| **Dependencies** | `cd agent_harness_examples && uv sync` |

All other model names in the examples (`nonexistent-model-xyz`, `fail-1`, etc.) are **intentionally broken** to trigger errors. No env variables, MongoDB, Redis, Logfire, or OTLP required — all providers are mock.

## Examples

### 01 — Basic Handler

`uv run 2-error_handling/01_basic_handler.py`

Five sub-examples demonstrating the fundamental handler patterns:

1. **Suppress LLM errors** — Handler returns a string, error is suppressed. `result.success=False`, `result.output` contains the fallback message. Prints `ErrorContext` fields (source, type, message, session, prompt, stack presence).

2. **Re-raise** — Handler returns `None`, exception propagates. Wrapped in `try/except` to show the re-raised `Exception`.

3. **Catch-all `on_error`** — Lambda catches any source not matched by a specific handler.

4. **Successful run** — Handler is registered but never triggered because the model works. Shows that error handlers are only called on failure.

5. **Per-source routing** — Registers handlers for `llm`, `memory`, `prompt`, `output`, plus a catch-all. Demonstrates that errors route to the correct callback based on source.

**Models:** `nonexistent-model-xyz`, `also-broken-model`, `will-definitely-fail` (broken), `gpt-oss:20b` (working), `routing-test-fail` (broken).

```
Handler decision flow:
──────────────────────
Error occurs in agent.run()
    │
    ▼
Match error source to handler
    │
    ├──▶ on_llm_error / on_tool_error / ... (specific)
    │
    └──▶ on_error (catch-all, if no specific match)
         │
         ▼
    Handler callback(ErrorContext)
         │
         ├──▶ return string  →  suppress error
         │                      result.success=False
         │                      result.output = returned string
         │
         └──▶ return None    →  re-raise exception
                                exception propagates up
```

---

### 02 — Source Routing

`uv run 2-error_handling/02_source_routing.py`

Registers a dedicated handler for all 8 error sources plus a catch-all. Only one agent actually runs (with a broken model), triggering the `on_llm_error` handler. The rest of the file prints a reference table showing where each source originates in `agent.run()`:

| Source | Origination Point |
|---|---|
| `llm` | `agent.run()` inside `asyncio.wait_for` |
| `tool` | Tool function execution |
| `validation` | Output validator `ModelRetry` exhausted |
| `guardrail` | Circuit breaker, token/cost limits, content filter |
| `memory` | `message_history.load()`, `save_to` `save_turn()` |
| `prompt` | `get_system_prompt()`, Jinja2 render |
| `evaluator` | `evaluator.evaluate()` |
| `output` | Usage parsing, `TurnData`, `extract_clean_output` |

**Model:** `this-model-does-not-exist` (broken).

---

### 03 — Custom Recovery

`uv run 2-error_handling/03_custom_recovery.py`

Three sub-examples showing recovery patterns with logging:

1. **Per-source recovery** — `llm_recovery`, `memory_recovery`, `output_recovery` handlers each return a tailored fallback message. A `log_recovery()` function appends entries to an in-memory `error_log` list.

2. **Multiple sequential failures** — Loop of 3 agents with different broken model names (`fail-1`, `fail-2`, `fail-3`). Each gets a different fallback output. Demonstrates that the same config handles repeated failures.

3. **Stack trace capture** — `stack_inspector` handler reads `ErrorContext.stack_trace` and prints the first 4 frames of the traceback. Returns `None` to let a catch-all handler decide the final outcome.

Prints a full recovery log summary at the end with timestamps, source, error type, and action taken.

**Models:** `recovery-test-model-fail`, `fail-1/2/3`, `stack-trace-test-model` (all broken).

---

### 04 — Tool Errors

`uv run 2-error_handling/04_tool_errors.py`

Registers two tools:

- `broken_divider(a, b)` — raises `ValueError` when `b == 0`
- `stable_echo(message)` — always succeeds

**Run 1:** Agent is asked to divide 10 by 0. Tool raises `ValueError("Cannot divide 10 by zero")`. `on_tool_error` catches it, prints diagnostics, returns a fallback message. `result.success=False`.

**Run 2:** Agent divides 42 by 6. Tool succeeds. `result.success=True`.

**Requires tool-calling support.** Uses `gpt-oss:20b` which supports tool/function calling.

```
Run 1: divide by zero
─────────────────────
Agent ──▶ broken_divider(10, 0)
              │
              ▼
         ValueError("Cannot divide 10 by zero")
              │
              ▼
         on_tool_error callback
              │
              ├──▶ prints ErrorContext
              └──▶ returns fallback string
                       │
                       ▼
              result.success=False
              result.output="I'm sorry, the calculation tool..."

Run 2: valid division
─────────────────────
Agent ──▶ broken_divider(42, 6)  ──▶ "42 / 6 = 7.0"
              │
              ▼
         result.success=True
         result.output="42 / 6 = 7.0"
```

---

### 05 — Guardrail Errors

`uv run 2-error_handling/05_guardrail_errors.py`

Three sub-examples:

1. **Circuit breaker opens** — `CircuitBreakerConfig` with `threshold=1`. First run fails (broken model), circuit trips. Second run immediately gets a guardrail error. Without `on_error` on the `CircuitBreakerConfig`, the `RuntimeError` propagates.

2. **Circuit breaker with `on_error` callback** — `CircuitBreakerConfig().on_error(cb_on_error)` fires **before** `ErrorHandlingConfig`. Returns a fallback string. Second run succeeds with the CB fallback.

3. **Content filter callback raises** — `broken_filter` raises `RuntimeError("Filter service unavailable")`. Without `on_error` on `ContentFilterConfig`, the error propagates.

**Guardrail error types:** `CircuitBreakerOpen`, `TokenLimitExceeded`, `CostLimitExceeded`, `TurnLimitExceeded`, callback exceptions.

**Models:** `this-will-fail-immediately`, `another-broken-model` (broken), `gpt-oss:20b` (working).

```
Circuit breaker lifecycle:
──────────────────────────
Run 1: model fails ──▶ CB records failure (1/1)
                        │
                        ▼
                   Circuit OPENS
                        │
Run 2: ─────────────────▶ CB is OPEN
         │
         ▼
    RuntimeError("Circuit breaker is open")
         │
         ├──▶ on_error on CB config (if set) ──▶ fallback
         │
         └──▶ on_guardrail_error on ErrorHandlingConfig
                  │
                  ├──▶ return value ──▶ suppress
                  └──▶ return None  ──▶ re-raise

Guardrail error sources:
────────────────────────
┌──────────────────────┬─────────────────────────────────┐
│ Guardrail Type       │ When it fires                   │
├──────────────────────┼─────────────────────────────────┤
│ CircuitBreakerOpen   │ N consecutive failures          │
│ TokenLimitExceeded   │ Token usage cap reached         │
│ CostLimitExceeded    │ Dollar cost cap reached         │
│ TurnLimitExceeded    │ Session turn cap reached        │
│ Callback exception   │ Content filter / PII raises     │
└──────────────────────┴─────────────────────────────────┘
```

---

### 06 — Evaluator Errors

`uv run 2-error_handling/06_evaluator_errors.py`

Defines two evaluators:

- `FailingEvaluator` — always raises `RuntimeError("external service timeout")`
- `WorkingEvaluator` — logs output length (succeeds)

Three sub-examples:

1. **Suppress** — `on_evaluator_error` returns a value, suppressing the failure. Agent output is the normal LLM response.

2. **Mixed evaluators** — First evaluator raises, handler returns `None` (re-raise). The `RuntimeError` propagates and the second evaluator (`WorkingEvaluator`) never runs.

3. **Suppress with fallback** — Handler returns `"[Evaluator note]: quality check failed..."` which becomes part of the output. Agent continues normally.

Evaluator failures previously were silently swallowed; now they propagate to the error handler with `source="evaluator"`.

**Model:** `gpt-oss:20b` (working).

---

### 07 — Memory Errors

`uv run 2-error_handling/07_memory_errors.py`

Defines two mock providers:

- `FailingMemoryProvider` — raises `ConnectionError` on `save_turn()` (simulates DB outage)
- `FailingLoadProvider` — raises `ConnectionError` on `load_turns()` (simulates connection pool exhaustion)

Three sub-examples:

1. **`save_turn()` fails** — Short-term memory works (load succeeds), but `save_to=[broken_save]` causes `ConnectionError` during save. Handler suppresses silently. Agent output is the normal LLM response.

2. **`load_turns()` fails** — `FailingLoadProvider` as short-term memory raises during `MessageHistory().load()`. Handler returns a warning string. Agent runs with empty context.

3. **Save failure with fallback output** — Handler returns a visible warning (`"[Warning: persistence unavailable...]"`) which appears in the output.

**Model:** `gpt-oss:20b` (working). No real MongoDB or Redis — all mock providers.

---

### 08 — Prompt Errors

`uv run 2-error_handling/08_prompt_errors.py`

`FailingPromptProvider` raises `RuntimeError` on `get_system_prompt()` (simulates MongoDB down or invalid Jinja2 template).

Two sub-examples:

1. **Re-raise** — Handler returns `None`. `RuntimeError` propagates. Prompts are treated as critical — can't continue without them.

2. **Suppress with fallback** — Handler returns `None` (suppress). Agent runs without a system prompt. Output shows the agent responded despite the missing prompt.

Also includes a conceptual section about two failure modes of `MongoPrompts`: `ConnectionError` (MongoDB unreachable) and `ValueError` (invalid Jinja2 template like `{{undefined_variable}}`).

**Model:** `gpt-oss:20b` (working). No real MongoDB needed — all mock.

---

### 09 — Pipeline Error Recovery

`uv run 2-error_handling/09_pipeline_error_recovery.py`

A three-agent pipeline (Research → Analysis → Summary):

1. **Research** — Agent runs successfully, produces facts about embeddings.

2. **Analysis** — `FailingPromptProvider` raises `RuntimeError("Agent 2 prompt service unavailable")`. `analysis_fallback` handler suppresses it and returns `"[Recovered] Analysis stage failed: ..."`. `r2.success=False`.

3. **Summary** — Agent runs with both `r1.output` (research) and `r2.output` (the recovery fallback message) concatenated into the prompt. Produces a summary.

`PipelineContext` tracks every stage and prints a full trace at the end showing success/failure status and outputs.

**Model:** `gpt-oss:20b` (all 3 agents).

```
Three-agent pipeline with mid-stage failure:
─────────────────────────────────────────────

Stage 1: Research
    │
    ▼
Agent.run("Give 3 key facts about embeddings...")
    │
    ▼
  r1.success=True
  r1.output="Embeddings are vector representations..."
    │
    ▼
Stage 2: Analysis  ◀── FailingPromptProvider (RuntimeError)
    │
    ▼
on_prompt_error ──▶ analysis_fallback
    │
    ▼
  r2.success=False
  r2.output="[Recovered] Analysis stage failed: ..."
    │
    ▼
Stage 3: Summary
    │
    ▼
Agent.run("Summarize the following...\n\nResearch: {r1}\nAnalysis: {r2}")
    │
    ▼
  r3.success=True
  r3.output="Embeddings are vector representations used in ML..."

PipelineContext trace:
┌──────────┬──────────┬──────────────────────────────────┐
│ Stage    │ Status   │ Output                           │
├──────────┼──────────┼──────────────────────────────────┤
│ Research │ OK       │ Embeddings are vector...          │
│ Analysis │ FAILED   │ [Recovered] Analysis stage failed │
│ Summary  │ OK       │ Embeddings are vector...          │
└──────────┴──────────┴──────────────────────────────────┘
```

## Running

All commands from the `agent_harness_examples/` directory:

```bash
cd agent_harness_examples
uv sync
uv run 2-error_handling/01_basic_handler.py
uv run 2-error_handling/02_source_routing.py
uv run 2-error_handling/03_custom_recovery.py
uv run 2-error_handling/04_tool_errors.py
uv run 2-error_handling/05_guardrail_errors.py
uv run 2-error_handling/06_evaluator_errors.py
uv run 2-error_handling/07_memory_errors.py
uv run 2-error_handling/08_prompt_errors.py
uv run 2-error_handling/09_pipeline_error_recovery.py
```
