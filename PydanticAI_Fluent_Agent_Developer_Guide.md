# PydanticAI-Fluent Agent Harness Examples
## Junior Developer AI Engineering & Production Guide

**Repository:** `myndfire/pydanticai-fluent`  
**Scope:** `agent_harness_examples/` on the `master` branch  
**Audience:** Junior developers learning AI engineering  
**Objective:** Explain the intent of every example folder and file, the fluent ManagedAgent configuration used in the examples, and the production engineering lessons behind each feature.

> **Teaching principle:** Do not learn these examples as syntax recipes. Learn the engineering problem each feature solves, how the harness composes the solution, how it can fail, and what must change before production deployment.

---

# 1. Why This Folder Exists

`agent_harness_examples` is a progressive set of runnable examples for learning how an AI-enabled application grows from a simple model call into a production-oriented agent.

A beginner often starts with:

```text
prompt -> model -> response
```

A production agent is closer to:

```text
User / API / Event
        |
        v
+-----------------------+
|     ManagedAgent      |
+-----------------------+
 |      |       |      |
 v      v       v      v
Model  Prompt  Memory  Tools
 |                      |
 +----------+-----------+
            v
       LLM execution
            |
   +--------+---------+---------+-----------+
   |                  |         |           |
   v                  v         v           v
Structured Output  Guardrails  Evals   Observability
   |                  |         |           |
   +------------------+---------+-----------+
                      |
                      v
             Application result
```

The examples teach the concerns that make this reliable:

- model configuration
- prompt management
- tool calling
- dependency/context injection
- memory
- structured outputs
- retries
- error handling
- guardrails
- human approval
- evaluation
- loops
- multi-agent orchestration
- retrieval/RAG
- logging
- traces and metrics
- asynchronous messaging

---

# 2. Repository Navigation Note

The current repository uses numbered topic directories:

```text
01-getting_started
02-error_handling
03-evaluators
04-guardrails
05-human_in_the_loop
06-loops
07-orchestration
08-structured_output
09-tool_calling
10-prompts
11-memory
12-observability
13-rag
14-logging
15-messaging
```

The root README still contains some older paths such as `loops/...`, `tools/...`, `memory/...`, and `messaging/rabbitmq/...`.

**When the README and actual directory tree disagree, treat the actual directory tree as authoritative.**

This matters for a junior developer because copying a stale path can make a correct example appear broken.

---

# 3. The ManagedAgent Fluent Mental Model

Most examples build an agent with method chaining:

```python
agent = (
    ManagedAgent()
    .with_model(...)
    .with_short_term_memory(...)
    .with_long_term_memory(...)
    .with_tools(...)
    .with_prompts(...)
    .with_output(...)
    .with_observability(...)
    .with_error_handling(...)
    .with_evaluators(...)
)
```

Read this as **composition**.

Each `.with_*()` call contributes one capability. The purpose of the fluent API is to make the agent's architecture visible at construction time.

## 3.1 `ManagedAgent()`

Creates the orchestration object.

It coordinates the configured model, memory, prompt provider, tools, output schema, guardrails, evaluators, and telemetry.

**Production rule:** `ManagedAgent` should orchestrate AI behavior. It should not become your database layer, authorization system, web API, or business domain.

## 3.2 `.with_model(ModelConfig(...))`

Example:

```python
.with_model(
    ModelConfig(
        provider="ollama",
        model_name=os.getenv("MODEL_NAME", "qwen2.5:3b"),
    )
)
```

Parameters:

| Parameter | Meaning |
|---|---|
| `provider` | Which model integration/provider to use. |
| `model_name` | Provider-specific model identifier. |

**Production rule:** Read model/provider settings from configuration. Do not hard-code model choices throughout business code.

## 3.3 `.with_short_term_memory(provider)`

Attaches conversation/session memory intended for recent turns.

Example:

```python
InMemoryProvider(max_turns=10)
```

`max_turns` bounds retained recent turns.

**Production rule:** In-memory state is process-local and disappears on restart. Use a shared/persistent backend when multiple application instances or durable sessions matter.

## 3.4 `.with_long_term_memory(provider)`

Adds longer-lived memory/history storage.

Examples use providers such as MongoDB.

**Production rule:** Long-term memory is user/application data. Apply authorization, tenant isolation, encryption, retention, deletion, and audit requirements.

## 3.5 `.with_tools(ToolRegistry(...))`

Registers Python functions the model may call.

```text
LLM chooses an action
      |
      v
typed tool request
      |
      v
deterministic Python/service code
```

**Production rule:** The model may choose *when* to invoke a tool, but authorization and side-effect controls must remain deterministic.

## 3.6 `.with_prompts(provider)`

Examples include:

```python
StaticPrompts("You are a helpful assistant")
```

and Mongo-backed prompt templates.

**Production rule:** Prompts are behavioral configuration. Version, review, test, and trace them.

## 3.7 `.with_output(Model, output_retries=N)`

Constrains the result to a Pydantic model.

`output_retries` bounds automatic correction attempts when generated output does not validate.

**Production rule:** Prefer structured output whenever another software component consumes the response.

## 3.8 `.with_observability(observability)`

Attaches logs, traces, and/or metrics.

**Production rule:** If you cannot observe an agent's model calls, tool calls, latency, failures, and token usage, you cannot reliably operate it.

## 3.9 `.with_error_handling(ErrorHandlingConfig(...))`

Configures failure routing and fallback behavior for sources such as:

- LLM
- tools
- validation
- guardrails
- memory
- prompts
- evaluators
- output

**Production rule:** Different failures require different responses. A database outage is not the same as a safety-policy rejection.

## 3.10 `.with_evaluators(...)`

Registers post-turn evaluators.

```python
.with_evaluators(eval1, eval2, eval3)
```

Evaluators observe the prompt/result/context after a run.

**Production rule:** Evaluation is not a substitute for deterministic validation or authorization.

---

# 4. Root-Level Files

## `README.md`

Intent:

- quick-start instructions
- infrastructure dependencies
- model setup
- observability stack
- build-your-own-project instructions
- example index

The root README demonstrates the minimal construction pattern:

```python
ManagedAgent()
    .with_model(...)
    .with_short_term_memory(...)
    .with_prompts(...)
```

It also documents infrastructure such as MongoDB, Redis, Elasticsearch, Grafana, Prometheus, Jaeger, OTel Collector, Pushgateway, and RabbitMQ.

**Junior lesson:** This file is the map of the examples, not a full production deployment manual.

## `.env.example`

Provides environment-variable examples.

**Junior lesson:** Configuration belongs outside source code.

**Production upgrade:** Secrets should come from a secret-management/deployment system, not committed `.env` files.

## `pyproject.toml`

Defines Python dependencies and project metadata.

**Junior lesson:** Understand exactly which packages are needed to run examples.

## `uv.lock`

Locks resolved dependency versions.

**Production lesson:** Lock files improve reproducibility.

## `data/`

Contains teaching fixtures used by examples.

**Production lesson:** Real user documents require validation, access control, retention, size limits, and data-classification policy.

---

# 5. `01-getting_started`

## Folder intent

Three introductory examples move from tools, to a more observable multi-turn agent, to typed output.

### Files

- `README.md`
- `agent_example-1.py`
- `agent_example-2.py`
- `agent_example-3.py`

## `README.md`

Explains prerequisites, environment variables, execution commands, and the conceptual difference among the three starter agents.

---

## `agent_example-1.py` — Minimal Agent with Tools

### Intent

Demonstrates the smallest useful ManagedAgent composition with two tools:

```python
repeat(text)
shout(text)
```

The prompt explicitly asks the model to call `repeat`, then pass the result into `shout`.

### Fluent construction

```python
ManagedAgent()
    .with_model(ModelConfig(...))
    .with_short_term_memory(short_term)
    .with_long_term_memory(long_term)
    .with_tools(tools)
    .with_prompts(StaticPrompts(...))
    .with_observability(Observability())
    .with_error_handling(ErrorHandlingConfig())
    .with_evaluators(PrintEvaluator())
```

### Every fluent element

| Call | Example value | What it contributes |
|---|---|---|
| `with_model` | `ModelConfig(provider="ollama", model_name=...)` | Chooses the LLM. |
| `with_short_term_memory` | `InMemoryProvider()` | Recent conversation state. |
| `with_long_term_memory` | `InMemoryProvider()` | Demonstrates a second memory role. |
| `with_tools` | `ToolRegistry().add_many(repeat, shout)` | Gives the model deterministic actions. |
| `with_prompts` | `StaticPrompts(...)` | Defines agent behavior. |
| `with_observability` | `Observability()` | Hooks telemetry into execution. |
| `with_error_handling` | `ErrorHandlingConfig()` | Enables explicit failure routing. |
| `with_evaluators` | `PrintEvaluator()` | Demonstrates post-run observation. |

### Related non-fluent APIs

```python
MessageHistory().load("demo-session", short_term)
```

Loads conversation history for a session.

```python
agent.run(prompt, history, "demo-session")
```

Runs one agent turn.

### Production upgrade

- Replace in-memory long-term storage.
- Add real telemetry providers.
- Add tool authorization.
- Add tool timeouts.
- Define failure callbacks.
- Test that the correct tool sequence occurs.

---

## `agent_example-2.py` — Memory + Observability + Error Handling

### Intent

Demonstrates a more operational agent:

- three sequential prompts
- same `session_id`
- bounded short-term memory
- optional MongoDB long-term memory
- console logging
- Logfire tracing
- OTEL tracing/metrics
- explicit error handling
- PydanticAI `ModelSettings`

### Key memory parameters

```python
InMemoryProvider(max_turns=10)
```

`max_turns` limits the short-term memory window.

Mongo memory parameters:

| Parameter | Purpose |
|---|---|
| `uri` | MongoDB connection string. |
| `database` | Database name. |
| `collection` | Collection holding conversation data. |

### Production upgrade

Define:

- session ownership
- tenant isolation
- retention policy
- timeout/retry behavior
- backend connection pooling
- what happens if long-term memory fails
- which telemetry fields are safe to record

---

## `agent_example-3.py` — Structured Invoice Output

### Intent

Converts natural-language invoice information into a typed Pydantic `Invoice`.

It also demonstrates extracting structured data from invoice content.

### Core fluent capability

```python
.with_output(Invoice)
```

The model must satisfy the Pydantic schema rather than return arbitrary prose.

### Why it matters

```text
Natural language
      |
      v
LLM
      |
      v
Pydantic validation
      |
      v
Invoice instance
```

### Production upgrade

Schema validity is only the first level of correctness. Add domain validation such as:

- subtotal equals service totals
- tax calculation is valid
- total amount due is consistent
- dates are sensible
- currency is allowed

---

# 6. `02-error_handling`

## Folder intent

Nine examples teach source-aware failure handling.

### Files

- `01_basic_handler.py`
- `02_source_routing.py`
- `03_custom_recovery.py`
- `04_tool_errors.py`
- `05_guardrail_errors.py`
- `06_evaluator_errors.py`
- `07_memory_errors.py`
- `08_prompt_errors.py`
- `09_pipeline_error_recovery.py`
- `README.md`

## Core API

```python
ErrorHandlingConfig()
    .on_llm_error(...)
    .on_tool_error(...)
    .on_validation_error(...)
    .on_guardrail_error(...)
    .on_memory_error(...)
    .on_prompt_error(...)
    .on_evaluator_error(...)
    .on_output_error(...)
    .on_error(...)
```

A handler receives `ErrorContext`.

Important context fields include the failure source, exception type/message, session/prompt context, and stack information.

A returned fallback value suppresses the failure; returning `None` allows the exception to propagate.

## `01_basic_handler.py`

Introduces source-specific callbacks, fallback values, and the catch-all handler.

**Junior lesson:** Handling an exception is a design decision, not merely `try/except`.

## `02_source_routing.py`

Shows different policies for different failure sources.

Example policy model:

```text
LLM outage      -> fallback/retry
tool failure    -> degraded feature
memory failure  -> stateless mode only if safe
guardrail       -> stop/redact
prompt failure  -> fallback prompt
evaluator fail  -> preserve user result + alert
```

## `03_custom_recovery.py`

Builds a richer recovery strategy that records diagnostic context and returns degraded output.

**Production rule:** Never silently recover. Record what happened.

## `04_tool_errors.py`

Triggers errors from registered tools and routes them to `on_tool_error`.

**Production rule:** Side-effecting tools may have partially succeeded. Use idempotency/transaction semantics.

## `05_guardrail_errors.py`

Demonstrates guardrail-originated failures.

**Production rule:** A policy rejection is expected control flow, not necessarily an application crash.

## `06_evaluator_errors.py`

Demonstrates evaluator exceptions routed through `on_evaluator_error`.

**Production rule:** Quality telemetry generally should not destroy an otherwise valid user result, but repeated evaluator failure must be visible.

## `07_memory_errors.py`

Uses a failing memory provider to exercise `on_memory_error`.

**Production rule:** Decide explicitly whether stateless degradation is safe.

## `08_prompt_errors.py`

Uses a failing prompt provider to exercise `on_prompt_error`.

**Production rule:** Validate prompt templates before deployment and retain known-good versions.

## `09_pipeline_error_recovery.py`

Demonstrates a multi-stage pipeline where one stage fails but the workflow continues.

**Production rule:** Define fail-fast, partial-success, resume, compensation, and retry semantics for each stage.

---

# 7. `03-evaluators`

## Folder intent

Five examples show post-turn evaluation.

### Files

- `01_quality_check.py`
- `02_safety_check.py`
- `03_custom_evaluator.py`
- `04_protocol_evaluator.py`
- `05_combined_evaluators.py`
- `README.md`

## Core fluent API

```python
.with_evaluators(eval1, eval2, ...)
```

## `01_quality_check.py`

Uses an LLM-as-judge quality evaluator.

Important concepts include:

- quality threshold
- judge model
- response scoring

**Production rule:** Calibrate judge behavior against human-labeled examples.

## `02_safety_check.py`

Uses a safety/moderation evaluator.

**Production rule:** Define what happens when the moderation service itself is unavailable.

## `03_custom_evaluator.py`

Demonstrates organization-specific evaluators such as:

- response length
- required terms
- turn counting

**Production rule:** Custom evaluators are useful for measurable business-quality requirements.

## `04_protocol_evaluator.py`

Shows that an evaluator can satisfy the expected protocol without inheriting from a particular framework base class.

**Engineering lesson:** Protocol-based design reduces coupling.

## `05_combined_evaluators.py`

Registers several evaluators on the same agent.

```python
.with_evaluators(eval1, eval2, eval3)
```

**Production rule:** Keep quality, safety, compliance, and telemetry evaluators separate.

---

# 8. `04-guardrails`

## Folder intent

Ten examples teach bounded, safer agent execution.

### Files

- `01_agent_retries.py`
- `02_tool_retries.py`
- `03_result_validator_retries.py`
- `04_content_filter.py`
- `05_pii_detection.py`
- `06_token_limits.py`
- `07_cost_limits.py`
- `08_circuit_breaker.py`
- `09_all_guardrails.py`
- `10_turn_limits.py`
- `README.md`

## `01_agent_retries.py`

Uses `AgentRetryConfig`.

Typical concepts:

- maximum attempts
- timeout
- retry callback
- error callback
- fallback model
- backoff

**Production rule:** Retry transient failures only.

## `02_tool_retries.py`

Uses both agent-level and tool-level retry policy.

**Production rule:** Retried side-effecting tools must be idempotent or protected by an idempotency key.

## `03_result_validator_retries.py`

Retries when the result violates validation rules.

**Production rule:** Use validation retries for correctable model errors, not for infrastructure failure.

## `04_content_filter.py`

Uses `ContentFilterConfig` and an `on_filter` callback.

**Production rule:** Regex-based demo filtering is not a complete enterprise moderation system.

## `05_pii_detection.py`

Uses `PIIDetectionConfig` and redaction behavior.

Common demo patterns include email, phone, SSN, card, and IP-like data.

**Production rule:** Real PII handling requires policy, jurisdiction awareness, DLP controls, and auditability.

## `06_token_limits.py`

Uses `TokenLimitsConfig`.

Conceptual parameters:

| Setting | Meaning |
|---|---|
| maximum input tokens | Bounds prompt/context size. |
| maximum output tokens | Bounds generated response size. |
| maximum total tokens | Bounds combined model usage. |
| token-limit callback | Defines behavior when exceeded. |

ManagedAgent integration:

```python
.with_token_limits(limits)
```

## `07_cost_limits.py`

Uses `CostLimitsConfig`.

Typical fields represent input/output token pricing and maximum allowed cost.

**Production rule:** Keep pricing in configuration and update it when provider prices change.

## `08_circuit_breaker.py`

Uses `CircuitBreakerConfig`.

Concepts:

- failure threshold
- open timeout
- half-open recovery

```text
CLOSED -> OPEN -> HALF_OPEN -> CLOSED
```

## `09_all_guardrails.py`

Composes multiple guardrail configurations on one agent.

**Junior lesson:** Production safety is layered.

## `10_turn_limits.py`

Uses `TurnLimitsConfig` to cap agent/session turns.

**Production rule:** Every autonomous or iterative agent needs a deterministic stopping bound.

---

# 9. `05-human_in_the_loop`

## Folder intent

Three examples show increasingly granular human intervention.

### Files

- `01_review_approval.py`
- `02_tool_review.py`
- `03_multistep_tools.py`
- `README.md`

## `01_review_approval.py`

Uses a filter/review callback to approve or replace the final model response.

**Production upgrade:** Persist approval requests. Do not block a web worker with terminal `input()`.

## `02_tool_review.py`

Adds a tool-driven workflow and human review around the resulting response.

**Production distinction:** Reviewing text is different from approving a consequential action.

## `03_multistep_tools.py`

Introduces approval at multiple tool steps.

**Production rule:** Approval should happen before irreversible side effects.

---

# 10. `06-loops`

## Folder intent

Five patterns show who controls repetition and how the loop stops.

### Files

- `01_interactive_loop.py`
- `02_iterative_refinement.py`
- `03_react_loop.py`
- `04_goal_seeking_loop.py`
- `05_planning_loop.py`
- `README.md`
- `text_to_summarize.txt`

## `01_interactive_loop.py`

User-driven REPL/chat loop.

Common infrastructure:

- `ManagedAgent`
- `InMemoryProvider`
- `MessageHistory`
- `StaticPrompts`

**Junior lesson:** The UI/application owns the conversation loop; the agent performs a turn.

## `02_iterative_refinement.py`

Program-driven refinement loop.

A deterministic condition/evaluator decides whether another attempt is needed.

**Production rule:** Prefer deterministic stopping rules where possible.

## `03_react_loop.py`

Agent-driven reasoning/action loop using tools.

**Production rule:** Bound tool calls, time, tokens, cost, and iterations.

## `04_goal_seeking_loop.py`

Continues until measurable success criteria are satisfied or a maximum attempt count is reached.

**Production rule:** Express success as deterministic data whenever possible.

## `05_planning_loop.py`

Separates planning from step execution.

```text
Goal -> Plan -> Execute steps -> Final result
```

**Production upgrade:** Persist the plan and each step's state for resumable workflows.

## `text_to_summarize.txt`

Input fixture used by a loop example.

---

# 11. `07-orchestration`

## Folder intent

Four patterns compose multiple ManagedAgents.

### Files

- `01_delegation.py`
- `02_sequential_pipeline.py`
- `03_routing.py`
- `04_parallel_fanout.py`
- `README.md`

## `01_delegation.py`

A coordinator delegates a specialist task through a tool.

Important concept:

```python
deps_type=SharedContext
```

`deps_type` defines typed runtime dependencies accessible through `RunContext`.

**Production use:** request-scoped services, tenant identity, authenticated user context, service clients.

**Security rule:** The model must not create or choose its own authorization context.

## `02_sequential_pipeline.py`

Runs agents in sequence:

```text
Researcher -> Writer -> Editor
```

**Production upgrade:** Persist stage status and define retry/resume behavior.

## `03_routing.py`

One agent classifies the request; normal Python dispatches to the selected specialist.

**Engineering lesson:** Structured classify-then-route is easier to reason about than giving one general agent every tool.

## `04_parallel_fanout.py`

Runs independent specialists concurrently and combines outputs.

**Production rule:** Add concurrency limits, total cost limits, timeouts, and partial-failure policy.

---

# 12. `08-structured_output`

## Folder intent

Four examples teach typed Pydantic outputs.

### Files

- `01_simple_model.py`
- `02_enums_literals.py`
- `03_nested_models.py`
- `04_validation_retries.py`
- `README.md`

## Core fluent API

```python
.with_output(SomeModel)
```

or:

```python
.with_output(SomeModel, output_retries=3)
```

## `01_simple_model.py`

Returns a typed model such as `WeatherReport`.

`output_retries` bounds attempts to correct invalid generated data.

## `02_enums_literals.py`

Uses finite values such as `Literal[...]`.

**Junior lesson:** If the answer belongs to a finite set, encode that constraint in the type.

## `03_nested_models.py`

Demonstrates hierarchical schemas such as recipes containing ingredients and instruction steps.

**Production lesson:** Nested schemas are excellent application contracts, but avoid unnecessary complexity.

## `04_validation_retries.py`

Adds custom business-rule validation and retry behavior.

**Production rule:** Separate:
1. type/schema validation;
2. domain/business validation.

---

# 13. `09-tool_calling`

## Folder intent

Four examples progress from plain Python functions to context-aware and external tools.

### Files

- `01_plain_tools.py`
- `02_context_tools.py`
- `03_mcp_server.py`
- `04_tool_combinations.py`
- `README.md`

## `01_plain_tools.py`

Uses `ToolRegistry`.

Important methods:

| API | Purpose |
|---|---|
| `.add(fn)` | Register one function. |
| `.add_many(fn1, fn2, ...)` | Register several functions. |
| `.clear()` | Remove registrations. |
| `.get_tools()` | Inspect current registrations. |

The registry uses type hints and docstrings to describe the callable to the LLM.

**Production rule:** Tool signatures are APIs. Treat them accordingly.

## `02_context_tools.py`

Uses:

```python
RunContext[UserDeps]
```

for context-aware tools.

**Production use:** Inject authenticated identity, tenant context, data-access services, correlation IDs, and policy services.

## `03_mcp_server.py`

Demonstrates MCP-provided tools. Example 4 combines two public MCP servers — Context7 (library documentation) and Complex server (data/user/order operations) — with `tool_prefix` disambiguation. Example 5 runs a live demo connecting to Context7, resolving a library ID and fetching a docs excerpt using MCP tools at runtime.

**Production rule:** MCP expands the agent's capability boundary. Use authentication, allow-lists, timeouts, network controls, and audit logging.

## `04_tool_combinations.py`

Combines tools with guards/evaluators/other capabilities.

**Engineering lesson:** Tools must be surrounded by policy, validation, and observability.

---

# 14. `10-prompts`

## Folder intent

Three examples separate prompt behavior from agent code.

### Files

- `01_static_prompts.py`
- `02_mongo_prompts.py`
- `03_prompt_variables.py`
- `README.md`

## `01_static_prompts.py`

Uses:

```python
StaticPrompts("...")
.with_prompts(prompts)
```

**Production rule:** Even static prompts should be version-controlled and tested.

## `02_mongo_prompts.py`

Uses Mongo-backed prompt templates with Jinja2-style rendering.

Concepts include:

- persistence
- prompt IDs
- versions
- dynamic templates
- caching
- CRUD

**Production rule:** Prompt stores need version history, rollback, change review, and environment promotion.

## `03_prompt_variables.py`

Passes variables used to render a selected prompt template.

**Production rule:** Treat external prompt variables as untrusted input and prevent them from bypassing system policy.

---

# 15. `11-memory`

## Folder intent

Nine examples teach session/history persistence across several providers.

### Files

- `01_in_memory.py`
- `02_message_history.py`
- `03_multi_provider.py`
- `04_memory_operations.py`
- `05_mongo_memory.py`
- `06_redis_memory.py`
- `07_combined_memory.py`
- `08_elasticsearch_memory.py`
- `09_reasoning_traces.py`
- `README.md`

## `01_in_memory.py`

Uses `InMemoryProvider`.

Important parameter:

```python
max_turns=N
```

Also demonstrates selecting memory destinations with a concept such as:

```python
save_to=[provider1, provider2]
```

## `02_message_history.py`

Uses:

```python
MessageHistory().load(session_id, provider)
```

to reconstruct previous messages.

**Production rule:** Do not blindly send all stored history back to the model. Apply selection/summarization/token budgeting.

## `03_multi_provider.py`

Uses multiple memory providers for distinct roles.

**Production lesson:** Recent context, durable history, and audit history may deserve different stores and retention policies.

## `04_memory_operations.py`

Demonstrates provider management operations such as reading, deletion, clearing, and trimming/limits.

**Production rule:** Data deletion and retention are first-class features.

## `05_mongo_memory.py`

Uses MongoDB for durable conversation data.

**Production upgrade:** Add indexes, pooling, retry policy, tenant filters, backups, retention, and access control.

## `06_redis_memory.py`

Uses Redis for fast session storage.

**Production rule:** Define TTL/eviction/durability expectations.

## `07_combined_memory.py`

Combines Redis and MongoDB.

Common architectural role:

```text
Redis -> recent fast state
Mongo -> durable archive
```

**Production rule:** Define consistency behavior if one storage write succeeds and another fails.

## `08_elasticsearch_memory.py`

Uses Elasticsearch for searchable history.

**Production rule:** Apply correct mappings, index lifecycle management, and mandatory tenant filters.

## `09_reasoning_traces.py`

Inspects reasoning-related stored message parts.

**Production warning:** Application logic should not depend on private chain-of-thought. Store auditable decisions, tool calls, summaries, and outcomes instead.

---

# 16. `12-observability`

## Folder intent

Nine examples build from simple logging to a complete OpenTelemetry pipeline.

### Files

- `01_logging.py`
- `02_tracing_metrics.py`
- `03_builder_logs_metrics.py`
- `04_composite_logs.py`
- `05_elasticsearch_logging.py`
- `06_otel_jaeger_logs_traces_metrics.py`
- `07_prometheus_logs_metrics.py`
- `08_live_agent_logs_metrics.py`
- `09_otel_oltp_logs_traces_metrics.py`
- `README.md`

## `01_logging.py`

Demonstrates console and file log providers.

For a rotating file logger, concepts include:

- file path
- rotation
- retention

**Production rule:** Prefer structured log fields.

## `02_tracing_metrics.py`

Uses in-memory tracer/metrics providers so a developer can inspect emitted telemetry directly.

**Production rule:** In-memory telemetry is primarily for tests and learning.

## `03_builder_logs_metrics.py`

Uses a fluent observability builder to combine log/metric providers, then attaches them to a real ManagedAgent.

**Engineering lesson:** Compose infrastructure in one place.

## `04_composite_logs.py`

Fans one event out to multiple logging destinations.

**Engineering lesson:** Business code should not know which log backend is active.

## `05_elasticsearch_logging.py`

Writes structured logs into Elasticsearch.

**Production upgrade:** Configure templates/mappings, ILM/data streams, authentication, restricted access, and retention.

## `06_otel_jaeger_logs_traces_metrics.py`

Shows OpenTelemetry-style telemetry and Jaeger integration.

**Production rule:** Correlate top-level agent runs, model calls, and tool spans.

## `07_prometheus_logs_metrics.py`

Demonstrates Prometheus-oriented metrics.

Useful production measurements:

- agent runs
- success/failure
- duration
- model latency
- input/output tokens
- tool calls
- retry count
- guardrail blocks
- evaluator failures

## `08_live_agent_logs_metrics.py`

Runs a real agent while emitting composed logging and metrics.

**Engineering lesson:** Telemetry must be tested on the real execution path.

## `09_otel_oltp_logs_traces_metrics.py`

Most complete observability example.

Conceptual architecture:

```text
ManagedAgent
    |
    v
OTel logs + traces + metrics
    |
    v
OTel Collector
  /     |        v      v        v
ES   Prometheus  Jaeger
 \      |        /
       Grafana
```

**Production starting point:** This is one of the strongest examples for operational diagnostics.

## OpenTelemetry & the OTel Collector

OpenTelemetry (OTEL) is the de-facto standard transport for logs, metrics, and
traces in this project. **Every OTEL backend — `OTELLogger`, `OTELTracer`,
`OTELMetrics` — exports over OTLP to a single OpenTelemetry Collector**, which
then fans each signal out to the appropriate backend. This is the architecture
used by `09_otel_oltp_logs_traces_metrics.py` and recommended for production.

### Why route through a collector?

Exporting telemetry straight from the application to each backend (logs to
Elasticsearch, traces to Jaeger, metrics to Prometheus) couples your code to
the backend topology and multiplies outbound connections, auth configs, and
failure modes. A collector centralizes that responsibility:

- **One egress point.** The app only needs to reach the collector over OTLP
  (gRPC `4317` / HTTP `4318`). The collector owns downstream routing, batching,
  retries, and backend credentials.
- **Backend substitution without code changes.** Swap Elasticsearch for Loki,
  or Jaeger for Tempo/Datadog, by editing the collector config — not your
  Python code.
- **Resilience.** The collector buffers and retries, so a briefly unavailable
  backend does not drop telemetry or back-pressure your agent.
- **Unified pipelines.** One receiver accepts all three signals and routes them
  through consistent processing (batching, resource attributes, multi-tenancy).

### The OTel Collector in this repository

`docker-compose.yml` publishes the collector on the standard OTLP ports:

| Port | Purpose |
|------|---------|
| `4317` | OTLP gRPC (logs, metrics, traces) |
| `4318` | OTLP HTTP (logs, metrics, traces) |

Jaeger is exposed on `14317`/`14318` **only as a downstream target of the
collector** — it is not meant to be hit directly by the application.

`otel-collector-config.yml` defines three pipelines off a single `otlp`
receiver:

```yaml
service:
  pipelines:
    traces:   receivers: [otlp]  exporters: [otlp/jaeger, otlphttp/elasticsearch]
    metrics:  receivers: [otlp]  exporters: [otlphttp/prometheus]
    logs:     receivers: [otlp]  exporters: [otlphttp/elasticsearch]
```

- **Logs** → Elasticsearch (`/_otlp/v1/logs`), stored in the
  `logs-generic.otel-default` data stream.
- **Metrics** → Prometheus via its native OTLP receiver
  (`/api/v1/otlp/v1/metrics`).
- **Traces** → Jaeger (`jaeger:4317`).

### Pointing the backends at the collector

All OTEL backends take an `otlp_endpoint`. The library default is
`localhost:4317` (gRPC), which is exactly where the collector listens:

```python
OTELMetrics(service_name="agent")                                   # → localhost:4317
OTELTracer(service_name="agent", otlp_endpoint="localhost:4317")
OTELLogger(service_name="agent", otlp_endpoint="localhost:4317")
```

In the examples this is driven by the `OTEL_COLLECTOR_ENDPOINT` environment
variable (the older `JAEGER_OTLP_ENDPOINT` is now deprecated).

### Log / trace correlation

Because logs and traces share the same OTLP stream, records emitted inside an
active span automatically carry `trace_id` / `span_id`. Jaeger's trace→logs link
and Elasticsearch queries on `trace_id` let you jump from a slow span to the
exact log lines that produced it.

### Production best practice

> **Use the OTel Collector as the single telemetry gateway in production.**
> Configure the application to export OTLP to the collector and let the
> collector own all backend connectivity. Do **not** point `OTELLogger` /
> `OTELTracer` / `OTELMetrics` directly at the individual open-source backends
> (Elasticsearch/Jaeger/Prometheus) in a production deployment — direct-to-
> backend export is acceptable for a local demo but forfeits the buffering,
> retry, and backend-portability benefits above. (Logfire is a managed service
> and exports to Logfire's cloud by design, which is expected.)

Operational notes:

- Run the collector as a sidecar or a dedicated service with its own resource
  limits; it is critical infrastructure for observability.
- Secure the OTLP endpoint (mTLS / network policy) — it receives potentially
  sensitive structured logs and traces.
- Keep `otel-collector-config.yml` in version control; it is your telemetry
  contract.

---

# 17. `13-rag`

## Folder intent

An end-to-end agentic retrieval example.

### Files

- `README.md`
- `agent.py`
- `pyproject.toml`
- `uv.lock`
- `sample_data/`

## `agent.py`

Builds a medical-assistant-style ManagedAgent with retrieval tools, guards, observability, and evaluators.

Retrieval tools expose specific categories of information.

Conceptual flow:

```text
Question
   |
   v
ManagedAgent
   |
   +--> retrieval tool
   |       |
   |       v
   |    source data
   |
   v
LLM synthesis
   |
   v
guards + evals + telemetry
```

### Production RAG upgrade

1. Authenticate the user.
2. Establish tenant scope.
3. Authorize documents before retrieval.
4. Search only authorized corpora.
5. Return source metadata.
6. Bound context size.
7. Defend against prompt injection in retrieved documents.
8. Trace retrieval/tool activity.
9. Redact sensitive telemetry.
10. Add evidence/citation behavior for factual responses.

## `sample_data/`

Teaching documents used by retrieval tools.

## `pyproject.toml` / `uv.lock`

Project-specific dependencies and lock state.

---

# 18. `14-logging`

## Folder intent

Three examples focus on structured **log enrichment**.

### Files

- `01_basic_log_context.py`
- `02_custom_enricher.py`
- `03_pipeline_logging.py`

## `01_basic_log_context.py`

Demonstrates:

```python
.with_log_enrichment(...)
```

with a `LogContext` and environment enrichment.

Typical pattern:

```python
LogContext()
    .with_("pipeline", "content-qa")
    .with_("agent_role", "assistant")
```

Per-run context can also be added during `agent.run(...)`.

### What the parameters mean

| Element | Purpose |
|---|---|
| `with_log_enrichment(*providers)` | Persistent enrichment applied across runs. |
| `LogContext().with_(key, value)` | Adds a structured field. |
| environment enricher | Adds runtime host/process/environment details. |
| run-level enrichment | Adds values specific to one execution. |

Recommended production fields:

- `tenant_id`
- `request_id`
- `session_id`
- `workflow`
- `stage`
- `agent_name`
- `agent_version`
- `tool_name`
- `model`

Never enrich logs with unrestricted secrets or private prompt/document data.

## `02_custom_enricher.py`

Shows custom structured enrichment.

**Production rule:** Enrichers should be fast, deterministic, and privacy-safe.

## `03_pipeline_logging.py`

Carries correlation metadata across a multi-stage/multi-agent workflow.

**Production lesson:** Correlation fields are essential when one user request produces many model/tool spans.

---

# 19. `15-messaging`

## Current folder contents

The actual numbered directory currently contains:

- `post_labs_message.sh`
- `setup_queues.sh`

The root README still refers to an older/different `messaging/rabbitmq/rabbitmq_agent.py` path. Do not assume that path exists in the current numbered directory.

## `setup_queues.sh`

Creates/configures RabbitMQ queues for the messaging workflow.

**Production upgrade:** Queue declarations should be idempotent and explicitly define:

- durability
- dead-letter exchange/queue
- retry strategy
- retention
- access control

## `post_labs_message.sh`

Publishes a sample labs-related message so the downstream workflow can be exercised.

A production message contract should contain metadata such as:

```json
{
  "message_id": "...",
  "tenant_id": "...",
  "user_id": "...",
  "correlation_id": "...",
  "event_type": "...",
  "schema_version": 1,
  "payload": {}
}
```

**Production rule:** Version message schemas.

---

# 20. Fluent API Cheat Sheet

| API | Purpose |
|---|---|
| `ManagedAgent()` | Agent orchestration object. |
| `.with_model(ModelConfig(...))` | Selects model/provider. |
| `.with_short_term_memory(provider)` | Recent/session memory. |
| `.with_long_term_memory(provider)` | Durable/history memory. |
| `.with_tools(registry)` | Exposes controlled Python tools. |
| `.with_prompts(provider)` | Selects system-prompt source. |
| `.with_output(Model, output_retries=N)` | Enforces typed output. |
| `.with_observability(obs)` | Adds logs/traces/metrics. |
| `.with_error_handling(config)` | Source-aware recovery. |
| `.with_evaluators(*evaluators)` | Post-turn evaluation. |
| `.with_token_limits(config)` | Token budget enforcement. |
| `.with_guardrails(...)` | Composes guardrail policies. |
| `.with_log_enrichment(...)` | Persistent structured log context. |
| `ToolRegistry().add(fn)` | Register one tool. |
| `ToolRegistry().add_many(...)` | Register several tools. |
| `MessageHistory().load(...)` | Reconstruct conversation history. |
| `deps_type=T` | Defines typed dependencies for context-aware tools. |
| `output_retries=N` | Bounds structured-output correction attempts. |
| `save_to=[...]` | Selects memory providers receiving a completed turn. |
| `run(..., enrichment=...)` | Adds run-scoped log context. |

---

# 21. How a Junior Developer Should Study the Examples

Recommended order:

```text
01 Getting Started
       |
       v
09 Tool Calling
       |
       v
10 Prompts
       |
       v
11 Memory
       |
       v
08 Structured Output
       |
       v
02 Error Handling
       |
       v
04 Guardrails
       |
       v
03 Evaluators
       |
       v
05 Human-in-the-Loop
       |
       v
06 Loops
       |
       v
07 Orchestration
       |
       v
12 Observability
       |
       v
14 Logging
       |
       v
13 RAG
       |
       v
15 Messaging
```

For every example:

1. Run it unchanged.
2. Draw the execution flow.
3. Identify every `.with_*()` capability.
4. Break one dependency intentionally.
5. Observe the error.
6. Add one production control.
7. Write a test.

Example flow to draw:

```text
request
 -> prompt render
 -> memory load
 -> model
 -> tool
 -> model
 -> output validation
 -> guardrail
 -> memory save
 -> evaluator
 -> logs/metrics/traces
```

---

# 22. Production-Grade Architecture After the Examples

Do not leave the whole application in a single example script.

A maintainable structure may look like:

```text
application/
├── api/
│   └── routes.py
├── agents/
│   ├── support_agent.py
│   └── agent_factory.py
├── tools/
│   ├── customer_tools.py
│   └── order_tools.py
├── prompts/
│   └── providers.py
├── memory/
│   └── providers.py
├── guards/
│   └── policies.py
├── evaluators/
│   └── quality.py
├── observability/
│   └── telemetry.py
├── domain/
│   ├── models.py
│   └── services.py
├── infrastructure/
│   ├── database.py
│   └── clients.py
└── tests/
    ├── unit/
    ├── integration/
    └── evaluations/
```

> The agent orchestrates AI behavior. The rest of the application still follows normal software-engineering principles.

---

# 23. Production Checklist

## Model
- configurable model/provider
- explicit timeout
- retry classification
- fallback behavior
- usage/cost measurement

## Prompts
- versioned
- reviewed
- tested
- prompt/version ID in telemetry
- variables validated

## Tools
- typed signatures
- clear docstrings
- deterministic authorization
- tenant/user scope from trusted context
- timeout
- retry/idempotency rules
- tool logging/tracing

## Memory
- tenant isolation
- authorization
- retention/deletion
- context-size limits
- backend failure policy
- encryption where appropriate

## Structured Output
- typed schemas
- business-rule validation
- bounded retries
- invalid-output telemetry

## Guardrails
- retry policy
- token limits
- cost limits
- turn limits
- PII policy
- content policy
- circuit breakers

## Error Handling
- source-specific policy
- safe user-facing fallback
- internal diagnostic context
- request/session correlation

## Evaluators
- quality criteria
- safety checks
- regression evaluation dataset
- evaluator failure monitoring

## Human Approval
- defined approval points
- approver identity
- persisted workflow state
- approval before irreversible actions

## Observability
- structured logs
- metrics
- distributed traces
- token usage
- latency
- tool spans
- retry counts
- guardrail events
- evaluator failures
- prompt/model/version metadata

## Security
- authentication
- authorization
- tenant isolation
- secret management
- least-privilege tools
- network restrictions
- audit logging
- prompt/tool injection defenses

## Reliability
- timeouts
- bounded retries
- backoff
- idempotency
- circuit breakers
- concurrency limits
- dead-letter queues
- graceful degradation

---

# 24. What the Student Should Be Able to Build

After completing these examples deliberately, a junior developer should be able to build an agentic application that:

- receives an authenticated request
- loads controlled system instructions
- maintains bounded session history
- invokes typed tools
- receives trusted request dependencies
- returns typed output
- retries transient failures
- respects token/cost/turn budgets
- applies content and PII controls
- records source-aware errors
- runs quality/safety evaluations
- requests human approval for sensitive operations
- composes specialist agents
- retrieves authorized data
- emits logs, metrics, and traces
- propagates correlation/session/tenant context
- participates in asynchronous workflows

The learning goal is to move from:

```text
"I can call an LLM."
```

to:

```text
"I can engineer, test, secure, observe, and operate an AI-enabled software system."
```

---

# 25. References

Primary repository:

- `https://github.com/myndfire/pydanticai-fluent`
- `https://github.com/myndfire/pydanticai-fluent/tree/master/agent_harness_examples`

Important repository documentation:

- `agent_harness_examples/README.md`
- each topic folder's `README.md`
- corresponding Python/shell examples
- repository `USAGE.md`
- observability documentation referenced by the examples

---

# 26. Final Teaching Principle

Teach each feature using this sequence:

```text
Problem -> Why AI makes it difficult -> Harness capability
        -> Failure modes -> Production controls -> Tests
```

Do **not** teach:

> “Here is the syntax for pydanticai-fluent.”

Teach:

> “Here is the engineering problem this syntax solves, and here is how to make that solution safe and production-ready.”

Framework APIs will evolve. The durable skills are:

- deterministic boundaries around probabilistic models
- typed contracts
- bounded execution
- explicit state
- least-privilege tools
- validation
- failure isolation
- evaluation
- observability
- security
- human control where consequences matter

---

# 27. Environment Variables — Authoritative Current Defaults

> **Important:** The tables in this section use the current `agent_harness_examples/.env.example` as the authoritative source for default values. Some individual folder READMEs still show older defaults. When the README and `.env.example` disagree, use `.env.example` unless the Python source explicitly overrides it.

## 27.1 Global / Shared Variables

| Variable | Current default/example value | Purpose |
|---|---:|---|
| `UV_PROJECT_VENV` | `.venv` | Virtual environment location used by the examples project. |
| `LOGFIRE_TOKEN` | `pylf_v1_us_z....` | Placeholder for a Pydantic Logfire token. Replace with a real token when using Logfire-backed telemetry. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible Ollama API endpoint. |
| `OPENAI_API_KEY` | `sk-proj...` | Placeholder OpenAI key; required only by examples that actually call OpenAI services, such as the moderation evaluator. |
| `MODEL_NAME` | `granite4.1:8b` | Shared default Ollama model for examples that use the generic model setting. |
| `MAX_TOKENS` | `256` | Shared maximum LLM output-token setting. |
| `LLM_PROVIDER` | `ollama` | Shared provider identifier, used by orchestration and other examples. |
| `HARNESS_DETAILED_LOG` | `false` | Enables/disables additional harness diagnostic logging. |
| `HARNESS_DEFAULT_TRACEBACK_FRAMES` | `0` | Controls default traceback-frame inclusion in harness diagnostics. |

### Junior-developer note

A value being present in `.env.example` does **not** mean every Python file reads it. Treat `.env.example` as the centralized configuration catalog. Each file-specific table below lists the variables relevant to that file or topic.

---

# 28. Expanded File-by-File Guide

This section expands the earlier descriptions so a junior developer can understand the **runtime flow**, not merely the feature name.

---

## 28.1 `01-getting_started`

### Environment variables for this folder

Current `.env.example` says this folder uses the shared variables:

| Variable | Default | Used for |
|---|---:|---|
| `MODEL_NAME` | `granite4.1:8b` | Model selected by the starter examples. |
| `MAX_TOKENS` | `256` | Maximum model output size where configured. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint when the provider integration reads the standard base URL. |
| `LOGFIRE_TOKEN` | placeholder | Relevant to examples using Logfire telemetry. |
| `MONGODB_URI` | `mongodb://localhost:27017` in the shared storage section | Optional persistent memory for example 2. |
| `MONGODB_DATABASE` | `agent_memory` in the memory section | MongoDB database used for agent memory. |
| `MONGODB_COLLECTION` | `conversations` in the memory section | MongoDB collection used for conversations. |

> **Repository mismatch to notice:** the folder README still says `MODEL_NAME=qwen2.5:3b`; the current root `.env.example` says `MODEL_NAME=granite4.1:8b`.

### `README.md`

This is the onboarding document for the first three agents. It explains the three progressively richer examples, prerequisites, optional telemetry/storage services, basic run commands, and visual execution flows.

A junior developer should use it to answer:

- What must be running before I execute an example?
- Which external services are optional?
- What is the difference between a basic tool-enabled agent, an observable multi-turn agent, and a structured-output agent?
- Which files should I run first?

It also illustrates how the harness moves from a minimal tool call to an operationally instrumented agent.

### `agent_example-1.py`

This file is the smallest example that demonstrates **actual agent behavior rather than just a model call**.

It defines two deterministic Python tools:

```text
repeat(text) -> returns text
shout(text)  -> uppercases text
```

The prompt asks the model to perform a two-step task. The interesting part is that the model has to decide to invoke the tools in sequence. The program therefore demonstrates the complete loop:

```text
user prompt
   |
   v
ManagedAgent
   |
   v
LLM decides to call repeat()
   |
   v
repeat() result returned to model
   |
   v
LLM decides to call shout()
   |
   v
shout() result returned to model
   |
   v
final response
```

The file also deliberately includes memory, observability, error handling, and an evaluator even though the task is simple. That is pedagogically important: it shows that the framework is designed around **composable operational capabilities**, not just tool calling.

A junior developer should inspect:

- how functions become tools;
- how `ToolRegistry` is populated;
- how prompt behavior is declared;
- how a session is loaded;
- how the same `ManagedAgent` can have optional operational components attached.

**Production lesson:** tool functions should normally call application services rather than containing large amounts of business logic directly.

### `agent_example-2.py`

This file turns the starter agent into something closer to a service you could debug in production.

It runs multiple prompts with the same session ID, which demonstrates that the second and third requests depend on conversation state created by the first request.

The flow is roughly:

```text
turn 1 -> memory save
turn 2 -> load prior history -> model -> memory save
turn 3 -> load prior history -> model -> memory save
```

At the same time, telemetry providers observe the run. The example introduces the idea that one logical agent run can produce:

- application logs;
- model/tool spans;
- metrics;
- token information;
- session correlation information.

It also optionally replaces long-term in-memory storage with MongoDB. This teaches a key production distinction:

```text
short-term context != durable historical storage
```

A junior developer should pay attention to the graceful fallback when MongoDB is not configured and understand why this is acceptable for a demo but may or may not be acceptable for a real application.

### `agent_example-3.py`

This file demonstrates **structured AI output** using a Pydantic `Invoice`.

Instead of asking the LLM to produce prose and then writing fragile string parsing, the program defines the expected object first.

Conceptually:

```text
natural-language invoice request
        |
        v
      model
        |
        v
Pydantic structured-output contract
        |
        v
      Invoice
```

The example also parses invoice-like source content into the same typed model. This teaches that structured output is useful for both generation and extraction.

A junior developer should notice that Pydantic validates **shape and types**, but not necessarily every accounting invariant. Production code still needs deterministic checks such as:

```text
subtotal == sum(line_items)
total == subtotal + tax
```

---

## 28.2 `02-error_handling`

### Environment variables

Current `.env.example` explicitly states:

**No environment variables are required.** The examples use hard-coded values and intentionally broken model/provider implementations where needed.

The only real model mentioned by the folder documentation is an Ollama `gpt-oss:20b` example path; broken model names are deliberate failure injectors.

### `01_basic_handler.py`

This file teaches the basic contract of `ErrorHandlingConfig`.

Rather than showing only one exception, it runs several small scenarios so the developer can see both possible handler outcomes:

```text
handler returns fallback -> error suppressed
handler returns None     -> exception propagates
```

It also prints fields from `ErrorContext`, demonstrating what diagnostic information is available when deciding how to recover.

This file is important because it teaches that “handling an error” means making an explicit product decision about whether the request can continue.

### `02_source_routing.py`

This example demonstrates that error policy should be based on **where the failure originated**.

The code registers different callbacks for different sources and lets the harness route the failure automatically.

The developer should understand why this is better than one giant catch-all:

```text
tool timeout != invalid output != memory outage != safety rejection
```

A production implementation can then map these categories to different:

- retry behavior;
- alerts;
- user messages;
- fallback providers;
- incident severity.

### `03_custom_recovery.py`

This file goes beyond a one-line fallback callback and demonstrates a richer recovery function.

It shows how a recovery handler can inspect diagnostic context, record what happened, choose a safe degraded result, and still allow the overall application to remain responsive.

The major lesson is that fallback behavior should be **observable**. If recovery hides every failure, operators may believe the system is healthy when it is repeatedly failing underneath.

### `04_tool_errors.py`

This file deliberately causes a registered tool to fail.

The model call itself can be perfectly healthy; the error occurs only after the model asks the application to perform an action.

This distinction matters because tools often cross high-risk boundaries:

```text
agent -> payment API
agent -> database
agent -> email
agent -> filesystem
```

The example teaches how those failures enter `on_tool_error`.

Production systems should additionally consider partial completion, idempotency, and compensation.

### `05_guardrail_errors.py`

This example triggers errors from guardrail logic and routes them to the guardrail-specific error callback.

The key lesson is that some “errors” are actually **deliberate policy enforcement**. A circuit breaker, token budget, or content policy may intentionally prevent execution.

Operators need to distinguish:

```text
system malfunction
vs.
system correctly blocked unsafe/expensive behavior
```

### `06_evaluator_errors.py`

This file intentionally causes an evaluator to fail.

Evaluators run after the main result, so the example shows that evaluation infrastructure is a separate reliability domain.

The production question is:

> Should a telemetry/quality-scoring failure prevent a user from receiving an otherwise valid response?

Often the answer is no, but the failure should still be monitored.

### `07_memory_errors.py`

This example supplies a memory provider that fails during memory operations.

It teaches the developer where memory sits in the request lifecycle:

```text
memory load -> model execution -> memory save
```

Either side can fail.

A production application should explicitly decide whether it can continue stateless. A casual Q&A bot might; a regulated workflow depending on conversation state might not.

### `08_prompt_errors.py`

This file demonstrates failure in the prompt provider rather than in the LLM.

Examples of equivalent real-world failures include:

- missing prompt version;
- bad template;
- missing Jinja variable;
- prompt database unavailable.

The lesson is that prompts are an application dependency and should have validation, versioning, and fallback strategy.

### `09_pipeline_error_recovery.py`

This is the most architectural error-handling example.

It creates multiple stages and intentionally fails one stage to demonstrate workflow-level recovery.

The developer should think of it as:

```text
stage A succeeds
    |
stage B fails
    |
recovery policy
    |
stage C may still run
```

That forces explicit thinking about partial success, downstream assumptions, and whether continuing produces a trustworthy result.

---

## 28.3 `03-evaluators`

### Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_NAME` | `granite4.1:8b` | Shared model used by agent examples unless code specifies otherwise. |
| `QUALITY_CHECK_MODEL` | `ollama:gpt-oss:20b` | Model used by `QualityCheck` as the judging model. |
| `SAFETY_CHECK_MODEL` | `omni-moderation-2024-09-26` | OpenAI moderation model used by safety evaluation. |
| `OPENAI_API_KEY` | placeholder | Required when actually running OpenAI moderation. |

### `01_quality_check.py`

This file introduces **LLM-as-a-judge** evaluation.

The main agent produces a response. A second model then scores that response against quality expectations.

That produces two different AI roles:

```text
primary model -> generates
judge model   -> evaluates
```

The file demonstrates several thresholds, including lenient and strict configurations, so the developer can see that the threshold is a product-quality policy.

Important lesson: an LLM judge is probabilistic. It should be validated against human-rated examples before it is trusted as a release gate.

### `02_safety_check.py`

This example sends prompt/response content to an OpenAI moderation model and reports flagged safety categories.

The agent itself can use a different model provider; the evaluator is independent.

That separation teaches a valuable production pattern:

```text
generation provider != moderation provider
```

The example also demonstrates graceful behavior when the optional moderation dependency is unavailable.

### `03_custom_evaluator.py`

This file demonstrates how to implement your own evaluator for organization-specific rules.

Instead of asking another LLM to judge everything, custom evaluators can perform deterministic checks such as:

- response length;
- required words;
- turn counts;
- known formatting rules.

This is usually cheaper, faster, and more predictable when the criterion is objectively measurable.

### `04_protocol_evaluator.py`

This example demonstrates loose coupling.

The evaluator does not need to inherit from a framework base class; it only needs to satisfy the expected async `evaluate(...)` interface.

That teaches the junior developer an important Python design principle: depend on behavior/protocols rather than concrete inheritance hierarchies.

### `05_combined_evaluators.py`

This file attaches multiple evaluators to the same agent.

Each evaluator sees the same completed turn and runs independently.

The production lesson is to avoid creating one enormous “do everything” evaluator. Separate concerns such as:

```text
quality
safety
compliance
telemetry
business KPI scoring
```

---

## 28.4 `04-guardrails`

### Shared environment variables

| Variable | Default |
|---|---:|
| `GUARDRAILS_MODEL_PROVIDER` | `ollama` |
| `GUARDRAILS_MODEL_NAME` | `gpt-oss:20b` |

### `01_agent_retries.py`

Environment:

| Variable | Default |
|---|---:|
| `AGENT_RETRIES_MAX_RETRIES` | `3` |
| `AGENT_RETRIES_TIMEOUT` | `10` |
| `AGENT_RETRIES_BACKOFF` | `2.0` |
| `AGENT_RETRIES_FALLBACK_MODEL` | `ollama:gpt-oss:20b` |

This file demonstrates retrying an entire agent/model operation rather than a single tool.

It shows the relationship among:

- max retries;
- timeout;
- exponential backoff;
- retry callbacks;
- terminal error handling;
- optional fallback model.

A junior developer should learn to distinguish retryable transient failures from deterministic failures. Invalid credentials, for example, normally should not be retried repeatedly.

### `02_tool_retries.py`

Environment:

| Variable | Default |
|---|---:|
| `TOOL_RETRIES_AGENT_MAX_RETRIES` | `2` |
| `TOOL_RETRIES_AGENT_TIMEOUT` | `30` |
| `TOOL_RETRIES_MAX_RETRIES` | `3` |
| `TOOL_RETRIES_BACKOFF` | `1.5` |

This file shows that tool retry policy can be separate from whole-agent retry policy.

That is important because a tool may fail while the model provider is healthy.

For side-effecting tools, retry safety must be designed explicitly.

### `03_result_validator_retries.py`

Environment:

| Variable | Default |
|---|---:|
| `VALIDATOR_AGENT_MAX_RETRIES` | `2` |
| `VALIDATOR_AGENT_TIMEOUT` | `30` |
| `VALIDATOR_MAX_RETRIES` | `3` |
| `VALIDATOR_BACKOFF` | `2.0` |

This example retries when the generated result violates an output/business validation rule.

The retry asks the model to correct its answer rather than treating the result as permanently failed.

This is useful for correctable generation mistakes, but it must remain bounded.

### `04_content_filter.py`

No file-specific environment variables are listed.

The file demonstrates a content transformation/filter callback. The example uses simple profanity-style filtering to make the callback behavior easy to see.

The lesson is the hook location:

```text
model output -> filter -> returned output
```

Production moderation should normally be more robust than a demo regex/list.

### `05_pii_detection.py`

No file-specific environment variables are listed.

The example demonstrates output redaction of recognizable sensitive-data patterns.

It shows categories such as email, phone, SSN, card-like numbers, and IP addresses.

The important production lesson is that privacy policy is broader than regex matching. The example demonstrates the integration point, not a complete DLP solution.

### `06_token_limits.py`

Environment:

| Variable | Default |
|---|---:|
| `TOKEN_LIMITS_EX1_MAX_TOTAL` | `50` |
| `TOKEN_LIMITS_EX1_MAX_OUTPUT` | `200` |
| `TOKEN_LIMITS_EX2_MAX_INPUT` | `500` |
| `TOKEN_LIMITS_EX2_MAX_OUTPUT` | `100` |
| `TOKEN_LIMITS_EX2_MAX_TOTAL` | `600` |
| `TOKEN_LIMITS_EX3_MAX_TOTAL` | `5` |

This file runs several scenarios showing input, output, and total token budgets.

The tiny limits are intentionally useful for triggering the guardrail.

The developer should understand tokens as both a technical resource and a cost/latency resource.

### `07_cost_limits.py`

Environment:

| Variable | Default |
|---|---:|
| `COST_LIMITS_INPUT_COST` | `0.000003` |
| `COST_LIMITS_OUTPUT_COST` | `0.000015` |
| `COST_LIMITS_EX1_MAX_TOTAL` | `0.005` |
| `COST_LIMITS_EX2_MAX_INPUT` | `0.001` |
| `COST_LIMITS_EX2_MAX_OUTPUT` | `0.003` |
| `COST_LIMITS_EX3_MAX_TOTAL` | `0.000001` |

This example converts token usage into an estimated monetary budget and rejects execution when the configured limit is exceeded.

The prices are example configuration, not permanent provider pricing. Production systems should source current pricing/configuration centrally.

### `08_circuit_breaker.py`

Environment:

| Variable | Default |
|---|---:|
| `CIRCUIT_BREAKER_BAD_MODEL_NAME` | `this-model-does-not-exist` |
| `CIRCUIT_BREAKER_THRESHOLD` | `3` |
| `CIRCUIT_BREAKER_TIMEOUT` | `5` |

The intentionally invalid model causes repeated failures until the circuit breaker opens.

This allows the developer to observe the state machine:

```text
CLOSED -> repeated failures -> OPEN
OPEN -> timeout -> HALF_OPEN
HALF_OPEN -> successful probe -> CLOSED
```

The key production purpose is to stop flooding an unhealthy dependency.

### `09_all_guardrails.py`

Environment:

| Variable | Default |
|---|---:|
| `ALL_GUARDRAILS_AGENT_MAX_RETRIES` | `3` |
| `ALL_GUARDRAILS_AGENT_TIMEOUT` | `30` |
| `ALL_GUARDRAILS_AGENT_BACKOFF` | `2.0` |
| `ALL_GUARDRAILS_TOKEN_MAX_TOTAL` | `500` |
| `ALL_GUARDRAILS_INPUT_COST` | `0.000003` |
| `ALL_GUARDRAILS_OUTPUT_COST` | `0.000015` |
| `ALL_GUARDRAILS_MAX_TOTAL_COST` | `0.01` |
| `ALL_GUARDRAILS_CIRCUIT_THRESHOLD` | `5` |
| `ALL_GUARDRAILS_CIRCUIT_TIMEOUT` | `30` |
| `ALL_GUARDRAILS_AGENT2_MAX_RETRIES` | `2` |
| `ALL_GUARDRAILS_AGENT2_TOKEN_MAX_TOTAL` | `300` |
| `ALL_GUARDRAILS_AGENT2_MAX_TOTAL_COST` | `0.005` |
| `ALL_GUARDRAILS_AGENT2_CIRCUIT_THRESHOLD` | `3` |

This is the “composition” example. It combines multiple protections so the developer can see how independent policies build a production envelope around the same agent.

### `10_turn_limits.py`

Environment:

| Variable | Default |
|---|---:|
| `TURN_LIMITS_EX1_MAX_TURNS` | `3` |
| `TURN_LIMITS_EX2_MAX_TURNS` | `2` |
| `TURN_LIMITS_EX3_MAX_TURNS` | `1` |

This file demonstrates per-session turn limits and what happens when the maximum is exceeded.

The deeper lesson is that every potentially autonomous loop must have a deterministic resource bound.

---

## 28.5 `05-human_in_the_loop`

### Environment variables

The current `.env.example` says **no environment variables are required**; the examples use hard-coded defaults.

### `01_review_approval.py`

The agent generates a complete answer first. A human review callback then intercepts it.

The person can approve the output or replace it.

This is the easiest HITL pattern to understand:

```text
agent work -> proposed output -> human review -> final output
```

It is appropriate when the main concern is publication/release of the generated text.

### `02_tool_review.py`

This file adds a deterministic calculation tool before human review.

The point is to show that the answer may be based on real tool results, yet a person still gets the opportunity to inspect the final composed response.

The developer should understand that human review of the text does not automatically equal authorization for the underlying action.

### `03_multistep_tools.py`

This is the most realistic HITL example in the folder.

It inserts human confirmation between dependent tool steps.

The design demonstrates a critical principle:

```text
approval must occur before the consequential step
```

For production, replace terminal `input()` with a persisted approval request and resumable workflow.

---

## 28.6 `06-loops`

### Environment variables

| Variable | Default |
|---|---:|
| `MODEL_NAME` | `granite4.1:8b` |
| `MAX_TOKENS` | `256` |
| `REACT_MODEL_NAME` | `qwen3.5:4b` |
| `REFINEMENT_MAX_WORDS` | `10` |
| `REFINEMENT_MAX_ATTEMPTS` | `3` |
| `REFINEMENT_TEMPERATURE` | `0.7` |
| `GOAL_MAX_ATTEMPTS` | `10` |
| `GOAL_LIFESPAN_MIN` | `7` |
| `GOAL_LIFESPAN_MAX` | `12` |
| `GOAL_TEMPERATURE` | `0.7` |
| `PLANNING_MODEL_NAME` | `llama3.1:8b` |
| `PLAN_MAX_STEPS` | `6` |

> The folder README still contains some older values; use the root `.env.example` above as the current default catalog.

### `01_interactive_loop.py`

This is an ordinary conversational shell around a ManagedAgent.

The loop itself is deterministic Python. On every iteration the application:

1. reads user input;
2. checks for exit commands;
3. loads message history;
4. calls the agent;
5. prints the result;
6. repeats.

This teaches that a chat product is **an application loop that calls an agent**, not one giant magical agent.

### `02_iterative_refinement.py`

The program asks the model to summarize text under a word limit. A deterministic evaluator counts words.

If the response is too long, Python creates corrective feedback and invokes the agent again.

This is a good introduction to **closed-loop AI systems**:

```text
generate -> measure -> feedback -> regenerate
```

The loop stops when quality passes or the attempt limit is reached.

### `03_react_loop.py`

A single `agent.run()` triggers an internal model/tool loop.

The model chooses multiple tool calls, sees the observations, and decides when it has enough information to stop.

This demonstrates the ReAct mental model without requiring the developer to write an explicit outer loop.

Production systems should still cap the total number of model/tool iterations.

### `04_goal_seeking_loop.py`

The model proposes an answer; deterministic Python checks whether it satisfies a lifespan constraint.

If it does not, the code tells the model why it failed and tries again.

Unlike free-form “self-reflection,” the success criterion here is measurable by normal code.

That is the preferred pattern whenever the desired outcome can be verified deterministically.

### `05_planning_loop.py`

This example separates planning from execution.

Phase 1 returns a typed `AgentPlan`. Python then iterates those plan steps and calls deterministic tools. A final phase synthesizes the collected results.

This is a much cleaner architecture than letting a single model improvise an unbounded workflow.

### `text_to_summarize.txt`

A deterministic fixture consumed by the refinement example. It exists so the loop example can focus on behavior rather than external document retrieval.

---

## 28.7 `07-orchestration`

### Environment variables

| Variable | Default |
|---|---:|
| `MODEL_NAME` | `granite4.1:8b` |
| `LLM_PROVIDER` | `ollama` |

No additional orchestration-specific variables are listed.

### `01_delegation.py`

A coordinator agent exposes a specialist as a callable capability/tool.

The coordinator decides that a subproblem belongs to the specialist, delegates it, receives the specialist's result, and incorporates that into the final answer.

This teaches **agent-as-a-capability**, not simply “run two agents.”

Typed dependency context is also shared so both layers can access trusted application state.

### `02_sequential_pipeline.py`

Multiple agents are called in a fixed deterministic order.

Each stage has one responsibility and consumes the previous stage's output.

The pattern is useful when the workflow itself is known in advance:

```text
research -> write -> edit
```

The model is used inside the steps; Python owns the workflow topology.

### `03_routing.py`

A router/classifier agent decides which specialist category matches the request. Python then selects the appropriate specialist.

This teaches a production-friendly split:

```text
AI classification
+
deterministic dispatch
```

That is safer and easier to test than letting one unrestricted agent dynamically call everything.

### `04_parallel_fanout.py`

Independent specialist requests are executed concurrently and their results are merged.

This improves latency when tasks do not depend on each other, but increases concurrency and provider load.

Production code should set concurrency and timeout limits rather than launching unlimited subtasks.

---

## 28.8 `08-structured_output`

### Environment variables

| Variable | Default |
|---|---:|
| `STRUCTURED_OUTPUT_MODEL_NAME` | `phi4-mini` |
| `STRUCTURED_OUTPUT_REASONING_MODEL` | `phi4-mini-reasoning` |
| `STRUCTURED_OUTPUT_MAX_TOKENS` | `512` |

### `01_simple_model.py`

Returns one simple Pydantic object rather than natural-language prose.

The purpose is to teach the developer to define the **software contract first**.

### `02_enums_literals.py`

Adds finite allowed values such as sentiment labels.

It demonstrates how type constraints improve reliability by making impossible values fail validation rather than silently flow downstream.

### `03_nested_models.py`

Builds richer hierarchical output with child Pydantic models and lists.

This shows that structured-output agents can produce realistic domain objects, not just flat JSON dictionaries.

### `04_validation_retries.py`

Adds custom validation beyond schema/type checking.

When the model returns structurally valid but semantically unacceptable data, the validator can request another model attempt.

The key lesson is:

```text
valid JSON != valid business result
```

---

## 28.9 `09-tool_calling`

### Environment variables

| Variable | Default |
|---|---:|
| `TOOL_CALLING_MODEL_NAME` | `gpt-oss:20b` |
| `TOOL_CALLING_MAX_TOKENS` | `512` |

### `01_plain_tools.py`

This is the tool-registration fundamentals example.

It demonstrates adding one or many regular Python functions to `ToolRegistry`, retrieving the registry contents, and clearing the registry.

A junior developer should study the function annotations and docstrings because those become part of the model-visible tool contract.

### `02_context_tools.py`

This example adds `RunContext`/typed dependencies to a tool.

The important distinction is that some tool parameters come from the **trusted application**, not from the LLM.

For example:

```text
LLM supplies: customer query
app supplies: authenticated user / tenant / service client
```

This is one of the most important security patterns in production agent design.

### `03_mcp_server.py`

This file introduces MCP as an external tool provider.

Rather than registering every function directly in the Python process, the agent can consume tools published by an MCP server. Example 4 combines two public, keyless servers — Context7 (`https://mcp.context7.com/mcp`) for library documentation and Complex server (`https://mcpplaygroundonline.com/mcp-complex-server`) for data operations — using `tool_prefix` to disambiguate tools across servers. No auth required, only network egress.

The developer should view MCP as another integration boundary that requires trust, authentication, allow-listing, timeout, and audit policy.

### `04_tool_combinations.py`

Combines tool calling with additional harness features.

The lesson is that a tool-enabled agent is not production-ready merely because the tool call works. The same call path should also participate in guardrails, evaluation, typed output, errors, and telemetry.

---

## 28.10 `10-prompts`

### Environment variables

| Variable | Default |
|---|---:|
| `PROMPTS_MODEL_NAME` | `gpt-oss:20b` |
| `PROMPTS_MAX_TOKENS` | `512` |
| `MONGODB_URI` | `mongodb://localhost:27017` |
| `MONGODB_DATABASE` | `agent_prompts` |
| `MONGODB_COLLECTION` | `prompts` |

### `01_static_prompts.py`

Shows the simplest prompt-provider strategy: instructions exist directly in application configuration.

The file demonstrates that changing the system prompt changes the behavior/persona of the same underlying agent.

Static prompts are appropriate when prompts are versioned with code and do not need runtime administration.

### `02_mongo_prompts.py`

Moves prompt templates into MongoDB.

This enables prompt CRUD, versions, and dynamic rendering without rebuilding application code.

The example is useful for teaching that prompts can have a lifecycle similar to configuration/data.

Production prompt administration should include review, version pinning, rollback, and audit history.

### `03_prompt_variables.py`

Demonstrates reusable prompt templates containing variables.

At run time, application values are supplied to render the final prompt.

The developer should understand the separation:

```text
template + trusted/untrusted variables -> rendered system prompt
```

External values must be sanitized and constrained so they cannot override core policy.

---

## 28.11 `11-memory`

### Environment variables

| Variable | Default |
|---|---:|
| `MEMORY_MODEL_NAME` | `gpt-oss:20b` |
| `REASONING_MODEL_NAME` | `phi4-mini-reasoning` |
| `MEMORY_MAX_TOKENS` | `512` |
| `MEMORY_SHORT_TERM_MAX_TURNS` | `10` |
| `MEMORY_LONG_TERM_MAX_TURNS` | `100` |
| `MEMORY_AUDIT_MAX_TURNS` | `10000` |
| `MEMORY_MONGODB_TIMEOUT_MS` | `2000` |
| `MEMORY_REDIS_CONNECT_TIMEOUT` | `2` |
| `MEMORY_REASONING_TIMEOUT` | `60` |
| `MONGODB_URI` | `mongodb://localhost:27017` |
| `MONGODB_DATABASE` | `agent_memory` |
| `MONGODB_COLLECTION` | `conversations` |
| `REDIS_HOST` | `localhost` |
| `REDIS_PORT` | `6379` |
| `REDIS_KEY_PREFIX` | `agent:memory:` |
| `ELASTICSEARCH_ENDPOINT` | `http://localhost:9200` |
| `ELASTICSEARCH_INDEX` | `agent-memory` |

### `01_in_memory.py`

Demonstrates short- and long-term memory roles using process-local storage.

Because both are in memory, the example isolates the API concepts from database setup.

It is excellent for learning/tests but not durable across application restarts.

### `02_message_history.py`

Focuses on turning persisted `TurnData` back into model-consumable conversation history.

This file teaches the crucial distinction between **stored application state** and **messages actually supplied to the model**.

Production context management may need summarization, filtering, truncation, and privacy controls.

### `03_multi_provider.py`

Writes conversation information to multiple providers that represent different purposes, such as short-term state, longer-term history, and audit retention.

This teaches that “memory” is not necessarily one database.

### `04_memory_operations.py`

Exercises management operations rather than just normal chat execution.

The developer sees how to inspect, delete, clear, or limit stored turns.

These capabilities matter for administration, privacy, troubleshooting, and automated retention.

### `05_mongo_memory.py`

Uses MongoDB for persistent conversation storage.

The file demonstrates the same memory contract with a durable provider.

Production additions include indexes, authentication, encryption, pool sizing, health checks, retry strategy, and tenant scoping.

### `06_redis_memory.py`

Uses Redis as a fast memory backend.

Redis is especially suitable for recent session state where TTL/expiration is useful.

The developer should understand that durability characteristics depend on Redis configuration.

### `07_combined_memory.py`

Combines Redis and MongoDB in one design.

This demonstrates a realistic split:

```text
fast recent session state -> Redis
durable conversation archive -> MongoDB
```

The production question becomes what to do when only one of those writes succeeds.

### `08_elasticsearch_memory.py`

Stores/indexes memory in Elasticsearch so historical turns can be searched.

This demonstrates that a memory backend can also serve query/debug/retrieval needs.

Production requires correct mappings, lifecycle management, and tenant filters on every query.

### `09_reasoning_traces.py`

Uses a reasoning-capable model and inspects reasoning-related message parts.

The teaching goal is to understand the shapes stored in model message history.

Production applications should not depend on hidden/private chain-of-thought for business logic; store explicit decisions, summaries, tool calls, and outcomes instead.

---

## 28.12 `12-observability`

### Environment variables

| Variable | Default |
|---|---:|
| `OBSERVABILITY_MODEL_NAME` | `gpt-oss:20b` |
| `OBSERVABILITY_MAX_TOKENS` | `512` |
| `ELASTICSEARCH_ENDPOINT` | `http://localhost:9200` |
| `JAEGER_OTLP_ENDPOINT` | `localhost:14317` *(deprecated — OTLP now flows through the collector)* |
| `PROMETHEUS_PUSH_GATEWAY` | `http://localhost:9091` |
| `OTEL_COLLECTOR_ENDPOINT` | `localhost:4317` |
| `OBSERVABILITY_SERVICE_NAME` | `all-in-one-observability-demo` |

### `01_logging.py`

Introduces logging providers in isolation.

It demonstrates console and file-oriented logging and file-rotation/retention concepts.

The purpose is to let the developer understand the logger contract before combining it with live model execution.

### `02_tracing_metrics.py`

Uses in-memory tracer and metrics implementations.

The generated data can be inspected immediately in Python, making this a good unit-test/learning environment.

### `03_builder_logs_metrics.py`

Uses an observability builder to compose multiple telemetry capabilities and then attaches the resulting object to a ManagedAgent.

This teaches centralized infrastructure configuration.

### `04_composite_logs.py`

Sends the same structured event to multiple logger backends.

This pattern allows local console visibility and centralized storage without duplicating application logging calls.

### `05_elasticsearch_logging.py`

Sends structured agent logs directly into Elasticsearch.

The developer should inspect how agent events turn into indexed fields rather than only human-readable strings.

### `06_otel_jaeger_logs_traces_metrics.py`

Introduces OpenTelemetry-based tracing and Jaeger visualization.

The important concept is correlation: one top-level agent run can include nested model/tool operations in the same trace.

### `07_prometheus_logs_metrics.py`

Demonstrates Prometheus-oriented metrics and Pushgateway behavior.

Metrics answer aggregate operational questions such as:

```text
How many runs failed?
How long are runs taking?
How many tokens are consumed?
```

### `08_live_agent_logs_metrics.py`

Combines real model execution with the composed telemetry stack.

This demonstrates that instrumentation works on the real execution path rather than only in isolated provider demos.

### `09_otel_oltp_logs_traces_metrics.py`

This is the most complete observability example.

All three signal types are emitted via OpenTelemetry to a collector, which can route them into specialized backends.

The developer should treat this as a reference architecture for production diagnostics rather than as only a demo.

---

## 28.13 `13-rag`

### Environment variables

The current `.env.example` states:

**No environment variables are required; the model is hard-coded in `agent.py`.**

### `README.md`

Explains the sample RAG use case, data fixtures, run behavior, and the purpose of the retrieval tools.

### `agent.py`

Creates a domain-specific assistant with retrieval tools.

The model does not receive every document automatically. Instead, it chooses among controlled retrieval functions that return relevant source data.

The developer should study this boundary carefully:

```text
LLM decides what information is needed
application/tool decides what data can actually be retrieved
```

That separation is the foundation for authorization-aware RAG.

### `sample_data/`

Contains deterministic source documents used by the example.

The folder makes the example reproducible without requiring a real search/vector/document platform.

### `pyproject.toml`

Defines the RAG example's dependency requirements.

### `uv.lock`

Locks those dependency versions.

---

## 28.14 `14-logging`

### Environment variables

Shared:

| Variable | Default |
|---|---:|
| `LOGGING_MODEL_PROVIDER` | `ollama` |
| `LOGGING_MODEL_NAME` | `gpt-oss:20b` |

`01_basic_log_context.py`:

| Variable | Default |
|---|---:|
| `LOGGING_01_PIPELINE` | `content-qa` |
| `LOGGING_01_AGENT_ROLE` | `assistant` |
| `LOGGING_01_STAGE_1` | `math-check` |
| `LOGGING_01_STAGE_2` | `knowledge-check` |

`02_custom_enricher.py`:

| Variable | Default |
|---|---:|
| `LOGGING_02_PIPELINE` | `enrichment-demo` |
| `LOGGING_02_APP_VERSION` | `2.1.0` |

`03_pipeline_logging.py`:

| Variable | Default |
|---|---:|
| `LOGGING_03_PIPELINE` | `content-qa` |
| `LOGGING_03_ROLE_RESEARCHER` | `researcher` |
| `LOGGING_03_ROLE_WRITER` | `writer` |
| `LOGGING_03_ROLE_EDITOR` | `editor` |

### `01_basic_log_context.py`

Introduces static/persistent and per-run structured context.

Instead of generating a vague line such as:

```text
agent ran
```

the log event can carry fields such as:

```text
pipeline=content-qa
agent_role=assistant
stage=math-check
session_id=...
```

This makes production filtering and correlation possible.

### `02_custom_enricher.py`

Shows how to create a custom enrichment provider.

This is useful when every log should automatically carry application-specific data such as version, deployment, or tenant context.

The developer should ensure enrichers do not make slow external calls.

### `03_pipeline_logging.py`

Uses structured log context across several pipeline roles/stages.

This is particularly useful in multi-agent workflows because otherwise logs from researcher/writer/editor runs can be difficult to associate with one top-level request.

---

## 28.15 `15-messaging`

### Environment variables

| Variable | Default |
|---|---:|
| `RABBITMQ_AGENT_MODEL_NAME` | `gemma4:12b-mlx` |
| `RABBITMQ_AGENT_MODEL_SETTINGS` | `{"thinking": false, "max_tokens": 512, "temperature": 0.1}` |

> The current numbered `15-messaging` directory contains shell helper scripts, while the root README and `.env.example` still reference a RabbitMQ agent path from an older/different layout. Treat this as repository documentation drift and verify the code tree before teaching a missing file.

### `setup_queues.sh`

Creates the RabbitMQ queue topology needed by the messaging demonstration.

The developer should understand this as **infrastructure setup**, not agent reasoning.

In production the queue topology should define:

- durable queues;
- retry/dead-letter behavior;
- message TTL if appropriate;
- access permissions;
- idempotent declaration.

### `post_labs_message.sh`

Publishes a sample labs message into the queue.

Its purpose is to provide a reproducible event that can trigger/drive the asynchronous agent workflow.

The deeper lesson is that an agent does not have to be invoked by an HTTP request or chat UI:

```text
business event -> queue -> worker -> ManagedAgent
```

Production messages should use a versioned schema and carry correlation and tenant/user identity metadata from trusted sources.

---

# 29. Configuration Precedence for Students

When you teach these examples, use this order to determine a configuration value:

1. **Look at the actual Python file.** If it contains an explicit hard-coded value that is not overridden, that wins.
2. **Look at `agent_harness_examples/.env.example`.** This is the best current centralized catalog of environment defaults.
3. **Look at the folder `README.md`.** Use it for explanation, but be aware that some defaults/paths lag behind the current code.
4. **Look at the root README.** It provides navigation and infrastructure guidance but contains some stale directory names.

This is also a useful real-world engineering lesson:

> Documentation can drift. Production engineers validate behavior against source and centralized configuration rather than assuming every README is synchronized.

---

# 30. Recommended `.env` Starter for the Curriculum

A junior developer running the local Ollama-based curriculum can begin with:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/v1

MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
LLM_PROVIDER=ollama

HARNESS_DETAILED_LOG=false
HARNESS_DEFAULT_TRACEBACK_FRAMES=0

GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b

QUALITY_CHECK_MODEL=ollama:gpt-oss:20b

STRUCTURED_OUTPUT_MODEL_NAME=phi4-mini
STRUCTURED_OUTPUT_REASONING_MODEL=phi4-mini-reasoning
STRUCTURED_OUTPUT_MAX_TOKENS=512

TOOL_CALLING_MODEL_NAME=gpt-oss:20b
TOOL_CALLING_MAX_TOKENS=512

PROMPTS_MODEL_NAME=gpt-oss:20b
PROMPTS_MAX_TOKENS=512

MEMORY_MODEL_NAME=gpt-oss:20b
REASONING_MODEL_NAME=phi4-mini-reasoning
MEMORY_MAX_TOKENS=512

OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512

LOGGING_MODEL_PROVIDER=ollama
LOGGING_MODEL_NAME=gpt-oss:20b
```

Then add external-service settings only for lessons that need them:

```dotenv
# OpenAI moderation
OPENAI_API_KEY=...
SAFETY_CHECK_MODEL=omni-moderation-2024-09-26

# MongoDB
MONGODB_URI=mongodb://localhost:27017

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_KEY_PREFIX=agent:memory:

# Elasticsearch
ELASTICSEARCH_ENDPOINT=http://localhost:9200

# Telemetry
JAEGER_OTLP_ENDPOINT=localhost:14317  # deprecated
PROMETHEUS_PUSH_GATEWAY=http://localhost:9091
OTEL_COLLECTOR_ENDPOINT=localhost:4317
OBSERVABILITY_SERVICE_NAME=all-in-one-observability-demo

# Logfire
LOGFIRE_TOKEN=...
```

Do not commit real API keys or telemetry tokens.

---

# 31. Instructor Guidance: What Students Should Explain Before Moving On

For each Python example, require the student to answer these questions:

1. **What starts the workflow?**
2. **What deterministic Python code executes?**
3. **What decision is delegated to the LLM?**
4. **What external dependency can fail?**
5. **What `.with_*()` methods assemble the ManagedAgent?**
6. **What does each fluent parameter change?**
7. **Which environment variables affect this example?**
8. **What are their current defaults?**
9. **What is safe for a demo but not production-ready?**
10. **How would I test this without relying on a live LLM?**

If a junior developer can answer those ten questions for every example in this repository, they are learning **AI engineering**, not merely learning how to copy agent code.

---

# 32. Per-File Environment Variable Checklist

This section is intentionally repetitive. Before running **any individual example file**, use the matching block below.

Legend:

- **Required** — the file needs this value or dependency to run as intended.
- **Optional** — the file can run without it or the feature is optional.
- **Default** — the current value documented in `agent_harness_examples/.env.example`.
- **External service** — something that must be running outside Python.

---

## 32.1 `01-getting_started/agent_example-1.py`

### Environment variables to set

```dotenv
MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
OLLAMA_BASE_URL=http://localhost:11434/v1
```

| Variable | Required? | Default | Purpose |
|---|---|---:|---|
| `MODEL_NAME` | Yes | `granite4.1:8b` | LLM used by the agent. |
| `MAX_TOKENS` | Usually | `256` | Maximum output-token setting where read by the example/model config. |
| `OLLAMA_BASE_URL` | Yes for local Ollama | `http://localhost:11434/v1` | Ollama-compatible API endpoint. |

External service:

- **Ollama** running locally.
- The selected model must already be installed/pulled.

Optional telemetry:

```dotenv
LOGFIRE_TOKEN=...
```

Only needed if the example is configured to export to Logfire.

---

## 32.2 `01-getting_started/agent_example-2.py`

### Environment variables to set

```dotenv
MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Optional persistent memory:

```dotenv
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_memory
MONGODB_COLLECTION=conversations
```

Optional Logfire:

```dotenv
LOGFIRE_TOKEN=...
```

| Variable | Required? | Default | Purpose |
|---|---|---:|---|
| `MODEL_NAME` | Yes | `granite4.1:8b` | Main model. |
| `MAX_TOKENS` | Usually | `256` | Output token cap. |
| `OLLAMA_BASE_URL` | Yes for Ollama | `http://localhost:11434/v1` | Local model endpoint. |
| `MONGODB_URI` | Optional | `mongodb://localhost:27017` | Enables durable long-term memory. |
| `MONGODB_DATABASE` | Optional | `agent_memory` | Mongo database. |
| `MONGODB_COLLECTION` | Optional | `conversations` | Conversation collection. |
| `LOGFIRE_TOKEN` | Optional | placeholder | Enables Logfire export if that provider is enabled. |

External services:

- Ollama required.
- MongoDB only if using the persistent-memory path.
- Any OTEL/Logfire backend only if enabled.

---

## 32.3 `01-getting_started/agent_example-3.py`

### Environment variables to set

```dotenv
MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
OLLAMA_BASE_URL=http://localhost:11434/v1
```

| Variable | Required? | Default | Purpose |
|---|---|---:|---|
| `MODEL_NAME` | Yes | `granite4.1:8b` | Model used to produce typed invoice output. |
| `MAX_TOKENS` | Usually | `256` | Output-token cap. |
| `OLLAMA_BASE_URL` | Yes for Ollama | `http://localhost:11434/v1` | Local model endpoint. |

External service:

- Ollama.

---

# 32.4 `02-error_handling`

The current `.env.example` indicates that the error-handling examples do **not require environment variables**.

That is intentional: many files inject failures directly with fake/broken providers.

### `01_basic_handler.py`

```text
No environment variables required.
```

### `02_source_routing.py`

```text
No environment variables required.
```

### `03_custom_recovery.py`

```text
No environment variables required.
```

### `04_tool_errors.py`

```text
No environment variables required.
```

### `05_guardrail_errors.py`

```text
No environment variables required.
```

### `06_evaluator_errors.py`

```text
No environment variables required.
```

### `07_memory_errors.py`

```text
No environment variables required.
```

### `08_prompt_errors.py`

```text
No environment variables required.
```

### `09_pipeline_error_recovery.py`

```text
No environment variables required.
```

Some examples may still use a real local model path in specific scenarios. If so, use:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/v1
MODEL_NAME=granite4.1:8b
```

unless the file itself hard-codes another model.

---

# 32.5 `03-evaluators`

## `01_quality_check.py`

### Environment variables to set

```dotenv
MODEL_NAME=granite4.1:8b
QUALITY_CHECK_MODEL=ollama:gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

| Variable | Required? | Default | Purpose |
|---|---|---:|---|
| `MODEL_NAME` | Yes | `granite4.1:8b` | Primary agent model. |
| `QUALITY_CHECK_MODEL` | Yes | `ollama:gpt-oss:20b` | LLM judge model. |
| `OLLAMA_BASE_URL` | Yes for Ollama | `http://localhost:11434/v1` | Ollama endpoint. |

External service:

- Ollama with both required models available.

## `02_safety_check.py`

### Environment variables to set

```dotenv
MODEL_NAME=granite4.1:8b
SAFETY_CHECK_MODEL=omni-moderation-2024-09-26
OPENAI_API_KEY=your-real-openai-key
OLLAMA_BASE_URL=http://localhost:11434/v1
```

| Variable | Required? | Default | Purpose |
|---|---|---:|---|
| `MODEL_NAME` | Yes | `granite4.1:8b` | Primary agent model. |
| `SAFETY_CHECK_MODEL` | Yes | `omni-moderation-2024-09-26` | OpenAI moderation model. |
| `OPENAI_API_KEY` | Yes for moderation | placeholder | Authenticates the moderation API call. |
| `OLLAMA_BASE_URL` | Yes if primary model is Ollama | `http://localhost:11434/v1` | Primary model endpoint. |

External services:

- Ollama for the main agent.
- OpenAI API access for moderation.

## `03_custom_evaluator.py`

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No evaluator-specific environment variables are required.

## `04_protocol_evaluator.py`

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No evaluator-specific environment variables are required.

## `05_combined_evaluators.py`

```dotenv
MODEL_NAME=granite4.1:8b
QUALITY_CHECK_MODEL=ollama:gpt-oss:20b
SAFETY_CHECK_MODEL=omni-moderation-2024-09-26
OPENAI_API_KEY=your-real-openai-key
OLLAMA_BASE_URL=http://localhost:11434/v1
```

This file needs the union of variables used by the configured evaluators.

---

# 32.6 `04-guardrails`

All guardrail examples use:

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_agent_retries.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
AGENT_RETRIES_MAX_RETRIES=3
AGENT_RETRIES_TIMEOUT=10
AGENT_RETRIES_BACKOFF=2.0
AGENT_RETRIES_FALLBACK_MODEL=ollama:gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_tool_retries.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
TOOL_RETRIES_AGENT_MAX_RETRIES=2
TOOL_RETRIES_AGENT_TIMEOUT=30
TOOL_RETRIES_MAX_RETRIES=3
TOOL_RETRIES_BACKOFF=1.5
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_result_validator_retries.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
VALIDATOR_AGENT_MAX_RETRIES=2
VALIDATOR_AGENT_TIMEOUT=30
VALIDATOR_MAX_RETRIES=3
VALIDATOR_BACKOFF=2.0
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `04_content_filter.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No content-filter-specific environment variables are currently listed.

## `05_pii_detection.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No PII-specific environment variables are currently listed.

## `06_token_limits.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
TOKEN_LIMITS_EX1_MAX_TOTAL=50
TOKEN_LIMITS_EX1_MAX_OUTPUT=200
TOKEN_LIMITS_EX2_MAX_INPUT=500
TOKEN_LIMITS_EX2_MAX_OUTPUT=100
TOKEN_LIMITS_EX2_MAX_TOTAL=600
TOKEN_LIMITS_EX3_MAX_TOTAL=5
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `07_cost_limits.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
COST_LIMITS_INPUT_COST=0.000003
COST_LIMITS_OUTPUT_COST=0.000015
COST_LIMITS_EX1_MAX_TOTAL=0.005
COST_LIMITS_EX2_MAX_INPUT=0.001
COST_LIMITS_EX2_MAX_OUTPUT=0.003
COST_LIMITS_EX3_MAX_TOTAL=0.000001
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `08_circuit_breaker.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
CIRCUIT_BREAKER_BAD_MODEL_NAME=this-model-does-not-exist
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=5
OLLAMA_BASE_URL=http://localhost:11434/v1
```

The bad model value is intentionally invalid so the circuit breaker can be demonstrated.

## `09_all_guardrails.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b

ALL_GUARDRAILS_AGENT_MAX_RETRIES=3
ALL_GUARDRAILS_AGENT_TIMEOUT=30
ALL_GUARDRAILS_AGENT_BACKOFF=2.0

ALL_GUARDRAILS_TOKEN_MAX_TOTAL=500

ALL_GUARDRAILS_INPUT_COST=0.000003
ALL_GUARDRAILS_OUTPUT_COST=0.000015
ALL_GUARDRAILS_MAX_TOTAL_COST=0.01

ALL_GUARDRAILS_CIRCUIT_THRESHOLD=5
ALL_GUARDRAILS_CIRCUIT_TIMEOUT=30

ALL_GUARDRAILS_AGENT2_MAX_RETRIES=2
ALL_GUARDRAILS_AGENT2_TOKEN_MAX_TOTAL=300
ALL_GUARDRAILS_AGENT2_MAX_TOTAL_COST=0.005
ALL_GUARDRAILS_AGENT2_CIRCUIT_THRESHOLD=3

OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `10_turn_limits.py`

```dotenv
GUARDRAILS_MODEL_PROVIDER=ollama
GUARDRAILS_MODEL_NAME=gpt-oss:20b
TURN_LIMITS_EX1_MAX_TURNS=3
TURN_LIMITS_EX2_MAX_TURNS=2
TURN_LIMITS_EX3_MAX_TURNS=1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

---

# 32.7 `05-human_in_the_loop`

The current `.env.example` lists no HITL-specific variables.

If the examples use the shared local model configuration, use:

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_review_approval.py`

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_tool_review.py`

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_multistep_tools.py`

```dotenv
MODEL_NAME=granite4.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No additional HITL-specific variables are currently documented.

---

# 32.8 `06-loops`

## `01_interactive_loop.py`

```dotenv
MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_iterative_refinement.py`

```dotenv
MODEL_NAME=granite4.1:8b
MAX_TOKENS=256
REFINEMENT_MAX_WORDS=10
REFINEMENT_MAX_ATTEMPTS=3
REFINEMENT_TEMPERATURE=0.7
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_react_loop.py`

```dotenv
REACT_MODEL_NAME=qwen3.5:4b
MAX_TOKENS=256
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `04_goal_seeking_loop.py`

```dotenv
MODEL_NAME=granite4.1:8b
GOAL_MAX_ATTEMPTS=10
GOAL_LIFESPAN_MIN=7
GOAL_LIFESPAN_MAX=12
GOAL_TEMPERATURE=0.7
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `05_planning_loop.py`

```dotenv
PLANNING_MODEL_NAME=llama3.1:8b
PLAN_MAX_STEPS=6
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `text_to_summarize.txt`

```text
No environment variables. This is a data fixture.
```

---

# 32.9 `07-orchestration`

All four examples use the same shared configuration:

```dotenv
MODEL_NAME=granite4.1:8b
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_delegation.py`

Use the block above.

## `02_sequential_pipeline.py`

Use the block above.

## `03_routing.py`

Use the block above.

## `04_parallel_fanout.py`

Use the block above.

There are currently no additional per-file orchestration variables in `.env.example`.

---

# 32.10 `08-structured_output`

All files use:

```dotenv
STRUCTURED_OUTPUT_MODEL_NAME=phi4-mini
STRUCTURED_OUTPUT_REASONING_MODEL=phi4-mini-reasoning
STRUCTURED_OUTPUT_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_simple_model.py`

```dotenv
STRUCTURED_OUTPUT_MODEL_NAME=phi4-mini
STRUCTURED_OUTPUT_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_enums_literals.py`

```dotenv
STRUCTURED_OUTPUT_MODEL_NAME=phi4-mini
STRUCTURED_OUTPUT_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_nested_models.py`

```dotenv
STRUCTURED_OUTPUT_MODEL_NAME=phi4-mini
STRUCTURED_OUTPUT_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `04_validation_retries.py`

```dotenv
STRUCTURED_OUTPUT_REASONING_MODEL=phi4-mini-reasoning
STRUCTURED_OUTPUT_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Use the reasoning model where the file explicitly selects it; otherwise use the standard structured-output model.

---

# 32.11 `09-tool_calling`

All files use:

```dotenv
TOOL_CALLING_MODEL_NAME=gpt-oss:20b
TOOL_CALLING_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_plain_tools.py`

Use the block above.

## `02_context_tools.py`

Use the block above.

## `03_mcp_server.py`

Use:

```dotenv
TOOL_CALLING_MODEL_NAME=gpt-oss:20b
TOOL_CALLING_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
MCP_HTTP_URL=https://mcp.context7.com/mcp
MCP_COMPLEX_URL=https://mcpplaygroundonline.com/mcp-complex-server
```

`MCP_HTTP_URL` defaults to Context7 (public, keyless). `MCP_COMPLEX_URL` defaults to the Complex server for multi-server demos. Override either to point at your own MCP server.

## `04_tool_combinations.py`

Use the shared block above.

---

# 32.12 `10-prompts`

All files:

```dotenv
PROMPTS_MODEL_NAME=gpt-oss:20b
PROMPTS_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_static_prompts.py`

```dotenv
PROMPTS_MODEL_NAME=gpt-oss:20b
PROMPTS_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_mongo_prompts.py`

```dotenv
PROMPTS_MODEL_NAME=gpt-oss:20b
PROMPTS_MAX_TOKENS=512
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_prompts
MONGODB_COLLECTION=prompts
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Ollama
- MongoDB

## `03_prompt_variables.py`

```dotenv
PROMPTS_MODEL_NAME=gpt-oss:20b
PROMPTS_MAX_TOKENS=512
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_prompts
MONGODB_COLLECTION=prompts
OLLAMA_BASE_URL=http://localhost:11434/v1
```

This example needs MongoDB if it loads the variable-driven prompt template from the Mongo prompt provider.

---

# 32.13 `11-memory`

Shared memory-model settings:

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
REASONING_MODEL_NAME=phi4-mini-reasoning
MEMORY_MAX_TOKENS=512
MEMORY_SHORT_TERM_MAX_TURNS=10
MEMORY_LONG_TERM_MAX_TURNS=100
MEMORY_AUDIT_MAX_TURNS=10000
MEMORY_MONGODB_TIMEOUT_MS=2000
MEMORY_REDIS_CONNECT_TIMEOUT=2
MEMORY_REASONING_TIMEOUT=60
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_in_memory.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
MEMORY_SHORT_TERM_MAX_TURNS=10
MEMORY_LONG_TERM_MAX_TURNS=100
OLLAMA_BASE_URL=http://localhost:11434/v1
```

No external storage service required.

## `02_message_history.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
MEMORY_SHORT_TERM_MAX_TURNS=10
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_multi_provider.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
MEMORY_SHORT_TERM_MAX_TURNS=10
MEMORY_LONG_TERM_MAX_TURNS=100
MEMORY_AUDIT_MAX_TURNS=10000
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services depend on which providers the file enables.

## `04_memory_operations.py`

```dotenv
MEMORY_SHORT_TERM_MAX_TURNS=10
MEMORY_LONG_TERM_MAX_TURNS=100
MEMORY_AUDIT_MAX_TURNS=10000
```

Model variables are only needed if the file also runs the agent rather than only provider operations.

## `05_mongo_memory.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_memory
MONGODB_COLLECTION=conversations
MEMORY_MONGODB_TIMEOUT_MS=2000
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Ollama
- MongoDB

## `06_redis_memory.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_KEY_PREFIX=agent:memory:
MEMORY_REDIS_CONNECT_TIMEOUT=2
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Ollama
- Redis

## `07_combined_memory.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_KEY_PREFIX=agent:memory:
MEMORY_REDIS_CONNECT_TIMEOUT=2

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_memory
MONGODB_COLLECTION=conversations
MEMORY_MONGODB_TIMEOUT_MS=2000

OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Ollama
- Redis
- MongoDB

## `08_elasticsearch_memory.py`

```dotenv
MEMORY_MODEL_NAME=gpt-oss:20b
MEMORY_MAX_TOKENS=512
ELASTICSEARCH_ENDPOINT=http://localhost:9200
ELASTICSEARCH_INDEX=agent-memory
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Ollama
- Elasticsearch

## `09_reasoning_traces.py`

```dotenv
REASONING_MODEL_NAME=phi4-mini-reasoning
MEMORY_MAX_TOKENS=512
MEMORY_REASONING_TIMEOUT=60
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External service:

- Ollama with the reasoning model available.

---

# 32.14 `12-observability`

Shared:

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_logging.py`

No model is necessarily required if running only the logging-provider demo.

For file logging, no environment variable is required unless the file explicitly reads one.

## `02_tracing_metrics.py`

No external telemetry service is required because the example uses in-memory tracer/metrics providers.

If the file runs a real agent:

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_builder_logs_metrics.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `04_composite_logs.py`

No required environment variable if only demonstrating log fan-out.

If attached to a live agent, also set the shared model variables.

## `05_elasticsearch_logging.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
ELASTICSEARCH_ENDPOINT=http://localhost:9200
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Elasticsearch
- Ollama if the live agent path is run

## `06_otel_jaeger_logs_traces_metrics.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
JAEGER_OTLP_ENDPOINT=localhost:14317  # deprecated
OBSERVABILITY_SERVICE_NAME=all-in-one-observability-demo
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Jaeger / OTLP receiver
- Ollama

## `07_prometheus_logs_metrics.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
PROMETHEUS_PUSH_GATEWAY=http://localhost:9091
OBSERVABILITY_SERVICE_NAME=all-in-one-observability-demo
OLLAMA_BASE_URL=http://localhost:11434/v1
```

External services:

- Prometheus Pushgateway
- Ollama if live-agent path is used

## `08_live_agent_logs_metrics.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Add backend-specific variables for whichever logger/metrics providers are enabled.

## `09_otel_oltp_logs_traces_metrics.py`

```dotenv
OBSERVABILITY_MODEL_NAME=gpt-oss:20b
OBSERVABILITY_MAX_TOKENS=512
OTEL_COLLECTOR_ENDPOINT=localhost:4317
OBSERVABILITY_SERVICE_NAME=all-in-one-observability-demo
OLLAMA_BASE_URL=http://localhost:11434/v1
```

If the collector forwards to other systems, those backend endpoints are usually configured in the collector rather than directly in this Python example.

---

# 32.15 `13-rag`

The current `.env.example` states that the RAG example does **not require environment variables** because the model is hard-coded in `agent.py`.

## `agent.py`

```text
No environment variables required by the current example.
```

External requirement:

- Whatever local model/runtime is referenced directly in `agent.py` must be available.

If you refactor the example to configuration-driven model selection, use:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/v1
MODEL_NAME=<model used by agent.py>
```

## `sample_data/*`

```text
No environment variables. These are source-data fixtures.
```

## `pyproject.toml`

```text
No environment variables. Dependency metadata only.
```

## `uv.lock`

```text
No environment variables. Dependency lock file only.
```

---

# 32.16 `14-logging`

Shared:

```dotenv
LOGGING_MODEL_PROVIDER=ollama
LOGGING_MODEL_NAME=gpt-oss:20b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `01_basic_log_context.py`

```dotenv
LOGGING_MODEL_PROVIDER=ollama
LOGGING_MODEL_NAME=gpt-oss:20b
LOGGING_01_PIPELINE=content-qa
LOGGING_01_AGENT_ROLE=assistant
LOGGING_01_STAGE_1=math-check
LOGGING_01_STAGE_2=knowledge-check
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `02_custom_enricher.py`

```dotenv
LOGGING_MODEL_PROVIDER=ollama
LOGGING_MODEL_NAME=gpt-oss:20b
LOGGING_02_PIPELINE=enrichment-demo
LOGGING_02_APP_VERSION=2.1.0
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## `03_pipeline_logging.py`

```dotenv
LOGGING_MODEL_PROVIDER=ollama
LOGGING_MODEL_NAME=gpt-oss:20b
LOGGING_03_PIPELINE=content-qa
LOGGING_03_ROLE_RESEARCHER=researcher
LOGGING_03_ROLE_WRITER=writer
LOGGING_03_ROLE_EDITOR=editor
OLLAMA_BASE_URL=http://localhost:11434/v1
```

---

# 32.17 `15-messaging`

## `setup_queues.sh`

No model variables are needed to create the queue topology.

RabbitMQ connection information may be embedded in the script or supplied by the local environment depending on the current script contents.

The current centralized model variables are:

```dotenv
RABBITMQ_AGENT_MODEL_NAME=gemma4:12b-mlx
RABBITMQ_AGENT_MODEL_SETTINGS={"thinking": false, "max_tokens": 512, "temperature": 0.1}
```

These are relevant to the RabbitMQ **agent** example referenced by the repository documentation, not necessarily to the queue-setup shell script itself.

## `post_labs_message.sh`

No LLM environment variables are required just to publish a sample message.

RabbitMQ connection values may be script-local or environment-driven depending on the shell script.

## Referenced RabbitMQ agent configuration

If using the agent path referenced by the root README / `.env.example`:

```dotenv
RABBITMQ_AGENT_MODEL_NAME=gemma4:12b-mlx
RABBITMQ_AGENT_MODEL_SETTINGS={"thinking": false, "max_tokens": 512, "temperature": 0.1}
```

The current numbered directory does not contain the referenced Python agent file, so verify the current repository tree before attempting to run it.

---

# 33. Fastest Way to Use This Section

Before running a file, search this document for the exact filename.

Example:

```text
Search: 06_token_limits.py
```

Then copy the `.env` block immediately beneath it.

That block is the **minimum practical configuration checklist for that example**, while the earlier sections explain why each variable and capability exists.
