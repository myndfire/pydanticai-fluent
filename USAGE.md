# agent_harness — Usage Guide

## 1. Installation & Setup

The project uses [uv](https://docs.astral.sh/uv/) for package management. There are two workspaces:

- **`agent_harness/`** — the core library (`agent-harness` package, src-layout)
- **`agent_harness_examples/`** — example scripts that depend on `agent_harness` via a local path source

```bash
# 1. Install core library dependencies
cd agent_harness
uv sync

# 2. Install examples dependencies (pulls in agent_harness as an editable dependency)
cd ../agent_harness_examples
uv sync
```

The examples project declares the dependency in `pyproject.toml`:

```toml
[tool.uv.sources]
agent-harness = { path = "../agent_harness" }
```

After syncing, run any example:

```bash
uv run agent_example-1.py
```

---

## 2. Core Concept — ManagedAgent & the Fluent API

`ManagedAgent` is the central entry point. Every `.with_*` method returns `self`, enabling method chaining:

```python
agent = (
    ManagedAgent()
    .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
    .with_short_term_memory(InMemoryProvider())
    .with_tools(ToolRegistry().add_many(tool1, tool2))
    .with_prompts(StaticPrompts("You are a helpful bot."))
    .with_observability(Observability())
    .with_error_handling(ErrorHandlingConfig())
)
```

To execute, call `agent.run(prompt, message_history, session_id)`:

```python
history = await MessageHistory().load("my-session", memory_provider)
result = await agent.run("Your prompt here", history, "my-session")
print(result.output)
```

### Defaults

When you omit a `.with_*` call, sensible defaults are applied:

| Concern | Default |
|---|---|
| Model | `ModelConfig(provider="ollama", model_name="gpt-oss:20b")` |
| Prompts | `StaticPrompts()` |
| Observability | `Observability()` → `ConsoleLogger` + `NoOpTracer` + `NoOpMetrics` |
| Tools | `ToolRegistry()` (empty) |
| Evaluators | `[]` (none) |
| Guards | `GuardConfig()` (3 retries, 120s timeout, no guardrails) |
| Memory | `None` (no persistence) |

---

## 3. Quick Start

Here's a complete working agent with two custom tools and an evaluator (based on `agent_example-1.py`):

```python
import asyncio
from agent_harness import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.tools import ToolRegistry
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability
from agent_harness.model_config import ModelConfig
from agent_harness.errorhandling import ErrorHandlingConfig
from agent_harness.evaluators import Evaluator


# Define tools as plain functions with type hints
def repeat(text: str) -> str:
    """Repeat tool — returns the provided text unchanged."""
    print("[tool:repeat] params:", text)
    return text


def shout(text: str) -> str:
    """Shout tool — returns the text in uppercase."""
    print("[tool:shout] params:", text)
    return text.upper()


# Custom evaluator (runs after every agent turn)
class PrintEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:
        print("[Evaluator] Prompt:", prompt)
        print("[Evaluator] Result:", getattr(result, "output", result))


async def main():
    short_term = InMemoryProvider()
    long_term = InMemoryProvider()
    tools = ToolRegistry().add_many(repeat, shout)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_tools(tools)
        .with_prompts(StaticPrompts("You are a helpful bot. Use the provided tools when instructed."))
        .with_observability(Observability())
        .with_error_handling(ErrorHandlingConfig())
        .with_evaluators(PrintEvaluator())
    )

    history = await MessageHistory().load("demo-session", short_term)
    result = await agent.run(
        "First, use repeat to echo 'hello world'. Then use shout on the result.",
        history,
        "demo-session",
    )
    print("\nAgent response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

**Key points:**
- Tools are plain Python functions — `ToolRegistry` inspects signatures and registers them with pydantic-ai.
- If a tool's first parameter is annotated with `RunContext`, it's registered as a context-aware tool; otherwise `tool_plain`.
- `MessageHistory.load(session_id, provider)` reconstructs prior turns from a memory provider so the agent has full conversation context.
- Evaluators run after every turn and receive the prompt, result, and a context dict.

---

## 4. Model Configuration

### `ModelConfig`

```python
from agent_harness.model_config import ModelConfig

ModelConfig(
    provider="openai",        # ProviderType literal (20 supported providers)
    model_name="gpt-4o",      # Model name
    api_key="sk-...",         # Optional — omit for auto-inference
    base_url=None,            # Optional — custom endpoint URL
)
```

### Supported Providers

`ollama`, `openai`, `anthropic`, `google`, `groq`, `mistral`, `bedrock`, `cohere`, `huggingface`, `openrouter`, `grok`, `deepseek`, `cerebras`, `fireworks`, `together`, `azure`, `vercel`, `moonshotai`, `github`, `heroku`

### Two resolution paths

**Auto-infer** (no `api_key` and no `base_url`):
```python
ModelConfig(provider="openai", model_name="gpt-4o")
# → pydantic-ai resolves "openai:gpt-4o" from env (OPENAI_API_KEY)
```

**Explicit** (provide `api_key` and/or `base_url`):
```python
ModelConfig(
    provider="ollama",
    model_name="gpt-oss:20b",
    base_url="http://localhost:11434/v1",
)
# → Constructs OllamaModel + OpenAIProvider explicitly
```

### Custom model settings on `run()`

Pass `model_settings` (a `pydantic_ai.settings.ModelSettings` dict) as kwargs to `run()`:
```python
result = await agent.run(
    prompt, history, session_id,
    model_settings={
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.9,
        "timeout": 30.0,
    },
)
```

---

## 5. Memory

### Memory Providers

All providers implement the `MemoryProvider` protocol:

| Provider | Persistence | Constructor |
|---|---|---|
| `InMemoryProvider` | None (process memory) | `InMemoryProvider(max_turns=100)` |
| `MongoMemory` | MongoDB | `MongoMemory(uri, database, collection)` |
| `RedisMemory` | Redis | `RedisMemory(host, port, db, password, key_prefix)` |
| `ElasticsearchMemory` | Elasticsearch | `ElasticsearchMemory(endpoint, index)` |

### Usage pattern

```python
from agent_harness.memory import InMemoryProvider, MongoMemory, MessageHistory

# Short-term = ephemeral, long-term = persistent
short_term = InMemoryProvider(max_turns=50)
long_term = MongoMemory(uri="mongodb://localhost:27017")

agent = (
    ManagedAgent()
    .with_short_term_memory(short_term)
    .with_long_term_memory(long_term)
)

# Load prior turns into a MessageHistory, then pass it to run()
history = await MessageHistory().load("session-123", short_term)
result = await agent.run("Hello!", history, "session-123")
```

### Persisting turns

Pass `save_to` providers to `run()`:
```python
result = await agent.run(
    prompt, history, session_id,
    save_to=[short_term, long_term],  # persists the turn after completion
)
```

### `MessageHistory`

`MessageHistory` loads turns from a provider and reconstructs them as pydantic-ai `ModelRequest`/`ModelResponse` objects:
```python
history = MessageHistory()
await history.load("session-123", provider)
messages = history.messages  # list[ModelMessage] ready for the agent
```

---

## 6. Tools

### Registering plain functions

```python
from agent_harness.tools import ToolRegistry

def my_tool(param: str) -> str:
    return f"processed: {param}"

tools = ToolRegistry().add(my_tool)
tools.add_many(tool_2, tool_3)
```

### Context-aware tools

If a tool's first parameter is type-annotated with `RunContext`, it's registered via `agent.tool()` instead of `agent.tool_plain()`:

```python
from pydantic_ai import RunContext

def tool_with_context(ctx: RunContext[MyDeps], param: str) -> str:
    # Access dependency injection
    return f"user: {ctx.deps.user_id}, param: {param}"
```

### MCP Server integration

```python
# Single MCP server
agent.with_mcp_server("http://localhost:8000", tool_prefix="mcp_")

# Multiple MCP servers
agent.with_mcp_servers(
    "http://localhost:8000",
    "http://localhost:8001",
    tool_prefix="mcp_",
)
```

---

## 7. Prompts

### Static prompts

```python
from agent_harness.prompts import StaticPrompts

agent.with_prompts(StaticPrompts("You are a helpful assistant."))
```

### MongoDB + Jinja2 templates

```python
from agent_harness.prompts import MongoPrompts

prompts = MongoPrompts(
    uri="mongodb://localhost:27017",
    database="agent_prompts",
    collection="prompts",
)
agent.with_prompts(prompts)
```

MongoDB document schema:
```json
{
    "_id": "customer_support",
    "template": "You are a {{role}} specialized in {{domain}}. Be concise.",
    "active": true,
    "version": 1,
    "metadata": { "tags": ["production"] }
}
```

### Prompt selection & variables

```python
# Select "customer_support" prompt and pass template variables
result = await agent.run(
    "Help with refund",
    history,
    session_id,
    prompt_id="customer_support",
    role="support agent",
    domain="e-commerce",
)
```

`StaticPrompts` ignores `prompt_id` — it always returns its stored string. `MongoPrompts` looks up the document by `prompt_id`, renders with Jinja2 using the kwargs.

---

## 8. Observability

`Observability` is a facade that coordinates **logging**, **tracing**, and **metrics**.

### Quick setup

```python
from agent_harness.observability import Observability
from agent_harness.logging import ConsoleLogger

agent.with_observability(Observability(logger=ConsoleLogger()))
```

### Fluent builder

```python
from agent_harness.observability import ObservabilityBuilder

obs = (
    ObservabilityBuilder("my-agent")
    .with_console_logging()
    .with_file_logging("agent.log")
    .with_logfire_tracing()
    .with_otel_tracing(otlp_endpoint="http://localhost:4317")
    .with_prometheus_metrics(push_gateway="localhost:9091")
    .build()
)
agent.with_observability(obs)
```

### Logging backends

| Class | Description |
|---|---|
| `ConsoleLogger()` | structlog to console |
| `FileLogger(log_file, rotation, retention)` | Timed/file-size rotating file logs |
| `ElasticsearchLogger(endpoint, index_prefix)` | Async ES daily indices |
| `LogfireLogger(service_name)` | Logfire platform |
| `CompositeLogger(*loggers)` | Fans out to multiple loggers |

### Tracing backends

| Class | Description |
|---|---|
| `NoOpTracer()` | No-op |
| `InMemoryTracer()` | Stores spans in memory (for testing) |
| `LogfireTracer(service_name)` | Logfire spans |
| `OTELTracer(service_name, otlp_endpoint)` | OpenTelemetry gRPC |
| `JaegerTracer(service_name, host, port)` | Jaeger UDP |

### Metrics backends

| Class | Description |
|---|---|
| `NoOpMetrics()` | No-op |
| `InMemoryMetrics()` | Stores in dicts (for testing) |
| `LogfireMetrics(service_name)` | Logfire metric events |
| `OTLPMetrics(service_name, otlp_endpoint)` | OpenTelemetry OTLP |
| `PrometheusMetrics(namespace, push_gateway)` | Prometheus client |
| `StatsdMetrics(host, port, prefix)` | StatsD |

---

## 9. Guards & Retries

### Retry configurations

```python
from agent_harness.guards import (
    GuardConfig, AgentRetryConfig, ToolRetryConfig,
    ResultValidatorRetryConfig,
)

# Agent-level retries
agent_retry = AgentRetryConfig(
    max_retries=3,
    timeout=120,              # seconds per attempt
    backoff_multiplier=2.0,
    fallback_model="openai:gpt-4o-mini",  # try cheaper model on exhaustion
)

# Tool-level retries
tool_retry = ToolRetryConfig(max_retries=2, backoff_multiplier=1.5)

# Structured output validation retries
validator_retry = ResultValidatorRetryConfig(max_retries=3)

agent = ManagedAgent(
    guards=GuardConfig(
        agent=agent_retry,
        tool=tool_retry,
        result_validator=validator_retry,
    )
)
```

### Fluent retry setters

```python
agent.with_agent_retries(
    AgentRetryConfig()
    .with_max_retries(5)
    .with_timeout(60)
    .with_fallback("openai:gpt-4o-mini")
    .on_retry(lambda ctx: print(f"Retrying: {ctx.error_type}"))
    .on_error(lambda ctx: backup_handler(ctx))
)
.with_tool_retries(ToolRetryConfig().with_max_retries(2))
.with_result_validator_retries(ResultValidatorRetryConfig().with_max_retries(3))
```

### Built-in guardrails

```python
agent.with_content_filter(ContentFilterConfig(enabled=True))
agent.with_pii_detection(PIIDetectionConfig(enabled=True))
agent.with_cost_limits(CostLimitsConfig(max_tokens_per_request=4096))
agent.with_circuit_breaker(CircuitBreakerConfig(
    enabled=True, failure_threshold=5, circuit_timeout=60,
))
```

Bulk guardrail setter:
```python
agent.with_guardrails(
    content_filter=ContentFilterConfig(),
    pii_detection=PIIDetectionConfig(),
    cost_limits=CostLimitsConfig(max_tokens_per_request=4096),
)
```

### GuardRunner flow

Inside `run()`, `GuardRunner.run_with_guards()`:
1. Executes `agent.run()` inside `asyncio.wait_for(timeout)`
2. On timeout/error: backs off exponentially, retries up to `max_retries`
3. Calls `on_retry` callback on each retry
4. After exhaustion: tries `fallback_model` if configured
5. If fallback also fails: calls `on_error` callback, returns `AgentRunResult(success=False)`

---

## 10. Error Handling

### Custom error handler

```python
from agent_harness.errorhandling import (
    ErrorHandlingConfig, AgentErrorContext,
)

class AgentErrorHandler:
    def __init__(self, obs):
        self.obs = obs

    def __call__(self, ctx: AgentErrorContext, exc: Exception) -> bool:
        # ctx.session_id, ctx.prompt, ctx.source, ctx.error_context
        self.obs.log_error("Agent failed", error=str(exc), session_id=ctx.session_id)
        return False  # False = re-raise, True = suppress and return failed result

error_config = ErrorHandlingConfig().with_error_handler(AgentErrorHandler(obs))
agent.with_error_handling(error_config)
```

### ErrorContext data

```python
@dataclass
class ErrorContext:
    error_type: str
    error_message: str
    source: str          # "tool", "memory", "llm", "unknown"
    session_id: str | None
    prompt: str | None
    stack_trace: str | None
    partial_result: AgentRunResult | None
    attempt: int
    max_attempts: int
    will_retry: bool
```

### Error source detection

`ErrorHandler.determine_error_source(exception)` classifies exceptions into:
- `"memory"` — database/connection errors
- `"llm"` — API errors, rate limits, model errors
- `"tool"` — tool execution failures
- `"unknown"` — everything else

---

## 11. Evaluators

Evaluators run **after every turn** and can inspect or score the agent's output.

### Custom evaluator

```python
from agent_harness.evaluators import Evaluator, CustomEvaluator

class MyEvaluator(Evaluator):
    async def evaluate(self, prompt: str, result, context: dict) -> None:
        print(f"[{context['session_id']}] Output length: {len(str(result.output))}")

agent.with_evaluators(MyEvaluator())
```

### Built-in evaluators

**`QualityCheck`** — LLM-as-judge scoring (0-10):
```python
from agent_harness.evaluators import QualityCheck

agent.with_evaluators(QualityCheck(threshold=7.0, judge_model="openai:gpt-4o-mini"))
```

**`SafetyCheck`** — OpenAI content moderation:
```python
from agent_harness.evaluators import SafetyCheck

agent.with_evaluators(SafetyCheck())
```

### CustomEvaluator base class

Provides `log_info()`, `log_warning()`, `log_error()` helpers:
```python
class MyEvaluator(CustomEvaluator, name="toxicity"):
    async def evaluate(self, prompt, result, context):
        if "bad word" in str(result.output):
            self.log_warning("Toxicity detected")
```

---

## 12. Structured Output

Use `.with_output()` to constrain the agent's response to a Pydantic model:

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    invoice_number: str = Field(..., description="Invoice number")
    date_issued: str
    due_date: str
    currency: str
    customer_name: str
    subtotal: float
    tax_amount: float
    total_amount_due: float

agent = (
    ManagedAgent()
    .with_model(ModelConfig(provider="openai", model_name="gpt-4o"))
    .with_output(Invoice, output_retries=3)
    .with_result_validator_retries(
        ResultValidatorRetryConfig().with_max_retries(3)
    )
)

result = await agent.run(
    "Generate an invoice for consulting services by Acme Corp to Globex Inc for $5000.",
    history, session_id,
)
invoice: Invoice = result.output  # typed!
print(f"Invoice #{invoice.invoice_number}: ${invoice.total_amount_due:.2f}")
```

The agent will retry up to `output_retries` (default 3) times if it fails to produce valid structured output. Configure additional validation retries via `GuardConfig.result_validator`.

---

## 13. RabbitMQ Integration

For message-driven agent workflows (see `document_classification_rabbitmq_agent.py`).

### Fluent queue/exchange configuration

```python
agent = (
    ManagedAgent()
    .with_rabbitmq(
        host="localhost",
        port=5672,
        username="guest",
        password="guest",
        virtual_host="/",
    )
    .with_input_queue("classification_requests")
    .with_input_exchange("classification_exchange")
    .with_output_queue("classification_results")
    .with_output_exchange("results_exchange")
    .with_dead_letter_queue("classification_dlq")
    .with_dead_letter_exchange("dlq_exchange")
)
```

### Manual messaging with `MessagingService`

```python
from agent_harness.rabbitmq import MessagingService

mq = MessagingService(host="localhost", port=5672)
await mq.connect()
await mq.declare_exchange("classification", "direct", durable=True)
await mq.declare_queue("input_queue", durable=True)

async for message in mq.consume("input_queue"):
    body = message.body.decode()
    result = await agent.run(body, history, session_id)
    await mq.publish("output_queue", result.output)
    await mq.ack(message)
```

### MessagingService API

| Method | Description |
|---|---|
| `connect()` | Establish RabbitMQ connection |
| `disconnect()` | Close connection |
| `declare_exchange(name, type, durable)` | Declare an exchange |
| `declare_queue(name, durable)` | Declare a queue |
| `consume(queue_name) -> AsyncIterator[IncomingMessage]` | Async generator |
| `publish(queue_name, message, exchange, delivery_mode)` | Send message |
| `ack(message)` | Acknowledge |
| `nack(message, requeue)` | Negatively acknowledge |
| `is_connected` | Property — connection status |

---

## 14. Environment Variables

The `AgentConfig` class (in `config.py`) is a `pydantic.BaseSettings` class that reads from a `.env` file. It's **not used internally** by `ManagedAgent` — it's offered as a convenience for centralising configuration:

```python
from agent_harness.config import AgentConfig

config = AgentConfig()  # reads .env from cwd
print(config.model_name)  # "ollama:gpt-oss:20b"
```

### Supported environment variables

```
model_name=ollama:gpt-oss:20b
openai_api_key=sk-...
groq_api_key=gsk_...
memory_type=in-memory              # or "mongodb"
mongodb_uri=mongodb://localhost:27017
mongodb_database=agent_memory
mongodb_collection=conversations
prompt_source=static               # or "mongodb"
prompt_mongodb_uri=mongodb://localhost:27017
prompt_database=agent_prompts
prompt_collection=prompts
default_system_prompt=You are a helpful assistant
enable_otel=true
otel_service_name=my-agent
otel_endpoint=http://localhost:4317
elasticsearch_endpoint=http://localhost:9200
elasticsearch_index_prefix=agent-logs
max_retries=3
timeout=120
fallback_model=openai:gpt-4o-mini
```

---

## 15. Running the Examples

All examples are in `agent_harness_examples/`. Run with `uv run`:

```bash
cd agent_harness_examples

# Example 1 — Basic agent with two tools and a custom evaluator
uv run agent_example-1.py

# Example 2 — Error handling, multi-turn conversation, Logfire observability
uv run agent_example-2.py

# Example 3 — Structured output (Invoice model) with .with_output()
uv run agent_example-3.py

# Document Classification — Full RabbitMQ pipeline
uv run document_classification_rabbitmq_agent.py
```

**Prerequisites:**
- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (for Ollama models) or API keys for cloud providers
- MongoDB (optional, for persistent memory in examples 2/3)
- RabbitMQ (optional, for the document classification example)

---

## 16. Full Fluent API Reference

### ManagedAgent

| Method | Signature | Description |
|---|---|---|
| `with_model` | `(model: ModelConfig) -> ManagedAgent` | Set the LLM model |
| `with_short_term_memory` | `(provider: MemoryProvider) -> ManagedAgent` | Ephemeral conversation memory |
| `with_long_term_memory` | `(provider: MemoryProvider \| None) -> ManagedAgent` | Persistent conversation memory |
| `with_deps_type` | `(deps_type: type) -> ManagedAgent` | Dependency injection type |
| `with_prompts` | `(provider: PromptProvider) -> ManagedAgent` | System prompt provider |
| `with_observability` | `(obs: Observability) -> ManagedAgent` | Logging, tracing, metrics facade |
| `with_tools` | `(registry: ToolRegistry) -> ManagedAgent` | Tool functions |
| `with_mcp_server` | `(url: str, **kwargs) -> ManagedAgent` | Single MCP server |
| `with_mcp_servers` | `(*urls: str, tool_prefix: str?) -> ManagedAgent` | Multiple MCP servers |
| `with_evaluators` | `(*evaluators: Evaluator) -> ManagedAgent` | Post-turn evaluators |
| `with_error_handling` | `(config: ErrorHandlingConfig) -> ManagedAgent` | Error handler config |
| `with_agent_retries` | `(config: AgentRetryConfig) -> ManagedAgent` | Agent-level retry config |
| `with_tool_retries` | `(config: ToolRetryConfig) -> ManagedAgent` | Tool-level retry config |
| `with_result_validator_retries` | `(config: ResultValidatorRetryConfig) -> ManagedAgent` | Output validation retries |
| `with_content_filter` | `(config: ContentFilterConfig) -> ManagedAgent` | Enable content filtering |
| `with_pii_detection` | `(config: PIIDetectionConfig) -> ManagedAgent` | Enable PII detection |
| `with_cost_limits` | `(config: CostLimitsConfig) -> ManagedAgent` | Max tokens per request |
| `with_circuit_breaker` | `(config: CircuitBreakerConfig) -> ManagedAgent` | Failure threshold + timeout |
| `with_guardrails` | `(content_filter, pii_detection, cost_limits) -> ManagedAgent` | Batch guardrail enable |
| `with_output` | `(output_type: Any, output_retries: int = 3) -> ManagedAgent` | Structured Pydantic output |
| `with_rabbitmq` | `(host, port, username, password, virtual_host) -> ManagedAgent` | RabbitMQ connection |
| `with_input_queue` | `(queue_name: str) -> ManagedAgent` | Input queue name |
| `with_input_exchange` | `(exchange_name: str) -> ManagedAgent` | Input exchange name |
| `with_output_queue` | `(queue_name: str) -> ManagedAgent` | Output queue name |
| `with_output_exchange` | `(exchange_name: str) -> ManagedAgent` | Output exchange name |
| `with_dead_letter_queue` | `(queue_name: str) -> ManagedAgent` | Dead letter queue name |
| `with_dead_letter_exchange` | `(exchange_name: str) -> ManagedAgent` | Dead letter exchange name |

### Agent.run()

```python
async def run(
    self,
    prompt: str,                        # User prompt
    message_history: MessageHistory,     # Loaded history (required)
    session_id: str,                     # Session key (required)
    save_to: list[MemoryProvider] | None = None,  # Persist turn to these providers
    deps: Any = None,                    # Dependency injection value
    **kwargs,                            # prompt_id, template vars, model_settings, etc.
) -> AgentRunResult
```

### AgentRunResult

```python
@dataclass
class AgentRunResult:
    output: Any                    # The response (str or Pydantic model)
    success: bool                  # Whether the run succeeded
    error_context: ErrorContext | None  # Error details if failed
    used_fallback: bool            # Whether fallback model was used
    new_messages: list[ModelMessage]    # Raw pydantic-ai messages
    usage: Any                     # Token usage data
```

---

## 17. Architecture & Data Flow

```
agent.run(prompt, message_history, session_id)
  │
  ├─ Observability.observe("agent_run")  ← async context manager
  │    ├─ loggers: "agent_run_started"
  │    ├─ metrics: counter("agent_runs_total")
  │    └─ tracers: span("agent_run")
  │
  ├─ MessageHistory.load(session_id, short_term_memory)
  ├─ MessageHistory.load(session_id, long_term_memory)
  │
  ├─ prompts.get_system_prompt(prompt_id, **kwargs)
  │
  ├─ GuardRunner.run_with_guards(agent, prompt, messages, deps)
  │    ├─ for attempt in range(max_retries):
  │    │    asyncio.wait_for(agent.run(), timeout)
  │    │    ✓ success → return AgentRunResult(success=True)
  │    │    ✗ timeout/error → backoff, on_retry callback, retry
  │    ├─ exhaust retries + fallback_model → run fallback
  │    └─ fallback fail + on_error → return AgentRunResult(success=False)
  │       or re-raise
  │
  ├─ extract_clean_output(result)  [if no structured output type]
  ├─ TurnData(messages, usage, duration, model, status)
  ├─ save_to providers → provider.save_turn(session_id, turn)
  ├─ observability → log token usage
  ├─ evaluators → for each evaluator: evaluate(prompt, result, context)
  │
  └─ [on exception] ErrorHandler.handle_error(exception, source, session_id, prompt)
       ├─ handler returns True  → return AgentRunResult(success=False)
       └─ handler returns False → re-raise
```
