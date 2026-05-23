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
| --- | --- |
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

| Provider | Model class | Provider class | Auth | Env var |
|---|---|---|---|---|
| `ollama` | `OpenAIChatModel` | `OllamaProvider` | None (local) | `OLLAMA_BASE_URL` |
| `openai` | `OpenAIChatModel` | `OpenAIProvider` | API key | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicModel` | `AnthropicProvider` | API key | `ANTHROPIC_API_KEY` |
| `google` | `GoogleModel` | `GoogleProvider` | API key | `GOOGLE_API_KEY` |
| `groq` | `GroqModel` | `GroqProvider` | API key | `GROQ_API_KEY` |
| `mistral` | `MistralModel` | `MistralProvider` | API key | `MISTRAL_API_KEY` |
| `bedrock` | `BedrockConverseModel` | `BedrockProvider` | AWS credentials | `AWS_ACCESS_KEY_ID` |
| `cohere` | `CohereModel` | `CohereProvider` | API key | `COHERE_API_KEY` |
| `huggingface` | `HuggingFaceModel` | `HuggingFaceProvider` | API key | `HUGGINGFACE_API_KEY` |
| `openrouter` | `OpenAIChatModel` | `OpenRouterProvider` | API key | `OPENROUTER_API_KEY` |
| `grok` | `OpenAIChatModel` | `GrokProvider` | API key | `GROK_API_KEY` |
| `deepseek` | `OpenAIChatModel` | `DeepSeekProvider` | API key | `DEEPSEEK_API_KEY` |
| `cerebras` | `OpenAIChatModel` | `CerebrasProvider` | API key | `CEREBRAS_API_KEY` |
| `fireworks` | `OpenAIChatModel` | `FireworksProvider` | API key | `FIREWORKS_API_KEY` |
| `together` | `OpenAIChatModel` | `TogetherProvider` | API key | `TOGETHER_API_KEY` |
| `azure` | `OpenAIChatModel` | `AzureProvider` | API key | `AZURE_API_KEY` |
| `vercel` | `OpenAIChatModel` | `VercelProvider` | API key | `VERCEL_API_KEY` |
| `moonshotai` | `OpenAIChatModel` | `MoonshotAIProvider` | API key | `MOONSHOTAI_API_KEY` |
| `github` | `OpenAIChatModel` | `GitHubProvider` | API key | `GITHUB_API_KEY` |
| `heroku` | `OpenAIChatModel` | `HerokuProvider` | API key | `HEROKU_API_KEY` |

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

All providers implement the `MemoryProvider` protocol with five async methods:

| Provider | Constructors | Persistence | Use case |
|---|---|---|---|
| `InMemoryProvider` | `(max_turns: int = 100)` | None — per session `defaultdict[str, list[TurnData]]` | Dev, testing, short-term conversation cache. Turns are lost on process exit. Automatically trims to `max_turns` per session. |
| `MongoMemory` | `(uri: str, database: str = "agent_memory", collection: str = "conversations")` | MongoDB (motor async driver) | Production long-term memory. Each turn is stored as a document `{session_id, turn}` with lazy connection via `AsyncIOMotorClient`. |
| `RedisMemory` | `(host: str = "localhost", port: int = 6379, db: int = 0, password: str \| None = None, key_prefix: str = "agent:memory:")` | Redis | Production memory with sub-millisecond reads. Turns stored as JSON strings in Redis lists (`RPUSH` + `LTRIM` to keep last 100). |
| `ElasticsearchMemory` | `(endpoint: str = "http://localhost:9200", index: str = "agent-memory")` | Elasticsearch | Production memory with full-text search. Auto-creates index with mappings. Document ID: `{session_id}:{turn_id}`. |

### MemoryProvider protocol methods

```python
class MemoryProvider(Protocol):
    async def save_turn(self, session_id: str, turn: TurnData) -> None: ...
    async def load_turns(self, session_id: str, limit: int | None = None) -> list[TurnData]: ...
    async def get_turn(self, session_id: str, turn_id: str) -> TurnData | None: ...
    async def delete_turn(self, session_id: str, turn_id: str) -> bool: ...
    async def clear(self, session_id: str) -> None: ...
```

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

### ToolRegistry API

All methods return `self` for chaining.

| Method | Signature | Description |
|---|---|---|
| `add` | `(func: Callable) -> ToolRegistry` | Register a single tool function. |
| `add_many` | `(*funcs: Callable) -> ToolRegistry` | Register multiple tool functions at once. |
| `add_mcp` | `(server: str, endpoint: str \| None = None) -> ToolRegistry` | Placeholder for MCP server tool discovery. |
| `clear` | `() -> ToolRegistry` | Remove all registered tools. |
| `get_tools` | `() -> list[Callable]` | Return a copy of the registered tool list. |
| `register_to_agent` | `(agent: pydantic_ai.Agent) -> None` | Registers all tools with the underlying PydanticAI agent. Detects context-aware tools by inspecting the first parameter annotation for `RunContext`. |

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

### PromptProvider backends

| Backend | Constructor | Description |
|---|---|---|
| `StaticPrompts` | `(system_prompt: str = "You are a helpful assistant")` | Returns the fixed string on every `get_system_prompt()` call. Ignores `prompt_id` and template variables. Simplest option for single-purpose agents. |
| `MongoPrompts` | `(uri: str, database: str = "agent_prompts", collection: str = "prompts")` | Loads Jinja2 templates from MongoDB. Caches compiled templates in memory. Supports `list_prompts()`, `create_prompt()`, `update_prompt()`, and `clear_cache()`. Ideal for multi-tenant or dynamically updated prompts. |

### MongoPrompts API

| Method | Signature | Description |
|---|---|---|
| `get_system_prompt` | `async (prompt_id: str = "default", **variables) -> str` | Fetches the document by `_id`, renders the `template` field with Jinja2 using the provided variables. |
| `list_prompts` | `async (active_only: bool = True) -> list[dict]` | List all prompt documents, optionally filtered to active ones only. |
| `create_prompt` | `async (prompt_id: str, template: str, version: int = 1, metadata: dict \| None = None) -> None` | Insert a new prompt document into MongoDB. |
| `update_prompt` | `async (prompt_id: str, template: str \| None = None, active: bool \| None = None, metadata: dict \| None = None) -> None` | Update an existing prompt's template, active status, or metadata. |
| `clear_cache` | `() -> None` | Clear the in-memory Jinja2 template cache. Templates are recompiled on next access. |

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

### Fluent builder API reference

All builder methods return `self` for chaining. Call `.build()` at the end to produce an `Observability` instance.

| Builder method | Signature | Backend | Description |
|---|---|---|---|
| `.with_console_logging()` | `() -> ObservabilityBuilder` | `ConsoleLogger` | Writes structured logs to stdout/stderr via structlog. Good for local dev. |
| `.with_file_logging()` | `(log_file: str = "agent.log") -> ObservabilityBuilder` | `FileLogger` | Writes logs to a rotating file. Rotation defaults to daily; use `"size"` for 10 MB rollover. Keeps 7 days / files by default. Ideal for production when no log aggregator is available. |
| `.with_elasticsearch_logging()` | `(endpoint: str, index_prefix: str = "agent-logs") -> ObservabilityBuilder` | `ElasticsearchLogger` | Ships logs to Elasticsearch with daily indices (`<index_prefix>-YYYY.MM.DD`). Auto-creates indices. Best for production when you use the ELK stack. |
| `.with_logfire_logging()` | `() -> ObservabilityBuilder` | `LogfireLogger` | Sends structured logs to [Logfire](https://logfire.pydantic.dev). Configures structlog with JSON renderer, timestamps, and caller info. Falls back to console if Logfire is unavailable. |
| `.with_logfire_tracing()` | `(send_to_logfire: bool = True, instrument_pydantic_ai: bool = True) -> ObservabilityBuilder` | `LogfireTracer` | Creates Logfire spans for every agent run. When `instrument_pydantic_ai=True`, automatically instruments the underlying PydanticAI agent for detailed LLM call tracing. The Logfire equivalent of OpenTelemetry distributed tracing. |
| `.with_otel_tracing()` | `(otlp_endpoint: str = "localhost:4317", sample_rate: float = 1.0) -> ObservabilityBuilder` | `OTELTracer` | Exports spans via OTLP gRPC to an OpenTelemetry collector (e.g. Grafana, Jaeger, Datadog). `sample_rate` controls trace sampling (1.0 = all traces). Requires `opentelemetry-api`, `opentelemetry-sdk`, and `opentelemetry-exporter-otlp-proto-grpc` packages. |
| `.with_jaeger_tracing()` | `(jaeger_host: str = "localhost", jaeger_port: int = 6831) -> ObservabilityBuilder` | `JaegerTracer` | Sends spans to a Jaeger agent via UDP over the compact Thrift protocol. Lightweight alternative to OTLP when you use Jaeger directly. |
| `.with_prometheus_metrics()` | `(push_gateway: str \| None = None) -> ObservabilityBuilder` | `PrometheusMetrics` | Records counters, gauges, and histograms using the Prometheus client library. If `push_gateway` is set, metrics are pushed to a Prometheus Pushgateway (useful for short-lived jobs). Otherwise, metrics are only accessible via the Python client API. |
| `.with_statsd_metrics()` | `(host: str = "localhost", port: int = 8125) -> ObservabilityBuilder` | `StatsdMetrics` | Sends metrics to a StatsD daemon (Datadog Agent, Telegraf, etc.). Uses `timing` for summary metrics. All metric names are prefixed with `prefix` (default `"agent"`). |
| `.with_in_memory_metrics()` | `() -> ObservabilityBuilder` | `InMemoryMetrics` | Stores all counters, gauges, and histograms in Python dicts. Accessible via `.get_metrics()` for inspection. Useful for unit/integration tests. |
| `.with_logfire_metrics()` | `() -> ObservabilityBuilder` | `LogfireMetrics` | Logs metric events to Logfire as structured info-level messages. No metric protocol — uses Logfire's event ingestion. |
| `.with_logfire_observability()` | `(send_to_logfire: bool = True, include_tracing: bool = True, include_metrics: bool = True) -> ObservabilityBuilder` | All three Logfire | Convenience method that adds Logfire logging, tracing, and metrics in one call. Toggle individual components with the `include_*` flags. |
| `.build()` | `() -> Observability` | — | Constructs and returns the `Observability` instance ready for `.with_observability()`. |

### Logging backends (standalone)

Use these when constructing `Observability(logger=...)` or `Observability(loggers=[...])` directly.

| Class | Constructor | Description |
|---|---|---|
| `ConsoleLogger` | `()` | Writes structured logs to stdout/stderr via structlog. No network dependencies. |
| `FileLogger` | `(log_file: str = "agent.log", rotation: str = "daily", retention: int = 7)` | Rotating file logger. `rotation`: `"daily"` uses `TimedRotatingFileHandler`, `"size"` uses `RotatingFileHandler` (10 MB). `retention`: number of backups to keep. |
| `ElasticsearchLogger` | `(endpoint: str, index_prefix: str = "agent-logs", service_name: str = "agent")` | Async Elasticsearch client. Writes to daily indices. Also mirrors logs locally via structlog. Close with `await logger.close()`. |
| `LogfireLogger` | `(service_name: str = "agent")` | Configures Logfire and structlog together. JSON-formatted output with timestamps, caller info, and stack traces. Gracefully falls back to console. |
| `CompositeLogger` | `(*loggers: Logger)` | Fans out all log calls to every logger in the list. Use when you need logs in multiple destinations simultaneously (e.g. console + file + ES). |

### Tracing backends (standalone)

Use these when constructing `Observability(tracer=...)` or `Observability(tracers=[...])` directly.

| Class | Constructor | Description |
|---|---|---|
| `NoOpTracer` | `()` | All methods are no-ops. Used internally as the default when no tracer is specified. |
| `InMemoryTracer` | `()` | Records spans in a list (`get_spans()`). Call `reset()` to clear. Use for testing or debugging span structure. |
| `LogfireTracer` | `(service_name: str, send_to_logfire: bool = True, instrument_pydantic_ai: bool = True)` | Full Logfire integration. Spans are named `{service_name}.{operation}`. When `instrument_pydantic_ai=True`, auto-instruments the PydanticAI agent for detailed LLM call traces. Supports `notice()`, `set_attribute()`, and `add_event()`. |
| `OTELTracer` | `(service_name: str, otlp_endpoint: str = "localhost:4317", sample_rate: float = 1.0)` | Pure OpenTelemetry tracer. Exports via OTLP gRPC. Creates spans with `{service_name}.{operation}` naming. Records exceptions and attributes. `sample_rate=0.1` traces 10% of runs. |
| `JaegerTracer` | `(service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 6831)` | Jaeger client tracer. Sends spans as Thrift-compact over UDP. Good for local Jaeger all-in-one deployments. |

### Metrics backends (standalone)

Use these when constructing `Observability(metrics=...)` or `Observability(metrics_list=[...])` directly.

| Class | Constructor | Description |
|---|---|---|
| `NoOpMetrics` | `()` | All counter/gauge/histogram/summary calls are no-ops. Default when no metrics backend is configured. |
| `InMemoryMetrics` | `()` | Stores metrics in Python dicts: `_counters`, `_gauges`, `_histograms`, `_summaries`. Access with `get_metrics()`, clear with `reset()`. Perfect for testing. |
| `LogfireMetrics` | `(service_name: str = "agent")` | Sends metric events to Logfire as info-level log entries. No dedicated metric protocol — uses Logfire's structured event system. |
| `OTLPMetrics` | `(service_name: str = "agent", otlp_endpoint: str = "localhost:4319")` | OpenTelemetry metrics via OTLP gRPC. Creates real OTel counters, gauges, and histograms with a `PeriodicExportingMetricReader`. Note: metrics use port 4319 (separate from tracing at 4317). |
| `PrometheusMetrics` | `(namespace: str = "agent", push_gateway: str \| None = None)` | Prometheus client library metrics. Supports `push_to_gateway(job_name)` for push-based workflows. Metric names follow Prometheus naming conventions. |
| `StatsdMetrics` | `(host: str = "localhost", port: int = 8125, prefix: str = "agent")` | Standard StatsD client. `summary()` maps to StatsD `timing()`. Compatible with Datadog Agent, Telegraf, and other StatsD-compatible collectors. |

### Standard metric names

`Observability.observe("agent_run")` automatically records these metrics:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `agent_runs_total` | Counter | `model`, `session_id` | Incremented on every agent run start |
| `agent_errors_total` | Counter | `error_type`, `operation` | Incremented on run failures |
| `agent_duration_seconds` | Histogram | `model`, `status` | Runtime of successful runs |
| `{operation}_total` | Counter | `model`, `session_id` | Generic counter for custom operations |
| `{operation}_duration_seconds` | Histogram | `model`, `status` | Generic histogram for custom operations |

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

### Configuration class reference

**`AgentRetryConfig`** — agent-level retry behaviour:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `3` | Maximum retry attempts for the entire agent run |
| `timeout` | `int` | `120` | Seconds before a single agent call times out |
| `backoff_multiplier` | `float` | `2.0` | Exponential backoff factor between retries |
| `fallback_model` | `str \| None` | `None` | Cheaper/faster model to try after all retries exhausted |
| `on_retry` | `Callable[[ErrorContext], None] \| None` | `None` | Callback invoked on each retry (receives error context) |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Final callback after all retries and fallback fail |

Fluent setters: `.with_max_retries(n)`, `.with_timeout(n)`, `.with_backoff(m)`, `.with_fallback(model)`, `.on_retry(callback)`, `.on_error(callback)`.

**`ToolRetryConfig`** — per-tool retry behaviour:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `3` | Maximum retries for individual tool executions |
| `backoff_multiplier` | `float` | `2.0` | Exponential backoff factor between tool retries |

Fluent setters: `.with_max_retries(n)`, `.with_backoff(m)`.

**`ResultValidatorRetryConfig`** — structured output validation retries:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `3` | Maximum retries when structured output fails validation |
| `backoff_multiplier` | `float` | `2.0` | Exponential backoff factor between validation retries |

Fluent setters: `.with_max_retries(n)`, `.with_backoff(m)`.

**`GuardConfig`** — combines all retry configs and guardrail toggles:

| Field | Type | Default | Description |
|---|---|---|---|
| `agent` | `AgentRetryConfig` | `AgentRetryConfig()` | Agent-level retry settings |
| `tool` | `ToolRetryConfig` | `ToolRetryConfig()` | Tool-level retry settings |
| `result_validator` | `ResultValidatorRetryConfig` | `ResultValidatorRetryConfig()` | Output validation retry settings |
| `enable_content_filter` | `bool` | `False` | Enable content filtering guardrail |
| `enable_pii_detection` | `bool` | `False` | Enable PII detection guardrail |
| `enable_cost_limits` | `bool` | `False` | Enable cost/token limiting |
| `max_tokens_per_request` | `int \| None` | `None` | Token cap when cost limits enabled |
| `enable_circuit_breaker` | `bool` | `False` | Enable circuit breaker |
| `failure_threshold` | `int` | `5` | Consecutive failures before circuit opens |
| `circuit_timeout` | `int` | `60` | Seconds before circuit half-opens for a trial request |

**`ContentFilterConfig`** — toggles content filtering:

| Field | Type | Default |
|---|---|---|
| `enabled` | `bool` | `True` |

Fluent setter: `.with_enabled(bool)`.

**`PIIDetectionConfig`** — toggles PII detection:

| Field | Type | Default |
|---|---|---|
| `enabled` | `bool` | `True` |

Fluent setter: `.with_enabled(bool)`.

**`CostLimitsConfig`** — caps token usage:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens_per_request` | `int \| None` | `None` | Hard token limit; `None` = unlimited |

Fluent setter: `.with_max_tokens(n)`.

**`CircuitBreakerConfig`** — failure-aware circuit breaker:

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Whether the circuit breaker is active |
| `failure_threshold` | `int` | `5` | Consecutive failures before the circuit opens |
| `circuit_timeout` | `int` | `60` | Seconds to wait before testing with a half-open request |

Fluent setters: `.with_enabled(bool)`, `.with_threshold(n)`, `.with_timeout(n)`.

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
Constructs a separate evaluation prompt asking a judge LLM to rate the response on accuracy and helpfulness. Logs a warning when the score falls below `threshold`. Default judge is `openai:gpt-4o-mini`.

**`SafetyCheck`** — OpenAI content moderation:
```python
from agent_harness.evaluators import SafetyCheck

agent.with_evaluators(SafetyCheck())
```
Sends both the prompt and the agent's output to [OpenAI's Moderations API](https://platform.openai.com/docs/guides/moderation). Logs warnings for any flagged categories (hate, harassment, violence, etc.) with per-category scores. Gracefully skips evaluation if the `openai` package is unavailable.

### Evaluator backends

| Class | Constructor | Description |
|---|---|---|
| `Evaluator` (protocol) | *interface only* | Implement `async def evaluate(self, prompt: str, result, context: dict) -> None`. Receives the raw prompt, the `AgentRunResult` (or `AgentRunResult.output`), and a context dict with `session_id`, `prompt_id`, `model`. |
| `QualityCheck` | `(threshold: float = 7.0, judge_model: str = "openai:gpt-4o-mini")` | LLM-as-judge: calls a separate model to score the output 0-10. Logs warnings below threshold. |
| `SafetyCheck` | `()` | OpenAI Moderations API integration. Flags harmful content with per-category details. No-op if `openai` package not installed. |
| `CustomEvaluator` | `(name: str = "custom")` | Base class providing `log_info()`, `log_warning()`, `log_error()` helpers (prefixed with `[{name}]`). Subclass and override `evaluate()`. |

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

```python
from agent_harness.rabbitmq import MessagingService

mq = MessagingService(
    host="localhost",         # or RABBITMQ_HOST env
    port=5672,                # or RABBITMQ_PORT env
    username="guest",         # or RABBITMQ_USER env
    password="guest",         # or RABBITMQ_PASSWORD env
    virtual_host="/",         # or RABBITMQ_VHOST env
)
```

| Method | Signature | Description |
|---|---|---|
| `connect` | `async () -> None` | Establish the `aio_pika` connection. Must be called before any queue/exchange operations. |
| `disconnect` | `async () -> None` | Gracefully close the connection. |
| `declare_exchange` | `async (name: str, exchange_type: str, durable: bool = True) -> None` | Declare an exchange (`"direct"`, `"topic"`, `"fanout"`, `"headers"`). |
| `declare_queue` | `async (name: str, durable: bool = True) -> None` | Declare a queue. |
| `consume` | `async (queue_name: str) -> AsyncIterator[aio_pika.IncomingMessage]` | Async generator yielding messages from a queue. Loop with `async for`. |
| `publish` | `async (queue_name: str, message: str, exchange: str \| None = None, delivery_mode: int = 2) -> None` | Publish a message to a queue (default routing) or exchange. `delivery_mode=2` = persistent. |
| `ack` | `async (message: aio_pika.IncomingMessage) -> None` | Acknowledge a consumed message (remove from queue). |
| `nack` | `async (message: aio_pika.IncomingMessage, requeue: bool = True) -> None` | Negatively acknowledge (requeue or dead-letter). |
| `is_connected` | `property -> bool` | Whether the RabbitMQ connection is currently established. |

---

## 14. Environment Variables

The `AgentConfig` class (in `config.py`) is a `pydantic.BaseSettings` class that reads from a `.env` file. It's **not used internally** by `ManagedAgent` — it's offered as a convenience for centralising configuration:

```python
from agent_harness.config import AgentConfig

config = AgentConfig()  # reads .env from cwd
print(config.model_name)  # "ollama:gpt-oss:20b"
```

### Supported environment variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"ollama:gpt-oss:20b"` | Default model used by `AgentConfig` |
| `openai_api_key` | `str \| None` | `None` | OpenAI API key |
| `groq_api_key` | `str \| None` | `None` | Groq API key (for GroqCloud) |
| `memory_type` | `str` | `"in-memory"` | Memory backend: `"in-memory"` or `"mongodb"` |
| `mongodb_uri` | `str \| None` | `None` | MongoDB connection string |
| `mongodb_database` | `str` | `"agent_memory"` | MongoDB database name for memory |
| `mongodb_collection` | `str` | `"conversations"` | MongoDB collection for conversation turns |
| `qdrant_url` | `str \| None` | `None` | Qdrant vector DB URL |
| `qdrant_collection` | `str` | `"agent_docs"` | Qdrant collection name |
| `prompt_source` | `str` | `"static"` | Prompt backend: `"static"` or `"mongodb"` |
| `prompt_mongodb_uri` | `str \| None` | `None` | MongoDB URI for prompt storage |
| `prompt_database` | `str` | `"agent_prompts"` | MongoDB database for prompts |
| `prompt_collection` | `str` | `"prompts"` | MongoDB collection for prompt documents |
| `default_system_prompt` | `str` | `"You are a helpful assistant"` | Fallback system prompt text |
| `enable_otel` | `bool` | `False` | Enable OpenTelemetry export |
| `otel_service_name` | `str` | `"agent"` | OTel service name for traces/metrics |
| `otel_endpoint` | `str` | `"http://localhost:4317"` | OTLP collector endpoint |
| `elasticsearch_endpoint` | `str \| None` | `None` | Elasticsearch endpoint for log shipping |
| `elasticsearch_index_prefix` | `str` | `"agent-logs"` | Prefix for ES daily log indices |
| `max_retries` | `int` | `3` | Default max retry attempts |
| `timeout` | `int` | `120` | Default agent timeout in seconds |
| `fallback_model` | `str \| None` | `None` | Fallback model when retries exhausted |
| `file_storage_mongodb_uri` | `str \| None` | `None` | MongoDB URI for GridFS file storage |
| `file_storage_database` | `str \| None` | `None` | MongoDB database for file storage |
| `file_storage_collection` | `str \| None` | `None` | MongoDB collection for file storage |

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

### ManagedAgent constructor

```python
ManagedAgent(
    model: ModelConfig | None = None,
    prompts: PromptProvider | None = None,
    observability: Observability | None = None,
    tools: ToolRegistry | None = None,
    evaluators: list[Evaluator] | None = None,
    guards: GuardConfig | None = None,
    deps_type: type | None = None,
)
```

All parameters are optional. Omitted parameters fall back to sensible defaults (see [Section 2 defaults](#2-core-concept--managedagent--the-fluent-api)).

### Fluent methods

| Method | Signature | Description |
|---|---|---|
| `with_model` | `(model: ModelConfig) -> ManagedAgent` | Replace the underlying LLM model and provider. |
| `with_short_term_memory` | `(provider: MemoryProvider) -> ManagedAgent` | Set ephemeral session memory (e.g. `InMemoryProvider`). |
| `with_long_term_memory` | `(provider: MemoryProvider \| None) -> ManagedAgent` | Set persistent session memory (e.g. `MongoMemory`). Pass `None` to disable. |
| `with_deps_type` | `(deps_type: type) -> ManagedAgent` | Set the PydanticAI dependency injection type for `RunContext[MyDeps]`. |
| `with_prompts` | `(provider: PromptProvider) -> ManagedAgent` | Replace the system prompt provider. |
| `with_observability` | `(obs: Observability) -> ManagedAgent` | Replace the logging/tracing/metrics facade. |
| `with_tools` | `(registry: ToolRegistry) -> ManagedAgent` | Replace the tool registry and register tools with the underlying agent. |
| `with_mcp_server` | `(url: str, **kwargs) -> ManagedAgent` | Add a single MCP SSE server. `tool_prefix` strips a prefix from tool names. |
| `with_mcp_servers` | `(*urls: str, tool_prefix: str \| None = None) -> ManagedAgent` | Add multiple MCP servers. Calls `with_mcp_server` for each URL. |
| `with_evaluators` | `(*evaluators: Evaluator) -> ManagedAgent` | Append evaluators to the list that runs after each turn. |
| `with_error_handling` | `(config: ErrorHandlingConfig) -> ManagedAgent` | Replace the error handling config and rebuild the error handler. |
| `with_agent_retries` | `(config: AgentRetryConfig) -> ManagedAgent` | Set agent-level retry behaviour (max retries, timeout, backoff, fallback). |
| `with_tool_retries` | `(config: ToolRetryConfig) -> ManagedAgent` | Set per-tool retry behaviour. |
| `with_result_validator_retries` | `(config: ResultValidatorRetryConfig) -> ManagedAgent` | Set structured output validation retry behaviour. |
| `with_content_filter` | `(config: ContentFilterConfig) -> ManagedAgent` | Enable or disable content filtering guardrail. |
| `with_pii_detection` | `(config: PIIDetectionConfig) -> ManagedAgent` | Enable or disable PII detection guardrail. |
| `with_cost_limits` | `(config: CostLimitsConfig) -> ManagedAgent` | Set the max tokens per request guardrail. |
| `with_circuit_breaker` | `(config: CircuitBreakerConfig) -> ManagedAgent` | Configure the circuit breaker (failure threshold + timeout). |
| `with_guardrails` | `(content_filter: ContentFilterConfig, pii_detection: PIIDetectionConfig, cost_limits: CostLimitsConfig) -> ManagedAgent` | Set all three guardrail configs at once. |
| `with_output` | `(output_type: type, output_retries: int = 3) -> ManagedAgent` | Set a Pydantic model as the structured output type. The agent will retry up to `output_retries` times to produce valid output. |
| `with_rabbitmq` | `(host: str, port: int, username: str, password: str, virtual_host: str = "/") -> ManagedAgent` | Store RabbitMQ connection parameters (not connected until `run()` is called with queue config). |
| `with_input_queue` | `(queue_name: str) -> ManagedAgent` | Set the RabbitMQ input queue name. |
| `with_input_exchange` | `(exchange_name: str) -> ManagedAgent` | Set the RabbitMQ input exchange name. |
| `with_output_queue` | `(queue_name: str) -> ManagedAgent` | Set the RabbitMQ output queue name. |
| `with_output_exchange` | `(exchange_name: str) -> ManagedAgent` | Set the RabbitMQ output exchange name. |
| `with_dead_letter_queue` | `(queue_name: str) -> ManagedAgent` | Set the RabbitMQ dead-letter queue name. |
| `with_dead_letter_exchange` | `(exchange_name: str) -> ManagedAgent` | Set the RabbitMQ dead-letter exchange name. |

### Properties

| Property | Type | Description |
|---|---|---|
| `last_turn` | `TurnData \| None` | The most recent turn from the last `run()` call, or `None` if `run()` hasn't been called yet. |
| `has_queue_config` | `bool` | Whether RabbitMQ configuration has been set (at minimum host + input queue). |

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
