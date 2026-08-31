# agent_harness — Usage Guide

## 1. Installation & Setup

### Install from a tagged release (recommended for stability)

**With pip:**
```bash
pip install "git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0"
```

**With uv:**
```bash
uv add "git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0"
```

**In your `pyproject.toml`:**
```toml
[project]
dependencies = [
    "pydanticai-fluent @ git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0",
]
```

### Install for local development

The project uses [uv](https://docs.astral.sh/uv/) for package management. There are two workspaces:

- **`agent_harness/`** — the core library (`pydanticai-fluent` package, src-layout)
- **`agent_harness_examples/`** — example scripts that depend on `pydanticai-fluent` via a local path source

```bash
# 1. Install core library dependencies
cd agent_harness
uv sync

# 2. Install examples dependencies (pulls in pydanticai-fluent as an editable dependency)
cd ../agent_harness_examples
uv sync
```

The examples project declares the dependency in `pyproject.toml`:

```toml
[tool.uv.sources]
pydanticai-fluent = { path = "../agent_harness" }
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

### 4.1 `ModelConfig`

`ModelConfig` selects the LLM provider and model. It is a typed dataclass with four fields:

```python
from agent_harness.model_config import ModelConfig

ModelConfig(
    provider="openai",        # ProviderType literal (20 supported providers)
    model_name="gpt-4o",      # Model name (without provider prefix)
    api_key="sk-...",         # Optional — omit for auto-inference from env
    base_url=None,            # Optional — custom endpoint URL
)
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `provider` | `ProviderType` | Yes | `"ollama"` | One of 20 supported providers (see table below) |
| `model_name` | `str` | Yes | `""` | Model identifier without provider prefix |
| `api_key` | `str \| None` | No | `None` | Explicit API key. Overrides the provider's env var. |
| `base_url` | `str \| None` | No | `None` | Custom endpoint URL (e.g., local Ollama) |

#### Supported Providers

| Provider | Model class | Provider class | Auth | Env var |
|---|---|---|---|---|
| `ollama` | `OpenAIChatModel` | `OllamaProvider` | None (local) | `OLLAMA_BASE_URL` |
| `openai` | `OpenAIChatModel` | `OpenAIProvider` | API key | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicModel` | `AnthropicProvider` | API key | `ANTHROPIC_API_KEY` |
| `google` | `GoogleModel` | `GoogleProvider` | API key | `GOOGLE_API_KEY` |
| `groq` | `GroqModel` | `GroqProvider` | API key | `GROQ_API_KEY` |
| `mistral` | `MistralModel` | `MistralProvider` | API key | `MISTRAL_API_KEY` |
| `bedrock` | `BedrockConverseModel` | `BedrockProvider` | AWS credentials | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
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

#### Two resolution paths

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

---

### 4.2 `ModelSettings`

`ModelSettings` (from `pydantic_ai.settings`) controls generation behavior on every inference call. Pass it to `.with_model_settings()` for agent-wide defaults, or to `agent.run()` for per-call overrides.

#### All `ModelSettings` fields

| Field | Type | Description | Supported By |
|---|---|---|---|
| `max_tokens` | `int` | Maximum tokens to generate before stopping | Gemini, Anthropic, OpenAI, Groq, Cohere, Mistral, Bedrock, MCP, xAI |
| `temperature` | `float` | Response randomness (0.0 = deterministic) | Gemini, Anthropic, OpenAI, Groq, Cohere, Mistral, Bedrock, xAI |
| `top_p` | `float` | Nucleus sampling cutoff (alternative to temperature) | Same as `temperature` |
| `top_k` | `int` | Sample only from the top K options for each token | Gemini, Anthropic, Cohere, Bedrock (Anthropic & Amazon Nova) |
| `timeout` | `int \| float \| Timeout` | Per-request timeout override (seconds) | Gemini, Anthropic, OpenAI, Groq, Mistral, xAI |
| `parallel_tool_calls` | `bool` | Allow the model to call multiple tools in parallel | OpenAI, Groq, Anthropic, xAI |
| `tool_choice` | `ToolChoice` | Control which function tools the model can use. Values: `None` (default), `'auto'`, `'none'`, `'required'`, `list[str]`, `ToolOrOutput` | OpenAI, Anthropic, Google, Groq, Mistral, HuggingFace, Bedrock, xAI |
| `seed` | `int` | Random seed for (near-)deterministic results | OpenAI, Groq, Cohere, Mistral, Gemini, xAI |
| `presence_penalty` | `float` | Penalize tokens that have already appeared | OpenAI, Groq, Cohere, Gemini, Mistral, xAI |
| `frequency_penalty` | `float` | Penalize tokens based on their frequency so far | Same as `presence_penalty` |
| `logit_bias` | `dict[str, int]` | Modify likelihood of specific tokens | OpenAI, Groq |
| `stop_sequences` | `list[str]` | Sequences that cause generation to stop | OpenAI, Anthropic, Bedrock, Mistral, Groq, Cohere, Google, xAI |
| `extra_headers` | `dict[str, str]` | Extra HTTP headers to send to the model | OpenAI, Anthropic, Gemini, Groq, xAI |
| `thinking` | `bool \| str` | Enable or configure reasoning/thinking. See dedicated section below. | Anthropic, OpenAI, Gemini, Groq, Bedrock, OpenRouter, Cerebras, xAI, Mistral |
| `service_tier` | `ServiceTier` | Cross-provider service tier (`'auto'`, `'default'`, `'flex'`, `'priority'`) | OpenAI, Anthropic, Bedrock, Google |
| `extra_body` | `object` | Extra fields to include in the request body | OpenAI, Anthropic, Groq |

> **Note:** Not all fields are supported by all providers. Unsupported fields are silently ignored by the provider. When both a unified field (e.g., `service_tier`) and a provider-specific field (e.g., `openai_service_tier`) are set, the provider-specific field takes precedence.

#### Setting `ModelSettings` on the agent

```python
from pydantic_ai.settings import ModelSettings
from agent_harness import ManagedAgent

agent = (
    ManagedAgent()
    .with_model(ModelConfig(provider="ollama", model_name="gemma4:4b-mlx"))
    .with_model_settings(
        ModelSettings(
            thinking=False,        # disable hidden reasoning
            max_tokens=512,        # cap output length
            temperature=0.1,       # low = faster, more deterministic
            timeout=30.0,
        )
    )
)
```

#### Overriding `ModelSettings` per `run()` call

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

#### The `thinking` field — performance impact

The `thinking` field enables hidden chain-of-thought reasoning. It can dramatically increase inference time (2-5x for local models) because the model generates reasoning tokens internally before producing the visible output.

| Value | Behavior |
|---|---|
| `True` | Enable thinking with the provider's default effort level |
| `False` | Disable thinking (silently ignored if the model always thinks) |
| `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` | Enable thinking at a specific effort level |

> **Performance tip:** For simple tasks like document classification, set `thinking=False` and cap `max_tokens` to the minimum needed for your output schema. This is the single biggest win for reducing latency on local models.

---

### 4.3 Environment Variable Configuration

You can configure models entirely through environment variables — no code changes required.

#### Provider API keys

Set only the keys for the providers you use. pydantic-ai reads them automatically when you use auto-infer mode (no explicit `api_key` in `ModelConfig`).

```bash
# Cloud providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GROK_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
FIREWORKS_API_KEY=fw-...
TOGETHER_API_KEY=...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
OPENROUTER_API_KEY=sk-or-...
AZURE_API_KEY=...
VERCEL_API_KEY=...
MOONSHOTAI_API_KEY=...
GITHUB_API_KEY=ghp_...
HEROKU_API_KEY=...

# AWS Bedrock (uses boto3 credential chain)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

#### Ollama (local models)

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1
```

#### Application-level model config

The `AgentConfig` class (see [Section 15](#15-environment-variables)) reads these variables:

| Variable | Type | Default | Description |
|---|---|---|---|
| `MODEL_NAME` | `str` | `"ollama:gpt-oss:20b"` | Full `provider:model_name` string |

#### Per-application env vars

Individual applications and examples may define their own env vars. See the README in each example directory for application-specific variables (e.g., `messaging/rabbitmq/README.md`).

#### JSON-string pattern for `ModelSettings` in `.env`

Any application that wants env-driven model settings can use the JSON-string pattern:

```python
# In your Python code
import os, json
from pydantic_ai.settings import ModelSettings

raw = os.getenv("MYAPP_MODEL_SETTINGS", '{"thinking": false}')
try:
    settings = ModelSettings(**json.loads(raw))
except (json.JSONDecodeError, TypeError):
    settings = ModelSettings(thinking=False)
```

```bash
# In your .env file
MYAPP_MODEL_SETTINGS={"thinking": false, "max_tokens": 512, "temperature": 0.1}
```

> **Tip:** Keep the JSON compact (no newlines) so `.env` parsers handle it correctly.

---

### 4.4 Model Selection Strategy

| Use case | Recommended providers | Notes |
|---|---|---|
| **Local development / testing** | `ollama` | Free, runs on your hardware. Latency depends on your machine (Apple Silicon via MLX, NVIDIA via CUDA). |
| **Fast cloud inference** | `groq`, `cerebras`, `fireworks` | Sub-second response times. Good for high-throughput applications. |
| **High-quality reasoning** | `openai`, `anthropic` | Best output quality. Higher cost and latency. |
| **Cost-sensitive** | `ollama`, `groq`, `together` | Local is free; Groq and Together offer competitive per-token pricing. |

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
    .with_otel_tracing(otlp_endpoint="localhost:4317")
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
| `.with_otel_logging()` | `(otlp_endpoint: str = "localhost:4317") -> ObservabilityBuilder` | `OTELLogger` | Exports structured logs via OTLP gRPC to an OpenTelemetry collector. Log records emitted inside a span automatically carry `trace_id`/`span_id` for log-trace correlation. |
| `.with_logfire_tracing()` | `(send_to_logfire: bool = True, instrument_pydantic_ai: bool = True) -> ObservabilityBuilder` | `LogfireTracer` | Creates Logfire spans for every agent run. When `instrument_pydantic_ai=True`, automatically instruments the underlying PydanticAI agent for detailed LLM call tracing. The Logfire equivalent of OpenTelemetry distributed tracing. |
| `.with_otel_tracing()` | `(otlp_endpoint: str = "localhost:4317", sample_rate: float = 1.0, create_spans: bool = False, record_failures: bool = True) -> ObservabilityBuilder` | `OTELTracer` | Exports spans via OTLP gRPC to an OpenTelemetry collector (e.g. Grafana, Jaeger, Datadog). `sample_rate` controls trace sampling (1.0 = all traces). By default (`create_spans=False`) the harness adds no spans — the trace stream is PydanticAI's native instrumentation only (`invoke_agent`, `execute_tool`, `chat`), so `gen_ai.*` labels are stable to query on. `record_failures=True` still surfaces failures as ERROR spans / exception events (see "Failure telemetry"). Set `create_spans=True` to also export harness spans named `{service_name}.{operation}`. Requires `opentelemetry-api`, `opentelemetry-sdk`, and `opentelemetry-exporter-otlp-proto-grpc` packages. |
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
| `OTELLogger` | `(service_name: str = "agent", otlp_endpoint: str = "localhost:4317")` | OpenTelemetry structured logging via OTLP gRPC. Emits records with `Logger.emit()`; attributes come from `**context` plus callsite `code.file.path` / `code.function` / `code.line.number`. Records emitted inside an active span inherit `trace_id`/`span_id`. Flush with `close()`. |
| `CompositeLogger` | `(*loggers: Logger)` | Fans out all log calls to every logger in the list. Use when you need logs in multiple destinations simultaneously (e.g. console + file + ES). |

### Tracing backends (standalone)

Use these when constructing `Observability(tracer=...)` or `Observability(tracers=[...])` directly.

| Class | Constructor | Description |
|---|---|---|
| `NoOpTracer` | `()` | All methods are no-ops. Used internally as the default when no tracer is specified. |
| `InMemoryTracer` | `()` | Records spans in a list (`get_spans()`). Call `reset()` to clear. Use for testing or debugging span structure. |
| `LogfireTracer` | `(service_name: str, send_to_logfire: bool = True, instrument_pydantic_ai: bool = True)` | Full Logfire integration. Spans are named `{service_name}.{operation}`. When `instrument_pydantic_ai=True`, auto-instruments the PydanticAI agent for detailed LLM call traces. Supports `notice()`, `set_attribute()`, and `add_event()`. |
| `OTELTracer` | `(service_name: str, otlp_endpoint: str = "localhost:4317", sample_rate: float = 1.0, create_spans: bool = False, record_failures: bool = True)` | Pure OpenTelemetry tracer. Exports via OTLP gRPC. With `create_spans=False` (default) it adds no spans of its own; the trace stream is PydanticAI's native instrumentation (`invoke_agent <name>` / `execute_tool <tool>` / `chat <model>`, plus `gen_ai.*` attributes). `record_failures=True` records escaping exceptions as ERROR + exception events (enriching a live span or emitting `{service}.{operation}:failed`). With `create_spans=True` it also creates spans named `{service_name}.{operation}`, records exceptions/attributes, and `sample_rate=0.1` traces 10% of runs. |
| `JaegerTracer` | `(service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 6831)` | Jaeger client tracer. Sends spans as Thrift-compact over UDP. Good for local Jaeger all-in-one deployments. |

### Metrics backends (standalone)

Use these when constructing `Observability(metrics=...)` or `Observability(metrics_list=[...])` directly.

| Class | Constructor | Description |
|---|---|---|
| `NoOpMetrics` | `()` | All counter/gauge/histogram/summary calls are no-ops. Default when no metrics backend is configured. |
| `InMemoryMetrics` | `()` | Stores metrics in Python dicts: `_counters`, `_gauges`, `_histograms`, `_summaries`. Access with `get_metrics()`, clear with `reset()`. Perfect for testing. |
| `LogfireMetrics` | `(service_name: str = "agent")` | Sends metric events to Logfire as info-level log entries. No dedicated metric protocol — uses Logfire's structured event system. |
| `OTELMetrics` | `(service_name: str = "agent", otlp_endpoint: str = "localhost:4317")` | OpenTelemetry metrics via OTLP gRPC. Creates real OTel counters, gauges, and histograms with a `PeriodicExportingMetricReader`. Same default OTLP gRPC port as tracing (`4317`); the collector routes metrics to Prometheus. |
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

### Failure telemetry in the trace stream

With `create_spans=False` (default), success traces contain PydanticAI native spans only. Failures are still surfaced via `record_failures=True` (the default):

- **PydanticAI-owned failure** (model/tool error inside the agent graph) → the canonical `invoke_agent` / `execute_tool` / `chat` spans get `status = ERROR` plus an `exception` event; the harness adds nothing.
- **Harness failure with a recording span active** → that span is marked `ERROR` and records the exception.
- **Harness failure with no active recording span** → a harness-owned span `{service}.{operation}:failed` is emitted with `status = ERROR`, `error.type`, `error.source` and the `exception` event. The `:failed` suffix is only a query convenience — the failure semantics come from the standard OTel status + exception event (`exception.type` / `exception.message` / `exception.stacktrace`).
- Successes never produce a harness span.

Set `record_failures=False` to opt out of all harness failure spans/enrichment.

Reference queries for these failures against `traces-generic.otel-default*` (Elasticsearch) are in **[`OBSERVABILITY.md`](OBSERVABILITY.md#43-trace-queries-traces-genericotel-default)** — e.g. `status.code: "STATUS_CODE_ERROR"`, `name: *:failed`, `error.type`/`error.source`.

Disable repeatedly-failing paths in a run: `create_spans=False, record_failures=False`.

### Code location and failure detail on log records

Every log record now carries the application callsite (skipping frames inside `agent_harness/`, the stdlib, and site-packages):

- **OpenTelemetry logs (ES)** — record attributes include `code.file.path`, `code.function`, `code.line.number`.
- **Console / file / Logfire records** — the same location is appended as structlog-style keys `pathname`, `func_name`, `lineno`.

Failure records — `{operation}_failed`, `error_handled`, and any `obs.error(msg, exception=e)` — additionally embed the exception (OTel `exception.*` shape) plus the **raise site**: `exception.type`, `exception.message`, `exception.stacktrace`, and `code.file.path` / `code.function` / `code.line.number` taken from the innermost traceback frame.

ES record shape and reference queries against `logs-generic.otel-default*` are in **[`OBSERVABILITY.md`](OBSERVABILITY.md#42-log-queries-logs-genericotel-default)** (`body.text`, `attributes.code.file.path`, `attributes.exception.stacktrace`, …).

### Visualizing telemetry (Elasticsearch, Jaeger, Prometheus, Grafana, Kibana)

Docs for running the observability stack and inspecting telemetry in **Elasticsearch** (log/trace queries), **Jaeger** (trace waterfall), **Prometheus** (PromQL), **Grafana** (single pane, Logs Drilldown), and the optional **Kibana** log-levels dashboard moved to **[`OBSERVABILITY.md`](OBSERVABILITY.md)**: stack startup, service/port reference, data-stream shapes, ES reference queries, Jaeger/Grafana/Prometheus usage, and Kibana provisioning.

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

Guardrails are callback-driven. Set a config to enable it; omit the config (or pass `None`) to disable. Config presence IS the toggle — no `enabled` booleans.

```python
# Content filter: transform output via callback
agent.with_content_filter(
    ContentFilterConfig()
    .on_filter(lambda text: text.replace("badword", "***"))
    .on_error(lambda ctx: f"Filter failed: {ctx.error_message}")
)

# PII detection: redact via callback
agent.with_pii_detection(
    PIIDetectionConfig()
    .on_redact(redact_pii_fn)
    .on_error(lambda ctx: f"PII redaction failed: {ctx.error_message}")
)

# Token limits: separate input/output/total caps
agent.with_token_limits(
    TokenLimitsConfig()
    .with_max_input_tokens(2000)
    .with_max_output_tokens(1000)
    .with_max_total_tokens(3000)
    .on_token_limit(lambda ctx: f"Token limit hit: {ctx.error_message}")
)

# Cost limits: dollar-based with per-token pricing
agent.with_cost_limits(
    CostLimitsConfig()
    .with_cost_per_input_token(0.000003)    # GPT-4o rate
    .with_cost_per_output_token(0.000015)
    .with_max_total_cost(0.01)
    .on_cost_limit(lambda ctx: f"Budget exceeded: {ctx.error_message}")
)

# Circuit breaker: block after N consecutive failures
agent.with_circuit_breaker(
    CircuitBreakerConfig()
    .with_threshold(5)
    .with_timeout(60)
    .on_error(lambda ctx: f"Circuit open: {ctx.error_message}")
)

# Turn limits: cap agent invocations per session
agent.with_turn_limits(
    TurnLimitsConfig()
    .with_max_turns(50)
    .on_turn_limit(lambda ctx: f"Session limit: {ctx.error_message}")
)
```

### Bulk guardrail setter

```python
agent.with_guardrails(
    content_filter=ContentFilterConfig().on_filter(my_filter),
    pii_detection=PIIDetectionConfig().on_redact(my_redactor),
    token_limits=TokenLimitsConfig().with_max_total_tokens(4096),
    cost_limits=CostLimitsConfig()
        .with_cost_per_input_token(0.000003)
        .with_cost_per_output_token(0.000015)
        .with_max_total_cost(0.01),
)
```

### Callback return contract

All guardrail callbacks follow the same pattern as `AgentRetryConfig`:

| Callback type | Signature | Returns |
|--------------|-----------|---------|
| Named callback (e.g. `on_filter`, `on_redact`, `on_token_limit`, `on_cost_limit`, `on_turn_limit`) | `(ErrorContext) -> Any` | Return a value → suppress. `None` would re-raise but the guardrail raises first anyway. |
| `on_error` (on every config) | `(ErrorContext) -> Any` | Return a value → suppress with that value as output. `None` → re-raise the exception. |

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

**`GuardConfig`** — combines retry configs and optional guardrail configs:

| Field | Type | Default | Description |
|---|---|---|---|
| `agent` | `AgentRetryConfig` | `AgentRetryConfig()` | Agent-level retry settings |
| `tool` | `ToolRetryConfig` | `ToolRetryConfig()` | Tool-level retry settings |
| `result_validator` | `ResultValidatorRetryConfig` | `ResultValidatorRetryConfig()` | Output validation retry settings |
| `content_filter` | `ContentFilterConfig \| None` | `None` | Content filtering (None = off, set = on) |
| `pii_detection` | `PIIDetectionConfig \| None` | `None` | PII detection (None = off, set = on) |
| `token_limits` | `TokenLimitsConfig \| None` | `None` | Token usage limits (None = off, set = on) |
| `cost_limits` | `CostLimitsConfig \| None` | `None` | Dollar cost limits (None = off, set = on) |
| `circuit_breaker` | `CircuitBreakerConfig \| None` | `None` | Circuit breaker (None = off, set = on) |
| `turn_limits` | `TurnLimitsConfig \| None` | `None` | Session turn cap (None = off, set = on) |

**`ContentFilterConfig`** — callback-driven content filtering:

| Field | Type | Default | Description |
|---|---|---|---|
| `on_filter` | `Callable[[str], str] \| None` | `None` | Transform the response text (e.g. profanity filter) |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback when the filter callback raises |

Fluent setters: `.on_filter(callback)`, `.on_error(callback)`.

**`PIIDetectionConfig`** — callback-driven PII redaction:

| Field | Type | Default | Description |
|---|---|---|---|
| `on_redact` | `Callable[[str], str] \| None` | `None` | Redact PII from the response text |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback when the redaction callback raises |

Fluent setters: `.on_redact(callback)`, `.on_error(callback)`.

**`TokenLimitsConfig`** — caps token usage per request:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_input_tokens` | `int \| None` | `None` | Cap on input token count |
| `max_output_tokens` | `int \| None` | `None` | Cap on output token count |
| `max_total_tokens` | `int \| None` | `None` | Cap on total token count |
| `on_token_limit` | `Callable[[ErrorContext], Any] \| None` | `None` | Callback when any limit is exceeded |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback for unexpected errors |

Fluent setters: `.with_max_input_tokens(n)`, `.with_max_output_tokens(n)`, `.with_max_total_tokens(n)`, `.on_token_limit(callback)`, `.on_error(callback)`.

**`CostLimitsConfig`** — dollar cost caps using per-token pricing:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_input_cost` | `float \| None` | `None` | Max dollar cost for input tokens |
| `max_output_cost` | `float \| None` | `None` | Max dollar cost for output tokens |
| `max_total_cost` | `float \| None` | `None` | Max total dollar cost |
| `cost_per_input_token` | `float \| None` | `None` | Pricing per input token (e.g. `0.000003` for GPT-4o) |
| `cost_per_output_token` | `float \| None` | `None` | Pricing per output token (e.g. `0.000015` for GPT-4o) |
| `on_cost_limit` | `Callable[[ErrorContext], Any] \| None` | `None` | Callback when cost exceeds a limit |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback for unexpected errors |

Fluent setters: `.with_max_input_cost(n)`, `.with_max_output_cost(n)`, `.with_max_total_cost(n)`, `.with_cost_per_input_token(n)`, `.with_cost_per_output_token(n)`, `.on_cost_limit(callback)`, `.on_error(callback)`.

**`CircuitBreakerConfig`** — failure-aware circuit breaker:

| Field | Type | Default | Description |
|---|---|---|---|
| `failure_threshold` | `int` | `5` | Consecutive failures before the circuit opens |
| `circuit_timeout` | `int` | `60` | Seconds before testing with a half-open request |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback when circuit is open |

Fluent setters: `.with_threshold(n)`, `.with_timeout(n)`, `.on_error(callback)`.

**`TurnLimitsConfig`** — session turn cap:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_turns` | `int \| None` | `None` | Maximum agent invocations per session |
| `on_turn_limit` | `Callable[[ErrorContext], Any] \| None` | `None` | Callback when turn limit is exceeded |
| `on_error` | `Callable[[ErrorContext], Any] \| None` | `None` | Fallback for unexpected errors |

Fluent setters: `.with_max_turns(n)`, `.on_turn_limit(callback)`, `.on_error(callback)`.

### GuardRunner flow

Inside `run()`, `GuardRunner.run_with_guards()`:
1. Checks circuit breaker — if open and timeout hasn't elapsed, blocks the request
2. Executes `agent.run()` inside `asyncio.wait_for(timeout)`
3. On timeout/error: backs off exponentially, retries up to `max_retries`, tracks consecutive failures for circuit breaker
4. After success: checks token limits → cost limits → content filter → PII redaction
5. Calls `on_retry` callback on each retry; `on_error` callback on exhaustion
6. After exhaustion: tries `fallback_model` if configured

---

## 10. Error Handling

### Per-source error callbacks

Errors are classified into 8 sources at the point of origin. Each source has a dedicated callback:

```python
from agent_harness.errorhandling import (
    ErrorHandlingConfig, ErrorContext,
)

config = (
    ErrorHandlingConfig()
    .on_llm_error(lambda ctx: f"LLM failed: {ctx.error_message}")     # network, auth, rate limit, model not found
    .on_tool_error(lambda ctx: f"Tool failed: {ctx.error_message}")    # tool function raises
    .on_validation_error(lambda ctx: None)                              # output validator ModelRetry exhausted (re-raise)
    .on_guardrail_error(lambda ctx: f"Blocked: {ctx.error_message}")   # circuit breaker, token/cost/turn limits
    .on_memory_error(lambda ctx: None)                                  # persistence failure (suppress, continue)
    .on_prompt_error(lambda ctx: None)                                  # template/render failure (re-raise)
    .on_evaluator_error(lambda ctx: None)                               # post-turn evaluator failure (suppress)
    .on_output_error(lambda ctx: f"Output error: {ctx.error_message}")  # usage parsing, extraction
    .on_error(lambda ctx: f"Unhandled [{ctx.source}]: {ctx.error_message}")  # catch-all
)

agent.with_error_handling(config)
```

### Callback return contract

All error callbacks share one signature:

```python
def handler(ctx: ErrorContext) -> Any | None:
    ...
```

| Return value | Effect |
|-------------|--------|
| `any_value` | Suppress the error — `any_value` becomes `AgentRunResult.output`. Returns `AgentRunResult(success=False, output=any_value)`. |
| `None` | Re-raise the exception — it propagates as normal. |

### Error source taxonomy

Eight sources, set explicitly at each origination point in `agent.run()`:

| Source | Origination point |
|--------|------------------|
| `llm` | `agent.run()` inside `asyncio.wait_for` — network, auth, rate limit, model not found, timeout |
| `tool` | Tool function execution inside pydantic_ai tool loop |
| `validation` | Output validator `ModelRetry` exhausted |
| `guardrail` | Circuit breaker open, token/cost/turn limits, content filter/PII callback exception |
| `memory` | `message_history.load()` or `provider.save_turn()` failures |
| `prompt` | `get_system_prompt()` Jinja2 render or MongoPrompts query failure |
| `evaluator` | `evaluator.evaluate()` raises (no longer silently swallowed) |
| `output` | Usage parsing, `TurnData` construction, `extract_clean_output()` failures |

### ErrorContext data

```python
@dataclass
class ErrorContext:
    error_type: str           # Python exception class name
    error_message: str        # Exception message
    source: str               # llm, tool, validation, guardrail, memory,
                              # prompt, evaluator, output
    session_id: str | None
    prompt: str | None
    stack_trace: str | None   # Full traceback for debugging
    attempt: int
    max_attempts: int
    will_retry: bool
```

### Pipeline error recovery example

See `error_handling/09_pipeline_error_recovery.py` for a complete three-agent pipeline where the middle agent deliberately fails via `FailingPromptProvider`, the error handler suppresses it, and the pipeline continues to the final agent. A shared `PipelineContext` tracks every stage's status and prints a full trace at the end.

---

## 11. Orchestration Patterns

Multi-agent orchestration patterns for composing pipelines. Each pattern is self-contained in `agent_harness_examples/orchestration/`.

### 11.1 Tool-Driven Delegation (`01_delegation.py`)

A coordinator agent delegates tasks to a specialist agent via a tool. A shared `PipelineContext` tracks all delegations across turns.

```
Coordinator Agent (tool: delegate_to_specialist)
    │
    │ tool call: "analyze Q3 revenue"
    ▼
Specialist Agent (finance)
    │
    ▼
SharedContext ←── delegation log updated
    │
    ▼
Coordinator returns final answer
```

```python
@dataclass
class SharedContext:
    delegation_log: list[dict] = field(default_factory=list)

# Specialist agent
specialist = ManagedAgent().with_model(model)

# Delegation tool — runs the specialist inside the tool
async def delegate_to_specialist(ctx: RunContext[SharedContext], task: str) -> str:
    sub_history = MessageHistory()
    result = await specialist.run(f"Task: {task}", sub_history, f"sub-{uuid4().hex[:8]}")
    ctx.deps.record(task, str(result.output))
    return f"[Specialist Report]\n{result.output}"

# Coordinator with the delegation tool
coordinator = (
    ManagedAgent(deps_type=SharedContext)
    .with_model(model)
    .with_tools(ToolRegistry().add(delegate_to_specialist))
)

ctx = SharedContext()
result = await coordinator.run(prompt, history, session_id, deps=ctx)
```

### 11.2 Sequential Pipeline (`02_sequential_pipeline.py`)

Three agents run in sequence — each agent's output feeds into the next agent as its prompt. No tools involved; orchestration is fully programmatic.

```
Researcher Agent ──(facts)──▶ Writer Agent ──(draft)──▶ Editor Agent
```

```python
# Build three agents
researcher = ManagedAgent().with_model(model)
writer = ManagedAgent().with_model(model)
editor = ManagedAgent().with_model(model)

# Run pipeline
r1 = await researcher.run(question, h1, session_id)
r2 = await writer.run(f"Write using: {r1.output}", h2, session_id)
r3 = await editor.run(f"Edit: {r2.output}", h3, session_id)
```

### 11.3 Classify and Route (`03_routing.py`)

A router agent classifies the user's request via a tool, then the program routes to the appropriate specialist agent.

```
User query ──▶ Router Agent (tool: classify_request)
                     │
               "billing" / "tech-support" / "general"
                     │
                     ▼
          BillingAgent / TechAgent / GeneralAgent
```

```python
# Router with classify tool
def classify_request(ctx: RunContext[SharedContext], text: str) -> str:
    """Classify a user request into a category."""
    if "bill" in text.lower(): return "billing"
    if "error" in text.lower(): return "tech-support"
    return "general"

router = (
    ManagedAgent(deps_type=SharedContext)
    .with_model(model)
    .with_tools(ToolRegistry().add(classify_request))
)

specialists = {"billing": billing_agent, "tech-support": tech_agent, "general": general_agent}

# Classify, then route
await router.run(query, h1, session_id, deps=ctx)
specialist = specialists[ctx.classification]
result = await specialist.run(query, h2, session_id)
```

### 11.4 Parallel Fan-Out / Fan-In (`04_parallel_fanout.py`)

A coordinator agent fans out a question to multiple specialist agents concurrently via `asyncio.gather`, then aggregates their perspectives.

```
Coordinator (tool: gather_perspectives)
    │
    ├──▶ Legal Analyst   ──┐
    ├──▶ Tech Analyst    ──┤── asyncio.gather ──▶ aggregate
    ├──▶ Business Analyst ──┘
```

```python
async def gather_perspectives(ctx: RunContext[SharedContext], question: str) -> str:
    async def ask_legal():
        r = await legal_agent.run(question, ...)
        return ("Legal", str(r.output))
    async def ask_tech():
        r = await tech_agent.run(question, ...)
        return ("Technical", str(r.output))
    async def ask_business():
        r = await business_agent.run(question, ...)
        return ("Business", str(r.output))

    results = await asyncio.gather(ask_legal(), ask_tech(), ask_business())
    return "\n\n".join(f"### {label}\n{out}" for label, out in results)
```

### 11.5 Pipeline Error Recovery (`error_handling/09_pipeline_error_recovery.py`)

A three-agent pipeline where the middle agent deliberately fails, the error handler suppresses it, and the pipeline continues. A shared `PipelineContext` tracks every stage and prints a full trace at the end.

```python
@dataclass
class PipelineContext:
    stages: list[dict] = field(default_factory=list)

    def post(self, name: str, success: bool, output: str, error: str = ""):
        self.stages.append({...})

    def display_trace(self):
        for s in self.stages:
            print(f"  {'✓' if s['success'] else '✗'} {s['name']}: {s['output'][:100]}")

# Agent 2 fails deterministically (prompt provider always raises)
agent2 = (
    ManagedAgent()
    .with_prompts(FailingPromptProvider())
    .with_error_handling(
        ErrorHandlingConfig().on_prompt_error(lambda ctx: f"[Recovered] {ctx.error_message}")
    )
)

# Pipeline continues despite Agent 2 failure
ctx.post("Research", r1.success, r1.output)
ctx.post("Analysis", r2.success, r2.output, r2.error_context.error_message if not r2.success else "")
ctx.post("Summary", r3.success, r3.output)
ctx.display_trace()
```

---

## 12. Evaluators

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

## 13. Structured Output

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

## 14. RabbitMQ Integration

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

## 15. Environment Variables

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
| `otel_endpoint` | `str` | `"localhost:4317"` | OTLP collector endpoint |
| `elasticsearch_endpoint` | `str \| None` | `None` | Elasticsearch endpoint for log shipping |
| `elasticsearch_index_prefix` | `str` | `"agent-logs"` | Prefix for ES daily log indices |
| `max_retries` | `int` | `3` | Default max retry attempts |
| `timeout` | `int` | `120` | Default agent timeout in seconds |
| `fallback_model` | `str \| None` | `None` | Fallback model when retries exhausted |
| `file_storage_mongodb_uri` | `str \| None` | `None` | MongoDB URI for GridFS file storage |
| `file_storage_database` | `str \| None` | `None` | MongoDB database for file storage |
| `file_storage_collection` | `str \| None` | `None` | MongoDB collection for file storage |

---

## 16. Running the Examples

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

# Orchestration patterns
uv run orchestration/01_delegation.py       # Tool-driven delegation
uv run orchestration/02_sequential_pipeline.py  # Sequential pipeline chain
uv run orchestration/03_routing.py           # Classify and route
uv run orchestration/04_parallel_fanout.py   # Parallel fan-out / fan-in

# Error handling — pipeline error recovery
uv run error_handling/09_pipeline_error_recovery.py

# Observability — OTel logs+traces+metrics → ES logs + Prometheus metrics + Jaeger traces
uv run observability/09_otel_oltp_logs_traces_metrics.py
```

**Prerequisites:**
- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (for Ollama models) or API keys for cloud providers
- MongoDB (optional, for persistent memory in examples 2/3)
- RabbitMQ (optional, for the document classification example)
- Elasticsearch + OTel Collector + Jaeger + Prometheus (+ Kibana/Grafana for the OTEL observability examples) — `docker compose -f docker-compose.yml up -d` from the repo root

---

## 17. Full Fluent API Reference

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
| `with_mcp_server` | `(url: str, **kwargs) -> ManagedAgent` | Add a single MCP Streamable HTTP server. `tool_prefix` strips a prefix from tool names. |
| `with_mcp_servers` | `(*urls: str, tool_prefix: str \| None = None) -> ManagedAgent` | Add multiple MCP servers. Calls `with_mcp_server` for each URL. |
| `with_evaluators` | `(*evaluators: Evaluator) -> ManagedAgent` | Append evaluators to the list that runs after each turn. |
| `with_error_handling` | `(config: ErrorHandlingConfig) -> ManagedAgent` | Replace the error handling config with per-source callbacks. |
| `with_agent_retries` | `(config: AgentRetryConfig) -> ManagedAgent` | Set agent-level retry behaviour (max retries, timeout, backoff, fallback). |
| `with_tool_retries` | `(config: ToolRetryConfig) -> ManagedAgent` | Set per-tool retry behaviour. |
| `with_result_validator_retries` | `(config: ResultValidatorRetryConfig) -> ManagedAgent` | Set structured output validation retry behaviour. |
| `with_content_filter` | `(config: ContentFilterConfig) -> ManagedAgent` | Set content filter config — omit or pass `None` to disable. |
| `with_pii_detection` | `(config: PIIDetectionConfig) -> ManagedAgent` | Set PII detection config — omit or pass `None` to disable. |
| `with_token_limits` | `(config: TokenLimitsConfig) -> ManagedAgent` | Set token usage limits (input/output/total caps). |
| `with_cost_limits` | `(config: CostLimitsConfig) -> ManagedAgent` | Set dollar-based cost limits with per-token pricing. |
| `with_circuit_breaker` | `(config: CircuitBreakerConfig) -> ManagedAgent` | Configure the circuit breaker (failure threshold + timeout). |
| `with_turn_limits` | `(config: TurnLimitsConfig) -> ManagedAgent` | Set a cap on agent invocations per session. |
| `with_guardrails` | `(content_filter: ContentFilterConfig, pii_detection: PIIDetectionConfig, token_limits: TokenLimitsConfig, cost_limits: CostLimitsConfig) -> ManagedAgent` | Set multiple guardrail configs in one call. |
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

## 18. Architecture & Data Flow

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
   │    ├─ Circuit breaker gateway check
   │    ├─ for attempt in range(max_retries):
   │    │    asyncio.wait_for(agent.run(), timeout)
   │    │    ✓ success → token limits → cost limits → content filter → PII redaction
   │    │               → return AgentRunResult(success=True)
   │    │    ✗ timeout/error → backoff, on_retry callback, track failure, retry
   │    ├─ exhaust retries + fallback_model → run fallback
   │    └─ guardrail/fallback fail + on_error → return AgentRunResult(success=False)
   │       or re-raise
  │
  ├─ extract_clean_output(result)  [if no structured output type]
  ├─ TurnData(messages, usage, duration, model, status)
  ├─ save_to providers → provider.save_turn(session_id, turn)
  ├─ observability → log token usage
  ├─ evaluators → for each evaluator: evaluate(prompt, result, context)
  │
   └─ [on exception] ErrorHandler.handle_error(exception, source, session_id, prompt)
        ├─ routes to source-specific callback (on_llm_error / on_tool_error / ...)
        ├─ fallback to on_error (catch-all)
        ├─ handler returns a value → return AgentRunResult(success=False, output=value)
        └─ handler returns None → re-raise
```
