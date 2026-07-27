# pydanticai-fluent

A fluent, builder-style API for configuring [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents with cross-cutting concerns.

## Design

`ManagedAgent` is the central orchestrator. It wraps a pydantic-ai `Agent` and layers on top:

- **Memory** — short-term and long-term conversation persistence via a pluggable `MemoryProvider` protocol (in-memory, MongoDB, Redis, Elasticsearch).
- **Observability** — unified facade combining logging (structlog, file, Elasticsearch, Logfire), tracing (OTEL, Logfire, Jaeger), and metrics (Prometheus, StatsD, OTEL, InMemory).
- **Guards** — retry logic with exponential backoff, fallback models, callbacks; circuit breaker; guardrails for content filtering, PII detection, and cost limits.
- **Error handling** — custom error handlers with source classification (LLM, tool, memory, unknown); pipeline error recovery.
- **Orchestration** — multi-agent patterns: tool-driven delegation, sequential pipelines, classify-and-route, parallel fan-out/fan-in.
- **Tools** — plain-function and context-aware tool registration with automatic `RunContext` detection.
- **Prompts** — static strings or Jinja2 templates from MongoDB.
- **Evaluators** — post-turn evaluation hooks with built-in LLM-as-judge (`QualityCheck`) and OpenAI moderation (`SafetyCheck`).
- **Structured output** — constrain responses to Pydantic models.
- **Messaging** — RabbitMQ integration for message-driven agent workflows.

Everything is configured via a fluent chain:

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

### Model Configuration

`ManagedAgent` supports **20 LLM providers** (Ollama, OpenAI, Anthropic, Groq, Cohere, Mistral, Bedrock, HuggingFace, OpenRouter, xAI, DeepSeek, Cerebras, Fireworks, Together, Azure, Vercel, MoonshotAI, GitHub, Heroku) via `ModelConfig`. You can configure models programmatically or entirely through environment variables.

```python
from agent_harness.model_config import ModelConfig
from pydantic_ai.settings import ModelSettings

# Programmatic
agent = (
    ManagedAgent()
    .with_model(ModelConfig(provider="ollama", model_name="gemma4:4b-mlx"))
    .with_model_settings(
        ModelSettings(
            thinking=False,        # disable hidden reasoning — 2-5x faster
            max_tokens=512,        # cap output length
            temperature=0.1,       # low = faster, more deterministic
        )
    )
)

# Or via .env — no code changes
#   MODEL_NAME=ollama:gemma4:4b-mlx
#   OPENAI_API_KEY=sk-...
```

See [`USAGE.md` Section 4](USAGE.md#4-model-configuration) for the complete reference on all `ModelSettings` fields (16+ options including `thinking`, `max_tokens`, `temperature`, `top_p`, `timeout`, `tool_choice`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stop_sequences`, `extra_headers`, `service_tier`, `extra_body`), provider-specific environment variables, JSON-string `.env` patterns, and model selection strategy.

## Using as a Library

Install `pydanticai-fluent` from a tagged GitHub release into your own project.

### With pip

```bash
pip install "git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0"
```

### With uv

```bash
uv add "git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0"
```

### In your `pyproject.toml`

```toml
[project]
dependencies = [
    "pydanticai-fluent @ git+https://github.com/myndfire/pydanticai-fluent.git@v0.1.0",
]
```

### Quick Start

```python
import asyncio
from agent_harness import ManagedAgent
from agent_harness.model_config import ModelConfig
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.prompts import StaticPrompts

async def main():
    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_short_term_memory(InMemoryProvider())
        .with_prompts(StaticPrompts("You are a helpful assistant"))
    )
    
    history = await MessageHistory().load("session-1", agent._short_term_memory)
    result = await agent.run("Hello!", history, "session-1")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

See [`USAGE.md`](USAGE.md) for the complete API reference.

## Project structure

```
pydanticai-fluent/
├── agent_harness/              # Core package (src-layout)
│   └── src/agent_harness/
│       ├── agent.py            # ManagedAgent
│       ├── model_config.py     # ModelConfig + build_model()
│       ├── memory.py           # MemoryProvider, MessageHistory, InMemory/Mongo/Redis/ES
│       ├── tools.py            # ToolRegistry
│       ├── prompts.py          # StaticPrompts, MongoPrompts
│       ├── observability.py    # Observability facade + builder
│       ├── logging.py          # ConsoleLogger, FileLogger, ElasticsearchLogger, etc.
│       ├── tracing.py          # OTelTracer, LogfireTracer, JaegerTracer, etc.
│       ├── metrics.py          # PrometheusMetrics, StatsdMetrics, OTLPMetrics, etc.
│       ├── guards.py           # GuardConfig, retry configs, guardrail configs
│       ├── errorhandling.py    # ErrorHandlingConfig, ErrorHandler
│       ├── evaluators.py       # Evaluator, QualityCheck, SafetyCheck
│       ├── rabbitmq.py         # MessagingService
│       └── file_storage.py     # MongoDB GridFS FileStorage
├── agent_harness_examples/     # Runnable examples
│   ├── orchestration/          # Multi-agent orchestration examples
│   ├── error_handling/         # Error handling examples
│   ├── tools/ memory/ guards/  # etc.
│   └── ...
├── agentic_rag/                # RAG agent example
└── USAGE.md                    # Full usage guide
```

## Installation

```bash
cd agent_harness && uv sync
cd ../agent_harness_examples && uv sync
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

> For detailed API docs, configuration, and examples, see [`USAGE.md`](USAGE.md).
