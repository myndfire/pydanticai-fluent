# pydanticai-fluent

A fluent, builder-style API for configuring [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents with cross-cutting concerns.

## Design

`ManagedAgent` is the central orchestrator. It wraps a pydantic-ai `Agent` and layers on top:

- **Memory** — short-term and long-term conversation persistence via a pluggable `MemoryProvider` protocol (in-memory, MongoDB, Redis, Elasticsearch).
- **Observability** — unified facade combining logging (structlog, file, Elasticsearch, Logfire), tracing (OTEL, Logfire, Jaeger), and metrics (Prometheus, StatsD, OTEL, InMemory).
- **Guards** — retry logic with exponential backoff, fallback models, callbacks; circuit breaker; guardrails for content filtering, PII detection, and cost limits.
- **Error handling** — custom error handlers with source classification (LLM, tool, memory, unknown).
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
├── agentic_rag/                # RAG agent example
└── USAGE.md                    # Full usage guide
```

## Installation

```bash
cd agent_harness && uv sync
cd ../agent_harness_examples && uv sync
```

---

> For detailed API docs, configuration, and examples, see [`USAGE.md`](USAGE.md).
