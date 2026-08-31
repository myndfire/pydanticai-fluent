# Observability

Log, trace, and measure agent behavior — from simple console logging to full OpenTelemetry pipelines with Elasticsearch, Jaeger, Prometheus, and Grafana.

## Overview

The `observability` module provides a composable `Observability` facade that fans out events to multiple logging, tracing, and metrics backends. `ObservabilityBuilder` offers a fluent API to compose backends. Each `agent.run()` automatically logs start/complete events, increments counters, records duration histograms, and creates trace spans.

```
Observability architecture:
────────────────────────────────────────────────────────────────────

  agent.run(prompt, history, session_id)
       │
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Observability facade                                        │
  │                                                               │
  │  .info("agent_run_started", ...)                             │
  │  .metrics.counter(AGENT_RUNS, ...)                           │
  │  .observe("agent_run", ...) → trace span + duration          │
  └──────────────────────────────────────────────────────────────┘
       │
       ├──▶ ConsoleLogger       → stderr (structlog)
       ├──▶ FileLogger          → daily/size-rotated files
       ├──▶ ElasticsearchLogger → ES daily indices
       ├──▶ OTELLogger          → OTLP gRPC → Collector
       │
       ├──▶ InMemoryTracer      → spans list (local inspection)
       ├──▶ OTELTracer          → OTLP gRPC → Jaeger
       │
       ├──▶ InMemoryMetrics     → counters/histograms/gauges (dicts)
       ├──▶ PrometheusMetrics   → push gateway
       └──▶ OTELMetrics         → OTLP gRPC → Prometheus
```

| Example | File | Backends | External services |
|---------|------|----------|-------------------|
| Logging basics | `01_logging.py` | ConsoleLogger, FileLogger | None |
| Tracing & metrics | `02_tracing_metrics.py` | InMemoryTracer, InMemoryMetrics | None |
| Builder pattern | `03_builder_logs_metrics.py` | Console, File, InMemory metrics | Ollama |
| Composite fan-out | `04_composite_logs.py` | CompositeLogger, multi-backend | None |
| Elasticsearch logs | `05_elasticsearch_logging.py` | ElasticsearchLogger | Elasticsearch, Ollama |
| OTEL + Jaeger traces | `06_otel_jaeger_logs_traces_metrics.py` | OTELTracer, InMemory | Jaeger, Ollama |
| Prometheus metrics | `07_prometheus_logs_metrics.py` | PrometheusMetrics, InMemory | Pushgateway, Ollama |
| Live agent full stack | `08_live_agent_logs_metrics.py` | Console, File, InMemory | Ollama |
| All-in-one OTLP | `09_otel_oltp_logs_traces_metrics.py` | OTELLogger, OTELTracer, OTELMetrics | ES, OTel Collector, Jaeger, Prometheus, Grafana, Ollama |

## Files

### 01_logging.py

`ConsoleLogger` (structlog to stderr) and `FileLogger` (daily/size rotation). All log methods accept `**kwargs` as structured key-value pairs.

```
ConsoleLogger()
    │
    ▼
console.info("agent_started", model="gpt-4o", session_id="abc123")
    │
    ▼
stderr: event=agent_started model=gpt-4o session_id=abc123

FileLogger(log_file="agent.log", rotation="daily", retention=7)
    │
    ▼
file_log.info("tool_invoked", tool="search", query="quantum")
    │
    ▼
agent.log: {"event": "tool_invoked", "tool": "search", "ts": "..."}
(rolls daily, keeps 7 days)
```

Key components:
- `ConsoleLogger()` — structlog-based stderr output
- `FileLogger(log_file, rotation, retention)` — daily or size-based rotation
- `Observability(loggers=[...])` — multiple loggers in one facade
- Log levels: `debug`, `info`, `warning`, `error`

### 02_tracing_metrics.py

`InMemoryTracer` records spans to a list. `InMemoryMetrics` stores counters, histograms, and gauges in dicts. Both support `reset()`.

```
InMemoryTracer:
    │
    ▼
async with tracer.span("agent_run", model="gpt-4o") as span:
    span["extra"] = "context"
    async with tracer.span("tool_call", tool="search"):
        pass
    │
    ▼
tracer.get_spans() → [
    {"name": "agent_run", "attributes": {"model": "gpt-4o", ...}},
    {"name": "tool_call", "attributes": {"tool": "search", ...}},
]

InMemoryMetrics:
    │
    ▼
metrics.counter(MetricNames.AGENT_RUNS, model="gpt-4o")  → +1
metrics.histogram(MetricNames.AGENT_DURATION, 1.2)        → [1.2]
metrics.gauge(MetricNames.ACTIVE_SESSIONS, 5)              → 5
    │
    ▼
metrics.get_metrics() → {
    "counters": {"agent_runs_total{model=gpt-4o}": 2},
    "histograms": {"agent_duration_seconds{...}": [1.2, 2.5]},
    "gauges": {"active_sessions": 5},
}
```

Key components:
- `InMemoryTracer()` — span recording with `get_spans()` / `reset()`
- `InMemoryMetrics()` — counters, histograms, gauges with `get_metrics()` / `reset()`
- `MetricNames` constants: `AGENT_RUNS`, `AGENT_DURATION`, `AGENT_ERRORS`, `ACTIVE_SESSIONS`, `MEMORY_SIZE`

### 03_builder_logs_metrics.py

`ObservabilityBuilder` fluent API to compose logging + tracing + metrics. Attach to agent via `.with_observability()`.

```
obs = (
    ObservabilityBuilder(service_name="demo-agent")
    .with_console_logging()
    .with_file_logging(log_file="agent.log")
    .with_in_memory_metrics()
    .build()
)
    │
    ▼
agent = ManagedAgent()
    .with_model(ModelConfig(...))
    .with_observability(obs)
    │
    ▼
agent.run("What is 2+2?")
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Auto-collected by observe():                                 │
│    Logs: agent_run_started, agent_run_completed               │
│    Counter: agent_runs_total{model=..., session_id=...} += 1 │
│    Histogram: agent_duration_seconds{...} = 1.23              │
│    Trace span: "agent_run" with attributes                    │
└──────────────────────────────────────────────────────────────┘
```

Key components:
- `ObservabilityBuilder(service_name)` — fluent builder
- `.with_console_logging()` / `.with_file_logging()` / `.with_in_memory_metrics()`
- `.build()` → `Observability` instance
- `agent.with_observability(obs)` — attach to agent
- `obs.observe("custom_op", ...)` — manual context manager for custom spans

### 04_composite_logs.py

`CompositeLogger` fans out log messages to multiple loggers. `Observability` with multiple loggers, tracers, and metrics backends.

```
CompositeLogger(console, file_log)
    │
    ▼
composite.info("test", source="example")
    │
    ├──▶ ConsoleLogger → stderr
    └──▶ FileLogger → file

Observability(
    loggers=[ConsoleLogger(), FileLogger(...)],
    tracers=[InMemoryTracer(), NoOpTracer()],
    metrics_list=[InMemoryMetrics()],
)
    │
    ▼
obs.observe("op") → ALL backends receive events
obs.logger  → first logger (convenience)
obs.tracer  → first tracer (convenience)
obs.metrics → first metrics (convenience)
```

Key components:
- `CompositeLogger(*loggers)` — fan-out to multiple loggers
- `Observability(loggers, tracers, metrics_list)` — multi-backend facade
- `.logger`, `.tracer`, `.metrics` — convenience properties (first backend)
- `observe()` — fans out to ALL backends simultaneously

### 05_elasticsearch_logging.py

`ElasticsearchLogger` writes structured logs to Elasticsearch with daily index patterns.

```
ElasticsearchLogger(endpoint, index_prefix="agent-logs", service_name)
    │
    ▼
es_logger.info("agent_run_started", model="gpt-4o", session_id="es-1")
    │
    ▼
POST /agent-logs-2025.01.15/_doc
{
    "timestamp": "2025-01-15T10:30:00Z",
    "service_name": "es-logging-demo",
    "level": "info",
    "event": "agent_run_started",
    "model": "gpt-4o",
    "session_id": "es-1"
}
    │
    ▼
Daily index: agent-logs-YYYY.MM.DD
Query: GET /agent-logs-*/_search?q=service_name:es-agent-demo
```

Key components:
- `ElasticsearchLogger(endpoint, index_prefix, service_name)` — ES structured logging
- Daily index pattern: `agent-logs-YYYY.MM.DD`
- Lazy connection with graceful fallback
- `close()` — flush pending tasks and clean up
- Prerequisite: `docker compose up -d elasticsearch`

### 06_otel_jaeger_logs_traces_metrics.py

`OTELTracer` exports traces via OTLP gRPC to Jaeger. Chained with `InMemoryTracer` for local inspection.

```
OTELTracer(service_name, otlp_endpoint="localhost:4317", sample_rate=1.0)
    │
    ▼
obs.observe("manual_span", op="test")
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Trace span exported to Jaeger via OTLP gRPC:                │
│    name: "manual_span"                                       │
│    trace_id: "abc123..."                                     │
│    span_id: "def456..."                                      │
│    attributes: {"op": "test", "value": 42}                   │
│    events: ["checkpoint_reached"]                             │
│                                                               │
│  Also captured by InMemoryTracer for local inspection:       │
│    spans = mem_tracer.get_spans()                             │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Jaeger UI: http://localhost:16686
  → Service: otel-jaeger-demo
```

Key components:
- `OTELTracer(service_name, otlp_endpoint, sample_rate, create_spans)` — OTLP gRPC export
- Chained tracers: `OTELTracer` + `InMemoryTracer` for local inspection
- `obs.add_span_event("checkpoint")` — add events to current span
- `obs.set_span_attribute("key", "value")` — add attributes
- Prerequisite: `docker compose up -d otel-collector` (forwards traces to Jaeger)

### 07_prometheus_logs_metrics.py

`PrometheusMetrics` with push gateway support. Counter, gauge, histogram, and summary metrics with labels.

```
PrometheusMetrics(namespace="agent", push_gateway="http://localhost:9091")
    │
    ▼
prom.counter(MetricNames.AGENT_RUNS, model="gpt-4o")
prom.gauge(MetricNames.ACTIVE_SESSIONS, 3)
prom.histogram(MetricNames.AGENT_DURATION, 1.2, status="success")
    │
    ▼
prom.push_to_gateway(job_name="agent")
    │
    ▼
POST http://localhost:9091/metrics/job/agent
  agent_agent_runs_total{model="gpt-4o"} 2
  agent_active_sessions 3
  agent_agent_duration_seconds{status="success"} 1.2
```

Key components:
- `PrometheusMetrics(namespace, push_gateway)` — Prometheus metrics with push gateway
- `.push_to_gateway(job_name)` — push metrics to Prometheus pushgateway
- Label-based metrics: model, session_id, status, error_type
- Agent run auto-collection via `observe()`
- Prerequisite: `docker compose up -d pushgateway`

### 08_live_agent_logs_metrics.py

Full observability stack with in-memory backends — no external services required. Shows multi-turn conversation with per-turn metrics.

```
ObservabilityBuilder(service_name="live-agent-demo")
    .with_console_logging()
    .with_file_logging(log_file="live_agent.log")
    .with_in_memory_metrics()
    .build()
    │
    ▼
agent = ManagedAgent()
    .with_model(ModelConfig(...))
    .with_observability(obs)
    │
    ▼
agent.run("My name is Carol...", save_to=[memory])
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Auto-collected per turn:                                     │
│    Logs: agent_run_started → agent_run_completed              │
│    Counter: agent_runs_total += 1                             │
│    Histogram: agent_duration_seconds = 0.85                   │
│    Token usage: input_tokens, output_tokens, total_tokens     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
After 3 turns:
  counters: {"agent_runs_total": 3}
  histograms: {"agent_duration_seconds": [0.85, 1.2, 0.9]}
  log file: live_agent.log (structured entries)
```

Key components:
- No external services — all in-memory/console/file
- Multi-turn conversation with per-turn observability
- Automatic metric collection: `agent_runs_total`, `agent_duration_seconds`
- Token usage logging: `input_tokens`, `output_tokens`, `total_tokens`
- `obs.observe("custom_workflow", ...)` — manual context manager

### 09_otel_oltp_logs_traces_metrics.py

All-in-one OTLP pipeline: logs → Elasticsearch, metrics → Prometheus, traces → Jaeger. Full agent instrumentation with tool calls and failure telemetry.

```
Architecture:
────────────────────────────────────────────────────────────────────

  agent_harness          otel-collector           backends
  ┌──────────────┐      ┌──────────────┐        ┌──────────────┐
  │ OTELLogger   │─OTLP─▶│ otlp receiver │──HTTP─▶│ Elasticsearch│
  │ OTELTracer   │─gRPC─▶│  :4317 (gRPC) │──gRPC─▶│ Jaeger       │
  │ OTELMetrics  │─OTLP─▶│  :4318 (HTTP) │──HTTP─▶│ Prometheus   │
  └──────────────┘      └──────────────┘        └──────────────┘
                                                          │
                                                     Grafana
                                                  (single pane)
```

Agent with tools:
```
agent = ManagedAgent()
    .with_model(ModelConfig(...))
    .with_tools(ToolRegistry().add_many(get_weather, calculator))
    .with_observability(obs)  # OTELLogger + OTELTracer + OTELMetrics
    │
    ▼
agent.run("Average temperature of Tokyo, London, NY?")
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Trace spans (PydanticAI native):                             │
│    invoke_agent                                               │
│      ├── execute_tool get_weather("tokyo")                    │
│      ├── execute_tool get_weather("london")                   │
│      ├── execute_tool get_weather("new york")                 │
│      ├── execute_tool calculator("(22+15+18)/3")             │
│      └── chat gpt-oss:20b                                    │
│                                                               │
│  Logs (→ ES via OTLP):                                        │
│    tool_call, tool_result, agent_run_started, token_usage     │
│                                                               │
│  Metrics (→ Prometheus via OTLP):                             │
│    agent_runs_total, agent_duration_seconds                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Failure telemetry:
  - tool raises ValueError → ERROR span + exception event
  - harness raises RuntimeError → <service>.guardrail_eval:failed span
```

Key components:
- `OTELLogger(service_name, otlp_endpoint)` — logs via OTLP gRPC
- `OTELTracer(service_name, otlp_endpoint, create_spans=False)` — native PydanticAI spans only
- `OTELMetrics(service_name, otlp_endpoint)` — metrics via OTLP gRPC
- `ToolDeps` — dependency container for context-aware tools with observability
- Failure telemetry: `DEMO_FAILURES=True` exercises error paths
- Log-trace correlation: log records carry `trace_id`/`span_id`
- Prerequisite: `docker compose up -d elasticsearch otel-collector jaeger prometheus grafana`

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally (for files 03, 05–09)
- Elasticsearch (for files 05, 09)
- Jaeger (for files 06, 09)
- Prometheus pushgateway (for file 07)
- Prometheus (for file 09)
- Grafana (for file 09)
- OTel Collector (for file 09)

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model (first time only)
ollama pull gpt-oss:20b

# 3. (Optional) Start observability stack
docker compose -f docker-compose.yml up -d elasticsearch jaeger pushgateway grafana

# 4. (Optional) Start full OTLP stack (for 09)
docker compose -f docker-compose.yml up -d elasticsearch otel-collector jaeger prometheus grafana

# 5. Install dependencies
cd agent_harness_examples
uv sync

# 6. (Optional) Copy and edit .env
cp .env.example .env
```

## Configuration

All variables are optional and read from `.env` via `python-dotenv`.

| Variable | File(s) | Default | Description |
|----------|---------|---------|-------------|
| `OBSERVABILITY_MODEL_NAME` | 03, 05–09 | `gpt-oss:20b` | LLM model name |
| `OBSERVABILITY_MAX_TOKENS` | 03, 05–09 | `512` | Max LLM output tokens |
| `ELASTICSEARCH_ENDPOINT` | 05, 09 | `http://localhost:9200` | Elasticsearch endpoint |
| `JAEGER_OTLP_ENDPOINT` | — | `localhost:14317` | **Deprecated** — Jaeger host OTLP port. All OTLP now flows through the OTel Collector; no example reads this. |
| `PROMETHEUS_PUSH_GATEWAY` | 07 | `http://localhost:9091` | Prometheus push gateway |
| `OTEL_COLLECTOR_ENDPOINT` | 06, 09 | `localhost:4317` | OTel Collector OTLP gRPC (logs, metrics, traces) |
| `OBSERVABILITY_SERVICE_NAME` | 09 | `all-in-one-observability-demo` | OTLP service name |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |

## Running

Each file is an independent entry point:

```bash
# Logging basics — ConsoleLogger + FileLogger
uv run python 12-observability/01_logging.py

# Tracing & metrics — InMemory backends
uv run python 12-observability/02_tracing_metrics.py

# Builder pattern — ObservabilityBuilder + agent integration
uv run python 12-observability/03_builder_logs_metrics.py

# Composite fan-out — multi-destination logging
uv run python 12-observability/04_composite_logs.py

# Elasticsearch — structured logging to ES
uv run python 12-observability/05_elasticsearch_logging.py

# OTEL + Jaeger — distributed tracing
uv run python 12-observability/06_otel_jaeger_logs_traces_metrics.py

# Prometheus — metrics with push gateway
uv run python 12-observability/07_prometheus_logs_metrics.py

# Live agent — full observability (no external services)
uv run python 12-observability/08_live_agent_logs_metrics.py

# All-in-one OTLP — ES + Jaeger + Prometheus + Grafana
uv run python 12-observability/09_otel_oltp_logs_traces_metrics.py
```

## Expected Output

**01_logging.py:** Console and file log output with structured key-value pairs. Log files: `agent_example.log`, `agent_size.log`, `combined.log`.

**02_tracing_metrics.py:** Span recording, counter/histogram/gauge inspection, MetricNames reference.

**03_builder_logs_metrics.py:** Builder composition, agent run with auto-collected metrics (AGENT_RUNS, AGENT_DURATION).

**04_composite_logs.py:** CompositeLogger fan-out, multi-backend observe(), convenience properties.

**05_elasticsearch_logging.py:** Structured logs in ES daily indices. Query: `GET /agent-logs-*/_search`.

**06_otel_jaeger_logs_traces_metrics.py:** Traces in Jaeger UI at `http://localhost:16686`.

**07_prometheus_logs_metrics.py:** Metrics pushed to pushgateway at `http://localhost:9091/metrics`.

**08_live_agent_logs_metrics.py:** 3-turn conversation with per-turn metrics, log file, token usage.

**09_otel_oltp_logs_traces_metrics.py:** Full pipeline: logs in ES, traces in Jaeger, metrics in Prometheus, all visible in Grafana.

## How It Works

1. **01_logging.py** — `ConsoleLogger` wraps structlog for stderr output. `FileLogger` adds daily/size rotation. Both accept `**kwargs` as structured context.

2. **02_tracing_metrics.py** — `InMemoryTracer` records spans as dicts. `InMemoryMetrics` stores counters (increment), histograms (append), and gauges (set) in dicts.

3. **03_builder_logs_metrics.py** — `ObservabilityBuilder` composes backends via `.with_*()` methods. `.build()` returns an `Observability` instance. `agent.with_observability(obs)` enables auto-instrumentation.

4. **04_composite_logs.py** — `CompositeLogger` wraps multiple loggers and fans out each call. `Observability` with multiple backends receives events from `observe()` on ALL backends.

5. **05_elasticsearch_logging.py** — `ElasticsearchLogger` writes structured documents to daily ES indices. Connection check with graceful fallback. `close()` flushes pending writes.

6. **06_otel_jaeger_logs_traces_metrics.py** — `OTELTracer` exports spans via OTLP gRPC to Jaeger. Chained with `InMemoryTracer` for local inspection. `add_span_event()` / `set_span_attribute()` enrich spans.

7. **07_prometheus_logs_metrics.py** — `PrometheusMetrics` records metrics with labels. `push_to_gateway()` pushes to Prometheus pushgateway. Auto-collection via `observe()`.

8. **08_live_agent_logs_metrics.py** — Full stack with in-memory backends. `agent.run()` automatically logs events, increments counters, records duration, and captures token usage.

9. **09_otel_oltp_logs_traces_metrics.py** — All-in-one: `OTELLogger` + `OTELTracer` + `OTEMetrics` export via OTLP gRPC to an OTel Collector, which routes to ES (logs), Jaeger (traces), and Prometheus (metrics). Tool calls log to ES. Failure telemetry records ERROR spans.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **Elasticsearch not reachable** — `docker compose up -d elasticsearch`.
- **Jaeger not reachable** — `docker compose up -d jaeger`. View at `http://localhost:16686`.
- **Pushgateway not reachable** — `docker compose up -d pushgateway`. View at `http://localhost:9091`.
- **OTel Collector not reachable** — `docker compose up -d otel-collector`.
- **Grafana dashboards missing** — Check `OBSERVABILITY.md` at repo root for provisioning instructions.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
