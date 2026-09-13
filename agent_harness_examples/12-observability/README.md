# Observability

OpenTelemetry-only logging, tracing, and metrics for `ManagedAgent`. Every
signal is emitted via OTLP to the collector; the collector fans out to the
backends. No application code talks to a backend SDK directly.

```
ManagedAgent  (OTel only: OTELLogger + OTELTracer + OTELMetrics + pydantic-ai native spans)
   │  structlog call sites are bridged to OTel
   │  local console rendered by the OTel Console{Log,Span,Metric}Exporter
   ▼ OTLP :4317
otel-collector ──▶ Elasticsearch / Kibana   (logs + metrics + traces)
               ──▶ Langfuse                (traces)
               ──▶ OpenObserve             (logs + metrics + traces)
```

## Telemetry granularity

`HARNESS_TELEMETRY_LEVEL` controls how much detail is emitted across logs,
metrics, traces, and memory:

| Level | Logs | Metrics | Traces | Memory |
|---|---|---|---|---|
| `minimal` | `run_summary` + WARN/ERROR only | not exported | native spans, no prompt content | not enriched |
| `standard` (library default) | + `agent_turn` per model iteration | native + `gen_ai.client.operation.duration` | native PydanticAI spans; no prompt content | `cost`/`latency`/`turn_count` populated |
| `verbose` (dev default in `.env`) | + retry attempts, `retry_wait`, lifecycle `*_started` | + retry/attempt metrics | + prompt/completion content (native spans), TTFT | full |

`HARNESS_TELEMETRY_CONSOLE=true` renders records to the local console through the
OTel console exporters.

Set `HARNESS_TELEMETRY_ENABLED=false` to disable all telemetry exporters and
local telemetry output. This is useful when running examples without an OTel
Collector.

## Error drill-down

Start in either the Elasticsearch/Kibana or OpenObserve Errors view. Filter on
`error.handled:false`, open the failed record, and follow its `trace_id` to the
failed span. Use `error.id` to find every log record for the same logical
failure and `run.id` to inspect the complete agent execution. The collector
exports the same fields, event names, and correlation IDs to both systems.

Investigation order:

1. Read `error.type`, `error.message`, and `error.source`.
2. Check `error.handled`, `error.retryable`, and attempt fields.
3. Open the linked trace and select the failed span.
4. Query `error.id` for the complete failure timeline.
5. Use `code.file.path`, `code.function`, and `code.line.number` to locate the source.

## Event model

All records carry `component` (the emitting subsystem, also used as the OTel
instrumentation scope) and `trace_id`/`span_id` when emitted inside a span.

| Event | Emitted | Key fields |
|---|---|---|
| `agent_turn` | once per internal model iteration | `turn.index`/`phase`/`tool_names`/`tool_call_count`, `token_usage.*` (incl. `cache_read_tokens`, `cache_write_tokens`, `cache_hit_ratio`), `cost.total_usd`/`input_usd`/`output_usd` + `source`, `latency.model_seconds`/`tool_seconds`/`total_seconds` |
| `run_summary` | once per agent request (`agent.run`) | `run.turn_count`/`request_count`/`tool_call_count`/`status`, per-turn avg/max latency + cost, `token_usage.*`, `cost.*`, `latency.*` breakdown (`memory_load`, `prompt_fetch`, `agent_core`, `memory_save`, `evaluators`, `overhead`, `model`, `tool`) |
| `model_request` | once per model request, **inside PydanticAI's `chat` span** | `model`, `provider`, `finish_reason`, `performance.duration_seconds`, `token_usage.*` |
| `agent_run` | once per agent run, **inside PydanticAI's `invoke_agent` span** | `agent_name`, `run_id`, `model`, `status`, `performance.duration_seconds` |
| `tool_call` / `tool_result` / `tool_error` | per tool invocation | `tool.name`/`parameters`, `performance.duration_seconds`, `component=tools` |
| `evaluator_started` / `evaluator_completed` / `evaluator_failed` | per evaluator | `evaluator`, `performance.duration_seconds`, `component=evaluators` |
| `retry_attempt` / `retry_wait` | on retries | `attempt`, `max_attempts`, `wait_seconds`, `error_type`, `component=guards` (retries visible at `verbose`) |
| `scenario_*`, `pipeline_stage_completed`, errors | as applicable | `component=app` / `pipeline` |

Failure records add `error.type` / `error.message` / `error.stacktrace` /
`error.source` / `error.handled` plus the `exception.*` mirror and, at
WARNING/ERROR, the caller location (`code.file.path`, `code.function`,
`code.line.number`, `code.namespace`). The same canonical fields are written to
the parent and child spans and to the `agent_errors_total` metric labels. When an
error is raised inside pydantic-ai's own asyncio task (no caller frame on the
stack), the harness uses the call site captured at `agent.run()` entry.

`model_request` and `agent_run` exist so backends that correlate a span to logs
by `span_id` — e.g. OpenObserve's trace **View Logs** — resolve the `chat` and
`invoke_agent` spans. They are compact correlation markers emitted inside those
spans; every agent the harness constructs (main agent, `QualityCheck` judge, guard
fallback) carries the capability, which is ordered inside PydanticAI's
instrumentation so the records inherit the span context. The full model data stays
on the spans' `gen_ai.*` attributes, and tool spans resolve against the existing
`tool_call`/`tool_result` records.

## Files

| File | What it shows |
|---|---|
| `01_otel_tracing.py` | OTLP tracing basics |
| `02_otel_full_stack.py` | Full OTLP stack (logs + traces + metrics) |
| `03_otel_builder.py` | `ObservabilityBuilder` fluent configuration |
| `04_otel_agent_run.py` | Agent-run telemetry (turns, tokens, latency) |
| `fluent_app.py` | End-to-end `ManagedAgent` demo (tools, guardrails, evaluators) with the default stack |

## Setup

```bash
# From the repo root — starts Langfuse, Elasticsearch, Kibana, OpenObserve, collector
docker compose up -d

# Provision Kibana data views/dashboards (incl. metrics) and OpenObserve
./kibana/provision-dashboards.sh
./openobserve/provision.sh

cd agent_harness_examples
uv sync
uv run python 12-observability/fluent_app.py
```

## Inspecting results

Shared UI login for Langfuse and OpenObserve: **`admin@example.com` / `Admin1234!`**
(Kibana has no login).

- **Langfuse** — <http://localhost:3000>: per-trace span waterfall, per-generation
  tokens/cost/latency, prompt content.
- **OpenObserve** — <http://localhost:5080>: unified logs, metrics, and traces
  (`agent_harness` streams), with SQL/PromQL and provisioned dashboards
  (**Agent Harness — Agent Runs / Token Usage / Cost / Latency / Errors**, each
  with a `Model` variable). `./openobserve/provision.sh` imports/refreshes them
  by title, so re-running it replaces rather than duplicates.
- **Kibana** — <http://localhost:5601>: `logs-generic.otel-default*`,
  `metrics-generic.otel-default*`; `trace_id` renders as a **View in Langfuse** link.
- **Elasticsearch**:
  ```bash
  curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=event_name:run_summary'
  curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=event_name:agent_turn'
  curl -s 'http://localhost:9200/traces-generic.otel-default*/_search?q=attributes.error.id:*'
  ```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HARNESS_TELEMETRY_ENABLED` | `true` | Enable OTLP telemetry and local telemetry output |
| `HARNESS_TELEMETRY_LEVEL` | `standard` | `minimal` / `standard` / `verbose` |
| `HARNESS_TELEMETRY_CONSOLE` | `true` | Render OTel records to the local console |
| `OTEL_COLLECTOR_ENDPOINT` | `localhost:4317` | OTLP gRPC collector endpoint |
| `OBSERVABILITY_SERVICE_NAME` | `agent-harness` | OTLP resource `service.name` |
| `LANGFUSE_UI_URL` / `LANGFUSE_PROJECT_ID` | `http://localhost:3000` / `local-project` | Log → Langfuse deep links |
| `OPENOBSERVE_UI_URL` | `http://localhost:5080` | OpenObserve UI |
| `OPENOBSERVE_ENDPOINT` | `http://openobserve:5080/api/default` | Collector → OpenObserve OTLP/HTTP endpoint |
| `OPENOBSERVE_AUTH` | — | `base64(email:password)` for the collector |
| `ZO_ROOT_USER_EMAIL` / `ZO_ROOT_USER_PASSWORD` | `admin@example.com` / `Admin1234!` | OpenObserve root credentials (shared UI login) |
| `ELASTICSEARCH_ENDPOINT` | `http://localhost:9200` | Elasticsearch endpoint (where applicable) |
