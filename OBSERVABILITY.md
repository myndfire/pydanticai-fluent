# OBSERVABILITY.md — Langfuse, Elasticsearch, Kibana & OpenObserve

How to run the observability stack and inspect agent telemetry in **Elasticsearch + Kibana** and **OpenObserve**. Both receive the same structured logs, metrics, and traces from the collector; **Langfuse** remains an additional trace-oriented UI.

> The Jaeger / Prometheus / Grafana sections at the end describe the **legacy**
> stack in `docker-compose.yml.old`; the default `docker-compose.yml` ships
> Langfuse (traces) + Elasticsearch/Kibana (logs) and routes metrics to the
> collector's `debug` exporter.

## 1. Overview & architecture

The OTEL backends (`OTELLogger`, `OTELTracer`, `OTELMetrics`) export over OTLP gRPC to the OpenTelemetry Collector, which routes each signal to a backend:

> **OTLP is the transport for all telemetry.** Every signal is sent to the OTel Collector at `localhost:4317` (gRPC) / `localhost:4318` (HTTP), which fans out the same data to Elasticsearch and OpenObserve.

```
agent_harness  --OTLP gRPC:4317-->  otel-collector  --otlphttp-->  Langfuse (traces)
    (logs + metrics + traces)        (otlp receiver:   --otlphttp-->  Elasticsearch (logs)
                                       grpc :4317,      \--debug----->  collector stdout (metrics)
                                       http :4318)
```

To make a run's logs link back to its trace, `fluent_app.py` enables `OTELTracer(create_spans=True)`, so the harness owns an `agent_run` span that stays active for the whole run (tools, guardrails, evaluators, error handling). Every in-run log record then carries `trace_id`, which Kibana renders as a "View in Langfuse" link (see §4.4 and §8).

## 2. Start the stack

From the repo root:

```bash
docker compose up -d
```

This starts Langfuse (plus its Postgres/ClickHouse/Redis/MinIO dependencies), the OTel Collector, Elasticsearch, and Kibana.

## 3. Service & port reference

| Service | Port | Role |
|---|---|---|
| `langfuse-web` | `3000` | Trace backend + UI (shared UI login, see below) |
| `elasticsearch` | `9200` | Logs, metrics, and traces backend — native OTLP/HTTP intake (`/_otlp`) |
| `kibana` | `5601` | Log/metrics browser; renders `trace_id` as a "View in Langfuse" link (no login) |
| `openobserve` | `5080`, `5081` | Unified logs + metrics + traces UI (OTLP/HTTP + gRPC; shared UI login) |
| `otel-collector` | `4317`, `4318` | Single OTLP receiver; traces → Langfuse + OpenObserve, logs/metrics → Elasticsearch + OpenObserve |
| `postgres`, `clickhouse`, `redis`, `minio` | internal | Langfuse storage backends |

**Shared UI login** for Langfuse and OpenObserve: `admin@example.com` / `Admin1234!`
(set via `LANGFUSE_INIT_USER_*` and `ZO_ROOT_USER_*` in `.env`; applied on first
startup, so changing it requires recreating the affected volumes). Kibana has no
login.

**OpenObserve dashboards** are provisioned idempotently (replace-by-title) by
`./openobserve/provision.sh`, which imports the bundles in
`openobserve/dashboards/*.json`:

| Dashboard | What it shows |
|---|---|
| **Agent Harness — Agent Runs** | Run volume over time, runs by model/component, status, avg turns/duration, top sessions |
| **Agent Harness — Token Usage** | Total tokens over time, input/output/reasoning, tokens by model/phase, cache hit ratio |
| **Agent Harness — Cost** | Cost over time, by model, input vs output, top sessions (priced models) |
| **Agent Harness — Latency** | Avg run latency over time, avg/max, per-segment breakdown, slowest runs, per-turn model latency |
| **Agent Harness — Errors** | Error volume, by event/component, and recent error records (`body` carries the error summary) |

Each dashboard defaults its time range to the last day
(`defaultDatetimeDuration`).

**Trace → log correlation.** OpenObserve's trace "View Logs" is span-scoped
(`span_id='…' AND trace_id='…'`). Every agent the harness constructs (the main
agent, the `QualityCheck` LLM judge, and the guard fallback) carries a capability
ordered inside PydanticAI's instrumentation, so each emits `model_request` inside
its `chat` span and `agent_run` inside its `invoke_agent` span, and those spans
resolve to logs; tool spans carry the existing `tool_call`/`tool_result` records.
Records are gated by `HARNESS_TELEMETRY_LEVEL` (`minimal` emits none). Set
`HARNESS_TELEMETRY_ENABLED=false` to disable all telemetry exporters when no
OTel Collector is running.

## 4. Elasticsearch

Log records land in Elasticsearch:

- Log records → data stream `logs-generic.otel-default` (backing indices `.ds-logs-generic.otel-default-…`).
- Metrics → data stream `metrics-generic.otel-default` (TSDS).
- Trace spans → **Langfuse** (and **OpenObserve**) in the default stack.

### 4.1 Data shape

- The OTel log message lands in `body.text`; the **`event_name`** keyword field
  carries the same value in an aggregatable/filterable form (mirrored as
  `attributes.event.name`).
- Structured context is **flattened into dotted, typed `attributes.*` fields** so
  it can be aggregated in Kibana Lens rather than read as an opaque string:
  - `attributes.turn.index` / `.phase` / `.tool_names` / `.tool_call_count`
  - `attributes.token_usage.total_tokens` / `.input_tokens` / `.output_tokens`
    / `.reasoning_tokens` / `.cache_read_tokens` / `.cache_write_tokens`
    / `.cache_hit_ratio`
  - `attributes.cost.total_usd` / `.input_usd` / `.output_usd` / `.source`
  - `attributes.latency.model_seconds` / `.tool_seconds` / `.total_seconds`
  - `attributes.run.turn_count` / `.avg_turn_latency_seconds` / `.max_turn_latency_seconds`
  - `attributes.performance.duration_seconds`
  - `attributes.model_settings.max_tokens`
  - `attributes.tool.name` / `attributes.tool.parameters.*`
  - `attributes.error.type` / `.message` / `.stacktrace` / `.source` / `.handled`
  - `attributes.exception.type` / `.message` / `.stacktrace` (OTel/ECS mirror)
  - `attributes.code.file.path` / `.function` / `.line.number` (caller location on errors)
  - `attributes.component` (emitting subsystem, also the OTel scope name)
- Deployment-wide facts are **resource attributes** (set once per process, not
  repeated on every record): `resource.attributes.service.name`,
  `resource.attributes.deployment.environment`, `resource.attributes.host.name`.
- Log records emitted **while a span is active** carry top-level `trace_id` / `span_id` so they correlate with traces.
- Span exception events are additionally extracted into the logs stream as `event_name: exception` docs.
- The collector (`filter/drop_noise`) drops DEBUG records, lifecycle `*_started`
  markers, and `retry_wait` backoff events before they reach Elasticsearch; the
  completion record, `retry_attempt`, and traces carry that information.

### 4.1.1 Canonical error schema

A failure is projected into one schema and emitted to **logs, spans (parent and
child), and metric labels**, so the same fields appear everywhere:

| Field | Meaning |
|---|---|
| `error.type` | Exception class or guardrail type (e.g. `TokenLimitExceeded`) |
| `error.message` | Human-readable message |
| `error.stacktrace` | Full traceback (unset/`None`/`0` = full; `N` = last `N` frames) |
| `error.source` | `llm` / `guardrail` / `tool` / `memory` / `prompt` / `evaluator` / `output` / `unknown` |
| `error.handled` | `true` when a guardrail/callback recovered the error |
| `code.file.path` / `.function` / `.line.number` / `.namespace` | Caller location (your `agent.run(...)`) |
| `exception.type` / `.message` / `.stacktrace` | OTel/ECS mirror of `error.*` |

pydantic-ai runs the agent in its own asyncio task, so errors raised there do not
carry your frame. The harness records the caller at `ManagedAgent.run()` entry
and uses it as the `code.*` fallback (on error records and the parent/child
spans only). Guardrail token/cost limits are enforced by the harness (not via
pydantic-ai `UsageLimits`) and emitted once as `token_limit_exceeded` /
`cost_limit_exceeded` with `error.handled=true`. Metric labels use the same
keys: `agent_errors_total` carries `error.type`, `error.source`, `error.handled`,
`operation`.

### 4.2 Log queries (`logs-generic.otel-default*`)

```text
event_name: "run_summary"                per agent request: turn_count, cost, latency breakdown
event_name: "agent_turn"                 per model iteration: tokens, cost, latency, phase
event_name: "agent_run_failed"          a failed operation by event name (keyword)
event_name: (retry_attempt or filter_error or agent_run_failed or token_limit_exceeded)
severity_text: "ERROR"                   all error records
attributes.component: guards             filter by emitting subsystem
attributes.error.type: ValueError        drill into cause by type
attributes.token_usage.total_tokens: >100   expensive turns
attributes.cost.total_usd: >0.01         expensive turns (priced models)
attributes.latency.agent_core_seconds: >5   slow runs, split by segment
attributes.code.file.path: *             records that carry a caller location (errors)
```

Raw curl:

```bash
curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=service.name:<service-name>'
curl -s 'http://localhost:9200/logs-generic.otel-default*/_search?q=trace_id:<span-trace-id>'
curl -s http://localhost:9200/_cat/indices/*generic.otel-default*
```

### 4.3 Trace queries (`traces-generic.otel-default*`)

> Legacy stack only — in the default stack trace spans go to **Langfuse** (§4.4), not Elasticsearch.

Reference queries for failure telemetry:

```text
status.code: "STATUS_CODE_ERROR"                 all failed spans
name: *:failed                                   harness-owned failures only
error.type: builtins.ValueError                  drill into cause by type
error.source: tool                               / by harness error source (memory, tool, guardrail, ...)
events.name: exception                           spans that recorded an exception event
```

Raw curl:

```bash
curl -s 'http://localhost:9200/traces-generic.otel-default*/_search?_source=name,status,attributes.error.type,attributes.error.source&q=name:%22*:failed%22'
curl -s 'http://localhost:9200/traces-generic.otel-default*/_search?_source=name,status,events&q=events.name:exception'
```

> **`-*` vs `*` gotcha** — a data-view pattern like `logs-generic.otel-default-*` matches nothing because the backing indices are hidden (`.ds-…`). Use `logs-generic.otel-default*` (no trailing hyphen) so ES resolves the data stream itself. The same applies to `traces-generic.otel-default*`.

### 4.4 Linking a log to its Langfuse trace

Log records emitted inside an active span carry a top-level `trace_id`. With `OTELTracer(create_spans=True)` (as used by `fluent_app.py`) the harness owns the `agent_run` span for the whole run, so in-run records — tool calls, guardrail errors (`filter_error`), evaluator failures, `error_handled` — all carry `trace_id`.

`kibana/provision-dashboards.sh` adds a URL field format to the `logs-generic.otel-default*` data view so Kibana renders `trace_id` as a clickable **"View in Langfuse"** link:

```text
{LANGFUSE_UI_URL}/project/{LANGFUSE_PROJECT_ID}/traces/{trace_id}
```

Both values come from `.env` (`LANGFUSE_UI_URL`, `LANGFUSE_PROJECT_ID`). Note that `LANGFUSE_HOST` (`http://langfuse-web:3000`) is the in-Docker hostname and is **not** browser-reachable.

```bash
./kibana/provision-dashboards.sh   # from the repo root
```

> A link only appears on records that carry `trace_id` (emitted while a span was active); bootstrap/out-of-run logs have none. In production, sampling matters too: with `sample_rate < 1.0` an unsampled record still carries a `trace_id`, but Langfuse will not have the trace.

## 5. Jaeger (legacy stack)

Trace backend with a native OTLP gRPC ingest (host `:14317`, forwarded from the collector) and UI at **http://localhost:16686**.

To view your runs:

1. Open `http://localhost:16686`.
2. Pick a service in the **Service** dropdown — PydanticAI native spans appear under its OTel service name (e.g. `pydantic-ai`), harness spans under the harness `service_name` (e.g. `all-in-one-observability-demo`) — and hit **Find Traces**.
3. Click a trace row for the waterfall: `invoke_agent <name>` / `execute_tool <tool>` / `chat <model>` native spans, plus `{service}.{operation}:failed` / manual harness spans on failures.

> There is a small ingest delay (collector → Jaeger batch export); refresh if a just-run trace isn't listed yet.

## 6. Prometheus (legacy stack)

Metrics backend with a native OTLP receiver (`/api/v1/otlp/v1/metrics`), UI at **http://localhost:9090**.

Example PromQL(`{__name__=~...}`):

```promql
sum(all_in_one_observability_demo_agent_runs_total)                                       # runs
sum(all_in_one_observability_demo_agent_errors_total)                                     # errors
sum(all_in_one_observability_demo_agent_duration_seconds_sum) / \
  sum(all_in_one_observability_demo_agent_duration_seconds_count)                          # avg run time
```

Raw curl:

```bash
curl -s 'http://localhost:9090/api/v1/query?query=sum({__name__=~"all_in_one_observability_demo_agent_runs_total"})'
```

## 7. Grafana (legacy stack)

Single pane at **http://localhost:3000** (`admin`/`admin`) in the legacy stack. Datasources (Elasticsearch, Prometheus, Jaeger) and the **"Agent Harness — OTel Telemetry"** dashboard are auto-provisioned (Dashboards → OTel).

> In the default stack, `localhost:3000` is **Langfuse**, not Grafana.

- **Logs like Kibana** — Logs Drilldown (`/a/grafana-lokiexplore-app`) on the Elasticsearch datasource, or the dashboard's *Logs (Elasticsearch)* panel.
- **Metrics like Grafana** — Prometheus datasource (`/a/explore-metrics`) or PromQL panels, e.g. `sum(all_in_one_observability_demo_agent_runs_total)`.
- **Traces like Jaeger** — Jaeger UI (http://localhost:16686) or Grafana Explore → Jaeger for the native waterfall; select a span → *View in logs* jumps to correlated ES log records by `trace_id`. The dashboard also shows *Span volume by span name (Jaeger)*.

**Correlation** — logs and traces share `trace_id`/`span_id` (Jaeger's trace→logs link maps spans to ES logs); metrics correlate by `service.name` + timestamp (standard OTel behavior).

> Correlation links only resolve for log records that carry `trace_id`/`span_id` (i.e. records emitted while a span was active — in-run logs like `tool_call`). Boundary logs (`agent_run_started`/`completed`/`failed`) emitted outside any span do not carry trace context.

## 8. Kibana

Kibana is the log browser. Provision the data views and dashboards idempotently from the repo root (Kibana must be running):

```bash
docker compose up -d kibana
./kibana/provision-dashboards.sh
```

What the script does:

1. **Waits** for Kibana `/api/status` → `available`.
2. **Upserts** the data views `logs-generic.otel-default*` and `traces-generic.otel-default*` (timeField `@timestamp`).
3. **Adds a URL field format** on `trace_id` in the logs data view, so Discover renders it as a **"View in Langfuse"** link to `{LANGFUSE_UI_URL}/project/{LANGFUSE_PROJECT_ID}/traces/{trace_id}` (values read from `.env`, see §4.4).
4. **Imports** the saved-object bundles in `kibana/saved-objects/*.ndjson` and prints the dashboard URLs.

Dashboards (all built on the **logs** data view; trace analytics live in **Langfuse**):

| Dashboard | URL path | What it shows |
|---|---|---|
| **Agent Harness — Errors** | `/app/dashboards#/view/errors-exceptions-dashboard` | ERROR-severity trend, errors by event, exception types (`attributes.exception.type`), errors by component (`attributes.component`), recent errors, and a retries/failures view (drill down via clickable `trace_id`) |
| **Agent Harness — Debug Logs** | `/app/dashboards#/view/log-levels-dashboard` | Severity overview, top events (`event_name`), log volume, recent logs |
| **Agent Harness — Agent Runs** | `/app/dashboards#/view/agent-runs-dashboard` | Run volume, duration (`attributes.performance.duration_seconds`), runs by model/session/environment |
| **Agent Harness — Token Usage** | `/app/dashboards#/view/token-usage-dashboard` | Token usage by model/phase (`attributes.token_usage.*`) |

> The old trace-based Kibana dashboards (**Errors & Exceptions**, **LLM Performance**, **Tool Calls**) were removed: traces now go to **Langfuse**, so those ES-backed views would be empty. Use Langfuse for trace/LLM/tool analytics.

**Finding errors in Discover:** widen the time picker (the Errors dashboard defaults to `now-24h`), select the `logs-generic.otel-default*` data view, and filter `severity_text: "ERROR"` or `event_name: (retry_attempt or filter_error or agent_run_failed)`. Each error record carries `trace_id` (rendered as a **View in Langfuse** link).

In **Discover**, filter `service.name: <your-service-name>` to scope to one app.

## 9. Quick start

```bash
docker compose up -d
./kibana/provision-dashboards.sh
cd agent_harness_examples
uv sync
uv run python 12-observability/fluent_app.py
```

Then:

- **Langfuse traces** — http://localhost:3000 (log in with the seeded admin user from `.env`).
- **Elasticsearch logs** — query the `logs-generic.otel-default*` data stream (see §4.2).
- **Kibana** — Discover on `logs-generic.otel-default*`; click **View in Langfuse** on a record's `trace_id` to jump to the trace where it happened.
