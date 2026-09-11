# OBSERVABILITY.md — Langfuse, Elasticsearch & Kibana

How to run the observability stack and inspect agent telemetry in **Langfuse** (traces), **Elasticsearch** (structured logs), and **Kibana** (log browser with a click-through link to the Langfuse trace where the event happened).

> The Jaeger / Prometheus / Grafana sections at the end describe the **legacy**
> stack in `docker-compose.yml.old`; the default `docker-compose.yml` ships
> Langfuse (traces) + Elasticsearch/Kibana (logs) and routes metrics to the
> collector's `debug` exporter.

## 1. Overview & architecture

The OTEL backends (`OTELLogger`, `OTELTracer`, `OTELMetrics`) export over OTLP gRPC to the OpenTelemetry Collector, which routes each signal to a backend:

> **OTLP is the de-facto transport for all telemetry.** Every OTEL backend ships logs, metrics, and traces to the OTel Collector at `localhost:4317` (gRPC) / `localhost:4318` (HTTP).

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
| `langfuse-web` | `3000` | Trace backend + UI (login with the seeded admin user from `.env`) |
| `elasticsearch` | `9200` | Logs backend — native OTLP/HTTP intake (`/_otlp/v1/logs`) → OTel data stream `logs-generic.otel-default` |
| `kibana` | `5601` | Log browser; renders `trace_id` as a "View in Langfuse" link |
| `otel-collector` | `4317`, `4318` | Single OTLP receiver; traces → Langfuse, logs → Elasticsearch, metrics → debug |
| `postgres`, `clickhouse`, `redis`, `minio` | internal | Langfuse storage backends |

## 4. Elasticsearch

Log records land in Elasticsearch:

- Log records → data stream `logs-generic.otel-default` (backing indices `.ds-logs-generic.otel-default-…`).
- Trace spans → **Langfuse** in the default stack (in the legacy stack they also went to `traces-generic.otel-default`, see §5-§7).

### 4.1 Data shape

- The OTel log message lands in `body.text` (the string body is wrapped in an object).
- All log context lands under `attributes.*` (`session_id`, `model`, `error`, `error_type`, `duration_seconds`, `code.*`, `exception.*`, …).
- Log records emitted **while a span is active** carry top-level `trace_id` / `span_id` so they correlate with traces.
- Span exception events are additionally extracted into the logs stream as `event_name: exception` docs carrying `attributes.exception.type`, `attributes.exception.message`, `attributes.exception.stacktrace`.

### 4.2 Log queries (`logs-generic.otel-default*`)

```text
body.text: "agent_run_failed"            a failed operation by message
attributes.code.file.path: *             records that carry a callsite
attributes.exception.stacktrace: *       failed/error_handled records with the full traceback
attributes.code.file.path: 09_otel_oltp_logs_traces_metrics.py AND attributes.code.line.number: >0
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
| **Agent Harness — Errors** | `/app/dashboards#/view/errors-exceptions-dashboard` | ERROR-severity trend, top messages (`attributes.error_message`), exception types, raise sites, and recent errors (with `langfuse_trace_url`) |
| **Agent Harness — Debug Logs** | `/app/dashboards#/view/log-levels-dashboard` | Severity overview, log volume, recent logs |
| **Agent Harness — Agent Runs** | `/app/dashboards#/view/agent-runs-dashboard` | Run volume/duration |
| **Agent Harness — Token Usage** | `/app/dashboards#/view/token-usage-dashboard` | Token usage by model/phase |

> The old trace-based Kibana dashboards (**Errors & Exceptions**, **LLM Performance**, **Tool Calls**) were removed: traces now go to **Langfuse**, so those ES-backed views would be empty. Use Langfuse for trace/LLM/tool analytics.

**Finding errors in Discover:** widen the time picker (the Errors dashboard defaults to `now-24h`), select the `logs-generic.otel-default*` data view, and filter `severity_text: "ERROR"`. Note `body.text` is analyzed — search `body.text: filter_error` (not `body.text: error`). Each error record carries `trace_id` and a `langfuse_trace_url` field.

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