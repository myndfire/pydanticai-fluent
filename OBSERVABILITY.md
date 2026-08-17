# OBSERVABILITY.md — Elasticsearch, Jaeger, Prometheus & Grafana

How to run the observability stack and inspect agent telemetry in **Elasticsearch** (logs + traces), **Jaeger** (trace waterfall), **Prometheus** (metrics), **Grafana** (single pane), and optionally **Kibana** (log dashboards).

## 1. Overview & architecture

The OTEL backends (`OTELLogger`, `OTELTracer`, `OTELMetrics`) export over OTLP gRPC to the OpenTelemetry Collector, which routes each signal over OTLP to a separate native-OTLP backend:

```
agent_harness  --OTLP gRPC:14317-->  otel-collector  --otlphttp-->  Elasticsearch (logs)
    (logs + metrics + traces)        (otlp receiver:   --otlphttp-->  Prometheus (metrics)
                                       grpc :4317,      \--otlp gRPC-->  Jaeger (traces)
                                       http :4318)
```

## 2. Start the stack

From the repo root:

```bash
docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana jaeger prometheus
```

Skip `kibana` if you don't need the optional specialist browser (Grafana Logs Drilldown covers it).

## 3. Service & port reference

| Service | Port | Role |
|---|---|---|
| `elasticsearch` | `9200` | Logs backend — native OTLP/HTTP intake (`/_otlp/v1/logs`) → OTel data stream `logs-generic.otel-default` |
| `grafana` | `3000` | Single pane — ES datasource (logs), Prometheus datasource (metrics), Jaeger datasource (trace waterfall) |
| `prometheus` | `9090` | Metrics backend — native OTLP receiver (`/api/v1/otlp/v1/metrics`) |
| `jaeger` | `16686` | Trace backend — native OTLP gRPC ingest (`:4317`), UI at `:16686` |
| `kibana` | `5601` | Optional specialist log browser (Grafana Logs Drilldown covers this) |
| `otel-collector` | `14317`, `14318` | Single OTLP receiver; traces → Jaeger, metrics → Prometheus, logs → Elasticsearch |

## 4. Elasticsearch

Both logs and traces land in Elasticsearch:

- Log records → data stream `logs-generic.otel-default` (backing indices `.ds-logs-generic.otel-default-…`).
- Trace spans → data stream `traces-generic.otel-default`.

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

## 5. Jaeger

Trace backend with a native OTLP gRPC ingest (`:4317`) and UI at **http://localhost:16686**.

To view your runs:

1. Open `http://localhost:16686`.
2. Pick a service in the **Service** dropdown — PydanticAI native spans appear under its OTel service name (e.g. `pydantic-ai`), harness spans under the harness `service_name` (e.g. `all-in-one-observability-demo`) — and hit **Find Traces**.
3. Click a trace row for the waterfall: `invoke_agent <name>` / `execute_tool <tool>` / `chat <model>` native spans, plus `{service}.{operation}:failed` / manual harness spans on failures.

> There is a small ingest delay (collector → Jaeger batch export); refresh if a just-run trace isn't listed yet.

## 6. Prometheus

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

## 7. Grafana

Single pane at **http://localhost:3000** (`admin`/`admin`). Datasources (Elasticsearch, Prometheus, Jaeger) and the **"Agent Harness — OTel Telemetry"** dashboard are auto-provisioned (Dashboards → OTel).

- **Logs like Kibana** — Logs Drilldown (`/a/grafana-lokiexplore-app`) on the Elasticsearch datasource, or the dashboard's *Logs (Elasticsearch)* panel.
- **Metrics like Grafana** — Prometheus datasource (`/a/explore-metrics`) or PromQL panels, e.g. `sum(all_in_one_observability_demo_agent_runs_total)`.
- **Traces like Jaeger** — Jaeger UI (http://localhost:16686) or Grafana Explore → Jaeger for the native waterfall; select a span → *View in logs* jumps to correlated ES log records by `trace_id`. The dashboard also shows *Span volume by span name (Jaeger)*.

**Correlation** — logs and traces share `trace_id`/`span_id` (Jaeger's trace→logs link maps spans to ES logs); metrics correlate by `service.name` + timestamp (standard OTel behavior).

> Correlation links only resolve for log records that carry `trace_id`/`span_id` (i.e. records emitted while a span was active — in-run logs like `tool_call`). Boundary logs (`agent_run_started`/`completed`/`failed`) emitted outside any span do not carry trace context.

## 8. Kibana (optional)

Kibana ships a built-in log viewer for log data streams, but to get a purpose-built **severity dashboard** (bar by severity, volume-over-time by severity, donut share, recent-logs table) you must provision saved objects — Kibana only supports file-based provisioning for data views, not for Lens panels/dashboards.

Provision it idempotently from the repo root (Kibana must be running):

```bash
docker compose up -d kibana
./kibana/provision-log-levels-dashboard.sh
```

What the script does:

1. **Waits** for Kibana `/api/status` → `available`.
2. **Upserts** the data view `logs-generic.otel-default*` (timeField `@timestamp`) — the OTel log data stream.
3. **Imports** `kibana/saved-objects/log-levels.ndjson` (`POST /api/saved_objects/_import?overwrite=true`) — 4 Lens panels + 1 dashboard, with `overwrite=true` so re-running is a no-op.
4. Prints the dashboard URL (`http://localhost:5601/app/dashboards#/view/log-levels-dashboard`, or find **Agent Harness — Log Levels**).

The panels:

| Panel | Type | What it shows |
|---|---|---|
| Logs by severity | Bar | Count of log records grouped by `severity_text` |
| Log volume over time by severity | Stacked area | `@timestamp` histogram split by `severity_text` |
| Severity share | Donut | Distribution of `severity_text` |
| Recent logs | Table | Time, severity, `resource.attributes.service.name`, `body.text` |

In **Discover**, use the `logs-generic.otel-default*` data view and filter `service.name: all-in-one-observability-demo`.

## 9. Quick start (all-in-one demo)

Run the all-in-one demo, then open the provisioned Grafana dashboard, the Logs Drilldown, or the Jaeger UI and search `service.name: all-in-one-observability-demo`:

```bash
cd agent_harness_examples
uv sync
uv run python observability/09_otel_oltp_logs_traces_metrics.py
```