# Examples

Runnable examples for `pydanticai-fluent`. Copy this directory anywhere and run.

---

## Quick Start

### 1. Copy the directory

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/myndfire/pydanticai-fluent.git
cd pydanticai-fluent
git sparse-checkout set agent_harness_examples
mv agent_harness_examples ../my-examples
cd ../my-examples
```

Or simply copy `agent_harness_examples/` from the GitHub web UI.

### 2. Install dependencies

```bash
uv sync
# Or: pip install -e .
```

This installs `pydanticai-fluent` as an editable install from the local `../` project root.

### 3. Run any example

```bash
uv run agent_example-1.py
uv run loops/01_interactive_loop.py
uv run messaging/rabbitmq/rabbitmq_agent.py
```

---

## Start Infrastructure

All example services come from the root compose file. From the repo root:

```bash
docker compose -f docker-compose.yml up -d            # all services
docker compose -f docker-compose.yml up -d mongo      # just what an example needs
```

Services:

| Service | Port(s) | Used by |
|---|---|---|
| `mongo` | `27017` | MongoMemory, MongoPrompts |
| `redis` | `6379` | RedisMemory |
| `elasticsearch` | `9200` | ElasticsearchMemory, ElasticsearchLogger, OTel logs backend |
| `kibana` | `5601` | Optional specialist — Kibana log browser (Grafana Logs Drilldown covers this) |
| `grafana` | `3000` | Single pane: logs from ES, metrics from Prometheus, trace waterfall from Jaeger |
| `prometheus` | `9090` | Metrics backend — native OTLP receiver (ingests the collector's OTLP metrics) |
| `jaeger` | `16686`, `14317`, `14318` | Trace backend — native OTLP gRPC ingest (host :14317/:14318), UI at :16686 |
| `otel-collector` | `4317`, `4318` | Single OTLP receiver → ES logs + Prometheus metrics + Jaeger traces (OTELLogger, OTELTracer, OTELMetrics) |
| `pushgateway` | `9091` | PrometheusMetrics |
| `rabbitmq` | `5672`, `15672` | Messaging examples |

Stop everything with `docker compose -f docker-compose.yml down -v`.

### Visualize OTel signals (Grafana single pane)

Start the observability stack, then run the all-in-one example:

```bash
docker compose -f docker-compose.yml up -d elasticsearch otel-collector kibana grafana jaeger prometheus
uv run python observability/09_otel_oltp_logs_traces_metrics.py
```

Full docs on using **Elasticsearch** (log/trace queries), **Jaeger**, **Prometheus**, and **Grafana** (datasources, Logs Drilldown, PromQL, Explore → Jaeger) live in **[`OBSERVABILITY.md`](../OBSERVABILITY.md)**. Quick links:

- Grafana: http://localhost:3000 (`admin`/`admin`) — datasources (Elasticsearch, Prometheus, Jaeger) and the **"Agent Harness — OTel Telemetry"** dashboard are auto-provisioned (Dashboards → OTel → Agent Harness — OTel Telemetry).
- Logs like Kibana — Logs Drilldown (`/a/grafana-lokiexplore-app`) on the Elasticsearch datasource.
- Traces like Jaeger — Jaeger UI (http://localhost:16686) or Grafana Explore → Jaeger.

### Kibana log-levels dashboard

Provision a purpose-built **severity dashboard** in Kibana (bar by severity, volume-over-time by severity, donut share, recent-logs table). Kibana only file-provisions data views (not Lens panels), so this pushes pre-built saved objects via the import API — idempotent, re-run anytime:

```bash
docker compose -f docker-compose.yml up -d kibana
./kibana/provision-log-levels-dashboard.sh
```

Opens at `http://localhost:5601/app/dashboards#/view/log-levels-dashboard` ("Agent Harness — Log Levels"). See **[`OBSERVABILITY.md §8`](../OBSERVABILITY.md#8-kibana-optional)** for the panel table, script steps, and the data-stream `-*` vs `*` gotcha.

---

## Build Your Own Project

After exploring examples, create a new independent project.

### Step 1: Scaffold a new project

```bash
mkdir my-agent-project
cd my-agent-project
uv init
```

### Step 2: Add pydanticai-fluent

```bash
uv add "git+https://github.com/myndfire/pydanticai-fluent.git@master"
```

Or in `pyproject.toml`:

```toml
[project]
dependencies = [
    "pydanticai-fluent @ git+https://github.com/myndfire/pydanticai-fluent.git@master",
]
```

### Step 3: Write your agent

Copy any example as a starter. Here's a minimal one:

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

### Step 4: Learn more

- [`USAGE.md`](https://github.com/myndfire/pydanticai-fluent/blob/main/USAGE.md) — Complete API reference
- [`README.md`](https://github.com/myndfire/pydanticai-fluent/blob/main/README.md) — Installation and overview

---

## Example Index

| Category | File/Directory | What It Shows |
|---|---|---|
| **Core** | `agent_example-1.py` | Basic agent with tools + evaluator |
| | `agent_example-2.py` | Error handling, multi-turn, observability |
| | `agent_example-3.py` | Structured output (Pydantic models) |
| **Loops** | `loops/01_interactive_loop.py` | Interactive chat loop |
| | `loops/02_iterative_refinement.py` | Self-improving loop |
| | `loops/03_react_loop.py` | ReAct pattern (reasoning + action) |
| | `loops/04_goal_seeking_loop.py` | Goal-directed agent |
| | `loops/05_planning_loop.py` | Planning + execution loop |
| **Orchestration** | `orchestration/01_delegation.py` | Tool-driven delegation |
| | `orchestration/02_sequential_pipeline.py` | Multi-agent pipeline |
| | `orchestration/03_routing.py` | Classify and route |
| | `orchestration/04_parallel_fanout.py` | Parallel fan-out / fan-in |
| **Tools** | `tools/01_plain_tools.py` | Register plain functions as tools |
| **Memory** | `memory/01_in_memory.py` | In-memory conversation storage |
| | `memory/02_mongo_memory.py` | MongoDB persistent memory |
| | `memory/03_redis_memory.py` | Redis memory |
| | `memory/04_elasticsearch_memory.py` | Elasticsearch memory |
| **Observability** | `observability/01_logging.py` | Structured logging (console/file) |
| | `observability/02_tracing_metrics.py` | In-memory spans & metrics inspection |
| | `observability/03_builder_logs_metrics.py` | Fluent ObservabilityBuilder (logs + metrics) |
| | `observability/04_composite_logs.py` | Multiple logging backends together |
| | `observability/05_elasticsearch_logging.py` | Direct ES structured logging (daily indices) |
| | `observability/06_otel_jaeger_logs_traces_metrics.py` | OTel logs+traces+metrics → Jaeger |
| | `observability/07_prometheus_logs_metrics.py` | Prometheus logs + metrics with push gateway |
| | `observability/08_live_agent_logs_metrics.py` | Live agent with composed logs + metrics |
| | `observability/09_otel_oltp_logs_traces_metrics.py` | OTel logs+traces+metrics → ES logs + Prometheus metrics + Jaeger traces (via one OTLP collector) |
| **Prompts** | `prompts/01_static_prompts.py` | Fixed system prompt |
| | `prompts/02_mongo_prompts.py` | Jinja2 templates from MongoDB |
| | `prompts/03_prompt_variables.py` | Dynamic prompt rendering |
| **Evaluators** | `evaluators/01_quality_check.py` | LLM-as-judge scoring |
| | `evaluators/02_safety_check.py` | OpenAI moderation |
| | `evaluators/03_custom_evaluator.py` | Write your own evaluator |
| | `evaluators/04_protocol_evaluator.py` | Protocol-based evaluator |
| | `evaluators/05_combined_evaluators.py` | Multiple evaluators together |
| **Error Handling** | `error_handling/01_basic_handler.py` | Per-source error callbacks |
| | `error_handling/09_pipeline_error_recovery.py` | Pipeline continues on failure |
| **Guardrails** | `guardrails/01_content_filter.py` | Content filtering |
| | `guardrails/02_pii_detection.py` | PII redaction |
| | `guardrails/03_token_limits.py` | Token usage caps |
| | `guardrails/04_cost_limits.py` | Dollar cost limits |
| **Structured Output** | `structured_output/01_basic_model.py` | Constrain to Pydantic model |
| | `structured_output/03_nested_models.py` | Nested Pydantic schemas |
| | `structured_output/04_validation_retries.py` | Retry on validation failure |
| **Messaging** | `messaging/rabbitmq/rabbitmq_agent.py` | Message-driven agent |
