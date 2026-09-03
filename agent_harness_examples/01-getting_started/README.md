# Getting Started Examples

Three examples demonstrating the `agent_harness` fluent API, from minimal tool usage to structured output with observability.

## Overview

```
example-1: Tools              example-2: Observability       example-3: Structured Output
┌──────────┐                  ┌──────────┐                   ┌──────────┐
│  Agent   │                  │  Agent   │                   │  Agent   │
│          │                  │          │                   │          │
│ ┌──────┐ │                  │ Tracer ──┼──▶ OTEL (OTLP)     │ .with_   │
│ │repeat│ │                  │ Logger ──┼──▶ OTEL (OTLP)     │ output() │
│ │shout │ │                  │ Metrics ─┼──▶ OTEL (OTLP)     │ (Invoice)│
│ └──────┘ │                  │ Memory ──┼──▶ MongoDB (opt)  └──────────┘
└──────────┘                  └──────────┘                        │
                                                                  ▼
  "repeat hello,                "what 2+2?"               Prompt → Invoice
   then shout"                 "add 1"                    (Pydantic model)
                                "add 2 more"
```

## Prerequisites

### Required (all examples)

| Requirement | Setup |
|---|---|
| **Ollama** | Install from [ollama.com](https://ollama.com), then run `ollama serve` |
| **Model** | `ollama pull qwen2.5:3b` (or set `MODEL_NAME` in `../.env` to a pulled model) |
| **Dependencies** | `cd agent_harness_examples && uv sync` |

### Optional (example-2 and example-3)

| Requirement | Purpose | Setup |
|---|---|---|
| **OTel Collector** | Logs, traces, and metrics backend | Run collector on `localhost:4317` (gRPC) |
| **MongoDB** | Persistent long-term memory | Set `MONGODB_URI` in `../.env`; if unset, falls back to in-memory storage |

## Environment Variables

All variables are read from `../.env`.

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `MODEL_NAME` | Yes | `qwen2.5:3b` | all | Ollama model identifier |
| `OTEL_COLLECTOR_ENDPOINT` | No | `localhost:4317` | example-2, example-3 | OTel Collector OTLP gRPC endpoint |
| `MONGODB_URI` | No | - | example-2 | MongoDB connection string (e.g., `mongodb://localhost:27017`) |
| `MONGODB_DATABASE` | No | `agent_memory` | example-2 | MongoDB database name |
| `MONGODB_COLLECTION` | No | `conversations` | example-2 | MongoDB collection name |

## Running

All commands from the `agent_harness_examples/` directory:

```bash
cd agent_harness_examples
uv sync
```

### example-1 -- Minimal agent with tools

```bash
uv run python 1-getting_started/agent_example-1.py
```

Registers `repeat` and `shout` tools, runs a multi-step prompt requiring both tools in sequence. No external services beyond Ollama needed.

```
Prompt: "repeat 'hello world', then shout the result"
    │
    ▼
Agent (LLM)
    │
    ├──▶ tool: repeat("hello world") ──▶ "hello world"
    │
    └──▶ tool:shout("hello world")   ──▶ "HELLO WORLD"
                                        │
                                        ▼
                                  Final output: "HELLO WORLD"
```

### example-2 -- Observability, tracing, and memory

```bash
uv run python 1-getting_started/agent_example-2.py
```

Full observability stack: OTEL logging, tracing, and metrics. Runs three sequential prompts in the same session to demonstrate conversation continuity. Optionally uses MongoDB for persistent long-term memory.

```
Observability stack:
┌─────────────────────────────────────────────────┐
│  Observability                                  │
│  ├── logger:  OTELLogger   ──▶ localhost:4317   │
│  ├── tracer:  OTELTracer   ──▶ localhost:4317   │
│  ├── metrics: OTELMetrics  ──▶ localhost:4317   │
│  └── memory:  MongoDB (optional)                │
└─────────────────────────────────────────────────┘

Session flow (3 turns, same session_id):
    │
    ├──▶ "what 2+2?"           ──▶ "4"
    ├──▶ "add 1, total?"       ──▶ "5"
    └──▶ "add 2 more, total?"  ──▶ "7"

Memory:
    ├── short_term: InMemoryProvider (ephemeral)
    └── long_term:  MongoMemory (optional, persistent)
```

### example-3 -- Structured output with Pydantic model

```bash
uv run python 1-getting_started/agent_example-3.py
```

Generates an invoice from natural language, then extracts structured data from a markdown invoice file using a Pydantic `Invoice` model. Requires a `data/invoice.md` file relative to the working directory.

```
Prompt 1: "generate invoice for Acme Inc..."
    │
    ▼
Agent (.with_output(Invoice))
    │
    ▼
Invoice (Pydantic model)
    ├── invoice_number: "INV-..."
    ├── date_issued: "2022-01-15"
    ├── services_provided: [...]
    ├── subtotal: 1625.0
    ├── tax_amount: 325.0
    └── total_amount_due: 1950.0

Prompt 2: "extract data from invoice.md"
    │
    ▼
Agent (reads markdown, returns Invoice)
    │
    ▼
Invoice (structured extraction)
```
