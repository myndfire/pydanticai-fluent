# Orchestration Patterns

Four patterns for coordinating multiple agents: delegation, sequential pipeline, classify-and-route, and parallel fan-out.

## Overview

Orchestration composes multiple `ManagedAgent` instances into larger workflows. Each pattern solves a different coordination problem.

```
Pattern 1: Delegation          Pattern 2: Pipeline
┌──────────┐                   ┌──────────┐     ┌──────────┐     ┌──────────┐
│Coordinator│──tool──▶         │Researcher│──▶  │  Writer  │──▶  │  Editor  │
└──────────┘                   └──────────┘     └──────────┘     └──────────┘
     │                                                                      │
     ▼                                                                      ▼
 Specialist                                                          Final answer

Pattern 3: Routing             Pattern 4: Fan-Out
┌──────────┐                   ┌──────────┐
│  Router  │──classify──▶      │Coordinator│──tool──▶┌──────────────────────┐
└──────────┘                   └──────────┘         │ Legal │ Tech │ Biz  │
     │                                               └───────┴──────┴──────┘
     ├──▶ Billing                                                    │
     ├──▶ Tech Support                                    asyncio.gather
     └──▶ General                                         ◀── aggregate ──▶
```

| Pattern | File | Coordination style |
|---------|------|--------------------|
| Tool-driven delegation | `01_delegation.py` | Coordinator delegates to specialist via tool |
| Sequential pipeline | `02_sequential_pipeline.py` | Output of agent N becomes input to agent N+1 |
| Classify and route | `03_routing.py` | Router classifies, program dispatches to specialist |
| Parallel fan-out | `04_parallel_fanout.py` | Coordinator fans out to multiple specialists concurrently |

## Files

### 01_delegation.py

A coordinator agent delegates tasks to a finance specialist via a `delegate_to_specialist` tool. A `SharedContext` dataclass records all delegations across turns so the coordinator can recall what the specialist has done.

```
User ──▶ Coordinator Agent
              │
              │ calls delegate_to_specialist(task)
              ▼
         Specialist Agent
              │
              │ returns result
              ▼
         SharedContext ◀── delegation_log updated
              │
              ▼
         Coordinator returns final answer
```

Key components:
- `SharedContext` with `delegation_log` — tracks task/result pairs
- `delegate_to_specialist(ctx, task)` — tool that runs the specialist and records to `SharedContext`
- Coordinator uses `deps_type=SharedContext` for dependency injection
- Three turns: Q3 revenue calculation, expense analysis, recall past delegations

### 02_sequential_pipeline.py

Three agents run in sequence — researcher, writer, editor — with no tools. Each agent's output feeds directly into the next as its prompt. A `SharedContext` holds intermediate outputs.

```
User question
    │
    ▼
Researcher Agent ──(facts)──▶ Writer Agent ──(draft)──▶ Editor Agent
                                                           │
                                                           ▼
                                                   Final polished answer
```

Key components:
- Three `ManagedAgent` instances (researcher, writer, editor) sharing one `ModelConfig`
- `SharedContext` with `research`, `draft`, `final` fields
- Fully programmatic flow — no tool calls, just prompt chaining
- Demo: research vector search → write explanation → edit for clarity

### 03_routing.py

A router agent classifies requests via a `classify_request` tool, then the program routes to the appropriate specialist (billing, tech support, or general). Classification is stored in `SharedContext`.

```
User query
    │
    ▼
Router Agent (classify_request tool)
    │
    │ returns: "billing" / "tech-support" / "general"
    ▼
Program reads SharedContext.classification
    │
    ├──"billing"──────▶ Billing Specialist
    ├──"tech-support"──▶ Tech Support Specialist
    └──"general"───────▶ General Specialist
                             │
                             ▼
                       Specialist output
```

Key components:
- `SharedContext` with `classification`, `specialist_used`, `routing_log`
- `classify_request(ctx, text)` — keyword-based classifier that sets `ctx.deps.classification`
- `specialist_map` dict maps classification → (agent, label)
- Three demo queries: billing complaint, tech support, general question

### 04_parallel_fanout.py

A coordinator agent has a `gather_perspectives` tool that fans out a question to three specialists (legal, tech, business) concurrently via `asyncio.gather`, then aggregates their perspectives.

```
Coordinator Agent (gather_perspectives tool)
         │
         │ fires: runs 3 specialists concurrently
         │
         ├──▶ Legal Analyst ──────┐
         ├──▶ Tech Analyst ───────┤── asyncio.gather ──▶ aggregate ──▶ return
         ├──▶ Business Analyst ───┘
         │
         ▼
SharedContext ◀── all perspectives recorded
```

Key components:
- `SharedContext` with `perspectives` list
- `gather_perspectives(ctx, question)` — runs three specialists in parallel
- `asyncio.gather(ask_legal(), ask_tech(), ask_business())` — concurrent execution
- Two turns: remote work policy, AI in customer support

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with `gpt-oss:20b` pulled

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull the model (first time only)
ollama pull gpt-oss:20b

# 3. Install dependencies
cd agent_harness_examples
uv sync

# 4. (Optional) Copy and edit .env
cp .env.example .env
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_NAME` | No | `gpt-oss:20b` | LLM model name |
| `LLM_PROVIDER` | No | `ollama` | LLM provider |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama endpoint |

## Running

```bash
# Delegation
uv run python 7-orchestration/01_delegation.py

# Sequential pipeline
uv run python 7-orchestration/02_sequential_pipeline.py

# Classify and route
uv run python 7-orchestration/03_routing.py

# Parallel fan-out
uv run python 7-orchestration/04_parallel_fanout.py
```

## Expected Output

**01_delegation.py:** Three turns showing coordinator→specialist delegation. Turn 1 calculates Q3 revenue, Turn 2 analyzes expenses, Turn 3 recalls prior work. Ends with delegation log summary.

**02_sequential_pipeline.py:** Three stages — researcher gathers facts, writer produces draft, editor polishes. Final output is the edited explanation of vector search.

**03_routing.py:** Three queries classified and routed — "charged twice" → billing specialist, "can't log in" → tech support, "office hours" → general specialist. Ends with routing log.

**04_parallel_fanout.py:** Two turns with concurrent specialist execution. Turn 1 gathers perspectives on remote work policy, Turn 2 on AI in customer support. Shows fan-out and fan-in timing.

## How It Works

1. **01_delegation.py** — The coordinator's `delegate_to_specialist` tool creates a fresh `ManagedAgent` for the specialist, runs it with a sub-session, and records the result to `SharedContext`. The coordinator sees the specialist's output in its context and incorporates it into its final answer.

2. **02_sequential_pipeline.py** — A simple Python pipeline: run researcher, capture output, inject into writer prompt, capture output, inject into editor prompt. No tools or callbacks — just sequential `agent.run()` calls with prompt chaining.

3. **03_routing.py** — The router agent calls `classify_request` which sets `ctx.deps.classification` via keyword matching. The program reads the classification and dispatches to the matching specialist agent. Each turn creates a fresh `SharedContext` to isolate routing decisions.

4. **04_parallel_fanout.py** — The `gather_perspectives` tool defines three async closures (one per specialist), runs them with `asyncio.gather`, and aggregates results into a single report string returned to the coordinator.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the model: `ollama pull gpt-oss:20b`.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
- **Slow parallel fan-out** — Sequential fallback if `asyncio.gather` is not used; ensure all specialist `agent.run()` calls are awaited together.
