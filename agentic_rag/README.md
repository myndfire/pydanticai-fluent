# agentic-rag

Retrieval-Augmented Generation (RAG) medical assistant agent built on the [`agent-harness`](https://github.com/myndfire/pydanticai-fluent) framework.

Demonstrates an end-to-end agentic RAG workflow using `ManagedAgent` with tool-based retrieval, guards, observability, and evaluators.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally (default: `gpt-oss:20b`) or another model provider

## Setup

```bash
# 1. Install dependencies
cd agentic_rag
uv sync

# 2. (Optional) Configure environment
cp .env.example .env
# Edit .env to add your LOGFIRE_TOKEN if using Logfire tracing
```

## Usage

Run the agent in a terminal chat loop:

```bash
uv run agent.py
```

The agent is a medical assistant with three retrieval tools. Example interactions:

```
You: What are my lipid panel results?
Agent: [calls get_labs(category='lipid panel')] ...

You: Do I have any liver issues?
Agent: [calls get_diagnosis(category='general')] ...

You: What does my chest x-ray show?
Agent: [calls get_findings(category='chest xray')] ...
```

## Project Structure

```
agentic_rag/
├── agent.py                    # Main agent with interactive chat loop
├── sample_data/
│   ├── labs.md                 # Sample lab report (Markdown)
│   ├── labs.pdf                # Sample lab report (PDF)
│   ├── docling.md              # Docling technical report (Markdown)
│   ├── docling.pdf             # Docling technical report (PDF)
│   └── invoice.md              # Sample invoice (Markdown)
├── pyproject.toml              # Dependencies and build configuration
├── uv.lock                     # Locked dependency versions
├── .env.example                # Environment variable template
└── README.md
```

## Tools

| Tool | Signature | Description |
|---|---|---|
| `get_labs` | `(category: str) -> list[str]` | Returns lab panel results (lipid profile, etc.) |
| `get_diagnosis` | `(category: str) -> list[str]` | Returns diagnosis information |
| `get_findings` | `(category: str) -> list[str]` | Returns imaging/x-ray findings |

All tools currently return mock data seeded in the agent code. Replace with real database or API calls for production use.

## Guards & Reliability

Configured via fluent API on the `ManagedAgent`:

| Guard | Setting |
|---|---|
| Agent retries | 3 attempts, 120s timeout |
| Tool retries | 3 attempts |
| Result validator retries | 3 attempts |
| Circuit breaker | Disabled by default |
| Content filter | Disabled by default |
| PII detection | Disabled by default |
| Cost limits | Disabled by default |

## Observability

- **Console logging** via structlog (always on)
- **Logfire tracing** (optional — set `LOGFIRE_TOKEN` in `.env`)
- **Evaluators**: `PrintEvaluator` (logs every turn) and `LLMJudgeEvaluator` (placeholder for LLM-as-judge scoring)

## License

Apache 2.0 — see [LICENSE](../../LICENSE).
