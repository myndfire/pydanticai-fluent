# Tool Calling

Register functions as tools the agent can invoke — from simple callables to context-aware tools with dependency injection, MCP servers, and combinations with guards/evaluators.

## Overview

All examples use `ToolRegistry` to register Python functions as tools the LLM can call. The registry inspects function signatures to determine tool type: if the first parameter is `RunContext[T]`, it registers as a context-aware tool (`agent.tool()`); otherwise it registers as a plain tool (`agent.tool_plain()`).

```
Tool registration pipeline:
────────────────────────────────────────────────────────────────────

  Python function       ToolRegistry            ManagedAgent          LLM decides
  ┌──────────────┐      ┌──────────────┐        ┌──────────────┐     ┌──────────────┐
  │ def calc(x)  │ ───▶ │ .add(calc)   │  ───▶  │ .with_tools  │ ──▶ │ "I need the  │
  │              │      │              │        │   (registry)  │     │  calculator" │
  │ def profile( │ ───▶ │ .add(profile)│  ───▶  │              │ ──▶ │ "I need the  │
  │   ctx: RC)   │      │              │        │              │     │  profile"    │
  └──────────────┘      └──────────────┘        └──────────────┘     └──────────────┘
                             │                       │                       │
                         Inspects signature      Passes tools to LLM    LLM picks tool
                         → plain or context-aware                      → calls function
                                                                     → observes result
                                                                     → repeats or answers
```

| Example | File | Key concept |
|---------|------|-------------|
| Plain tools | `01_plain_tools.py` | `.add()` / `.add_many()` / `.clear()` — stateless callables |
| Context tools | `02_context_tools.py` | `RunContext[T]` — dependency injection + audit logging |
| MCP servers | `03_mcp_server.py` | `with_mcp_server()` — external tool discovery over HTTP |
| Combinations | `04_tool_combinations.py` | Tools + guards + evaluators + structured output |

## Files

### 01_plain_tools.py

Stateless functions the agent can call. `ToolRegistry` inspects function signatures and registers them via `agent.tool_plain()`. Type hints drive the tool schema the LLM sees.

```
ToolRegistry()
    │
    ├──▶ .add(calculator)
    │        ↓
    │    agent.tool_plain(calculator)
    │    schema: {"expression": str}
    │
    ├──▶ .add_many(get_weather, convert_currency)
    │        ↓
    │    agent.tool_plain(get_weather)
    │    agent.tool_plain(convert_currency)
    │
    └──▶ .clear()  →  empty registry
```

Four sub-examples:
```
Example 1: Single tool          Example 2: Multiple tools
  .add(calculator)                .add_many(calculator, get_weather, convert_currency)
       │                                │
       ▼                                ▼
  "sqrt(144)/2"                   "weather in Tokyo + 100 USD to JPY"
  → calculator("144**0.5 / 2")   → get_weather("Tokyo")
  → Result: 6.0                   → convert_currency(100, "USD", "JPY")

Example 3: Fluent chaining      Example 4: Clear + rebuild
  .add(calculator)                .add_many(calculator, get_weather)
    .add_many(get_weather,          .clear()  → 0 tools
              convert_currency)     .add(convert_currency) → 1 tool
```

Key components:
- `ToolRegistry().add(fn)` — register a single tool
- `ToolRegistry().add_many(fn1, fn2, ...)` — bulk registration
- `.clear()` — remove all tools
- `.get_tools()` — list registered tool functions
- Demo: calculator, weather, currency conversion

### 02_context_tools.py

Tools that receive `RunContext[T]` to access a dependency container. `ToolRegistry` detects `RunContext` in the first parameter and registers via `agent.tool()` instead of `agent.tool_plain()`.

```
UserDeps (dependency container)
┌──────────────────────────┐
│ user_id: "usr_abc123"    │
│ username: "alice"        │
│ role: "member"           │
│ api_calls: []            │
│ log_call(name, params)   │
└──────────┬───────────────┘
           │ injected into
           ▼
┌──────────────────────────────────────────────────────────────┐
│  ToolRegistry().add_many(get_profile, set_role,              │
│                          get_audit_log, echo)                 │
│                                                               │
│  Tool inspection:                                             │
│    get_profile(ctx: RunContext[UserDeps])                     │
│      → agent.tool()       (context-aware)                    │
│    set_role(ctx: RunContext[UserDeps], new_role: str)         │
│      → agent.tool()       (context-aware)                    │
│    get_audit_log(ctx: RunContext[UserDeps], count: int)       │
│      → agent.tool()       (context-aware)                    │
│    echo(message: str)                                        │
│      → agent.tool_plain() (plain)                            │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│  Run 1: "Check my user profile"                              │
│    → get_profile(ctx)                                        │
│    → deps.log_call("get_profile", "")                        │
│    → returns user_id, username, role, api_calls count         │
│                                                               │
│  Run 2: "Change my role to 'admin'"                          │
│    → set_role(ctx, "admin")                                  │
│    → deps.role = "admin"                                     │
│    → deps.log_call("set_role", "admin")                      │
│                                                               │
│  Run 3: "What is my current role?"                           │
│    → get_profile(ctx)  → role: "admin" (mutated!)            │
│                                                               │
│  Run 4: "Show last 3 audit log entries"                      │
│    → get_audit_log(ctx, 3)                                   │
│    → returns: ["get_profile()", "set_role(admin)", ...]      │
└──────────────────────────────────────────────────────────────┘
```

Key components:
- `UserDeps` dataclass — dependency container with `user_id`, `username`, `role`, `api_calls`
- `RunContext[UserDeps]` — first parameter in context-aware tools
- `ManagedAgent(deps_type=UserDeps)` — declares dependency type
- `agent.run(..., deps=deps)` — injects dependencies at runtime
- Demo: profile lookup → role change → verify → audit log

### 03_mcp_server.py

MCP (Model Context Protocol) server integration — add external tool servers over HTTP. The file demonstrates the API patterns and includes a live demo that connects to Context7 (a public, keyless MCP server for library documentation).

```
MCP integration options:
────────────────────────────────────────────────────────────────────

  Single MCP server:
    agent.with_mcp_server("http://localhost:8000/mcp/filesystem")

  Multiple MCP servers:
    agent.with_mcp_servers(
        "http://localhost:8000/mcp/filesystem",
        "http://localhost:8001/mcp/postgres",
        "http://localhost:8002/mcp/github",
    )

  MCP + tool_prefix (disambiguation):
    agent.with_mcp_server("http://localhost:8000/mcp/filesystem",
                           tool_prefix="fs_")
    agent.with_mcp_server("http://localhost:8001/mcp/s3",
                           tool_prefix="s3_")

  MCP + custom tools:
    agent.with_tools(ToolRegistry().add(echo))
    agent.with_mcp_servers(
        "http://localhost:8000/mcp/filesystem",
        "http://localhost:8001/mcp/web-search",
        tool_prefix="mcp_",
    )
```

Live demo (requires network — connects to Context7 + Complex server):
```
MCP_HTTP_URL defaults to https://mcp.context7.com/mcp
MCP_COMPLEX_URL defaults to https://mcpplaygroundonline.com/mcp-complex-server

agent = ManagedAgent()
    .with_model(...)
    .with_tools(ToolRegistry().add(echo))
    .with_mcp_server(MCP_HTTP_URL, tool_prefix="ctx7_")
    .with_mcp_server(MCP_COMPLEX_URL, tool_prefix="complex_")

# Tools available: ctx7_*, complex_*, echo
```

Key components:
- `with_mcp_server(url)` — add a single MCP server as toolset
- `with_mcp_servers(url1, url2, ...)` — add multiple MCP servers
- `tool_prefix` — prefix tool names to avoid collisions
- `MCP_HTTP_URL` env var — primary MCP server URL (defaults to Context7)
- `MCP_COMPLEX_URL` env var — second MCP server (defaults to Complex server)
- Demo: two MCP servers + custom echo tool combined

### 04_tool_combinations.py

Combining tools with guards, evaluators, and structured output — a full-stack agent with retries, content filtering, token limits, and tool usage tracking.

```
ManagedAgent(deps_type=SearchDeps)
    │
    ├──▶ .with_tools(ToolRegistry)
    │        ├── search_docs(query)         → plain tool
    │        ├── calculate_rating(reviews)  → plain tool
    │        └── search_with_context(ctx, query, limit)  → context-aware
    │
    ├──▶ .with_agent_retries(AgentRetryConfig)
    │        max_retries=2, timeout=30
    │
    ├──▶ .with_tool_retries(ToolRetryConfig)
    │        max_retries=2, backoff=1.5
    │
    ├──▶ .with_content_filter(ContentFilterConfig)
    │        on_filter: clean_tool_output (regex replaces "definitely" → "certainly")
    │
    ├──▶ .with_token_limits(TokenLimitsConfig)
    │        max_total_tokens=500
    │
    └──▶ .with_evaluators(ToolUsageEvaluator)
             logs prompt, output length, session
```

Tool execution flow:
```
Run 1: "What is Flask? Search the docs."
    │
    ▼
LLM decides: search_docs("flask")
    │
    ▼
Tool returns: "Flask is a micro web framework for Python."
    │
    ▼
Content filter: clean_tool_output(text)  →  no changes
    │
    ▼
LLM final answer

Run 4: "Is Flask definitely a good framework?"
    │
    ▼
LLM decides: search_docs("definitely")
    │
    ▼
Tool returns: "Most definitely!"
    │
    ▼
Content filter: clean_tool_output(text)
    "Most definitely!" → "Most certainly!"
    │
    ▼
LLM final answer (with cleaned text)
```

Key components:
- `SearchDeps` — dependency container with `search_engine`, `max_results`, `queries_made`
- `clean_tool_output()` — regex-based content filter for tool responses
- `ToolUsageEvaluator` — logs prompt, output length, session after each turn
- `SearchResult(BaseModel)` — structured output model (not used in runs, but available)
- Four runs: search, context-aware search, calculate rating, content filter demo

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the required model pulled

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model (first time only)
ollama pull gpt-oss:20b

# 3. Install dependencies
cd agent_harness_examples
uv sync

# 4. (Optional) Copy and edit .env
cp .env.example .env
```

## Configuration

All variables are optional and read from `.env` via `python-dotenv`.

| Variable | File(s) | Default | Description |
|----------|---------|---------|-------------|
| `TOOL_CALLING_MODEL_NAME` | all | `gpt-oss:20b` | LLM model name |
| `TOOL_CALLING_MAX_TOKENS` | all | `512` | Max LLM output tokens |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |
| `MCP_HTTP_URL` | `03_mcp_server.py` | `https://mcp.context7.com/mcp` | Primary MCP server URL (public keyless) |
| `MCP_COMPLEX_URL` | `03_mcp_server.py` | `https://mcpplaygroundonline.com/mcp-complex-server` | Second MCP server for multi-server demo (public keyless) |

## Running

Each file is an independent entry point:

```bash
# Plain tools — register stateless callables
uv run python 09-tool_calling/01_plain_tools.py

# Context tools — dependency injection with RunContext
uv run python 09-tool_calling/02_context_tools.py

# MCP servers — external tool integration over HTTP
uv run python 09-tool_calling/03_mcp_server.py

# Tool combinations — tools + guards + evaluators + structured output
uv run python 09-tool_calling/04_tool_combinations.py
```

## Expected Output

**01_plain_tools.py:** Four sub-examples: single tool (calculator), multiple tools (weather + currency), fluent chaining, and clear/rebuild. Shows tool registration and invocation.

**02_context_tools.py:** Four runs: get profile, change role, verify profile (shows mutation), audit log. Demonstrates dependency injection and state mutation across turns.

**03_mcp_server.py:** API pattern examples + live demo connecting to Context7 (library docs) and Complex server (data/user/order tools) with `tool_prefix` disambiguation. Agent uses MCP tools to resolve library ID and fetch docs excerpt. Requires network egress.

**04_tool_combinations.py:** Four runs: search docs, context-aware search, calculate ratings, content filter demo. Shows the full stack with guards, evaluators, and content filtering.

## How It Works

1. **01_plain_tools.py** — `ToolRegistry` inspects function signatures. Plain functions (no `RunContext`) register via `agent.tool_plain()`. Type hints become the JSON schema the LLM sees.

2. **02_context_tools.py** — Functions with `RunContext[T]` as first parameter register via `agent.tool()`. The dependency container `T` is injected at runtime via `agent.run(..., deps=deps)`. Tools can mutate deps state.

3. **03_mcp_server.py** — `with_mcp_server()` / `with_mcp_servers()` add external HTTP tool servers via `MCPToolset(FastMCPClient(url))`. `tool_prefix` avoids name collisions via `.prefixed()`. MCP tools are discovered lazily at `agent.run()` time. Example 4 combines two public servers (Context7 for docs, Complex server for data operations) with `tool_prefix` disambiguation. Example 5 runs a live demo with Context7 + custom echo tool.

4. **04_tool_combinations.py** — Full-stack agent combining `ToolRegistry` with `AgentRetryConfig`, `ToolRetryConfig`, `ContentFilterConfig`, `TokenLimitsConfig`, and `Evaluator`. Tools run through the content filter before the LLM sees the result.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **Tool not called** — The LLM may not recognize it needs the tool. Rephrase the prompt to be more explicit (e.g., "Use the calculator to...").
- **MCP server not connecting** — Ensure network egress is available (Context7 requires internet). Check firewall/proxy settings. Set `MCP_HTTP_URL` to a different MCP server if needed.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
