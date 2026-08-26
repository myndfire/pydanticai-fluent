# Prompts

Shape agent personality with static system prompts, MongoDB-backed Jinja2 templates, and dynamic variable injection.

## Overview

All examples use the `prompts` module to control the system prompt the LLM sees. `StaticPrompts` is a fixed string — simple and direct. `MongoPrompts` stores Jinja2 templates in MongoDB, supporting dynamic variables, versioning, and caching.

```
Prompt providers:
────────────────────────────────────────────────────────────────────

  StaticPrompts                   MongoPrompts
  ┌──────────────────┐           ┌──────────────────┐
  │ Fixed string     │           │ MongoDB           │
  │ "You are a ..."  │           │   └─ Jinja2       │
  │                  │           │       templates    │
  │ No variables     │           │       {{role}}     │
  │ No rendering     │           │       {{domain}}   │
  │ Instant setup    │           │       {% if %}     │
  └──────────────────┘           │       {% for %}    │
                                 │                   │
                                 │ Caching, versioning│
                                 │ CRUD operations    │
                                 └──────────────────┘
                                         │
                                    kwargs flow
                                    ┌─────┴─────┐
                                    │ prompt_id  │ → selects template
                                    │ role="..." │ → {{role}}
                                    │ city="..." │ → {{city}}
                                    └───────────┘
```

| Example | File | Key concept |
|---------|------|-------------|
| Static prompts | `01_static_prompts.py` | `StaticPrompts("You are a ...")` — fixed personality |
| Mongo prompts | `02_mongo_prompts.py` | `MongoPrompts` — Jinja2 templates in MongoDB |
| Prompt variables | `03_prompt_variables.py` | `prompt_id` switching + kwargs as template vars |

## Files

### 01_static_prompts.py

Fixed system prompts to shape agent personality. `StaticPrompts` wraps a string that becomes the system message. No variables, no rendering — just a static instruction the LLM follows.

```
StaticPrompts("You are a French chef...")
    │
    ▼
agent = ManagedAgent()
    .with_model(ModelConfig(...))
    .with_prompts(chef_prompt)
    │
    ▼
agent.run("What is the meaning of life?")
    │
    ▼
┌─────────────────────────────────────────────┐
│  System prompt:                              │
│  "You are a world-renowned French chef..."  │
│                                              │
│  User prompt:                                │
│  "What is the meaning of life?"              │
│                                              │
│  LLM response (in character):                │
│  "Ze meaning of life, like a perfect         │
│   soufflé, requires patience and balance..." │
│   Bon appetit!"                              │
└─────────────────────────────────────────────┘
```

Five sub-examples:
```
Example 1: Default         Example 2: French Chef     Example 3: Shakespeare
  "You are a                "You are a world-           "Thou art William
   helpful assistant"        renowned French chef"       Shakespeare himself"
       │                         │                          │
       ▼                         ▼                          ▼
  Neutral response          Culinary metaphors          Iambic pentameter,
  Standard tone             French terms                Early Modern English

Example 4: Comparison       Example 5: DBA
  Side-by-side output         "You are a terse database
  of all 3 personalities      administrator. Bullet points
       │                      only. No pleasantries."
       ▼                           │
  Shows personality              ▼
  differences                   Short, direct answers
```

Key components:
- `StaticPrompts("...")` — fixed system prompt string
- `ManagedAgent().with_prompts(prompts)` — attaches prompts to agent
- Default prompt (no `with_prompts()`) — "You are a helpful assistant"
- `.with_model().with_prompts()` — fluent builder pipeline
- Demo: default, French chef, Shakespeare, DBA personalities

### 02_mongo_prompts.py

MongoDB-backed Jinja2 templates with CRUD, caching, and dynamic variables. Templates are stored in MongoDB, rendered with variables via `get_system_prompt()`, and cached for performance.

```
MongoDB schema:
{
    "_id": "prompt_id",
    "template": "You are a {{role}} specialized in {{domain}}...",
    "active": true,
    "version": 1,
    "metadata": {"tags": [...]}
}

CRUD operations:
────────────────────────────────────────────────────────────────────

  create_prompt("doctor", template, version, metadata)
       │
       ▼
  MongoDB ← stores document
       │
  get_system_prompt("doctor", specialty="cardiology", language="simple")
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │  Jinja2 rendering:                                    │
  │  Template: "You are a {{specialty}} medical..."       │
  │  Variables: {specialty: "cardiology", language: "simple"} │
  │  Output:   "You are a cardiology medical..."         │
  │            "Use plain, non-technical language."       │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  Cache ← stores rendered prompt
       │
  update_prompt("doctor", template=...)
       │
       ▼
  Cache invalidated → next render re-fetches from MongoDB
```

Template features:
```
Jinja2 syntax supported:
────────────────────────────────────────────────────────────────────

  Variables:       {{role}}, {{domain}}
  Conditionals:    {% if language == 'simple' %}...{% endif %}
  Loops:           {% for rule in rules %}- {{rule}}{% endfor %}
  Complex objects: {{config.host}}:{{config.port}}

Three seeded templates:
  "doctor"  → {{specialty}}, {{language}}    + conditional
  "coder"   → {{language}}, {{years}}, {{rules}}  + for loop
  "poet"    → {{style}}, {{meter}}, {{tone}}  + updated with {{mood}}
```

Agent integration:
```
agent.run(
    "Write a Fibonacci function",
    prompt_id="coder",      ← selects template
    language="Python",      ← {{language}}
    years=5,                ← {{years}}
    rules=["Use type hints", "Add error handling"],  ← {% for rule in rules %}
)
    │
    ▼
MongoPrompts.get_system_prompt("coder", language="Python", years=5, rules=[...])
    │
    ▼
Rendered: "You are an expert Python programmer with 5 years...
           - Use type hints
           - Add error handling"
    │
    ▼
LLM responds with Python code following the rendered prompt
```

Key components:
- `MongoPrompts(uri, database, collection)` — MongoDB-backed prompt store
- `.create_prompt(prompt_id, template, version, metadata)` — seed templates
- `.get_system_prompt(prompt_id, **kwargs)` — render Jinja2 template with variables
- `.update_prompt(prompt_id, template=...)` — update template (invalidates cache)
- `.list_prompts(active_only=True)` — browse available prompts
- `.clear_cache()` — force cache refresh
- `agent.run(..., prompt_id="coder", language="Python")` — kwargs flow to template
- Prerequisite: MongoDB running (`docker compose -f docker-compose.yml up -d mongo`)

### 03_prompt_variables.py

How `prompt_id` and kwargs flow through `run()` to template rendering. Demonstrates switching prompts mid-session, reserved kwargs, and how `StaticPrompts` gracefully ignores extra variables.

```
run(prompt, history, session_id, **kwargs)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  kwargs processing:                                   │
│                                                       │
│  prompt_id="coder"  ← consumed by run()              │
│  _internal="hidden" ← underscore-prefixed, filtered  │
│  role="engineer"    ← passed as template variable     │
│  city="Tokyo"       ← passed as template variable     │
│                                                       │
│  Reserved kwargs (consumed, not templated):           │
│    prompt_id   — selects the prompt template          │
│    _prefix     — underscore-prefixed keys             │
│                                                       │
│  Template kwargs (passed to Jinja2):                  │
│    role, city, language, etc.                         │
└──────────────────────────────────────────────────────┘
```

Multi-turn prompt switching:
```
Turn 1: prompt_id="formal"
    → System: "You speak in formal, professional tone"
    → "Introduce yourself." → Formal response

Turn 2: prompt_id="casual"
    → System: "You speak in casual, friendly tone"
    → "Explain quantum computing." → Casual response

Turn 3: prompt_id=default
    → System: "You are a helpful assistant."
    → "What did I just ask?" → Context-aware response
```

StaticPrompts vs MongoPrompts variable handling:
```
StaticPrompts("You are a helpful assistant.")
    │
    ▼
agent.run("Say hello.", role="doctor", domain="cardiology")
    │
    ▼
StaticPrompts ignores role, domain — no error
System prompt: "You are a helpful assistant." (unchanged)

MongoPrompts
    │
    ▼
agent.run("Say hello.", prompt_id="doctor", specialty="cardiology")
    │
    ▼
MongoPrompts renders: "You are a cardiology medical professional..."
Variables consumed by Jinja2 template
```

Key components:
- `prompt_id` kwarg — selects which prompt template to use
- Default: no `prompt_id` → uses "default" template
- Multi-turn switching: different `prompt_id` each turn
- `StaticPrompts` ignores extra kwargs (no errors)
- `MongoPrompts` renders kwargs as Jinja2 variables
- Reserved kwargs: `prompt_id`, `_prefix` keys (filtered out)

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the required model pulled
- MongoDB (for `02_mongo_prompts.py` only)

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model (first time only)
ollama pull gpt-oss:20b

# 3. (Optional) Start MongoDB (only for 02_mongo_prompts.py)
docker compose -f docker-compose.yml up -d mongo

# 4. Install dependencies
cd agent_harness_examples
uv sync

# 5. (Optional) Copy and edit .env
cp .env.example .env
```

## Configuration

All variables are optional and read from `.env` via `python-dotenv`.

| Variable | File(s) | Default | Description |
|----------|---------|---------|-------------|
| `PROMPTS_MODEL_NAME` | all | `gpt-oss:20b` | LLM model name |
| `PROMPTS_MAX_TOKENS` | all | `512` | Max LLM output tokens |
| `MONGODB_URI` | 02 | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | 02 | `agent_prompts` | MongoDB database name |
| `MONGODB_COLLECTION` | 02 | `prompts` | MongoDB collection name |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |

## Running

Each file is an independent entry point:

```bash
# Static prompts — fixed personality system prompts
uv run python 10-prompts/01_static_prompts.py

# Mongo prompts — MongoDB-backed Jinja2 templates
uv run python 10-prompts/02_mongo_prompts.py

# Prompt variables — prompt_id switching & kwargs flow
uv run python 10-prompts/03_prompt_variables.py
```

## Expected Output

**01_static_prompts.py:** Five sub-examples showing default, French chef, Shakespeare, and DBA personalities. Side-by-side comparison of responses.

**02_mongo_prompts.py:** Seeds 3 templates (doctor, coder, poet), renders them with variables, demonstrates update/cache invalidation, and runs an agent with the coder template. Requires MongoDB.

**03_prompt_variables.py:** Six examples showing default prompt_id, mid-session switching, variable flow, structured data as vars, StaticPrompts ignoring extra kwargs, and reserved kwargs.

## How It Works

1. **01_static_prompts.py** — `StaticPrompts("...")` wraps a fixed string as the system prompt. `ManagedAgent().with_prompts(prompt)` attaches it. The LLM sees the prompt as its system message, shaping tone and behavior.

2. **02_mongo_prompts.py** — `MongoPrompts(uri, db, collection)` stores Jinja2 templates in MongoDB. `get_system_prompt(prompt_id, **kwargs)` renders templates with variables. `update_prompt()` invalidates cache. `agent.run(..., prompt_id="coder", language="Python")` flows kwargs to template rendering.

3. **03_prompt_variables.py** — `run()` pops `prompt_id` from kwargs (selects template), filters underscore-prefixed keys, and passes remaining kwargs as template variables. `StaticPrompts` ignores extra kwargs gracefully. `MongoPrompts` renders them via Jinja2.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **MongoDB not reachable** — Start MongoDB: `docker compose -f docker-compose.yml up -d mongo`.
- **Template rendering error** — Check Jinja2 syntax in the template. Missing variables cause `UndefinedError`.
- **Cache stale** — Call `prompts.clear_cache()` or `update_prompt()` to invalidate.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
