# Memory

Store and retrieve conversation history — from ephemeral in-memory caches to persistent MongoDB, Redis, and Elasticsearch backends.

## Overview

All examples use `MemoryProvider` implementations to persist `TurnData` after each `agent.run()`. The agent loads prior context from memory before each turn, enabling multi-turn conversations. Memory can be split across providers: short-term (fast, bounded) and long-term (durable, full archive).

```
Memory architecture:
────────────────────────────────────────────────────────────────────

  agent.run(prompt, history, session_id, save_to=[provider1, provider2])
       │                                          │
       │                                          ▼
       │                              ┌──────────────────────┐
       │                              │  TurnData persisted   │
       │                              │  to each provider     │
       │                              └──────────────────────┘
       ▼
  MessageHistory().load(session_id, provider)
       │
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Prior turns loaded from provider                            │
  │  → reconstructed as ModelRequest/ModelResponse messages      │
  │  → injected into LLM context for next turn                   │
  └──────────────────────────────────────────────────────────────┘
```

| Provider | File | Backend | Use case |
|----------|------|---------|----------|
| `InMemoryProvider` | `01_in_memory.py` | Python dict | Dev, testing, short-term cache |
| `MessageHistory` | `02_message_history.py` | Any provider | Context loading & inspection |
| Multi-provider | `03_multi_provider.py` | 3× InMemory | Short + long + audit split |
| CRUD operations | `04_memory_operations.py` | InMemoryProvider | get, delete, clear, limit |
| `MongoMemory` | `05_mongo_memory.py` | MongoDB | Durable long-term archival |
| `RedisMemory` | `06_redis_memory.py` | Redis | Fast short-term with TTL |
| Combined | `07_combined_memory.py` | Redis + Mongo | Fast reads + durable archive |
| `ElasticsearchMemory` | `08_elasticsearch_memory.py` | Elasticsearch | Full-text search over history |
| Reasoning traces | `09_reasoning_traces.py` | InMemoryProvider | Inspect hidden thinking parts |

## Files

### 01_in_memory.py

Ephemeral `InMemoryProvider` for short-term and long-term memory. Demonstrates `save_to` for multi-provider persistence, `max_turns` for automatic trimming, and session isolation.

```
agent = ManagedAgent()
    .with_short_term_memory(InMemoryProvider(max_turns=10))
    .with_long_term_memory(InMemoryProvider(max_turns=100))
    │
    ▼
agent.run("My name is Alice", history, "session-42",
          save_to=[short_term, long_term])
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  TurnData saved to BOTH providers:                            │
│    short_term["session-42"] → [turn1]   (max 10 turns)       │
│    long_term["session-42"]  → [turn1]   (max 100 turns)      │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
agent.run("What is my name?", history, "session-42", save_to=[...])
    │
    ▼
History loaded from short_term → prior turn injected into LLM context
    │
    ▼
Response: "Your name is Alice."
```

Key components:
- `InMemoryProvider(max_turns=N)` — bounded ephemeral storage
- `save_to=[provider1, provider2]` — persist to multiple providers
- `agent.last_turn` — inspect most recent `TurnData` (status, model, usage, duration)
- `max_turns` trimming — oldest turns auto-evicted when limit exceeded
- Session isolation — each `session_id` has its own turn list

### 02_message_history.py

`MessageHistory` loads and reconstructs conversation context from any provider. Shows how prior turns become `ModelRequest`/`ModelResponse` objects the LLM can use.

```
MessageHistory().load("session-id", provider)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Load turns from provider:                                    │
│    TurnData[0] → ModelRequest(parts=[UserPrompt("My name..."))│
│    TurnData[1] → ModelResponse(parts=[TextPart("Your name...")│
│    TurnData[2] → ModelRequest(parts=[UserPrompt("What is..."))│
│    TurnData[3] → ModelResponse(parts=[TextPart("Alice")])    │
│                                                               │
│  Result: MessageHistory with .messages = [Request, Response,  │
│          Request, Response]                                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
agent.run(prompt, history, session_id)
    │
    ▼
LLM sees full conversation context → context-aware response
```

Key components:
- `MessageHistory().load(session_id, provider)` — reconstruct messages from turns
- `filter_thinking_parts(messages)` — serialize messages without internal thinking parts
- Multi-turn: each `agent.run()` loads prior context, appends new turn
- Fresh session: `load("new-id", provider)` returns empty history

### 03_multi_provider.py

Split persistence across three providers: short-term (fast context), long-term (full archive), and audit log (compliance). Shows `TurnData` inspection with `to_dict()`/`from_dict()` round-trip.

```
save_to=[short_term, long_term, audit_log]
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  short_term (max_turns=10)     ← fast context retrieval      │
│    [turn1, turn2, turn3]                                    │
│                                                               │
│  long_term (max_turns=1000)    ← full archive                │
│    [turn1, turn2, turn3]                                    │
│                                                               │
│  audit_log (max_turns=10000)   ← compliance trail            │
│    [turn1, turn2, turn3]                                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
TurnData inspection:
  turn_id:    "abc-123..."
  status:     "success"
  model:      "gpt-oss:20b"
  duration:   1.23s
  messages:   4
  usage:      in=150 out=42 total=192
  timestamp:  2025-01-15T10:30:00Z
    │
    ▼
to_dict() → JSON-serializable dict → from_dict() → round-trip OK
```

Key components:
- Three providers: `short_term`, `long_term`, `audit_log`
- `TurnData` inspection: `turn_id`, `status`, `model`, `duration_seconds`, `usage`
- `UsageData`: `input_tokens`, `output_tokens`, `total_tokens`
- `to_dict()` / `from_dict()` — serialization round-trip

### 04_memory_operations.py

CRUD operations on `InMemoryProvider`: load with limit, get by ID, delete, clear, and session isolation.

```
┌──────────────────────────────────────────────────────────────┐
│  CRUD operations:                                            │
│                                                               │
│  load_turns(session_id, limit=N)                             │
│    → returns N most recent turns                             │
│                                                               │
│  get_turn(session_id, turn_id)                               │
│    → returns specific TurnData or None                       │
│                                                               │
│  delete_turn(session_id, turn_id)                            │
│    → removes single turn, returns bool                      │
│                                                               │
│  clear(session_id)                                           │
│    → wipes ALL turns for a session                           │
│    → other sessions unaffected                               │
│                                                               │
│  agent.last_turn                                             │
│    → direct access to most recent TurnData                   │
└──────────────────────────────────────────────────────────────┘
```

Key components:
- `load_turns(session_id, limit=N)` — retrieve N most recent turns
- `get_turn(session_id, turn_id)` — fetch specific turn by ID
- `delete_turn(session_id, turn_id)` — remove single turn
- `clear(session_id)` — wipe all turns for session (isolated)
- `agent.last_turn` — direct access to most recent `TurnData`

### 05_mongo_memory.py

`MongoMemory` persists turns to MongoDB. Shows connection check, multi-turn conversation, context restoration from a new agent instance, and CRUD operations.

```
MongoMemory(uri, database, collection)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  MongoDB document per turn:                                   │
│  {                                                           │
│    "_id": ObjectId("..."),                                   │
│    "session_id": "mongo-demo",                               │
│    "turn_id": "abc-123...",                                  │
│    "timestamp": ISODate("2025-01-15T10:30:00Z"),            │
│    "turn_data": {                                            │
│      "messages": [...],                                      │
│      "status": "success",                                    │
│      "model": "gpt-oss:20b",                                │
│      "usage": {"input_tokens": 150, "output_tokens": 42}    │
│    }                                                         │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Context restoration:
  new_agent = ManagedAgent()
  history = await MessageHistory().load("mongo-demo", mongo)
  result = await new_agent.run("What is my name?", history, "mongo-demo")
  → Response: "Your name is Alice." (loaded from MongoDB)
```

Key components:
- `MongoMemory(uri, database, collection)` — MongoDB-backed persistent storage
- Connection check with graceful fallback
- Context restoration: new agent loads from MongoDB
- CRUD: `load_turns`, `get_turn`, `delete_turn`, `clear`
- Prerequisite: `docker compose -f docker-compose.yml up -d mongo`

### 06_redis_memory.py

`RedisMemory` persists turns to Redis with key prefix isolation. Shows connection check, multi-turn conversation, key prefix inspection, and CRUD operations.

```
RedisMemory(host, port, key_prefix)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Redis key structure:                                         │
│                                                               │
│  key_prefix + session_id = Redis list                        │
│  "agent:memory:redis-demo" → [turn1, turn2, turn3]          │
│                                                               │
│  Each session gets its own Redis list, isolated by prefix.   │
│  "agent:memory:other-session" → []  (empty, different session)│
└──────────────────────────────────────────────────────────────┘
    │
    ▼
save_to=[short_term, redis_mem]
    │
    ▼
Turns stored in Redis as serialized TurnData in a list.
Key prefix ensures no collisions between different agents/apps.
```

Key components:
- `RedisMemory(host, port, key_prefix)` — Redis-backed persistent storage
- Key prefix isolation: `agent:memory:<session_id>`
- Session isolation: different sessions → different Redis lists
- CRUD: `load_turns`, `get_turn`, `delete_turn`, `clear`
- Prerequisite: `docker compose -f docker-compose.yml up -d redis`

### 07_combined_memory.py

Redis for fast short-term reads + MongoDB for durable long-term archival. Shows context union (load from BOTH providers), cross-provider CRUD, and provider comparison.

```
agent = ManagedAgent()
    .with_short_term_memory(RedisMemory(...))
    .with_long_term_memory(MongoMemory(...))
    │
    ▼
agent.run(prompt, history, session, save_to=[redis_mem, mongo])
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Context union: agent loads from BOTH providers              │
│                                                               │
│  Redis:   [turn1, turn2, turn3]  ← fast reads               │
│  MongoDB: [turn1, turn2, turn3]  ← durable archive           │
│                                                               │
│  Combined: [turn1, turn2, turn3]  ← full context             │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Cross-provider CRUD:
  delete from Redis → MongoDB still has it
  Load from MongoDB alone → proves Redis not required for persistence
```

Key components:
- `RedisMemory` for short-term (fast reads)
- `MongoMemory` for long-term (durable archive)
- Context union: `MessageHistory().load()` merges from both providers
- Cross-provider CRUD: delete from one, verify the other
- Prerequisite: `docker compose -f docker-compose.yml up -d mongo redis`

### 08_elasticsearch_memory.py

`ElasticsearchMemory` persists turns to Elasticsearch with auto-index creation and full-text search capabilities. Shows document ID format, index mappings, and CRUD operations.

```
ElasticsearchMemory(endpoint, index)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Auto-created index with mappings:                            │
│    session_id: keyword                                       │
│    turn_id:    keyword                                       │
│    timestamp:  date                                          │
│    turn_data:  object                                        │
│                                                               │
│  Document ID format: {session_id}:{turn_id}                  │
│  Example: "es-demo:abc-123-def-456"                          │
│                                                               │
│  load_turns() → sorted by timestamp asc                      │
│  Full-text search over turn_data possible                     │
└──────────────────────────────────────────────────────────────┘
```

Key components:
- `ElasticsearchMemory(endpoint, index)` — Elasticsearch-backed persistent storage
- Auto-index creation with session_id, turn_id, timestamp mappings
- Document ID: `{session_id}:{turn_id}`
- `load_turns()` returns turns sorted by timestamp ascending
- Prerequisite: `docker compose -f docker-compose.yml up -d elasticsearch`

### 09_reasoning_traces.py

Inspect the model's hidden reasoning — `ThinkingPart` (chain-of-thought) vs `TextPart` (visible answer). Shows how `agent_harness` strips thinking parts from stored turns.

```
agent.run(prompt, model_settings={"thinking": True})
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  MODEL RESPONSE (one turn):                                   │
│                                                               │
│  🧠 ThinkingPart   ← hidden reasoning                        │
│     "Let me think about Seattle's climate..."                │
│                                                               │
│  📝 TextPart       ← visible answer                          │
│     "Yes, you should bring an umbrella."                     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
result.new_messages → contains BOTH ThinkingPart and TextPart
    │
    ▼
result.output → TextPart only (thinking stripped)
TurnData.messages → TextPart only (thinking stripped)
    │
    ▼
filter_thinking_parts(messages) → serialized without thinking parts
```

Key components:
- `model_settings={"thinking": True}` — enables hidden reasoning
- `ThinkingPart` — internal chain-of-thought (hidden from user)
- `TextPart` — visible response
- `result.new_messages` — raw messages with both types
- `result.output` — clean text only (thinking stripped)
- `filter_thinking_parts()` — serialize without thinking parts
- `AgentRetryConfig().with_timeout(60)` — prevents hangs on unsupported models

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the required model pulled
- MongoDB (for `05`, `07`)
- Redis (for `06`, `07`)
- Elasticsearch (for `08`)

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models (first time only)
ollama pull gpt-oss:20b           # for 01–08
ollama pull phi4-mini-reasoning   # for 09 (reasoning traces)

# 3. (Optional) Start services
docker compose -f docker-compose.yml up -d mongo redis elasticsearch

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
| `MEMORY_MODEL_NAME` | 01–08 | `gpt-oss:20b` | LLM model name |
| `REASONING_MODEL_NAME` | 09 | `phi4-mini-reasoning` | LLM model for reasoning traces |
| `MEMORY_MAX_TOKENS` | all | `512` | Max LLM output tokens |
| `MONGODB_URI` | 05, 07 | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | 05, 07 | `agent_memory` | MongoDB database name |
| `MONGODB_COLLECTION` | 05, 07 | `conversations` | MongoDB collection name |
| `REDIS_HOST` | 06, 07 | `localhost` | Redis host |
| `REDIS_PORT` | 06, 07 | `6379` | Redis port |
| `REDIS_KEY_PREFIX` | 06, 07 | `agent:memory:` | Redis key prefix |
| `ELASTICSEARCH_ENDPOINT` | 08 | `http://localhost:9200` | Elasticsearch endpoint |
| `ELASTICSEARCH_INDEX` | 08 | `agent-memory` | Elasticsearch index name |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |

## Running

Each file is an independent entry point:

```bash
# InMemoryProvider — short/long term, max_turns, session isolation
uv run python 11-memory/01_in_memory.py

# MessageHistory — context loading, filtering, multi-turn
uv run python 11-memory/02_message_history.py

# Multi-provider — short + long + audit split
uv run python 11-memory/03_multi_provider.py

# CRUD operations — get, delete, clear, limit
uv run python 11-memory/04_memory_operations.py

# MongoMemory — MongoDB persistent memory
uv run python 11-memory/05_mongo_memory.py

# RedisMemory — Redis persistent memory
uv run python 11-memory/06_redis_memory.py

# Combined — Redis (short) + MongoDB (long)
uv run python 11-memory/07_combined_memory.py

# ElasticsearchMemory — full-text search memory
uv run python 11-memory/08_elasticsearch_memory.py

# Reasoning traces — inspect hidden thinking parts
uv run python 11-memory/09_reasoning_traces.py
```

## Expected Output

**01_in_memory.py:** 3-turn conversation with short/long term providers, max_turns trimming demo, session isolation proof.

**02_message_history.py:** 4-turn conversation, message inspection (Request/Response), filter_thinking_parts demo, fresh session.

**03_multi_provider.py:** 3-turn conversation saved to 3 providers, TurnData inspection, storage comparison, to_dict/from_dict round-trip.

**04_memory_operations.py:** 5 turns populated, then CRUD demos: load with limit, get by ID, delete, clear, session isolation.

**05_mongo_memory.py:** 3-turn conversation persisted to MongoDB, context restoration from new agent, CRUD operations. Requires MongoDB.

**06_redis_memory.py:** 3-turn conversation persisted to Redis, key prefix inspection, CRUD operations. Requires Redis.

**07_combined_memory.py:** 4-turn conversation saved to both Redis and MongoDB, context union verification, cross-provider CRUD. Requires both.

**08_elasticsearch_memory.py:** 4-turn conversation persisted to Elasticsearch, context restoration, CRUD operations. Requires Elasticsearch.

**09_reasoning_traces.py:** Single turn with `thinking=True`, shows hidden reasoning vs visible response, summary stats.

## How It Works

1. **01_in_memory.py** — `InMemoryProvider` stores `TurnData` in a Python dict keyed by `session_id`. `max_turns` evicts oldest turns. `save_to=[...]` persists to multiple providers simultaneously.

2. **02_message_history.py** — `MessageHistory().load(session_id, provider)` fetches turns and reconstructs `ModelRequest`/`ModelResponse` objects. These become the conversation context the LLM sees.

3. **03_multi_provider.py** — Three `InMemoryProvider` instances serve different purposes. `TurnData` includes `turn_id`, `status`, `model`, `usage`, `duration`, `timestamp`. `to_dict()`/`from_dict()` enable serialization.

4. **04_memory_operations.py** — CRUD: `load_turns(limit=N)` for recent turns, `get_turn(turn_id)` for specific turns, `delete_turn(turn_id)` to remove, `clear(session_id)` to wipe. `clear()` is session-isolated.

5. **05_mongo_memory.py** — `MongoMemory` stores turns as MongoDB documents. Connection check with graceful fallback. Context restoration: a new agent loads history from MongoDB independently.

6. **06_redis_memory.py` — `RedisMemory` stores turns in Redis lists keyed by `key_prefix + session_id`. Key prefix ensures no collisions. Fast reads for short-term context.

7. **07_combined_memory.py** — Redis for fast short-term reads, MongoDB for durable long-term archival. `MessageHistory().load()` merges context from both providers. Cross-provider CRUD shows independence.

8. **08_elasticsearch_memory.py** — `ElasticsearchMemory` auto-creates an index with session_id, turn_id, timestamp mappings. Document ID: `{session_id}:{turn_id}`. `load_turns()` returns sorted by timestamp.

9. **09_reasoning_traces.py** — `model_settings={"thinking": True}` enables hidden `ThinkingPart` in responses. `agent_harness` strips these from `result.output` and stored turns, but `result.new_messages` preserves them.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **MongoDB not reachable** — Start MongoDB: `docker compose -f docker-compose.yml up -d mongo`.
- **Redis not reachable** — Start Redis: `docker compose -f docker-compose.yml up -d redis`.
- **Elasticsearch not reachable** — Start Elasticsearch: `docker compose -f docker-compose.yml up -d elasticsearch`.
- **Thinking hangs (09)** — Model doesn't support `thinking=True`. Use `phi4-mini-reasoning` or `qwen2.5`.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
