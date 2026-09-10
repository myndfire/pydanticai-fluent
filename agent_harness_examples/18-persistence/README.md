# 18 — Step Persistence

Append-only step log + continuable snapshots + tool-effect ledger, via the
fluent API. All tuning comes from the environment — no hardcoded thresholds
in code.

| Example | What it shows |
|---|---|
| `01_memory_steps.py` | `with_memory_steps(...)`, `list_runs`/`list_events`, `continue_run` |
| `02_file_steps.py` | `with_file_steps(...)`, `list_unresolved_tool_effects`, `fork_run` |
| `03_conversation_grouping.py` | `conversation_id` defaults to `session_id`; `with_step_persistence_from_env()` |

```bash
uv run python 18-persistence/01_memory_steps.py
```

Model comes from shared `MODEL_NAME` / `LLM_PROVIDER`. Step-persistence keys
(`PERSISTENCE_*` per-example, `HARNESS_PERSISTENCE_*` shared) are listed in
`.env.example` under `18-persistence`. MongoDB (`with_mongo_steps`) needs a
reachable MongoDB — see `HARNESS_PERSISTENCE_MONGO_URL`.
