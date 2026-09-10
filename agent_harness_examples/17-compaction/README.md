# 17 — Compaction

Model-agnostic context management via the fluent API. All tuning comes from
the environment — no hardcoded thresholds in code.

| Example | What it shows |
|---|---|
| `01_clear_tool_results.py` | `with_clear_tool_results(...)` — blank old tool results, keep recent pairs |
| `02_sliding_window.py` | `with_sliding_window(...)` — keep only the recent tail |
| `03_warn_and_report.py` | `with_warn_near_limits(...)` + `with_report_context_usage(...)` — observe, don't rewrite |
| `04_tiered_from_env.py` | `with_tiered_compaction_from_env(tiers)` — recommended default, budget from env |

```bash
uv run python 17-compaction/01_clear_tool_results.py
```

Model comes from shared `MODEL_NAME` / `LLM_PROVIDER`. Per-example keys
(`COMPACTION_*`) and shared budget keys (`HARNESS_COMPACTION_*`) are listed
in `.env.example` under `17-compaction`.
