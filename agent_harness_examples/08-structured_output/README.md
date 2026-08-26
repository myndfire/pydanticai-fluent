# Structured Output

Constrain LLM responses to typed Pydantic models — guaranteed fields, validated types, and automatic retries.

## Overview

All examples use `.with_output(SomeModel)` to tell the agent to always return a validated Pydantic object. The LLM generates JSON, pydantic validates it against the model schema, and `result.output` is a typed instance — no manual parsing needed.

```
Validation pipeline:
────────────────────────────────────────────────────────────────────

  LLM generates        Pydantic schema         Custom validator     Typed result
  raw JSON             validation              (business rules)
  ┌──────────┐         ┌──────────────┐        ┌──────────────┐     ┌──────────────┐
  │ {"city":  │   ───▶  │ Check types: │  ───▶  │ Check rules: │───▶ │ report.city  │
  │  "Tokyo", │         │ str, int,    │        │ end > start  │     │ report.temp  │
  │  "temp":  │         │ list[str],   │        │ ge, le, etc  │     │ (typed)      │
  │  22}      │         │ Literal[...] │        │              │     │              │
  └──────────┘         └──────────────┘        └──────────────┘     └──────────────┘
                             │                       │
                             ▼                       ▼
                         FAIL: retry             FAIL: ModelRetry
                         (output_retries)       → retry with error msg
```

| Example | File | Output type | Key concept |
|---------|------|-------------|-------------|
| Simple model | `01_simple_model.py` | `WeatherReport` | `.with_output()` with typed fields |
| Enums / Literals | `02_enums_literals.py` | `SentimentResult` | `Literal["positive","negative","neutral"]` constraint |
| Nested models | `03_nested_models.py` | `Recipe` | `Recipe` → `Ingredient[]` + `InstructionStep[]` |
| Validation retries | `04_validation_retries.py` | `MeetingScheduler` | Custom `@output_validator` + `ModelRetry` |

## Files

### 01_simple_model.py

The simplest structured output example. A `WeatherReport` model with 5 typed fields. The agent returns a validated object — no string parsing required.

```
User: "What is the current weather in Tokyo?"
    │
    ▼
agent.run(prompt)
    │
    ▼
┌─────────────────────────────────────────┐
│  .with_output(WeatherReport)            │
│                                          │
│  LLM generates JSON:                     │
│  {"city": "Tokyo",                       │
│   "temperature_f": 72,                   │
│   "conditions": "sunny",                 │
│   "humidity_percent": 65,                │
│   "wind_mph": 8}                         │
│                                          │
│  Pydantic validates:                     │
│    city: str ✓                           │
│    temperature_f: int ✓                  │
│    conditions: str ✓                     │
│    humidity_percent: int ✓               │
│    wind_mph: int ✓                       │
└─────────────────────────────────────────┘
    │
    ▼
result.output → WeatherReport instance
    │
    ▼
report.city          # "Tokyo"       (str)
report.temperature_f # 72            (int)
report.conditions    # "sunny"       (str)
report.humidity_percent  # 65        (int)
report.wind_mph      # 8             (int)
```

Key components:
- `WeatherReport(BaseModel)` — 5 typed fields with `Field(description=...)`
- `.with_output(WeatherReport, output_retries=3)` — constrains LLM + retries on invalid JSON
- Two independent runs: Tokyo and London (separate `MessageHistory` sessions)
- Demo: get weather for two cities with typed access

### 02_enums_literals.py

Classification with `Literal` types — restricts the LLM to a fixed set of valid values. If the LLM returns `"happy"` instead of `"positive"`, pydantic rejects it and the agent retries. This makes classification tasks reliable and consistent.

```
User: "Analyze this review: 'Absolutely love this product!'"
    │
    ▼
agent.run(prompt)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  .with_output(SentimentResult)                           │
│                                                          │
│  LLM generates JSON:                                     │
│  {"sentiment": "positive",                               │
│   "confidence": 0.95,                                    │
│   "keywords": ["love", "perfectly", "best"],             │
│   "reasoning": "Strong positive language throughout"}     │
│                                                          │
│  Pydantic validates:                                     │
│    sentiment: Literal["positive","negative","neutral"]   │
│      ✓ "positive" matches one of the 3 allowed values   │
│                                                          │
│    ← If LLM returned "happy" → REJECTED → retry         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
result.output → SentimentResult instance
    │
    ▼
r.sentiment   # "positive" (guaranteed to be one of 3 values)
r.confidence  # 0.95       (float, 0.0–1.0)
r.keywords    # ["love", "perfectly", "best"] (list[str])
r.reasoning   # "Strong positive language..." (str)
```

Key components:
- `SentimentResult` with `Literal["positive", "negative", "neutral"]` constraint
- `confidence: float = Field(ge=0.0, le=1.0)` — range validation
- Three runs: positive, negative, and neutral reviews
- Demo: sentiment analysis with guaranteed valid labels

### 03_nested_models.py

Complex hierarchical output — a full recipe with nested `Ingredient` and `InstructionStep` sub-models. Each sub-model is independently validated. If one ingredient is missing a `unit`, pydantic catches it and the agent retries.

```
User: "Give me a recipe for chocolate chip cookies"
    │
    ▼
agent.run(prompt)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  .with_output(Recipe)                                         │
│                                                               │
│  Recipe(title="Chocolate Chip Cookies",                       │
│         prep_time_minutes=15,                                 │
│         cook_time_minutes=12,                                 │
│         servings=24,                                          │
│         ingredients=[                                         │
│           Ingredient(name="flour", amount=2.25, unit="cups"), │ ← each Ingredient
│           Ingredient(name="butter", amount=1, unit="cup"),   │   validated independently
│           Ingredient(name="sugar", amount=0.75, unit="cup"), │
│           ...                                                 │
│         ],                                                    │
│         instructions=[                                        │
│           InstructionStep(step=1, action="Preheat oven..."),  │ ← each step validated
│           InstructionStep(step=2, action="Cream butter..."),  │
│           ...                                                 │
│         ])                                                    │
│                                                               │
│  Nested validation:                                           │
│    Recipe.ingredients → list[Ingredient]                      │
│      └─ Ingredient.name: str ✓                                │
│      └─ Ingredient.amount: float ✓                            │
│      └─ Ingredient.unit: str ✓                                │
│    Recipe.instructions → list[InstructionStep]                │
│      └─ InstructionStep.step: int ✓                           │
│      └─ InstructionStep.action: str ✓                         │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
recipe.ingredients[0].name   # "flour" (nested access works)
recipe.instructions[-1].action # "Enjoy!" (typed at every level)
```

Key components:
- 3 models: `Recipe`, `Ingredient`, `InstructionStep`
- Nested `list[Ingredient]` and `list[InstructionStep]` — each sub-item validated
- `.with_output(Recipe, output_retries=3)` — retries if any sub-model fails
- Demo: chocolate chip cookie recipe with typed nested access

### 04_validation_retries.py

Custom business rules via `@output_validator`. Pydantic validates field types, but for rules like "end_time must be after start_time", use a custom validator that raises `ModelRetry`. Each `ModelRetry` triggers the agent to try again with a new response.

```
User: "Schedule a meeting from 3pm to 1pm"
    │
    ▼
agent.run(prompt)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: LLM generates JSON                                  │
│  {"title": "Marketing Meeting",                              │
│   "start_time": "15:00",                                     │
│   "end_time": "13:00",                                       │
│   "attendees": ["Alice", "Bob"]}                             │
│                                                               │
│  Step 2: Pydantic schema validation                          │
│    title: str ✓                                              │
│    start_time: str ✓                                         │
│    end_time: str ✓                                           │
│    attendees: list[str] ✓                                    │
│                                                               │
│  Step 3: Custom @output_validator                            │
│    parse start_time → 15:00                                  │
│    parse end_time   → 13:00                                  │
│    13:00 < 15:00 → FAIL                                      │
│                                                               │
│  Step 4: raise ModelRetry(                                   │
│    "End time 13:00 must be after start time 15:00")          │
│                                                               │
│  Step 5: Agent sees error → tries again                      │
│    (up to output_retries=5 times)                            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
After 5 retries → exception raised (end > start impossible)
```

Two paths:
```
Valid meeting:                           Impossible meeting:
  "1-hour review at 2pm"                   "3pm to 1pm"
       │                                        │
       ▼                                        ▼
  Pydantic ✓                              Pydantic ✓
  Validator: 15:00 > 14:00 ✓             Validator: 13:00 < 15:00 ✗
       │                                        │
       ▼                                        ▼
  Return MeetingScheduler               raise ModelRetry → retry ×5
       │                                        │
       ▼                                        ▼
  title: "Project Review"               exception raised
  start: "14:00"
  end:   "15:00"
  attendees: ["Alice","Bob","Charlie"]
```

Key components:
- `MeetingScheduler(BaseModel)` — 4 fields including `start_time`/`end_time` as `HH:MM` strings
- `@raw_agent.output_validator` — custom function that parses times and checks `end > start`
- `ModelRetry` — tells pydantic-ai to retry with the error message visible to the LLM
- `output_retries=5` — max attempts before giving up
- Two runs: valid meeting (passes) and impossible meeting (exhausts retries)

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the required model pulled

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models (first time only)
ollama pull phi4-mini                # for 01
ollama pull phi4-mini-reasoning      # for 02, 03, 04

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
| `STRUCTURED_OUTPUT_MODEL_NAME` | 01 | `phi4-mini` | LLM model for simple model example |
| `STRUCTURED_OUTPUT_REASONING_MODEL` | 02, 03, 04 | `phi4-mini-reasoning` | LLM model with reasoning for classification/nested/validation |
| `STRUCTURED_OUTPUT_MAX_TOKENS` | all | `512` | Max LLM output tokens |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |

## Running

Each file is an independent entry point:

```bash
# Simple model — typed weather report
uv run python 08-structured_output/01_simple_model.py

# Enums/Literals — constrained sentiment classification
uv run python 08-structured_output/02_enums_literals.py

# Nested models — hierarchical recipe output
uv run python 08-structured_output/03_nested_models.py

# Validation retries — business rules with ModelRetry
uv run python 08-structured_output/04_validation_retries.py
```

## Expected Output

**01_simple_model.py:** Two weather reports (Tokyo, London) printed with typed fields. Shows `isinstance(result.output, WeatherReport)` is `True`.

**02_enums_literals.py:** Three sentiment analyses (positive, negative, neutral) with confidence scores and keywords. Shows that `Literal` guarantees valid labels.

**03_nested_models.py:** One chocolate chip cookie recipe with ingredients list and step-by-step instructions. Shows nested access like `recipe.ingredients[0].name`.

**04_validation_retries.py:** First run succeeds (valid meeting). Second run fails after retries (impossible time range). Shows the ModelRetry error message.

## How It Works

1. **01_simple_model.py** — `.with_output(WeatherReport)` tells pydantic-ai to constrain the LLM to JSON matching the `WeatherReport` schema. Pydantic validates field types. `result.output` is a `WeatherReport` instance with typed access.

2. **02_enums_literals.py** — `Literal["positive", "negative", "neutral"]` restricts the `sentiment` field to exactly 3 valid values. If the LLM returns anything else, pydantic rejects it and the agent retries (up to `output_retries=3`).

3. **03_nested_models.py** — `Recipe` contains `list[Ingredient]` and `list[InstructionStep]`. Each sub-model is independently validated — a missing `unit` or non-integer `step` triggers a retry.

4. **04_validation_retries.py** — After pydantic validates types, a custom `@output_validator` function checks business rules (end time > start time). If the rule fails, it raises `ModelRetry` with a descriptive error. The LLM sees the error and tries to fix its response.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **Validation keeps retrying** — The LLM may struggle with the schema. Use a larger model (e.g., `qwen3.5:4b` instead of `phi4-mini`).
- **Malformed JSON** — The model may not support structured output well. Increase `output_retries` or use a different model.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
