# Loop Patterns

Five agent loop patterns ranging from simple conversation to structured plan-then-execute workflows.

## Overview

All examples demonstrate how to build repeatable agent cycles using `agent_harness`. They share common infrastructure (`ManagedAgent`, `InMemoryProvider`, `MessageHistory`) but differ in who drives the loop and how termination is decided.

```
Loop drivers:
────────────────────────────────────────────────────────────────────

  User-driven              Program-driven            Agent-driven
  ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
  │ 01 Interactive│         │ 02 Refinement│          │ 03 ReAct     │
  │    REPL      │         │ 04 Goal-Seek │          │              │
  │              │         │ 05 Planning  │          │              │
  └──────────────┘         └──────────────┘          └──────────────┘
       │                        │                         │
       ▼                        ▼                         ▼
  while True              for / while              agent.run()
  input() → agent         eval → feedback          (internal tool loop)
  quit to exit            max attempts             LLM decides sequence
```

| Pattern | File | Loop driver | Termination |
|---------|------|-------------|-------------|
| Interactive REPL | `01_interactive_loop.py` | User | User types quit |
| Iterative refinement | `02_iterative_refinement.py` | Python | Evaluator passes or max attempts |
| ReAct | `03_react_loop.py` | LLM (internal) | LLM stops calling tools |
| Goal-seeking | `04_goal_seeking_loop.py` | Python | Constraint satisfied or max attempts |
| Planning | `05_planning_loop.py` | Python | Plan steps exhausted |

## Files

### 01_interactive_loop.py

A conversational REPL where the agent remembers the full conversation across turns. Each iteration loads `MessageHistory` from an `InMemoryProvider`, runs the agent, and prints the response. The loop exits on `quit`, `exit`, or `bye`.

```
while True:
    │
    ▼
user_input = input("You: ")
    │
    ├──▶ "quit" / "exit" / "bye"  ──▶ break
    │
    ▼
history = MessageHistory().load(session_id, memory)
    │
    ▼
result = agent.run(user_input, history, session_id)
    │
    ▼
print(f"Agent: {result.output}")
    │
    ▼
(save turn to memory)
    │
    └──▶ loop back to input()
```

Key components:
- `ManagedAgent` with `StaticPrompts`, `Observability`, and short-term memory
- `InMemoryProvider` for ephemeral session persistence
- Graceful exit on `EOFError`/`KeyboardInterrupt`
- Demo: open-ended conversation (e.g., "What is the capital of France?" → "What is its population?")

### 02_iterative_refinement.py

An external `WordCountEvaluator` checks whether the agent's summary fits within a word limit. If not, a feedback message ("Too long — try again") becomes the next prompt. The agent sees its previous failed attempts via `MessageHistory` and self-corrects.

```
prompt = "Summarize in ≤10 words: ..."
    │
    ▼
for attempt in range(1, MAX_ATTEMPTS + 1):
    │
    ▼
result = agent.run(prompt, history, session_id)
    │
    ▼
evaluator.evaluate(result)
    │
    ├──▶ word_count <= 10  ──▶ PASS ──▶ break (success!)
    │
    └──▶ word_count > 10   ──▶ FAIL
              │
              ▼
         feedback = "Your summary was 42 words. Shorten to ≤10."
         prompt = feedback
              │
              ▼
         (loop: next attempt)
```

Key components:
- `WordCountEvaluator` (subclasses `Evaluator`) — counts words, stores pass/fail
- `ManagedAgent` with `.with_evaluators(evaluator)`
- `text_to_summarize.txt` — input text (Python language description)
- Configurable: `REFINEMENT_MAX_WORDS`, `REFINEMENT_MAX_ATTEMPTS`, `REFINEMENT_TEMPERATURE`
- Demo: summarize text in ≤10 words

### 03_react_loop.py

The classic ReAct (Reason → Act → Observe) pattern where the *agent itself* drives the loop. A single `agent.run()` call triggers pydantic-ai's internal tool loop: the LLM reasons about what it needs, calls tools, observes results, and repeats until it has a final answer.

```
agent.run("What is the average temperature of Tokyo, London, NY?")
    │
    ▼
┌─────────────────────────────────────────────┐
│  pydantic-ai internal tool loop (LLM-driven)│
│                                              │
│  LLM: "I need Tokyo's weather"              │
│      ▼                                       │
│  tool: get_weather("Tokyo")  → "22°C"       │
│      ▼                                       │
│  LLM: "Now London..."                       │
│      ▼                                       │
│  tool: get_weather("London") → "15°C"       │
│      ▼                                       │
│  LLM: "Now New York..."                     │
│      ▼                                       │
│  tool: get_weather("New York") → "18°C"     │
│      ▼                                       │
│  LLM: "Calculate average"                   │
│      ▼                                       │
│  tool: calculator("(22+15+18)/3") → "18.33" │
│      ▼                                       │
│  LLM: "Final answer: 18.33°C"               │
└─────────────────────────────────────────────┘
    │
    ▼
result.output = "The average temperature is 18.33°C"
```

Key components:
- `ToolRegistry` with `get_weather` and `calculator` tools
- `ManagedAgent` with `.with_tools(tools)`
- Agent-driven: no external loop code — the LLM decides tool sequence
- Demo: average temperature of Tokyo, London, and New York

### 04_goal_seeking_loop.py

A program-controlled loop where the agent guesses an animal and Python validates the guess against a lifespan dictionary. Feedback ("Too short!", "Too long!") is injected into the next prompt. The loop terminates when the constraint is met or the attempt budget is exhausted.

```
prompt = "Find an animal whose lifespan is between 7 and 12 years"
    │
    ▼
for attempt in range(1, MAX_ATTEMPTS + 1):
    │
    ▼
result = agent.run(prompt, history, session_id)
    │
    ▼
animal = extract_animal(result.output)  # regex: first word
    │
    ▼
success, feedback = check_lifespan(animal)
    │
    ├──▶ success=True  ──▶ break (goal achieved!)
    │
    └──▶ success=False
              │
              ├──▶ "Too short! Try longer-lived."
              ├──▶ "Too long! Try shorter-lived."
              └──▶ "Unknown animal. Try common pet."
              │
              ▼
         prompt = f"Previous: {feedback}\nTry a {direction} animal."
              │
              ▼
         (loop: next attempt)
```

Key components:
- `extract_animal()` — regex parser for agent output
- `check_lifespan()` — dictionary lookup with constraint check
- `ANIMAL_LIFESPANS` — hardcoded lookup table
- Configurable: `GOAL_LIFESPAN_MIN`, `GOAL_LIFESPAN_MAX`, `GOAL_MAX_ATTEMPTS`, `GOAL_TEMPERATURE`
- Demo: find an animal whose lifespan is between 7 and 12 years

### 05_planning_loop.py

A plan-then-execute workflow with structured output. Phase 1: the agent outputs an `AgentPlan` (Pydantic model with `PlanStep` objects). Phase 2: Python iterates the plan and calls tools directly. Phase 3: results are synthesized into a final answer.

```
Phase 1: Plan Generation
────────────────────────
planner_agent.run("Create a plan to research frameworks...")
    │
    ▼
.with_output(AgentPlan)  →  JSON constrained to Pydantic model
    │
    ▼
AgentPlan(steps=[
    PlanStep(action="search", target="Flask"),
    PlanStep(action="search", target="FastAPI"),
    PlanStep(action="search", target="Django"),
    PlanStep(action="calculate", target="ratings"),
])

Phase 2: Step Execution
───────────────────────
for step in plan.steps:
    │
    ├──▶ search_docs("flask")      → "Flask: 4.2/5"
    ├──▶ search_docs("fastapi")    → "FastAPI: 4.7/5"
    ├──▶ search_docs("django")     → "Django: 4.5/5"
    └──▶ calculate_average("4.2,4.7,4.5") → "Average: 4.47"

Phase 3: Synthesis
─────────────────
Final answer: "Average rating: 4.47/5"
```

Key components:
- `AgentPlan` and `PlanStep` — Pydantic models for structured plan output
- `.with_output(AgentPlan)` — constrains LLM to valid JSON
- Two-agent design: `planner_agent` (no tools) + `executor_agent` (with tools)
- `search_docs` and `calculate_average` tools (simulated)
- Configurable: `PLAN_MAX_STEPS`, `PLANNING_MODEL_NAME`
- Demo: research Flask, FastAPI, Django and calculate average rating

### text_to_summarize.txt

Input data for `02_iterative_refinement.py`. Contains a one-paragraph description of the Python programming language.

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the required model pulled

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models (first time only)
ollama pull qwen2.5:3b      # for 01, 02, 04
ollama pull qwen3.5:4b      # for 03
ollama pull llama3.1:8b     # for 05

# 3. Install dependencies
cd agent_harness_examples
uv sync

# 4. (Optional) Copy and edit .env
cp .env.example .env
```

## Configuration

All variables are optional and read from `.env` via `python-dotenv`.

| Variable | Files | Default | Description |
|----------|-------|---------|-------------|
| `MODEL_NAME` | 01, 02, 04 | `qwen2.5:3b` | LLM model name |
| `REACT_MODEL_NAME` | 03 | `qwen3.5:4b` | LLM model for ReAct |
| `PLANNING_MODEL_NAME` | 05 | `llama3.1:8b` | LLM model for planning |
| `MAX_TOKENS` | all | 128–512 | Max LLM output tokens |
| `OLLAMA_BASE_URL` | all | `http://localhost:11434/v1` | Ollama endpoint |
| `REFINEMENT_MAX_WORDS` | 02 | `10` | Word count limit for summary |
| `REFINEMENT_MAX_ATTEMPTS` | 02 | `3` | Max retry attempts |
| `REFINEMENT_TEMPERATURE` | 02 | `0.7` | LLM temperature |
| `GOAL_LIFESPAN_MIN` | 04 | `7` | Min animal lifespan (years) |
| `GOAL_LIFESPAN_MAX` | 04 | `12` | Max animal lifespan (years) |
| `GOAL_MAX_ATTEMPTS` | 04 | `5` | Max retry attempts |
| `GOAL_TEMPERATURE` | 04 | `0.7` | LLM temperature |
| `PLAN_MAX_STEPS` | 05 | `6` | Max plan steps (safety cap) |

## Running

Each file is an independent entry point:

```bash
# Interactive REPL
uv run python 6-loops/01_interactive_loop.py

# Iterative refinement (word count constraint)
uv run python 6-loops/02_iterative_refinement.py

# ReAct loop (agent-driven tool calls)
uv run python 6-loops/03_react_loop.py

# Goal-seeking loop (constraint satisfaction)
uv run python 6-loops/04_goal_seeking_loop.py

# Planning loop (plan → execute → synthesize)
uv run python 6-loops/05_planning_loop.py
```

## Expected Output

**01_interactive_loop.py:** Open-ended conversation. Prints a banner, accepts user input, and responds. Shows memory state each turn.

**02_iterative_refinement.py:** 1–3 attempts to summarize text. Each attempt shows word count and pass/fail. Ends with success or "gave up" message.

**03_react_loop.py:** Single run with internal tool calls printed (3 weather lookups + 1 calculation). Final answer: average temperature.

**04_goal_seeking_loop.py:** 1–5 attempts to find an animal in the target lifespan range. Each attempt shows the guess and feedback ("Too short!", "Too long!").

**05_planning_loop.py:** Three phases — plan creation (JSON), step-by-step execution (4 steps), and final synthesis with framework ratings and average.

## How It Works

1. **01_interactive_loop.py** — Classic `while True` loop. Each iteration loads conversation history, runs the agent, saves the turn, and prints the response. The `InMemoryProvider` retains context across turns.

2. **02_iterative_refinement.py** — Python controls the loop. After each `agent.run()`, the `WordCountEvaluator` checks word count. If it fails, a feedback prompt replaces the original prompt. The agent sees previous attempts via `MessageHistory`.

3. **03_react_loop.py** — The LLM drives the loop internally. A single `agent.run()` triggers pydantic-ai's built-in tool loop: the LLM decides which tool to call, observes the result, and repeats until it has a final answer. No external loop code needed.

4. **04_goal_seeking_loop.py** — Python controls the loop. After each `agent.run()`, the output is parsed (regex), validated (dictionary lookup), and feedback is injected as the next prompt. The agent converges on a valid answer through iteration.

5. **05_planning_loop.py** — Two-phase design. Phase 1: a planner agent (no tools) produces a structured `AgentPlan` via `.with_output(AgentPlan)`. Phase 2: Python iterates the plan and calls tools directly. Phase 3: results are synthesized into a final answer.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the required model (see Setup section).
- **Malformed JSON (05)** — The planner model may not support structured output. Use `llama3.1:8b` or larger.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
- **Empty `.env`** — All variables have defaults. A missing `.env` file is fine.
