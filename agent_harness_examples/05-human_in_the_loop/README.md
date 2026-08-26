# Human-in-the-Loop Examples

Three patterns for inserting human review into an `agent_harness` pipeline, ranging from simple end-of-response approval to per-tool-step intervention.

## Overview

Human-in-the-loop (HITL) lets a person review, approve, or modify agent output before it is returned. These examples use the existing `ContentFilterConfig.on_filter` callback and inline `input()` prompts inside tools to implement HITL without any new framework code.

```
Pattern 1: End Review         Pattern 2: Tool Review        Pattern 3: Per-Step Review
┌──────────┐                  ┌──────────┐                  ┌──────────┐
│  Agent   │                  │  Agent   │                  │  Agent   │
│  (LLM)   │                  │  (LLM)   │                  │  (LLM)   │
└────┬─────┘                  └────┬─────┘                  └────┬─────┘
     │                             │                             │
     ▼                             ▼                             ▼
 output                        tool call                     tool call 1
     │                        (mortgage)                   (flight price)
     ▼                             │                             │
 [A]pprove                      output                        [Y]es?
 [M]odify                          │                             │
     │                             ▼                             ▼
 final                        [A]pprove                     tool call 2
                              [M]odify                     (calculate total)
                                  │                             │
                                  ▼                             ▼
                              final                         [Y]es?
                                                              │
                                                              ▼
                                                          tool call 3
                                                          (discount)
                                                              │
                                                              ▼
                                                          final
```

**Patterns covered:**

| Pattern | File | When human reviews |
|---------|------|--------------------|
| End-of-response review | `01_review_approval.py` | After the full pipeline (LLM + tools) completes |
| Tool-augmented review | `02_tool_review.py` | After the LLM composes a response that includes tool results |
| Per-step tool review | `03_multistep_tools.py` | Between tool calls, before downstream tools execute |

## Files

### 01_review_approval.py

The simplest HITL pattern. The agent generates a company mission statement for "Green Threads" (a sustainable fashion startup), then a `human_review` callback intercepts the output. The human can **[A]pprove** it as-is or **[M]odify** it by typing a replacement.

```
agent.run(prompt)
    │
    ▼
LLM generates response
    │
    ▼
ContentFilterConfig.on_filter fires
    │
    ▼
┌──────────────────────────┐
│  human_review(text)      │
│                          │
│  print("Agent says:", t) │
│  input("[A]pprove/[M]odify") │
│      │                   │
│      ├── [A] ──▶ return text (original)  │
│      └── [M] ──▶ input("Your version:")  │
│                  return edited text       │
└──────────────────────────┘
    │
    ▼
final output returned
```

Key components:
- `ManagedAgent` with `ContentFilterConfig().on_filter(human_review)`
- `InMemoryProvider` and `MessageHistory` for session persistence
- Demo prompt: "Write a one-sentence company mission statement for a sustainable fashion startup called Green Threads."

### 02_tool_review.py

Extends the first example by adding a `mortgage_calculator` tool. The agent calls the tool with loan parameters, the LLM composes a final response incorporating the tool's result, and then the human reviews the composed output.

```
agent.run("Calculate mortgage...")
    │
    ▼
LLM calls tool: mortgage_calculator(500000, 0.065, 30)
    │
    ▼
tool returns "$3,160.34/month"
    │
    ▼
LLM composes: "The monthly payment is $3,160.34..."
    │
    ▼
ContentFilterConfig.on_filter fires
    │
    ▼
┌──────────────────────────────────┐
│  human_review(composed_text)     │
│                                  │
│  Agent says: "The monthly..."    │
│  [A]pprove  [M]odify             │
│      │                           │
│      ├── [A] ──▶ return original │
│      └── [M] ──▶ return edited   │
└──────────────────────────────────┘
    │
    ▼
final output returned
```

Key components:
- `ToolRegistry` with a plain `mortgage_calculator` function
- `ManagedAgent` with `.with_tools()` and `.with_content_filter()`
- `mortgage_calculator(loan_amount, annual_rate, years)` computes monthly payment
- Demo prompt: "Calculate the monthly payment for a $500,000 mortgage at 6.5% annual interest over 30 years."

### 03_multistep_tools.py

The most granular pattern. The human reviews **between** tool calls. Three tools run in sequence:

1. `get_flight_price(destination)` — human approves or corrects the price
2. `calculate_total(flight_price, hotel_per_night, nights)` — human approves or corrects the total
3. `apply_discount(total, discount_percent)` — automatic, no human review

Each tool that prompts for approval uses `input()` inline. The human's correction replaces the tool's output before it is passed to the next tool or the LLM.

```
agent.run("Plan trip to Tokyo...")
    │
    ▼
Tool 1: get_flight_price("Tokyo")
    │
    ▼
┌──────────────────────────────────────┐
│  print("Found: $1,200.00")           │
│  input("Approve? [Y]es / correction")│
│      │                               │
│      ├── [Y] ──▶ return "$1,200.00"  │
│      └── [other] ──▶ return typed    │
└──────────────────────────────────────┘
    │
    ▼
Tool 2: calculate_total(flight, hotel, nights)
    │
    ▼
┌──────────────────────────────────────┐
│  print("Breakdown: ... Total: $2,200")│
│  input("Approve? [Y]es / correction")│
│      │                               │
│      ├── [Y] ──▶ return "$2,200.00"  │
│      └── [other] ──▶ return typed    │
└──────────────────────────────────────┘
    │
    ▼
Tool 3: apply_discount(total, 10)
    │
    ▼
return "$1,980.00 (10% off)"  (no human review)
    │
    ▼
LLM composes final response
```

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with the `gpt-oss:20b` model pulled

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull the model (first time only)
ollama pull gpt-oss:20b

# 3. Install dependencies
cd agent_harness_examples
uv sync
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama API endpoint |

No API keys are needed when using Ollama as the provider.

## Running

Each file is an independent entry point:

```bash
# Pattern 1: End-of-response review
uv run python 5-human_in_the_loop/01_review_approval.py

# Pattern 2: Tool-augmented review
uv run python 5-human_in_the_loop/02_tool_review.py

# Pattern 3: Per-step tool review
uv run python 5-human_in_the_loop/03_multistep_tools.py
```

## Expected Output

**01_review_approval.py:**
```
[agent] Generating response...

Agent says: Green Threads is committed to...

Approve or modify? [A]pprove  [M]odify:
```
After you approve or modify, the final output is printed.

**02_tool_review.py:**
```
[agent] Generating response...
[tool:mortgage_calculator] $500,000 at 6.5% for 30yr = $3,160.34/month

Agent says: The monthly payment for a $500,000 mortgage at 6.5% over 30 years is $3,160.34.

Approve or modify? [A]pprove  [M]odify:
```

**03_multistep_tools.py:**
```
[agent] Generating response...

[tool:get_flight_price] Looking up price for Tokyo...
[tool:get_flight_price] Found: $1,200.00
[tool:get_flight_price] Approve? [Y]es  enter correction:

[tool:calculate_total] Breakdown:
  Flight:       $1,200.00
  Hotel:        $200.00 × 5 nights
  Total before discount: $2,200.00
[tool:calculate_total] Approve? [Y]es  enter correction:

Final output: Your 5-night trip to Tokyo costs $1,980.00 (10% off).
```

## How It Works

1. **01_review_approval.py** — The `ContentFilterConfig.on_filter` callback fires after the LLM completes. The callback receives the raw output string, displays it, and returns either the original or a human-supplied replacement.

2. **02_tool_review.py** — Same callback mechanism, but the output the human reviews already includes the tool's computed result (the mortgage payment). The LLM composes natural language around the tool output before the filter fires.

3. **03_multistep_tools.py** — Review is embedded inside the tools themselves using `input()`. Each tool prompts for approval before returning. The human's correction replaces the tool's output, which is then passed to the next tool or the LLM for final composition.

## Troubleshooting

- **"Connection refused"** — Ollama is not running. Start it with `ollama serve`.
- **Model not found** — Pull the model: `ollama pull gpt-oss:20b`.
- **Wrong endpoint** — Set `OLLAMA_BASE_URL` if Ollama is running on a non-default host/port.
