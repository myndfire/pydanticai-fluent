# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Goal-seeking loop — external verification drives repeated attempts.

This example demonstrates a **goal-seeking** or **convergent** loop where
the *program* controls the loop condition and the *agent* provides candidate
solutions. The goal is to find an animal whose typical lifespan falls between
7 and 12 years. After each guess, Python looks up the lifespan in a
hardcoded dictionary, checks the constraint, and feeds back a corrective hint
("Too short!", "Too long!", or "Unknown animal") that becomes the next prompt.
The agent therefore learns from its mistakes across loop iterations.

Key Concepts Demonstrated
-------------------------
- **Program-Controlled Loop**: The ``for`` loop decides when to stop,
  not the LLM. This is useful when you have hard constraints that are
  cheaper to verify in code than to explain in natural language.
- **Closed-Loop Feedback**: The agent's output is parsed, validated, and
  the validation result is injected back into the conversation history as a
  new user message.
- **Output Parsing**: A regex extracts the first animal name from the agent's
  response so the program can perform a dictionary lookup.
- **Constraint Satisfaction**: The loop terminates as soon as the constraint
  is satisfied or the attempt budget is exhausted.

What You Will See
-----------------
A typical run converges in 2–4 attempts::

    $ uv run python loops/04_goal_seeking_loop.py
    ============================================================
    Goal-Seeking Loop
    Model: qwen2.5:3b
    ============================================================
    Goal: Find an animal whose typical lifespan is between 7 and 12 years.

    --- Attempt 1/5 ---
      Agent guess: dog
      A dog lives ~13 years. Too long! Try a shorter-lived animal.

    --- Attempt 2/5 ---
      Agent guess: hamster
      A hamster lives ~3 years. Too short! Try a longer-lived animal.

    --- Attempt 3/5 ---
      Agent guess: rabbit
      A rabbit lives ~9 years. That's between 7 and 12. ✓

    ✓ Goal achieved on attempt 3!

Architecture
------------
::

    Initial prompt: "Name an animal..."
        │
        ▼
    Agent.run()  →  returns text
        │
        ▼
    extract_animal()  →  animal name
        │
        ▼
    check_lifespan(animal)  →  (success?, feedback)
        │
        ├──► True   →  print success, break loop
        └──► False  →  print feedback
                │
                ▼
        Reload MessageHistory (includes prior guesses)
                │
                ▼
        Agent.run(feedback)  →  next guess
                │
                ▼
        Repeat up to MAX_ATTEMPTS

Configuration
-------------
- ``GOAL_LIFESPAN_MIN`` and ``GOAL_LIFESPAN_MAX`` — Set in ``.env`` to change
  the target lifespan range (defaults: 7, 12 years).
- ``GOAL_MAX_ATTEMPTS`` — Set in ``.env`` for harder constraints or weaker
  models (default: 5).
- ``MAX_TOKENS`` — Set in ``.env`` to cap LLM output per run (default: 128).
- Model name is read from ``MODEL_NAME`` in ``.env`` (defaults to
  ``qwen2.5:3b``).

Usage
-----
Run from the ``agent_harness_examples`` directory::

    uv run python loops/04_goal_seeking_loop.py

Setup
-----
1. Start Ollama (or your preferred local LLM server)::

       ollama serve

2. Install dependencies::

       cd agent_harness_examples
       uv sync

3. (Optional) Edit ``.env`` to change the model or target range.

Tips
----
- If the agent hallucinates unknown animals, the feedback says "I don't know
  that animal's lifespan" and prompts it to try a common pet or farm animal.
- You can replace ``extract_animal`` and ``check_lifespan`` with domain-specific
  validators (e.g., JSON schema validation, unit-test assertions, or Pydantic
  model checks) for any constrained-generation task.
"""

import os
import asyncio
import re

import structlog
from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts



load_dotenv()

log = structlog.get_logger()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")

LIFESPAN_MIN = int(os.getenv("GOAL_LIFESPAN_MIN", "7"))
LIFESPAN_MAX = int(os.getenv("GOAL_LIFESPAN_MAX", "12"))
MAX_ATTEMPTS = int(os.getenv("GOAL_MAX_ATTEMPTS", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("GOAL_TEMPERATURE", "0.7"))

ANIMAL_LIFESPANS = {
    "dog": 13,
    "cat": 15,
    "goat": 15,
    "rabbit": 9,
    "parrot": 50,
    "elephant": 70,
    "horse": 25,
    "hamster": 3,
    "goldfish": 2,
    "turtle": 80,
    "guineapig": 8,
    "ferret": 8,
    "chinchilla": 15,
    "gerbil": 4,
    "lizard": 6,
    "snake": 15,
}


def extract_animal(text: str) -> str | None:
    """Extract the first animal name from the agent's response."""
    match = re.search(r"\b([a-zA-Z]+)\b", text.strip())
    if match:
        return match.group(1).lower()
    return None


def check_lifespan(animal: str) -> tuple[bool, str]:
    """Check if animal's lifespan is in target range. Returns (success, feedback)."""
    lifespan = ANIMAL_LIFESPANS.get(animal)
    if lifespan is None:
        return (
            False,
            f"I don't know a {animal}'s lifespan. Try a common pet or farm animal.",
        )
    if LIFESPAN_MIN < lifespan < LIFESPAN_MAX:
        return (
            True,
            f"A {animal} lives ~{lifespan} years. That's between {LIFESPAN_MIN} and {LIFESPAN_MAX}. ✓",
        )
    elif lifespan <= LIFESPAN_MIN:
        return (
            False,
            f"A {animal} lives ~{lifespan} years. Too short! Try a longer-lived animal.",
        )
    else:
        return (
            False,
            f"A {animal} lives ~{lifespan} years. Too long! Try a shorter-lived animal.",
        )


async def main():
    """Run the goal-seeking loop demo.

    Setup:
        - Ollama must be running (`ollama serve`).
        - Model must be pulled (default: `ollama pull qwen2.5:3b`).
        - `MODEL_NAME` may override the model in `.env`.
        - `GOAL_LIFESPAN_MIN`, `GOAL_LIFESPAN_MAX`, `GOAL_MAX_ATTEMPTS`,
          `GOAL_TEMPERATURE`, `MAX_TOKENS` may be set in `.env`.
        - `OLLAMA_BASE_URL` may configure the Ollama endpoint
          (default: `http://localhost:11434/v1`).
    """
    log.debug("separator")
    log.debug("section", title="Goal-Seeking Loop")
    log.debug("model", name=MODEL_NAME)
    log.debug("separator")
    log.debug("goal", min=LIFESPAN_MIN, max=LIFESPAN_MAX, message="Find an animal whose typical lifespan is between 7 and 12 years.")

    memory = InMemoryProvider()
    session_id = "goal-seeking-session"

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_prompts(
            StaticPrompts(
                "You are an animal-guessing assistant. "
                "Your task is to name an animal whose lifespan falls within the target range. "
                "Consider any feedback from previous guesses. "
                "Respond with a single animal name as your final answer."
            )
        )
        .with_short_term_memory(memory)
    )

    history = await MessageHistory().load(session_id, memory)
    prompt = (
        f"Task: Find an animal whose typical lifespan is between {LIFESPAN_MIN} and {LIFESPAN_MAX} years. "
        f"Reply with only the animal name."
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        turns = await memory.load_turns(session_id)
        turn_count = len(turns)
        log.debug("attempt", number=attempt, max_attempts=MAX_ATTEMPTS)
        log.debug("memory_loaded", turn_count=turn_count)

        result = await agent.run(
            prompt,
            history,
            session_id,
            model_settings={"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
            save_to=[memory]
        )
        log.debug("agent_guess", output=result.output)

        guess = extract_animal(result.output)
        if guess is None:
            log.debug("parse_failed", message="Could not parse an animal name from the response.")
            feedback = (
                "Your response did not contain a valid animal name. "
                "Please reply with only a single animal name."
            )
        else:
            success, feedback = check_lifespan(guess)
            log.debug("feedback", text=feedback)
            if success:
                log.debug("goal_achieved", attempt=attempt)
                log.debug("separator")
                log.debug("section", title="CONCEPTS DEMONSTRATED")
                log.debug("separator")
                log.debug("concept", description="Program-controlled loop: Python decided when to stop")
                log.debug("concept", description="Closed-loop feedback: validation result injected into next prompt")
                log.debug("concept", description="Output parsing: regex extracted animal name for dictionary lookup")
                log.debug("concept", description="Constraint satisfaction: loop ended when range was met")
                break

        if "Too short" in feedback:
            direction = "longer-lived"
        elif "Too long" in feedback:
            direction = "shorter-lived"
        else:
            direction = "different"
        prompt = (
            f"Task: Find an animal whose typical lifespan is between {LIFESPAN_MIN} and {LIFESPAN_MAX} years.\n"
            f"Previous result: {feedback}\n"
            f"Try a {direction} animal. Reply with only your next guess."
        )
        history = await MessageHistory().load(session_id, memory)
    else:
        log.debug("goal_not_achieved", max_attempts=MAX_ATTEMPTS)
        log.debug("separator")
        log.debug("section", title="CONCEPTS DEMONSTRATED")
        log.debug("separator")
        log.debug("concept", description="Program-controlled loop: Python enforced MAX_ATTEMPTS cap")
        log.debug("concept", description="Closed-loop feedback: validation drove repeated attempts")
        log.debug("concept", description="Output parsing: regex extracted animal name for lookup")
        log.debug("concept", description="Constraint satisfaction: budget exhausted before goal met")


if __name__ == "__main__":
    asyncio.run(main())
