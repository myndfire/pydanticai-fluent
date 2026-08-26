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

"""Iterative refinement loop — external evaluator drives repeated attempts.

This example demonstrates a programmatic loop where an external evaluator
checks the agent's output quality and feeds back a corrective prompt when
the result does not meet the criteria. The agent is asked to summarize a
block of text in 20 words or fewer. If the output is too long, the
``WordCountEvaluator`` notes the word count and the script constructs a
feedback message that becomes the next prompt. The agent then tries again,
learning from its previous attempt via the updated ``MessageHistory``.

Key Concepts Demonstrated
-------------------------
- **External Quality Gate**: The loop condition is driven by Python code
  (``evaluator.passed``), not by the LLM itself.
- **Custom Evaluator**: Subclasses ``Evaluator`` to count words and store
  pass/fail state after every turn.
- **Feedback as Prompt**: Failed attempts are converted into follow-up
  prompts ("Too long — try again") so the agent can self-correct.
- **Attempt Cap**: A ``for`` loop with a ``MAX_ATTEMPTS`` limit prevents
  infinite retries when the model consistently fails the constraint.

What You Will See
-----------------
A typical run looks like this::

    $ uv run python loops/02_iterative_refinement.py
    ============================================================
    Iterative Refinement Loop
    Model: qwen2.5:3b
    ============================================================

    --- Attempt 1/3 ---
      Output: Python, created by Guido van Rossum in 1991, is a ...
      [evaluator] Word count: 42 / 10 (FAIL)
      Feedback: Your summary was 42 words. Please shorten it ...

    --- Attempt 2/3 ---
      Output: Python is a versatile language created in 1991 ...
      [evaluator] Word count: 28 / 10 (FAIL)
      Feedback: Your summary was 28 words. Please shorten it ...

    --- Attempt 3/3 ---
      Output: Python, a readable language from 1991, is widely ...
      [evaluator] Word count: 9 / 10 (PASS)

    ✓ Success on attempt 3!

Architecture
------------
::

    Initial prompt ("Summarize in ≤20 words")
        │
        ▼
    Agent.run()
        │
        └──► WordCountEvaluator.evaluate()
                │
                ├──► PASS  →  break loop, report success
                └──► FAIL  →  build feedback prompt
                        │
                        ▼
                Reload MessageHistory (includes failed attempt)
                        │
                        ▼
                Next agent.run() with feedback as prompt
                        │
                        ▼
                Repeat up to MAX_ATTEMPTS

Configuration
-------------
- ``REFINEMENT_MAX_WORDS`` — Set in ``.env`` to make the constraint stricter
  or looser (default: 10). Lower to 5 or 3 to force retries.
- ``REFINEMENT_MAX_ATTEMPTS`` — Set in ``.env`` for more retries on weaker
  models (default: 3).
- ``REFINEMENT_TEMPERATURE`` — Set in ``.env`` to increase output randomness.
  Higher values (e.g. 0.9) make the model more verbose, increasing the
  chance of failing the word limit so the feedback loop is visible.
- ``data/text_to_summarize.txt`` — Replace the contents of this file with
  your own text to summarize.
- ``MAX_TOKENS`` — Set in ``.env`` to cap LLM output per run (default: 256).
- Model name is read from ``MODEL_NAME`` in ``.env`` (defaults to
  ``qwen2.5:3b``).

Usage
-----
Run from the ``agent_harness_examples`` directory::

    uv run python loops/02_iterative_refinement.py

Setup
-----
1. Start Ollama (or your preferred local LLM server)::

       ollama serve

2. Install dependencies::

       cd agent_harness_examples
       uv sync

3. (Optional) Edit ``.env`` to change the model.

Tips
----
- If the model consistently overshoots the word limit, try lowering
  ``temperature`` via ``model_settings`` on ``agent.run()``.
- You can make the evaluator more sophisticated (e.g., check for keyword
  inclusion, readability scores, or factual accuracy) by expanding the
  ``evaluate()`` method.
- Because failed attempts are saved to ``MessageHistory``, the agent sees
  its own mistakes and can learn from them on subsequent turns.
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts
from agent_harness.observability import Observability
from agent_harness.evaluators import Evaluator


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")
MAX_WORDS = int(os.getenv("REFINEMENT_MAX_WORDS", "10"))
MAX_ATTEMPTS = int(os.getenv("REFINEMENT_MAX_ATTEMPTS", "3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("REFINEMENT_TEMPERATURE", "0.7"))


TEXT_TO_SUMMARIZE = Path(__file__).with_name("text_to_summarize.txt").read_text()


class WordCountEvaluator(Evaluator):
    """Counts words in the agent output and decides if it passes."""

    def __init__(self, max_words: int):
        self.max_words = max_words
        self.last_count = 0
        self.passed = False

    async def evaluate(self, prompt: str, result, context: dict) -> None:  # type: ignore[override]
        output_text = str(getattr(result, "output", result))
        word_count = len(output_text.split())
        self.last_count = word_count
        self.passed = word_count <= self.max_words

        print(
            f"  [evaluator] Word count: {word_count} / {self.max_words} "
            f"({'PASS' if self.passed else 'FAIL'})"
        )


async def main():
    """Run the iterative refinement demo.

    Setup:
        - Ollama must be running (`ollama serve`).
        - Model must be pulled (default: `ollama pull qwen2.5:3b`).
        - `MODEL_NAME` may override the model in `.env`.
        - `REFINEMENT_MAX_WORDS`, `REFINEMENT_MAX_ATTEMPTS`,
          `REFINEMENT_TEMPERATURE`, `MAX_TOKENS` may be set in `.env`.
        - `OLLAMA_BASE_URL` may configure the Ollama endpoint
          (default: `http://localhost:11434/v1`).
    """
    print("=" * 60)
    print("Iterative Refinement Loop")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    print(f"\nTask: Summarize text in {MAX_WORDS} words or fewer")
    print(f"Text length: {len(TEXT_TO_SUMMARIZE)} characters")
    print(f"  Preview: {TEXT_TO_SUMMARIZE[:100]}...")

    memory = InMemoryProvider()
    session_id = "refinement-session"

    evaluator = WordCountEvaluator(max_words=MAX_WORDS)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME))
        .with_prompts(
            StaticPrompts(
                "You are a concise summarizer. "
                "Your summaries must be accurate and extremely brief."
            )
        )
        .with_observability(Observability())
        .with_short_term_memory(memory)
        .with_evaluators(evaluator)
    )

    # Initial prompt
    history = await MessageHistory().load(session_id, memory)
    prompt = (
        f"Summarize the following text in {MAX_WORDS} words or fewer:\n\n"
        f"{TEXT_TO_SUMMARIZE}"
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")

        # Load history and show accumulated context
        history = await MessageHistory().load(session_id, memory)
        turns = await memory.load_turns(session_id)
        turn_count = len(turns)
        print(f"  [Memory] Loaded {turn_count} prior turn(s) (includes failed attempts)")

        result = await agent.run(prompt, history, session_id, model_settings={"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}, save_to=[memory])
        print(f"  Output: {result.output}")

        if evaluator.passed:
            print(f"\n✓ Success on attempt {attempt}!")
            break

        # Feed back the failure so the next attempt can improve
        feedback = (
            f"Your summary was {evaluator.last_count} words. "
            f"Please shorten it to {MAX_WORDS} words or fewer. "
            f"Keep all key points but use fewer words."
        )
        print(f"  Feedback: {feedback}")
        prompt = feedback
    else:
        print(f"\n✗ Gave up after {MAX_ATTEMPTS} attempts.")
        print(f"  Final word count: {evaluator.last_count}")

    print(f"\n{'=' * 60}")
    print("CONCEPTS DEMONSTRATED")
    print(f"{'=' * 60}")
    print("✓ External evaluator (WordCountEvaluator) enforced constraint")
    print("✓ Feedback from failed attempts became next prompt")
    print(f"✓ MessageHistory retained {attempt} attempts for learning")
    print("✓ Attempt cap prevented infinite loop")


if __name__ == "__main__":
    asyncio.run(main())
