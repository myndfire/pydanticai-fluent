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

"""Nested Pydantic models — complex hierarchical structured output.

The LLM produces a full recipe with nested ingredients and instructions.
Each sub-model is independently validated — if one ingredient is missing
a unit, pydantic catches it and the agent retries until everything is valid.

Usage:
    uv run python 03_nested_models.py

Setup
-----
    1. Start Ollama (if using local models):
        ollama serve
    2. Install dependencies and run:
        cd agent_harness_examples
        uv sync
        uv run python structured_output/03_nested_models.py
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_harness.agent import ManagedAgent
from agent_harness.memory import InMemoryProvider, MessageHistory
from agent_harness.model_config import ModelConfig

load_dotenv()

MODEL_NAME = os.getenv("STRUCTURED_OUTPUT_REASONING_MODEL", "phi4-mini-reasoning")
MAX_TOKENS = int(os.getenv("STRUCTURED_OUTPUT_MAX_TOKENS", "512"))


class Ingredient(BaseModel):
    """A single recipe ingredient."""
    name: str = Field(description="Ingredient name, e.g. 'flour'")
    amount: float = Field(description="Quantity as a number")
    unit: str = Field(description="Unit of measurement, e.g. 'cups'")


class InstructionStep(BaseModel):
    """A single step in the recipe instructions."""
    step: int = Field(description="Step number (1, 2, 3, ...)")
    action: str = Field(description="What to do in this step")


class Recipe(BaseModel):
    """A complete recipe with ingredients and instructions."""
    title: str = Field(description="Recipe name")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    cook_time_minutes: int = Field(description="Cooking time in minutes")
    servings: int = Field(description="Number of servings")
    ingredients: list[Ingredient] = Field(description="Full ingredient list")
    instructions: list[InstructionStep] = Field(description="Step-by-step instructions")


async def main():
    """Run the nested models structured output example.

    Setup
    -----
        1. Start Ollama: ollama serve
        2. Pull model: ollama pull phi4-mini-reasoning
        3. Install deps: cd agent_harness_examples && uv sync
    """
    print("=" * 60)
    print("Nested Models — Hierarchical Structured Output")
    print("=" * 60)

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name=MODEL_NAME, max_tokens=MAX_TOKENS))
        .with_output(Recipe, output_retries=3)
    )

    memory = InMemoryProvider()

    # ── Generate a recipe ───────────────────────────────────────
    print("\n--- Chocolate chip cookies ---")
    history = await MessageHistory().load("recipe-1", memory)
    result = await agent.run(
        "Give me a recipe for classic chocolate chip cookies. "
        "Include ingredients and step-by-step instructions.",
        history, "recipe-1",
    )

    recipe = result.output
    print(f"\n  {recipe.title}")
    print(f"  Prep: {recipe.prep_time_minutes} min | "
          f"Cook: {recipe.cook_time_minutes} min | "
          f"Servings: {recipe.servings}")

    # ── Ingredients ─────────────────────────────────────────────
    print(f"\n  Ingredients ({len(recipe.ingredients)} items):")
    for ing in recipe.ingredients:
        print(f"    - {ing.amount} {ing.unit} {ing.name}")

    # ── Instructions ────────────────────────────────────────────
    print(f"\n  Instructions ({len(recipe.instructions)} steps):")
    for step in recipe.instructions:
        print(f"    {step.step}. {step.action}")

    # ── Nested access ───────────────────────────────────────────
    print(f"\n--- Nested access ---")
    print(f"  recipe.ingredients[0] → {recipe.ingredients[0]}")
    print(f"  recipe.ingredients[0].name → '{recipe.ingredients[0].name}'")
    print(f"  recipe.instructions[-1] → {recipe.instructions[-1]}")
    print(f"  All entries validated against Pydantic schemas")


if __name__ == "__main__":
    asyncio.run(main())
