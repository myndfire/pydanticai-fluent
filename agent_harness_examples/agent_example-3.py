import os
import asyncio
import uuid
import logfire
from dotenv import load_dotenv

# Fluent agent imports
from agent_harness import (
    ManagedAgent,
    StaticPrompts,
    Observability,
    MessageHistory,
    InMemoryProvider,
)

# Pydantic model describing the expected invoice output
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    invoice_number: str = Field(
        ..., description="The unique identifier for the invoice."
    )
    date_issued: str = Field(..., description="The date when the invoice was issued.")
    due_date: str = Field(..., description="The date by which payment is expected.")
    currency: str = Field(
        ..., description="The currency in which the invoice is denominated."
    )
    customer_name: str = Field(..., description="The name of the customer.")
    company: str = Field(..., description="The company associated with the customer.")
    address: str = Field(..., description="The address of the customer.")
    services_provided: list[str] = Field(
        [], description="A detailed breakdown of services provided."
    )
    subtotal: float = Field(..., description="Total before tax.")
    tax_rate: float = Field(..., description="Tax rate applied.")
    tax_amount: float = Field(..., description="Calculated tax amount.")
    total_amount_due: float = Field(..., description="Final amount after tax.")
    payment_instructions: dict = Field(
        {}, description="Instructions for making payment."
    )


load_dotenv()
logfire.configure()

# Build the agent using the fluent API
obs = Observability()
short_term = InMemoryProvider(max_turns=10)
agent = (
    ManagedAgent()
    .with_model("ollama:gpt-oss:20b")
    .with_prompts(StaticPrompts("You are a helpful assistant"))
    .with_observability(obs)
    .with_short_term_memory(short_term)
    .with_long_term_memory(None)
    .with_output(Invoice)
)


# ---------------------------------------------------------------------------
# Async entry point – runs two separate LLM calls using the same fluent agent
# ---------------------------------------------------------------------------
async def main():
    # Prompt 1 – generate an invoice from a natural‑language description
    prompt_invoice = (
        "Can you generate an invoice for a consulting service provided to Acme Inc. on 2022-01-15 "
        "with a due date of 2022-02-15? The invoice should include the following services: "
        "10 hours of consulting at $100 per hour, 5 hours of research at $50 per hour, "
        "and 3 hours of report writing at $75 per hour. The tax rate is 20 percent and the "
        "payment should be made via bank transfer."
    )
    session_id = f"invoice_{uuid.uuid4().hex[:8]}"
    result = await agent.run_sync(
        prompt_invoice,
        message_history=MessageHistory(),
        session_id=session_id,
    )
    logfire.notice("Text prompt LLM results: {result}", result=str(result.output))
    logfire.info("Result type: {result}", result=type(result.output))

    # Read the markdown file generated previously (or any sample file)
    with open("data/invoice.md", "r") as file:
        invoice_data = file.read()

    # Prompt 2 – extract structured data from the markdown invoice
    prompt_extract = f"Can you extract the following information from the invoice? The raw data is {invoice_data}"
    session_id2 = f"invoice_{uuid.uuid4().hex[:8]}"
    result2 = await agent.run_sync(
        prompt_extract,
        message_history=MessageHistory(),
        session_id=session_id2,
    )
    logfire.notice(
        "Invoice markdown prompt LLM results: {result}", result=str(result2.output)
    )
    logfire.info("Result type: {result}", result=type(result2.output))


if __name__ == "__main__":
    asyncio.run(main())
