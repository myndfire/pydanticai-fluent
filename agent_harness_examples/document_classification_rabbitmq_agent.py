import asyncio
import json
import os
import uuid

from dotenv import load_dotenv

from agent_harness import (
    ManagedAgent,
    ModelConfig,
    MessagingService,
    MessageHistory,
    StaticPrompts,
    Observability,
    LogfireTracer,
    InMemoryProvider,
    MongoMemory,
)
from agent_harness.errorhandling import (
    ErrorHandlingConfig,
    ErrorHandler,
    AgentErrorContext,
)
from pydantic import BaseModel, Field
from typing import Literal


class ClassificationData(BaseModel):
    """Document classification data from LLM."""

    document_type: Literal["medical", "finance", "resume", "unknown"] = Field(
        ...,
        description="The type of document: medical (medical lab results), finance (billing invoice), resume (job resume) or unknown",
    )
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(
        ..., description="Brief explanation of why this document type was selected"
    )
    document_summary: str = Field(
        ..., description="A brief summary of the document contents"
    )


class DocumentClassificationResult(BaseModel):
    """Result message to be posted to the output queue."""

    filename: str = Field(..., description="Original filename")
    session_id: str = Field(..., description="Session ID")
    tenant_id: str | None = Field(None, description="Tenant ID")
    user_id: str | None = Field(None, description="User ID")
    transaction_id: str | None = Field(None, description="Transaction ID")
    classification: ClassificationData = Field(..., description="Classification result")
    status: Literal["success", "error"] = Field(..., description="Processing status")


def create_memory_providers():
    """Create short and long-term memory from .env config."""
    short_term = InMemoryProvider(max_turns=10)
    long_term = None

    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri:
        long_term = MongoMemory(
            uri=mongo_uri,
            database=os.getenv("MONGODB_DATABASE", "agent_memory"),
            collection=os.getenv("MONGODB_COLLECTION", "conversations"),
        )
    return short_term, long_term


class DocumentErrorHandler:
    """Error handler for document classification agent."""

    def __init__(self, obs: Observability):
        self._obs = obs

    def __call__(self, ctx: AgentErrorContext, exc: Exception) -> bool:
        print(
            f"[ERROR] {ctx.source}: {ctx.error_context.error_type} - {ctx.error_context.error_message}"
        )
        print(f"  Session: {ctx.session_id}")

        if hasattr(self._obs, "tracer"):
            self._obs.tracer.error(
                f"{ctx.error_context.error_type}: {ctx.error_context.error_message}",
                source=ctx.source,
                session_id=ctx.session_id or "unknown",
                prompt=ctx.prompt or "unknown",
                stack_trace=ctx.error_context.stack_trace or "",
            )

        return False


load_dotenv()


def create_agent(short_term, long_term, obs):
    """Create the managed agent with all configurations."""
    return (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(StaticPrompts("You are a helpful assistant"))
        .with_observability(obs)
        .with_short_term_memory(short_term)
        .with_long_term_memory(long_term)
        .with_error_handling(
            ErrorHandlingConfig().with_error_handler(DocumentErrorHandler(obs))
        )
        .with_rabbitmq(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
        )
        .with_input_queue("document_input_queue")
        .with_input_exchange("document_workflow_exchange")
        .with_output_queue("classified_document_queue")
        .with_output_exchange("document_workflow_exchange")
        .with_dead_letter_queue("dead_letter_queue")
        .with_dead_letter_exchange("dead_letter_exchange")
        .with_output(ClassificationData)
    )


async def process_message(
    message_data: dict, agent, save_to
) -> DocumentClassificationResult:
    """Process a document classification message."""
    filename = message_data.get("filename")
    session_id = message_data.get("session_id", f"doc_{uuid.uuid4().hex[:8]}")
    tenant_id = message_data.get("tenant_id")
    user_id = message_data.get("user_id")
    transaction_id = message_data.get("transaction_id")

    obs = agent.observability
    obs.logger.info(f"Reading file: {filename}", filename=filename)

    try:
        with open(f"data/{filename}", "r") as f:
            content = f.read()
    except Exception as e:
        obs.logger.error(f"Failed to read file: {e}", exc=str(e))
        raise

    prompt = (
        f"Provide a brief summary of this document in a single paragraph. "
        f"The summary should be detailed enough so that anyone should be able to know what type of document it is "
        f"— whether it's a financial, medical, business, resume, receipt or invoice document: {content}"
    )

    obs.logger.info(
        f"Running classification for session: {session_id}", session=session_id
    )

    result = await agent.run(
        prompt,
        MessageHistory(),
        session_id,
        save_to=save_to,
    )

    obs.logger.info(
        f"Classification complete: document_type={result.output.document_type}, "
        f"confidence={result.output.confidence}, "
        f"summary={result.output.document_summary}"
    )

    return DocumentClassificationResult(
        filename=filename,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        transaction_id=transaction_id,
        classification=result.output,
        status="success",
    )


async def main():
    short_term, long_term = create_memory_providers()
    obs = Observability(tracer=LogfireTracer(service_name="document_classifier"))
    obs.logger.info("Starting document classification agent")

    agent = create_agent(short_term, long_term, obs)
    save_to = [p for p in [short_term, long_term] if p]

    rabbitmq_config = agent._rabbitmq_config
    messenger = MessagingService(**rabbitmq_config)
    await messenger.connect()

    input_exchange = agent._input_exchange
    output_exchange = agent._output_exchange
    dead_letter_exchange = agent._dead_letter_exchange
    input_queue = agent._input_queue
    output_queue = agent._output_queue
    dead_letter_queue = agent._dead_letter_queue

    for exchange in [input_exchange, output_exchange, dead_letter_exchange]:
        if exchange:
            await messenger.declare_exchange(exchange)

    for queue in [input_queue, output_queue, dead_letter_queue]:
        if queue:
            await messenger.declare_queue(queue)

    obs.logger.info(f"Listening on queue: {input_queue}")

    message_data = None
    try:
        async for message in messenger.consume(input_queue):
            try:
                message_data = json.loads(message.body)
                obs.logger.info(f"Processing message: {message_data}")

                output = await process_message(message_data, agent, save_to)

                await messenger.publish(
                    output_queue,
                    output.model_dump(),
                    exchange=output_exchange,
                )
                obs.logger.info(
                    f"Published to queue: {output_queue}, exchange: {output_exchange}"
                )
                await messenger.ack(message)

                obs.logger.info(f"Result: {output}")

            except Exception as e:
                obs.logger.error(f"Error processing message: {e}", error=str(e))

                filename = message_data.get("filename") if message_data else "unknown"
                session_id = (
                    message_data.get("session_id") if message_data else "unknown"
                )
                tenant_id = message_data.get("tenant_id")
                user_id = message_data.get("user_id")
                transaction_id = message_data.get("transaction_id")
                error_output = DocumentClassificationResult(
                    filename=filename,
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    transaction_id=transaction_id,
                    classification=ClassificationData(
                        document_type="unknown",
                        confidence=0.0,
                        reasoning=f"Error: {str(e)}",
                        document_summary="",
                    ),
                    status="error",
                )
                await messenger.publish(
                    dead_letter_queue,
                    error_output.model_dump(),
                    exchange=dead_letter_exchange,
                )
                await messenger.nack(message, requeue=False)

    finally:
        await messenger.disconnect()
        obs.logger.info("Disconnected from RabbitMQ")


if __name__ == "__main__":
    asyncio.run(main())
