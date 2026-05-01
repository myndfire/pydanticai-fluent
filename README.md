# pydanticai-fluent
Fluent syntax for configuring agents with pydantic AI

## Overview

`pydanticai-fluent` is a project that provides a fluent, builder-style API for configuring and managing AI agents using the [pydantic-ai](https://github.com/pydantic/pydantic-ai) library. The project includes an `agent_harness` library that offers a clean, composable interface for setting up agents with cross-cutting concerns like memory, observability, error handling, and messaging.

## Project Structure

```
pydanticai-fluent/
├── agent_harness/           # Core library with fluent API
│   ├── src/agent_harness/
│   │   ├── agent.py         # Main ManagedAgent class
│   │   ├── memory.py        # Memory providers
│   │   ├── tools.py         # Tool registry
│   │   ├── prompts.py       # Prompt providers
│   │   ├── observability.py # Logging, tracing, metrics
│   │   ├── errorhandling.py # Error handling configuration
│   │   ├── guards.py        # Guardrails
│   │   ├── evaluators.py    # Evaluation components
│   │   ├── rabbitmq.py      # RabbitMQ messaging
│   │   ├── tracing.py       # Tracing implementations
│   │   ├── logging.py       # Logging utilities
│   │   ├── config.py        # Configuration
│   │   ├── file_storage.py  # File storage
│   │   ├── metrics.py       # Metrics collection
│   │   └── __init__.py
│   └── pyproject.toml       # Package configuration
├── agent_harness_examples/  # Example applications
│   ├── agent_example-1.py   # Basic agent with tools
│   ├── agent_example-LINE OLDER VERSION OF FILE.ts   # Arch index.ts (probably obsolete)
│   ├── agent_example-2.py   # Agent with memory and error handling
│   ├── agent_example-3.py   # Structured output with Pydantic models
│   └── document_classification_rabbitmq_agent.py  # RabbitMQ messaging example
└── README.md
```

## Core Features

- **Fluent API**: Chainable builder methods for configuring agents
- **Memory Management**: Short-term (in-memory) and long-term (MongoDB) memory providers
- **Observability**: Built-in logging, tracing, and metrics
- **Error Handling**: Configurable error handling with fallback strategies
- **Guards**: Guardrails for safety and compliance
- **Tool Integration**: Easy registration of tools and MCP servers
- **Structured Output**: Support for Pydantic model outputs
- **Messaging**: RabbitMQ integration for message-based workflows

## Installation

```bash
# Install the agent_harness library locally
cd agent_harness
pip install -e .

# Or install directly from the directory
pip install ./agent_harness
```

## Running the Examples

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (for Ollama models)
- Optional: MongoDB for long-term memory (examples 2 and 3)
- Optional: RabbitMQ for messaging (document classification example)

### Example 1: Basic Agent with Tools

```bash
cd agent_harness_examples
python agent_example-1.py
```

This example demonstrates:
- Basic agent configuration with fluent API
- Tool registration (`repeat` and `shout` functions)
- Simple prompt execution

### Example 2: Agent with Memory and Error Handling

```bash
cd agent_harness_examples
python agent_example-2.py
```

This example demonstrates:
- Short-term and optional long-term (MongoDB) memory
- Error handling configuration
- Multiple sequential agent runs
- Observability with Logfire tracing

### Example 3: Structured Output with Pydantic Models

```bash
cd agent_harness_examples
python agent_example-3.py
```

This example demonstrates:
- Structured output using Pydantic models
- Invoice data extraction
- Logfire integration for observability

### Document Classification with RabbitMQ

```bash
cd agent_harness_examples
python document_classification_rabbitmq_agent.py
```

This example demonstrates:
- RabbitMQ message-based workflow
- Document classification with structured output
- Error handling with dead-letter queues
- Multi-tenant support with session management

## Configuration

### Environment Variables

Create a `.env` file in the examples directory:

```env
# Ollama configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434/v1

# MongoDB for long-term memory (optional)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agent_memory
MONGODB_COLLECTION=conversations

# RabbitMQ (for messaging examples)
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
```

### Model Configuration

The fluent API supports various model configurations:

```python
from agent_harness import ManagedAgent

agent = (
    ManagedAgent()
    .with_model("ollama:gpt-oss:20b")  # Local Ollama model
    # or .with_model("gpt-4")          # OpenAI model
    .with_short_term_memory(InMemoryProvider())
    .with_tools(ToolRegistry().add_many(tool1, tool2))
    .with_prompts(StaticPrompts("You are a helpful assistant"))
    .with_observability(Observability())
    .with_error_handling(ErrorHandlingConfig())
)
```

## Advanced Usage

### MCP Server Integration

```python
agent.with_mcp_server("http://localhost:8000", tool_prefix="mcp_")
```

### Custom Error Handlers

```python
from agent_harness.errorhandling import ErrorHandlingConfig, AgentErrorContext

class CustomErrorHandler:
    def __call__(self, ctx: AgentErrorContext, exc: Exception) -> bool:
        # Handle error and return True if fallback was used
        return False

agent.with_error_handling(
    ErrorHandlingConfig().with_error_handler(CustomErrorHandler())
)
```

### Structured Output Types

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    invoice_number: str = Field(..., description="Invoice number")
    total_amount: float = Field(..., description="Total amount due")

agent.with_output(Invoice)
```

### RabbitMQ Messaging

```python
agent.with_rabbitmq(
    host="localhost",
    port=5672,
    username="guest",
    password="guest"
).with_input_queue("input_queue")
.with_output_queue("output_queue")
.with_dead_letter_queue("dead_letter_queue")
```

## Development

### Dependencies

See `agent_harness/pyproject.toml` for the full dependency list:
- `pydantic-ai`
- `python-dotenv`
- `structlog`
- `jinja2`
- `pydantic-settings`
- `aio-pika`

### Running Tests

```bash
cd agent_harness
pytest tests/
```

## License

MIT License