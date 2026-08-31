from dataclasses import dataclass

from pydantic import BaseModel

from agent_harness import ManagedAgent
from agent_harness.model_config import ModelConfig
from agent_harness.prompts import StaticPrompts, MongoPrompts
from agent_harness.tools import ToolRegistry
from agent_harness.memory import (
    MessageHistory,
    InMemoryProvider,
    MongoMemory,
    ElasticsearchMemory,
    RedisMemory,
)
from agent_harness.evaluators import (
    QualityCheck,
    SafetyCheck,
    CustomEvaluator,
)
from agent_harness.guards import (
    AgentRetryConfig,
    ToolRetryConfig,
    ResultValidatorRetryConfig,
    ContentFilterConfig,
    PIIDetectionConfig,
    TokenLimitsConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
    TurnLimitsConfig,
)
from agent_harness.errorhandling import ErrorHandlingConfig
from agent_harness.observability import Observability
from agent_harness.logging import (
    ConsoleLogger,
    ElasticsearchLogger,
    OTELLogger,
    LogfireLogger,
)
from agent_harness.tracing import (
    NoOpTracer,
    InMemoryTracer,
    LogfireTracer,
    OTELTracer,
)
from agent_harness.metrics import (
    NoOpMetrics,
    InMemoryMetrics,
    LogfireMetrics,
    OTELMetrics,
    PrometheusMetrics,
)
from agent_harness.log_enrichment import (
    LogContext,
    EnvEnricher,
)


# -------------------------------------------------------------------
# Structured output
# -------------------------------------------------------------------

class AgentOutput(BaseModel):
    summary: str
    confidence: float


# -------------------------------------------------------------------
# Dependency injection
# -------------------------------------------------------------------

@dataclass
class AppDeps:
    tenant_id: str
    user_id: str


# -------------------------------------------------------------------
# Example tools
# -------------------------------------------------------------------

async def search_tool(query: str) -> str:
    return f"Search results for: {query}"


async def save_tool(value: str) -> str:
    return f"Saved: {value}"


tool_registry = (
    ToolRegistry()
    .add(search_tool)
    .add(save_tool)
)

# Equivalent:
#
# tool_registry = ToolRegistry().add_many(
#     search_tool,
#     save_tool,
# )


# -------------------------------------------------------------------
# Prompt providers
# -------------------------------------------------------------------

static_prompts = StaticPrompts(
    system_prompt="You are a helpful assistant."
)

mongo_prompts = MongoPrompts(
    uri="mongodb://localhost:27017",
    database="agent_prompts",
    collection="prompts",
)


# -------------------------------------------------------------------
# Memory provider examples
# -------------------------------------------------------------------

in_memory = InMemoryProvider(
    max_turns=100
)

mongo_memory = MongoMemory(
    uri="mongodb://localhost:27017",
    database="agent_memory",
    collection="conversations",
)

elasticsearch_memory = ElasticsearchMemory(
    endpoint="http://localhost:9200",
    index="agent-memory",
)

redis_memory = RedisMemory(
    host="localhost",
    port=6379,
    db=0,
    password=None,
    key_prefix="agent:memory:",
)


# -------------------------------------------------------------------
# Evaluators
# -------------------------------------------------------------------

quality_check = QualityCheck(
    threshold=7.0,
    judge_model="openai:gpt-4o-mini",
)

safety_check = SafetyCheck()

custom_evaluator = CustomEvaluator(
    name="custom"
)


# -------------------------------------------------------------------
# Retry configuration
# -------------------------------------------------------------------

def on_retry(*args, **kwargs):
    pass


def on_agent_error(*args, **kwargs):
    pass


agent_retries = (
    AgentRetryConfig()
    .with_max_retries(3)
    .with_timeout(120)
    .with_backoff(2.0)
    .with_fallback("ollama:backup")
    .on_retry(on_retry)
    .on_error(on_agent_error)
)

tool_retries = (
    ToolRetryConfig()
    .with_max_retries(3)
    .with_backoff(2.0)
)

result_validator_retries = (
    ResultValidatorRetryConfig()
    .with_max_retries(3)
    .with_backoff(2.0)
)


# -------------------------------------------------------------------
# Guardrail callbacks
# -------------------------------------------------------------------

def on_filter(*args, **kwargs):
    pass


def on_redact(*args, **kwargs):
    pass


def on_token_limit(*args, **kwargs):
    pass


def on_cost_limit(*args, **kwargs):
    pass


def on_turn_limit(*args, **kwargs):
    pass


def on_guardrail_error(*args, **kwargs):
    pass


content_filter = (
    ContentFilterConfig()
    .on_filter(on_filter)
    .on_error(on_guardrail_error)
)

pii_detection = (
    PIIDetectionConfig()
    .on_redact(on_redact)
    .on_error(on_guardrail_error)
)

token_limits = (
    TokenLimitsConfig()
    .with_max_input_tokens(4000)
    .with_max_output_tokens(4000)
    .with_max_total_tokens(8000)
    .on_token_limit(on_token_limit)
    .on_error(on_guardrail_error)
)

cost_limits = (
    CostLimitsConfig()
    .with_max_input_cost(0.05)
    .with_max_output_cost(0.05)
    .with_max_total_cost(0.10)
    .with_cost_per_input_token(0.000001)
    .with_cost_per_output_token(0.000002)
    .on_cost_limit(on_cost_limit)
    .on_error(on_guardrail_error)
)

circuit_breaker = (
    CircuitBreakerConfig()
    .with_threshold(5)
    .with_timeout(60)
    .on_error(on_guardrail_error)
)

turn_limits = (
    TurnLimitsConfig()
    .with_max_turns(20)
    .on_turn_limit(on_turn_limit)
    .on_error(on_guardrail_error)
)


# -------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------

def on_llm_error(ctx):
    return None


def on_tool_error(ctx):
    return None


def on_validation_error(ctx):
    return None


def on_memory_error(ctx):
    return None


def on_prompt_error(ctx):
    return None


def on_evaluator_error(ctx):
    return None


def on_output_error(ctx):
    return None


def on_error(ctx):
    return None


error_handling = (
    ErrorHandlingConfig()
    .on_llm_error(on_llm_error)
    .on_tool_error(on_tool_error)
    .on_validation_error(on_validation_error)
    .on_guardrail_error(on_guardrail_error)
    .on_memory_error(on_memory_error)
    .on_prompt_error(on_prompt_error)
    .on_evaluator_error(on_evaluator_error)
    .on_output_error(on_output_error)
    .on_error(on_error)
)


# -------------------------------------------------------------------
# Observability — backends
#
# The OTEL backends (OTELMetrics / OTELTracer) are created FIRST so they
# register the global OpenTelemetry providers with their OTLP readers and
# span processors. The Logfire backends are created afterwards and attach to
# those already-registered providers (OpenTelemetry rejects overriding an
# existing provider), which avoids the "Overriding of current ...Provider"
# warnings.
# -------------------------------------------------------------------

# --- OTEL backends (registered first so the global providers carry the OTLP
#     readers/processors; the Logfire backends below then attach to them) ---
otel_metrics = OTELMetrics(
    service_name="agent",
    otlp_endpoint="localhost:4317",
)

otel_tracer = OTELTracer(
    service_name="agent",
    otlp_endpoint="localhost:4317",
    sample_rate=1.0,
    create_spans=False,
    record_failures=True,
)

# --- Metrics ---
noop_metrics = NoOpMetrics()

in_memory_metrics = InMemoryMetrics()

logfire_metrics = LogfireMetrics(
    service_name="agent",
)

prometheus_metrics = PrometheusMetrics(
    namespace="agent",
    push_gateway=None,
)

# --- Tracing ---
noop_tracer = NoOpTracer()

in_memory_tracer = InMemoryTracer()

logfire_tracer = LogfireTracer(
    service_name="agent",
    send_to_logfire=True,
    instrument_pydantic_ai=True,
)

# --- Logging (Logfire attaches to the providers registered above) ---
console_logger = ConsoleLogger()

elasticsearch_logger = ElasticsearchLogger(
    endpoint="http://localhost:9200",
    index_prefix="agent-logs",
    service_name="agent",
)

otel_logger = OTELLogger(
    service_name="agent",
    otlp_endpoint="localhost:4317",
)

logfire_logger = LogfireLogger(
    service_name="agent",
    logfire_instance=None,
)


# -------------------------------------------------------------------
# Observability facade
# -------------------------------------------------------------------

observability = Observability(
    # Single-backend convenience settings
    logger=console_logger,
    tracer=otel_tracer,
    metrics=otel_metrics,

    service_name="agent",

    # Multi-backend fan-out alternatives
    loggers=[
        console_logger,
        elasticsearch_logger,
        otel_logger,
        logfire_logger,
    ],
    tracers=[
        noop_tracer,
        in_memory_tracer,
        logfire_tracer,
        otel_tracer,
    ],
    metrics_list=[
        noop_metrics,
        in_memory_metrics,
        logfire_metrics,
        otel_metrics,
        prometheus_metrics,
    ],

    traceback_frame_limit=None,
)


# -------------------------------------------------------------------
# Log enrichment
# -------------------------------------------------------------------

log_context = (
    LogContext()
    .with_("tenant", "acme")
    .with_("application", "support")
    .with_many(
        region="us-central",
        environment="development",
    )
)

env_enricher = EnvEnricher()


# -------------------------------------------------------------------
# ManagedAgent — kitchen-sink composition
# -------------------------------------------------------------------

agent = (
    ManagedAgent()

    # ModelConfig options:
    # provider defaults to "ollama"
    # model_name defaults to ""
    # api_key defaults to None
    # base_url defaults to None
    .with_model(
        ModelConfig(
            provider="ollama",
            model_name="gpt-oss:20b",
            api_key=None,
            base_url="http://localhost:11434/v1",
        )
    )

    # Intentionally open-ended PydanticAI/provider settings.
    .with_model_settings(
        {
            "temperature": 0.2,
            "max_tokens": 2048,
        }
    )

    # PromptProvider
    .with_prompts(
        static_prompts
    )

    # Function tools
    .with_tools(
        tool_registry
    )

    # Direct MCP integration
    .with_mcp_server(
        "http://localhost:8001/mcp",
        tool_prefix="crm",
    )

    .with_mcp_servers(
        "http://localhost:8002/mcp",
        "http://localhost:8003/mcp",
        tool_prefix="external",
    )

    # Memory
    .with_short_term_memory(
        in_memory
    )

    .with_long_term_memory(
        mongo_memory
    )

    # Dependency injection type
    .with_deps_type(
        AppDeps
    )

    # Evaluators — execute sequentially
    .with_evaluators(
        quality_check,
        safety_check,
        custom_evaluator,
    )

    # Retry scopes
    .with_agent_retries(
        agent_retries
    )

    .with_tool_retries(
        tool_retries
    )

    .with_result_validator_retries(
        result_validator_retries
    )

    # Individual guardrails
    .with_content_filter(
        content_filter
    )

    .with_pii_detection(
        pii_detection
    )

    .with_token_limits(
        token_limits
    )

    .with_cost_limits(
        cost_limits
    )

    .with_circuit_breaker(
        circuit_breaker
    )

    .with_turn_limits(
        turn_limits
    )

    # Instead of the four individual common guardrails above,
    # these can also be supplied with:
    #
    # .with_guardrails(
    #     content_filter=content_filter,
    #     pii_detection=pii_detection,
    #     token_limits=token_limits,
    #     cost_limits=cost_limits,
    # )

    # Error routing
    .with_error_handling(
        error_handling
    )

    # Structured result
    .with_output(
        AgentOutput,
        output_retries=3,
    )

    # Observability
    .with_observability(
        observability
    )

    # Agent-level log enrichment
    .with_log_enrichment(
        env_enricher,
        log_context,
    )

    # Traceback configuration.
    # Pick ONE of these styles:
    .with_traceback_frame_limit(2)

    # Alternatives:
    # .with_minimal_traceback()
    # .with_full_traceback()

    # RabbitMQ connection
    .with_rabbitmq(
        host="localhost",
        port=5672,
        username="guest",
        password="guest",
        virtual_host="/",
    )

    .with_input_queue(
        "agent.in"
    )

    .with_input_exchange(
        "agent.input"
    )

    .with_output_queue(
        "agent.out"
    )

    .with_output_exchange(
        "agent.output"
    )

    .with_dead_letter_queue(
        "agent.dlq"
    )

    .with_dead_letter_exchange(
        "agent.dlx"
    )
)