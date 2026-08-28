PydanticAI Fluent — Agent Harness Fluent API Reference

Implementation-oriented reference for agent_harness and
agent_harness_examples

Repository snapshot reviewed: master branch, August 27, 2026

# Purpose

This document catalogs the configuration surface of the
pydanticai-fluent agent harness. It separates the top-level ManagedAgent
fluent API from the subordinate configuration/building APIs used for
tools, prompts, memory, retries, guardrails, error handling, structured
output, observability, tracing, metrics, messaging, log enrichment, and
multi-agent pipelines.

The implementation under agent_harness/src/agent_harness is treated as
the source of truth. The examples under agent_harness_examples are used
to confirm intended composition and usage patterns.

# At-a-glance architecture

| Area                 | Primary type(s)                                                                                                      | How attached/configured                                                             |
|----------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Core agent           | ManagedAgent                                                                                                         | ManagedAgent().with\_\* fluent chain                                                |
| Model                | ModelConfig, model settings                                                                                          | with_model(...), with_model_settings(...)                                           |
| Prompts              | PromptProvider, StaticPrompts, MongoPrompts                                                                          | with_prompts(...)                                                                   |
| Tools                | ToolRegistry, MCP server toolsets                                                                                    | with_tools(...), with_mcp_server(s)(...)                                            |
| Memory               | MessageHistory, InMemoryProvider, MongoMemory, ElasticsearchMemory, RedisMemory                                      | with_short_term_memory(...), with_long_term_memory(...)                             |
| Evaluators           | QualityCheck, SafetyCheck, CustomEvaluator, Evaluator protocol                                                       | with_evaluators(...)                                                                |
| Retries              | AgentRetryConfig, ToolRetryConfig, ResultValidatorRetryConfig                                                        | with_agent_retries(...), with_tool_retries(...), with_result_validator_retries(...) |
| Guardrails           | ContentFilterConfig, PIIDetectionConfig, TokenLimitsConfig, CostLimitsConfig, CircuitBreakerConfig, TurnLimitsConfig | individual with\_\* calls or with_guardrails(...)                                   |
| Error handling       | ErrorHandlingConfig                                                                                                  | with_error_handling(...)                                                            |
| Structured output    | Pydantic output model                                                                                                | with_output(...)                                                                    |
| Observability        | Observability + logger/tracer/metrics implementations                                                                | with_observability(...)                                                             |
| Log enrichment       | LogContext, EnvEnricher, custom provider                                                                             | with_log_enrichment(...)                                                            |
| Pipeline             | PipelineContext                                                                                                      | PipelineContext().with_observability(...).with_memory(...).with_log_enrichment(...) |
| Messaging            | RabbitMQ settings                                                                                                    | with_rabbitmq(...) + queue/exchange fluent methods                                  |
| Dependency injection | deps_type / deps                                                                                                     | with_deps_type(...), run(..., deps=...)                                             |
| Traceback controls   | traceback frame limit                                                                                                | with_traceback_frame_limit(), with_minimal_traceback(), with_full_traceback()       |

# 1. ManagedAgent — complete top-level fluent API

ManagedAgent is the main composition root. The following methods return
the same ManagedAgent instance and are designed to be chained.

| Fluent method                                                                                 | Parameter(s)                   | Purpose                                                                                                                   | Typical value                                       |
|-----------------------------------------------------------------------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| with_model(model)                                                                             | ModelConfig                    | Rebuilds the PydanticAI Agent using the requested provider/model.                                                         | ModelConfig(provider='openai', model_name='gpt-4o') |
| with_model_settings(model_settings)                                                           | Any / PydanticAI ModelSettings | Sets model-generation settings such as temperature, max_tokens, thinking, etc.; preserves current toolsets/output config. | {'temperature': 0.2, 'max_tokens': 2048}            |
| with_log_enrichment(\*providers)                                                              | LogEnrichmentProvider...       | Adds agent-level structured context providers; applied to logs, traces and metrics.                                       | LogContext().with\_('tenant','acme')                |
| with_short_term_memory(provider)                                                              | MemoryProvider                 | Loads session history from the configured short-term store on each run.                                                   | InMemoryProvider()                                  |
| with_long_term_memory(provider=None)                                                          | MemoryProvider \| None         | Adds a second memory source, loaded after short-term memory.                                                              | MongoMemory(...)                                    |
| with_deps_type(deps_type)                                                                     | type                           | Defines the PydanticAI dependency-injection type.                                                                         | AppDeps                                             |
| with_prompts(provider)                                                                        | PromptProvider                 | Selects the source used to resolve the system prompt.                                                                     | StaticPrompts('...')                                |
| with_observability(observability)                                                             | Observability                  | Replaces the observability facade and propagates it to the tool registry.                                                 | Observability(...)                                  |
| with_tools(registry)                                                                          | ToolRegistry                   | Attaches registered function tools and injects observability into the registry.                                           | ToolRegistry().add_many(...)                        |
| with_mcp_server(url, \*\*kwargs)                                                              | str; tool_prefix optional      | Adds one MCP Streamable HTTP server directly as a PydanticAI toolset.                                                     | url, tool_prefix='crm'                              |
| with_mcp_servers(\*urls, tool_prefix=None)                                                    | str...                         | Adds multiple MCP servers using the same optional tool prefix.                                                            | 'http://a','http://b'                               |
| with_evaluators(\*evaluators)                                                                 | Evaluator...                   | Appends post-run evaluators; evaluators execute sequentially.                                                             | QualityCheck(), SafetyCheck()                       |
| with_error_handling(config)                                                                   | ErrorHandlingConfig            | Installs source-specific/catch-all error callbacks.                                                                       | ErrorHandlingConfig().on_tool_error(...)            |
| with_agent_retries(config)                                                                    | AgentRetryConfig               | Sets agent/model-call retry behavior and rebuilds GuardRunner.                                                            | AgentRetryConfig().with_max_retries(5)              |
| with_tool_retries(config)                                                                     | ToolRetryConfig                | Sets tool retry behavior and rebuilds GuardRunner.                                                                        | ToolRetryConfig().with_max_retries(2)               |
| with_result_validator_retries(config)                                                         | ResultValidatorRetryConfig     | Sets structured/result validation retry behavior.                                                                         | ResultValidatorRetryConfig().with_max_retries(3)    |
| with_content_filter(config)                                                                   | ContentFilterConfig            | Installs output content filtering callback(s).                                                                            | ContentFilterConfig().on_filter(fn)                 |
| with_pii_detection(config)                                                                    | PIIDetectionConfig             | Installs PII detection/redaction callback(s).                                                                             | PIIDetectionConfig().on_redact(fn)                  |
| with_token_limits(config)                                                                     | TokenLimitsConfig              | Caps input/output/total tokens.                                                                                           | TokenLimitsConfig().with_max_total_tokens(8000)     |
| with_cost_limits(config)                                                                      | CostLimitsConfig               | Caps input/output/total request cost using configured token pricing.                                                      | CostLimitsConfig().with_max_total_cost(0.10)        |
| with_circuit_breaker(config)                                                                  | CircuitBreakerConfig           | Stops repeated calls after failure threshold until timeout.                                                               | CircuitBreakerConfig().with_threshold(5)            |
| with_turn_limits(config)                                                                      | TurnLimitsConfig               | Caps agent invocations by session id.                                                                                     | TurnLimitsConfig().with_max_turns(20)               |
| with_guardrails(content_filter=None, pii_detection=None, token_limits=None, cost_limits=None) | optional configs               | Convenience method to attach four common guardrail configurations in one call.                                            | with_guardrails(token_limits=..., cost_limits=...)  |
| with_traceback_frame_limit(limit)                                                             | int \| None                    | Controls how many traceback frames are exposed; None means full.                                                          | 2                                                   |
| with_minimal_traceback()                                                                      | none                           | Equivalent to traceback frame limit 0.                                                                                    |                                                     |
| with_full_traceback()                                                                         | none                           | Equivalent to traceback frame limit None.                                                                                 |                                                     |
| with_output(output_type, output_retries=3)                                                    | Pydantic model/type, int       | Configures structured output and PydanticAI validation retries.                                                           | with_output(MyResult, 3)                            |
| with_rabbitmq(host=None, port=None, username=None, password=None, virtual_host=None)          | connection fields              | Stores RabbitMQ connection configuration for messaging use.                                                               | host='localhost', port=5672                         |
| with_input_queue(queue_name)                                                                  | str                            | Sets inbound queue.                                                                                                       | 'agent.in'                                          |
| with_input_exchange(exchange_name)                                                            | str                            | Sets inbound exchange.                                                                                                    | 'agent.input'                                       |
| with_output_queue(queue_name)                                                                 | str                            | Sets outbound queue.                                                                                                      | 'agent.out'                                         |
| with_output_exchange(exchange_name)                                                           | str                            | Sets outbound exchange.                                                                                                   | 'agent.output'                                      |
| with_dead_letter_queue(queue_name)                                                            | str                            | Sets dead-letter queue.                                                                                                   | 'agent.dlq'                                         |
| with_dead_letter_exchange(exchange_name)                                                      | str                            | Sets dead-letter exchange.                                                                                                | 'agent.dlx'                                         |

## Minimal composition example

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_model_settings({"temperature": 0.2})
        .with_prompts(StaticPrompts("You are a helpful assistant."))
        .with_tools(ToolRegistry().add_many(search_tool, save_tool))
        .with_short_term_memory(InMemoryProvider())
        .with_evaluators(QualityCheck(threshold=7.0))
    )

# 2. Model configuration

ModelConfig is provider-agnostic. Authentication normally falls back to
provider environment variables when api_key is not explicitly supplied.

| Setting    | Type         | Default      | Meaning                                                                                        |
|------------|--------------|--------------|------------------------------------------------------------------------------------------------|
| provider   | ProviderType | ollama       | LLM provider identifier.                                                                       |
| model_name | str          | empty string | Provider-specific model identifier without provider prefix.                                    |
| api_key    | str \| None  | None         | Explicit provider API key.                                                                     |
| base_url   | str \| None  | None         | Custom provider endpoint; for Ollama defaults to OLLAMA_BASE_URL or http://localhost:11434/v1. |

## Supported provider values

ollama, openai, anthropic, google, groq, mistral, bedrock, cohere,
huggingface, openrouter, grok, deepseek, cerebras, fireworks, together,
azure, vercel, moonshotai, github, heroku

## Model settings

with_model_settings() forwards a PydanticAI-compatible model-settings
object/dict. The harness intentionally does not constrain this
dictionary, so available keys depend on the selected PydanticAI/provider
model implementation. Common examples include temperature, max_tokens
and provider-specific thinking/reasoning options.

# 3. Prompts

Prompts are abstracted behind
PromptProvider.get_system_prompt(\*\*context). ManagedAgent resolves the
prompt at run time and passes prompt_id plus user-supplied render
variables.

| Provider/API                                                      | Settings                               | Behavior                                                                    |
|-------------------------------------------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| StaticPrompts(system_prompt='You are a helpful assistant')        | system_prompt: str                     | Always returns the same system prompt.                                      |
| MongoPrompts(uri, database='agent_prompts', collection='prompts') | uri, database, collection              | Loads active prompt records from MongoDB and renders templates with Jinja2. |
| PromptProvider protocol                                           | get_system_prompt(\*\*context) -\> str | Implement this protocol for any custom prompt source.                       |
| ManagedAgent.with_prompts(provider)                               | PromptProvider                         | Makes the provider active for the agent.                                    |
| ManagedAgent.run(..., prompt_id='default', \*\*prompt_vars)       | prompt_id + arbitrary render variables | Selects a prompt and supplies variables to the provider.                    |

## Mongo prompt shape observed in the implementation

    {
      "_id": "prompt_id",
      "template": "You are a {{ role }} specialized in {{ domain }}...",
      "active": true,
      "version": 1,
      "created_at": "...",
      "metadata": {"tags": ["production"], "description": "..."}
    }

# 4. Tools and MCP

ToolRegistry is itself fluent. Functions are wrapped for
invocation/result/error logging and can be synchronous or asynchronous.

| ToolRegistry method            | Parameters                     | Purpose                                                                                                       |
|--------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------|
| add(func)                      | Callable                       | Add one function tool; returns registry.                                                                      |
| add_many(\*funcs)              | Callable...                    | Add multiple function tools; returns registry.                                                                |
| add_mcp(server, endpoint=None) | server name, optional endpoint | Placeholder discovery-style MCP integration in ToolRegistry; current implementation logs intended connection. |
| clear()                        | none                           | Remove all registered tools; returns registry.                                                                |
| get_tools()                    | none                           | Returns a copy of registered wrapped functions; not fluent.                                                   |
| register_to_agent(agent)       | PydanticAI Agent               | Registers tools. Uses agent.tool() if first parameter is RunContext; otherwise agent.tool_plain().            |

## Direct MCP toolsets on ManagedAgent

| Method                                     | Settings                              | Notes                                                           |
|--------------------------------------------|---------------------------------------|-----------------------------------------------------------------|
| with_mcp_server(url, \*\*kwargs)           | url; optional tool_prefix             | Uses MCPServerStreamableHTTP and adds it to the agent toolsets. |
| with_mcp_servers(\*urls, tool_prefix=None) | multiple URLs; optional common prefix | Convenience loop around with_mcp_server().                      |

# 5. Memory and conversation history

MessageHistory is explicit at run time. ManagedAgent may also be given
short- and long-term memory providers; both are loaded into the supplied
MessageHistory before the model call.

| Provider            | Constructor settings                                                         | Notes                                                                                           |
|---------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| InMemoryProvider    | max_turns=100                                                                | Process-local turn store; trims oldest turns beyond max_turns.                                  |
| MongoMemory         | uri; database='agent_memory'; collection='conversations'                     | One stored document per conversation turn.                                                      |
| ElasticsearchMemory | endpoint='http://localhost:9200'; index='agent-memory'                       | Stores turns in an Elasticsearch index keyed by session and turn id.                            |
| RedisMemory         | host='localhost'; port=6379; db=0; password=None; key_prefix='agent:memory:' | Stores serialized turns in a Redis list; implementation trims to the most recent 100 entries.   |
| MessageHistory      | no constructor settings                                                      | load(session_id, from_memory) rebuilds PydanticAI messages; .messages exposes the current list. |

## Memory-provider protocol

    save_turn(session_id, turn)
    load_turns(session_id, limit=None)
    get_turn(session_id, turn_id)
    delete_turn(session_id, turn_id)
    clear(session_id)

# 6. Evaluators

Evaluators run after a successful guarded agent call and after turn
construction/saving. Each receives (prompt_text, result, context).

| Evaluator                                  | Settings                                        | Behavior                                                                        |
|--------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------|
| QualityCheck                               | threshold=7.0; judge_model='openai:gpt-4o-mini' | LLM-as-judge quality scoring on a 0–10 scale; logs pass/low-quality outcome.    |
| SafetyCheck                                | none                                            | Uses OpenAI moderation when available; logs safety evaluation.                  |
| CustomEvaluator                            | name='custom'                                   | Base helper for custom evaluator implementations with named structured logging. |
| Evaluator protocol/custom object           | evaluate(prompt, result, context)               | Any compatible evaluator can be attached.                                       |
| ManagedAgent.with_evaluators(\*evaluators) | one or more evaluator objects                   | Appends evaluators; execution order is the order registered.                    |

Evaluator failures are routed with source='evaluator' through the
harness error-handling path rather than silently becoming normal model
output.

# 7. Retry configuration

There are three independently configurable retry scopes.

| Config                     | Constructor defaults                                                                                  | Fluent methods                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| AgentRetryConfig           | max_retries=3; timeout=120; backoff_multiplier=2.0; fallback_model=None; on_retry=None; on_error=None | with_max_retries(n); with_timeout(seconds); with_backoff(multiplier); with_fallback(model); on_retry(cb); on_error(cb) |
| ToolRetryConfig            | max_retries=3; backoff_multiplier=2.0                                                                 | with_max_retries(n); with_backoff(multiplier)                                                                          |
| ResultValidatorRetryConfig | max_retries=3; backoff_multiplier=2.0                                                                 | with_max_retries(n); with_backoff(multiplier)                                                                          |

## Example

    agent.with_agent_retries(
        AgentRetryConfig()
          .with_max_retries(5)
          .with_timeout(60)
          .with_backoff(2.0)
          .with_fallback("ollama:backup")
          .on_retry(on_retry)
          .on_error(on_agent_error)
    )

# 8. Guardrails

Guardrails are optional configurations managed by GuardRunner. The agent
exposes individual attachment methods and a with_guardrails()
convenience method for content, PII, token and cost controls.

| Config               | Constructor settings                                                                                                                                     | Fluent API                                                                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ContentFilterConfig  | on_filter=None; on_error=None                                                                                                                            | on_filter(callback); on_error(callback)                                                                                                                               |
| PIIDetectionConfig   | on_redact=None; on_error=None                                                                                                                            | on_redact(callback); on_error(callback)                                                                                                                               |
| TokenLimitsConfig    | max_input_tokens=None; max_output_tokens=None; max_total_tokens=None; on_token_limit=None; on_error=None                                                 | with_max_input_tokens(n); with_max_output_tokens(n); with_max_total_tokens(n); on_token_limit(cb); on_error(cb)                                                       |
| CostLimitsConfig     | max_input_cost=None; max_output_cost=None; max_total_cost=None; cost_per_input_token=None; cost_per_output_token=None; on_cost_limit=None; on_error=None | with_max_input_cost(n); with_max_output_cost(n); with_max_total_cost(n); with_cost_per_input_token(n); with_cost_per_output_token(n); on_cost_limit(cb); on_error(cb) |
| CircuitBreakerConfig | failure_threshold=5; circuit_timeout=60; on_error=None                                                                                                   | with_threshold(n); with_timeout(seconds); on_error(cb)                                                                                                                |
| TurnLimitsConfig     | max_turns=None; on_turn_limit=None; on_error=None                                                                                                        | with_max_turns(n); on_turn_limit(cb); on_error(cb)                                                                                                                    |

# 9. Error handling

ErrorHandlingConfig uses fluent callback registration by failure source.
A callback receives ErrorContext. Returning a value suppresses the
exception and makes that value the AgentRunResult output; returning None
allows the exception to propagate.

| Fluent callback         | Source handled                                                   |
|-------------------------|------------------------------------------------------------------|
| on_llm_error(cb)        | LLM/model errors                                                 |
| on_tool_error(cb)       | tool failures                                                    |
| on_validation_error(cb) | validation failures                                              |
| on_guardrail_error(cb)  | guardrail failures                                               |
| on_memory_error(cb)     | memory load/save failures                                        |
| on_prompt_error(cb)     | prompt provider/rendering failures                               |
| on_evaluator_error(cb)  | evaluator failures                                               |
| on_output_error(cb)     | turn/result construction/output processing failures              |
| on_error(cb)            | catch-all fallback when no source-specific handler is configured |

## ErrorContext fields

| Field                  | Meaning                                                                 |
|------------------------|-------------------------------------------------------------------------|
| error_type             | Exception/error type name.                                              |
| error_message          | Human-readable message.                                                 |
| source                 | llm, tool, validation, guardrail, memory, prompt, evaluator, or output. |
| session_id             | Session associated with failure.                                        |
| prompt                 | Prompt associated with failure.                                         |
| stack_trace            | Captured traceback text.                                                |
| partial_result         | Partial AgentRunResult when available.                                  |
| attempt / max_attempts | Retry attempt context.                                                  |
| will_retry             | Whether another retry is expected.                                      |

# 10. Structured output

Use ManagedAgent.with_output(output_type, output_retries=3). The harness
rebuilds the underlying PydanticAI Agent with output_type and retries.
The output type is normally a Pydantic model, giving schema-constrained
responses and validation.

    class Answer(BaseModel):
        summary: str
        confidence: float

    agent = ManagedAgent().with_output(Answer, output_retries=3)

# 11. Observability

Observability is a facade over logging, tracing and metrics. It supports
either a single backend for each signal or lists of backends for
fan-out. If no logger is supplied it defaults to ConsoleLogger; if no
tracer is supplied it defaults to NoOpTracer.

| Observability setting | Type/default                     | Purpose                                             |
|-----------------------|----------------------------------|-----------------------------------------------------|
| logger                | Logger \| None                   | Single logger convenience parameter.                |
| tracer                | Tracer \| None                   | Single tracer convenience parameter.                |
| metrics               | MetricsCollector \| None         | Single metrics convenience parameter.               |
| service_name          | str = 'agent'                    | Service name attached to telemetry.                 |
| loggers               | list\[Logger\] \| None           | Multiple logging destinations.                      |
| tracers               | list\[Tracer\] \| None           | Multiple tracing destinations.                      |
| metrics_list          | list\[MetricsCollector\] \| None | Multiple metric destinations.                       |
| traceback_frame_limit | int \| None                      | Maximum captured traceback frames; None means full. |

## Logging backends

| Backend             | Constructor settings                                      |
|---------------------|-----------------------------------------------------------|
| ConsoleLogger       | none                                                      |
| ElasticsearchLogger | endpoint; index_prefix='agent-logs'; service_name='agent' |
| OTELLogger          | service_name='agent'; otlp_endpoint='localhost:4317'      |
| LogfireLogger       | service_name='agent'; logfire_instance=None               |

## Tracing backends

| Backend        | Constructor settings / behavior                                                                                |
|----------------|----------------------------------------------------------------------------------------------------------------|
| NoOpTracer     | No configuration; minimal overhead.                                                                            |
| InMemoryTracer | No configuration; records spans in memory for testing/development.                                             |
| LogfireTracer  | service_name; send_to_logfire=True; instrument_pydantic_ai=True                                                |
| OTELTracer     | service_name; otlp_endpoint='http://localhost:4317'; sample_rate=1.0; create_spans=False; record_failures=True |

OTELTracer defaults to create_spans=False so PydanticAI native
instrumentation is the canonical span tree. record_failures=True still
records failures even when the harness is not creating a span for every
successful operation.

## Metrics backends

| Backend           | Constructor settings                                 |
|-------------------|------------------------------------------------------|
| NoOpMetrics       | none                                                 |
| InMemoryMetrics   | none; supports get_metrics() and reset()             |
| LogfireMetrics    | service_name='agent'                                 |
| OTELMetrics       | service_name='agent'; otlp_endpoint='localhost:4319' |
| PrometheusMetrics | namespace='agent'; push_gateway=None                 |

# 12. Log enrichment

Log enrichment adds stable business/runtime dimensions to the telemetry
context. Agent-level providers are merged first; per-run enrichment then
wins on key conflicts.

| API                                               | Purpose                                                     |
|---------------------------------------------------|-------------------------------------------------------------|
| LogContext().with\_(key, value)                   | Add one context field.                                      |
| LogContext().with_many(\*\*kwargs)                | Add multiple fields.                                        |
| LogContext().merge(other)                         | Merge another LogContext; other wins on conflicts.          |
| LogContext.as_dict() / enrich()                   | Export a plain dict.                                        |
| EnvEnricher()                                     | Adds host, env (APP_ENV; default local), and process id.    |
| ManagedAgent.with_log_enrichment(\*providers)     | Attach reusable agent-level enrichment.                     |
| ManagedAgent.run(..., enrichment=LogContext(...)) | Attach one-run enrichment; overrides same agent-level keys. |

# 13. Pipeline / orchestration

PipelineContext is a lightweight multi-agent stage coordinator with
fluent infrastructure attachment.

| Fluent method                    | Parameter                | Purpose                                            |
|----------------------------------|--------------------------|----------------------------------------------------|
| with_observability(obs)          | Observability            | Auto-log stage completion.                         |
| with_memory(provider)            | MemoryProvider           | Provide memory for run_stage() session management. |
| with_log_enrichment(\*providers) | LogEnrichmentProvider... | Propagate enrichment to stage runs.                |

## Operational methods

| Method                                     | Purpose                                                                                                     |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| run_stage(agent, name, prompt, \*\*kwargs) | Runs a ManagedAgent as a named pipeline stage, handling history/session/enrichment and recording the stage. |
| post(name, success, output, error='')      | Records/logs a stage result manually.                                                                       |
| display_trace()                            | Displays accumulated pipeline stage trace (as demonstrated by the implementation/examples).                 |

# 14. RabbitMQ messaging

| ManagedAgent fluent method      | Setting                                      |
|---------------------------------|----------------------------------------------|
| with_rabbitmq(...)              | host, port, username, password, virtual_host |
| with_input_queue(name)          | input queue                                  |
| with_input_exchange(name)       | input exchange                               |
| with_output_queue(name)         | output queue                                 |
| with_output_exchange(name)      | output exchange                              |
| with_dead_letter_queue(name)    | dead-letter queue                            |
| with_dead_letter_exchange(name) | dead-letter exchange                         |

# 15. Dependency injection

with_deps_type(MyDeps) configures the dependency type on the underlying
PydanticAI Agent. The concrete dependency object is supplied per
execution using run(..., deps=deps). Tools that declare RunContext are
registered with agent.tool(); plain functions are registered with
agent.tool_plain().

# 16. ManagedAgent.run() runtime settings

| Argument         | Required/default           | Meaning                                                              |
|------------------|----------------------------|----------------------------------------------------------------------|
| prompt           | required                   | String or sequence of PydanticAI UserContent for multimodal input.   |
| message_history  | required                   | MessageHistory instance.                                             |
| session_id       | required                   | Key for memory, turn limits and telemetry.                           |
| save_to          | None                       | One or more MemoryProvider targets to persist the completed turn.    |
| deps             | None                       | Dependency object for PydanticAI dependency injection.               |
| enrichment       | None                       | Per-run LogContext; overrides duplicate agent-level enrichment keys. |
| prompt_id        | 'default' (via \*\*kwargs) | Prompt identifier passed to PromptProvider.                          |
| other \*\*kwargs | none                       | Non-underscore keys are passed as prompt-render variables.           |

# 17. Recommended composition patterns

## Production-oriented base agent

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="openai", model_name="gpt-4o"))
        .with_model_settings({"temperature": 0.1})
        .with_prompts(prompt_provider)
        .with_tools(tool_registry)
        .with_short_term_memory(short_memory)
        .with_long_term_memory(long_memory)
        .with_agent_retries(
            AgentRetryConfig()
            .with_max_retries(3)
            .with_timeout(120)
            .with_backoff(2.0)
        )
        .with_tool_retries(ToolRetryConfig().with_max_retries(2))
        .with_token_limits(TokenLimitsConfig().with_max_total_tokens(16000))
        .with_turn_limits(TurnLimitsConfig().with_max_turns(50))
        .with_evaluators(QualityCheck(threshold=7.0))
        .with_observability(observability)
        .with_log_enrichment(EnvEnricher(), LogContext().with_("app", "support"))
    )

## Structured-output agent

    agent = (
        ManagedAgent()
        .with_model(ModelConfig(provider="ollama", model_name="gpt-oss:20b"))
        .with_prompts(StaticPrompts("Return the requested structured result."))
        .with_output(ResultModel, output_retries=3)
        .with_result_validator_retries(
            ResultValidatorRetryConfig().with_max_retries(3)
        )
    )

# 18. Important implementation notes

- ManagedAgent defaults to ModelConfig(provider='ollama',
  model_name='gpt-oss:20b') when no model is provided.

<!-- -->

- ManagedAgent defaults to StaticPrompts(), Observability(), an empty
  ToolRegistry, an empty evaluator list, and GuardConfig defaults.

<!-- -->

- ToolRegistry.add_mcp() is explicitly a placeholder in the reviewed
  implementation; ManagedAgent.with_mcp_server() / with_mcp_servers()
  are the concrete MCP Streamable HTTP integration.

<!-- -->

- Model settings are intentionally passed through to PydanticAI rather
  than exhaustively enumerated by the harness; provider-specific keys
  therefore belong to the selected PydanticAI model/provider contract.

<!-- -->

- ThinkingPart objects are filtered from persisted message history by
  the memory serialization path.

<!-- -->

- Evaluator execution is sequential, in registration order.

<!-- -->

- Short-term memory is loaded before long-term memory.

<!-- -->

- Per-run LogContext enrichment overrides duplicate keys contributed by
  agent-level enrichment providers.

<!-- -->

- OTELTracer defaults to PydanticAI-native trace spans instead of
  creating a duplicate harness span for every successful operation.

# 19. Master fluent API checklist

| Fluent API                                  | Verified |
|---------------------------------------------|----------|
| ManagedAgent.with_model                     | ✓        |
| ManagedAgent.with_model_settings            | ✓        |
| ManagedAgent.with_log_enrichment            | ✓        |
| ManagedAgent.with_short_term_memory         | ✓        |
| ManagedAgent.with_long_term_memory          | ✓        |
| ManagedAgent.with_deps_type                 | ✓        |
| ManagedAgent.with_prompts                   | ✓        |
| ManagedAgent.with_observability             | ✓        |
| ManagedAgent.with_tools                     | ✓        |
| ManagedAgent.with_mcp_server                | ✓        |
| ManagedAgent.with_mcp_servers               | ✓        |
| ManagedAgent.with_evaluators                | ✓        |
| ManagedAgent.with_error_handling            | ✓        |
| ManagedAgent.with_agent_retries             | ✓        |
| ManagedAgent.with_tool_retries              | ✓        |
| ManagedAgent.with_result_validator_retries  | ✓        |
| ManagedAgent.with_content_filter            | ✓        |
| ManagedAgent.with_pii_detection             | ✓        |
| ManagedAgent.with_token_limits              | ✓        |
| ManagedAgent.with_cost_limits               | ✓        |
| ManagedAgent.with_circuit_breaker           | ✓        |
| ManagedAgent.with_turn_limits               | ✓        |
| ManagedAgent.with_guardrails                | ✓        |
| ManagedAgent.with_traceback_frame_limit     | ✓        |
| ManagedAgent.with_minimal_traceback         | ✓        |
| ManagedAgent.with_full_traceback            | ✓        |
| ManagedAgent.with_output                    | ✓        |
| ManagedAgent.with_rabbitmq                  | ✓        |
| ManagedAgent.with_input_queue               | ✓        |
| ManagedAgent.with_input_exchange            | ✓        |
| ManagedAgent.with_output_queue              | ✓        |
| ManagedAgent.with_output_exchange           | ✓        |
| ManagedAgent.with_dead_letter_queue         | ✓        |
| ManagedAgent.with_dead_letter_exchange      | ✓        |
| ToolRegistry.add                            | ✓        |
| ToolRegistry.add_many                       | ✓        |
| ToolRegistry.add_mcp                        | ✓        |
| ToolRegistry.clear                          | ✓        |
| LogContext.with\_                           | ✓        |
| LogContext.with_many                        | ✓        |
| LogContext.merge                            | ✓        |
| AgentRetryConfig.with_max_retries           | ✓        |
| AgentRetryConfig.with_timeout               | ✓        |
| AgentRetryConfig.with_backoff               | ✓        |
| AgentRetryConfig.with_fallback              | ✓        |
| AgentRetryConfig.on_retry                   | ✓        |
| AgentRetryConfig.on_error                   | ✓        |
| ToolRetryConfig.with_max_retries            | ✓        |
| ToolRetryConfig.with_backoff                | ✓        |
| ResultValidatorRetryConfig.with_max_retries | ✓        |
| ResultValidatorRetryConfig.with_backoff     | ✓        |
| ContentFilterConfig.on_filter               | ✓        |
| ContentFilterConfig.on_error                | ✓        |
| PIIDetectionConfig.on_redact                | ✓        |
| PIIDetectionConfig.on_error                 | ✓        |
| TokenLimitsConfig.with_max_input_tokens     | ✓        |
| TokenLimitsConfig.with_max_output_tokens    | ✓        |
| TokenLimitsConfig.with_max_total_tokens     | ✓        |
| TokenLimitsConfig.on_token_limit            | ✓        |
| TokenLimitsConfig.on_error                  | ✓        |
| CostLimitsConfig.with_max_input_cost        | ✓        |
| CostLimitsConfig.with_max_output_cost       | ✓        |
| CostLimitsConfig.with_max_total_cost        | ✓        |
| CostLimitsConfig.with_cost_per_input_token  | ✓        |
| CostLimitsConfig.with_cost_per_output_token | ✓        |
| CostLimitsConfig.on_cost_limit              | ✓        |
| CostLimitsConfig.on_error                   | ✓        |
| CircuitBreakerConfig.with_threshold         | ✓        |
| CircuitBreakerConfig.with_timeout           | ✓        |
| CircuitBreakerConfig.on_error               | ✓        |
| TurnLimitsConfig.with_max_turns             | ✓        |
| TurnLimitsConfig.on_turn_limit              | ✓        |
| TurnLimitsConfig.on_error                   | ✓        |
| ErrorHandlingConfig.on_llm_error            | ✓        |
| ErrorHandlingConfig.on_tool_error           | ✓        |
| ErrorHandlingConfig.on_validation_error     | ✓        |
| ErrorHandlingConfig.on_guardrail_error      | ✓        |
| ErrorHandlingConfig.on_memory_error         | ✓        |
| ErrorHandlingConfig.on_prompt_error         | ✓        |
| ErrorHandlingConfig.on_evaluator_error      | ✓        |
| ErrorHandlingConfig.on_output_error         | ✓        |
| ErrorHandlingConfig.on_error                | ✓        |
| PipelineContext.with_observability          | ✓        |
| PipelineContext.with_memory                 | ✓        |
| PipelineContext.with_log_enrichment         | ✓        |

# 20. References

These references support the API inventory and usage guidance in this
document. The myndfire/pydanticai-fluent implementation is the primary
source of truth. Upstream PydanticAI documentation is included where the
harness delegates to the underlying framework.

R1. Agent harness implementation -
https://github.com/myndfire/pydanticai-fluent/tree/master/agent_harness/src/agent_harness
Primary implementation directory.

R2. Agent harness examples -
https://github.com/myndfire/pydanticai-fluent/tree/master/agent_harness_examples
Intended usage and fluent composition examples.

R3. ManagedAgent - agent.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/agent.py
ManagedAgent.with\_\* methods and run-time behavior.

R4. ModelConfig - model_config.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/model_config.py
Model configuration and provider support.

R5. Prompts - prompts.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/prompts.py
PromptProvider, StaticPrompts, MongoPrompts.

R6. Tools - tools.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/tools.py
ToolRegistry and function-tool wrapping.

R7. Memory - memory.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/memory.py
MessageHistory and memory providers.

R8. Evaluators - evaluators.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/evaluators.py
Evaluator implementations and protocol.

R9. Guardrails/retries - guards.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/guards.py
Retry and guardrail configuration.

R10. Error handling - errorhandling.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/errorhandling.py
ErrorHandlingConfig and ErrorContext.

R11. Observability - observability.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/observability.py
Observability facade.

R12. Logging - logging.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/logging.py
Logging backends.

R13. Tracing - tracing.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/tracing.py
Tracing backends and OTEL settings.

R14. Metrics - metrics.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/metrics.py
Metrics backends.

R15. Log enrichment - log_enrichment.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/log_enrichment.py
LogContext and enrichment providers.

R16. Pipeline - pipeline.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/pipeline.py
PipelineContext and stage execution.

R17. RabbitMQ - rabbitmq.py -
https://github.com/myndfire/pydanticai-fluent/blob/master/agent_harness/src/agent_harness/rabbitmq.py
Messaging integration.

R18. PydanticAI Agents - https://ai.pydantic.dev/agents/ Upstream agent,
instructions, structured output, dependency, model, and model-settings
concepts.

R19. PydanticAI Tools - https://ai.pydantic.dev/tools/ Upstream
tools/toolsets and RunContext concepts.

R20. PydanticAI MCP - https://ai.pydantic.dev/mcp/ Upstream MCP
integration.

R21. PydanticAI source repository -
https://github.com/pydantic/pydantic-ai Underlying framework source.

## Reference map by area

| Area                             | References         |
|----------------------------------|--------------------|
| ManagedAgent / run API           | R1, R3, R18        |
| Models / model settings          | R3, R4, R18        |
| Prompts                          | R3, R5, R18        |
| Tools / MCP                      | R3, R6, R19, R20   |
| Memory                           | R3, R7             |
| Evaluators                       | R3, R8             |
| Retries / guardrails             | R3, R9             |
| Error handling                   | R3, R10            |
| Structured output / dependencies | R3, R18            |
| Observability                    | R11, R12, R13, R14 |
| Log enrichment                   | R15                |
| Pipeline                         | R16                |
| RabbitMQ                         | R17                |
| Examples                         | R2                 |

Scope note. This reference documents the myndfire/pydanticai-fluent
agent_harness API. It should not be confused with the separately
maintained official pydantic-ai-harness package.
