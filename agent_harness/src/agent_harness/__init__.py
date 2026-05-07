"""
Agent Template - Elegant PydanticAI agent with crosscutting concerns.

Simple, elegant composition-based design with fluent API.
"""

from .agent import ManagedAgent
from .config import AgentConfig
from .memory import (
    MemoryProvider,
    InMemoryProvider,
    MongoMemory,
    RedisMemory,
    ElasticsearchMemory,
    TurnData,
    UsageData,
    MessageHistory,
)
from .prompts import PromptProvider, StaticPrompts, MongoPrompts
from .logging import (
    Logger,
    ConsoleLogger,
    ElasticsearchLogger,
    FileLogger,
    CompositeLogger,
)
from .tracing import (
    Tracer,
    NoOpTracer,
    InMemoryTracer,
    LogfireTracer,
    OTELTracer,
    JaegerTracer,
)
from .metrics import (
    MetricsCollector,
    NoOpMetrics,
    InMemoryMetrics,
    OTLPMetrics,
    PrometheusMetrics,
    StatsdMetrics,
    MetricNames,
)
from .observability import Observability, ObservabilityBuilder
from .tools import ToolRegistry
from .guards import (
    GuardConfig,
    GuardRunner,
    ErrorContext,
    AgentRunResult,
    AgentRetryConfig,
    ToolRetryConfig,
    ResultValidatorRetryConfig,
    ContentFilterConfig,
    PIIDetectionConfig,
    CostLimitsConfig,
    CircuitBreakerConfig,
)
from .evaluators import Evaluator, QualityCheck, SafetyCheck, CustomEvaluator
from .file_storage import FileStorage
from .rabbitmq import MessagingService

__version__ = "0.1.0"

__all__ = [
    # Core
    "ManagedAgent",
    "AgentConfig",
    # Memory
    "MemoryProvider",
    "InMemoryProvider",
    "MongoMemory",
    "RedisMemory",
    "ElasticsearchMemory",
    "TurnData",
    "UsageData",
    # Prompts
    "PromptProvider",
    "StaticPrompts",
    "MongoPrompts",
    # Logging
    "Logger",
    "ConsoleLogger",
    "ElasticsearchLogger",
    "FileLogger",
    "CompositeLogger",
    # Tracing
    "Tracer",
    "NoOpTracer",
    "InMemoryTracer",
    "LogfireTracer",
    "OTELTracer",
    "JaegerTracer",
    # Metrics
    "MetricsCollector",
    "NoOpMetrics",
    "InMemoryMetrics",
    "OTLPMetrics",
    "PrometheusMetrics",
    "StatsdMetrics",
    "MetricNames",
    # Observability (Unified)
    "Observability",
    "ObservabilityBuilder",
    # Tools
    "ToolRegistry",
    # Guards
    "GuardConfig",
    "GuardRunner",
    "ErrorContext",
    "AgentRunResult",
    "AgentRetryConfig",
    "ToolRetryConfig",
    "ResultValidatorRetryConfig",
    "ContentFilterConfig",
    "PIIDetectionConfig",
    "CostLimitsConfig",
    "CircuitBreakerConfig",
    # Evaluators
    "Evaluator",
    "QualityCheck",
    "SafetyCheck",
    "CustomEvaluator",
    # File Storage
    "FileStorage",
    # Messaging
    "MessagingService",
]
