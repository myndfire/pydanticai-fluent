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

"""
Agent Template - Elegant PydanticAI agent with crosscutting concerns.

Simple, elegant composition-based design with fluent API.
"""

from .agent import ManagedAgent
from .config import AgentConfig
from .model_config import ModelConfig, build_model
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
    "ModelConfig",
    "build_model",
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
