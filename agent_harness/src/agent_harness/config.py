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

"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AgentConfig(BaseSettings):
    """Configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model Configuration
    model_name: str = "ollama:gpt-oss:20b"
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Memory Configuration
    memory_type: str = "in-memory"  # "in-memory" | "mongodb" | "qdrant"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "agent_memory"
    mongodb_collection: str = "conversations"

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_memory"

    # Prompt Configuration
    prompt_source: str = "static"  # "static" | "mongodb"
    prompt_mongodb_uri: str = "mongodb://localhost:27017"
    prompt_database: str = "agent_prompts"
    prompt_collection: str = "prompts"
    default_system_prompt: str = "You are a helpful assistant"

    # Logging Configuration
    enable_otel: bool = False
    otel_service_name: str = "my-agent"
    otel_endpoint: str = "http://localhost:4317"
    elasticsearch_endpoint: Optional[str] = None
    elasticsearch_index_prefix: str = "agent-logs"

    # Guard Configuration
    max_retries: int = 3
    timeout: int = 30
    fallback_model: Optional[str] = None

    # File Storage Configuration
    file_storage_mongodb_uri: str = "mongodb://localhost:27017"
    file_storage_database: str = "agent_files"
    file_storage_collection: str = "files"
