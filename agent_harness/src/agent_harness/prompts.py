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

"""Prompt management with MongoDB backend and Jinja2 templates."""

from typing import Protocol, Any, Optional
from jinja2 import Environment, BaseLoader, Template


class PromptProvider(Protocol):
    """Protocol for prompt providers."""

    async def get_system_prompt(self, **context) -> str:
        """Get system prompt with optional context variables."""
        ...


class StaticPrompts:
    """Simple static prompt provider (default)."""

    def __init__(self, system_prompt: str = "You are a helpful assistant"):
        """
        Initialize static prompt provider.

        Args:
            system_prompt: The static system prompt to use
        """
        self._prompt = system_prompt

    async def get_system_prompt(self, **context) -> str:
        """Return the static system prompt."""
        return self._prompt


class MongoPrompts:
    """MongoDB-backed prompt management with Jinja2 templating."""

    def __init__(
        self, uri: str, database: str = "agent_prompts", collection: str = "prompts"
    ):
        """
        Initialize MongoDB prompt provider.

        MongoDB Schema:
        {
            "_id": "prompt_id",
            "template": "You are a {{role}} specialized in {{domain}}...",
            "active": true,
            "version": 1,
            "created_at": ISODate("2024-01-01"),
            "metadata": {"tags": ["production"], "description": "..."}
        }

        Args:
            uri: MongoDB connection URI
            database: Database name
            collection: Collection name
        """
        self._uri = uri
        self._database_name = database
        self._collection_name = collection
        self._client = None
        self._collection = None
        self._connected = False
        self._jinja_env = Environment(loader=BaseLoader())
        self._cache: dict[str, Template] = {}

    async def _ensure_connected(self):
        """Ensure MongoDB connection is established (fail fast)."""
        if self._connected:
            return

        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._client = AsyncIOMotorClient(self._uri)
            self._collection = self._client[self._database_name][self._collection_name]

            # Test connection
            await self._client.admin.command("ping")
            self._connected = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MongoDB for prompts at {self._uri}: {str(e)}"
            )

    async def get_system_prompt(self, prompt_id: str = "default", **variables) -> str:
        """
        Get and render system prompt from MongoDB.

        Args:
            prompt_id: Prompt ID to fetch
            **variables: Template variables for Jinja2 rendering

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt not found or inactive
        """
        await self._ensure_connected()

        # Check cache first
        if prompt_id in self._cache:
            template = self._cache[prompt_id]
            return template.render(**variables)

        # Fetch from MongoDB
        doc = await self._collection.find_one({"_id": prompt_id, "active": True})

        if not doc:
            raise ValueError(f"Prompt '{prompt_id}' not found or inactive in MongoDB")

        # Validate template
        try:
            template = self._jinja_env.from_string(doc["template"])
        except Exception as e:
            raise ValueError(
                f"Invalid Jinja2 template in prompt '{prompt_id}': {str(e)}"
            )

        # Cache template
        self._cache[prompt_id] = template

        # Render with variables
        try:
            return template.render(**variables)
        except Exception as e:
            raise ValueError(f"Failed to render prompt '{prompt_id}': {str(e)}")

    async def list_prompts(self, active_only: bool = True) -> list[dict[str, Any]]:
        """
        List available prompts.

        Args:
            active_only: Only return active prompts

        Returns:
            List of prompt metadata
        """
        await self._ensure_connected()

        query = {"active": True} if active_only else {}
        cursor = self._collection.find(query)

        prompts = []
        async for doc in cursor:
            prompts.append(
                {
                    "prompt_id": doc["_id"],
                    "version": doc.get("version", 1),
                    "active": doc.get("active", True),
                    "created_at": doc.get("created_at"),
                    "metadata": doc.get("metadata", {}),
                }
            )

        return prompts

    async def create_prompt(
        self,
        prompt_id: str,
        template: str,
        version: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create a new prompt in MongoDB.

        Args:
            prompt_id: Unique prompt ID
            template: Jinja2 template string
            version: Version number
            metadata: Optional metadata
        """
        await self._ensure_connected()

        from datetime import datetime

        # Validate template syntax
        try:
            self._jinja_env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {str(e)}")

        doc = {
            "_id": prompt_id,
            "template": template,
            "active": True,
            "version": version,
            "created_at": datetime.now(),
            "metadata": metadata or {},
        }

        await self._collection.insert_one(doc)

    async def update_prompt(
        self,
        prompt_id: str,
        template: Optional[str] = None,
        active: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Update an existing prompt.

        Args:
            prompt_id: Prompt ID to update
            template: New template (optional)
            active: Active status (optional)
            metadata: New metadata (optional)
        """
        await self._ensure_connected()

        update_doc = {}

        if template is not None:
            # Validate template
            try:
                self._jinja_env.from_string(template)
            except Exception as e:
                raise ValueError(f"Invalid Jinja2 template: {str(e)}")
            update_doc["template"] = template

            # Invalidate cache
            if prompt_id in self._cache:
                del self._cache[prompt_id]

        if active is not None:
            update_doc["active"] = active

        if metadata is not None:
            update_doc["metadata"] = metadata

        if update_doc:
            result = await self._collection.update_one(
                {"_id": prompt_id}, {"$set": update_doc}
            )

            if result.matched_count == 0:
                raise ValueError(f"Prompt '{prompt_id}' not found")

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()
