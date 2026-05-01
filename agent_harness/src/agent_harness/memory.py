from beanie import Document, init_beanie
from pydantic.dataclasses import dataclass as pydantic_dataclass
import uuid
from collections import defaultdict
from dataclasses import field, asdict
from datetime import datetime
from typing import Protocol, Any, Optional, List

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@pydantic_dataclass
class UsageData:
    """Token usage for a turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@pydantic_dataclass
class TurnData:
    """Complete data for a single conversation turn."""

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    messages: List[dict] = field(default_factory=list)
    usage: Optional[UsageData] = None
    duration_seconds: float = 0.0
    cost: Optional[float] = None
    model: Optional[str] = None
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for any storage that expects JSON‑compatible data."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.usage:
            data["usage"] = asdict(self.usage)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnData":
        """Create a TurnData instance from a dict (e.g. loaded from DB)."""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("usage") and isinstance(data["usage"], dict):
            data["usage"] = UsageData(**data["usage"])  # type: ignore[arg-type]
        return cls(**data)


# ---------------------------------------------------------------------------
# Helper for serialising messages
# ---------------------------------------------------------------------------


def filter_thinking_parts(messages: List[ModelMessage]) -> List[dict]:
    """Filter out internal ThinkingPart objects and return a JSON‑friendly list."""
    from pydantic_ai.messages import ThinkingPart

    serialized: List[dict] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            serialized.append(
                {
                    "kind": "request",
                    "parts": [
                        {
                            "type": p.__class__.__name__,
                            "content": getattr(p, "content", ""),
                        }
                        for p in msg.parts
                    ],
                }
            )
        elif isinstance(msg, ModelResponse):
            filtered_parts = [
                {"type": p.__class__.__name__, "content": getattr(p, "content", "")}
                for p in msg.parts
                if not isinstance(p, ThinkingPart)
            ]
            usage_dict = None
            if msg.usage:
                usage_dict = {
                    "input_tokens": getattr(msg.usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(msg.usage, "output_tokens", 0) or 0,
                    "total_tokens": getattr(msg.usage, "total_tokens", 0) or 0,
                    "prompt_tokens": getattr(msg.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(msg.usage, "completion_tokens", 0)
                    or 0,
                }
            serialized.append(
                {
                    "kind": "response",
                    "parts": filtered_parts,
                    "usage": usage_dict,
                    "model_name": getattr(msg, "model_name", None),
                    "finish_reason": getattr(msg, "finish_reason", None),
                }
            )
    return serialized


# ---------------------------------------------------------------------------
# Protocol for storage back‑ends
# ---------------------------------------------------------------------------


class MemoryProvider(Protocol):
    async def save_turn(self, session_id: str, turn: TurnData) -> None: ...
    async def load_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[TurnData]: ...
    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]: ...
    async def delete_turn(self, session_id: str, turn_id: str) -> bool: ...
    async def clear(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Helper to rebuild a MessageHistory from stored turns
# ---------------------------------------------------------------------------


class MessageHistory:
    def __init__(self):
        self._messages: List[ModelMessage] = []

    async def load(
        self, session_id: str, from_memory: MemoryProvider
    ) -> "MessageHistory":
        turns = await from_memory.load_turns(session_id)
        for turn in turns:
            for msg_dict in turn.messages:
                if msg_dict.get("kind") == "request":
                    from pydantic_ai.messages import UserPromptPart, ModelRequest

                    parts = [
                        UserPromptPart(content=p["content"])
                        for p in msg_dict.get("parts", [])
                    ]
                    self._messages.append(ModelRequest(parts=parts))
                elif msg_dict.get("kind") == "response":
                    from pydantic_ai.messages import TextPart, ModelResponse

                    parts = [
                        TextPart(content=p["content"])
                        for p in msg_dict.get("parts", [])
                    ]
                    msg = ModelResponse(parts=parts)
                    if msg_dict.get("usage"):
                        from pydantic_ai.usage import RequestUsage

                        u = msg_dict["usage"]
                        msg.usage = RequestUsage(
                            input_tokens=u.get("input_tokens", 0),
                            output_tokens=u.get("output_tokens", 0),
                        )
                    self._messages.append(msg)
        return self

    @property
    def messages(self) -> List[ModelMessage]:
        return self._messages


# ---------------------------------------------------------------------------
# In‑memory implementation (keeps TurnData objects directly)
# ---------------------------------------------------------------------------


class InMemoryProvider:
    def __init__(self, max_turns: int = 100):
        self._storage: dict[str, List[TurnData]] = defaultdict(list)
        self._max_turns = max_turns

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        self._storage[session_id].append(turn)
        if len(self._storage[session_id]) > self._max_turns:
            self._storage[session_id] = self._storage[session_id][-self._max_turns :]

    async def load_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[TurnData]:
        turns = self._storage.get(session_id, [])
        if limit:
            turns = turns[-limit:]
        return turns

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        for turn in self._storage.get(session_id, []):
            if turn.turn_id == turn_id:
                return turn
        return None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        lst = self._storage.get(session_id, [])
        for i, turn in enumerate(lst):
            if turn.turn_id == turn_id:
                lst.pop(i)
                return True
        return False

    async def clear(self, session_id: str) -> None:
        if session_id in self._storage:
            del self._storage[session_id]


# ---------------------------------------------------------------------------
# MongoDB implementation using Beanie – stores each turn as its own document.
# ---------------------------------------------------------------------------


class TurnDocument(Document):
    session_id: str
    turn: TurnData

    class Settings:
        name = "turns"

    def to_turn(self) -> TurnData:
        return self.turn


class MongoMemory:
    def __init__(
        self,
        uri: str,
        database: str = "agent_memory",
        collection: str = "conversations",
    ):
        self._uri = uri
        self._database_name = database
        self._collection_name = collection
        self._client = None
        self._db = None
        self._collection = None
        self._connected = False

    async def _ensure_connected(self):
        if self._connected:
            return
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(self._uri)
        self._db = self._client[self._database_name]
        self._collection = self._db[self._collection_name]
        await self._client.admin.command("ping")
        self._connected = True

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        await self._ensure_connected()
        doc = {"session_id": session_id, "turn": turn.to_dict()}
        await self._collection.insert_one(doc)

    async def load_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[TurnData]:
        await self._ensure_connected()
        query = self._collection.find({"session_id": session_id}).sort(
            "turn.timestamp", 1
        )
        if limit:
            query = query.limit(limit)
        docs = await query.to_list()
        return [TurnData.from_dict(doc["turn"]) for doc in docs]

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        await self._ensure_connected()
        doc = await self._collection.find_one(
            {"session_id": session_id, "turn.turn_id": turn_id}
        )
        return TurnData.from_dict(doc["turn"]) if doc else None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        await self._ensure_connected()
        result = await self._collection.delete_one(
            {"session_id": session_id, "turn.turn_id": turn_id}
        )
        return result.deleted_count > 0

    async def clear(self, session_id: str) -> None:
        await self._ensure_connected()
        await self._collection.delete_many({"session_id": session_id})


# ---------------------------------------------------------------------------
# Elasticsearch implementation (unchanged – still uses JSON dicts)
# ---------------------------------------------------------------------------


class ElasticsearchMemory:
    def __init__(
        self, endpoint: str = "http://localhost:9200", index: str = "agent-memory"
    ):
        self._endpoint = endpoint
        self._index = index
        self._es_client = None
        self._connected = False

    async def _ensure_connected(self):
        if self._connected:
            return
        try:
            from elasticsearch import AsyncElasticsearch

            self._es_client = AsyncElasticsearch([self._endpoint])
            await self._es_client.info()
            exists = await self._es_client.indices.exists(index=self._index)
            if not exists:
                await self._es_client.indices.create(
                    index=self._index,
                    mappings={
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "turn_id": {"type": "keyword"},
                            "timestamp": {"type": "date"},
                            "turn_data": {"type": "object"},
                        }
                    },
                )
            self._connected = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Elasticsearch at {self._endpoint}: {str(e)}"
            )

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        await self._ensure_connected()
        await self._es_client.index(
            index=self._index,
            id=f"{session_id}:{turn.turn_id}",
            document={
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp.isoformat(),
                "turn_data": turn.to_dict(),
            },
        )

    async def load_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[TurnData]:
        await self._ensure_connected()
        query = {
            "query": {"term": {"session_id": session_id}},
            "sort": [{"timestamp": {"order": "asc"}}],
            "size": limit or 100,
        }
        response = await self._es_client.search(index=self._index, body=query)
        hits = response["hits"]["hits"]
        return [TurnData.from_dict(hit["_source"]["turn_data"]) for hit in hits]

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        await self._ensure_connected()
        try:
            resp = await self._es_client.get(
                index=self._index, id=f"{session_id}:{turn_id}"
            )
            return TurnData.from_dict(resp["_source"]["turn_data"])
        except Exception:
            return None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        await self._ensure_connected()
        try:
            await self._es_client.delete(
                index=self._index, id=f"{session_id}:{turn_id}"
            )
            return True
        except Exception:
            return False

    async def clear(self, session_id: str) -> None:
        await self._ensure_connected()
        await self._es_client.delete_by_query(
            index=self._index,
            body={"query": {"term": {"session_id": session_id}}},
        )


# ---------------------------------------------------------------------------
# Redis implementation (unchanged – still stores JSON strings)
# ---------------------------------------------------------------------------


class RedisMemory:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "agent:memory:",
    ):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._key_prefix = key_prefix
        self._client = None

    def _get_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    async def _ensure_connected(self):
        if self._client:
            return
        try:
            import redis.asyncio as redis

            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                decode_responses=True,
            )
            await self._client.ping()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Redis at {self._host}:{self._port}: {str(e)}"
            )

    async def save_turn(self, session_id: str, turn: TurnData) -> None:
        import json

        await self._ensure_connected()
        key = self._get_key(session_id)
        await self._client.rpush(key, json.dumps(turn.to_dict()))
        await self._client.ltrim(key, -100, -1)

    async def load_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[TurnData]:
        import json

        await self._ensure_connected()
        key = self._get_key(session_id)
        data = await self._client.lrange(key, 0, -1)
        if not data:
            return []
        turns = [TurnData.from_dict(json.loads(t)) for t in data]
        if limit:
            turns = turns[-limit:]
        return turns

    async def get_turn(self, session_id: str, turn_id: str) -> Optional[TurnData]:
        import json

        await self._ensure_connected()
        key = self._get_key(session_id)
        data = await self._client.lrange(key, 0, -1)
        for t in data:
            turn = TurnData.from_dict(json.loads(t))
            if turn.turn_id == turn_id:
                return turn
        return None

    async def delete_turn(self, session_id: str, turn_id: str) -> bool:
        import json

        await self._ensure_connected()
        key = self._get_key(session_id)
        data = await self._client.lrange(key, 0, -1)
        new_data = [
            t for t in data if TurnData.from_dict(json.loads(t)).turn_id != turn_id
        ]
        if len(new_data) == len(data):
            return False
        await self._client.delete(key)
        if new_data:
            await self._client.rpush(key, *new_data)
        return True

    async def clear(self, session_id: str) -> None:
        await self._ensure_connected()
        key = self._get_key(session_id)
        await self._client.delete(key)
