"""File storage in MongoDB using GridFS."""

from typing import BinaryIO, Any, Optional, Union
from datetime import datetime
import io


class FileStorage:
    """MongoDB GridFS-based file storage."""

    def __init__(
        self, uri: str, database: str = "agent_files", collection: str = "files"
    ):
        """
        Initialize file storage.

        Args:
            uri: MongoDB connection URI
            database: Database name
            collection: GridFS collection prefix (creates files.files and files.chunks)
        """
        self._uri = uri
        self._database_name = database
        self._collection_prefix = collection
        self._client = None
        self._fs = None
        self._connected = False

    async def _ensure_connected(self):
        """Ensure MongoDB GridFS connection is established (fail fast)."""
        if self._connected:
            return

        try:
            from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

            self._client = AsyncIOMotorClient(self._uri)
            db = self._client[self._database_name]
            self._fs = AsyncIOMotorGridFSBucket(db, bucket_name=self._collection_prefix)

            # Test connection
            await self._client.admin.command("ping")
            self._connected = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MongoDB GridFS at {self._uri}: {str(e)}"
            )

    async def store_file(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Store a file in MongoDB GridFS.

        Args:
            file_data: File bytes or file-like object
            filename: Original filename
            metadata: Optional metadata dict
            content_type: MIME type

        Returns:
            File ID (string)
        """
        await self._ensure_connected()

        # Convert to bytes if needed
        if isinstance(file_data, bytes):
            data = file_data
        else:
            data = file_data.read()

        # Prepare metadata
        full_metadata = {
            "filename": filename,
            "uploaded_at": datetime.now().isoformat(),
            "content_type": content_type or "application/octet-stream",
            "size": len(data),
        }

        if metadata:
            full_metadata.update(metadata)

        # Upload to GridFS
        file_id = await self._fs.upload_from_stream(
            filename, data, metadata=full_metadata
        )

        return str(file_id)

    async def retrieve_file(self, file_id: str) -> tuple[bytes, dict[str, Any]]:
        """
        Retrieve a file from MongoDB GridFS.

        Args:
            file_id: File ID from store_file()

        Returns:
            Tuple of (file_bytes, metadata)
        """
        await self._ensure_connected()

        from bson import ObjectId

        try:
            # Get file metadata
            grid_out = await self._fs.open_download_stream(ObjectId(file_id))

            # Read file data
            data = await grid_out.read()

            # Get metadata
            metadata = {
                "filename": grid_out.filename,
                "content_type": grid_out.metadata.get("content_type"),
                "size": grid_out.length,
                "uploaded_at": grid_out.metadata.get("uploaded_at"),
                **{
                    k: v
                    for k, v in grid_out.metadata.items()
                    if k not in ["filename", "content_type", "size", "uploaded_at"]
                },
            }

            return data, metadata

        except Exception as e:
            raise FileNotFoundError(f"File with ID {file_id} not found: {str(e)}")

    async def delete_file(self, file_id: str) -> None:
        """
        Delete a file from MongoDB GridFS.

        Args:
            file_id: File ID from store_file()
        """
        await self._ensure_connected()

        from bson import ObjectId

        try:
            await self._fs.delete(ObjectId(file_id))
        except Exception as e:
            raise FileNotFoundError(f"File with ID {file_id} not found: {str(e)}")

    async def list_files(
        self, filter_metadata: Optional[dict[str, Any]] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        List files with optional metadata filtering.

        Args:
            filter_metadata: Optional metadata filter (e.g., {"session_id": "123"})
            limit: Maximum number of files to return

        Returns:
            List of file metadata dicts with 'file_id', 'filename', etc.
        """
        await self._ensure_connected()

        # Build query
        query = {}
        if filter_metadata:
            for key, value in filter_metadata.items():
                query[f"metadata.{key}"] = value

        # Query files collection
        collection = self._client[self._database_name][
            f"{self._collection_prefix}.files"
        ]
        cursor = collection.find(query).limit(limit)

        files = []
        async for doc in cursor:
            file_info = {
                "file_id": str(doc["_id"]),
                "filename": doc["filename"],
                "length": doc["length"],
                "upload_date": doc["uploadDate"].isoformat(),
                "metadata": doc.get("metadata", {}),
            }
            files.append(file_info)

        return files

    async def get_file_metadata(self, file_id: str) -> dict[str, Any]:
        """
        Get file metadata without downloading the file.

        Args:
            file_id: File ID from store_file()

        Returns:
            Metadata dict
        """
        await self._ensure_connected()

        from bson import ObjectId

        collection = self._client[self._database_name][
            f"{self._collection_prefix}.files"
        ]
        doc = await collection.find_one({"_id": ObjectId(file_id)})

        if not doc:
            raise FileNotFoundError(f"File with ID {file_id} not found")

        return {
            "file_id": str(doc["_id"]),
            "filename": doc["filename"],
            "length": doc["length"],
            "upload_date": doc["uploadDate"].isoformat(),
            "metadata": doc.get("metadata", {}),
        }
