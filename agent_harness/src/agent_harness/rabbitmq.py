"""RabbitMQ messaging service for async queue operations."""

import os
import json
from typing import Any, AsyncIterator, Optional

import aio_pika
from aio_pika import IncomingMessage
from aio_pika.abc import AbstractIncomingMessage


class MessagingService:
    """Async RabbitMQ messenger for queue operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        virtual_host: Optional[str] = None,
    ):
        """
        Initialize MessagingService with connection parameters.

        Args:
            host: RabbitMQ host (default: from RABBITMQ_HOST env or "localhost")
            port: RabbitMQ port (default: from RABBITMQ_PORT env or 5672)
            username: RabbitMQ username (default: from RABBITMQ_USER env or "guest")
            password: RabbitMQ password (default: from RABBITMQ_PASSWORD env or "guest")
            virtual_host: RabbitMQ virtual host (default: from RABBITMQ_VHOST env or "/")
        """
        self.host = host or os.getenv("RABBITMQ_HOST", "localhost")
        self.port = port or int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = username or os.getenv("RABBITMQ_USER", "guest")
        self.password = password or os.getenv("RABBITMQ_PASSWORD", "guest")
        self.virtual_host = virtual_host or os.getenv("RABBITMQ_VHOST", "/")

        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.Channel] = None
        self._exchanges: dict[str, aio_pika.Exchange] = {}

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        if self._connection is None:
            self._connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password,
                virtualhost=self.virtual_host,
            )
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=1)

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        if self._connection is None:
            self._connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password,
                virtualhost=self.virtual_host,
            )
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=1)

    async def declare_exchange(
        self,
        name: str,
        exchange_type: str = "direct",
        durable: bool = True,
    ) -> None:
        """
        Declare an exchange.

        Args:
            name: Exchange name
            exchange_type: Exchange type (direct, fanout, topic, headers)
            durable: Whether exchange should persist across broker restarts
        """
        if self._channel is None:
            raise RuntimeError("Not connected. Call connect() first.")
        exchange = await self._channel.declare_exchange(
            name,
            type=exchange_type,
            durable=durable,
        )
        self._exchanges[name] = exchange

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None

    async def declare_queue(self, name: str, durable: bool = True) -> None:
        """
        Declare a queue.

        Args:
            name: Queue name
            durable: Whether queue should persist across broker restarts
        """
        if self._channel is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._channel.declare_queue(name, durable=durable)

    async def consume(self, queue_name: str) -> AsyncIterator[IncomingMessage]:
        """
        Consume messages from a queue.

        Args:
            queue_name: Queue to consume from

        Yields:
            IncomingMessage objects
        """
        if self._channel is None:
            raise RuntimeError("Not connected. Call connect() first.")

        queue = await self._channel.get_queue(queue_name)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                yield message

    async def publish(
        self,
        queue_name: str,
        message: dict,
        exchange: Optional[str] = None,
        delivery_mode: bool = True,
    ) -> None:
        """
        Publish a message to a queue.

        Args:
            queue_name: Queue to publish to
            message: Message dictionary to publish
            exchange: Exchange to use (default: use default exchange)
            delivery_mode: Whether message should persist (default: True)
        """
        if self._channel is None:
            raise RuntimeError("Not connected. Call connect() first.")

        exch = (
            self._exchanges.get(exchange)
            if exchange
            else self._channel.default_exchange
        )
        await exch.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                if delivery_mode
                else aio_pika.DeliveryMode.NOT_PERSISTENT,
            ),
            routing_key=queue_name,
        )

    async def ack(self, message: AbstractIncomingMessage) -> None:
        """
        Acknowledge a message.

        Args:
            message: Message to acknowledge
        """
        await message.ack()

    async def nack(
        self, message: AbstractIncomingMessage, requeue: bool = False
    ) -> None:
        """
        Negative acknowledge a message.

        Args:
            message: Message to reject
            requeue: Whether to requeue the message
        """
        await message.nack(requeue=requeue)

    @property
    def is_connected(self) -> bool:
        """Check if connected to RabbitMQ."""
        return self._connection is not None and not self._connection.is_closed
