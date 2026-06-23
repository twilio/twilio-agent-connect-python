"""
Server-Sent Events (SSE) manager for real-time browser updates.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class SSEManager:
    """Manages SSE connections and broadcasts events to all clients."""

    def __init__(self) -> None:
        self.clients: set[asyncio.Queue[str]] = set()

    async def event_stream(self) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self.clients.add(queue)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
        finally:
            self.clients.discard(queue)

    def broadcast(self, event_type: str, data: dict) -> None:
        message = json.dumps({"type": event_type, "data": data})
        for client_queue in list(self.clients):
            try:
                client_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass


sse_manager = SSEManager()
