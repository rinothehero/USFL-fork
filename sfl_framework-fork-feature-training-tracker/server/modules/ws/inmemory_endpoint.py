import asyncio
from typing import Callable, Dict

import orjson
from server_args import Config

from .inmemory_connection import DisconnectedError, InMemoryConnection


class InMemoryEndpoint:
    def __init__(self, config: "Config", connection: InMemoryConnection):
        self.config = config
        self.connection = connection
        self.handlers: Dict[str, Callable] = {}
        self.callback = {}

    def add_handler(self, event_name: str, handler: Callable):
        self.handlers[event_name] = handler

    async def mock_endpoint(self):
        while True:
            await asyncio.sleep(10)

    async def simulate_endpoint(self, client_id: int):
        try:
            while True:
                message = await self.connection.receive_message(client_id)
                if message is None:
                    break

                try:
                    message_data = orjson.loads(message)
                except orjson.JSONDecodeError:
                    print(f"Invalid JSON from client {client_id}")
                    continue

                event = message_data.get("event")
                params = message_data.get("params", {})

                if event not in self.handlers:
                    print(f"Event {event} not found (client_id: {client_id})")
                    continue

                handler = self.handlers[event]
                response: dict = await handler({"client_id": client_id, **params})

                if response is not None:
                    await self.connection.send_bytes(
                        orjson.dumps(response), client_id, logging=False
                    )

        except DisconnectedError:
            pass
        finally:
            self.connection.disconnect(client_id)
