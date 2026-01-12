from typing import TYPE_CHECKING

import orjson
from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from typing import Callable

    from modules.ws.connection import Connection
    from server_args import Config


class Endpoint:
    def __init__(self, config: "Config", connection: "Connection"):
        self.config = config
        self.connection = connection

        self.handlers = {}
        self.callback = {}

    def add_handler(self, event_name: str, handler: "Callable"):
        self.handlers[event_name] = handler

    async def websocket_endpoint(
        self,
        websocket: "WebSocket",
        client_id: int,
    ):
        await self.connection.connect(websocket, client_id)

        try:
            while True:
                message = await self.connection.receive_message(client_id)
                message_data = orjson.loads(message)

                event = message_data.get("event")
                params = message_data.get("params", {})

                if event not in self.handlers:
                    print(f"Event {event} not found (client_id: {client_id})")
                    continue

                handler = self.handlers[event]
                message: dict = await handler(
                    {
                        "client_id": client_id,
                        **params,
                    }
                )

                if message is not None:
                    await self.connection.send_bytes(
                        orjson.dumps(message), client_id, logging=False
                    )

        except WebSocketDisconnect:
            self.connection.disconnect(client_id)
