import asyncio
from typing import TYPE_CHECKING

import brotli
import orjson
import websockets
from tqdm import tqdm
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from utils.log_utils import vprint, TQDM_DISABLED

if TYPE_CHECKING:
    from client_args import Config


class Connection:
    def __init__(self, config: "Config"):
        self.config = config
        self.websocket: websockets.WebSocketClientProtocol = None

    async def connect(self):
        uri = f"ws://{self.config.server_uri}/ws/{self.config.client_id}"

        while self.websocket is None:
            try:
                self.websocket = await websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=None,
                    max_size=10 * 1024 * 1024,
                )
            except Exception as e:
                vprint(f"Connection failed: {e}. Retrying...", 2)
                await asyncio.sleep(5)

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_json(self, data: dict, logging=True):
        if self.websocket:
            await self.send_bytes(orjson.dumps(data), logging=logging)

    async def send_bytes(self, data: bytes, logging=True):
        start_marker = b"<START>"
        end_marker = b"<END>"
        chunk_size = 5 * 1024 * 1024

        try:
            await self.websocket.send(start_marker)

            if logging:
                for i in tqdm(
                    range(0, len(data), chunk_size),
                    desc="Sending data to server",
                    unit="chunk",
                    disable=TQDM_DISABLED,
                ):
                    chunk = data[i : i + chunk_size]
                    await self.websocket.send(chunk)
            else:
                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]
                    await self.websocket.send(chunk)

            await self.websocket.send(end_marker)
        except (ConnectionClosedError, OSError) as e:
            await self.connect()
            await self.send_bytes(data, logging=logging)

    async def receive_message(self) -> str:
        data = b""
        start_marker = b"<START>"
        end_marker = b"<END>"
        receiving = False

        try:
            while True:
                chunk = await self.websocket.recv()
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")

                if start_marker in chunk:
                    receiving = True
                    data = b""
                    chunk = chunk.replace(start_marker, b"")
                if end_marker in chunk:
                    chunk = chunk.replace(end_marker, b"")
                    data += chunk
                    break
                if receiving:
                    data += chunk
        except (ConnectionClosedError, OSError) as e:
            await self.connect()
            return await self.receive_message()
        except Exception as e:
            return None

        return data.decode("utf-8")
