import asyncio
from typing import TYPE_CHECKING, Dict

import brotli
import orjson
from fastapi import WebSocket
from starlette.websockets import WebSocketState
from tqdm import tqdm

from .compressor import compress, decompress
from utils.log_utils import vprint

if TYPE_CHECKING:
    from modules.global_dict.global_dict import GlobalDict
    from server_args import Config


class DisconnectedError(Exception):
    def __init__(self, message="Client is disconnected"):
        self.message = message
        super().__init__(self.message)


class Connection:
    def __init__(self, config: "Config", global_dict: "GlobalDict"):
        self.config = config
        self.global_dict = global_dict
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.global_dict.set_waiting_clients(client_id, False)
        vprint(f"Client {client_id} connected", 2)

    def disconnect(self, client_id: int):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.global_dict.remove_waiting_client(client_id)

        vprint(f"Client {client_id} disconnected", 2)

    def total_connections(self):
        return len(self.active_connections)

    def get_all_clients(self):
        return list(self.active_connections.keys())

    async def send_json(self, data: dict, client_id: int):
        await self.send_bytes(orjson.dumps(data), client_id)

    async def broadcast(self, data: bytes):
        for client_id in self.active_connections.keys():
            await self.send_bytes(data, client_id)

    async def broadcast_json(self, data: dict):
        await self.broadcast(orjson.dumps(data))

    async def send_bytes_batch(
        self,
        data: list[bytes],
        client_ids: list[int],
        logging=True,
        on_start=None,
        on_end=None,
    ):
        if bool(self.config.networking_fairness):
            vprint("Using networking fairness", 2)
            return await self.send_bytes_round_robin(
                data, client_ids, logging, on_start, on_end
            )
        else:
            vprint("Using concurrent sending", 2)
            return await self.send_bytes_concurrently(
                data, client_ids, logging, on_start, on_end
            )

    async def send_bytes_round_robin(
        self,
        data: list[bytes],
        client_ids: list[int],
        logging=True,
        on_start=None,
        on_end=None,
    ):
        start_marker = b"<START>"
        end_marker = b"<END>"
        chunk_size = 5 * 1024 * 1024

        compressed_data = [compress(d) for d in data]

        client_chunk_lengths = [
            len(compressed_data[i]) // chunk_size
            + (1 if len(compressed_data[i]) % chunk_size != 0 else 0)
            for i in range(len(compressed_data))
        ]
        sent_chunks = [0] * len(client_ids)
        finished_clients = set()

        progress_bars = [
            tqdm(
                total=client_chunk_lengths[i],
                desc=f"Client {client_id}",
                unit="chunk",
                leave=False,
                position=i,
                ncols=100,
            )
            for i, client_id in enumerate(client_ids)
        ]

        for i, client_id in enumerate(client_ids):
            websocket = self.active_connections[client_id]
            await websocket.send_bytes(start_marker)

            if on_start:
                on_start(len(compressed_data[i]), client_id)

        while len(finished_clients) < len(client_ids):
            for i, client_id in enumerate(client_ids):
                if i in finished_clients:
                    continue

                websocket = self.active_connections[client_id]
                if (
                    not websocket
                    or websocket.client_state == WebSocketState.DISCONNECTED
                ):
                    vprint(f"Websocket {client_id} is not open, try to connect again", 0)
                    self.active_connections[client_id] = None
                    raise DisconnectedError(f"Client {client_id} is not connected")

                if sent_chunks[i] < client_chunk_lengths[i]:
                    chunk = compressed_data[i][
                        sent_chunks[i] * chunk_size : (sent_chunks[i] + 1) * chunk_size
                    ]
                    await websocket.send_bytes(chunk)
                    sent_chunks[i] += 1

                    if logging:
                        progress_bars[i].update(1)

                    if sent_chunks[i] == client_chunk_lengths[i]:
                        await websocket.send_bytes(end_marker)
                        finished_clients.add(i)
                        progress_bars[i].close()

                        if on_end:
                            on_end(len(compressed_data[i]), client_id)

        if logging:
            vprint("All clients have received their compressed data.", 2)

        return True

    async def send_bytes_concurrently(
        self,
        data: list[bytes],
        client_ids: list[int],
        logging=True,
        on_start=None,
        on_end=None,
    ):
        tasks = []

        for i, client_id in enumerate(client_ids):
            tasks.append(
                asyncio.create_task(
                    self.send_bytes(
                        data[i],
                        client_id,
                        logging=logging,
                        on_start=on_start,
                        on_end=on_end,
                    )
                )
            )

        await asyncio.gather(*tasks)

        if logging:
            vprint("All clients have received their data concurrently.", 2)

        return True

    async def send_bytes(
        self, data: bytes, client_id: int, logging=True, on_start=None, on_end=None
    ):
        compressed_data = compress(data)

        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]

            if not websocket or websocket.client_state == WebSocketState.DISCONNECTED:
                vprint(f"Websocket {client_id} is not open, try to connect again", 0)
                self.active_connections[client_id] = None
                raise DisconnectedError(f"Client {client_id} is not connected")

            start_marker = b"<START>"
            end_marker = b"<END>"
            chunk_size = 5 * 1024 * 1024

            await websocket.send_bytes(start_marker)

            if on_start:
                on_start(len(compressed_data), client_id)

            if logging:
                for i in tqdm(
                    range(0, len(compressed_data), chunk_size),
                    desc=f"Sending compressed data to client {client_id}",
                    unit="chunk",
                ):
                    chunk = compressed_data[i : i + chunk_size]
                    await websocket.send_bytes(chunk)
            else:
                for i in range(0, len(compressed_data), chunk_size):
                    chunk = compressed_data[i : i + chunk_size]
                    await websocket.send_bytes(chunk)

            await websocket.send_bytes(end_marker)

            if on_end:
                on_end(len(compressed_data), client_id)

            return True
        else:
            raise DisconnectedError(f"Client {client_id} is not connected")

    async def receive_message(self, client_id: int):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]

            if not websocket or websocket.client_state == WebSocketState.DISCONNECTED:
                vprint(f"Websocket {client_id} is not open, try to connect again", 0)
                self.active_connections[client_id] = None
                raise DisconnectedError(f"Client {client_id} is not connected")

            data = b""
            start_marker = b"<START>"
            end_marker = b"<END>"
            receiving = False

            try:
                while True:
                    chunk = await websocket.receive_bytes()
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
            except Exception as e:
                vprint(f"Error receiving data from client {client_id}: {e}", 0)
                return None

            try:
                decompressed_data = decompress(data)

                return decompressed_data.decode("utf-8")
            except brotli.error as e:
                vprint(f"Error decompressing data from client {client_id}: {e}", 0)
                return None
        else:
            raise DisconnectedError(f"Client {client_id} is not connected")
