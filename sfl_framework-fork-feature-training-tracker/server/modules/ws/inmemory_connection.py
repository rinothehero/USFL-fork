import asyncio
from typing import TYPE_CHECKING, Callable, Dict

import orjson
from tqdm import tqdm

from utils.log_utils import vprint

if TYPE_CHECKING:
    from modules.global_dict.global_dict import GlobalDict
    from server_args import Config


class DisconnectedError(Exception):
    def __init__(self, message="Client is disconnected"):
        self.message = message
        super().__init__(self.message)


class InMemoryConnection:
    def __init__(self, config: "Config", global_dict: "GlobalDict"):
        self.config = config
        self.global_dict = global_dict
        self.active_connections: Dict[int, Dict] = {}

    async def connect(self, client_id: int):
        self.active_connections[client_id] = {
            "to_client_queue": asyncio.Queue(),
            "from_client_queue": asyncio.Queue(),
            "connected": True,
        }
        self.global_dict.set_waiting_clients(client_id, False)
        vprint(f"Server: Client {client_id} connected", 2)
        return self.active_connections[client_id]

    def disconnect(self, client_id: int):
        if client_id in self.active_connections:
            self.active_connections[client_id]["connected"] = False
            del self.active_connections[client_id]
            self.global_dict.remove_waiting_client(client_id)
        vprint(f"Server: Client {client_id} disconnected", 2)

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
            raise NotImplementedError(
                "Networking fairness is not implemented for inmemory connection"
            )
        else:
            return await self.send_bytes_concurrently(
                data, client_ids, logging, on_start, on_end
            )

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
            vprint("Server: All clients have received their data concurrently.", 2)

        return True

    async def send_bytes(
        self, data: bytes, client_id: int, logging=True, on_start=None, on_end=None
    ):

        conn = self.active_connections.get(client_id)
        if not conn or not conn["connected"]:
            raise DisconnectedError(f"Client {client_id} is not connected")

        start_marker = b"<START>"
        end_marker = b"<END>"
        chunk_size = 5 * 1024 * 1024

        await conn["to_client_queue"].put(start_marker)

        if on_start:
            on_start(len(data), client_id)

        await conn["to_client_queue"].put(data)
        await conn["to_client_queue"].put(end_marker)

        if on_end:
            on_end(len(data), client_id)

        # self.global_dict.add_event(
        #     "SEND_BYTES", {"client_id": client_id, "data_size": len(data)}
        # )

        return True

    async def receive_message(self, client_id: int):
        conn = self.active_connections.get(client_id)
        if not conn or not conn["connected"]:
            raise DisconnectedError(f"Client {client_id} is not connected")

        data = b""
        start_marker = b"<START>"
        end_marker = b"<END>"
        receiving = False

        try:
            while True:
                chunk = await conn["from_client_queue"].get()
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
            vprint(f"Server: Error receiving data from client {client_id}: {e}", 0)
            return None

        # self.global_dict.add_event(
        #     "RECEIVE_MESSAGE",
        #     {"client_id": client_id, "data_size": len(data)},
        # )

        return data.decode("utf-8")
