import orjson
from client_args import Config
from tqdm import tqdm

from utils.log_utils import vprint


class InMemoryConnection:
    def __init__(self, config: "Config", server):
        self.config = config
        self.server = server
        self.client_id = self.config.client_id
        self.conn = None

    async def connect(self):
        self.conn = await self.server.connect(self.client_id)
        vprint(f"Client: Connected as Client {self.client_id}", 2)

    async def disconnect(self):
        if self.conn and self.conn["connected"]:
            self.server.disconnect(self.client_id)
            self.conn = None

    async def send_json(self, data: dict, logging=True):
        await self.send_bytes(orjson.dumps(data), logging=logging)

    async def send_bytes(self, data: bytes, logging=True):
        if not self.conn or not self.conn["connected"]:
            vprint("Client: Not connected, reconnecting...", 2)
            await self.connect()

        start_marker = b"<START>"
        end_marker = b"<END>"
        chunk_size = 5 * 1024 * 1024

        await self.conn["from_client_queue"].put(start_marker)
        await self.conn["from_client_queue"].put(data)
        await self.conn["from_client_queue"].put(end_marker)

    async def receive_message(self) -> str:
        if not self.conn or not self.conn["connected"]:
            vprint("Client: Not connected, reconnecting...", 2)
            await self.connect()

        data = b""
        start_marker = b"<START>"
        end_marker = b"<END>"
        receiving = False

        try:
            while True:
                chunk = await self.conn["to_client_queue"].get()
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
            vprint(f"Client: Error receiving data: {e}", 0)
            return None

        return data.decode("utf-8")
