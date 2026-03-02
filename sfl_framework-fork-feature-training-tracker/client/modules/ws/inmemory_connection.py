from client_args import Config

from utils.log_utils import vprint


class InMemoryConnection:
    """Zero-serialization in-memory connection for simulation mode (client side).

    All data is passed as Python objects directly through asyncio.Queue.
    No pickle, no hex encoding, no START/END markers.
    """

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
        """Send dict directly (no serialization in simulation mode)."""
        if not self.conn or not self.conn["connected"]:
            vprint("Client: Not connected, reconnecting...", 2)
            await self.connect()
        await self.conn["from_client_queue"].put(data)

    async def send_bytes(self, data, logging=True):
        """Send data directly through queue (no START/END markers)."""
        if not self.conn or not self.conn["connected"]:
            vprint("Client: Not connected, reconnecting...", 2)
            await self.connect()
        await self.conn["from_client_queue"].put(data)

    async def receive_message(self):
        """Receive message directly from queue.

        Returns whatever the server put: dict, bytes, or string.
        """
        if not self.conn or not self.conn["connected"]:
            vprint("Client: Not connected, reconnecting...", 2)
            await self.connect()
        return await self.conn["to_client_queue"].get()
