from typing import TYPE_CHECKING

from client.modules.trainer.trainer import Trainer
from client.modules.ws.inmemory_connection import InMemoryConnection

from utils.log_utils import vprint

if TYPE_CHECKING:
    from client_args import Config

import traceback


class SimulationClient:
    def __init__(self, config: "Config", connection: "InMemoryConnection"):
        self.config = config
        self.connection = connection
        self.trainer = Trainer(config, connection)

    async def run(self):
        try:
            await self.connection.connect()
            vprint("Connected to server", 2)

            await self.trainer.train()
        except Exception as e:

            vprint(f"Error occurred: {e}", 0)
            traceback.print_exc()
