from typing import TYPE_CHECKING

from client.modules.trainer.trainer import Trainer
from client.modules.ws.inmemory_connection import InMemoryConnection

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
            print("Connected to server")

            await self.trainer.train()
        except Exception as e:

            print(f"Error occurred: {e}")
            traceback.print_exc()
