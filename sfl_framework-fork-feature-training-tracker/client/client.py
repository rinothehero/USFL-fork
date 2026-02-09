from typing import TYPE_CHECKING

from modules.trainer.trainer import Trainer
from modules.ws.connection import Connection

from utils.log_utils import vprint

if TYPE_CHECKING:
    from client_args import Config


class Client:
    def __init__(self, config: "Config"):
        self.config = config
        self.connection = Connection(config)
        self.trainer = Trainer(config, self.connection)

    async def run(self):
        await self.connection.connect()
        vprint("Connected to server", 2)

        await self.trainer.train()
