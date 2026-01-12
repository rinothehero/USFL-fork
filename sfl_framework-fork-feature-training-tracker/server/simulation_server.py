import asyncio
import random
import traceback
from typing import TYPE_CHECKING

import numpy as np
import torch
from fastapi import FastAPI, WebSocket

from server.modules.global_dict.global_dict import GlobalDict
from server.modules.trainer.trainer import Trainer
from server.modules.ws.handler.handler import get_handler
from server.modules.ws.inmemory_connection import InMemoryConnection
from server.modules.ws.inmemory_endpoint import InMemoryEndpoint

if TYPE_CHECKING:
    from server_args import Config


class SimulationServer:
    def __init__(
        self,
        config: "Config",
        connection: "InMemoryConnection",
    ) -> None:
        self.config = config
        self.global_dict = GlobalDict(self.config)
        self.connection = connection
        self.connection.global_dict = self.global_dict
        self.endpoint = InMemoryEndpoint(self.config, self.connection)
        self.trainer = Trainer(self.config, self.connection, self.global_dict)

        self.handler = get_handler(self.config, self.global_dict)

    async def run_server(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        handlers = self.handler.get_all_handler()

        for name, handler in handlers.items():
            self.endpoint.add_handler(name, handler)

        client_ids = list(self.connection.active_connections.keys())
        endpoint_tasks = []
        for client_id in client_ids:
            endpoint_tasks.append(self.endpoint.simulate_endpoint(client_id))

        if len(endpoint_tasks) == 0:
            endpoint_tasks.append(self.endpoint.mock_endpoint())
        await asyncio.gather(
            *endpoint_tasks,
        )

    async def run(self):
        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        await self.trainer.initialize()

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(self.run_server()),
                asyncio.create_task(self.trainer.train()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            print(f"Task completed: {task}")
            try:
                result = task.result()
                print(f"Task result: {result}")
            except Exception as e:

                print(f"Task raised an exception: {e}")
                print(f"Exception traceback: {traceback.format_exc()}")

        for task in pending:
            print(f"Task still pending: {task}")
