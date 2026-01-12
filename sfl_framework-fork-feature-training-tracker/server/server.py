import asyncio
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from modules.global_dict.global_dict import GlobalDict
from modules.trainer.trainer import Trainer
from modules.ws.connection import Connection
from modules.ws.endpoint import Endpoint
from modules.ws.handler.handler import get_handler

if TYPE_CHECKING:
    from server_args import Config


class Server:
    def __init__(self, config: "Config") -> None:
        self.app = FastAPI()
        self.config = config
        self.global_dict = GlobalDict(self.config)
        self.connection = Connection(self.config, self.global_dict)
        self.endpoint = Endpoint(self.config, self.connection)
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

        @self.app.websocket("/ws/{client_id}")
        async def _(websocket: WebSocket, client_id: str):
            await self.endpoint.websocket_endpoint(
                websocket,
                str(client_id),
            )

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            ws_ping_interval=20,
            ws_ping_timeout=None,
            ws_max_size=10 * 1024 * 1024,
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run(self):
        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        await self.trainer.initialize()

        tasks = [
            asyncio.create_task(self.run_server()),
            asyncio.create_task(self.trainer.train()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
