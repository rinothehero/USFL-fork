import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client_args import Config, ServerConfig
    from modules.trainer.model_trainer.base_model_trainer import BaseModelTrainer


class InRound:
    def __init__(self, config: "Config", server_config: "ServerConfig"):
        self.config = config
        self.server_config = server_config

    async def train(
        self,
        model_trainer: "BaseModelTrainer",
        params: dict = {},
    ):
        return await model_trainer.train(params=params)
