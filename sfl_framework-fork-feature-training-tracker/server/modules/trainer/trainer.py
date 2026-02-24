import asyncio
import os
import random
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..dataset.dataset import get_dataset
from ..model.model import get_model
from .aggregator.aggregator import get_aggregator
from .distributer.distributer import get_distributer
from .seletor.selector import get_selector
from .splitter.splitter import get_splitter
from .stage.stage_organizer import get_stage_organizer
from utils.log_utils import vprint

if TYPE_CHECKING:
    from modules.global_dict.global_dict import GlobalDict
    from modules.ws.connection import Connection
    from server_args import Config


class Trainer:
    def __init__(
        self,
        config: "Config",
        connection: "Connection",
        global_dict: "GlobalDict",
    ):
        self.stage = "PRE_ROUND"  # PRE_ROUND, IN_ROUND, POST_ROUND, FINISH
        self.round = 0

        self.config = config

        self.connection = connection
        self.global_dict = global_dict

        self.aggregator = get_aggregator(config)
        self.selector = get_selector(config)
        self.splitter = get_splitter(config)

        self.dataset = get_dataset(config)
        self.distributer = get_distributer(config)
        self.model = get_model(config, self.dataset.get_num_classes())

        self.stage_organizer = None

    async def initialize(self):
        self.dataset.initialize()

        self.global_dict.set(
            "client_data_masks",
            self.distributer.distribute(
                self.dataset.get_trainset(), [i for i in range(self.config.num_clients)]
            ),
        )

        self.stage_organizer = get_stage_organizer(
            self.config,
            self.connection,
            self.global_dict,
            self.aggregator,
            self.model,
            self.dataset,
            self.selector,
            self.splitter,
        )

    async def pre_round(self):
        self.round += 1
        self.global_dict.set_round(self.round)
        self.global_dict.add_event("PRE_ROUND_START")

        await self.stage_organizer.run_pre_round(self.round)

        self.global_dict.add_event("PRE_ROUND_END")
        self.stage = "IN_ROUND"

    async def in_round(self):
        self.global_dict.add_event("IN_ROUND_START")

        await self.stage_organizer.run_in_round(self.round)

        self.global_dict.add_event("IN_ROUND_END")
        self.stage = "POST_ROUND"

    async def post_round(self):
        self.global_dict.add_event("POST_ROUND_START")

        await self.stage_organizer.run_post_round(self.round)

        self.global_dict.add_event("POST_ROUND_END")
        self.global_dict.save_metric()

        if self.round == self.config.global_round:
            self.stage = "FINISH"
        else:
            self.stage = "PRE_ROUND"

    async def finish(self):
        vprint("Training finished. Terminating process.", 1)
        self.global_dict.add_event("FINISH_START")
        all_clients = self.connection.get_all_clients()

        await asyncio.gather(
            *[
                self.connection.send_json(
                    data={"event": "kill_round"}, client_id=client
                )
                for client in all_clients
            ]
        )

        self.global_dict.add_event("FINISH_END")

    async def train(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

        while True:
            if self.stage == "PRE_ROUND":
                vprint("=" * 50, 2)
                await self.pre_round()
            elif self.stage == "IN_ROUND":
                await self.in_round()
            elif self.stage == "POST_ROUND":
                await self.post_round()
            elif self.stage == "FINISH":
                await self.finish()
                break
