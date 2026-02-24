import asyncio
import os
import random
import time
from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import torch

from client_args import ServerConfig

from utils.log_utils import vprint

from ..dataset.dataset import get_dataset
from .apis.common import CommonAPI
from .stage.stage_organizer import get_stage_organizer

if TYPE_CHECKING:
    from client_args import Config

    from ..ws.connection import Connection


class Trainer:
    def __init__(self, config: "Config", connection: "Connection"):
        self.stage = "INITIALIZE"  # INITIALIZE, PRE_ROUND, IN_ROUND, POST_ROUND, FINISH
        self.config = config
        self.server_config = None
        self.connection = connection

        self.api = CommonAPI(connection)

        self.model = None
        self.dataset = None
        self.stage_organizer = None

    def _set_client_seed(self, seed: int) -> None:
        # Make per-client RNG stream deterministic and independent.
        seed = int(seed) + int(getattr(self.config, "client_id", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        vprint(f"[Client] Deterministic seed set: {seed}", 2)

    async def initialize(self):
        server_config = await self.api.request_server_config()
        # Server config may include server-only extension keys that clients do
        # not consume. Filter unknown keys to keep client bootstrapping robust.
        allowed_keys = {f.name for f in fields(ServerConfig)}
        filtered_server_config = {
            k: v for k, v in server_config.items() if k in allowed_keys
        }
        dropped_keys = sorted(
            [k for k in server_config.keys() if k not in allowed_keys]
        )
        if dropped_keys:
            vprint(
                f"[Client] Ignoring unknown server config keys: {', '.join(dropped_keys)}",
                2,
            )
        self.server_config = ServerConfig(**filtered_server_config)
        self._set_client_seed(getattr(self.server_config, "seed", 42))

        vprint("Initialized server config", 2)

        self.dataset = get_dataset(self.server_config)
        self.dataset.initialize()

        vprint("Initialized dataset", 2)

        self.stage_organizer = get_stage_organizer(
            self.config,
            self.server_config,
            self.api,
            self.dataset,
        )
        vprint("Initialized stage organizer", 2)
        self.stage = "PRE_ROUND"

    async def pre_round(self):
        kill = await self.stage_organizer.run_pre_round()
        if kill is True:
            self.stage = "FINISH"
        else:
            self.stage = "IN_ROUND"

    async def in_round(self):
        await self.stage_organizer.run_in_round()
        self.stage = "POST_ROUND"

    async def post_round(self):
        await self.stage_organizer.run_post_round()
        self.stage = "PRE_ROUND"

    async def train(self):
        while True:
            if self.stage == "INITIALIZE":
                await self.initialize()
            elif self.stage == "PRE_ROUND":
                await self.pre_round()
            elif self.stage == "IN_ROUND":
                await self.in_round()
            elif self.stage == "POST_ROUND":
                await self.post_round()
            elif self.stage == "FINISH":
                vprint("Training finished. Terminating process in client", 1)
                break
