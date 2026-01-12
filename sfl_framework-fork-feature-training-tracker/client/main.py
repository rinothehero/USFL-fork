import asyncio
from typing import TYPE_CHECKING

import torch
from client_args import parse_args

from client import Client

if TYPE_CHECKING:
    from client_args import Config


def _validate_device(config: "Config"):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is not available on this system")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device is not available on this system")


def validate_config(config: "Config"):
    _validate_device(config)


def main(config: "Config"):
    client_instance = Client(config)
    asyncio.run(client_instance.run())


if __name__ == "__main__":
    config = parse_args()
    validate_config(config)
    main(config)
