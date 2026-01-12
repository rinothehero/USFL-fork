from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from server_args import ServerConfig


class BaseModelTrainer(metaclass=ABCMeta):
    def get_criterion(self, config: "ServerConfig"):
        if config.criterion == "ce":
            return nn.CrossEntropyLoss()
        elif config.criterion == "mse":
            return nn.MSELoss()
        elif config.criterion == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown criterion {config.criterion}")

    def get_optimizer(self, config: "ServerConfig"):
        if config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
            )
        elif config.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")

    @abstractmethod
    async def train(self, params: dict):
        pass
