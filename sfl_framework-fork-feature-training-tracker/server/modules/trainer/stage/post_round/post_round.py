from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from utils.log_utils import vprint

from ....global_dict.model_queue.model_queue import ModelQueue
from ....model.base_model import BaseModel
from ....trainer.aggregator.base_aggregator import BaseAggregator

if TYPE_CHECKING:
    from server_args import Config

    from ....global_dict.global_dict import GlobalDict


class PostRound:
    def __init__(self, config: "Config", global_dict: "GlobalDict"):
        self.config = config
        self.global_dict = global_dict

    def aggregate_torch_models(
        self,
        aggregator: "BaseAggregator",
        models: list["torch.nn.Module"],
        params: list[dict],
    ):
        if len(models) == 0:
            vprint("No models to aggregate", 2)
            return None
        if self.config.method == "nestfl" and self.config.aggregator == "fedavg":
            updated_model = aggregator.aggregate(models, params, True)
        else:
            updated_model = aggregator.aggregate(models, params)

        return updated_model

    def aggregate_models(self, aggregator: "BaseAggregator", model_queue: "ModelQueue"):
        ids, models, params = model_queue.get_all_models()

        if len(models) == 0:
            vprint("No models to aggregate", 2)
            return None
        else:
            updated_model = aggregator.aggregate(models, params)
            return updated_model

    def update_global_model(self, updated_model: "torch.nn.Module", model: "BaseModel"):
        torch_model = model.get_torch_model()
        torch_model.load_state_dict(updated_model.state_dict())

    def evaluate_global_model(self, model: "BaseModel", testloader: "DataLoader"):
        accuracy = model.evaluate(testloader)
        return accuracy
