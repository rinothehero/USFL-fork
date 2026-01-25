from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

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
            print("No models to aggregate")
            return None
        if self.config.method == "nestfl" and self.config.aggregator == "fedavg":
            updated_model = aggregator.aggregate(models, params, True)
        else:
            updated_model = aggregator.aggregate(models, params)

        return updated_model

    def aggregate_models(self, aggregator: "BaseAggregator", model_queue: "ModelQueue"):
        ids, models, params = model_queue.get_all_models()

        if len(models) == 0:
            print("No models to aggregate")
            return None
        else:
            updated_model = aggregator.aggregate(models, params)
            return updated_model

    def update_global_model(self, updated_model: "torch.nn.Module", model: "BaseModel"):
        if updated_model is None:
            return

        if (
            hasattr(model, "client_model")
            and hasattr(model, "server_model")
            and hasattr(model, "sync_full_model_from_split")
        ):
            model.client_model.load_state_dict(updated_model.state_dict())
            model.sync_full_model_from_split()
            return

        torch_model = model.get_torch_model()
        torch_model.load_state_dict(updated_model.state_dict())

    def evaluate_global_model(self, model: "BaseModel", testloader: "DataLoader"):
        accuracy = model.evaluate(testloader)
        return accuracy
