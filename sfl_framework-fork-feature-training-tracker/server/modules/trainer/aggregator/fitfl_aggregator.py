import copy
import math
from typing import TYPE_CHECKING, List

import torch

from .base_aggregator import BaseAggregator

if TYPE_CHECKING:
    from typing import List


class FitFLAggregator(BaseAggregator):
    def _except_zero(
        self, models: List[torch.nn.Module], dataset_sizes: List[int], epochs: List[int]
    ):
        if not models:
            raise ValueError("No models to aggregate")

        epochs = [math.log(epoch) + 1 for epoch in epochs]
        total_datasets = {
            key: torch.zeros_like(params)
            for key, params in models[0].state_dict().items()
        }
        total_epochs = {
            key: torch.zeros_like(params)
            for key, params in models[0].state_dict().items()
        }

        for model, dataset_size, epoch in zip(models, dataset_sizes, epochs):
            model_state_dict = model.state_dict()
            for key, params in model_state_dict.items():
                total_dataset = total_datasets[key]
                mask = params != 0
                total_dataset[mask] += dataset_size
                total_datasets[key] = total_dataset

                total_epoch = total_epochs[key]
                total_epoch[mask] += epoch
                total_epochs[key] = total_epoch

        aggregated_state_dict = {}
        for key, param in models[0].state_dict().items():
            aggregated_state_dict[key] = torch.zeros_like(param, dtype=torch.float32)

        for model, dataset_size, epoch in zip(models, dataset_sizes, epochs):
            model_state_dict = model.state_dict()
            for key, param in model_state_dict.items():
                total_dataset = total_datasets[key]
                total_epoch = total_epochs[key]

                with torch.no_grad():
                    weight = torch.where(
                        total_dataset != 0,
                        ((dataset_size / total_dataset) + (epoch / total_epoch)) / 2,
                        torch.zeros_like(total_dataset),
                    )

                aggregated_state_dict[key] += param.float() * weight

        aggregated_model = copy.deepcopy(models[0])
        aggregated_model.load_state_dict(aggregated_state_dict)

        return aggregated_model

    def aggregate(
        self,
        models: List[torch.nn.Module],
        params: List[dict],
        except_zero=False,
    ):
        print(f"Params: {params}")
        dataset_sizes = [param["dataset_size"] for param in params]
        trained_epochs = [param["trained_epochs"] for param in params]
        print(f"Trained epochs: {trained_epochs}")

        if except_zero:
            print("Except zero parameters")
            return self._except_zero(models, dataset_sizes, trained_epochs)

        if not models:
            raise ValueError("No models to aggregate")

        epochs = [math.log(epoch) + 1 for epoch in trained_epochs]
        total_data_size = sum(dataset_sizes)
        total_epochs = sum(epochs)

        dataset_weights = [data_size / total_data_size for data_size in dataset_sizes]
        epoch_weights = [epoch / total_epochs for epoch in epochs]
        weights = [
            (dataset_weight + epoch_weight) / 2
            for dataset_weight, epoch_weight in zip(dataset_weights, epoch_weights)
        ]

        print(f"Dataset sizes: {dataset_sizes}")
        print(f"Epochs: {epochs}")
        print(f"Weights: {weights}")

        aggregated_state_dict = {}
        for key, param in models[0].state_dict().items():
            aggregated_state_dict[key] = torch.zeros_like(param, dtype=torch.float32)

        for model, weight in zip(models, weights):
            model_state_dict = model.state_dict()
            for key, param in model_state_dict.items():
                aggregated_state_dict[key] += param.float() * weight

        aggregated_model = copy.deepcopy(models[0])
        aggregated_model.load_state_dict(aggregated_state_dict)

        return aggregated_model
