import copy
from typing import TYPE_CHECKING, List

import torch

from .base_aggregator import BaseAggregator
from ..utils.training_tracker import TrainingTracker

if TYPE_CHECKING:
    from typing import List


class FedAvgAggregator(BaseAggregator):
    def _except_zero(self, models: List[torch.nn.Module], params: List[dict]):
        if not models:
            raise ValueError("No models to aggregate")

        dataset_sizes = [param["dataset_size"] for param in params]

        total_datasets = {
            key: torch.zeros_like(params)
            for key, params in models[0].state_dict().items()
        }

        for model, dataset_size in zip(models, dataset_sizes):
            model_state_dict = model.state_dict()
            for key, params in model_state_dict.items():
                total_dataset = total_datasets[key]
                mask = params != 0
                total_dataset[mask] += dataset_size
                total_datasets[key] = total_dataset

        aggregated_state_dict = {}
        for key, param in models[0].state_dict().items():
            aggregated_state_dict[key] = torch.zeros_like(param, dtype=torch.float32)

        for model, dataset_size in zip(models, dataset_sizes):
            model_state_dict = model.state_dict()
            for key, param in model_state_dict.items():
                total_dataset = total_datasets[key]

                with torch.no_grad():
                    weight = torch.where(
                        total_dataset != 0,
                        dataset_size / total_dataset,
                        torch.zeros_like(total_dataset),
                    )

                aggregated_state_dict[key] += param.float() * weight

        aggregated_model = copy.deepcopy(models[0])
        aggregated_model.load_state_dict(aggregated_state_dict)

        return aggregated_model

    def aggregate(
        self, models: List[torch.nn.Module], params: List[dict], except_zero=False
    ):
        if except_zero:
            print("Except zero parameters")
            return self._except_zero(models, params)

        if not models:
            raise ValueError("No models to aggregate")

        # Extract round_number and client_ids from params
        round_number = params[0].get("round_number", 0) if params else 0
        client_ids = [p.get("client_id", i) for i, p in enumerate(params)]

        # Use augmented (actual used) data for weight calculation
        augmented_dists = [p.get("augmented_label_counts", {}) for p in params]
        dataset_sizes = [sum(d.values()) if d else p.get("dataset_size", 0) 
                         for d, p in zip(augmented_dists, params)]

        total_data_size = sum(dataset_sizes)
        weights = [data_size / total_data_size for data_size in dataset_sizes] if total_data_size > 0 else [1.0 / len(params)] * len(params)

        # Extract label distributions for logging (original)
        label_dists = [p.get("label_distribution", {}) for p in params]
        labels = sorted({lab for d in label_dists for lab in d})
        total_per_label = {
            lab: sum(d.get(lab, 0) for d in label_dists) for lab in labels
        }

        # Extract augmented (actual used) label distributions
        augmented_dists = [p.get("augmented_label_counts", {}) for p in params]
        augmented_total_per_label = {
            lab: sum(d.get(lab, 0) for d in augmented_dists) for lab in labels
        }

        # --- Log Aggregation Weights ---
        TrainingTracker.log_aggregation_weights(
            round_number=round_number,
            client_ids=client_ids,
            client_weights=weights,
            label_distributions=label_dists,
            total_per_label=total_per_label,
            augmented_label_distributions=augmented_dists,
            augmented_total_per_label=augmented_total_per_label,
        )
        # --- End Log ---

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
