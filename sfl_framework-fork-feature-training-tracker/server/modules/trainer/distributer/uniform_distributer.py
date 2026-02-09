from typing import TYPE_CHECKING

import numpy as np
import torch

from .base_distributer import BaseDistributer
from utils.log_utils import vprint

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import Dataset


class UniformDistributer(BaseDistributer):
    def __init__(self, config: "Config"):
        self.config = config

    def distribute(self, dataset: "Dataset", clients: list[str]):
        dataset_size = len(dataset)
        if hasattr(dataset, "targets"):
            tmp_t = dataset.targets
        elif hasattr(dataset, "labels"):
            tmp_t = dataset.labels
        elif "label" in dataset.column_names:
            tmp_t = dataset["label"]
        else:
            raise ValueError("Dataset does not have targets or labels attribute")
        
        if isinstance(tmp_t, list):
            tmp_t = np.array(tmp_t)
        if isinstance(tmp_t, torch.Tensor):
            tmp_t = tmp_t.numpy()

        if self.config.delete_fraction_of_data:
            tmp_t = self._remove_fraction_of_labels(tmp_t, 0.5)

        indices = [i for i, x in enumerate(tmp_t) if x != int("9999999")]
        indices_size = len(indices)
        np.random.shuffle(indices)

        split_size = indices_size // len(clients)
        remainder = indices_size % len(clients)

        client_indices_list = []
        start_idx = 0
        for i in range(len(clients)):
            end_idx = start_idx + split_size + (1 if i < remainder else 0)
            client_indices = indices[start_idx:end_idx]
            client_indices_list.append(client_indices)
            start_idx = end_idx

        total_assigned_samples = 0
        for i, idx_j in enumerate(client_indices_list):
            num_samples = len(idx_j)
            total_assigned_samples += num_samples
            vprint(f"Client {clients[i]}: {num_samples} samples", 2)

        vprint(f"Total assigned samples: {total_assigned_samples}", 2)
        vprint(f"Total dataset size: {dataset_size}", 2)

        return client_indices_list
