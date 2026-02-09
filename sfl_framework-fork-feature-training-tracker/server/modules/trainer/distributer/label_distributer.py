from typing import TYPE_CHECKING

import numpy as np
import torch

from .base_distributer import BaseDistributer
from utils.log_utils import vprint

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import Dataset


class LabelDistributer(BaseDistributer):
    def __init__(self, config: "Config"):
        self.config = config

    def distribute(self, dataset: "Dataset", clients: list[str]):
        dataset_size = len(dataset)
        num_clients = len(clients)

        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        elif "label" in dataset.column_names:
            targets = dataset["label"]
        else:
            raise ValueError("Dataset does not have targets or labels attribute")

        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        if self.config.delete_fraction_of_data:
            targets = self._remove_fraction_of_labels(targets, 0.5)

        num_classes = len(set(targets) - {int("9999999")})
        vprint(f"num_classes: {num_classes}", 2)

        labels_per_client = self.config.labels_per_client
        seed = self.config.seed if hasattr(self.config, "seed") else 42
        prng = np.random.default_rng(seed)

        times = [0 for _ in range(num_classes)]
        contains = []

        class_list = np.arange(num_classes)
        prng.shuffle(class_list)
        class_cycle = np.tile(
            class_list, (labels_per_client * num_clients // num_classes + 1)
        )

        class_idx = 0
        for i in range(num_clients):
            current = []
            j = 0
            while j < labels_per_client:
                index = class_cycle[class_idx % len(class_cycle)]
                class_idx += 1

                if index not in current:
                    current.append(index)
                    times[index] += 1
                    j += 1
            contains.append(current)

        idx_clients = [[] for _ in range(num_clients)]

        for i in range(num_classes):
            if times[i] == 0:
                vprint(f"Class {i} was not assigned to any client.", 0)
                continue

            idx_k = np.where(targets == i)[0]
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(num_clients):
                if i in contains[j]:
                    idx_clients[j] += idx_k_split[ids].tolist()
                    ids += 1

        total_assigned_samples = 0
        for i, idx_j in enumerate(idx_clients):
            num_samples = len(idx_j)
            total_assigned_samples += num_samples
            vprint(f"Client {clients[i]}: {num_samples} samples, labels: {contains[i]}", 2)
        vprint(f"Total assigned samples: {total_assigned_samples}", 2)
        vprint(f"Total dataset size: {dataset_size}", 2)

        client_indices_list = idx_clients
        return client_indices_list
