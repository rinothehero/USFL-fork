from typing import TYPE_CHECKING

import numpy as np
import torch

from .base_distributer import BaseDistributer
from utils.log_utils import vprint

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import Dataset


def random_split_indices(
    indices: np.ndarray, k: int, prng: np.random.Generator, alpha: float = 1.0
):
    n = len(indices)
    if k <= 1 or n == 0:
        return [indices]

    prng.shuffle(indices)
    random_props = prng.dirichlet(np.full(k, alpha), 1)[0]
    sizes = np.floor(random_props * n).astype(int)

    zeros = np.sum(sizes == 0)

    current_sum = sizes.sum()
    remainder = n - current_sum

    if zeros > 0 and remainder >= zeros:
        for i in range(k):
            if sizes[i] == 0:
                sizes[i] = 1
                remainder -= 1

    for i in range(remainder):
        sizes[i % k] += 1

    result = []
    start = 0
    for size in sizes:
        result.append(indices[start : start + size])
        start += size

    return result


class LabelDirichletDistributer(BaseDistributer):
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
        seed = getattr(self.config, "seed", 42)
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
            idx_k_splits = random_split_indices(
                idx_k, times[i], prng, self.config.dirichlet_alpha
            )

            ids = 0
            for j in range(num_clients):
                if i in contains[j]:
                    idx_clients[j] += idx_k_splits[ids].tolist()
                    ids += 1

        for j in range(num_clients):
            if len(idx_clients[j]) == 0:
                vprint(
                    f"Client {clients[j]} has no samples. Finding samples to assign...", 0
                )
                max_samples_client = max(
                    range(num_clients), key=lambda x: len(idx_clients[x])
                )
                if len(idx_clients[max_samples_client]) > 1:
                    idx_clients[j].append(idx_clients[max_samples_client].pop())
                    vprint(
                        f"  Assigned 1 sample from client {clients[max_samples_client]} to client {clients[j]}", 2
                    )
                else:
                    vprint(f"  Could not find samples to assign to client {clients[j]}", 0)

        total_assigned_samples = 0
        for i, idx_j in enumerate(idx_clients):
            num_samples = len(idx_j)
            total_assigned_samples += num_samples
            vprint(f"Client {clients[i]}: {num_samples} samples, labels: {contains[i]}", 2)
        vprint(f"Total assigned samples: {total_assigned_samples}", 2)
        vprint(f"Total dataset size: {dataset_size}", 2)

        return idx_clients
