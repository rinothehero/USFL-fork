from typing import TYPE_CHECKING

import numpy as np
import torch

from .base_distributer import BaseDistributer

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import Dataset


class DirichletDistributer(BaseDistributer):
    def __init__(self, config: "Config"):
        self.config = config

    def distribute(self, dataset: "Dataset", clients: list[str]):
        dataset_size = len(dataset)
        num_clients = len(clients)

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

        num_classes = len(set(tmp_t) - {int("9999999")})
        total_samples = len([x for x in tmp_t if x != int("9999999")])

        min_required_samples_per_client = 1
        min_samples = 0
        seed = self.config.seed if hasattr(self.config, "seed") else 42
        prng = np.random.default_rng(seed)

        while min_samples < min_required_samples_per_client:
            idx_clients = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(tmp_t == k)[0]
                prng.shuffle(idx_k)
                proportions = prng.dirichlet(
                    np.repeat(self.config.dirichlet_alpha, num_clients)
                )

                proportions = np.array(
                    [
                        p * (len(idx_j) < total_samples / num_clients)
                        for p, idx_j in zip(proportions, idx_clients)
                    ]
                )

                total_proportions = proportions.sum()
                if total_proportions == 0:
                    proportions = np.ones(num_clients) / num_clients
                else:
                    proportions = proportions / total_proportions

                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_k_split = np.split(idx_k, proportions)

                idx_clients = [
                    idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
                ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

        total_assigned_samples = 0
        for i, idx_j in enumerate(idx_clients):
            num_samples = len(idx_j)
            total_assigned_samples += num_samples
            print(f"Client {clients[i]}: {num_samples} samples")
        print(f"Total assigned samples: {total_assigned_samples}")
        print(f"Total dataset size: {dataset_size}")

        client_indices_list = idx_clients
        return client_indices_list
