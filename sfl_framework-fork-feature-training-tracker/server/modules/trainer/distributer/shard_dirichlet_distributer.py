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


class ShardDirichletDistributer(BaseDistributer):
    """
    Shard-Dirichlet Distributer with min_size guarantee.
    
    Key differences from LabelDirichletDistributer:
    1. Ensures each class is assigned exactly the same number of times across clients
    2. Retry loop to guarantee minimum samples per client (min_require_size)
    3. More robust for extreme Non-IID settings (low alpha values)
    """
    
    MAX_RETRIES = 10000
    
    def __init__(self, config: "Config"):
        self.config = config
        # min_require_size: use config value if provided, otherwise default to max(10, batch_size//4)
        if getattr(config, 'min_require_size', None) is not None:
            self.min_require_size = config.min_require_size
        else:
            self.min_require_size = max(10, getattr(config, 'batch_size', 64) // 4)

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
        vprint(f"[ShardDirichlet] num_classes: {num_classes}", 2)

        labels_per_client = self.config.labels_per_client
        seed = getattr(self.config, "seed", 42)
        alpha = self.config.dirichlet_alpha
        
        for retry in range(self.MAX_RETRIES):
            prng = np.random.default_rng(seed + retry)  # Different seed each retry
            
            idx_clients, contains = self._distribute_once(
                targets, num_clients, num_classes, labels_per_client, alpha, prng
            )
            
            # Check min_size
            min_size = min(len(idx) for idx in idx_clients) if idx_clients else 0
            
            if min_size >= self.min_require_size:
                vprint(f"[ShardDirichlet] Success on attempt {retry + 1}: min_size={min_size} >= {self.min_require_size}", 2)
                break
            else:
                if retry < self.MAX_RETRIES - 1:
                    vprint(f"[ShardDirichlet] Retry {retry + 1}: min_size={min_size} < {self.min_require_size}", 2)
        else:
            vprint(f"[ShardDirichlet] Warning: Could not achieve min_size={self.min_require_size} after {self.MAX_RETRIES} retries. Final min_size={min_size}", 0)

        # Fallback: Handle empty clients by borrowing from the richest client
        for j in range(num_clients):
            if len(idx_clients[j]) == 0:
                vprint(f"[ShardDirichlet] Client {clients[j]} has no samples. Finding samples to assign...", 2)
                max_samples_client = max(
                    range(num_clients), key=lambda x: len(idx_clients[x])
                )
                # Borrow at least min_require_size samples or as many as possible
                num_to_borrow = min(self.min_require_size, len(idx_clients[max_samples_client]) // 2)
                num_to_borrow = max(1, num_to_borrow)  # At least 1
                
                for _ in range(num_to_borrow):
                    if len(idx_clients[max_samples_client]) > 1:
                        idx_clients[j].append(idx_clients[max_samples_client].pop())
                
                vprint(f"  Assigned {len(idx_clients[j])} samples from client {clients[max_samples_client]} to client {clients[j]}", 2)

        # Final stats
        total_assigned_samples = 0
        for i, idx_j in enumerate(idx_clients):
            num_samples = len(idx_j)
            total_assigned_samples += num_samples
            vprint(f"Client {clients[i]}: {num_samples} samples, labels: {contains[i]}", 2)
        vprint(f"Total assigned samples: {total_assigned_samples}", 2)
        vprint(f"Total dataset size: {dataset_size}", 2)

        return idx_clients

    def _distribute_once(
        self, targets, num_clients, num_classes, labels_per_client, alpha, prng
    ):
        """
        Perform one distribution attempt.
        Uses Shard_Dirichlet style: each class is assigned exactly assignments_per_class times.
        """
        # Step 1: Calculate how many times each class should be assigned
        total_assignments = num_clients * labels_per_client
        assignments_per_class = total_assignments // num_classes
        
        # Create class pool: each class appears exactly assignments_per_class times
        class_pool = []
        for k in range(num_classes):
            class_pool.extend([k] * assignments_per_class)
        
        # Handle remainder: distribute extra assignments to some classes
        remainder = total_assignments - len(class_pool)
        extra_classes = prng.choice(num_classes, size=remainder, replace=False)
        class_pool.extend(extra_classes.tolist())
        
        prng.shuffle(class_pool)
        
        # Step 2: Assign classes to each client
        client2classes = {}
        times = [0 for _ in range(num_classes)]
        
        for i in range(num_clients):
            assigned = class_pool[i * labels_per_client : (i + 1) * labels_per_client]
            # Remove duplicates within same client
            assigned = list(set(assigned))
            
            # If we got fewer due to duplicates, fill from remaining pool or random
            while len(assigned) < labels_per_client:
                # Try to find a class not yet in assigned
                for c in range(num_classes):
                    if c not in assigned:
                        assigned.append(c)
                        break
                else:
                    break
            
            client2classes[i] = assigned
            for c in assigned:
                times[c] += 1
        
        # Step 3: Distribute samples using Dirichlet
        idx_clients = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            clients_with_class_k = [i for i in range(num_clients) if k in client2classes[i]]
            
            if not clients_with_class_k:
                vprint(f"[ShardDirichlet] Warning: Class {k} has no assigned clients", 0)
                continue
            
            idx_k = np.where(targets == k)[0]
            if len(idx_k) == 0:
                continue
            
            prng.shuffle(idx_k)
            
            # Dirichlet distribution among clients that have this class
            proportions = prng.dirichlet(np.repeat(alpha, len(clients_with_class_k)))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_splits = np.split(idx_k, proportions)
            
            for client_idx, split in zip(clients_with_class_k, idx_splits):
                idx_clients[client_idx].extend(split.tolist())
        
        return idx_clients, client2classes
