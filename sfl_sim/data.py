"""
Dataset loading, client data distribution, and maskable dataset for USFL balancing.

Supports: CIFAR-10, CIFAR-100, FMNIST, MNIST, SVHN
Distribution: uniform (IID), dirichlet, shard_dirichlet
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Per-dataset normalization constants
_NORM = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "fmnist":   ((0.2860,), (0.3530,)),
    "mnist":    ((0.1307,), (0.3081,)),
    "svhn":     ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}

_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "fmnist": 10,
    "mnist": 10,
    "svhn": 10,
}


def load_dataset(
    name: str,
    data_dir: str = "./data",
) -> Tuple[Dataset, Dataset, int]:
    """
    Load train/test datasets with standard transforms.

    Returns: (trainset, testset, num_classes)
    """
    if name not in _NORM:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(_NORM)}")

    mean, std = _NORM[name]
    num_classes = _NUM_CLASSES[name]

    if name in ("cifar10", "cifar100"):
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        trainset = cls(root=data_dir, train=True, download=True, transform=train_tf)
        testset = cls(root=data_dir, train=False, download=True, transform=test_tf)

    elif name in ("fmnist", "mnist"):
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        cls = torchvision.datasets.FashionMNIST if name == "fmnist" else torchvision.datasets.MNIST
        trainset = cls(root=data_dir, train=True, download=True, transform=tf)
        testset = cls(root=data_dir, train=False, download=True, transform=tf)

    elif name == "svhn":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        trainset = torchvision.datasets.SVHN(
            root=data_dir, split="train", download=True, transform=train_tf)
        testset = torchvision.datasets.SVHN(
            root=data_dir, split="test", download=True, transform=test_tf)

    return trainset, testset, num_classes


def get_targets(dataset: Dataset) -> np.ndarray:
    """Extract label array from a dataset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    raise ValueError("Dataset has no targets or labels attribute")


def get_label_distribution(dataset: Dataset, indices) -> Dict[int, int]:
    """Compute per-label counts for a subset of a dataset."""
    from collections import Counter
    targets = get_targets(dataset)
    labels = [int(targets[i]) for i in indices]
    return dict(Counter(labels))


def get_testloader(testset: Dataset, batch_size: int = 128) -> DataLoader:
    """Create test DataLoader."""
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# Data distribution
# ---------------------------------------------------------------------------

ClientDataMasks = Dict[int, List[int]]


def distribute(
    trainset: Dataset,
    num_clients: int,
    method: str,
    *,
    alpha: float = 0.3,
    labels_per_client: int = 2,
    min_require_size: int = 10,
    seed: int = 42,
) -> ClientDataMasks:
    """
    Partition training data among clients.

    Args:
        trainset: Training dataset
        num_clients: Total number of clients
        method: "uniform", "dirichlet", or "shard_dirichlet"
        alpha: Dirichlet concentration parameter (lower = more Non-IID)
        labels_per_client: Number of classes per client (shard_dirichlet)
        min_require_size: Minimum samples per client
        seed: Random seed

    Returns: Dict mapping client_id to list of sample indices
    """
    rng = np.random.RandomState(seed)
    targets = get_targets(trainset)
    num_classes = len(np.unique(targets))

    if method == "uniform":
        return _distribute_uniform(targets, num_clients, rng)
    elif method == "dirichlet":
        return _distribute_dirichlet(targets, num_clients, num_classes, alpha,
                                     min_require_size, rng)
    elif method == "shard_dirichlet":
        return _distribute_shard_dirichlet(targets, num_clients, num_classes,
                                           labels_per_client, alpha,
                                           min_require_size, rng)
    else:
        raise ValueError(f"Unknown distribution method: {method}")


def _distribute_uniform(
    targets: np.ndarray,
    num_clients: int,
    rng: np.random.RandomState,
) -> ClientDataMasks:
    """IID uniform distribution."""
    indices = rng.permutation(len(targets))
    splits = np.array_split(indices, num_clients)
    return {i: splits[i].tolist() for i in range(num_clients)}


def _distribute_dirichlet(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    min_require_size: int,
    rng: np.random.RandomState,
) -> ClientDataMasks:
    """Non-IID Dirichlet distribution."""
    client_indices: ClientDataMasks = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)

        # Enforce minimum size: set tiny proportions to 0
        proportions = np.array([
            p if p * len(class_indices) >= min_require_size else 0.0
            for p in proportions
        ])
        total = proportions.sum()
        if total > 0:
            proportions /= total

        # Split indices according to proportions
        splits = (proportions * len(class_indices)).astype(int)
        # Distribute remainder
        remainder = len(class_indices) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1

        offset = 0
        for cid in range(num_clients):
            count = splits[cid]
            if count > 0:
                client_indices[cid].extend(class_indices[offset:offset + count].tolist())
            offset += count

    return client_indices


def _distribute_shard_dirichlet(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    labels_per_client: int,
    alpha: float,
    min_require_size: int,
    rng: np.random.RandomState,
) -> ClientDataMasks:
    """
    Shard + Dirichlet: each client gets exactly `labels_per_client` classes,
    within-class distribution follows Dirichlet(alpha).

    1. Assign classes to clients (each client gets exactly lpc classes)
    2. For each class, split data among assigned clients using Dirichlet
    """
    # Step 1: Assign classes to clients
    # Ensure each class is assigned to at least one client
    client_classes: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # Create a pool of (client_id, slot) pairs
    # Each client needs exactly labels_per_client classes
    all_slots = []
    for cid in range(num_clients):
        for _ in range(labels_per_client):
            all_slots.append(cid)
    rng.shuffle(all_slots)

    # Assign classes round-robin from shuffled slots
    class_clients: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    slot_idx = 0
    classes_shuffled = list(range(num_classes))

    # First pass: ensure every class has at least one client
    for c in classes_shuffled:
        if slot_idx >= len(all_slots):
            break
        cid = all_slots[slot_idx]
        if c not in client_classes[cid]:
            client_classes[cid].append(c)
            class_clients[c].append(cid)
            slot_idx += 1

    # Second pass: fill remaining slots
    while slot_idx < len(all_slots):
        cid = all_slots[slot_idx]
        needed = labels_per_client - len(client_classes[cid])
        if needed <= 0:
            slot_idx += 1
            continue
        # Pick a random class not yet assigned to this client
        available = [c for c in range(num_classes) if c not in client_classes[cid]]
        if not available:
            slot_idx += 1
            continue
        c = rng.choice(available)
        client_classes[cid].append(c)
        class_clients[c].append(cid)
        slot_idx += 1

    # Step 2: For each class, distribute data among assigned clients via Dirichlet
    client_indices: ClientDataMasks = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        assigned = class_clients[c]
        if not assigned:
            continue

        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)

        n_assigned = len(assigned)
        if n_assigned == 1:
            client_indices[assigned[0]].extend(class_idx.tolist())
            continue

        # Dirichlet split among assigned clients
        proportions = rng.dirichlet([alpha] * n_assigned)
        counts = (proportions * len(class_idx)).astype(int)

        # Ensure minimum size
        for i in range(n_assigned):
            counts[i] = max(counts[i], min(min_require_size, len(class_idx) // n_assigned))

        # Fix total to match available data
        total = counts.sum()
        if total > len(class_idx):
            # Scale down proportionally
            counts = (counts * len(class_idx) / total).astype(int)
        remainder = len(class_idx) - counts.sum()
        for i in range(abs(remainder)):
            counts[i % n_assigned] += 1 if remainder > 0 else -1

        offset = 0
        for i, cid in enumerate(assigned):
            count = max(0, counts[i])
            client_indices[cid].extend(class_idx[offset:offset + count].tolist())
            offset += count

    return client_indices


# ---------------------------------------------------------------------------
# MaskableDataset for USFL data balancing (trimming/replication)
# ---------------------------------------------------------------------------


class MaskableDataset(Dataset):
    """
    Dataset wrapper supporting per-label trimming and replication.

    - Trimming: randomly select target samples from available (not always first N)
    - Replication: cycle from random start position (not always position 0)
    - Uses numpy rng for full reproducibility
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: List[int],
        rng: Optional[np.random.RandomState] = None,
    ):
        self.base = base_dataset
        self.original_indices = list(indices)
        self.rng = rng if rng is not None else np.random.RandomState()

        # Extract targets
        if hasattr(base_dataset, "targets"):
            targets = base_dataset.targets
        elif hasattr(base_dataset, "labels"):
            targets = base_dataset.labels
        else:
            raise ValueError("Dataset has no targets or labels")

        # Build label -> original index mapping
        self._label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            label = int(targets[idx])
            self._label_to_indices[label].append(idx)

        # Active indices (modified by update_amount_per_label)
        self.active_indices = list(indices)

    def update_amount_per_label(self, amounts: Dict[int, int]):
        """
        Set target sample count per label.

        Trim: random subset of available samples (varies each call)
        Replicate: cycle from random start position
        """
        new_indices = []

        for label, target in amounts.items():
            available = self._label_to_indices.get(label, [])
            if not available or target <= 0:
                continue

            n = len(available)
            if target <= n:
                # Trim: random subset (not always first N)
                perm = self.rng.permutation(n)
                new_indices.extend(available[i] for i in perm[:target])
            else:
                # Replicate: all samples + cycle from random start for remainder
                new_indices.extend(available)
                remainder = target - n
                start = self.rng.randint(0, n)
                for i in range(remainder):
                    new_indices.append(available[(start + i) % n])

        self.rng.shuffle(new_indices)
        self.active_indices = new_indices

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        return self.base[self.active_indices[idx]]

    @property
    def label_distribution(self) -> Dict[int, int]:
        """Current label distribution of active indices."""
        if hasattr(self.base, "targets"):
            targets = self.base.targets
        else:
            targets = self.base.labels

        dist: Dict[int, int] = defaultdict(int)
        for idx in self.active_indices:
            dist[int(targets[idx])] += 1
        return dict(dist)
