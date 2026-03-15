"""
Dataset loading and client data distribution.

Supports: CIFAR-10, CIFAR-100, FMNIST, MNIST, SVHN
Distribution: uniform (IID), dirichlet, shard_dirichlet
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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
