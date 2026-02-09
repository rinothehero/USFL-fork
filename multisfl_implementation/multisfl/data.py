from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision
import torchvision.transforms as transforms

from .log_utils import vprint


@dataclass
class ClientData:
    client_id: int
    dataset: Any
    class_to_indices: Dict[int, List[int]]


class SyntheticImageDataset(Dataset):
    def __init__(self, n: int, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        self.targets = self.y.tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.x[idx], int(self.y[idx])


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        augment: bool = True,
        download: bool = True,
        use_sfl_transform: bool = False,
    ):
        if use_sfl_transform:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            )

            if train and augment:
                transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                )

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.targets = self.dataset.targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self.dataset[idx]


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        augment: bool = True,
        download: bool = True,
    ):
        # FMNIST mean and std (grayscale)
        normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])

        if train and augment:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    # Pad to 32x32 to match CIFAR/AlexNet input size expectations
                    transforms.Resize(32),
                    # Keep 1 channel for FMNIST models
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                    transforms.Resize(32),
                    # Keep 1 channel for FMNIST models
                ]
            )

        self.dataset = torchvision.datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )
        self.targets = self.dataset.targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self.dataset[idx]


class SubsetWithTargets(Subset):
    def __init__(self, dataset: Any, indices: List[int], targets: List[int]):
        super().__init__(dataset, indices)
        self.targets = targets


def build_class_to_indices(
    targets: Sequence[int], num_classes: int
) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        if 0 <= label < num_classes:
            class_to_indices[label].append(idx)
    return class_to_indices


def _get_targets_from_dataset(dataset: Any) -> List[int]:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.tolist()
        return list(targets)
    return [int(dataset[i][1]) for i in range(len(dataset))]


def partition_iid(
    dataset: Any, num_clients: int, num_classes: int = 10, seed: int = 0
) -> List[ClientData]:
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)

    all_targets = _get_targets_from_dataset(dataset)

    clients = []
    for cid, idxs in enumerate(splits):
        idxs_list = idxs.tolist()
        subset_targets = [all_targets[i] for i in idxs_list]
        subset = SubsetWithTargets(dataset, idxs_list, subset_targets)
        c2i = build_class_to_indices(subset_targets, num_classes)
        clients.append(ClientData(client_id=cid, dataset=subset, class_to_indices=c2i))

    return clients


def partition_dirichlet(
    dataset: Any,
    num_clients: int,
    num_classes: int = 10,
    alpha: float = 0.3,
    seed: int = 0,
    min_samples_per_client: int = 10,
) -> List[ClientData]:
    """
    Dirichlet non-IID partitioning: for each class c, draw proportions ~ Dir(alpha)
    and allocate samples proportionally. Lower alpha = more heterogeneous.
    """
    rng = np.random.default_rng(seed)
    all_targets = np.array(_get_targets_from_dataset(dataset))

    min_size = 0
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    while min_size < min_samples_per_client:
        class_indices = {c: np.where(all_targets == c)[0] for c in range(num_classes)}
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            c_idxs = class_indices[c].copy()
            rng.shuffle(c_idxs)

            proportions = rng.dirichlet([alpha] * num_clients)
            n_samples = len(c_idxs)
            counts = (proportions * n_samples).astype(int)

            diff = n_samples - counts.sum()
            if diff > 0:
                top_clients = np.argsort(-proportions)[:diff]
                for tc in top_clients:
                    counts[tc] += 1
            elif diff < 0:
                for _ in range(-diff):
                    for tc in np.argsort(proportions):
                        if counts[tc] > 0:
                            counts[tc] -= 1
                            break

            start = 0
            for client_id, count in enumerate(counts):
                end = start + count
                client_indices[client_id].extend(c_idxs[start:end].tolist())
                start = end

        sizes = [len(ci) for ci in client_indices]
        min_size = min(sizes)

    clients = []
    for cid, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            idxs = [0]

        subset_targets = [int(all_targets[i]) for i in idxs]
        subset = SubsetWithTargets(dataset, idxs, subset_targets)
        c2i = build_class_to_indices(subset_targets, num_classes)
        clients.append(ClientData(client_id=cid, dataset=subset, class_to_indices=c2i))

    return clients


def partition_shard_dirichlet(
    dataset: Any,
    num_clients: int,
    num_classes: int = 10,
    shards: int = 2,
    alpha: float = 0.3,
    seed: int = 0,
    min_samples_per_client: int = 10,
) -> List[ClientData]:
    """
    Shard-Dirichlet partitioning:
    1. Each client is assigned exactly `shards` distinct classes.
    2. For each class c, its samples are distributed among the clients who "own" that class
       using a Dirichlet distribution with parameter `alpha`.
    """
    rng = np.random.default_rng(seed)
    all_targets = np.array(_get_targets_from_dataset(dataset))

    min_size = 0
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    while min_size < min_samples_per_client:
        # 1. Assign classes to clients (ensure every class is covered at least once)
        client_classes = [set() for _ in range(num_clients)]

        for cid in range(num_clients):
            chosen = rng.choice(num_classes, size=shards, replace=False)
            client_classes[cid].update(chosen)

        class_owners = {c: [] for c in range(num_classes)}
        for cid, classes in enumerate(client_classes):
            for c in classes:
                class_owners[c].append(cid)

        for c in range(num_classes):
            if not class_owners[c]:
                cid = rng.integers(0, num_clients)
                client_classes[cid].add(c)
                class_owners[c].append(cid)

        # 2. Distribute samples
        class_indices = {c: np.where(all_targets == c)[0] for c in range(num_classes)}
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            owners = class_owners[c]
            if not owners:
                continue

            c_idxs = class_indices[c].copy()
            rng.shuffle(c_idxs)

            n_samples = len(c_idxs)
            n_owners = len(owners)

            proportions = rng.dirichlet([alpha] * n_owners)
            counts = (proportions * n_samples).astype(int)

            diff = n_samples - counts.sum()
            if diff > 0:
                top_owners = np.argsort(-proportions)[:diff]
                for i in top_owners:
                    counts[i] += 1
            elif diff < 0:
                for _ in range(-diff):
                    for i in np.argsort(proportions):
                        if counts[i] > 0:
                            counts[i] -= 1
                            break

            start = 0
            for i, count in enumerate(counts):
                owner_id = owners[i]
                end = start + count
                client_indices[owner_id].extend(c_idxs[start:end].tolist())
                start = end

        sizes = [len(ci) for ci in client_indices]
        min_size = min(sizes)

    clients = []
    for cid, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            idxs = [0]

        rng.shuffle(idxs)

        subset_targets = [int(all_targets[i]) for i in idxs]
        subset = SubsetWithTargets(dataset, idxs, subset_targets)
        c2i = build_class_to_indices(subset_targets, num_classes)
        clients.append(ClientData(client_id=cid, dataset=subset, class_to_indices=c2i))

    return clients


def print_partition_stats(clients: List[ClientData], num_classes: int = 10) -> None:
    vprint("\n" + "=" * 70, 2)
    vprint("PARTITION STATISTICS", 2)
    vprint("=" * 70, 2)

    total_samples = sum(len(cd.dataset) for cd in clients)
    sizes = [len(cd.dataset) for cd in clients]
    vprint(f"Total clients: {len(clients)}, Total samples: {total_samples}", 2)
    vprint(
        f"Samples per client: min={min(sizes)}, max={max(sizes)}, mean={total_samples / len(clients):.1f}", 2
    )

    vprint("\nPer-client class distribution (top 5 classes by count):", 2)
    vprint("-" * 70, 2)

    for cd in clients[: min(10, len(clients))]:
        class_counts = np.array(
            [len(cd.class_to_indices[c]) for c in range(num_classes)]
        )
        total = int(class_counts.sum())

        top5_classes = np.argsort(-class_counts)[:5]
        top5_str = ", ".join(
            [f"c{c}:{class_counts[c]}" for c in top5_classes if class_counts[c] > 0]
        )

        max_frac = class_counts.max() / max(total, 1)
        vprint(
            f"Client {cd.client_id:3d}: n={total:4d}, max_class_frac={max_frac:.2f}, top5=[{top5_str}]", 2
        )

    if len(clients) > 10:
        vprint(f"... and {len(clients) - 10} more clients", 2)

    global_counts = np.zeros(num_classes, dtype=int)
    for cd in clients:
        for c in range(num_classes):
            global_counts[c] += len(cd.class_to_indices[c])

    vprint(f"\nGlobal class distribution: {global_counts.tolist()}", 2)
    vprint("=" * 70 + "\n", 2)


def get_cifar10_test_loader(
    batch_size: int = 128, root: str = "./data", use_sfl_transform: bool = False
) -> DataLoader:
    test_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        augment=False,
        download=True,
        use_sfl_transform=use_sfl_transform,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def get_fmnist_test_loader(batch_size: int = 128, root: str = "./data") -> DataLoader:
    test_dataset = FashionMNISTDataset(
        root=root, train=False, augment=False, download=True
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def get_synthetic_test_loader(
    n: int = 1000, num_classes: int = 10, batch_size: int = 128, seed: int = 999
) -> DataLoader:
    test_dataset = SyntheticImageDataset(n=n, num_classes=num_classes, seed=seed)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
