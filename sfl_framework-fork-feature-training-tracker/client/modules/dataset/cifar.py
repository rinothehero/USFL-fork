from typing import TYPE_CHECKING, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseDataset
from .maskable_dataset import MaskableDataset

if TYPE_CHECKING:
    from client_args import ServerConfig


class CIFAR(BaseDataset):
    def __init__(self, config: "ServerConfig") -> None:
        self.initialized = False

        self.config = config
        self.mask_ids = config.mask_ids
        self.trainset: MaskableDataset | None = None
        self.trainloader: DataLoader | None = None
        self._loader_seed = self._build_loader_seed()
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(self._loader_seed)

    def _build_loader_seed(self) -> int:
        base_seed = int(getattr(self.config, "seed", 0))
        mask_ids = list(getattr(self.config, "mask_ids", []) or [])
        checksum = 0
        for idx, value in enumerate(mask_ids[:2048]):
            checksum = (checksum + (idx + 1) * int(value)) % 2147483647
        return (base_seed + checksum) % 2147483647

    def _download_dataset(self) -> MaskableDataset:
        transform = None

        if self.config.model == "mobilenet":
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        else:
            transform = transforms.ToTensor()

        if self.config.dataset == "cifar10":
            train_dataset = MaskableDataset(
                datasets.CIFAR10(
                    root=self.config.dataset_path,
                    train=True,
                    download=True,
                    transform=transform,
                ),
                self.mask_ids,
            )
        elif self.config.dataset == "cifar100":
            train_dataset = MaskableDataset(
                datasets.CIFAR100(
                    root=self.config.dataset_path,
                    train=True,
                    download=True,
                    transform=transform,
                ),
                self.mask_ids,
            )
        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")

        return train_dataset

    def _create_loader(self) -> DataLoader:
        if self.config.use_dynamic_batch_scheduler:
            # Dynamic scheduler: Fetch one sample at a time for manual batching
            train_loader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=1,  # Fetch one sample at a time for manual batching
                shuffle=False,  # Shuffling is handled by MaskableDataset's limited_indices
                drop_last=False,  # Not needed for manual batching
                pin_memory=False,
            )
        else:
            # Original scheduler: Use configured batch size with shuffling
            train_loader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._loader_generator,
                drop_last=False,  # Server handles batch_size=1 with eval mode
                pin_memory=False,
            )

        return train_loader

    def update_batch_size(self, batch_size: int) -> None:
        self.config.batch_size = batch_size
        self.trainloader = self._create_loader()

    # It limits the dataset amount per each label
    def update_amount_per_label(self, amount_limit_per_label: dict[int, int]) -> None:
        self.trainset.update_amount_per_label(amount_limit_per_label)
        self.trainloader = self._create_loader()

    def reshuffle_dataset(self) -> None:
        self.trainset.update_limited_indices()
        self.trainloader = self._create_loader()

    def initialize(self) -> None:
        self.trainset = self._download_dataset()
        self.trainloader = self._create_loader()
        self.initialized = True

    def get_trainset(self) -> Dataset:
        return self.trainset

    def get_trainloader(self) -> DataLoader:
        return self.trainloader

    def get_num_classes(self) -> int:
        if self.config.dataset == "cifar10":
            return 10
        elif self.config.dataset == "cifar100":
            return 100
        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")
