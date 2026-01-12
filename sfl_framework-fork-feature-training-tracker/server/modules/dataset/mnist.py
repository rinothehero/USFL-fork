from typing import TYPE_CHECKING, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseDataset

if TYPE_CHECKING:
    from server_args import Config

class MNIST(BaseDataset):
    def __init__(self, config: "Config") -> None:
        self.initialized = False

        self.config = config
        self.trainset: Dataset | None = None
        self.testset: Dataset | None = None

        self.trainloader: DataLoader | None = None
        self.testloader: DataLoader | None = None

    def _download_dataset(self) -> Tuple[Dataset, Dataset]:
        if self.config.dataset == "mnist":
            train_dataset = datasets.MNIST(
                root=self.config.dataset_path,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )

            test_dataset = datasets.MNIST(
                root=self.config.dataset_path,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )

        elif self.config.dataset == "fmnist":
            train_dataset = datasets.FashionMNIST(
                root=self.config.dataset_path,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )

            test_dataset = datasets.FashionMNIST(
                root=self.config.dataset_path,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )

        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")

        return train_dataset, test_dataset

    def _create_loader(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = torch.utils.data.DataLoader(
            dataset=self.trainset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        return train_loader, test_loader

    def initialize(self) -> None:
        self.trainset, self.testset = self._download_dataset()
        self.trainloader, self.testloader = self._create_loader()
        self.initialized = True

    def get_trainset(self) -> Dataset:
        return self.trainset

    def get_testset(self) -> Dataset:
        return self.testset

    def get_trainloader(self) -> DataLoader:
        return self.trainloader

    def get_testloader(self) -> DataLoader:
        return self.testloader

    def get_num_classes(self) -> int:
        if self.config.dataset in ["mnist", "fmnist"]:
            return 10
        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")
