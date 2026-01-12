from typing import TYPE_CHECKING, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseDataset

if TYPE_CHECKING:
    from server_args import Config


class CIFAR(BaseDataset):
    def __init__(self, config: "Config") -> None:
        self.initialized = False

        self.config = config
        self.trainset: Dataset | None = None
        self.testset: Dataset | None = None

        self.trainloader: DataLoader | None = None
        self.testloader: DataLoader | None = None

    def _download_dataset(self) -> Tuple[Dataset, Dataset]:
        transform = None

        if self.config.model == "alexnet_legacy":  # Renamed to avoid triggering for custom AlexNetCifar
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
        elif self.config.model == "mobilenet":
            if self.config.dataset == "cifar10":
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                        ),
                    ]
                )
        else:
            transform = transforms.ToTensor()

        if self.config.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                root=self.config.dataset_path,
                train=True,
                download=True,
                transform=transform,
            )

            test_dataset = datasets.CIFAR10(
                root=self.config.dataset_path,
                train=False,
                download=True,
                transform=transform,
            )

        elif self.config.dataset == "cifar100":

            train_dataset = datasets.CIFAR100(
                root=self.config.dataset_path,
                train=True,
                download=True,
                transform=transform,
            )

            test_dataset = datasets.CIFAR100(
                root=self.config.dataset_path,
                train=False,
                download=True,
                transform=transform,
            )

        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")

        return train_dataset, test_dataset

    def _create_loader(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = torch.utils.data.DataLoader(
            dataset=self.trainset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
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
        if self.config.dataset == "cifar10":
            return 10
        elif self.config.dataset == "cifar100":
            return 100
        else:
            raise ValueError(f"Unknown dataset {self.config.dataset}")
