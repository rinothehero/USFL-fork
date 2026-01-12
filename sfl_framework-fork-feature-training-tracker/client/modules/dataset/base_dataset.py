from abc import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client_args import ServerConfig
    from torch.utils.data import DataLoader, Dataset


class BaseDataset(metaclass=ABCMeta):
    def get_class_name(self):
        return self.__class__.__name__

    @abstractmethod
    def __init__(self, config: "ServerConfig") -> None:
        pass

    @abstractmethod
    def _download_dataset(self) -> None:
        pass

    @abstractmethod
    def _create_loader(self) -> None:
        pass

    @abstractmethod
    def update_batch_size(self, batch_size: int) -> None:
        pass

    @abstractmethod
    def update_amount_per_label(self, dataset_size: dict[str, int]) -> None:
        pass

    @abstractmethod
    def reshuffle_dataset(self) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def get_trainset(self) -> "Dataset":
        pass

    @abstractmethod
    def get_trainloader(self) -> "DataLoader":
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass
