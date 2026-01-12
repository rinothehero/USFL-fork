from abc import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader, Dataset


class BaseDataset(metaclass=ABCMeta):
    def get_class_name(self):
        return self.__class__.__name__

    @abstractmethod
    def __init__(self, config: "Config") -> None:
        pass

    @abstractmethod
    def _download_dataset(self) -> None:
        pass

    @abstractmethod
    def _create_loader(self) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def get_trainset(self) -> "Dataset":
        pass

    @abstractmethod
    def get_testset(self) -> "Dataset":
        pass

    @abstractmethod
    def get_trainloader(self) -> "DataLoader":
        pass

    @abstractmethod
    def get_testloader(self) -> "DataLoader":
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass
