from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, model: "Module", params: dict):
        pass
