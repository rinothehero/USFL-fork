from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


class BaseStrategy(ABC):
    @abstractmethod
    def find_split_points(self, model: "Module", params: dict):
        pass
