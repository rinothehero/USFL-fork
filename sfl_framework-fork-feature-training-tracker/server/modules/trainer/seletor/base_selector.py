from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from typing import List


class BaseSelector(ABC):
    @abstractmethod
    def select(self, n: int, client_ids: List[str], data):
        pass
