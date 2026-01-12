from abc import ABC, abstractmethod
from typing import Callable


class BaseHandler(ABC):
    @abstractmethod
    def get_all_handler(self) -> dict[str, Callable]:
        pass
