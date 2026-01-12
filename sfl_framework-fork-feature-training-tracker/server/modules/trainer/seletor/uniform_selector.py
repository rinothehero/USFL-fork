import random
from typing import TYPE_CHECKING, List

from .base_selector import BaseSelector

if TYPE_CHECKING:
    from typing import List


class UniformSelector(BaseSelector):
    def select(self, n: int, client_ids: List[int], data=None):
        random.shuffle(client_ids)
        return client_ids[:n]
