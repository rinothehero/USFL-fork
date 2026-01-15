from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union, Optional

import torch

if TYPE_CHECKING:
    from torch.nn import Module


class BasePropagator(ABC, torch.nn.Module):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, params: Optional[dict] = None
    ) -> Union[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def backward(self, grads: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        pass
