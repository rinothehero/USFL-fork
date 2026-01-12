from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


class BaseModel(torch.nn.Module):
    def get_class_name(self):
        return self.__class__.__name__

    def disable_inplace(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))
            elif isinstance(child, nn.ReLU6):
                setattr(module, name, nn.ReLU6(inplace=False))
            elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(child, "inplace"):
                    child.inplace = False
            else:
                self.disable_inplace(child)

    def __init__(self, config: "Config") -> None:
        super().__init__()
        pass

    def forward(self, inputs):
        pass

    def predict(self, inputs) -> torch.Tensor:
        pass

    def save_model(self, save_path: str) -> None:
        pass

    def load_model(self, load_path: str) -> None:
        pass

    def get_torch_model(self):
        pass
