from collections import deque
from typing import TYPE_CHECKING

import torch
from torch import nn

from .base_splitter import BaseSplitter

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module

    from .strategy.base_strategy import BaseStrategy


class SequentialModuleDict(nn.ModuleDict):
    """ModuleDict with a forward() that iterates layers sequentially.

    This preserves .items() access for VGGPropagator while allowing
    the module to be called directly (e.g. for probe direction computation).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, layer in self.items():
            x = layer(x)
            # Flatten after AdaptiveAvgPool (matches VGGPropagator behavior)
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                x = torch.flatten(x, 1)
        return x


class VGGSplitter(BaseSplitter):
    def __init__(self, config: "Config", strategy: "BaseStrategy"):
        self.config = config
        self.strategy = strategy

    def split(self, model: "Module", params: dict):
        split_points = deque(self.strategy.find_split_points(model, params))
        submodels = []
        current_model = SequentialModuleDict()

        def recurse_split(parent, prefix=""):
            nonlocal current_model
            for name, child in parent.named_children():
                full_name = prefix + name

                if len(list(child.named_children())) != 0:
                    recurse_split(child, full_name + ".")
                else:
                    modified_name = full_name.replace(".", "-")
                    current_model[modified_name] = child

                    if len(split_points) > 0 and full_name == split_points[0]:
                        submodels.append(current_model)
                        split_points.popleft()
                        current_model = SequentialModuleDict()

        recurse_split(model)

        if len(current_model) > 0:
            submodels.append(current_model)

        return submodels
