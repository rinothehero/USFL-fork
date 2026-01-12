from collections import deque
from typing import TYPE_CHECKING

from torch import nn

from .base_splitter import BaseSplitter

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module

    from .strategy.base_strategy import BaseStrategy


class VGGSplitter(BaseSplitter):
    def __init__(self, config: "Config", strategy: "BaseStrategy"):
        self.config = config
        self.strategy = strategy

    def split(self, model: "Module", params: dict):
        split_points = deque(self.strategy.find_split_points(model, params))
        submodels = []
        current_model = nn.ModuleDict()

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
                        current_model = nn.ModuleDict()

        recurse_split(model)

        if len(current_model) > 0:
            submodels.append(current_model)

        return submodels
