from typing import TYPE_CHECKING
from .base_strategy import BaseStrategy

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class NameStrategy(BaseStrategy):
    def __init__(self, config: "Config"):
        self.config = config

    def find_split_points(self, model: "Module", params: dict):
        if not hasattr(self.config, 'split_layer') or not self.config.split_layer:
            # Check params if config doesn't have it (fallback)
            if 'split_layer' in params:
                 return [params['split_layer']]
            raise ValueError("split_layer is not defined in config for layer_name strategy")
        return [self.config.split_layer]
