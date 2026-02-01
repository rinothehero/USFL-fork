from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from typing import List


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        models: List[torch.nn.Module],
    ):
        pass
    
    def model_reshape(self, model: torch.nn.ModuleDict) -> torch.nn.Module:
        re_model = torch.nn.ModuleDict()
        
        def recurse_reshape(
            layer: torch.nn.Module, layer_name: str, parent: torch.nn.ModuleDict
        ):
            split_index = layer_name.find('-')
            if split_index != -1:
                current_name = layer_name[:split_index]
                child_name = layer_name[split_index + 1:]
                
                if current_name not in parent:
                    parent.update({current_name: torch.nn.ModuleDict()})
                
                recurse_reshape(layer, child_name, parent[current_name])
            else:
                parent.update({layer_name: layer})
        
        for layer_name, layer in model.items():
            split_index = layer_name.find('-')
            if split_index == -1:
                re_model.update({layer_name: layer})
            else:
                current_name = layer_name[:split_index]
                if current_name not in re_model:
                    re_model.update({current_name: torch.nn.ModuleDict()})
                child_name = layer_name[split_index + 1:]
                recurse_reshape(layer, child_name, re_model[current_name])
        
        return re_model