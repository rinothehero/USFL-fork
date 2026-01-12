from typing import TYPE_CHECKING

import torch

from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class VGGPropagator(BasePropagator):
    def __init__(self, model: torch.nn.ModuleDict, config: "Config"):
        super().__init__()
        self.model = model
        self.outputs: torch.Tensor = None
        self.config = config

        self.forward_mapper = {
            "avgpool": self.avgpool_forward,
        }

    def forward(self, x: torch.Tensor, params: dict = None) -> torch.Tensor:
        for layer_name, layer in self.model.items():
            layer_name = layer_name.split("-")[-1]
            mapped_forward = self.forward_mapper.get(layer_name)
            x = layer(x) if (mapped_forward is None) else mapped_forward(layer, x)

        self.outputs = x
        return self.outputs

    # Except for client, which contains the last part of the model.
    def backward(self, grads: torch.Tensor):
        grads = grads.to(self.config.device)

        self.outputs.requires_grad_(True)
        self.outputs.backward(grads)

    def avgpool_forward(
        self, layer: torch.nn.AdaptiveAvgPool2d, x: torch.Tensor
    ) -> torch.Tensor:
        x = layer(x)
        output = torch.flatten(x, 1)

        return output
