from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn import functional as F

from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class MobileNetPropagator(BasePropagator):
    def __init__(self, model: torch.nn.ModuleDict, config: "Config"):
        super().__init__()
        self.model = model
        self.outputs: torch.Tensor = None
        self.config = config

        self.forward_mapper = {
            "classifier-0": self.classifier_forward,
        }

    def forward(self, x: torch.Tensor, params: dict = None) -> torch.Tensor:
        for layer_name, layer in self.model.items():
            mapped_forward = self.forward_mapper.get(layer_name)
            x = layer(x) if (mapped_forward is None) else mapped_forward(layer, x)

        self.outputs = x
        return self.outputs

    # Except for client, which contains the last part of the model.
    def backward(self, grads: torch.Tensor):
        grads = grads.to(self.config.device)

        self.outputs.requires_grad_(True)
        self.outputs.backward(grads)

    def classifier_forward(
        self, layer: torch.nn.Dropout, x: torch.Tensor
    ) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = layer(x)

        return x
