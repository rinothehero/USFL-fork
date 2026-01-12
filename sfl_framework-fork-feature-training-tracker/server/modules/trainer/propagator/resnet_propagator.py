import copy
from typing import TYPE_CHECKING, Tuple, Union

import torch

from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class ResnetPropagator(BasePropagator):
    def __init__(self, model: torch.nn.ModuleDict, config: "Config"):
        super().__init__()
        self.model = model
        self.outputs: torch.Tensor = None
        self.config = config

        self.forward_mapper = {
            "conv1": self.conv1_forward,
            "bn1": self.bn1_forward,
            "relu": self.relu_forward,
            "conv2": self.conv2_forward,
            "bn2": self.bn2_forward,
            "downsample0": self.downsample_0_forward,
            "downsample1": self.downsample_1_forward,
            "avgpool": self.avgpool_forward,
        }

    def forward(
        self, x: torch.Tensor, params: dict = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        for layer_name, layer in self.model.items():
            split_name = (
                layer_name.split("-")[-2] + layer_name.split("-")[-1]
                if "downsample" in layer_name
                else layer_name.split("-")[-1]
            )

            mapped_forward = None
            if ("layer" in layer_name) or ("avgpool" in layer_name):
                mapped_forward = self.forward_mapper.get(split_name)

            if mapped_forward is None:
                x = layer(x)
            else:
                if (
                    mapped_forward in [self.avgpool_forward, self.conv1_forward]
                ) and isinstance(x, Tuple):
                    x = x[0] + x[1]
                    relu = torch.nn.ReLU(inplace=True)
                    x = relu(x)
                x = mapped_forward(layer, x)

        self.outputs = x
        return self.outputs

    # Except for client, which contains the last part of the model.
    def backward(self, grads: torch.Tensor):
        if isinstance(self.outputs, Tuple):
            self.outputs = self.outputs[0]

        grads = grads.to(self.config.device)

        self.outputs.requires_grad_(True)
        self.outputs.backward(grads)

    # (conv1): Conv2d() in BasicBlock
    def conv1_forward(
        self, layer: torch.nn.Conv2d, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        output = layer(x)

        return output, identity

    # (bn1): BatchNorm2d() in BasicBlock
    def bn1_forward(
        self, layer: torch.nn.BatchNorm2d, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, identity = input_info
        output = layer(x)

        return output, identity

    # (relu): ReLU() in BasicBlock
    def relu_forward(
        self, layer: torch.nn.ReLU, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, identity = input_info
        output = layer(x)

        return output, identity

    # (conv2): Conv2d() in BasicBlock
    def conv2_forward(
        self, layer: torch.nn.Conv2d, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, identity = input_info
        output = layer(x)

        return output, identity

    # (bn2): BatchNorm2d() in BasicBlock
    def bn2_forward(
        self, layer: torch.nn.BatchNorm2d, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, identity = input_info
        output = layer(x)

        return output, identity

    # (downsample): Sequential() in BasicBlock
    def downsample_0_forward(
        self, layer: torch.nn.Conv2d, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, x = input_info
        identity = layer(x)

        return output, identity

    # (downsample): Sequential() in BasicBlock
    def downsample_1_forward(
        self, layer: torch.nn.BatchNorm2d, input_info: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, x = input_info
        identity = layer(x)

        return output, identity

    # (avgpool): AdaptiveAvgPool2d()
    def avgpool_forward(
        self, layer: torch.nn.AdaptiveAvgPool2d, x: torch.Tensor
    ) -> torch.Tensor:
        x = layer(x)
        output = torch.flatten(x, 1)

        return output
