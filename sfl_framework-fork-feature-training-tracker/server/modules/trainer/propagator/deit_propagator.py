"""
DeiT Propagator (Server) — handles Tensor-only activation.

DeiT splits always produce plain tensors (batch, num_patches+1, embed_dim),
never tuples. This is a simplified version of FlexibleResnetPropagator.
"""

from typing import TYPE_CHECKING

import torch

from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class DeiTPropagator(BasePropagator):
    def __init__(self, model: "Module", config: "Config"):
        super().__init__()
        self.model = model
        self.config = config
        self.outputs: torch.Tensor = None
        self.inputs: torch.Tensor = None

    def forward(self, x: torch.Tensor, params: dict = None) -> torch.Tensor:
        self.inputs = x.to(self.config.device).requires_grad_(True)
        self.outputs = self.model(self.inputs)
        return self.outputs

    def backward(self, grads: torch.Tensor) -> torch.Tensor:
        if self.outputs is None:
            raise RuntimeError("Call forward() before backward()")
        grads = grads.to(self.config.device)
        self.outputs.backward(grads)
        if self.inputs.grad is not None:
            return self.inputs.grad.detach()
        return torch.zeros_like(self.inputs)
