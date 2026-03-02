"""
DeiT Propagator (Client) — handles Tensor-only output.

DeiT client always outputs a plain Tensor (batch, num_patches+1, embed_dim).
Simplified version of FlexibleResnetPropagator (no tuple handling).
"""

from typing import TYPE_CHECKING, Optional

import torch

from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from client_args import Config
    from torch.nn import Module


class DeiTPropagator(BasePropagator):
    def __init__(self, model: "Module", config: "Config"):
        super().__init__()
        self.model = model
        self.config = config
        self.outputs: torch.Tensor = None

    def forward(self, x: torch.Tensor, params: Optional[dict] = None) -> torch.Tensor:
        self.outputs = self.model(x)
        return self.outputs.clone().detach().requires_grad_(True)

    def backward(self, grads: torch.Tensor):
        if self.outputs is None:
            raise RuntimeError("Call forward() before backward()")
        grads = grads.to(self.config.device)
        self.outputs.requires_grad_(True)
        self.outputs.backward(grads)
