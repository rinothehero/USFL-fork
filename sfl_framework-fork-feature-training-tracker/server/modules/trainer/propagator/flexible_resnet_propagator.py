"""
FlexibleResnetPropagator (Server) - Propagator for FlexibleResNetServer.

Handles both:
- Tensor input (layer boundary splits)
- Tuple input (mid-block splits with residual identity)

Unlike ResnetPropagator which manually handles each layer with ModuleDict,
FlexibleResnetPropagator directly calls the FlexibleResNetServer forward method.
"""

from typing import TYPE_CHECKING, Tuple, Union

import torch
from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from server_args import Config
    from torch.nn import Module


class FlexibleResnetPropagator(BasePropagator):
    """
    Server-side propagator for FlexibleResNetServer.

    Key differences from ResnetPropagator:
    - FlexibleResNetServer handles all residual logic internally
    - No need for complex forward_mapper
    - Supports both Tensor and Tuple inputs directly
    """

    def __init__(self, model: "Module", config: "Config"):
        super().__init__()
        self.model = model
        self.config = config
        self.outputs: torch.Tensor = None
        self.inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], params: dict = None
    ) -> torch.Tensor:
        """
        Forward pass through server model.

        Args:
            x: Input from client
                - Tensor for layer boundary splits
                - Tuple[activation, identity] for mid-block splits
            params: Optional parameters (unused)

        Returns:
            Model output tensor
        """
        # Store inputs for backward pass
        if isinstance(x, tuple):
            # Mid-block split: ensure both tensors require gradients
            self.inputs = tuple(
                t.to(self.config.device).requires_grad_(True)
                if isinstance(t, torch.Tensor) else t
                for t in x
            )
        else:
            # Layer boundary split: single tensor
            self.inputs = x.to(self.config.device).requires_grad_(True)

        self.outputs = self.model(self.inputs)
        return self.outputs

    def backward(self, grads: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Backward pass through server model.

        Args:
            grads: Gradients from loss

        Returns:
            Gradients for client
                - Tensor for layer boundary splits
                - Tuple[grad_activation, grad_identity] for mid-block splits
        """
        if self.outputs is None:
            raise RuntimeError("Propagator outputs not initialized. Call forward() first.")

        grads = grads.to(self.config.device)
        self.outputs.backward(grads)

        # Return gradients for client
        if isinstance(self.inputs, tuple):
            # Mid-block split: return gradients for both activation and identity
            return tuple(
                t.grad.detach() if t.grad is not None else torch.zeros_like(t)
                for t in self.inputs
            )
        else:
            # Layer boundary split: return gradient for single tensor
            if self.inputs.grad is not None:
                return self.inputs.grad.detach()
            return torch.zeros_like(self.inputs)
