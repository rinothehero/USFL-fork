"""
FlexibleResnetPropagator - Propagator for FlexibleResNet models.

Handles both:
- Tensor output (layer boundary splits)
- Tuple output (mid-block splits with residual identity)

Unlike ResnetPropagator which manually handles residual connections,
FlexibleResnetPropagator relies on FlexibleResNetClient/Server to handle
the residual logic internally.
"""

from typing import TYPE_CHECKING, Tuple, Union, Optional

import torch
from .base_propagator import BasePropagator

if TYPE_CHECKING:
    from client_args import Config
    from torch.nn import Module


class FlexibleResnetPropagator(BasePropagator):
    """
    Propagator for FlexibleResNet split models.

    Key differences from ResnetPropagator:
    - FlexibleResNetClient outputs Tensor or Tuple directly (no ModuleDict traversal)
    - Layer boundary splits (layer2, layer3, layer4) return Tensor -> simple handling
    - Mid-block splits (layer1.1.bn2) return Tuple -> both elements need gradients
    """

    def __init__(self, model: "Module", config: "Config"):
        super().__init__()
        self.model = model
        self.config = config
        self.outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None

    def forward(
        self, x: torch.Tensor, params: Optional[dict] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through client model.

        Returns:
            - Tensor for layer boundary splits (layer2, layer3, layer4)
            - Tuple[activation, identity] for mid-block splits (layer1.1.bn2)
        """
        self.outputs = self.model(x)

        # Prepare outputs for gradient tracking
        if isinstance(self.outputs, tuple):
            # Mid-block split: both activation and identity need gradients
            detached_outputs = []
            for item in self.outputs:
                if isinstance(item, torch.Tensor):
                    detached = item.clone().detach().requires_grad_(True)
                    detached_outputs.append(detached)
                else:
                    detached_outputs.append(item)
            return tuple(detached_outputs)
        else:
            # Layer boundary split: single tensor
            return self.outputs.clone().detach().requires_grad_(True)

    def backward(self, grads: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        """
        Backward pass through client model.

        Args:
            grads: Gradients from server
                - Tensor for layer boundary splits
                - Tuple[grad_activation, grad_identity] for mid-block splits
        """
        if self.outputs is None:
            raise RuntimeError("Propagator outputs not initialized. Call forward() first.")

        if isinstance(self.outputs, tuple):
            # Mid-block split: backward through both paths
            if not isinstance(grads, tuple):
                raise ValueError("Expected tuple gradients for tuple outputs")

            grads = tuple(g.to(self.config.device) for g in grads)
            torch.autograd.backward(list(self.outputs), list(grads))
        else:
            # Layer boundary split: simple backward
            if isinstance(grads, tuple):
                raise ValueError("Expected tensor gradients for tensor output")

            grads = grads.to(self.config.device)
            self.outputs.requires_grad_(True)
            self.outputs.backward(grads)
