"""
Base method hook interface.

All methods (SFL, USFL, SCAFFOLD, Mix2SFL, GAS, MultiSFL) inherit from this.
Override only what differs from the default behavior.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..trainer import SimTrainer
    from ..client_ops import RoundContext, RoundResult, ClientResult


class BaseMethodHook(ABC):
    """
    Strategy interface for method-specific behavior.

    Lifecycle:
        for round in rounds:
            ctx = hook.pre_round(trainer, round)           # Setup
            results = trainer.run_sfl_round(ctx)            # Training (shared)
            round_result = hook.post_round(trainer, ...)    # Finalize
    """

    def __init__(self, config, trainer: "SimTrainer"):
        self.config = config
        self.trainer = trainer

    @abstractmethod
    def pre_round(self, trainer: "SimTrainer", round_number: int) -> "RoundContext":
        """
        Prepare for a training round.

        Select clients, split model, create ClientStates, return RoundContext.
        """
        ...

    @abstractmethod
    def post_round(
        self,
        trainer: "SimTrainer",
        round_number: int,
        round_ctx: "RoundContext",
        client_results: List["ClientResult"],
    ) -> "RoundResult":
        """
        Finalize a training round.

        Aggregate client models, update global model, evaluate, return RoundResult.
        """
        ...

    # --- In-round hooks (override to customize) ---

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        ctx: "RoundContext",
        client_id: int | None = None,
    ) -> torch.Tensor:
        """Compute loss. Override for GAS logit adjustment etc."""
        return criterion(logits, labels)

    def process_gradients(
        self,
        activation_grads: torch.Tensor,
        ctx: "RoundContext",
    ) -> torch.Tensor:
        """Process activation gradients before sending to clients. Override for USFL gradient shuffle."""
        return activation_grads

    def process_activations(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        ctx: "RoundContext",
    ):
        """Process activations before server forward. Override for GAS feature generation, Mix2SFL SmashMix."""
        return activations, labels
