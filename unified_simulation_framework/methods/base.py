"""
BaseMethodHook — Strategy interface for method-specific SFL behavior.

All 7 methods implement this ABC. The SimTrainer calls these hooks
at well-defined points during training, while the core SFL loop
(_run_sfl_round) remains identical across all methods.

ERRATA-aware design:
- server_training_mode distinguishes per_client vs concatenated patterns
- compute_loss allows per-client loss modification (GAS logit adjustment)
- process_activations allows activation mixing (Mix2SFL SmashMix)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from ..client_ops import ClientResult, RoundContext, RoundResult


class BaseMethodHook(ABC):
    """
    Lifecycle for single-branch methods (SFL, USFL, SCAFFOLD, Mix2SFL, GAS):
        for round in rounds:
            ctx = hook.pre_round(trainer, round)
            results = trainer._run_sfl_round(ctx)
            round_result = hook.post_round(trainer, round, ctx, results)

    Lifecycle for multi-branch methods (MultiSFL):
        for round in rounds:
            hook.pre_round(trainer, round)
            for branch in range(B):
                ctx = hook.pre_branch(trainer, round, branch)
                results = trainer._run_sfl_round(ctx)
                hook.post_branch(trainer, round, branch, ctx, results)
            round_result = hook.post_round_multi(trainer, round)
    """

    def __init__(self, config, resources: dict):
        self.config = config
        self.resources = resources

    # --- Server training mode (ERRATA 1) ---

    @property
    def server_training_mode(self) -> str:
        """
        Return 'per_client' or 'concatenated'.

        per_client:    Server forward/backward/step for EACH client sequentially.
                       Used by SFL, SCAFFOLD, GAS, MultiSFL.
        concatenated:  Wait for ALL clients, concatenate activations, single
                       forward/backward/step. Used by USFL, Mix2SFL.
        """
        return "per_client"

    # --- Round lifecycle ---

    @abstractmethod
    def pre_round(self, trainer, round_number: int) -> "RoundContext":
        """
        Prepare for a round:
        - Select clients
        - Split model
        - Create ClientStates
        - Return RoundContext
        """
        ...

    @abstractmethod
    def post_round(
        self,
        trainer,
        round_number: int,
        round_ctx: "RoundContext",
        client_results: List["ClientResult"],
    ) -> "RoundResult":
        """
        Finalize a round:
        - Aggregate client models
        - Update global model
        - Evaluate
        """
        ...

    # --- In-round hooks (optional overrides) ---

    def process_activations(
        self,
        activations: List[torch.Tensor],
        labels: List[torch.Tensor],
        ctx: "RoundContext",
    ) -> tuple:
        """
        Post-process activations before server forward.
        Default: pass-through.
        Override for: Mix2SFL SmashMix.
        """
        return activations, labels

    def process_gradients(
        self,
        activation_grads: torch.Tensor,
        activations: List[torch.Tensor],
        labels: torch.Tensor,
        ctx: "RoundContext",
    ) -> torch.Tensor:
        """
        Post-process activation gradients before sending to clients.
        Default: pass-through.
        Override for: USFL gradient shuffle, Mix2SFL GradMix.
        """
        return activation_grads

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion: torch.nn.Module,
        ctx: "RoundContext",
        client_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute loss. Default: criterion(logits, labels).
        Override for: GAS logit adjustment.
        """
        return criterion(logits, labels)

    # --- Multi-branch support (MultiSFL only) ---

    @property
    def is_multi_branch(self) -> bool:
        """Whether this method uses multi-branch training."""
        return False

    def pre_branch(self, trainer, round_number: int, branch: int) -> "RoundContext":
        """Prepare one branch's training context."""
        raise NotImplementedError("Only MultiSFL uses multi-branch")

    def post_branch(
        self, trainer, round_number: int, branch: int,
        round_ctx: "RoundContext", client_results: List["ClientResult"],
    ):
        """Finalize one branch's training."""
        raise NotImplementedError("Only MultiSFL uses multi-branch")

    def post_round_multi(self, trainer, round_number: int) -> "RoundResult":
        """Finalize a multi-branch round (master aggregation, evaluate)."""
        raise NotImplementedError("Only MultiSFL uses multi-branch")
