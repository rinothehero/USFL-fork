"""
GAS (Gradient Adjustment Scheme) hook.

Currently a minimal stub equivalent to SFL (per_client mode, FedAvg,
pre-allocated pool). The core GAS features are not yet implemented.

TODO: Implement core GAS features:
  - Fake activation generation (label-wise feature stats + Gaussian sampling)
  - Logit adjustment (gradient quality-based correction)
  - G-measurement-based client selection
  - V-value scoring for client ranking
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List

import torch

from .base import BaseMethodHook
from ..client_ops import (
    RoundContext, RoundResult, ClientResult, ClientState,
    create_client_state, create_server_optimizer, create_criterion,
    snapshot_model, create_optimizer,
)
from ..aggregation import aggregate

if TYPE_CHECKING:
    from ..trainer import SimTrainer


class GASHook(BaseMethodHook):
    """
    GAS: Gradient Adjustment Scheme for Split Federated Learning.

    TODO: Core GAS features (fake activation generation, logit adjustment)
    are not yet implemented. Currently behaves identically to SFL.
    """

    @property
    def server_training_mode(self) -> str:
        return "per_client"

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)
        # Pre-allocate client model pool
        self._client_pool: list = []

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _ensure_pool(self, n: int, model, device):
        """Ensure pool has at least n client model copies."""
        while len(self._client_pool) < n:
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

    # ------------------------------------------------------------------
    # Pre-round
    # ------------------------------------------------------------------

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Select clients (from pre-computed schedule)
        selected = trainer.selection_schedule[round_number - 1]

        # 2. Snapshot current client model state
        client_base_state = snapshot_model(model.client_model)

        # 3. Reuse pre-allocated models
        self._ensure_pool(len(selected), model, device)

        client_states = {}
        for i, cid in enumerate(selected):
            client_model = self._client_pool[i]
            client_model.load_state_dict(client_base_state)

            indices = trainer.client_data_masks[cid]
            state = create_client_state(
                client_id=cid,
                client_model=client_model,
                trainset=trainer.trainset,
                data_indices=indices,
                config=config,
            )
            client_states[cid] = state

        # 4. Server model + optimizer + criterion
        server_model = model.server_model
        server_model.to(device)
        server_optimizer = create_server_optimizer(server_model, config)
        criterion = create_criterion(config)

        return RoundContext(
            round_number=round_number,
            selected_client_ids=selected,
            client_states=client_states,
            server_model=server_model,
            server_optimizer=server_optimizer,
            criterion=criterion,
            iterations=0,  # per_client mode: trainer computes its own loop count
            device=device,
            extra={"client_base_state": client_base_state},
        )

    # ------------------------------------------------------------------
    # Post-round
    # ------------------------------------------------------------------

    def post_round(
        self,
        trainer: "SimTrainer",
        round_number: int,
        round_ctx: RoundContext,
        client_results: List[ClientResult],
    ) -> RoundResult:
        model = trainer.model
        device = trainer.device

        # 1. Aggregate client models (FedAvg)
        state_dicts = [r.model_state_dict for r in client_results]
        weights = [float(r.dataset_size) for r in client_results]
        agg_state = aggregate(self.config.aggregator, state_dicts, weights)

        # 2. Update global client model
        model.client_model.load_state_dict(agg_state)

        # 3. Evaluate
        accuracy = model.evaluate(trainer.testloader, device)
        avg_loss = round_ctx.extra.get("avg_loss", 0.0)

        print(
            f"[Round {round_number:3d}] "
            f"Acc: {accuracy:.4f}  Loss: {avg_loss:.4f}",
            flush=True,
        )

        return RoundResult(
            round_number=round_number,
            accuracy=accuracy,
            loss=avg_loss,
            metrics={
                "selected_clients": round_ctx.selected_client_ids,
            },
        )
