"""
Mix2SFL hook — Split Federated Learning with SmashMix activation mixing.

SmashMix augments training by creating mixed activations from pairs of clients:
    mixed_act = lambda * act_i + (1 - lambda) * act_j
    mixed_label = lambda * onehot_i + (1 - lambda) * onehot_j

Lambda is sampled from Beta(alpha, alpha) distribution.
Mixed samples are appended to the concatenated activation batch.
Soft cross-entropy is used for mixed labels.

Uses concatenated server training mode (same as USFL).
"""
from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from .base import BaseMethodHook
from ..client_ops import (
    RoundContext, RoundResult, ClientResult, ClientState,
    create_client_state, create_server_optimizer, create_criterion,
    snapshot_model, create_optimizer,
)
from ..aggregation import aggregate


if TYPE_CHECKING:
    from ..trainer import SimTrainer


class Mix2SFLHook(BaseMethodHook):
    """Mix2SFL with SmashMix activation mixing."""

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)
        self._client_pool: list = []

        # SmashMix config (read from config or use defaults)
        self.smashmix_alpha = getattr(config, "mix2sfl_beta_alpha", 1.0)
        self.smashmix_ratio = getattr(config, "mix2sfl_smashmix_ratio", 1.0)

    @property
    def server_training_mode(self) -> str:
        return "concatenated"

    def _ensure_pool(self, n: int, model, device):
        while len(self._client_pool) < n:
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

    def process_activations(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        ctx: RoundContext,
    ):
        """Apply SmashMix: mix activations from different clients.

        For each client, sample partner(s) and create mixed activation+label pairs.
        Mixed samples are appended to the batch. Labels become soft (one-hot mixed).
        """
        num_classes = ctx.extra.get("num_classes", 10)
        client_batch_sizes = ctx.extra.get("client_batch_sizes", [])
        client_order = ctx.extra.get("client_order", [])

        if len(client_order) <= 1:
            return activations, labels

        # Determine how many partners per client
        ns = self._get_ns(len(client_order))
        if ns <= 0:
            return activations, labels

        # Split activations by client
        client_acts = {}
        client_labels = {}
        offset = 0
        for i, cid in enumerate(client_order):
            bs = client_batch_sizes[i] if i < len(client_batch_sizes) else 0
            if bs > 0:
                client_acts[cid] = activations[offset:offset + bs]
                client_labels[cid] = labels[offset:offset + bs]
            offset += bs

        mixed_acts_list = []
        mixed_labels_list = []

        for cid in client_order:
            if cid not in client_acts:
                continue

            candidates = [c for c in client_order if c != cid and c in client_acts]
            num_partners = min(ns, len(candidates))
            if num_partners <= 0:
                continue

            partners = random.sample(candidates, num_partners)
            for pid in partners:
                act_i = client_acts[cid]
                act_j = client_acts[pid]

                # Batch sizes must match for mixing
                min_bs = min(act_i.size(0), act_j.size(0))
                if min_bs == 0:
                    continue

                act_i_slice = act_i[:min_bs]
                act_j_slice = act_j[:min_bs]
                labels_i = client_labels[cid][:min_bs]
                labels_j = client_labels[pid][:min_bs]

                lam = self._sample_lambda()

                mixed_act = lam * act_i_slice + (1.0 - lam) * act_j_slice

                # Create soft labels (one-hot mixed)
                onehot_i = F.one_hot(labels_i, num_classes=num_classes).float()
                onehot_j = F.one_hot(labels_j, num_classes=num_classes).float()
                mixed_label = lam * onehot_i + (1.0 - lam) * onehot_j

                mixed_acts_list.append(mixed_act.detach())
                mixed_labels_list.append(mixed_label)

        if not mixed_acts_list:
            return activations, labels

        # Append mixed samples
        mixed_acts = torch.cat(mixed_acts_list, dim=0)
        mixed_labels_soft = torch.cat(mixed_labels_list, dim=0)

        # Convert original labels to one-hot for uniform loss computation
        original_soft = F.one_hot(labels, num_classes=num_classes).float()

        combined_acts = torch.cat([activations, mixed_acts], dim=0)
        combined_labels = torch.cat([original_soft, mixed_labels_soft], dim=0)

        # Store flag: labels are now soft
        ctx.extra["soft_labels"] = True

        return combined_acts, combined_labels

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        criterion,
        ctx: RoundContext,
        client_id=None,
    ) -> torch.Tensor:
        """Use soft cross-entropy when SmashMix is active."""
        if ctx.extra.get("soft_labels", False) and labels.dim() > 1:
            # Soft cross-entropy: -sum(target * log_softmax(logits))
            log_probs = F.log_softmax(logits, dim=-1)
            return -(labels * log_probs).sum(dim=-1).mean()
        return criterion(logits, labels)

    def _get_ns(self, active_count: int) -> int:
        """Number of mixing partners per client."""
        ns = int(active_count * self.smashmix_ratio)
        if active_count <= 1:
            return 0
        return min(ns, active_count - 1)

    def _sample_lambda(self) -> float:
        """Sample mixing coefficient from Beta distribution."""
        alpha = self.smashmix_alpha
        if alpha <= 0:
            return random.random()
        return random.betavariate(alpha, alpha)

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Select clients (from pre-computed schedule)
        selected = trainer.selection_schedule[round_number - 1]

        # 2. Snapshot client model state
        client_base_state = snapshot_model(model.client_model)

        # 3. Create client states
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

        # 5. Compute iterations
        max_batches = max(len(s.dataloader) for s in client_states.values())
        iterations = config.local_epochs * max_batches

        return RoundContext(
            round_number=round_number,
            selected_client_ids=selected,
            client_states=client_states,
            server_model=server_model,
            server_optimizer=server_optimizer,
            criterion=criterion,
            iterations=iterations,
            device=device,
            extra={
                "client_base_state": client_base_state,
                "num_classes": trainer.num_classes,
            },
        )

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
