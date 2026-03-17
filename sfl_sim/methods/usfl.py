"""
USFL (Unified Split Federated Learning) hook.

6 core optimizations for Non-IID environments:
1. Data Balancing (trimming/replication/target)
2. Gradient Shuffle (random/inplace/average/adaptive_alpha)
3. USFL Selector (missing label + freshness)
4. USFL Aggregator (label-capped weighted)
5. Dynamic Batch Scheduler
6. Cumulative Usage Tracking
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseMethodHook
from ..client_ops import (
    RoundContext, RoundResult, ClientResult,
    create_server_optimizer, create_criterion,
    snapshot_model, get_label_distribution,
)
from ..aggregation import aggregate
from ..selection import select_clients
from ..maskable_dataset import MaskableDataset
from ..batch_scheduler import create_schedule

if TYPE_CHECKING:
    from ..trainer import SimTrainer


class USFLHook(BaseMethodHook):
    """USFL with all 6 Non-IID optimizations."""

    @property
    def server_training_mode(self) -> str:
        # Config override (e.g., "concatenated_fused") or default "concatenated"
        override = self.config.server_training_mode
        return override if override else "concatenated"

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)
        # Persistent state across rounds
        self.cumulative_usage: Dict[int, Dict[int, Dict[int, int]]] = {}
        # {client_id: {label: {bin_key: count}}}

    # ------------------------------------------------------------------
    # Pre-round
    # ------------------------------------------------------------------

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Build client label distributions
        client_label_dists = {}
        for cid in trainer.all_client_ids:
            indices = trainer.client_data_masks[cid]
            client_label_dists[cid] = get_label_distribution(trainer.trainset, indices)

        # 2. Select clients (USFL selector)
        selected = select_clients(
            config.selector,
            config.num_clients_per_round,
            trainer.all_client_ids,
            rng=trainer.rng,
            client_label_dists=client_label_dists,
            cumulative_usage=self.cumulative_usage,
            use_fresh_scoring=config.use_fresh_scoring,
        )

        # 3. Snapshot client model state
        client_base_state = snapshot_model(model.client_model)

        # 4. Calculate augmented sizes (data balancing)
        augmented_sizes = {}
        if config.balancing_strategy != "none":
            augmented_sizes = self._calculate_augmented_sizes(
                selected, client_label_dists, config
            )

        # 5. Create client states with MaskableDataset
        from torch.utils.data import DataLoader
        client_states = {}
        client_data_sizes = {}

        for cid in selected:
            client_copy = copy.deepcopy(model.client_model)
            client_copy.load_state_dict(client_base_state)
            client_copy.to(device)

            indices = trainer.client_data_masks[cid]

            # Apply data balancing if configured
            if cid in augmented_sizes:
                dataset = MaskableDataset(trainer.trainset, indices)
                dataset.update_amount_per_label(augmented_sizes[cid])
            else:
                from torch.utils.data import Subset
                dataset = Subset(trainer.trainset, indices)

            dataloader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
            )
            from ..client_ops import create_optimizer
            optimizer = create_optimizer(client_copy, config)

            from ..client_ops import ClientState
            client_states[cid] = ClientState(
                client_id=cid,
                client_model=client_copy,
                optimizer=optimizer,
                dataloader=dataloader,
                data_iter=iter(dataloader),
                label_distribution=client_label_dists[cid],
                dataset_size=len(dataset),
            )
            client_data_sizes[cid] = len(dataset)

        # 6. Compute iterations
        if config.use_dynamic_batch_scheduler:
            # Target = global batch size (all clients combined per iteration)
            target_bs = config.batch_size * config.num_clients_per_round
            k, schedule = create_schedule(target_bs, client_data_sizes)
            # k = iterations for 1 pass through data, multiply by local_epochs
            iterations = k * config.local_epochs
        else:
            max_batches = max(len(s.dataloader) for s in client_states.values())
            iterations = config.local_epochs * max_batches

        # 7. Server setup
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
            iterations=iterations,
            device=device,
            extra={
                "client_base_state": client_base_state,
                "client_label_dists": {cid: client_label_dists[cid] for cid in selected},
            },
        )

    # ------------------------------------------------------------------
    # Gradient shuffle
    # ------------------------------------------------------------------

    def process_gradients(self, activation_grads: torch.Tensor, ctx: RoundContext) -> torch.Tensor:
        if not self.config.gradient_shuffle:
            return activation_grads

        strategy = self.config.gradient_shuffle_strategy
        batch_sizes = ctx.extra.get("client_batch_sizes", [])
        labels = ctx.extra.get("concat_labels", None)

        if strategy == "random":
            return self._shuffle_random(activation_grads)
        elif strategy == "inplace":
            return self._shuffle_inplace(activation_grads, batch_sizes, labels)
        elif strategy == "average":
            return self._shuffle_average(activation_grads, batch_sizes)
        elif strategy == "average_adaptive_alpha":
            return self._shuffle_adaptive(activation_grads, batch_sizes)
        return activation_grads

    def _shuffle_random(self, grads: torch.Tensor) -> torch.Tensor:
        """Random permutation of gradient rows."""
        perm = torch.randperm(grads.size(0), device=grads.device)
        return grads[perm]

    def _shuffle_inplace(
        self, grads: torch.Tensor, batch_sizes: List[int], labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Class-balanced shuffle: redistribute so each client gets balanced classes."""
        if labels is None:
            return self._shuffle_random(grads)

        n_clients = len(batch_sizes)
        total = grads.size(0)

        # Group gradient indices by label
        label_indices: Dict[int, List[int]] = defaultdict(list)
        for i in range(total):
            label_indices[int(labels[i].item())].append(i)

        # Round-robin distribute across clients
        new_order = [[] for _ in range(n_clients)]
        for label, indices in label_indices.items():
            for i, idx in enumerate(indices):
                client_idx = i % n_clients
                new_order[client_idx].append(idx)

        # Flatten in client order
        flat_order = []
        for client_indices in new_order:
            flat_order.extend(client_indices)

        # Pad or trim to match original size
        if len(flat_order) < total:
            flat_order.extend(range(len(flat_order), total))
        flat_order = flat_order[:total]

        return grads[torch.tensor(flat_order, device=grads.device)]

    def _shuffle_average(self, grads: torch.Tensor, batch_sizes: List[int]) -> torch.Tensor:
        """Mix each client's gradients with global mean gradient."""
        alpha = self.config.gradient_average_weight
        mean_grad = grads.mean(dim=0, keepdim=True)

        mixed = alpha * grads + (1 - alpha) * mean_grad.expand_as(grads)
        return mixed

    def _shuffle_adaptive(self, grads: torch.Tensor, batch_sizes: List[int]) -> torch.Tensor:
        """Adaptive alpha based on cosine similarity with global mean."""
        beta = self.config.adaptive_alpha_beta
        mean_grad = grads.mean(dim=0, keepdim=True)

        # Compute per-client mean gradient, then cosine similarity
        offset = 0
        result = grads.clone()
        for bs in batch_sizes:
            if bs == 0:
                continue
            client_grads = grads[offset:offset + bs]
            client_mean = client_grads.mean(dim=0, keepdim=True)

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                client_mean.flatten().unsqueeze(0),
                mean_grad.flatten().unsqueeze(0),
            ).item()

            # alpha: high similarity → keep more original, low → mix more
            alpha = 1.0 / (1.0 + math.exp(-beta * cos_sim))

            result[offset:offset + bs] = alpha * client_grads + (1 - alpha) * mean_grad.expand_as(client_grads)
            offset += bs

        return result

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
        config = self.config

        # 1. USFL aggregation (label-capped weighted)
        state_dicts = [r.model_state_dict for r in client_results]
        label_dists = [r.label_distribution for r in client_results]
        weights = [float(r.dataset_size) for r in client_results]

        if config.aggregator == "usfl":
            agg_state = aggregate("usfl", state_dicts, weights, label_distributions=label_dists)
        else:
            agg_state = aggregate(config.aggregator, state_dicts, weights)

        # 2. Update global client model
        model.client_model.load_state_dict(agg_state)

        # 3. Update cumulative usage
        if config.use_cumulative_usage:
            self._update_cumulative_usage(round_ctx, client_results)

        # 4. Evaluate
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

    # ------------------------------------------------------------------
    # Data balancing
    # ------------------------------------------------------------------

    def _calculate_augmented_sizes(
        self,
        selected: List[int],
        client_label_dists: Dict[int, Dict[int, int]],
        config,
    ) -> Dict[int, Dict[int, int]]:
        """
        Calculate per-client per-label target sizes based on balancing strategy.

        Returns: {client_id: {label: target_count}}
        """
        result = {}
        strategy = config.balancing_strategy
        target_str = config.balancing_target

        for cid in selected:
            dist = client_label_dists.get(cid, {})
            if not dist:
                continue

            counts = list(dist.values())
            if not counts:
                continue

            # Determine target per label
            if strategy == "trimming":
                target = min(counts)
            elif strategy == "replication":
                target = max(counts)
            elif strategy == "target":
                if target_str == "mean":
                    target = int(sum(counts) / len(counts))
                elif target_str == "median":
                    sorted_counts = sorted(counts)
                    mid = len(sorted_counts) // 2
                    target = sorted_counts[mid]
                else:
                    target = int(target_str)
            else:
                continue

            target = max(1, target)
            result[cid] = {label: target for label in dist}

        return result

    # ------------------------------------------------------------------
    # Cumulative usage tracking
    # ------------------------------------------------------------------

    def _update_cumulative_usage(
        self, ctx: RoundContext, client_results: List[ClientResult]
    ):
        """Update exponential bin tracking for selected clients."""
        for result in client_results:
            cid = result.client_id
            if cid not in self.cumulative_usage:
                self.cumulative_usage[cid] = {}

            for label, count in result.label_distribution.items():
                if label not in self.cumulative_usage[cid]:
                    self.cumulative_usage[cid][label] = {}

                bins = self.cumulative_usage[cid][label]
                # Find current total usage for this label
                total = sum(bins.values())
                bin_key = self._get_exponential_bin(total)
                bins[bin_key] = bins.get(bin_key, 0) + count

    @staticmethod
    def _get_exponential_bin(usage_count: int) -> int:
        """Map usage count to exponential bin key. Bins: 0, 1, 2, 4, 8, 16, ..."""
        if usage_count <= 1:
            return usage_count
        return 2 ** (int(math.log2(usage_count)))
