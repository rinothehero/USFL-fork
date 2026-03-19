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
    snapshot_model,
)
from ..data import get_label_distribution
from ..aggregation import aggregate
from ..data import MaskableDataset
from ..client_ops import create_schedule

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
        # Pre-allocated client model pool
        self._client_pool: list = []

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

        # 2. Select clients (from pre-computed schedule)
        selected = trainer.selection_schedule[round_number - 1]

        # 3. Snapshot client model state
        client_base_state = snapshot_model(model.client_model)

        # 4. Calculate augmented sizes (data balancing)
        augmented_sizes = {}
        if config.balancing_strategy != "none":
            augmented_sizes = self._calculate_augmented_sizes(
                selected, client_label_dists, config
            )

        # 5. Build datasets and compute data sizes (before DBS)
        from torch.utils.data import DataLoader, Subset
        from ..client_ops import create_optimizer, ClientState

        datasets = {}
        client_data_sizes = {}
        for cid in selected:
            indices = trainer.client_data_masks[cid]
            if cid in augmented_sizes:
                ds = MaskableDataset(trainer.trainset, indices, rng=trainer.rng)
                ds.update_amount_per_label(augmented_sizes[cid])
            else:
                ds = Subset(trainer.trainset, indices)
            datasets[cid] = ds
            client_data_sizes[cid] = len(ds)

        # 6. Compute DBS schedule (determines per-client batch sizes)
        dbs_batch_sizes = {}  # {cid: batch_size}
        if config.use_dynamic_batch_scheduler:
            target_bs = config.batch_size * config.num_clients_per_round
            k, _ = create_schedule(target_bs, client_data_sizes)
            batches_per_epoch = k
            for cid in selected:
                dbs_batch_sizes[cid] = max(1, client_data_sizes[cid] // k)

            # Debug: DBS schedule
            if round_number == 1:
                sizes = [client_data_sizes[c] for c in selected]
                bs_list = [dbs_batch_sizes[c] for c in selected]
                concat_per_iter = sum(bs_list)
                print(
                    f"[DBS] k={k}, target={target_bs}, concat/iter={concat_per_iter}\n"
                    f"  data_sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes)//len(sizes)}\n"
                    f"  batch_sizes: min={min(bs_list)} max={max(bs_list)} mean={sum(bs_list)//len(bs_list)}\n"
                    f"  dataloader_batches: all clients → {k} batches each",
                    flush=True,
                )
        else:
            batches_per_epoch = None  # computed after DataLoader creation

        # 7. Create client states with per-client batch sizes
        client_states = {}

        # Ensure pool has enough models
        while len(self._client_pool) < len(selected):
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

        for i, cid in enumerate(selected):
            client_model = self._client_pool[i]
            client_model.load_state_dict(client_base_state)

            bs = dbs_batch_sizes.get(cid, config.batch_size)
            dataloader = DataLoader(
                datasets[cid], batch_size=bs, shuffle=True, drop_last=False
            )
            optimizer = create_optimizer(client_model, config)

            client_states[cid] = ClientState(
                client_id=cid,
                client_model=client_model,
                optimizer=optimizer,
                dataloader=dataloader,
                data_iter=iter(dataloader),
                label_distribution=client_label_dists[cid],
                dataset_size=len(datasets[cid]),
            )

        if batches_per_epoch is None:
            batches_per_epoch = max(len(s.dataloader) for s in client_states.values())

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
            device=device,
            extra={
                "client_base_state": client_base_state,
                "client_label_dists": {cid: client_label_dists[cid] for cid in selected},
                "batches_per_epoch": batches_per_epoch,
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

        Balancing is computed at the GLOBAL level (union of all selected clients),
        then each client's per-label amount is scaled proportionally.

        Example (target=mean):
          Global: {class 0: 2000, class 3: 500, class 7: 3000}
          target_per_class = mean(2000, 500, 3000) = 1833
          class 0 scale = 1833/2000 = 0.917 → Client A's class 0: 800 → 733
          class 3 scale = 1833/500  = 3.667 → Client B's class 3: 200 → 733

        Returns: {client_id: {label: target_count}}
        """
        strategy = config.balancing_strategy
        target_str = config.balancing_target

        # 1. Compute global label distribution across all selected clients
        global_dist: Dict[int, int] = {}
        for cid in selected:
            for label, count in client_label_dists.get(cid, {}).items():
                global_dist[label] = global_dist.get(label, 0) + count

        if not global_dist:
            return {}

        global_counts = list(global_dist.values())

        # 2. Compute target per class (global level)
        if strategy == "trimming":
            target_per_class = min(global_counts)
        elif strategy == "replication":
            target_per_class = max(global_counts)
        elif strategy == "target":
            if target_str == "mean":
                target_per_class = int(sum(global_counts) / len(global_counts))
            elif target_str == "median":
                sorted_counts = sorted(global_counts)
                mid = len(sorted_counts) // 2
                target_per_class = sorted_counts[mid]
            else:
                target_per_class = int(target_str)
        else:
            return {}

        target_per_class = max(1, target_per_class)

        # 3. Compute per-class scale factor
        scale = {}
        for label, global_count in global_dist.items():
            scale[label] = target_per_class / max(global_count, 1)

        # 4. Apply scale to each client proportionally
        result = {}
        for cid in selected:
            dist = client_label_dists.get(cid, {})
            if not dist:
                continue
            augmented = {}
            for label, count in dist.items():
                augmented[label] = max(1, int(count * scale.get(label, 1.0)))
            result[cid] = augmented

        return result
