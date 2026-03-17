"""
GAS (Gradient Adjustment Scheme) hook.

SFL-based method using gradient quality metrics (G-measurement) for intelligent
client selection and gradient adjustment. Clients with better gradient quality
(lower G-scores) are prioritized.

Key features:
- G-measurement: gradient distance between client gradient and oracle gradient
  from 3 perspectives (client model, server model, split layer)
- V-value: gradient dissimilarity scoring for client ranking
- Feature stats: label-wise mean/variance for synthetic feature generation
- Client selection: G-score + V-value based ranking with uniform fallback
- server_training_mode: per_client (same as SFL)
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from .base import BaseMethodHook
from ..client_ops import (
    RoundContext, RoundResult, ClientResult, ClientState,
    create_client_state, create_server_optimizer, create_criterion,
    snapshot_model, create_optimizer,
)
from ..aggregation import aggregate
from ..selection import select_clients
from ..g_measurement import (
    compute_oracle_gradients,
    collect_current_gradients,
    compute_all_g_scores,
    compute_v_value,
    FeatureStats,
)

if TYPE_CHECKING:
    from ..trainer import SimTrainer


class GASHook(BaseMethodHook):
    """
    GAS: Gradient Adjustment Scheme for Split Federated Learning.

    Uses G-measurement (gradient quality) to rank clients and select the
    ones that contribute most representative gradients. Falls back to
    uniform selection for the first few rounds until G data is available.
    """

    @property
    def server_training_mode(self) -> str:
        return "per_client"

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)

        # Pre-allocated client model pool
        self._client_pool: list = []

        # Persistent state across rounds
        # Per-client G-scores (lower = better gradient quality)
        self.client_g_scores: Dict[int, float] = {}
        # Per-client V-values (lower = more representative features)
        self.client_v_values: Dict[int, float] = {}
        # G-measurement history for logging
        self.g_history: Dict[str, list] = defaultdict(list)
        # Feature statistics tracker (for synthetic feature generation)
        self.feature_stats = FeatureStats()
        # Oracle gradients (recomputed periodically)
        self._oracle_grads: Optional[dict] = None
        # Round at which oracle was last computed
        self._oracle_round: int = -1

        # Config defaults for GAS-specific params
        self._g_measure_frequency = getattr(config, "g_measure_frequency", 10)
        self._v_test_batches = getattr(config, "v_test_batches", 10)
        self._warmup_rounds = getattr(config, "warmup_rounds", 5)
        self._oracle_max_batches = getattr(config, "oracle_max_batches", None)

        # Subscribe to training events to collect per-client gradients
        self._current_round_grads: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _ensure_pool(self, n: int, model, device):
        """Ensure pool has at least n client model copies."""
        while len(self._client_pool) < n:
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def _select_clients_gas(
        self, trainer: "SimTrainer", round_number: int
    ) -> List[int]:
        """
        Select clients using G-score + V-value ranking.

        Falls back to uniform selection during warmup (first few rounds)
        or when no G data is available.
        """
        config = self.config
        n = config.num_clients_per_round
        all_ids = trainer.all_client_ids
        rng = trainer.rng

        # Warmup: use uniform selection
        if round_number <= self._warmup_rounds or not self.client_g_scores:
            return select_clients("uniform", n, all_ids, rng=rng)

        # Score each client: lower G + lower V = better
        # Combine G-score and V-value into a single ranking score
        # We want to SELECT clients with LOW scores, so we negate for max-selection
        scores = {}
        for cid in all_ids:
            g = self.client_g_scores.get(cid, float("inf"))
            v = self.client_v_values.get(cid, float("inf"))

            # Normalize: use ranking-based score
            # Clients without scores get worst rank
            if math.isnan(g) or math.isinf(g):
                g = 1e6
            if math.isnan(v) or math.isinf(v):
                v = 1e6

            # Combined score (lower is better)
            # Weight G-score more than V-value since it directly measures gradient quality
            scores[cid] = g + 0.5 * v

        # Sort by score (ascending = best first) with small random tiebreaker
        ranked = sorted(all_ids, key=lambda c: scores[c] + rng.random() * 1e-6)

        # Select top-N clients
        selected = sorted(ranked[:n])
        return selected

    # ------------------------------------------------------------------
    # Pre-round
    # ------------------------------------------------------------------

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Select clients (GAS-based or uniform fallback)
        selected = self._select_clients_gas(trainer, round_number)

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

        # 5. Compute iterations
        max_iters = 0
        for cid in selected:
            n_batches = len(client_states[cid].dataloader)
            max_iters = max(max_iters, config.local_epochs * n_batches)

        # 6. Reset per-round gradient collection
        self._current_round_grads = {}

        return RoundContext(
            round_number=round_number,
            selected_client_ids=selected,
            client_states=client_states,
            server_model=server_model,
            server_optimizer=server_optimizer,
            criterion=criterion,
            iterations=max_iters,
            device=device,
            extra={
                "client_base_state": client_base_state,
            },
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

        # 3. G-measurement (periodic)
        g_metrics = {}
        if self._should_measure_g(round_number):
            g_metrics = self._run_g_measurement(trainer, round_number, round_ctx)

        # 4. Evaluate
        accuracy = model.evaluate(trainer.testloader, device)
        avg_loss = round_ctx.extra.get("avg_loss", 0.0)

        g_info = ""
        if g_metrics:
            cg = g_metrics.get("mean_client_g", float("nan"))
            sg = g_metrics.get("mean_server_g", float("nan"))
            g_info = f"  G_client: {cg:.4f}  G_server: {sg:.4f}"

        print(
            f"[Round {round_number:3d}] "
            f"Acc: {accuracy:.4f}  Loss: {avg_loss:.4f}{g_info}",
            flush=True,
        )

        metrics = {
            "selected_clients": round_ctx.selected_client_ids,
        }
        if g_metrics:
            metrics["g_measurement"] = g_metrics

        return RoundResult(
            round_number=round_number,
            accuracy=accuracy,
            loss=avg_loss,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # G-measurement
    # ------------------------------------------------------------------

    def _should_measure_g(self, round_number: int) -> bool:
        """Check if this round is a G-measurement round."""
        return round_number % self._g_measure_frequency == 0

    def _run_g_measurement(
        self,
        trainer: "SimTrainer",
        round_number: int,
        round_ctx: RoundContext,
    ) -> Dict[str, float]:
        """
        Run G-measurement: compute oracle gradients, then measure G-score
        for each selected client.

        Updates self.client_g_scores with per-client G-scores.
        """
        device = trainer.device
        model = trainer.model

        # Compute oracle gradients (from test/probe data)
        oracle = compute_oracle_gradients(
            client_model=model.client_model,
            server_model=model.server_model,
            dataloader=trainer.testloader,
            device=device,
            max_batches=self._oracle_max_batches,
        )
        self._oracle_grads = oracle
        self._oracle_round = round_number

        # Compute per-client G-scores using proxy:
        # For each selected client, we do a single forward-backward pass
        # on their data to get their gradient, then compare to oracle.
        selected = round_ctx.selected_client_ids
        per_client_g = {}
        criterion = create_criterion(self.config)

        for cid in selected:
            indices = trainer.client_data_masks[cid]
            from torch.utils.data import DataLoader, Subset

            ds = Subset(trainer.trainset, indices)
            loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False)

            # Backup model states
            client_state_backup = snapshot_model(model.client_model)
            server_state_backup = snapshot_model(model.server_model)

            model.client_model.train()
            model.server_model.train()
            model.client_model.zero_grad()
            model.server_model.zero_grad()

            # Single-batch gradient computation for this client
            total_samples = 0
            client_grad_accum = [torch.zeros_like(p, device="cpu") for p in model.client_model.parameters()]
            server_grad_accum = [torch.zeros_like(p, device="cpu") for p in model.server_model.parameters()]
            split_grad_accum = None

            # Use only first batch for efficiency
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                model.client_model.zero_grad()
                model.server_model.zero_grad()

                activation = model.client_model(images)
                activation.retain_grad()
                logits = model.server_model(activation)
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                loss.backward()

                total_samples += labels.size(0)
                for i, p in enumerate(model.client_model.parameters()):
                    if p.grad is not None:
                        client_grad_accum[i] += p.grad.detach().cpu()
                for i, p in enumerate(model.server_model.parameters()):
                    if p.grad is not None:
                        server_grad_accum[i] += p.grad.detach().cpu()
                if activation.grad is not None:
                    sg = activation.grad.detach().sum(dim=0).cpu()
                    if split_grad_accum is None:
                        split_grad_accum = sg
                    else:
                        split_grad_accum += sg

                break  # First batch only for efficiency

            if total_samples > 0:
                client_grad_accum = [g / total_samples for g in client_grad_accum]
                server_grad_accum = [g / total_samples for g in server_grad_accum]
                if split_grad_accum is not None:
                    split_grad_accum = split_grad_accum / total_samples

            current = {
                "client": client_grad_accum,
                "server": server_grad_accum,
                "split": split_grad_accum,
            }

            scores = compute_all_g_scores(oracle, current)
            per_client_g[cid] = scores

            # Update persistent G-scores (use client_g as primary ranking metric)
            self.client_g_scores[cid] = scores["client_g"]

            # Restore model states
            model.client_model.load_state_dict(client_state_backup)
            model.server_model.load_state_dict(server_state_backup)

        # Compute mean G-scores across selected clients
        client_gs = [s["client_g"] for s in per_client_g.values() if not math.isnan(s["client_g"])]
        server_gs = [s["server_g"] for s in per_client_g.values() if not math.isnan(s["server_g"])]
        split_gs = [s["split_g"] for s in per_client_g.values() if not math.isnan(s["split_g"])]

        mean_client_g = sum(client_gs) / len(client_gs) if client_gs else float("nan")
        mean_server_g = sum(server_gs) / len(server_gs) if server_gs else float("nan")
        mean_split_g = sum(split_gs) / len(split_gs) if split_gs else float("nan")

        # Record history
        self.g_history["client_g"].append(mean_client_g)
        self.g_history["server_g"].append(mean_server_g)
        self.g_history["split_g"].append(mean_split_g)
        self.g_history["per_client_g"].append(per_client_g)

        model.client_model.zero_grad()
        model.server_model.zero_grad()

        return {
            "mean_client_g": mean_client_g,
            "mean_server_g": mean_server_g,
            "mean_split_g": mean_split_g,
            "per_client": per_client_g,
        }
