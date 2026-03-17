"""
MultiSFL (Multi-Branch Split Federated Learning) hook.

Multiple branch servers each maintain their own server model.
A main server coordinates and aggregates via soft-pull.
Clients are assigned to branches round-robin.
Knowledge replay: inactive clients provide replay data to branches.
Sampling proportions are adjusted dynamically per branch.

Key difference from SFL/USFL: overrides run_round() for full control
because MultiSFL needs N separate server models trained in parallel.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .base import BaseMethodHook
from ..client_ops import (
    ClientResult,
    ClientState,
    RoundContext,
    RoundResult,
    create_client_state,
    create_criterion,
    create_optimizer,
    create_server_optimizer,
    get_label_distribution,
    get_next_batch,
    snapshot_model,
)
from ..aggregation import fedavg
from ..selection import select_clients

if TYPE_CHECKING:
    from ..trainer import SimTrainer


# ---------------------------------------------------------------------------
# Branch state management
# ---------------------------------------------------------------------------


@dataclass
class BranchState:
    """State for one branch server."""
    branch_id: int
    server_model: nn.Module
    server_optimizer: torch.optim.Optimizer
    accuracy_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Score vector tracker (per-branch label distribution tracking)
# ---------------------------------------------------------------------------


class ScoreVectorTracker:
    """Track per-branch label distributions for replay planning."""

    def __init__(self, num_branches: int, num_classes: int, gamma: float):
        self.B = num_branches
        self.C = num_classes
        self.gamma = gamma
        self.L_hist: List[List[np.ndarray]] = [[] for _ in range(self.B)]

    def append_label_dist(self, b: int, label_dist: Dict[int, int]) -> None:
        """Append a label distribution (sparse dict) for branch b."""
        dense = np.zeros(self.C, dtype=np.float64)
        total = sum(label_dist.values())
        if total > 0:
            for label, count in label_dist.items():
                if 0 <= label < self.C:
                    dense[label] = count / total
        self.L_hist[b].append(dense)

    def score_vector(self, b: int) -> np.ndarray:
        """Compute exponentially-weighted score vector for branch b."""
        hist = self.L_hist[b]
        if len(hist) == 0:
            return np.ones(self.C, dtype=np.float64) / self.C
        r = len(hist) - 1
        weights = np.array(
            [self.gamma ** (r - j) for j in range(r + 1)], dtype=np.float64
        )
        weights = weights / (weights.sum() + 1e-12)
        stacked = np.stack(hist, axis=0)
        return (weights[:, None] * stacked).sum(axis=0)


# ---------------------------------------------------------------------------
# Knowledge replay planner
# ---------------------------------------------------------------------------


class ReplayPlanner:
    """Plan per-class replay quotas based on score vector deficits."""

    def __init__(self, num_classes: int):
        self.C = num_classes

    def plan(
        self, sv: np.ndarray, p_r: float, base_count: int, min_total: int = 0
    ) -> np.ndarray:
        """Return per-class integer quotas for replay."""
        mean_sv = float(sv.mean())
        prior = np.maximum(0.0, mean_sv - sv)
        total = int(round(float(base_count) * float(p_r)))
        if total < min_total and p_r > 0:
            total = min_total
        if total <= 0:
            return np.zeros(self.C, dtype=np.int64)

        sum_prior = float(prior.sum())
        if sum_prior <= 0.0:
            return np.zeros(self.C, dtype=np.int64)

        q_float = total * (prior / sum_prior)
        q = np.round(q_float).astype(np.int64)

        # Rounding correction
        diff = int(total - q.sum())
        if diff != 0:
            order = np.argsort(-prior)
            idx = 0
            while diff != 0 and idx < len(order):
                c = int(order[idx])
                if diff > 0:
                    q[c] += 1
                    diff -= 1
                elif q[c] > 0:
                    q[c] -= 1
                    diff += 1
                idx = (idx + 1) if (idx + 1) < len(order) else 0

        return np.maximum(q, 0)


# ---------------------------------------------------------------------------
# Sampling proportion scheduler
# ---------------------------------------------------------------------------


class SamplingScheduler:
    """Dynamically adjust replay proportion p_r based on gradient norm changes."""

    def __init__(
        self,
        p0: float,
        p_min: float,
        p_max: float,
        eps: float,
        mode: str = "abs_ratio",
        delta_clip: float = 0.2,
    ):
        self.p = float(p0)
        self.p_min = max(float(p_min), 1e-4)
        self.p_max = float(p_max)
        self.eps = float(eps)
        self.mode = mode
        self.delta_clip = float(delta_clip)
        self.fgn_hist: List[float] = []

    def update(self, fgn_r: float) -> float:
        """Update p based on functional gradient norm."""
        fgn_prev = self.fgn_hist[-1] if self.fgn_hist else None
        self.fgn_hist.append(float(fgn_r))

        if fgn_prev is None:
            return self.p

        if self.mode == "paper":
            denom = (
                fgn_prev
                if abs(fgn_prev) > self.eps
                else (self.eps if fgn_prev >= 0 else -self.eps)
            )
            factor = (fgn_r - fgn_prev) / denom
            p_new = self.p * (1.0 + factor)
        elif self.mode == "abs_ratio":
            factor = abs(fgn_r) / (abs(fgn_prev) + self.eps)
            p_new = self.p * factor
        elif self.mode == "one_plus_delta":
            delta = (fgn_r - fgn_prev) / (abs(fgn_prev) + self.eps)
            delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))
            p_new = self.p * (1.0 + delta)
        else:
            raise ValueError(f"Unknown p update mode: {self.mode}")

        self.p = float(np.clip(p_new, self.p_min, self.p_max))
        return self.p


# ---------------------------------------------------------------------------
# Helper: average and blend state dicts
# ---------------------------------------------------------------------------


def _average_state_dicts(state_dicts: List[dict]) -> dict:
    """Simple average of multiple state_dicts."""
    n = len(state_dicts)
    avg = {}
    for key in state_dicts[0]:
        avg[key] = sum(sd[key].float() for sd in state_dicts).to(
            state_dicts[0][key].dtype
        ) / n
    return avg


def _blend_state_dict(current: dict, target: dict, alpha: float) -> dict:
    """Blend current toward target: new = (1-alpha)*current + alpha*target."""
    blended = {}
    for key in current:
        blended[key] = (
            (1 - alpha) * current[key].float() + alpha * target[key].float()
        ).to(current[key].dtype)
    return blended


# ---------------------------------------------------------------------------
# Helper: build class-to-indices map for a dataset subset
# ---------------------------------------------------------------------------


def _build_class_to_indices(
    trainset, indices: List[int], num_classes: int
) -> Dict[int, List[int]]:
    """Build {class_label: [local_indices_within_subset]} mapping."""
    if hasattr(trainset, "targets"):
        targets = trainset.targets
    elif hasattr(trainset, "labels"):
        targets = trainset.labels
    else:
        return {c: [] for c in range(num_classes)}

    mapping: Dict[int, List[int]] = defaultdict(list)
    for local_idx, global_idx in enumerate(indices):
        label = int(targets[global_idx])
        mapping[label].append(local_idx)
    return dict(mapping)


# ---------------------------------------------------------------------------
# MultiSFLHook
# ---------------------------------------------------------------------------


class MultiSFLHook(BaseMethodHook):
    """
    Multi-branch Split Federated Learning.

    Overrides run_round() to handle N branch servers, each with its own
    server model and assigned client. After local training, branch server
    models are averaged into a master and soft-pulled back.
    """

    def __init__(self, config, trainer: "SimTrainer"):
        super().__init__(config, trainer)

        self.num_branches = config.num_branches
        self.alpha = config.alpha_master_pull
        device = trainer.device

        # Pre-allocate branch server models (deepcopy of base server model)
        self.branches: List[BranchState] = []
        for b in range(self.num_branches):
            server_copy = copy.deepcopy(trainer.model.server_model).to(device)
            server_opt = self._create_branch_server_optimizer(server_copy, config)
            self.branches.append(
                BranchState(
                    branch_id=b,
                    server_model=server_copy,
                    server_optimizer=server_opt,
                )
            )

        # Master state dicts (computed after each round)
        self._master_client_sd: Optional[dict] = None
        self._master_server_sd: Optional[dict] = None

        # Pre-allocate client model pool
        self._client_pool: list = []

        # Replay components
        self.score_tracker = ScoreVectorTracker(
            self.num_branches, trainer.num_classes, config.gamma
        )
        self.replay_planner = ReplayPlanner(trainer.num_classes)
        self.sampling_scheduler = SamplingScheduler(
            p0=config.p0,
            p_min=config.p_min,
            p_max=config.p_max,
            eps=config.p_eps,
            mode=config.p_update,
            delta_clip=config.p_delta_clip,
        )

        # Track class-to-indices for replay sampling
        self._class_to_indices: Dict[int, Dict[int, List[int]]] = {}

    def _create_branch_server_optimizer(
        self, model: nn.Module, config
    ) -> torch.optim.Optimizer:
        """Create optimizer for a branch server model."""
        lr = config.learning_rate
        if config.server_learning_rate is not None:
            lr = config.server_learning_rate

        if config.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer in ("adam", "adamw"):
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _ensure_pool(self, n: int, model, device):
        """Ensure client model pool has at least n copies."""
        while len(self._client_pool) < n:
            self._client_pool.append(copy.deepcopy(model.client_model).to(device))

    def _get_class_to_indices(
        self, cid: int, trainer: "SimTrainer"
    ) -> Dict[int, List[int]]:
        """Lazily build and cache class-to-indices for a client."""
        if cid not in self._class_to_indices:
            indices = trainer.client_data_masks[cid]
            self._class_to_indices[cid] = _build_class_to_indices(
                trainer.trainset, indices, trainer.num_classes
            )
        return self._class_to_indices[cid]

    # ------------------------------------------------------------------
    # Pre-round: select clients, assign to branches
    # ------------------------------------------------------------------

    def pre_round(self, trainer: "SimTrainer", round_number: int) -> RoundContext:
        config = self.config
        device = trainer.device
        model = trainer.model

        # 1. Select clients
        selected = select_clients(
            config.selector,
            config.num_clients_per_round,
            trainer.all_client_ids,
            rng=trainer.rng,
        )

        # 2. Assign clients to branches (round-robin)
        branch_mapping: Dict[int, int] = {}
        for b in range(self.num_branches):
            cid = selected[b % len(selected)]
            branch_mapping[b] = cid

        # 3. Snapshot client model
        client_base_state = snapshot_model(model.client_model)

        # 4. Create client states for all selected
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

        # 5. Compute iterations (local_epochs * max_batches)
        max_iters = 0
        for cid in selected:
            n_batches = len(client_states[cid].dataloader)
            max_iters = max(max_iters, config.local_epochs * n_batches)

        # 6. Use dummy server model/optimizer for the context (not used by run_round)
        server_model = model.server_model
        server_optimizer = create_server_optimizer(server_model, config)
        criterion = create_criterion(config)

        # Inactive clients (for replay)
        inactive_ids = [c for c in trainer.all_client_ids if c not in selected]

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
                "branch_mapping": branch_mapping,
                "inactive_ids": inactive_ids,
            },
        )

    # ------------------------------------------------------------------
    # Run round: full multi-branch training override
    # ------------------------------------------------------------------

    def run_round(
        self, trainer: "SimTrainer", ctx: RoundContext
    ) -> Optional[List[ClientResult]]:
        """Multi-branch SFL training round."""
        config = self.config
        device = ctx.device
        criterion = ctx.criterion
        branch_mapping: Dict[int, int] = ctx.extra["branch_mapping"]
        inactive_ids: List[int] = ctx.extra["inactive_ids"]
        p_r = self.sampling_scheduler.p

        total_loss = 0.0
        loss_count = 0
        grad_norm_sq_list: List[float] = []

        for b in range(self.num_branches):
            branch = self.branches[b]
            cid = branch_mapping[b]
            state = ctx.client_states.get(cid)

            if state is None:
                continue

            server_model = branch.server_model
            server_optimizer = branch.server_optimizer

            # Track label distribution for replay scoring
            self.score_tracker.append_label_dist(b, state.label_distribution)

            # Plan replay for this branch (first step only)
            sv = self.score_tracker.score_vector(b)
            base_count = state.dataset_size
            if config.replay_budget_mode == "batch":
                base_count = config.batch_size
            q_remaining = self.replay_planner.plan(
                sv, p_r, base_count, min_total=config.replay_min_total
            )

            # Collect replay activations from inactive clients
            replay_acts: List[torch.Tensor] = []
            replay_labels: List[torch.Tensor] = []
            if q_remaining.sum() > 0 and len(inactive_ids) > 0:
                replay_acts, replay_labels = self._collect_replay(
                    trainer=trainer,
                    client_model=state.client_model,
                    q_remaining=q_remaining,
                    inactive_ids=inactive_ids,
                    device=device,
                )

            # Per-branch local training
            branch_grad_norm_sq = 0.0
            n_batches = len(state.dataloader)
            steps_to_run = config.local_epochs * n_batches

            for local_step in range(steps_to_run):
                batch = get_next_batch(state, device, "cycling")
                if batch is None:
                    break
                images, labels = batch

                # Client forward (eval mode, detach)
                state.client_model.eval()
                with torch.no_grad():
                    activation = state.client_model(images)
                activation = activation.detach()

                # Combine with replay activations (first step only)
                if local_step == 0 and replay_acts:
                    f_main = activation.detach().requires_grad_(True)
                    f_replay = torch.cat(replay_acts, dim=0).to(device)
                    y_replay = torch.cat(replay_labels, dim=0).to(device)
                    f_all = torch.cat([f_main, f_replay.detach()], dim=0)
                    y_all = torch.cat([labels, y_replay], dim=0)
                else:
                    f_main = activation.detach().requires_grad_(True)
                    f_all = f_main
                    y_all = labels

                # Server forward + backward
                if f_all.size(0) == 1:
                    server_model.eval()
                else:
                    server_model.train()

                server_optimizer.zero_grad()
                logits = server_model(f_all)
                loss = criterion(logits, y_all)
                loss.backward()

                total_loss += loss.item()
                loss_count += 1

                # Collect grad norm for scheduler
                for p in server_model.parameters():
                    if p.grad is not None:
                        branch_grad_norm_sq += float(
                            (p.grad.detach() ** 2).sum().item()
                        )

                if config.clip_grad:
                    nn.utils.clip_grad_norm_(
                        server_model.parameters(), config.clip_grad_max_norm
                    )
                server_optimizer.step()

                # Client backward with activation gradient
                grad_f_main = f_main.grad
                if grad_f_main is not None:
                    state.client_model.train()
                    state.optimizer.zero_grad()
                    act = state.client_model(images)
                    act.backward(grad_f_main.detach())

                    if config.clip_grad:
                        nn.utils.clip_grad_norm_(
                            state.client_model.parameters(),
                            config.clip_grad_max_norm,
                        )
                    state.optimizer.step()

            grad_norm_sq_list.append(
                branch_grad_norm_sq / max(steps_to_run, 1)
            )

        # Update sampling proportion
        lr_server = config.server_learning_rate or config.learning_rate
        fgn_r = (
            float(np.mean([-lr_server * g2 for g2 in grad_norm_sq_list]))
            if grad_norm_sq_list
            else 0.0
        )
        self.sampling_scheduler.update(fgn_r)

        # Aggregate branch server models -> master -> soft pull
        self._aggregate_branches(trainer)

        ctx.extra["avg_loss"] = total_loss / max(loss_count, 1)
        ctx.extra["p_r"] = self.sampling_scheduler.p
        ctx.extra["fgn_r"] = fgn_r

        # Collect client results (for post_round aggregation)
        return self._collect_client_results(ctx)

    def _collect_replay(
        self,
        trainer: "SimTrainer",
        client_model: nn.Module,
        q_remaining: np.ndarray,
        inactive_ids: List[int],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect replay activations from inactive clients."""
        config = self.config
        replay_acts: List[torch.Tensor] = []
        replay_labels: List[torch.Tensor] = []
        q_rem = q_remaining.copy()

        trials = 0
        rng = trainer.rng
        while q_rem.sum() > 0 and trials < config.max_assistant_trials:
            assist_id = int(rng.choice(inactive_ids))
            trials += 1

            c2i = self._get_class_to_indices(assist_id, trainer)
            indices = trainer.client_data_masks[assist_id]

            # Sample by quota
            x_list: List[torch.Tensor] = []
            y_list: List[int] = []
            provided = np.zeros_like(q_rem)

            for c in range(len(q_rem)):
                needed = int(q_rem[c])
                if needed <= 0:
                    continue
                avail = c2i.get(c, [])
                if not avail:
                    continue
                n_sample = min(needed, len(avail))
                sampled = rng.choice(avail, size=n_sample, replace=False).tolist()
                for local_idx in sampled:
                    global_idx = indices[local_idx]
                    sample_x, sample_y = trainer.trainset[global_idx]
                    if not isinstance(sample_x, torch.Tensor):
                        sample_x = torch.tensor(sample_x)
                    x_list.append(sample_x)
                    y_list.append(int(sample_y) if not isinstance(sample_y, int) else sample_y)
                    provided[c] += 1
                    if len(x_list) >= 256:
                        break
                if len(x_list) >= 256:
                    break

            if x_list:
                x_batch = torch.stack(x_list, dim=0).to(device)
                y_batch = torch.tensor(y_list, dtype=torch.long, device=device)

                # Forward through client model to get activations
                client_model.eval()
                with torch.no_grad():
                    act = client_model(x_batch)
                replay_acts.append(act.detach())
                replay_labels.append(y_batch)
                q_rem = np.maximum(0, q_rem - provided)

        return replay_acts, replay_labels

    def _aggregate_branches(self, trainer: "SimTrainer"):
        """Aggregate branch server models and client models via soft-pull."""
        # Server: average branch servers -> master -> soft pull
        server_sds = [b.server_model.state_dict() for b in self.branches]
        master_server = _average_state_dicts(server_sds)
        self._master_server_sd = master_server

        for branch in self.branches:
            blended = _blend_state_dict(
                branch.server_model.state_dict(), master_server, self.alpha
            )
            branch.server_model.load_state_dict(blended)

    def _collect_client_results(self, ctx: RoundContext) -> List[ClientResult]:
        """Collect client model snapshots after training."""
        results = []
        for cid in sorted(ctx.client_states.keys()):
            state = ctx.client_states[cid]
            results.append(
                ClientResult(
                    client_id=cid,
                    model_state_dict=snapshot_model(state.client_model),
                    dataset_size=state.dataset_size,
                    label_distribution=state.label_distribution,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Post-round: aggregate client models, evaluate
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
        agg_state = fedavg(state_dicts, weights)

        # 2. Update global client model
        model.client_model.load_state_dict(agg_state)

        # 3. Also soft-pull client models in branches (for next round consistency)
        # The global client model acts as the master for client-side
        # Each round starts from the aggregated global client model anyway

        # 4. Evaluate using master server + aggregated client
        # Temporarily use the master server for evaluation
        original_server_sd = snapshot_model(model.server_model)
        if self._master_server_sd is not None:
            model.server_model.load_state_dict(self._master_server_sd)

        accuracy = model.evaluate(trainer.testloader, device)

        # Restore server model
        model.server_model.load_state_dict(original_server_sd)

        avg_loss = round_ctx.extra.get("avg_loss", 0.0)
        p_r = round_ctx.extra.get("p_r", 0.0)

        print(
            f"[Round {round_number:3d}] "
            f"Acc: {accuracy:.4f}  Loss: {avg_loss:.4f}  "
            f"p_r: {p_r:.6f}  branches: {self.num_branches}",
            flush=True,
        )

        return RoundResult(
            round_number=round_number,
            accuracy=accuracy,
            loss=avg_loss,
            metrics={
                "selected_clients": round_ctx.selected_client_ids,
                "branch_mapping": round_ctx.extra.get("branch_mapping", {}),
                "p_r": p_r,
                "fgn_r": round_ctx.extra.get("fgn_r", 0.0),
            },
        )
