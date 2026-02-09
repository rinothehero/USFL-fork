"""
Client & Server Drift Measurement Utility (SCAFFOLD-style) for MultiSFL

Measures drift from round-start global model to track
how much local updates deviate during training.

Key Metrics:
- G_drift_client: Average client (branch) trajectory drift energy
- G_drift_server: Average server (branch) trajectory drift energy
- G_end_client/G_end_server: End-point drift
- G_drift_norm: Normalized drift (prevents "update suppression" criticism)

Usage:
1. on_round_start(master_client_sd, master_server_sd): Save x^{t,0} for both
2. accumulate_branch_drift(b, branch_client_model): After each client local step
3. accumulate_server_drift(b, branch_server_model): After each server step
4. finalize_branch(b, branch_client_model, branch_server_model): When branch completes
5. on_round_end(new_master_client_sd, new_master_server_sd): Compute G metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.update_alignment import flatten_delta, compute_update_alignment
from .log_utils import vprint


@dataclass
class DriftMetrics:
    """Drift measurement results for a round"""

    # Client-side drift (aggregated from all branches)
    G_drift_client: float = 0.0  # (1/|B|) Σ (S_n / B_n)
    G_end_client: float = 0.0  # (1/|B|) Σ E_n
    G_drift_norm_client: float = 0.0  # G_drift_client / (||Δx_c||² + ε)
    delta_client_norm_sq: float = 0.0  # ||x_c^{t+1,0} - x_c^{t,0}||²

    # Server-side drift (aggregated from all branches)
    G_drift_server: float = 0.0  # (1/|B|) Σ (S_s / B_s)
    G_end_server: float = 0.0  # (1/|B|) Σ E_s
    G_drift_norm_server: float = 0.0  # G_drift_server / (||Δx_s||² + ε)
    delta_server_norm_sq: float = 0.0  # ||x_s^{t+1,0} - x_s^{t,0}||²

    # Combined metrics
    G_drift_total: float = 0.0  # G_drift_client + G_drift_server
    G_end_total: float = 0.0  # G_end_client + G_end_server

    num_branches: int = 0
    server_steps: int = 0

    # Update alignment metrics
    A_cos_client: float = float("nan")
    M_norm_client: float = 0.0
    A_cos_server: float = float("nan")
    M_norm_server: float = 0.0
    n_valid_alignment: int = 0

    def to_dict(self) -> dict:
        return {
            # Client metrics
            "G_drift_client": self.G_drift_client,
            "G_end_client": self.G_end_client,
            "G_drift_norm_client": self.G_drift_norm_client,
            "delta_client_norm_sq": self.delta_client_norm_sq,
            # Server metrics
            "G_drift_server": self.G_drift_server,
            "G_end_server": self.G_end_server,
            "G_drift_norm_server": self.G_drift_norm_server,
            "delta_server_norm_sq": self.delta_server_norm_sq,
            # Combined
            "G_drift_total": self.G_drift_total,
            "G_end_total": self.G_end_total,
            # Legacy compatibility (client-only names)
            "G_drift": self.G_drift_client,
            "G_end": self.G_end_client,
            "G_drift_norm": self.G_drift_norm_client,
            "delta_global_norm_sq": self.delta_client_norm_sq,
            # Counts
            "num_clients": self.num_branches,
            "num_branches": self.num_branches,
            "server_steps": self.server_steps,
            # Update alignment
            "A_cos_client": self.A_cos_client,
            "M_norm_client": self.M_norm_client,
            "A_cos_server": self.A_cos_server,
            "M_norm_server": self.M_norm_server,
            "n_valid_alignment": self.n_valid_alignment,
        }


@dataclass
class RoundDriftMeasurement:
    """Per-round drift measurement with branch details"""
    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_branch_client: Dict[int, dict] = field(default_factory=dict)
    per_branch_server: Dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_branch_client": {str(k): v for k, v in self.per_branch_client.items()},
            "per_branch_server": {str(k): v for k, v in self.per_branch_server.items()},
        }


class BranchDriftState:
    """Per-branch drift accumulation state"""
    def __init__(self):
        self.trajectory_sum: float = 0.0  # S_n(t)
        self.batch_steps: int = 0  # B_n(t)
        self.endpoint_drift: float = 0.0  # E_n(t)
        self.client_id: Optional[int] = None

    def reset(self):
        self.trajectory_sum = 0.0
        self.batch_steps = 0
        self.endpoint_drift = 0.0
        self.client_id = None

    def to_dict(self) -> dict:
        return {
            "S": self.trajectory_sum,
            "B": self.batch_steps,
            "E": self.endpoint_drift,
            "client_id": self.client_id,
        }


class MultiSFLDriftTracker:
    """
    Tracks client and server drift across rounds for MultiSFL architecture.

    In MultiSFL, multiple branches train concurrently, each with its own client and server.
    Drift is measured per-branch for both client and server models and aggregated.

    Workflow:
    1. on_round_start(master_client_sd, master_server_sd): Save master model snapshots
    2. accumulate_branch_drift(b, branch_client_model): After each client local step
    3. accumulate_server_drift(b, branch_server_model): After each server step
    4. finalize_branch(b, branch_client, branch_server, client_id): When branch completes
    5. on_round_end(new_master_client_sd, new_master_server_sd): Compute G metrics
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        sample_interval: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            epsilon: Small constant to prevent division by zero
            sample_interval: Accumulate drift every N steps (1 = every step)
            device: Device for tensor operations
        """
        self.epsilon = epsilon
        self.sample_interval = sample_interval
        self.device = device

        # Round-start master model snapshots
        self._round_start_client_params: Dict[str, torch.Tensor] = {}
        self._round_start_server_params: Dict[str, torch.Tensor] = {}

        # Per-branch drift states (separate for client and server)
        self._branch_client_states: Dict[int, BranchDriftState] = {}
        self._branch_server_states: Dict[int, BranchDriftState] = {}

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

        # For update alignment (A_cos) computation
        self._branch_deltas: List[Tuple[int, torch.Tensor]] = []
        self._branch_server_deltas: List[Tuple[int, torch.Tensor]] = []

    def _is_trainable_param(self, name: str) -> bool:
        """Check if parameter name corresponds to trainable weights (not buffers)"""
        # Exclude BatchNorm running statistics and tracking buffers
        buffer_keywords = ("running_mean", "running_var", "num_batches_tracked")
        return not any(kw in name for kw in buffer_keywords)

    def on_round_start(
        self,
        master_client_state_dict: dict,
        master_server_state_dict: Optional[dict] = None,
    ):
        """
        Save master model snapshots at round start (x^{t,0})

        Args:
            master_client_state_dict: The master client model state dict
            master_server_state_dict: The master server model state dict (optional)
        """
        # Only save trainable parameters (exclude BatchNorm buffers)
        self._round_start_client_params = {
            name: param.clone().detach().cpu()
            for name, param in master_client_state_dict.items()
            if self._is_trainable_param(name)
        }

        if master_server_state_dict is not None:
            self._round_start_server_params = {
                name: param.clone().detach().cpu()
                for name, param in master_server_state_dict.items()
                if self._is_trainable_param(name)
            }
        else:
            self._round_start_server_params = {}

        # Reset per-round accumulators
        self._branch_client_states.clear()
        self._branch_server_states.clear()
        self._branch_deltas = []
        self._branch_server_deltas = []

    def _compute_drift_from_start(
        self,
        current_state_dict: dict,
        start_params: Dict[str, torch.Tensor],
    ) -> float:
        """Compute ||x^{t,b} - x^{t,0}||² for trainable parameters only"""
        if not start_params:
            return 0.0
        drift_sq = 0.0
        for name, param in current_state_dict.items():
            if name in start_params and self._is_trainable_param(name):
                if isinstance(param, torch.Tensor):
                    diff = param.cpu() - start_params[name]
                else:
                    diff = torch.tensor(param).cpu() - start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def accumulate_branch_drift(self, branch_id: int, branch_client_model: nn.Module):
        """
        Accumulate client drift after a branch's local step.

        Args:
            branch_id: Branch identifier
            branch_client_model: Branch's client model after optimizer step
        """
        if branch_id not in self._branch_client_states:
            self._branch_client_states[branch_id] = BranchDriftState()

        state = self._branch_client_states[branch_id]
        state.batch_steps += 1

        # Sample according to interval
        if state.batch_steps % self.sample_interval == 0:
            current_sd = branch_client_model.state_dict()
            drift = self._compute_drift_from_start(current_sd, self._round_start_client_params)
            state.trajectory_sum += drift

    def accumulate_server_drift(self, branch_id: int, branch_server_model: nn.Module):
        """
        Accumulate server drift after a branch's server step.

        Args:
            branch_id: Branch identifier
            branch_server_model: Branch's server model after optimizer step
        """
        if not self._round_start_server_params:
            return

        if branch_id not in self._branch_server_states:
            self._branch_server_states[branch_id] = BranchDriftState()

        state = self._branch_server_states[branch_id]
        state.batch_steps += 1

        current_sd = branch_server_model.state_dict()
        drift = self._compute_drift_from_start(current_sd, self._round_start_server_params)
        state.trajectory_sum += drift
        state.endpoint_drift = drift

    def finalize_branch(
        self,
        branch_id: int,
        branch_client_model: nn.Module,
        branch_server_model: Optional[nn.Module] = None,
        client_id: Optional[int] = None,
    ):
        """
        Finalize drift measurement for a branch when it completes local training.

        Args:
            branch_id: Branch identifier
            branch_client_model: Final branch client model
            branch_server_model: Final branch server model (optional)
            client_id: Optional client ID that was assigned to this branch
        """
        # Finalize client drift
        if branch_id not in self._branch_client_states:
            self._branch_client_states[branch_id] = BranchDriftState()

        client_state = self._branch_client_states[branch_id]
        client_state.client_id = client_id

        # Compute endpoint drift for client
        current_client_sd = branch_client_model.state_dict()
        client_state.endpoint_drift = self._compute_drift_from_start(
            current_client_sd, self._round_start_client_params
        )

        # Extrapolate if sampling was used
        if self.sample_interval > 1 and client_state.batch_steps > 0:
            sampled_steps = client_state.batch_steps // self.sample_interval
            if sampled_steps > 0:
                client_state.trajectory_sum = (
                    client_state.trajectory_sum * client_state.batch_steps / sampled_steps
                )

        # Finalize server drift
        if branch_server_model is not None and self._round_start_server_params:
            if branch_id not in self._branch_server_states:
                self._branch_server_states[branch_id] = BranchDriftState()

            server_state = self._branch_server_states[branch_id]
            server_state.client_id = client_id

            current_server_sd = branch_server_model.state_dict()
            server_state.endpoint_drift = self._compute_drift_from_start(
                current_server_sd, self._round_start_server_params
            )

    def collect_branch_delta(self, branch_id: int, branch_client_state_dict: dict):
        """Compute and store client Δ_b for A_cos computation."""
        delta = flatten_delta(branch_client_state_dict, self._round_start_client_params)
        if delta is not None:
            self._branch_deltas.append((branch_id, delta))

    def collect_branch_server_delta(self, branch_id: int, branch_server_state_dict: dict):
        """Compute and store server Δ_b for A_cos computation."""
        delta = flatten_delta(branch_server_state_dict, self._round_start_server_params)
        if delta is not None:
            self._branch_server_deltas.append((branch_id, delta))

    def on_round_end(
        self,
        round_number: int,
        new_master_client_state_dict: dict,
        new_master_server_state_dict: Optional[dict] = None,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_master_client_state_dict: The aggregated master client model state dict
            new_master_server_state_dict: The aggregated master server model state dict (optional)

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_branch_client = {
            bid: state.to_dict() for bid, state in self._branch_client_states.items()
        }
        result.per_branch_server = {
            bid: state.to_dict() for bid, state in self._branch_server_states.items()
        }

        # --- Client drift metrics ---
        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_client_norm_sq = 0.0
        for name, param in new_master_client_state_dict.items():
            if name in self._round_start_client_params:
                if isinstance(param, torch.Tensor):
                    diff = param.cpu() - self._round_start_client_params[name]
                else:
                    diff = torch.tensor(param).cpu() - self._round_start_client_params[name]
                delta_client_norm_sq += (diff ** 2).sum().item()

        # G_drift_client = (1/|B|) Σ (S_n / B_n)
        G_drift_client = 0.0
        valid_branches_client = 0
        for state in self._branch_client_states.values():
            if state.batch_steps > 0:
                G_drift_client += state.trajectory_sum / state.batch_steps
                valid_branches_client += 1

        if valid_branches_client > 0:
            G_drift_client /= valid_branches_client

        # G_end_client = (1/|B|) Σ E_n
        G_end_client = sum(s.endpoint_drift for s in self._branch_client_states.values())
        if self._branch_client_states:
            G_end_client /= len(self._branch_client_states)

        # --- Server drift metrics ---
        delta_server_norm_sq = 0.0
        G_drift_server = 0.0
        G_end_server = 0.0
        total_server_steps = 0

        if new_master_server_state_dict is not None and self._round_start_server_params:
            # Compute delta server norm
            for name, param in new_master_server_state_dict.items():
                if name in self._round_start_server_params:
                    if isinstance(param, torch.Tensor):
                        diff = param.cpu() - self._round_start_server_params[name]
                    else:
                        diff = torch.tensor(param).cpu() - self._round_start_server_params[name]
                    delta_server_norm_sq += (diff ** 2).sum().item()

            # G_drift_server = (1/|B|) Σ (S_s / B_s)
            valid_branches_server = 0
            for state in self._branch_server_states.values():
                if state.batch_steps > 0:
                    G_drift_server += state.trajectory_sum / state.batch_steps
                    valid_branches_server += 1
                    total_server_steps += state.batch_steps

            if valid_branches_server > 0:
                G_drift_server /= valid_branches_server

            # G_end_server = (1/|B|) Σ E_s
            G_end_server = sum(s.endpoint_drift for s in self._branch_server_states.values())
            if self._branch_server_states:
                G_end_server /= len(self._branch_server_states)

        # --- Adaptive epsilon ---
        total_delta = delta_client_norm_sq + delta_server_norm_sq
        if len(self._early_delta_norms) < 10:
            self._early_delta_norms.append(total_delta)
            if len(self._early_delta_norms) == 10:
                sorted_deltas = sorted(self._early_delta_norms)
                median_delta = sorted_deltas[len(sorted_deltas) // 2]
                self._adaptive_epsilon = 1e-3 * median_delta
                vprint(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})", 2
                )

        epsilon = self._adaptive_epsilon if self._adaptive_epsilon is not None else self.epsilon

        # --- Normalized drift ---
        G_drift_norm_client = G_drift_client / (delta_client_norm_sq + epsilon)
        G_drift_norm_server = (
            G_drift_server / (delta_server_norm_sq + epsilon)
            if delta_server_norm_sq > 0
            else 0.0
        )

        # --- Combined metrics ---
        G_drift_total = G_drift_client + G_drift_server
        G_end_total = G_end_client + G_end_server

        # Update Alignment (A_cos + M_norm) — client side
        alignment_client = compute_update_alignment(self._branch_deltas)
        # Update Alignment — server side (unique to MultiSFL)
        alignment_server = compute_update_alignment(self._branch_server_deltas)

        result.metrics = DriftMetrics(
            # Client
            G_drift_client=G_drift_client,
            G_end_client=G_end_client,
            G_drift_norm_client=G_drift_norm_client,
            delta_client_norm_sq=delta_client_norm_sq,
            # Server
            G_drift_server=G_drift_server,
            G_end_server=G_end_server,
            G_drift_norm_server=G_drift_norm_server,
            delta_server_norm_sq=delta_server_norm_sq,
            # Combined
            G_drift_total=G_drift_total,
            G_end_total=G_end_total,
            # Counts
            num_branches=len(self._branch_client_states),
            server_steps=total_server_steps,
            # Update alignment
            A_cos_client=alignment_client.A_cos,
            M_norm_client=alignment_client.M_norm,
            A_cos_server=alignment_server.A_cos,
            M_norm_server=alignment_server.M_norm,
            n_valid_alignment=alignment_client.n_valid,
        )

        self.measurements.append(result)

        vprint(
            f"[Drift] Round {round_number}: "
            f"Client(G_drift={G_drift_client:.6f}, G_end={G_end_client:.6f}) "
            f"Server(G_drift={G_drift_server:.6f}, G_end={G_end_server:.6f}, steps={total_server_steps}) "
            f"Total(G_drift={G_drift_total:.6f}) "
            f"A_cos(c={alignment_client.A_cos:.4f}, s={alignment_server.A_cos:.4f})", 1
        )

        return result

    def get_history(self) -> dict:
        """Return drift measurement history"""
        return {
            # Client metrics
            "G_drift_client": [m.metrics.G_drift_client for m in self.measurements],
            "G_end_client": [m.metrics.G_end_client for m in self.measurements],
            "G_drift_norm_client": [m.metrics.G_drift_norm_client for m in self.measurements],
            "delta_client_norm_sq": [m.metrics.delta_client_norm_sq for m in self.measurements],
            # Server metrics
            "G_drift_server": [m.metrics.G_drift_server for m in self.measurements],
            "G_end_server": [m.metrics.G_end_server for m in self.measurements],
            "G_drift_norm_server": [m.metrics.G_drift_norm_server for m in self.measurements],
            "delta_server_norm_sq": [m.metrics.delta_server_norm_sq for m in self.measurements],
            # Combined
            "G_drift_total": [m.metrics.G_drift_total for m in self.measurements],
            "G_end_total": [m.metrics.G_end_total for m in self.measurements],
            # Legacy compatibility
            "G_drift": [m.metrics.G_drift_client for m in self.measurements],
            "G_end": [m.metrics.G_end_client for m in self.measurements],
            "G_drift_norm": [m.metrics.G_drift_norm_client for m in self.measurements],
            "delta_global_norm_sq": [m.metrics.delta_client_norm_sq for m in self.measurements],
            # Update alignment
            "A_cos_client": [m.metrics.A_cos_client for m in self.measurements],
            "M_norm_client": [m.metrics.M_norm_client for m in self.measurements],
            "A_cos_server": [m.metrics.A_cos_server for m in self.measurements],
            "M_norm_server": [m.metrics.M_norm_server for m in self.measurements],
            # Per-round details
            "per_round": [m.to_dict() for m in self.measurements],
        }

    def clear(self):
        """Clear all state"""
        self._round_start_client_params = {}
        self._round_start_server_params = {}
        self._branch_client_states.clear()
        self._branch_server_states.clear()
        self._branch_deltas = []
        self._branch_server_deltas = []
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
