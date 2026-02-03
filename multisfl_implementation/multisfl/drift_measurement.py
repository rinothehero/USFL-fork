"""
Client Drift Measurement Utility (SCAFFOLD-style) for MultiSFL

Measures client drift from round-start global model to track
how much local updates deviate during training.

Key Metrics:
- G_drift: Average trajectory drift energy (SCAFFOLD-style)
- G_end: End-point drift (simpler, lighter)
- G_drift_norm: Normalized drift (prevents "update suppression" criticism)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class DriftMetrics:
    """Drift measurement results for a round"""
    G_drift: float = 0.0  # (1/|P_t|) Σ (S_n / B_n)
    G_end: float = 0.0  # (1/|P_t|) Σ E_n
    G_drift_norm: float = 0.0  # G_drift / (||Δx_c^t||² + ε)
    delta_global_norm_sq: float = 0.0  # ||x_c^{t+1,0} - x_c^{t,0}||²
    num_clients: int = 0

    def to_dict(self) -> dict:
        return {
            "G_drift": self.G_drift,
            "G_end": self.G_end,
            "G_drift_norm": self.G_drift_norm,
            "delta_global_norm_sq": self.delta_global_norm_sq,
            "num_clients": self.num_clients,
        }


@dataclass
class RoundDriftMeasurement:
    """Per-round drift measurement with client/branch details"""
    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_branch: Dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_branch": {str(k): v for k, v in self.per_branch.items()},
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
    Tracks client drift across rounds for MultiSFL architecture.

    In MultiSFL, multiple branches train concurrently, each with its own client.
    Drift is measured per-branch and aggregated.

    Workflow:
    1. on_round_start(master_client_model): Save master client model snapshot
    2. accumulate_branch_drift(branch_id, branch_client_model): After each local step
    3. finalize_branch(branch_id, branch_client_model, client_id): When branch completes
    4. on_round_end(new_master_client_model): Compute G metrics
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

        # Round-start master client model snapshot
        self._round_start_params: Dict[str, torch.Tensor] = {}

        # Per-branch drift states
        self._branch_states: Dict[int, BranchDriftState] = {}

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

    def on_round_start(self, master_client_state_dict: dict):
        """
        Save master client model snapshot at round start (x_c^{t,0})

        Args:
            master_client_state_dict: The master client model state dict
        """
        self._round_start_params = {
            name: param.clone().detach().cpu()
            for name, param in master_client_state_dict.items()
        }
        self._branch_states.clear()

    def _compute_drift_from_start(self, current_state_dict: dict) -> float:
        """Compute ||x_c^{t,b} - x_c^{t,0}||²"""
        if not self._round_start_params:
            return 0.0
        drift_sq = 0.0
        for name, param in current_state_dict.items():
            if name in self._round_start_params:
                if isinstance(param, torch.Tensor):
                    diff = param.cpu() - self._round_start_params[name]
                else:
                    diff = torch.tensor(param).cpu() - self._round_start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def accumulate_branch_drift(self, branch_id: int, branch_client_model: nn.Module):
        """
        Accumulate drift after a branch's local step.

        Args:
            branch_id: Branch identifier
            branch_client_model: Branch's client model after optimizer step
        """
        if branch_id not in self._branch_states:
            self._branch_states[branch_id] = BranchDriftState()

        state = self._branch_states[branch_id]
        state.batch_steps += 1

        # Sample according to interval
        if state.batch_steps % self.sample_interval == 0:
            current_sd = branch_client_model.state_dict()
            drift = self._compute_drift_from_start(current_sd)
            state.trajectory_sum += drift

    def finalize_branch(
        self,
        branch_id: int,
        branch_client_model: nn.Module,
        client_id: Optional[int] = None,
    ):
        """
        Finalize drift measurement for a branch when it completes local training.

        Args:
            branch_id: Branch identifier
            branch_client_model: Final branch client model
            client_id: Optional client ID that was assigned to this branch
        """
        if branch_id not in self._branch_states:
            self._branch_states[branch_id] = BranchDriftState()

        state = self._branch_states[branch_id]
        state.client_id = client_id

        # Compute endpoint drift
        current_sd = branch_client_model.state_dict()
        state.endpoint_drift = self._compute_drift_from_start(current_sd)

        # Extrapolate if sampling was used
        if self.sample_interval > 1 and state.batch_steps > 0:
            sampled_steps = state.batch_steps // self.sample_interval
            if sampled_steps > 0:
                state.trajectory_sum = (
                    state.trajectory_sum * state.batch_steps / sampled_steps
                )

    def on_round_end(
        self,
        round_number: int,
        new_master_client_state_dict: dict,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_master_client_state_dict: The aggregated master client model state dict

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_branch = {
            bid: state.to_dict() for bid, state in self._branch_states.items()
        }

        if not self._branch_states:
            self.measurements.append(result)
            return result

        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_norm_sq = 0.0
        for name, param in new_master_client_state_dict.items():
            if name in self._round_start_params:
                if isinstance(param, torch.Tensor):
                    diff = param.cpu() - self._round_start_params[name]
                else:
                    diff = torch.tensor(param).cpu() - self._round_start_params[name]
                delta_norm_sq += (diff ** 2).sum().item()

        # Track early deltas for adaptive epsilon
        if len(self._early_delta_norms) < 10:
            self._early_delta_norms.append(delta_norm_sq)
            if len(self._early_delta_norms) == 10:
                sorted_deltas = sorted(self._early_delta_norms)
                median_delta = sorted_deltas[len(sorted_deltas) // 2]
                self._adaptive_epsilon = 1e-3 * median_delta
                print(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})"
                )

        epsilon = self._adaptive_epsilon if self._adaptive_epsilon is not None else self.epsilon

        # G_drift = (1/|P_t|) Σ (S_n / B_n)
        G_drift = 0.0
        valid_branches = 0
        for state in self._branch_states.values():
            if state.batch_steps > 0:
                G_drift += state.trajectory_sum / state.batch_steps
                valid_branches += 1

        if valid_branches > 0:
            G_drift /= valid_branches

        # G_end = (1/|P_t|) Σ E_n
        G_end = sum(s.endpoint_drift for s in self._branch_states.values())
        if self._branch_states:
            G_end /= len(self._branch_states)

        # G_drift_norm = G_drift / (||Δx_c^t||² + ε)
        G_drift_norm = G_drift / (delta_norm_sq + epsilon)

        result.metrics = DriftMetrics(
            G_drift=G_drift,
            G_end=G_end,
            G_drift_norm=G_drift_norm,
            delta_global_norm_sq=delta_norm_sq,
            num_clients=len(self._branch_states),
        )

        self.measurements.append(result)

        print(
            f"[Drift] Round {round_number}: "
            f"G_drift={G_drift:.6f}, G_end={G_end:.6f}, "
            f"G_drift_norm={G_drift_norm:.4f}, "
            f"Δ||x||²={delta_norm_sq:.6f}, "
            f"branches={len(self._branch_states)}"
        )

        return result

    def get_history(self) -> dict:
        """Return drift measurement history"""
        return {
            "G_drift": [m.metrics.G_drift for m in self.measurements],
            "G_end": [m.metrics.G_end for m in self.measurements],
            "G_drift_norm": [m.metrics.G_drift_norm for m in self.measurements],
            "delta_global_norm_sq": [m.metrics.delta_global_norm_sq for m in self.measurements],
            "per_round": [m.to_dict() for m in self.measurements],
        }

    def clear(self):
        """Clear all state"""
        self._round_start_params = {}
        self._branch_states.clear()
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
