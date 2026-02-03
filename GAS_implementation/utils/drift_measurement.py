"""
Client Drift Measurement Utility (SCAFFOLD-style) for GAS

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
import copy


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
    """Per-round drift measurement with client details"""
    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_client: Dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
        }


class ClientDriftState:
    """Per-client drift accumulation state"""
    def __init__(self):
        self.trajectory_sum: float = 0.0  # S_n(t)
        self.batch_steps: int = 0  # B_n(t)
        self.endpoint_drift: float = 0.0  # E_n(t)

    def reset(self):
        self.trajectory_sum = 0.0
        self.batch_steps = 0
        self.endpoint_drift = 0.0

    def to_dict(self) -> dict:
        return {
            "S": self.trajectory_sum,
            "B": self.batch_steps,
            "E": self.endpoint_drift,
        }


class DriftMeasurementTracker:
    """
    Tracks client drift across rounds using SCAFFOLD-style metrics.

    Workflow (for GAS-style training):
    1. on_round_start(global_model_state_dict): Save x_c^{t,0}
    2. accumulate_client_drift(client_id, current_model_state_dict): After each optimizer.step()
    3. finalize_client(client_id, current_model_state_dict): When client finishes local training
    4. on_round_end(new_global_model_state_dict): Compute G metrics
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

        # Round-start global client model snapshot
        self._round_start_params: Dict[str, torch.Tensor] = {}

        # Per-client drift states
        self._client_states: Dict[int, ClientDriftState] = {}

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

    def on_round_start(self, global_model_state_dict: dict):
        """
        Save global client model snapshot at round start (x_c^{t,0})

        Args:
            global_model_state_dict: The global client-side model state dict
        """
        self._round_start_params = {
            name: param.clone().detach().cpu()
            for name, param in global_model_state_dict.items()
        }
        self._client_states.clear()

    def _compute_drift_from_start(self, current_state_dict: dict) -> float:
        """Compute ||x_c^{t,b} - x_c^{t,0}||²"""
        if not self._round_start_params:
            return 0.0
        drift_sq = 0.0
        for name, param in current_state_dict.items():
            if name in self._round_start_params:
                diff = param.cpu() - self._round_start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def accumulate_client_drift(self, client_id: int, current_state_dict: dict):
        """
        Accumulate drift after a client optimizer step.

        Args:
            client_id: Client identifier
            current_state_dict: Client model state dict after optimizer.step()
        """
        if client_id not in self._client_states:
            self._client_states[client_id] = ClientDriftState()

        state = self._client_states[client_id]
        state.batch_steps += 1

        # Sample according to interval
        if state.batch_steps % self.sample_interval == 0:
            drift = self._compute_drift_from_start(current_state_dict)
            state.trajectory_sum += drift

    def finalize_client(self, client_id: int, current_state_dict: dict):
        """
        Finalize drift measurement for a client when it completes local training.

        Args:
            client_id: Client identifier
            current_state_dict: Final client model state dict
        """
        if client_id not in self._client_states:
            self._client_states[client_id] = ClientDriftState()

        state = self._client_states[client_id]

        # Compute endpoint drift
        state.endpoint_drift = self._compute_drift_from_start(current_state_dict)

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
        new_global_model_state_dict: dict,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_global_model_state_dict: The aggregated global model state dict

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_client = {
            cid: state.to_dict() for cid, state in self._client_states.items()
        }

        if not self._client_states:
            self.measurements.append(result)
            return result

        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_norm_sq = 0.0
        for name, param in new_global_model_state_dict.items():
            if name in self._round_start_params:
                diff = param.cpu() - self._round_start_params[name]
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
        valid_clients = 0
        for state in self._client_states.values():
            if state.batch_steps > 0:
                G_drift += state.trajectory_sum / state.batch_steps
                valid_clients += 1

        if valid_clients > 0:
            G_drift /= valid_clients

        # G_end = (1/|P_t|) Σ E_n
        G_end = sum(s.endpoint_drift for s in self._client_states.values())
        if self._client_states:
            G_end /= len(self._client_states)

        # G_drift_norm = G_drift / (||Δx_c^t||² + ε)
        G_drift_norm = G_drift / (delta_norm_sq + epsilon)

        result.metrics = DriftMetrics(
            G_drift=G_drift,
            G_end=G_end,
            G_drift_norm=G_drift_norm,
            delta_global_norm_sq=delta_norm_sq,
            num_clients=len(self._client_states),
        )

        self.measurements.append(result)

        print(
            f"[Drift] Round {round_number}: "
            f"G_drift={G_drift:.6f}, G_end={G_end:.6f}, "
            f"G_drift_norm={G_drift_norm:.4f}, "
            f"Δ||x||²={delta_norm_sq:.6f}, "
            f"clients={len(self._client_states)}"
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
        self._client_states.clear()
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
