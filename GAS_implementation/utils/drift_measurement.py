"""
Client & Server Drift Measurement Utility (SCAFFOLD-style) for GAS

Measures drift from round-start global model to track
how much local updates deviate during training.

Key Metrics:
- G_drift_client: Average client trajectory drift energy
- G_drift_server: Server model trajectory drift energy
- G_end_client/G_end_server: End-point drift
- G_drift_norm: Normalized drift (prevents "update suppression" criticism)

Usage:
1. on_round_start(client_state_dict, server_state_dict): Save x^{t,0} for both
2. accumulate_server_drift(server_state_dict): Call after each server optimizer.step()
3. accumulate_client_drift(client_id, client_state_dict): After each client optimizer.step()
4. finalize_client(client_id, client_state_dict): When client finishes local training
5. on_round_end(new_client_state_dict, new_server_state_dict): Compute G metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import copy


@dataclass
class DriftMetrics:
    """Drift measurement results for a round"""

    # Client-side drift (aggregated from all participating clients)
    G_drift_client: float = 0.0  # (1/|P_t|) Σ (S_n / B_n)
    G_end_client: float = 0.0  # (1/|P_t|) Σ E_n
    G_drift_norm_client: float = 0.0  # G_drift_client / (||Δx_c||² + ε)
    delta_client_norm_sq: float = 0.0  # ||x_c^{t+1,0} - x_c^{t,0}||²

    # Server-side drift (single server model)
    G_drift_server: float = 0.0  # S_server / B_server
    G_end_server: float = 0.0  # E_server
    G_drift_norm_server: float = 0.0  # G_drift_server / (||Δx_s||² + ε)
    delta_server_norm_sq: float = 0.0  # ||x_s^{t+1,0} - x_s^{t,0}||²

    # Combined metrics
    G_drift_total: float = 0.0  # G_drift_client + G_drift_server
    G_end_total: float = 0.0  # G_end_client + G_end_server

    num_clients: int = 0
    server_steps: int = 0

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
            "num_clients": self.num_clients,
            "server_steps": self.server_steps,
        }


@dataclass
class RoundDriftMeasurement:
    """Per-round drift measurement with client details"""
    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_client: Dict[int, dict] = field(default_factory=dict)
    server_drift: dict = field(default_factory=dict)  # {S, B, E} for server

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
            "server_drift": self.server_drift,
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
    Tracks client and server drift across rounds using SCAFFOLD-style metrics.

    Workflow (for GAS-style training):
    1. on_round_start(client_state_dict, server_state_dict): Save x^{t,0} for both
    2. accumulate_server_drift(server_state_dict): After each server optimizer.step()
    3. accumulate_client_drift(client_id, client_state_dict): After each client optimizer.step()
    4. finalize_client(client_id, client_state_dict): When client finishes local training
    5. on_round_end(new_client_state_dict, new_server_state_dict): Compute G metrics
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

        # Round-start model snapshots
        self._round_start_client_params: Dict[str, torch.Tensor] = {}
        self._round_start_server_params: Dict[str, torch.Tensor] = {}

        # Per-client drift states
        self._client_states: Dict[int, ClientDriftState] = {}

        # Server drift accumulation for current round
        self._server_trajectory_sum: float = 0.0
        self._server_batch_steps: int = 0
        self._server_endpoint_drift: float = 0.0

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

    def _is_trainable_param(self, name: str) -> bool:
        """Check if parameter name corresponds to trainable weights (not buffers)"""
        # Exclude BatchNorm running statistics and tracking buffers
        buffer_keywords = ("running_mean", "running_var", "num_batches_tracked")
        return not any(kw in name for kw in buffer_keywords)

    def on_round_start(
        self,
        client_state_dict: dict,
        server_state_dict: Optional[dict] = None,
    ):
        """
        Save global model snapshots at round start (x^{t,0})

        Args:
            client_state_dict: The global client-side model state dict
            server_state_dict: The server-side model state dict (optional)
        """
        # Only save trainable parameters (exclude BatchNorm buffers)
        self._round_start_client_params = {
            name: param.clone().detach().cpu()
            for name, param in client_state_dict.items()
            if self._is_trainable_param(name)
        }

        if server_state_dict is not None:
            self._round_start_server_params = {
                name: param.clone().detach().cpu()
                for name, param in server_state_dict.items()
                if self._is_trainable_param(name)
            }
        else:
            self._round_start_server_params = {}

        # Reset per-round accumulators
        self._client_states.clear()
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0

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
                diff = param.cpu() - start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def accumulate_server_drift(self, server_state_dict: dict):
        """
        Accumulate server drift after each server optimizer.step()

        Args:
            server_state_dict: Server model state dict after optimizer.step()
        """
        if not self._round_start_server_params:
            return

        self._server_batch_steps += 1
        drift = self._compute_drift_from_start(
            server_state_dict, self._round_start_server_params
        )
        self._server_trajectory_sum += drift
        self._server_endpoint_drift = drift

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
            drift = self._compute_drift_from_start(
                current_state_dict, self._round_start_client_params
            )
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
        state.endpoint_drift = self._compute_drift_from_start(
            current_state_dict, self._round_start_client_params
        )

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
        new_client_state_dict: dict,
        new_server_state_dict: Optional[dict] = None,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_client_state_dict: The aggregated global client model state dict
            new_server_state_dict: The server model state dict after round (optional)

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_client = {
            cid: state.to_dict() for cid, state in self._client_states.items()
        }
        result.server_drift = {
            "S": self._server_trajectory_sum,
            "B": self._server_batch_steps,
            "E": self._server_endpoint_drift,
        }

        # --- Client drift metrics ---
        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_client_norm_sq = 0.0
        for name, param in new_client_state_dict.items():
            if name in self._round_start_client_params:
                diff = param.cpu() - self._round_start_client_params[name]
                delta_client_norm_sq += (diff ** 2).sum().item()

        # G_drift_client = (1/|P_t|) Σ (S_n / B_n)
        G_drift_client = 0.0
        valid_clients = 0
        for state in self._client_states.values():
            if state.batch_steps > 0:
                G_drift_client += state.trajectory_sum / state.batch_steps
                valid_clients += 1

        if valid_clients > 0:
            G_drift_client /= valid_clients

        # G_end_client = (1/|P_t|) Σ E_n
        G_end_client = sum(s.endpoint_drift for s in self._client_states.values())
        if self._client_states:
            G_end_client /= len(self._client_states)

        # --- Server drift metrics ---
        delta_server_norm_sq = 0.0
        if new_server_state_dict is not None and self._round_start_server_params:
            for name, param in new_server_state_dict.items():
                if name in self._round_start_server_params:
                    diff = param.cpu() - self._round_start_server_params[name]
                    delta_server_norm_sq += (diff ** 2).sum().item()

        # G_drift_server = S_server / B_server
        G_drift_server = 0.0
        if self._server_batch_steps > 0:
            G_drift_server = self._server_trajectory_sum / self._server_batch_steps

        G_end_server = self._server_endpoint_drift

        # --- Adaptive epsilon ---
        total_delta = delta_client_norm_sq + delta_server_norm_sq
        if len(self._early_delta_norms) < 10:
            self._early_delta_norms.append(total_delta)
            if len(self._early_delta_norms) == 10:
                sorted_deltas = sorted(self._early_delta_norms)
                median_delta = sorted_deltas[len(sorted_deltas) // 2]
                self._adaptive_epsilon = 1e-3 * median_delta
                print(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})"
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
            num_clients=len(self._client_states),
            server_steps=self._server_batch_steps,
        )

        self.measurements.append(result)

        print(
            f"[Drift] Round {round_number}: "
            f"Client(G_drift={G_drift_client:.6f}, G_end={G_end_client:.6f}) "
            f"Server(G_drift={G_drift_server:.6f}, G_end={G_end_server:.6f}, steps={self._server_batch_steps}) "
            f"Total(G_drift={G_drift_total:.6f})"
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
            # Per-round details
            "per_round": [m.to_dict() for m in self.measurements],
        }

    def clear(self):
        """Clear all state"""
        self._round_start_client_params = {}
        self._round_start_server_params = {}
        self._client_states.clear()
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
