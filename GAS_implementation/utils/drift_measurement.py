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
from typing import Dict, List, Optional, Tuple
import torch
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.update_alignment import flatten_delta, compute_update_alignment
from shared.experiment_a_metrics import compute_experiment_a_metrics
from log_utils import vprint


@dataclass
class DriftMetrics:
    """Drift measurement results for a round"""

    # Client-side drift (aggregated from all participating clients)
    G_drift_client: float = 0.0  # (1/|P_t|) Σ (S_n / B_n)
    # Step-weighted variant: ΣS / ΣB (useful when clients have different step counts)
    G_drift_client_stepweighted: float = 0.0
    G_end_client: float = 0.0  # (1/|P_t|) Σ E_n
    # In GAS, aggregation is uniform over participating clients, so this equals G_end_client.
    G_end_client_weighted: float = 0.0
    G_drift_norm_client: float = 0.0  # G_drift_client / (||Δx_c||² + ε)
    delta_client_norm_sq: float = 0.0  # ||x_c^{t+1,0} - x_c^{t,0}||²
    # Update disagreement around the aggregated update μ (uniform mean in GAS):
    # D_dir = E[||Δ_i||²] - ||E[Δ_i]||²
    D_dir_client_weighted: float = 0.0
    D_rel_client_weighted: float = 0.0

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

    # Update Alignment (A_cos)
    A_cos: float = float("nan")
    M_norm: float = 0.0
    n_valid_alignment: int = 0

    def to_dict(self) -> dict:
        return {
            # Client metrics
            "G_drift_client": self.G_drift_client,
            "G_drift_client_stepweighted": self.G_drift_client_stepweighted,
            "G_end_client": self.G_end_client,
            "G_end_client_weighted": self.G_end_client_weighted,
            "G_drift_norm_client": self.G_drift_norm_client,
            "delta_client_norm_sq": self.delta_client_norm_sq,
            "D_dir_client_weighted": self.D_dir_client_weighted,
            "D_rel_client_weighted": self.D_rel_client_weighted,
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
            # Update Alignment
            "A_cos": self.A_cos,
            "M_norm": self.M_norm,
            "n_valid_alignment": self.n_valid_alignment,
        }


@dataclass
class RoundDriftMeasurement:
    """Per-round drift measurement with client details"""
    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_client: Dict[int, dict] = field(default_factory=dict)
    server_drift: dict = field(default_factory=dict)  # {S, B, E} for server
    experiment_a: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
            "server_drift": self.server_drift,
            "experiment_a": self.experiment_a,
        }


class ClientDriftState:
    """Per-client drift accumulation state"""
    def __init__(self):
        self.trajectory_sum: float = 0.0  # S_n(t)
        self.batch_steps: int = 0  # B_n(t)
        self.sample_count: int = 0  # main samples processed in round
        self.endpoint_drift: float = 0.0  # E_n(t)

    def reset(self):
        self.trajectory_sum = 0.0
        self.batch_steps = 0
        self.sample_count = 0
        self.endpoint_drift = 0.0

    def to_dict(self) -> dict:
        return {
            "S": self.trajectory_sum,
            "B": self.batch_steps,
            "N": self.sample_count,
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

        # Client deltas for A_cos computation
        self._client_deltas: List[Tuple[int, torch.Tensor]] = []
        # Experiment A client deltas (Δ_i from each participant's own round-start state)
        self._experiment_a_client_deltas: Dict[int, torch.Tensor] = {}
        self._client_start_params: Dict[int, Dict[str, torch.Tensor]] = {}

        # Experiment A probe directions
        self._probe_client_direction: Optional[torch.Tensor] = None
        self._probe_server_direction: Optional[torch.Tensor] = None
        self._probe_meta: Dict[str, float] = {}
        self._per_client_probe_directions: Dict[int, torch.Tensor] = {}

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
        self._client_deltas = []
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0
        self._probe_client_direction = None
        self._probe_server_direction = None
        self._probe_meta = {}
        self._per_client_probe_directions = {}
        self._experiment_a_client_deltas = {}
        self._client_start_params = {}

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

    def accumulate_client_drift(
        self,
        client_id: int,
        current_state_dict: dict,
        batch_samples: Optional[int] = None,
    ):
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
        if batch_samples is not None:
            try:
                state.sample_count += max(int(batch_samples), 0)
            except (TypeError, ValueError):
                pass

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

    def collect_client_delta(self, client_id: int, client_state_dict: dict):
        """Compute and store Δ_i = θ_end - θ_start for A_cos computation."""
        delta = flatten_delta(client_state_dict, self._round_start_client_params)
        if delta is not None:
            self._client_deltas.append((client_id, delta))
        start_params = self._client_start_params.get(int(client_id))
        if start_params is None:
            start_params = self._round_start_client_params
        exp_delta = flatten_delta(client_state_dict, start_params)
        if exp_delta is not None:
            self._experiment_a_client_deltas[int(client_id)] = exp_delta

    def record_client_start_state(self, client_id: int, client_state_dict: dict):
        """Record x_{c,i}^{t,0} for Experiment A in async settings."""
        self._client_start_params[int(client_id)] = {
            name: param.clone().detach().cpu()
            for name, param in client_state_dict.items()
            if self._is_trainable_param(name)
        }

    def set_probe_directions(
        self,
        client_direction: Optional[torch.Tensor],
        server_direction: Optional[torch.Tensor],
        meta: Optional[dict] = None,
    ):
        self._probe_client_direction = (
            client_direction.detach().cpu().float()
            if isinstance(client_direction, torch.Tensor)
            else None
        )
        self._probe_server_direction = (
            server_direction.detach().cpu().float()
            if isinstance(server_direction, torch.Tensor)
            else None
        )
        self._probe_meta = dict(meta or {})

    def set_per_client_probe_directions(self, per_client: Dict[int, torch.Tensor]):
        self._per_client_probe_directions = {}
        for cid, direction in (per_client or {}).items():
            if isinstance(direction, torch.Tensor):
                self._per_client_probe_directions[int(cid)] = (
                    direction.detach().cpu().float()
                )

    def _flatten_state_delta(
        self,
        current_state_dict: dict,
        start_params: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not start_params:
            return None
        vecs = []
        for name, start in start_params.items():
            if name not in current_state_dict:
                continue
            end = current_state_dict[name]
            if isinstance(end, torch.Tensor):
                diff = end.detach().cpu().float() - start.float()
            else:
                diff = torch.tensor(end, dtype=torch.float32) - start.float()
            vecs.append(diff.reshape(-1))
        if not vecs:
            return None
        return torch.cat(vecs)

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
        sum_S = 0.0
        sum_B = 0.0
        for state in self._client_states.values():
            if state.batch_steps > 0:
                G_drift_client += state.trajectory_sum / state.batch_steps
                valid_clients += 1
                sum_S += state.trajectory_sum
                sum_B += state.batch_steps

        if valid_clients > 0:
            G_drift_client /= valid_clients
        G_drift_client_stepweighted = (sum_S / sum_B) if sum_B > 0 else 0.0

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
                vprint(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})", 1
                )

        epsilon = self._adaptive_epsilon if self._adaptive_epsilon is not None else self.epsilon

        # --- Normalized drift ---
        G_drift_norm_client = G_drift_client / (delta_client_norm_sq + epsilon)
        G_drift_norm_server = (
            G_drift_server / (delta_server_norm_sq + epsilon)
            if delta_server_norm_sq > 0
            else 0.0
        )

        # --- Update disagreement (uniform mean update in GAS) ---
        # In GAS, the global client model is updated by uniform averaging over the
        # participating clients, so:
        #   D_dir = E[||Δ_i||²] - ||Δ_global||²
        # where ||Δ_global||² == delta_client_norm_sq.
        G_end_client_weighted = G_end_client
        D_dir_client_weighted = G_end_client_weighted - delta_client_norm_sq
        D_rel_client_weighted = D_dir_client_weighted / (delta_client_norm_sq + epsilon)

        # --- Combined metrics ---
        G_drift_total = G_drift_client + G_drift_server
        G_end_total = G_end_client + G_end_server

        # Update Alignment (A_cos + M_norm)
        alignment = compute_update_alignment(self._client_deltas)

        result.metrics = DriftMetrics(
            # Client
            G_drift_client=G_drift_client,
            G_drift_client_stepweighted=G_drift_client_stepweighted,
            G_end_client=G_end_client,
            G_end_client_weighted=G_end_client_weighted,
            G_drift_norm_client=G_drift_norm_client,
            delta_client_norm_sq=delta_client_norm_sq,
            D_dir_client_weighted=D_dir_client_weighted,
            D_rel_client_weighted=D_rel_client_weighted,
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
            # Update Alignment
            A_cos=alignment.A_cos,
            M_norm=alignment.M_norm,
            n_valid_alignment=alignment.n_valid,
        )

        client_delta_map: Dict[int, torch.Tensor] = {
            int(cid): delta.float()
            for cid, delta in self._experiment_a_client_deltas.items()
        }
        if not client_delta_map:
            for cid, delta in self._client_deltas:
                client_delta_map[int(cid)] = delta.float()

        server_delta_vec = None
        if new_server_state_dict is not None and self._round_start_server_params:
            server_delta_vec = self._flatten_state_delta(
                new_server_state_dict, self._round_start_server_params
            )

        client_weights = {
            int(cid): float(state.sample_count)
            for cid, state in self._client_states.items()
            if state.sample_count > 0
        }

        experiment_a = compute_experiment_a_metrics(
            client_deltas=client_delta_map,
            client_weights=client_weights if client_weights else None,
            client_probe_direction=self._probe_client_direction,
            per_client_probe_directions=self._per_client_probe_directions
            if self._per_client_probe_directions
            else None,
            server_delta=server_delta_vec,
            server_probe_direction=self._probe_server_direction,
            server_steps=self._server_batch_steps,
            epsilon=epsilon,
        )
        experiment_a["probe"] = {
            "used_batches": int(self._probe_meta.get("used_batches", 0)),
            "used_samples": int(self._probe_meta.get("used_samples", 0)),
        }
        experiment_a["B_i"] = {
            str(cid): float(state.sample_count)
            for cid, state in self._client_states.items()
        }
        experiment_a["R_i"] = {}
        result.experiment_a = experiment_a

        self.measurements.append(result)

        vprint(
            f"[Drift] Round {round_number}: "
            f"Client(G_drift={G_drift_client:.6f}, G_end={G_end_client:.6f}) "
            f"Server(G_drift={G_drift_server:.6f}, G_end={G_end_server:.6f}, steps={self._server_batch_steps}) "
            f"Total(G_drift={G_drift_total:.6f}) "
            f"A_cos={alignment.A_cos:.4f}", 1
        )

        return result

    def get_history(self) -> dict:
        """Return drift measurement history"""
        return {
            # Client metrics
            "G_drift_client": [m.metrics.G_drift_client for m in self.measurements],
            "G_drift_client_stepweighted": [m.metrics.G_drift_client_stepweighted for m in self.measurements],
            "G_end_client": [m.metrics.G_end_client for m in self.measurements],
            "G_end_client_weighted": [m.metrics.G_end_client_weighted for m in self.measurements],
            "G_drift_norm_client": [m.metrics.G_drift_norm_client for m in self.measurements],
            "delta_client_norm_sq": [m.metrics.delta_client_norm_sq for m in self.measurements],
            "D_dir_client_weighted": [m.metrics.D_dir_client_weighted for m in self.measurements],
            "D_rel_client_weighted": [m.metrics.D_rel_client_weighted for m in self.measurements],
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
            # Update Alignment
            "A_cos": [m.metrics.A_cos for m in self.measurements],
            "M_norm": [m.metrics.M_norm for m in self.measurements],
            # Experiment A
            "expA_A_c_ratio": [m.experiment_a.get("A_c_ratio") for m in self.measurements],
            "expA_A_c_rel": [m.experiment_a.get("A_c_rel") for m in self.measurements],
            "expA_B_c": [m.experiment_a.get("B_c") for m in self.measurements],
            "expA_C_c": [m.experiment_a.get("C_c") for m in self.measurements],
            "expA_C_c_per_client_probe": [
                m.experiment_a.get("C_c_per_client_probe") for m in self.measurements
            ],
            "expA_B_s": [m.experiment_a.get("B_s") for m in self.measurements],
            "expA_m2_c": [m.experiment_a.get("m2_c") for m in self.measurements],
            "expA_u2_c": [m.experiment_a.get("u2_c") for m in self.measurements],
            "expA_var_c": [m.experiment_a.get("var_c") for m in self.measurements],
            "expA_server_mag_per_step": [
                m.experiment_a.get("server_mag_per_step") for m in self.measurements
            ],
            "expA_server_mag_per_step_sq": [
                m.experiment_a.get("server_mag_per_step_sq")
                for m in self.measurements
            ],
            "experiment_a": [m.experiment_a for m in self.measurements],
            # Per-round details
            "per_round": [m.to_dict() for m in self.measurements],
        }

    def clear(self):
        """Clear all state"""
        self._round_start_client_params = {}
        self._round_start_server_params = {}
        self._client_states.clear()
        self._client_deltas = []
        self._experiment_a_client_deltas = {}
        self._client_start_params = {}
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0
        self._probe_client_direction = None
        self._probe_server_direction = None
        self._probe_meta = {}
        self._per_client_probe_directions = {}
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
