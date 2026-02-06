"""
Client & Server Drift Measurement Utility (SCAFFOLD-style)

Measures drift from round-start global model to track
how much local updates deviate during training.

Key Metrics:
- G_drift_client: Average client trajectory drift energy
- G_drift_server: Server model trajectory drift energy
- G_end_client/G_end_server: End-point drift
- G_drift_norm: Normalized drift (prevents "update suppression" criticism)

Usage:
1. on_round_start(client_model, server_model): Save x^{t,0} for both
2. accumulate_server_drift(): Call after each server optimizer.step()
3. collect_client_drift(client_id, S_n, B_n, E_n): Collect from clients
4. on_round_end(): Compute G metrics
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# shared/update_alignment.py lives at repo root: USFL-fork/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared.update_alignment import flatten_delta, compute_update_alignment


@dataclass
class DriftMetrics:
    """Drift measurement results for a round"""

    # Client-side drift (aggregated from all participating clients)
    G_drift_client: float = 0.0  # (1/|P_t|) Σ (S_n / B_n)
    # Step-weighted variant: ΣS / ΣB (prevents "1 client = 1 vote" distortion when B differs)
    G_drift_client_stepweighted: float = 0.0
    G_end_client: float = 0.0  # (1/|P_t|) Σ E_n
    # Client-weighted endpoint drift (weights should match aggregation weights when possible)
    G_end_client_weighted: float = 0.0
    G_drift_norm_client: float = 0.0  # G_drift_client / (||Δx_c||² + ε)
    delta_client_norm_sq: float = 0.0  # ||x_c^{t+1,0} - x_c^{t,0}||²
    # Client update disagreement around the aggregated update vector μ:
    # D_dir = E_w[||Δ_i||²] - ||E_w[Δ_i]||²  (variance identity; weights w should match aggregation)
    D_dir_client_weighted: float = 0.0
    # Scale-invariant relative disagreement
    D_rel_client_weighted: float = 0.0

    # Server-side drift (single server model)
    G_drift_server: float = 0.0  # S_server / B_server
    G_end_server: float = 0.0  # E_server
    G_drift_norm_server: float = 0.0  # G_drift_server / (||Δx_s||² + ε)
    delta_server_norm_sq: float = 0.0  # ||x_s^{t+1,0} - x_s^{t,0}||²

    # Combined metrics
    G_drift_total: float = 0.0  # G_drift_client + G_drift_server
    G_end_total: float = 0.0  # G_end_client + G_end_server

    # Update alignment (A_cos) metrics
    A_cos: float = float("nan")
    M_norm: float = 0.0
    n_valid_alignment: int = 0

    num_clients: int = 0
    server_steps: int = 0

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
            # Update alignment
            "A_cos": self.A_cos,
            "M_norm": self.M_norm,
            "n_valid_alignment": self.n_valid_alignment,
            # Counts
            "num_clients": self.num_clients,
            "server_steps": self.server_steps,
        }


@dataclass
class RoundDriftMeasurement:
    """Per-round drift measurement with client details"""

    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_client: Dict[int, dict] = field(default_factory=dict)  # {client_id: {S, B, E}}
    server_drift: dict = field(default_factory=dict)  # {S, B, E} for server

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
            "server_drift": self.server_drift,
        }


class DriftMeasurementTracker:
    """
    Tracks client and server drift across rounds using SCAFFOLD-style metrics.

    Workflow:
    1. on_round_start(client_model, server_model): Save snapshots
    2. accumulate_server_drift(server_model): Call after each server optimizer.step()
    3. collect_client_drift(): Collect S_n, B_n, E_n from each client
    4. on_round_end(): Compute G_drift, G_end, G_drift_norm for both
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small constant to prevent division by zero in G_drift_norm
        """
        self.epsilon = epsilon

        # Round-start model snapshots
        self._round_start_client_params: Dict[str, torch.Tensor] = {}
        self._round_start_server_params: Dict[str, torch.Tensor] = {}

        # Per-client drift data for current round
        self._client_drifts: Dict[int, dict] = {}  # {client_id: {S, B, E}}
        # Optional per-client weights for fair comparisons (e.g., aggregation weights).
        # Intended meaning: w_i ∝ "effective" samples used by the client in the round.
        self._client_weights: Dict[int, float] = {}

        # Update alignment (A_cos) accumulators
        self._client_deltas: List[Tuple[int, torch.Tensor]] = []
        self._alignment_weights: Dict[int, float] = {}

        # Server drift accumulation for current round
        self._server_trajectory_sum: float = 0.0
        self._server_batch_steps: int = 0
        self._server_endpoint_drift: float = 0.0

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon based on early rounds
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

    def _compute_param_drift(
        self, model: nn.Module, start_params: Dict[str, torch.Tensor]
    ) -> float:
        """Compute ||x^{t,b} - x^{t,0}||² for a model"""
        drift_sq = 0.0
        for name, param in model.named_parameters():
            if name in start_params:
                diff = param.data.cpu() - start_params[name]
                drift_sq += (diff ** 2).sum().item()
        return drift_sq

    def on_round_start(
        self, client_model: nn.Module, server_model: Optional[nn.Module] = None
    ):
        """
        Save global model snapshots at round start (x^{t,0})

        Args:
            client_model: The global client-side model at round start
            server_model: The server-side model at round start (optional)
        """
        self._round_start_client_params = {
            name: param.data.clone().detach().cpu()
            for name, param in client_model.named_parameters()
        }

        if server_model is not None:
            self._round_start_server_params = {
                name: param.data.clone().detach().cpu()
                for name, param in server_model.named_parameters()
            }
        else:
            self._round_start_server_params = {}

        # Debug: Log parameter counts and model types
        print(
            f"[Drift] on_round_start: client_model={type(client_model).__name__} "
            f"({len(self._round_start_client_params)} params), "
            f"server_model={type(server_model).__name__ if server_model else 'None'} "
            f"({len(self._round_start_server_params)} params)"
        )

        # Reset per-round accumulators
        self._client_drifts = {}
        self._client_weights = {}
        self._client_deltas = []
        self._alignment_weights = {}
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0

    def accumulate_server_drift(self, server_model: nn.Module):
        """
        Accumulate server drift after each server optimizer.step()

        Args:
            server_model: The server model after optimizer.step()
        """
        if not self._round_start_server_params:
            return

        self._server_batch_steps += 1
        drift = self._compute_param_drift(server_model, self._round_start_server_params)
        self._server_trajectory_sum += drift
        self._server_endpoint_drift = drift

    def collect_client_drift(
        self,
        client_id: int,
        drift_trajectory_sum: float,
        drift_batch_steps: int,
        drift_endpoint: float,
        client_weight: Optional[float] = None,
    ):
        """
        Collect drift scalars from a client.

        Args:
            client_id: Client identifier
            drift_trajectory_sum: S_n(t) = Σ||x^{t,b} - x^{t,0}||²
            drift_batch_steps: B_{n,t} = number of batch steps
            drift_endpoint: E_n(t) = ||x^{t,B} - x^{t,0}||²
        """
        self._client_drifts[client_id] = {
            "S": drift_trajectory_sum,
            "B": drift_batch_steps,
            "E": drift_endpoint,
        }
        if client_weight is not None:
            try:
                self._client_weights[client_id] = float(client_weight)
            except (TypeError, ValueError):
                # Keep it absent if the value is not castable.
                pass

    def collect_client_model(
        self,
        client_id: int,
        state_dict: dict,
        client_weight: float = None,
    ):
        """
        Collect a client's post-training state_dict for A_cos computation.

        Computes Δ_i = flatten(θ_end - θ_start) and stores it for later
        pairwise cosine alignment calculation.
        """
        delta = flatten_delta(state_dict, self._round_start_client_params)
        if delta is not None:
            self._client_deltas.append((client_id, delta))
        if client_weight is not None:
            self._alignment_weights[client_id] = float(client_weight)

    def on_round_end(
        self,
        round_number: int,
        new_client_model: nn.Module,
        new_server_model: Optional[nn.Module] = None,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_client_model: The aggregated global client model (x_c^{t+1,0})
            new_server_model: The server model after round (x_s^{t+1,0})

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_client = dict(self._client_drifts)
        result.server_drift = {
            "S": self._server_trajectory_sum,
            "B": self._server_batch_steps,
            "E": self._server_endpoint_drift,
        }

        # --- Client drift metrics ---
        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_client_norm_sq = 0.0
        client_matched_params = 0
        client_total_params = 0
        for name, param in new_client_model.named_parameters():
            client_total_params += 1
            if name in self._round_start_client_params:
                client_matched_params += 1
                diff = param.data.cpu() - self._round_start_client_params[name]
                delta_client_norm_sq += (diff ** 2).sum().item()

        # G_drift_client = (1/|P_t|) Σ (S_n / B_n)
        G_drift_client = 0.0
        valid_clients = 0
        sum_S = 0.0
        sum_B = 0.0
        for client_id, drift in self._client_drifts.items():
            if drift["B"] > 0:
                G_drift_client += drift["S"] / drift["B"]
                valid_clients += 1
                sum_S += drift["S"]
                sum_B += drift["B"]

        if valid_clients > 0:
            G_drift_client /= valid_clients
        G_drift_client_stepweighted = (sum_S / sum_B) if sum_B > 0 else 0.0

        # G_end_client = (1/|P_t|) Σ E_n
        G_end_client = sum(d["E"] for d in self._client_drifts.values())
        if self._client_drifts:
            G_end_client /= len(self._client_drifts)

        # --- Client update disagreement (weighted) ---
        # Use per-client weights if available, otherwise fall back to uniform weights.
        # NOTE: This metric assumes the aggregated global client model is a weighted average
        #       of client models using the same weights (e.g., FedAvg weights).
        weights = {
            cid: self._client_weights.get(cid, None) for cid in self._client_drifts.keys()
        }
        if all(w is not None for w in weights.values()):
            weight_sum = sum(float(w) for w in weights.values() if w is not None)
        else:
            weight_sum = 0.0

        if weight_sum <= 0.0:
            # Fallback: uniform weights across participating clients
            weights = {cid: 1.0 for cid in self._client_drifts.keys()}
            weight_sum = float(len(weights)) if weights else 0.0

        # Weighted mean of endpoint drift norms: E_w[||Δ_i||²]
        G_end_client_weighted = 0.0
        if weight_sum > 0.0:
            for cid, drift in self._client_drifts.items():
                w = float(weights.get(cid, 0.0))
                G_end_client_weighted += (w / weight_sum) * float(drift.get("E", 0.0))

        # Weighted directional disagreement around μ (the aggregated update):
        # D_dir = E_w[||Δ_i||²] - ||μ||², where ||μ||² == delta_client_norm_sq.
        D_dir_client_weighted = G_end_client_weighted - delta_client_norm_sq

        # --- Server drift metrics ---
        delta_server_norm_sq = 0.0
        server_matched_params = 0
        server_total_params = 0
        if new_server_model is not None and self._round_start_server_params:
            for name, param in new_server_model.named_parameters():
                server_total_params += 1
                if name in self._round_start_server_params:
                    server_matched_params += 1
                    diff = param.data.cpu() - self._round_start_server_params[name]
                    delta_server_norm_sq += (diff ** 2).sum().item()

        # Debug: Log parameter matching stats
        print(
            f"[Drift] on_round_end: client_matched={client_matched_params}/{client_total_params}, "
            f"server_matched={server_matched_params}/{server_total_params}"
        )

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

        epsilon = (
            self._adaptive_epsilon
            if self._adaptive_epsilon is not None
            else self.epsilon
        )

        # --- Normalized drift ---
        G_drift_norm_client = G_drift_client / (delta_client_norm_sq + epsilon)
        G_drift_norm_server = (
            G_drift_server / (delta_server_norm_sq + epsilon)
            if delta_server_norm_sq > 0
            else 0.0
        )

        # Relative disagreement (scale-invariant proxy)
        D_rel_client_weighted = D_dir_client_weighted / (delta_client_norm_sq + epsilon)

        # --- Combined metrics ---
        G_drift_total = G_drift_client + G_drift_server
        G_end_total = G_end_client + G_end_server

        # --- Update alignment (A_cos) ---
        alignment = compute_update_alignment(
            self._client_deltas,
            weights=self._alignment_weights if self._alignment_weights else None,
        )

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
            # Update alignment
            A_cos=alignment.A_cos,
            M_norm=alignment.M_norm,
            n_valid_alignment=alignment.n_valid,
            # Counts
            num_clients=len(self._client_drifts),
            server_steps=self._server_batch_steps,
        )

        self.measurements.append(result)

        # Log results
        print(
            f"[Drift] Round {round_number}: "
            f"Client(G_drift={G_drift_client:.6f}, G_end={G_end_client:.6f}) "
            f"Server(G_drift={G_drift_server:.6f}, G_end={G_end_server:.6f}, steps={self._server_batch_steps}) "
            f"Total(G_drift={G_drift_total:.6f}) "
            f"A_cos={alignment.A_cos:.6f} M_norm={alignment.M_norm:.6f} (n_valid={alignment.n_valid})"
        )

        return result

    def get_all_measurements(self) -> List[dict]:
        """Return all measurements as list of dicts"""
        return [m.to_dict() for m in self.measurements]

    def get_latest_measurement(self) -> Optional[RoundDriftMeasurement]:
        """Return the most recent measurement"""
        return self.measurements[-1] if self.measurements else None

    def clear(self):
        """Clear all state"""
        self._round_start_client_params = {}
        self._round_start_server_params = {}
        self._client_drifts = {}
        self._client_weights = {}
        self._client_deltas = []
        self._alignment_weights = {}
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
