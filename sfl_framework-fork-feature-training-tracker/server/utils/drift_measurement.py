"""
Client Drift Measurement Utility (SCAFFOLD-style)

Measures client drift from round-start global model to track
how much local updates deviate during training.

Key Metrics:
- G_drift: Average trajectory drift energy (SCAFFOLD-style)
- G_end: End-point drift (simpler, lighter)
- G_drift_norm: Normalized drift (prevents "update suppression" criticism)

Usage:
1. on_round_start(client_model): Save x_c^{t,0}
2. collect_client_drift(client_id, S_n, B_n, E_n): Collect scalars from clients
3. on_round_end(new_global_params): Compute G metrics
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
    """Per-round drift measurement with client details"""

    round_number: int
    metrics: DriftMetrics = field(default_factory=DriftMetrics)
    per_client: Dict[int, dict] = field(default_factory=dict)  # {client_id: {S, B, E}}

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
        }


class DriftMeasurementTracker:
    """
    Tracks client drift across rounds using SCAFFOLD-style metrics.

    Workflow:
    1. on_round_start(): Save global client model snapshot
    2. collect_client_drift(): Collect S_n, B_n, E_n from each client
    3. on_round_end(): Compute G_drift, G_end, G_drift_norm
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small constant to prevent division by zero in G_drift_norm
        """
        self.epsilon = epsilon

        # Round-start global client model snapshot
        self._round_start_params: Dict[str, torch.Tensor] = {}

        # Per-client drift data for current round
        self._client_drifts: Dict[int, dict] = {}  # {client_id: {S, B, E}}

        # Historical measurements
        self.measurements: List[RoundDriftMeasurement] = []

        # For adaptive epsilon based on early rounds
        self._early_delta_norms: List[float] = []
        self._adaptive_epsilon: Optional[float] = None

    def on_round_start(self, client_model: nn.Module):
        """
        Save global client model snapshot at round start (x_c^{t,0})

        Args:
            client_model: The global client-side model at round start
        """
        self._round_start_params = {
            name: param.data.clone().detach().cpu()
            for name, param in client_model.named_parameters()
        }
        self._client_drifts = {}

    def collect_client_drift(
        self,
        client_id: int,
        drift_trajectory_sum: float,
        drift_batch_steps: int,
        drift_endpoint: float,
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

    def on_round_end(
        self,
        round_number: int,
        new_global_model: nn.Module,
    ) -> RoundDriftMeasurement:
        """
        Compute drift metrics at round end.

        Args:
            round_number: Current round number
            new_global_model: The aggregated global client model (x_c^{t+1,0})

        Returns:
            RoundDriftMeasurement with computed metrics
        """
        result = RoundDriftMeasurement(round_number=round_number)
        result.per_client = dict(self._client_drifts)

        if not self._client_drifts:
            self.measurements.append(result)
            return result

        # Compute Δx_c^t = x_c^{t+1,0} - x_c^{t,0}
        delta_norm_sq = 0.0
        for name, param in new_global_model.named_parameters():
            if name in self._round_start_params:
                diff = param.data.cpu() - self._round_start_params[name]
                delta_norm_sq += (diff ** 2).sum().item()

        # Track early deltas for adaptive epsilon
        if len(self._early_delta_norms) < 10:
            self._early_delta_norms.append(delta_norm_sq)
            if len(self._early_delta_norms) == 10:
                # Set adaptive epsilon as 1e-3 * median of early deltas
                sorted_deltas = sorted(self._early_delta_norms)
                median_delta = sorted_deltas[len(sorted_deltas) // 2]
                self._adaptive_epsilon = 1e-3 * median_delta
                print(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})"
                )

        # Use adaptive epsilon if available, otherwise fixed
        epsilon = (
            self._adaptive_epsilon
            if self._adaptive_epsilon is not None
            else self.epsilon
        )

        # G_drift = (1/|P_t|) Σ (S_n / B_n)
        G_drift = 0.0
        valid_clients = 0
        for client_id, drift in self._client_drifts.items():
            if drift["B"] > 0:
                G_drift += drift["S"] / drift["B"]
                valid_clients += 1

        if valid_clients > 0:
            G_drift /= valid_clients

        # G_end = (1/|P_t|) Σ E_n
        G_end = sum(d["E"] for d in self._client_drifts.values())
        if self._client_drifts:
            G_end /= len(self._client_drifts)

        # G_drift_norm = G_drift / (||Δx_c^t||² + ε)
        G_drift_norm = G_drift / (delta_norm_sq + epsilon)

        result.metrics = DriftMetrics(
            G_drift=G_drift,
            G_end=G_end,
            G_drift_norm=G_drift_norm,
            delta_global_norm_sq=delta_norm_sq,
            num_clients=len(self._client_drifts),
        )

        self.measurements.append(result)

        # Log results
        print(
            f"[Drift] Round {round_number}: "
            f"G_drift={G_drift:.6f}, G_end={G_end:.6f}, "
            f"G_drift_norm={G_drift_norm:.4f}, "
            f"Δ||x||²={delta_norm_sq:.6f}, "
            f"clients={len(self._client_drifts)}"
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
        self._round_start_params = {}
        self._client_drifts = {}
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
