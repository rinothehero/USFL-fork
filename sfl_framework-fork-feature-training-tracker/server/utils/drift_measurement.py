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
from shared.experiment_a_metrics import (
    compute_experiment_a_metrics,
    compute_weighted_client_consensus_update,
    safe_cosine_distance,
)
from shared.expa_iid_mu_reference import load_iid_mu_reference, save_iid_mu_round
from utils.log_utils import vprint


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
    experiment_a: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            **self.metrics.to_dict(),
            "per_client": {str(k): v for k, v in self.per_client.items()},
            "server_drift": self.server_drift,
            "experiment_a": self.experiment_a,
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

    def __init__(
        self,
        epsilon: float = 1e-8,
        iid_mu_reference_load_path: str = "",
        iid_mu_reference_save_dir: str = "",
    ):
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

        # Experiment A probe directions
        self._probe_client_direction: Optional[torch.Tensor] = None
        self._probe_server_direction: Optional[torch.Tensor] = None
        self._probe_meta: Dict[str, float] = {}
        self._per_client_probe_directions: Dict[int, torch.Tensor] = {}

        # Optional IID reference for consensus update direction:
        # B_c_vs_sfl_iid_mu^t = 1 - cos(mu_c^t, mu_c_iid^t)
        self._iid_mu_reference_load_path = iid_mu_reference_load_path or ""
        self._iid_mu_reference_save_dir = iid_mu_reference_save_dir or ""
        self._iid_mu_reference: Dict[int, torch.Tensor] = load_iid_mu_reference(
            self._iid_mu_reference_load_path
        )
        if self._iid_mu_reference_load_path:
            vprint(
                f"[Drift][ExpA-IID] Loaded IID mu refs: {len(self._iid_mu_reference)} rounds "
                f"from {self._iid_mu_reference_load_path}",
                1,
            )
        if self._iid_mu_reference_save_dir:
            vprint(
                f"[Drift][ExpA-IID] Will save IID mu refs per round to {self._iid_mu_reference_save_dir}",
                1,
            )

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
        vprint(
            f"[Drift] on_round_start: client_model={type(client_model).__name__} "
            f"({len(self._round_start_client_params)} params), "
            f"server_model={type(server_model).__name__ if server_model else 'None'} "
            f"({len(self._round_start_server_params)} params)", 2
        )

        # Reset per-round accumulators
        self._client_drifts = {}
        self._client_weights = {}
        self._client_deltas = []
        self._alignment_weights = {}
        self._server_trajectory_sum = 0.0
        self._server_batch_steps = 0
        self._server_endpoint_drift = 0.0
        self._probe_client_direction = None
        self._probe_server_direction = None
        self._probe_meta = {}
        self._per_client_probe_directions = {}

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
                self._client_drifts[client_id]["B_main"] = float(client_weight)
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

    def _flatten_model_delta(
        self, model: nn.Module, start_params: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        vecs = []
        for name, param in model.named_parameters():
            if name not in start_params:
                continue
            diff = param.data.detach().cpu().float() - start_params[name].float()
            vecs.append(diff.reshape(-1))
        if not vecs:
            return None
        return torch.cat(vecs)

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
        vprint(
            f"[Drift] on_round_end: client_matched={client_matched_params}/{client_total_params}, "
            f"server_matched={server_matched_params}/{server_total_params}", 2
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
                vprint(
                    f"[Drift] Adaptive epsilon set to {self._adaptive_epsilon:.6e} "
                    f"(median delta: {median_delta:.6f})", 2
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

        client_delta_map: Dict[int, torch.Tensor] = {}
        for cid, delta in self._client_deltas:
            client_delta_map[int(cid)] = delta.float()

        server_delta_vec = None
        if new_server_model is not None and self._round_start_server_params:
            server_delta_vec = self._flatten_model_delta(
                new_server_model, self._round_start_server_params
            )

        experiment_a = compute_experiment_a_metrics(
            client_deltas=client_delta_map,
            client_weights=self._alignment_weights or self._client_weights,
            client_probe_direction=self._probe_client_direction,
            per_client_probe_directions=self._per_client_probe_directions
            if self._per_client_probe_directions
            else None,
            server_delta=server_delta_vec,
            server_probe_direction=self._probe_server_direction,
            server_steps=self._server_batch_steps,
            epsilon=epsilon,
        )
        mu_c = compute_weighted_client_consensus_update(
            client_delta_map,
            self._alignment_weights or self._client_weights,
        )
        iid_mu_ref = self._iid_mu_reference.get(int(round_number))
        b_c_vs_sfl_iid_mu = safe_cosine_distance(mu_c, iid_mu_ref, float(epsilon))
        cos_c_vs_sfl_iid_mu = (
            float(1.0 - b_c_vs_sfl_iid_mu)
            if b_c_vs_sfl_iid_mu is not None
            else None
        )
        experiment_a["B_c_vs_sfl_iid_mu"] = (
            float(b_c_vs_sfl_iid_mu) if b_c_vs_sfl_iid_mu is not None else None
        )
        experiment_a["cos_c_vs_sfl_iid_mu"] = cos_c_vs_sfl_iid_mu
        experiment_a["sfl_iid_mu_round_available"] = iid_mu_ref is not None

        if mu_c is not None and self._iid_mu_reference_save_dir:
            saved_path = save_iid_mu_round(
                self._iid_mu_reference_save_dir, int(round_number), mu_c
            )
            if saved_path is not None:
                experiment_a["sfl_iid_mu_saved_path"] = saved_path

        experiment_a["probe"] = {
            "used_batches": int(self._probe_meta.get("used_batches", 0)),
            "used_samples": int(self._probe_meta.get("used_samples", 0)),
        }
        experiment_a["B_i"] = {
            str(cid): float(w) for cid, w in self._client_weights.items()
        }
        experiment_a["R_i"] = {}
        result.experiment_a = experiment_a

        self.measurements.append(result)

        # Log results
        vprint(
            f"[Drift] Round {round_number}: "
            f"Client(G_drift={G_drift_client:.6f}, G_end={G_end_client:.6f}) "
            f"Server(G_drift={G_drift_server:.6f}, G_end={G_end_server:.6f}, steps={self._server_batch_steps}) "
            f"Total(G_drift={G_drift_total:.6f}) "
            f"A_cos={alignment.A_cos:.6f} M_norm={alignment.M_norm:.6f} (n_valid={alignment.n_valid})", 1
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
        self._probe_client_direction = None
        self._probe_server_direction = None
        self._probe_meta = {}
        self._per_client_probe_directions = {}
        self.measurements = []
        self._early_delta_norms = []
        self._adaptive_epsilon = None
