"""
G (Gradient Distance to Oracle) Measurement for MultiSFL

Measures gradient distance between current training state and Oracle (ideal) state.
Uses the same measurement protocol as GAS-fork and sfl_framework-fork:
- Oracle calculation: reduction='mean' + divide by num_batches
- Unified for fair comparison across frameworks

Metrics:
- Client G: Distance of client model gradient from Oracle
- Server G: Distance of server model gradient from Oracle (with replay features)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class GMetrics:
    G: float = 0.0
    G_rel: float = 0.0
    D_cosine: float = 0.0

    def to_dict(self) -> dict:
        return {"G": self.G, "G_rel": self.G_rel, "D_cosine": self.D_cosine}


@dataclass
class RoundGResult:
    round_idx: int
    client_g: GMetrics = field(default_factory=GMetrics)
    server_g: GMetrics = field(default_factory=GMetrics)
    per_client_g: Dict[int, GMetrics] = field(default_factory=dict)
    per_branch_server_g: Dict[int, GMetrics] = field(default_factory=dict)
    variance_client_g: float = float("nan")
    variance_client_g_rel: float = float("nan")
    variance_server_g: float = float("nan")
    variance_server_g_rel: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "round": self.round_idx,
            "client_g": self.client_g.to_dict(),
            "server_g": self.server_g.to_dict(),
            "per_client_g": {str(k): v.to_dict() for k, v in self.per_client_g.items()},
            "per_branch_server_g": {
                str(k): v.to_dict() for k, v in self.per_branch_server_g.items()
            },
            "variance_client_g": self.variance_client_g,
            "variance_client_g_rel": self.variance_client_g_rel,
            "variance_server_g": self.variance_server_g,
            "variance_server_g_rel": self.variance_server_g_rel,
        }


def backup_bn_stats(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    backup = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[name] = {
                "running_mean": (
                    module.running_mean.clone()
                    if module.running_mean is not None
                    else None
                ),
                "running_var": (
                    module.running_var.clone()
                    if module.running_var is not None
                    else None
                ),
                "num_batches_tracked": (
                    module.num_batches_tracked.clone()
                    if module.num_batches_tracked is not None
                    else None
                ),
            }
    return backup


def restore_bn_stats(model: nn.Module, backup: Dict[str, Dict[str, torch.Tensor]]):
    for name, module in model.named_modules():
        if name in backup:
            stats = backup[name]
            if stats["running_mean"] is not None and module.running_mean is not None:
                module.running_mean.copy_(stats["running_mean"])
            if stats["running_var"] is not None and module.running_var is not None:
                module.running_var.copy_(stats["running_var"])
            if (
                stats["num_batches_tracked"] is not None
                and module.num_batches_tracked is not None
            ):
                module.num_batches_tracked.copy_(stats["num_batches_tracked"])


def gradient_to_vector(
    grad_dict: Dict[str, torch.Tensor], reference_names: Optional[List[str]] = None
) -> torch.Tensor:
    if not grad_dict:
        return torch.zeros(1)

    if reference_names is None:
        reference_names = sorted(grad_dict.keys())

    vectors = []
    for name in reference_names:
        if name in grad_dict:
            vectors.append(grad_dict[name].flatten().float())

    if not vectors:
        return torch.zeros(1)

    return torch.cat(vectors)


def compute_g_metrics(
    g_tilde: Dict[str, torch.Tensor],
    g_oracle: Dict[str, torch.Tensor],
    epsilon: float = 1e-8,
) -> GMetrics:
    if not g_tilde or not g_oracle:
        return GMetrics()

    tilde_keys = set(g_tilde.keys())
    oracle_keys = set(g_oracle.keys())
    if tilde_keys != oracle_keys:
        missing = sorted(oracle_keys - tilde_keys)
        extra = sorted(tilde_keys - oracle_keys)
        raise ValueError(
            "[G Measurement] Param key mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}, "
            f"missing_count={len(missing)}, extra_count={len(extra)}"
        )

    common_keys = sorted(tilde_keys)
    if not common_keys:
        return GMetrics()

    g_tilde_flat = gradient_to_vector(g_tilde, common_keys)
    g_oracle_flat = gradient_to_vector(g_oracle, common_keys)

    if torch.isnan(g_tilde_flat).any() or torch.isinf(g_tilde_flat).any():
        return GMetrics(G=float("nan"), G_rel=float("nan"), D_cosine=1.0)
    if torch.isnan(g_oracle_flat).any() or torch.isinf(g_oracle_flat).any():
        return GMetrics(G=float("nan"), G_rel=float("nan"), D_cosine=1.0)

    diff = g_tilde_flat - g_oracle_flat
    G = torch.dot(diff, diff).item()

    oracle_norm_sq = torch.dot(g_oracle_flat, g_oracle_flat).item()
    G_rel = G / (oracle_norm_sq + epsilon)

    tilde_norm = torch.norm(g_tilde_flat).item()
    oracle_norm = torch.norm(g_oracle_flat).item()
    if tilde_norm > epsilon and oracle_norm > epsilon:
        dot_product = torch.dot(g_tilde_flat, g_oracle_flat).item()
        cos_sim = dot_product / (tilde_norm * oracle_norm)
        D_cosine = 1.0 - cos_sim
    else:
        D_cosine = 1.0

    D_cosine = max(0.0, min(2.0, D_cosine))

    return GMetrics(G=G, G_rel=G_rel, D_cosine=D_cosine)


class OracleCalculator:
    def __init__(self, full_dataloader: DataLoader, device: str = "cpu"):
        self.full_dataloader = full_dataloader
        self.device = device

    def compute_oracle_gradient(
        self, client_model: nn.Module, server_model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute Oracle gradient using full training data.
        Uses reduction='mean' + divide by num_batches.
        """
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)

        client_bn_backup = backup_bn_stats(client_model)
        server_bn_backup = backup_bn_stats(server_model)

        client_model.train()
        server_model.train()

        client_grad_accum: Dict[str, torch.Tensor] = {}
        server_grad_accum: Dict[str, torch.Tensor] = {}
        total_samples = 0
        num_batches = 0

        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue

            client_model.zero_grad(set_to_none=True)
            server_model.zero_grad(set_to_none=True)

            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            num_batches += 1

            activation = client_model(data)
            if isinstance(activation, tuple):
                activation = activation[0]

            activation_detached = activation.detach().requires_grad_(True)
            logits = server_model(activation_detached)

            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()

            activation.backward(activation_detached.grad)

            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    grad_cpu = param.grad.clone().detach().cpu()
                    if name not in client_grad_accum:
                        client_grad_accum[name] = grad_cpu
                    else:
                        client_grad_accum[name] += grad_cpu

            for name, param in server_model.named_parameters():
                if param.grad is not None:
                    grad_cpu = param.grad.clone().detach().cpu()
                    if name not in server_grad_accum:
                        server_grad_accum[name] = grad_cpu
                    else:
                        server_grad_accum[name] += grad_cpu

            total_samples += batch_size

            del data, labels, activation, activation_detached, logits, loss

        divisor = num_batches if num_batches > 0 else 1
        if num_batches > 0:
            for name in client_grad_accum:
                client_grad_accum[name] /= divisor
            for name in server_grad_accum:
                server_grad_accum[name] /= divisor

        client_model.zero_grad(set_to_none=True)
        server_model.zero_grad(set_to_none=True)

        restore_bn_stats(client_model, client_bn_backup)
        restore_bn_stats(server_model, server_bn_backup)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return client_grad_accum, server_grad_accum


class GMeasurementSystem:
    def __init__(
        self,
        full_dataloader: DataLoader,
        device: str = "cpu",
        diagnostic_frequency: int = 10,
        use_variance_g: bool = False,
    ):
        self.full_dataloader = full_dataloader
        self.device = device
        self.diagnostic_frequency = diagnostic_frequency
        self.use_variance_g = use_variance_g

        self.oracle_calculator = OracleCalculator(full_dataloader, device)
        self.oracle_client_grad: Optional[Dict[str, torch.Tensor]] = None
        self.oracle_server_grad: Optional[Dict[str, torch.Tensor]] = None

        self.measurements: List[RoundGResult] = []

    def is_diagnostic_round(self, round_idx: int) -> bool:
        return (round_idx + 1) % self.diagnostic_frequency == 0

    def compute_oracle(self, client_model: nn.Module, server_model: nn.Module):
        print(f"[G Measurement] Computing Oracle gradient...")
        self.oracle_client_grad, self.oracle_server_grad = (
            self.oracle_calculator.compute_oracle_gradient(client_model, server_model)
        )
        print(
            f"[G Measurement] Oracle computed: "
            f"client={len(self.oracle_client_grad)} params, "
            f"server={len(self.oracle_server_grad)} params"
        )

    def measure_client_g(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        x_list: List[torch.Tensor],
        y_list: List[torch.Tensor],
        client_ids: List[int],
        client_weights: Optional[List[int]] = None,
    ) -> Tuple[GMetrics, Dict[int, GMetrics], float, float]:
        """
        Measure Client G using ONLY THE FIRST BATCH for each client.
        This aligns with sfl_framework-fork's behavior (1-step measurement).
        """
        if self.oracle_client_grad is None:
            return GMetrics(), {}, float("nan"), float("nan")

        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)

        client_bn_backup = backup_bn_stats(client_model)
        server_bn_backup = backup_bn_stats(server_model)

        client_model.train()
        server_model.train()

        per_client_g: Dict[int, GMetrics] = {}
        per_client_vecs: Dict[int, torch.Tensor] = {}
        all_g_values: List[float] = []
        all_g_rel_values: List[float] = []
        all_d_cosine_values: List[float] = []

        if client_weights is None:
            client_weights = [1 for _ in client_ids]
        client_weight_map: Dict[int, float] = {}

        oracle_keys = sorted(self.oracle_client_grad.keys())

        # Track which clients have been measured
        measured_clients = set()

        for idx, (x, y, client_id) in enumerate(zip(x_list, y_list, client_ids)):
            # Skip if this client has already been measured (1-step only)
            if client_id in measured_clients:
                continue

            measured_clients.add(client_id)
            if idx < len(client_weights):
                client_weight_map[client_id] = float(client_weights[idx])
            else:
                client_weight_map[client_id] = 1.0

            client_model.zero_grad(set_to_none=True)
            server_model.zero_grad(set_to_none=True)

            x = x.to(self.device)
            y = y.to(self.device)

            activation = client_model(x)
            if isinstance(activation, tuple):
                activation = activation[0]

            activation_detached = activation.detach().requires_grad_(True)
            logits = server_model(activation_detached)

            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()

            activation.backward(activation_detached.grad)

            current_client_grad = {
                name: param.grad.clone().detach().cpu()
                for name, param in client_model.named_parameters()
                if param.grad is not None
            }

            metrics = compute_g_metrics(current_client_grad, self.oracle_client_grad)
            per_client_g[client_id] = metrics
            all_g_values.append(metrics.G)
            all_g_rel_values.append(metrics.G_rel)
            all_d_cosine_values.append(metrics.D_cosine)
            per_client_vecs[client_id] = gradient_to_vector(
                current_client_grad, oracle_keys
            )
            del x, y, activation, activation_detached, logits, loss, current_client_grad

        restore_bn_stats(client_model, client_bn_backup)
        restore_bn_stats(server_model, server_bn_backup)

        if all_g_values:
            avg_g = sum(all_g_values) / len(all_g_values)
            avg_g_rel = sum(all_g_rel_values) / len(all_g_rel_values)
            avg_d_cosine = sum(all_d_cosine_values) / len(all_d_cosine_values)
            avg_metrics = GMetrics(G=avg_g, G_rel=avg_g_rel, D_cosine=avg_d_cosine)
        else:
            avg_metrics = GMetrics()

        variance_client_g = float("nan")
        variance_client_g_rel = float("nan")
        if self.use_variance_g and per_client_vecs:
            total_weight = sum(client_weight_map.values())
            if total_weight > 0:
                oracle_vec = gradient_to_vector(self.oracle_client_grad, oracle_keys)
                Vc = 0.0
                for client_id, vec in per_client_vecs.items():
                    weight = client_weight_map.get(client_id, 1.0) / total_weight
                    diff = vec - oracle_vec
                    Vc += weight * torch.dot(diff, diff).item()
                variance_client_g = Vc
                oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
                variance_client_g_rel = (
                    Vc / oracle_norm_sq if oracle_norm_sq > 0 else float("nan")
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_metrics, per_client_g, variance_client_g, variance_client_g_rel

    def measure_server_g(
        self,
        server_model: nn.Module,
        f_list: List[torch.Tensor],
        y_list: List[torch.Tensor],
        server_weights: Optional[List[int]] = None,
    ) -> Tuple[GMetrics, Dict[int, GMetrics], float, float]:
        """
        Measure Server G using each collected batch.
        This aligns with sfl_framework-fork's behavior (1-step measurement).
        """
        if self.oracle_server_grad is None or not f_list:
            return GMetrics(), {}, float("nan"), float("nan")

        server_model = server_model.to(self.device)
        server_bn_backup = backup_bn_stats(server_model)

        server_model.train()

        if server_weights is None:
            server_weights = [1 for _ in f_list]

        oracle_keys = sorted(self.oracle_server_grad.keys())
        per_batch_metrics: List[GMetrics] = []
        per_batch_vecs: List[torch.Tensor] = []
        per_batch_weights: List[float] = []

        for idx, (f_batch, y_batch) in enumerate(zip(f_list, y_list)):
            server_model.zero_grad(set_to_none=True)

            f_batch = f_batch.to(self.device).detach().requires_grad_(True)
            y_batch = y_batch.to(self.device)

            logits = server_model(f_batch)
            loss = F.cross_entropy(logits, y_batch, reduction="mean")
            loss.backward()

            current_server_grad = {
                name: param.grad.clone().detach().cpu()
                for name, param in server_model.named_parameters()
                if param.grad is not None
            }

            metrics = compute_g_metrics(current_server_grad, self.oracle_server_grad)
            per_batch_metrics.append(metrics)
            per_batch_vecs.append(gradient_to_vector(current_server_grad, oracle_keys))
            per_batch_weights.append(
                float(server_weights[idx]) if idx < len(server_weights) else 1.0
            )

            del f_batch, y_batch, logits, loss, current_server_grad

        per_branch_metrics = {
            idx: metrics for idx, metrics in enumerate(per_batch_metrics)
        }

        if per_batch_metrics:
            avg_g = sum(m.G for m in per_batch_metrics) / len(per_batch_metrics)
            avg_g_rel = sum(m.G_rel for m in per_batch_metrics) / len(per_batch_metrics)
            avg_d_cosine = sum(m.D_cosine for m in per_batch_metrics) / len(
                per_batch_metrics
            )
            result_metrics = GMetrics(G=avg_g, G_rel=avg_g_rel, D_cosine=avg_d_cosine)
        else:
            result_metrics = GMetrics()

        variance_server_g = float("nan")
        variance_server_g_rel = float("nan")
        if self.use_variance_g and per_batch_vecs:
            total_weight = sum(per_batch_weights)
            if total_weight > 0:
                oracle_vec = gradient_to_vector(self.oracle_server_grad, oracle_keys)
                Vs = 0.0
                for vec, weight in zip(per_batch_vecs, per_batch_weights):
                    w = weight / total_weight
                    diff = vec - oracle_vec
                    Vs += w * torch.dot(diff, diff).item()
                variance_server_g = Vs
                oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
                variance_server_g_rel = (
                    Vs / oracle_norm_sq if oracle_norm_sq > 0 else float("nan")
                )

        server_model.zero_grad(set_to_none=True)
        restore_bn_stats(server_model, server_bn_backup)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            result_metrics,
            per_branch_metrics,
            variance_server_g,
            variance_server_g_rel,
        )

    def measure_round(
        self,
        round_idx: int,
        client_model: nn.Module,
        server_model: nn.Module,
        client_ids: List[int],
        x_all: List[torch.Tensor],
        y_all_client: List[torch.Tensor],
        f_all: List[torch.Tensor],
        y_all_server: List[torch.Tensor],
        client_weights: Optional[List[int]] = None,
        server_weights: Optional[List[int]] = None,
    ) -> RoundGResult:
        """
        Full G measurement for a diagnostic round.
        Models passed here should be pre-update (start of round).
        Data passed here should be exactly what was used in the first training step.
        """
        # Oracle is already computed (called separately before training loop)
        # self.compute_oracle(client_model, server_model)

        client_g, per_client_g, variance_client_g, variance_client_g_rel = (
            self.measure_client_g(
                client_model,
                server_model,
                x_all,
                y_all_client,
                client_ids,
                client_weights=client_weights,
            )
        )

        (
            server_g,
            per_branch_server_g,
            variance_server_g,
            variance_server_g_rel,
        ) = self.measure_server_g(
            server_model, f_all, y_all_server, server_weights=server_weights
        )

        result = RoundGResult(
            round_idx=round_idx,
            client_g=client_g,
            server_g=server_g,
            per_client_g=per_client_g,
            per_branch_server_g=per_branch_server_g,
            variance_client_g=variance_client_g,
            variance_client_g_rel=variance_client_g_rel,
            variance_server_g=variance_server_g,
            variance_server_g_rel=variance_server_g_rel,
        )

        self.measurements.append(result)

        self.oracle_client_grad = None
        self.oracle_server_grad = None

        print(
            f"[G Measurement] Round {round_idx + 1}: "
            f"Client G={client_g.G:.6f} (G_rel={client_g.G_rel:.4f}), "
            f"Server G={server_g.G:.6f} (G_rel={server_g.G_rel:.4f})"
        )
        if per_branch_server_g:
            for branch_idx in sorted(per_branch_server_g.keys()):
                branch_metrics = per_branch_server_g[branch_idx]
                print(
                    f"[G Measurement] Round {round_idx + 1} Branch {branch_idx}: "
                    f"G={branch_metrics.G:.6f} (G_rel={branch_metrics.G_rel:.4f})"
                )
        if self.use_variance_g:
            print(
                f"[G Measurement] Variance Client G={variance_client_g:.6f} "
                f"(G_rel={variance_client_g_rel:.6f}), "
                f"Variance Server G={variance_server_g:.6f} "
                f"(G_rel={variance_server_g_rel:.6f})"
            )

        return result

    def clear(self):
        self.oracle_client_grad = None
        self.oracle_server_grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_all_measurements(self) -> List[dict]:
        return [m.to_dict() for m in self.measurements]

    def get_summary(self) -> Dict[str, List[float]]:
        return {
            "rounds": [m.round_idx for m in self.measurements],
            "client_g": [m.client_g.G for m in self.measurements],
            "client_g_rel": [m.client_g.G_rel for m in self.measurements],
            "server_g": [m.server_g.G for m in self.measurements],
            "server_g_rel": [m.server_g.G_rel for m in self.measurements],
            "variance_client_g": [m.variance_client_g for m in self.measurements],
            "variance_client_g_rel": [
                m.variance_client_g_rel for m in self.measurements
            ],
            "variance_server_g": [m.variance_server_g for m in self.measurements],
            "variance_server_g_rel": [
                m.variance_server_g_rel for m in self.measurements
            ],
        }
