"""
G (Gradient Distance to Oracle) Measurement Utility - V2

핵심 프로토콜 (진단 라운드):
1. pre_round 시작 (θ_ref 확정)
2. Oracle 계산 (train 모드 + BN stats 백업/복구)
3. 측정용 1-step (optimizer.step() 없음)
4. G 계산
5. 실제 학습 (in_round)

4가지 필수 보장:
1. _snapshot_model(): deep snapshot (파라미터 + BN 버퍼)
2. measurement step: 동기화된 모델 배포 이후 수행
3. _split_gradient(): 실제 client/server 모델의 param name set으로 분리
4. flatten 시 동일 parameter 순서 보장 (sorted by name)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from server.modules.trainer.propagator.propagator import get_propagator


@dataclass
class GMetrics:
    """G 측정 지표"""

    G: float = 0.0  # L2 norm: ||g̃ - g*||
    G_rel: float = 0.0  # 상대 오차: G / (||g*|| + ε)
    D_cosine: float = 0.0  # 코사인 거리: 1 - cos(g̃, g*)

    def to_dict(self) -> dict:
        return {"G": self.G, "G_rel": self.G_rel, "D_cosine": self.D_cosine}


@dataclass
class RoundGMeasurement:
    """라운드별 G 측정 결과"""

    round_number: int
    is_diagnostic: bool
    server: GMetrics = field(default_factory=GMetrics)
    split_layer: GMetrics = field(default_factory=GMetrics)  # Split layer G
    clients: Dict[int, GMetrics] = field(default_factory=dict)
    client_G_mean: float = 0.0
    client_G_max: float = 0.0
    client_D_mean: float = 0.0
    variance_client_g: float = float("nan")
    variance_client_g_rel: float = float("nan")
    variance_server_g: float = float("nan")
    variance_server_g_rel: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "is_diagnostic": self.is_diagnostic,
            "server": self.server.to_dict(),
            "split_layer": self.split_layer.to_dict(),
            "clients": {str(k): v.to_dict() for k, v in self.clients.items()},
            "client_summary": {
                "G_mean": self.client_G_mean,
                "G_max": self.client_G_max,
                "D_mean": self.client_D_mean,
            },
            "variance_client_g": self.variance_client_g,
            "variance_client_g_rel": self.variance_client_g_rel,
            "variance_server_g": self.variance_server_g,
            "variance_server_g_rel": self.variance_server_g_rel,
        }


# ============================================================
# 요구사항 1: Deep Snapshot (파라미터 + BN 버퍼)
# ============================================================


def snapshot_model(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    모델의 complete state 스냅샷 (파라미터 + BN running stats)

    Returns:
        Dict containing all parameters and buffers
    """
    snapshot = {}

    # Parameters
    for name, param in model.named_parameters():
        snapshot[f"param.{name}"] = param.data.clone().detach()

    # Buffers (BN running_mean, running_var, etc.)
    for name, buffer in model.named_buffers():
        if buffer is not None:
            snapshot[f"buffer.{name}"] = buffer.clone().detach()

    return snapshot


def restore_model(model: nn.Module, snapshot: Dict[str, torch.Tensor]):
    """
    스냅샷에서 모델 상태 복구
    """
    # Restore parameters
    for name, param in model.named_parameters():
        key = f"param.{name}"
        if key in snapshot:
            param.data.copy_(snapshot[key])

    # Restore buffers
    for name, buffer in model.named_buffers():
        key = f"buffer.{name}"
        if key in snapshot and buffer is not None:
            buffer.copy_(snapshot[key])


def backup_bn_stats(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    BN running stats만 백업 (oracle 계산 시 사용)
    """
    backup = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            backup[name] = {
                "running_mean": module.running_mean.clone()
                if module.running_mean is not None
                else None,
                "running_var": module.running_var.clone()
                if module.running_var is not None
                else None,
                "num_batches_tracked": module.num_batches_tracked.clone()
                if module.num_batches_tracked is not None
                else None,
            }
    return backup


def restore_bn_stats(model: nn.Module, backup: Dict[str, Dict[str, torch.Tensor]]):
    """
    BN running stats 복구
    """
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


# ============================================================
# 요구사항 3: Split Gradient by Param Names
# ============================================================


def split_gradient_by_names(
    full_gradient: Dict[str, torch.Tensor],
    client_param_names: Set[str],
    server_param_names: Set[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Full gradient를 client/server param name set 기준으로 분리

    Args:
        full_gradient: 전체 모델 gradient
        client_param_names: 클라이언트 모델의 parameter names
        server_param_names: 서버 모델의 parameter names

    Returns:
        (client_grad, server_grad) 둘 다 Dict[param_name, tensor]
    """
    client_grad = {}
    server_grad = {}

    for name, grad in full_gradient.items():
        if name in client_param_names:
            client_grad[name] = grad
        elif name in server_param_names:
            server_grad[name] = grad
        # else: 양쪽에 없는 파라미터는 무시

    return client_grad, server_grad


def get_param_names(model: nn.Module) -> Set[str]:
    """모델의 모든 parameter names 추출"""
    return set(name for name, _ in model.named_parameters())


# ============================================================
# 요구사항 4: Consistent Parameter Ordering (sorted by name)
# ============================================================


def normalize_grad_keys(grad_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("-", "."): v for k, v in grad_dict.items()}


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
    g_star: Dict[str, torch.Tensor],
    epsilon: float = 1e-8,
) -> GMetrics:
    """
    G 측정 지표 계산

    CRITICAL: 동일한 parameter name 순서로 flatten (sorted)
    """
    if not g_tilde or not g_star:
        print(
            f"[DEBUG] compute_g_metrics: Empty input - g_tilde={len(g_tilde) if g_tilde else 0}, g_star={len(g_star) if g_star else 0}"
        )
        return GMetrics()

    # Key normalization helper
    g_tilde_norm = normalize_grad_keys(g_tilde)
    g_star_norm = normalize_grad_keys(g_star)

    tilde_keys = set(g_tilde_norm.keys())
    star_keys = set(g_star_norm.keys())
    if tilde_keys != star_keys:
        missing = sorted(star_keys - tilde_keys)
        extra = sorted(tilde_keys - star_keys)
        raise ValueError(
            "[G Measurement] Param key mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}, "
            f"missing_count={len(missing)}, extra_count={len(extra)}"
        )

    common_keys = sorted(tilde_keys)

    if not common_keys:
        print(f"[DEBUG] compute_g_metrics: No common keys!")
        print(f"[DEBUG]   g_tilde keys: {list(g_tilde_norm.keys())[:5]}...")
        print(f"[DEBUG]   g_star keys: {list(g_star_norm.keys())[:5]}...")
        return GMetrics()

    # 동일한 순서로 flatten
    g_tilde_flat = gradient_to_vector(g_tilde_norm, common_keys)
    g_star_flat = gradient_to_vector(g_star_norm, common_keys)

    # Check for NaN/Inf
    if torch.isnan(g_tilde_flat).any() or torch.isinf(g_tilde_flat).any():
        print(f"[DEBUG] compute_g_metrics: g_tilde contains NaN/Inf!")
        return GMetrics(G=float("nan"), G_rel=float("nan"), D_cosine=1.0)
    if torch.isnan(g_star_flat).any() or torch.isinf(g_star_flat).any():
        print(f"[DEBUG] compute_g_metrics: g_star contains NaN/Inf!")
        return GMetrics(G=float("nan"), G_rel=float("nan"), D_cosine=1.0)

    diff = g_tilde_flat - g_star_flat
    G = torch.dot(diff, diff).item()

    g_star_norm_val = torch.norm(g_star_flat).item()
    g_star_norm_sq = torch.dot(g_star_flat, g_star_flat).item()

    G_rel = G / (g_star_norm_sq + epsilon)

    # D_cosine = 1 - cos(g̃, g*)
    g_tilde_norm_val = torch.norm(g_tilde_flat).item()

    if g_tilde_norm_val > epsilon and g_star_norm_val > epsilon:
        dot_product = torch.dot(g_tilde_flat, g_star_flat).item()
        cos_sim = dot_product / (g_tilde_norm_val * g_star_norm_val)
        D_cosine = 1.0 - cos_sim
    else:
        D_cosine = 1.0

    D_cosine = max(0.0, min(2.0, D_cosine))

    return GMetrics(G=G, G_rel=G_rel, D_cosine=D_cosine)


# ============================================================
# Oracle Calculator (train 모드 + BN backup)
# ============================================================


class OracleGradientCalculator:
    """
    Oracle gradient 계산기

    특징:
    - train 모드로 계산 (학습과 동일한 조건)
    - BN running stats 백업/복구
    - reduction='mean' + 배치 수로 나눔 (GAS 정합)
    """

    def __init__(self, full_dataloader: DataLoader, device: str = "cuda"):
        self.full_dataloader = full_dataloader
        self.device = device

    def compute_oracle_gradient(
        self, model: nn.Module, return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        전체 데이터셋에서 oracle gradient 계산

        특징:
        1. train 모드 유지 (학습과 동일한 gradient 계산)
        2. BN stats 백업/복구 (θ_ref 보존)
        3. reduction='sum' + 전체 샘플 수로 나눔

        Returns:
            {param_name: gradient_tensor}
        """
        model = model.to(self.device)

        bn_backup = backup_bn_stats(model)
        model.train()

        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        grad_accum = {}

        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue

            model.zero_grad(set_to_none=True)

            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            total_samples += batch_size
            num_batches += 1

            outputs = model(data)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        grad_accum[name] += param.grad.clone().detach().cpu()

            total_loss += loss.item()
            del outputs, loss, data, labels

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        divisor = total_samples if total_samples > 0 else 1
        oracle_grad = {name: grad / divisor for name, grad in grad_accum.items()}

        model.zero_grad(set_to_none=True)
        restore_bn_stats(model, bn_backup)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(
            f"[Oracle] Computed from {total_samples} samples over {num_batches} batches, avg_loss={avg_loss:.4f}"
        )

        if return_loss:
            return oracle_grad, avg_loss
        return oracle_grad

    def compute_oracle_with_split_hook(
        self,
        full_model: nn.Module,
        client_param_names: Set[str],
        split_layer_name: str = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ]:
        full_model = full_model.to(self.device)

        split_layer = None
        if split_layer_name is not None:
            for name, module in full_model.named_modules():
                if name == split_layer_name:
                    split_layer = module
                    break

        split_activation_storage = {"activations": []}
        hook_handle = None

        def _clone_output(tensor: torch.Tensor) -> torch.Tensor:
            out_clone = tensor.clone()
            out_clone.requires_grad_(True)
            out_clone.retain_grad()
            return out_clone

        def forward_hook(_module, _input, output):
            if isinstance(output, tuple):
                cloned = []
                for item in output:
                    if isinstance(item, torch.Tensor):
                        cloned.append(_clone_output(item))
                    else:
                        cloned.append(item)
                split_activation_storage["activations"].append(tuple(cloned))
                return tuple(cloned)
            output_clone = _clone_output(output)
            split_activation_storage["activations"].append(output_clone)
            return output_clone

        if split_layer is not None:
            hook_handle = split_layer.register_forward_hook(forward_hook)
            print(
                f"[Oracle Split Hook] Registered forward hook on '{split_layer_name}'"
            )

        bn_backup = backup_bn_stats(full_model)

        original_inplace_states = {}
        for name, module in full_model.named_modules():
            if hasattr(module, "inplace"):
                original_inplace_states[name] = module.inplace
                module.inplace = False

        full_model.train()

        total_samples = 0
        num_batches = 0
        grad_accum = {}

        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue

            full_model.zero_grad(set_to_none=True)

            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            total_samples += batch_size
            num_batches += 1

            outputs = full_model(data)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            loss.backward()

            for name, param in full_model.named_parameters():
                if param.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        grad_accum[name] += param.grad.clone().detach().cpu()

            del outputs, loss, data, labels

        if hook_handle is not None:
            hook_handle.remove()

        divisor = total_samples if total_samples > 0 else 1
        oracle_grad = {name: grad / divisor for name, grad in grad_accum.items()}

        full_model.zero_grad(set_to_none=True)

        for name, module in full_model.named_modules():
            if name in original_inplace_states:
                module.inplace = original_inplace_states[name]

        restore_bn_stats(full_model, bn_backup)

        oracle_client_grad = {}
        oracle_server_grad = {}
        client_param_names_dots = {n.replace("-", ".") for n in client_param_names}

        for name, grad in oracle_grad.items():
            name_dot = name.replace("-", ".")
            if name_dot in client_param_names_dots:
                oracle_client_grad[name] = grad
            else:
                oracle_server_grad[name] = grad

        oracle_split_grad = None
        if split_activation_storage["activations"]:
            split_grad_sum: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
            for activation in split_activation_storage["activations"]:
                if isinstance(activation, tuple):
                    grads = []
                    missing_grad = False
                    for item in activation:
                        if not isinstance(item, torch.Tensor) or item.grad is None:
                            missing_grad = True
                            break
                        grads.append(item.grad.detach().sum(dim=0).cpu())
                    if missing_grad:
                        continue
                    if split_grad_sum is None:
                        split_grad_sum = grads
                    else:
                        for idx, grad_item in enumerate(grads):
                            split_grad_sum[idx] = split_grad_sum[idx] + grad_item
                else:
                    if activation.grad is not None:
                        batch_grad_sum = activation.grad.detach().sum(dim=0)
                        if split_grad_sum is None:
                            split_grad_sum = batch_grad_sum.cpu()
                        else:
                            split_grad_sum = split_grad_sum + batch_grad_sum.cpu()

            if split_grad_sum is not None:
                if isinstance(split_grad_sum, list):
                    oracle_split_grad = tuple(g / divisor for g in split_grad_sum)
                    split_shapes = [g.shape for g in oracle_split_grad]
                else:
                    oracle_split_grad = split_grad_sum / divisor
                    split_shapes = [oracle_split_grad.shape]
                print(f"[Oracle Split Hook] Split layer grad shape: {split_shapes}")

        print(
            f"[Oracle Split Hook] Split: client={len(oracle_client_grad)}, "
            f"server={len(oracle_server_grad)}, "
            f"split_layer={'yes' if oracle_split_grad is not None else 'no'}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return oracle_client_grad, oracle_server_grad, oracle_split_grad

    def compute_oracle_split_gradient(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        config,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ]:
        """
        Split 모델로 Oracle 계산 + Split Layer Gradient 수집

        Returns:
            (oracle_client_grad, oracle_server_grad, oracle_split_grad)

        Note: Split grad는 batch dimension에서 평균화된 형태 [C, H, W]
        """
        client_model = client_model.to(self.device)
        server_model = server_model.to(self.device)

        # BN stats 백업
        client_bn_backup = backup_bn_stats(client_model)
        server_bn_backup = backup_bn_stats(server_model)

        client_model.train()
        server_model.train()

        total_samples = 0
        num_batches = 0
        split_grad_sum = None
        client_grad_accum = {}
        server_grad_accum = {}

        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue

            client_model.zero_grad(set_to_none=True)
            server_model.zero_grad(set_to_none=True)

            data = data.to(self.device)
            labels = labels.to(self.device)
            batch_size = data.size(0)
            num_batches += 1

            attention_mask = batch[2] if len(batch) >= 3 else None
            params = {
                "attention_mask": attention_mask.to(self.device)
                if attention_mask is not None
                else None
            }

            client_propagator = get_propagator(config, client_model)
            server_propagator = get_propagator(config, server_model)

            client_propagator.forward(data, params)
            activation = client_propagator.outputs
            act_detached = None
            id_detached = None

            if isinstance(activation, tuple):
                act, identity = activation
                act_detached = act.detach().requires_grad_(True)
                id_detached = identity.detach().requires_grad_(True)
                logits = server_propagator.forward((act_detached, id_detached), params)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                loss.backward()

                if act_detached.grad is None or id_detached.grad is None:
                    raise RuntimeError("Missing gradients for tuple split output")

                torch.autograd.backward(
                    [act, identity], [act_detached.grad, id_detached.grad]
                )
            else:
                activation.requires_grad_(True)
                activation.retain_grad()  # Activation gradient 유지

                logits = server_propagator.forward(activation, params)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                loss.backward()

            # Accumulate client gradients
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    if name not in client_grad_accum:
                        client_grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        client_grad_accum[name] += param.grad.clone().detach().cpu()

            # Accumulate server gradients
            for name, param in server_model.named_parameters():
                if param.grad is not None:
                    if name not in server_grad_accum:
                        server_grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        server_grad_accum[name] += param.grad.clone().detach().cpu()

            if isinstance(activation, tuple):
                if act_detached is None or id_detached is None:
                    raise RuntimeError("Missing tuple activations for split grad")
                assert act_detached is not None and id_detached is not None
                if act_detached.grad is not None and id_detached.grad is not None:
                    batch_split_grad = (
                        act_detached.grad.detach().sum(dim=0).cpu(),
                        id_detached.grad.detach().sum(dim=0).cpu(),
                    )
                    if split_grad_sum is None:
                        split_grad_sum = [batch_split_grad[0], batch_split_grad[1]]
                    else:
                        split_grad_sum[0] = split_grad_sum[0] + batch_split_grad[0]
                        split_grad_sum[1] = split_grad_sum[1] + batch_split_grad[1]
            else:
                if activation.grad is not None:
                    batch_split_grad = activation.grad.detach().sum(dim=0)
                    if split_grad_sum is None:
                        split_grad_sum = batch_split_grad.cpu()
                    else:
                        split_grad_sum = split_grad_sum + batch_split_grad.cpu()

            total_samples += batch_size

            # Clear intermediate tensors
            del activation, logits, loss, data, labels

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        divisor = total_samples if total_samples > 0 else 1

        oracle_client_grad = {}
        for name, grad in client_grad_accum.items():
            oracle_client_grad[name] = grad / divisor

        oracle_server_grad = {}
        for name, grad in server_grad_accum.items():
            oracle_server_grad[name] = grad / divisor

        if split_grad_sum is None:
            oracle_split_grad = None
        elif isinstance(split_grad_sum, list):
            oracle_split_grad = tuple(g / divisor for g in split_grad_sum)
        else:
            oracle_split_grad = split_grad_sum / divisor

        client_model.zero_grad(set_to_none=True)
        server_model.zero_grad(set_to_none=True)
        restore_bn_stats(client_model, client_bn_backup)
        restore_bn_stats(server_model, server_bn_backup)

        print(f"[Oracle Split] Computed from {total_samples} samples")
        if oracle_split_grad is not None:
            if isinstance(oracle_split_grad, tuple):
                split_shapes = [g.shape for g in oracle_split_grad]
            else:
                split_shapes = [oracle_split_grad.shape]
            print(f"[Oracle Split] Split grad shape: {split_shapes}")

        return oracle_client_grad, oracle_server_grad, oracle_split_grad


# ============================================================
# Gradient Accumulator (가중 평균용)
# ============================================================


class GradientAccumulator:
    """배치별 gradient 누적 (가중 평균 계산용)"""

    def __init__(self):
        self.gradient_sum: Dict[str, torch.Tensor] = {}
        self.total_weight: float = 0.0

    def reset(self):
        self.gradient_sum = {}
        self.total_weight = 0.0

    def add(self, gradient: Dict[str, torch.Tensor], weight: float = 1.0):
        """Gradient 누적 (weight = batch size)"""
        for name, grad in gradient.items():
            grad_cpu = (
                grad.clone().detach().cpu().float()
                if grad.is_cuda
                else grad.clone().detach().float()
            )
            weighted = grad_cpu * weight

            if name not in self.gradient_sum:
                self.gradient_sum[name] = weighted
            else:
                self.gradient_sum[name] += weighted

        self.total_weight += weight

    def get_weighted_average(self) -> Dict[str, torch.Tensor]:
        """가중 평균 gradient 반환"""
        if self.total_weight == 0:
            return {}

        return {
            name: grad / self.total_weight for name, grad in self.gradient_sum.items()
        }


# ============================================================
# 통합 G 측정 시스템
# ============================================================


class GMeasurementSystem:
    """
    Oracle 기반 G 측정 통합 시스템

    사용 흐름:
    1. initialize(full_dataloader) - 초기화
    2. on_diagnostic_round_start(model, client_names, server_names) - oracle 계산
    3. on_measurement_step(server_grad, client_grads) - 1-step grad 저장
    4. compute_g() - G 계산

    Measurement Modes:
    - "single": 기존 1-step 방식 (첫 번째 배치만 측정)
    - "k_batch": 첫 K개 배치 gradient 누적 후 평균
    - "accumulated": 전체 라운드 gradient 누적 후 평균 (Oracle과 동일한 정규화)
    """

    def __init__(
        self,
        diagnostic_frequency: int = 10,
        device: str = "cuda",
        use_variance_g: bool = False,
        measurement_mode: str = "single",  # "single" | "k_batch" | "accumulated"
        measurement_k: int = 5,  # Number of batches for k_batch mode
    ):
        self.diagnostic_frequency = diagnostic_frequency
        self.device = device
        self.use_variance_g = use_variance_g
        self.measurement_mode = measurement_mode
        self.measurement_k = measurement_k

        self.oracle_calculator: Optional[OracleGradientCalculator] = None

        # Cached oracle gradients (split into client/server)
        self.oracle_client_grad: Optional[Dict[str, torch.Tensor]] = None
        self.oracle_server_grad: Optional[Dict[str, torch.Tensor]] = None

        # Param name sets for split
        self.client_param_names: Set[str] = set()
        self.server_param_names: Set[str] = set()

        # 1-step measurement results (single mode)
        self.server_g_tildes: List[Dict[str, torch.Tensor]] = []
        self.server_weights: List[float] = []
        self.client_g_tildes: Dict[int, Dict[str, torch.Tensor]] = {}

        # Accumulated/K-batch mode: gradient accumulators
        self._server_accumulator: GradientAccumulator = GradientAccumulator()
        self._client_accumulators: Dict[int, GradientAccumulator] = {}
        self._accumulated_round_active: bool = False
        self._server_batch_count: int = 0  # For k_batch mode

        # Split layer gradient (activation gradient at split point)
        self.oracle_split_grad: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        ] = None
        self.split_g_tilde: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = (
            None
        )
        self.split_g_tildes: Dict[int, torch.Tensor] = {}  # Per-client split gradients

        # Results
        self.measurements: List[RoundGMeasurement] = []

        if measurement_mode == "accumulated":
            print("[G Measurement] Mode: ACCUMULATED (full round average)")
        elif measurement_mode == "k_batch":
            print(f"[G Measurement] Mode: K_BATCH (first {measurement_k} batches)")
        else:
            print("[G Measurement] Mode: SINGLE (1-step)")

    def is_diagnostic_round(self, round_number: int) -> bool:
        return (round_number + 1) % self.diagnostic_frequency == 0

    # ============================================================
    # Accumulated / K-Batch Mode Methods
    # ============================================================

    def start_accumulated_round(self):
        """
        Accumulated/K-batch 모드에서 라운드 시작 시 호출
        - Accumulator 초기화
        - Batch counter 초기화 (k_batch 모드)
        """
        if self.measurement_mode not in ("accumulated", "k_batch"):
            return

        self._server_accumulator.reset()
        self._client_accumulators = {}
        self._accumulated_round_active = True
        self._server_batch_count = 0
        mode_str = f"K_BATCH (K={self.measurement_k})" if self.measurement_mode == "k_batch" else "ACCUMULATED"
        print(f"[G Measurement] {mode_str} round started - accumulators reset")

    def accumulate_server_gradient(
        self, server_grad: Dict[str, torch.Tensor], batch_size: int
    ) -> bool:
        """
        Accumulated/K-batch 모드에서 매 iteration마다 서버 gradient 누적

        Args:
            server_grad: 서버 모델의 gradient (reduction='mean' 기준)
            batch_size: 현재 배치 크기 (weight로 사용)

        Returns:
            bool: True if gradient was accumulated, False if skipped (k_batch limit reached)

        Note:
            Oracle은 reduction='sum' + total_samples로 나눔
            학습은 reduction='mean'이므로:
            accumulated = Σ(grad_mean * batch_size) / Σ(batch_size)
                        = Σ(grad_sum) / total_samples
            → Oracle과 동일한 정규화
        """
        if self.measurement_mode not in ("accumulated", "k_batch"):
            return False
        if not self._accumulated_round_active:
            return False

        # K-batch mode: check if we've collected enough batches
        if self.measurement_mode == "k_batch":
            if self._server_batch_count >= self.measurement_k:
                return False  # Already collected K batches
            self._server_batch_count += 1

        # grad_mean * batch_size = grad_sum (reduction='sum'과 동일)
        self._server_accumulator.add(server_grad, weight=float(batch_size))
        return True

    def accumulate_client_gradient(
        self, client_id: int, client_grad: Dict[str, torch.Tensor], batch_size: int
    ):
        """
        Accumulated/K-batch 모드에서 매 iteration마다 클라이언트 gradient 누적

        Args:
            client_id: 클라이언트 ID
            client_grad: 클라이언트 모델의 gradient
            batch_size: 현재 배치 크기

        Note: K-batch limit는 클라이언트별로 적용됨
        """
        if self.measurement_mode not in ("accumulated", "k_batch"):
            return
        if not self._accumulated_round_active:
            return

        if client_id not in self._client_accumulators:
            self._client_accumulators[client_id] = GradientAccumulator()

        # K-batch mode: check batch count per client
        accumulator = self._client_accumulators[client_id]
        if self.measurement_mode == "k_batch":
            # Use batch_count stored in accumulator (we need to track this)
            current_count = getattr(accumulator, '_batch_count', 0)
            if current_count >= self.measurement_k:
                return  # Already collected K batches for this client
            accumulator._batch_count = current_count + 1

        accumulator.add(client_grad, weight=float(batch_size))

    def finalize_accumulated_round(self):
        """
        Accumulated/K-batch 모드에서 라운드 끝에 호출
        - 누적된 gradient를 평균 내어 g_tildes로 저장
        """
        if self.measurement_mode not in ("accumulated", "k_batch"):
            return
        if not self._accumulated_round_active:
            return

        mode_str = f"K_BATCH (K={self.measurement_k})" if self.measurement_mode == "k_batch" else "ACCUMULATED"

        # Server gradient average
        server_avg = self._server_accumulator.get_weighted_average()
        if server_avg:
            self.server_g_tildes = [server_avg]
            self.server_weights = [self._server_accumulator.total_weight]
            batch_info = f" ({self._server_batch_count} batches)" if self.measurement_mode == "k_batch" else ""
            print(
                f"[G Measurement] Server {mode_str}: {self._server_accumulator.total_weight:.0f} samples{batch_info}"
            )

        # Client gradient averages
        self.client_g_tildes = {}
        for client_id, accumulator in self._client_accumulators.items():
            client_avg = accumulator.get_weighted_average()
            if client_avg:
                self.client_g_tildes[client_id] = client_avg
                batch_count = getattr(accumulator, '_batch_count', 'N/A')
                batch_info = f" ({batch_count} batches)" if self.measurement_mode == "k_batch" else ""
                print(
                    f"[G Measurement] Client {client_id} {mode_str}: {accumulator.total_weight:.0f} samples{batch_info}"
                )

        self._accumulated_round_active = False
        print(
            f"[G Measurement] {mode_str} round finalized: server + {len(self.client_g_tildes)} clients"
        )

    def initialize(self, full_dataloader: DataLoader):
        """Oracle calculator 초기화"""
        self.oracle_calculator = OracleGradientCalculator(full_dataloader, self.device)
        print(
            f"[G Measurement] Initialized with {len(full_dataloader.dataset)} samples"
        )

    def set_param_names(self, client_model: nn.Module, server_model: nn.Module):
        """
        요구사항 3: 실제 client/server 모델의 param name set 설정
        """
        self.client_param_names = get_param_names(client_model)
        self.server_param_names = get_param_names(server_model)
        print(
            f"[G Measurement] Param names set: client={len(self.client_param_names)}, server={len(self.server_param_names)}"
        )

    def compute_oracle_for_round(self, full_model: nn.Module):
        """
        진단 라운드에서 oracle gradient 계산

        Args:
            full_model: Client + Server 결합된 full model
        """
        if self.oracle_calculator is None:
            print("[G Measurement] Error: Oracle calculator not initialized")
            return

        print("[G Measurement] Computing oracle gradient from full dataset...")

        # Full model에서 oracle 계산
        full_oracle = self.oracle_calculator.compute_oracle_gradient(full_model)

        # Client/Server로 split (요구사항 3)
        self.oracle_client_grad, self.oracle_server_grad = split_gradient_by_names(
            full_oracle, self.client_param_names, self.server_param_names
        )

        print(
            f"[G Measurement] Oracle split: client={len(self.oracle_client_grad)}, server={len(self.oracle_server_grad)}"
        )

    def compute_oracle_split_for_round(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        full_model: nn.Module = None,
        split_layer_name: str = None,
        config=None,
    ):
        """
        Oracle gradient 계산 with split layer gradient

        Note: client_model/server_model are ModuleDict, so we use full_model
        with backward hook to capture split layer gradient

        Args:
            client_model: 클라이언트 측 모델 (param names만 사용)
            server_model: 서버 측 모델 (param names만 사용)
            full_model: Full model (optional, for split layer gradient)
            split_layer_name: Name of the split layer to hook
        """
        if self.oracle_calculator is None:
            print("[G Measurement] Error: Oracle calculator not initialized")
            return

        print("[G Measurement] Computing oracle gradient with split layer...")

        # Get client param names for split layer detection
        client_param_names = set(name for name, _ in client_model.named_parameters())

        if full_model is not None:
            # Use full model oracle calculation with backward hook
            (
                self.oracle_client_grad,
                self.oracle_server_grad,
                self.oracle_split_grad,
            ) = self.oracle_calculator.compute_oracle_with_split_hook(
                full_model, client_param_names, split_layer_name
            )
        else:
            # Fallback: compute regular oracle without split layer gradient
            print(
                "[G Measurement] Warning: full_model not provided, split layer gradient not available"
            )
            full_oracle = self.oracle_calculator.compute_oracle_gradient(
                return_loss=False
            )
            self.oracle_client_grad = {
                n: g for n, g in full_oracle.items() if n in client_param_names
            }
            self.oracle_server_grad = {
                n: g for n, g in full_oracle.items() if n not in client_param_names
            }
            self.oracle_split_grad = None

        def _client_split_output_is_tuple() -> bool:
            if self.oracle_calculator is None or config is None:
                return False
            dataloader = self.oracle_calculator.full_dataloader
            device = self.oracle_calculator.device
            data = None
            attention_mask = None
            for batch in dataloader:
                if len(batch) >= 2:
                    data = batch[0]
                    attention_mask = batch[2] if len(batch) >= 3 else None
                    break
            if data is None:
                return False

            data = data.to(device)
            params = {
                "attention_mask": attention_mask.to(device)
                if attention_mask is not None
                else None
            }

            client_propagator = get_propagator(config, client_model)
            was_training = client_model.training
            client_model.eval()
            with torch.no_grad():
                client_propagator.forward(data, params)
            if was_training:
                client_model.train()
            return isinstance(client_propagator.outputs, tuple)

        if (
            config is not None
            and _client_split_output_is_tuple()
            and not isinstance(self.oracle_split_grad, tuple)
        ):
            (
                _,
                _,
                split_grad,
            ) = self.oracle_calculator.compute_oracle_split_gradient(
                client_model, server_model, config
            )
            if split_grad is not None:
                self.oracle_split_grad = split_grad

        client_oracle_norm = None
        server_oracle_norm = None
        client_numel = None
        server_numel = None

        if self.oracle_client_grad:
            client_oracle_vec = gradient_to_vector(
                normalize_grad_keys(self.oracle_client_grad)
            )
            client_oracle_norm = torch.norm(client_oracle_vec).item()
            client_numel = client_oracle_vec.numel()
        if self.oracle_server_grad:
            server_oracle_vec = gradient_to_vector(
                normalize_grad_keys(self.oracle_server_grad)
            )
            server_oracle_norm = torch.norm(server_oracle_vec).item()
            server_numel = server_oracle_vec.numel()

        split_shape = "none"
        if self.oracle_split_grad is not None:
            if isinstance(self.oracle_split_grad, tuple):
                split_shape = [g.shape for g in self.oracle_split_grad]
            else:
                split_shape = [self.oracle_split_grad.shape]

        if self.oracle_calculator is not None:
            full_loader = self.oracle_calculator.full_dataloader
            total_samples = len(full_loader.dataset)
            num_batches = len(full_loader)
        else:
            total_samples = 0
            num_batches = 0

        print(
            f"[G] Oracle: samples={total_samples}, batches={num_batches}, "
            f"split_layer={split_layer_name or 'none'}, split_shape={split_shape}"
        )
        client_oracle_norm = client_oracle_norm or 0.0
        server_oracle_norm = server_oracle_norm or 0.0
        client_numel = client_numel or 0
        server_numel = server_numel or 0
        print(
            f"[G] Oracle Norms: client={client_oracle_norm:.4f} (numel={client_numel}), "
            f"server={server_oracle_norm:.4f} (numel={server_numel})"
        )

    def store_measurement_gradient(
        self,
        server_grad: Dict[str, torch.Tensor],
        client_grads: Dict[int, Dict[str, torch.Tensor]],
    ):
        """
        측정용 1-step에서 수집된 gradient 저장

        Args:
            server_grad: 서버 모델의 gradient (1-step)
            client_grads: {client_id: gradient} - 각 클라이언트의 gradient (1-step)
        """
        self.server_g_tildes = [
            {
                name: grad.clone().detach().cpu()
                if grad.is_cuda
                else grad.clone().detach()
                for name, grad in server_grad.items()
            }
        ]
        self.server_weights = [1.0]

        self.client_g_tildes = {}
        for client_id, grad in client_grads.items():
            self.client_g_tildes[client_id] = {
                name: g.clone().detach().cpu() if g.is_cuda else g.clone().detach()
                for name, g in grad.items()
            }

        print(
            f"[G Measurement] Stored 1-step gradients: server + {len(self.client_g_tildes)} clients"
        )

    def store_server_gradient(
        self, server_grad: Dict[str, torch.Tensor], weight: float
    ):
        grad_cpu = {
            name: grad.clone().detach().cpu() if grad.is_cuda else grad.clone().detach()
            for name, grad in server_grad.items()
        }
        self.server_g_tildes.append(grad_cpu)
        self.server_weights.append(weight)

    def _compute_split_g_metrics(
        self,
        g_tilde: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        g_oracle: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        epsilon: float = 1e-8,
    ) -> GMetrics:
        """
        Split layer (activation) gradient에 대한 G 계산

        Args:
            g_tilde: 실제 측정된 split gradient [평균화된 형태]
            g_oracle: Oracle split gradient [평균화된 형태]
        """

        def _flatten_split_grad(
            grad: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        ) -> torch.Tensor:
            if isinstance(grad, tuple):
                flats = [g.flatten().float() for g in grad]
                return torch.cat(flats) if flats else torch.tensor([])
            return grad.flatten().float()

        # Flatten to 1D
        g_tilde_flat = _flatten_split_grad(g_tilde)
        g_oracle_flat = _flatten_split_grad(g_oracle)

        G = torch.dot(g_tilde_flat - g_oracle_flat, g_tilde_flat - g_oracle_flat).item()

        oracle_norm = torch.norm(g_oracle_flat).item()
        oracle_norm_sq = torch.dot(g_oracle_flat, g_oracle_flat).item()
        G_rel = G / (oracle_norm_sq + epsilon)

        # D_cosine = 1 - cos(g_tilde, g_oracle)
        tilde_norm = torch.norm(g_tilde_flat).item()
        if tilde_norm > epsilon and oracle_norm > epsilon:
            dot_product = torch.dot(g_tilde_flat, g_oracle_flat).item()
            cos_sim = dot_product / (tilde_norm * oracle_norm)
            D_cosine = 1.0 - cos_sim
        else:
            D_cosine = 1.0

        D_cosine = max(0.0, min(2.0, D_cosine))

        return GMetrics(G=G, G_rel=G_rel, D_cosine=D_cosine)

    def compute_g(
        self, round_number: int, client_weights: Optional[Dict[int, float]] = None
    ) -> RoundGMeasurement:
        """
        G 계산 (요구사항 4: sorted keys로 일관된 순서)
        """
        result = RoundGMeasurement(
            round_number=round_number,
            is_diagnostic=self.is_diagnostic_round(round_number),
        )

        if not result.is_diagnostic:
            return result

        client_sizes = []
        if client_weights:
            client_sizes = [
                int(client_weights[cid]) for cid in sorted(client_weights.keys())
            ]
        server_sizes = (
            [int(w) for w in self.server_weights] if self.server_weights else []
        )
        print(f"[G] Batch Sizes: client={client_sizes}, server={server_sizes}")

        # Server G
        if self.server_g_tildes and self.oracle_server_grad:
            server_metrics = []
            for idx, server_grad in enumerate(self.server_g_tildes):
                metrics = compute_g_metrics(server_grad, self.oracle_server_grad)
                server_metrics.append(metrics)
                print(
                    f"[G] Server {idx}: G={metrics.G:.6f}, G_rel={metrics.G_rel:.4f}, D={metrics.D_cosine:.4f}"
                )

            if server_metrics:
                result.server = GMetrics(
                    G=sum(m.G for m in server_metrics) / len(server_metrics),
                    G_rel=sum(m.G_rel for m in server_metrics) / len(server_metrics),
                    D_cosine=sum(m.D_cosine for m in server_metrics)
                    / len(server_metrics),
                )
                print(
                    f"[G] Server Summary: G={result.server.G:.6f}, G_rel={result.server.G_rel:.4f}, D={result.server.D_cosine:.4f}"
                )

        # Client G
        client_Gs = []
        client_G_rel = []
        client_Ds = []

        for client_id, g_tilde in self.client_g_tildes.items():
            if self.oracle_client_grad:
                metrics = compute_g_metrics(g_tilde, self.oracle_client_grad)
                result.clients[client_id] = metrics
                client_Gs.append(metrics.G)
                client_G_rel.append(metrics.G_rel)
                client_Ds.append(metrics.D_cosine)
                print(
                    f"[G] Client {client_id}: G={metrics.G:.6f}, G_rel={metrics.G_rel:.4f}, D={metrics.D_cosine:.4f}"
                )

        if client_Gs:
            result.client_G_mean = sum(client_Gs) / len(client_Gs)
            result.client_G_max = max(client_Gs)
            result.client_D_mean = sum(client_Ds) / len(client_Ds)
            client_G_rel_mean = sum(client_G_rel) / len(client_G_rel)
            print(
                f"[G] Client Summary: G={result.client_G_mean:.6f}, G_rel={client_G_rel_mean:.4f}, D={result.client_D_mean:.4f}"
            )

        if self.use_variance_g:
            variance_client_g = float("nan")
            variance_client_g_rel = float("nan")
            variance_server_g = float("nan")
            variance_server_g_rel = float("nan")

            if client_weights is None:
                client_weights = {
                    client_id: 1.0 for client_id in self.client_g_tildes.keys()
                }

            if self.oracle_client_grad and self.client_g_tildes:
                oracle_grad_norm = normalize_grad_keys(self.oracle_client_grad)
                oracle_keys = sorted(oracle_grad_norm.keys())
                oracle_vec = gradient_to_vector(oracle_grad_norm, oracle_keys)
                total_weight = sum(
                    client_weights.get(client_id, 1.0)
                    for client_id in self.client_g_tildes.keys()
                )
                if total_weight > 0:
                    Vc = 0.0
                    for client_id, g_tilde in self.client_g_tildes.items():
                        g_tilde_norm = normalize_grad_keys(g_tilde)
                        vec = gradient_to_vector(g_tilde_norm, oracle_keys)
                        weight = client_weights.get(client_id, 1.0) / total_weight
                        diff = vec - oracle_vec
                        Vc += weight * torch.dot(diff, diff).item()
                    variance_client_g = Vc
                    oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
                    variance_client_g_rel = (
                        Vc / oracle_norm_sq if oracle_norm_sq > 0 else float("nan")
                    )

            if self.oracle_server_grad and self.server_g_tildes:
                oracle_grad_norm = normalize_grad_keys(self.oracle_server_grad)
                server_keys = sorted(oracle_grad_norm.keys())
                oracle_vec = gradient_to_vector(oracle_grad_norm, server_keys)
                weights = self.server_weights
                if len(weights) != len(self.server_g_tildes):
                    weights = [1.0] * len(self.server_g_tildes)
                total_weight = sum(weights)
                if total_weight > 0:
                    Vs = 0.0
                    for server_grad, weight in zip(self.server_g_tildes, weights):
                        server_grad_norm = normalize_grad_keys(server_grad)
                        server_vec = gradient_to_vector(server_grad_norm, server_keys)
                        scaled_weight = weight / total_weight
                        diff = server_vec - oracle_vec
                        Vs += scaled_weight * torch.dot(diff, diff).item()
                    variance_server_g = Vs
                    oracle_norm_sq = torch.dot(oracle_vec, oracle_vec).item()
                    variance_server_g_rel = (
                        Vs / oracle_norm_sq if oracle_norm_sq > 0 else float("nan")
                    )

            result.variance_client_g = variance_client_g
            result.variance_client_g_rel = variance_client_g_rel
            result.variance_server_g = variance_server_g
            result.variance_server_g_rel = variance_server_g_rel

            print(
                f"[G] Variance Client: G={variance_client_g:.6f}, G_rel={variance_client_g_rel:.6f}"
            )
            print(
                f"[G] Variance Server: G={variance_server_g:.6f}, G_rel={variance_server_g_rel:.6f}"
            )

        self.measurements.append(result)
        return result

    def clear_round_data(self):
        """라운드 데이터 초기화 (메모리 해제)"""
        self.server_g_tildes = []
        self.server_weights = []
        self.client_g_tildes = {}

        # Clear split layer gradients
        self.split_g_tilde = None
        self.split_g_tildes = {}

        # Also clear oracle gradients to release memory
        self.oracle_client_grad = None
        self.oracle_server_grad = None
        self.oracle_split_grad = None

        # Also clear oracle calculator to release dataloader memory
        # It will be recreated when needed via lazy initialization
        # Note: This means oracle needs to be recomputed each diagnostic round
        # which is intentional since model weights change
        self.oracle_calculator = None

        # Force garbage collection
        import gc

        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_all_measurements(self) -> List[dict]:
        return [m.to_dict() for m in self.measurements]
