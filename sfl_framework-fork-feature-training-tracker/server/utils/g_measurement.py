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
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class GMetrics:
    """G 측정 지표"""
    G: float = 0.0          # L2 norm: ||g̃ - g*||
    G_rel: float = 0.0      # 상대 오차: G / (||g*|| + ε)
    D_cosine: float = 0.0   # 코사인 거리: 1 - cos(g̃, g*)
    
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
            }
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
                'running_mean': module.running_mean.clone() if module.running_mean is not None else None,
                'running_var': module.running_var.clone() if module.running_var is not None else None,
                'num_batches_tracked': module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None
            }
    return backup


def restore_bn_stats(model: nn.Module, backup: Dict[str, Dict[str, torch.Tensor]]):
    """
    BN running stats 복구
    """
    for name, module in model.named_modules():
        if name in backup:
            stats = backup[name]
            if stats['running_mean'] is not None and module.running_mean is not None:
                module.running_mean.copy_(stats['running_mean'])
            if stats['running_var'] is not None and module.running_var is not None:
                module.running_var.copy_(stats['running_var'])
            if stats['num_batches_tracked'] is not None and module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(stats['num_batches_tracked'])


# ============================================================
# 요구사항 3: Split Gradient by Param Names
# ============================================================

def split_gradient_by_names(
    full_gradient: Dict[str, torch.Tensor],
    client_param_names: Set[str],
    server_param_names: Set[str]
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

def gradient_to_vector(
    gradient: Dict[str, torch.Tensor],
    reference_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Gradient dict를 1D vector로 flatten
    
    Args:
        gradient: {name: tensor} dict
        reference_names: 순서 기준 (None이면 sorted keys 사용)
        
    Returns:
        Flattened 1D tensor
        
    CRITICAL: reference_names가 주어지면 그 순서대로,
              아니면 sorted(keys)로 일관된 순서 보장
    """
    if not gradient:
        return torch.zeros(1)
    
    if reference_names is None:
        reference_names = sorted(gradient.keys())
    
    vectors = []
    for name in reference_names:
        if name in gradient:
            vectors.append(gradient[name].flatten().float())
    
    if not vectors:
        return torch.zeros(1)
    
    return torch.cat(vectors)


def compute_g_metrics(
    g_tilde: Dict[str, torch.Tensor],
    g_star: Dict[str, torch.Tensor],
    epsilon: float = 1e-8
) -> GMetrics:
    """
    G 측정 지표 계산
    
    CRITICAL: 동일한 parameter name 순서로 flatten (sorted)
    """
    if not g_tilde or not g_star:
        print(f"[DEBUG] compute_g_metrics: Empty input - g_tilde={len(g_tilde) if g_tilde else 0}, g_star={len(g_star) if g_star else 0}")
        return GMetrics()
    
    # Key normalization helper
    def normalize_keys(d):
        return {k.replace("-", "."): v for k, v in d.items()}

    g_tilde_norm = normalize_keys(g_tilde)
    g_star_norm = normalize_keys(g_star)

    # 공통 키만 사용 (순서 보장을 위해 sorted)
    common_keys = sorted(set(g_tilde_norm.keys()) & set(g_star_norm.keys()))
    
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
        return GMetrics(G=float('nan'), G_rel=float('nan'), D_cosine=1.0)
    if torch.isnan(g_star_flat).any() or torch.isinf(g_star_flat).any():
        print(f"[DEBUG] compute_g_metrics: g_star contains NaN/Inf!")
        return GMetrics(G=float('nan'), G_rel=float('nan'), D_cosine=1.0)
    
    # G = ||g̃ - g*||
    diff = g_tilde_flat - g_star_flat
    G = torch.norm(diff).item()
    
    # ||g*||
    g_star_norm_val = torch.norm(g_star_flat).item()
    
    # G_rel = G / (||g*|| + ε)
    G_rel = G / (g_star_norm_val + epsilon)
    
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
    - reduction='sum' + 샘플 수로 나눔
    """
    
    def __init__(
        self, 
        full_dataloader: DataLoader,
        device: str = "cuda"
    ):
        self.full_dataloader = full_dataloader
        self.device = device
    
    def compute_oracle_gradient(
        self, 
        model: nn.Module,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        전체 데이터셋에서 oracle gradient 계산
        
        특징:
        1. train 모드 유지 (학습과 동일한 gradient 계산)
        2. BN stats 백업/복구 (θ_ref 보존)
        3. reduction='sum' + 샘플 수로 나눔 (정확한 평균)
        
        Returns:
            {param_name: gradient_tensor}
        """
        model = model.to(self.device)
        
        # 1. BN stats 백업
        bn_backup = backup_bn_stats(model)
        
        # 2. train 모드 (학습과 동일)
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Initialize gradient accumulators
        grad_accum = {}
        
        # 3. 전체 데이터에 대해 gradient 누적 (배치별 평균)
        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue
            
            model.zero_grad(set_to_none=True)
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            outputs = model(data)
            
            # reduction='mean' to match training (in_round.py uses nn.CrossEntropyLoss() default)
            loss = F.cross_entropy(outputs, labels, reduction='mean')
            loss.backward()
            
            # Accumulate gradients (per-batch mean)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        grad_accum[name] += param.grad.clone().detach().cpu()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear intermediate tensors to prevent OOM
            del outputs, loss, data, labels
        
        # Clear CUDA cache after oracle computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. 배치 수로 나눠서 평균 gradient 추출 (matches training behavior)
        oracle_grad = {}
        for name, grad in grad_accum.items():
            oracle_grad[name] = grad / num_batches
        
        model.zero_grad(set_to_none=True)
        
        # 5. BN stats 복구
        restore_bn_stats(model, bn_backup)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"[Oracle] Computed from {total_samples} samples, avg_loss={avg_loss:.4f}")
        
        if return_loss:
            return oracle_grad, avg_loss
        return oracle_grad
    
    def compute_oracle_with_split_hook(
        self,
        full_model: nn.Module,
        client_param_names: Set[str],
        split_layer_name: str = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Full model로 Oracle 계산 + Split layer에 backward hook을 걸어 정확한 split gradient 수집
        
        Args:
            full_model: Full model (client + server combined)
            client_param_names: 클라이언트 모델의 파라미터 이름들
            split_layer_name: Split layer 이름 (예: "layer1.1.conv2"). None이면 자동 감지
        
        Returns:
            (oracle_client_grad, oracle_server_grad, oracle_split_grad)
        """
        print("[Oracle Split Hook] Computing oracle gradient with split layer hook...")
        
        # Find split layer name from client_param_names if not provided
        if split_layer_name is None:
            # Get the "deepest" layer in client model (last layer before split)
            client_layers = set()
            for name in client_param_names:
                parts = name.rsplit('.', 1)
                if len(parts) > 1:
                    client_layers.add(parts[0])
            
            sorted_layers = sorted(client_layers, key=lambda x: (x.count('.') + x.count('-'), x), reverse=True)
            if sorted_layers:
                split_layer_name = sorted_layers[0]
                # Convert '-' to '.' (ModuleDict uses '-' but full_model uses '.')
                split_layer_name = split_layer_name.replace('-', '.')
                print(f"[Oracle Split Hook] Auto-detected split layer: {split_layer_name}")
        
        full_model = full_model.to(self.device)
        
        # Find the split layer module
        split_layer = None
        for name, module in full_model.named_modules():
            if name == split_layer_name:
                split_layer = module
                break
        
        if split_layer is None:
            print(f"[Oracle Split Hook] Warning: Could not find split layer '{split_layer_name}'")
        
        # Storage for split layer gradient (using forward hook + retain_grad)
        split_activation_storage = {'activations': []}
        hook_handle = None
        
        def forward_hook(module, input, output):
            """Forward hook to capture split layer output for gradient computation"""
            # Clone output to avoid inplace modification issues
            output_clone = output.clone()
            output_clone.requires_grad_(True)
            output_clone.retain_grad()
            split_activation_storage['activations'].append(output_clone)
            return output_clone  # Return cloned tensor to break the view chain
        

        # Register hook
        if split_layer is not None:
            hook_handle = split_layer.register_forward_hook(forward_hook)
            print(f"[Oracle Split Hook] Registered forward hook on '{split_layer_name}'")
        
        # BN stats 백업
        bn_backup = backup_bn_stats(full_model)
        
        # Helper to disable inplace ops (fixes RuntimeError with backward hook)
        original_inplace_states = {}
        for name, module in full_model.named_modules():
            if hasattr(module, 'inplace'):
                original_inplace_states[name] = module.inplace
                module.inplace = False
        
        # train 모드
        full_model.train()
        
        num_batches = 0
        grad_accum = {}
        
        # 전체 데이터로 gradient 계산 (배치별 평균)
        for batch in self.full_dataloader:
            if len(batch) >= 2:
                data, labels = batch[0], batch[1]
            else:
                continue
            
            full_model.zero_grad(set_to_none=True)
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            
            outputs = full_model(data)
            
            # Debug shapes
            if num_batches == 0:
                print(f"[DEBUG] Oracle Input shape: {data.shape}")
                print(f"[DEBUG] Oracle Output shape: {outputs.shape}")
                print(f"[DEBUG] Oracle Labels shape: {labels.shape}")
            
            # reduction='mean' to match training
            loss = F.cross_entropy(outputs, labels, reduction='mean')
            loss.backward()
            
            # Accumulate gradients
            for name, param in full_model.named_parameters():
                if param.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = param.grad.clone().detach().cpu()
                    else:
                        grad_accum[name] += param.grad.clone().detach().cpu()
            
            num_batches += 1
            
            del outputs, loss, data, labels
        
        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()
        
        # 배치 수로 나눠서 평균 gradient 추출
        oracle_grad = {}
        for name, grad in grad_accum.items():
            oracle_grad[name] = grad / num_batches
        
        full_model.zero_grad(set_to_none=True)
        
        # Restore inplace states
        for name, module in full_model.named_modules():
            if name in original_inplace_states:
                module.inplace = original_inplace_states[name]
        
        # BN stats 복구
        restore_bn_stats(full_model, bn_backup)
        
        # Split gradients by param names
        oracle_client_grad = {}
        oracle_server_grad = {}
        
        # Normalize client param names to dots for comparison
        client_param_names_dots = {n.replace("-", ".") for n in client_param_names}
        
        for name, grad in oracle_grad.items():
            # name from full_model already has dots (usually)
            # but just in case, normalize it too
            name_dot = name.replace("-", ".")
            
            if name_dot in client_param_names_dots:
                oracle_client_grad[name] = grad
            else:
                oracle_server_grad[name] = grad
        
        # Split layer gradient from forward hook (평균)
        oracle_split_grad = None
        if split_activation_storage['activations']:
            split_grad_sum = None
            split_count = 0
            for activation in split_activation_storage['activations']:
                if activation.grad is not None:
                    batch_grad = activation.grad.detach()
                    batch_grad_sum = batch_grad.sum(dim=0)
                    if split_grad_sum is None:
                        split_grad_sum = batch_grad_sum.cpu()
                    else:
                        split_grad_sum = split_grad_sum + batch_grad_sum.cpu()
                    split_count += batch_grad.size(0)
            
            if split_count > 0:
                oracle_split_grad = split_grad_sum / split_count
                print(f"[Oracle Split Hook] Split layer grad shape: {oracle_split_grad.shape}")
        
        print(f"[Oracle Split Hook] Split: client={len(oracle_client_grad)}, "
              f"server={len(oracle_server_grad)}, "
              f"split_layer={'yes' if oracle_split_grad is not None else 'no'}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return oracle_client_grad, oracle_server_grad, oracle_split_grad
    
    def compute_oracle_split_gradient(
        self,
        client_model: nn.Module,
        server_model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
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
            
            # Client forward
            activation = client_model(data)
            activation.requires_grad_(True)
            activation.retain_grad()  # Activation gradient 유지
            
            # Server forward
            logits = server_model(activation)
            # reduction='mean' to match training
            loss = F.cross_entropy(logits, labels, reduction='mean')
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
            
            # Split gradient 누적 (batch dimension에서 평균 후 누적)
            if activation.grad is not None:
                # Mean over batch dimension, keep [C, H, W] 
                batch_split_grad = activation.grad.detach().mean(dim=0)
                if split_grad_sum is None:
                    split_grad_sum = batch_split_grad.cpu()
                else:
                    split_grad_sum = split_grad_sum + batch_split_grad.cpu()
            
            num_batches += 1
            
            # Clear intermediate tensors
            del activation, logits, loss, data, labels
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 배치 수로 나눠서 평균 gradient 추출
        oracle_client_grad = {}
        for name, grad in client_grad_accum.items():
            oracle_client_grad[name] = grad / num_batches
        
        oracle_server_grad = {}
        for name, grad in server_grad_accum.items():
            oracle_server_grad[name] = grad / num_batches
        
        # Split gradient 평균
        oracle_split_grad = split_grad_sum / num_batches if split_grad_sum is not None else None
        
        # 초기화 및 복구
        client_model.zero_grad(set_to_none=True)
        server_model.zero_grad(set_to_none=True)
        restore_bn_stats(client_model, client_bn_backup)
        restore_bn_stats(server_model, server_bn_backup)
        
        print(f"[Oracle Split] Computed from {num_batches} batches")
        if oracle_split_grad is not None:
            print(f"[Oracle Split] Split grad shape: {oracle_split_grad.shape}")
        
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
            grad_cpu = grad.clone().detach().cpu().float() if grad.is_cuda else grad.clone().detach().float()
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
        
        return {name: grad / self.total_weight for name, grad in self.gradient_sum.items()}


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
    """
    
    def __init__(
        self,
        diagnostic_rounds: List[int] = [1, 3, 5],
        device: str = "cuda"
    ):
        self.diagnostic_rounds = set(diagnostic_rounds)
        self.device = device
        
        self.oracle_calculator: Optional[OracleGradientCalculator] = None
        
        # Cached oracle gradients (split into client/server)
        self.oracle_client_grad: Optional[Dict[str, torch.Tensor]] = None
        self.oracle_server_grad: Optional[Dict[str, torch.Tensor]] = None
        
        # Param name sets for split
        self.client_param_names: Set[str] = set()
        self.server_param_names: Set[str] = set()
        
        # 1-step measurement results
        self.server_g_tilde: Optional[Dict[str, torch.Tensor]] = None
        self.client_g_tildes: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Split layer gradient (activation gradient at split point)
        self.oracle_split_grad: Optional[torch.Tensor] = None
        self.split_g_tilde: Optional[torch.Tensor] = None
        self.split_g_tildes: Dict[int, torch.Tensor] = {}  # Per-client split gradients
        
        # Results
        self.measurements: List[RoundGMeasurement] = []
    
    def is_diagnostic_round(self, round_number: int) -> bool:
        return round_number in self.diagnostic_rounds
    
    def initialize(self, full_dataloader: DataLoader):
        """Oracle calculator 초기화"""
        self.oracle_calculator = OracleGradientCalculator(full_dataloader, self.device)
        print(f"[G Measurement] Initialized with {len(full_dataloader.dataset)} samples")
    
    def set_param_names(self, client_model: nn.Module, server_model: nn.Module):
        """
        요구사항 3: 실제 client/server 모델의 param name set 설정
        """
        self.client_param_names = get_param_names(client_model)
        self.server_param_names = get_param_names(server_model)
        print(f"[G Measurement] Param names set: client={len(self.client_param_names)}, server={len(self.server_param_names)}")
    
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
            full_oracle, 
            self.client_param_names, 
            self.server_param_names
        )
        
        print(f"[G Measurement] Oracle split: client={len(self.oracle_client_grad)}, server={len(self.oracle_server_grad)}")
    
    def compute_oracle_split_for_round(
        self, 
        client_model: nn.Module, 
        server_model: nn.Module,
        full_model: nn.Module = None,
        split_layer_name: str = None
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
                self.oracle_split_grad
            ) = self.oracle_calculator.compute_oracle_with_split_hook(
                full_model,
                client_param_names,
                split_layer_name
            )
        else:
            # Fallback: compute regular oracle without split layer gradient
            print("[G Measurement] Warning: full_model not provided, split layer gradient not available")
            full_oracle = self.oracle_calculator.compute_oracle_gradient(return_loss=False)
            self.oracle_client_grad = {n: g for n, g in full_oracle.items() if n in client_param_names}
            self.oracle_server_grad = {n: g for n, g in full_oracle.items() if n not in client_param_names}
            self.oracle_split_grad = None
        
        print(f"[G Measurement] Oracle computed: client={len(self.oracle_client_grad)}, "
              f"server={len(self.oracle_server_grad)}, "
              f"split_layer={'yes' if self.oracle_split_grad is not None else 'no'}")
    
    def store_measurement_gradient(
        self,
        server_grad: Dict[str, torch.Tensor],
        client_grads: Dict[int, Dict[str, torch.Tensor]]
    ):
        """
        측정용 1-step에서 수집된 gradient 저장
        
        Args:
            server_grad: 서버 모델의 gradient (1-step)
            client_grads: {client_id: gradient} - 각 클라이언트의 gradient (1-step)
        """
        self.server_g_tilde = {
            name: grad.clone().detach().cpu() if grad.is_cuda else grad.clone().detach()
            for name, grad in server_grad.items()
        }
        
        self.client_g_tildes = {}
        for client_id, grad in client_grads.items():
            self.client_g_tildes[client_id] = {
                name: g.clone().detach().cpu() if g.is_cuda else g.clone().detach()
                for name, g in grad.items()
            }
        
        print(f"[G Measurement] Stored 1-step gradients: server + {len(self.client_g_tildes)} clients")
    
    def _compute_split_g_metrics(
        self, 
        g_tilde: torch.Tensor, 
        g_oracle: torch.Tensor,
        epsilon: float = 1e-8
    ) -> GMetrics:
        """
        Split layer (activation) gradient에 대한 G 계산
        
        Args:
            g_tilde: 실제 측정된 split gradient [평균화된 형태]
            g_oracle: Oracle split gradient [평균화된 형태]
        """
        # Flatten to 1D
        g_tilde_flat = g_tilde.flatten().float()
        g_oracle_flat = g_oracle.flatten().float()
        
        # G = L2 norm
        G = torch.norm(g_tilde_flat - g_oracle_flat).item()
        
        # G_rel = G / ||oracle||
        oracle_norm = torch.norm(g_oracle_flat).item()
        G_rel = G / (oracle_norm + epsilon)
        
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
    
    def compute_g(self, round_number: int) -> RoundGMeasurement:
        """
        G 계산 (요구사항 4: sorted keys로 일관된 순서)
        """
        result = RoundGMeasurement(
            round_number=round_number,
            is_diagnostic=self.is_diagnostic_round(round_number)
        )
        
        if not result.is_diagnostic:
            return result
        
        # Server G
        if self.server_g_tilde and self.oracle_server_grad:
            result.server = compute_g_metrics(self.server_g_tilde, self.oracle_server_grad)
            print(f"[G] Server: G={result.server.G:.6f}, G_rel={result.server.G_rel:.4f}, D={result.server.D_cosine:.4f}")
        
        # Split Layer G (aggregated)
        if self.split_g_tilde is not None and self.oracle_split_grad is not None:
            split_metrics = self._compute_split_g_metrics(self.split_g_tilde, self.oracle_split_grad)
            result.split_layer = split_metrics
            print(f"[G] Split Layer: G={split_metrics.G:.6f}, G_rel={split_metrics.G_rel:.4f}, D={split_metrics.D_cosine:.4f}")
        
        # Client G
        client_Gs = []
        client_Ds = []
        
        for client_id, g_tilde in self.client_g_tildes.items():
            if self.oracle_client_grad:
                # DEBUG: Print first few values to verify gradients are unique
                first_key = next(iter(g_tilde.keys())) if g_tilde else None
                if first_key and first_key in g_tilde:
                    grad_sample = g_tilde[first_key].flatten()[:5].tolist()
                    print(f"[DEBUG] Client {client_id} grad sample: {grad_sample}")
                
                metrics = compute_g_metrics(g_tilde, self.oracle_client_grad)
                result.clients[client_id] = metrics
                client_Gs.append(metrics.G)
                client_Ds.append(metrics.D_cosine)
                print(f"[G] Client {client_id}: G={metrics.G:.6f}, D={metrics.D_cosine:.4f}")
        
        # Summary
        if client_Gs:
            result.client_G_mean = sum(client_Gs) / len(client_Gs)
            result.client_G_max = max(client_Gs)
            result.client_D_mean = sum(client_Ds) / len(client_Ds)
            print(f"[G] Summary: G_mean={result.client_G_mean:.6f}, G_max={result.client_G_max:.6f}")
        
        self.measurements.append(result)
        return result
    
    def clear_round_data(self):
        """라운드 데이터 초기화 (메모리 해제)"""
        self.server_g_tilde = None
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
