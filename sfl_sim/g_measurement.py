"""
G-measurement module for GAS (Gradient Adjustment Scheme).

Measures gradient quality from three perspectives:
1. Client G: gradient distance in client model (bias from client-only training)
2. Server G: gradient distance in server model (influence of split architecture)
3. Split G: gradient at split layer (distortion at activation cut point)

Oracle: full-data gradient computed on probe/test data.
G-score: L2 distance between current gradient and oracle gradient.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BatchNorm stats backup/restore (oracle computation must not corrupt BN)
# ---------------------------------------------------------------------------


def _backup_bn_stats(model: nn.Module) -> Dict[str, Tuple]:
    """Backup BatchNorm running statistics (memory-efficient)."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            stats[name] = (
                module.running_mean.clone().detach() if module.running_mean is not None else None,
                module.running_var.clone().detach() if module.running_var is not None else None,
            )
    return stats


def _restore_bn_stats(model: nn.Module, stats: Dict[str, Tuple]):
    """Restore BatchNorm running statistics from backup."""
    for name, module in model.named_modules():
        if name in stats:
            mean, var = stats[name]
            if mean is not None:
                module.running_mean.copy_(mean)
            if var is not None:
                module.running_var.copy_(var)


# ---------------------------------------------------------------------------
# Oracle gradient computation
# ---------------------------------------------------------------------------


def compute_oracle_gradients(
    client_model: nn.Module,
    server_model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Compute oracle gradients using full data (probe or test set).

    Runs client -> server forward, cross-entropy loss with reduction='sum',
    accumulates gradients, then divides by total samples.

    Returns:
        dict with keys 'client', 'server', 'split' — each a list of CPU tensors
        (or single tensor for split).
    """
    # Backup BN stats so oracle computation doesn't corrupt them
    client_bn = _backup_bn_stats(client_model)
    server_bn = _backup_bn_stats(server_model)

    client_model.train()
    server_model.train()

    client_grad_accum = [torch.zeros_like(p, device="cpu") for p in client_model.parameters()]
    server_grad_accum = [torch.zeros_like(p, device="cpu") for p in server_model.parameters()]
    split_grad_accum = None

    total_samples = 0
    batch_count = 0

    for images, labels in dataloader:
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        total_samples += batch_size

        client_model.zero_grad()
        server_model.zero_grad()

        # Forward: client -> server
        activation = client_model(images)
        activation.retain_grad()

        logits = server_model(activation)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        loss.backward()

        # Accumulate client gradients
        for i, p in enumerate(client_model.parameters()):
            if p.grad is not None:
                client_grad_accum[i] += p.grad.detach().cpu()

        # Accumulate server gradients
        for i, p in enumerate(server_model.parameters()):
            if p.grad is not None:
                server_grad_accum[i] += p.grad.detach().cpu()

        # Split layer gradient (accumulate across all batches)
        if activation.grad is not None:
            batch_split = activation.grad.detach().sum(dim=0).cpu()
            if split_grad_accum is None:
                split_grad_accum = batch_split
            else:
                split_grad_accum += batch_split

        batch_count += 1

        # Cleanup
        del images, labels, activation, logits, loss

    # Normalize by total samples
    if total_samples > 0:
        client_grad_accum = [g / total_samples for g in client_grad_accum]
        server_grad_accum = [g / total_samples for g in server_grad_accum]
        if split_grad_accum is not None:
            split_grad_accum = split_grad_accum / total_samples

    # Restore BN stats
    _restore_bn_stats(client_model, client_bn)
    _restore_bn_stats(server_model, server_bn)

    client_model.zero_grad()
    server_model.zero_grad()

    return {
        "client": client_grad_accum,
        "server": server_grad_accum,
        "split": split_grad_accum,
    }


# ---------------------------------------------------------------------------
# Current gradient collection (called after loss.backward, before step)
# ---------------------------------------------------------------------------


def collect_current_gradients(
    client_model: nn.Module,
    server_model: nn.Module,
    activation_grad: Optional[torch.Tensor] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Collect current gradients from models after loss.backward().

    Returns same format as compute_oracle_gradients.
    """
    client_grads = []
    for p in client_model.parameters():
        if p.grad is not None:
            client_grads.append(p.grad.detach().cpu())
        else:
            client_grads.append(torch.zeros_like(p, device="cpu"))

    server_grads = []
    for p in server_model.parameters():
        if p.grad is not None:
            server_grads.append(p.grad.detach().cpu())
        else:
            server_grads.append(torch.zeros_like(p, device="cpu"))

    split_grad = None
    if activation_grad is not None:
        split_grad = activation_grad.detach().sum(dim=0).cpu()

    return {
        "client": client_grads,
        "server": server_grads,
        "split": split_grad,
    }


# ---------------------------------------------------------------------------
# G-score computation
# ---------------------------------------------------------------------------


def compute_g_score(
    oracle_grads,
    current_grads,
) -> float:
    """
    Compute G-score: squared L2 distance between flattened gradient vectors.

    Args:
        oracle_grads: list of gradient tensors OR single tensor (for split)
        current_grads: list of gradient tensors OR single tensor

    Returns:
        float: G-score (squared L2 norm of difference)
    """
    if oracle_grads is None or current_grads is None:
        return float("nan")

    if isinstance(oracle_grads, list) and isinstance(current_grads, list):
        if len(oracle_grads) != len(current_grads):
            return float("nan")
        oracle_flat = torch.cat([g.flatten().float() for g in oracle_grads])
        current_flat = torch.cat([g.flatten().float() for g in current_grads])
    else:
        oracle_flat = oracle_grads.flatten().float()
        current_flat = current_grads.flatten().float()

    diff = current_flat - oracle_flat
    return torch.dot(diff, diff).item()


def compute_all_g_scores(
    oracle: Dict[str, List[torch.Tensor]],
    current: Dict[str, List[torch.Tensor]],
) -> Dict[str, float]:
    """
    Compute G-scores for all three perspectives.

    Returns: dict with 'client_g', 'server_g', 'split_g'
    """
    return {
        "client_g": compute_g_score(oracle["client"], current["client"]),
        "server_g": compute_g_score(oracle["server"], current["server"]),
        "split_g": compute_g_score(oracle.get("split"), current.get("split")),
    }


# ---------------------------------------------------------------------------
# V-value computation (gradient dissimilarity for client ranking)
# ---------------------------------------------------------------------------


def compute_v_value(
    server_model: nn.Module,
    client_model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
    num_minibatches: int = 10,
) -> float:
    """
    Compute V-value: gradient dissimilarity between sampled/generated features
    and true test data gradients.

    Lower V-value = client provides more representative features.

    Args:
        server_model: Server-side model
        client_model: Client-side model
        features: Concatenated activation features from client(s)
        labels: Labels corresponding to features
        test_loader: DataLoader for test/probe data
        criterion: Loss function
        device: torch device
        num_minibatches: Number of test batches for estimating true gradient

    Returns:
        float: V-value (gradient dissimilarity)
    """
    # Estimate true gradient from test data
    total_grads = [torch.zeros_like(p) for p in server_model.parameters()]
    test_iter = iter(test_loader)

    for _ in range(num_minibatches):
        try:
            images, lbls = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            images, lbls = next(test_iter)

        images, lbls = images.to(device), lbls.to(device)
        with torch.no_grad():
            act = client_model(images)
        act = act.detach().requires_grad_(False)

        # Need gradients for server params
        logits = server_model(act)
        loss = criterion(logits, lbls)
        grads = torch.autograd.grad(
            loss, server_model.parameters(), retain_graph=False, allow_unused=True
        )
        for tg, g in zip(total_grads, grads):
            if g is not None:
                tg += g.detach()
        server_model.zero_grad()

    grads_real = [tg / num_minibatches for tg in total_grads]

    # Compute gradient from sampled/generated features
    features = features.detach().requires_grad_(False)
    logits = server_model(features)
    loss = criterion(logits, labels)
    grads_sampled = torch.autograd.grad(
        loss, server_model.parameters(), retain_graph=False, allow_unused=True
    )
    grads_sampled = [
        g.detach() if g is not None else torch.zeros_like(p)
        for g, p in zip(grads_sampled, server_model.parameters())
    ]
    server_model.zero_grad()

    # V-value: mean squared L2 distance across parameter groups
    v_value = sum(
        (torch.norm(gs - gr) ** 2).item()
        for gs, gr in zip(grads_sampled, grads_real)
    ) / max(len(grads_sampled), 1)

    return v_value


# ---------------------------------------------------------------------------
# Feature generation from label-wise statistics
# ---------------------------------------------------------------------------


class FeatureStats:
    """
    Tracks per-label mean and variance of activation features.

    Used to generate synthetic features for missing labels via Gaussian sampling.
    """

    def __init__(self):
        self._mean: Dict[int, torch.Tensor] = {}
        self._var: Dict[int, torch.Tensor] = {}
        self._count: Dict[int, int] = {}

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update running statistics with a batch of features.

        Uses Welford-style online update for numerical stability.
        """
        for label in labels.unique().tolist():
            mask = labels == label
            feats = features[mask].detach().cpu()
            flat_feats = feats.view(feats.size(0), -1)  # Flatten spatial dims

            batch_mean = flat_feats.mean(dim=0)
            batch_var = flat_feats.var(dim=0, unbiased=False) if flat_feats.size(0) > 1 else torch.zeros_like(batch_mean)
            batch_count = flat_feats.size(0)

            if label not in self._mean:
                self._mean[label] = batch_mean
                self._var[label] = batch_var
                self._count[label] = batch_count
            else:
                old_count = self._count[label]
                new_count = old_count + batch_count
                old_mean = self._mean[label]

                # Combined mean
                new_mean = (old_mean * old_count + batch_mean * batch_count) / new_count
                # Combined variance (parallel algorithm)
                delta = batch_mean - old_mean
                new_var = (
                    self._var[label] * old_count
                    + batch_var * batch_count
                    + delta ** 2 * old_count * batch_count / new_count
                ) / new_count

                self._mean[label] = new_mean
                self._var[label] = new_var
                self._count[label] = new_count

    def has_label(self, label: int) -> bool:
        return label in self._mean

    def generate(
        self,
        label: int,
        num_samples: int,
        activation_shape: Tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate synthetic features for a label using diagonal Gaussian.

        Args:
            label: Class label
            num_samples: Number of features to generate
            activation_shape: Shape of a single activation (excluding batch dim)
            device: Target device

        Returns:
            Tensor of shape (num_samples, *activation_shape)
        """
        if label not in self._mean:
            # No stats available; return random noise
            return torch.randn(num_samples, *activation_shape, device=device) * 0.01

        mean = self._mean[label].to(device)
        std = torch.sqrt(self._var[label].to(device) + 1e-5)

        mean_exp = mean.unsqueeze(0).expand(num_samples, -1)
        std_exp = std.unsqueeze(0).expand(num_samples, -1)
        flat_samples = torch.normal(mean=mean_exp, std=std_exp)

        # Reshape back to activation shape
        return flat_samples.view(num_samples, *activation_shape)
