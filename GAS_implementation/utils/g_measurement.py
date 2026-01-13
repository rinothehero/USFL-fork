"""
G Measurement Module for GAS (USFL-style 3-Perspective Gradient Distance)

This module provides functions to measure the gradient distance (G) between
the current training state and the ideal (Oracle) state from three perspectives:
1. Client-side G: Bias in individual client gradients
2. Server-side G: Server model's view of global distribution
3. Split-layer G: Distortion at the cut point between client and server

Memory Optimization Strategies:
- Lazy Evaluation: Only compute on diagnostic rounds
- First-batch-only: For split-layer, only capture first batch
- CPU Offloading: Move Oracle gradients to CPU immediately
- Immediate Cleanup: del intermediate tensors, empty_cache()
- BN Stats Backup: Only backup running_mean/running_var
"""

import torch
import torch.nn as nn
import copy


class GradientCollector:
    """
    Collects and stores gradients for G measurement.
    Uses CPU offloading to minimize GPU memory usage.
    """

    def __init__(self, device):
        self.device = device
        self.oracle_client_grad = None
        self.oracle_server_grad = None
        self.oracle_split_grad = None
        self.current_client_grad = None
        self.current_server_grad = None
        self.current_split_grad = None

    def clear(self):
        """Clear all stored gradients to free memory."""
        self.oracle_client_grad = None
        self.oracle_server_grad = None
        self.oracle_split_grad = None
        self.current_client_grad = None
        self.current_server_grad = None
        self.current_split_grad = None
        torch.cuda.empty_cache()


def get_param_names(model):
    return [name for name, _ in model.named_parameters()]


def assert_param_name_alignment(expected_names, model, label):
    current_names = get_param_names(model)
    if current_names != expected_names:
        missing = [name for name in expected_names if name not in current_names]
        extra = [name for name in current_names if name not in expected_names]
        raise ValueError(
            "[G Measurement] Param name mismatch for "
            f"{label}: missing={missing[:5]}, extra={extra[:5]}, "
            f"missing_count={len(missing)}, extra_count={len(extra)}"
        )


def backup_bn_stats(model):
    """
    Backup only BatchNorm running statistics (memory-efficient).
    Returns a dict of {module_name: (running_mean, running_var)}.
    """
    bn_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_stats[name] = (
                module.running_mean.clone().cpu()
                if module.running_mean is not None
                else None,
                module.running_var.clone().cpu()
                if module.running_var is not None
                else None,
            )
    return bn_stats


def restore_bn_stats(model, bn_stats, device):
    """
    Restore BatchNorm running statistics from backup.
    """
    for name, module in model.named_modules():
        if name in bn_stats:
            mean, var = bn_stats[name]
            if mean is not None:
                module.running_mean.copy_(mean.to(device))
            if var is not None:
                module.running_var.copy_(var.to(device))


def compute_oracle_gradients(
    user_model, server_model, train_loader, criterion, device, max_batches=None
):
    """
    Compute Oracle gradients using full training data.

    This function computes the "ideal" gradients that would be obtained
    if we could see all training data at once.

    Args:
        user_model: Client-side model
        server_model: Server-side model
        train_loader: DataLoader for full training data
        criterion: Loss function
        device: torch device
        max_batches: Optional limit on batches to process (for memory)

    Returns:
        dict: {
            'client': list of gradient tensors (on CPU),
            'server': list of gradient tensors (on CPU),
            'split': gradient tensor at split layer (on CPU)
        }
    """
    # Backup BN stats
    user_bn_stats = backup_bn_stats(user_model)
    server_bn_stats = backup_bn_stats(server_model)

    user_model.train()
    server_model.train()

    client_param_names = get_param_names(user_model)
    server_param_names = get_param_names(server_model)

    client_grad_accum = [torch.zeros_like(p).cpu() for p in user_model.parameters()]
    server_grad_accum = [torch.zeros_like(p).cpu() for p in server_model.parameters()]
    split_grad_accum = None
    batch_count = 0
    for images, labels in train_loader:
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with gradient tracking for split layer
        split_output = user_model(images)

        # Handle tuple output from fine-grained split (activation, identity)
        if isinstance(split_output, tuple):
            activation, identity = split_output
            activation.retain_grad()  # Enable gradient capture at split layer
            server_output = server_model(split_output)
        else:
            split_output.retain_grad()  # Enable gradient capture at split layer
            activation = split_output
            server_output = server_model(split_output)

        # Use reduction='mean' to match sfl_framework (per-batch mean, then average over batches)
        loss = torch.nn.functional.cross_entropy(
            server_output, labels.long(), reduction="mean"
        )

        # Backward pass
        loss.backward()

        # Accumulate client gradients (per-batch mean)
        for i, p in enumerate(user_model.parameters()):
            if p.grad is not None:
                client_grad_accum[i] += p.grad.cpu()

        # Accumulate server gradients
        for i, p in enumerate(server_model.parameters()):
            if p.grad is not None:
                server_grad_accum[i] += p.grad.cpu()

        # Accumulate split layer gradient (first batch only for memory)
        # Take mean over batch dimension to make shape batch-size independent
        if split_grad_accum is None and activation.grad is not None:
            split_grad_accum = activation.grad.mean(dim=0).clone().cpu()

        batch_count += 1

        # Clean up
        user_model.zero_grad()
        server_model.zero_grad()
        del images, labels, split_output, server_output, loss

    if batch_count > 0:
        client_grad_accum = [g / batch_count for g in client_grad_accum]
        server_grad_accum = [g / batch_count for g in server_grad_accum]

    # Restore BN stats
    restore_bn_stats(user_model, user_bn_stats, device)
    restore_bn_stats(server_model, server_bn_stats, device)

    # Clear GPU cache
    torch.cuda.empty_cache()

    return {
        "client": client_grad_accum,
        "server": server_grad_accum,
        "split": split_grad_accum,
        "client_names": client_param_names,
        "server_names": server_param_names,
    }


def collect_current_gradients(user_model, server_model, split_output, loss):
    """
    Collect current gradients during normal training step.

    This function should be called AFTER loss.backward() but BEFORE optimizer.step().

    Args:
        user_model: Client-side model (after backward)
        server_model: Server-side model (after backward)
        split_output: The split layer output tensor (must have retain_grad() called before backward)
        loss: The loss tensor (for reference, not used directly)

    Returns:
        dict: {
            'client': list of gradient tensors (on CPU),
            'server': list of gradient tensors (on CPU),
            'split': gradient tensor at split layer (on CPU)
        }
    """
    current_grads = {"client": [], "server": [], "split": None}

    # Collect client gradients
    for p in user_model.parameters():
        if p.grad is not None:
            current_grads["client"].append(p.grad.clone().cpu())
        else:
            current_grads["client"].append(torch.zeros_like(p).cpu())

    # Collect server gradients
    for p in server_model.parameters():
        if p.grad is not None:
            current_grads["server"].append(p.grad.clone().cpu())
        else:
            current_grads["server"].append(torch.zeros_like(p).cpu())

    # Collect split layer gradient
    # Handle tuple output from fine-grained split
    if isinstance(split_output, tuple):
        activation = split_output[0]
    else:
        activation = split_output

    if activation.grad is not None:
        current_grads["split"] = activation.grad.clone().cpu()

    return current_grads


def compute_g_score(oracle_grads, current_grads, return_details=False):
    """
    Compute G score: || flatten(g_tilde) - flatten(g_star) ||

    USFL-style: Flatten all gradient tensors into one vector, then compute L2 norm.

    Args:
        oracle_grads: list of Oracle gradient tensors (or single tensor)
        current_grads: list of Current gradient tensors (or single tensor)
        return_details: if True, return dict with oracle_norm, current_norm, G, G_rel

    Returns:
        float: G score (L2 norm of flattened difference)
        OR dict: {'oracle_norm', 'current_norm', 'G', 'G_rel'} if return_details=True
    """
    if oracle_grads is None or current_grads is None:
        if return_details:
            return {
                "oracle_norm": float("nan"),
                "current_norm": float("nan"),
                "G": float("nan"),
                "G_rel": float("nan"),
            }
        return float("nan")

    if isinstance(oracle_grads, list) and isinstance(current_grads, list):
        if len(oracle_grads) != len(current_grads):
            if return_details:
                return {
                    "oracle_norm": float("nan"),
                    "current_norm": float("nan"),
                    "G": float("nan"),
                    "G_rel": float("nan"),
                }
            return float("nan")

        # Flatten all gradients into one vector
        oracle_flat = torch.cat([g.flatten().float() for g in oracle_grads])
        current_flat = torch.cat([g.flatten().float() for g in current_grads])
    else:
        # Single tensor (split layer)
        oracle_flat = oracle_grads.flatten().float()
        current_flat = current_grads.flatten().float()

    # Compute norms
    oracle_norm = torch.norm(oracle_flat).item()
    current_norm = torch.norm(current_flat).item()
    oracle_norm_sq = torch.dot(oracle_flat, oracle_flat).item()
    diff = current_flat - oracle_flat
    G = torch.dot(diff, diff).item()
    G_rel = G / (oracle_norm_sq + 1e-10)

    # Compute Cosine Distance (D_cosine)
    if current_norm > 1e-10 and oracle_norm > 1e-10:
        dot_product = torch.dot(current_flat, oracle_flat).item()
        cos_sim = dot_product / (current_norm * oracle_norm)
        D_cosine = 1.0 - cos_sim
    else:
        D_cosine = 1.0

    # Clamp to [0, 2]
    D_cosine = max(0.0, min(2.0, D_cosine))

    if return_details:
        return {
            "oracle_norm": oracle_norm,
            "current_norm": current_norm,
            "G": G,
            "G_rel": G_rel,
            "D_cosine": D_cosine,
        }
    return G


def compute_all_g_scores(oracle, current):
    """
    Compute all three G scores.

    Args:
        oracle: dict with 'client', 'server', 'split' Oracle gradients
        current: dict with 'client', 'server', 'split' Current gradients

    Returns:
        dict: {'client_g': float, 'server_g': float, 'split_g': float}
    """
    return {
        "client_g": compute_g_score(oracle["client"], current["client"]),
        "server_g": compute_g_score(oracle["server"], current["server"]),
        "split_g": compute_g_score(oracle["split"], current["split"]),
    }


class GMeasurementManager:
    """
    Manager class for G measurement in GAS training.

    Handles the diagnostic round pattern and memory-efficient computation.
    """

    def __init__(self, device, measure_frequency=10):
        """
        Args:
            device: torch device
            measure_frequency: How often to measure G (every N epochs)
        """
        self.device = device
        self.measure_frequency = measure_frequency
        self.collector = GradientCollector(device)
        self.oracle_computed = False
        self.oracle_grads = None
        self.g_history = {"client_g": [], "server_g": [], "split_g": []}

    def should_measure(self, epoch):
        """Check if this epoch is a diagnostic round."""
        return (epoch + 1) % self.measure_frequency == 0

    def compute_oracle(self, user_model, server_model, train_loader, criterion):
        """
        Compute and store Oracle gradients.
        Should be called once per diagnostic round before training step.
        """
        print("[G Measurement] Computing Oracle gradients...")
        self.oracle_grads = compute_oracle_gradients(
            user_model, server_model, train_loader, criterion, self.device
        )
        self.oracle_computed = True
        print("[G Measurement] Oracle computation complete.")

    def measure_and_record(self, user_model, server_model, split_output, loss, epoch):
        """
        Measure G scores during training and record them.

        This should be called AFTER loss.backward() but BEFORE optimizer.step().
        """
        if not self.oracle_computed:
            print("[G Measurement] Warning: Oracle not computed, skipping measurement.")
            return None

        # Collect current gradients
        current_grads = collect_current_gradients(
            user_model, server_model, split_output, loss
        )

        # Compute G scores
        g_scores = compute_all_g_scores(self.oracle_grads, current_grads)

        # Record
        self.g_history["client_g"].append(g_scores["client_g"])
        self.g_history["server_g"].append(g_scores["server_g"])
        self.g_history["split_g"].append(g_scores["split_g"])

        print(
            f"[G Measurement] Epoch {epoch + 1}: "
            f"Client G = {g_scores['client_g']:.6f}, "
            f"Server G = {g_scores['server_g']:.6f}, "
            f"Split G = {g_scores['split_g']:.6f}"
        )

        # Clean up for next round
        self.oracle_computed = False
        self.oracle_grads = None
        self.collector.clear()

        return g_scores

    def get_history(self):
        """Return the history of G measurements."""
        return self.g_history
