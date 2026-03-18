"""
Client-side SFL operations, data structures, and dynamic batch scheduling.

Pure functions — no async, no queues, no communication protocol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClientState:
    """Per-client training state for one round."""
    client_id: int
    client_model: nn.Module
    optimizer: torch.optim.Optimizer
    dataloader: DataLoader
    data_iter: Iterator
    label_distribution: Dict[int, int]
    dataset_size: int
    exhausted: bool = False  # True when client has no more data
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientResult:
    """Result of one client's training in a round."""
    client_id: int
    model_state_dict: dict
    dataset_size: int
    label_distribution: Dict[int, int]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundContext:
    """Everything needed to execute one SFL round."""
    round_number: int
    selected_client_ids: List[int]
    client_states: Dict[int, ClientState]
    server_model: nn.Module
    server_optimizer: torch.optim.Optimizer
    criterion: nn.Module
    iterations: int
    device: torch.device
    batch_schedule: Optional[dict] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundResult:
    """Final result of a training round."""
    round_number: int
    accuracy: float
    loss: float
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model state utilities
# ---------------------------------------------------------------------------


def snapshot_model(model: nn.Module) -> dict:
    """Create a detached copy of model's state_dict."""
    return {k: v.clone().detach() for k, v in model.state_dict().items()}




# ---------------------------------------------------------------------------
# Label distribution (canonical implementation in data.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Optimizer / Criterion factories
# ---------------------------------------------------------------------------


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create client optimizer from config."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer in ("adam", "adamw"):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_server_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create server-side optimizer (may use scaled LR)."""
    if config.scale_server_lr:
        lr = config.learning_rate * config.num_clients_per_round
    elif config.server_learning_rate is not None:
        lr = config.server_learning_rate
    else:
        lr = config.learning_rate

    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer in ("adam", "adamw"):
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_criterion(config) -> nn.Module:
    """Create loss function from config."""
    if config.criterion == "ce":
        return nn.CrossEntropyLoss()
    elif config.criterion == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown criterion: {config.criterion}")


# ---------------------------------------------------------------------------
# Client state factory
# ---------------------------------------------------------------------------


def create_client_state(
    client_id: int,
    client_model: nn.Module,
    trainset,
    data_indices: List[int],
    config,
    batch_size: Optional[int] = None,
) -> ClientState:
    """Create a ClientState for one client in one round."""
    dataset = Subset(trainset, data_indices)
    bs = batch_size or config.batch_size
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    optimizer = create_optimizer(client_model, config)
    label_dist = get_label_distribution(trainset, data_indices)

    return ClientState(
        client_id=client_id,
        client_model=client_model,
        optimizer=optimizer,
        dataloader=dataloader,
        data_iter=iter(dataloader),
        label_distribution=label_dist,
        dataset_size=len(dataset),
    )


# ---------------------------------------------------------------------------
# Client forward / backward
# ---------------------------------------------------------------------------


def client_forward(
    state: ClientState,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Client-side forward pass.

    Returns (activation, labels) where activation is detached from client graph
    but has grad enabled for server backward.
    """
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    state.client_model.train()
    activation = state.client_model(images)

    # Detach from client graph, enable grad for server backward
    activation = activation.detach().requires_grad_(True)
    activation.retain_grad()

    return activation, labels


def client_backward(
    state: ClientState,
    images: torch.Tensor,
    activation_grad: torch.Tensor,
    config,
):
    """
    Client-side backward pass: re-forward and backprop using
    the activation gradient received from server.
    """
    state.optimizer.zero_grad()

    state.client_model.train()
    act = state.client_model(images)
    act.backward(activation_grad)

    if config.clip_grad:
        nn.utils.clip_grad_norm_(
            state.client_model.parameters(), config.clip_grad_max_norm
        )

    state.optimizer.step()


def get_next_batch(
    state: ClientState, device: torch.device, policy: str = "cycling"
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Get next batch from client's dataloader.

    Returns None if client is exhausted and policy is "skip" or "break".

    Policy:
        "cycling": restart dataloader from beginning (default, legacy)
        "skip": return None when exhausted (client skips this iteration)
        "break": return None when exhausted (caller should stop all clients)
        "dbs": same as cycling (DBS ensures all exhaust simultaneously)
    """
    if state.exhausted:
        if policy in ("skip", "break"):
            return None
        # cycling / dbs: reset
        state.data_iter = iter(state.dataloader)
        state.exhausted = False

    try:
        batch = next(state.data_iter)
    except StopIteration:
        state.exhausted = True
        if policy in ("skip", "break"):
            return None
        # cycling / dbs: restart
        state.data_iter = iter(state.dataloader)
        state.exhausted = False
        batch = next(state.data_iter)

    images, labels = batch
    return images.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# Dynamic batch scheduler (for USFL)
# ---------------------------------------------------------------------------


def create_schedule(
    target_batch_size: int,
    client_data_sizes: Dict[int, int],
) -> Tuple[int, List[Dict[int, int]]]:
    """
    Compute optimal iteration count and per-client per-iteration batch sizes.

    Phase 1: Find k that minimizes |sum(ceil(C_i/k)) - B|
    Phase 2: Distribute batches proportionally per iteration

    Args:
        target_batch_size: B (desired total batch size per iteration)
        client_data_sizes: {client_id: data_count}

    Returns:
        (k, schedule) where:
        - k: number of iterations
        - schedule[iteration][client_id] = batch_size
    """
    client_ids = sorted(client_data_sizes.keys())
    C = [client_data_sizes[cid] for cid in client_ids]

    if not C or all(c == 0 for c in C):
        return 0, []

    max_c = max(C)

    # Phase 1: Find best k
    best_k = 1
    best_diff = float("inf")

    for k in range(1, max_c + 1):
        total = sum(math.ceil(c / k) for c in C)
        diff = abs(total - target_batch_size)
        if diff < best_diff:
            best_diff = diff
            best_k = k

    # Phase 2: Compute per-client per-iteration batch sizes
    schedule = []
    for it in range(best_k):
        batch_sizes = {}
        for i, cid in enumerate(client_ids):
            c = C[i]
            per_iter = math.ceil(c / best_k)
            used = it * per_iter
            remaining = c - used
            batch_sizes[cid] = max(0, min(per_iter, remaining))
        schedule.append(batch_sizes)

    return best_k, schedule
