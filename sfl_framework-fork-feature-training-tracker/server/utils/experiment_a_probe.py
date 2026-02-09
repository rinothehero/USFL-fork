from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def _extract_batch(
    batch: Any, device: str
) -> Tuple[Tuple[Any, ...], Dict[str, Any], Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        labels = batch.get("labels", batch.get("label"))
        kwargs = {}
        for k, v in batch.items():
            if k in ("labels", "label"):
                continue
            kwargs[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        return tuple(), kwargs, labels

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x = batch[0]
        y = batch[1]
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        if isinstance(y, torch.Tensor):
            y = y.to(device)
        return (x,), {}, y

    return tuple(), {}, None


def _to_logits(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    return None


def _loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.dtype in (torch.float16, torch.float32, torch.float64):
        if logits.dim() == labels.dim():
            return F.mse_loss(logits, labels)
        if logits.dim() == labels.dim() + 1 and logits.size(-1) == 1:
            return F.mse_loss(logits.squeeze(-1), labels.float())
    return F.cross_entropy(logits, labels.long())


def _accumulate_grad(
    model: torch.nn.Module,
    grad_sum: Dict[str, torch.Tensor],
    batch_weight: int,
) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        g = param.grad.detach().cpu().float() * float(batch_weight)
        if name in grad_sum:
            grad_sum[name] += g
        else:
            grad_sum[name] = g


def _flatten_avg_grad(
    model: torch.nn.Module,
    grad_sum: Dict[str, torch.Tensor],
    total_weight: int,
) -> Optional[torch.Tensor]:
    if total_weight <= 0:
        return None
    out = []
    inv = 1.0 / float(total_weight)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        g = grad_sum.get(name)
        if g is None:
            g = torch.zeros_like(param.detach().cpu().float())
        else:
            g = g * inv
        out.append(g.reshape(-1))
    if not out:
        return None
    return torch.cat(out)


def _load_probe_indices(indices_path: str) -> List[int]:
    path = (indices_path or "").strip()
    if not path:
        return []

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = (
                payload.get("indices")
                or payload.get("probe_indices")
                or payload.get("q_indices")
                or []
            )
        if not isinstance(payload, list):
            return []
        raw_values = payload
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_values = [line.strip() for line in f if line.strip()]

    out: List[int] = []
    seen = set()
    for value in raw_values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx in seen:
            continue
        out.append(idx)
        seen.add(idx)
    return out


def build_probe_loader(
    default_loader: Optional[DataLoader],
    train_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    source: str = "test",
    indices_path: str = "",
    num_samples: int = 0,
    batch_size: int = 0,
    seed: int = 0,
) -> Tuple[Optional[DataLoader], dict]:
    source_norm = str(source or "test").lower()
    if source_norm not in ("test", "train"):
        source_norm = "test"

    dataset = test_dataset if source_norm == "test" else train_dataset
    if dataset is None and default_loader is not None:
        dataset = default_loader.dataset
    if dataset is None:
        return None, {"source": source_norm, "selected_samples": 0}

    total_samples = len(dataset)
    selected_indices: Optional[List[int]] = None

    loaded_indices = _load_probe_indices(indices_path)
    if loaded_indices:
        filtered = [idx for idx in loaded_indices if 0 <= idx < total_samples]
        if filtered:
            selected_indices = filtered
    elif int(num_samples) > 0 and total_samples > 0:
        k = min(int(num_samples), total_samples)
        all_indices = list(range(total_samples))
        random.Random(int(seed)).shuffle(all_indices)
        selected_indices = all_indices[:k]

    probe_dataset = Subset(dataset, selected_indices) if selected_indices is not None else dataset

    if batch_size and int(batch_size) > 0:
        probe_batch_size = int(batch_size)
    elif default_loader is not None and getattr(default_loader, "batch_size", None):
        probe_batch_size = int(default_loader.batch_size)
    else:
        probe_batch_size = 1

    loader_kwargs = {
        "dataset": probe_dataset,
        "batch_size": max(probe_batch_size, 1),
        "shuffle": False,
        "drop_last": False,
    }
    if default_loader is not None:
        loader_kwargs["num_workers"] = int(getattr(default_loader, "num_workers", 0))
        loader_kwargs["pin_memory"] = bool(getattr(default_loader, "pin_memory", False))
        collate_fn = getattr(default_loader, "collate_fn", None)
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn

    probe_loader = DataLoader(**loader_kwargs)
    return probe_loader, {
        "source": source_norm,
        "total_samples": int(total_samples),
        "selected_samples": int(len(probe_dataset)),
        "batch_size": int(max(probe_batch_size, 1)),
        "indices_path": indices_path or "",
        "num_samples": int(max(int(num_samples), 0)),
        "seed": int(seed),
    }


def compute_split_probe_directions(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    probe_loader: Any,
    device: str,
    max_batches: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    """
    Compute central directions c_c, c_s from a fixed probe loader:
      c = -∇ L_Q(x^{t,0})
    """
    if probe_loader is None:
        return None, None, {"used_batches": 0, "used_samples": 0}

    max_batches = max(int(max_batches), 1)

    was_client_training = client_model.training
    was_server_training = server_model.training
    client_model.eval()
    server_model.eval()

    client_grad_sum: Dict[str, torch.Tensor] = {}
    server_grad_sum: Dict[str, torch.Tensor] = {}
    used_batches = 0
    used_samples = 0

    try:
        for batch in probe_loader:
            args, kwargs, labels = _extract_batch(batch, device)
            if labels is None:
                continue

            if labels.dim() == 0:
                batch_size = 1
            else:
                batch_size = int(labels.shape[0])
            if batch_size <= 0:
                continue

            client_model.zero_grad(set_to_none=True)
            server_model.zero_grad(set_to_none=True)

            if kwargs:
                client_out = client_model(**kwargs)
            else:
                client_out = client_model(*args)
            server_out = server_model(client_out)
            logits = _to_logits(server_out)
            if logits is None:
                continue

            loss = _loss_fn(logits, labels)
            loss.backward()

            _accumulate_grad(client_model, client_grad_sum, batch_size)
            _accumulate_grad(server_model, server_grad_sum, batch_size)

            used_batches += 1
            used_samples += batch_size

            if used_batches >= max_batches:
                break
    finally:
        client_model.zero_grad(set_to_none=True)
        server_model.zero_grad(set_to_none=True)
        client_model.train(was_client_training)
        server_model.train(was_server_training)

    client_grad = _flatten_avg_grad(client_model, client_grad_sum, used_samples)
    server_grad = _flatten_avg_grad(server_model, server_grad_sum, used_samples)

    c_client = (-client_grad) if client_grad is not None else None
    c_server = (-server_grad) if server_grad is not None else None

    return c_client, c_server, {
        "used_batches": int(used_batches),
        "used_samples": int(used_samples),
    }
