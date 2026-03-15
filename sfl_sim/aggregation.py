"""
Model aggregation strategies.

Supported: fedavg, usfl (label-capped weighted aggregation)
"""
from __future__ import annotations

from typing import Dict, List

import torch


def fedavg(
    state_dicts: List[dict],
    weights: List[float],
) -> dict:
    """
    Federated Averaging: weighted average of model state dicts.

    Args:
        state_dicts: List of model state_dicts from clients
        weights: Per-client weights (typically dataset_size)

    Returns: Aggregated state_dict
    """
    total = sum(weights)
    avg = {}
    for key in state_dicts[0]:
        avg[key] = sum(
            sd[key].float() * (w / total)
            for sd, w in zip(state_dicts, weights)
        ).to(state_dicts[0][key].dtype)
    return avg


def usfl_aggregate(
    state_dicts: List[dict],
    label_distributions: List[Dict[int, int]],
) -> dict:
    """
    USFL label-capped weighted aggregation.

    1. Compute global label distribution (union of selected clients)
    2. Per-label weight cap: w_hat[l] = n_l_global / N_global
    3. Per-client weight: w_j = sum_l w_hat[l] * (n_l_j / n_l)
    4. Normalize and weighted average
    """
    # Global distribution
    global_dist: Dict[int, int] = {}
    for dist in label_distributions:
        for label, count in dist.items():
            global_dist[label] = global_dist.get(label, 0) + count
    N_global = sum(global_dist.values())

    if N_global == 0:
        n = len(state_dicts)
        return fedavg(state_dicts, [1.0] * n)

    # Per-label cap
    w_hat = {l: count / N_global for l, count in global_dist.items()}

    # Per-client weight
    weights = []
    for dist in label_distributions:
        w = 0.0
        for label, count in dist.items():
            n_l = global_dist.get(label, 1)
            w += w_hat.get(label, 0) * (count / n_l)
        weights.append(w)

    # Normalize
    total_w = sum(weights)
    if total_w > 0:
        weights = [w / total_w for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    return fedavg(state_dicts, weights)


def aggregate(
    method: str,
    state_dicts: List[dict],
    weights: List[float],
    **kwargs,
) -> dict:
    """
    Dispatch to aggregation method.

    Args:
        method: "fedavg" or "usfl"
        state_dicts: Client model state_dicts
        weights: Per-client weights (ignored for usfl, uses label_distributions)

    Returns: Aggregated state_dict
    """
    if method == "fedavg":
        return fedavg(state_dicts, weights)
    if method == "usfl":
        label_dists = kwargs.get("label_distributions", [])
        return usfl_aggregate(state_dicts, label_dists)
    raise ValueError(f"Unknown aggregator: {method}")
