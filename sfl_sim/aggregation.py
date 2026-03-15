"""
Model aggregation strategies.

Supported: fedavg (weighted average by dataset size)
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


def aggregate(
    method: str,
    state_dicts: List[dict],
    weights: List[float],
    **kwargs,
) -> dict:
    """
    Dispatch to aggregation method.

    Args:
        method: "fedavg" (more to come in Phase 2+)
        state_dicts: Client model state_dicts
        weights: Per-client weights

    Returns: Aggregated state_dict
    """
    if method == "fedavg":
        return fedavg(state_dicts, weights)
    raise ValueError(f"Unknown aggregator: {method}")
