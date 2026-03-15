"""
Client selection strategies.

Supported: uniform, usfl (missing label + freshness scoring)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def uniform_select(
    num_to_select: int,
    client_ids: List[int],
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Randomly select clients uniformly without replacement."""
    if rng is None:
        rng = np.random.RandomState()
    selected = rng.choice(client_ids, size=num_to_select, replace=False)
    return sorted(selected.tolist())


def usfl_select(
    num_to_select: int,
    client_ids: List[int],
    client_label_dists: Dict[int, Dict[int, int]],
    cumulative_usage: Optional[Dict] = None,
    use_fresh_scoring: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """
    USFL multi-phase client selection.

    Phase 1: Missing label scoring — favor clients with rare labels
    Phase 2: Freshness scoring — favor clients with less-used data
    Greedy selection with overlap penalty.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Collect all labels across all clients
    all_labels = set()
    for dist in client_label_dists.values():
        all_labels.update(dist.keys())

    # Phase 1: Label rarity score
    # Count how many clients have each label
    label_coverage = {l: 0 for l in all_labels}
    for dist in client_label_dists.values():
        for l in dist:
            label_coverage[l] += 1

    scores = {}
    for cid in client_ids:
        dist = client_label_dists.get(cid, {})
        # Score = sum of inverse coverage for each label this client has
        score = sum(1.0 / max(label_coverage.get(l, 1), 1) for l in dist)
        scores[cid] = score

    # Phase 2: Freshness scoring
    if use_fresh_scoring and cumulative_usage:
        for cid in client_ids:
            usage = cumulative_usage.get(cid, {})
            total_usage = 0
            for label_bins in usage.values():
                total_usage += sum(label_bins.values())
            freshness = 1.0 / (1.0 + total_usage)
            scores[cid] = scores.get(cid, 0) + freshness

    # Greedy selection with overlap penalty
    selected = []
    remaining = set(client_ids)

    for _ in range(min(num_to_select, len(client_ids))):
        if not remaining:
            break

        # Pick highest scoring
        best = max(remaining, key=lambda c: scores.get(c, 0) + rng.random() * 0.01)
        selected.append(best)
        remaining.remove(best)

        # Penalize remaining clients that overlap in labels
        best_labels = set(client_label_dists.get(best, {}).keys())
        for cid in remaining:
            overlap = best_labels & set(client_label_dists.get(cid, {}).keys())
            if overlap:
                scores[cid] *= 0.5

    return sorted(selected)


def select_clients(
    method: str,
    num_to_select: int,
    client_ids: List[int],
    rng: Optional[np.random.RandomState] = None,
    **kwargs,
) -> List[int]:
    """
    Dispatch to selection method.

    Args:
        method: "uniform" or "usfl"
        num_to_select: Number of clients to select
        client_ids: Pool of available client IDs
        rng: Random state for reproducibility

    Returns: Sorted list of selected client IDs
    """
    if method == "uniform":
        return uniform_select(num_to_select, client_ids, rng)
    if method == "usfl":
        return usfl_select(
            num_to_select, client_ids,
            client_label_dists=kwargs.get("client_label_dists", {}),
            cumulative_usage=kwargs.get("cumulative_usage"),
            use_fresh_scoring=kwargs.get("use_fresh_scoring", False),
            rng=rng,
        )
    raise ValueError(f"Unknown selector: {method}")
