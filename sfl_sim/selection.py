"""
Client selection strategies.

Supported: uniform (random selection)
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
        method: "uniform" (more to come in Phase 2+)
        num_to_select: Number of clients to select
        client_ids: Pool of available client IDs
        rng: Random state for reproducibility

    Returns: Sorted list of selected client IDs
    """
    if method == "uniform":
        return uniform_select(num_to_select, client_ids, rng)
    raise ValueError(f"Unknown selector: {method}")
