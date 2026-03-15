"""
Dynamic batch scheduler for USFL.

Computes optimal iteration count to fully utilize all client data
while keeping total batch size close to target.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


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
