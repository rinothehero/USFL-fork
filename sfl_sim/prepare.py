"""
Pre-compute data distribution and selection schedule before training.

Usage:
    python -m sfl_sim.prepare \\
        -d cifar10 -nc 100 -ncpr 10 -gr 100 \\
        --distribution shard_dirichlet --alpha 0.3 --lpc 2 \\
        --selector usfl --seed 42

    # Custom output directory:
    python -m sfl_sim.prepare ... -o schedules/my_experiment

Outputs:
    data_map.json   — {client_id: [data_indices]}
    schedule.json   — {round: [client_ids]}
    meta.json       — generation parameters + coverage/fairness stats

These files are shared across all methods for fair comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data import load_dataset, distribute, get_label_distribution


# ---------------------------------------------------------------------------
# Selectors (single-round)
# ---------------------------------------------------------------------------


def uniform_select(
    num_to_select: int,
    client_ids: List[int],
    rng: np.random.RandomState,
) -> List[int]:
    """Randomly select clients uniformly without replacement."""
    selected = rng.choice(client_ids, size=num_to_select, replace=False)
    return sorted(selected.tolist())


def usfl_select(
    num_to_select: int,
    client_ids: List[int],
    client_label_sets: Dict[int, set],
    all_labels: set,
    selection_count: Dict[int, int],
    rng: np.random.RandomState,
) -> List[int]:
    """
    Coverage-first, fairness-fill client selection.

    Phase 1 (Coverage): Greedily pick clients to cover all classes.
    Phase 2 (Fairness): Fill remaining slots with least-selected clients.

    Args:
        num_to_select: Number of clients to pick
        client_ids: Pool of all client IDs
        client_label_sets: {cid: set of labels} (precomputed)
        all_labels: set of all labels in the dataset
        selection_count: {cid: times_selected} (mutated in-place)
        rng: random state
    """
    max_count = max(selection_count.values()) if selection_count else 0
    norm = max(max_count, 1)

    selected = []
    remaining = set(client_ids)
    uncovered = set(all_labels)

    for _ in range(min(num_to_select, len(client_ids))):
        if not remaining:
            break

        if uncovered:
            # --- Coverage phase ---
            candidates = [
                cid for cid in remaining
                if client_label_sets[cid] & uncovered
            ]

            if not candidates:
                # Coverage impossible for remaining classes
                uncovered = set()
            else:
                def coverage_score(cid):
                    n_covers = len(client_label_sets[cid] & uncovered)
                    fair = selection_count.get(cid, 0) / norm
                    return (n_covers, -fair, rng.random() * 0.001)

                best = max(candidates, key=coverage_score)
                selected.append(best)
                remaining.remove(best)
                uncovered -= client_label_sets[best]
                continue

        # --- Fairness phase ---
        def fairness_score(cid):
            fair = selection_count.get(cid, 0) / norm
            return (-fair, rng.random() * 0.001)

        best = max(remaining, key=fairness_score)
        selected.append(best)
        remaining.remove(best)

    return sorted(selected)


# ---------------------------------------------------------------------------
# Schedule generation (pre-computed before training)
# ---------------------------------------------------------------------------


def generate_selection_schedule(
    method: str,
    num_rounds: int,
    num_per_round: int,
    client_ids: List[int],
    trainset,
    client_data_masks: Dict[int, List[int]],
    seed: int,
) -> List[List[int]]:
    """
    Pre-compute client selection for all rounds before training starts.

    This decouples selection logic from training hooks, so any method
    can use any selector strategy.

    Args:
        method: "uniform" or "usfl"
        num_rounds: Total training rounds
        num_per_round: Clients per round
        client_ids: All client IDs
        trainset: Training dataset (for label extraction)
        client_data_masks: {cid: [indices]} from data distribution
        seed: Random seed for reproducibility

    Returns:
        schedule[round_idx] = sorted list of selected client IDs
        (round_idx 0-based, i.e. schedule[0] = round 1 selection)
    """
    rng = np.random.RandomState(seed)
    schedule = []

    if method == "uniform":
        for _ in range(num_rounds):
            selected = uniform_select(num_per_round, client_ids, rng)
            schedule.append(selected)
        return schedule

    if method == "usfl":
        # Precompute label distributions and label sets
        client_label_dists = {}
        for cid in client_ids:
            client_label_dists[cid] = get_label_distribution(
                trainset, client_data_masks[cid]
            )

        client_label_sets = {
            cid: set(dist.keys()) for cid, dist in client_label_dists.items()
        }
        all_labels = set()
        for s in client_label_sets.values():
            all_labels |= s

        # Track selection counts for fairness across rounds
        selection_count = {cid: 0 for cid in client_ids}

        for _ in range(num_rounds):
            selected = usfl_select(
                num_per_round, client_ids,
                client_label_sets, all_labels,
                selection_count, rng,
            )
            # Update counts
            for cid in selected:
                selection_count[cid] += 1
            schedule.append(selected)

        return schedule

    raise ValueError(f"Unknown selector: {method}")


# ---------------------------------------------------------------------------
# CLI prepare tool
# ---------------------------------------------------------------------------


def _parse_prepare_args(argv=None):
    """Parse arguments for schedule preparation (no method/model/training args)."""
    p = argparse.ArgumentParser(
        description="Pre-compute data distribution and selection schedule"
    )

    # Dataset
    p.add_argument("-d", "--dataset", default="cifar10",
                   help="cifar10 | cifar100 | fmnist | mnist | svhn")

    # Clients
    p.add_argument("-nc", "--num-clients", type=int, default=100)
    p.add_argument("-ncpr", "--clients-per-round", type=int, default=10)

    # Rounds
    p.add_argument("-gr", "--rounds", type=int, default=100)

    # Distribution
    p.add_argument("--distribution", default="shard_dirichlet",
                   help="uniform | dirichlet | shard_dirichlet")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Dirichlet alpha (lower = more Non-IID)")
    p.add_argument("--lpc", "--labels-per-client", dest="labels_per_client",
                   type=int, default=2)
    p.add_argument("--min-require-size", type=int, default=10)

    # Selector
    p.add_argument("-s", "--selector", default="usfl",
                   help="uniform | usfl")

    # Seed
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (auto-generated if not set)")

    return p.parse_args(argv)


def prepare(args) -> Path:
    """Generate data_map.json and schedule.json."""

    # Determine output directory
    if args.output_dir is None:
        name = f"seed{args.seed}_{args.dataset}_{args.num_clients}c_{args.distribution}"
        if args.distribution == "shard_dirichlet":
            name += f"_a{args.alpha}_lpc{args.labels_per_client}"
        elif args.distribution == "dirichlet":
            name += f"_a{args.alpha}"
        name += f"_{args.selector}"
        out_dir = Path("sfl_sim/schedules") / name
    else:
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    trainset, _, num_classes = load_dataset(args.dataset, data_dir="./data")

    # 2. Distribute data
    print(f"[prepare] Distributing {args.dataset} to {args.num_clients} clients "
          f"({args.distribution}, alpha={args.alpha}, lpc={args.labels_per_client}, "
          f"seed={args.seed})...")
    client_data_masks = distribute(
        trainset,
        args.num_clients,
        args.distribution,
        alpha=args.alpha,
        labels_per_client=args.labels_per_client,
        min_require_size=args.min_require_size,
        seed=args.seed,
    )

    # 3. Save data_map.json
    data_map = {str(cid): indices for cid, indices in client_data_masks.items()}
    with open(out_dir / "data_map.json", "w") as f:
        json.dump(data_map, f)
    print(f"[prepare] data_map.json saved ({len(client_data_masks)} clients)")

    # 4. Generate selection schedule
    client_ids = list(range(args.num_clients))
    print(f"[prepare] Generating {args.rounds}-round schedule "
          f"(selector={args.selector}, {args.clients_per_round}/round)...")
    schedule = generate_selection_schedule(
        method=args.selector,
        num_rounds=args.rounds,
        num_per_round=args.clients_per_round,
        client_ids=client_ids,
        trainset=trainset,
        client_data_masks=client_data_masks,
        seed=args.seed,
    )

    # 5. Save schedule.json (1-based round keys)
    schedule_dict = {str(r + 1): clients for r, clients in enumerate(schedule)}
    with open(out_dir / "schedule.json", "w") as f:
        json.dump(schedule_dict, f)
    print(f"[prepare] schedule.json saved ({len(schedule)} rounds)")

    # 6. Save meta.json
    label_dists = {}
    for cid in client_ids:
        label_dists[cid] = get_label_distribution(trainset, client_data_masks[cid])

    all_labels = set()
    for dist in label_dists.values():
        all_labels.update(dist.keys())

    coverage_failures = 0
    for selected in schedule:
        round_labels = set()
        for cid in selected:
            round_labels.update(label_dists[cid].keys())
        if round_labels != all_labels:
            coverage_failures += 1

    selection_counts = {cid: 0 for cid in client_ids}
    for selected in schedule:
        for cid in selected:
            selection_counts[cid] += 1
    counts = list(selection_counts.values())

    meta = {
        "seed": args.seed,
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "clients_per_round": args.clients_per_round,
        "rounds": args.rounds,
        "distribution": args.distribution,
        "alpha": args.alpha,
        "labels_per_client": args.labels_per_client,
        "selector": args.selector,
        "num_classes": len(all_labels),
        "coverage_failures": coverage_failures,
        "selection_count_min": min(counts),
        "selection_count_max": max(counts),
        "selection_count_mean": round(sum(counts) / len(counts), 2),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[prepare] Output: {out_dir}/")
    print(f"  Classes: {len(all_labels)}")
    print(f"  Coverage failures: {coverage_failures}/{len(schedule)} rounds")
    print(f"  Selection count: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f}")

    return out_dir


def main():
    args = _parse_prepare_args()
    prepare(args)


if __name__ == "__main__":
    main()
