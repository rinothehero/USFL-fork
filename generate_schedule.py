"""
Generate fixed client schedule by simulating USFL selector logic.

Replicates the USFL selector's alpha-based scoring (missing label coverage +
selection frequency balancing) without running actual training. Produces a
JSON schedule file usable by all methods via client_schedule_path.

Usage:
    python generate_schedule.py
    python generate_schedule.py --output experiment_configs/client_schedule.json
    python generate_schedule.py --rounds 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List

import numpy as np


# ── Shard-Dirichlet Distribution ────────────────────────────────────

def shard_dirichlet_distribute(
    targets: np.ndarray,
    num_clients: int,
    labels_per_client: int,
    alpha: float,
    seed: int,
    min_require_size: int = 12,
    max_retries: int = 10000,
) -> List[List[int]]:
    """Replicate ShardDirichletDistributer from the SFL framework."""
    num_classes = len(set(targets.tolist()))

    for retry in range(max_retries):
        prng = np.random.default_rng(seed + retry)

        # Step 1: class assignment pool (each class assigned equally)
        total_assignments = num_clients * labels_per_client
        per_class = total_assignments // num_classes
        class_pool = [k for k in range(num_classes) for _ in range(per_class)]
        remainder = total_assignments - len(class_pool)
        if remainder > 0:
            extra = prng.choice(num_classes, size=remainder, replace=False)
            class_pool.extend(extra.tolist())
        prng.shuffle(class_pool)

        # Step 2: assign classes to each client
        client2classes: Dict[int, List[int]] = {}
        for i in range(num_clients):
            assigned = list(set(class_pool[i * labels_per_client: (i + 1) * labels_per_client]))
            c = 0
            while len(assigned) < labels_per_client:
                if c not in assigned:
                    assigned.append(c)
                c += 1
            client2classes[i] = assigned

        # Step 3: distribute samples via Dirichlet
        idx_clients: List[List[int]] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            clients_k = [i for i in range(num_clients) if k in client2classes[i]]
            idx_k = np.where(targets == k)[0].copy()
            prng.shuffle(idx_k)
            if not clients_k:
                continue
            props = prng.dirichlet(np.repeat(alpha, len(clients_k)))
            splits = np.split(idx_k, (np.cumsum(props) * len(idx_k)).astype(int)[:-1])
            for ci, sp in zip(clients_k, splits):
                idx_clients[ci].extend(sp.tolist())

        min_size = min(len(v) for v in idx_clients)
        if min_size >= min_require_size:
            return idx_clients

    # Fallback: borrow from richest client
    for j in range(num_clients):
        if len(idx_clients[j]) == 0:
            richest = max(range(num_clients), key=lambda x: len(idx_clients[x]))
            borrow = max(1, min(min_require_size, len(idx_clients[richest]) // 2))
            for _ in range(borrow):
                if len(idx_clients[richest]) > 1:
                    idx_clients[j].append(idx_clients[richest].pop())
    return idx_clients


# ── USFL Selector Simulator ─────────────────────────────────────────

class USFLSelectorSim:
    """Simulates USFL selector alpha-based scoring without training."""

    def __init__(self, num_clients: int, clients_per_round: int,
                 num_classes: int, batch_size: int):
        self.num_clients = num_clients
        self.n = clients_per_round
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.selected_counts: Dict[str, int] = {str(l): 0 for l in range(num_classes)}
        self.client_selected_counts: Dict[int, int] = {c: 0 for c in range(num_clients)}
        self.history: deque = deque(maxlen=num_clients // clients_per_round)

    @staticmethod
    def _minmax(val: float, lo: float, hi: float) -> float:
        return 0.0 if hi == lo else (val - lo) / (hi - lo)

    @staticmethod
    def _kl(counts: Dict[str, int], nc: int) -> float:
        total = sum(counts.values())
        if total == 0:
            return float("inf")
        return sum((c / total) * math.log((c / total) * nc)
                   for c in counts.values() if c > 0)

    def select(self, label_dists: Dict[int, Dict[str, int]],
               alpha: float = 0.5) -> List[int]:
        nc = self.num_classes

        # Valid clients: total samples >= num_classes
        valid = [c for c in range(self.num_clients)
                 if sum(label_dists[c].values()) >= nc]

        for _retry in range(1000):
            agg = {str(l): 0 for l in range(nc)}
            ds_sizes: Dict[int, int] = {}
            sel: List[int] = []
            cands = valid.copy()
            random.shuffle(cands)

            # First pick
            first = cands.pop(0)
            sel.append(first)
            ds_sizes[first] = sum(label_dists[first].values())
            for l, c in label_dists[first].items():
                agg[l] += c

            missing = True
            under_bs = True

            while cands and (len(sel) < self.n or missing or under_bs):
                # Alpha-based greedy scoring
                scores = []
                for cand in cands:
                    csb = sum(1 for past in self.history if cand in past)
                    temp = agg.copy()
                    for l, c in label_dists[cand].items():
                        temp[l] += c
                    tmin = min(temp.values())
                    scores.append((cand, tmin, csb + 1))

                if not scores:
                    break

                mins = [s[1] for s in scores]
                csbs = [s[2] for s in scores]
                lo_m, hi_m = min(mins), max(mins)
                lo_c, hi_c = min(csbs), max(csbs)

                best_s, best_c = -float("inf"), None
                for cand, tmin, tcsb in scores:
                    s = alpha * self._minmax(tmin, lo_m, hi_m) + \
                        (1 - alpha) * (1 - self._minmax(tcsb, lo_c, hi_c))
                    if s > best_s:
                        best_s, best_c = s, cand

                if best_c is None:
                    break

                cands.remove(best_c)
                sel.append(best_c)
                ds_sizes[best_c] = sum(label_dists[best_c].values())
                for l, c in label_dists[best_c].items():
                    agg[l] += c

                # Batch-size pruning (only when batch_size > 64)
                if self.batch_size > 64:
                    total_ds = sum(ds_sizes.values())
                    g_bs = int(self.batch_size * len(sel) / self.n) \
                        if len(sel) >= self.n else self.batch_size
                    for cid in sel.copy():
                        cid_bs = int(g_bs * ds_sizes[cid] / total_ds) if total_ds > 0 else 0
                        if cid_bs < nc:
                            sel.remove(cid)
                            for l, c in label_dists[cid].items():
                                agg[l] -= c
                            del ds_sizes[cid]

                g_bs = int(self.batch_size * len(sel) / self.n) \
                    if len(sel) >= self.n else self.batch_size
                missing = any(v == 0 for v in agg.values())
                min_agg = min(agg.values()) if agg else 0
                min_crit = max(1, g_bs // nc) if g_bs > 0 else 1
                under_bs = not (len(sel) >= self.n and min_agg >= min_crit)

            if len(sel) >= self.n and not missing and not under_bs:
                self.history.append(list(sel))
                for l in range(nc):
                    self.selected_counts[str(l)] += agg[str(l)]
                for cid in sel:
                    self.client_selected_counts[cid] += 1
                return sel

        raise RuntimeError("USFLSelectorSim: failed after 1000 retries")


# ── Dataset targets ─────────────────────────────────────────────────

def get_targets(dataset: str) -> np.ndarray:
    """Get dataset targets (labels) for distribution."""
    if dataset == "cifar10":
        try:
            from torchvision.datasets import CIFAR10
            ds = CIFAR10(root="./data", train=True, download=True)
            return np.array(ds.targets)
        except Exception:
            print("[WARN] torchvision not available, using synthetic CIFAR-10 targets")
            return np.array([c for c in range(10) for _ in range(5000)])
    elif dataset == "fmnist":
        try:
            from torchvision.datasets import FashionMNIST
            ds = FashionMNIST(root="./data", train=True, download=True)
            return np.array(ds.targets)
        except Exception:
            return np.array([c for c in range(10) for _ in range(6000)])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate USFL client schedule (selection simulation)")
    parser.add_argument("--output", default="experiment_configs/client_schedule.json",
                        help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--total-clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--labels-per-client", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== USFL Schedule Generator ===")
    print(f"  Dataset: {args.dataset}")
    print(f"  Clients: {args.total_clients}, per round: {args.clients_per_round}")
    print(f"  Dirichlet alpha: {args.alpha}, labels/client: {args.labels_per_client}")
    print(f"  Rounds: {args.rounds}, seed: {args.seed}")
    print()

    # 1. Get dataset targets
    targets = get_targets(args.dataset)
    num_classes = len(set(targets.tolist()))
    print(f"  Samples: {len(targets)}, classes: {num_classes}")

    # 2. Distribute data (shard_dirichlet)
    min_req = max(10, args.batch_size // 4)
    idx_clients = shard_dirichlet_distribute(
        targets, args.total_clients, args.labels_per_client,
        args.alpha, args.seed, min_req,
    )
    sizes = [len(v) for v in idx_clients]
    print(f"  Distribution: min={min(sizes)}, max={max(sizes)}, "
          f"avg={np.mean(sizes):.0f} samples/client")

    # 3. Compute per-client label distributions
    label_dists: Dict[int, Dict[str, int]] = {}
    for cid in range(args.total_clients):
        counts = Counter(int(targets[i]) for i in idx_clients[cid])
        label_dists[cid] = {str(k): v for k, v in counts.items()}
    print()

    # 4. Simulate USFL selection
    selector = USFLSelectorSim(
        args.total_clients, args.clients_per_round, num_classes, args.batch_size)

    schedule: List[List[int]] = []
    for r in range(args.rounds):
        selected = selector.select(label_dists)
        schedule.append(selected)
        if (r + 1) % 50 == 0 or r == 0:
            print(f"  Round {r+1:3d}/{args.rounds}: {selected}")

    # 5. Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

    # 6. Stats
    freq = Counter()
    for rnd in schedule:
        freq.update(rnd)
    never = args.total_clients - len(freq)
    print(f"\n=== Done ===")
    print(f"  Saved: {out}")
    print(f"  Selection frequency: min={min(freq.values())}, "
          f"max={max(freq.values())}, never_selected={never}")
    print(f"\nTo use: set client_schedule_path in experiment_configs/common.json")


if __name__ == "__main__":
    main()
